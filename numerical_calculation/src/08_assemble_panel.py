"""08_assemble_panel.py
=========================================================
合并全部 29 个因子 + 月度超额收益 + 控制变量 → 最终因子面板。

修正点（对应审计）：
- 收益改超额收益：ret_excess = ret_m − rf_m，ret_next_excess 同理；
  rf_m 用 1 月期 SHIBOR 年化转月化。回归目标用超额收益。
- log_size：ETF 用 log(份额×收盘价)（fund_share）；商品用 log(月末持仓×收盘×乘数)
  （fut_daily oi）。缺数据时回退到月成交额并标注。
- 标的覆盖 ETF + 商品期权，按 config.UNDERLYINGS 统一。

输入: data/processed/factors_*.parquet, data/raw/fund_daily_*.parquet,
       fut_daily_*.parquet, shibor.parquet, fund_share_*.parquet
输出: data/processed/factor_panel.parquet
=========================================================
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import UNDERLYINGS
from utils import to_datetime

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_monthly_returns() -> pd.DataFrame:
    """每个标的月度对数收益 + 月成交额 + 月末持仓（商品）。"""
    rows = []
    for u in UNDERLYINGS:
        code = u["code"]
        if u["asset"] == "etf":
            f = os.path.join(RAW_DIR, f"fund_daily_{code.replace('.', '_')}.parquet")
            oi_col = None
        else:
            f = os.path.join(RAW_DIR, f"fut_daily_{u['fut_symbol'].replace('.', '_')}.parquet")
            oi_col = "oi"
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        df['trade_date'] = to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['ym'] = df['trade_date'].dt.to_period('M')
        agg = {'close': 'last', 'trade_date': 'last', 'vol': 'sum'}
        if oi_col and oi_col in df.columns:
            agg[oi_col] = 'last'
        last = df.groupby('ym').agg(**{k: (k, v) for k, v in agg.items()}).reset_index()
        last['underlying'] = code
        last['ret_m'] = np.log(last['close'] / last['close'].shift(1))
        last['vol_m'] = last['vol']
        if oi_col and oi_col in last.columns:
            last['oi_end'] = last[oi_col]
        else:
            last['oi_end'] = np.nan
        rows.append(last)
    return pd.concat(rows, ignore_index=True)


def load_monthly_rf() -> pd.DataFrame:
    """1 月期 SHIBOR 年化 → 月化无风险利率，按月末对齐。"""
    f = os.path.join(RAW_DIR, "shibor.parquet")
    if not os.path.exists(f):
        return pd.DataFrame(columns=['ym', 'rf_m'])
    sh = pd.read_parquet(f)
    sh['date'] = to_datetime(sh['date'])
    sh['ym'] = sh['date'].dt.to_period('M')
    sh['ann'] = sh['1m'] / 100.0
    m = sh.groupby('ym')['ann'].mean().reset_index()
    m['rf_m'] = (1 + m['ann']) ** (1 / 12) - 1
    return m[['ym', 'rf_m']]


def load_etf_shares() -> pd.DataFrame:
    """ETF 月末份额 → [underlying, ym, shares]。"""
    rows = []
    for u in UNDERLYINGS:
        if u["asset"] != "etf":
            continue
        f = os.path.join(RAW_DIR, f"fund_share_{u['code'].replace('.', '_')}.parquet")
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        datecol = 'trade_date' if 'trade_date' in d.columns else 'ann_date'
        d['trade_date'] = to_datetime(d[datecol])
        d['ym'] = d['trade_date'].dt.to_period('M')
        # 份额列兼容
        sc = [c for c in d.columns if 'share' in c.lower()]
        if not sc:
            continue
        s = d.groupby('ym')[sc[0]].last().reset_index().rename(columns={sc[0]: 'shares'})
        s['underlying'] = u['code']
        rows.append(s[['underlying', 'ym', 'shares']])
    if not rows:
        return pd.DataFrame(columns=['underlying', 'ym', 'shares'])
    return pd.concat(rows, ignore_index=True)


def main():
    print("[08] 合并因子面板 ...")
    fiv   = pd.read_parquet(os.path.join(PROC_DIR, "factors_iv.parquet"))
    fvar  = pd.read_parquet(os.path.join(PROC_DIR, "factors_var.parquet"))
    fterm = pd.read_parquet(os.path.join(PROC_DIR, "factors_term.parquet"))
    fbeta = pd.read_parquet(os.path.join(PROC_DIR, "factors_beta.parquet"))
    for d in [fiv, fvar, fterm, fbeta]:
        d['trade_date'] = pd.to_datetime(d['trade_date'])
        d['ym'] = d['trade_date'].dt.to_period('M')

    panel = fiv.drop(columns=['trade_date']) \
        .merge(fvar.drop(columns=['trade_date']), on=['underlying', 'ym'], how='outer') \
        .merge(fterm.drop(columns=['trade_date']), on=['underlying', 'ym'], how='outer') \
        .merge(fbeta.drop(columns=['trade_date']), on=['underlying', 'ym'], how='outer')

    rets = load_monthly_returns()
    rets['ym'] = rets['ym'].astype('period[M]')
    rf = load_monthly_rf()
    shares = load_etf_shares()
    shares['ym'] = shares['ym'].astype('period[M]')

    panel = panel.merge(rets[['underlying', 'ym', 'trade_date', 'close',
                              'vol_m', 'oi_end', 'ret_m']],
                        on=['underlying', 'ym'], how='left')
    panel = panel.merge(rf, on='ym', how='left')
    panel = panel.merge(shares, on=['underlying', 'ym'], how='left')

    # 超额收益
    panel['rf_m'] = panel['rf_m'].fillna(0.0)
    panel = panel.sort_values(['underlying', 'ym']).reset_index(drop=True)
    panel['ret_excess'] = panel['ret_m'] - panel['rf_m']
    panel['ret_next'] = panel.groupby('underlying')['ret_m'].shift(-1)
    panel['ret_next_excess'] = panel.groupby('underlying')['ret_excess'].shift(-1)
    panel['ret_lag1'] = panel.groupby('underlying')['ret_m'].shift(1)

    # log_size：ETF 用份额×价；商品用持仓×价×乘数；缺则回退月成交额
    mult_map = {u['code']: u['mult'] for u in UNDERLYINGS}
    asset_map = {u['code']: u['asset'] for u in UNDERLYINGS}
    panel['mult'] = panel['underlying'].map(mult_map)
    panel['asset'] = panel['underlying'].map(asset_map)
    etf_size = np.where(panel['shares'].notna() & (panel['shares'] > 0),
                        panel['shares'] * panel['close'], np.nan)
    comm_size = panel['oi_end'] * panel['close'] * panel['mult']
    size = np.where(panel['asset'] == 'etf', etf_size, comm_size)
    # 回退：用月成交额（成交 notional）作规模代理
    fallback = panel['close'] * panel['vol_m']
    size = np.where(np.isnan(size) | (size <= 0), fallback, size)
    size = pd.Series(size).replace(0, np.nan)
    panel['log_size'] = np.log(size)

    panel = panel.rename(columns={'O_S': 'O_S'})

    factor_cols = [
        'CIV', 'PIV', 'dCIV', 'dPIV',
        'IVS_ATM', 'IVS_OTM', 'dIVS_ATM', 'dIVS_OTM', 'SKEW', 'AVAR',
        'VARQ', 'VAR_PLUS', 'VAR_MINUS', 'VRP', 'KT',
        'TSCALL', 'TSPUT', 'dTSCALL', 'dTSPUT', 'VOV',
        'O_S', 'betaVIX', 'betaSkew', 'betaVRP', 'betaIVSOTM',
        'betaTSCALL', 'betaTSPUT', 'betaJump', 'betaVol',
    ]
    keep = ['underlying', 'ym', 'trade_date', 'ret_m', 'ret_excess',
            'ret_next', 'ret_next_excess', 'ret_lag1', 'log_size'] + factor_cols
    panel = panel[keep]

    out = os.path.join(PROC_DIR, "factor_panel.parquet")
    panel.to_parquet(out, index=False)
    print(f"  ✓ 输出 {len(panel)} 行 × {len(panel.columns)} 列")
    print("  各标的样本量：")
    print(panel.groupby('underlying').size())
    print("  非缺失率（各因子）：")
    print(panel[factor_cols].notna().mean().sort_values().to_string())


if __name__ == "__main__":
    main()
