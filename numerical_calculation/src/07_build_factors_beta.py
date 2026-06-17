"""07_build_factors_beta.py
=========================================================
流动性因子 O/S 与 8 个 beta 因子。

设计修正（对应审计）：
- 滚动窗口 60 → 250 交易日（对齐原论文与 docstring）。
- 市场代理：不再硬编码 50ETF。改为每个标的用其自身的期权隐含因子序列
  （VIX/SKEW/IVS_OTM/TSCALL/TSPUT/VRP/JUMP/VOL）作为"市场因子"，
  估计该标的日收益对自身期权因子变动的滚动 beta。这样 ETF 与商品期权
  统一可算，且 beta 的经济含义是"收益-隐含信息敏感度"——正是横截面 ML
  所需的资产级特征（详见报告第四部分对截面 vs 时序的讨论）。
- O/S 合约乘数：ETF 用 10000（合约单位），商品期权标的为期货、单位相同故
  乘数抵消，O/S = log(期货成交量 / 期权成交量)。

输入: data/processed/iv_surface.parquet, iv_panel.parquet,
       data/raw/fund_daily_*.parquet, fut_daily_*.parquet
输出: data/processed/factors_beta.parquet
=========================================================
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import UNDERLYINGS
from utils import to_datetime, monthly_last, carr_madan_variance

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

WINDOW = 250  # 滚动窗口（交易日）


def load_returns() -> pd.DataFrame:
    rows = []
    for u in UNDERLYINGS:
        if u["asset"] == "etf":
            f = os.path.join(RAW_DIR, f"fund_daily_{u['code'].replace('.', '_')}.parquet")
        else:
            f = os.path.join(RAW_DIR, f"fut_daily_{u['fut_symbol'].replace('.', '_')}.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        df['trade_date'] = to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['ret'] = np.log(df['close'] / df['close'].shift(1))
        df['underlying'] = u['code']
        df = df.rename(columns={'vol': 'udl_vol'})
        rows.append(df[['underlying', 'trade_date', 'close', 'udl_vol', 'ret']])
    return pd.concat(rows, ignore_index=True)


def compute_OS(iv_panel: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """O/S = log(标的成交量 / 期权成交量×乘数)。

    ETF：标的是份额，期权合约单位=mult 份/张 → 除以 mult。
    商品：标的是期货手，期权单位与期货相同 → 乘数抵消，不除。
    """
    op = iv_panel[iv_panel['T_days'].between(5, 30)].copy()
    op['ym'] = op['trade_date'].dt.to_period('M')
    op_vol = op.groupby(['underlying', 'ym'])['vol'].sum().reset_index().rename(
        columns={'vol': 'option_vol'})

    rets = rets.copy()
    rets['ym'] = rets['trade_date'].dt.to_period('M')
    s_vol = rets.groupby(['underlying', 'ym'])['udl_vol'].sum().reset_index().rename(
        columns={'udl_vol': 'stock_vol_m'})

    df = s_vol.merge(op_vol, on=['underlying', 'ym'], how='inner')
    last_day = rets.groupby(['underlying', 'ym'])['trade_date'].max().reset_index()
    df = df.merge(last_day, on=['underlying', 'ym'], how='left')

    mult_map = {u['code']: u['mult'] for u in UNDERLYINGS}
    asset_map = {u['code']: u['asset'] for u in UNDERLYINGS}
    df['mult'] = df['underlying'].map(mult_map)
    df['asset'] = df['underlying'].map(asset_map)
    # ETF 除以 mult；商品不除（单位抵消）
    opt_eff = np.where(df['asset'] == 'etf',
                       df['option_vol'] * df['mult'],
                       df['option_vol'])
    df['OS'] = np.log(df['stock_vol_m'] / pd.Series(opt_eff).replace(0, np.nan))
    return df[['underlying', 'trade_date', 'OS']]


def own_surface_factors(surface: pd.DataFrame, code: str) -> pd.DataFrame:
    """某标的自身 surface → 日度期权因子序列。"""
    s = surface[surface['underlying'] == code].sort_values('trade_date').reset_index(drop=True)
    if s.empty:
        return pd.DataFrame()
    s['VIX']      = s['iv_30d_call_d50']
    s['IVS_OTM']  = s['iv_30d_put_d20'] - s['iv_30d_call_d50']
    s['TSCALL']   = s['iv_30d_call_d50'] - s['iv_180d_call_d50']
    s['TSPUT']    = s['iv_30d_put_d50']  - s['iv_180d_put_d50']
    put_cols  = ['iv_30d_put_d20',  'iv_30d_put_d30',  'iv_30d_put_d40']
    call_cols = ['iv_30d_call_d20', 'iv_30d_call_d30', 'iv_30d_call_d40']
    s['SKEW'] = np.log(s[put_cols].mean(axis=1) / s[call_cols].mean(axis=1))
    return s[['trade_date', 'VIX', 'SKEW', 'IVS_OTM', 'TSCALL', 'TSPUT']]


def own_vrp(iv_panel: pd.DataFrame, code: str, rets: pd.DataFrame) -> pd.DataFrame:
    """某标的自身 VRP 日度序列。"""
    sub = iv_panel[(iv_panel['underlying'] == code) &
                   (iv_panel['T_days'].between(7, 120))].copy()
    rows = []
    for (td, mat), g in sub.groupby(['trade_date', 'maturity_date']):
        S = g['S'].iloc[0]; r = g['r'].iloc[0]; T = g['T'].iloc[0]
        q = g['q'].iloc[0] if 'q' in g.columns else 0.0
        F = S * np.exp((r - q) * T)
        otm = g[((g['call_put'] == 'C') & (g['exercise_price'] >= F)) |
                ((g['call_put'] == 'P') & (g['exercise_price'] <= F))]
        if len(otm) < 4:
            continue
        otm = otm.sort_values('exercise_price').drop_duplicates('exercise_price', keep='first')
        v = carr_madan_variance(otm['exercise_price'].values, otm['settle'].values,
                                (otm['call_put'] == 'C').values, S, r, T, q=q)
        rows.append({'trade_date': td, 'VARQ': v['VARQ']})
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.DataFrame(columns=['trade_date', 'VRP'])
    daily = d.groupby('trade_date')['VARQ'].mean().reset_index()
    rv = rets[rets['underlying'] == code].sort_values('trade_date').reset_index(drop=True)
    rv['RV'] = rv['ret'].rolling(20).var() * 252
    daily = daily.merge(rv[['trade_date', 'RV']], on='trade_date', how='left')
    daily['VRP'] = daily['VARQ'] - daily['RV']
    return daily[['trade_date', 'VRP']]


def own_jump_vol(rets: pd.DataFrame, code: str) -> pd.DataFrame:
    m = rets[rets['underlying'] == code].sort_values('trade_date').reset_index(drop=True)
    m['JUMP'] = -(m['ret'].clip(upper=0)) ** 2
    m['VOL'] = m['ret'].rolling(20).var() * 252
    return m[['trade_date', 'JUMP', 'VOL']]


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var


def main():
    print("[07] 构造 O/S 与 beta 类因子 ...")
    iv_panel = pd.read_parquet(os.path.join(PROC_DIR, "iv_panel.parquet"))
    surface  = pd.read_parquet(os.path.join(PROC_DIR, "iv_surface.parquet"))
    surface['trade_date'] = pd.to_datetime(surface['trade_date'])
    iv_panel['trade_date'] = pd.to_datetime(iv_panel['trade_date'])
    rets = load_returns()

    print("  → O/S")
    os_df = compute_OS(iv_panel, rets)

    print(f"  → 滚动 beta（窗口 {WINDOW} 天，各标的自身期权因子）")
    rows = []
    for u in UNDERLYINGS:
        code = u['code']
        e = rets[rets['underlying'] == code][['trade_date', 'ret']].copy()
        m = own_surface_factors(surface, code)
        v = own_vrp(iv_panel, code, rets)
        j = own_jump_vol(rets, code)
        if m.empty:
            continue
        merged = e.merge(m, on='trade_date', how='left') \
                  .merge(v, on='trade_date', how='left') \
                  .merge(j, on='trade_date', how='left') \
                  .sort_values('trade_date').reset_index(drop=True)
        merged['underlying'] = code
        for col in ['VIX', 'SKEW', 'IVS_OTM', 'TSCALL', 'TSPUT', 'VRP']:
            merged[f'{col}_d'] = merged[col].diff()
        merged['betaVIX']    = rolling_beta(merged['ret'], merged['VIX_d'],    WINDOW)
        merged['betaSkew']   = rolling_beta(merged['ret'], merged['SKEW_d'],   WINDOW)
        merged['betaVRP']    = rolling_beta(merged['ret'], merged['VRP_d'],    WINDOW)
        merged['betaIVSOTM'] = rolling_beta(merged['ret'], merged['IVS_OTM_d'], WINDOW)
        merged['betaTSCALL'] = rolling_beta(merged['ret'], merged['TSCALL_d'], WINDOW)
        merged['betaTSPUT']  = rolling_beta(merged['ret'], merged['TSPUT_d'],  WINDOW)
        merged['betaJump']   = rolling_beta(merged['ret'], merged['JUMP'],     WINDOW)
        merged['betaVol']    = rolling_beta(merged['ret'], merged['VOL'].diff(), WINDOW)
        rows.append(merged)
    if not rows:
        print("  ⚠️  无可用标的")
        return
    beta_panel = pd.concat(rows, ignore_index=True)
    monthly_beta = monthly_last(beta_panel, date_col='trade_date')
    keep_betas = ['betaVIX', 'betaSkew', 'betaVRP', 'betaIVSOTM',
                  'betaTSCALL', 'betaTSPUT', 'betaJump', 'betaVol']
    out_betas = monthly_beta[['underlying', 'trade_date'] + keep_betas]
    out = out_betas.merge(os_df, on=['underlying', 'trade_date'], how='outer')
    out = out.rename(columns={'OS': 'O_S'})
    final = os.path.join(PROC_DIR, "factors_beta.parquet")
    out.to_parquet(final, index=False)
    print(f"  ✓ 输出 {len(out)} 行 → {final}")


if __name__ == "__main__":
    main()
