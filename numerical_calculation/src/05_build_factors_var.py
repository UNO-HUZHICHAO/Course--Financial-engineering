"""05_build_factors_var.py
=========================================================
风险中性方差与 VRP 类因子（共 6 个）：
    VARQ      = Carr-Madan (1998) model-free 风险中性方差（年化）
    VAR_PLUS  = call 端半方差
    VAR_MINUS = put 端半方差
    AVAR      = VAR_MINUS - VAR_PLUS （方差不对称）
    VRP       = VARQ - RV²           （RV 用滚动 20 日年化已实现方差）
    KT        = Kadan-Tang (2020) / Martin (2017) 期望收益下界（前向、领先阶）

修正点：
- VARQ 以 30 天期限为主（取距 30 天最近的到期月份），跨期限均值作稳健性备查；
  不再做跨期限简单平均作为主值。
- KT 不再等于 VARQ，改用 kadan_tang_lb（量纲为持有期收益下界）。
- RV 兼容商品期货主力连续收益。

输入: data/processed/iv_panel.parquet, data/raw/fund_daily_*.parquet, fut_daily_*.parquet
输出: data/processed/factors_var.parquet
=========================================================
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import UNDERLYINGS
from utils import carr_madan_variance, kadan_tang_lb, monthly_last, to_datetime

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def daily_var_one_group(g: pd.DataFrame) -> dict:
    """单个 (标的×交易日×到期月份) 的 Carr-Madan 方差。仅用 OTM 合约。"""
    if len(g) < 4:
        return {'VARQ': np.nan, 'VAR_PLUS': np.nan, 'VAR_MINUS': np.nan,
                'r': np.nan, 'T': np.nan}
    S = g['S'].iloc[0]; r = g['r'].iloc[0]; T = g['T'].iloc[0]
    q = g['q'].iloc[0] if 'q' in g.columns else 0.0
    F = S * np.exp((r - q) * T)
    otm = g[((g['call_put'] == 'C') & (g['exercise_price'] >= F)) |
            ((g['call_put'] == 'P') & (g['exercise_price'] <= F))]
    if len(otm) < 4:
        return {'VARQ': np.nan, 'VAR_PLUS': np.nan, 'VAR_MINUS': np.nan,
                'r': r, 'T': T}
    otm = otm.sort_values('exercise_price').drop_duplicates('exercise_price', keep='first')
    return {**carr_madan_variance(otm['exercise_price'].values,
                                  otm['settle'].values,
                                  (otm['call_put'] == 'C').values,
                                  S, r, T, q=q),
            'r': r, 'T': T}


def load_underlying_returns() -> pd.DataFrame:
    """所有标的（ETF + 商品期货主力）日对数收益。"""
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
        rows.append(df[['underlying', 'trade_date', 'ret']])
    return pd.concat(rows, ignore_index=True)


def main():
    print("[05] 构造方差类因子 ...")
    iv_panel = pd.read_parquet(os.path.join(PROC_DIR, "iv_panel.parquet"))
    iv_panel = iv_panel[iv_panel['T_days'].between(7, 120)]

    print("  计算 Carr-Madan 风险中性方差（按到期月份） ...")
    rows = []
    for (u, td, mat), g in iv_panel.groupby(['underlying', 'trade_date', 'maturity_date']):
        v = daily_var_one_group(g)
        v['underlying'] = u; v['trade_date'] = td; v['maturity_date'] = mat
        v['T_days'] = g['T_days'].iloc[0]
        rows.append(v)
    var_by_mat = pd.DataFrame(rows)

    # 主值：每个 (标的, 交易日) 取距 30 天最近的到期月份
    var_by_mat['dist30'] = (var_by_mat['T_days'] - 30).abs()
    idx = var_by_mat.groupby(['underlying', 'trade_date'])['dist30'].idxmin()
    primary = var_by_mat.loc[idx].copy()
    primary['AVAR'] = primary['VAR_MINUS'] - primary['VAR_PLUS']
    primary['KT'] = primary.apply(
        lambda r: kadan_tang_lb(r['VARQ'], r['r'], r['T']), axis=1)

    # 跨期限均值（稳健性，备查）
    cross = var_by_mat.groupby(['underlying', 'trade_date'])[
        ['VARQ', 'VAR_PLUS', 'VAR_MINUS']].mean().reset_index()
    cross = cross.rename(columns={'VARQ': 'VARQ_cross', 'VAR_PLUS': 'VAR_PLUS_cross',
                                  'VAR_MINUS': 'VAR_MINUS_cross'})

    daily_var = primary[['underlying', 'trade_date', 'VARQ', 'VAR_PLUS',
                         'VAR_MINUS', 'AVAR', 'KT']].merge(cross, on=['underlying', 'trade_date'])

    # RV
    print("  计算已实现方差 RV ...")
    rets = load_underlying_returns()
    rets = rets.sort_values(['underlying', 'trade_date']).reset_index(drop=True)
    rets['RV'] = rets.groupby('underlying')['ret'].transform(
        lambda x: x.rolling(20).var() * 252)
    daily_var = daily_var.merge(rets[['underlying', 'trade_date', 'RV']],
                                on=['underlying', 'trade_date'], how='left')
    daily_var['VRP'] = daily_var['VARQ'] - daily_var['RV']

    monthly = monthly_last(daily_var, date_col='trade_date')
    keep = ['underlying', 'trade_date', 'VARQ', 'VAR_PLUS', 'VAR_MINUS',
            'AVAR', 'VRP', 'KT']
    out = os.path.join(PROC_DIR, "factors_var.parquet")
    monthly[keep].to_parquet(out, index=False)
    print(f"  ✓ 输出 {len(monthly)} 行 → {out}")
    print(monthly[keep].describe())


if __name__ == "__main__":
    main()
