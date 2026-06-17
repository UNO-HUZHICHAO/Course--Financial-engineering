"""06_build_factors_term.py
=========================================================
期限结构与波动率波动率因子（共 5 个）：
    TSCALL  = iv_30d_call_d50 - iv_180d_call_d50    （注意符号习惯：原论文用 1月-6月）
    TSPUT   = iv_30d_put_d50  - iv_180d_put_d50
    dTSCALL = TSCALL 月度差分
    dTSPUT  = TSPUT  月度差分
    VOV     = (cv(call IV) + cv(put IV)) / 2  （月内日度 IV 序列的 cv，call+put 平均）

输入: data/processed/iv_surface.parquet
输出: data/processed/factors_term.parquet
=========================================================
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from utils import monthly_last

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main():
    print("[06] 构造期限结构与 VOV ...")
    surface = pd.read_parquet(os.path.join(PROC_DIR, "iv_surface.parquet"))
    surface = surface.sort_values(['underlying', 'trade_date']).reset_index(drop=True)

    surface['TSCALL'] = surface['iv_30d_call_d50'] - surface['iv_180d_call_d50']
    surface['TSPUT']  = surface['iv_30d_put_d50']  - surface['iv_180d_put_d50']

    # VOV：月内 30 天 ATM IV 日度一阶差分（变动）的标准差，call+put 平均
    # （对齐原文 "IV 的波动率" 含义，而非变异系数）
    print("  计算月内 VOV（IV 变动标准差） ...")
    s = surface.copy()
    s = s.sort_values(['underlying', 'trade_date']).reset_index(drop=True)
    s['d_iv_call'] = s.groupby('underlying')['iv_30d_call_d50'].diff()
    s['d_iv_put']  = s.groupby('underlying')['iv_30d_put_d50'].diff()
    s['ym'] = s['trade_date'].dt.to_period('M')

    def std_chg(x):
        x = x.dropna()
        if len(x) < 5:
            return np.nan
        return x.std()

    vov_call = s.groupby(['underlying', 'ym'])['d_iv_call'].apply(std_chg).rename('vov_call')
    vov_put  = s.groupby(['underlying', 'ym'])['d_iv_put'].apply(std_chg).rename('vov_put')
    vov = pd.concat([vov_call, vov_put], axis=1).reset_index()
    vov['VOV'] = (vov['vov_call'] + vov['vov_put']) / 2

    # 月末取期限结构
    monthly = monthly_last(surface, date_col='trade_date')
    monthly['ym'] = monthly['trade_date'].dt.to_period('M')

    monthly = monthly.merge(vov[['underlying', 'ym', 'VOV']],
                            on=['underlying', 'ym'], how='left')

    monthly = monthly.sort_values(['underlying', 'trade_date']).reset_index(drop=True)
    monthly['dTSCALL'] = monthly.groupby('underlying')['TSCALL'].diff()
    monthly['dTSPUT']  = monthly.groupby('underlying')['TSPUT'].diff()

    keep = ['underlying', 'trade_date',
            'TSCALL', 'TSPUT', 'dTSCALL', 'dTSPUT', 'VOV']
    out = os.path.join(PROC_DIR, "factors_term.parquet")
    monthly[keep].to_parquet(out, index=False)
    print(f"  ✓ 输出 {len(monthly)} 行 → {out}")
    print(monthly[keep].describe())


if __name__ == "__main__":
    main()
