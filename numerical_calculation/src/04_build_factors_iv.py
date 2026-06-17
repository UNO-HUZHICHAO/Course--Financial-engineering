"""04_build_factors_iv.py
=========================================================
基于标准化 IV 曲面构造 IV 类与 IV spread 类因子（共 9 个日度因子）。

日度因子：
    CIV       = iv_30d_call_d50           # An et al. (2014)
    PIV       = iv_30d_put_d50
    IVS_ATM   = iv_30d_put_d50 - iv_30d_call_d50  # Cremers-Weinbaum (2010)
    IVS_OTM   = iv_30d_put_d20 - iv_30d_call_d50  # Xing et al. (2010)
    SKEW      = avg(put 20-40 delta) - avg(call 20-40 delta)  # Yan (2011)

月度因子（取月末日度因子值）：
    CIV, PIV, IVS_ATM, IVS_OTM, SKEW
    ΔCIV, ΔPIV, ΔIVS_ATM, ΔIVS_OTM   = 月度差分

输入: data/processed/iv_surface.parquet
输出: data/processed/factors_iv.parquet —— 月度因子面板
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
    print("[04] 构造 IV / IV spread 因子 ...")
    surface = pd.read_parquet(os.path.join(PROC_DIR, "iv_surface.parquet"))
    surface = surface.sort_values(['underlying', 'trade_date']).reset_index(drop=True)

    # 日度因子
    print("  计算日度因子 ...")
    surface['CIV']     = surface['iv_30d_call_d50']
    surface['PIV']     = surface['iv_30d_put_d50']
    # IVS_ATM：Cremers-Weinbaum (2010) 的 ATM call-put IV 价差，对齐 delta 0.25
    surface['IVS_ATM'] = surface['iv_30d_put_d25']  - surface['iv_30d_call_d25']
    # IVS_OTM：Xing-Zhang-Zhao (2010) OTM put(Δ0.20) − ATM call(Δ0.50)
    surface['IVS_OTM'] = surface['iv_30d_put_d20']  - surface['iv_30d_call_d50']

    # SKEW：Yan (2011) 风格的看跌-看涨 IV 偏度，取对数比（跨 delta 20-40 均值）
    put_cols  = ['iv_30d_put_d20',  'iv_30d_put_d30',  'iv_30d_put_d40']
    call_cols = ['iv_30d_call_d20', 'iv_30d_call_d30', 'iv_30d_call_d40']
    put_avg  = surface[put_cols].mean(axis=1)
    call_avg = surface[call_cols].mean(axis=1)
    surface['SKEW'] = np.log(put_avg / call_avg)

    # 月末取值
    print("  转月度（取月末日度值） ...")
    monthly = monthly_last(surface, date_col='trade_date')

    # 月度差分
    monthly = monthly.sort_values(['underlying', 'trade_date']).reset_index(drop=True)
    for f in ['CIV', 'PIV', 'IVS_ATM', 'IVS_OTM']:
        monthly[f'd{f}'] = monthly.groupby('underlying')[f].diff()

    keep = ['underlying', 'trade_date',
            'CIV', 'PIV', 'IVS_ATM', 'IVS_OTM', 'SKEW',
            'dCIV', 'dPIV', 'dIVS_ATM', 'dIVS_OTM']
    out = os.path.join(PROC_DIR, "factors_iv.parquet")
    monthly[keep].to_parquet(out, index=False)
    print(f"  ✓ 输出 {len(monthly)} 行 → {out}")
    print(monthly[keep].describe())


if __name__ == "__main__":
    main()
