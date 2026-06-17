"""09_run_regressions.py
=========================================================
单变量 Newey-West 预测回归（经典 Neuhierl 表式对照）+ 与原论文方向对比。

作为 09_run_ml.py 的"可解释单变量"对照：
- 每个标的 × 每个因子：ret_next_excess ~ factor，Newey-West(4) HAC 标准误。
- 与原论文的方向对比：仅比较预测系数的"方向（符号）"是否一致——原论文为美股
  横截面 FF6 alpha，中国市场为时序/面板 beta，量纲不可直接比，故只比方向与
  显著性，并在报告第四部分给出机制解释。不再使用手抄的"精确 alpha 百分数"。

输入: data/processed/factor_panel.parquet, result/tables/ml_feature_importance.csv(可选)
输出: result/tables/ts_regression.csv, comparison_with_paper.csv
=========================================================
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from utils import newey_west_se

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TBL_DIR   = os.path.join(os.path.dirname(__file__), "..", "result", "tables")
os.makedirs(TBL_DIR, exist_ok=True)

FACTOR_COLS = [
    'CIV', 'PIV', 'dCIV', 'dPIV',
    'IVS_ATM', 'IVS_OTM', 'dIVS_ATM', 'dIVS_OTM', 'SKEW', 'AVAR',
    'VARQ', 'VAR_PLUS', 'VAR_MINUS', 'VRP', 'KT',
    'TSCALL', 'TSPUT', 'dTSCALL', 'dTSPUT', 'VOV',
    'O_S', 'betaVIX', 'betaSkew', 'betaVRP', 'betaIVSOTM',
    'betaTSCALL', 'betaTSPUT', 'betaJump', 'betaVol',
]
TARGET = 'ret_next_excess'

# 原论文（Neuhierl et al. 2024）报告的预测系数方向（- 负向预测 / + 正向预测 / 0 不显著）。
# 来源：论文 Table 2/3 对下月收益的横截面预测符号。仅用于方向对比。
PAPER_SIGN = {
    'IVS_ATM': -1, 'IVS_OTM': -1, 'dIVS_ATM': -1, 'dIVS_OTM': -1,
    'SKEW': -1, 'AVAR': -1,
    'CIV': -1, 'PIV': -1, 'dCIV': +1, 'dPIV': -1,
    'O_S': -1, 'VRP': +1, 'VARQ': -1, 'VAR_PLUS': -1, 'VAR_MINUS': -1, 'KT': -1,
    'VOV': -1, 'TSCALL': -1, 'TSPUT': -1, 'dTSCALL': +1, 'dTSPUT': -1,
    'betaJump': +1, 'betaVol': +1, 'betaVIX': -1, 'betaSkew': -1,
    'betaVRP': -1, 'betaIVSOTM': -1, 'betaTSCALL': -1, 'betaTSPUT': -1,
}
PAPER_ROBUST6 = ['IVS_ATM', 'IVS_OTM', 'dIVS_ATM', 'dIVS_OTM', 'SKEW', 'AVAR']


def time_series_regressions(panel):
    """每个标的 × 每个因子：ret_next_excess ~ factor，Newey-West(4)。"""
    rows = []
    for u in panel['underlying'].unique():
        sub = panel[panel['underlying'] == u]
        for f in FACTOR_COLS:
            res = newey_west_se(sub[f].values, sub[TARGET].values, lags=4)
            rows.append({'underlying': u, 'factor': f, **res})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TBL_DIR, "ts_regression.csv"), index=False)
    print("  ✓ 时间序列回归 → ts_regression.csv")
    return df


def comparison_with_paper(ts_reg):
    """方向对比：把我们的时序 beta 符号（多数标的一致方向计为有效）与原论文方向比。"""
    # 每个因子：统计在各标的中 beta 符号与原论文一致的比例
    g = ts_reg.dropna(subset=['beta']).copy()
    g['china_sign'] = np.sign(g['beta'])
    g['paper_sign'] = g['factor'].map(PAPER_SIGN)
    g['agree'] = (g['china_sign'] == g['paper_sign']).astype(int)
    agg = g.groupby('factor').agg(
        china_sign_majority=('china_sign', lambda x: int(np.sign(x.sum())) if x.sum() != 0 else 0),
        agree_ratio=('agree', 'mean'),
        mean_t=('beta_t', 'mean'),
        n_underlyings=('underlying', 'nunique'),
    ).reset_index()
    agg['paper_sign'] = agg['factor'].map(PAPER_SIGN)
    agg['paper_robust6'] = agg['factor'].isin(PAPER_ROBUST6)
    agg['direction_match'] = agg['china_sign_majority'] == agg['paper_sign']
    # 合并 ML 是否选出
    ml_path = os.path.join(TBL_DIR, "ml_feature_importance.csv")
    if os.path.exists(ml_path):
        ml = pd.read_csv(ml_path)[['factor', 'lasso_selected', 'rf_perm']]
        agg = agg.merge(ml, on='factor', how='left')
    agg = agg[['factor', 'paper_sign', 'china_sign_majority', 'direction_match',
               'agree_ratio', 'mean_t', 'paper_robust6', 'lasso_selected', 'rf_perm']]
    agg.to_csv(os.path.join(TBL_DIR, "comparison_with_paper.csv"), index=False)
    print("  ✓ 中外方向对比 → comparison_with_paper.csv")
    return agg


def main():
    print("[09-TS] 单变量 Newey-West 预测回归 ...")
    panel = pd.read_parquet(os.path.join(PROC_DIR, "factor_panel.parquet"))
    panel = panel.dropna(subset=[TARGET])
    ts = time_series_regressions(panel)
    comparison_with_paper(ts)
    print("[OK] 单变量回归与对比完成。")


if __name__ == "__main__":
    main()
