"""10_make_plots.py
=========================================================
全部可视化输出：
1. 因子时间序列：各标的 IVS_OTM / SKEW / AVAR / VRP
2. IV smile 示例：某标的某日的 IV-行权价曲线
3. 因子-下期超额收益散点（合池）
4. ML 特征重要性条形图（Gini + 排列重要性）
5. 样本外预测 vs 实际（前向 CV）
6. 因子相关性热图
7. 单变量 panel t 值条形图（含 Bonferroni 阈值）

输出: result/figures/*.png
=========================================================
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import UNDERLYINGS
from utils import to_datetime

for f in ['PingFang SC', 'Heiti SC', 'STHeiti', 'Hiragino Sans GB', 'Microsoft YaHei']:
    if f in [ff.name for ff in mpl.font_manager.fontManager.ttflist]:
        plt.rcParams['font.family'] = f
        break
plt.rcParams['axes.unicode_minus'] = False

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TBL_DIR  = os.path.join(os.path.dirname(__file__), "..", "result", "tables")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "result", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

NAME_MAP = {u["code"]: u["name"] for u in UNDERLYINGS}


def _codes_in(panel):
    return [c for c in NAME_MAP if c in panel['underlying'].unique()]


def plot_factor_time_series(panel):
    factors = ['IVS_OTM', 'SKEW', 'AVAR', 'VRP']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    codes = _codes_in(panel)
    for ax, f in zip(axes.flat, factors):
        for code in codes:
            sub = panel[panel['underlying'] == code].sort_values('trade_date')
            ax.plot(sub['trade_date'], sub[f], label=NAME_MAP[code], lw=0.9, alpha=0.8)
        ax.set_title(f, fontsize=12)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc='best', ncol=2)
    fig.suptitle("中国期权 IV spread / 方差类因子时间序列", fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "factor_time_series.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {out}")


def plot_iv_smile_example():
    iv_panel = pd.read_parquet(os.path.join(PROC_DIR, "iv_panel.parquet"))
    # 选一个有充足合约的标的与日期
    for code in NAME_MAP:
        sub = iv_panel[(iv_panel['underlying'] == code) & (iv_panel['T_days'].between(20, 45))]
        if len(sub) < 6:
            continue
        td = sub['trade_date'].max()
        g = sub[sub['trade_date'] == td]
        if len(g) < 6:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for cp, m, c in [('C', 'o', 'tab:red'), ('P', 's', 'tab:blue')]:
            s = g[g['call_put'] == cp].sort_values('exercise_price')
            if len(s):
                ax.plot(s['exercise_price'], s['iv'], marker=m, color=c,
                        label='看涨 Call' if cp == 'C' else '看跌 Put', lw=1.2)
        ax.set_xlabel("行权价"); ax.set_ylabel("隐含波动率 (年化)")
        ax.set_title(f"{NAME_MAP[code]} 期权 IV smile - {pd.Timestamp(td).date()}")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(FIG_DIR, "iv_smile_example.png")
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  ✓ {out}")
        return


def plot_scatter_ivs_return(panel):
    sub = panel[['IVS_OTM', 'ret_next_excess', 'underlying']].dropna()
    if len(sub) < 5:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    codes = _codes_in(panel)
    for c in codes:
        s = sub[sub['underlying'] == c]
        ax.scatter(s['IVS_OTM'], s['ret_next_excess'] * 100, alpha=0.6, s=22,
                   label=NAME_MAP[c])
    coefs = np.polyfit(sub['IVS_OTM'], sub['ret_next_excess'] * 100, 1)
    xs = np.linspace(sub['IVS_OTM'].min(), sub['IVS_OTM'].max(), 100)
    ax.plot(xs, np.polyval(coefs, xs), 'k--', lw=1.5, label='OLS')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel("IVS_OTM (月末)"); ax.set_ylabel("下月超额对数收益 (%)")
    ax.set_title("IV spread (OTM) 对未来超额收益的预测：合池散点")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "ivs_vs_future_return.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {out}")


def plot_feature_importance():
    path = os.path.join(TBL_DIR, "ml_feature_importance.csv")
    if not os.path.exists(path):
        return
    fi = pd.read_csv(path).sort_values('rf_perm', ascending=True).dropna(subset=['rf_perm'])
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ['tab:red' if s else 'steelblue' for s in fi['lasso_selected']]
    ax.barh(fi['factor'], fi['rf_perm'], color=colors)
    ax.set_xlabel("Random Forest 排列重要性（OOS）")
    ax.set_title("ML 特征重要性：哪些期权特征预测力最强\n(红=LASSO 亦选出)")
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "ml_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {out}")


def plot_oos_predictions():
    path = os.path.join(TBL_DIR, "ml_predictions.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df['ym'] = pd.PeriodIndex(df['ym'], freq='M').to_timestamp()
    agg = df.groupby('ym')[['actual', 'pred_lasso', 'pred_ridge', 'pred_rf']].mean().sort_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(agg.index, agg['actual'], 'k-o', ms=3, lw=1.5, label='实际下月超额收益(均值)')
    for col, c in [('pred_lasso', 'tab:red'), ('pred_ridge', 'tab:green'), ('pred_rf', 'tab:blue')]:
        ax.plot(agg.index, agg[col], '-', color=c, lw=1.2, label=col)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel("月份"); ax.set_ylabel("月度超额收益(均值)")
    ax.set_title("前向 walk-forward 样本外预测 vs 实际（截面均值）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "oos_predictions.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {out}")


def plot_correlation_heatmap():
    path = os.path.join(TBL_DIR, "factor_correlations.csv")
    if not os.path.exists(path):
        return
    corr = pd.read_csv(path, index_col=0)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("29 个期权特征的相关性矩阵")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "factor_correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {out}")


def plot_panel_t_stat():
    # 优先用 univariate_panel.csv（含多重检验）；否则回退 panel_regression.csv
    for fn in ["univariate_panel.csv", "panel_regression.csv"]:
        path = os.path.join(TBL_DIR, fn)
        if os.path.exists(path):
            df = pd.read_csv(path).dropna(subset=['beta_t'])
            break
    else:
        return
    df = df.sort_values('beta_t')
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df['factor'], df['beta_t'],
            color=['tab:red' if abs(t) >= 1.96 else 'tab:gray' for t in df['beta_t']])
    ax.axvline(0, color='black', lw=0.5)
    ax.axvline(1.96, color='black', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(-1.96, color='black', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel("单变量 panel 回归 t 值（按月聚类）")
    ax.set_title("29 个期权特征对未来超额收益的 panel 预测显著性")
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "panel_t_stats.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {out}")


def plot_paper_comparison():
    path = os.path.join(TBL_DIR, "comparison_with_paper.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path).dropna(subset=['paper_sign', 'china_sign_majority'])
    df['paper_sign'] = df['paper_sign'].astype(int)
    df['china_sign_majority'] = df['china_sign_majority'].astype(int)
    fig, ax = plt.subplots(figsize=(8, 8))
    jitter = np.random.uniform(-0.15, 0.15, len(df))
    colors = ['tab:red' if r else 'tab:gray' for r in df['paper_robust6']]
    ax.scatter(df['paper_sign'] + jitter, df['china_sign_majority'] + jitter,
               c=colors, s=70, alpha=0.7)
    for _, r in df.iterrows():
        ax.annotate(r['factor'], (r['paper_sign'], r['china_sign_majority']),
                    fontsize=7, alpha=0.8, xytext=(4, 4), textcoords='offset points')
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("原论文预测方向 (-1/0/+1)")
    ax.set_ylabel("中国市场时序 beta 多数方向 (-1/0/+1)")
    ax.set_title("原论文 vs 中国市场：预测方向对比\n(红=原论文 6 个稳健 IV spread 因子)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "paper_vs_china_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {out}")


def main():
    print("[10] 生成可视化 ...")
    panel = pd.read_parquet(os.path.join(PROC_DIR, "factor_panel.parquet"))
    panel['trade_date'] = pd.to_datetime(panel['trade_date'])
    plot_iv_smile_example()
    plot_factor_time_series(panel)
    plot_scatter_ivs_return(panel)
    plot_feature_importance()
    plot_oos_predictions()
    plot_correlation_heatmap()
    plot_panel_t_stat()
    plot_paper_comparison()
    print("[OK] 全部图表已输出。")


if __name__ == "__main__":
    main()
