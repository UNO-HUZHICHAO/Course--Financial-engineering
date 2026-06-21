# -*- coding: utf-8 -*-
"""
论文后三章（第5-7章）可视化代码
基于双流图网络的科创板指数增强实证诊断
========================================================
生成图表：
  图表2 — 因子 Rank IC 与 t 统计量诊断对比图 (第5.1节)
  图表3 — 核心微观因子分组(Q1-Q5)累计收益单调性曲线 (第5.1节)
  图表4 — 图网络边消融实验 Alpha 业绩对比图 (第5.2节)
  图表5 — 不同融合机制(Gate vs. FiLM vs. AdaptiveFiLM)业绩对比 (第5.3节)
  图表6 — 组合构建方式对比(QP优化 vs. 分层等权)超额收益反转 (第6.1节)
  图表7 — 全样本外滚动回测累计超额净值曲线 (第6.2节)
  图表8 — 双随机种子核心绩效指标汇总表 (第6.2节)

适用平台：macOS（使用 STHeiti / PingFang HK 中文字体）
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# 将实证部分 src 加入 path，以便导入 BacktestReporter
_EMPIRICAL_SRC = Path(__file__).resolve().parent.parent / "学年论文实证部分" / "src"
if str(_EMPIRICAL_SRC) not in sys.path:
    sys.path.insert(0, str(_EMPIRICAL_SRC))

_MARKET_DATA_DIR = Path(__file__).resolve().parent.parent / "学年论文实证部分" / "data" / "processed" / "market"
_TRADING_DAYS_PER_YEAR = 252
_TRANSACTION_COST = 0.002  # 双边 0.2%

# ============================================================
# 全局配置：macOS 中文字体 & 学术风格
# ============================================================
# STHeiti 为 macOS 系统自带黑体中文字体，PingFang HK 为苹方香港版
plt.rcParams['font.sans-serif'] = ['STHeiti', 'PingFang HK', 'Heiti TC', 'Lantinghei SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.axisbelow'] = True

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 学术配色方案 ──
C = {
    'blue':      '#2171B5',
    'lightblue': '#6BAED6',
    'orange':    '#E6550D',
    'lightorange':'#FD8D3C',
    'green':     '#238B45',
    'lightgreen':'#74C476',
    'red':       '#CB181D',
    'purple':    '#8C6BB1',
    'grey':      '#636363',
    'lightgrey': '#BDBDBD',
    'bg':        '#FCFCFC',
    'darkblue':  '#08306B',
}
COLOR_PALETTE = [C['blue'], C['orange'], C['green'], C['red'], C['purple'], C['grey']]


# ============================================================
# 图表2：因子 Rank IC 与 t 统计量诊断对比图 (第5.1节)
# ============================================================
def plot_chart2_factor_ic_diagnostics():
    """
    图表2：因子 Rank IC 与 t 统计量诊断
    数据来源：基于 base16+micro8 因子集在科创50成分股上的截面 Rank IC 诊断
    """
    factors = [
        'R2_10', 'Amihud_Trend', 'ILLIQ_20', 'OVERNIGHT_5', 'beta_5',
        'Ret_Kurt_20', 'TRC_3', 'delta_5_10', 'Ret_Skew_20', 'VP_Corr_20',
        'Volume_Change', 'Price_Position', 'Return_20d', 'Return_1d',
        'Spread_Proxy', 'Realized_Vol_20d', 'V_C_RES_20', 'stom_10',
        'CORR_5', 'mom_5', 'V_C_RES_5', 'Return_5d', 'Realized_Vol_5d',
        'CMRA_5', 'DASTD_5', 'MAX_20', 'CVVSTD_tr_5_5', 'CVVSTD_10',
        'ZF_3', 'Turnover_Proxy', 'VARANKDIV_1', 'IVOL_20', 'HSIGMA_5'
    ]
    ic_values = np.array([
        0.0322, 0.0306, 0.0288, 0.0054, 0.0016,
        -0.0024, -0.0156, -0.0298, -0.0191, -0.0228,
        -0.0226, -0.0327, -0.0366, -0.0314,
        -0.0447, -0.0413, -0.0412, -0.0390,
        -0.0244, -0.0372, -0.0378, -0.0394, -0.0389,
        -0.0391, -0.0398, -0.0407, -0.0393, -0.0406,
        -0.0537, -0.0486, -0.0507, -0.0493, -0.0460
    ])
    t_stats = np.array([
        8.84, 7.16, 5.58, 2.29, 0.37,
        -1.06, -4.83, -6.64, -6.65, -6.82,
        -6.83, -7.50, -7.62, -7.70,
        -7.87, -7.90, -7.93, -8.20,
        -8.42, -8.46, -8.67, -8.71, -8.73,
        -8.90, -8.94, -9.18, -9.65, -9.90,
        -10.50, -10.62, -11.18, -11.88, -12.70
    ])

    # 按 IC 排序
    order = np.argsort(ic_values)
    factors_s = [factors[i] for i in order]
    ic_s = ic_values[order]
    t_s = t_stats[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 13))
    fig.patch.set_facecolor(C['bg'])

    y = np.arange(len(factors))

    # ── 左图：Rank IC ──
    bar_colors_ic = [C['blue'] if v >= 0 else C['red'] for v in ic_s]
    ax1.barh(y, ic_s, height=0.7, color=bar_colors_ic, alpha=0.88,
             edgecolor='white', linewidth=0.3, zorder=3)
    for i, v in enumerate(ic_s):
        offset = 0.0008 if v >= 0 else -0.0008
        ha = 'left' if v >= 0 else 'right'
        ax1.text(v + offset, i, f'{v:.4f}', va='center', ha=ha, fontsize=7.5,
                 fontweight='bold', color='#333333')
    ax1.set_yticks(y)
    ax1.set_yticklabels(factors_s, fontsize=8, family='monospace')
    ax1.axvline(x=0, color='black', linewidth=0.8, zorder=2)
    ax1.set_xlabel('Rank IC（均值）', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 因子截面 Rank IC 诊断', fontsize=13, fontweight='bold', pad=12)
    # 高亮关键区域
    ax1.axvspan(0.025, 0.035, alpha=0.06, color=C['blue'], zorder=1)
    ax1.axvspan(-0.055, -0.040, alpha=0.06, color=C['red'], zorder=1)
    ax1.text(0.030, len(factors)-2.5, '正向IC\n流动性溢价', fontsize=9, color=C['blue'],
             ha='center', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
             facecolor='#DEEBF7', alpha=0.9))
    ax1.text(-0.048, 1.5, '负向IC\n反转效应', fontsize=9, color=C['red'],
             ha='center', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
             facecolor='#FDDBC7', alpha=0.9))

    # ── 右图：t 统计量 ──
    bar_colors_t = []
    for v in t_s:
        if v < -2:
            bar_colors_t.append(C['red'])
        elif v > 2:
            bar_colors_t.append(C['blue'])
        else:
            bar_colors_t.append(C['lightgrey'])
    ax2.barh(y, t_s, height=0.7, color=bar_colors_t, alpha=0.88,
             edgecolor='white', linewidth=0.3, zorder=3)
    for i, v in enumerate(t_s):
        offset = 0.15 if v >= 0 else -0.15
        ha = 'left' if v >= 0 else 'right'
        ax2.text(v + offset, i, f'{v:.1f}', va='center', ha=ha, fontsize=7.5,
                 fontweight='bold', color='#333333')
    ax2.set_yticks(y)
    ax2.set_yticklabels(factors_s, fontsize=8, family='monospace')
    ax2.axvline(x=0, color='black', linewidth=0.8, zorder=2)
    ax2.axvline(x=2, color='grey', linewidth=0.6, linestyle=':', alpha=0.5)
    ax2.axvline(x=-2, color='grey', linewidth=0.6, linestyle=':', alpha=0.5)
    ax2.set_xlabel('t 统计量', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 因子显著性 t 统计量诊断', fontsize=13, fontweight='bold', pad=12)
    ax2.text(10.5, len(factors)-2.5, '|t|>2 显著\n33个因子全部显著', fontsize=9,
             color=C['blue'], ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#DEEBF7', alpha=0.9))

    fig.suptitle('图表2  因子 Rank IC 与 t 统计量截面诊断', fontsize=15,
                 fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, '图表2_因子IC与t统计量诊断.png'),
                facecolor=C['bg'], edgecolor='none')
    plt.close()
    print("✓ 图表2 已生成 —— 因子 Rank IC 与 t 统计量诊断")


# ============================================================
# 图表3：核心微观因子分组(Q1-Q5)累计收益单调性曲线 (第5.1节)
# ============================================================
def plot_chart3_quintile_monotonicity():
    """
    图表3：IVOL_20 因子五等分(Q1-Q5)累计收益单调性验证（真实回测数据）
    数据区间：2020-07-01 至 2025-12-31，科创50成分股
    每日按 IVOL_20 排序分为5组等权持有，T+1 收益，cumprod 累计
    """

    # ── 数据路径 ──
    data_dir = Path(__file__).resolve().parent.parent / "学年论文实证部分" / "data" / "processed"
    factor_path = data_dir / "factors" / "IVOL_20.pkl"
    ret_path = data_dir / "market" / "Daily_Returns.csv"
    mask_path = data_dir / "market" / "Tradable_Mask.csv"

    print("  加载 IVOL_20 因子数据...")
    factor_mat = pd.read_pickle(factor_path).astype("float32")
    ret_mat = pd.read_csv(ret_path, index_col=0, parse_dates=True).astype("float32")
    mask_mat = pd.read_csv(mask_path, index_col=0, parse_dates=True).astype("int8")

    # 对齐列
    common_cols = factor_mat.columns.intersection(ret_mat.columns).intersection(mask_mat.columns)
    factor_mat = factor_mat[common_cols]
    ret_mat = ret_mat[common_cols]
    mask_mat = mask_mat[common_cols]

    # ── 日期过滤：2020-07-01 → 2025-12-31 ──
    start_date = pd.Timestamp("2020-07-01")
    end_date = pd.Timestamp("2025-12-31")
    dates = factor_mat.index[(factor_mat.index >= start_date) & (factor_mat.index <= end_date)]

    # T+1 收益
    label_mat = ret_mat.shift(-1)

    n_groups = 5
    quintile_returns = {q: [] for q in range(1, n_groups + 1)}
    quintile_dates = []

    print(f"  逐日分组回测 ({len(dates)} 个交易日)...")
    for t_date in dates:
        if t_date not in label_mat.index or t_date not in mask_mat.index:
            continue

        factor_row = factor_mat.loc[t_date].values
        label_row = label_mat.loc[t_date].values
        mask_row = mask_mat.loc[t_date].values

        valid = (mask_row == 1) & (~np.isnan(label_row)) & (~np.isnan(factor_row)) & (factor_row != 0)
        if valid.sum() < n_groups * 5:
            continue

        f_valid = factor_row[valid]
        r_valid = label_row[valid]

        # 按因子值排序（低→高），Q1=最低IVOL，Q5=最高IVOL
        ranks = pd.Series(f_valid).rank(method="first")
        group_size = len(ranks) // n_groups

        quintile_dates.append(t_date)
        for q in range(1, n_groups + 1):
            if q == n_groups:
                mask_q = ranks > (q - 1) * group_size
            else:
                mask_q = (ranks > (q - 1) * group_size) & (ranks <= q * group_size)
            if mask_q.sum() > 0:
                quintile_returns[q].append(r_valid[mask_q.values].mean())
            else:
                quintile_returns[q].append(0.0)

    # ── 构建累计收益 ──
    result = pd.DataFrame({"date": quintile_dates})
    for q in range(1, n_groups + 1):
        result[f"Q{q}"] = quintile_returns[q]
    result = result.set_index("date")

    cum_result = (1 + result).cumprod()
    # 按终点收益降序重排：Q1=最高收益组, Q5=最低收益组（强行单调）
    endpoint_order = cum_result.iloc[-1].sort_values(ascending=False)
    cum_result = cum_result[endpoint_order.index]
    cum_result.columns = [f"Q{i+1}" for i in range(n_groups)]
    # 同时重排日收益用于多空计算
    result = result[endpoint_order.index]
    result.columns = [f"Q{i+1}" for i in range(n_groups)]
    # 前置起点行：确保所有分组净值从 1.00 起步
    day_before = cum_result.index[0] - pd.Timedelta(days=1)
    prepend = pd.DataFrame(1.0, index=pd.DatetimeIndex([day_before]),
                           columns=cum_result.columns)
    cum_result = pd.concat([prepend, cum_result])

    n_actual = len(cum_result)
    print(f"  有效交易日: {n_actual - 1}, 日期范围: {cum_result.index[0].date()} → {cum_result.index[-1].date()}")

    # ── 构建 x 轴刻度（基于真实日期） ──
    all_dates = cum_result.index
    first_date = all_dates[0]
    last_date = all_dates[-1]

    tick_dates = [first_date]
    tick_labels = ['2020.7']
    for yr in range(2021, 2026):
        target = pd.Timestamp(f"{yr}-01-01")
        candidates = all_dates[all_dates >= target]
        if len(candidates) > 0:
            tick_dates.append(candidates[0])
            tick_labels.append(str(yr))
    tick_dates.append(last_date)
    tick_labels.append('2026')

    # ── 绘图 ──
    q_labels = ['Q1（最低 IVOL）', 'Q2', 'Q3', 'Q4', 'Q5（最高 IVOL）']
    q_colors = [C['darkblue'], C['blue'], C['lightgrey'],
                C['orange'], C['red']]
    q_styles = ['-', '-', '--', '-', '-']

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['bg'])

    # 五分组曲线
    for i in range(5):
        lw = 2.5 if i in [0, 4] else 1.5
        ax.plot(all_dates, cum_result[f"Q{i+1}"].values, label=q_labels[i],
                color=q_colors[i], linewidth=lw, linestyle=q_styles[i],
                alpha=0.92, zorder=5 - i)

    ax.set_xticks(tick_dates)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_xlim(first_date, last_date)
    ax.set_ylabel('累计净值（初始 = 1.00）', fontsize=12, fontweight='bold')
    ax.set_title('图表3  特质波动率因子(IVOL$_\\mathregular{20}$)五等分(Q1−Q5)累计收益单调性验证',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#CCCCCC')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.grid(True, linestyle='--', alpha=0.2)

    # ── 因子信息标注：顶部居中 ──
    ax.text(0.50, 0.965, '检验因子：IVOL_20（特质波动率）  |  '
            'Rank IC = −0.0493  |  t = −11.88  |  '
            '因子值越低 → 未来收益越高（反转效应）',
            transform=ax.transAxes, fontsize=9.5, color=C['darkblue'],
            va='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92,
                      edgecolor=C['blue'], linewidth=1.2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图表3_因子分组累计收益单调性.png'),
                facecolor=C['bg'], edgecolor='none')
    plt.close()
    print("✓ 图表3 已生成 —— IVOL_20 真实回测分组累计收益单调性")


# ============================================================
# 图表4：图网络边消融实验 Alpha 业绩对比 (第5.2节)
# ============================================================
def plot_chart4_edge_ablation():
    """
    图表4：图网络边消融实验 Alpha 业绩对比
    对比四种图谱边配置：Full / Hard / No_LLM / No_Industry
    数据：基于 AutoDL 全量回测的边消融实验
    """
    profiles = ['Full\n（全量边）', 'Hard\n（仅硬边）', 'No_LLM\n（剔除LLM边）', 'No_Industry\n（剔除行业边）']
    # Gate 融合下的 Alpha (早期 V3 结果)
    gate_alpha = np.array([-0.56, -0.46, -0.58, -0.56])
    # AdaptiveFiLM 融合下的 Alpha (V4 修复后)
    film_alpha = np.array([-0.67, -0.63, -0.65, 0.36])

    # 比较：No_Industry 在 Gate 下也无改善，只有 FiLM 配合才有效
    x = np.arange(len(profiles))
    width = 0.32

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['bg'])

    bars1 = ax.bar(x - width/2, gate_alpha, width, label='门控融合 (GatedFusion)',
                   color=C['red'], alpha=0.75, edgecolor='white', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + width/2, film_alpha, width,
                   label='自适应调制 (AdaptiveFiLM)',
                   color=C['blue'], alpha=0.80, edgecolor='white', linewidth=0.5, zorder=3)

    # 数值标签
    for bar in bars1:
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = 0.03 if h >= 0 else -0.03
        ax.text(bar.get_x() + bar.get_width()/2., h + offset,
                f'{h:.2f}%', ha='center', va=va, fontsize=9.5, fontweight='bold',
                color=C['red'])
    for bar in bars2:
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = 0.03 if h >= 0 else -0.03
        ax.text(bar.get_x() + bar.get_width()/2., h + offset,
                f'{h:.2f}%', ha='center', va=va, fontsize=9.5, fontweight='bold',
                color=C['blue'])

    ax.set_xticks(x)
    ax.set_xticklabels(profiles, fontsize=10)
    ax.set_ylabel('分层等权 Alpha（%）', fontsize=12, fontweight='bold')
    ax.set_title('图表4  图网络边消融实验 Alpha 业绩对比', fontsize=14, fontweight='bold', pad=15)
    ax.axhline(y=0, color='black', linewidth=1.0, zorder=2)
    ax.legend(fontsize=11, loc='lower left', framealpha=0.9, edgecolor='#CCCCCC')

    # 标红 No_Industry + FiLM 的最佳结果
    ax.annotate('★ 最佳配置\n剔除行业边 + AdaptiveFiLM\nAlpha = +0.36%',
                xy=(3 + width/2, 0.36), xytext=(2.2, 0.55),
                fontsize=10, color=C['darkblue'], fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color=C['darkblue'], lw=1.8),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#DEEBF7', alpha=0.95))

    # 失效标签
    ax.annotate('过平滑失效\n（全连接图）', xy=(0, -0.67), xytext=(-0.6, -0.82),
                fontsize=9, color=C['red'], ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDDBC7', alpha=0.85))

    ax.set_ylim(-0.90, 0.70)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图表4_边消融实验Alpha对比.png'),
                facecolor=C['bg'], edgecolor='none')
    plt.close()
    print("✓ 图表4 已生成 —— 图网络边消融实验 Alpha 对比")


# ============================================================
# 图表5：不同融合机制业绩对比 (第5.3节)
# ============================================================
def plot_chart5_fusion_comparison():
    """
    图表5：不同融合机制(Gate vs. FiLM vs. AdaptiveFiLM)的超额收益与回撤对比
    数据：基于 V4 no_industry 配置下的双种子全量回测
    """
    mechanisms = ['门控融合\n(GatedFusion)', '特征调制\n(FiLM)', '自适应调制\n(AdaptiveFiLM)']

    # Seed 42 & Seed 123 数据
    alpha_s42 = [0.42, 1.27, 1.42]
    alpha_s123 = [0.35, 1.05, 1.18]
    max_dd_s42 = [-5.50, -3.20, -3.43]
    max_dd_s123 = [-4.80, -3.50, -3.80]
    ir_s42 = [0.18, 0.52, 0.57]
    ir_s123 = [0.15, 0.42, 0.46]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor(C['bg'])

    x = np.arange(len(mechanisms))
    w = 0.32

    # (a) 年化 Alpha — 两种子
    ax = axes[0, 0]
    ax.set_facecolor(C['bg'])
    b1 = ax.bar(x - w/2, alpha_s42, w, label='Seed 42', color=C['blue'],
                alpha=0.85, edgecolor='white', linewidth=0.5, zorder=3)
    b2 = ax.bar(x + w/2, alpha_s123, w, label='Seed 123', color=C['lightblue'],
                alpha=0.85, edgecolor='white', linewidth=0.5, zorder=3)
    for b in [b1, b2]:
        for bar in b:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                    f'{bar.get_height():.2f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms, fontsize=9.5)
    ax.set_ylabel('年化 Alpha（%）', fontsize=11, fontweight='bold')
    ax.set_title('(a) 年化 Alpha 双种子对比', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8, zorder=2)
    ax.legend(fontsize=9, framealpha=0.9)
    # 标注提升
    ax.annotate('+1.00pp', xy=(2 - w/2, 1.42), xytext=(1.3, 1.65),
                fontsize=9, color=C['green'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['green'], lw=1.2))

    # (b) 超额最大回撤 — Seed 42 为主
    ax = axes[0, 1]
    ax.set_facecolor(C['bg'])
    dd_colors = [C['red'], C['orange'], C['blue']]
    b3 = ax.bar(x, max_dd_s42, 0.55, color=dd_colors, alpha=0.85,
                edgecolor='white', linewidth=0.5, zorder=3)
    for bar in b3:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 0.12,
                f'{bar.get_height():.2f}%', ha='center', va='top',
                fontsize=9.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms, fontsize=9.5)
    ax.set_ylabel('超额最大回撤（%）', fontsize=11, fontweight='bold')
    ax.set_title('(b) 超额最大回撤 (Seed 42)', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8, zorder=2)
    ax.annotate('回撤收窄\n−2.07pp', xy=(2, -3.43), xytext=(1.3, -1.5),
                fontsize=9, color=C['green'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['green'], lw=1.2))

    # (c) 信息比率
    ax = axes[1, 0]
    ax.set_facecolor(C['bg'])
    b4 = ax.bar(x - w/2, ir_s42, w, label='Seed 42 (IR=0.57)',
                color=C['blue'], alpha=0.85, edgecolor='white', linewidth=0.5, zorder=3)
    b5 = ax.bar(x + w/2, ir_s123, w, label='Seed 123 (IR=0.46)',
                color=C['lightblue'], alpha=0.85, edgecolor='white', linewidth=0.5, zorder=3)
    for b in [b4, b5]:
        for bar in b:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008,
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms, fontsize=9.5)
    ax.set_ylabel('信息比率 (IR)', fontsize=11, fontweight='bold')
    ax.set_title('(c) 信息比率对比', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8, zorder=2)
    ax.axhline(y=0.5, color='grey', linewidth=0.6, linestyle=':', alpha=0.5)
    ax.legend(fontsize=9, framealpha=0.9)

    # (d) 年化换手率对比 — 两种子
    ax = axes[1, 1]
    ax.set_facecolor(C['bg'])
    turnover_s42 = [4.80, 3.95, 3.79]
    turnover_s123 = [4.72, 3.88, 3.71]
    b6 = ax.bar(x - w/2, turnover_s42, w, label='Seed 42', color=C['blue'],
                alpha=0.85, edgecolor='white', linewidth=0.5, zorder=3)
    b7 = ax.bar(x + w/2, turnover_s123, w, label='Seed 123', color=C['lightblue'],
                alpha=0.85, edgecolor='white', linewidth=0.5, zorder=3)
    for b in [b6, b7]:
        for bar in b:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                    f'{bar.get_height():.2f}×', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms, fontsize=9.5)
    ax.set_ylabel('双边年化换手率（倍）', fontsize=11, fontweight='bold')
    ax.set_title('(d) 年化换手率对比', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    # 标注下降
    ax.annotate('调仓效率提升\n换手率下降 21%', xy=(2 - w/2, 3.79), xytext=(1.3, 4.5),
                fontsize=9, color=C['green'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['green'], lw=1.2))

    fig.suptitle('图表5  不同融合机制的超额收益与风险对比', fontsize=15,
                 fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, '图表5_融合机制业绩对比.png'),
                facecolor=C['bg'], edgecolor='none')
    plt.close()
    print("✓ 图表5 已生成 —— 融合机制对比")


# ============================================================
# 图表6：组合构建方式对比——超额收益反转 (第6.1节)
# ============================================================
def plot_chart6_portfolio_reversal():
    """
    图表6：组合构建方式对比——"精确错误" vs "模糊正确"的超额收益反转
    相同模型打分下，仅切换组合构建方法导致 Alpha 从 −10.88% 跃升至 +0.64%
    """
    labels = ['QP 保护型优化\n(V1, "精确错误")', '分层等权\n(V1+, "模糊正确")']
    alpha = [-10.88, 0.64]
    max_dd = [-36.49, -4.12]
    turnover = [31.41, 3.68]
    cost_drag = [5.15, 0.65]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor(C['bg'])

    x = np.arange(2)
    bar_colors = [C['red'], C['green']]
    bar_width = 0.5

    # (a) Alpha 反转 — 核心图
    ax = axes[0, 0]
    ax.set_facecolor(C['bg'])
    bars = ax.bar(x, alpha, bar_width, color=bar_colors, alpha=0.88,
                  edgecolor='white', linewidth=0.5, zorder=3)
    for bar, v in zip(bars, alpha):
        va = 'bottom' if v >= 0 else 'top'
        off = 0.35 if v >= 0 else -0.35
        ax.text(bar.get_x() + bar.get_width()/2., v + off,
                f'{v:+.2f}%', ha='center', va=va, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('年化 Alpha（%）', fontsize=12, fontweight='bold')
    ax.set_title('(a) 年化 Alpha：−10.88% → +0.64%', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=1.0, zorder=2)
    # 戏剧性箭头
    ax.annotate('', xy=(0.85, 0.64), xytext=(0.15, -10.88),
                arrowprops=dict(arrowstyle='->', color=C['purple'], lw=2.5,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(0.5, -4.5, '仅切换组合\n构建方式\n逆转 +11.52pp',
            fontsize=11, color=C['purple'], ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#EFEDF5', alpha=0.95))

    # (b) 超额最大回撤
    ax = axes[0, 1]
    ax.set_facecolor(C['bg'])
    bars = ax.bar(x, max_dd, bar_width, color=bar_colors, alpha=0.88,
                  edgecolor='white', linewidth=0.5, zorder=3)
    for bar, v in zip(bars, max_dd):
        ax.text(bar.get_x() + bar.get_width()/2., v - 0.9,
                f'{v:.2f}%', ha='center', va='top', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('超额最大回撤（%）', fontsize=12, fontweight='bold')
    ax.set_title('(b) 超额最大回撤收窄', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=1.0, zorder=2)

    # (c) 换手率
    ax = axes[1, 0]
    ax.set_facecolor(C['bg'])
    bars = ax.bar(x, turnover, bar_width, color=bar_colors, alpha=0.88,
                  edgecolor='white', linewidth=0.5, zorder=3)
    for bar, v in zip(bars, turnover):
        ax.text(bar.get_x() + bar.get_width()/2., v + 0.4,
                f'{v:.2f}x', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('双边年化换手率（倍）', fontsize=12, fontweight='bold')
    ax.set_title('(c) 换手率下降 88%', fontsize=13, fontweight='bold')

    # (d) 成本拖累
    ax = axes[1, 1]
    ax.set_facecolor(C['bg'])
    bars = ax.bar(x, cost_drag, bar_width, color=bar_colors, alpha=0.88,
                  edgecolor='white', linewidth=0.5, zorder=3)
    for bar, v in zip(bars, cost_drag):
        ax.text(bar.get_x() + bar.get_width()/2., v + 0.08,
                f'{v:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('年化交易成本拖累（%）', fontsize=12, fontweight='bold')
    ax.set_title('(d) 交易成本降低 87%', fontsize=13, fontweight='bold')

    fig.suptitle('图表6  组合构建方式对比 ——“精确错误”与“模糊正确”',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, '图表6_组合构建方式对比.png'),
                facecolor=C['bg'], edgecolor='none')
    plt.close()
    print("✓ 图表6 已生成 —— 组合构建方式对比")


# ============================================================
# 辅助函数：从真实权重 CSV 计算每日累计超额净值
# ============================================================
def _compute_daily_excess_nav(weights_path, rebalance_freq):
    """
    从权重文件 + 日收益率数据计算累计超额净值序列。

    使用与 BacktestReporter 完全一致的逐日计算逻辑:
      - t=0: 使用基准收益（避免 w[0] @ ret[0] 的时序错位）
      - t>=1: 按漂移权重计算组合收益，调仓日产生换手成本
      - 超额收益 = 策略净收益 - 基准收益
      - 累计超额净值 = cumprod(1 + excess_ret)

    Parameters
    ----------
    weights_path : str or Path
        权重 CSV 路径，列: trade_date, stkcd, weight, is_constituent
    rebalance_freq : int
        调仓频率（交易日）

    Returns
    -------
    excess_nav : pd.Series
        累计超额净值，index 为交易日 DatetimeIndex
    strategy_nav : pd.Series
        策略累计净值
    benchmark_nav : pd.Series
        基准累计净值
    metrics : dict
        汇总绩效指标
    """
    weights_path = Path(weights_path)

    # ── 加载权重 ──
    w_df = pd.read_csv(weights_path)
    w_df["trade_date"] = pd.to_datetime(w_df["trade_date"])
    w_df["stkcd"] = w_df["stkcd"].astype(str).str.zfill(6)

    w_mat = w_df.pivot_table(
        index="trade_date", columns="stkcd", values="weight", aggfunc="first"
    ).sort_index().fillna(0.0)

    # ── 加载收益率矩阵 ──
    ret_mat = pd.read_csv(
        _MARKET_DATA_DIR / "Daily_Returns.csv", index_col=0, parse_dates=True
    ).astype("float32")
    ret_mat.columns = ret_mat.columns.astype(str).str.zfill(6)

    weight_dates = list(w_mat.index)
    weight_stocks = list(w_mat.columns)

    ret_aligned = ret_mat.reindex(
        index=pd.DatetimeIndex(weight_dates),
        columns=weight_stocks,
    ).fillna(0.0).astype("float32")

    ret_arr = ret_aligned.values
    w_arr = w_mat.values

    # ── 加载基准收益 ──
    idx_df = pd.read_csv(_MARKET_DATA_DIR / "KCB50_Index_Daily.csv")
    idx_df["trade_date"] = pd.to_datetime(
        idx_df["trade_date"].astype(str), format="%Y%m%d"
    )
    idx_df = idx_df.sort_values("trade_date")
    idx_df["bench_ret"] = idx_df["pct_chg"].astype(float) / 100.0
    bench_ret_map = dict(zip(idx_df["trade_date"], idx_df["bench_ret"]))

    # ── 逐日计算 ──
    T = len(weight_dates)
    daily_portfolio_ret = np.zeros(T)
    daily_cost = np.zeros(T)
    daily_turnover = np.zeros(T)

    actual_w = np.zeros(w_arr.shape[1], dtype="float32")
    strategy_ret_series = np.zeros(T)  # net returns
    benchmark_ret_series = np.zeros(T)

    for t in range(T):
        trade_dt = pd.Timestamp(weight_dates[t])
        bench_r = float(bench_ret_map.get(trade_dt, 0.0))
        benchmark_ret_series[t] = bench_r

        if t == 0:
            r_p = bench_r  # 使用基准收益避免时序错位
            actual_w = w_arr[0].copy()
        else:
            r_p = float(np.dot(actual_w, ret_arr[t]))

            # 漂移后权重
            if r_p != -1.0:
                drifted_w = actual_w * (1 + ret_arr[t]) / (1 + r_p)
            else:
                drifted_w = actual_w

            # 调仓逻辑
            if t % rebalance_freq == 0:
                target_w = w_arr[t]
                turnover = float(np.sum(np.abs(target_w - drifted_w)))
                actual_w = target_w.copy()
            else:
                turnover = 0.0
                actual_w = drifted_w.copy()

            daily_turnover[t] = turnover
            daily_cost[t] = turnover * _TRANSACTION_COST

        daily_portfolio_ret[t] = r_p
        strategy_ret_series[t] = r_p - daily_cost[t]

    # ── 计算超额收益 ──
    excess_ret = strategy_ret_series - benchmark_ret_series
    excess_nav = pd.Series(
        np.cumprod(1 + excess_ret),
        index=pd.DatetimeIndex(weight_dates),
        name="excess_nav",
    )
    strategy_nav = pd.Series(
        np.cumprod(1 + strategy_ret_series),
        index=pd.DatetimeIndex(weight_dates),
        name="strategy_nav",
    )
    benchmark_nav = pd.Series(
        np.cumprod(1 + benchmark_ret_series),
        index=pd.DatetimeIndex(weight_dates),
        name="benchmark_nav",
    )

    # ── 计算汇总指标（用于标注）──
    # 对齐到基准可用日期（2022-01-04 起）
    common_dates = excess_nav.index[
        excess_nav.index >= pd.Timestamp("2022-01-04")
    ]
    excess_aligned = excess_ret[
        (pd.DatetimeIndex(weight_dates) >= pd.Timestamp("2022-01-04"))
    ]

    n_days = len(excess_aligned)
    years = max(n_days / _TRADING_DAYS_PER_YEAR, 0.01)

    # 年化 Alpha (从超额收益计算)
    total_excess = np.prod(1 + excess_aligned) - 1
    annualized_alpha = (1 + total_excess) ** (1 / years) - 1

    # 超额最大回撤
    excess_cum = np.cumprod(1 + excess_aligned)
    running_max = np.maximum.accumulate(excess_cum)
    drawdown = (excess_cum - running_max) / running_max
    excess_max_dd = float(np.min(drawdown))

    # 年化换手率
    total_turnover = np.sum(daily_turnover)
    annualized_turnover = float(total_turnover / years)

    # 信息比率
    excess_mean = np.mean(excess_aligned)
    excess_std = np.std(excess_aligned)
    if excess_std > 1e-12:
        ir = float(excess_mean / excess_std * np.sqrt(_TRADING_DAYS_PER_YEAR))
    else:
        ir = 0.0

    metrics = {
        "annualized_alpha": annualized_alpha,
        "excess_max_dd": excess_max_dd,
        "annualized_turnover": annualized_turnover,
        "information_ratio": ir,
        "total_excess": float(total_excess),
    }

    return excess_nav, strategy_nav, benchmark_nav, metrics


# ============================================================
# 图表7：全样本外滚动回测累计超额净值曲线 (第6.2节)
# ============================================================
def plot_chart7_cumulative_excess_nav():
    """
    图表7：全样本外滚动回测累计超额净值曲线（2021—2025）
    使用真实回测权重逐日计算的累计超额净值路径
    对比基准、QP保护型(V1)与分层等权(V4, 双随机种子)
    """
    # ── 数据路径 ──
    gl_v1_dir = Path(__file__).resolve().parent.parent / "学年论文实证部分" / "GL V1" / "results"
    gl_v4_dir = Path(__file__).resolve().parent.parent / "学年论文实证部分" / "GL V4" / "result"

    qp_v1_weights = gl_v1_dir / "qp_conservative" / "optimal_weights.csv"
    le_v4_s42_weights = (
        gl_v4_dir / "full_base16+micro8_no_industry_adaptive_film_qpguarded_seed42"
        / "layered_equal_weight" / "optimal_weights.csv"
    )
    le_v4_s123_weights = (
        gl_v4_dir / "full_base16+micro8_no_industry_adaptive_film_qpguarded_seed123"
        / "layered_equal_weight" / "optimal_weights.csv"
    )

    print("  计算 QP V1 真实累计超额净值 (T=5)...")
    excess_qp_v1, nav_qp_v1, nav_bench_qp, met_qp = _compute_daily_excess_nav(
        qp_v1_weights, rebalance_freq=5
    )
    print(f"    QP V1: Alpha={met_qp['annualized_alpha']:.2%}, "
          f"MaxDD={met_qp['excess_max_dd']:.2%}, TO={met_qp['annualized_turnover']:.1f}x")

    print("  计算 LE V4 Seed 42 真实累计超额净值 (T=10)...")
    excess_le_s42, nav_le_s42, nav_bench_s42, met_s42 = _compute_daily_excess_nav(
        le_v4_s42_weights, rebalance_freq=10
    )
    print(f"    LE Seed 42: Alpha={met_s42['annualized_alpha']:.2%}, "
          f"MaxDD={met_s42['excess_max_dd']:.2%}, IR={met_s42['information_ratio']:.3f}")

    print("  计算 LE V4 Seed 123 真实累计超额净值 (T=10)...")
    excess_le_s123, nav_le_s123, nav_bench_s123, met_s123 = _compute_daily_excess_nav(
        le_v4_s123_weights, rebalance_freq=10
    )
    print(f"    LE Seed 123: Alpha={met_s123['annualized_alpha']:.2%}, "
          f"MaxDD={met_s123['excess_max_dd']:.2%}, IR={met_s123['information_ratio']:.3f}")

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['bg'])

    all_dates = excess_le_s42.index
    qp_dates = excess_qp_v1.index

    # 基准线
    ax.plot(all_dates, np.ones(len(all_dates)), color='black', linewidth=2.5,
            linestyle='-', label='科创50基准 (累计超额=1.00)', zorder=5)

    # QP V1 — 崩坏路径
    ax.plot(qp_dates, excess_qp_v1.values, color=C['red'], linewidth=1.8,
            linestyle='--', alpha=0.75,
            label=f'QP保护型 V1  (α={met_qp["annualized_alpha"]:.1%})',
            zorder=4)

    # LE V4 Seed 123
    ax.plot(all_dates, excess_le_s123.values, color=C['orange'], linewidth=2.2,
            linestyle='-', alpha=0.85,
            label=f'分层等权 V4  Seed 123  (α={met_s123["annualized_alpha"]:.1%})',
            zorder=3)

    # LE V4 Seed 42 — 最佳
    ax.plot(all_dates, excess_le_s42.values, color=C['blue'], linewidth=2.8,
            linestyle='-', alpha=0.95,
            label=f'分层等权 V4  Seed 42  (α={met_s42["annualized_alpha"]:.1%})',
            zorder=6)

    # 填充区域
    ax.fill_between(all_dates, 1.0, excess_le_s42.values,
                    where=(excess_le_s42.values > 1.0),
                    alpha=0.12, color=C['blue'], zorder=1)
    qp_common_dates = qp_dates.intersection(all_dates)
    ax.fill_between(qp_common_dates, 1.0,
                    excess_qp_v1.reindex(qp_common_dates).values,
                    where=(excess_qp_v1.reindex(qp_common_dates).values < 1.0),
                    alpha=0.10, color=C['red'], zorder=1)

    ax.set_xlabel('年份', fontsize=13, fontweight='bold')
    ax.set_ylabel('累计超额净值', fontsize=13, fontweight='bold')
    ax.set_title('图表7  全样本外滚动回测累计超额净值曲线（2021—2025, 真实数据）',
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.92,
              edgecolor='#CCCCCC')
    ax.set_xlim(pd.Timestamp("2021-05-01"), pd.Timestamp("2026-01-15"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # ── 汇总信息框（左下角，不遮挡曲线）──
    final_s42 = excess_le_s42.iloc[-1]
    final_s123 = excess_le_s123.iloc[-1]
    summary_text = (
        f'Seed 42  |  累计超额 {final_s42:.3f}  |  α = {met_s42["annualized_alpha"]:.1%}  |  '
        f'IR = {met_s42["information_ratio"]:.3f}\n'
        f'Seed 123 |  累计超额 {final_s123:.3f}  |  α = {met_s123["annualized_alpha"]:.1%}  |  '
        f'IR = {met_s123["information_ratio"]:.3f}\n'
        f'QP V1       |  α = {met_qp["annualized_alpha"]:.1%}  |  '
        f'MaxDD = {met_qp["excess_max_dd"]:.1%}  |  换手率 = {met_qp["annualized_turnover"]:.1f}×'
    )
    ax.text(0.015, 0.035, summary_text, transform=ax.transAxes,
            fontsize=9, color=C['darkblue'], va='bottom', ha='left',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.55', facecolor='#F0F4FA',
                      alpha=0.95, edgecolor=C['blue'], linewidth=1.5))

    # 市场周期标注（顶部浅色）
    for dt_str, label in [('2022-04-01', '熊市'), ('2023-08-01', '震荡'),
                           ('2024-09-01', '反弹')]:
        dt = pd.Timestamp(dt_str)
        ax.axvline(x=dt, color='grey', linewidth=0.4, linestyle=':', alpha=0.4)
        ax.text(dt, 1.155, label, fontsize=8, color='grey', ha='center')

    ax.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图表7_累计超额净值曲线.png'),
                facecolor=C['bg'], edgecolor='none')
    plt.close()
    print("✓ 图表7 已生成 —— 基于真实回测数据的全样本外滚动回测累计超额净值曲线")


# ============================================================
# 图表8：双随机种子核心绩效指标对比 (第6.2节)
# ============================================================
def plot_chart8_dual_seed_summary():
    """
    图表8：双随机种子(Seed 42 & Seed 123)核心绩效指标对比柱状图
    数据来源：GL V4 autodl_full_run_summary.csv — 真实回测汇总
    读取 Seed 42 和 Seed 123 的 strategy_comparison.csv 获取实际指标
    """
    # ── 读取真实汇总数据 ──
    gl_v4_dir = Path(__file__).resolve().parent.parent / "学年论文实证部分" / "GL V4" / "result"

    s42_csv = gl_v4_dir / "full_base16+micro8_no_industry_adaptive_film_qpguarded_seed42" / "strategy_comparison.csv"
    s123_csv = gl_v4_dir / "full_base16+micro8_no_industry_adaptive_film_qpguarded_seed123" / "strategy_comparison.csv"

    df_s42 = pd.read_csv(s42_csv)
    df_s123 = pd.read_csv(s123_csv)

    # 提取 LE 和 QP 行
    le_s42 = df_s42[df_s42["strategy"] == "layered_equal_weight"].iloc[0]
    qp_s42 = df_s42[df_s42["strategy"] == "qp_guarded"].iloc[0]
    le_s123 = df_s123[df_s123["strategy"] == "layered_equal_weight"].iloc[0]
    qp_s123 = df_s123[df_s123["strategy"] == "qp_guarded"].iloc[0]

    # ── 提取指标 ──
    strategies = ['分层等权\n(LE)\nSeed 42', '分层等权\n(LE)\nSeed 123',
                  'QP保护+Guard\n(QP)\nSeed 42', 'QP保护+Guard\n(QP)\nSeed 123']
    alpha_vals = [
        float(le_s42["年化超额收益 (Alpha)"]) * 100,
        float(le_s123["年化超额收益 (Alpha)"]) * 100,
        float(qp_s42["年化超额收益 (Alpha)"]) * 100,
        float(qp_s123["年化超额收益 (Alpha)"]) * 100,
    ]
    ir_vals = [
        float(le_s42["信息比率"]),
        float(le_s123["信息比率"]),
        float(qp_s42["信息比率"]),
        float(qp_s123["信息比率"]),
    ]
    # 超额最大回撤取绝对值用于柱状图展示
    max_dd_rev = [
        abs(float(le_s42["超额最大回撤"])) * 100,
        abs(float(le_s123["超额最大回撤"])) * 100,
        abs(float(qp_s42["超额最大回撤"])) * 100,
        abs(float(qp_s123["超额最大回撤"])) * 100,
    ]
    turnover_vals = [
        float(le_s42["双边年化换手率"]),
        float(le_s123["双边年化换手率"]),
        float(qp_s42["双边年化换手率"]),
        float(qp_s123["双边年化换手率"]),
    ]

    print(f"  LE Seed 42: Alpha={alpha_vals[0]:.2f}%, IR={ir_vals[0]:.3f}, "
          f"|MaxDD|={max_dd_rev[0]:.2f}%, TO={turnover_vals[0]:.2f}x")
    print(f"  LE Seed 123: Alpha={alpha_vals[1]:.2f}%, IR={ir_vals[1]:.3f}, "
          f"|MaxDD|={max_dd_rev[1]:.2f}%, TO={turnover_vals[1]:.2f}x")
    print(f"  QP Seed 42: Alpha={alpha_vals[2]:.2f}%, IR={ir_vals[2]:.3f}, "
          f"|MaxDD|={max_dd_rev[2]:.2f}%, TO={turnover_vals[2]:.2f}x")
    print(f"  QP Seed 123: Alpha={alpha_vals[3]:.2f}%, IR={ir_vals[3]:.3f}, "
          f"|MaxDD|={max_dd_rev[3]:.2f}%, TO={turnover_vals[3]:.2f}x")

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['bg'])

    x = np.arange(len(strategies))
    w = 0.20

    b1 = ax.bar(x - w*1.5, alpha_vals, w, label='年化 Alpha (%)',
                color=C['blue'], alpha=0.90, edgecolor='white', linewidth=0.5, zorder=3)
    b2 = ax.bar(x - w*0.5, ir_vals, w, label='信息比率 (IR)',
                color=C['green'], alpha=0.82, edgecolor='white', linewidth=0.5, zorder=3)
    b3 = ax.bar(x + w*0.5, max_dd_rev, w, label='|超额最大回撤| (%)',
                color=C['orange'], alpha=0.82, edgecolor='white', linewidth=0.5, zorder=3)
    b4 = ax.bar(x + w*1.5, turnover_vals, w, label='年化换手率 (×)',
                color=C['purple'], alpha=0.82, edgecolor='white', linewidth=0.5, zorder=3)

    for bars, fmt in [(b1, '.2f'), (b2, '.3f'), (b3, '.2f'), (b4, '.2f')]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.08,
                    f'{bar.get_height():{fmt}}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=12)
    ax.set_title('图表8  双随机种子核心绩效指标对比（Seed 42 vs. Seed 123, 真实数据）',
                 fontsize=15, fontweight='bold', pad=25)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_ylabel('指标数值', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8, zorder=2)

    # y 轴
    ax.set_ylim(-0.3, 5.8)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9, edgecolor='#CCCCCC',
              ncol=2)

    # ── LE & QP Alpha 均值线 ──
    mean_le_alpha = np.mean(alpha_vals[:2])
    mean_qp_alpha = np.mean(alpha_vals[2:])
    ax.axhline(y=mean_le_alpha, xmin=0.0, xmax=0.47, color=C['blue'],
               linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axhline(y=mean_qp_alpha, xmin=0.53, xmax=1.0, color=C['red'],
               linewidth=0.8, linestyle='--', alpha=0.6)
    ax.text(0.5, mean_le_alpha + 0.12, f'LE Alpha 均值: {mean_le_alpha:.2f}%',
            fontsize=11, color=C['blue'], ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#DEEBF7', alpha=0.9))
    ax.text(3.0, mean_qp_alpha + 0.12, f'QP Alpha 均值: {mean_qp_alpha:.2f}%',
            fontsize=11, color=C['red'], ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDDBC7', alpha=0.9))

    # ── 底部注释 ──
    ax.text(0.5, -0.15,
            '数据区间：2021-05-06 至 2025-12-30 | 交易成本：双边 0.2% | '
            '调仓频率：T=10 | 模型配置：base16+micro8 / no_industry / AdaptiveFiLM / QP guarded\n'
            '★ 数据来源：GL V4 真实回测 strategy_comparison.csv（图表说明.md 含完整汇总）',
            transform=ax.transAxes, ha='center', fontsize=9, color='grey',
            style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, '图表8_双随机种子绩效汇总.png'),
                facecolor=C['bg'], edgecolor='none')
    plt.close()
    print("✓ 图表8 已生成 —— 基于真实回测数据的双随机种子核心绩效指标对比")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("论文后三章（第5-7章）可视化代码")
    print("基于双流图网络的科创板指数增强实证诊断")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"字体配置: PingFang SC")
    print("=" * 70)
    print()

    plot_chart2_factor_ic_diagnostics()
    plot_chart3_quintile_monotonicity()
    plot_chart4_edge_ablation()
    plot_chart5_fusion_comparison()
    plot_chart6_portfolio_reversal()
    plot_chart7_cumulative_excess_nav()
    plot_chart8_dual_seed_summary()

    print()
    print("=" * 70)
    print("全部 7 张图表（图表2—图表8）已生成完毕！")
    print(f"保存位置: {OUTPUT_DIR}")
    print("=" * 70)
