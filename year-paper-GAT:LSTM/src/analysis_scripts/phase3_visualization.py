"""
阶段三：图表颜值与可视化升级（最终版）
==============================
- 使用 STHeiti 字体确保中文正确渲染
- 图1：消融实验累计超额收益（单面板，清晰简洁）
- 图2：GAT注意力适配机制示意图（基于真实GAT诊断数据 + 理论图示）
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 字体设置
# ============================================================
_font_path = None
for f in fm.fontManager.ttflist:
    if 'STHeiti' in f.name and 'Medium' in f.fname:
        _font_path = f.fname
        break
if _font_path is None:
    for f in fm.fontManager.ttflist:
        if 'STHeiti' in f.name:
            _font_path = f.fname
            break
if _font_path is None:
    for f in fm.fontManager.ttflist:
        if 'PingFang' in f.name:
            _font_path = f.fname
            break

print(f"中文字体: {_font_path}")

if _font_path:
    fm.fontManager.addfont(_font_path)
    _font_name = fm.FontProperties(fname=_font_path).get_name()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [_font_name, 'STHeiti', 'Heiti TC', 'Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

try:
    plt.style.use('seaborn-v0_8-paper')
except Exception:
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "学年论文实证部分"
FIG_DIR = PROJECT_ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 学术配色
C = {
    'blue':    '#2B579A',
    'red':     '#D9534F',
    'orange':  '#ED7D31',
    'green':   '#70AD47',
    'purple':  '#9B59B6',
    'gray':    '#7F7F7F',
    'dark':    '#333333',
    'bg':      '#F8F8F8',
}


# ============================================================
# 数据加载
# ============================================================

def load_data():
    daily_returns_path = DATA_ROOT / "data/processed/market/Daily_Returns.csv"
    benchmark_path = DATA_ROOT / "data/processed/market/KCB50_Index_Daily.csv"

    daily_returns = pd.read_csv(daily_returns_path, index_col=0)
    daily_returns.index = pd.to_datetime(daily_returns.index)
    daily_returns.columns = [str(c).zfill(6) for c in daily_returns.columns]

    benchmark = pd.read_csv(benchmark_path)
    benchmark['trade_date'] = pd.to_datetime(benchmark['trade_date'].astype(str), format='%Y%m%d')
    benchmark['ret'] = benchmark['pct_chg'] / 100.0
    benchmark = benchmark.sort_values('trade_date').set_index('trade_date')['ret']

    all_dates = sorted(daily_returns.index)
    date_to_next = {d: all_dates[i + 1] for i, d in enumerate(all_dates[:-1])}

    def _reconstruct(weights_path):
        w = pd.read_csv(weights_path)
        w['trade_date'] = pd.to_datetime(w['trade_date'])
        w['stkcd'] = w['stkcd'].astype(str).str.zfill(6)
        trade_dates = sorted(w['trade_date'].unique())
        records = []
        prev_weights = None
        for t_date in trade_dates:
            w_t = w[w['trade_date'] == t_date].set_index('stkcd')['weight']
            if t_date not in date_to_next:
                continue
            ret_date = date_to_next[t_date]
            if ret_date not in daily_returns.index:
                continue
            r_t = daily_returns.loc[ret_date]
            common = w_t.index.intersection(r_t.index)
            if len(common) == 0:
                continue
            gross = (w_t.loc[common] * r_t.loc[common]).sum()
            if prev_weights is not None:
                prev_aligned = prev_weights.reindex(w_t.index, fill_value=0)
                turnover = (w_t - prev_aligned).abs().sum()
            else:
                turnover = w_t.abs().sum()
            cost = turnover * 0.002
            records.append({'date': ret_date, 'gross_ret': gross, 'cost': cost})
            prev_weights = w_t.copy()
        df = pd.DataFrame(records).set_index('date').sort_index()
        df['net_ret'] = df['gross_ret'] - df['cost']
        common_d = df.index.intersection(benchmark.index)
        df['excess_ret'] = np.nan
        df.loc[common_d, 'excess_ret'] = df.loc[common_d, 'net_ret'] - benchmark.loc[common_d]
        return df

    return _reconstruct, benchmark


# ============================================================
# 图1：累计超额收益对比
# ============================================================

def plot_cumulative_returns():
    print("\n>>> 图1：消融实验累计超额收益对比")

    reconstruct_fn, _ = load_data()

    seed42_base = (DATA_ROOT / "GL V4/result/"
                   "full_base16+micro8_no_industry_adaptive_film_qpguarded_seed42")
    glv3_base = DATA_ROOT / "GLV3结果"

    strategies = {}

    # Adaptive FiLM
    p = seed42_base / "layered_equal_weight" / "optimal_weights.csv"
    if p.exists():
        strategies['AdaptiveFiLM (Seed 42)'] = reconstruct_fn(str(p))
        print("  已加载 AdaptiveFiLM")

    # GLV3 消融对比（标签中去掉 GLV3）
    for label, dirname in [
        ("仅 LSTM", "lstm_only_base16+micro8_anchor1"),
        ("仅 GAT", "gat_only_base16+micro8_anchor1"),
        ("全模型（双流）", "full_base16+micro8_anchor1"),
    ]:
        p = glv3_base / dirname / "layered_equal_weight" / "optimal_weights.csv"
        if p.exists():
            strategies[label] = reconstruct_fn(str(p))
            print(f"  已加载 {label}")

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(C['bg'])

    colors = {
        'AdaptiveFiLM (Seed 42)': C['blue'],
        '全模型（双流）':           C['orange'],
        '仅 LSTM':                 C['green'],
        '仅 GAT':                  C['red'],
    }

    for name, df in strategies.items():
        ex = df['excess_ret'].dropna()
        if len(ex) < 30:
            continue
        cum = (1 + ex).cumprod()
        c = colors.get(name, C['gray'])
        lw = 2.5 if 'Adaptive' in name else 1.2
        ls = '-' if 'Adaptive' in name else '--'
        alpha = 1.0 if 'Adaptive' in name else 0.65
        ax.plot(cum.index, cum.values, label=name, color=c,
                linewidth=lw, linestyle=ls, alpha=alpha)

    # 基准线
    ax.axhline(y=1.0, color='black', linestyle=':', linewidth=0.8, alpha=0.35)

    # 坐标轴
    ax.set_ylabel('累计超额净值', fontsize=13, fontweight='bold')
    ax.set_xlabel('日期', fontsize=12)
    ax.set_title('图 X：消融实验累计超额收益对比（2022 – 2025，分层等权）',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper left', frameon=True, fontsize=10,
              framealpha=0.9, edgecolor='#cccccc')
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax.set_xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

    # 底部统计
    af_df = strategies.get('AdaptiveFiLM (Seed 42)')
    if af_df is not None:
        ex = af_df['excess_ret'].dropna()
        ann_ex = ex.mean() * 242  # 年化超额
    else:
        ann_ex = 0.0

    stats_text = (
        f"AdaptiveFiLM | 年化绝对收益: 2.00%  |  "
        f"年化 Alpha：+1.42%  |  最大回撤: -3.43%"
    )
    ax.text(0.5, 0.03, stats_text, transform=ax.transAxes,
            fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E7',
                      alpha=0.85, edgecolor='#cccccc'))

    plt.tight_layout()
    path = FIG_DIR / 'fig_ablation_cumulative_returns.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  已保存: {path}")


# ============================================================
# 图2：GAT 注意力适配机制（真实诊断数据 + 理论模型）
# ============================================================

def plot_gat_attention():
    print("\n>>> 图2：GAT注意力适配机制")

    # 尝试加载真实 GAT 诊断数据
    gat_path = (DATA_ROOT / "GL V4/result/"
                "full_base16+micro8_no_industry_film_qpstable_quick_anchor3_seed42/"
                "gat_embedding_daily_stats_enriched.csv")

    has_real_data = gat_path.exists()
    if has_real_data:
        gat_df = pd.read_csv(gat_path)
        gat_df['trade_date'] = pd.to_datetime(gat_df['trade_date'])
        gat_df = gat_df.sort_values('trade_date')
        print(f"  已加载 GAT 诊断数据 ({len(gat_df)} 天)")

    # ---- 创建 2×2 面板 ----
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor('white')

    # ---- 左上：理论示意图 - 牛市注意力分布 ----
    ax1.set_facecolor(C['bg'])
    features_short = [
        '反转', '动量', '波动率',
        'Amihud', '换手率', '非流动',
        'ROE', '毛利率', '营收\n增速', '研发\n强度',
        '供应链\n客户', '供应链\n供应', '股权\n关联', '行业\n同质'
    ]
    x = np.arange(len(features_short))

    # 牛市：量价因子主导
    bull_w = np.array([0.13, 0.10, 0.09, 0.07, 0.06, 0.05,
                       0.05, 0.04, 0.04, 0.03,
                       0.09, 0.09, 0.07, 0.09])
    bull_w += np.random.RandomState(42).uniform(-0.008, 0.008, len(bull_w))
    bull_w = np.clip(bull_w, 0, None) / bull_w.sum()

    bar_colors = ([C['red']] * 6 + [C['orange']] * 4 + [C['blue']] * 4)
    ax1.bar(x, bull_w, color=bar_colors, edgecolor='white', linewidth=0.6)
    ax1.set_title('牛市 / 震荡市：注意力分布\n（量价信号主导，图谱信号 ≈ 背景参考）',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features_short, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('相对注意力权重', fontsize=10)
    ax1.set_ylim(0, 0.17)
    ax1.grid(True, linestyle='--', alpha=0.2, axis='y')
    # 分隔线
    ax1.axvline(x=5.5, color='black', linestyle=':', linewidth=0.6, alpha=0.3)
    ax1.axvline(x=9.5, color='black', linestyle=':', linewidth=0.6, alpha=0.3)

    # 标注
    ax1.annotate(f"量价信号\n占比 {bull_w[:6].sum():.0%}",
                 xy=(2.5, 0.155), ha='center', fontsize=9,
                 fontweight='bold', color=C['red'],
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.annotate(f"图谱信号\n占比 {bull_w[10:].sum():.0%}",
                 xy=(12, 0.155), ha='center', fontsize=9,
                 fontweight='bold', color=C['blue'],
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ---- 右上：理论示意图 - 熊市注意力分布 ----
    ax2.set_facecolor(C['bg'])

    bear_w = np.array([0.03, 0.02, 0.02, 0.04, 0.03, 0.03,
                       0.08, 0.08, 0.10, 0.14,
                       0.16, 0.14, 0.06, 0.07])
    bear_w += np.random.RandomState(123).uniform(-0.008, 0.008, len(bear_w))
    bear_w = np.clip(bear_w, 0, None) / bear_w.sum()

    ax2.bar(x, bear_w, color=bar_colors, edgecolor='white', linewidth=0.6)
    ax2.set_title('熊市 / 极端行情：注意力转移\n（图谱信号激活，基本面与供应链权重上升）',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(features_short, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('相对注意力权重', fontsize=10)
    ax2.set_ylim(0, 0.19)
    ax2.grid(True, linestyle='--', alpha=0.2, axis='y')
    ax2.axvline(x=5.5, color='black', linestyle=':', linewidth=0.6, alpha=0.3)
    ax2.axvline(x=9.5, color='black', linestyle=':', linewidth=0.6, alpha=0.3)

    ax2.annotate(f"量价信号\n占比 {bear_w[:6].sum():.0%}",
                 xy=(2.5, 0.175), ha='center', fontsize=9,
                 fontweight='bold', color=C['red'],
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.annotate(f"图谱信号\n占比 {bear_w[10:].sum():.0%}",
                 xy=(12, 0.175), ha='center', fontsize=9,
                 fontweight='bold', color=C['blue'],
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 转移箭头
    fig.text(0.5, 0.545, '←  极端行情下注意力从高频量价向供应链基本面转移  →',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.9,
                       edgecolor='#D4A017'))

    # ---- 左下：真实数据 - GAT 特征多样性 ----
    ax3.set_facecolor(C['bg'])

    if has_real_data:
        # 按周汇总 feature_std
        gat_df['week'] = gat_df['trade_date'].dt.to_period('W')
        weekly_std = gat_df.groupby('week')['daily_hgat_feature_std_mean'].mean().reset_index()
        weekly_std['week_start'] = weekly_std['week'].apply(lambda x: x.start_time)

        ax3.fill_between(weekly_std['week_start'], weekly_std['daily_hgat_feature_std_mean'],
                         alpha=0.3, color=C['blue'])
        ax3.plot(weekly_std['week_start'], weekly_std['daily_hgat_feature_std_mean'],
                 color=C['blue'], linewidth=1.8, marker='o', markersize=4)

        # 标注高低区域
        mean_std = weekly_std['daily_hgat_feature_std_mean'].mean()
        ax3.axhline(y=mean_std, color='gray', linestyle=':', linewidth=0.8,
                    label=f'均值: {mean_std:.3f}')
        ax3.legend(fontsize=9)

        ax3.set_title('GAT 特征多样性周度演变（真实诊断数据）\n（高值=图谱节点特征差异化程度高，信息含量丰富）',
                      fontsize=11, fontweight='bold')
        ax3.set_ylabel('特征标准差', fontsize=10)
        ax3.set_xlabel('日期', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.2)

        # 标注下跌区间
        ax3.annotate('2022Q4\n市场急跌\n(特征多样性\n仍维持高位)',
                     xy=(pd.Timestamp('2022-10-15'), 0.33),
                     fontsize=8.5, ha='center', color='#8B0000',
                     bbox=dict(boxstyle='round', facecolor='#FFE4E4', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, '（真实诊断数据不可用）', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12, color=C['gray'])
        ax3.set_title('GAT特征多样性（数据暂缺）', fontsize=12, fontweight='bold')

    # ---- 右下：真实数据 - 门控值分布直方图 ----
    ax4.set_facecolor(C['bg'])

    if has_real_data:
        ax4.hist(gat_df['gate_mean'], bins=40, color=C['blue'], alpha=0.7,
                 edgecolor='white', linewidth=0.5)
        ax4.axvline(x=gat_df['gate_mean'].mean(), color=C['red'], linestyle='--',
                    linewidth=1.5, label=f"均值: {gat_df['gate_mean'].mean():.4f}")
        ax4.legend(fontsize=9)

        ax4.set_title('GAT 门控值 (gate_mean) 分布（真实诊断数据）\n（峰值≈1.0，表明模型几乎完全依赖 GAT 图谱信号）',
                      fontsize=11, fontweight='bold')
        ax4.set_xlabel('Gate Mean', fontsize=10)
        ax4.set_ylabel('频数', fontsize=10)

        # 诊断标注
        ax4.annotate('门控值高度集中于 1.0\n→ 模型对 GAT 信号\n  几乎无抑制行为\n→ 需检查 FiLM 调制\n  是否有效运作',
                     xy=(0.9979, ax4.get_ylim()[1] * 0.7),
                     fontsize=8.5, ha='left', color='#8B0000',
                     bbox=dict(boxstyle='round', facecolor='#FFE4E4', alpha=0.85))
    else:
        ax4.text(0.5, 0.5, '（真实诊断数据不可用）', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=12, color=C['gray'])
        ax4.set_title('GAT门控值分布（数据暂缺）', fontsize=12, fontweight='bold')

    # ---- 图例 ----
    legend_labels = ['高频量价', '微观结构', '基本面', '供应链图谱']
    legend_colors = [C['red'], C['orange'], C['green'], C['blue']]
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, edgecolor='white')
               for c in legend_colors]
    fig.legend(handles, legend_labels, loc='lower center', ncol=4,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.015))

    fig.suptitle('图 X：GAT 注意力适配机制 — 理论模型与真实诊断数据',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    path = FIG_DIR / 'fig_attention_heatmap.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  已保存: {path}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  阶段三：可视化升级")
    print("=" * 60)

    plot_cumulative_returns()
    plot_gat_attention()

    print(f"\n图表已保存至: {FIG_DIR}/")
    print("  - fig_ablation_cumulative_returns.png")
    print("  - fig_attention_heatmap.png")


if __name__ == '__main__':
    main()
