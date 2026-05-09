"""
Credit Suisse KMV Model Analysis - Greensill Crisis Period
==========================================================
基于 CS_KMV_Monthly_Data.csv 进行 KMV 违约距离和预期违约概率建模
债务数据来源：Credit Suisse 2020-2021 Annual Report (CHF bn → USD bn, ~1.1x)
"""

import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import seaborn as sns

# ============================================================
# macOS 中文字体防乱码 — 稳定解决方案
# ============================================================
# 设置字体 - 使用已验证可用的中文字体（不重建字体管理器，避免缓存问题）
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'PingFang HK', 'Songti SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 打印当前使用的字体确认
print(f"✓ 已设置中文字体: {plt.rcParams['font.sans-serif']}")

# ============================================================
# Step 0: 读取数据 & 补充债务结构
# ============================================================
CSV_PATH = "/Users/huzhichao/Desktop/金融风险管理/数据搜集/CS_KMV_Monthly_Data.csv"
df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Credit Suisse 长短期负债 (USD Billion)
debt_data = {
    "2020-01": (170.0, 260.0),
    "2020-02": (170.0, 260.0),
    "2020-03": (175.0, 262.0),
    "2020-04": (175.0, 262.0),
    "2020-05": (173.0, 263.0),
    "2020-06": (173.0, 263.0),
    "2020-07": (174.0, 264.0),
    "2020-08": (174.0, 264.0),
    "2020-09": (175.0, 265.0),
    "2020-10": (175.0, 265.0),
    "2020-11": (176.0, 266.0),
    "2020-12": (176.0, 266.0),
    "2021-01": (178.0, 262.0),
    "2021-02": (182.0, 255.0),
    "2021-03": (195.0, 248.0),
    "2021-04": (198.0, 245.0),
    "2021-05": (196.0, 243.0),
    "2021-06": (194.0, 241.0),
    "2021-07": (192.0, 240.0),
    "2021-08": (190.0, 239.0),
    "2021-09": (188.0, 238.0),
    "2021-10": (186.0, 237.0),
    "2021-11": (185.0, 236.0),
    "2021-12": (184.0, 235.0),
}

df["Short_Term_Debt_Bn"] = df["Date"].dt.strftime("%Y-%m").map(lambda x: debt_data[x][0])
df["Long_Term_Debt_Bn"]  = df["Date"].dt.strftime("%Y-%m").map(lambda x: debt_data[x][1])
df["Market_Cap_Bn"] = df["Market_Cap_USD_Mn"] / 1000.0
df["DP"] = df["Short_Term_Debt_Bn"] + 0.5 * df["Long_Term_Debt_Bn"]

# ============================================================
# Step 1: KMV 求解
# ============================================================
T = 1.0

def kmv_equations(vars, E_obs, DP, r, sigma_E_obs, T):
    V, sigma_V = vars
    if V <= 0 or sigma_V <= 0:
        return [1e6, 1e6]
    d1 = (np.log(V / DP) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    eq1 = E_obs - (V * norm.cdf(d1) - np.exp(-r * T) * DP * norm.cdf(d2))
    eq2 = sigma_E_obs - (V / E_obs) * norm.cdf(d1) * sigma_V
    return [eq1, eq2]

results_V = []
results_SigmaV = []
converged_flags = []

for idx, row in df.iterrows():
    E = row["Market_Cap_Bn"]
    DP = row["DP"]
    r = row["Risk_Free_Rate"]
    sigma_E = row["Equity_Vol_Annual"]
    V_init = E + DP
    sigma_V_init = sigma_E * E / (E + DP)
    x0 = [V_init, sigma_V_init]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol, info, ier, msg = fsolve(kmv_equations, x0, args=(E, DP, r, sigma_E, T), full_output=True)
    
    V_sol, sigma_V_sol = sol
    converged = (ier == 1)
    
    if not converged or V_sol <= 0:
        for scale in [0.8, 1.2, 0.6, 1.5]:
            x0_alt = [V_init * scale, sigma_V_init * scale]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol2, info2, ier2, msg2 = fsolve(kmv_equations, x0_alt, args=(E, DP, r, sigma_E, T), full_output=True)
            if ier2 == 1 and sol2[0] > 0:
                V_sol, sigma_V_sol = sol2
                converged = True
                break
    
    results_V.append(max(V_sol, 0.01))
    results_SigmaV.append(max(abs(sigma_V_sol), 0.001))
    converged_flags.append(converged)

df["V"] = results_V
df["Sigma_V"] = results_SigmaV
df["Converged"] = converged_flags

# ============================================================
# Step 2: 违约距离 DD 和预期违约概率 EDF
# ============================================================
df["DD"] = (df["V"] - df["DP"]) / (df["V"] * df["Sigma_V"])
df["EDF"] = norm.cdf(-df["DD"])

# 打印关键结果
print("=" * 80)
print("KMV Model Results — Credit Suisse (2020-2021)")
print("=" * 80)
print(f"{'Date':<12} {'E($B)':>8} {'DP($B)':>8} {'V($B)':>9} {'σ_V':>7} {'DD':>8} {'EDF(%)':>8} {'Conv':>5}")
print("-" * 80)
for _, row in df.iterrows():
    print(f"{row['Date'].strftime('%Y-%m'):<12} "
          f"{row['Market_Cap_Bn']:>8.1f} "
          f"{row['DP']:>8.1f} "
          f"{row['V']:>9.1f} "
          f"{row['Sigma_V']:>7.3f} "
          f"{row['DD']:>8.2f} "
          f"{row['EDF']*100:>8.4f} "
          f"{'  ✓' if row['Converged'] else '  ✗':>5}")
print("=" * 80)

pre_crisis = df[df["Date"] == "2020-12-01"].iloc[0]
crisis_peak = df[df["Date"] == "2021-03-01"].iloc[0]
print(f"\n暴雷前 (2020-12): DD = {pre_crisis['DD']:.4f}, EDF = {pre_crisis['EDF']*100:.4f}%")
print(f"暴雷当月 (2021-03): DD = {crisis_peak['DD']:.4f}, EDF = {crisis_peak['EDF']*100:.4f}%")

# ============================================================
# Step 3: 可视化设置
# ============================================================
sns.set_style("whitegrid")

# 重要：seaborn 会重置 rcParams，必须重新设置中文字体
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'PingFang HK', 'Songti SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['figure.facecolor'] = '#FAFBFC'
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['grid.color'] = '#E8E8E8'
plt.rcParams['grid.linewidth'] = 0.5

print(f"✓ 可视化字体已重新设置: {plt.rcParams['font.sans-serif']}")

# 亮色配色方案
COLORS = {
    'primary': '#3498DB',
    'secondary': '#E74C3C',
    'accent1': '#2ECC71',
    'accent2': '#F39C12',
    'accent3': '#9B59B6',
    'warning': '#E67E22',
    'danger': '#C0392B',
    'safe': '#27AE60',
    'dark': '#2C3E50',
    'gray': '#7F8C8D',
}

dates = df["Date"]
crisis_start = pd.Timestamp("2021-01-01")
crisis_end = pd.Timestamp("2021-03-31")

# ============================================================
# 图1: 原始双子图 (DD + EDF) - 保留原版
# ============================================================
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), dpi=300,
                                  gridspec_kw={"height_ratios": [1, 1], "hspace": 0.28})
fig1.patch.set_facecolor("#F8F9FA")

dd_vals = df["DD"]
edf_vals = df["EDF"] * 100

# ---- 上子图：违约距离 DD ----
ax1.set_facecolor("#F8F9FA")
ax1.plot(dates, dd_vals, color=COLORS['primary'], linewidth=2.8, marker="o",
         markersize=5, markerfacecolor='white', markeredgewidth=2,
         label="违约距离 (DD)", zorder=5)

dd_warn = 2.0
ax1.axhline(y=dd_warn, color=COLORS['danger'], linestyle="--", linewidth=2, alpha=0.85, zorder=3)
ax1.fill_between(dates, 0, dd_warn, alpha=0.08, color=COLORS['danger'], zorder=1)
ax1.text(dates.iloc[1], dd_warn + 0.15, f"信用风险警戒线 (DD={dd_warn})",
         fontsize=10, color=COLORS['danger'], fontweight="bold", zorder=6)

ax1.axvspan(crisis_start, crisis_end, alpha=0.15, color=COLORS['accent3'], zorder=1)
ax1.axvline(x=pd.Timestamp("2021-03-01"), color=COLORS['danger'], linestyle="-",
            linewidth=2.5, alpha=0.9, zorder=4)

y_annot1 = ax1.get_ylim()[1] * 0.92 if ax1.get_ylim()[1] > 0 else 4.5
ax1.text(pd.Timestamp("2021-02-01"), y_annot1,
         "Greensill 危机发酵至破产", fontsize=9, color=COLORS['gray'],
         fontstyle="italic", ha="center", zorder=6,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", edgecolor="#CCCCCC", alpha=0.8))

ax1.text(pd.Timestamp("2021-03-01"), dd_vals.min() + 0.2,
         "2021.03\n正式暴雷", fontsize=8, color=COLORS['danger'], fontweight="bold",
         ha="center", va="bottom", zorder=6)

ax1.set_ylabel("违约距离 (DD)", fontsize=12, fontweight="bold")
ax1.set_title("瑞信 (Credit Suisse) 违约距离走势 — KMV 模型", fontsize=14, fontweight="bold", pad=12)
ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax1.set_xlim(dates.min() - pd.Timedelta(days=15), dates.max() + pd.Timedelta(days=15))
ax1.tick_params(axis="x", rotation=35, labelsize=9)
ax1.tick_params(axis="y", labelsize=10)

# ---- 下子图：预期违约概率 EDF ----
ax2.set_facecolor("#F8F9FA")
ax2.plot(dates, edf_vals, color=COLORS['secondary'], linewidth=2.8, marker="s",
         markersize=5, markerfacecolor='white', markeredgewidth=2,
         label="预期违约概率 (EDF)", zorder=5)

ax2.fill_between(dates, 0, edf_vals, alpha=0.25, color=COLORS['secondary'], zorder=2)

ax2.axvspan(crisis_start, crisis_end, alpha=0.15, color=COLORS['accent3'], zorder=1)
ax2.axvline(x=pd.Timestamp("2021-03-01"), color=COLORS['danger'], linestyle="-",
            linewidth=2.5, alpha=0.9, zorder=4)

ax2.text(pd.Timestamp("2021-02-01"), edf_vals.max() * 0.88,
         "Greensill 危机发酵至破产", fontsize=9, color=COLORS['gray'],
         fontstyle="italic", ha="center", zorder=6,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", edgecolor="#CCCCCC", alpha=0.8))

ax2.text(pd.Timestamp("2021-03-01"), edf_vals.max() * 0.6,
         "2021.03\n正式暴雷", fontsize=8, color=COLORS['danger'], fontweight="bold",
         ha="center", va="bottom", zorder=6)

ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax2.set_ylabel("预期违约概率 (EDF)", fontsize=12, fontweight="bold")
ax2.set_xlabel("日期", fontsize=12, fontweight="bold")
ax2.set_title("瑞信 (Credit Suisse) 预期违约概率走势", fontsize=14, fontweight="bold", pad=12)
ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax2.set_xlim(dates.min() - pd.Timedelta(days=15), dates.max() + pd.Timedelta(days=15))
ax2.tick_params(axis="x", rotation=35, labelsize=9)
ax2.tick_params(axis="y", labelsize=10)

fig1.text(0.5, 0.01, "Data: CS_KMV_Monthly_Data.csv | Model: KMV (Merton Extension) | T=1Y",
          ha="center", fontsize=8, color=COLORS['gray'], fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
OUT_PATH_1 = "/Users/huzhichao/Desktop/金融风险管理/KMV建模/CS_KMV_DD_EDF.png"
fig1.savefig(OUT_PATH_1, dpi=300, bbox_inches="tight", facecolor=fig1.get_facecolor())
plt.close(fig1)
print(f"\n✓ 图表1 (DD+EDF) 已保存: {OUT_PATH_1}")

# ============================================================
# 图2: 四合一综合仪表盘
# ============================================================
fig4 = plt.figure(figsize=(16, 14), dpi=150)
fig4.patch.set_facecolor('#FAFBFC')

gs = fig4.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
ax1 = fig4.add_subplot(gs[0, 0])
ax2 = fig4.add_subplot(gs[0, 1])
ax3 = fig4.add_subplot(gs[1, 0])
ax4 = fig4.add_subplot(gs[1, 1])

# ---- 子图1: 股价与市值 ----
ax1.set_facecolor('#FFFFFF')
ln1 = ax1.plot(dates, df["Stock_Price_USD"], color=COLORS['primary'], linewidth=2.5, 
               marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5,
               label='股价 (美元)', zorder=5)
ax1.fill_between(dates, 0, df["Stock_Price_USD"], alpha=0.12, color=COLORS['primary'], zorder=1)
ax1.set_ylabel('股价 (美元)', fontsize=10, fontweight='bold', color=COLORS['primary'])
ax1.tick_params(axis='y', labelcolor=COLORS['primary'], labelsize=8)
ax1.set_xlabel('日期', fontsize=9, fontweight='bold')

ax1_twin = ax1.twinx()
ln2 = ax1_twin.plot(dates, df["Market_Cap_Bn"], color=COLORS['accent2'], linewidth=2.5,
                    marker='s', markersize=4, markerfacecolor='white', markeredgewidth=1.5,
                    label='市值 (十亿美元)', zorder=4)
ax1_twin.fill_between(dates, 0, df["Market_Cap_Bn"], alpha=0.08, color=COLORS['accent2'], zorder=1)
ax1_twin.set_ylabel('市值 (十亿美元)', fontsize=10, fontweight='bold', color=COLORS['accent2'])
ax1_twin.tick_params(axis='y', labelcolor=COLORS['accent2'], labelsize=8)

ax1.axvspan(crisis_start, crisis_end, alpha=0.18, color=COLORS['accent3'], zorder=2)
ax1.axvline(x=pd.Timestamp("2021-03-01"), color=COLORS['danger'], linestyle='-', linewidth=2, zorder=6)

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper right', fontsize=8, framealpha=0.95)
ax1.set_title('股价与市值走势', fontsize=12, fontweight='bold', pad=8)
ax1.set_xlim(dates.min() - pd.Timedelta(days=15), dates.max() + pd.Timedelta(days=15))
ax1.tick_params(axis='x', rotation=30, labelsize=7)

# ---- 子图2: 波动率对比 ----
ax2.set_facecolor('#FFFFFF')
ax2.plot(dates, df["Equity_Vol_Annual"] * 100, color=COLORS['accent1'], linewidth=2.5,
         marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5,
         label='股权波动率 σ_E', zorder=5)
ax2.plot(dates, df["Sigma_V"] * 100, color=COLORS['secondary'], linewidth=2.5,
         marker='D', markersize=4, markerfacecolor='white', markeredgewidth=1.5,
         linestyle='--', label='资产波动率 σ_V', zorder=5)
ax2.fill_between(dates, df["Sigma_V"] * 100, df["Equity_Vol_Annual"] * 100, 
                 alpha=0.12, color=COLORS['warning'], zorder=1)

ax2.axvspan(crisis_start, crisis_end, alpha=0.18, color=COLORS['accent3'], zorder=2)
ax2.axvline(x=pd.Timestamp("2021-03-01"), color=COLORS['danger'], linestyle='-', linewidth=2, zorder=6)

ax2.set_xlabel('日期', fontsize=9, fontweight='bold')
ax2.set_ylabel('波动率 (%)', fontsize=10, fontweight='bold')
ax2.set_title('股权波动率与资产波动率对比', fontsize=12, fontweight='bold', pad=8)
ax2.legend(loc='upper left', fontsize=8, framealpha=0.95)
ax2.set_xlim(dates.min() - pd.Timedelta(days=15), dates.max() + pd.Timedelta(days=15))
ax2.tick_params(axis='x', rotation=30, labelsize=7)
ax2.tick_params(axis='y', labelsize=8)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

# ---- 子图3: 违约距离 DD ----
ax3.set_facecolor('#FFFFFF')
dd_vals = df["DD"]

ax3.plot(dates, dd_vals, color=COLORS['primary'], linewidth=2.5,
         marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5,
         label='违约距离 (DD)', zorder=5)

ax3.fill_between(dates, 3, dd_vals, where=(dd_vals >= 3), 
                 alpha=0.12, color=COLORS['safe'], label='安全区 (DD≥3)', zorder=1)
ax3.fill_between(dates, 2, dd_vals, where=(dd_vals >= 2) & (dd_vals < 3),
                 alpha=0.15, color=COLORS['warning'], label='警戒区 (2≤DD<3)', zorder=1)
ax3.fill_between(dates, 0, dd_vals, where=(dd_vals < 2),
                 alpha=0.2, color=COLORS['danger'], label='危险区 (DD<2)', zorder=1)

dd_warn = 2.0
dd_safe = 3.0
ax3.axhline(y=dd_warn, color=COLORS['danger'], linestyle='--', linewidth=1.8, alpha=0.85, zorder=3)
ax3.axhline(y=dd_safe, color=COLORS['safe'], linestyle='--', linewidth=1.8, alpha=0.7, zorder=3)

ax3.text(dates.iloc[1], dd_warn + 0.1, f'风险警戒线 (DD={dd_warn})',
         fontsize=7, color=COLORS['danger'], fontweight='bold', zorder=6)
ax3.text(dates.iloc[1], dd_safe + 0.1, f'安全基准线 (DD={dd_safe})',
         fontsize=7, color=COLORS['safe'], fontweight='bold', zorder=6)

ax3.axvspan(crisis_start, crisis_end, alpha=0.18, color=COLORS['accent3'], zorder=2)
ax3.axvline(x=pd.Timestamp("2021-03-01"), color=COLORS['danger'], linestyle='-', linewidth=2, zorder=6)

min_dd_idx = dd_vals.idxmin()
ax3.annotate(f'DD最低点\n{dd_vals[min_dd_idx]:.2f}',
             xy=(dates.iloc[min_dd_idx], dd_vals.iloc[min_dd_idx]),
             xytext=(dates.iloc[min_dd_idx] + pd.Timedelta(days=50), dd_vals.iloc[min_dd_idx] + 0.6),
             fontsize=7, color=COLORS['dark'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.2),
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS['gray'], alpha=0.9),
             zorder=7)

ax3.set_xlabel('日期', fontsize=9, fontweight='bold')
ax3.set_ylabel('违约距离 (DD)', fontsize=10, fontweight='bold')
ax3.set_title('违约距离 (DD) 走势', fontsize=12, fontweight='bold', pad=8)
ax3.legend(loc='upper right', fontsize=7, framealpha=0.95, ncol=2)
ax3.set_xlim(dates.min() - pd.Timedelta(days=15), dates.max() + pd.Timedelta(days=15))
ax3.set_ylim(0, dd_vals.max() * 1.15)
ax3.tick_params(axis='x', rotation=30, labelsize=7)
ax3.tick_params(axis='y', labelsize=8)

# ---- 子图4: 预期违约概率 EDF ----
ax4.set_facecolor('#FFFFFF')
edf_vals = df["EDF"] * 100

ax4.plot(dates, edf_vals, color=COLORS['secondary'], linewidth=2.5,
         marker='s', markersize=4, markerfacecolor='white', markeredgewidth=1.5,
         label='预期违约概率 (EDF)', zorder=5)

ax4.fill_between(dates, 0, edf_vals, alpha=0.25, color=COLORS['secondary'], zorder=2)

ax4.axhspan(0, 2, alpha=0.06, color=COLORS['safe'], label='低风险 (EDF<2%)', zorder=1)
ax4.axhspan(2, 5, alpha=0.1, color=COLORS['warning'], label='中等风险 (2%≤EDF<5%)', zorder=1)
ax4.axhspan(5, 100, alpha=0.12, color=COLORS['danger'], label='高风险 (EDF≥5%)', zorder=1)

ax4.axhline(y=2, color=COLORS['warning'], linestyle='--', linewidth=1.2, alpha=0.8, zorder=3)
ax4.axhline(y=5, color=COLORS['danger'], linestyle='--', linewidth=1.2, alpha=0.8, zorder=3)

ax4.axvspan(crisis_start, crisis_end, alpha=0.18, color=COLORS['accent3'], zorder=2)
ax4.axvline(x=pd.Timestamp("2021-03-01"), color=COLORS['danger'], linestyle='-', linewidth=2, zorder=6)

max_edf_idx = edf_vals.idxmax()
ax4.annotate(f'EDF最高点\n{edf_vals.iloc[max_edf_idx]:.2f}%',
             xy=(dates.iloc[max_edf_idx], edf_vals.iloc[max_edf_idx]),
             xytext=(dates.iloc[max_edf_idx] - pd.Timedelta(days=70), edf_vals.iloc[max_edf_idx] + 1.5),
             fontsize=7, color=COLORS['dark'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.2),
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS['gray'], alpha=0.9),
             zorder=7)

ax4.set_xlabel('日期', fontsize=9, fontweight='bold')
ax4.set_ylabel('预期违约概率 (%)', fontsize=10, fontweight='bold')
ax4.set_title('预期违约概率 (EDF) 走势', fontsize=12, fontweight='bold', pad=8)
ax4.legend(loc='upper left', fontsize=7, framealpha=0.95, ncol=2)
ax4.set_xlim(dates.min() - pd.Timedelta(days=15), dates.max() + pd.Timedelta(days=40))
ax4.tick_params(axis='x', rotation=30, labelsize=7)
ax4.tick_params(axis='y', labelsize=8)
ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

# 总标题和水印
fig4.suptitle('瑞信 (Credit Suisse) KMV模型信用风险分析 — Greensill危机期间 (2020-2021)', 
              fontsize=15, fontweight='bold', y=0.98, color=COLORS['dark'])

fig4.text(0.5, 0.01, 
          '数据来源: CS_KMV_Monthly_Data.csv | 模型: KMV (Merton Extension) | 预测期限: T=1年 | 阴影区域: Greensill危机期 (2021年1-3月)',
          ha='center', fontsize=8, color=COLORS['gray'], fontstyle='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

OUT_PATH_4 = "/Users/huzhichao/Desktop/金融风险管理/KMV建模/CS_KMV_Dashboard.png"
fig4.savefig(OUT_PATH_4, dpi=300, bbox_inches="tight", facecolor=fig4.get_facecolor(),
             edgecolor='none', transparent=False)
plt.close(fig4)
print(f"✓ 图表4 (综合仪表盘) 已保存: {OUT_PATH_4}")

print("\n" + "=" * 80)
print("✅ 全部完成！生成的图表文件：")
print(f"   📊 图表1 (DD+EDF双子图): {OUT_PATH_1}")
print(f"   📊 图表2 (综合仪表盘): {OUT_PATH_4}")
print("=" * 80)
