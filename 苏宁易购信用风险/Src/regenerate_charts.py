"""
重新生成图表 - 使用硬编码数据并修复中文显示
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import os
import shutil

# ============================================================
# Step 0: 清除matplotlib字体缓存（关键步骤！）
# ============================================================
cache_dir = matplotlib.get_cachedir()
print(f"[清理] matplotlib缓存目录: {cache_dir}")
if os.path.exists(cache_dir):
    try:
        # 删除字体缓存文件
        font_cache = os.path.join(cache_dir, 'fontlist-v330.json')
        if os.path.exists(font_cache):
            os.remove(font_cache)
            print(f"[清理] 已删除字体缓存: {font_cache}")
        # 尝试删除整个缓存目录
        for f in os.listdir(cache_dir):
            if 'font' in f.lower():
                fp = os.path.join(cache_dir, f)
                try:
                    os.remove(fp)
                    print(f"[清理] 已删除: {fp}")
                except:
                    pass
    except Exception as e:
        print(f"[警告] 清理缓存时出错: {e}")

# 重建字体管理器
fm._load_fontmanager(try_read_cache=False)

# ============================================================
# Step 1: 设置中文字体
# ============================================================
print("\n[字体] 设置中文字体...")

# macOS系统字体路径
font_paths = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]

for fp in font_paths:
    if os.path.exists(fp):
        try:
            fm.fontManager.addfont(fp)
            print(f"[字体] 已注册: {fp}")
        except Exception as e:
            print(f"[警告] 注册失败 {fp}: {e}")

# 设置rcParams
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'PingFang SC', 'Heiti TC', 'STHeiti', 'Songti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print(f"[字体] 当前设置: {plt.rcParams['font.sans-serif'][:3]}")

# ============================================================
# Step 2: 硬编码数据（来自之前运行的真实输出）
# ============================================================
print("\n[数据] 加载硬编码数据...")

# 从之前成功运行时输出的数据
data_records = [
    {"Date": "2019-12-31", "Close": 10.06, "E": 9.365860e+10, "DP": 1.350000e+11, "V": 2.259854e+11, "Sigma_V": 0.034998, "DD": 11.503859, "EDF": 6.307317e-31, "Sigma_E": 0.084446},
    {"Date": "2020-01-31", "Close": 9.95, "E": 9.263450e+10, "DP": 1.350000e+11, "V": 2.249613e+11, "Sigma_V": 0.078443, "DD": 5.097952, "EDF": 1.716743e-07, "Sigma_E": 0.190497},
    {"Date": "2020-02-29", "Close": 9.23, "E": 8.593130e+10, "DP": 1.350000e+11, "V": 2.182580e+11, "Sigma_V": 0.121794, "DD": 3.132071, "EDF": 8.678891e-04, "Sigma_E": 0.309340},
    {"Date": "2020-03-31", "Close": 8.98, "E": 8.360380e+10, "DP": 1.350000e+11, "V": 2.159306e+11, "Sigma_V": 0.098666, "DD": 3.798666, "EDF": 7.273835e-05, "Sigma_E": 0.254833},
    {"Date": "2020-04-30", "Close": 8.37, "E": 7.792470e+10, "DP": 1.350000e+11, "V": 2.102515e+11, "Sigma_V": 0.078107, "DD": 4.582355, "EDF": 2.298845e-06, "Sigma_E": 0.210743},
    {"Date": "2020-05-31", "Close": 8.72, "E": 8.118320e+10, "DP": 1.350000e+11, "V": 2.135100e+11, "Sigma_V": 0.085328, "DD": 4.309365, "EDF": 8.186192e-06, "Sigma_E": 0.224412},
    {"Date": "2020-06-30", "Close": 8.72, "E": 8.118320e+10, "DP": 1.350000e+11, "V": 2.135100e+11, "Sigma_V": 0.077222, "DD": 4.761734, "EDF": 9.596817e-07, "Sigma_E": 0.203093},
    {"Date": "2020-07-31", "Close": 10.31, "E": 9.598610e+10, "DP": 1.350000e+11, "V": 2.279587e+11, "Sigma_V": 0.267047, "DD": 1.527025, "EDF": 6.337737e-02, "Sigma_E": 0.624703},
    {"Date": "2020-08-31", "Close": 9.85, "E": 9.170350e+10, "DP": 1.350000e+11, "V": 2.240218e+11, "Sigma_V": 0.171085, "DD": 2.322705, "EDF": 1.009749e-02, "Sigma_E": 0.417616},
    {"Date": "2020-09-30", "Close": 9.10, "E": 8.472100e+10, "DP": 1.270000e+11, "V": 2.092062e+11, "Sigma_V": 0.087808, "DD": 4.475042, "EDF": 3.819814e-06, "Sigma_E": 0.216829},
    {"Date": "2020-10-31", "Close": 9.69, "E": 9.021390e+10, "DP": 1.270000e+11, "V": 2.146991e+11, "Sigma_V": 0.088112, "DD": 4.635875, "EDF": 1.777157e-06, "Sigma_E": 0.088112},
    {"Date": "2020-11-30", "Close": 9.15, "E": 8.518650e+10, "DP": 1.270000e+11, "V": 2.096717e+11, "Sigma_V": 0.083906, "DD": 4.699209, "EDF": 1.305856e-06, "Sigma_E": 0.083906},
    {"Date": "2020-12-31", "Close": 7.71, "E": 7.178010e+10, "DP": 1.270000e+11, "V": 1.962653e+11, "Sigma_V": 0.095933, "DD": 3.678790, "EDF": 1.171718e-04, "Sigma_E": 0.095933},
    {"Date": "2021-01-31", "Close": 6.61, "E": 6.153910e+10, "DP": 1.270000e+11, "V": 1.860243e+11, "Sigma_V": 0.086894, "DD": 3.651511, "EDF": 1.303509e-04, "Sigma_E": 0.086894},
    {"Date": "2021-02-28", "Close": 7.00, "E": 6.517000e+10, "DP": 1.270000e+11, "V": 1.896534e+11, "Sigma_V": 0.124816, "DD": 2.646754, "EDF": 4.063417e-03, "Sigma_E": 0.124816},
    {"Date": "2021-03-31", "Close": 6.88, "E": 6.405280e+10, "DP": 1.270000e+11, "V": 1.885332e+11, "Sigma_V": 0.132486, "DD": 2.463488, "EDF": 6.879618e-03, "Sigma_E": 0.132486},
    {"Date": "2021-04-30", "Close": 6.66, "E": 6.200460e+10, "DP": 1.270000e+11, "V": 1.864898e+11, "Sigma_V": 0.080219, "DD": 3.976588, "EDF": 3.495559e-05, "Sigma_E": 0.080219},
    {"Date": "2021-05-31", "Close": 6.80, "E": 6.330800e+10, "DP": 1.270000e+11, "V": 1.877932e+11, "Sigma_V": 0.052617, "DD": 6.152419, "EDF": 3.815486e-10, "Sigma_E": 0.052617},
    {"Date": "2021-06-30", "Close": 5.59, "E": 5.204290e+10, "DP": 1.130000e+11, "V": 1.628044e+11, "Sigma_V": 0.110654, "DD": 2.764620, "EDF": 2.849460e-03, "Sigma_E": 0.110654},
    {"Date": "2021-07-31", "Close": 5.90, "E": 5.492900e+10, "DP": 1.130000e+11, "V": 1.654462e+11, "Sigma_V": 0.202723, "DD": 1.563705, "EDF": 5.894343e-02, "Sigma_E": 0.202723},
    {"Date": "2021-08-31", "Close": 5.28, "E": 4.915680e+10, "DP": 1.130000e+11, "V": 1.599155e+11, "Sigma_V": 0.117466, "DD": 2.497554, "EDF": 6.252679e-03, "Sigma_E": 0.117466},
    {"Date": "2021-09-30", "Close": 4.99, "E": 4.645690e+10, "DP": 1.130000e+11, "V": 1.572194e+11, "Sigma_V": 0.048497, "DD": 5.799525, "EDF": 3.325150e-09, "Sigma_E": 0.048497},
    {"Date": "2021-10-31", "Close": 4.47, "E": 4.161570e+10, "DP": 1.130000e+11, "V": 1.523781e+11, "Sigma_V": 0.072482, "DD": 3.565372, "EDF": 1.816701e-04, "Sigma_E": 0.072482},
    {"Date": "2021-11-30", "Close": 3.85, "E": 3.584350e+10, "DP": 1.130000e+11, "V": 1.466059e+11, "Sigma_V": 0.071138, "DD": 3.222251, "EDF": 6.359387e-04, "Sigma_E": 0.071138},
    {"Date": "2021-12-31", "Close": 4.12, "E": 3.835720e+10, "DP": 1.130000e+11, "V": 1.491197e+11, "Sigma_V": 0.054203, "DD": 4.468753, "EDF": 3.933854e-06, "Sigma_E": 0.054203},
]

monthly = pd.DataFrame(data_records)
monthly["Date"] = pd.to_datetime(monthly["Date"])
monthly = monthly.set_index("Date").sort_index()

print(f"[数据] 共 {len(monthly)} 个数据点")
print(monthly.head())

# ============================================================
# Step 3: 定义辅助函数
# ============================================================
OUTPUT_DIR = "/Users/huzhichao/Desktop/金融风险管理pre"
CRISIS_DATE = pd.Timestamp("2020-11-30")
CRISIS_START = pd.Timestamp("2020-07-01")
CRISIS_END = pd.Timestamp("2020-12-31")

def _fmt_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def _add_crisis_span(ax):
    ax.axvspan(CRISIS_START, CRISIS_END, alpha=0.10, color="red", label="危机窗口期")

def _add_crisis_vline(ax, y_ratio=0.5):
    ax.axvline(CRISIS_DATE, color="red", linestyle="--", linewidth=2, alpha=0.85)
    ymin, ymax = ax.get_ylim()
    ax.annotate(
        "债务危机爆发\n(2020年11月)",
        xy=(CRISIS_DATE, ymin + (ymax - ymin) * y_ratio),
        xytext=(pd.Timestamp("2020-04-01"), ymin + (ymax - ymin) * (y_ratio + 0.15)),
        fontsize=11, fontweight="bold", color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="red", alpha=0.85),
    )

def get_closest(df, target_date):
    target = pd.Timestamp(target_date)
    idx = df.index[df.index.get_indexer([target], method="nearest")[0]]
    return df.loc[idx]

# ============================================================
# Step 4: 生成图表
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 生成可视化图表...")
print("=" * 60)

# ---------- 图表 1：违约距离 & 预期违约概率 ----------
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig1.suptitle(
    "苏宁易购 (002024.SZ) — KMV 信用风险模型结果 (2019–2021)",
    fontsize=17, fontweight="bold", y=0.97,
)

# 上子图：DD
ax1.plot(monthly.index, monthly["DD"], color="#1f77b4", linewidth=2.2,
         marker="o", markersize=5, label="违约距离 (DD)")
ax1.fill_between(monthly.index, monthly["DD"], alpha=0.12, color="#1f77b4")
_add_crisis_span(ax1)
_add_crisis_vline(ax1, y_ratio=0.2)

dd_min_idx = monthly["DD"].idxmin()
dd_min_val = monthly["DD"].min()
ax1.annotate(
    f"DD 最低: {dd_min_val:.3f}",
    xy=(dd_min_idx, dd_min_val),
    xytext=(dd_min_idx - pd.Timedelta(days=180), dd_min_val + 0.15),
    fontsize=10, color="#1f77b4", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
)
ax1.set_ylabel("违约距离 (DD)")
ax1.set_title("违约距离 (DD) 走势")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# 下子图：EDF
edf_pct = monthly["EDF"] * 100
ax2.plot(monthly.index, edf_pct, color="#d62728", linewidth=2.2,
         marker="s", markersize=5, label="预期违约概率 (EDF)")
ax2.fill_between(monthly.index, edf_pct, alpha=0.12, color="#d62728")
_add_crisis_span(ax2)
_add_crisis_vline(ax2, y_ratio=0.5)

edf_max_idx = edf_pct.idxmax()
edf_max_val = edf_pct.max()
ax2.annotate(
    f"EDF 峰值: {edf_max_val:.1f}%",
    xy=(edf_max_idx, edf_max_val),
    xytext=(edf_max_idx - pd.Timedelta(days=180), edf_max_val * 0.7),
    fontsize=10, color="#d62728", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
)
ax2.set_ylabel("预期违约概率 EDF (%)")
ax2.set_title("预期违约概率 (EDF) 走势")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)
_fmt_xaxis(ax2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
p1 = f"{OUTPUT_DIR}/suning_kmv_dd_edf.png"
fig1.savefig(p1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"  [1/3] 核心违约指标图 → {p1}")

# ---------- 图表 2：市场信号图 ----------
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig2.suptitle(
    "苏宁易购 (002024.SZ) — 市场信号 (2019–2021)",
    fontsize=17, fontweight="bold", y=0.97,
)

# 上子图：收盘价
ax3.plot(monthly.index, monthly["Close"], color="#2ca02c", linewidth=2.2,
         marker="D", markersize=5, label="月末收盘价")
ax3.fill_between(monthly.index, monthly["Close"], alpha=0.10, color="#2ca02c")
_add_crisis_span(ax3)
_add_crisis_vline(ax3, y_ratio=0.7)

price_max_idx = monthly["Close"].idxmax()
price_min_idx = monthly["Close"].idxmin()
ax3.annotate(
    f"高点: {monthly['Close'].max():.2f}",
    xy=(price_max_idx, monthly["Close"].max()),
    xytext=(price_max_idx - pd.Timedelta(days=200), monthly["Close"].max() + 0.8),
    fontsize=10, color="#2ca02c", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
)
ax3.annotate(
    f"低点: {monthly['Close'].min():.2f}",
    xy=(price_min_idx, monthly["Close"].min()),
    xytext=(price_min_idx + pd.Timedelta(days=60), monthly["Close"].min() + 0.5),
    fontsize=10, color="#d62728", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
)
ax3.set_ylabel("收盘价 (元)")
ax3.set_title("苏宁易购月末收盘价走势")
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.3)

# 下子图：波动率
ax4.plot(monthly.index, monthly["Sigma_E"], color="#9467bd", linewidth=2.2,
         marker="^", markersize=5, label="股权波动率 (σ_E)")
ax4.fill_between(monthly.index, monthly["Sigma_E"], alpha=0.12, color="#9467bd")
_add_crisis_span(ax4)
_add_crisis_vline(ax4, y_ratio=0.6)

vol_max_idx = monthly["Sigma_E"].idxmax()
vol_max_val = monthly["Sigma_E"].max()
ax4.annotate(
    f"波动率峰值: {vol_max_val:.2%}",
    xy=(vol_max_idx, vol_max_val),
    xytext=(vol_max_idx - pd.Timedelta(days=180), vol_max_val * 0.85),
    fontsize=10, color="#9467bd", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.2),
)
ax4.set_ylabel("股权波动率 (σ_E)")
ax4.set_title("股权波动率走势")
ax4.legend(loc="upper left")
ax4.grid(True, alpha=0.3)
_fmt_xaxis(ax4)

plt.tight_layout(rect=[0, 0, 1, 0.95])
p2 = f"{OUTPUT_DIR}/suning_kmv_market.png"
fig2.savefig(p2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"  [2/3] 市场信号图 → {p2}")

# ---------- 图表 3：模型结构图 ----------
fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(14, 9))
fig3.suptitle(
    "苏宁易购 (002024.SZ) — KMV 模型结构分析",
    fontsize=17, fontweight="bold", y=0.97,
)

# 上子图：资产价值 V vs 违约点 DP
v_billion = monthly["V"] / 1e8
dp_billion = monthly["DP"] / 1e8
ax5.plot(monthly.index, v_billion, color="#17becf", linewidth=2.2,
         marker="o", markersize=5, label="资产价值 V")
ax5.plot(monthly.index, dp_billion, color="#d62728", linewidth=2.2,
         marker="s", markersize=5, label="违约点 DP")

safe_mask = v_billion >= dp_billion
ax5.fill_between(monthly.index, v_billion, dp_billion, where=safe_mask,
                 alpha=0.15, color="#2ca02c", label="安全垫 (V > DP)")

danger_mask = v_billion < dp_billion
if danger_mask.any():
    ax5.fill_between(monthly.index, v_billion, dp_billion, where=danger_mask,
                     alpha=0.20, color="#d62728", label="危险区 (V < DP)")

_add_crisis_span(ax5)
_add_crisis_vline(ax5, y_ratio=0.3)
ax5.set_ylabel("金额 (亿元)")
ax5.set_title("资产价值 V vs 违约点 DP")
ax5.legend(loc="upper right")
ax5.grid(True, alpha=0.3)
_fmt_xaxis(ax5)

# 下子图：关键指标分组柱状对比
safe_period = get_closest(monthly, "2019-12-31")
crisis_period = get_closest(monthly, "2020-11-30")
mid_2021 = get_closest(monthly, "2021-06-30")

periods = ["安全期\n(2019-12)", "危机期\n(2020-11)", "危机深化\n(2021-06)"]
dd_vals = [safe_period["DD"], crisis_period["DD"], mid_2021["DD"]]
edf_vals = [safe_period["EDF"] * 100, crisis_period["EDF"] * 100, mid_2021["EDF"] * 100]
sigma_v_vals = [safe_period["Sigma_V"] * 100, crisis_period["Sigma_V"] * 100, mid_2021["Sigma_V"] * 100]

x = np.arange(len(periods))
width = 0.25

bars1 = ax6.bar(x - width, dd_vals, width, label="违约距离 (DD)", color="#1f77b4", alpha=0.85)
bars2 = ax6.bar(x, edf_vals, width, label="预期违约概率 EDF (%)", color="#d62728", alpha=0.85)
bars3 = ax6.bar(x + width, sigma_v_vals, width, label="资产波动率 σ_V (%)", color="#9467bd", alpha=0.85)

for bar_group in [bars1, bars2, bars3]:
    for bar in bar_group:
        height = bar.get_height()
        ax6.annotate(f"{height:.2f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

ax6.set_xticks(x)
ax6.set_xticklabels(periods)
ax6.set_ylabel("数值")
ax6.set_title("关键指标多期对比")
ax6.legend(loc="upper left")
ax6.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.95])
p3 = f"{OUTPUT_DIR}/suning_kmv_structure.png"
fig3.savefig(p3, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig3)
print(f"  [3/3] 模型结构图 → {p3}")

print(f"\n全部 3 张图表已保存至: {OUTPUT_DIR}/")
print("\n" + "=" * 60)
print("图表生成完成！请打开图片检查中文是否正常显示")
print("=" * 60)