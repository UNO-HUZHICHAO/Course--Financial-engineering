"""
模块五：学术级因子诊断体系
==========================
对 LSTM 16 维正交化技术因子和 GAT 节点因子进行全面诊断分析。

功能:
  1. 单因子 IC/ICIR 分析（Rank IC, t-stat, IC>0 占比）
  2. 分位数组合分析（Q1-Q5 多空收益差，单调性验证）
  3. 因子相关性热力图（验证正交化效果）
  4. 综合模型 Score 的分位数分析

用法:
  python -m analysis.factor_diagnostics --mode ic           # 单因子 IC/ICIR
  python -m analysis.factor_diagnostics --mode quintile     # 分位数组合
  python -m analysis.factor_diagnostics --mode corr         # 相关性热力图
  python -m analysis.factor_diagnostics --mode score --predictions result/predictions.csv
  python -m analysis.factor_diagnostics --mode all          # 全部诊断
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 路径配置 ────────────────────────────────────────────────────────────────
def _find_data_root() -> Path:
    """向上搜索包含 data/processed 的项目根目录。"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data" / "processed").exists():
            return parent
    raise FileNotFoundError("未找到包含 data/processed 的项目根目录。")


_DATA_ROOT = _find_data_root()
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# 因子和市场数据路径（使用根项目的 processed 数据）
FACTOR_DIR = _DATA_ROOT / "data" / "processed" / "factors"
MARKET_DIR = _DATA_ROOT / "data" / "processed" / "market"
OUTPUT_DIR = PROJECT_ROOT / "result" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
#  1. 单因子 IC/ICIR 分析
# ============================================================================

def compute_factor_ic(
    factor_dir: Path = FACTOR_DIR,
    market_dir: Path = MARKET_DIR,
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    计算每个因子的每日截面 Rank IC = SpearmanCorr(factor_t, return_{t+1})。

    Returns
    -------
    pd.DataFrame 列: [factor, ic_mean, ic_std, icir, ic_positive_pct, t_stat]
    """
    print("=" * 60)
    print("单因子 IC/ICIR 分析")
    print("=" * 60)

    # 加载因子文件
    factor_files = sorted(factor_dir.glob("*.pkl"))
    if not factor_files:
        raise FileNotFoundError(f"未找到因子文件: {factor_dir}")
    print(f"  因子文件: {len(factor_files)} 个")

    # 加载收益率矩阵
    ret_mat = pd.read_csv(market_dir / "Daily_Returns.csv", index_col=0, parse_dates=True)
    ret_mat = ret_mat.astype("float32")

    # 加载可交易掩码
    mask_mat = pd.read_csv(market_dir / "Tradable_Mask.csv", index_col=0, parse_dates=True)
    mask_mat = mask_mat.astype("int8")

    # 构建 T+1 标签（原始收益率，非平滑版，用于 IC 诊断）
    label_mat = ret_mat.shift(-1)

    # 日期范围过滤
    dates = ret_mat.index
    date_mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
    dates = dates[date_mask]

    # 加载所有因子面板，并对齐到市场数据的股票列表
    factor_panels: Dict[str, pd.DataFrame] = {}
    common_cols = ret_mat.columns  # 以市场数据的股票列表为基准
    for fp in factor_files:
        name = fp.stem
        mat = pd.read_pickle(fp).astype("float32")
        mat = mat.reindex(columns=common_cols).fillna(0.0)
        factor_panels[name] = mat

    factor_names = list(factor_panels.keys())
    print(f"  因子列表: {factor_names}")

    # 逐日计算截面 Rank IC
    ic_records: Dict[str, List[float]] = {name: [] for name in factor_names}
    ic_dates: List[pd.Timestamp] = []

    for t_date in dates:
        if t_date not in label_mat.index:
            continue
        if t_date not in mask_mat.index:
            continue

        label_row = label_mat.loc[t_date].values  # [N]
        mask_row = mask_mat.loc[t_date].values     # [N]

        # 可交易且标签非 NaN 的股票
        valid = (mask_row == 1) & (~np.isnan(label_row))
        if valid.sum() < 20:
            continue

        label_valid = label_row[valid]
        ic_dates.append(t_date)

        for name in factor_names:
            panel = factor_panels[name]
            if t_date not in panel.index:
                ic_records[name].append(np.nan)
                continue

            factor_row = panel.loc[t_date].values[valid]

            # Rank IC = Spearman correlation
            # 使用排名后的 Pearson 相关
            from scipy import stats
            try:
                ic, _ = stats.spearmanr(factor_row, label_valid)
            except Exception:
                ic = np.nan
            ic_records[name].append(ic)

    # 汇总统计
    results = []
    for name in factor_names:
        ics = np.array(ic_records[name])
        ics = ics[~np.isnan(ics)]
        if len(ics) < 10:
            continue
        ic_mean = ics.mean()
        ic_std = ics.std()
        icir = ic_mean / (ic_std + 1e-8)
        ic_pos_pct = (ics > 0).mean()
        t_stat = ic_mean / (ic_std / np.sqrt(len(ics)) + 1e-8)
        results.append({
            "factor": name,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "ic_positive_pct": ic_pos_pct,
            "t_stat": t_stat,
            "n_days": len(ics),
        })

    df = pd.DataFrame(results).sort_values("icir", ascending=False)
    df = df.round(4)

    # 保存
    out_path = output_dir / "factor_ic_icir.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  结果已保存: {out_path}")
    print(df.to_string(index=False))

    # 绘制 IC 时间序列图
    try:
        _plot_ic_timeseries(ic_records, ic_dates, factor_names, output_dir)
    except Exception as e:
        print(f"  [WARN] IC 时序图绘制失败: {e}")

    return df


def _plot_ic_timeseries(
    ic_records: Dict[str, List[float]],
    ic_dates: List[pd.Timestamp],
    factor_names: List[str],
    output_dir: Path,
) -> None:
    """绘制因子 IC 时间序列（20 日滚动均值）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 4, figsize=(20, 12), dpi=100)
    axes = axes.flatten()

    for i, name in enumerate(factor_names[:16]):
        ax = axes[i]
        ics = pd.Series(ic_records[name], index=ic_dates[:len(ic_records[name])])
        ics_rolling = ics.rolling(20, min_periods=5).mean()
        ax.bar(ics.index, ics.values, alpha=0.3, width=2, color="steelblue")
        ax.plot(ics_rolling.index, ics_rolling.values, color="red", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(output_dir / "ic_timeseries.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  IC 时序图已保存: {output_dir / 'ic_timeseries.png'}")


# ============================================================================
#  2. 分位数组合分析（单调性验证）
# ============================================================================

def compute_quintile_analysis(
    factor_name: str = "Return_20d",
    factor_dir: Path = FACTOR_DIR,
    market_dir: Path = MARKET_DIR,
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
    n_groups: int = 5,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    对单因子做分位数组合分析。

    每日按因子值排序，等分为 n_groups 组，计算每组等权收益。
    输出: Q1-Q5 累计收益曲线，多空收益差 (Q5-Q1)。

    Returns
    -------
    pd.DataFrame 列: [date, Q1, Q2, ..., Q5, Q5-Q1]
    """
    print("=" * 60)
    print(f"分位数组合分析: {factor_name} ({n_groups} 组)")
    print("=" * 60)

    # 加载收益率
    ret_mat = pd.read_csv(market_dir / "Daily_Returns.csv", index_col=0, parse_dates=True)
    ret_mat = ret_mat.astype("float32")

    # 加载因子
    factor_path = factor_dir / f"{factor_name}.pkl"
    if not factor_path.exists():
        raise FileNotFoundError(f"因子文件不存在: {factor_path}")
    factor_mat = pd.read_pickle(factor_path).astype("float32")
    # 对齐到市场数据股票列表
    factor_mat = factor_mat.reindex(columns=ret_mat.columns).fillna(0.0)

    # 加载掩码
    mask_mat = pd.read_csv(market_dir / "Tradable_Mask.csv", index_col=0, parse_dates=True)
    mask_mat = mask_mat.astype("int8")

    # 日期范围
    dates = factor_mat.index
    date_mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
    dates = dates[date_mask]

    # T+1 收益
    label_mat = ret_mat.shift(-1)

    # 逐日计算分位数收益
    quintile_returns: Dict[int, List[float]] = {q: [] for q in range(1, n_groups + 1)}
    quintile_dates: List[pd.Timestamp] = []

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

        # 按因子值排序，分组
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

    # 构建结果 DataFrame
    result = pd.DataFrame({"date": quintile_dates})
    for q in range(1, n_groups + 1):
        result[f"Q{q}"] = quintile_returns[q]
    result["Q5_minus_Q1"] = result[f"Q{n_groups}"] - result["Q1"]
    result = result.set_index("date")

    # 累计收益
    cum_result = (1 + result).cumprod()

    # 保存
    out_path = output_dir / f"quintile_{factor_name}.csv"
    cum_result.to_csv(out_path)

    # 打印统计
    annual_ret = result.mean() * 252
    print(f"\n  年化收益:")
    for col in result.columns:
        print(f"    {col}: {annual_ret[col]:.4f}")

    # 绘图
    try:
        _plot_quintile(cum_result, factor_name, n_groups, output_dir)
    except Exception as e:
        print(f"  [WARN] 分位数图绘制失败: {e}")

    return result


def compute_score_quintile(
    predictions_path: Path,
    market_dir: Path = MARKET_DIR,
    n_groups: int = 5,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    对模型综合 Score 做分位数组合分析。
    """
    print("=" * 60)
    print(f"模型 Score 分位数分析 ({n_groups} 组)")
    print("=" * 60)

    preds = pd.read_csv(predictions_path)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])

    # 加载 T+1 收益
    ret_mat = pd.read_csv(market_dir / "Daily_Returns.csv", index_col=0, parse_dates=True)
    ret_mat = ret_mat.astype("float32")
    label_mat = ret_mat.shift(-1)

    dates = sorted(preds["trade_date"].unique())
    quintile_returns: Dict[int, List[float]] = {q: [] for q in range(1, n_groups + 1)}
    quintile_dates: List[pd.Timestamp] = []

    for t_date in dates:
        if t_date not in label_mat.index:
            continue

        day_preds = preds[preds["trade_date"] == t_date].copy()
        day_preds = day_preds[day_preds["stkcd"].isin(label_mat.columns)]

        if len(day_preds) < n_groups * 5:
            continue

        # 获取每只股票的 T+1 收益
        scores = day_preds["score"].values
        stkcds = day_preds["stkcd"].values
        labels = np.array([label_mat.loc[t_date, s] if s in label_mat.columns else np.nan for s in stkcds])

        valid = ~np.isnan(labels) & ~np.isnan(scores)
        if valid.sum() < n_groups * 5:
            continue

        scores_v = scores[valid]
        labels_v = labels[valid]

        ranks = pd.Series(scores_v).rank(method="first")
        group_size = len(ranks) // n_groups

        quintile_dates.append(t_date)
        for q in range(1, n_groups + 1):
            if q == n_groups:
                mask_q = ranks > (q - 1) * group_size
            else:
                mask_q = (ranks > (q - 1) * group_size) & (ranks <= q * group_size)
            if mask_q.sum() > 0:
                quintile_returns[q].append(labels_v[mask_q.values].mean())
            else:
                quintile_returns[q].append(0.0)

    result = pd.DataFrame({"date": quintile_dates})
    for q in range(1, n_groups + 1):
        result[f"Q{q}"] = quintile_returns[q]
    result["Q5_minus_Q1"] = result[f"Q{n_groups}"] - result["Q1"]
    result = result.set_index("date")

    cum_result = (1 + result).cumprod()

    out_path = output_dir / "quintile_model_score.csv"
    cum_result.to_csv(out_path)

    annual_ret = result.mean() * 252
    print(f"\n  年化收益:")
    for col in result.columns:
        print(f"    {col}: {annual_ret[col]:.4f}")

    try:
        _plot_quintile(cum_result, "Model_Score", n_groups, output_dir)
    except Exception as e:
        print(f"  [WARN] 分位数图绘制失败: {e}")

    return result


def _plot_quintile(
    cum_result: pd.DataFrame,
    factor_name: str,
    n_groups: int,
    output_dir: Path,
) -> None:
    """绘制分位数组合累计收益曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=100)

    # 上图：各分位累计收益
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_groups))
    for i, q in enumerate(range(1, n_groups + 1)):
        col = f"Q{q}"
        if col in cum_result.columns:
            ax1.plot(cum_result.index, cum_result[col], label=col, color=colors[i], linewidth=1.2)
    ax1.set_title(f"{factor_name} — Quintile Portfolio Cumulative Returns", fontsize=12)
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 下图：多空收益差
    if "Q5_minus_Q1" in cum_result.columns:
        ax2.plot(cum_result.index, cum_result["Q5_minus_Q1"], color="darkblue", linewidth=1.5)
        ax2.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
        ax2.set_title(f"{factor_name} — Long-Short Spread (Q{n_groups} - Q1)", fontsize=12)
        ax2.set_ylabel("Cumulative Spread")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"quintile_{factor_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  分位数图已保存: {output_dir / f'quintile_{factor_name}.png'}")


# ============================================================================
#  3. 因子相关性热力图
# ============================================================================

def compute_factor_correlation(
    factor_dir: Path = FACTOR_DIR,
    market_dir: Path = MARKET_DIR,
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    计算因子间的平均截面相关性矩阵，验证对称正交化效果。

    Returns
    -------
    pd.DataFrame 相关性矩阵 [n_factors x n_factors]
    """
    print("=" * 60)
    print("因子相关性分析")
    print("=" * 60)

    factor_files = sorted(factor_dir.glob("*.pkl"))
    if not factor_files:
        raise FileNotFoundError(f"未找到因子文件: {factor_dir}")

    mask_mat = pd.read_csv(market_dir / "Tradable_Mask.csv", index_col=0, parse_dates=True)
    mask_mat = mask_mat.astype("int8")

    # 加载因子面板并对齐
    panels: Dict[str, pd.DataFrame] = {}
    for fp in factor_files:
        mat = pd.read_pickle(fp).astype("float32")
        mat = mat.reindex(columns=mask_mat.columns).fillna(0.0)
        panels[fp.stem] = mat

    factor_names = list(panels.keys())
    dates = panels[factor_names[0]].index
    date_mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
    dates = dates[date_mask]

    # 逐日计算截面相关性，取平均
    n_f = len(factor_names)
    corr_sum = np.zeros((n_f, n_f), dtype="float64")
    count = 0

    for t_date in dates:
        if t_date not in mask_mat.index:
            continue
        mask_row = mask_mat.loc[t_date].values
        valid = mask_row == 1
        if valid.sum() < 30:
            continue

        # 构建 [N_valid, n_f] 矩阵
        factor_matrix = np.zeros((valid.sum(), n_f), dtype="float32")
        for i, name in enumerate(factor_names):
            if t_date in panels[name].index:
                factor_matrix[:, i] = panels[name].loc[t_date].values[valid]

        # 计算相关性矩阵
        with np.errstate(all="ignore"):
            c = np.corrcoef(factor_matrix, rowvar=False)
        c = np.nan_to_num(c, nan=0.0)
        corr_sum += c
        count += 1

    avg_corr = corr_sum / max(count, 1)
    corr_df = pd.DataFrame(avg_corr, index=factor_names, columns=factor_names)
    corr_df = corr_df.round(3)

    # 保存
    out_path = output_dir / "factor_correlation.csv"
    corr_df.to_csv(out_path)
    print(f"\n  相关性矩阵已保存: {out_path}")

    # 打印高相关因子对
    print("\n  高相关因子对 (|corr| > 0.5):")
    for i in range(n_f):
        for j in range(i + 1, n_f):
            if abs(avg_corr[i, j]) > 0.5:
                print(f"    {factor_names[i]} <-> {factor_names[j]}: {avg_corr[i, j]:.3f}")

    # 绘制热力图
    try:
        _plot_correlation_heatmap(corr_df, output_dir)
    except Exception as e:
        print(f"  [WARN] 热力图绘制失败: {e}")

    return corr_df


def _plot_correlation_heatmap(corr_df: pd.DataFrame, output_dir: Path) -> None:
    """绘制因子相关性热力图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    n = len(corr_df)
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_df.index, fontsize=8)

    # 在格子里显示数值
    for i in range(n):
        for j in range(n):
            val = corr_df.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Factor Cross-Sectional Correlation Matrix", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "factor_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  热力图已保存: {output_dir / 'factor_correlation_heatmap.png'}")


# ============================================================================
#  4. 子区间稳定性分析
# ============================================================================

def compute_subperiod_stability(
    predictions_path: Path,
    market_dir: Path = MARKET_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    将 OOS 区间按半年拆分，计算每段的 Rank IC、ICIR。
    """
    print("=" * 60)
    print("子区间稳定性分析")
    print("=" * 60)

    preds = pd.read_csv(predictions_path)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])

    ret_mat = pd.read_csv(market_dir / "Daily_Returns.csv", index_col=0, parse_dates=True)
    label_mat = ret_mat.shift(-1)

    # 定义子区间
    subperiods = []
    for year in range(2021, 2026):
        for half in [1, 2]:
            if half == 1:
                start = f"{year}-01-01"
                end = f"{year}-06-30"
            else:
                start = f"{year}-07-01"
                end = f"{year}-12-31"
            subperiods.append({"name": f"{year}H{half}", "start": start, "end": end})

    from scipy import stats

    results = []
    for sp in subperiods:
        sp_preds = preds[
            (preds["trade_date"] >= sp["start"]) & (preds["trade_date"] <= sp["end"])
        ]
        if len(sp_preds) < 100:
            continue

        ics = []
        for t_date in sp_preds["trade_date"].unique():
            if t_date not in label_mat.index:
                continue
            day = sp_preds[sp_preds["trade_date"] == t_date]
            scores = day["score"].values
            stkcds = day["stkcd"].values
            labels = np.array([
                label_mat.loc[t_date, s] if s in label_mat.columns else np.nan
                for s in stkcds
            ])
            valid = ~np.isnan(labels) & ~np.isnan(scores)
            if valid.sum() < 20:
                continue
            try:
                ic, _ = stats.spearmanr(scores[valid], labels[valid])
                ics.append(ic)
            except Exception:
                pass

        if len(ics) < 5:
            continue

        ics = np.array(ics)
        results.append({
            "subperiod": sp["name"],
            "ic_mean": ics.mean(),
            "ic_std": ics.std(),
            "icir": ics.mean() / (ics.std() + 1e-8),
            "t_stat": ics.mean() / (ics.std() / np.sqrt(len(ics)) + 1e-8),
            "n_days": len(ics),
        })

    df = pd.DataFrame(results).round(4)
    out_path = output_dir / "subperiod_stability.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  子区间分析已保存: {out_path}")
    print(df.to_string(index=False))
    return df


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="因子诊断体系")
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["ic", "quintile", "corr", "score", "subperiod", "all"],
        help="诊断模式: ic=IC/ICIR, quintile=分位数, corr=相关性, score=模型Score, subperiod=子区间, all=全部"
    )
    parser.add_argument("--factor", type=str, default="Return_20d", help="分位数分析的目标因子名")
    parser.add_argument("--predictions", type=str, default=None, help="模型预测文件路径 (score/quintile 模式)")
    parser.add_argument("--start", type=str, default="2021-01-01", help="分析起始日期")
    parser.add_argument("--end", type=str, default="2025-12-31", help="分析结束日期")
    parser.add_argument("--n-groups", type=int, default=5, help="分位数组数")
    parser.add_argument("--factor-dir", type=str, default=None, help="因子文件目录")
    parser.add_argument("--market-dir", type=str, default=None, help="市场数据目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    factor_dir = Path(args.factor_dir) if args.factor_dir else FACTOR_DIR
    market_dir = Path(args.market_dir) if args.market_dir else MARKET_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  因子目录: {factor_dir}")
    print(f"  市场目录: {market_dir}")
    print(f"  输出目录: {output_dir}")
    print()

    mode = args.mode

    if mode in ("ic", "all"):
        compute_factor_ic(factor_dir, market_dir, args.start, args.end, output_dir)
        print()

    if mode in ("quintile", "all"):
        compute_quintile_analysis(args.factor, factor_dir, market_dir, args.start, args.end, args.n_groups, output_dir)
        print()

    if mode in ("corr", "all"):
        compute_factor_correlation(factor_dir, market_dir, args.start, args.end, output_dir)
        print()

    if mode in ("score", "all"):
        preds_path = args.predictions
        if preds_path is None:
            preds_path = PROJECT_ROOT / "result" / "predictions.csv"
        preds_path = Path(preds_path)
        if preds_path.exists():
            compute_score_quintile(preds_path, market_dir, args.n_groups, output_dir)
            compute_subperiod_stability(preds_path, market_dir, output_dir)
        else:
            print(f"  [SKIP] 预测文件不存在: {preds_path}")
        print()

    print("=" * 60)
    print("诊断完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
