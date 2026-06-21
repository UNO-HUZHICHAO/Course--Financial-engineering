"""
GAT 表征诊断脚本
================

读取回测输出的 H_GAT 诊断文件，生成表征塌缩报告。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(x: float | int | None, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.2%}" if pct else f"{x:.4f}"


def _safe_corr(df: pd.DataFrame, left: str, right: str) -> float:
    if left not in df.columns or right not in df.columns:
        return float("nan")
    sub = df[[left, right]].dropna()
    if len(sub) < 5:
        return float("nan")
    return float(sub[left].corr(sub[right]))


def generate_gat_embedding_report(result_dir: str | Path) -> Path:
    result_dir = Path(result_dir)
    predictions_path = result_dir / "predictions.csv"
    daily_stats_path = result_dir / "gat_embedding_daily_stats.csv"
    snapshot_stats_path = result_dir / "gat_embedding_snapshot_stats.csv"
    strategy_path = result_dir / "strategy_comparison.csv"

    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    daily_stats = pd.read_csv(daily_stats_path) if daily_stats_path.exists() else pd.DataFrame()
    snapshot_stats = pd.read_csv(snapshot_stats_path, dtype={"snap_key": str}) if snapshot_stats_path.exists() else pd.DataFrame()
    strategy_df = pd.read_csv(strategy_path) if strategy_path.exists() else pd.DataFrame()

    if not predictions.empty:
        predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
        predictions["gat_available"] = predictions.get("gat_available", 0)
        predictions["year"] = predictions["trade_date"].dt.year

    if not daily_stats.empty:
        daily_stats["trade_date"] = pd.to_datetime(daily_stats["trade_date"])
        daily_stats["year"] = daily_stats["trade_date"].dt.year
    if not snapshot_stats.empty:
        snapshot_stats["snap_key"] = snapshot_stats["snap_key"].astype(str).str.replace(".0", "", regex=False)

    sample_summary = {}
    if not predictions.empty:
        sample_summary = {
            "n_rows": int(len(predictions)),
            "gat_available_ratio": float(predictions["gat_available"].mean()) if "gat_available" in predictions.columns else float("nan"),
            "h_gat_norm_mean": float(predictions["h_gat_norm"].mean()) if "h_gat_norm" in predictions.columns else float("nan"),
            "score_hgat_corr": _safe_corr(predictions, "score", "h_gat_norm"),
            "gate_hgat_corr": _safe_corr(predictions, "gate_value", "h_gat_norm"),
        }

    daily_summary = {}
    if not daily_stats.empty:
        daily_summary = {
            "n_days": int(len(daily_stats)),
            "feature_std_mean": float(daily_stats["daily_hgat_feature_std_mean"].mean()),
            "feature_std_median": float(daily_stats["daily_hgat_feature_std_mean"].median()),
            "to_mean_l2_mean": float(daily_stats["daily_hgat_to_mean_l2_mean"].mean()),
            "cosine_mean": float(daily_stats["daily_hgat_pairwise_cosine_sample_mean"].dropna().mean()),
            "valid_gat_mean": float(daily_stats["n_valid_gat"].mean()),
        }

    snapshot_summary = {}
    if not snapshot_stats.empty:
        grouped = snapshot_stats.groupby("snap_key", as_index=False).agg(
            n_days=("trade_date", "size"),
            n_nodes_used_mean=("n_nodes_used", "mean"),
            hgat_feature_std_mean=("hgat_feature_std_mean", "mean"),
            hgat_to_mean_l2_mean=("hgat_to_mean_l2_mean", "mean"),
            hgat_pairwise_cosine_sample_mean=("hgat_pairwise_cosine_sample_mean", "mean"),
        )
        grouped = grouped.sort_values("hgat_feature_std_mean")
        snapshot_summary["table"] = grouped
        snapshot_summary["worst_snap"] = grouped.iloc[0].to_dict() if not grouped.empty else None
        snapshot_summary["best_snap"] = grouped.iloc[-1].to_dict() if not grouped.empty else None

    collapse_hint = "无法判断"
    if daily_summary:
        std_mean = daily_summary["feature_std_mean"]
        cosine_mean = daily_summary["cosine_mean"]
        if std_mean < 0.05 and cosine_mean > 0.90:
            collapse_hint = "强烈疑似塌缩：横截面方差极低且余弦相似度极高"
        elif std_mean < 0.10 and cosine_mean > 0.75:
            collapse_hint = "中度疑似塌缩：表征分散度偏低，需结合对照组判断"
        else:
            collapse_hint = "未见明显塌缩：至少从横截面分散度看仍有区分度"

    report_lines = [
        "# GAT 表征诊断报告",
        "",
        f"- 结果目录：`{result_dir}`",
        f"- 塌缩判断：**{collapse_hint}**",
        "",
        "## 1. 样本级概览",
        "",
        f"- 样本数：**{sample_summary.get('n_rows', 0)}**",
        f"- GAT 可用率：**{_fmt(sample_summary.get('gat_available_ratio'), pct=True)}**",
        f"- `h_gat_norm` 均值：**{_fmt(sample_summary.get('h_gat_norm_mean'))}**",
        f"- `score` 与 `h_gat_norm` 相关性：**{_fmt(sample_summary.get('score_hgat_corr'))}**",
        f"- `gate_value` 与 `h_gat_norm` 相关性：**{_fmt(sample_summary.get('gate_hgat_corr'))}**",
        "",
        "## 2. 日度表征分散度",
        "",
        f"- 交易日数：**{daily_summary.get('n_days', 0)}**",
        f"- `daily_hgat_feature_std_mean` 均值：**{_fmt(daily_summary.get('feature_std_mean'))}**",
        f"- `daily_hgat_feature_std_mean` 中位数：**{_fmt(daily_summary.get('feature_std_median'))}**",
        f"- 节点到日均值向量距离均值：**{_fmt(daily_summary.get('to_mean_l2_mean'))}**",
        f"- 抽样两两余弦相似度均值：**{_fmt(daily_summary.get('cosine_mean'))}**",
        f"- 日均有效 GAT 节点数：**{_fmt(daily_summary.get('valid_gat_mean'))}**",
        "",
        "## 3. 快照级对比",
        "",
    ]

    if snapshot_summary.get("table") is not None:
        report_lines.append(snapshot_summary["table"].to_markdown(index=False))
        report_lines.extend([
            "",
            f"- 最差快照：`{snapshot_summary['worst_snap']['snap_key']}` | std={_fmt(snapshot_summary['worst_snap']['hgat_feature_std_mean'])} | cosine={_fmt(snapshot_summary['worst_snap']['hgat_pairwise_cosine_sample_mean'])}",
            f"- 最优快照：`{snapshot_summary['best_snap']['snap_key']}` | std={_fmt(snapshot_summary['best_snap']['hgat_feature_std_mean'])} | cosine={_fmt(snapshot_summary['best_snap']['hgat_pairwise_cosine_sample_mean'])}",
        ])
    else:
        report_lines.append("- 无快照级统计数据")

    if not strategy_df.empty and "年化超额收益 (Alpha)" in strategy_df.columns:
        report_lines.extend([
            "",
            "## 4. 结果层参考",
            "",
            strategy_df[["strategy", "年化超额收益 (Alpha)", "信息比率", "双边年化换手率"]].to_markdown(index=False),
        ])

    report_lines.extend([
        "",
        "## 5. 解释建议",
        "",
        "- 若 `daily_hgat_feature_std_mean` 很低且余弦相似度长期接近 1，优先怀疑图过密或关系噪声导致过度平滑。",
        "- 若硬边图相较全图提高了分散度或降低了余弦相似度，说明 `Industry/LLM` 很可能是主要噪声源。",
        "- 若表征不再塌缩但 `full` 仍弱于 `lstm_only`，下一步优先测试 `FiLM`，而不是继续堆边或堆因子。",
    ])

    report_path = result_dir / "gat_embedding_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    if not daily_stats.empty:
        daily_enriched = daily_stats.copy()
        daily_enriched["collapse_flag"] = (
            (daily_enriched["daily_hgat_feature_std_mean"] < 0.05)
            & (daily_enriched["daily_hgat_pairwise_cosine_sample_mean"] > 0.90)
        ).astype(int)
        daily_enriched.to_csv(result_dir / "gat_embedding_daily_stats_enriched.csv", index=False)

    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 GAT 表征诊断报告")
    parser.add_argument("--result-dir", type=str, required=True, help="某次回测结果目录")
    args = parser.parse_args()
    path = generate_gat_embedding_report(args.result_dir)
    print(f"[Save] GAT 诊断报告: {path}")


if __name__ == "__main__":
    main()
