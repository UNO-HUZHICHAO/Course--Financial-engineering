"""
基于已有 predictions.csv 重建双策略结果。

用途：
1. 调整组合构建逻辑后，快速复算 QP / 分层等权结果；
2. 不重新训练模型，只验证组合层改动是否改善回测结果。
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from backtest.portfolio_optimizer import (
    build_layered_equal_weight_portfolio,
    optimize_portfolio,
)
from backtest.reporter import BacktestReporter

RESULT_DIR = PROJECT_ROOT / "result"
PREDICTIONS_PATH = RESULT_DIR / "predictions.csv"


def rebuild() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH)

    strategies = {
        "qp_conservative": {
            "weights": optimize_portfolio(
                predictions,
                rebalance_freq=5,
                smoothing_window=3,
                verbose=False,
                diagnostics_output_path=RESULT_DIR / "qp_conservative" / "optimizer_diagnostics.csv",
            ),
            "rebalance_freq": 5,
        },
        "layered_equal_weight": {
            "weights": build_layered_equal_weight_portfolio(
                predictions,
                rebalance_freq=10,
                tier_sizes=(20, 20, 20),
                tier_allocations=(0.4, 0.35, 0.25),
                active_share=0.10,
                smoothing_window=10,
                constituent_only=True,
                verbose=False,
                diagnostics_output_path=RESULT_DIR / "layered_equal_weight" / "optimizer_diagnostics.csv",
            ),
            "rebalance_freq": 10,
        },
    }

    rows = []
    for name, cfg in strategies.items():
        out_dir = RESULT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_df = cfg["weights"]
        weights_df.to_csv(out_dir / "optimal_weights.csv", index=False)

        reporter = BacktestReporter(
            weights_df=weights_df,
            rebalance_freq=cfg["rebalance_freq"],
            output_dir=out_dir,
        )
        reporter.generate_report(gate_values=predictions)
        rows.append({"strategy": name, **reporter.metrics})

    comparison_df = pd.DataFrame(rows)
    comparison_path = RESULT_DIR / "strategy_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    best_idx = comparison_df["年化超额收益 (Alpha)"].astype(float).idxmax()
    best_strategy = comparison_df.loc[best_idx, "strategy"]
    best_dir = RESULT_DIR / best_strategy
    shutil.copyfile(best_dir / "optimal_weights.csv", RESULT_DIR / "optimal_weights.csv")
    shutil.copyfile(best_dir / "backtest_report.md", RESULT_DIR / "backtest_report.md")

    summary_lines = [
        "# 组合策略对比汇总",
        "",
        f"- 推荐主结果: **{best_strategy}**",
        "",
        comparison_df[
            [
                "strategy",
                "年化收益率(计费前)",
                "年化收益率",
                "基准年化收益",
                "年化成本拖累",
                "年化超额收益 (计费前)",
                "年化超额收益 (Alpha)",
                "超额最大回撤",
                "信息比率",
                "夏普比率",
                "双边年化换手率",
            ]
        ].to_markdown(index=False),
    ]
    (RESULT_DIR / "strategy_comparison.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return comparison_df


if __name__ == "__main__":
    df = rebuild()
    print(df.to_string(index=False))
