#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONUNBUFFERED=1

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-1e-3}"
LOG_DIR="${SCRIPT_DIR}/logs"
RESULT_ROOT="${SCRIPT_DIR}/result"

mkdir -p "${LOG_DIR}" "${RESULT_ROOT}"

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(42 123)
fi

COMMON_ARGS=(
  --ablation full
  --factor-set base16+micro8
  --graph-profile no_industry
  --fusion-mode adaptive_film
  --qp-profile guarded
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --device "${DEVICE}"
)

echo "========================================"
echo " GLV4 AutoDL Full Run"
echo " Workspace: ${WORKSPACE_ROOT}"
echo " Project:   ${SCRIPT_DIR}"
echo " Device:    ${DEVICE}"
echo " Seeds:     ${SEEDS[*]}"
echo "========================================"

for seed in "${SEEDS[@]}"; do
  run_name="full_base16+micro8_no_industry_adaptive_film_qpguarded_seed${seed}"
  run_log="${LOG_DIR}/${run_name}.log"

  echo ""
  echo "[Run] seed=${seed} -> ${run_name}"
  echo "[Log] ${run_log}"

  python3 src/run_backtest.py \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" 2>&1 | tee "${run_log}"

  echo "[Done] seed=${seed}"
done

python3 - <<'PY'
from pathlib import Path
import pandas as pd

project_root = Path.cwd()
result_root = project_root / "result"
run_dirs = sorted(result_root.glob("full_base16+micro8_no_industry_adaptive_film_qpguarded_seed*"))

rows = []
for run_dir in run_dirs:
    csv_path = run_dir / "strategy_comparison.csv"
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    df["run_dir"] = run_dir.name
    rows.append(df)

if rows:
    summary = pd.concat(rows, ignore_index=True)
    keep_cols = [
        "run_dir",
        "strategy",
        "年化超额收益 (Alpha)",
        "年化收益率",
        "超额最大回撤",
        "信息比率",
        "夏普比率",
        "双边年化换手率",
    ]
    keep_cols = [c for c in keep_cols if c in summary.columns]
    summary = summary[keep_cols]
    summary_csv = result_root / "autodl_full_run_summary.csv"
    summary_md = result_root / "autodl_full_run_summary.md"
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(
        "# AutoDL Full Run Summary\n\n" + summary.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(f"[Summary] {summary_csv}")
    print(f"[Summary] {summary_md}")
else:
    print("[Summary] No finished run directories found.")
PY

echo ""
echo "========================================"
echo " All requested seeds finished."
echo " Result root: ${RESULT_ROOT}"
echo " Logs:        ${LOG_DIR}"
echo "========================================"
