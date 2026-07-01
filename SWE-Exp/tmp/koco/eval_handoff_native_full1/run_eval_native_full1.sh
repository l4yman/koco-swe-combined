#!/usr/bin/env bash
set -euo pipefail

# Native SWE-Exp KOCO infer1 partial handoff.
# Only the frameworks that were fully (or effectively) finished at snapshot time
# are included: raganything (8/8) and open-r1 (23/25).
# smolagents / tensorrt_model_optimizer / verl are still running on the dev box
# and are intentionally excluded from this handoff.

LABEL="ali-deepseek-v3.2_native_full1_iter100_c5"
HANDOFF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_EXP_DIR="$(cd "$HANDOFF_DIR/../../.." && pwd)"
KOCO_DIR="$SWE_EXP_DIR/../KOCO-bench/KOCO-bench-en/domain_code_generation"
PYTHON_BIN="${PYTHON_BIN:-python}"

frameworks=(
  "raganything"
  "open-r1"
)

# 1) Stage handoff outputs into the KOCO scripts/data tree expected by eval.
for framework in "${frameworks[@]}"; do
  src="$HANDOFF_DIR/outputs/$framework/swe_exp/$LABEL"
  dst="$KOCO_DIR/scripts/data/$framework/swe_exp/$LABEL"
  if [[ ! -d "$src" ]]; then
    echo "Missing handoff output directory: $src" >&2
    exit 1
  fi
  mkdir -p "$dst"
  cp -f "$src"/*_output.jsonl "$dst"/
done

export KOCO_SWE_EXP_OUTPUT_ROOT="$KOCO_DIR/scripts/data"

# 2) Run KOCO docker eval per framework.
cd "$SWE_EXP_DIR"
for framework in "${frameworks[@]}"; do
  echo "=== Evaluating $framework ==="
  "$PYTHON_BIN" workflow_koco.py eval \
    --framework "$framework" \
    --model ali-deepseek-v3.2 \
    --output-label "$LABEL"
done

echo "=== Metrics files ==="
find "$KOCO_DIR/scripts/data" -path "*/swe_exp/$LABEL/*_result.metrics.json" -print | sort
