#!/usr/bin/env bash
set -euo pipefail

LABEL="ali-deepseek-v3.2_51c3eaf2_all_full1_iter100_c5"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWE_EXP_DIR="$ROOT_DIR"
KOCO_DIR="$SWE_EXP_DIR/../KOCO-bench/KOCO-bench-en/domain_code_generation"
HANDOFF_DIR="$SWE_EXP_DIR/tmp/koco/eval_handoff_full1"

frameworks=(
  "open-r1"
  "raganything"
  "smolagents"
  "tensorrt_model_optimizer"
  "verl"
)

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

cd "$SWE_EXP_DIR"
for framework in "${frameworks[@]}"; do
  echo "=== Evaluating $framework ==="
  .venv311/bin/python workflow_koco.py eval \
    --framework "$framework" \
    --model ali-deepseek-v3.2 \
    --output-label "$LABEL"
done

echo "=== Metrics files ==="
find "$KOCO_DIR/scripts/data" -path "*/swe_exp/$LABEL/*_result.metrics.json" -print | sort
