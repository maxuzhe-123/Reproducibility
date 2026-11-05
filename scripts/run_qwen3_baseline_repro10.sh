#!/usr/bin/env bash
# Qwen3-coder-next baseline — 10 independent repeated runs (generation + evaluation + cross-run CodeBLEU summary)
#
# Usage:
#   scripts/run_qwen3_baseline_repro10.sh            # run all 10 repetitions (run_01..run_10)
#   scripts/run_qwen3_baseline_repro10.sh 3           # run only the first 3 (for debugging)
#
# Each repetition is a complete independent generation (164 tasks × 10 samples) with no resume/shared cache,
# so it reflects the real variation of "running the same configuration repeatedly" rather than reusing one set of generations.
# Output paths are fully isolated from other experiments:
#   runs_v2/preds_repro10/run_XX/            — generation results of this repetition
#   runs_v2/evaluations_repro10/run_XX/      — evaluation results of this repetition (including codebleu_no_docstring_summary.json)
#   runs_v2/comparisons_repro10/             — cross-run distribution after aggregating the 10 runs
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/qwen3_baseline_repro10.yaml"
GEN_ROOT="runs_v2/preds_repro10"
EVAL_ROOT="runs_v2/evaluations_repro10"
N="${1:-10}"

for i in $(seq -w 1 "$N"); do
  RUN="run_${i}"
  echo "=== [$RUN] generation ==="
  python -m src.pipeline_v2.preds \
    --config "$CONFIG" \
    --run-dir "$GEN_ROOT/$RUN"

  echo "=== [$RUN] evaluation ==="
  python -m src.pipeline_v2.evaluator \
    --config "$CONFIG" \
    --generation-run "$GEN_ROOT/$RUN" \
    --output-dir "$EVAL_ROOT/$RUN"
done

echo "=== aggregating CodeBLEU distribution across ${N} runs ==="
python -m src.pipeline_v2.analyze_repro10_codebleu \
  --eval-root "$EVAL_ROOT" \
  --out "runs_v2/comparisons_repro10/qwen3_baseline_codebleu_repro10"

echo "done."
