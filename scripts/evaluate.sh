#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CHECKPOINT="${1:?Usage: bash scripts/evaluate.sh CHECKPOINT [CONFIG] [OUTPUT]}"
CONFIG="${2:-configs/pilot_cuda.yaml}"
OUT="${3:-outputs/evaluation}"
mkdir -p "$OUT"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/compare.py --checkpoint "$CHECKPOINT" --config "$CONFIG" --output "$OUT/matched" --seeds 810101 810202 810303 810404 810505 2>&1 | tee "$OUT/matched_console.log"
python scripts/stress.py --checkpoint "$CHECKPOINT" --config "$CONFIG" --output "$OUT/stress_cut" --intervention-mode minimal_feasible_cut --seeds 810101 810202 810303 810404 810505 2>&1 | tee "$OUT/stress_console.log"
python scripts/summarize.py --run "$OUT" --output "$OUT/summary.json"
printf 'Evaluation completed: %s\n' "$OUT"
