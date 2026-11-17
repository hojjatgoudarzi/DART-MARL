#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/validate.sh
OUT="outputs/smoke_$(date +%Y%m%d_%H%M%S)"
bash scripts/train.sh configs/smoke_cpu.yaml "$OUT"
python -m hr_mappo.experiment evaluate --config configs/smoke_cpu.yaml --env configs/rover_12.yaml --output "$OUT/evaluation" --checkpoint "$OUT/checkpoint.pt" --seeds 810101 --method actor
printf 'Smoke test finished (not a learning-quality or GPU validation).\n'
