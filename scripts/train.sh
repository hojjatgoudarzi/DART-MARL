#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/pilot_cuda.yaml}"
OUT="${2:-outputs/pilot}"
mkdir -p "$OUT"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
printf 'HR-MAPPO standalone training | config=%s | output=%s\n' "$CONFIG" "$OUT"
sha256sum "$CONFIG" configs/rover_12.yaml | tee "$OUT/input_hashes.txt"
python -m hr_mappo.experiment train --config "$CONFIG" --env configs/rover_12.yaml --output "$OUT" 2>&1 | tee "$OUT/terminal.log"
printf 'Training completed. Core outputs: %s/{terminal.log,train_history.jsonl,checkpoint.pt,manifest.json,research_readiness.json}\n' "$OUT"
if [[ -f "$OUT/best_checkpoint.pt" ]]; then printf 'Development-selected checkpoint: %s/best_checkpoint.pt\n' "$OUT"; fi
