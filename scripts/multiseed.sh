#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/study_cuda.yaml}"
OUT="${2:-outputs/multiseed}"
mkdir -p "$OUT/configs"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
for SEED in 101 202 303; do
  python - "$CONFIG" "$OUT/configs/seed_${SEED}.yaml" "$SEED" <<'PYBLOCK'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf8'))
cfg['seed']=int(sys.argv[3])
with open(sys.argv[2],'w',encoding='utf8') as f: yaml.safe_dump(cfg,f,sort_keys=False)
PYBLOCK
  bash scripts/train.sh "$OUT/configs/seed_${SEED}.yaml" "$OUT/seed_${SEED}"
done
printf 'Completed three independent training seeds. Evaluate using each best checkpoint, not only the last one.\n'
