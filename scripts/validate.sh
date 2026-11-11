#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python --version
python -m pytest -q
python -m compileall -q src scripts
python -m dart_marl inspect --config configs/env/dev_roles.yaml
python -m dart_marl inspect --config configs/env/professor_parity.yaml
python -m dart_marl inspect --config configs/env/rover_team_12_showcase.yaml
printf '\nValidation passed. Full multi-seed training is intentionally not part of this quick check.\n'
