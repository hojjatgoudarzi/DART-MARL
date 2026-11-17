#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m compileall -q src/hr_mappo
python -m pytest -q tests/test_core.py tests/test_credit_efficiency.py
printf '\nStandalone HR-MAPPO core validation passed. Run python -m pytest -q tests for extended tests.\n'
