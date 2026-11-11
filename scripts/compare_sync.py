#!/usr/bin/env python3
"""Run post-hoc legacy_study sensitivity analysis on an existing legacy_study pilot.

No training. No held-out evaluations. Does not mutate legacy_study pilot files.
"""
from __future__ import annotations

import argparse
import json

from dart_marl.research.bounded_sync import run_development_comparison


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, help='Existing outputs/study_pilot directory')
    p.add_argument('--out', required=True, help='New directory OUTSIDE pilot root')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--variants', nargs='+', help='Optional pilot subset: e.g. fault_robust')
    p.add_argument('--seeds', nargs='+', type=int, help='Optional pilot subset: e.g. 22')
    args = p.parse_args()
    print(json.dumps(run_development_comparison(
        args.root, args.out, variants=args.variants, seeds=args.seeds,
        device=args.device), indent=2))


if __name__ == '__main__':
    main()
