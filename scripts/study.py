#!/usr/bin/env python3
"""legacy_study research protocol: init, train, select, evaluate-dev, summarize-dev,
freeze, evaluate-test (one-time), summarize-test, plot.

Usage: python scripts/study.py --help
All commands except init take --root RUN_DIR. Training is sequential by design.
"""
from __future__ import annotations

import argparse
import json

from dart_marl.research.study import (
    cohort_status, evaluate_cohort, freeze_development, initialize, select_cohort, summarize, train_cohort, train_cft_cohort,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    init = sub.add_parser('init', help='Register fixed seeds/variants/scenarios/ablation matrix')
    init.add_argument('--spec', required=True)
    init.add_argument('--root', required=True)
    for name in ('status', 'train', 'select', 'train-cft', 'evaluate-dev', 'summarize-dev', 'freeze',
                 'evaluate-test', 'summarize-test', 'plot-dev', 'plot-test'):
        sp = sub.add_parser(name)
        sp.add_argument('--root', required=True)
        if name in ('train', 'select', 'train-cft'):
            sp.add_argument('--variants', nargs='+')
            sp.add_argument('--seeds', nargs='+', type=int)
        if name == 'evaluate-test':
            sp.add_argument('--unlock-test', action='store_true')
            sp.add_argument('--resume-test', action='store_true')
    args = p.parse_args()
    if args.cmd == 'init':
        result = initialize(args.spec, args.root)
        result = {'protocol_sha256': result['protocol_sha256'], 'root': args.root,
                  'training_seeds': result['spec']['training_seeds']}
    elif args.cmd == 'status':
        result = cohort_status(args.root)
    elif args.cmd == 'train':
        result = train_cohort(args.root, variants=args.variants, seeds=args.seeds)
    elif args.cmd == 'select':
        result = select_cohort(args.root, variants=args.variants, seeds=args.seeds)
    elif args.cmd == 'train-cft':
        result = train_cft_cohort(args.root, variants=args.variants, seeds=args.seeds)
    elif args.cmd.startswith('evaluate-'):
        result = evaluate_cohort(args.root, stage=args.cmd.removeprefix('evaluate-'),
                                 unlock_test=getattr(args, 'unlock_test', False),
                                 resume_test=getattr(args, 'resume_test', False))
    elif args.cmd.startswith('summarize-'):
        report = summarize(args.root, stage=args.cmd.removeprefix('summarize-'))
        result = {'stage': report['stage'], 'report': f"{args.root}/{report['stage']}_summary.json",
                  'n_training_seeds': report['independent_training_seed_count']}
    elif args.cmd == 'freeze':
        result = freeze_development(args.root)
    elif args.cmd.startswith('plot-'):
        from dart_marl.research.plots import plot_study
        from pathlib import Path
        stage = args.cmd.removeprefix('plot-')
        result = {'figures': [str(x) for x in plot_study(
            Path(args.root) / (stage + '_summary.json'), Path(args.root) / ('figures/' + stage))]}
    else:
        raise AssertionError('unexpected command')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
