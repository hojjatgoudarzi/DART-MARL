#!/usr/bin/env python3
"""Strictly combine per-method, restartable, same-controller evaluation reports.

Every method has *already* been evaluated against the same episode tape;
never average unmatched experiments or recompute any lost model trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def merge(parts: dict[str,Path], out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f'refusing to replace existing report: {out}')
    if len(parts)<2 or len(set(parts))!=len(parts):
        raise ValueError('at least two distinct methods required')
    reports={m:json.loads(p.read_text()) for m,p in parts.items()}
    ref=next(iter(reports.values()))
    fingerprints=('environment','evaluation_seeds','controller','allocator_settings','profile_latency')
    ids=ref['evaluation_seeds']
    for name,report in reports.items():
        if any(report[k]!=ref[k] for k in fingerprints):
            raise ValueError(f'mismatched controller/scenario/seeds/settings for method {name}')
        if set(report['methods'])!={name}:
            raise ValueError(f'each part must contain only its declared method: {name}')
        episodes=report['methods'][name]['per_episode']
        if [row['seed'] for row in episodes]!=ids:
            raise ValueError(f'episode seeds not matched for {name}')
    merged=dict(ref)
    merged['methods']={m:r['methods'][m] for m,r in reports.items()}
    merged['evaluation_kind']='analytic_recovery'
    merged['research_claim']='one frozen controller; no training-seed confidence intervals'
    merged['paired_differences']={}
    # Compare all methods against switch_aware, and R-CART against every other method.
    pairs=set()
    if 'switch_aware' in parts:
        pairs.update((m,'switch_aware') for m in parts if m!='switch_aware')
    if 'r_cart' in parts:
        pairs.update(('r_cart',m) for m in parts if m!='r_cart')
    for a,b in sorted(pairs):
        left=merged['methods'][a]['per_episode'];right=merged['methods'][b]['per_episode']
        diff=np.asarray([l['return']-r['return'] for l,r in zip(left,right)],dtype=float)
        merged['paired_differences'][f'{a}_minus_{b}']={
            'mean_difference':float(diff.mean()),
            'std_difference':float(diff.std(ddof=1)) if len(diff)>1 else None,
            'per_seed_differences':diff.tolist(),
        }
    merged['merged_from_verified_parts']={m:{'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                                           for m,p in parts.items()}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(merged,indent=2,allow_nan=False)+'\n')
    return {'output':str(out),'methods':list(parts),'episodes_per_method':len(ids),
            'controller_sha256':ref['controller']['checkpoint_sha256']}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parts',type=Path,required=True)
    ap.add_argument('--methods',nargs='+',required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    files={method:a.parts/method/'allocator_report.json' for method in a.methods}
    missing=[str(p) for p in files.values() if not p.is_file()]
    if missing:ap.error('missing completed method reports: '+', '.join(missing))
    print(json.dumps(merge(files,a.out),indent=2))


if __name__=='__main__':main()
