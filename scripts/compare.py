#!/usr/bin/env python3
"""Evaluate matched seeded high-level selectors and write descriptive per-episode comparisons."""
import argparse
import csv
from pathlib import Path
import numpy as np
from hr_mappo.experiment import cfg_read,evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--env',default='configs/rover_12.yaml')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[900101,900202,900303,900404,900505])
    args=p.parse_args();cfg=cfg_read(args.config)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    methods=('actor','untrained','always_idle','random','nearest','value_travel','persistent','coalition_local','static_highest_value_repeatable','fixed_initial_value','fixed_initial_nearest','window_aware')
    scores={}
    for m in methods:
        evaluate(cfg,args.env,args.checkpoint,out,args.seeds,m,True)
        with (out/f'{m}_episodes.csv').open() as f:
            scores[m]=[float(r['return_raw']) for r in csv.DictReader(f)]
    with (out/'paired_episode_differences.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','actor']+list(methods[1:]))
        for i,seed in enumerate(args.seeds):
            w.writerow([seed,scores['actor'][i]]+[scores['actor'][i]-scores[m][i] for m in methods[1:]])
    print('These episodes are nested within ONE checkpoint; they are NOT independent RL training seeds.')

if __name__=='__main__':main()
