#!/usr/bin/env python3
"""Only actual saved experiments; never interpolate fabricated learning curves."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',required=True,help='one train_history.jsonl per independent seed')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    fig,ax=plt.subplots(figsize=(9,5))
    for file in args.runs:
        raw=[json.loads(x) for x in Path(file).read_text().splitlines() if x.strip()]
        if not raw:raise ValueError(f'empty history: {file}')
        xs=np.array([r['update'] for r in raw]);ys=np.array([r['mean_return'] for r in raw])
        ax.plot(xs,ys,alpha=.65,label=Path(file).parent.name)
    ax.set(xlabel='PPO update',ylabel='Mean raw physical team return',title='Independent HR-MAPPO training runs')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    print(args.output)

if __name__=='__main__':main()
