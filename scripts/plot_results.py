#!/usr/bin/env python3
"""Recreate the archived held-out mean-return figure from the released CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results/heldout_episode_returns.csv')
    p.add_argument('--out',default='outputs/heldout_returns.png')
    a=p.parse_args()
    rows=list(csv.DictReader(open(a.input,encoding='utf-8')))
    arms=['nominal','fault_robust'];methods=['switch_aware','r_cart']
    data=np.asarray([[np.mean([float(r['team_return']) for r in rows
                         if r['controller_variant']==arm and r['method']==method])
                      for method in methods] for arm in arms])
    fig,ax=plt.subplots(figsize=(10,5.0),dpi=150)
    fig.patch.set_facecolor('#091729');ax.set_facecolor('#102238')
    xs=np.arange(len(arms));width=.27
    for j,(label,color) in enumerate([('Switch-Aware','#4fd4e6'),('Original R-CART','#fb8d9a')]):
        bars=ax.bar(xs+width*(j-.5),data[:,j],width,color=color,label=label)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.7,f'{bar.get_height():.2f}',
                    ha='center',va='bottom',color='#EDF4FE',fontweight='bold',fontsize=11)
    ax.set_xticks(xs,['Nominal controller','Fault-robust controller'],color='#D6E9F6',fontsize=12)
    ax.set_ylim(0,112);ax.set_ylabel('Mean team return',color='#D6E9F6',fontsize=12)
    ax.grid(axis='y',alpha=.12,color='#BED5E5');ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(axis='y',colors='#A8C1D3')
    ax.set_title('HELD-OUT EVALUATION  ·  10 INDEPENDENT TRAINING SEEDS PER CONTROLLER',
                 loc='left',color='#EDF4FE',fontweight='bold',fontsize=12,pad=18)
    leg=ax.legend(loc='upper right',frameon=False,labelcolor='#D6E9F6')
    fig.text(.12,.02,'Paired episode conditions; R-CART superiority was not established on the primary fault-robust arm.',
             color='#ADC3D7',fontsize=9)
    fig.subplots_adjust(left=.11,right=.97,bottom=.17,top=.83)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out,facecolor=fig.get_facecolor())
    plt.close(fig)
    print(a.out)

if __name__=='__main__': main()
