#!/usr/bin/env python3
"""Create four evidence-grounded, coordinated multi-panel paper figures.

Inputs are *recorded*, matched 12-rover evaluations, a real actor trajectory,
and the training JSONL. One trained controller != multiple training seeds.
No invented stress levels, fabricated intervals, or model rollouts occur here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

FORMATS = ('png', 'pdf', 'svg')
METHODS = ('fixed', 'greedy', 'switch_aware', 'r_cart',
           'r_cart_no_sync', 'r_cart_no_switch',
           'r_cart_no_hysteresis', 'r_cart_bounded_sync', 'r_cart_service')
CORE = ('fixed', 'switch_aware', 'r_cart', 'r_cart_service')
COLORS = {'fixed':'#8796A6', 'greedy':'#A1A76D', 'switch_aware':'#087F77',
          'r_cart':'#C14D56', 'r_cart_no_sync':'#6B72BF',
          'r_cart_no_switch':'#BC815D', 'r_cart_no_hysteresis':'#A56A9D',
          'r_cart_bounded_sync':'#3C9DB3', 'r_cart_service':'#C18C24'}
LABELS = {'fixed':'Fixed', 'greedy':'Greedy', 'switch_aware':'Switch-Aware',
          'r_cart':'Original R-CART', 'r_cart_no_sync':'No sync',
          'r_cart_no_switch':'No switch cost', 'r_cart_no_hysteresis':'No hysteresis',
          'r_cart_bounded_sync':'Bounded sync', 'r_cart_service':'Service-aware'}


def _hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _plot_backend():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9.4,
        'axes.titlesize': 11.5, 'axes.titleweight': 'bold',
        'axes.labelsize': 9.7, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.edgecolor': '#98A2AC',
        'axes.labelcolor': '#273745', 'xtick.color': '#526171',
        'ytick.color': '#526171', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'grid.alpha': .14, 'lines.linewidth': 1.9,
    })
    return plt


def _decorate(axes, subtitle: str) -> None:
    for i, ax in enumerate(axes.flat):
        title=ax.get_title()
        ax.set_title('', loc='center')
        ax.set_title(f'({chr(65+i)})  {title}', loc='left', pad=10)
        ax.grid(axis='y', color='#92A4B7', alpha=.18, linewidth=.7)
        ax.set_axisbelow(True)


def _save(fig, plt, out: Path, stem: str, registry: list[str]) -> None:
    for ext in FORMATS:
        name = f'{stem}.{ext}'
        fig.savefig(out / name, dpi=240, bbox_inches='tight', facecolor='white')
        registry.append(name)
    plt.close(fig)


def _series(history: list[dict], key: str) -> np.ndarray:
    return np.asarray([np.nan if r.get(key) is None else float(r[key]) for r in history])


def _smooth(y: np.ndarray, width: int = 9) -> np.ndarray:
    """Simple display-only moving mean, preserving missing-data gaps."""
    if width <= 1:
        return y.copy()
    ok=np.isfinite(y).astype(float)
    yy=np.nan_to_num(y, nan=0.)
    ker=np.ones(width, dtype=float)
    n=np.convolve(ok, ker, mode='same')
    sums=np.convolve(yy,ker,mode='same')
    return np.divide(sums,n,out=np.full_like(y,np.nan),where=n>0)


def _training_figure(path: Path, init_path: Path, out: Path, plt, registry: list[str]):
    history=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if not history: raise ValueError('missing training data')
    steps=_series(history,'env_steps')
    if not np.all(np.diff(steps)>0): raise ValueError('non-increasing training steps')
    values=[
        ('completed_return_mean','Training return (raw)', 'Raw team return'),
        ('value_explained_variance','Critic explained variance','Explained variance'),
        ('value_rmse','Critic value error','PPO-target RMSE'),
        ('value_loss','Critic loss','Normalized loss'),
        ('approx_kl','PPO approximate KL','Approx. KL'),
        ('clip_fraction','PPO clipping','Clipped fraction'),
    ]
    fig, axes=plt.subplots(2,3,figsize=(16.8,9.0))
    for ax,(key,title,ylabel) in zip(axes.flat,values):
        y=_series(history,key)
        if np.isfinite(y).any():
            ax.plot(steps/1000,y,alpha=.30,lw=.85,color='#95B0CA',label='Raw updates')
            ax.plot(steps/1000,_smooth(y,9),lw=2.1,color='#2064A5',label='9-update moving mean')
            if key=='completed_return_mean':
                ev=[(float(r['env_steps'])/1000,float(r['evaluation']['mean_return']))
                    for r in history if r.get('evaluation') and r['evaluation'].get('mean_return') is not None]
                if ev:
                    ax.plot(*zip(*ev),color=COLORS['switch_aware'],marker='o',ms=2.6,lw=1.45,label='Nominal validation')
                if init_path.is_file():
                    initial=float(_load(init_path)['mean_return'])
                    ax.axhline(initial,color='#C88229',ls=':',lw=1.5,label='Pre-training validation')
            if key=='value_explained_variance': ax.axhline(0,color='#B7BEC6',lw=.8)
        else:
            ax.text(.5,.5,'Not logged',transform=ax.transAxes,ha='center',va='center')
        ax.set(xlabel='Environment steps (thousands)',ylabel=ylabel,title=title)
        if key=='completed_return_mean':
            ax.legend(fontsize=7,frameon=False,loc='lower right')
        elif key=='value_explained_variance':
            ax.legend(fontsize=7,frameon=False,loc='lower right')
    _decorate(axes, '')
    fig.suptitle('Training and optimization diagnostics | one trained controller',fontsize=15.5,fontweight='bold',y=.985)
    fig.subplots_adjust(left=.075,right=.978,top=.905,bottom=.085,hspace=.39,wspace=.31)
    _save(fig,plt,out,'training_overview_2x3',registry)
    return len(history)


def _check_report(report: dict):
    methods=report.get('methods',{})
    if not all(k in methods for k in CORE):
        raise ValueError('missing core allocator results')
    ref=[int(x['seed']) for x in methods['switch_aware']['per_episode']]
    if len(ref)<2 or len(set(ref))!=len(ref):
        raise ValueError('need multiple distinct evaluation episode seeds')
    for method,m in methods.items():
        rows=m['per_episode']
        seeds=[int(x['seed']) for x in rows]
        if seeds != ref:raise ValueError(f'unmatched evaluation seeds: {method}')
        if not np.isclose(float(np.mean([e['return'] for e in rows])), float(m['mean_return'])):
            raise ValueError(f'inconsistent mean return: {method}')
    return methods,ref


def _allocator_figure(report: dict, out: Path, plt, registry: list[str]):
    blocks, seeds=_check_report(report)
    keys=[x for x in METHODS if x in blocks]
    colors=[COLORS[x] for x in keys]
    fig,axes=plt.subplots(2,3,figsize=(17.4,10.5))
    axs=axes.flat
    x=np.arange(len(keys))
    means=[float(blocks[k]['mean_return']) for k in keys]
    axs[0].bar(x, means,color=colors,alpha=.85,width=.72)
    for xi,k in enumerate(keys):
        y=[float(e['return']) for e in blocks[k]['per_episode']]
        axs[0].scatter(np.full(len(y),xi)+np.linspace(-.15,.15,len(y)), y,s=12,
                       color='#203747',alpha=.65,zorder=4)
    axs[0].set_xticks(x,[LABELS[k] for k in keys],rotation=40,ha='right',fontsize=7)
    axs[0].set(ylabel='Team return',title='Mean + episode points (9 methods)')
    axs[0].set_ylim(bottom=0)

    episode_idx=np.arange(1,len(seeds)+1)
    for k in CORE:
        axs[1].plot(episode_idx,[float(e['return']) for e in blocks[k]['per_episode']],
            '-o',ms=3,color=COLORS[k],label=LABELS[k])
    axs[1].set(xlabel='Matched evaluation episode',ylabel='Team return',title='Paired episode performance')
    axs[1].set_xticks(episode_idx)

    base=np.asarray([e['return'] for e in blocks['switch_aware']['per_episode']],dtype=float)
    for k in ('fixed','r_cart','r_cart_service'):
        vals=np.asarray([e['return'] for e in blocks[k]['per_episode']],dtype=float)-base
        axs[2].plot(episode_idx,vals,marker='o',ms=3,color=COLORS[k],label=LABELS[k]+' minus Switch')
    axs[2].axhline(0,color='#1F3448',lw=.85)
    axs[2].set(xlabel='Matched evaluation episode',ylabel='Return difference',title='Episode-level contrasts')
    axs[2].set_xticks(episode_idx)

    for k in CORE:
        axs[3].plot(episode_idx,[int(e['role_switches']) for e in blocks[k]['per_episode']],
            '-o',ms=3,color=COLORS[k],label=LABELS[k])
    axs[3].set(xlabel='Matched evaluation episode',ylabel='Role switches',title='Reassignment workload')
    axs[3].set_xticks(episode_idx)

    frac=[float(blocks[k]['recovery_metrics']['mean_post_event_repeat_service_fraction']) for k in keys]
    axs[4].bar(x,frac,color=colors,alpha=.85)
    axs[4].set_xticks(x,[LABELS[k] for k in keys],rotation=40,ha='right',fontsize=7)
    axs[4].set(ylabel='Service fraction',title='Repeat-task service after disruption')
    axs[4].set_ylim(0,1.05)

    for k in CORE:
        axs[5].plot(episode_idx,
            [float(e['after_first_disruption_return']) for e in blocks[k]['per_episode']],
            '-o',ms=3,color=COLORS[k],label=LABELS[k])
    axs[5].set(xlabel='Matched evaluation episode',ylabel='Cumulative post-first-fault return',
               title='Reward following first disruption')
    axs[5].set_xticks(episode_idx)
    _decorate(axes, '')
    handles,labels=axs[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.008),ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Fault-aware allocator comparison | same frozen MAPPO controller',fontsize=15.5,fontweight='bold',y=.982)
    fig.subplots_adjust(left=.07,right=.985,top=.895,bottom=.17,hspace=.53,wspace=.30)
    _save(fig,plt,out,'allocator_comparison_2x3',registry)

    table_rows=[]
    for k in keys:
        vals=np.asarray([e['return'] for e in blocks[k]['per_episode']],dtype=float)
        table_rows.append((k,float(vals.mean()),float(vals.std(ddof=1)),float((vals-base).mean()),
              float(blocks[k]['mean_role_switches']),
              float(blocks[k]['recovery_metrics']['mean_post_event_repeat_service_fraction'])))
    with (out/'allocator_summary.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.writer(f);wr.writerow(('method','mean_return','episode_sd','mean_delta_vs_switch_aware',
                   'mean_role_switches','post_disruption_service_fraction'))
        wr.writerows(table_rows)
    with (out/'allocator_summary.tex').open('w',encoding='utf-8') as f:
        f.write('% Descriptive results, one frozen controller; NOT training-seed CIs.\n')
        f.write('\\begin{tabular}{lrrrrr}\\toprule\nMethod & Return & Episode SD & $\\Delta$ vs Switch & Switches & Post-fault service \\\\\n\\midrule\n')
        for k,mean,sd,delta,sw,frac in table_rows:
            f.write(f'{LABELS[k]} & {mean:.2f} & {sd:.2f} & {delta:+.2f} & {sw:.2f} & {frac:.3f} \\\\\n')
        f.write('\\bottomrule\\end{tabular}\n')
    with (out/'paired_episode_returns.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.writer(f);wr.writerow(['environment_seed']+keys)
        for i,seed in enumerate(seeds):
            wr.writerow([seed]+[float(blocks[k]['per_episode'][i]['return']) for k in keys])
    return len(seeds)


def _per_step(trace:dict):
    left,right=trace['left'],trace['right']
    if len(left['states'])!=len(right['states']): raise ValueError('unpaired simulation trace')
    if left.get('seed')!=right.get('seed'): raise ValueError('unpaired starting seed')
    if left.get('controller_sha256')!=right.get('controller_sha256'):
        raise ValueError('different trained actors across trace sides')
    if len(left['states'])<2:raise ValueError('empty trace')
    events={}
    for side in (left,right):
        for step in side['states']:
            if step['events']:
                events.setdefault(int(step['step']),set()).update(e['kind'] for e in step['events'])
    return left,right,events


def _trajectory_figure(trace:dict,out:Path,plt,registry:list[str]):
    left,right,events=_per_step(trace)
    fig,axes=plt.subplots(2,3,figsize=(17.2,10.2))
    trace_colors=[COLORS['switch_aware'],COLORS['fixed'] if right['method']=='fixed' else COLORS['r_cart']]
    names=[LABELS.get(left['method'],left['method']),LABELS.get(right['method'],right['method'])]
    ss=[left['states'],right['states']]
    for st,c,name in zip(ss,trace_colors,names):
        t=np.asarray([r['step'] for r in st],dtype=int)
        axes[0,0].plot(t,[r['return'] for r in st],color=c,label=name)
        axes[0,1].plot(t,[sum(r['active']) for r in st],color=c,label=name)
        axes[0,2].plot(t,[sum(r['served']) for r in st],color=c,label=name)
        # Incremental replan flags are recorded in the trace, not inferred from commanded roles.
        replan=np.cumsum([bool(r.get('replanned',False)) for r in st])
        axes[1,0].plot(t,replan,color=c,label=name)
    for ax in list(axes[0])+[axes[1,0]]:
        for step in sorted(events):ax.axvline(step,color='#D99A44',lw=1,ls=':',alpha=.38)
        ax.set_xlabel('Simulation step')
    axes[0,0].set(title='Cumulative reward',ylabel='Team return')
    axes[0,1].set(title='Active roster size',ylabel='Active rovers')
    axes[0,2].set(title='Tasks serviced each step',ylabel='POIs physically served')
    axes[1,0].set(title='Role replan events',ylabel='Cumulative replans')
    for ax, st, name in [(axes[1,1],ss[0],names[0]),(axes[1,2],ss[1],names[1])]:
        arr=np.asarray([r['served'] for r in st[1:]],dtype=float).T
        im=ax.imshow(arr,origin='lower',interpolation='nearest',aspect='auto',cmap='YlGnBu',vmin=0,vmax=1,
               extent=[0,len(st)-1,-.5,arr.shape[0]-.5])
        ax.set(ylabel='POI index',xlabel='Simulation step',title=f'Physical service: {name}')
        ax.set_yticks(range(arr.shape[0]))
        for step in sorted(events):ax.axvline(step,color='#E09B38',lw=1,ls=':',alpha=.6)
    _decorate(axes,'')
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.012),frameon=False,ncol=2)
    fig.suptitle('Real trained-policy rollout | common initial state and fault tape',fontsize=15.5,fontweight='bold',y=.985)
    fig.subplots_adjust(left=.065,right=.985,top=.9,bottom=.105,hspace=.38,wspace=.29)
    _save(fig,plt,out,'recovery_timeline_2x3',registry)
    with (out/'demo_events.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','events'])
        for step,kinds in sorted(events.items()):writer.writerow([step,'; '.join(sorted(kinds))])
    return len(events)


def _historical_figure(path:Path,out:Path,plt,registry:list[str]):
    rows=list(csv.DictReader(path.open(encoding='utf-8')))
    groups={}
    for r in rows:
        v=r['controller_variant'];m=r['method'];key=(v,int(r['training_seed']),m)
        groups.setdefault(key,{})[int(r['episode_seed'])]=float(r['team_return'])
    variants=sorted({k[0] for k in groups})
    fig,axes=plt.subplots(1,len(variants),figsize=(6.5*len(variants),5.1),squeeze=False)
    records=[]
    for ax,variant in zip(axes.flat,variants):
        keys=sorted({seed for vv,seed,_ in groups if vv==variant})
        for seed in keys:
            lhs=groups[(variant,seed,'switch_aware')]
            rhs=groups[(variant,seed,'r_cart')]
            if set(lhs)!=set(rhs):raise ValueError('not matched within independent controller')
            x=float(np.mean(list(lhs.values())))
            y=float(np.mean(list(rhs.values())))
            records.append((variant,seed,x,y,y-x))
        vals=[r for r in records if r[0]==variant]
        for _,seed,x,y,_ in vals:
            ax.plot([0,1],[x,y],color='#9BACBE',lw=.8,alpha=.75,zorder=2)
            ax.scatter([0,1],[x,y],color=[COLORS['switch_aware'],COLORS['r_cart']],s=24,zorder=3)
        ax.set_xticks([0,1],['Switch-Aware','Original R-CART'])
        ax.set(ylabel='Mean team return per training seed',title=f'{variant.replace("_"," ").title()} (n={len(keys)})')
        ax.grid(axis='y',alpha=.2)
    for i,ax in enumerate(axes.flat):
        title=ax.get_title();ax.set_title('',loc='center')
        ax.set_title(f'({chr(65+i)})  {title}',loc='left',pad=10)
    fig.suptitle('Historical six-rover study | independent training controllers',fontweight='bold',fontsize=14,y=.982)
    fig.subplots_adjust(left=.09,right=.97,bottom=.19,top=.84,wspace=.23)
    _save(fig,plt,out,'historical_controller_pairs',registry)
    return len(records)


def build(source:Path,out:Path)->dict:
    data=source/'data'; media=source/'media'
    training=data/'train_history.jsonl'
    initial=data/'initial_evaluation.json'
    report=data/'allocator_report.json'
    # Bundle root auto-detected from results/team12 source if present.
    repo=source.parent.parent if source.name=='team12' else source
    trace=repo/'media'/'rover_reallocation_vs_fixed_12_trace.json'
    historical=repo/'results'/'heldout_episode_returns.csv'
    needed=[training,initial,report,trace,historical]
    for p in needed:
        if not p.is_file():raise FileNotFoundError(f'Required factual input absent: {p}')
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Refusing overwrite of nonempty directory {out}')
    out.mkdir(parents=True,exist_ok=True)
    plt=_plot_backend(); registry=[]
    updates=_training_figure(training,initial,out,plt,registry)
    episodes=_allocator_figure(_load(report),out,plt,registry)
    events=_trajectory_figure(_load(trace),out,plt,registry)
    controllers=_historical_figure(historical,out,plt,registry)
    sources={str(p.relative_to(repo)):_hash(p) for p in needed}
    manifest={'format':'dart-marl-figure-suite-v1','scope':'one 12-rover trained controller plus separate archival six-rover multi-controller study',
        'training_updates':updates,'paired_evaluation_episodes':episodes,'demo_event_boundaries':events,
        'historical_controller_variant_seed_rows':controllers,
        'limitations':['No 12-rover training-seed confidence interval','No intensity sweep/scale comparison without new measured evaluations',
                       'No inference from eight correlated evaluation episodes to independent trained controllers',
                       'PPO-target explained variance does not establish return improvement'],
        'source_sha256':sources,'figures':registry,
        'artifacts_sha256':{p.name:_hash(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('results/team12'),help='Published results/team12 directory')
    p.add_argument('--out',type=Path,default=Path('outputs/paper_figures'),help='New empty output directory')
    args=p.parse_args()
    manifest=build(args.source.resolve(),args.out.resolve())
    print(json.dumps({'status':'regenerated_from_recorded_evidence',
        'figures':len(manifest['figures'])//3,'panels':20,'out':str(args.out),
        'evaluation_episodes':manifest['paired_evaluation_episodes']},indent=2))

if __name__=='__main__':main()
