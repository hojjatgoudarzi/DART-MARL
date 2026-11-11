#!/usr/bin/env python3
"""Produce publication-style *descriptive* figures/tables from recorded experiments.

No simulations, training, fabricated missing values, or inferential confidence
intervals from episodes. Historical multi-controller CIs resample controllers.
The script rejects inconsistent paired keys and always records source hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tex(value: str) -> str:
    for a, b in [('\\', r'\textbackslash{}'), ('_', r'\_'), ('%', r'\%'), ('&', r'\&'), ('#', r'\#')]:
        value = value.replace(a, b)
    return value


def _start_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.titleweight': 'bold',
        'axes.labelsize': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.edgecolor': '#8993A0',
        'axes.labelcolor': '#273747', 'xtick.color': '#3C4B5A',
        'ytick.color': '#3C4B5A', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })
    return plt


def _save(plt, fig, out: Path, name: str, manifest: dict) -> None:
    fig.tight_layout(pad=1.8)
    for ext in ('png', 'pdf', 'svg'):
        path = out / f'{name}.{ext}'
        fig.savefig(path, dpi=220, bbox_inches='tight')
        manifest['figures'].append(path.name)
    plt.close(fig)


def _read_jsonl(path: Path) -> list[dict]:
    data = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not data or any('env_steps' not in row for row in data):
        raise ValueError('empty/invalid training history')
    return data


def training_assets(train: Path, out: Path, plt, manifest: dict) -> None:
    path = train / 'train_history.jsonl'
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest['sources'][str(path)] = _hash(path)
    data = _read_jsonl(path)
    steps = np.asarray([r['env_steps'] for r in data], dtype=float)
    if np.any(np.diff(steps) <= 0):
        raise ValueError('training steps must be strictly increasing')
    returns = np.asarray([float(r['completed_return_mean']) if r.get('completed_return_mean') is not None else np.nan for r in data])
    evaluations = [(r['env_steps'], r['evaluation']) for r in data if r.get('evaluation')]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(steps, returns, color='#7890A7', alpha=.8, label='Training episodes (stochastic)')
    if evaluations:
        ax.plot([x for x, _ in evaluations], [e['mean_return'] for _, e in evaluations],
                '-o', ms=4, color='#0D9488', lw=2, label='Nominal validation (deterministic)')
    initial = train / 'initial_evaluation.json'
    if initial.is_file():
        manifest['sources'][str(initial)] = _hash(initial)
        v = json.loads(initial.read_text())['mean_return']
        ax.axhline(v, color='#EA580C', linestyle=':', lw=1.6, label=f'Before training: {v:.1f}')
    ax.set(xlabel='Environment steps', ylabel='Raw team return', title='MAPPO learning curve | one training seed')
    ax.grid(axis='y', alpha=.18); ax.legend(frameon=False, fontsize=9, loc='best')
    _save(plt, fig, out, 'training_return', manifest)
    metrics = [('value_loss','Normalized value loss'),('policy_loss','Policy loss'),
               ('approx_kl','Approximate KL'),('value_explained_variance','PPO-target explained variance'),
               ('value_rmse','PPO-target value RMSE'),('clip_fraction','PPO clipped fraction')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    for ax,(key,label) in zip(axes.flat, metrics):
        vals = np.asarray([float(r.get(key)) if r.get(key) is not None else np.nan for r in data])
        if np.isfinite(vals).any():
            ax.plot(steps, vals, color='#2563EB', lw=1.8)
        else:
            ax.text(.5,.5,'Not logged',ha='center',va='center',transform=ax.transAxes)
        ax.set(xlabel='Environment steps',ylabel=label)
        ax.grid(axis='y',alpha=.17)
    fig.suptitle('Training diagnostics | one training seed (PPO targets, not held-out MC)',fontweight='bold')
    _save(plt, fig, out, 'training_diagnostics', manifest)
    coverage = [(s, e['mean_poi_service_steps']) for s,e in evaluations
                if e.get('mean_poi_service_steps') is not None]
    if coverage:
        xs = [s for s,_ in coverage]
        v = np.asarray([row for _,row in coverage], dtype=float)
        fig, ax = plt.subplots(figsize=(9.2,4.5))
        for i in range(v.shape[1]):
            ax.plot(xs,v[:,i],'-o',markersize=3,label=f'POI {i+1}')
        ax.set(xlabel='Environment steps',ylabel='Mean physically served steps',
               title='Per-POI nominal validation | one training seed')
        ax.grid(axis='y',alpha=.18)
        ax.legend(frameon=False,ncol=min(6,v.shape[1]),bbox_to_anchor=(0.5,-.22),loc='upper center')
        _save(plt, fig, out, 'training_poi_service',manifest)
    meta = {'training_seed': None,'environment_steps':int(steps[-1]),'validation_points':len(evaluations)}
    train_cfg = train/'train_config.json'
    if train_cfg.is_file():
        meta['training_seed']=json.loads(train_cfg.read_text()).get('seed')
        manifest['sources'][str(train_cfg)]=_hash(train_cfg)
    manifest['training']=meta


def allocator_assets(report_path: Path, out: Path, plt, manifest: dict) -> None:
    report=json.loads(report_path.read_text())
    manifest['sources'][str(report_path)]=_hash(report_path)
    methods=report['methods']
    names=list(methods)
    if not names:
        raise ValueError('no allocator results')
    base=methods[names[0]]['per_episode']
    n=len(base)
    if n<1 or len({e['seed'] for e in base})!=n:
        raise ValueError('invalid evaluation seeds')
    for method,block in methods.items():
        if len(block['per_episode'])!=n or [e['seed'] for e in block['per_episode']]!=[e['seed'] for e in base]:
            raise ValueError(f'not matched by evaluation episode seed: {method}')
    av=[methods[name]['mean_return'] for name in names]
    sd=[methods[name]['std_return'] for name in names]
    fig, ax=plt.subplots(figsize=(max(9, len(names)*1.2),5.2))
    x=np.arange(len(names))
    ax.bar(x,av,color=['#0F766E' if m=='switch_aware' else '#2563EB' if m=='r_cart' else '#93A9BC' for m in names])
    for i,v in enumerate(av):
        ax.text(i,v+max(av)*.009,f'{v:.1f}',ha='center',fontsize=8)
    ax.set_xticks(x,[s.replace('_','\n') for s in names]);ax.set_ylabel('Mean team return')
    ax.set_ylim(0,max(av)*1.17 if max(av)>0 else 1)
    ax.set_title(f'Allocator comparison | one controller, {n} matched episodes')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    _save(plt,fig,out,'allocator_returns_descriptive',manifest)
    if 'r_cart' in methods and 'switch_aware' in methods:
        deltas=[a['return']-b['return'] for a,b in zip(methods['r_cart']['per_episode'],methods['switch_aware']['per_episode'])]
        fig,ax=plt.subplots(figsize=(8.6,4.0))
        ax.bar(range(n),deltas,color=['#0F766E' if d>=0 else '#DC654A' for d in deltas])
        ax.axhline(0,color='#3D4E5F',lw=1)
        ax.set(xlabel='Matched episode index',ylabel='R-CART − Switch-Aware return',
               title='Paired episode differences | descriptive, NOT independent training seeds')
        ax.grid(axis='y',alpha=.15)
        _save(plt,fig,out,'paired_episode_differences',manifest)
    fig, ax=plt.subplots(figsize=(max(8,len(names)*1.1),4.6))
    service=np.asarray([methods[m].get('mean_poi_service_steps',[]) for m in names],dtype=float)
    if service.ndim==2 and service.shape[1]>0:
        image=ax.imshow(service,aspect='auto',cmap='YlGnBu')
        ax.set_yticks(range(len(names)),[x.replace('_',' ') for x in names]);ax.set_xticks(range(service.shape[1]),[f'POI {i+1}' for i in range(service.shape[1])])
        fig.colorbar(image,ax=ax,label='Mean physically served steps')
        ax.set_title('Per-task physical service | one controller')
        _save(plt,fig,out,'allocator_poi_service',manifest)
    recovery_keys=[('restoration_rate_among_at_risk','Restoration rate'),
                   ('mean_post_event_repeat_service_fraction','Post-event repeat service')]
    if any(all(methods[m].get('recovery_metrics',{}).get(k) is not None for m in names) for k,_ in recovery_keys):
        fig,axes=plt.subplots(1,2,figsize=(max(10,len(names)*1.4),4.5))
        for ax,(field,label) in zip(axes,recovery_keys):
            vals=[methods[m].get('recovery_metrics',{}).get(field) for m in names]
            if all(v is not None for v in vals):
                ax.bar(np.arange(len(names)),list(map(float,vals)),color='#3C8DBC')
                ax.set_ylim(0,1.08)
            else:
                ax.text(.5,.5,'Metric unavailable',ha='center',va='center',transform=ax.transAxes)
            ax.set_xticks(np.arange(len(names)),[m.replace('_','\n') for m in names],fontsize=8)
            ax.set(ylabel=label,title=label);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        fig.suptitle('Post-disruption recovery | descriptive, one trained controller',fontweight='bold')
        _save(plt,fig,out,'allocator_recovery_descriptive',manifest)
    switches=[methods[m].get('mean_role_switches') for m in names]
    if all(v is not None for v in switches):
        fig,ax=plt.subplots(figsize=(max(8,len(names)*1.0),4.1))
        ax.bar(range(len(names)),switches,color='#536F90')
        ax.set_xticks(range(len(names)),[m.replace('_','\n') for m in names],fontsize=8)
        ax.set(ylabel='Mean role switches per episode',title='Reallocation activity | same trained controller')
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        _save(plt,fig,out,'allocator_switches_descriptive',manifest)
    with (out/'allocator_table.tex').open('w',encoding='utf-8') as f:
        f.write('% Descriptive: episodes share the SAME trained controller; do not use as training-seed CIs.\n')
        f.write('\\begin{tabular}{lrrr}\n\\toprule\nMethod & Mean return & Episode SD & Mean switches \\\\\n\\midrule\n')
        for method in names:
            r=methods[method]
            f.write(f'{_tex(method)} & {r["mean_return"]:.3f} & {r["std_return"]:.3f} & {r.get("mean_role_switches",float("nan")):.2f} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    with (out/'allocator_table.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f);wr.writerow(['method','mean_return','episode_sd','mean_role_switches','mean_unique_pois_served'])
        for m in names:
            r=methods[m];wr.writerow([m,r['mean_return'],r['std_return'],r.get('mean_role_switches'),r.get('mean_unique_pois_served')])
    manifest['evaluation']={'episodes':n,'methods':names,'unit':'one frozen controller; episodes are descriptive only'}


def historical_assets(path: Path,out: Path,plt,manifest:dict)->None:
    from dart_marl.research.statistics import seed_summary
    manifest['sources'][str(path)]=_hash(path)
    rows=list(csv.DictReader(path.open(encoding='utf-8')))
    groups=defaultdict(lambda:defaultdict(dict))
    for r in rows:
        key=(r['controller_variant'],r['method'],int(r['training_seed']))
        groups[(r['controller_variant'],r['method'])][int(r['training_seed'])][int(r['episode_seed'])]=float(r['team_return'])
    variants=sorted(set(k[0] for k in groups))
    methods=['switch_aware','r_cart']
    fig,axes=plt.subplots(1,len(variants),figsize=(5.5*len(variants),4.7),squeeze=False)
    records=[]
    for ax,variant in zip(axes.flat,variants):
        seeds0=groups.get((variant,methods[0]),{});seeds1=groups.get((variant,methods[1]),{})
        if set(seeds0)!=set(seeds1) or not seeds0:
            raise ValueError('historical comparison requires matched training seeds')
        seeds=sorted(seeds0)
        for seed in seeds:
            if seeds0[seed].keys()!=seeds1[seed].keys():
                raise ValueError('historical evaluation episodes must be paired')
        vals={m:np.array([np.mean(list(groups[(variant,m)][seed].values())) for seed in seeds]) for m in methods}
        x=np.arange(2)
        means=[vals[m].mean() for m in methods]
        intervals=[seed_summary(vals[m].tolist())['ci95_training_seed_bootstrap_percentile'] for m in methods]
        errors=np.asarray([[mean-intv[0],intv[1]-mean] for mean,intv in zip(means,intervals)]).T if len(seeds)>1 else None
        ax.bar(x,means,yerr=errors,capsize=5,color=['#0F766E','#2563EB'])
        ax.set_xticks(x,['Switch-Aware','Original R-CART']);ax.set_ylabel('Mean team return')
        ax.set_title(f'{variant.replace("_"," ").title()} | {len(seeds)} training seeds')
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for seed,a,b in zip(seeds,vals[methods[0]],vals[methods[1]]):
            records.append((variant,seed,a,b,b-a))
    _save(plt,fig,out,'historical_independent_seed_comparison',manifest)
    with (out/'historical_paired_seed_table.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['controller_variant','training_seed','switch_aware','r_cart','r_cart_minus_switch_aware'])
        w.writerows(records)
    with (out/'historical_paired_seed_table.tex').open('w',encoding='utf-8') as f:
        f.write('% Each row is one independent training seed; episode results are averaged within controller.\n')
        f.write('\\begin{tabular}{lrrrr}\n\\toprule\nController & Seed & Switch-Aware & R-CART & Difference \\\\\n\\midrule\n')
        for variant,seed,a,b,d in records:
            f.write(f'{_tex(variant)} & {seed} & {a:.3f} & {b:.3f} & {d:+.3f} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    manifest['historical']={'training_seeds_per_variant':{v:len(set(s for a,s,*_ in records if a==v)) for v in variants},
        'metric':'descriptive percentile bootstrap of independent training controllers'}


def trajectory_assets(path:Path,out:Path,plt,manifest:dict)->None:
    manifest['sources'][str(path)]=_hash(path)
    trace=json.loads(path.read_text())
    left,right=trace['left'],trace['right']
    if len(left['states'])!=len(right['states']):
        raise ValueError('unpaired trace lengths')
    fig,axes=plt.subplots(2,1,figsize=(11,6.5),sharex=True)
    for side,series,color in [(left,'Switch-Aware','#0F766E'),(right,'R-CART','#2563EB')]:
        steps=[r['step'] for r in side['states']]
        axes[0].plot(steps,[r['return'] for r in side['states']],label=series,color=color,lw=2)
        axes[1].plot(steps,[sum(r['active']) for r in side['states']],label=series,color=color,lw=1.6)
    events=defaultdict(set)
    for s in (left,right):
        for r in s['states']:
            for e in r['events']:
                events[int(r['step'])].add(e['kind'])
    for ax in axes:
        for step in sorted(events):
            ax.axvline(step,color='#D97706',alpha=.26,lw=1)
        ax.grid(axis='y',alpha=.15)
    axes[0].set(ylabel='Cumulative team return',title='Real actor rollout | paired fault schedule')
    axes[1].set(xlabel='Decision step',ylabel='Active rover count')
    axes[0].legend(frameon=False);axes[1].legend(frameon=False)
    _save(plt,fig,out,'rollout_recovery_timeline',manifest)
    total=len(left['states'])-1
    mat=[];labels=[]
    for side,name in [(left,'Switch-Aware'),(right,'R-CART')]:
        arr=np.asarray([s['served'] for s in side['states'][1:]],dtype=int).T
        for i in range(arr.shape[0]):
            mat.append(arr[i]);labels.append(f'{name} · POI {i+1}')
    fig,ax=plt.subplots(figsize=(12, max(4, .38*len(mat)+1.4)))
    ax.imshow(np.asarray(mat),cmap='YlGn',aspect='auto',vmin=0,vmax=1,interpolation='nearest')
    ax.set_yticks(range(len(labels)),labels,fontsize=8)
    ax.set(xlabel='Decision step',title='Actual per-POI service timeline (1 = physically served)')
    _save(plt,fig,out,'rollout_service_timeline',manifest)
    with (out/'rollout_events.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['decision_boundary','event_types'])
        for k in sorted(events):writer.writerow([k,' | '.join(sorted(events[k]))])
    manifest['trace']={'steps':total,'event_boundaries':len(events),'left_return':left['final_return'],
                       'right_return':right['final_return'],'controller_sha256':left['controller_sha256']}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--training',type=Path,help='One completed MAPPO training output directory')
    ap.add_argument('--evaluation',type=Path,help='One evaluate-allocators report JSON')
    ap.add_argument('--trace',type=Path,help='Paired demo JSON trajectory trace')
    ap.add_argument('--historical-csv',type=Path,help='Archived independently trained held-out episode CSV')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if not any([args.training,args.evaluation,args.trace,args.historical_csv]):
        ap.error('at least one source is required; absent sources are NOT filled with invented data')
    if args.out.exists():
        ap.error(f'output directory already exists: {args.out}')
    args.out.mkdir(parents=True)
    plt=_start_plot()
    manifest={'format':'dart-marl-paper-assets-v1',
              'sources':{},'figures':[],
              'caution':'Per-episode results from ONE trained controller do not provide training-seed inference; historical CIs resample independently trained controllers.'}
    if args.training:training_assets(args.training,args.out,plt,manifest)
    if args.evaluation:allocator_assets(args.evaluation,args.out,plt,manifest)
    if args.trace:trajectory_assets(args.trace,args.out,plt,manifest)
    if args.historical_csv:historical_assets(args.historical_csv,args.out,plt,manifest)
    manifest['artifact_sha256']={p.name:_hash(p) for p in sorted(args.out.iterdir()) if p.is_file()}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'generated_from_recorded_data','out':str(args.out),
                      'figures':len(manifest['figures'])//3,'tables':len(list(args.out.glob('*.tex'))),
                      'sources':len(manifest['sources'])},indent=2))


if __name__=='__main__':
    main()
