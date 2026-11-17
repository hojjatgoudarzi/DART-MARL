#!/usr/bin/env python3
"""Pre-training v2.4 IDENTIFIABILITY audit. No neural training needed.

Matched explicit service physics and exactly identical pre-seeded environments.
Diagnostic *privileged* current-state planner is a FEASIBILITY witness, not a
fair decentralized baseline. It uses present full rover state, no future faults.
Local baselines use only the public map and per-rover sensor observation.
All methods use the identical low-level controller and 4-tick decisions.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from hr_mappo.envs.config import load_config
from hr_mappo.scenarios import create_scenario
from hr_mappo.physics import DeclaredIntentEnv
from hr_mappo.observations import observe_agents
from hr_mappo.baselines import make_selector
from hr_mappo.control import control
from hr_mappo.metrics import physical_metrics


def current_state_feasibility_planner(env):
    """TRAINING/DIAGNOSTIC ONLY: reads GLOBAL ACTIVE positions and skills.

    This intentionally violates decentralized execution: never present it as
    a same-information classical comparator. No future event tape is read.
    """
    roles=np.full(env.config.max_agents,-1,np.int32)
    t=env.step_index
    for skill in (0,1):
        tasks=[k for k,p in enumerate(env.config.pois[:4]) if p.required_skill==skill
               and p.window[1]>=t]
        agents=[int(i) for i in np.flatnonzero(env.active) if env.skills[i]==skill]
        if not tasks or not agents: continue
        if len(tasks)==1:
            roles[agents]=tasks[0]
            continue
        a,b=tasks
        # If six are available, serve *both* coupling-3 tasks, by choosing
        # position-aware subset (not a fixed index / global assignment oracle
        # provided to any learned policy).
        if len(agents)>=6:
            candidates=[]
            for group in itertools.combinations(agents,3):
                other=[i for i in agents if i not in group]
                dist=sum(np.linalg.norm(env.positions[i]-env.poi_positions[a]) for i in group)
                dist+=sum(np.linalg.norm(env.positions[i]-env.poi_positions[b]) for i in other)
                candidates.append((dist,group,other))
            _,left,right=min(candidates,key=lambda x:x[0])
            roles[list(left)]=a;roles[list(right)]=b
        else:
            # With five capable rovers, 3+3 is impossible: serve the more
            # valuable *currently active* remaining task. If neither is
            # active, consider arrival before its public window closes.
            active_tasks=[k for k in tasks if env.config.pois[k].window[0]<=t]
            options=active_tasks if active_tasks else tasks
            best=max(options,key=lambda k:(env.config.pois[k].value,
                                           env.config.pois[k].window[1]-t))
            roles[agents]=best
    return roles


def run_one(scenario,seed,method,k=4):
    env=DeclaredIntentEnv(scenario)
    env.reset(seed=seed)
    previous=np.full(scenario.max_agents,-1,np.int32)
    selector=make_selector(method,np.random.default_rng(seed+101010)) if not method.startswith('privileged_') else None
    initial_oracle_roles=None
    trace=[]
    while not env.is_done:
        t=int(env.step_index)
        if t%k==0:
            if method.startswith('privileged_'):
                if initial_oracle_roles is None:
                    initial_oracle_roles=current_state_feasibility_planner(env)
                proposed=(initial_oracle_roles.copy() if method=='privileged_frozen'
                          else current_state_feasibility_planner(env))
            else:
                obs=observe_agents(env,previous)
                proposed=selector(obs,None,previous)
            previous=np.asarray(proposed,dtype=np.int32)
        previous[~env.active]=-1
        env.set_roles(previous)
        _,reward,_,_,info=env.step(control(env,previous))
        trace.append(dict(step=int(info['transition_step']),reward=float(reward),
                          served=info['poi_served'].tolist(),
                          serving_agents=info['serving_agents'].tolist(),
                          active=info['pre_active_mask'].tolist(),
                          choices=previous.tolist(),
                          events=info['events_next_boundary']))
    return physical_metrics(trace,scenario)


def analyze(base, seeds, variant='coordination_v24'):
    methods=('always_idle','fixed_initial_value','fixed_initial_nearest',
             'static_highest_value_repeatable','nearest','window_aware',
             'privileged_frozen','privileged_feasibility')
    rows=[]
    for seed in seeds:
        scenario=create_scenario(base,int(seed),variant)
        for method in methods:
            m=run_one(scenario,int(seed),method)
            rows.append(dict(seed=int(seed),method=method,raw_return=m['team_return'],
                coupled_steps=m['coupled_service_steps'],
                disrupted=m['disrupted_service_events'],
                active_restored=m['recovery_active_reallocation'],
                natural_reentry=m['recovery_natural_reentry'],
                censored=m['recovery_censored'],
                uninterrupted=m['faults_without_immediate_service_drop'],
                idle_fraction=m['idle_fraction'],
                one_shot=m['one_shot_completed']))
    summaries={}
    for method in methods:
        rr=[r for r in rows if r['method']==method]
        summaries[method]={key:float(np.mean([r[key] for r in rr])) for key in (
            'raw_return','coupled_steps','disrupted','active_restored','censored',
            'uninterrupted','idle_fraction','one_shot')}
    # This is a *necessary* benchmark preflight, not a sufficient proof of MARL.
    idle_zero=all(r['raw_return']==0 for r in rows if r['method']=='always_idle')
    static_best=max(summaries[x]['raw_return'] for x in
                    ('fixed_initial_value','fixed_initial_nearest',
                     'static_highest_value_repeatable'))
    witness=summaries['privileged_feasibility']['raw_return']
    frozen=summaries['privileged_frozen']['raw_return']
    report=dict(variant=variant,seeds=list(map(int,seeds)),
                metrics=rows,means=summaries,
                checks=dict(always_idle_zero=idle_zero,
                            reachable_positive=summaries['nearest']['raw_return']>0,
                            feasible_adaptation_gain_over_own_frozen_plan=witness>frozen,
                            privileged_feasibility_beats_static_local=witness>static_best,
                            diagnostic_planner_has_disrupted_tasks=
                              sum(r['disrupted'] for r in rows if r['method']=='privileged_feasibility')>0,
                            diagnostic_planner_has_active_restoration=
                              sum(r['active_restored'] for r in rows if r['method']=='privileged_feasibility')>0),
                interpretation='Privileged feasibility planner reads current global positions/skills. NOT a fair local baseline. No trained actor was run. Episode seeds are not independent RL training seeds.')
    return report


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--env',default='configs/rover_12.yaml')
    a.add_argument('--out',default='experiments/hr/v24_preflight')
    a.add_argument('--variant',default='coordination_v24',choices=['coordination_v24','coordination_v24_nominal','coordination_v24_stress'])
    a.add_argument('--seeds',nargs='+',type=int,default=[810101,810202,810303,810404,810505,810606,810707,810808])
    args=a.parse_args()
    result=analyze(load_config(args.env),args.seeds,args.variant)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'benchmark_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'matched_episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result['metrics'][0]));w.writeheader();w.writerows(result['metrics'])
    print(json.dumps(result['checks'],indent=2))
    for method,m in result['means'].items():
        print(f'{method:33s} return={m["raw_return"]:.3f}, disrupted={m["disrupted"]:.2f}, restored={m["active_restored"]:.2f}')
    if not result['checks']['always_idle_zero'] or not result['checks']['reachable_positive']:
        raise SystemExit('FAILED hard benchmark validity condition')
    if not result['checks']['feasible_adaptation_gain_over_own_frozen_plan']:
        raise SystemExit('WARNING: dynamic planner failed to beat fixed assignment; revise benchmark before RL')
if __name__=='__main__':main()
