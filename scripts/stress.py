#!/usr/bin/env python3
"""Conditional intervention diagnostic (NOT comparable fixed-fault return).

Trained policy may lack eligible serviced tasks at the predeclared boundary;
that yields no intervention and undefined recovery rate, not a failure.
All original v1.3.1 and v2.3 benchmarks remain unchanged.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from hr_mappo.envs.config import load_config
from hr_mappo.experiment import cfg_read,setup
from hr_mappo.scenarios import create_scenario
from hr_mappo.interventions import ConditionalCoalitionFaultEnv
from hr_mappo.rollout import trajectory
from hr_mappo.baselines import make_selector
from hr_mappo.metrics import physical_metrics


def run(cfg,envpath,checkpoint,out,seeds,methods,intervention_mode="single"):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    base=load_config(envpath)
    actor,critic,_=setup(cfg)
    data=torch.load(checkpoint,map_location=cfg.device,weights_only=False)
    actor.load_state_dict(data['actor'] if 'actor' in data else data)
    if 'critic' in data:critic.load_state_dict(data['critic'])
    actor.eval();critic.eval()
    results=[]
    for seed in seeds:
        sc=create_scenario(base,int(seed),'coordination_v24_nominal')
        for method in methods:
            env=ConditionalCoalitionFaultEnv(sc,mode=intervention_mode)
            selector=None if method=='actor' else make_selector(method,np.random.default_rng(seed+12049))
            ep=trajectory(actor,critic,env,int(seed),cfg,deterministic=True,selector=selector)
            met=physical_metrics(ep['physical'],sc)
            status=env.intervention_status
            assert status is not None
            row=dict(method=method,seed=int(seed),return_raw=float(ep['team_return']),
                     opportunity=bool(status['opportunity']),
                     immediate_service_drop=bool(met['recovery_opportunities']>0),
                     active_restoration=int(met['recovery_active_reallocation']),
                     natural_reentry=int(met['recovery_natural_reentry']),
                     censored=int(met['recovery_censored']),
                     recovery_rate=met['active_recovery_rate'],
                     intervention_step=int(status['step']),
                     intervention_poi=status.get('task'),
                     target_rover=status.get('rover'),
                     victims=status.get('victims',[]),mode=intervention_mode,
                     structural_feasibility_after_cut=status.get('structural_feasibility_after_cut'))
            results.append(row)
            (out/f'{method}_{seed}_metrics.json').write_text(json.dumps(dict(intervention=status,
                      metrics=met),indent=2)+'\n')
            print(row,flush=True)
    with (out/'intervention_episode_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    summary={}
    for method in methods:
        rs=[r for r in results if r['method']==method]
        exposure=[r for r in rs if r['immediate_service_drop']]
        summary[method]=dict(episodes=len(rs),opportunities=sum(r['opportunity'] for r in rs),
             drops=len(exposure),active_recovery_count=sum(r['active_restoration'] for r in rs),
             conditional_recovery_rate=(sum(r['active_restoration']>0 for r in exposure)/len(exposure)
                                         if exposure else None),
             feasible_interventions=sum(r['opportunity'] for r in rs),
             unconditional_fault_continuity=sum(r['opportunity'] and not r['immediate_service_drop'] for r in rs),
             no_feasible_intervention=sum(not r['opportunity'] for r in rs),
             mean_raw_return=float(np.mean([r['return_raw'] for r in rs])))
    (out/'intervention_summary.json').write_text(json.dumps(dict(summary=summary,
       mode=intervention_mode,
       warning='Targets are POLICY-CONDITIONED, sometimes MULTI-ROVER, and may differ between policies. Structural feasibility does not guarantee recovery. Never use conditional raw returns to rank policies or pool with fixed-fault results.'),indent=2)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--env',default='configs/rover_12.yaml')
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[810101,810202,810303,810404,810505])
    ap.add_argument('--intervention-mode',choices=['single','minimal_feasible_cut'],default='single')
    ap.add_argument('--methods',nargs='+',default=['actor','fixed_initial_value','fixed_initial_nearest','nearest','window_aware'])
    a=ap.parse_args()
    result=run(cfg_read(a.config),a.env,a.checkpoint,a.output,a.seeds,a.methods,a.intervention_mode)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
