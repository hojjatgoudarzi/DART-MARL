"""Pre-training causal checks on an existing HR-MAPPO checkpoint.

Diagnostics use copies of the environment only OUTSIDE deployed-actor inference.
The constructive centralized repair is a feasibility witness, NOT an execution
baseline and NEVER an input or demonstration supplied to the actor.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from hr_mappo.envs.config import load_config
from .control import control
from .experiment import cfg_read, setup
from .metrics import physical_metrics
from .rollout import trajectory
from .physics import DeclaredIntentEnv
from .scenarios import create_scenario
from .interventions import ConditionalCoalitionFaultEnv
from .baselines import make_selector


def repair_witness(env, task_id: int, *, horizon: int = 65, sustain: int = 2):
    """Try a simple privileged *physical feasibility witness* after the cut.

    All agent choice changes happen in the CLONED simulator, never the rollout.
    Keeps existing coalition members, recruits nearby compatible survivors.
    A witness success is constructive; witness failure does not prove infeasibility.
    """
    model = copy.deepcopy(env)
    p = model.config.pois[task_id]
    compatible = [int(i) for i in np.flatnonzero(model.active)
                  if p.required_skill is None or model.skills[i] == p.required_skill]
    if len(compatible) < p.coupling:
        return dict(constructive_success=False, reason='insufficient_surviving_compatible_rovers',
                    compatible_survivors=len(compatible), horizon_ticks=0, first_restoration_step=None)
    model_roles = model.roles.copy()
    present = [i for i in compatible if model_roles[i] == task_id]
    present.sort(key=lambda i: np.linalg.norm(model.positions[i] - model.poi_positions[task_id]))
    remaining = [i for i in compatible if i not in present]
    remaining.sort(key=lambda i: np.linalg.norm(model.positions[i] - model.poi_positions[task_id]))
    chosen = (present + remaining)[:p.coupling]
    model_roles[chosen] = int(task_id)
    first = None
    consecutive = 0
    limit = min(int(model.config.horizon), int(p.window[1])+1, int(model.step_index)+int(horizon))
    tested = 0
    while not model.is_done and model.step_index < limit:
        model_roles[~model.active] = -1
        model.set_roles(model_roles)
        _, _, term, trunc, info = model.step(control(model, model_roles))
        tested += 1
        if bool(info['poi_served'][task_id]):
            if consecutive == 0:
                first = int(info['transition_step'])
            consecutive += 1
            if consecutive >= sustain:
                return dict(constructive_success=True, reason='physical_repair_constructed',
                            compatible_survivors=len(compatible), witness_rovers=chosen,
                            horizon_ticks=tested, first_restoration_step=first)
        else:
            first = None
            consecutive = 0
        if term or trunc:
            break
    return dict(constructive_success=False, reason='constructive_repair_not_found_within_window',
                compatible_survivors=len(compatible), witness_rovers=chosen,
                horizon_ticks=tested, first_restoration_step=None)


def _observable_snapshot(obs, i):
    """Only per-agent actor input, excluding privileged critic/global state."""
    return {k: np.asarray(obs[k][i]).copy() for k in
            ('ego','tasks','neighbors','neighbor_mask','status','status_mask','legal')}


def _change_keys(a, b):
    if a is None or b is None:
        return []
    return [k for k in a if not np.allclose(a[k],b[k],atol=1e-6)]


def diagnose_checkpoint(config, env_path, checkpoint, output, seeds=(810101,810202,810303),
                        intervention_step=50):
    """Run actual checkpoint, a no-fault counterfactual, and local baselines."""
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    cfg=replace(cfg_read(config),device='cpu')  # preflight is CPU and architecture identical
    base=load_config(env_path)
    actor,critic,_=setup(cfg)
    data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if data.get('config',{}).get('hidden') != cfg.hidden:
        raise ValueError('checkpoint architecture does not match config')
    actor.load_state_dict(data['actor'])
    critic.load_state_dict(data['critic'])
    actor.eval();critic.eval()
    rows=[]
    all_details={}
    for seed in seeds:
        nominal=create_scenario(base,int(seed),'coordination_v24_nominal')
        records={}
        # At every tick retain ONLY each agent's legitimate local observation.
        # Counterfactual difference is exact at t=50 since prefault tapes match.
        def collector(label):
            def hook(*,env,t,obs,decide,previous,signatures):
                if intervention_step-1 <= t <= intervention_step+20:
                    records.setdefault(label,{})[int(t)] = dict(
                        active=np.asarray(obs['active']).copy(),
                        decide=np.asarray(decide).copy(),
                        choices=np.asarray(previous).copy(),
                        views=[_observable_snapshot(obs,i) for i in range(cfg.n_agents)])
                if label=='fault' and t==intervention_step and env.intervention_status and env.intervention_status.get('opportunity'):
                    target=int(env.intervention_status['task'])
                    records['witness']=repair_witness(env,target)
            return hook
        no_fault=trajectory(actor,critic,
            DeclaredIntentEnv(nominal),
            int(seed),cfg,deterministic=True,audit_hook=collector('no_fault'))
        fault_env=ConditionalCoalitionFaultEnv(nominal,intervention_step=intervention_step,
                                               mode='minimal_feasible_cut')
        fault=trajectory(actor,critic,fault_env,int(seed),cfg,deterministic=True,
                         audit_hook=collector('fault'))
        # The counterfactual comparison at the interruption is only causal
        # if both policies experienced identical pre-fault local histories.
        f_before=records.get('fault',{}).get(intervention_step-1)
        n_before=records.get('no_fault',{}).get(intervention_step-1)
        if f_before is None or n_before is None:
            raise AssertionError('missing pre-intervention observations')
        for i in range(cfg.n_agents):
            if not np.array_equal(f_before['active'],n_before['active']) or \
               _change_keys(f_before['views'][i],n_before['views'][i]):
                raise AssertionError('pre-fault local histories differ: causal comparison invalid')
        metric=physical_metrics(fault['physical'],nominal)
        status=fault_env.intervention_status or {}
        victims=set(map(int,status.get('victims',[])))
        # At the intervention step compare true local observations of each
        # surviving rover under counterfactual no-fault vs real fault.
        tf=records.get('fault',{}).get(intervention_step)
        tn=records.get('no_fault',{}).get(intervention_step)
        own_change=[];any_change=[];due_observer=[];first_signal={};first_due={}
        for i in range(cfg.n_agents):
            if i in victims or tf is None or not tf['active'][i]:
                continue
            keys=_change_keys(tf['views'][i],tn['views'][i]) if tn is not None else []
            if keys:
                any_change.append(i)
                if tf['decide'][i]:due_observer.append(i)
                own_change.append(dict(rover=i,changed_features=keys,decision_due=bool(tf['decide'][i]),
                                       decision_task=int(tf['choices'][i])))
            for t in range(intervention_step,intervention_step+21):
                f=records.get('fault',{}).get(t)
                n=records.get('no_fault',{}).get(t)
                if f is None or n is None or not f['active'][i]:continue
                if i not in first_signal and _change_keys(f['views'][i],n['views'][i]):first_signal[i]=t
                if i not in first_due and f['decide'][i]:first_due[i]=t
        # Monitor the causal record, not merely whether some fault happened.
        affected=int(status['task']) if status.get('opportunity') else None
        after=[x for x in fault['physical'] if intervention_step <= int(x['step'])<=intervention_step+20]
        after_decision_changes=sum(bool(x.get('decision_rovers')) for x in after)
        # Immediate threshold-game credit is zero for an interrupted task
        # until physical coupling resumes. We record *both* credit designs
        # and incomplete coalitions as diagnostic signals (not actor inputs).
        target_partial_steps=0
        target_shapley_positive_steps=0
        target_pivotal_positive_steps=0
        target_actual_rewards=0.
        target_reassignment_choices=0
        different_task_decision_steps=0
        if affected is not None:
            coupling=int(nominal.pois[affected].coupling)
            for sample in after:
                idx=int(sample['step'])
                n=int(sample['poi_coverage'][affected])
                served=bool(sample['served'][affected])
                r=float(sample['poi_rewards'][affected])
                target_partial_steps+=int(0<n<coupling and not served)
                target_shapley_positive_steps+=int(served and r>0)
                target_pivotal_positive_steps+=int(served and r>0 and n==coupling)
                target_actual_rewards+=r
                cur=records.get('fault',{}).get(idx)
                pre=records.get('no_fault',{}).get(idx)
                if cur and pre:
                    for i in range(cfg.n_agents):
                        if i in victims or not cur['active'][i]:continue
                        different_task_decision_steps+=int(cur['decide'][i] and
                             int(cur['choices'][i])!=int(pre['choices'][i]))
                        if cur['decide'][i] and int(cur['choices'][i])==affected and \
                           int(pre['choices'][i])!=affected:
                            target_reassignment_choices+=1
        row=dict(seed=int(seed),intervention_opportunity=bool(status.get('opportunity')),
                 target_poi=affected,victims=sorted(victims),
                 structurally_feasible=status.get('structural_feasibility_after_cut'),
                 immediate_drop=bool(metric['disrupted_service_events']>0),
                 actual_active_restoration=int(metric['recovery_active_reallocation']),
                 recovery_rate=metric['active_recovery_rate'],
                 first_tick_survivors_with_changed_local_input=len(any_change),
                 first_tick_changed_and_deciding=len(due_observer),
                 any_survivor_with_changed_local_input=len(first_signal),
                 due_decisions_21_ticks=after_decision_changes,
                 post_fault_actor_return=float(fault['team_return']),
                 no_fault_actor_return=float(no_fault['team_return']),
                 constructive_repair_possible=records.get('witness',{}).get('constructive_success'),
                 witness_restoration_step=records.get('witness',{}).get('first_restoration_step'),
                 postfault_partial_target_service_steps=target_partial_steps,
                 postfault_positive_shapley_credit_steps=target_shapley_positive_steps,
                 postfault_positive_pivotal_credit_steps=target_pivotal_positive_steps,
                 postfault_physical_reward_from_target=target_actual_rewards,
                 postfault_task_choices_differing_from_no_fault=different_task_decision_steps,
                 postfault_reassignments_to_target=target_reassignment_choices)
        rows.append(row)
        all_details[str(seed)]=dict(row=row,intervention=status,
            constructive_witness=records.get('witness',{}),
            first_tick_observable_changes=own_change,
            first_signal_steps={str(k):v for k,v in first_signal.items()},
            first_redecision_steps={str(k):v for k,v in first_due.items()},
            recovery_events=metric['recovery_events'],
            warning='Witness uses full simulator for DIAGNOSTIC ONLY; failed witness is not an infeasibility proof.')
    # Evidence gate: we do not pretend that agent observation necessarily
    # reveals an absent rover remotely or guarantee physical feasibility.
    opportunities=sum(int(x['intervention_opportunity']) for x in rows)
    drops=sum(int(x['immediate_drop']) for x in rows)
    witness=sum(x['constructive_repair_possible'] is True for x in rows)
    observers=sum(int(x['any_survivor_with_changed_local_input']>0) for x in rows)
    decisions=sum(int(x['due_decisions_21_ticks']>0) for x in rows)
    summary=dict(checkpoint=str(checkpoint),seeds=list(map(int,seeds)),cases=len(rows),
        valid_interventions=opportunities,actual_drops=drops,
        physical_witness_successes=witness,observable_change_cases=observers,
        post_fault_decision_cases=decisions,
        audit_integrity_passed=all(x['due_decisions_21_ticks']>0 for x in rows),
        training_ready=bool(drops>0 and witness>0 and observers>0 and decisions>0),
        caveat='training_ready means diagnostic checks found *some* feasible, observable events; it is NOT proof of recovery learning or numerical correctness across all cases.')
    (out/'phase_a_details.json').write_text(json.dumps(all_details,indent=2,default=str)+'\n')
    (out/'phase_a_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import csv
    with (out/'phase_a_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return summary
