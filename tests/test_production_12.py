"""12 initially active rovers + 2 replacement slots with paper/demo provenance."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from dart_marl.algorithms.mappo.diagnostics import role_following_actions
from dart_marl.envs.config import load_config
from dart_marl.envs.rover import RoverEnv
from dart_marl.roles.allocator import AllocationConfig
from dart_marl.roles.coordinator import RoleCoordinator
from dart_marl.roles.evaluation import _same_task_problem

BASE = Path('configs/env/rover_team_12.yaml')
SHOW = Path('configs/env/rover_team_12_showcase.yaml')


def test_twelve_rovers_plus_replacement_reserve():
    cfg=load_config(BASE);env=RoverEnv(cfg)
    assert (len(cfg.rovers),cfg.max_agents,len(cfg.pois),cfg.horizon)==(12,14,6,120)
    assert int(env.active.sum())==12 and not env.active[12] and not env.active[13]


def test_showcase_compatibility_and_full_fault_tape():
    original=load_config(BASE);cfg=load_config(SHOW)
    assert _same_task_problem(original,cfg) and len(cfg.events)==8
    env=RoverEnv(cfg);env.reset(seed=31)
    received=[]
    while not env.is_done:
        _,_,done,truncated,info=env.step(role_following_actions(env))
        received.extend(info['events_next_boundary'])
    assert env.step_index==120
    assert sum(e['kind']=='replacement_agent' for e in received)==2
    assert sum(e['kind']=='automatic_reentry' for e in received)==3
    assert bool(env._ever_occupied[12]) and bool(env._ever_occupied[13])
    assert all(len(row)==2 for row in env.positions)


def test_allocator_padded_fourteen_slots_after_fault():
    cfg=load_config(SHOW);env=RoverEnv(cfg);env.reset(seed=31)
    coordinator=RoleCoordinator(AllocationConfig(method='switch_aware'))
    coordinator.reset(env)
    found=False
    while env.step_index<20:
        _,_,_,_,_=env.step(role_following_actions(env))
        result=coordinator.maybe_reassign(env)
        if env.step_index==19:
            assert result is not None
            assert len(env.roles)==14
            assert all(-1<=r<6 for r in env.roles)
            found=True
    assert found


def test_paper_generator_rejects_fabricated_unpaired_episodes(tmp_path):
    from scripts.make_paper_assets import allocator_assets, _start_plot
    data={'methods':{
        'switch_aware':{'per_episode':[{'seed':1,'return':3}]},
        'r_cart':{'per_episode':[{'seed':2,'return':2}]},
    }}
    p=tmp_path/'report.json';p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='not matched'):
        allocator_assets(p,tmp_path,_start_plot(),{'sources':{},'figures':[]})


def test_historical_figure_bootstrap_requires_controller_seed_pairs(tmp_path):
    from scripts.make_paper_assets import historical_assets, _start_plot
    p=tmp_path/'rows.csv'
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['controller_variant','training_seed','episode_seed','method','team_return'])
        writer.writeheader()
        writer.writerows([
            {'controller_variant':'fault_robust','training_seed':11,'episode_seed':9,'method':'switch_aware','team_return':1},
            {'controller_variant':'fault_robust','training_seed':22,'episode_seed':9,'method':'r_cart','team_return':2},
        ])
    with pytest.raises(ValueError,match='matched training seeds'):
        historical_assets(p,tmp_path,_start_plot(),{'sources':{},'figures':[]})


def test_default_demo_uses_true_initial_roster():
    cfg=load_config(SHOW)
    assert int(sum(1 for e in cfg.events if e.kind=='replacement_agent'))==2
    assert int(np.sum(np.asarray([cfg.max_agents - len(cfg.rovers)])))==2


def test_restartable_report_merge_verifies_seeds_controller(tmp_path):
    from scripts.merge_allocator_reports import merge
    parts={}
    for method, value in [('switch_aware',2.0),('r_cart',3.5)]:
        payload={
            'environment':'configs/env/rover_team_12_showcase.yaml',
            'evaluation_seeds':[10,20],
            'controller':{'checkpoint_sha256':'sha', 'training_seed':314159},
            'allocator_settings':{'forecast_horizon':24},
            'profile_latency':False,
            'methods':{method:{'per_episode':[{'seed':10,'return':value},{'seed':20,'return':value}]}}
        }
        p=tmp_path/(method+'.json');p.write_text(json.dumps(payload));parts[method]=p
    target=tmp_path/'merged.json'
    result=merge(parts,target)
    assert result['episodes_per_method']==2
    assert json.loads(target.read_text())['paired_differences']['r_cart_minus_switch_aware']['mean_difference']==pytest.approx(1.5)
    with pytest.raises(FileExistsError):
        merge(parts,target)
    altered=json.loads(parts['r_cart'].read_text())
    altered['controller']['checkpoint_sha256']='different'
    parts['r_cart'].write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='mismatched'):
        merge(parts,tmp_path/'other.json')
