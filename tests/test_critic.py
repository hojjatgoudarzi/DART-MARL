"""legacy_study: finite-horizon correctness, critic learning, and fair fault suites."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dart_marl.algorithms.mappo.critic import TemporalTeamCritic, build_critic
from dart_marl.algorithms.mappo.ppo import PPOConfig
from dart_marl.algorithms.mappo.trainer import Collector, TrainConfig, load_train_config, train
from dart_marl.envs.config import config_from_mapping, load_config
from dart_marl.roles.generalization import evaluate_fault_suite


def test_v2_masking_agent_permutation_and_team_broadcast():
    torch.manual_seed(7)
    model = TemporalTeamCritic(hidden=24, return_scale=64.)
    agents = torch.rand(2, 5, 9)
    tasks = torch.rand(2, 4, 10)
    active = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 0, 0]], dtype=torch.bool)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    val = model(agents, tasks, active, mask)
    assert torch.all(val[~active] == 0)
    assert torch.allclose(val[0, active[0]], val[0, active[0]][0].expand(3))
    altered_agents = agents.clone(); altered_agents[~active] = 1e5
    altered_tasks = tasks.clone(); altered_tasks[~mask] = -1e5
    torch.testing.assert_close(val, model(altered_agents, altered_tasks, active, mask))
    perm = torch.tensor([4, 2, 0, 1, 3])
    torch.testing.assert_close(val[:, perm], model(agents[:, perm], tasks, active[:, perm], mask))


def test_v2_time_and_position_sensitivity():
    env_cfg = load_config('configs/env/dev_roles.yaml')
    from dart_marl.envs.rover import RoverEnv
    from dart_marl.algorithms.mappo.state import critic_input
    env = RoverEnv(env_cfg); env.reset(seed=551)
    from dart_marl.algorithms.mappo.roles import assign_initial_roles
    assign_initial_roles(env, 'initial_coupling')
    state = critic_input(env)
    x = torch.tensor(state.agents[None]); y=torch.tensor(state.pois[None])
    a=torch.tensor(state.active[None]); t=torch.tensor(state.task_mask[None])
    z = build_critic(architecture='temporal_team_v2', hidden=24, return_scale=64.)
    v=z(x,y,a,t)
    late=x.clone(); late[...,8]=late[...,8]+.6
    assert not torch.allclose(v,z(late,y,a,t),atol=1e-7)
    assert isinstance(build_critic(architecture='agent_v1'),torch.nn.Module)
    with pytest.raises(ValueError): build_critic(architecture='unknown')


def test_finite_horizon_is_terminal_for_new_collector():
    original = load_config('configs/env/learning_sanity.yaml')
    data = asdict(original)
    data['schema_version'] = 1
    data['horizon'] = 1
    for p in data['pois']:
        p['window'] = (0,0)
    env = config_from_mapping(data)
    # The collector's `terminated` means *no bootstrap* for GAE, even if the
    # environment technically returned truncated=True at a known finite horizon.
    base=TrainConfig(env_config='unused',num_envs=1,rollout_steps=1,role_mode='none',
                     treat_horizon_as_terminal=True)
    from dart_marl.algorithms.mappo.actor import SharedActor
    from dart_marl.algorithms.mappo.ppo import PPOLearner
    from dart_marl.envs.rover import RoverEnv
    agent=PPOLearner(SharedActor(RoverEnv(env).observation_size, hidden=16),
                     build_critic(hidden=16),PPOConfig(),torch.device('cpu'))
    batch,_=Collector(env,base).collect(agent)
    assert bool(batch['done'][0,0])
    assert bool(batch['terminated'][0,0])
    assert bool(batch['next_active'][0,0].any())
    old_batch,_=Collector(env,replace(base,treat_horizon_as_terminal=False)).collect(agent)
    assert not bool(old_batch['terminated'][0,0])


def test_configs_and_legacy_compatibility():
    v2=load_train_config('configs/train/mappo_team_value.yaml')
    assert v2.critic_arch=='temporal_team_v2'
    assert v2.treat_horizon_as_terminal
    assert v2.ppo.value_target_scale == 64.
    old=load_train_config('configs/train/mappo_role_guided.yaml')
    assert old.critic_arch=='agent_v1' and not old.treat_horizon_as_terminal
    assert old.ppo.value_target_scale == 1.
    with pytest.raises(ValueError): replace(v2, ppo=replace(v2.ppo,value_target_scale=1.))
    with pytest.raises(ValueError): PPOConfig(value_target_scale=0)


@pytest.fixture(scope='module')
def small_train(tmp_path_factory):
    path = tmp_path_factory.mktemp('critic')
    base=asdict(load_train_config('configs/train/mappo_team_smoke.yaml'))
    base.update(num_envs=2,rollout_steps=8,total_env_steps=32,eval_episodes=2,
                eval_every=2,save_every=2)
    base['ppo'].update(ppo_epochs=1,num_minibatches=1)
    config=path/'config.yaml'
    config.write_text(yaml.safe_dump({'schema_version':1,**base}))
    result=train(config,path/'training')
    return path,result


def test_v2_end_to_end_checkpoint_and_audit(small_train):
    path, summary = small_train
    assert summary['env_steps']==32
    import json
    from dart_marl.algorithms.mappo.critic_audit import audit_checkpoint
    hist=[json.loads(s) for s in (path/'training/train_history.jsonl').read_text().splitlines()]
    assert np.isfinite([r['value_explained_variance'] for r in hist]).all()
    report = audit_checkpoint(path/'training/checkpoints/latest.pt','configs/env/dev_roles.yaml',
                              'configs/env/dev_roles.yaml',episodes=2,seed=76420,device='cpu')
    assert report['critic_arch']=='temporal_team_v2'
    assert np.isfinite(report['critic_rmse'])


def test_fault_suite_pairing_protection(small_train,tmp_path):
    path,_=small_train
    checkpoint=path/'training/checkpoints/latest.pt'
    kwargs=dict(checkpoints=[checkpoint],training_env_config='configs/env/dev_roles.yaml',
                scenarios=['configs/env/dev_failure.yaml','configs/env/analytic_repeated_faults.yaml'],
                episodes=2,seed=991001,device='cpu',methods=('fixed','switch_aware','r_cart'))
    first=evaluate_fault_suite(out=tmp_path/'first.json',**kwargs)
    second=evaluate_fault_suite(out=tmp_path/'second.json',**kwargs)
    assert first==second
    assert first['training_seed_count']==1
    assert all(v['r_cart']['training_seed_sample_std'] is None for v in first['summary'].values())
    assert len(first['results'])==2
    with pytest.raises(ValueError,match='duplicate independent training seed'):
        evaluate_fault_suite(out=tmp_path/'duplicate.json',**(kwargs|{'checkpoints':[checkpoint,checkpoint.parent/'update_000002.pt']}))
    with pytest.raises(ValueError,match='TEST_ONLY'):
        evaluate_fault_suite(out=tmp_path/'leak.json',**(kwargs|{'scenarios':['configs/env/historical_fault_stress.yaml']}))
    with pytest.raises(ValueError,match='geometry/dynamics'):
        evaluate_fault_suite(out=tmp_path/'invalid.json',**(kwargs|{'scenarios':['configs/env/stress_12.yaml']}))


def test_backward_compatible_v1_resume_strict_settings(tmp_path):
    from dart_marl.algorithms.mappo.trainer import _load_checkpoint, set_seeds
    from dart_marl.algorithms.mappo.ppo import PPOLearner
    from dart_marl.algorithms.mappo.actor import SharedActor
    from dart_marl.algorithms.mappo.trainer import _env_sha
    env_cfg=load_config('configs/env/learning_sanity.yaml')
    cfg=TrainConfig(env_config='configs/env/learning_sanity.yaml',
                    role_mode='none',num_envs=1,rollout_steps=2,hidden_size=16)
    set_seeds(1)
    from dart_marl.envs.rover import RoverEnv
    agent=PPOLearner(SharedActor(RoverEnv(env_cfg).observation_size,16),build_critic(hidden=16),
                     cfg.ppo,torch.device('cpu'))
    collector=Collector(env_cfg,cfg)
    from dart_marl.algorithms.mappo.trainer import _save_checkpoint
    path=tmp_path/'v1.pt'
    sha=_env_sha(cfg.env_config)
    _save_checkpoint(path,agent,collector,cfg,sha,2,4)
    old=torch.load(path,map_location='cpu',weights_only=False)
    for key in ('critic_arch','treat_horizon_as_terminal'):
        old['train_config'].pop(key)
    for key in ('critic_learning_rate','value_target_scale'):
        old['train_config']['ppo'].pop(key)
    torch.save(old,path)
    assert _load_checkpoint(path,agent,collector,cfg,sha)==(2,4)
    with pytest.raises(ValueError, match='treat_horizon_as_terminal'):
        _load_checkpoint(path,agent,collector,replace(cfg,treat_horizon_as_terminal=True),sha)


def test_fault_diverse_training_sampling_and_visibility():
    from dart_marl.disruptions.sampling import diverse_scenario
    from dart_marl.envs.rover import RoverEnv
    from dart_marl.algorithms.mappo.state import critic_input
    base=load_config('configs/env/dev_roles.yaml')
    sizes=set(); kinds=set()
    for seed in range(800,816):
        cfg=diverse_scenario(base,seed)
        assert cfg.events==diverse_scenario(base,seed).events
        sizes.add(len(cfg.events));kinds.update(e.kind for e in cfg.events)
        a=RoverEnv(base);b=RoverEnv(cfg)
        a.reset(seed=seed);b.reset(seed=seed)
        np.testing.assert_array_equal(a.observe(), b.observe())
        np.testing.assert_array_equal(critic_input(a).agents,critic_input(b).agents)
    assert sizes=={0,1,2,3}
    assert len(kinds)>=3


def test_fault_collector_checkpoint_restores_episode_tapes():
    from dart_marl.disruptions.sampling import diverse_scenario
    cfg=load_train_config('configs/train/mappo_fault_robust.yaml')
    cfg=replace(cfg,num_envs=4,rollout_steps=4)
    base=load_config(cfg.env_config)
    collector=Collector(base,cfg)
    assert len({len(env.config.events) for env in collector.envs})>=2
    original_tapes=[e.config.events for e in collector.envs]
    snapshots=collector.checkpoint_state()
    fresh=Collector(base,cfg)
    fresh.load_state(snapshots)
    assert [env.config.events for env in fresh.envs]==original_tapes
    assert all(e.snapshot().step_index == f.snapshot().step_index for e,f in zip(collector.envs,fresh.envs))
    assert all(np.array_equal(e.positions,f.positions) for e,f in zip(collector.envs,fresh.envs))
    assert cfg.fault_mode=='diverse'
    with pytest.raises(ValueError,match='missing episode scenario'):
        legacy=dict(snapshots);legacy.pop('scenario_configs')
        fresh.load_state(legacy)


def test_fault_robust_resume_matches_uninterrupted_cpu(tmp_path):
    from dart_marl.algorithms.mappo.trainer import train
    original=asdict(load_train_config('configs/train/mappo_fault_smoke.yaml'))
    original.update(env_config='configs/env/dev_roles.yaml',seed=44,num_envs=2,
                    rollout_steps=8,total_env_steps=32,torch_threads=2,eval_episodes=2,
                    eval_every=2,save_every=2)
    original['ppo'].update(ppo_epochs=1,num_minibatches=1)
    full=tmp_path/'full.yaml';full.write_text(yaml.safe_dump({'schema_version':1,**original}))
    train(full,tmp_path/'full')
    original['total_env_steps']=16
    half=tmp_path/'half.yaml';half.write_text(yaml.safe_dump({'schema_version':1,**original}))
    train(half,tmp_path/'first')
    original['total_env_steps']=32
    half.write_text(yaml.safe_dump({'schema_version':1,**original}))
    train(half,tmp_path/'second',resume=tmp_path/'first/checkpoints/latest.pt')
    a=torch.load(tmp_path/'full/checkpoints/latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(tmp_path/'second/checkpoints/latest.pt',map_location='cpu',weights_only=False)
    for section in ('actor','critic'):
        for param in a[section]:
            torch.testing.assert_close(a[section][param],b[section][param],rtol=0,atol=0)
    for i,j in zip(a['collector']['scenario_configs'], b['collector']['scenario_configs']):
        assert i.events==j.events


def test_development_only_checkpoint_selection(small_train,tmp_path):
    from dart_marl.algorithms.mappo.selection import select_critic_checkpoint
    path,_=small_train
    checkpoint=path/'training/checkpoints/latest.pt'
    kw=dict(checkpoints=[checkpoint], training_env_config='configs/env/dev_roles.yaml',
        scenarios=['configs/env/dev_failure.yaml'],episodes=2,seed=819000,device='cpu')
    selected=select_critic_checkpoint(out=tmp_path/'selection.json',**kw)
    assert selected['selected_training_step']==32
    assert np.isfinite(selected['entries'][0]['mean_relative_rmse'])
    assert selected['selected_checkpoint_sha256']
    with pytest.raises(ValueError,match='TEST_ONLY'):
        select_critic_checkpoint(out=tmp_path/'leak.json',**(kw|{
            'scenarios':['configs/env/historical_fault_stress.yaml']}))
    with pytest.raises(ValueError,match='duplicate checkpoint'):
        select_critic_checkpoint(out=tmp_path/'duplicate.json',**(kw|{
            'checkpoints':[checkpoint, checkpoint]}))
