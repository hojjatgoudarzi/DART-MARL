"""Additional independent probes of observation transport and progression."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from hr_mappo.envs.config import load_config
from hr_mappo.envs.rover import RoverEnv
from hr_mappo.config import HRConfig
from hr_mappo.communication import LocalChannel, CommConfig
from hr_mappo.curriculum import STAGES, GateState, stage_scenario
from hr_mappo.observations import observe_agents
from hr_mappo.models import TaskActor
from hr_mappo.rollout import tensorize, trajectory
from hr_mappo.experiment import setup
from hr_mappo.baselines import choose

BASE=load_config('configs/rover_12.yaml')


def test_delayed_message_private_and_stale_expiry():
    env=RoverEnv(BASE);env.reset(seed=12)
    env.positions[0]=[10,10];env.positions[1]=[11,10]
    env.positions[2]=[39,39]
    channel=LocalChannel(14,6,CommConfig(True,2,0.,3,5.),100)
    choices=np.full(14,-1);choices[0]=4
    channel.advance(env,choices)
    status,mask=channel.observations(0)
    assert not mask[1,0] and not mask[2,0]
    env.step_index=1;channel.advance(env,choices)
    assert not channel.observations(1)[1][1,0]
    env.positions[0]=[39,39]
    env.step_index=2;channel.advance(env,choices)
    status,mask=channel.observations(2)
    assert mask[1,0] and not mask[2,0]
    assert np.isclose(status[1,0,1],5/7)
    # Old information is a belief, not the ground-truth current state.
    env.active[0]=False
    env.step_index=6;channel.advance(env,choices)
    assert not channel.observations(6)[1][1,0]


def test_all_messages_dropped_and_no_oracle():
    env=RoverEnv(BASE)
    channel=LocalChannel(14,6,CommConfig(True,0,1.,5,100.),11)
    channel.advance(env,np.zeros(14,int))
    assert not channel.observations(0)[1].any()
    obs=observe_agents(env,np.full(14,-1),channel)
    assert obs['status_mask'].shape==(14,14)
    assert not obs['status_mask'].any()


def test_poi_completion_memory_is_private():
    env=RoverEnv(BASE);env.reset(seed=9)
    channel=LocalChannel(14,6,CommConfig(),4)
    env.positions[0]=np.asarray(BASE.pois[4].position)
    env.sensor_radii[0]=1
    env.positions[1]=[39,39];env.sensor_radii[1]=.01
    env.poi_completed[4]=True
    channel.advance(env,np.full(14,-1))
    env.positions[0]=[39,39];env.sensor_radii[0]=.01
    view=observe_agents(env,np.full(14,-1),channel)
    assert not view['legal'][0,4]
    assert view['legal'][1,4]


def test_observation_still_identical_under_hidden_remote_fault():
    env=RoverEnv(BASE);env.reset(seed=1)
    env.positions[0]=[0,0];env.sensor_radii[0]=.01
    channel=LocalChannel(14,6,CommConfig(True,3,0.,5,2.),4)
    channel.advance(env,np.full(14,-1))
    obs=observe_agents(env,np.full(14,-1),channel)
    env.active[11]=False;env.positions[11]=[39,39]
    view=observe_agents(env,np.full(14,-1),channel)
    for field in obs:
        np.testing.assert_array_equal(obs[field][0],view[field][0])
    actor=TaskActor(16)
    with torch.no_grad():
        x,_=actor.forward_step(tensorize(obs,'cpu'),torch.zeros(14,16))
        y,_=actor.forward_step(tensorize(view,'cpu'),torch.zeros(14,16))
    assert torch.equal(x[0],y[0])


def test_curriculum_config_and_gates():
    assert len(STAGES)==5
    for stage in STAGES:
        e=stage_scenario(BASE,101,stage)
        assert len(e.rovers)==12 and e.max_agents==14
    assert all(p.coupling==1 for p in stage_scenario(BASE,111,STAGES[0]).pois)
    assert all(not p.required_skill for p in stage_scenario(BASE,111,STAGES[1]).pois)
    assert not stage_scenario(BASE,111,STAGES[2]).events
    assert len(stage_scenario(BASE,111,STAGES[4]).events)==4
    gate=GateState()
    for _ in range(8):gate.record_update()
    assert not gate.assess(1,2,8,0)
    assert gate.index==0
    assert gate.assess(3,2,8,0)
    assert gate.index==1 and gate.updates_in_stage==0


def test_mask_expired_window_but_not_future_opening():
    env=RoverEnv(BASE);env.reset(seed=7)
    obs=observe_agents(env,np.full(14,-1))
    # POIs with future windows are legal to approach before they open.
    for j,p in enumerate(BASE.pois):
        if p.window[0]>0 and p.required_skill is None:
            assert obs['legal'][0,j]
    env.step_index=BASE.horizon-1
    later=observe_agents(env,np.full(14,-1))
    for j,p in enumerate(BASE.pois):
        if p.window[1]<env.step_index:
            assert not later['legal'][0,j]


def test_random_baseline_idle_possible_and_all_legal():
    env=RoverEnv(BASE);obs=observe_agents(env,np.full(14,-1))
    chosen=[]
    rng=np.random.default_rng(13)
    for _ in range(100):
        choice=choose(obs,np.full(14,-1),'random',rng)
        chosen.extend(choice[:12].tolist())
        for i,v in enumerate(choice):
            assert obs['legal'][i, v if v>=0 else -1]
    assert -1 in chosen


def test_nonzero_switch_cost_only_affects_train_reward():
    cfg=replace(HRConfig(),hidden=16,epochs=1,decision_interval=1,
                switch_penalty=.2)
    actor,critic,_=setup(cfg)
    # No need to claim any learning from this random network.
    ep=trajectory(actor,critic,RoverEnv(stage_scenario(BASE,400,'single_service')),400,cfg)
    total=sum(x['switch_cost'] for x in ep['records'])
    assert total>=0
    assert np.isclose(sum(x['raw_reward'] for x in ep['records']),ep['team_return'])
    assert np.isclose(sum(x['shaped_reward'] for x in ep['records']),ep['team_return']-total)


def test_stress_seed_reproducibility_and_skills():
    from hr_mappo.scenarios import create_scenario
    a=create_scenario(BASE,300,'stress')
    b=create_scenario(BASE,300,'stress')
    assert a==b and any(p.required_skill is not None for p in a.pois)
    assert len(a.events)==4


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):HRConfig(message_dropout=1.2)
    with pytest.raises(ValueError):HRConfig(curriculum_eval_every=0)
    with pytest.raises(ValueError):CommConfig(delay=-1)


def test_physical_attribution_neutral_ignores_roles():
    from hr_mappo.physics import make_env
    cfg=replace(HRConfig(),service_attribution='neutral')
    env=make_env(BASE,cfg)
    env.reset(seed=1)
    env.positions[0]=BASE.pois[0].position
    env.positions[1]=BASE.pois[0].position
    env.positions[2]=BASE.pois[0].position
    env.set_roles(np.array([0,0,0]+[-1]*11))
    a=env.preview_service()['reward']
    env.set_roles(np.array([-1]*14))
    b=env.preview_service()['reward']
    assert a==b
    with pytest.raises(ValueError):HRConfig(service_attribution='unknown')


def test_curriculum_checkpoint_resume_bitwise_cpu(tmp_path,monkeypatch):
    """Two-update split run must equal uninterrupted 2-update run."""
    from hr_mappo.experiment import train
    from hr_mappo.ppo import Learner
    cfg=replace(HRConfig(),hidden=16,epochs=1,updates=2,
                train_episodes_per_update=1,seed=917,
                curriculum=True,curriculum_eval_every=1,
                curriculum_min_updates=1,communication=True,
                message_delay=2,message_dropout=.15)
    env='configs/rover_12.yaml'
    full=tmp_path/'full'; split=tmp_path/'split'
    train(cfg,env,full)
    original=Learner.update
    calls=[0]
    def cutoff(self,episodes):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError('simulated interruption')
        return original(self,episodes)
    with monkeypatch.context() as m:
        m.setattr(Learner,'update',cutoff)
        with pytest.raises(RuntimeError,match='simulated interruption'):
            train(cfg,env,split)
    train(cfg,env,split,resume=split/'checkpoint.pt')
    for key in ('actor','critic'):
        left=torch.load(full/'checkpoint.pt',weights_only=False)[key]
        right=torch.load(split/'checkpoint.pt',weights_only=False)[key]
        assert all(torch.equal(left[k],right[k]) for k in left)
    assert len((split/'train_history.jsonl').read_text().splitlines())==2
