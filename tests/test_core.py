"""Correctness gates: isolation, masking, SMDP, physics, reproducibility."""
import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from hr_mappo.envs.config import load_config,POISpec,EnvironmentConfig,RoverSpec
from hr_mappo.envs.rover import RoverEnv
from hr_mappo.config import HRConfig
from hr_mappo.observations import observe_agents,critic_state
from hr_mappo.models import TaskActor,TeamCritic
from hr_mappo.rollout import tensorize,trajectory
from hr_mappo.gae import macro_gae
from hr_mappo.control import control
from hr_mappo.scenarios import create_scenario
from hr_mappo.experiment import setup

BASE=load_config('configs/rover_12.yaml')


def test_original_baseline_and_contract():
    assert len(BASE.rovers)==12 and BASE.max_agents==14 and BASE.max_pois==6
    assert BASE.horizon==120
    e=RoverEnv(BASE); x=observe_agents(e,np.full(14,-1))
    assert x['ego'].shape==(14,8)
    assert x['tasks'].shape==(14,6,12)
    assert x['neighbors'].shape==(14,14,5)
    assert x['legal'].shape==(14,7)
    assert x['legal'][:12,-1].all() and (~x['legal'][12:,:-1]).all()


def test_legal_mask_and_entropy():
    e=RoverEnv(BASE);o=observe_agents(e,np.full(14,-1));o['legal'][:,:6]=False
    a=TaskActor(32)
    dist,h=a.distribution(tensorize(o,'cpu'),torch.zeros(14,32))
    assert (dist.sample()==6).all()
    assert torch.allclose(dist.entropy(),torch.zeros(14),atol=1e-5)
    assert torch.isfinite(h).all()


def test_local_counterfactual_firewall():
    e=RoverEnv(BASE)
    e.reset(seed=42)
    e.sensor_radii[0]=.1
    o1=observe_agents(e,np.full(14,-1))
    # A distant teammate's actual state, skill, activity and future fault tape
    # must not affect rover 0's observable data.
    e.positions[11]=[39,39];e.skills[11]=999;e.active[11]=False
    o2=observe_agents(e,np.full(14,-1))
    for name in ('ego','tasks','neighbors','neighbor_mask','legal'):
        np.testing.assert_array_equal(o1[name][0],o2[name][0])
    a=TaskActor(32)
    with torch.no_grad():
        l1,_=a.forward_step(tensorize(o1,'cpu'),torch.zeros(14,32))
        l2,_=a.forward_step(tensorize(o2,'cpu'),torch.zeros(14,32))
    assert torch.equal(l1[0],l2[0])


def test_hidden_one_shot_completion_not_globally_reported():
    e=RoverEnv(BASE);e.reset(seed=1)
    e.sensor_radii[0]=.01
    a=observe_agents(e,np.full(14,-1))
    e.poi_completed[4]=True
    b=observe_agents(e,np.full(14,-1))
    np.testing.assert_array_equal(a['legal'][0],b['legal'][0])
    np.testing.assert_array_equal(a['tasks'][0],b['tasks'][0])


def test_gae_option_discount_and_truncation():
    a,r=macro_gae([1+.9*2],[0.],[10.],[2],[False],.9,.95)
    assert np.isclose(a[0],2.8+.9**2*10)
    a2,r2=macro_gae([2.8],[0.],[10.],[2],[True],.9,.95)
    assert np.isclose(a2[0],2.8)
    _,full=macro_gae([2.,3.],[1.,1.],[1.,0.],[1,4],[False,True],.9,1.)
    assert np.isclose(full[0],2+.9*3)


def test_real_physical_coupling_and_roles():
    e=RoverEnv(BASE);e.reset(seed=2)
    e.positions[:3]=np.asarray(BASE.pois[0].position)
    e.set_roles(np.array([0]*3+[-1]*11))
    prev=e.preview_service()
    assert prev['served'][0]
    e.positions[2]=[20,20]
    assert not e.preview_service()['served'][0]
    # Merely marking 3 agents as assigned at a distant place must not pay.
    _,rew,_,_,info=e.step(np.zeros((14,2)))
    assert not info['poi_served'][0]


def test_deterministic_controller_no_task_alloc():
    e=RoverEnv(BASE);e.reset(seed=3)
    a=np.full(14,-1)
    np.testing.assert_array_equal(control(e,a),np.zeros((14,2)))
    a[0]=0
    assert np.linalg.norm(control(e,a)[0])>0


def test_scenario_seed_and_fault_independence():
    a=create_scenario(BASE,105,'random_faults')
    b=create_scenario(BASE,105,'random_faults')
    c=create_scenario(BASE,106,'random_faults')
    assert a==b and a!=c
    assert len(a.rovers)==12 and a.max_agents==14
    assert len(a.events)==4 and {e.kind for e in a.events} == {
      'temporary_disable','capability_degradation','replacement_agent','permanent_failure'}
    assert all(e.rover<12 for e in a.events)
    h=create_scenario(BASE,5,'heterogeneous')
    assert {p.required_skill for p in h.pois[:4]}=={0,1}


def test_forward_backward_and_masked_gradient():
    e=RoverEnv(BASE)
    ob=tensorize(observe_agents(e,np.full(14,-1)),'cpu')
    cs=tensorize(critic_state(e),'cpu')
    a=TaskActor(32);c=TeamCritic(32)
    dist,_=a.distribution(ob,torch.zeros(14,32))
    value,_=c.forward_step(cs,torch.zeros(1,32))
    loss=-dist.log_prob(torch.full((14,),6)).mean()+value.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None)
    assert all(torch.isfinite(p.grad).all() for p in c.parameters() if p.grad is not None)


def test_rollout_replay_ppo_and_agent_interrupt():
    cfg=replace(HRConfig(),hidden=32,epochs=1)
    a,c,l=setup(cfg)
    e=RoverEnv(create_scenario(BASE,101,'random_faults'))
    ep=trajectory(a,c,e,101,cfg)
    assert len(ep['physical'])==120 and len(ep['records'])==30
    assert any(not rec['memory_keep'].all() for rec in ep['records'])
    assert all(1<=r['duration']<=4 for r in ep['records'])
    d=l.update([ep]);assert np.isfinite(list(d.values())).all()


def test_rollout_fixed_seed_reproducible():
    cfg=replace(HRConfig(),hidden=32,epochs=1)
    a,c,_=setup(cfg)
    e1=RoverEnv(create_scenario(BASE,101))
    e2=RoverEnv(create_scenario(BASE,101))
    torch.manual_seed(222)
    p=trajectory(a,c,e1,101,cfg)
    torch.manual_seed(222)
    q=trajectory(a,c,e2,101,cfg)
    assert p['team_return']==q['team_return']
    assert [r['actions'].tolist() for r in p['records']]==[r['actions'].tolist() for r in q['records']]


def test_log_probability_replay_exact_pre_update():
    cfg=replace(HRConfig(),hidden=32,epochs=1)
    actor,critic,_=setup(cfg)
    ep=trajectory(actor,critic,RoverEnv(create_scenario(BASE,11)),11,cfg)
    h=torch.zeros(14,cfg.hidden)
    with torch.no_grad():
        for record in ep['records']:
            distribution,h=actor.distribution(tensorize(record['obs'],'cpu'),h)
            logged=distribution.log_prob(torch.tensor(record['actions']))
            np.testing.assert_allclose(logged.numpy(),record['old_logp'],atol=2e-6)
            h=h*torch.tensor(record['memory_keep']).unsqueeze(-1)


def test_physical_metric_censoring():
    from hr_mappo.metrics import physical_metrics
    trace=[dict(step=t,reward=1.,served=[True,False,False,False,False,False] if t<2 else
                        [False]*6,active=[True]*12+[False]*2,choices=[0]*12+[-1]*2,
                events=[{'step':2,'kind':'permanent_failure'}] if t==1 else []) for t in range(6)]
    metric=physical_metrics(trace,BASE)
    assert metric['team_return']==6
    assert metric['recovery_censored']==1
    assert metric['recovery_events'][0]['recovery_lag'] is None


def test_reentry_memory_preserve_ablation_actor_contract():
    actor=TaskActor(32)
    env=RoverEnv(BASE)
    obs=observe_agents(env,np.full(14,-1))
    before=torch.zeros((14,32));before[0]=1.
    env.active[0]=False
    obs2=tensorize(observe_agents(env,np.full(14,-1)),'cpu')
    _,after=actor.forward_step(obs2,before)
    assert torch.equal(after[0],before[0])
    # Reset policy is implemented via collector memory_keep instead of an
    # implicit reset in the Actor, so preserve and reset can be ablated.
