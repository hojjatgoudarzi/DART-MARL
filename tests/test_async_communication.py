"""Research correctness gates for v2.2 asynchronous CTDE and recovery task."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from hr_mappo.envs.config import load_config
from hr_mappo.config import HRConfig
from hr_mappo.experiment import setup
from hr_mappo.models import TaskActor
from hr_mappo.scheduler import DecisionScheduler
from hr_mappo.observations import observe_agents
from hr_mappo.scenarios import create_scenario
from hr_mappo.physics import make_env
from hr_mappo.rollout import trajectory, tensorize
from hr_mappo.gae import macro_gae
from hr_mappo.metrics import physical_metrics
from hr_mappo.baselines import choose

BASE=load_config('configs/rover_12.yaml')


def config(**kwargs):
    return replace(HRConfig(),hidden=16,epochs=1,decision_interval=8,
                   decision_mode='asynchronous',async_min_interval=4,**kwargs)


def test_local_trigger_cannot_read_future_fault_or_remote_truth():
    env=make_env(BASE,config())
    env.reset(seed=11)
    env.positions[0]=[1,1]
    env.sensor_radii[0]=2
    env.positions[11]=[38,38]
    obs=observe_agents(env,np.full(14,-1))
    scheduler=DecisionScheduler(14,8,'asynchronous',2)
    mask,sigs=scheduler.ready(obs,0,np.full(14,-1))
    assert mask[:12].all()
    scheduler.commit(0,mask,sigs,env.active)
    # Remote truth and the hidden event tape changed; local view unchanged.
    env.active[11]=False
    obs2=observe_agents(env,np.full(14,-1))
    np.testing.assert_array_equal(obs['ego'][0],obs2['ego'][0])
    decide,_=scheduler.ready(obs2,2,np.full(14,-1))
    assert not decide[0]
    # A directly sensed teammate suddenly disappears: a valid local trigger.
    env.positions[1]=[1.5,1.]
    obs3=observe_agents(env,np.full(14,-1))
    decide,_=scheduler.ready(obs3,2,np.full(14,-1))
    assert decide[0] and scheduler.pending_reasons[0]=='observable_change'


def test_async_macro_episode_and_actor_replay_exact():
    cfg=config(communication=True,message_delay=2,message_dropout=.1,chunk_length=11,burn_in=3)
    actor,critic,learner=setup(cfg)
    ep=trajectory(actor,critic,make_env(create_scenario(BASE,113,'recovery_challenge'),cfg),113,cfg)
    assert sum(r['duration'] for r in ep['records'])==120
    assert len(ep['physical'])==120
    assert ep['decisions']>120//cfg.decision_interval
    assert any(0<r['decide'].sum()<12 for r in ep['records'])
    assert all(r['duration']>=1 for r in ep['records'])
    assert any(x['reason']=='physical_failure_interrupt' for x in ep['options'])
    assert all(x['duration']>0 for x in ep['options'])
    assert all(x['duration']<=cfg.decision_interval for x in ep['options'])
    with torch.no_grad():
        hidden=torch.zeros(14,cfg.hidden)
        for r in ep['records']:
            dist,proposal=actor.distribution(tensorize(r['obs'],'cpu'),hidden)
            logged=dist.log_prob(torch.tensor(r['actions']))
            np.testing.assert_allclose(logged[r['decide']].numpy(),
                                       r['old_logp'][r['decide']],atol=2e-6)
            hidden=torch.where(torch.tensor(r['decide']).unsqueeze(-1),proposal,hidden)
            hidden=hidden*torch.tensor(r['memory_keep']).unsqueeze(-1)
    stats=learner.update([ep])
    assert all(np.isfinite(float(v)) for v in stats.values())
    assert stats['actor_grad_norm']>0


def test_chunked_vs_full_single_chunk_identical_on_same_seed():
    base=config(communication=False,chunk_length=0,burn_in=0)
    first,critic,learn=setup(base)
    ep=trajectory(first,critic,make_env(create_scenario(BASE,332,'recovery_challenge'),base),332,base)
    second,other,otherlearner=setup(replace(base,chunk_length=300,burn_in=4))
    second.load_state_dict(first.state_dict());other.load_state_dict(critic.state_dict())
    a=learn.update([ep]);b=otherlearner.update([ep])
    np.testing.assert_allclose(a['policy_loss'],b['policy_loss'],rtol=1e-6,atol=1e-6)
    for left,right in zip(first.parameters(),second.parameters()):
        torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)
    for left,right in zip(critic.parameters(),other.parameters()):
        torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)


def test_lambda_duration_is_physical_clock_when_asynchronous():
    gamma=.9;lam=.8
    a,_=macro_gae([1.,2.],[0.,0.],[0.,0.],[2,3],[False,True],gamma,lam,
                  lambda_per_step=True)
    np.testing.assert_allclose(a,[1.+gamma**2*lam**2*2.,2.],atol=1e-6)


def test_recovery_challenge_is_feasible_but_competitive():
    for seed in (2,113,2002):
        scene=create_scenario(BASE,seed,'recovery_challenge')
        assert len(scene.rovers)==12 and scene.max_agents==14
        assert [p.coupling for p in scene.pois]==[4,4,3,3,1,1]
        assert len(scene.events)==4
        assert sum(r.skill==0 for r in scene.rovers)==6
        assert sum(r.skill==1 for r in scene.rovers)==6
        assert scene.pois[0].required_skill==scene.pois[2].required_skill
        assert scene.pois[1].required_skill==scene.pois[3].required_skill
        assert scene.pois[0].coupling+scene.pois[2].coupling>6
        assert scene.pois[1].coupling+scene.pois[3].coupling>6
        assert create_scenario(BASE,seed,'recovery_challenge')==scene
        assert create_scenario(BASE,seed,'recovery_heldout')!=scene
    cfg=config()
    a,c,_=setup(cfg)
    scene=create_scenario(BASE,101,'recovery_challenge')
    ep=trajectory(a,c,make_env(scene,cfg),101,cfg,True,
                  lambda obs,env,prev:choose(obs,prev,'nearest'))
    met=physical_metrics(ep['physical'],scene)
    assert met['coupled_service_rate']<.85  # not a saturated benchmark
    assert met['recovery_events']
    assert any(x['recovered'] for x in met['recovery_events'])
    assert met['coupled_eligible_service_rate']>=met['coupled_service_rate']


def test_no_magic_knowledge_padded_slots_and_new_rover():
    cfg=config(communication=False)
    actor,critic,_=setup(cfg)
    a=trajectory(actor,critic,make_env(create_scenario(BASE,119,'recovery_challenge'),cfg),119,cfg,
                 deterministic=True)
    assert all(not r['decide'][13] for r in a['records'])
    assert all(r['actions'][13]==cfg.n_pois for r in a['records'])
    assert any('12' in r['decision_reasons'] for r in a['records'])
    assert all(int(x['duration'])<=cfg.decision_interval for x in a['options'])


def test_config_synchrony_and_chunk_guards():
    assert HRConfig(decision_interval=1).decision_mode=='synchronous'
    with pytest.raises(ValueError):HRConfig(decision_mode='asynchronous',decision_interval=2,async_min_interval=4)
    with pytest.raises(ValueError):HRConfig(chunk_length=-1)
    with pytest.raises(ValueError):HRConfig(curriculum_min_recovery_rate=1.5)


def test_async_two_update_interrupted_resume_exact(tmp_path,monkeypatch):
    from hr_mappo.experiment import train
    from hr_mappo.ppo import Learner
    cfg=replace(config(),hidden=16,updates=2,train_episodes_per_update=1,
                chunk_length=10,burn_in=3,communication=True,seed=1727,
                scenario='recovery_challenge')
    env='configs/rover_12.yaml'
    full=tmp_path/'complete';split=tmp_path/'restarted'
    train(cfg,env,full)
    orig=Learner.update
    times=[0]
    def break_second(self,episodes):
        times[0]+=1
        if times[0]==2:raise RuntimeError('intentional cutoff')
        return orig(self,episodes)
    with monkeypatch.context() as m:
        m.setattr(Learner,'update',break_second)
        with pytest.raises(RuntimeError,match='intentional cutoff'):
            train(cfg,env,split)
    train(cfg,env,split,resume=split/'checkpoint.pt')
    for name in ('actor','critic'):
        x=torch.load(full/'checkpoint.pt',weights_only=False)[name]
        y=torch.load(split/'checkpoint.pt',weights_only=False)[name]
        assert all(torch.equal(x[k],y[k]) for k in x)
    assert (split/'research_readiness.json').is_file()
