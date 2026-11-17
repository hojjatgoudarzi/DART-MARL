"""v2.3 contract, controlled causal recovery, diagnostics, PPO and provenance."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from hr_mappo.envs.config import load_config
from hr_mappo.disruptions.events import DisruptionEvent
from hr_mappo.config import HRConfig
from hr_mappo.experiment import setup
from hr_mappo.scenarios import create_scenario
from hr_mappo.scenarios_heterogeneous import scenario_v23,STAGE_COUPLING
from hr_mappo.curriculum import stage_scenario,STAGES
from hr_mappo.physics import make_env,marginal_physical_credit,service_potential
from hr_mappo.control import control
from hr_mappo.rollout import trajectory
from hr_mappo.metrics import physical_metrics
from hr_mappo.baselines import choose

BASE=load_config('configs/rover_12.yaml')


def cfg(**kw):
    return replace(HRConfig(),hidden=16,epochs=1,decision_interval=8,
                   decision_mode='asynchronous',async_min_interval=4,
                   service_attribution='declared',scenario='coordination_v23',**kw)


def test_no_prepositioned_coalition_and_strict_12_plus_2():
    for seed in (101,202,303,404,900101):
        sc=scenario_v23(BASE,seed)
        assert len(sc.rovers)==12 and sc.max_agents==14
        assert [p.coupling for p in sc.pois]==[4,4,3,3,1,1]
        assert [sum(r.skill==v for r in sc.rovers) for v in (0,1)]==[6,6]
        assert sc.events==scenario_v23(BASE,seed).events
        assert sc==create_scenario(BASE,seed,'coordination_v23')
        for r in sc.rovers:
            assert all(np.linalg.norm(np.asarray(r.start)-np.asarray(p.position))>p.radius+2.
                       for p in sc.pois)
        env=make_env(sc,cfg());env.reset(seed=seed)
        assert env.preview_service()['reward']==0
        assert scenario_v23(BASE,seed,heldout=True)!=sc
        # Even after permanent + temporary failure there are enough
        # skilled survivors to service either task of each skill.
        for p in sc.pois[:4]:
            assert sum(r.skill==p.required_skill for r in sc.rovers)-2>=p.coupling


def test_always_idle_reward_exactly_zero_while_local_selector_can_score():
    conf=cfg()
    actor,critic,_=setup(conf)
    for seed in (810101,810202):
        sc=scenario_v23(BASE,seed)
        idle=trajectory(actor,critic,make_env(sc,conf),seed,conf,True,
                        lambda obs,env,prev:choose(obs,prev,'always_idle'))
        assert idle['team_return']==0
        assert all(not any(x['served']) for x in idle['physical'])
    sc=scenario_v23(BASE,810101)
    nearest=trajectory(actor,critic,make_env(sc,conf),810101,conf,True,
                       lambda obs,env,prev:choose(obs,prev,'nearest'))
    assert nearest['team_return']>0


@pytest.mark.parametrize('needed',[1,2,3,4])
def test_declared_intent_requires_exact_physical_coupling(needed):
    scene=scenario_v23(BASE,119,faults=False)
    p=replace(scene.pois[0],coupling=needed,window=(0,119))
    scene=replace(scene,pois=(p,*scene.pois[1:]))
    env=make_env(scene,cfg());env.reset(seed=119)
    pool=[i for i,r in enumerate(scene.rovers) if r.skill==p.required_skill]
    coords=np.array(p.position)
    env.positions[pool[:needed]]=coords+np.arange(needed)[:,None]*np.array([.12,.08])
    roles=np.full(14,-1,np.int32)
    roles[pool[:needed-1]]=0
    env.set_roles(roles)
    assert not env.preview_service()['served'][0]
    # IDLE inside radius cannot serve even if enough bodies physically there.
    roles[pool[needed-1]]=0
    env.set_roles(roles)
    assert env.preview_service()['served'][0]
    env.positions[pool[0]]=coords+np.array([p.radius+1.,0.])
    assert not env.preview_service()['served'][0]


def test_original_mode_preserves_idle_physical_presence():
    scene=scenario_v23(BASE,121,faults=False)
    legacy=make_env(scene,replace(cfg(),service_attribution='intent'))
    strict=make_env(scene,cfg())
    skill=scene.pois[0].required_skill
    indices=[i for i,r in enumerate(scene.rovers) if r.skill==skill][:4]
    for env in (legacy,strict):
        env.reset(seed=121)
        env.positions[indices]=np.array(scene.pois[0].position)
        env.set_roles(np.full(14,-1,np.int32))
    assert legacy.preview_service()['served'][0]
    assert not strict.preview_service()['served'][0]


def test_overlap_requires_declared_single_choice():
    scene=scenario_v23(BASE,123,faults=False)
    # Two same-skill tasks share a circle; one rover cannot count twice.
    pois=list(scene.pois)
    pois[2]=replace(pois[2],position=pois[0].position,radius=pois[0].radius,coupling=1,window=(0,119))
    pois[0]=replace(pois[0],coupling=1)
    scene=replace(scene,pois=tuple(pois))
    env=make_env(scene,cfg());env.reset(seed=123)
    rover=next(i for i,r in enumerate(scene.rovers) if r.skill==pois[0].required_skill)
    env.positions[rover]=pois[0].position
    roles=np.full(14,-1,np.int32);roles[rover]=2
    env.set_roles(roles)
    _,served,_=env._physical_coverage(0)
    assert served[2] and not served[0]
    roles[rover]=0;env.set_roles(roles)
    _,served,_=env._physical_coverage(0)
    assert served[0] and not served[2]


def scripted_recovery(kind='permanent_failure'):
    base=scenario_v23(BASE,203,faults=False)
    pool=[i for i,r in enumerate(base.rovers) if r.skill==base.pois[0].required_skill]
    chosen=pool[:4];spare=pool[4]
    event=DisruptionEvent(step=2,kind=kind,rover=chosen[0],
                          duration=3 if kind=='temporary_disable' else None)
    scene=replace(base,events=(event,))
    env=make_env(scene,cfg());env.reset(seed=203)
    center=np.array(scene.pois[0].position)
    env.positions[chosen]=center
    env.positions[spare]=center+np.array([scene.pois[0].radius+.3,0.])
    trace=[]
    for t in range(8):
        roles=np.full(14,-1,np.int32)
        roles[chosen]=0
        if kind=='permanent_failure' and t>=2:
            roles[chosen[0]]=-1
            if t>=3: roles[spare]=0
        if kind=='temporary_disable' and 2<=t<5:
            roles[chosen[0]]=-1
        env.set_roles(roles)
        _,reward,_,_,info=env.step(control(env,roles))
        trace.append(dict(step=int(info['transition_step']),reward=reward,
                          served=info['poi_served'].tolist(),
                          active=info['pre_active_mask'].tolist(),
                          choices=roles.tolist(),serving_agents=info['serving_agents'].tolist(),
                          events=info['events_next_boundary']))
    return trace,scene,spare


def test_recovery_by_new_coalition_is_not_natural_reentry():
    trace,scene,spare=scripted_recovery()
    m=physical_metrics(trace,scene,sustain_steps=2)
    x=[x for x in m['recovery_events'] if x['poi']==0]
    assert len(x)==1
    assert x[0]['step']==2 and x[0]['recovered']
    assert x[0]['recovery_lag']<=2
    assert x[0]['recovery_type']=='active_coalition_reallocation'
    assert spare in x[0]['new_members']
    assert m['recovery_active_reallocation']==1
    assert m['recovery_natural_reentry']==0


def test_reentry_is_separate_from_reallocation():
    trace,scene,_=scripted_recovery('temporary_disable')
    m=physical_metrics(trace,scene,sustain_steps=2)
    x=[x for x in m['recovery_events'] if x['poi']==0]
    assert len(x)==1
    assert x[0]['recovered'] and x[0]['recovery_lag']==3
    assert x[0]['recovery_type']=='natural_reentry'
    assert m['recovery_active_reallocation']==0


def test_pivotal_credit_is_training_only_and_zero_for_idle():
    sc=scenario_v23(BASE,107,faults=False)
    n=sc.max_agents
    info=dict(poi_coverage=np.zeros(6,int),poi_served=np.zeros(6,bool),
              serving_agents=np.full(n,-1,int),poi_rewards=np.zeros(6,float))
    info['poi_coverage'][0]=4;info['poi_served'][0]=True
    info['poi_rewards'][0]=5.0;info['serving_agents'][:4]=0
    vals=marginal_physical_credit(info,sc)
    np.testing.assert_array_equal(vals[:4],np.full(4,5.))
    assert not vals[4:].any()
    info['poi_coverage'][0]=5;info['serving_agents'][4]=0
    assert not marginal_physical_credit(info,sc).any()


def test_potential_based_shaping_telescopes_and_keeps_raw_return():
    sc=scenario_v23(BASE,110,faults=False)
    conf=cfg(switch_penalty=0.,potential_shaping_weight=.4)
    a,c,_=setup(conf)
    selector=lambda o,e,p:choose(o,p,'nearest')
    shaped=trajectory(a,c,make_env(sc,conf),110,conf,True,selector)
    controlcfg=replace(conf,potential_shaping_weight=0.)
    plain=trajectory(a,c,make_env(sc,controlcfg),110,controlcfg,True,selector)
    assert shaped['team_return']==plain['team_return']
    def total(ep):
        return sum(conf.gamma**r['boundary_step']*r['reward'] for r in ep['records'])
    # Start state and terminal potential are both zero: discounted returns
    # telescope exactly, without changing the raw physical evaluation metric.
    assert abs(total(shaped)-total(plain))<1e-3


def test_v23_curriculum_increases_coupling_and_preserves_old_distribution():
    for index,stage in enumerate(STAGES):
        scene=stage_scenario(BASE,91,stage,distribution='v23')
        assert [p.coupling for p in scene.pois]==list(STAGE_COUPLING[stage])
        assert scene.events if index>=3 else not scene.events
    old=stage_scenario(BASE,91,STAGES[0])
    assert old.name.startswith('hr_single_service')
    assert old!=stage_scenario(BASE,91,STAGES[0],distribution='v23')


def test_value_scaling_and_marginal_ppo_grads_remain_finite():
    conf=cfg(value_output_scale=100.,marginal_credit_weight=.2,
             adaptive_entropy=True,entropy=.06,chunk_length=11,burn_in=3)
    a,c,l=setup(conf)
    ep=trajectory(a,c,make_env(scenario_v23(BASE,122),conf),122,conf)
    assert all('marginal_return' in r for r in ep['records'])
    assert all(np.isfinite(r['return']) for r in ep['records'])
    stats=l.update([ep])
    assert all(np.isfinite(float(v)) for v in stats.values())
    assert stats['actor_grad_norm']>0
    assert stats['value_loss_raw_units']>=stats['value_loss']
    assert 0<=stats['marginal_credit_nonzero_fraction']<=1
    assert 0<=stats['normalized_entropy']<=1+1e-4
    state=l.state_dict()
    a2,c2,l2=setup(conf);l2.load_state_dict(state)
    assert l2.entropy_coef==l.entropy_coef


def test_invalid_reward_and_attribution_config():
    for params in (dict(value_output_scale=0),dict(marginal_credit_weight=1.1),
                   dict(potential_shaping_weight=-1),dict(idle_decision_penalty=-.01),
                   dict(service_attribution='unrecognized')):
        with pytest.raises(ValueError):replace(cfg(),**params)


def test_v23_resume_preserves_entropy_critic_and_actor(tmp_path,monkeypatch):
    from hr_mappo.experiment import train
    from hr_mappo.ppo import Learner
    conf=cfg(updates=2,train_episodes_per_update=1,seed=1815,
             value_output_scale=100.,marginal_credit_weight=.2,
             potential_shaping_weight=.3,adaptive_entropy=True,
             chunk_length=7,burn_in=2)
    env='configs/rover_12.yaml'
    train(conf,env,tmp_path/'full')
    original=Learner.update
    counter=[0]
    def interrupted(self,episodes):
        counter[0]+=1
        if counter[0]==2:raise RuntimeError('simulate interruption')
        return original(self,episodes)
    with monkeypatch.context() as m:
        m.setattr(Learner,'update',interrupted)
        with pytest.raises(RuntimeError,match='simulate interruption'):
            train(conf,env,tmp_path/'partial')
    train(conf,env,tmp_path/'partial',resume=tmp_path/'partial'/'checkpoint.pt')
    x=torch.load(tmp_path/'full'/'checkpoint.pt',weights_only=False)
    y=torch.load(tmp_path/'partial'/'checkpoint.pt',weights_only=False)
    for name in ('actor','critic'):
        assert all(torch.equal(x[name][k],y[name][k]) for k in x[name])
    assert x['optimizer']['entropy_coef']==y['optimizer']['entropy_coef']


def test_curriculum_v23_training_and_gate_metrics_end_to_end(tmp_path):
    from hr_mappo.experiment import train
    conf=cfg(updates=1,train_episodes_per_update=1,curriculum=True,
             curriculum_eval_every=1,curriculum_min_updates=1,
             curriculum_dev_seeds=(700101,),curriculum_require_local=True,
             curriculum_local_fraction=.5,seed=204,
             value_output_scale=100.,adaptive_entropy=True)
    train(conf,'configs/rover_12.yaml',tmp_path/'pilot')
    import json
    history=[json.loads(x) for x in (tmp_path/'pilot'/'train_history.jsonl').read_text().splitlines()]
    assert len(history)==1
    assert 'gate_dev_idle_mean' in history[0]
    assert 'gate_dev_coupled_rate' in history[0]
    assert 'gate_dev_active_reallocation' in history[0]
    assert (tmp_path/'pilot'/'checkpoint.pt').exists()
