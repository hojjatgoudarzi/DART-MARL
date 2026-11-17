"""Research contracts for v2.5: no fake reward, no fake recovery, no oracle.

Small synthetic physical records and an integration training run are intentional
so the tests cannot pass just because a class or config key exists.
"""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from hr_mappo.envs.config import load_config
from hr_mappo.config import HRConfig
from hr_mappo.physics import (physical_overstaff, marginal_physical_credit,
                                        shapley_service_credit, make_env)
from hr_mappo.metrics import physical_metrics
from hr_mappo.scenarios_adaptive import scenario_v24
from hr_mappo.interventions import ConditionalCoalitionFaultEnv
from hr_mappo.experiment import dev_selection_score, train

BASE=load_config('configs/rover_12.yaml')


def sample_info(s,task=0,count=5,reward=4.0,served=True):
    members=np.full(s.max_agents,-1,dtype=np.int32)
    members[:count]=task
    rewards=np.zeros(s.max_pois,dtype=np.float32)
    rewards[task]=reward if served else 0.
    is_served=np.zeros(s.max_pois,dtype=bool);is_served[task]=served
    counts=np.zeros(s.max_pois,dtype=int);counts[task]=count
    return dict(serving_agents=members,poi_coverage=counts,
                poi_served=is_served,poi_rewards=rewards)


def test_physical_shapley_conserves_exact_service_reward_with_surplus():
    s=scenario_v24(BASE,101,faults=False)
    for count in (3,4,5,6):
        info=sample_info(s,count=count,reward=9.6)
        sh=shapley_service_credit(info,s)
        assert sh.shape==(14,)
        np.testing.assert_allclose(sh[:count],9.6/count,rtol=1e-6)
        assert sh[count:].sum()==0
        assert float(sh.sum())==pytest.approx(9.6)
        pivotal=marginal_physical_credit(info,s)
        assert (pivotal.sum()>0)==(count==3)
        assert physical_overstaff(info,s)==count-3


def test_physical_credit_never_awards_intent_only_or_incomplete_tasks():
    s=scenario_v24(BASE,101,faults=False)
    for count in (0,1,2):
        info=sample_info(s,count=count,served=False)
        assert not shapley_service_credit(info,s).any()
        assert physical_overstaff(info,s)==0


def test_multiple_tasks_shapley_split_and_true_surplus():
    s=scenario_v24(BASE,101,faults=False)
    info=sample_info(s,count=4,reward=8.)
    info['serving_agents'][4:7]=1
    info['poi_coverage'][1]=3
    info['poi_served'][1]=True
    info['poi_rewards'][1]=6.
    c=shapley_service_credit(info,s)
    np.testing.assert_allclose(c[:4],2.)
    np.testing.assert_allclose(c[4:7],2.)
    assert c.sum()==pytest.approx(14.)
    assert physical_overstaff(info,s)==1


def test_training_only_overstaff_coefficient_and_checkpoint_selection():
    cfg=HRConfig(overstaff_penalty_weight=.04,best_dev_overstaff_weight=.1,
                 credit_mode='shapley_service',best_dev_checkpoint=True)
    assert dev_selection_score(810,400,cfg)==pytest.approx(770.)
    assert dev_selection_score(800,200,cfg)==pytest.approx(780.)
    with pytest.raises(ValueError):HRConfig(credit_mode='D++')
    with pytest.raises(ValueError):HRConfig(overstaff_penalty_weight=-.01)


def test_efficiency_metrics_task_coverage_without_role_oracle():
    s=scenario_v24(BASE,101,faults=False)
    trace=[]
    for t in range(3):
        roles=[0]*5+[1]*3+[-1]*6
        trace.append(dict(step=t,reward=10.,served=[True,True,False,False,False,False],
            serving_agents=roles,choices=roles,active=[True]*12+[False]*2,events=[]))
    m=physical_metrics(trace,s)
    assert m['total_surplus_contributor_steps']==6
    assert m['surplus_contributor_steps_per_poi'][:2]==[6,0]
    assert m['steps_with_2plus_repeatable_pois']==3
    assert m['concurrent_2plus_fraction']==1.0
    assert m['mean_concurrent_repeatable_pois']==2.0
    assert m['team_return']==30.
    assert m['active_recovery_rate'] is None
    assert m['nominal_unserved_task_value_upper_bound']>0


def test_simultaneous_multi_rover_fault_is_one_outage_not_three():
    s=scenario_v24(BASE,101,faults=False)
    # 3-person coalition before fault, 2/3 victims fail at the SAME boundary.
    trace=[]
    for t in range(7):
        has_service=t<2 or t>=5
        served=[has_service]+[False]*5
        contributors=[0,0,0]+[-1]*11 if t<2 else (
            [0,0,-1]+[-1]*11 if t<5 else [0,0,0]+[-1]*11)
        ev=[dict(step=2,kind='permanent_failure',rover=i) for i in (0,1)] if t==1 else []
        trace.append(dict(step=t,reward=4. if has_service else 0.,served=served,
               serving_agents=contributors,choices=[0]*3+[-1]*11,
               active=[True]*12+[False]*2,events=ev))
    m=physical_metrics(trace,s,sustain_steps=2)
    assert m['fault_count']==2
    assert m['disrupted_service_events']==1
    assert m['recovery_events'][0]['trigger_rovers']==[0,1]
    assert m['recovery_events'][0]['recovery_lag']==3
    assert m['recovery_active_reallocation']==0  # no new member, no fake active transfer


def _prepared_env(seed=501,staff_count=5,mode='minimal_feasible_cut'):
    s=scenario_v24(BASE,seed,faults=False)
    env=ConditionalCoalitionFaultEnv(s,mode=mode)
    env.reset(seed=seed)
    chosen=[i for i,r in enumerate(s.rovers) if r.skill==s.pois[0].required_skill]
    assert len(chosen)==6
    env.positions[chosen[:staff_count]]=s.pois[0].position
    env.step_index=49
    roles=np.full(14,-1,np.int32)
    roles[chosen[:staff_count]]=0
    env.set_roles(roles)
    return env


def test_minimal_feasible_cut_forces_drop_with_spare_compatible_survivors():
    env=_prepared_env()
    _,r,_,_,info=env.step(np.zeros((14,2),np.float32))
    st=env.intervention_status
    assert r>0 and info['poi_served'][0]
    assert st['opportunity'] and st['target_cut_size']==3
    assert len(st['victims'])==3
    assert st['structural_feasibility_after_cut']
    assert len(info['events_next_boundary'])==3
    assert sum(env.active[i] and env.skills[i]==env.config.pois[0].required_skill
               for i in range(14))==3
    _,_,_,_,after=env.step(np.zeros((14,2),np.float32))
    assert not after['poi_served'][0]


def test_minimal_cut_rejects_irrecoverable_all_six_overstaffed():
    env=_prepared_env(staff_count=6)
    env.step(np.zeros((14,2),np.float32))
    st=env.intervention_status
    assert not st['opportunity']
    assert st['had_serviced_task']
    assert len(env.config.events)==0


def test_counterfactual_intervention_never_persists_after_reset():
    env=_prepared_env()
    env.step(np.zeros((14,2),np.float32))
    assert len(env.config.events)==3
    env.reset(seed=501)
    assert len(env.config.events)==0
    assert env.intervention_status is None


def test_small_real_ppo_rollout_has_separate_training_and_physical_return():
    from hr_mappo.experiment import setup
    from hr_mappo.rollout import trajectory
    from hr_mappo.scenarios import create_scenario
    cfg=HRConfig(device='cpu',scenario='coordination_v24',service_attribution='declared',
                 hidden=32,epochs=1,updates=1,train_episodes_per_update=1,
                 decision_mode='asynchronous',decision_interval=8,async_min_interval=4,
                 overstaff_penalty_weight=.08,marginal_credit_weight=.15,
                 credit_mode='shapley_service')
    actor,critic,learner=setup(cfg)
    s=create_scenario(BASE,606,'coordination_v24')
    ep=trajectory(actor,critic,make_env(s,cfg),606,cfg)
    assert ep['team_return']==pytest.approx(sum(x['reward'] for x in ep['physical']))
    assert ep['overstaff_cost']==pytest.approx(.08*ep['overstaff_rover_steps'])
    assert ep['team_return']>=0
    st=learner.update([ep])
    assert np.isfinite(st['value_loss']) and np.isfinite(st['policy_loss'])


def test_best_checkpoint_only_final_stage_and_can_be_reloaded(tmp_path):
    cfg=HRConfig(device='cpu',scenario='coordination_v24',service_attribution='declared',
        hidden=32,epochs=1,updates=5,train_episodes_per_update=1,
        curriculum=True,curriculum_mode='fixed',curriculum_eval_every=1,
        curriculum_dev_seeds=(700101,),curriculum_require_local=False,
        best_dev_checkpoint=True,best_dev_overstaff_weight=.05,
        overstaff_penalty_weight=.04,credit_mode='shapley_service',
        marginal_credit_weight=.1,decision_interval=8,decision_mode='asynchronous',
        async_min_interval=4)
    train(cfg,'configs/rover_12.yaml',tmp_path)
    ck=torch.load(tmp_path/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    assert ck['update']==5 and ck['best_dev_update']==5
    assert (tmp_path/'best_development.json').exists()
    assert (tmp_path/'checkpoint.pt').exists()
    assert (tmp_path/'actor_only.pt').exists()
    assert (tmp_path/'best_actor_only.pt').exists()
    from hr_mappo.experiment import setup
    actor,_,_=setup(cfg)
    actor.load_state_dict(ck['actor'])


def test_overstaff_penalty_cannot_change_physical_return_of_fixed_policy():
    from hr_mappo.experiment import setup
    from hr_mappo.rollout import trajectory
    from hr_mappo.baselines import make_selector
    s=scenario_v24(BASE,810101,faults=False)
    cfg=HRConfig(device='cpu',service_attribution='declared',hidden=32,
                 overstaff_penalty_weight=0.)
    a,c,l=setup(cfg)
    ep0=trajectory(a,c,make_env(s,cfg),810101,cfg,deterministic=True,
                   selector=make_selector('static_highest_value_repeatable'))
    weighted=replace(cfg,overstaff_penalty_weight=.1)
    ep1=trajectory(a,c,make_env(s,weighted),810101,weighted,deterministic=True,
                   selector=make_selector('static_highest_value_repeatable'))
    assert ep0['team_return']==pytest.approx(ep1['team_return'])
    assert ep0['overstaff_cost']==0
    assert ep1['overstaff_rover_steps']>0
    assert ep1['overstaff_cost']==pytest.approx(.1*ep1['overstaff_rover_steps'])
    assert not ep1['on_policy_compatible']
    with pytest.raises(ValueError,match='cannot PPO-update'):
        l.update([ep1])
