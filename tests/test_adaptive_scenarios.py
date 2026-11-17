"""v2.4 preflight oracles, nondegenerate task distribution and compat tests."""
from dataclasses import replace
import numpy as np
import pytest
from hr_mappo.envs.config import load_config
from hr_mappo.scenarios_adaptive import scenario_v24
from hr_mappo.scenarios import create_scenario
from hr_mappo.curriculum import STAGES,stage_scenario
from hr_mappo.config import HRConfig
from hr_mappo.physics import make_env
from hr_mappo.baselines import choose,make_selector
from hr_mappo.observations import observe_agents,critic_state
from hr_mappo.metrics import physical_metrics

BASE=load_config('configs/rover_12.yaml')


def test_seeded_discriminative_public_task_distribution():
    early_indices=set()
    for seed in (101,202,303,404,505,606,707,808):
        s=scenario_v24(BASE,seed)
        assert s == create_scenario(BASE,seed,'coordination_v24')
        assert s == scenario_v24(BASE,seed)
        assert len(s.rovers)==12 and s.max_agents==14 and len(s.pois)==6
        assert [p.coupling for p in s.pois]==[3,3,3,3,1,1]
        assert [sum(r.skill==v for r in s.rovers) for v in (0,1)]==[6,6]
        assert len(s.events)==4
        assert [ev.kind for ev in s.events]==['permanent_failure','temporary_disable',
                                               'replacement_agent','capability_degradation']
        assert not any(np.linalg.norm(np.asarray(r.start)-np.asarray(p.position))<=p.radius+2.
                       for r in s.rovers for p in s.pois)
        for group in ((0,2),(1,3)):
            a,b=(s.pois[j] for j in group)
            assert a.required_skill==b.required_skill
            assert a.window[0]==0 or b.window[0]==0
            assert a.window[0]>0 or b.window[0]>0
            assert max(a.window[0],b.window[0])<min(a.window[1],b.window[1])
            assert sum(r.skill==a.required_skill for r in s.rovers)-1 >= max(a.coupling,b.coupling)
            early_indices.add(next(j for j in group if s.pois[j].window[0]==0))
        env=make_env(s,replace(HRConfig(),service_attribution='declared'))
        env.reset(seed=seed)
        assert env.preview_service()['reward']==0.
    assert len(early_indices)==4  # no permanently assigned early task IDs


def test_train_dev_and_heldout_have_distinct_geometry():
    for seed in (101,202):
        a=scenario_v24(BASE,seed)
        b=scenario_v24(BASE,seed,heldout=True)
        assert a != b
        assert any(np.linalg.norm(np.asarray(x.position)-np.asarray(y.position))>2
                   for x,y in zip(a.pois,b.pois))


def test_stage_coupling_progression_and_hidden_faults():
    for j,stage in enumerate(STAGES):
        s=stage_scenario(BASE,303,stage,distribution='v24')
        assert len(s.rovers)==12
        assert set(p.coupling for p in s.pois[:4])==({1} if j==0 else {2} if j==1 else {3})
        assert bool(s.events)==(j>=3)
        assert all(p.required_skill is None for p in s.pois[:4])==(j<2)


def test_frozen_baseline_cannot_replan_a_closed_window():
    s=scenario_v24(BASE,101,faults=False)
    env=make_env(s,replace(HRConfig(),service_attribution='declared'))
    env.reset(seed=101)
    prev=np.full(14,-1,np.int32)
    obs=observe_agents(env,prev)
    selector=make_selector('fixed_initial_nearest')
    before=selector(obs,None,prev)
    # Future schedule changes the task's legality. A fixed assignment may
    # only become IDLE; it never opportunistically changes POIs.
    env.step_index=110
    later=observe_agents(env,before)
    after=selector(later,None,before)
    assert all(y in (-1,x) for x,y in zip(before,after))
    for i in range(12):
        if before[i]>=0 and not later['legal'][i,before[i]]:
            assert after[i]==-1


def test_new_baselines_use_only_observation_and_choose_legal():
    s=scenario_v24(BASE,606)
    env=make_env(s,replace(HRConfig(),service_attribution='declared'))
    env.reset(seed=606)
    prev=np.full(14,-1,np.int32);obs=observe_agents(env,prev)
    for method in ('always_idle','fixed_initial_value','fixed_initial_nearest',
                   'static_highest_value_repeatable','nearest','window_aware'):
        sel=make_selector(method)
        result=sel(obs,None,prev)
        assert result.shape==(14,)
        assert all(obs['legal'][i,int(a) if a>=0 else -1] for i,a in enumerate(result))


def test_hidden_remote_position_cannot_change_local_observation():
    s=scenario_v24(BASE,323,faults=False)
    env=make_env(s,replace(HRConfig(),service_attribution='declared'))
    env.reset(seed=323)
    prev=np.full(14,-1,np.int32)
    i=0
    distant=[j for j in range(1,12) if np.linalg.norm(env.positions[j]-env.positions[i])>
             env.sensor_radii[i]+2.]
    assert distant
    j=distant[0]
    first=observe_agents(env,prev)
    state=critic_state(env)
    env.positions[j]=np.asarray(s.rovers[j].start)+np.array([.2,.1])
    second=observe_agents(env,prev)
    for k in ('ego','tasks','neighbors','neighbor_mask','status','status_mask','legal'):
        np.testing.assert_array_equal(first[k][i],second[k][i])
    assert not np.array_equal(state['agent'],critic_state(env)['agent'])


def test_no_recovery_opportunity_means_undefined_rate():
    trace=[dict(step=t,reward=0.,served=[False]*6,choices=[-1]*14,
                active=[True]*12+[False]*2,serving_agents=[-1]*14,events=[])
           for t in range(120)]
    m=physical_metrics(trace,scenario_v24(BASE,100,faults=False))
    assert m['fault_count']==0
    assert m['recovery_opportunities']==0
    assert m['active_recovery_rate'] is None
    assert m['restored_service_rate'] is None
    assert m['recovery_opportunity_absent']


def test_policy_independent_preflight_shows_feasible_adaptation():
    # Distinguish initial global planning from subsequent global replanning.
    # The privileged diagnostic is a feasibility witness, NOT a fair baseline.
    from scripts.benchmark_preflight import analyze
    result=analyze(BASE,[810101,810202,810303,810404,810505,810606,810707,810808])
    assert result['checks']['always_idle_zero']
    assert result['checks']['reachable_positive']
    assert result['checks']['feasible_adaptation_gain_over_own_frozen_plan']
    assert result['checks']['diagnostic_planner_has_disrupted_tasks']
    assert result['checks']['diagnostic_planner_has_active_restoration']
    assert result['means']['fixed_initial_value']['raw_return']>0


def test_conditional_intervention_is_only_at_next_boundary_and_is_labeled():
    from hr_mappo.interventions import ConditionalCoalitionFaultEnv
    scene=scenario_v24(BASE,501,faults=False)
    env=ConditionalCoalitionFaultEnv(scene)
    env.reset(seed=501)
    poi=0
    agents=[i for i,r in enumerate(scene.rovers) if r.skill==scene.pois[poi].required_skill][:3]
    env.positions[agents]=np.asarray(scene.pois[poi].position)
    env.step_index=49
    roles=np.full(14,-1,np.int32);roles[agents]=poi
    env.set_roles(roles)
    # The movement scored at t=49 uses the PRE-fault team; the fault is
    # applied before action at t=50, never retroactively at t=49.
    _,reward,_,_,info=env.step(np.zeros((14,2),np.float32))
    assert info['poi_served'][poi]
    assert reward>0
    assert env.intervention_status['opportunity']
    assert env.intervention_status['task']==poi
    assert env.intervention_status['rover'] in agents
    assert not env.active[env.intervention_status['rover']]
    assert any(ev['kind']=='permanent_failure' and ev['step']==50
               for ev in info['events_next_boundary'])
    # Next observation/actor has no access to the private future schedule.
    assert not hasattr(observe_agents(env,np.full(14,-1,np.int32)),'events')


def test_intervention_no_service_is_not_a_fake_recovery_failure():
    from hr_mappo.interventions import ConditionalCoalitionFaultEnv
    scene=scenario_v24(BASE,501,faults=False)
    env=ConditionalCoalitionFaultEnv(scene)
    env.reset(seed=501)
    env.step_index=49
    env.set_roles(np.full(14,-1,np.int32))
    _,_,_,_,info=env.step(np.zeros((14,2),np.float32))
    assert env.intervention_status['opportunity'] is False
    assert not info['events_next_boundary']


def test_conditional_fault_never_leaks_across_episode_reset():
    from hr_mappo.interventions import ConditionalCoalitionFaultEnv
    scene=scenario_v24(BASE,509,faults=False)
    env=ConditionalCoalitionFaultEnv(scene)
    env.reset(seed=509)
    skill=scene.pois[0].required_skill
    chosen=[i for i,r in enumerate(scene.rovers) if r.skill==skill][:3]
    env.step_index=49
    env.positions[chosen]=scene.pois[0].position
    roles=np.full(14,-1,np.int32);roles[chosen]=0
    env.set_roles(roles)
    env.step(np.zeros((14,2),np.float32))
    assert env.intervention_status['opportunity']
    assert len(env.config.events)==1
    env.reset(seed=509)
    assert env.intervention_status is None
    assert len(env.config.events)==0 and len(env._event_applied)==0
    env.step_index=49
    env.set_roles(np.full(14,-1,np.int32))
    _,_,_,_,info=env.step(np.zeros((14,2),np.float32))
    assert env.intervention_status['opportunity'] is False
    assert info['events_next_boundary']==[]
