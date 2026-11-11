"""legacy_study service-aware redesign: structural engineering tests only.

NO test here executes the sealed legacy_study fault schedule. The method was
proposed after the legacy_study test was opened, so it requires new confirmation.
"""
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import yaml

from dart_marl.envs.config import EnvironmentConfig, POISpec, RoverSpec, load_config
from dart_marl.envs.rover import RoverEnv
from dart_marl.research.study import _assert_spec, cohort_status, initialize
from dart_marl.roles.allocator import AllocationConfig, allocate_roles
from dart_marl.roles.recovery import (SERVICE_SPEED_FACTORS, SERVICE_SPEED_WEIGHTS,
    enumerate_recovery_coalitions, expected_service_value,
    score_recovery_assignment, task_value_in_horizon)


def rover_problem(rovers, tasks, *, horizon=20):
    cfg = EnvironmentConfig('service_test', (30, 30), horizon=horizon,
        rovers=tuple(RoverSpec(start=tuple(start)) for start in rovers),
        pois=tuple(tasks), max_agents=12, max_pois=6)
    env = RoverEnv(cfg)
    env.reset(seed=73)
    return env


def empty():
    return np.full(12, -1, np.int32)


def test_expected_service_is_an_explicit_discounted_horizon_expectation():
    env = rover_problem([(0, 0), (0, 0)], [POISpec((10, 0), 1, 2, 12., (0, 10), False)])
    # At nominal speed eta is 9; with speed 0.8 and 0.6 it misses window 10.
    observed = expected_service_value(env, 0, 9., 11, 1.)
    assert SERVICE_SPEED_FACTORS == (1.0, 0.8, 0.6)
    assert sum(SERVICE_SPEED_WEIGHTS) == pytest.approx(1.)
    assert observed == pytest.approx(6.)
    assert observed <= task_value_in_horizon(env, 0, 9, 11, 1.)
    assert expected_service_value(env, 0, 0., 11, 1.) == pytest.approx(12.)


def test_service_aware_selects_reliable_task_over_fragile_window_value():
    env = rover_problem([(0, 0), (0, 0)], [
        POISpec((10, 0), 1, 2, 12., (0, 10), False),
        POISpec((0, 2), 1, 2, 1., (0, 10), True)], horizon=20)
    roster, previous = env.reported_active(), empty()
    baseline = allocate_roles(env, roster, previous, AllocationConfig(
        'r_cart_no_sync', forecast_horizon=11, forecast_discount=1.,
        travel_weight=0., switch_weight=0.))
    corrected = allocate_roles(env, roster, previous, AllocationConfig(
        'r_cart_service', forecast_horizon=11, forecast_discount=1.,
        travel_weight=0., switch_weight=0.))
    assert np.flatnonzero(baseline.roles == 0).tolist() == [0, 1]
    assert np.flatnonzero(corrected.roles == 1).tolist() == [0, 1]
    assert corrected.components['synchronization_penalty_basis'] == 0
    assert corrected.components['forecast_task_return'] < baseline.components['forecast_task_return']


def test_service_aware_does_not_penalize_early_partner_wait_twice():
    env = rover_problem([(0, 0), (11, 0), (0, 11)], [
        POISpec((0, 0), 1, 2, 3., (0, 19), True)])
    previous = empty()
    roster = env.reported_active()
    new = allocate_roles(env, roster, previous, AllocationConfig(
        'r_cart_service', forecast_horizon=20, forecast_discount=.98,
        sync_weight=500., travel_weight=.1, switch_weight=1.))
    new_zero = allocate_roles(env, roster, previous, AllocationConfig(
        'r_cart_service', forecast_horizon=20, forecast_discount=.98,
        sync_weight=0., travel_weight=.1, switch_weight=1.))
    assert np.array_equal(new.roles, new_zero.roles)
    assert new.objective == pytest.approx(new_zero.objective)
    assert new.components['synchronization_penalty_basis'] == 0
    assert 0 in np.flatnonzero(new.roles == 0)


def test_service_aware_dp_agrees_with_independent_bruteforce():
    env = rover_problem([(0, 0), (1, 0), (9, 0), (2, 9)], [
        POISpec((0, 0), 1, 2, 3., (0, 19), True),
        POISpec((10, 0), 1, 2, 2.5, (0, 19), True)])
    roster = env.reported_active()
    previous = empty(); previous[0] = previous[1] = 0
    conf = AllocationConfig('r_cart_service', forecast_horizon=15,
        forecast_discount=.95, travel_weight=.3, switch_weight=1.5, sync_weight=25.)
    got = allocate_roles(env, roster, previous, conf)
    optimum = -float('inf')
    for combo in product((-1, 0, 1), repeat=4):
        roles = empty(); roles[:4] = combo
        score = score_recovery_assignment(
            env, roster, previous, roles, horizon_steps=15, discount=.95,
            travel_weight=.3, switch_weight=1.5, sync_weight=25., sync_model='service')
        if score.feasible:
            optimum = max(optimum, score.objective)
    assert got.objective == pytest.approx(optimum)
    assert got.components['objective'] == pytest.approx(optimum)


def test_service_aware_respects_reported_roster_not_future_failure_mask():
    env = rover_problem([(0, 0), (2, 0), (8, 0)], [
        POISpec((1, 0), 1, 2, 4., (0, 19), True)])
    roster = env.reported_active().copy()
    before = empty()
    prior = allocate_roles(env, roster, before, AllocationConfig('r_cart_service'))
    # Physical failures are unobservable until the status report changes.
    env.active[0] = False
    same = allocate_roles(env, roster, before, AllocationConfig('r_cart_service'))
    assert np.array_equal(prior.roles, same.roles)
    assert prior.objective == pytest.approx(same.objective)
    roster[0] = False
    changed = allocate_roles(env, roster, before, AllocationConfig('r_cart_service'))
    assert changed.roles[0] == -1


def test_original_baselines_are_bit_for_bit_unchanged_in_direct_scoring():
    env = rover_problem([(0, 0), (11, 0), (0, 11)], [
        POISpec((0, 0), 1, 2, 8., (0, 19), True)])
    prior = empty(); roster = env.reported_active()
    original = allocate_roles(env, roster, prior, AllocationConfig('r_cart',
        forecast_horizon=20, travel_weight=.01, switch_weight=0., sync_weight=.75))
    normalized = allocate_roles(env, roster, prior, AllocationConfig('r_cart_bounded_sync',
        forecast_horizon=20, travel_weight=.01, switch_weight=0., sync_weight=.75))
    assert np.flatnonzero(original.roles == 0).tolist() == [1, 2]
    assert np.flatnonzero(normalized.roles == 0).tolist() == [1, 2]


def full_spec():
    return yaml.safe_load(Path('configs/experiments/prospective_service_study.yaml').read_text())


def test_service_aware_preregistration_uses_new_seeds_and_new_unexecuted_test():
    spec = _assert_spec(full_spec())  # schema validation ONLY
    old = yaml.safe_load(Path('configs/experiments/completed_study_protocol.yaml').read_text())
    assert spec['primary_method'] == 'r_cart_service'
    assert spec['confirmatory']['primary_variant'] == 'fault_robust'
    assert spec['test_scenarios'] == ['configs/env/service_aware_new_test_only.yaml']
    assert not set(spec['training_seeds']) & set(old['training_seeds'])
    assert not set(spec['training_seeds']) & {11, 22, 33, 42}
    assert {'r_cart_service', 'r_cart', 'switch_aware', 'r_cart_no_sync',
            'r_cart_bounded_sync'} <= set(spec['allocation']['methods'])
    assert 'cft' not in spec['allocation']['methods']
    assert load_config(spec['test_scenarios'][0]).name.endswith('TEST_ONLY')


@pytest.mark.parametrize('replacement,pattern', [
    ({'training_seeds': [101] + [3202,3303,3404,3505,3606,3707,3808,3909,4010]}, 'legacy_study'),
    ({'primary_method':'r_cart'}, 'confirmatory'),
    ({'test_scenarios':['configs/env/primary_study_final_test_only.yaml']}, 'confirmatory'),
    ({'test_scenarios':['configs/env/historical_stress_test.yaml']}, 'confirmatory'),
])
def test_service_aware_rejects_posthoc_reuse(replacement,pattern):
    spec = full_spec(); spec.update(deepcopy(replacement))
    with pytest.raises(ValueError, match=pattern):
        _assert_spec(spec)


def test_service_aware_protocol_registration_does_not_open_test_or_train(tmp_path):
    root=tmp_path/'p74'
    initialize('configs/experiments/prospective_service_study.yaml', root)
    status=cohort_status(root)
    assert status['study_kind'] == 'service_aware'
    assert status['progress']['registered_controllers'] == 20
    assert status['progress']['trained_controllers'] == 0
    assert status['progress']['expected_dev_evaluations'] == 100
    assert status['sealed_test_opened'] is False
    assert not (root/'sealed_test_started.json').exists()


def test_service_aware_pilot_is_distinct_and_can_register_without_test_access(tmp_path):
    pilot = yaml.safe_load(Path('configs/experiments/service_development_pilot.yaml').read_text())
    assert len(pilot['training_seeds']) == 2
    assert not set(pilot['training_seeds']) & set(full_spec()['training_seeds'])
    _assert_spec(pilot)
    initialize('configs/experiments/service_development_pilot.yaml', tmp_path/'pilot')
    assert not (tmp_path/'pilot/sealed_test_started.json').exists()


def test_service_aware_cannot_change_test_and_confirmatory_together():
    spec=full_spec()
    old='configs/env/primary_study_final_test_only.yaml'
    spec['test_scenarios']=[old]
    spec['confirmatory']['primary_scenario']=old
    with pytest.raises(ValueError, match='legacy_study'):
        _assert_spec(spec)


def test_service_aware_algorithm_sources_are_immutable_after_registration(tmp_path, monkeypatch):
    import json
    from dart_marl.research import study
    root=tmp_path/'p74'
    study.initialize('configs/experiments/prospective_service_study.yaml',root)
    recorded=json.loads((root/'protocol.json').read_text())
    assert 'src/dart_marl/roles/recovery.py' in recorded['algorithm_source_sha256']
    assert 'src/dart_marl/research/study.py' in recorded['algorithm_source_sha256']
    target='src/dart_marl/roles/recovery.py'
    original=study.sha
    def modified(path):
        return 'a'*64 if str(path)==target else original(path)
    monkeypatch.setattr(study,'sha',modified)
    with pytest.raises(ValueError, match='algorithm source changed'):
        study._verify_sources(recorded,root)


def test_service_aware_confirmatory_report_uses_service_aware_contrast_without_running_holdout(tmp_path, monkeypatch):
    from dart_marl.research import study
    root = tmp_path/'study'
    study.initialize('configs/experiments/prospective_service_study.yaml', root)
    (root/'sealed_test_started.json').write_text('{}')  # Synthetic guard fixture ONLY
    name = load_config('configs/env/service_aware_new_test_only.yaml').name
    rows=[]
    for v in ('nominal','fault_robust'):
        for seed in full_spec()['training_seeds']:
            m={}
            for method,value in [('switch_aware',8.),('r_cart',7.),('r_cart_no_sync',8.),
                                 ('r_cart_bounded_sync',8.),('r_cart_service',9.)]:
                m[method]={
                    'mean_return': value+seed/10000,
                    'mean_role_switches': 0.,
                    'mean_after_first_disruption_return': value/2,
                    'mean_poi_service_steps':[4.,8.,2.],
                    'recovery_metrics':{
                        'restoration_rate_among_at_risk':.75,
                        'mean_post_event_repeat_service_fraction':.5,
                        'mean_first_restoration_lag_observed_only':3.,
                    },
                }
            rows.append({'variant':v,'scenario':name,'training_seed':seed,
                         'evaluation_seeds':list(range(8)),'methods':m})
    monkeypatch.setattr(study,'_collect',lambda *args:rows)
    result=study.summarize(root,stage='test')
    primary=result['predeclared_primary_result']
    assert primary['contrast']=='r_cart_service_minus_switch_aware'
    assert primary['seed_level_estimate_and_ci']['mean']==pytest.approx(1.)
    assert primary['seed_level_estimate_and_ci']['n_independent_training_seeds']==10
