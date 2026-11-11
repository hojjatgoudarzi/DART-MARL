"""legacy_study CFT: paired branches, decision-time information, split, selection, reload."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from dart_marl.cft.candidates import assignment_candidates
from dart_marl.cft.features import visible_features, task_problem_hash
from dart_marl.cft.rollouts import branch_return, generate_data, sampled_scenario
from dart_marl.cft.training import CFTEstimator, train_predictor
from dart_marl.disruptions.events import DisruptionEvent
from dart_marl.envs.config import load_config
from dart_marl.envs.rover import RoverEnv
from dart_marl.roles.allocator import AllocationConfig, allocate_roles
from dart_marl.roles.evaluation import evaluate_allocators
from dart_marl.roles.recovery import score_recovery_assignment


@pytest.fixture(scope='module')
def base():
    return load_config('configs/env/dev_roles.yaml')


def test_future_fault_tape_hidden_from_visible_features(base):
    a = RoverEnv(replace(base, events=()))
    b = RoverEnv(replace(base, events=(DisruptionEvent(
        step=10, kind='permanent_failure', rover=0, detection_delay=2),)))
    a.reset(seed=101); b.reset(seed=101)
    prev = a.roles.copy()
    proposal = np.full(base.max_agents, -1, dtype=np.int32)
    proposal[:6] = [0, 0, 1, 1, 2, 2]
    x = visible_features(a, a.reported_active(), prev, proposal, horizon=24, discount=.98)
    y = visible_features(b, b.reported_active(), prev, proposal, horizon=24, discount=.98)
    assert np.array_equal(x, y)
    # Even if physical actuator state secretly changes while reports are delayed,
    # planner input may not change solely because of true active mask.
    b.active[0] = False
    y2 = visible_features(b, b.reported_active(), prev, proposal,
                          horizon=24, discount=.98)
    assert np.array_equal(y, y2)
    assert task_problem_hash(a.config) == task_problem_hash(b.config)


def test_branch_counterfactual_is_deterministic_and_does_not_mutate_original(base):
    env = RoverEnv(sampled_scenario(base, 507))
    env.reset(seed=507)
    orig = env.snapshot()
    roles = np.full(base.max_agents, -1, dtype=np.int32)
    roles[:6] = [0, 0, 1, 1, 2, 2]
    r1 = branch_return(env, roles, horizon=10, discount=.98)
    r2 = branch_return(env, roles, horizon=10, discount=.98)
    assert r1 == pytest.approx(r2)
    assert np.array_equal(env.positions, orig.positions)
    assert np.array_equal(env.active, orig.active)
    assert np.array_equal(env.roles, orig.roles)
    assert env.step_index == orig.step_index
    # Two branches have the same exogenous disturbances and RNG path after reset.
    left, right = env.clone(), env.clone()
    for _ in range(15):
        x = np.zeros((base.max_agents, 2))
        l = left.step(x)
        r = right.step(x)
        assert l[1] == r[1]
        assert l[4]['events_next_boundary'] == r[4]['events_next_boundary']


def test_candidate_set_contains_exact_optimum_is_feasible_and_side_effect_free(base):
    env = RoverEnv(base)
    env.reset(seed=55)
    before = env.snapshot()
    roster = env.reported_active()
    previous = np.full(base.max_agents, -1, dtype=np.int32)
    opts = assignment_candidates(env, roster, previous, k=7, beam=60)
    exact = allocate_roles(env, roster, previous, AllocationConfig('r_cart'))
    assert any(np.array_equal(r, exact.roles) for r in opts)
    assert len(opts) == len({tuple(x) for x in opts})
    for x in opts:
        assert np.all(x[~roster] == -1)
        score = score_recovery_assignment(
            env, roster, previous, x, horizon_steps=24, discount=.98,
            travel_weight=.1, switch_weight=3, sync_weight=.75)
        assert score.feasible
    assert np.array_equal(env.positions, before.positions)
    assert env.step_index == before.step_index
    with pytest.raises(ValueError, match='requires'):
        allocate_roles(env, roster, previous, AllocationConfig(
            'cft', cft_model_path='not-used.pt'))


@pytest.fixture(scope='module')
def toy_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp('cft_dataset') / 'dataset'
    generate_data('configs/env/dev_roles.yaml', out, controller='scripted',
                  episodes_train=2, episodes_val=1, episodes_test=1,
                  horizon=8, sample_period=12, candidates=3, beam=40, seed=400777)
    return out


def test_dataset_split_and_physical_return_bounds(toy_dataset):
    manifest = json.loads((toy_dataset / 'manifest.json').read_text())
    with np.load(toy_dataset / 'counterfactuals.npz', allow_pickle=False) as z:
        splits = z['split']; groups=z['episode_id']; snaps=z['snapshot_id']
        assert set(splits.tolist()) == {'train','val','test'}
        for a,b in [('train','val'), ('train','test'), ('val','test')]:
            assert set(groups[splits==a]).isdisjoint(set(groups[splits==b]))
        assert len(set(groups)) == 4
        assert len(set(snaps)) < len(snaps)
        assert np.all(z['target'] >= -1e-5)
        assert np.all(z['target'] <= z['potential'] + 1e-4)
        assert np.isfinite(z['x']).all()
        assert z['x'].shape[1] > 250
    assert manifest['event_mode'] == 'sampled'
    assert len(set(p['fingerprint'] for p in manifest['episode_protocols'])) > 1
    assert manifest['controller_checkpoint_sha256'] is None


@pytest.fixture(scope='module')
def toy_model(tmp_path_factory, toy_dataset):
    out = tmp_path_factory.mktemp('cft_model') / 'model'
    train_predictor(toy_dataset, out, seed=89, epochs=12,
                    batch_size=32, hidden=32, patience=4)
    return out


def test_model_training_metrics_finite_and_provenance_enforced(base, toy_model):
    report = json.loads((toy_model / 'training_report.json').read_text())
    for fold in ('train','val','test'):
        assert np.isfinite(report['metrics'][fold]['rmse'])
        assert report['metrics'][fold]['paired_snapshot_groups'] >= 1
        assert report['metrics'][fold]['candidate_selection_mean_regret'] >= -1e-5
    est = CFTEstimator(toy_model / 'cft_model.pt', expected_config=base,
                       expected_controller_sha=None, strict_controller=True,
                       horizon=8, discount=.98)
    x = np.zeros((2, len(est.mean)), dtype=np.float32)
    assert np.isfinite(est.residuals(x)).all()
    with pytest.raises(ValueError, match='different frozen'):
        CFTEstimator(toy_model / 'cft_model.pt', expected_controller_sha='wrong',
                     strict_controller=True)
    with pytest.raises(ValueError, match='horizon'):
        CFTEstimator(toy_model / 'cft_model.pt', horizon=24)
    with pytest.raises(ValueError, match='task problem'):
        CFTEstimator(toy_model / 'cft_model.pt', expected_config=load_config(
            'configs/env/main_slack.yaml'))


def test_reproducible_paired_cft_evaluation_and_model_requirements(toy_model):
    kwargs = dict(controller='scripted', capture_trace=False,
                  methods=('fixed', 'r_cart', 'cft'),
                  cft_model=toy_model / 'cft_model.pt', episodes=1,
                  seed=940101, forecast_horizon=8)
    x = evaluate_allocators('configs/env/analytic_repeated_faults.yaml', **kwargs)
    y = evaluate_allocators('configs/env/analytic_repeated_faults.yaml', **kwargs)
    assert x == y
    assert x['evaluation_kind'] == 'learned_cft'
    assert 'cft_minus_r_cart' in x['paired_differences']
    assert x['methods']['cft']['per_episode'][0]['role_decisions']
    with pytest.raises(ValueError, match='trained --cft-model'):
        evaluate_allocators('configs/env/analytic_repeated_faults.yaml',
                            methods=('cft',), episodes=1)
    with pytest.raises(ValueError, match='horizon'):
        evaluate_allocators('configs/env/analytic_repeated_faults.yaml',
                            **{**kwargs, 'forecast_horizon':24})
