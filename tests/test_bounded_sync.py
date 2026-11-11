"""legacy_study bounded synchronization: retains legacy R-CART for comparisons."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from dart_marl.envs.config import EnvironmentConfig, POISpec, RoverSpec, load_config
from dart_marl.envs.rover import RoverEnv
from dart_marl.roles.allocator import AllocationConfig, allocate_roles
from dart_marl.roles.recovery import (enumerate_recovery_coalitions,
                                      optimize_recovery_assignment,
                                      score_recovery_assignment)


def make_env(starts, *, task=(0, 0), radius=1.0, horizon=40):
    cfg = EnvironmentConfig(
        name='bounded_sync_regression', dimensions=(30, 30), horizon=horizon,
        max_agents=12, max_pois=6,
        rovers=tuple(RoverSpec(start=tuple(x)) for x in starts),
        pois=(POISpec(position=task, radius=radius, coupling=2, value=8.0,
                      window=(0, horizon-1), repeat=True),),
    )
    env = RoverEnv(cfg)
    env.reset(seed=48)
    return env


def empty_roles(n=12):
    return np.full(n, -1, dtype=np.int32)


def test_penalty_is_bounded_and_distinct_from_zero_sync():
    env = make_env([(0, 0), (11, 0), (0, 11)])
    env.positions[:3] = np.array([[0., 0.], [11., 0.], [0., 11.]])
    options = enumerate_recovery_coalitions(env, env.reported_active(), empty_roles(),
                                            horizon_steps=24, discount=1.)[0]
    for c in options:
        assert 0 <= c.normalized_synchronization <= 1.0 + 1e-10
        assert c.normalized_synchronization <= c.synchronization + 1e-10
    # Normalization does NOT simply discard synchronization: with sufficiently
    # cheap travel, equal arrival times still beat one near/one distant partner.
    bounded = allocate_roles(env, env.reported_active(), empty_roles(), AllocationConfig(
        'r_cart_bounded_sync', travel_weight=.01, switch_weight=0,
        sync_weight=.75, forecast_horizon=24))
    zero = allocate_roles(env, env.reported_active(), empty_roles(), AllocationConfig(
        'r_cart_no_sync', travel_weight=.01, switch_weight=0,
        sync_weight=.75, forecast_horizon=24))
    assert np.flatnonzero(bounded.roles == 0).tolist() == [1, 2]
    assert 0 in np.flatnonzero(zero.roles == 0)


def test_seed22_step17_failure_geometry_without_any_checkpoint():
    """Observed development geometry only; not an independent test example."""
    env = RoverEnv(load_config('configs/env/dev_failure.yaml'))
    env.reset(seed=980000)
    env.step_index = 17
    env.positions[:6] = np.array([
        [5.1, 4.3], [5.1, 4.0], [19.2, 5.4],
        [12.3, 18.9], [12.4, 18.2], [19.2, 5.0],
    ])
    env.poi_completed[2] = True
    env.active[0] = False
    reported = np.array([False, True, True, True, True, True] + [False]*6)
    prior = np.array([-1, 0, 1, -1, -1, 1] + [-1]*6, dtype=np.int32)
    before = env.positions.copy()
    old = allocate_roles(env, reported, prior, AllocationConfig('r_cart'))
    new = allocate_roles(env, reported, prior, AllocationConfig('r_cart_bounded_sync'))
    no_sync = allocate_roles(env, reported, prior, AllocationConfig('r_cart_no_sync'))
    assert np.flatnonzero(old.roles == 0).tolist() == [3, 4]
    assert np.flatnonzero(new.roles == 0).tolist() == [1, 4]
    assert np.array_equal(new.roles, no_sync.roles)
    assert new.components['synchronization_wait_sum'] > 12
    assert new.components['synchronization_penalty_basis'] <= 2
    assert np.array_equal(before, env.positions)


def test_bounded_optimizer_matches_independent_brute_force():
    env = make_env([(0, 0), (2, 0), (10, 0), (1, 10)], horizon=25)
    env.positions[:4] = np.array([[0., 0.], [2., 0.], [10., 0.], [1., 10.]])
    prior = empty_roles()
    prior[0] = prior[1] = 0
    cfg = AllocationConfig('r_cart_bounded_sync', forecast_horizon=15,
                           forecast_discount=.97, travel_weight=.2, switch_weight=1.,
                           sync_weight=2.)
    chosen = allocate_roles(env, env.reported_active(), prior, cfg)
    optimal = -float('inf')
    for assignment in product((-1, 0), repeat=4):
        candidate = empty_roles()
        candidate[:4] = assignment
        score = score_recovery_assignment(
            env, env.reported_active(), prior, candidate,
            horizon_steps=cfg.forecast_horizon, discount=cfg.forecast_discount,
            travel_weight=cfg.travel_weight, switch_weight=cfg.switch_weight,
            sync_weight=cfg.sync_weight, sync_model='normalized')
        if score.feasible:
            optimal = max(optimal, score.objective)
    assert chosen.objective == pytest.approx(optimal)
    assert chosen.components['objective'] == pytest.approx(chosen.objective)
    assert chosen.components['synchronization_wait_sum'] >= chosen.components['synchronization_penalty_basis']


def test_zero_sync_weight_produces_same_roles_for_both_variants():
    env = make_env([(0, 0), (11, 0), (0, 11)])
    env.positions[:3] = np.array([[0., 0.], [11., 0.], [0., 11.]])
    before = empty_roles()
    cfg = AllocationConfig('r_cart', forecast_horizon=24, sync_weight=0)
    legacy = allocate_roles(env, env.reported_active(), before, cfg)
    revised = allocate_roles(env, env.reported_active(), before,
                             replace(cfg, method='r_cart_bounded_sync'))
    assert np.array_equal(legacy.roles, revised.roles)
    assert legacy.objective == pytest.approx(revised.objective)


def test_invalid_sync_model_rejected_and_legacy_default_unchanged():
    env = make_env([(0, 0), (11, 0)])
    before = empty_roles()
    options = enumerate_recovery_coalitions(env, env.reported_active(), before,
                                            horizon_steps=20, discount=.98)
    roles = empty_roles()
    roles[:2] = 0
    args = dict(horizon_steps=20, discount=.98, travel_weight=.1,
                switch_weight=3, sync_weight=.75, candidates=options)
    old = score_recovery_assignment(env, env.reported_active(), before, roles, **args)
    raw = score_recovery_assignment(env, env.reported_active(), before, roles,
                                    sync_model='raw', **args)
    assert old == raw
    with pytest.raises(ValueError, match='sync_model'):
        score_recovery_assignment(env, env.reported_active(), before, roles,
                                  sync_model='invalid', **args)
    with pytest.raises(ValueError, match='sync_model'):
        optimize_recovery_assignment(env, env.reported_active(), before,
                                     horizon_steps=20, discount=.98, travel_weight=.1,
                                     switch_weight=3, sync_weight=.75, sync_model='invalid')


def test_bounded_sync_pilot_rejects_nonpilot_and_same_root_without_running(tmp_path):
    from dart_marl.research.bounded_sync import run_development_comparison
    pilot = tmp_path / 'pilot'
    pilot.mkdir()
    (pilot / 'protocol.json').write_text('{"format":"dart-marl-study-protocol-v1", "spec":{"name":"not_pilot"}}')
    with pytest.raises((ValueError, KeyError)):
        run_development_comparison(pilot, tmp_path / 'result')


def test_selected_checkpoint_can_be_rebased_without_trusting_filename(tmp_path):
    import hashlib
    import json
    from dart_marl.research.bounded_sync import _selected_from_pilot

    root = tmp_path / 'old_pilot'
    target = root / 'training/fault_robust/seed_22/checkpoints/update_000064.pt'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'test data (no real pickle)')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    record = root / 'selections/fault_robust/seed_22.json'
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({
        'selected_checkpoint': 'outputs/study_pilot/training/fault_robust/seed_22/checkpoints/update_000064.pt',
        'selected_checkpoint_sha256': digest,
    }))
    chosen, sha = _selected_from_pilot(root, 'fault_robust', 22)
    assert chosen == target and sha == digest
    target.write_bytes(b'changed!')
    with pytest.raises(FileNotFoundError, match='SHA-256'):
        _selected_from_pilot(root, 'fault_robust', 22)


def test_runner_blocks_test_only_scenario_without_touching_it(tmp_path):
    import json
    from dart_marl.research.bounded_sync import run_development_comparison
    root = tmp_path / 'old_pilot'
    root.mkdir()
    (root / 'protocol.json').write_text(json.dumps({
        'format': 'dart-marl-study-preregistered-v1',
        'protocol_sha256': 'dummy',
        'source_sha256': {}, 'resolved_config_sha256': {},
        'spec': {
            'name': 'malformed_DEVELOPMENT_ONLY',
            'training_variants': {'nominal': 'unused.yaml'},
            'training_seeds': [11],
            'validation_scenarios': ['configs/env/historical_stress_test.yaml'],
        }
    }))
    target = tmp_path / 'new_results'
    with pytest.raises(ValueError, match='test-only scenario'):
        run_development_comparison(root, target)
    assert not target.exists()


def test_frozen_pilot_rejected_without_evaluating(tmp_path):
    import json
    from dart_marl.research.bounded_sync import run_development_comparison
    root = tmp_path / 'old_pilot'
    root.mkdir()
    (root / 'frozen_development.json').write_text('{}')
    (root / 'protocol.json').write_text(json.dumps({
        'format': 'dart-marl-study-preregistered-v1',
        'protocol_sha256': 'dummy',
        'source_sha256': {}, 'resolved_config_sha256': {},
        'spec': {'name': 'DEVELOPMENT_ONLY',
                 'training_variants': {'nominal': 'unused.yaml'},
                 'training_seeds': [11], 'validation_scenarios': []},
    }))
    with pytest.raises(PermissionError, match='frozen'):
        run_development_comparison(root, tmp_path / 'new_results')


def test_bounded_sync_full_protocol_uses_fresh_seeds_and_explicit_primary():
    import yaml
    from pathlib import Path
    from dart_marl.research.study import _assert_spec
    old = yaml.safe_load(Path('configs/experiments/development_pilot.yaml').read_text())
    new = _assert_spec(yaml.safe_load(
        Path('configs/experiments/bounded_sync_study.yaml').read_text()))
    assert not set(old['training_seeds']) & set(new['training_seeds'])
    assert new['primary_method'] == 'r_cart_bounded_sync'
    assert {'r_cart', 'r_cart_no_sync', 'r_cart_bounded_sync', 'switch_aware'} <= set(
        new['allocation']['methods'])
    assert len(new['training_seeds']) == 10
