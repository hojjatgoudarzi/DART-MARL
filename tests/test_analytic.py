"""legacy_study R-CART: exact forecast optimization, eligibility, timing, no oracle."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from dart_marl.disruptions.events import DisruptionEvent
from dart_marl.envs.config import EnvironmentConfig, POISpec, RoverSpec
from dart_marl.envs.rover import RoverEnv
from dart_marl.roles.allocator import AllocationConfig, allocate_roles
from dart_marl.roles.coordinator import RoleCoordinator
from dart_marl.roles.evaluation import evaluate_allocators
from dart_marl.roles.metrics import aggregate_recovery, event_recovery_metrics
from dart_marl.roles.recovery import (
    optimize_recovery_assignment,
    score_recovery_assignment,
    task_value_in_horizon,
)


def world(rovers, pois, *, events=(), horizon=30):
    return RoverEnv(EnvironmentConfig(
        name="analytic_test", dimensions=(25, 25), horizon=horizon,
        max_agents=12, max_pois=6,
        rovers=tuple(RoverSpec(start=tuple(point)) for point in rovers),
        pois=tuple(pois), events=tuple(events),
    ))


def poi(x, y, *, coupling=2, value=2.0, window=(0, 29), repeat=True, skill=None):
    return POISpec(position=(x, y), radius=1.0, coupling=coupling,
                   value=value, window=window, repeat=repeat, required_skill=skill)


def previous(env, *initial):
    return np.asarray([*initial, *([-1] * (env.config.max_agents - len(initial)))],
                      dtype=np.int32)


def test_horizon_prediction_respects_deadlines_repeat_completion_and_discount():
    env = world([(0, 0), (0, 0)], [
        poi(0, 0, value=2, window=(2, 6)),
        poi(0, 0, value=3, window=(0, 6), repeat=False),
    ])
    assert task_value_in_horizon(env, 0, 0, 4, 1.0) == pytest.approx(4.0)
    assert task_value_in_horizon(env, 0, 0, 4, 0.5) == pytest.approx(2 * (0.25 + 0.125))
    assert task_value_in_horizon(env, 1, 0, 4, 1.0) == pytest.approx(3.0)
    assert task_value_in_horizon(env, 1, 5, 4, 1.0) == 0
    env.poi_completed[1] = True
    assert task_value_in_horizon(env, 1, 0, 4, 1.0) == 0


def test_forecast_selects_near_service_when_horizon_precludes_distant_task():
    env = world([(1, 1), (1, 1)], [poi(1, 1, value=1.5),
                                       poi(9, 1, value=9.0)])
    prior = previous(env, -1, -1)
    short = allocate_roles(env, env.reported_active(), prior, AllocationConfig(
        "r_cart", forecast_horizon=4, forecast_discount=1.0))
    long = allocate_roles(env, env.reported_active(), prior, AllocationConfig(
        "switch_aware"))
    assert short.projected_tasks == (0,)
    assert long.projected_tasks == (1,)
    assert short.components["forecast_task_return"] > 0
    assert short.components["forecast_opportunity_loss"] > 0


def test_sync_penalty_changes_coalition_without_changing_task_selection():
    env = world([(0, 0), (3, 0), (0, 3)], [poi(0, 0, value=10)])
    prior = previous(env, -1, -1, -1)
    normal = allocate_roles(env, env.reported_active(), prior, AllocationConfig(
        "r_cart", forecast_horizon=8, forecast_discount=1,
        travel_weight=0.1, sync_weight=5, switch_weight=0))
    without = allocate_roles(env, env.reported_active(), prior, AllocationConfig(
        "r_cart_no_sync", forecast_horizon=8, forecast_discount=1,
        travel_weight=0.1, sync_weight=5, switch_weight=0))
    assert normal.projected_tasks == without.projected_tasks == (0,)
    assert tuple(np.flatnonzero(normal.roles == 0)) == (1, 2)
    assert 0 in np.flatnonzero(without.roles == 0)
    assert normal.components["synchronization_wait_sum"] == pytest.approx(0.0)


def test_exact_optimizer_matches_independent_brute_assignment_enumeration():
    env = world([(1, 1), (3, 2), (12, 1), (12, 2)],
                [poi(1, 1, value=2.0), poi(12, 1, value=2.5)])
    old = previous(env, 0, 1, 0, 1)
    cfg = AllocationConfig("r_cart", travel_weight=0.7, switch_weight=2.0,
                           sync_weight=0.25, forecast_horizon=9, forecast_discount=0.94)
    chosen = allocate_roles(env, env.reported_active(), old, cfg)
    optimum = float("-inf")
    for assignment in product((-1, 0, 1), repeat=4):
        roles = previous(env, *assignment)
        scored = score_recovery_assignment(
            env, env.reported_active(), old, roles,
            horizon_steps=cfg.forecast_horizon, discount=cfg.forecast_discount,
            travel_weight=cfg.travel_weight, switch_weight=cfg.switch_weight,
            sync_weight=cfg.sync_weight)
        if scored.feasible:
            optimum = max(optimum, scored.objective)
    assert chosen.objective == pytest.approx(optimum)
    assert chosen.components["objective"] == pytest.approx(chosen.objective)


def test_r_cart_never_activates_partial_or_incompatible_coalitions():
    env = world([(1, 1), (2, 1), (3, 1)], [poi(1, 1, coupling=2, skill=1)])
    before = env.snapshot()
    solution = allocate_roles(env, env.reported_active(), previous(env, -1, -1, -1),
                              AllocationConfig("r_cart"))
    assert solution.projected_tasks == ()
    assert np.all(solution.roles == -1)
    assert np.array_equal(env.positions, before.positions)
    assert np.array_equal(env.roles, before.roles)


def test_incompatible_assignment_is_not_protected_by_hysteresis():
    env = world([(1, 1), (1, 1), (1, 1)], [poi(1, 1, coupling=2)])
    prior = previous(env, 0, 0, -1)
    scored = score_recovery_assignment(
        env, np.array([True, False, True] + [False] * 9), prior, prior,
        horizon_steps=5, discount=1, travel_weight=0.1, switch_weight=3,
        sync_weight=0.5)
    assert not scored.feasible
    assert scored.objective == float("-inf")


def test_cooldown_holds_feasible_plan_but_not_after_dwell():
    env = world([(0, 0), (0, 0)],
                [poi(0, 0, value=20), poi(7, 0, value=1)])
    cfg = AllocationConfig("r_cart", periodic_steps=1, min_dwell_steps=3,
                           hysteresis_margin=0, forecast_horizon=12)
    controller = RoleCoordinator(cfg)
    controller.reset(env)
    wrong = previous(env, 1, 1)
    env.set_roles(wrong)
    controller.previous_commanded = wrong.copy()
    held = []
    for _ in range(3):
        env.step(np.zeros((12, 2)))
        decision = controller.maybe_reassign(env)
        assert decision is not None
        held.append(decision.get("hysteresis_held"))
    assert held == [True, True, False]
    assert env.roles[:2].tolist() == [0, 0]


def test_delayed_fault_not_visible_in_rcart_decision_before_report():
    env = world([(2, 2), (2, 2), (2, 2)], [poi(2, 2)], events=[
        DisruptionEvent(step=2, kind="permanent_failure", rover=0, detection_delay=3),
    ])
    planner = RoleCoordinator(AllocationConfig("r_cart"))
    planner.reset(env)
    for t in range(5):
        env.step(np.zeros((12, 2)))
        report = planner.maybe_reassign(env)
        if t < 4:
            assert report is None
        else:
            assert report["step"] == 5
            assert "roster_report" in report["trigger"]
            assert report["commanded_roles"][0] == -1
            assert report["hysteresis_held"] is False


def test_future_failure_event_schedule_is_not_used_by_planner():
    rovers = [(5, 5), (5, 6), (17, 17)]
    tasks = [poi(5, 5), poi(17, 17, coupling=1)]
    first = world(rovers, tasks)
    second = world(rovers, tasks, events=[
        DisruptionEvent(step=10, kind="permanent_failure", rover=1),
    ])
    prior = previous(first, 0, 0, 1)
    a = allocate_roles(first, first.reported_active(), prior, AllocationConfig("r_cart"))
    b = allocate_roles(second, second.reported_active(), prior, AllocationConfig("r_cart"))
    assert np.array_equal(a.roles, b.roles)
    assert a.objective == pytest.approx(b.objective)


def test_offline_recovery_metrics_censor_and_report_restoration():
    cfg = world([(1, 1), (1, 1)], [poi(1, 1)], events=[
        DisruptionEvent(2, "permanent_failure", rover=0),
    ], horizon=7).config
    service = [[1], [1], [0], [0], [1], [1], [1]]
    recovery = event_recovery_metrics(cfg, service, [1] * 7, horizon_steps=4)
    assert len(recovery) == 1
    assert recovery[0]["measurement_end_exclusive"] == 6
    assert recovery[0]["task_recovery"][0]["first_restoration_lag"] == 2
    assert recovery[0]["task_recovery"][0]["sustained_two_step_lag"] == 2
    metrics = aggregate_recovery([{"recovery_events": recovery}])
    assert metrics["num_first_restoration_censored"] == 0
    assert metrics["restoration_rate_among_at_risk"] == 1.0
    no_recovery = event_recovery_metrics(cfg, [[1], [1], [0], [0], [0], [0], [0]],
                                         [1] * 7, horizon_steps=4)
    assert no_recovery[0]["task_recovery"][0]["first_restoration_censored"] is True
    assert aggregate_recovery([{"recovery_events": no_recovery}])[
        "mean_first_restoration_lag_observed_only"] is None


def test_analytic_evaluation_reproducible_and_legacy_baselines_unchanged():
    kwargs = dict(episodes=2, seed=900000, controller="scripted", capture_trace=False,
                  methods=("fixed", "greedy", "switch_aware", "r_cart"))
    a = evaluate_allocators("configs/env/dev_failure.yaml", **kwargs)
    b = evaluate_allocators("configs/env/dev_failure.yaml", **kwargs)
    assert a == b
    assert a["evaluation_kind"] == "analytic_recovery"
    assert len(a["methods"]["r_cart"]["per_episode"]) == 2
    assert "r_cart_minus_switch_aware" in a["paired_differences"]
    assert a["methods"]["fixed"]["mean_return"] == pytest.approx(102.9)
    for policy in a["methods"].values():
        assert policy["recovery_metrics"]["num_events"] == 4


@pytest.mark.parametrize("kw", [
    {"forecast_horizon": 0}, {"forecast_discount": 1.2},
    {"forecast_discount": float("nan")}, {"sync_weight": -1},
    {"min_dwell_steps": -1}, {"hysteresis_margin": -1},
])
def test_invalid_planner_parameters_are_rejected(kw):
    with pytest.raises(ValueError):
        AllocationConfig("r_cart", **kw)
