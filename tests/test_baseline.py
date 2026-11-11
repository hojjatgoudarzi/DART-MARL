"""Allocator correctness, delayed information, strict task feasibility, reproducibility."""

from __future__ import annotations


import numpy as np
import pytest

from dart_marl.disruptions.events import DisruptionEvent
from dart_marl.envs.config import EnvironmentConfig, POISpec, RoverSpec, load_config
from dart_marl.envs.rover import RoverEnv
from dart_marl.roles.allocator import AllocationConfig, allocate_roles
from dart_marl.roles.coordinator import RoleCoordinator
from dart_marl.roles.evaluation import evaluate_allocators


def _env(rovers, pois, events=(), horizon=20):
    return RoverEnv(EnvironmentConfig(
        name="baseline_test", dimensions=(20, 20), horizon=horizon,
        rovers=tuple(rovers), pois=tuple(pois),
        max_agents=12, max_pois=6, events=tuple(events)))


def _rover(x, y, skill=0):
    return RoverSpec(start=(x, y), skill=skill)


def _poi(x, y, coupling=2, value=2, required_skill=None):
    return POISpec(position=(x, y), radius=1.0, coupling=coupling, value=value,
                   window=(0, 19), required_skill=required_skill)


@pytest.mark.parametrize("method", ["greedy", "switch_aware"])
def test_no_partial_coalitions_or_incompatible_skills(method):
    env = _env([_rover(0, 0, 0), _rover(0, 0, 1), _rover(0, 0, 0)],
               [_poi(0, 0, coupling=2, required_skill=1),
                _poi(10, 0, coupling=2, required_skill=0)])
    prev = np.full(12, -1, np.int32)
    before = env.snapshot()
    result = allocate_roles(env, env.reported_active(), prev, AllocationConfig(method))
    assert result.projected_tasks == (1,)
    assert list(result.roles[:3]) == [1, -1, 1]
    assert np.array_equal(env.roles, before.roles)
    assert np.array_equal(env.positions, before.positions)
    assert np.array_equal(env.active, before.active)


def test_joint_allocator_avoids_greedy_skill_trap():
    env = _env([_rover(0.1, 0, 1), _rover(0.2, 0), _rover(1, 0)],
               [_poi(0, 0, coupling=2, value=5),
                _poi(0, 0, coupling=1, value=4, required_skill=1)])
    previous = np.full(12, -1, dtype=np.int32)
    greedy = allocate_roles(env, env.reported_active(), previous, AllocationConfig("greedy"))
    dp = allocate_roles(env, env.reported_active(), previous, AllocationConfig("switch_aware"))
    assert greedy.projected_tasks == (0,)
    assert dp.projected_tasks == (0, 1)
    assert dp.objective > greedy.objective


def test_switch_cost_avoids_needless_reassignment():
    env = _env([_rover(14, 5), _rover(14, 6), _rover(5, 5), _rover(5, 6)],
               [_poi(5, 5, value=3), _poi(14, 5, value=3)])
    prev = np.array([0, 0, 1, 1] + [-1] * 8, np.int32)
    greedy = allocate_roles(env, env.reported_active(), prev, AllocationConfig(
        "greedy", travel_weight=1, switch_weight=100))
    aware = allocate_roles(env, env.reported_active(), prev, AllocationConfig(
        "switch_aware", travel_weight=1, switch_weight=100))
    assert greedy.roles[:4].tolist() == [1, 1, 0, 0]
    assert aware.roles[:4].tolist() == [0, 0, 1, 1]
    assert aware.switches == 0


def test_delayed_failure_report_does_not_trigger_early_reallocation():
    env = _env([_rover(2, 2), _rover(2, 2), _rover(2, 2)],
               [_poi(2, 2, 2)], events=[DisruptionEvent(
                   step=2, kind="permanent_failure", rover=0, detection_delay=3)])
    env.reset(seed=17)
    planner = RoleCoordinator(AllocationConfig("switch_aware"))
    planner.reset(env)
    assert int(planner.previous_commanded[0]) == 0
    for _ in range(5):
        current = env.step_index
        _, _, _, _, _ = env.step(np.zeros((12, 2)))
        decision = planner.maybe_reassign(env)
        if current < 4:
            assert decision is None  # even though rover died before t=2 action
        else:
            assert decision is not None
            assert decision["step"] == 5
            assert "roster_report" in decision["trigger"]
            assert decision["commanded_roles"][0] == -1
    assert planner.summary()["decision_steps"] == [5]


def test_replacement_waits_until_reported_then_receives_role():
    env = _env([_rover(2, 2), _rover(2, 2)],
               [_poi(2, 2, 2)], events=[DisruptionEvent(
                   step=1, kind="replacement_agent", rover=0, new_slot=2,
                   spawn=(2, 2), detection_delay=2)])
    planner = RoleCoordinator(AllocationConfig("greedy"))
    planner.reset(env)
    for _ in range(3):
        env.step(np.zeros((12, 2)))
        d = planner.maybe_reassign(env)
    assert d is not None and d["step"] == 3
    assert env.roles[2] == 0
    assert env.roles[0] == -1
    assert planner.summary()["role_switches"] >= 1


def test_inactive_and_expired_tasks_never_receive_roles():
    env = _env([_rover(1, 1), _rover(1, 1)],
               [_poi(1, 1, coupling=2)], horizon=20)
    env.poi_completed[0] = True
    result = allocate_roles(env, env.reported_active(), np.full(12, -1),
                            AllocationConfig("switch_aware"))
    assert (result.roles == -1).all()
    assert result.projected_tasks == ()


def test_delayed_coordinator_not_affected_by_physical_alive_mask_when_report_unchanged():
    env = _env([_rover(1, 1), _rover(1, 1)], [_poi(1, 1)],
               events=[DisruptionEvent(1, "permanent_failure", 0, detection_delay=4)])
    c = RoleCoordinator(AllocationConfig("greedy", periodic_steps=0))
    c.reset(env)
    env.step(np.zeros((12, 2)))
    assert env.active[0] == 0 and env.reported_active()[0] == 1
    assert c.maybe_reassign(env) is None
    assert c.previous_commanded[0] == 0
    assert env.roles[0] == -1


def test_fixed_never_reallocates_and_initial_roles_shared():
    cfg = load_config("configs/env/dev_failure.yaml")
    roles = []
    for method in ("fixed", "greedy", "switch_aware"):
        env = RoverEnv(cfg)
        env.reset(seed=444)
        c = RoleCoordinator(AllocationConfig(method))
        c.reset(env)
        roles.append(env.roles.copy())
        if method == "fixed":
            for _ in range(35):
                env.step(np.zeros((cfg.max_agents, 2)))
                assert c.maybe_reassign(env) is None
            assert c.summary()["decision_count"] == 0
    assert all(np.array_equal(roles[0], r) for r in roles[1:])


def test_repeatable_paired_evaluation_and_identical_seeds():
    kw = dict(episodes=2, seed=99, controller="scripted", capture_trace=False)
    a = evaluate_allocators("configs/env/dev_failure.yaml", **kw)
    b = evaluate_allocators("configs/env/dev_failure.yaml", **kw)
    assert a == b
    for method in ("fixed", "greedy", "switch_aware"):
        assert [r["seed"] for r in a["methods"][method]["per_episode"]] == [99, 1108]
        assert a["methods"][method]["mean_unique_pois_served"] >= 0
    assert "switch_aware_minus_fixed" in a["paired_differences"]
    assert len(a["paired_differences"]["switch_aware_minus_fixed"]["per_seed_differences"]) == 2


def test_parameter_validation():
    with pytest.raises(ValueError):
        AllocationConfig("something")
    with pytest.raises(ValueError):
        AllocationConfig("greedy", travel_weight=float("nan"))
    with pytest.raises(ValueError):
        evaluate_allocators("configs/env/dev_roles.yaml", episodes=0)
    with pytest.raises(ValueError):
        evaluate_allocators("configs/env/dev_roles.yaml", controller="checkpoint")


def test_switch_aware_exact_matches_brute_force_small_problem():
    """Independent exhaustive enumeration of assignments tests the DP optimizer."""
    from itertools import product

    env = _env([_rover(2, 2), _rover(3, 2), _rover(12, 3), _rover(13, 3)],
               [_poi(2, 2, value=2), _poi(13, 3, value=3)])
    previous = np.array([1, 0, 0, 1] + [-1] * 8, np.int32)
    settings = AllocationConfig("switch_aware", travel_weight=0.7, switch_weight=4.0)
    solution = allocate_roles(env, env.reported_active(), previous, settings)
    best = float("-inf")
    for assignment in product((-1, 0, 1), repeat=4):
        score, valid = 0.0, True
        for p, poi in enumerate(env.config.pois):
            members = [i for i, role in enumerate(assignment) if role == p]
            if not members:
                continue
            if len(members) != poi.coupling:
                valid = False
                break
            times = [max(0, np.linalg.norm(env.positions[i] - env.poi_positions[p])
                         - poi.radius + 1e-6) / env.speeds[i] for i in members]
            arrival = int(np.ceil(max(times) - 1e-9))
            if arrival > poi.window[1]:
                valid = False
                break
            score += poi.value * (poi.window[1] - arrival + 1)
            score -= settings.travel_weight * sum(times)
        if valid:
            score -= settings.switch_weight * sum(
                old >= 0 and new != old for old, new in zip(previous[:4], assignment))
            best = max(best, score)
    assert solution.objective == pytest.approx(best, abs=1e-9)


def test_periodic_replan_is_explicit_and_does_not_need_fault_oracle():
    env = _env([_rover(2, 2), _rover(2, 2)], [_poi(2, 2)])
    c = RoleCoordinator(AllocationConfig("greedy", periodic_steps=3))
    c.reset(env)
    actions = np.zeros((env.config.max_agents, 2), dtype=np.float32)
    for i in range(3):
        env.step(actions)
        record = c.maybe_reassign(env)
        if i < 2:
            assert record is None
        else:
            assert record is not None and record["trigger"] == "periodic"
