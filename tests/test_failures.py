from dataclasses import replace

import numpy as np
import pytest

from dart_marl.disruptions import DisruptionEvent
from dart_marl.envs import RoverEnv


def _with_events(env, *events):
    return RoverEnv(replace(env.config, events=events))


def test_permanent_failure_at_t_zero_before_action(reference_env):
    env = _with_events(reference_env, DisruptionEvent(step=0, kind="permanent_failure", rover=1))
    o, info = env.reset(seed=9)
    assert not env.active[1] and not env.reported_active()[1]
    assert info["events_at_boundary"][0]["kind"] == "permanent_failure"
    assert (o[1] == 0).all()
    pos = env.positions[1].copy()
    a = np.zeros((12, 2))
    a[1] = [1, 0]
    env.step(a)
    np.testing.assert_allclose(env.positions[1], pos)


def test_failure_at_t_two_applied_before_next_observation(reference_env):
    env = _with_events(reference_env, DisruptionEvent(step=2, kind="permanent_failure", rover=1))
    env.reset(seed=1)
    a = np.zeros((12, 2))
    assert env.step(a)[4]["events_next_boundary"] == []
    result = env.step(a)
    assert result[4]["events_next_boundary"][0]["step"] == 2
    assert not env.active[1]
    assert (result[0][1] == 0).all()


def test_report_delay_does_not_delay_physical_failure(reference_env):
    env = _with_events(reference_env, DisruptionEvent(step=0, kind="permanent_failure",
                                                    rover=0, detection_delay=2))
    env.reset(seed=1)
    assert not env.active[0]
    assert env.reported_active()[0]
    z = np.zeros((12, 2))
    env.step(z)
    assert env.reported_active()[0]
    env.step(z)
    assert not env.reported_active()[0]


def test_temporary_disable_reenters_and_clears_role(reference_env):
    env = _with_events(reference_env, DisruptionEvent(step=1, kind="temporary_disable",
                                                    rover=0, duration=2))
    env.reset(seed=3)
    roles = np.full(12, -1)
    roles[0] = 1
    env.set_roles(roles)
    z = np.zeros((12, 2))
    env.step(z)
    assert not env.active[0] and env.roles[0] == -1
    env.step(z)
    assert not env.active[0]
    result = env.step(z)
    assert env.active[0]
    assert result[4]["events_next_boundary"][0]["kind"] == "automatic_reentry"
    assert env.roles[0] == -1


def test_replacement_uses_preallocated_slot(reference_env):
    ev = DisruptionEvent(step=0, kind="replacement_agent", rover=0,
                         new_slot=4, spawn=(4, 5))
    env = _with_events(reference_env, ev)
    env.reset(seed=0)
    assert not env.active[0] and env.active[4]
    np.testing.assert_array_equal(env.positions[4], [4, 5])
    assert env.speeds[4] == 1
    assert env.observe()[4].sum() > 0
    assert env.state()["active"][4]


def test_capability_degradation_excludes_speed_and_sensor_change(reference_env):
    ev = DisruptionEvent(step=0, kind="capability_degradation", rover=0,
                         speed_factor=0.5, sensor_factor=0.25, new_skill=2)
    env = _with_events(reference_env, ev)
    env.reset(seed=1)
    assert env.speeds[0] == 0.5 and env.sensor_radii[0] == 3
    assert env.skills[0] == 2
    a = np.zeros((12, 2))
    a[0] = [1, 0]
    env.step(a)
    assert env.positions[0, 0] == pytest.approx(10.5)


def test_inactive_roles_are_invalid(reference_env):
    env = _with_events(reference_env, DisruptionEvent(step=0, kind="permanent_failure", rover=0))
    roles = np.full(12, -1)
    roles[0] = 0
    with pytest.raises(ValueError, match="inactive"):
        env.set_roles(roles)


def test_fault_boundary_post_move_state_not_conflated_with_replacement(reference_env):
    event = DisruptionEvent(step=1, kind="replacement_agent", rover=0,
                            new_slot=4, spawn=(5, 6))
    env = _with_events(reference_env, event)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros((12, 2)))
    np.testing.assert_array_equal(info["post_positions"][4], [0, 0])
    np.testing.assert_array_equal(info["next_decision_positions"][4], [5, 6])
    assert not info["post_active_mask"][4]
    assert info["active_mask"][4]


def test_replacement_slot_cannot_be_recycled_without_explicit_lifecycle(reference_env):
    events = (
        DisruptionEvent(step=0, kind="replacement_agent", rover=0, new_slot=4),
        DisruptionEvent(step=2, kind="replacement_agent", rover=1, new_slot=4),
    )
    env = _with_events(reference_env, *events)
    env.reset(seed=0)
    with pytest.raises(RuntimeError, match="unused"):
        for _ in range(2):
            env.step(np.zeros((12, 2)))


def test_rejects_unknown_event_kind():
    with pytest.raises(ValueError, match="unknown"):
        DisruptionEvent(step=0, rover=0, kind="mystery")
