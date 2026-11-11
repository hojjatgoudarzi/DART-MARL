import math

import numpy as np
import pytest


def test_reference_layout_observation_and_initial_reward(reference_env):
    env = reference_env
    obs, info = env.reset(seed=11)
    assert obs.shape == (12, 8)
    np.testing.assert_array_equal(info["active_mask"][:4], [True] * 4)
    np.testing.assert_array_equal(obs[4:], 0)
    assert env.preview_service()["reward"] == 0
    # Reference POIs are 6 units away in north/south cones: density exp(-6/2).
    assert obs[0, 1] == pytest.approx(math.exp(-3), rel=1e-6)
    assert obs[0, 3] == pytest.approx(math.exp(-3), rel=1e-6)
    np.testing.assert_array_equal(obs[0, 4:], [1, 0, 0, 0])  # 3 coincident rovers


def test_reference_both_coalitions_reward_two(reference_env):
    env = reference_env
    env.positions[:4] = [[10, 4], [10, 4], [10, 16], [10, 16]]
    state_before = env.poi_completed.copy()
    assert env.preview_service()["reward"] == 2
    np.testing.assert_array_equal(env.poi_completed, state_before)  # pure preview
    _, r, terminated, truncated, info = env.step(np.zeros((12, 2)))
    assert r == 2 and not terminated and not truncated
    np.testing.assert_array_equal(info["poi_rewards"][:2], [1, 1])


def test_failure_drops_two_to_one_even_with_fixed_positions(reference_env):
    env = reference_env
    env.positions[:4] = [[10, 4], [10, 4], [10, 16], [10, 16]]
    env.active[3] = False
    assert env.preview_service()["reward"] == 1
    _, reward, *_ = env.step(np.zeros((12, 2)))
    assert reward == 1


def test_speed_magnitude_not_forced_to_max(reference_env):
    env = reference_env
    action = np.zeros((12, 2))
    action[0] = [0.2, 0]
    action[1] = [1, 1]
    env.step(action)
    np.testing.assert_allclose(env.positions[0], [10.2, 10])
    np.testing.assert_allclose(env.positions[1], [10 + math.sqrt(0.5)] * 2)


def test_strict_radius_boundary(reference_env):
    env = reference_env
    env.positions[:4] = [[10, 4], [12, 4], [10, 16], [10, 16]]
    assert env.preview_service()["reward"] == 1  # distance exactly 2 excluded


def test_directional_density_and_count_consistency(reference_env):
    env = reference_env
    env.positions[:4] = [[10, 10], [10, 10], [10, 10], [10, 10]]
    density = env.observe()
    assert density.shape[1] == 8
    assert density[0, 4] == pytest.approx(1.0)
    assert density[0, 1] == pytest.approx(math.exp(-3))
