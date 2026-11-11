from dataclasses import replace

import numpy as np

from dart_marl.disruptions import DisruptionEvent
from dart_marl.envs import RoverEnv


def test_snapshot_independent_branches(reference_env):
    cfg = replace(reference_env.config,
                  events=(DisruptionEvent(step=2, kind="permanent_failure", rover=0),),
                  spawn_jitter=0.1)
    env = RoverEnv(cfg)
    env.reset(seed=42)
    snapshot = env.snapshot()
    a = env.clone()
    b = env.clone()
    act_a = np.zeros((12, 2))
    act_b = np.zeros((12, 2))
    act_a[0] = [1, 0]
    act_b[0] = [0, 1]
    a.step(act_a)
    b.step(act_b)
    assert a.positions[0, 0] != b.positions[0, 0]
    np.testing.assert_array_equal(env.positions, snapshot.positions)
    env.restore(snapshot)
    np.testing.assert_array_equal(env.positions, snapshot.positions)
    a.restore(snapshot)
    b.restore(snapshot)
    np.testing.assert_array_equal(a.step(act_a)[0], env.step(act_a)[0])
    np.testing.assert_array_equal(b.step(act_a)[0], env.observe())


def test_reset_seed_repeatable_and_snapshots_copy_buffers(reference_env):
    env = RoverEnv(replace(reference_env.config, spawn_jitter=0.5))
    obs1, _ = env.reset(seed=37)
    pos1 = env.positions.copy()
    for _ in range(4):
        env.step(np.zeros((12, 2)))
    obs2, _ = env.reset(seed=37)
    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(env.positions, pos1)
    snapshot = env.snapshot()
    snapshot.positions[0] = [0, 0]
    assert env.positions[0, 0] != 0
    env.restore(snapshot)
    snapshot.positions[0] = [9, 9]
    assert env.positions[0, 0] != 9


def test_two_branches_same_actions_same_results(reference_env):
    env = reference_env
    env.reset(seed=4)
    x, y = env.clone(), env.clone()
    actions = np.random.default_rng(13).uniform(-1, 1, (12, 2))
    for _ in range(5):
        ox, rx, tx, ux, ix = x.step(actions)
        oy, ry, ty, uy, iy = y.step(actions)
        np.testing.assert_array_equal(ox, oy)
        np.testing.assert_array_equal(ix["post_positions"], iy["post_positions"])
        assert (rx, tx, ux) == (ry, ty, uy)
