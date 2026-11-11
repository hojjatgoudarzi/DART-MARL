from dataclasses import replace

import numpy as np
import pytest

from dart_marl.envs import RoverEnv
from dart_marl.envs.config import POISpec


def test_one_shot_task_is_not_repeated(reference_env):
    cfg = reference_env.config
    pois = (replace(cfg.pois[0], repeat=False), cfg.pois[1])
    env = RoverEnv(replace(cfg, pois=pois))
    env.positions[:4] = [[10, 4], [10, 4], [10, 16], [10, 16]]
    _, first, *_ = env.step(np.zeros((12, 2)))
    assert first == 2
    assert env.poi_completed[0]
    assert env.preview_service()["reward"] == 1
    _, second, *_ = env.step(np.zeros((12, 2)))
    assert second == 1


def test_one_shot_completion_can_terminate(reference_env):
    cfg = reference_env.config
    env = RoverEnv(replace(cfg, pois=(replace(cfg.pois[0], repeat=False),)))
    env.positions[:2] = [10, 4]
    _, reward, terminated, truncated, _ = env.step(np.zeros((12, 2)))
    assert reward == 1 and terminated and not truncated
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros((12, 2)))


def test_step_uses_post_action_positions_and_current_t_clock(reference_env):
    cfg = reference_env.config
    p = POISpec(position=(10, 9), radius=0.4, coupling=2, value=7,
                window=(0, 0), repeat=False)
    env = RoverEnv(replace(cfg, pois=(p,)))
    a = np.zeros((12, 2))
    a[:2] = [0, -1]
    _, r, done, _, info = env.step(a)
    assert info["transition_step"] == 0
    assert r == 7 and done


def test_one_shot_expired_not_paid_out(reference_env):
    cfg = reference_env.config
    p = POISpec(position=(10, 4), radius=2, coupling=2, value=7,
                window=(0, 0), repeat=False)
    env = RoverEnv(replace(cfg, pois=(p,)))
    env.step(np.zeros((12, 2)))
    env.positions[:2] = [10, 4]
    assert env.preview_service()["reward"] == 0
    assert env.step(np.zeros((12, 2)))[1] == 0


def test_overlap_never_double_counts_same_rover(reference_env):
    cfg = reference_env.config
    p0 = replace(cfg.pois[0], position=(10, 10), radius=3)
    p1 = replace(cfg.pois[1], position=(10, 10), radius=3)
    env = RoverEnv(replace(cfg, pois=(p0, p1)))
    assert env.preview_service()["reward"] == 1  # all 4 go to POI 0 by tie
    roles = np.full(12, -1)
    roles[:2] = 0
    roles[2:4] = 1
    env.set_roles(roles)
    assert env.preview_service()["reward"] == 2  # two distinct physical coalitions


def test_skill_matching(reference_env):
    cfg = reference_env.config
    p0 = replace(cfg.pois[0], required_skill=1)
    env = RoverEnv(replace(cfg, pois=(p0, cfg.pois[1])))
    env.positions[:4] = [[10, 4], [10, 4], [10, 16], [10, 16]]
    assert env.preview_service()["reward"] == 1
    env.skills[:2] = 1
    assert env.preview_service()["reward"] == 2


def test_research_obs_role_target_and_padding():
    from pathlib import Path

    from dart_marl.envs import load_config

    root = Path(__file__).resolve().parents[1]
    env = RoverEnv(load_config(root / "configs/env/dev_roles.yaml"))
    env.reset(seed=3)
    assert env.observe().shape == (12, 23)
    assert np.all(env.observe()[6:] == 0)
    roles = np.full(12, -1, dtype=int)
    roles[:2] = 0
    env.set_roles(roles)
    o = env.observe()
    assert o[0, -1] == 1  # valid target flag
    assert o[2, -1] == 0  # idle rover
