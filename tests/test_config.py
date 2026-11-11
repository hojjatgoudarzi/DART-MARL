from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dart_marl.disruptions import DisruptionEvent
from dart_marl.envs import RoverEnv, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,expected_agents,expected_pois", [
    ("professor_parity", 4, 2), ("dev_roles", 6, 3),
    ("dev_failure", 6, 3), ("main_mixed", 8, 5),
    ("main_slack", 8, 4), ("stress_12", 12, 6),
])
def test_all_configs(name, expected_agents, expected_pois):
    cfg = load_config(ROOT / f"configs/env/{name}.yaml")
    assert len(cfg.rovers) == expected_agents
    assert len(cfg.pois) == expected_pois
    env = RoverEnv(cfg)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (cfg.max_agents, env.observation_size)
    assert env.state()["task_features"].shape == (cfg.max_pois, 10)
    while not env.is_done:
        obs, _, terminated, truncated, _ = env.step(np.zeros((cfg.max_agents, 2)))
    assert terminated or truncated


def test_rejects_invalid_config_and_action(reference_env):
    with pytest.raises(ValueError):
        RoverEnv(replace(reference_env.config, max_agents=3))
    with pytest.raises(ValueError):
        RoverEnv(replace(reference_env.config, observation_mode="bad"))
    with pytest.raises(ValueError, match="shape"):
        reference_env.step(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="finite"):
        reference_env.step(np.full((12, 2), np.nan))
    with pytest.raises(ValueError):
        DisruptionEvent(step=0, kind="temporary_disable", rover=0)
