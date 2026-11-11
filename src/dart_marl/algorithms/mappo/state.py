"""Explicit, padded CTDE state encoding. Never supplied to the local actor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dart_marl.envs.rover import RoverEnv

AGENT_FEATURES = 9
POI_FEATURES = 10


@dataclass(frozen=True)
class CriticInput:
    agents: np.ndarray  # [N, 9]
    pois: np.ndarray  # [P, 10]
    active: np.ndarray  # [N] physical roster, not delayed coordinator report
    task_mask: np.ndarray  # [P]


def critic_input(env: RoverEnv) -> CriticInput:
    """Build an independent copy; zeros in non-active entity slots."""
    s = env.state()
    cfg = env.config
    alive = np.asarray(s["active"], dtype=np.bool_)
    n = cfg.max_agents
    agents = np.zeros((n, AGENT_FEATURES), dtype=np.float32)
    agents[:, 0:2] = np.asarray(s["positions"]) / np.asarray(cfg.dimensions)
    agents[:, 2] = alive
    agents[:, 3] = s["reported_active"]
    agents[:, 4] = np.asarray(s["speeds"]) / max(1.0, max(r.speed for r in cfg.rovers))
    agents[:, 5] = np.asarray(s["sensor_radii"]) / float(np.linalg.norm(cfg.dimensions))
    agents[:, 6] = np.asarray(s["skills"]) / max(1, max(r.skill for r in cfg.rovers) + 1)
    agents[:, 7] = (np.asarray(s["roles"]) + 1) / (cfg.max_pois + 1)
    agents[:, 8] = float(s["step"]) / cfg.horizon
    agents[~alive] = 0
    pois = np.asarray(s["task_features"], dtype=np.float32).copy()
    # Values are bounded by the maximum configured POI value, avoiding scale changes by layout.
    pois[:, 4] /= max(1.0, max(p.value for p in cfg.pois))
    return CriticInput(
        agents=agents,
        pois=pois,
        active=alive.copy(),
        task_mask=np.asarray(s["task_mask"], dtype=np.bool_).copy(),
    )
