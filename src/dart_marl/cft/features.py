"""Causal, fixed-shape coordinator-visible features for role-assignment outcomes.

Do not introduce `env.active`, events, pending report queues or future reward here.
The visible roster can be stale due to delayed status detection. Rover positions,
speeds and skills are treated as last-available telemetry, as in legacy_study.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np

from dart_marl.envs.config import EnvironmentConfig
from dart_marl.envs.rover import RoverEnv

SCHEMA = 'dart-cft-visible-v1'


def task_problem_hash(config: EnvironmentConfig) -> str:
    """Bind predictor to task dynamics, not to a particular failure schedule."""
    obj = asdict(config)
    obj.pop('name', None)
    obj.pop('events', None)
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def visible_features(env: RoverEnv, reported: np.ndarray, previous: np.ndarray,
                     proposed: np.ndarray, *, horizon: int, discount: float) -> np.ndarray:
    """No access to hidden active mask or future scheduled faults.

    Stable slot schema across experiments with identical max_agents/max_pois.
    Policy actions are NOT included in this predictor input.
    """
    cfg = env.config
    n, m = cfg.max_agents, cfg.max_pois
    reported = np.asarray(reported, dtype=bool)
    previous = np.asarray(previous, dtype=np.int32)
    proposed = np.asarray(proposed, dtype=np.int32)
    if any(x.shape != (n,) for x in (reported, previous, proposed)):
        raise ValueError('role/roster shape mismatch')
    if np.any(proposed[~reported] != -1):
        raise ValueError('unreported agents must not receive candidate roles')
    if np.any((previous < -1) | (previous >= len(cfg.pois))) or np.any(
        (proposed < -1) | (proposed >= len(cfg.pois))):
        raise ValueError('invalid candidate roles')
    diag = float(np.linalg.norm(cfg.dimensions))
    sp_norm = max(1., max(r.speed for r in cfg.rovers))
    skill_norm = max(1., max(r.skill for r in cfg.rovers) + 1)
    max_value = max(1., max(p.value for p in cfg.pois))
    columns = []
    for i in range(n):
        # Never include env.active[i]. reported[i] can be stale on purpose.
        x = [float(reported[i]), float(i < len(cfg.rovers)),
             float(env.positions[i, 0] / cfg.dimensions[0]),
             float(env.positions[i, 1] / cfg.dimensions[1]),
             float(env.speeds[i] / sp_norm), float(env.skills[i] / skill_norm)]
        x += [float(previous[i] == k) for k in range(-1, m)]
        x += [float(proposed[i] == k) for k in range(-1, m)]
        p = int(proposed[i])
        eta = (float(np.linalg.norm(env.positions[i] - env.poi_positions[p])) /
               max(float(env.speeds[i]), 1e-9) if p >= 0 else 0.)
        x += [min(eta / max(diag, 1), 4.)]
        columns.extend(x)
    for p in range(m):
        if p < len(cfg.pois):
            task = cfg.pois[p]
            columns.extend([1., task.position[0] / cfg.dimensions[0],
                            task.position[1] / cfg.dimensions[1],
                            task.radius / diag, task.coupling / n,
                            task.value / max_value,
                            task.window[0] / cfg.horizon,
                            task.window[1] / cfg.horizon,
                            float(task.repeat), float(env.poi_completed[p]),
                            -1. if task.required_skill is None else task.required_skill / skill_norm])
        else:
            columns.extend([0.] * 11)
    remaining = max(0, cfg.horizon - env.step_index)
    switches = int(np.sum(reported & (previous >= 0) & (previous != proposed)))
    columns.extend([env.step_index / cfg.horizon, min(horizon, remaining) / cfg.horizon,
                    discount, float(reported.mean()), switches / n,
                    float(np.mean(env.poi_completed[:len(cfg.pois)]))])
    return np.asarray(columns, dtype=np.float32)
