"""Task-level learned-policy diagnostics and an explicitly non-learning reference."""

from __future__ import annotations

from typing import Any

import numpy as np

from dart_marl.envs.rover import RoverEnv


def role_following_actions(env: RoverEnv) -> np.ndarray:
    """Transparent *privileged* feasibility controller; uses exact world coordinates.

    This is a test/reference, not a fair local-observation MAPPO baseline.
    It neither transfers/reassigns roles nor observes future stochastic events.
    """
    actions = np.zeros((env.config.max_agents, 2), dtype=np.float32)
    for i in np.flatnonzero(env.active):
        p = int(env.roles[i])
        if p < 0 or env.poi_completed[p] or env.step_index > env.config.pois[p].window[1]:
            continue
        delta = env.poi_positions[p] - env.positions[i]
        distance = float(np.linalg.norm(delta))
        if distance > 1e-9:
            actions[i] = (delta / max(distance, 1.0)).astype(np.float32)
    return actions


class EpisodeDiagnostics:
    """Measure physical service (not role intent) from true simulator step info."""

    def __init__(self, env: RoverEnv):
        self.cfg = env.config
        self.total_return = 0.0
        self.length = 0
        self.events = 0
        n = len(self.cfg.pois)
        self.poi_service_steps = np.zeros(n, dtype=np.int64)
        self.poi_reward = np.zeros(n, dtype=np.float64)
        self.poi_coverage_total = np.zeros(n, dtype=np.int64)
        self.poi_eligible_steps = np.zeros(n, dtype=np.int64)
        self.previous_completed = env.poi_completed[:n].copy()
        self.role_positions = 0
        self.role_eligible = 0
        self.final_completed = self.previous_completed.copy()

    def record(self, env: RoverEnv, reward: float, info: dict[str, Any]) -> None:
        if self.length >= self.cfg.horizon:
            raise ValueError("diagnostic episode is longer than the environment horizon")
        n = len(self.cfg.pois)
        t = int(info["transition_step"])
        earned = np.asarray(info["poi_rewards"], dtype=np.float64)[:n]
        served = np.asarray(info["poi_served"], dtype=bool)[:n]
        coverage = np.asarray(info["poi_coverage"], dtype=np.int64)[:n]
        if not np.isclose(float(earned.sum()), reward, rtol=1e-6, atol=1e-6):
            raise ValueError("per-POI reward does not agree with scalar shared reward")
        eligible = np.array([p.window[0] <= t <= p.window[1] and not self.previous_completed[j]
                             for j, p in enumerate(self.cfg.pois)], dtype=bool)
        self.poi_service_steps += served
        self.poi_reward += earned
        self.poi_coverage_total += coverage
        self.poi_eligible_steps += eligible
        active = np.asarray(info["pre_active_mask"], dtype=bool)
        assigned = np.asarray(info["pre_roles"], dtype=np.int32)
        positions = np.asarray(info["post_positions"], dtype=np.float64)
        for i in np.flatnonzero(active):
            p = int(assigned[i])
            if p < 0 or not eligible[p]:
                continue
            self.role_eligible += 1
            poi = self.cfg.pois[p]
            if np.linalg.norm(positions[i] - env.poi_positions[p]) < poi.radius:
                self.role_positions += 1
        self.previous_completed = np.asarray(info["poi_completed"], dtype=bool)[:n].copy()
        self.final_completed = self.previous_completed.copy()
        self.events += len(info["events_next_boundary"])
        self.total_return += float(reward)
        self.length += 1

    def summary(self) -> dict[str, Any]:
        return {
            "return": self.total_return, "length": self.length,
            "events_after_decision": self.events,
            "poi_service_steps": self.poi_service_steps.tolist(),
            "poi_rewards": self.poi_reward.tolist(),
            "poi_coverage_total": self.poi_coverage_total.tolist(),
            "poi_eligible_steps": self.poi_eligible_steps.tolist(),
            "poi_served_at_least_once": (self.poi_service_steps > 0).astype(int).tolist(),
            "poi_completed": self.final_completed.astype(int).tolist(),
            "unique_pois_served": int((self.poi_service_steps > 0).sum()),
            "role_presence_fraction": (self.role_positions / self.role_eligible
                                       if self.role_eligible else None),
        }

    @staticmethod
    def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
        if not reports:
            raise ValueError("evaluation must contain at least one episode")
        returns = [r["return"] for r in reports]
        scalar = {"mean_return": float(np.mean(returns)),
                  "std_return": float(np.std(returns)), "returns": returns,
                  "mean_length": float(np.mean([r["length"] for r in reports])),
                  "episodes": len(reports),
                  "events_after_decision": sum(r["events_after_decision"] for r in reports),
                  "mean_unique_pois_served": float(np.mean([r["unique_pois_served"] for r in reports]))}
        for item in ("poi_service_steps", "poi_rewards", "poi_coverage_total",
                     "poi_eligible_steps", "poi_served_at_least_once", "poi_completed"):
            scalar["mean_" + item] = np.mean(np.asarray([r[item] for r in reports],
                                                            dtype=np.float64), axis=0).tolist()
        presence = [r["role_presence_fraction"] for r in reports
                    if r["role_presence_fraction"] is not None]
        scalar["mean_role_presence_fraction"] = float(np.mean(presence)) if presence else None
        scalar["per_episode"] = reports
        return scalar
