"""Independent, finite-horizon Rover Domain implementation for DART-MARL.

Time convention: ``reset`` returns an observation for decision t=0 and applies
scheduled pre-action faults at t=0. ``step(a_t)`` moves currently active agents,
scores POST-movement positions at professor clock t, then applies pre-action
faults for the NEXT decision t+1 before returning its observation. Thus info
contains ``events_next_boundary`` and the returned observation sees them.

Reward is a SINGLE shared scalar, not N copies to be summed. Coupling uses
strict ``distance < radius``. Each active rover counts at most once per step;
when POI circles overlap we prioritize an in-range assigned POI, otherwise the
nearest normalized-distance POI, breaking ties by POI index. Role assignments
DO NOT automatically award reward; physical presence and skill are required.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import numpy as np

from dart_marl.envs.config import EnvironmentConfig
from dart_marl.envs.observation import directional_sensors
from dart_marl.envs.snapshot import RoverSnapshot


class RoverEnv:
    """Fixed-slot research simulator with Gym-like ``reset/step`` signatures."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self._rng = np.random.default_rng()
        self.step_index = 0
        self.positions = np.zeros((config.max_agents, 2), dtype=np.float64)
        self.active = np.zeros(config.max_agents, dtype=bool)
        self.speeds = np.zeros(config.max_agents, dtype=np.float64)
        self.sensor_radii = np.zeros(config.max_agents, dtype=np.float64)
        self.skills = np.zeros(config.max_agents, dtype=np.int32)
        self.roles = np.full(config.max_agents, -1, dtype=np.int32)
        self.poi_completed = np.zeros(config.max_pois, dtype=bool)
        self._event_applied = np.zeros(len(config.events), dtype=bool)
        self._reentry_schedule: list[tuple[int, int, int]] = []
        self._status_reports: list[tuple[int, int, bool]] = []
        self._initial_active = np.zeros(config.max_agents, dtype=bool)
        self._initial_active[: len(config.rovers)] = True
        self._ever_occupied = self._initial_active.copy()
        self._latest_events: list[dict[str, Any]] = []
        self.cumulative_reward = 0.0
        self.poi_positions = np.zeros((config.max_pois, 2), dtype=np.float64)
        self.poi_objectives = np.zeros(config.max_pois, dtype=np.int32)
        for p, poi in enumerate(config.pois):
            self.poi_positions[p] = poi.position
            self.poi_objectives[p] = poi.objective
        self.reset()

    @property
    def observation_size(self) -> int:
        core = self.config.num_sensors * (self.config.num_objectives + 1)
        return core if self.config.observation_mode.startswith("professor_") else core + 15

    @property
    def is_done(self) -> bool:
        all_one_shot = all(not p.repeat for p in self.config.pois)
        return (all_one_shot and bool(np.all(self.poi_completed[:len(self.config.pois)]))) \
            or self.step_index >= self.config.horizon

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to config and optionally reseed; returned observation is for action at t=0."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.config
        self.step_index = 0
        self.positions.fill(0)
        self.active[:] = self._initial_active
        self._ever_occupied[:] = self._initial_active
        self.speeds.fill(0)
        self.sensor_radii.fill(0)
        self.skills.fill(0)
        self.roles.fill(-1)
        self.poi_completed.fill(False)
        self._event_applied.fill(False)
        self._reentry_schedule.clear()
        self._status_reports.clear()
        self._latest_events.clear()
        self.cumulative_reward = 0.0
        for i, rover in enumerate(cfg.rovers):
            jitter = (self._rng.uniform(-cfg.spawn_jitter, cfg.spawn_jitter, size=2)
                      if cfg.spawn_jitter else 0.0)
            self.positions[i] = np.clip(np.asarray(rover.start) + jitter,
                                        0, cfg.dimensions)
            self.speeds[i] = rover.speed
            self.sensor_radii[i] = rover.sensor_radius
            self.skills[i] = rover.skill
        self._latest_events = self._apply_boundary_events()
        return self.observe(), {"decision_step": 0, "events_at_boundary": copy.deepcopy(self._latest_events),
                                "active_mask": self.active.copy()}

    def set_roles(self, roles: np.ndarray | list[int]) -> None:
        """Set each active rover's task ID or -1 (idle), without changing world reward.

        Failed slots must be idle. A planner may assign fewer than a POI's
        coupling, but only a feasible coalition can earn reward in the world.
        """
        roles = np.asarray(roles, dtype=np.int32)
        if roles.shape != (self.config.max_agents,):
            raise ValueError(f"roles shape must be ({self.config.max_agents},)")
        if np.any((roles < -1) | (roles >= len(self.config.pois))):
            raise ValueError("invalid POI ID")
        if np.any((~self.active) & (roles != -1)):
            raise ValueError("inactive rover cannot hold a role")
        self.roles[:] = roles

    def _apply_boundary_events(self) -> list[dict[str, Any]]:
        log: list[dict[str, Any]] = []
        for expiry, slot, report_delay in sorted(self._reentry_schedule):
            if expiry == self.step_index:
                self.active[slot] = True
                self._status_reports.append((self.step_index + report_delay, slot, True))
                log.append({"kind": "automatic_reentry", "step": self.step_index, "rover": slot})
        self._reentry_schedule = [e for e in self._reentry_schedule if e[0] != self.step_index]
        for idx, event in enumerate(self.config.events):
            if self._event_applied[idx] or event.step != self.step_index:
                continue
            i = event.rover
            if event.kind in {"permanent_failure", "temporary_disable", "replacement_agent"}:
                if not self.active[i]:
                    raise RuntimeError(f"fault target slot {i} is not active at t={self.step_index}")
                self.active[i] = False
                self.roles[i] = -1
                self._status_reports.append((self.step_index + event.detection_delay, i, False))
                if event.kind == "temporary_disable":
                    assert event.duration is not None
                    self._reentry_schedule.append((self.step_index + event.duration, i,
                                                   event.detection_delay))
                elif event.kind == "replacement_agent":
                    assert event.new_slot is not None
                    j = event.new_slot
                    if self._ever_occupied[j]:
                        raise RuntimeError(f"replacement slot {j} must be an unused padded slot")
                    self.active[j] = True
                    self._ever_occupied[j] = True
                    self.positions[j] = (self.positions[i] if event.spawn is None
                                         else np.asarray(event.spawn))
                    self.speeds[j] = self.speeds[i]
                    self.sensor_radii[j] = self.sensor_radii[i]
                    self.skills[j] = self.skills[i] if event.new_skill is None else event.new_skill
                    self.roles[j] = -1
                    self._status_reports.append((self.step_index + event.detection_delay, j, True))
            else:
                if not self.active[i]:
                    raise RuntimeError("cannot degrade inactive rover")
                self.speeds[i] *= event.speed_factor
                self.sensor_radii[i] *= event.sensor_factor
                if event.new_skill is not None:
                    self.skills[i] = event.new_skill
            self._event_applied[idx] = True
            log.append({"kind": event.kind, "step": event.step, "rover": i,
                        "detection_step": event.step + event.detection_delay,
                        "replacement_slot": event.new_slot})
        return log

    def reported_active(self) -> np.ndarray:
        """Coordinator-visible roster; fault reports obey configured detection delay."""
        visible = self._initial_active.copy()
        for report_at, slot, value in self._status_reports:
            if report_at <= self.step_index:
                visible[slot] = value
        return visible

    def _eligible_pois(self, t: int) -> np.ndarray:
        eligible = np.zeros(self.config.max_pois, dtype=bool)
        for p, poi in enumerate(self.config.pois):
            eligible[p] = (poi.window[0] <= t <= poi.window[1]
                           and not self.poi_completed[p])
        return eligible

    def _physical_coverage(self, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Non-mutating, exclusive rover-to-POI observation matching."""
        assigned = np.full(self.config.max_agents, -1, dtype=np.int32)
        observed_counts = np.zeros(self.config.max_pois, dtype=np.int32)
        eligible = self._eligible_pois(t)
        for i in np.flatnonzero(self.active):
            matches: list[tuple[int, float]] = []
            for p in np.flatnonzero(eligible):
                poi = self.config.pois[p]
                if poi.required_skill is not None and self.skills[i] != poi.required_skill:
                    continue
                d = float(np.linalg.norm(self.positions[i] - self.poi_positions[p]))
                if d < poi.radius:
                    matches.append((int(p), d / poi.radius))
            if matches:
                preferred = int(self.roles[i])
                if any(p == preferred for p, _ in matches):
                    chosen = preferred
                else:
                    chosen = min(matches, key=lambda item: (item[1], item[0]))[0]
                assigned[i] = chosen
                observed_counts[chosen] += 1
        served = np.zeros(self.config.max_pois, dtype=bool)
        for p, poi in enumerate(self.config.pois):
            served[p] = eligible[p] and observed_counts[p] >= poi.coupling
        return observed_counts, served, assigned

    def preview_service(self) -> dict[str, np.ndarray | float]:
        """Pure inspection at current decision t; NEVER marks a task complete."""
        counts, served, assigned = self._physical_coverage(self.step_index)
        values = np.array([p.value for p in self.config.pois], dtype=np.float64)
        return {"coverage": counts, "served": served,
                "serving_agents": assigned,
                "reward": float(np.dot(served[: len(values)], values))}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one synchronized joint action. Inactive action rows are ignored."""
        if self.is_done:
            raise RuntimeError("cannot step after episode finished; call reset")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.config.max_agents, 2):
            raise ValueError(f"actions shape must be ({self.config.max_agents}, 2)")
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        t = self.step_index
        pre_positions = self.positions.copy()
        pre_active = self.active.copy()
        pre_roles = self.roles.copy()
        clipped = np.clip(actions, -1, 1)
        norms = np.linalg.norm(clipped, axis=1)
        scales = np.minimum(1, np.divide(self.speeds, norms,
                                         out=np.ones_like(norms), where=norms > 0))
        deltas = clipped * scales[:, None]
        deltas[~self.active] = 0
        self.positions = np.clip(self.positions + deltas, 0, self.config.dimensions)
        counts, served, serving = self._physical_coverage(t)
        poi_rewards = np.zeros(self.config.max_pois, dtype=np.float64)
        for p, poi in enumerate(self.config.pois):
            if served[p]:
                poi_rewards[p] = poi.value
                if not poi.repeat:
                    self.poi_completed[p] = True
        reward = float(poi_rewards.sum())
        self.cumulative_reward += reward
        post_move_positions = self.positions.copy()
        post_move_active = self.active.copy()
        self.step_index = t + 1
        terminated = all(not p.repeat for p in self.config.pois) and bool(
            np.all(self.poi_completed[:len(self.config.pois)]))
        truncated = self.step_index >= self.config.horizon and not terminated
        events_next = []
        if not (terminated or truncated):
            events_next = self._apply_boundary_events()
        self._latest_events = events_next
        info = {
            "transition_step": t, "pre_positions": pre_positions,
            "post_positions": post_move_positions, "pre_active_mask": pre_active,
            "post_active_mask": post_move_active, "next_decision_positions": self.positions.copy(),
            "pre_roles": pre_roles, "poi_rewards": poi_rewards,
            "poi_coverage": counts, "poi_served": served,
            "serving_agents": serving, "poi_completed": self.poi_completed.copy(),
            "active_mask": self.active.copy(), "reported_active_mask": self.reported_active(),
            "events_next_boundary": copy.deepcopy(events_next),
            "team_reward": reward, "cumulative_reward": self.cumulative_reward,
        }
        return self.observe(), reward, terminated, truncated, info

    def observe(self) -> np.ndarray:
        """Local observation rows; inactive and padded slots are all zeros."""
        cfg = self.config
        parity = cfg.observation_mode.startswith("professor_")
        visible = np.zeros(cfg.max_pois, dtype=bool)
        for p, poi in enumerate(cfg.pois):
            visible[p] = parity or (not self.poi_completed[p] and self.step_index <= poi.window[1])
        raw = directional_sensors(
            self.positions, self.active, self.sensor_radii,
            self.poi_positions, visible, self.poi_objectives,
            n_sensors=cfg.num_sensors, n_objectives=cfg.num_objectives,
            poi_temperature=cfg.poi_sensor_temperature,
            agent_temperature=cfg.rover_sensor_temperature,
            density=cfg.observation_mode != "professor_count",
            normalizer_agents=len(cfg.rovers), normalizer_pois=len(cfg.pois),
        )
        if parity:
            return raw
        extra = np.zeros((cfg.max_agents, 15), dtype=np.float32)
        diag = float(np.linalg.norm(cfg.dimensions))
        max_value = max(p.value for p in cfg.pois) or 1.0
        for i in np.flatnonzero(self.active):
            extra[i, :2] = self.positions[i] / cfg.dimensions
            extra[i, 2] = self.speeds[i] / max(1e-9, max((r.speed for r in cfg.rovers), default=1))
            extra[i, 3] = self.sensor_radii[i] / diag
            extra[i, 4] = 1
            extra[i, 12] = self.skills[i] / max(1, max(r.skill for r in cfg.rovers) + 1)
            p = int(self.roles[i])
            if p >= 0:
                poi = cfg.pois[p]
                offset = self.poi_positions[p] - self.positions[i]
                extra[i, 5:7] = offset / cfg.dimensions
                extra[i, 7] = np.linalg.norm(offset) / diag
                extra[i, 8] = poi.value / max_value
                extra[i, 9] = poi.coupling / cfg.max_agents
                extra[i, 10] = max(0, poi.window[1] - self.step_index) / cfg.horizon
                extra[i, 11] = float(not self.poi_completed[p] and
                                     self.step_index <= poi.window[1])
                extra[i, 14] = 1
                extra[i, 13] = (0 if poi.required_skill is None else
                                (poi.required_skill + 1) / max(1, max(r.skill for r in cfg.rovers) + 2))
        return np.concatenate([raw, extra], axis=1)

    def state(self) -> dict[str, np.ndarray | int]:
        """Centralized full state (for future critic); never pass to local actor."""
        cfg = self.config
        task_features = np.zeros((cfg.max_pois, 10), dtype=np.float32)
        task_mask = np.zeros(cfg.max_pois, dtype=bool)
        for p, poi in enumerate(cfg.pois):
            task_mask[p] = True
            task_features[p] = [
                *np.divide(poi.position, cfg.dimensions), poi.radius / max(cfg.dimensions),
                poi.coupling / cfg.max_agents, poi.value,
                poi.window[0] / cfg.horizon, poi.window[1] / cfg.horizon,
                float(poi.repeat), float(self.poi_completed[p]),
                float(poi.required_skill if poi.required_skill is not None else -1),
            ]
        return {
            "step": self.step_index,
            "positions": self.positions.copy().astype(np.float32),
            "active": self.active.copy(), "reported_active": self.reported_active(),
            "speeds": self.speeds.copy().astype(np.float32),
            "sensor_radii": self.sensor_radii.copy().astype(np.float32),
            "skills": self.skills.copy(), "roles": self.roles.copy(),
            "task_features": task_features, "task_mask": task_mask,
            "poi_completed": self.poi_completed.copy(),
        }

    def snapshot(self) -> RoverSnapshot:
        return RoverSnapshot(
            step_index=self.step_index,
            positions=self.positions.copy(), active=self.active.copy(),
            speeds=self.speeds.copy(), sensor_radii=self.sensor_radii.copy(),
            skills=self.skills.copy(), roles=self.roles.copy(),
            poi_completed=self.poi_completed.copy(), event_applied=self._event_applied.copy(),
            reentry_schedule=copy.deepcopy(self._reentry_schedule),
            status_reports=copy.deepcopy(self._status_reports),
            ever_occupied=self._ever_occupied.copy(),
            observation_rng_state=copy.deepcopy(self._rng.bit_generator.state),
            cumulative_reward=self.cumulative_reward,
            latest_events=copy.deepcopy(self._latest_events),
        )

    def restore(self, snapshot: RoverSnapshot) -> None:
        """Restore deep independent branches; do not reuse writable snapshot buffers."""
        self.step_index = snapshot.step_index
        for name in ("positions", "active", "speeds", "sensor_radii", "skills",
                     "roles", "poi_completed"):
            setattr(self, name, getattr(snapshot, name).copy())
        self._event_applied = snapshot.event_applied.copy()
        self._reentry_schedule = copy.deepcopy(snapshot.reentry_schedule)
        self._status_reports = copy.deepcopy(snapshot.status_reports)
        self._ever_occupied = snapshot.ever_occupied.copy()
        self._rng.bit_generator.state = copy.deepcopy(snapshot.observation_rng_state)
        self.cumulative_reward = snapshot.cumulative_reward
        self._latest_events = copy.deepcopy(snapshot.latest_events)

    def clone(self) -> RoverEnv:
        """Counterfactual branch shares config but no mutable episode state or RNG."""
        other = RoverEnv(self.config)
        other.restore(self.snapshot())
        return other

    def resolved_config(self) -> dict[str, Any]:
        return asdict(self.config)
