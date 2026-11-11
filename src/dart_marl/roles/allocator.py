"""Deterministic coalition-feasible allocations without state mutation.

legacy_study baselines are preserved unchanged. legacy_study dispatches to R-CART
finite-horizon analytic recovery cost; no learned cost predictor exists yet.

The planner receives coordinator-visible roster, exact last-known rover
positions/skills/speeds, and current task status. The simulator physically masks
undelivered commands when an undisclosed failure occurs; the planner itself
never consults env.active. Capability telemetry is assumed immediately visible
when a planning call is made; no capability-report-delay model yet exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, isfinite

import numpy as np

from dart_marl.envs.rover import RoverEnv


@dataclass(frozen=True, slots=True)
class AllocationConfig:
    method: str = "switch_aware"
    travel_weight: float = 0.1
    switch_weight: float = 3.0
    periodic_steps: int = 0  # zero means pure event/task-change triggered
    forecast_horizon: int = 24  # R-CART, not used by legacy_study methods
    forecast_discount: float = 0.98
    sync_weight: float = 0.75
    hysteresis_margin: float = 1.0
    min_dwell_steps: int = 3
    cft_model_path: str | None = None
    cft_candidates: int = 8
    cft_beam: int = 300

    def __post_init__(self) -> None:
        if self.method not in {"fixed", "greedy", "switch_aware", "r_cart",
                               "r_cart_no_sync", "r_cart_no_switch",
                               "r_cart_no_hysteresis", "r_cart_bounded_sync", "r_cart_service", "cft", "cft_rank"}:
            raise ValueError("unknown role-allocation method")
        if not all(isfinite(float(v)) and v >= 0 for v in (
            self.travel_weight, self.switch_weight, self.sync_weight,
            self.hysteresis_margin,
        )) or self.periodic_steps < 0 or self.min_dwell_steps < 0 or self.forecast_horizon < 1 or not (
            isfinite(float(self.forecast_discount)) and 0 < self.forecast_discount <= 1
        ):
            raise ValueError("cost weights and periodic_steps must be nonnegative and finite")
        if self.cft_candidates < 1 or self.cft_beam < self.cft_candidates:
            raise ValueError("invalid CFT candidate/beam settings")
        if self.method in {"cft", "cft_rank"} and not self.cft_model_path:
            raise ValueError("CFT method requires --cft-model")


@dataclass(frozen=True, slots=True)
class Coalition:
    task: int
    members: tuple[int, ...]
    mask: int
    value: float
    travel: float
    retention: int


@dataclass(frozen=True, slots=True)
class Allocation:
    roles: np.ndarray
    method: str
    objective: float
    projected_tasks: tuple[int, ...]
    estimated_travel: float
    switches: int
    feasible_coalitions: int
    components: dict[str, float] | None = None


def _candidate_coalitions(env: RoverEnv, roster: np.ndarray, previous: np.ndarray,
                          config: AllocationConfig) -> list[list[Coalition]]:
    """Enumerate only fully staffed, compatible, reachable coalitions per task."""
    result: list[list[Coalition]] = []
    now = int(env.step_index)
    for p, task in enumerate(env.config.pois):
        candidates = []
        if env.poi_completed[p] or now > task.window[1]:
            result.append(candidates)
            continue
        eligible = [int(i) for i in np.flatnonzero(roster)
                    if task.required_skill is None or int(env.skills[i]) == task.required_skill]
        if len(eligible) < task.coupling:
            result.append(candidates)
            continue
        for members in combinations(eligible, task.coupling):
            travel_times = []
            for i in members:
                distance = max(0.0, float(np.linalg.norm(
                    env.positions[i] - env.poi_positions[p])) - task.radius + 1e-6)
                speed = float(env.speeds[i])
                if speed <= 0 and distance > 0:
                    break
                travel_times.append(distance / max(speed, 1e-12))
            if len(travel_times) != task.coupling:
                continue
            arrival = max(now + ceil(t - 1e-9) for t in travel_times)
            service_start = max(task.window[0], arrival)
            if service_start > task.window[1]:
                continue
            value = task.value * (task.window[1] - service_start + 1 if task.repeat else 1)
            candidates.append(Coalition(
                task=p, members=members, mask=sum(1 << i for i in members),
                value=float(value), travel=float(sum(travel_times)),
                retention=sum(int(previous[i] == p) for i in members),
            ))
        result.append(candidates)
    return result


def _better(score: float, roles: tuple[int, ...],
            incumbent: tuple[float, tuple[int, ...]] | None) -> bool:
    """Deterministic tie-breaking: task ID order before idle slots (-1 last)."""
    if incumbent is None or score > incumbent[0] + 1e-10:
        return True
    if abs(score - incumbent[0]) > 1e-10:
        return False
    return tuple(x if x >= 0 else 999 for x in roles) < tuple(
        x if x >= 0 else 999 for x in incumbent[1])


def allocate_roles(env: RoverEnv, roster: np.ndarray, previous: np.ndarray,
                   config: AllocationConfig) -> Allocation:
    """Return a deterministic assignment; do not modify the environment.

    `roster` MUST be the coordinator-reported mask, not the true active mask.
    `previous` is the coordinator's last commanded assignment including
    unacknowledged failures, not `env.roles` after automatic failure clearing.
    The exact DP uses one bit per slot and supports up to 16 reserved slots.
    """
    n = env.config.max_agents
    if n > 16:
        raise ValueError("exact coalition allocator supports max_agents <= 16")
    roster = np.asarray(roster)
    previous = np.asarray(previous)
    if roster.shape != (n,) or roster.dtype != np.bool_ or previous.shape != (n,):
        raise ValueError("invalid roster or previous role shape/dtype")
    if not np.issubdtype(previous.dtype, np.integer) or np.any(
        (previous < -1) | (previous >= len(env.config.pois))
    ):
        raise ValueError("invalid previous roles")
    if config.method in ("cft", "cft_rank"):
        raise ValueError("CFT allocation requires a fitted estimator through RoleCoordinator")
    if config.method.startswith("r_cart"):
        from dart_marl.roles.recovery import optimize_recovery_assignment
        effective_sync = (0.0 if config.method == "r_cart_no_sync" else config.sync_weight)
        effective_switch = (0.0 if config.method == "r_cart_no_switch" else config.switch_weight)
        roles, prediction, count = optimize_recovery_assignment(
            env, roster, previous, horizon_steps=config.forecast_horizon,
            discount=config.forecast_discount, travel_weight=config.travel_weight,
            switch_weight=effective_switch, sync_weight=effective_sync,
            sync_model=("service" if config.method == "r_cart_service"
                        else "normalized" if config.method == "r_cart_bounded_sync" else "raw"))
        return Allocation(roles, config.method, prediction.objective,
                          prediction.selected_tasks, prediction.travel,
                          prediction.switches, count, prediction.components())
    if config.method == "fixed":
        roles = previous.copy().astype(np.int32)
        roles[~roster] = -1
        return Allocation(roles, "fixed", 0.0, tuple(sorted(set(roles[roles >= 0]))),
                          0.0, 0, 0)
    candidates = _candidate_coalitions(env, roster, previous, config)
    count = sum(len(c) for c in candidates)
    if config.method == "greedy":
        roles = np.full(n, -1, dtype=np.int32)
        used = 0
        # Largest attainable projected value first; tie by POI ID. Then nearest coalition.
        order = sorted(range(len(candidates)), key=lambda p: (
            -max((c.value for c in candidates[p]), default=0.0), p))
        for p in order:
            choices = [c for c in candidates[p] if not c.mask & used]
            if not choices:
                continue
            best = min(choices, key=lambda c: (c.travel, c.members))
            if best.value <= config.travel_weight * best.travel:
                continue
            used |= best.mask
            for i in best.members:
                roles[i] = p
    else:
        # DP is globally optimal for the defined additive surrogate objective,
        # not the original non-additive time-dependent team-return objective.
        # State: allocated-rover bitmask -> (objective, immutable role tuple).
        base = tuple([-1] * n)
        dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, base)}
        for task_options in candidates:
            next_dp = dp.copy()  # skip this task
            for used, (score, current_roles) in dp.items():
                for c in task_options:
                    if used & c.mask:
                        continue
                    roles = list(current_roles)
                    for i in c.members:
                        roles[i] = c.task
                    new_roles = tuple(roles)
                    # Retention credit is equivalent to symmetric switching
                    # cost after subtracting the constant initial old-role cost.
                    bonus = (c.value - config.travel_weight * c.travel
                             + config.switch_weight * c.retention)
                    new_mask = used | c.mask
                    proposed = (score + bonus, new_roles)
                    if _better(proposed[0], proposed[1], next_dp.get(new_mask)):
                        next_dp[new_mask] = proposed
            dp = next_dp
        best: tuple[float, tuple[int, ...]] | None = None
        for value in dp.values():
            if _better(value[0], value[1], best):
                best = value
        assert best is not None
        roles = np.asarray(best[1], dtype=np.int32)
    # Recompute informative diagnostics of the selected assignment.
    total_value, total_travel = 0.0, 0.0
    fulfilled = []
    for p, options in enumerate(candidates):
        members = tuple(int(i) for i in np.flatnonzero(roles == p))
        selected = next((c for c in options if c.members == members), None)
        if selected is not None:
            fulfilled.append(p)
            total_value += selected.value
            total_travel += selected.travel
    switches = int(np.sum(roster & (previous >= 0) & (previous != roles)))
    objective = total_value - config.travel_weight * total_travel
    if config.method == "switch_aware":
        objective -= config.switch_weight * switches
    return Allocation(roles, config.method, float(objective), tuple(fulfilled),
                      float(total_travel), switches, count)
