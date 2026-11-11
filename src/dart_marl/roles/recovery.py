"""legacy_study R-CART: interpretable finite-horizon recovery-cost-aware role transfer.

This is an *analytic forecast* rather than a learned recovery-cost model. CFT and
learned C_phi belong to legacy_study. The optimizer never reads event schedules,
physical-active masks, future positions, or future rewards. It is conditional on
coordinator-reported rover availability and current telemetry; known task
windows are public problem inputs.

The model forecasts coalition readiness from the slowest required partner,
then values service that can occur over the next H decisions. It explicitly
accounts for opportunity loss, switching, travel and synchronization wait.
Predictions are not claimed to be calibrated returns of the actual actor.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil

import numpy as np

from dart_marl.envs.rover import RoverEnv


@dataclass(frozen=True, slots=True)
class RecoveryCoalition:
    task: int
    members: tuple[int, ...]
    mask: int
    travel: float
    synchronization: float
    arrival_steps: int
    forecast_value: float
    retained: int
    # Dimensionless fraction of time a partner waits for the last arrival.
    normalized_synchronization: float = 0.0
    # Expected service under declared navigation-speed uncertainty (legacy_study).
    service_aware_value: float = 0.0


@dataclass(frozen=True, slots=True)
class RecoveryScore:
    """All terms have raw task-reward or explicitly weighted time units."""
    feasible: bool
    projected_return: float
    potential_return: float
    projected_loss: float
    travel: float
    synchronization: float
    switches: int
    objective: float
    ready_at_max: int
    selected_tasks: tuple[int, ...]
    synchronization_penalty_basis: float = 0.0
    nominal_projected_return: float = 0.0

    def components(self) -> dict[str, float]:
        return {
            "forecast_task_return": self.projected_return,
            "nominal_forecast_task_return": self.nominal_projected_return,
            "potential_task_return": self.potential_return,
            "forecast_opportunity_loss": self.projected_loss,
            "travel_time_sum": self.travel,
            "synchronization_wait_sum": self.synchronization,
            "synchronization_penalty_basis": self.synchronization_penalty_basis,
            "role_switches": float(self.switches),
            "max_coalition_ready_steps": float(self.ready_at_max),
            "objective": self.objective,
        }


def _eligible_task(env: RoverEnv, p: int) -> bool:
    task = env.config.pois[p]
    return not env.poi_completed[p] and env.step_index <= task.window[1]


def _horizon(env: RoverEnv, horizon_steps: int) -> int:
    return min(horizon_steps, env.config.horizon - env.step_index)


def _discounted_sum(start: int, end: int, now: int, discount: float) -> float:
    """Inclusive [start, end], with exponents relative to current decision."""
    if end < start:
        return 0.0
    return float(sum(discount ** (t - now) for t in range(start, end + 1)))


def task_value_in_horizon(env: RoverEnv, p: int, readiness: int,
                          horizon_steps: int, discount: float) -> float:
    """Task score if a fully staffed coalition arrives after `readiness` steps.

    A rover can move and earn service at the current step; an ETA of 0 allows
    service at `now`. For conservative integer ETA we use ceil(distance/speed).
    No future fault or simulator trajectory is queried.
    """
    if not _eligible_task(env, p):
        return 0.0
    task = env.config.pois[p]
    now = env.step_index
    end = min(now + _horizon(env, horizon_steps) - 1, task.window[1])
    first = max(now + readiness, task.window[0])
    if first > end:
        return 0.0
    if task.repeat:
        return task.value * _discounted_sum(first, end, now, discount)
    return task.value * discount ** (first - now)


# Fixed *design assumptions*, not fitted on the opened legacy_study test.
# This is a small discrete quadrature of possible navigation slowdowns; the
# allocator does not observe future actions or faults. 0.6/0.8/1.0 are
# explicit modeling assumptions to be checked, not calibrated probabilities.
SERVICE_SPEED_FACTORS = (1.0, 0.8, 0.6)
SERVICE_SPEED_WEIGHTS = (0.5, 0.3, 0.2)


def expected_service_value(env: RoverEnv, task: int, max_eta: float,
                           horizon_steps: int, discount: float) -> float:
    """Expected discounted *served-task reward*, using ETA uncertainty once.

    Each speed scenario affects the slowest partner's arrival, and therefore
    readiness; no extra early-arrival waiting cost is charged.  The factors
    approximate unmodeled navigation slowdown, *not* future failure forecasts.
    """
    return float(sum(prob * task_value_in_horizon(
        env, task, max(0, ceil(max_eta / factor - 1e-9)), horizon_steps, discount)
        for factor, prob in zip(SERVICE_SPEED_FACTORS, SERVICE_SPEED_WEIGHTS)))


def enumerate_recovery_coalitions(env: RoverEnv, roster: np.ndarray,
                                  previous: np.ndarray, *, horizon_steps: int,
                                  discount: float) -> list[list[RecoveryCoalition]]:
    """Only complete, skill-compatible, prospectively reachable coalitions."""
    result: list[list[RecoveryCoalition]] = []
    for p, task in enumerate(env.config.pois):
        options: list[RecoveryCoalition] = []
        if not _eligible_task(env, p):
            result.append(options)
            continue
        eligible = [int(i) for i in np.flatnonzero(roster)
                    if task.required_skill is None or env.skills[i] == task.required_skill]
        for members in combinations(eligible, task.coupling):
            arrivals = []
            for i in members:
                # Strict radius threshold, matching the simulator.
                distance = max(0.0, float(np.linalg.norm(
                    env.positions[i] - env.poi_positions[p])) - task.radius + 1e-6)
                speed = float(env.speeds[i])
                if speed <= 0 and distance > 0:
                    break
                arrivals.append(distance / max(speed, 1e-12))
            if len(arrivals) != task.coupling:
                continue
            max_arrival = max(arrivals)
            readiness = max(0, ceil(max_arrival - 1e-9))
            value = task_value_in_horizon(env, p, readiness, horizon_steps, discount)
            if value <= 0:
                continue
            options.append(RecoveryCoalition(
                task=p, members=members, mask=sum(1 << i for i in members),
                travel=float(sum(arrivals)),
                synchronization=float(sum(max_arrival - a for a in arrivals)),
                arrival_steps=readiness, forecast_value=float(value),
                retained=sum(int(previous[i] == p) for i in members),
                normalized_synchronization=float(
                    sum((max_arrival - a) / max(1.0, max_arrival) for a in arrivals)),
                service_aware_value=expected_service_value(
                    env, p, max_arrival, horizon_steps, discount),
            ))
        result.append(options)
    return result


def score_recovery_assignment(
    env: RoverEnv, roster: np.ndarray, previous: np.ndarray,
    roles: np.ndarray, *, horizon_steps: int, discount: float,
    travel_weight: float, switch_weight: float, sync_weight: float,
    candidates: list[list[RecoveryCoalition]] | None = None,
    sync_model: str = "raw",
) -> RecoveryScore:
    """Score one assignment, rejecting incomplete or infeasible coalitions.

    `potential_return` is a **task-service upper bound** assuming zero travel,
    not a policy oracle. All components are computed on reported roster only.
    """
    if sync_model not in ("raw", "normalized", "service"):
        raise ValueError("sync_model must be raw, normalized or service")
    n = env.config.max_agents
    roster, previous, roles = map(np.asarray, (roster, previous, roles))
    if (roles.shape != (n,) or not np.issubdtype(roles.dtype, np.integer)
            or np.any((roles < -1) | (roles >= len(env.config.pois)))
            or np.any(roles[~roster] != -1)):
        feasible = False
    else:
        feasible = True
    if candidates is None:
        candidates = enumerate_recovery_coalitions(
            env, roster, previous, horizon_steps=horizon_steps, discount=discount)
    ideal = sum(task_value_in_horizon(env, p, 0, horizon_steps, discount)
                for p in range(len(env.config.pois)))
    value = travel = sync = sync_basis = nominal_value = 0.0
    max_ready = 0
    tasks = []
    if feasible:
        for p, options in enumerate(candidates):
            members = tuple(int(i) for i in np.flatnonzero(roles == p))
            if not members:
                continue
            chosen = next((c for c in options if c.members == members), None)
            if chosen is None:
                feasible = False
                break
            tasks.append(p)
            nominal_value += chosen.forecast_value
            value += (chosen.service_aware_value if sync_model == "service"
                      else chosen.forecast_value)
            travel += chosen.travel
            sync += chosen.synchronization
            sync_basis += (chosen.synchronization if sync_model == "raw"
                           else chosen.normalized_synchronization if sync_model == "normalized"
                           else 0.0)
            max_ready = max(max_ready, chosen.arrival_steps)
    switches = int(np.sum(roster & (previous >= 0) & (previous != roles)))
    objective = (value - travel_weight * travel - switch_weight * switches
                 - sync_weight * sync_basis) if feasible else float("-inf")
    return RecoveryScore(feasible, float(value), float(ideal), float(ideal - value),
                         float(travel), float(sync), switches, float(objective),
                         max_ready, tuple(tasks), float(sync_basis), float(nominal_value))


def optimize_recovery_assignment(env: RoverEnv, roster: np.ndarray, previous: np.ndarray,
                                 *, horizon_steps: int, discount: float,
                                 travel_weight: float, switch_weight: float,
                                 sync_weight: float, sync_model: str = "raw"):
    """Exact subset DP on <=16 padded rover slots for the defined additive forecast.

    Returns selected assignment, score and number of fully-feasible coalitions.
    Complexity scales with masks/candidates, not a learned return oracle.
    """
    from dart_marl.roles.allocator import _better

    if sync_model not in ("raw", "normalized", "service"):
        raise ValueError("sync_model must be raw, normalized or service")
    n = env.config.max_agents
    options_by_task = enumerate_recovery_coalitions(
        env, roster, previous, horizon_steps=horizon_steps, discount=discount)
    total_options = sum(map(len, options_by_task))
    empty = tuple([-1] * n)
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, empty)}
    for options in options_by_task:
        new_dp = dp.copy()
        for occupied, (objective, old_roles) in dp.items():
            for choice in options:
                if occupied & choice.mask:
                    continue
                current = list(old_roles)
                for i in choice.members:
                    current[i] = choice.task
                new_roles = tuple(current)
                # A constant -switch_weight*N(previously assigned reported)
                # converts the retention credit to true switch cost.
                service_return = (choice.service_aware_value if sync_model == "service"
                                  else choice.forecast_value)
                sync_basis = (choice.synchronization if sync_model == "raw" else
                              choice.normalized_synchronization if sync_model == "normalized" else 0.0)
                new_objective = (objective + service_return
                                 - travel_weight * choice.travel
                                 - sync_weight * sync_basis
                                 + switch_weight * choice.retained)
                mask = occupied | choice.mask
                candidate = (new_objective, new_roles)
                if _better(new_objective, new_roles, new_dp.get(mask)):
                    new_dp[mask] = candidate
        dp = new_dp
    best: tuple[float, tuple[int, ...]] | None = None
    for option in dp.values():
        if _better(option[0], option[1], best):
            best = option
    assert best is not None
    roles = np.asarray(best[1], dtype=np.int32)
    score = score_recovery_assignment(
        env, roster, previous, roles, horizon_steps=horizon_steps,
        discount=discount, travel_weight=travel_weight,
        switch_weight=switch_weight, sync_weight=sync_weight,
        candidates=options_by_task, sync_model=sync_model)
    assert score.feasible
    return roles, score, total_options
