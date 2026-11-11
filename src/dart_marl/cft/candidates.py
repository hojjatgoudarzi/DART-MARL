"""Diverse feasible assignment shortlist for counterfactual rollout and deployment.

Bounded search approximates the top assignment set. The analytic optimum is
always inserted explicitly; learned selection is restricted to this same
shortlist (not an unbounded oracle).
"""
from __future__ import annotations

import numpy as np

from dart_marl.envs.rover import RoverEnv
from dart_marl.roles.recovery import (
    enumerate_recovery_coalitions, optimize_recovery_assignment,
    score_recovery_assignment,
)


def assignment_candidates(env: RoverEnv, roster: np.ndarray, previous: np.ndarray, *,
                          horizon: int = 24, discount: float = .98,
                          travel_weight: float = .1, switch_weight: float = 3.,
                          sync_weight: float = .75, k: int = 8,
                          beam: int = 300) -> list[np.ndarray]:
    if k < 1 or beam < k or env.config.max_agents > 12:
        raise ValueError('candidate count / search beam invalid')
    n = env.config.max_agents
    options = enumerate_recovery_coalitions(
        env, roster, previous, horizon_steps=horizon, discount=discount)
    # Search all task subsets with beam truncation. Same additive surrogate as
    # analytic optimizer; use explicit exact optimum insertion below.
    states: list[tuple[int, tuple[int,...], float]] = [(0, (-1,) * n, 0.)]
    for task_options in options:
        expanded = list(states)
        for used, roles, objective in states:
            for c in task_options:
                if used & c.mask:
                    continue
                new_roles = list(roles)
                for i in c.members:
                    new_roles[i] = c.task
                surrogate = (objective + c.forecast_value
                             - travel_weight * c.travel - sync_weight * c.synchronization
                             + switch_weight * c.retained)
                expanded.append((used | c.mask, tuple(new_roles), surrogate))
        # Retain multiple allocations for equal occupied masks; they can differ
        # materially in future outcomes and produce training supervision.
        expanded.sort(key=lambda a: (-a[2], a[1]))
        seen: set[tuple[int,...]] = set()
        states = []
        for entry in expanded:
            if entry[1] not in seen:
                states.append(entry)
                seen.add(entry[1])
                if len(states) == beam:
                    break
    exact, _, _ = optimize_recovery_assignment(
        env, roster, previous, horizon_steps=horizon, discount=discount,
        travel_weight=travel_weight, switch_weight=switch_weight, sync_weight=sync_weight)
    ranked = [exact]
    # Include the feasible incumbent when it exists, even if much worse than
    # analytic optimum, to teach genuine role-transfer value.
    incumbent = previous.astype(np.int32).copy()
    incumbent[~roster] = -1
    score = score_recovery_assignment(
        env, roster, previous, incumbent, horizon_steps=horizon, discount=discount,
        travel_weight=travel_weight, switch_weight=switch_weight,
        sync_weight=sync_weight, candidates=options)
    if score.feasible and not np.array_equal(incumbent, exact):
        ranked.append(incumbent)
    # Encourage exploration of diverse role vectors while favouring high-value
    # candidates. No counterfactual labels used in this selection.
    choices = [(np.asarray(a[1], dtype=np.int32), a[2]) for a in states]
    while len(ranked) < k:
        alternatives = [(roles, value) for roles, value in choices
                        if not any(np.array_equal(roles, chosen) for chosen in ranked)]
        if not alternatives:
            break
        max_score = max(v for _, v in alternatives)
        best = max(alternatives, key=lambda e: (
            (e[1] - max_score) / max(1., abs(max_score)) * 0.3 +
            min(int(np.sum(e[0] != a)) for a in ranked) / n,
            e[1], -sum((i + 1) * (p + 2) for i, p in enumerate(e[0]))))
        ranked.append(best[0])
    return ranked
