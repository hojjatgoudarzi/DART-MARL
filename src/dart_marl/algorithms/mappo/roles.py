"""Transparent fixed *training* role initialization, NOT DART or a role planner."""

from __future__ import annotations

import numpy as np

from dart_marl.envs.rover import RoverEnv


def assign_initial_roles(env: RoverEnv, mode: str = "initial_coupling") -> None:
    """Assign physically/skill-feasible initial coalitions, with stable tie-breaking.

    The assignment is frozen for the whole episode. On disruption, the RoverEnv
    clears an inactive rover's role and we do not redistribute it. Therefore
    this is a legitimate no-reallocation baseline, not the future DART method.
    """
    roles = np.full(env.config.max_agents, -1, dtype=np.int32)
    if mode == "none":
        env.set_roles(roles)
        return
    if mode != "initial_coupling":
        raise ValueError(f"unknown role initialization: {mode}")
    available = set(np.flatnonzero(env.active).tolist())
    ordered = sorted(range(len(env.config.pois)),
                     key=lambda p: (-env.config.pois[p].value, p))
    for p in ordered:
        poi = env.config.pois[p]
        eligible = [i for i in available
                    if poi.required_skill is None or env.skills[i] == poi.required_skill]
        eligible.sort(key=lambda i, target=poi.position: (
            float(np.linalg.norm(env.positions[i] - target)), i))
        if len(eligible) < poi.coupling:
            continue
        for i in eligible[:poi.coupling]:
            roles[i] = p
            available.remove(i)
    # Remaining team members provide redundancy to a nearby compatible task.
    for i in sorted(available):
        options = [p for p in range(len(env.config.pois))
                   if env.config.pois[p].required_skill is None
                   or env.skills[i] == env.config.pois[p].required_skill]
        if options:
            roles[i] = min(options,
                           key=lambda p, rover=i: (
                               float(np.linalg.norm(env.positions[rover] - env.poi_positions[p])), p))
    env.set_roles(roles)


def role_potential(env: RoverEnv) -> float:
    """Negative valid-role distance potential for optional episode-boundary PBRS."""
    eligible = np.zeros(env.config.max_pois, dtype=bool)
    for p, poi in enumerate(env.config.pois):
        eligible[p] = not env.poi_completed[p] and env.step_index <= poi.window[1]
    valid_roles = (env.roles >= 0)
    valid_roles[valid_roles] = eligible[env.roles[valid_roles]]
    participating = np.flatnonzero(env.active & valid_roles)
    if not len(participating):
        return 0.0
    d = np.linalg.norm(env.positions[participating] - env.poi_positions[env.roles[participating]],
                       axis=1) / float(np.linalg.norm(env.config.dimensions))
    return -float(d.sum() / max(1, len(env.config.rovers)))
