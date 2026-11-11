"""Reproducible fault-pattern training curriculum; never sample held-out test YAML."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from dart_marl.disruptions.events import DisruptionEvent
from dart_marl.envs.config import EnvironmentConfig


def diverse_scenario(base: EnvironmentConfig, seed: int) -> EnvironmentConfig:
    """Predeclared 0–3-event distribution; only event schedules change.

    The episode seed controls the event tape. The actor and critic receive
    only current observations/state, not this future tape. Different physical
    targets avoid overlapping failure/reentry events on one rover.
    """
    rng = np.random.default_rng(seed ^ 0xB17A0)
    n = len(base.rovers)
    if n < 3 or base.horizon < 14:
        raise ValueError("diverse scenarios require >=3 rover slots and horizon >=14")
    count = seed % 4
    order = rng.permutation(n)[:count]
    times = np.sort(rng.choice(np.arange(4, base.horizon - 4), size=count,
                               replace=False))
    events = []
    replacement_used = False
    for slot, tick in zip(order, times):
        kind = ("permanent_failure", "temporary_disable", "replacement_agent",
                "capability_degradation")[int(rng.integers(4))]
        kw = {}
        if kind == "replacement_agent":
            if replacement_used or n >= base.max_agents:
                kind = "permanent_failure"
            else:
                kw = {"new_slot": n, "spawn": tuple(float(x) for x in base.rovers[int(slot)].start)}
                replacement_used = True
        if kind == "temporary_disable":
            kw["duration"] = min(int(rng.integers(3, 9)), base.horizon - int(tick) - 1)
        if kind == "capability_degradation":
            kw.update({"speed_factor": float(rng.choice([.25, .5, .75])),
                       "sensor_factor": float(rng.choice([.5, .8, 1.]))})
        events.append(DisruptionEvent(step=int(tick), rover=int(slot), kind=kind,
                                      detection_delay=int(rng.integers(0, 6)), **kw))
    return replace(base, name=f"cft_diverse_{seed}", events=tuple(events))
