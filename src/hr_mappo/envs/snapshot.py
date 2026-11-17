"""Deep snapshot of mutable simulator state and NumPy RNG state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class RoverSnapshot:
    step_index: int
    positions: np.ndarray
    active: np.ndarray
    speeds: np.ndarray
    sensor_radii: np.ndarray
    skills: np.ndarray
    roles: np.ndarray
    poi_completed: np.ndarray
    event_applied: np.ndarray
    reentry_schedule: list[tuple[int, int, int]]
    status_reports: list[tuple[int, int, bool]]
    ever_occupied: np.ndarray
    observation_rng_state: dict[str, Any]
    cumulative_reward: float
    latest_events: list[dict[str, Any]]
