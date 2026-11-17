"""Directional professor-compatible sensing and role-conditioned local features.

No actor-facing sensor feature reveals all teammates' exact positions. For the
research variant, extra target coordinates come from explicit role messages.
"""

from __future__ import annotations

import math

import numpy as np


def directional_sensors(
    positions: np.ndarray,
    active: np.ndarray,
    sensor_radii: np.ndarray,
    poi_positions: np.ndarray,
    poi_visible: np.ndarray,
    poi_objectives: np.ndarray,
    *,
    n_sensors: int,
    n_objectives: int,
    poi_temperature: float,
    agent_temperature: float,
    density: bool,
    normalizer_agents: int,
    normalizer_pois: int,
) -> np.ndarray:
    """Return ``[slot, (num_objectives+1)*num_sensors]`` float32 features.

    Compass cones are measured counterclockwise from positive x using atan2;
    distance ``<= sensor_radius`` is visible (as in professor's observation).
    ``poi_visible`` controls task filtering; parity mode passes all POIs.
    Inactive rover slots neither see nor are seen.
    """
    n = len(active)
    output = np.zeros((n, (n_objectives + 1) * n_sensors), dtype=np.float32)
    cone_width = 2 * math.pi / n_sensors
    for i in np.flatnonzero(active):
        poi_counts = np.zeros((n_objectives, n_sensors), dtype=np.float64)
        poi_densities = np.zeros_like(poi_counts)
        agents_count = np.zeros(n_sensors, dtype=np.float64)
        agents_density = np.zeros(n_sensors, dtype=np.float64)
        for p in np.flatnonzero(poi_visible):
            delta = poi_positions[p] - positions[i]
            distance = float(np.linalg.norm(delta))
            if distance <= sensor_radii[i]:
                cone = min(int((math.atan2(delta[1], delta[0]) % (2 * math.pi))
                               // cone_width), n_sensors - 1)
                obj = int(poi_objectives[p])
                poi_counts[obj, cone] += 1
                poi_densities[obj, cone] += math.exp(-distance / poi_temperature)
        for j in np.flatnonzero(active):
            if i == j:
                continue
            delta = positions[j] - positions[i]
            distance = float(np.linalg.norm(delta))
            if distance <= sensor_radii[i]:
                cone = min(int((math.atan2(delta[1], delta[0]) % (2 * math.pi))
                               // cone_width), n_sensors - 1)
                agents_count[cone] += 1
                agents_density[cone] += math.exp(-distance / agent_temperature)
        if density:
            poi_values = np.divide(poi_densities, poi_counts,
                                   out=np.zeros_like(poi_counts), where=poi_counts > 0)
            agent_values = np.divide(agents_density, agents_count,
                                     out=np.zeros_like(agents_count), where=agents_count > 0)
        else:
            poi_values = poi_counts / max(normalizer_pois, 1)
            agent_values = agents_count / max(normalizer_agents, 1)
        output[i] = np.concatenate([poi_values.ravel(), agent_values]).astype(np.float32)
    return output
