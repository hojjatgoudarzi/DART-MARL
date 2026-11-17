"""Explicit execution-information firewall: only own state, public task metadata,
nearby teammates, and own locally sensed completion. No coordinator or reports.
"""
import numpy as np

EGO = 8
TASK = 12
NEIGHBOR = 5
STATUS = 3

def observe_agents(env, previous, channel=None):
    cfg = env.config
    n, pmax = cfg.max_agents, cfg.max_pois
    ego = np.zeros((n, EGO), np.float32)
    tasks = np.zeros((n, pmax, TASK), np.float32)
    neighbors = np.zeros((n, n, NEIGHBOR), np.float32)
    seen = np.zeros((n, n), bool)
    legal = np.zeros((n, pmax + 1), bool)
    legal[:, -1] = True
    scale = np.asarray(cfg.dimensions)
    for i in np.flatnonzero(env.active):
        pos = env.positions[i]
        radius = env.sensor_radii[i]
        ego[i] = [*(pos / scale), env.speeds[i] / 2, radius / max(scale),
                  env.skills[i] / 4, env.step_index / cfg.horizon,
                  (previous[i] + 1) / (pmax+1), 1.]
        for j in np.flatnonzero(env.active):
            if i == j: continue
            d = env.positions[j] - pos
            if np.linalg.norm(d) <= radius:
                seen[i,j] = True
                # Only relative local position and observable presence, not remote skills/activity.
                neighbors[i,j] = [*(d / scale), np.linalg.norm(d)/max(scale), 1.,
                                  0.]  # no implicit teammate-intention message]
        for k, poi in enumerate(cfg.pois):
            delta = np.asarray(poi.position) - pos
            dist = float(np.linalg.norm(delta))
            # Completion of one-shot objective observable only inside own sensor range.
            completed_seen = bool((dist <= radius and env.poi_completed[k]) or
                                  (channel is not None and channel.completed[i, k]))
            # Outside sensor range, dynamically completed status remains UNKNOWN.
            # Future-window travel is legal: approaching before opening is necessary.
            available_by_clock = env.step_index <= poi.window[1]
            compatible = poi.required_skill is None or poi.required_skill == env.skills[i]
            legal[i,k] = available_by_clock and compatible and not completed_seen
            tasks[i,k] = [*(delta/scale), dist/max(scale), poi.radius/max(scale),
                          poi.coupling/float(n), poi.value/10,
                          poi.window[0]/cfg.horizon, poi.window[1]/cfg.horizon,
                          float(poi.repeat),
                          -1. if poi.required_skill is None else poi.required_skill/4,
                          float(completed_seen), float(previous[i] == k)]
    if channel is None:
        status=np.zeros((n,n,STATUS),np.float32)
        status_mask=np.zeros((n,n),bool)
    else:
        status,status_mask=channel.observations(env.step_index)
    return dict(ego=ego, tasks=tasks, neighbors=neighbors, neighbor_mask=seen,
                status=status, status_mask=status_mask,
                legal=legal, active=env.active.copy())


def critic_state(env):
    """Privileged *training only*. Never accepted by actor API."""
    cfg=env.config
    agent = np.zeros((cfg.max_agents, 7), np.float32)
    poi = np.zeros((cfg.max_pois, 9), np.float32)
    d = np.asarray(cfg.dimensions)
    for i in range(cfg.max_agents):
        if env.active[i]:
            agent[i] = [*(env.positions[i]/d), env.speeds[i]/2,
                        env.sensor_radii[i]/max(d), env.skills[i]/4,
                        (env.roles[i]+1)/(cfg.max_pois+1), 1.]
    for k,p in enumerate(cfg.pois):
        poi[k] = [*(env.poi_positions[k]/d), p.radius/max(d),
                  p.coupling/cfg.max_agents, p.value/10, p.window[0]/cfg.horizon,
                  p.window[1]/cfg.horizon, float(env.poi_completed[k]),
                  -1. if p.required_skill is None else p.required_skill/4]
    return dict(agent=agent, poi=poi, active=env.active.copy(),
                poi_mask=np.arange(cfg.max_pois) < len(cfg.pois),
                clock=np.array([env.step_index/cfg.horizon],np.float32))
