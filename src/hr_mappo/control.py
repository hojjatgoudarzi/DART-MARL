"""Shared, non-privileged deterministic navigation for all task selectors."""
import numpy as np

def control(env, choices):
    output=np.zeros((env.config.max_agents,2),np.float64)
    # Only ego, public static map and own speed; no centralized optimization.
    for i in np.flatnonzero(env.active):
        p=int(choices[i]);
        if p < 0 or p >= len(env.config.pois): continue
        delta=np.asarray(env.config.pois[p].position)-env.positions[i]
        norm=np.linalg.norm(delta)
        if norm > 1e-10:
            output[i]=delta/norm * min(float(env.speeds[i]), norm, 1.)
    return output
