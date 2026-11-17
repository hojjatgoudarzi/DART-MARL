"""Local-information-only event-triggered macro scheduler.

Each rover chooses at t=0 (or first activation) and at most K ticks later.
Earlier decisions require an *observable* change in own capability, task
validity, visible neighbor membership, or known peer intent/staleness.
Movement and heartbeat AGE alone never trigger a decision.  Source/target
status of remote unsensed rovers and env.events are never inputs.
"""
from dataclasses import dataclass
import numpy as np


def local_signature(obs, i):
    # A compact immutable representation that contains ONLY the agent view.
    ego = obs['ego'][i]
    # own speed, own sensor range, own skill; NOT position/clock (changes every tick)
    own = tuple(np.round(ego[[2, 3, 4]], 6).tolist())
    neighbors = tuple(np.flatnonzero(obs['neighbor_mask'][i]).tolist())
    valid_tasks = tuple(np.flatnonzero(obs['legal'][i, :-1]).tolist())
    known_completed = tuple(np.flatnonzero(obs['tasks'][i, :, -2] > 0.5).tolist())
    peer_mask = obs['status_mask'][i]
    peer_intents = tuple((int(j), round(float(obs['status'][i, j, 1]), 5))
                         for j in np.flatnonzero(peer_mask))
    return own, neighbors, valid_tasks, known_completed, peer_intents


@dataclass
class DecisionScheduler:
    n: int
    k: int
    mode: str = 'synchronous'
    min_interval: int = 2

    def __post_init__(self):
        self.last_time = np.full(self.n, -10**6, dtype=np.int32)
        self.signature = [None] * self.n
        self.initialized = np.zeros(self.n, dtype=bool)
        self.previous_active = np.zeros(self.n, dtype=bool)
        self.pending_reasons = [''] * self.n

    def ready(self, obs, t, choices):
        active = obs['active']
        result = np.zeros(self.n, dtype=bool)
        reasons = [''] * self.n
        signatures = [None] * self.n
        for i in np.flatnonzero(active):
            signatures[i] = local_signature(obs, i)
            if self.mode == 'synchronous':
                if t % self.k == 0:
                    result[i] = True; reasons[i] = 'scheduled'
                continue
            if not self.initialized[i] or not self.previous_active[i]:
                result[i] = True
                reasons[i] = 'activate_or_reenter'
                continue
            old_task = int(choices[i])
            invalid = old_task >= 0 and not obs['legal'][i, old_task]
            if invalid:
                result[i] = True; reasons[i] = 'known_task_invalid'
            elif t - self.last_time[i] >= self.k:
                result[i] = True; reasons[i] = 'scheduled'
            elif t - self.last_time[i] >= self.min_interval and signatures[i] != self.signature[i]:
                result[i] = True; reasons[i] = 'observable_change'
        self.pending_reasons = reasons
        return result, signatures

    def commit(self, t, selected, signatures, active):
        selected = np.asarray(selected, dtype=bool)
        for i in np.flatnonzero(selected):
            self.last_time[i] = t
            self.signature[i] = signatures[i]
            self.initialized[i] = True
        self.previous_active[:] = active

    def step_activity(self, active):
        # Memory of temporary inactivity survives until reentry, which triggers
        # a new decision. Keeping track requires no knowledge of *remote* agent
        # status in the Actor; this scheduler uses own-activation status per slot.
        self.previous_active[:] = active
