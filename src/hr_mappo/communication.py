"""Optional local, delayed, lossy rover-to-rover heartbeats.

A transmission is generated only from the *sender's own* current status, and
only for a receiver physically inside its communication range at send time.
Reception is delayed; the receiver never reads other rovers' truth directly.
No centralized activity reports or future event tapes are accessed.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CommConfig:
    enabled: bool = False
    delay: int = 3
    dropout: float = .1
    ttl: int = 10
    range: float = 12.

    def __post_init__(self):
        if self.delay < 0 or self.ttl < 1 or self.range <= 0 or not 0 <= self.dropout <= 1:
            raise ValueError('invalid communication controls')


class LocalChannel:
    """Each receiver has private memory; messages cannot reveal later faults."""
    def __init__(self, n, p, config: CommConfig, seed: int):
        self.n = n
        self.p = p
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.pending = []
        self.last = np.full((n, n), -10**8, dtype=np.int64)
        self.task = np.full((n, n), -1, dtype=np.int32)
        self.completed = np.zeros((n, p), dtype=bool)
        self.clock = -1

    def advance(self, env, selected):
        """Once per physical decision boundary; send AFTER events occurred."""
        t = int(env.step_index)
        if t <= self.clock:
            raise ValueError('nonmonotone communication clock')
        self.clock = t
        if self.config.enabled:
            keep = []
            for arrival, sender, recipient, chosen, sent in self.pending:
                if arrival <= t:
                    self.last[recipient, sender] = sent
                    self.task[recipient, sender] = chosen
                else:
                    keep.append((arrival, sender, recipient, chosen, sent))
            self.pending = keep
        # A completion is observed only while the agent physically senses the POI.
        for i in np.flatnonzero(env.active):
            for j, poi in enumerate(env.config.pois):
                if np.linalg.norm(env.positions[i] - env.poi_positions[j]) <= env.sensor_radii[i]:
                    if not poi.repeat and env.poi_completed[j]:
                        self.completed[i, j] = True
        if not self.config.enabled:
            return
        for sender in np.flatnonzero(env.active):
            for recipient in np.flatnonzero(env.active):
                if sender == recipient:
                    continue
                if np.linalg.norm(env.positions[sender]-env.positions[recipient]) > min(
                    self.config.range, env.sensor_radii[recipient]):
                    continue
                if self.rng.random() < self.config.dropout:
                    continue
                arrival = t+self.config.delay
                packet=(arrival,int(sender),int(recipient),int(selected[sender]),t)
                if arrival == t:
                    self.last[recipient, sender] = t
                    self.task[recipient, sender] = int(selected[sender])
                else:
                    self.pending.append(packet)

    def observations(self, t):
        status = np.zeros((self.n, self.n, 3), dtype=np.float32)
        mask = np.zeros((self.n, self.n), dtype=bool)
        if not self.config.enabled:
            return status, mask
        ages = t-self.last
        mask = (ages >= 0) & (ages <= self.config.ttl)
        status[:, :, 0] = np.clip(ages / self.config.ttl, 0, 1)
        status[:, :, 1] = (self.task+1)/(self.p+1)
        status[:, :, 2] = 1
        status[~mask] = 0
        np.fill_diagonal(mask, False)
        return status, mask
