"""Joint on-policy rollout storage and episode/time-limit-correct per-rover GAE."""

from __future__ import annotations

import numpy as np


def compute_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
                active: np.ndarray, next_active: np.ndarray, done: np.ndarray,
                terminated: np.ndarray, gamma: float, gae_lambda: float,
                ) -> tuple[np.ndarray, np.ndarray]:
    """GAE for [T,E,N]. True termination zeros bootstrap; truncation bootstraps.

    `done` includes time-limit truncations, stopping advantage recursion across
    episode resets. `next_active` stops credit crossing a rover's death. A
    reactivated rover begins a fresh valid segment; no cross-death gradient.
    """
    if rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError("rewards/values/next_values must have equal [T,E,N] shape")
    t, e, n = rewards.shape
    if active.shape != (t, e, n) or next_active.shape != (t, e, n):
        raise ValueError("active masks must match [T,E,N]")
    if done.shape != (t, e) or terminated.shape != (t, e):
        raise ValueError("done and terminated must be [T,E]")
    adv = np.zeros_like(values, dtype=np.float32)
    tail = np.zeros((e, n), dtype=np.float32)
    for step in range(t - 1, -1, -1):
        can_bootstrap = next_active[step] & ~terminated[step, :, None]
        delta = rewards[step] + gamma * next_values[step] * can_bootstrap - values[step]
        can_recurse = next_active[step] & ~done[step, :, None]
        tail = (delta + gamma * gae_lambda * tail * can_recurse) * active[step]
        adv[step] = tail
    return adv, adv + values


class RolloutBuffer:
    """Stores whole synchronized team transitions, not independently shuffled rovers."""

    def __init__(self):
        self._data: dict[str, list[np.ndarray]] = {}

    def append(self, **fields: np.ndarray) -> None:
        if self._data and set(fields) != set(self._data):
            raise ValueError("rollout transition fields changed")
        for key, value in fields.items():
            self._data.setdefault(key, []).append(np.array(value, copy=True))

    def stack(self) -> dict[str, np.ndarray]:
        if not self._data:
            raise ValueError("cannot use empty rollout buffer")
        return {name: np.stack(values, axis=0) for name, values in self._data.items()}
