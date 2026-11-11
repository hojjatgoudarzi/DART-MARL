"""Offline observed-service recovery metrics (never exposed to the allocator).

A task counts as *at risk* only if it was being served in the decision before
a configured disruption and remains a repeatable, eligible task. The end of
an event's recovery window is censored by the next configured disruption, the
episode end, and the declared measurement horizon. Missing restoration is
recorded as censored, not assigned a fabricated recovery time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dart_marl.envs.config import EnvironmentConfig


def event_recovery_metrics(
    cfg: EnvironmentConfig, poi_served_by_step: list[list[int]],
    rewards_by_step: list[float], *, horizon_steps: int = 12,
) -> list[dict[str, Any]]:
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    length = len(rewards_by_step)
    if len(poi_served_by_step) != length:
        raise ValueError("misaligned POI and reward series")
    if any(len(row) < len(cfg.pois) for row in poi_served_by_step):
        raise ValueError("missing task service measurements")
    steps = sorted({ev.step for ev in cfg.events if 0 <= ev.step < length})
    output = []
    for k, onset in enumerate(steps):
        next_event = steps[k + 1] if k + 1 < len(steps) else length
        cutoff = min(next_event, onset + horizon_steps, length)
        if cutoff <= onset:
            continue
        affected = [p for p, poi in enumerate(cfg.pois)
                    if poi.repeat and onset > 0 and poi.window[0] <= onset <= poi.window[1]
                    and bool(poi_served_by_step[onset - 1][p])]
        measurements = []
        for p in affected:
            served = [bool(poi_served_by_step[t][p]) for t in range(onset, cutoff)]
            first = next((i for i, active in enumerate(served) if active), None)
            consecutive = next((i for i in range(len(served) - 1)
                                if served[i] and served[i + 1]), None)
            measurements.append({
                "task": p,
                "first_restoration_lag": first,  # 0 means no loss at onset
                "first_restoration_censored": first is None,
                "sustained_two_step_lag": consecutive,
                "sustained_two_step_censored": consecutive is None,
                "observed_outage_steps": (len(served) if first is None else first),
                "served_steps_in_window": int(sum(served)),
                "eligible_steps_in_window": int(sum(
                    cfg.pois[p].window[0] <= t <= cfg.pois[p].window[1]
                    for t in range(onset, cutoff))),
            })
        # Aggregate service fraction only over repeatable, eligible task-step
        # pairs; this is observable coverage, NOT a recovery-time surrogate.
        eligible = [(t, p) for t in range(onset, cutoff)
                    for p, task in enumerate(cfg.pois)
                    if task.repeat and task.window[0] <= t <= task.window[1]]
        fraction = (sum(bool(poi_served_by_step[t][p]) for t, p in eligible)
                    / len(eligible)) if eligible else None
        output.append({
            "event_step": onset,
            "event_kinds": [e.kind for e in cfg.events if e.step == onset],
            "measurement_end_exclusive": cutoff,
            "window_length": cutoff - onset,
            "pre_event_served_repeat_tasks": affected,
            "task_recovery": measurements,
            "post_event_reward_window": float(sum(rewards_by_step[onset:cutoff])),
            "post_event_repeat_service_fraction": fraction,
        })
    return output


def aggregate_recovery(episode_results: list[dict[str, Any]]) -> dict[str, Any]:
    events = [event for ep in episode_results for event in ep["recovery_events"]]
    affected = [row for event in events for row in event["task_recovery"]]
    observed = [r["first_restoration_lag"] for r in affected
                if r["first_restoration_lag"] is not None]
    sustained = [r["sustained_two_step_lag"] for r in affected
                 if r["sustained_two_step_lag"] is not None]
    fractions = [e["post_event_repeat_service_fraction"] for e in events
                 if e["post_event_repeat_service_fraction"] is not None]
    return {
        "num_events": len(events),
        "num_pre_event_served_task_instances": len(affected),
        "num_restored": len(observed),
        "num_first_restoration_censored": len(affected) - len(observed),
        "restoration_rate_among_at_risk": (len(observed) / len(affected)
                                           if affected else None),
        "mean_first_restoration_lag_observed_only": (
            float(np.mean(observed)) if observed else None),
        "num_sustained_two_step_restored": len(sustained),
        "mean_sustained_two_step_lag_observed_only": (
            float(np.mean(sustained)) if sustained else None),
        "mean_post_event_repeat_service_fraction": (
            float(np.mean(fractions)) if fractions else None),
        "mean_post_event_window_reward": (
            float(np.mean([e["post_event_reward_window"] for e in events]))
            if events else None),
    }
