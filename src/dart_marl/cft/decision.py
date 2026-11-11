"""Snapshot-grouped decision diagnostics and validation-only score mixing.

Labels are fixed-role counterfactual returns; penalties are planner-time estimates
and not physical rewards. NEVER use test groups to select a model or mixture.
"""
from __future__ import annotations

import numpy as np

BLEND_GRID = (0., .25, .5, .75, 1.)


def snapshot_index_groups(snapshots: np.ndarray, subset: np.ndarray) -> list[np.ndarray]:
    """Stable sorted groups, each consisting of candidates from one snapshot."""
    subset = np.asarray(subset, dtype=np.int64)
    return [subset[snapshots[subset] == s] for s in np.unique(snapshots[subset])]


def decision_metrics(labels: np.ndarray, analytic: np.ndarray,
                     prediction: np.ndarray, penalties: np.ndarray,
                     snapshots: np.ndarray, subset: np.ndarray,
                     alpha: float = 1.) -> dict:
    """Observed regret of choosing with analytic/learned blended objective.

    Exact tie resolution uses candidate order, which contains the analytic
    optimum first. This treats each snapshot as one paired decision, not rows
    as independent observations.
    """
    if not 0 <= alpha <= 1:
        raise ValueError('blend must be between 0 and 1')
    regrets, analytic_regrets, learned_wins, oracle_values = [], [], 0, []
    groups = snapshot_index_groups(snapshots, subset)
    for group in groups:
        if len(group) < 2:
            continue
        actual = labels[group] + penalties[group]
        analytic_score = analytic[group] + penalties[group]
        mixed = ((1. - alpha) * analytic[group] + alpha * prediction[group]
                 + penalties[group])
        oracle = float(actual.max())
        r = oracle - float(actual[np.argmax(mixed)])
        ra = oracle - float(actual[np.argmax(analytic_score)])
        regrets.append(max(0., r)); analytic_regrets.append(max(0., ra))
        learned_wins += r < ra - 1e-6
        oracle_values.append(oracle)
    return {
        'paired_snapshot_groups': len(regrets),
        'candidate_selection_mean_regret': float(np.mean(regrets)) if regrets else None,
        'analytic_candidate_selection_mean_regret': (
            float(np.mean(analytic_regrets)) if analytic_regrets else None),
        'fraction_better_than_analytic': learned_wins / len(regrets) if regrets else None,
        'oracle_mean_decision_value': float(np.mean(oracle_values)) if regrets else None,
    }


def select_validation_blend(labels, analytic, prediction, penalties,
                            snapshots, subset, grid=BLEND_GRID) -> tuple[float, float]:
    """Pick alpha by VAL ONLY; ties favour smaller alpha (analytic fallback)."""
    best = (float('inf'), 0.)
    for alpha in grid:
        metrics = decision_metrics(labels, analytic, prediction, penalties,
                                   snapshots, subset, alpha=alpha)
        regret = metrics['candidate_selection_mean_regret']
        if regret is None:
            raise ValueError('validation needs paired candidate snapshots')
        candidate = (regret, float(alpha))
        if candidate < best:
            best = candidate
    return best[1], best[0]
