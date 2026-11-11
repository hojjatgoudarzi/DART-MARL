"""Learned allocation inference: score SAME shortlist as analytic R-CART.

The CFT model estimates H-step realized RETURN RESIDUAL for each feasible
assignment; this is a conditional surrogate and contains no future oracle.
"""
from __future__ import annotations

import numpy as np

from dart_marl.cft.candidates import assignment_candidates
from dart_marl.cft.features import visible_features
from dart_marl.roles.allocator import Allocation, AllocationConfig
from dart_marl.roles.recovery import score_recovery_assignment


def learned_score(env, roster, previous, proposed, cfg: AllocationConfig, estimator):
    score = score_recovery_assignment(
        env, roster, previous, proposed,
        horizon_steps=cfg.forecast_horizon, discount=cfg.forecast_discount,
        travel_weight=cfg.travel_weight, switch_weight=cfg.switch_weight,
        sync_weight=cfg.sync_weight)
    if not score.feasible:
        return None
    x = visible_features(env, roster, previous, proposed,
                         horizon=cfg.forecast_horizon, discount=cfg.forecast_discount)
    residual = float(estimator.residuals(x[None])[0])
    if not np.isfinite(residual):
        raise ValueError('non-finite learned CFT recovery estimate')
    # Transfer/switch and synchronization penalties remain from analytic R-CART;
    # learned label models raw physical rollout return, not arbitrary planner costs.
    raw_prediction = float(np.clip(score.projected_return + residual,
                                   0., score.potential_return))
    prediction = float((1. - estimator.blend_alpha) * score.projected_return
                       + estimator.blend_alpha * raw_prediction)
    residual = prediction - score.projected_return
    objective = float(score.objective + residual)
    return score, prediction, residual, objective


def select_learned_assignment(env, roster, previous, cfg: AllocationConfig, estimator):
    choices = assignment_candidates(
        env, roster, previous, horizon=cfg.forecast_horizon,
        discount=cfg.forecast_discount, travel_weight=cfg.travel_weight,
        switch_weight=cfg.switch_weight, sync_weight=cfg.sync_weight,
        k=cfg.cft_candidates, beam=cfg.cft_beam)
    batch_x = np.stack([visible_features(
        env, roster, previous, roles, horizon=cfg.forecast_horizon,
        discount=cfg.forecast_discount) for roles in choices])
    residuals = estimator.residuals(batch_x)
    if not np.isfinite(residuals).all():
        raise ValueError('non-finite CFT predictions')
    scored = []
    for roles, residual in zip(choices, residuals):
        score = score_recovery_assignment(
            env, roster, previous, roles, horizon_steps=cfg.forecast_horizon,
            discount=cfg.forecast_discount, travel_weight=cfg.travel_weight,
            switch_weight=cfg.switch_weight, sync_weight=cfg.sync_weight)
        if not score.feasible:
            raise AssertionError('candidate generator returned infeasible assignment')
        raw_prediction = float(np.clip(score.projected_return + residual,
                                       0., score.potential_return))
        bounded_return = float((1. - estimator.blend_alpha) * score.projected_return
                               + estimator.blend_alpha * raw_prediction)
        bounded_residual = bounded_return - score.projected_return
        scored.append((float(score.objective + bounded_residual), roles, score,
                       float(bounded_residual)))
    scored.sort(key=lambda row:(-row[0], tuple(int(a) for a in row[1])))
    objective, roles, score, residual = scored[0]
    components = score.components()
    components.update({
        'learned_return_residual':residual,
        'predicted_realized_horizon_return':float(score.projected_return + residual),
        'learned_objective':objective,
        'candidate_count':float(len(choices)),
        'validation_blend_alpha':float(estimator.blend_alpha),
        'analytic_shortlist_best':float(max(c[2].objective for c in scored)),
    })
    return Allocation(roles.copy(), cfg.method, objective, score.selected_tasks,
                      score.travel, score.switches, len(choices), components)
