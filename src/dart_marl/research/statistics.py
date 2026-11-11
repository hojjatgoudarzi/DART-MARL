"""Training-seed-level descriptive summaries and deterministic paired bootstrap.

The independent unit is a *trained controller*, not an evaluation episode,
allocator decision, candidate, or POI.  No p-values are produced.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def seed_summary(values: list[float] | np.ndarray, *, bootstrap_seed: int = 271828,
                 replicates: int = 4000) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 1 or not np.isfinite(x).all() or replicates < 100:
        raise ValueError('finite nonempty 1D training-seed data and >=100 bootstrap replicates required')
    summary: dict[str, Any] = {
        'n_independent_training_seeds': int(x.size),
        'mean': float(x.mean()),
        'sample_std': float(x.std(ddof=1)) if x.size > 1 else None,
        'ci95_training_seed_bootstrap_percentile': None,
        'units': 'one mean per independently initialized/trained controller',
    }
    if x.size >= 2:
        rng = np.random.default_rng(bootstrap_seed)
        indices = rng.integers(0, x.size, size=(replicates, x.size))
        means = x[indices].mean(axis=1)
        summary['ci95_training_seed_bootstrap_percentile'] = [
            float(np.quantile(means, .025)), float(np.quantile(means, .975))]
    if x.size < 5:
        summary['caution'] = ('Exploratory small-n estimate. Bootstrap intervals may be unstable; '
                              'do not infer significance or publication-grade generalization.')
    return summary


def report_statistics(rows: list[dict[str, Any]], *, bootstrap_seed: int = 271828,
                      replicates: int = 4000) -> dict[str, Any]:
    """Aggregate paired methods and variants, explicitly nesting eval seeds."""
    if not rows:
        raise ValueError('no completed evaluations')
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r['variant'], r['scenario']), []).append(r)
    output: dict[str, Any] = {'by_variant_scenario': {}, 'between_variants': {}}
    for (variant, scenario), entries in sorted(groups.items()):
        entries = sorted(entries, key=lambda r: r['training_seed'])
        ids = [int(r['training_seed']) for r in entries]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate training seed within variant/scenario')
        methods = list(entries[0]['methods'])
        evaluation_seeds = entries[0]['evaluation_seeds']
        if any(set(r['methods']) != set(methods) or r['evaluation_seeds'] != evaluation_seeds
               for r in entries):
            raise ValueError('unequal method sets or unmatched evaluation seeds')
        block = {'training_seeds': ids, 'eval_seeds': evaluation_seeds,
                 'methods': {}, 'paired_contrasts': {}}
        for method in methods:
            metrics = [('return', 'mean_return'), ('post_disruption_return', 'mean_after_first_disruption_return'),
                       ('role_switches', 'mean_role_switches')]
            block['methods'][method] = {}
            for metric, field in metrics:
                vals = [r['methods'][method].get(field) for r in entries]
                if all(v is not None for v in vals):
                    block['methods'][method][metric] = seed_summary(
                        vals, bootstrap_seed=bootstrap_seed, replicates=replicates)
                else:
                    block['methods'][method][metric] = None
            for metric in ('mean_post_event_repeat_service_fraction',
                           'mean_first_restoration_lag_observed_only'):
                values = [r['methods'][method]['recovery_metrics'].get(metric) for r in entries]
                block['methods'][method][metric] = (
                    seed_summary(values, bootstrap_seed=bootstrap_seed, replicates=replicates)
                    if all(v is not None for v in values) else None)
            # Censored recovery rate is a separate statistic; observed-only lag is
            # explicitly conditional on restoration and should not be treated as unconditional.
            block['methods'][method]['restoration_rate_among_at_risk'] = (
                seed_summary([r['methods'][method]['recovery_metrics']['restoration_rate_among_at_risk']
                              for r in entries], bootstrap_seed=bootstrap_seed, replicates=replicates)
                if all(r['methods'][method]['recovery_metrics'].get('restoration_rate_among_at_risk')
                       is not None for r in entries) else None)
        contrasts = [
            ('r_cart', 'switch_aware'), ('r_cart', 'greedy'),
            ('r_cart', 'r_cart_no_sync'), ('r_cart', 'r_cart_no_switch'),
            ('r_cart', 'r_cart_no_hysteresis'), ('r_cart_bounded_sync', 'switch_aware'),
            ('r_cart_bounded_sync', 'r_cart'), ('r_cart_bounded_sync', 'r_cart_no_sync'),
            ('cft', 'r_cart'), ('cft_rank', 'r_cart'),
            ('r_cart_service', 'switch_aware'), ('r_cart_service', 'r_cart'),
            ('r_cart_service', 'r_cart_no_sync'), ('r_cart_service', 'r_cart_bounded_sync'),
        ]
        for proposed, baseline in contrasts:
            if proposed in methods and baseline in methods:
                delta = [r['methods'][proposed]['mean_return'] -
                         r['methods'][baseline]['mean_return'] for r in entries]
                block['paired_contrasts'][f'{proposed}_minus_{baseline}'] = {
                    'per_training_seed': dict(zip(map(str, ids), map(float, delta))),
                    **seed_summary(delta, bootstrap_seed=bootstrap_seed, replicates=replicates),
                }
        # Secondary contrasts use the SAME independent controller-level unit.
        # Recovery probabilities have method-specific at-risk denominators;
        # these contrasts are descriptive, not patient/task-matched causal effects.
        block['secondary_paired_contrasts'] = {}
        secondary_fields = {
            'post_disruption_return': ('mean_after_first_disruption_return', None),
            'role_switches': ('mean_role_switches', None),
            'poi_service_steps': ('mean_poi_service_steps', None),
            'restoration_rate_among_at_risk': ('recovery_metrics', 'restoration_rate_among_at_risk'),
            'mean_post_event_repeat_service_fraction': ('recovery_metrics', 'mean_post_event_repeat_service_fraction'),
            'observed_first_restoration_lag': ('recovery_metrics', 'mean_first_restoration_lag_observed_only'),
        }
        for proposed, baseline in contrasts:
            if proposed not in methods or baseline not in methods:
                continue
            pair = f'{proposed}_minus_{baseline}'
            metrics_report = {}
            for metric, (outer, inner) in secondary_fields.items():
                pairs = []
                for r in entries:
                    a, b = r['methods'][proposed], r['methods'][baseline]
                    a = a.get(outer) if inner is None else a.get(outer, {}).get(inner)
                    b = b.get(outer) if inner is None else b.get(outer, {}).get(inner)
                    if a is None or b is None:
                        break
                    # POI service vector is reported per task; use a per-controller
                    # mean over the same task set rather than treating POIs as seeds.
                    if metric == 'poi_service_steps':
                        av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
                        if av.shape != bv.shape or av.size == 0:
                            raise ValueError('unmatched POI service dimensions')
                        a, b = float(av.mean()), float(bv.mean())
                    pairs.append(float(a) - float(b))
                if len(pairs) == len(entries):
                    metrics_report[metric] = {
                        'per_training_seed': dict(zip(map(str, ids), pairs)),
                        **seed_summary(pairs, bootstrap_seed=bootstrap_seed, replicates=replicates),
                    }
                else:
                    metrics_report[metric] = None
            block['secondary_paired_contrasts'][pair] = metrics_report
        output['by_variant_scenario'].setdefault(variant, {})[scenario] = block
    # Same seed IDs across training variants are paired by design, not by ordering.
    variants = sorted({v for v, _ in groups})
    if len(variants) == 2:
        a, b = variants
        for scenario in sorted({s for _, s in groups}):
            if (a, scenario) not in groups or (b, scenario) not in groups:
                continue
            by_a = {r['training_seed']: r for r in groups[a, scenario]}
            by_b = {r['training_seed']: r for r in groups[b, scenario]}
            if set(by_a) != set(by_b):
                raise ValueError('training seeds not matched between algorithm variants')
            if any(by_a[k]['evaluation_seeds'] != by_b[k]['evaluation_seeds'] for k in by_a):
                raise ValueError('evaluation seeds not paired between variants')
            common_methods = sorted(set(next(iter(by_a.values()))['methods']) &
                                    set(next(iter(by_b.values()))['methods']))
            entry = {}
            for method in common_methods:
                delta = [by_b[k]['methods'][method]['mean_return'] -
                         by_a[k]['methods'][method]['mean_return'] for k in sorted(by_a)]
                entry[f'{b}_minus_{a}_{method}'] = seed_summary(
                    delta, bootstrap_seed=bootstrap_seed, replicates=replicates)
            output['between_variants'][scenario] = entry
    return output
