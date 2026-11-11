#!/usr/bin/env python3
"""Validate public twelve-rover release provenance without retraining or pickle loads.

The report and trace record the original *full* training checkpoint SHA; the
public checkpoint is an actor-only export with a different SHA. The actor was
checked for tensor equality against the source during the export step.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def _sha(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def verify_release() -> dict:
    prov = _json('results/team12/provenance.json')
    public = _json('examples/checkpoints/manifest.json')
    charts = _json('results/team12/paper_assets_manifest.json')
    report = _json('results/team12/data/allocator_report.json')
    initial = _json('results/team12/data/initial_evaluation.json')
    train_config = _json('results/team12/data/train_config.json')
    original_demo = _json('media/rover_recovery_12_trace.json')
    fixed_demo = _json('media/rover_reallocation_vs_fixed_12_trace.json')

    assert _sha('examples/checkpoints/team12_seed_314159.pt') == prov['released_actor_only_checkpoint_sha256']
    assert _sha('configs/env/rover_team_12.yaml') == prov['training_config_sha256']
    assert _sha('results/team12/data/train_history.jsonl') == prov['training_history_sha256']
    assert _sha('results/team12/data/allocator_report.json') == prov['paired_evaluation_sha256']
    assert _sha('media/rover_recovery_12_trace.json') == prov['demo_trace_sha256']
    assert train_config['seed'] == 314159

    exported = next(x for x in public['models'] if x['seed'] == 314159)
    assert exported['original_checkpoint_sha256'] == prov['original_training_checkpoint_sha256']
    assert exported['exported_actor_sha256'] == prov['released_actor_only_checkpoint_sha256']
    assert report['controller']['checkpoint_sha256'] == prov['original_training_checkpoint_sha256']

    sources = {
        'outputs/team12_training/train_history.jsonl': 'results/team12/data/train_history.jsonl',
        'outputs/team12_training/train_config.json': 'results/team12/data/train_config.json',
        'outputs/team12_training/initial_evaluation.json': 'results/team12/data/initial_evaluation.json',
        'outputs/team12_publication/evaluation/allocator_report.json': 'results/team12/data/allocator_report.json',
        'outputs/team12_publication/rover_recovery_12_trace.json': 'media/rover_recovery_12_trace.json',
        'results/heldout_episode_returns.csv': 'results/heldout_episode_returns.csv',
    }
    for source, actual in sources.items():
        assert charts['sources'][source] == _sha(actual), (source, actual)
    for filename, digest in charts['artifact_sha256'].items():
        folder = 'figures' if filename.lower().endswith(('.png','.pdf','.svg')) else 'tables'
        assert digest == _sha(f'results/team12/{folder}/{filename}'), filename

    rows = list(csv.DictReader((ROOT / 'results/team12/data/episode_results.csv').open()))
    assert len(rows) == 72
    assert set(report['methods']) == set(x['method'] for x in rows)
    assert len(report['methods']) == 9 and len(report['evaluation_seeds']) == 8
    for method, stats in report['methods'].items():
        episodes = [x for x in rows if x['method'] == method]
        assert len(episodes) == 8
        assert sorted(int(x['episode_seed']) for x in episodes) == sorted(report['evaluation_seeds'])
        mean = sum(float(x['team_return']) for x in episodes) / 8
        assert abs(mean - stats['mean_return']) < 1e-5

    assert original_demo['left']['controller_sha256'] == prov['original_training_checkpoint_sha256']
    assert fixed_demo['left']['controller_sha256'] == prov['released_actor_only_checkpoint_sha256']
    assert original_demo['left']['method'] == fixed_demo['left']['method'] == 'switch_aware'
    assert original_demo['right']['method'] == 'r_cart'
    assert fixed_demo['right']['method'] == 'fixed'
    assert original_demo['left']['seed'] == fixed_demo['left']['seed'] == 901110
    assert abs(original_demo['left']['final_return'] - 786.7) < 1e-5
    assert abs(original_demo['right']['final_return'] - 786.7) < 1e-5
    assert abs(fixed_demo['right']['final_return'] - 639.5) < 1e-5
    # CPU/PyTorch floating-point rounding may differ slightly between runs.
    # Semantic states (roles, events, activity masks, service and return) must agree.
    for recorded, reproduced in zip(original_demo['left']['states'], fixed_demo['left']['states']):
        for key in ('step','active','roles','reported_active','served','completed','events','replanned'):
            assert recorded[key] == reproduced[key], (key, recorded['step'])
        assert abs(recorded['return'] - reproduced['return']) < 1e-4
        assert abs(recorded['reward'] - reproduced['reward']) < 1e-4
        assert max(abs(a-b) for row_a,row_b in zip(recorded['positions'],reproduced['positions'])
                   for a,b in zip(row_a,row_b)) < 1e-4
    assert len(fixed_demo['left']['states']) == len(original_demo['left']['states']) == 121
    assert len([e for s in fixed_demo['left']['states'] for e in s['events']]) == 11
    assert abs(initial['mean_return'] - 890.075) < 1e-6

    return {
        'status': 'verified',
        'model_training_seed': 314159,
        'model_training_steps': 131072,
        'trained_controllers': 1,
        'evaluation_episodes_per_method': 8,
        'evaluated_methods': 9,
        'verified_plots': len(charts['figures']),
        'verified_source_files': len(sources),
        'switch_aware_mean_return': report['methods']['switch_aware']['mean_return'],
        'original_r_cart_mean_return': report['methods']['r_cart']['mean_return'],
        'demo_fixed_return': fixed_demo['right']['final_return'],
        'demo_switch_return': fixed_demo['left']['final_return'],
        'note': 'Original source checkpoint is not bundled. Public actor-only export SHA differs by design.'
    }


if __name__ == '__main__':
    print(json.dumps(verify_release(), indent=2))
