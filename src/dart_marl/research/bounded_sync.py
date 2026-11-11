"""legacy_study *post-hoc development* comparison using an immutable legacy_study pilot.

Never evaluates TEST_ONLY or modifies pilot runs, checkpoints or selection files.
This analysis was designed after inspecting fault_robust/seed_22 and is NOT
an independent confirmation of the proposed bounded-synchronization model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dart_marl.envs.config import load_config
from dart_marl.research.study import _protocol, _verify_sources, sha
from dart_marl.research.statistics import report_statistics
from dart_marl.roles.evaluation import evaluate_allocators

METHODS = ('switch_aware', 'r_cart', 'r_cart_no_sync', 'r_cart_bounded_sync')
FORMAT = 'dart-marl-bounded_sync-development-v1'


def _new_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite experimental output: {path}')
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def _minimal(report: dict, *, variant: str, seed: int, scenario: str,
             digest: str, protocol_sha: str, scenario_sha: str) -> dict:
    return {
        'format': FORMAT, 'stage': 'dev', 'variant': variant, 'training_seed': seed,
        'checkpoint_sha256': digest, 'protocol_sha256': protocol_sha,
        'scenario': scenario, 'scenario_sha256': scenario_sha,
        'evaluation_seeds': report['evaluation_seeds'],
        'allocator_settings': report['allocator_settings'],
        'methods': {
            k: {
                'mean_return': v['mean_return'], 'returns': v['returns'],
                'mean_after_first_disruption_return': v['mean_after_first_disruption_return'],
                'mean_role_switches': v['mean_role_switches'],
                'mean_role_replans': v['mean_role_replans'],
                'mean_poi_service_steps': v['mean_poi_service_steps'],
                'recovery_metrics': v['recovery_metrics'],
                'first_episode_decisions': v['per_episode'][0]['role_decisions'],
            } for k, v in report['methods'].items()
        },
    }



def _selected_from_pilot(root: Path, variant: str, seed: int) -> tuple[Path, str]:
    """Resolve a registered checkpoint even when the checkpoint run lives in a separate directory.

    legacy_study saved relative checkpoint paths anchored to its original launch CWD.
    We derive an alternate location *only* inside the supplied pilot training root,
    then demand the exact registered hash. No checkpoint substitution is allowed.
    """
    record = json.loads((root / 'selections' / variant / f'seed_{seed}.json').read_text())
    saved = Path(record['selected_checkpoint'])
    candidates = [saved, root / 'training' / variant / f'seed_{seed}' /
                  'checkpoints' / saved.name]
    expected = record['selected_checkpoint_sha256']
    for candidate in candidates:
        if candidate.exists() and sha(candidate) == expected:
            return candidate, expected
    raise FileNotFoundError(
        f'Cannot find selected checkpoint with SHA-256 {expected}: '
        f'{variant} seed {seed}; preserve the original legacy_study training folder')


def run_development_comparison(pilot_root: str | Path, out: str | Path, *,
                               variants: list[str] | None = None,
                               seeds: list[int] | None = None,
                               device: str = 'cpu') -> dict[str, Any]:
    root, spec, protocol = _protocol(pilot_root)
    out = Path(out)
    if out.resolve() == root.resolve() or out.resolve().is_relative_to(root.resolve()):
        raise ValueError('legacy_study outputs must be outside the immutable legacy_study pilot root')
    if 'DEVELOPMENT_ONLY' not in spec['name']:
        raise ValueError('legacy_study comparison requires the registered DEVELOPMENT_ONLY pilot')
    if (root / 'sealed_test_started.json').exists() or (root / 'frozen_development.json').exists():
        raise PermissionError('a frozen/test-opened experiment cannot be modified or rerun for tuning')
    _verify_sources(protocol, root)
    chosen_variants = list(spec['training_variants']) if variants is None else list(variants)
    chosen_seeds = list(spec['training_seeds']) if seeds is None else list(seeds)
    if (not chosen_variants or not chosen_seeds or
            len(set(chosen_variants)) != len(chosen_variants) or
            len(set(chosen_seeds)) != len(chosen_seeds) or
            any(v not in spec['training_variants'] for v in chosen_variants) or
            any(s not in spec['training_seeds'] for s in chosen_seeds)):
        raise ValueError('only unique preregistered training variants/seeds allowed')
    scenarios = [Path(x) for x in spec['validation_scenarios']]
    if any('TEST_ONLY' in load_config(path).name for path in scenarios):
        raise ValueError('test-only scenario must not appear in development evaluation')
    expected = {
        'format': FORMAT, 'protocol_sha256': protocol['protocol_sha256'],
        'variants': chosen_variants, 'training_seeds': chosen_seeds,
        'methods': list(METHODS), 'development_scenario_sha256':
        {str(p): sha(p) for p in scenarios}, 'evaluation_settings': spec['allocation'],
        'warning': ('POST-HOC EXPLORATORY, designed after observing fault_robust seed 22. '
                    'Do not use this as independent generalization or held-out test evidence.'),
    }
    metadata = out / 'bounded_sync_manifest.json'
    if metadata.exists():
        if json.loads(metadata.read_text()) != expected:
            raise ValueError('cannot resume legacy_study with changed protocol/seeds/scenarios')
    else:
        _new_json(metadata, expected)
    results = []
    for variant in chosen_variants:
        for seed in chosen_seeds:
            checkpoint, checkpoint_hash = _selected_from_pilot(root, variant, seed)
            for scenario in scenarios:
                scenario_name = load_config(scenario).name
                original_file = root / 'evaluations' / 'dev' / variant / f'seed_{seed}' / f'{scenario_name}.json'
                original = json.loads(original_file.read_text())
                if (original.get('stage') != 'dev' or
                        original.get('protocol_sha256') != protocol['protocol_sha256'] or
                        original.get('checkpoint_sha256') != checkpoint_hash or
                        original.get('scenario_sha256') != sha(scenario)):
                    raise ValueError(f'original legacy_study evaluation provenance mismatch: {original_file}')
                if original['evaluation_seeds'] != [int(spec['allocation']['development_seed']) + 1009*i
                                                    for i in range(int(spec['allocation']['episodes']))]:
                    raise ValueError('original legacy_study evaluation seeds do not match registered pilot')
                dest = out / 'evaluations' / variant / f'seed_{seed}' / f'{scenario_name}.json'
                if dest.exists():
                    row = json.loads(dest.read_text())
                    if (row['checkpoint_sha256'] != checkpoint_hash or
                            row['scenario_sha256'] != sha(scenario) or
                            row['protocol_sha256'] != protocol['protocol_sha256'] or
                            set(row['methods']) != set(METHODS)):
                        raise ValueError(f'changed or incomplete legacy_study evaluation: {dest}')
                else:
                    report = evaluate_allocators(
                        scenario, controller='checkpoint', checkpoint=checkpoint,
                        training_env_config=spec['training_environment'], methods=METHODS,
                        episodes=int(spec['allocation']['episodes']),
                        seed=int(spec['allocation']['development_seed']), device=device,
                        capture_trace=False,
                        forecast_horizon=int(spec['allocation']['forecast_horizon']))
                    if report['controller']['checkpoint_sha256'] != checkpoint_hash:
                        raise ValueError('controller hash changed during evaluation')
                    row = _minimal(report, variant=variant, seed=seed,
                                   scenario=scenario_name, digest=checkpoint_hash,
                                   protocol_sha=protocol['protocol_sha256'], scenario_sha=sha(scenario))
                    # Baselines must precisely match the *already observed* development results.
                    for method in ('switch_aware', 'r_cart', 'r_cart_no_sync'):
                        if not np.allclose(row['methods'][method]['returns'],
                                           original['methods'][method]['returns'],
                                           atol=1e-7, rtol=0):
                            raise RuntimeError(f'{method} changed versus legacy_study pilot: '
                                               f'{variant} seed={seed}, {scenario_name}')
                    _new_json(dest, row)
                results.append(row)
    report = {
        **expected,
        'primary_method': 'r_cart_bounded_sync',
        'training_seed_statistical_unit': 'independent MAPPO training seed, not evaluation episode',
        'completed_evaluations': len(results),
        'statistics': report_statistics(results, bootstrap_seed=int(spec['statistics']['bootstrap_seed']),
                                        replicates=int(spec['statistics']['bootstrap_replicates'])),
        'interpretation': 'Development sensitivity analysis only; not a final research conclusion',
    }
    summary = out / 'bounded_sync_dev_summary.json'
    if summary.exists():
        if json.loads(summary.read_text()) != report:
            raise ValueError('existing legacy_study summary differs: keep historical data immutable')
    else:
        _new_json(summary, report)
    return {'summary': str(summary), 'evaluations': len(results),
            'training_seeds': chosen_seeds, 'variants': chosen_variants,
            'status': 'post_hoc_development_only'}
