"""legacy_study preregistered, independent-seed evaluation workflow.

The only experimental replication unit is a fresh MAPPO training seed.  This
runner records the entire evaluation protocol before touching the sealed test.
It never tunes algorithms on the test scenario and never overwrites artifacts.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from dart_marl.algorithms.mappo.trainer import load_train_config
from dart_marl.envs.config import load_config
from dart_marl.roles.evaluation import METHODS, _same_task_problem, evaluate_allocators
from dart_marl.research.statistics import report_statistics


SCHEMA = 'dart-marl-study-preregistered-v1'


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Immutable output already exists: {path}')
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _assert_spec(raw: dict) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get('schema_version') != 1:
        raise ValueError('legacy_study requires schema_version: 1')
    variants = raw.get('training_variants', {})
    seeds = raw.get('training_seeds', [])
    vals = raw.get('validation_scenarios', [])
    tests = raw.get('test_scenarios', [])
    allocation = raw.get('allocation', {})
    stats = raw.get('statistics', {})
    select = raw.get('checkpoint_selection', {})
    if not variants or not seeds or not vals or not tests or len(set(seeds)) != len(seeds):
        raise ValueError('nonempty variants/seeds/validation/test and unique training seeds required')
    if any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError('training seeds must be nonnegative integers')
    if len(set(variants)) != len(variants) or len(set(vals + tests)) != len(vals + tests):
        raise ValueError('duplicate scenario paths')
    if not allocation.get('methods') or len(set(allocation['methods'])) != len(allocation['methods']):
        raise ValueError('nonempty unique methods required')
    if any(x not in METHODS for x in allocation['methods']):
        raise ValueError('unknown allocation method')
    primary = raw.get('primary_method', 'r_cart')
    if ('switch_aware' not in allocation['methods'] or 'r_cart' not in allocation['methods']
            or primary not in allocation['methods'] or primary not in
            ('r_cart', 'r_cart_bounded_sync', 'r_cart_service')):
        raise ValueError('primary method must be a registered R-CART variant; baseline methods required')
    if any(m in allocation['methods'] for m in ('cft', 'cft_rank')):
        cft = raw.get('cft', {})
        if not cft or min(int(cft.get(x, 0)) for x in
                          ('train_episodes','val_episodes','test_episodes','sample_period',
                           'candidates','beam','epochs','patience')) < 1:
            raise ValueError('CFT methods require a complete preregistered per-controller training protocol')
        if cft.get('event_mode') not in ('diverse', 'sampled'):
            raise ValueError('CFT events must be randomized only from the development distribution')
        if int(cft['candidates']) > int(cft['beam']):
            raise ValueError('CFT candidate limit must not exceed beam limit')
        if cft.get('device') not in ('cpu','cuda','auto'):
            raise ValueError('CFT device missing or invalid')
    if int(select.get('episodes', 0)) < 2 or select.get('criterion') != 'relative_critic_rmse':
        raise ValueError('checkpoint selection must use development relative_critic_rmse')
    if int(allocation.get('episodes', 0)) < 1 or int(allocation.get('forecast_horizon', 0)) < 1:
        raise ValueError('positive evaluation episodes and horizon required')
    if int(stats.get('bootstrap_replicates', 0)) < 100:
        raise ValueError('bootstrap_replicates must be >= 100')
    confirm = raw.get('confirmatory')
    if confirm is not None:
        if not isinstance(confirm, dict):
            raise ValueError('confirmatory preregistration must be a mapping')
        mandatory = ('primary_variant', 'primary_scenario', 'primary_method',
                     'baseline_method', 'primary_metric', 'minimum_training_seeds')
        if any(k not in confirm for k in mandatory):
            raise ValueError('incomplete confirmatory primary hypothesis')
        if (confirm['primary_variant'] not in variants or
                confirm['primary_scenario'] != tests[0] or
                confirm['primary_method'] != primary or
                confirm['baseline_method'] != 'switch_aware' or
                confirm['primary_metric'] != 'mean_return' or
                confirm['primary_method'] not in allocation['methods'] or
                confirm['baseline_method'] not in allocation['methods']):
            raise ValueError('confirmatory hypothesis must match the registered primary variant, '
                             'first sealed test scenario and method contrast')
        min_seeds = confirm['minimum_training_seeds']
        if (not isinstance(min_seeds, int) or min_seeds < 5 or len(seeds) < min_seeds):
            raise ValueError('insufficient fresh training seeds for confirmatory protocol')
        if set(seeds) & {11, 22, 33, 42}:
            raise ValueError('confirmatory study cannot reuse consulted development/controller seeds')
        if raw.get('name', '').startswith('service_aware_'):
            previous_seeds = {101, 202, 303, 404, 505, 606, 707, 808, 909, 1010}
            if set(seeds) & previous_seeds:
                raise ValueError('legacy_study confirmation must not reuse legacy_study trained controllers')
            if (primary != 'r_cart_service' or
                    confirm['primary_variant'] != 'fault_robust' or
                    tests != ['configs/env/service_aware_new_test_only.yaml']):
                raise ValueError('legacy_study confirmatory primary/test protocol changed')
            if 'r_cart_no_sync' not in allocation['methods'] or 'r_cart_bounded_sync' not in allocation['methods']:
                raise ValueError('legacy_study requires explicit synchronization ablations')
        if any('primary_study_final_test_only' in path for path in tests) and raw.get('name', '').startswith('service_aware_'):
            raise ValueError('legacy_study opened test cannot be reused in legacy_study confirmation')
        if any('historical_fault_stress' in path or 'historical_stress_test' in path for path in tests):
            raise ValueError('previously exercised historical TEST_ONLY scenario is not a blind held-out test')
    base = load_config(raw['training_environment'])
    for name, path in [('validation', p) for p in vals] + [('test', p) for p in tests]:
        cfg = load_config(path)
        if not _same_task_problem(base, cfg):
            raise ValueError(f'{name} geometry/dynamics differ from training: {path}')
        if (name == 'test') != ('TEST_ONLY' in cfg.name):
            raise ValueError(f'{name} scenario label must be DEV or TEST_ONLY: {path}')
        if any(p != path and load_config(p).name == cfg.name for p in vals + tests):
            raise ValueError('duplicate scenario names')
    for variant, path in variants.items():
        train = load_train_config(path)
        if Path(train.env_config).resolve() != Path(raw['training_environment']).resolve():
            raise ValueError(f'{variant} training environment not the declared environment')
    return raw


def initialize(spec_path: str | Path, root: str | Path) -> dict:
    spec_path = Path(spec_path)
    spec = _assert_spec(yaml.safe_load(spec_path.read_text()))
    root = Path(root)
    if root.exists():
        raise FileExistsError(f'legacy_study experiment directory must be new: {root}')
    all_sources = [spec_path, Path(spec['training_environment'])]
    all_sources.extend(Path(v) for v in spec['training_variants'].values())
    all_sources.extend(Path(v) for v in spec['validation_scenarios'] + spec['test_scenarios'])
    fingerprints = {str(p): sha(p) for p in all_sources}
    # legacy_study immutability includes implementation bytes, not merely YAML and
    # checkpoint identities. Old registered protocols remain backward compatible.
    source_code = {}
    if spec.get('name', '').startswith('service_aware_'):
        source_files = sorted(Path('src/dart_marl').rglob('*.py')) + [Path('scripts/study.py')]
        source_code = {str(p): sha(p) for p in source_files}
    digest = hashlib.sha256(json.dumps({'spec': spec, 'inputs_sha256': fingerprints,
                                        'algorithm_sha256': source_code},
                                        sort_keys=True).encode()).hexdigest()
    metadata = {'format': SCHEMA, 'spec': spec, 'spec_sha256': sha(spec_path),
                'algorithm_source_sha256': source_code,
                'protocol_sha256': digest, 'source_sha256': fingerprints,
                'primary_comparison': f"{spec.get('primary_method', 'r_cart')} minus switch_aware on test return, paired per training seed",
                'confirmatory': spec.get('confirmatory'),
                'status': 'development',
                'warning': 'Do not inspect TEST_ONLY until development and model selection are frozen.'}
    root.mkdir(parents=True, exist_ok=False)
    (root / 'input_snapshot').mkdir()
    for idx, p in enumerate(all_sources):
        shutil.copyfile(p, root / 'input_snapshot' / f'{idx:02d}_{p.name}')
    for variant, path in spec['training_variants'].items():
        data = yaml.safe_load(Path(path).read_text())
        for seed in spec['training_seeds']:
            data_seed = {**data, 'seed': seed}
            dest = root / 'resolved_configs' / variant / f'seed_{seed}.yaml'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(yaml.safe_dump(data_seed, sort_keys=False))
    metadata['resolved_config_sha256'] = {
        str(p.relative_to(root)): sha(p) for p in
        sorted((root / 'resolved_configs').rglob('*.yaml'))}
    _write_new(root / 'protocol.json', metadata)
    return metadata


def _protocol(root: str | Path) -> tuple[Path, dict, dict]:
    root = Path(root)
    data = json.loads((root / 'protocol.json').read_text())
    if data.get('format') != SCHEMA:
        raise ValueError('invalid legacy_study protocol')
    return root, data['spec'], data


def _verify_sources(data: dict, root: Path | None = None) -> None:
    for path, expected in data.get('algorithm_source_sha256', {}).items():
        if sha(path) != expected:
            raise ValueError(f'algorithm source changed after experiment registration: {path}')
    for path, expected in data['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'source changed since legacy_study registration: {path}')
    if root is not None:
        for path, expected in data['resolved_config_sha256'].items():
            if sha(root / path) != expected:
                raise ValueError(f'resolved training YAML changed since registration: {path}')


def _train_path(root: Path, variant: str, seed: int) -> Path:
    return root / 'training' / variant / f'seed_{seed}'


def _selection_path(root: Path, variant: str, seed: int) -> Path:
    return root / 'selections' / variant / f'seed_{seed}.json'


def _selected(root: Path, variant: str, seed: int) -> tuple[Path, str]:
    record = json.loads(_selection_path(root, variant, seed).read_text())
    ckpt = Path(record['selected_checkpoint'])
    if sha(ckpt) != record['selected_checkpoint_sha256']:
        raise ValueError('selected checkpoint was changed after model selection')
    return ckpt, record['selected_checkpoint_sha256']


def train_cohort(root: str | Path, *, variants: list[str] | None = None,
                 seeds: list[int] | None = None) -> dict:
    root, spec, data = _protocol(root)
    _verify_sources(data, root)
    if (root / 'frozen_development.json').exists():
        raise RuntimeError('development frozen: training is no longer permitted')
    variants = variants or list(spec['training_variants'])
    seeds = seeds or spec['training_seeds']
    if any(v not in spec['training_variants'] for v in variants) or any(
            s not in spec['training_seeds'] for s in seeds):
        raise ValueError('requested variant or seed was not preregistered')
    done = []
    for variant in variants:
        for seed in seeds:
            out = _train_path(root, variant, seed)
            config = root / 'resolved_configs' / variant / f'seed_{seed}.yaml'
            if out.exists():
                if not (out / 'summary.json').exists() or not (out / 'checkpoints/latest.pt').exists():
                    raise RuntimeError(f'incomplete training output at {out}; preserve and repair it manually')
                done.append({'variant': variant, 'seed': seed, 'status': 'already_complete'})
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            logs = root / 'logs' / 'training' / variant
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / f'seed_{seed}.log').open('w') as logfile:
                subprocess.run([sys.executable, '-m', 'dart_marl', 'train',
                                '--config', str(config.resolve()), '--out', str(out.resolve())],
                               check=True, stdout=logfile, stderr=subprocess.STDOUT)
            if not (out / 'summary.json').exists():
                # Some releases write a different summary filename: checkpoint is the hard requirement.
                if not (out / 'checkpoints/latest.pt').exists():
                    raise RuntimeError(f'training did not create checkpoint: {out}')
            done.append({'variant': variant, 'seed': seed, 'status': 'trained'})
    return {'training': done, 'complete': len(done)}


def select_cohort(root: str | Path, *, variants: list[str] | None = None,
                  seeds: list[int] | None = None) -> dict:
    from dart_marl.algorithms.mappo.selection import select_critic_checkpoint
    root, spec, data = _protocol(root)
    _verify_sources(data, root)
    if (root / 'frozen_development.json').exists():
        raise RuntimeError('development frozen: checkpoint selection forbidden')
    variants = variants or list(spec['training_variants'])
    seeds = seeds or spec['training_seeds']
    if any(v not in spec['training_variants'] for v in variants) or any(
            s not in spec['training_seeds'] for s in seeds):
        raise ValueError('requested variant or seed was not preregistered')
    selected = []
    settings = spec['checkpoint_selection']
    for variant in variants:
        for seed in seeds:
            target = _selection_path(root, variant, seed)
            if target.exists():
                ckpt, digest = _selected(root, variant, seed)
                selected.append({'variant': variant, 'seed': seed, 'checkpoint': str(ckpt),
                                 'sha256': digest, 'status': 'existing'})
                continue
            folder = _train_path(root, variant, seed)
            candidates = sorted((folder / 'checkpoints').glob('update_*.pt'))
            if not candidates:
                raise FileNotFoundError(f'no saved checkpoints for {variant} seed {seed}')
            result = select_critic_checkpoint(
                checkpoints=candidates, training_env_config=spec['training_environment'],
                scenarios=spec['validation_scenarios'], episodes=int(settings['episodes']),
                seed=int(settings['seed']), device=settings['device'], out=target)
            selected.append({'variant': variant, 'seed': seed,
                             'checkpoint': result['selected_checkpoint'],
                             'sha256': result['selected_checkpoint_sha256'], 'status': 'selected'})
    return {'selection': selected}



def _cft_model_path(root: Path, variant: str, seed: int, method: str) -> Path:
    return root / 'cft' / variant / f'seed_{seed}' / method / 'cft_model.pt'


def _verified_cft_models(root: Path, spec: dict, variant: str, seed: int,
                         controller_sha: str) -> dict[str, Path]:
    from dart_marl.cft.training import CFTEstimator
    base = load_config(spec['training_environment'])
    found = {}
    for method in ('cft','cft_rank'):
        if method not in spec['allocation']['methods']:
            continue
        target = _cft_model_path(root, variant, seed, method)
        if not target.exists():
            raise FileNotFoundError(f'CFT model missing for {variant}, training seed {seed}: {target}')
        # Loading with weights_only=True, and enforce controller/task/horizon/discount identity.
        CFTEstimator(target, expected_config=base, expected_controller_sha=controller_sha,
                     horizon=int(spec['allocation']['forecast_horizon']), discount=.98,
                     strict_controller=True)
        found[method] = target
    return found


def train_cft_cohort(root: str | Path, *, variants: list[str] | None = None,
                     seeds: list[int] | None = None) -> dict:
    """Per-checkpoint CFT datasets and models; never reuse another controller's model."""
    from dart_marl.cft.rollouts import generate_data
    from dart_marl.cft.training import train_predictor
    from dart_marl.cft.rank_training import train_rank_predictor
    from dart_marl.algorithms.mappo.trainer import resolve_device

    root, spec, data = _protocol(root)
    _verify_sources(data, root)
    if (root / 'frozen_development.json').exists():
        raise RuntimeError('cannot train CFT models after development freeze')
    if not any(m in spec['allocation']['methods'] for m in ('cft','cft_rank')):
        raise ValueError('CFT objectives not preregistered for this experiment')
    cft = spec['cft']
    variants = variants or list(spec['training_variants'])
    seeds = seeds or spec['training_seeds']
    if any(v not in spec['training_variants'] for v in variants) or any(
            s not in spec['training_seeds'] for s in seeds):
        raise ValueError('seed or variant not in preregistration')
    outputs = []
    for variant in variants:
        for seed in seeds:
            checkpoint, digest = _selected(root, variant, seed)
            base = root / 'cft' / variant / f'seed_{seed}'
            data_dir = base / 'dataset'
            if not data_dir.exists():
                generate_data(spec['training_environment'], data_dir, controller='checkpoint',
                              checkpoint=checkpoint, training_env_config=spec['training_environment'],
                              device='cpu', episodes_train=int(cft['train_episodes']),
                              episodes_val=int(cft['val_episodes']),
                              episodes_test=int(cft['test_episodes']),
                              seed=int(cft['seed_base']) + seed * 1000, event_mode=cft['event_mode'],
                              horizon=int(spec['allocation']['forecast_horizon']),
                              discount=.98, sample_period=int(cft['sample_period']),
                              candidates=int(cft['candidates']), beam=int(cft['beam']))
            else:
                metadata = json.loads((data_dir / 'manifest.json').read_text())
                if metadata['controller_checkpoint_sha256'] != digest:
                    raise RuntimeError('existing CFT dataset was made from a different checkpoint')
            for method in ('cft','cft_rank'):
                if method not in spec['allocation']['methods']:
                    continue
                target = base / method
                if not target.exists():
                    trainer = train_predictor if method == 'cft' else train_rank_predictor
                    extra = ({'rank_weight': float(cft.get('rank_weight',.65)),
                              'rank_temperature':float(cft.get('rank_temperature',5.))}
                             if method == 'cft_rank' else {})
                    trainer(data_dir, target, seed=int(cft['model_seed']),
                            epochs=int(cft['epochs']), patience=int(cft['patience']),
                            device=str(resolve_device(cft['device'])), **extra)
            models = _verified_cft_models(root, spec, variant, seed, digest)
            outputs.append({'variant': variant, 'seed':seed, 'model_sha256':
                            {k:sha(p) for k,p in models.items()},
                            'dataset_sha256':sha(data_dir/'counterfactuals.npz')})
    return {'completed': outputs}


def _eval_path(root: Path, stage: str, variant: str, seed: int, scenario: str) -> Path:
    return root / 'evaluations' / stage / variant / f'seed_{seed}' / (scenario + '.json')


def _scenario_paths(spec: dict, stage: str) -> list[str]:
    return spec['validation_scenarios'] if stage == 'dev' else spec['test_scenarios']


def evaluate_cohort(root: str | Path, *, stage: str, unlock_test: bool = False,
                    resume_test: bool = False) -> dict:
    import torch
    root, spec, data = _protocol(root)
    _verify_sources(data, root)
    if stage not in {'dev', 'test'}:
        raise ValueError('stage must be dev or test')
    frozen = root / 'frozen_development.json'
    test_gate = root / 'sealed_test_started.json'
    if stage == 'dev' and frozen.exists():
        raise RuntimeError('development was frozen; no further development evaluation')
    if stage == 'test':
        if not unlock_test or not frozen.exists():
            raise PermissionError('test requires --unlock-test after freeze')
        manifest = json.loads(frozen.read_text())
        if sha(root / 'dev_summary.json') != manifest['development_summary_sha256']:
            raise RuntimeError('development report was changed after freeze')
        if manifest.get('confirmatory') != spec.get('confirmatory'):
            raise RuntimeError('primary hypothesis changed after freeze')
        for key, digest in manifest['selected_checkpoint_sha256'].items():
            variant, seed_s = key.split('|')
            _, live = _selected(root, variant, int(seed_s))
            if live != digest:
                raise RuntimeError('selected checkpoint differs from frozen version')
        for key, model_hashes in manifest['cft_model_sha256'].items():
            variant, seed_s = key.split('|')
            _, controller_sha = _selected(root, variant, int(seed_s))
            if {m:sha(p) for m,p in _verified_cft_models(
                    root, spec, variant, int(seed_s), controller_sha).items()} != model_hashes:
                raise RuntimeError('CFT model differs from frozen model')
        if test_gate.exists():
            if not resume_test:
                raise RuntimeError('held-out test already started; use explicit --resume-test only to finish incomplete evaluation')
            if json.loads(test_gate.read_text())['protocol_sha256'] != data['protocol_sha256']:
                raise RuntimeError('test-run protocol mismatch')
        else:
            _write_new(test_gate, {'protocol_sha256': data['protocol_sha256'],
                                   'warning': 'FINAL test opened once; no tuning after observing these results'})
    count = 0
    scenarios = _scenario_paths(spec, stage)
    for variant in spec['training_variants']:
        for seed in spec['training_seeds']:
            ckpt, digest = _selected(root, variant, seed)
            checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
            if int(checkpoint['train_config']['seed']) != seed:
                raise ValueError('checkpoint training seed disagrees with experiment protocol')
            config = yaml.safe_load((root / 'resolved_configs' / variant / f'seed_{seed}.yaml').read_text())
            recorded = dict(checkpoint['train_config'])
            if int(config['seed']) != int(recorded['seed']):
                raise ValueError('resolved YAML seed differs from checkpoint')
            for scenario in scenarios:
                scenario_name = load_config(scenario).name
                path = _eval_path(root, stage, variant, seed, scenario_name)
                if path.exists():
                    existing = json.loads(path.read_text())
                    if (existing.get('checkpoint_sha256') != digest or
                            existing.get('protocol_sha256') != data['protocol_sha256'] or
                            existing.get('scenario_sha256') != sha(scenario) or
                            existing.get('stage') != stage or
                            existing.get('training_seed') != seed or
                            existing.get('variant') != variant or
                            set(existing.get('methods', {})) != set(spec['allocation']['methods']) or
                            len(existing.get('evaluation_seeds', [])) != int(spec['allocation']['episodes']) or
                            any(len(m.get('returns', [])) != int(spec['allocation']['episodes'])
                                for m in existing.get('methods', {}).values()) or
                            existing.get('cft_model_sha256', {}) != {m: sha(p) for m,p in
                                _verified_cft_models(root, spec, variant, seed, digest).items()}):
                        raise ValueError('existing experiment report does not match frozen protocol')
                    count += 1
                    continue
                cft_models = _verified_cft_models(root, spec, variant, seed, digest)
                report = evaluate_allocators(
                    scenario, controller='checkpoint', checkpoint=ckpt,
                    training_env_config=spec['training_environment'],
                    methods=tuple(spec['allocation']['methods']),
                    episodes=int(spec['allocation']['episodes']),
                    seed=int(spec['allocation']['development_seed' if stage == 'dev' else 'test_seed']),
                    device=spec['allocation']['device'], capture_trace=False,
                    forecast_horizon=int(spec['allocation']['forecast_horizon']),
                    cft_model=cft_models.get('cft'), cft_rank_model=cft_models.get('cft_rank'),
                    cft_candidates=int(spec.get('cft', {}).get('candidates', 6)),
                    cft_beam=int(spec.get('cft', {}).get('beam', 200)))
                if report['controller']['checkpoint_sha256'] != digest:
                    raise ValueError('controller identity mismatch')
                minimal = {
                    'format': 'dart-marl-study-eval-v1', 'stage': stage,
                    'protocol_sha256': data['protocol_sha256'],
                    'variant': variant, 'training_seed': seed, 'checkpoint_sha256': digest,
                    'cft_model_sha256': {m:sha(p) for m,p in cft_models.items()},
                    'scenario': scenario_name, 'scenario_sha256': sha(scenario),
                    'evaluation_seeds': report['evaluation_seeds'],
                    'allocator_settings': report['allocator_settings'],
                    'methods': {
                        k: {'mean_return': v['mean_return'], 'returns': v['returns'],
                            'mean_after_first_disruption_return': v['mean_after_first_disruption_return'],
                            'mean_role_switches': v['mean_role_switches'],
                            'mean_role_replans': v['mean_role_replans'],
                            'mean_poi_service_steps': v['mean_poi_service_steps'],
                            'recovery_metrics': v['recovery_metrics']}
                        for k, v in report['methods'].items()},
                }
                _write_new(path, minimal)
                count += 1
    return {'evaluated_or_verified': count, 'stage': stage,
            'total_expected': len(spec['training_variants']) * len(spec['training_seeds']) * len(scenarios)}


def _collect(root: Path, spec: dict, data: dict, stage: str) -> list[dict]:
    rows = []
    for variant in spec['training_variants']:
        for seed in spec['training_seeds']:
            ckpt, digest = _selected(root, variant, seed)
            for path in _scenario_paths(spec, stage):
                name = load_config(path).name
                p = _eval_path(root, stage, variant, seed, name)
                row = json.loads(p.read_text())
                cft_models = _verified_cft_models(root, spec, variant, seed, digest)
                if (row.get('cft_model_sha256', {}) != {m:sha(p) for m,p in cft_models.items()}):
                    raise ValueError(f'CFT model provenance mismatch: {p}')
                if (row['stage'] != stage or row['protocol_sha256'] != data['protocol_sha256'] or
                        row['scenario_sha256'] != sha(path) or row['checkpoint_sha256'] != digest or
                        row['training_seed'] != seed or row['variant'] != variant):
                    raise ValueError(f'provenance mismatch: {p}')
                if set(row['methods']) != set(spec['allocation']['methods']):
                    raise ValueError('method mismatch versus preregistration')
                expected_eval_seeds = [
                    int(spec['allocation']['development_seed' if stage == 'dev' else 'test_seed']) + 1009*i
                    for i in range(int(spec['allocation']['episodes']))]
                if row['evaluation_seeds'] != expected_eval_seeds:
                    raise ValueError('evaluation seeds differ from the preregistered paired design')
                for method in row['methods'].values():
                    if len(method['returns']) != spec['allocation']['episodes']:
                        raise ValueError('incomplete per-episode method returns')
                rows.append(row)
    return rows


def summarize(root: str | Path, *, stage: str) -> dict:
    root, spec, data = _protocol(root)
    if stage not in {'dev', 'test'}:
        raise ValueError('stage must be dev or test')
    if stage == 'test' and not (root / 'sealed_test_started.json').exists():
        raise PermissionError('held-out test has not been opened')
    rows = _collect(root, spec, data, stage)
    stats = report_statistics(rows, bootstrap_seed=spec['statistics']['bootstrap_seed'],
                              replicates=spec['statistics']['bootstrap_replicates'])
    output = {'format': 'dart-marl-study-summary-v1', 'stage': stage,
              'protocol_sha256': data['protocol_sha256'],
              'training_seeds': spec['training_seeds'],
              'independent_training_seed_count': len(spec['training_seeds']),
              'evaluation_episodes_per_method_per_controller': spec['allocation']['episodes'],
              'bootstrap_unit': 'training seed (within-seed evaluation mean)',
              'statistics': stats,
              'interpretation': ('Exploratory development only; NEVER use for publication significance' if stage == 'dev'
                                 else 'Held-out descriptive estimates; uncertainty is over independently trained controllers'),
              'limitations': ('Fixed role-navigation prior remains an engineered controller component. '
                              'These results cover fault-schedule shift, not unseen maps/task geometries. '
                              'Learned CFT (when preregistered) is trained separately for each checkpoint. '
                              'Bootstrap percentile intervals are not a guarantee of power/significance.'),
              'primary_method': spec.get('primary_method', 'r_cart'),
              'primary_outcome': f"Paired difference in mean team return: {spec.get('primary_method', 'r_cart')} minus switch_aware",
              'predeclared_primary_scenario': load_config(spec['test_scenarios'][0]).name}
    confirm = spec.get('confirmatory')
    if confirm is not None:
        output['confirmatory'] = confirm
        if stage == 'test':
            scenario = load_config(confirm['primary_scenario']).name
            contrast_name = (f"{confirm['primary_method']}_minus_{confirm['baseline_method']}")
            block = stats['by_variant_scenario'][confirm['primary_variant']][scenario]
            if contrast_name not in block['paired_contrasts']:
                raise ValueError('predeclared primary contrast missing from statistical report')
            result = block['paired_contrasts'][contrast_name]
            if result['n_independent_training_seeds'] < confirm['minimum_training_seeds']:
                raise ValueError('incomplete independent-seed primary test')
            output['predeclared_primary_result'] = {
                'variant': confirm['primary_variant'], 'scenario': scenario,
                'contrast': contrast_name, 'metric': confirm['primary_metric'],
                'seed_level_estimate_and_ci': result,
                'interpretation': ('Descriptive paired estimate across independent training seeds. '
                                   'A bootstrap interval is not a guarantee of significance or power.'),
            }
    path = root / f'{stage}_summary.json'
    _write_new(path, output)
    return output


def freeze_development(root: str | Path) -> dict:
    root, spec, data = _protocol(root)
    if (root / 'frozen_development.json').exists():
        raise FileExistsError('development already frozen')
    if 'DEVELOPMENT_ONLY' in spec.get('name',''):
        raise PermissionError('development-only pilot must never unlock the final test')
    _verify_sources(data, root)
    # Complete development matrix and a saved summary required before opening TEST_ONLY.
    _collect(root, spec, data, 'dev')
    summary = root / 'dev_summary.json'
    if not summary.exists():
        raise ValueError('run development summary before freeze')
    saved_summary = json.loads(summary.read_text())
    if (saved_summary.get('stage') != 'dev' or
            saved_summary.get('protocol_sha256') != data['protocol_sha256'] or
            saved_summary.get('primary_method') != spec.get('primary_method', 'r_cart') or
            saved_summary.get('confirmatory') != spec.get('confirmatory')):
        # Historical legacy_study reports have no 'confirmatory' key, matching None.
        raise ValueError('development summary does not match registered primary hypothesis')
    selected = {}
    cft_model_hashes = {}
    for variant in spec['training_variants']:
        for seed in spec['training_seeds']:
            _, digest = _selected(root, variant, seed)
            selected[f'{variant}|{seed}'] = digest
            cft_model_hashes[f'{variant}|{seed}'] = {m:sha(p) for m,p in
                _verified_cft_models(root, spec, variant, seed, digest).items()}
    record = {'protocol_sha256': data['protocol_sha256'],
              'development_summary_sha256': sha(summary),
              'selected_checkpoint_sha256': selected,
              'cft_model_sha256': cft_model_hashes,
              'primary_outcome': f"paired {spec.get('primary_method', 'r_cart')} minus Switch-Aware return",
              'confirmatory': spec.get('confirmatory'),
              'test_scenarios': {p: sha(p) for p in spec['test_scenarios']},
              'notice': 'Frozen selections and analyses; no tuning on held-out test outcomes'}
    _write_new(root / 'frozen_development.json', record)
    return record


def cohort_status(root: str | Path) -> dict:
    """Read-only progress inventory; never evaluates or accesses TEST_ONLY outcomes."""
    root, spec, data = _protocol(root)
    _verify_sources(data, root)
    counts = {'registered_controllers': len(spec['training_variants']) * len(spec['training_seeds']),
              'trained_controllers': 0, 'selected_controllers': 0,
              'cft_datasets': 0, 'cft_models': 0,
              'dev_evaluations': 0, 'expected_dev_evaluations':
              len(spec['training_variants']) * len(spec['training_seeds']) * len(spec['validation_scenarios'])}
    missing = {'training': [], 'selection': [], 'cft': [], 'development_evaluation': []}
    for variant in spec['training_variants']:
        for seed in spec['training_seeds']:
            tag = f'{variant}/seed_{seed}'
            training = _train_path(root, variant, seed)
            if (training / 'checkpoints/latest.pt').exists() and (training / 'summary.json').exists():
                counts['trained_controllers'] += 1
            else:
                missing['training'].append(tag)
            selection = _selection_path(root, variant, seed)
            if selection.exists():
                _selected(root, variant, seed)  # Demand checksum identity even in read-only status.
                counts['selected_controllers'] += 1
            else:
                missing['selection'].append(tag)
            if any(m in spec['allocation']['methods'] for m in ('cft', 'cft_rank')):
                ds = root / 'cft' / variant / f'seed_{seed}' / 'dataset'
                if (ds / 'manifest.json').exists() and (ds / 'counterfactuals.npz').exists():
                    counts['cft_datasets'] += 1
                else:
                    missing['cft'].append(f'{tag}/dataset')
                for method in ('cft', 'cft_rank'):
                    if method in spec['allocation']['methods']:
                        if _cft_model_path(root, variant, seed, method).exists():
                            counts['cft_models'] += 1
                        else:
                            missing['cft'].append(f'{tag}/{method}')
            for scenario in spec['validation_scenarios']:
                name = load_config(scenario).name
                if _eval_path(root, 'dev', variant, seed, name).exists():
                    counts['dev_evaluations'] += 1
                else:
                    missing['development_evaluation'].append(f'{tag}/{name}')
    return {'study_kind': ('service_aware' if spec.get('name', '').startswith('service_aware_') else 'allocator_benchmark'),
            'protocol_sha256': data['protocol_sha256'],
            'primary_hypothesis': spec.get('confirmatory'),
            'progress': counts, 'missing': missing,
            'development_summary_exists': (root / 'dev_summary.json').exists(),
            'frozen': (root / 'frozen_development.json').exists(),
            'sealed_test_opened': (root / 'sealed_test_started.json').exists(),
            'note': 'Status is read-only and does not run or report held-out test outcomes.'}
