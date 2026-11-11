"""legacy_study protocol, pairing and sealed-test regression checks."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dart_marl.research.study import (
    _assert_spec, _collect, _selected, _train_path, evaluate_cohort,
    freeze_development, initialize, select_cohort, summarize, train_cohort,
)
from dart_marl.research.statistics import report_statistics, seed_summary


def _toy_spec(tmp_path):
    original = yaml.safe_load(Path('configs/train/mappo_team_smoke.yaml').read_text())
    original.update(seed=5, device='cpu', num_envs=2, rollout_steps=8,
                    total_env_steps=16, hidden_size=16, eval_episodes=2, eval_every=1,
                    save_every=1, torch_threads=1)
    original['ppo'].update(ppo_epochs=1, num_minibatches=1)
    config_a = tmp_path / 'nominal.yaml'
    config_a.write_text(yaml.safe_dump(original))
    config_b = tmp_path / 'fault.yaml'
    original['fault_mode'] = 'diverse'
    config_b.write_text(yaml.safe_dump(original))
    spec = yaml.safe_load(Path('configs/experiments/development_study.yaml').read_text())
    spec['name'] = 'toy_research_checks'
    spec['training_variants'] = {'nominal': str(config_a), 'fault': str(config_b)}
    spec['training_seeds'] = [17]
    spec['validation_scenarios'] = ['configs/env/dev_failure.yaml']
    spec['checkpoint_selection']['episodes'] = 2
    spec['allocation']['episodes'] = 1
    spec['allocation']['methods'] = ['fixed', 'switch_aware', 'r_cart', 'r_cart_no_sync']
    spec['allocation']['device'] = 'cpu'
    spec['statistics']['bootstrap_replicates'] = 100
    dest = tmp_path / 'protocol.yaml'
    dest.write_text(yaml.safe_dump(spec))
    return dest


def test_seed_summary_nested_uncertainty():
    one = seed_summary([4.0])
    assert one['ci95_training_seed_bootstrap_percentile'] is None
    assert one['n_independent_training_seeds'] == 1
    two = seed_summary([3.0, 7.0], bootstrap_seed=123, replicates=500)
    assert two == seed_summary([3.0, 7.0], bootstrap_seed=123, replicates=500)
    assert two['sample_std'] == pytest.approx(np.sqrt(8))
    with pytest.raises(ValueError):
        seed_summary([float('nan'), 1])


def test_pairing_detects_duplicate_and_unmatched():
    def row(seed, method_seed):
        return {'variant': 'v', 'scenario': 's', 'training_seed': seed,
                'evaluation_seeds': [method_seed],
                'methods': {name: {'mean_return': value,
                                   'mean_after_first_disruption_return': value / 2,
                                   'mean_role_switches': 0,
                                   'recovery_metrics': {'restoration_rate_among_at_risk': .5}}
                            for name, value in [('switch_aware', 10 + seed),
                                                ('r_cart', 11 + seed)]}}
    good = report_statistics([row(1, 777), row(2, 777)], replicates=100)
    contrast = good['by_variant_scenario']['v']['s']['paired_contrasts']['r_cart_minus_switch_aware']
    assert contrast['mean'] == 1
    assert contrast['n_independent_training_seeds'] == 2
    with pytest.raises(ValueError, match='duplicate'):
        report_statistics([row(1, 777), row(1, 777)], replicates=100)
    with pytest.raises(ValueError, match='unmatched'):
        report_statistics([row(1, 777), row(2, 888)], replicates=100)


def test_protocol_rejects_heldout_as_validation(tmp_path):
    spec = yaml.safe_load(_toy_spec(tmp_path).read_text())
    spec['validation_scenarios'] = ['configs/env/historical_fault_stress.yaml']
    with pytest.raises(ValueError, match='duplicate scenario|label'):
        _assert_spec(spec)
    spec = yaml.safe_load(_toy_spec(tmp_path).read_text())
    spec['training_seeds'] = [17, 17]
    with pytest.raises(ValueError, match='unique'):
        _assert_spec(spec)


@pytest.fixture(scope='module')
def study_fixture(tmp_path_factory):
    path = tmp_path_factory.mktemp('studyfull')
    spec = _toy_spec(path)
    root = path / 'study'
    initialized = initialize(spec, root)
    trained = train_cohort(root)
    selected = select_cohort(root)
    return root, spec, initialized, trained, selected


def test_train_and_development_outputs(study_fixture):
    root, _, initialized, trained, selected = study_fixture
    assert initialized['status'] == 'development'
    assert trained['complete'] == 2
    assert len(selected['selection']) == 2
    assert train_cohort(root)['training'][0]['status'] == 'already_complete'
    ckpt, _ = _selected(root, 'nominal', 17)
    assert ckpt.exists()
    assert json.loads((root/'protocol.json').read_text())['primary_comparison'].startswith('r_cart')


def test_development_and_sealed_test(study_fixture):
    root, _, _, _, _ = study_fixture
    with pytest.raises(PermissionError):
        evaluate_cohort(root, stage='test', unlock_test=True)
    dev = evaluate_cohort(root, stage='dev')
    assert dev['total_expected'] == 2
    assert evaluate_cohort(root, stage='dev')['evaluated_or_verified'] == 2
    report = summarize(root, stage='dev')
    assert report['independent_training_seed_count'] == 1
    assert report['statistics']['by_variant_scenario']['nominal']
    frozen = freeze_development(root)
    assert frozen['selected_checkpoint_sha256']
    with pytest.raises(RuntimeError, match='frozen'):
        train_cohort(root)
    with pytest.raises(RuntimeError, match='frozen'):
        select_cohort(root)
    with pytest.raises(RuntimeError, match='frozen'):
        evaluate_cohort(root, stage='dev')
    with pytest.raises(PermissionError):
        evaluate_cohort(root, stage='test')
    result = evaluate_cohort(root, stage='test', unlock_test=True)
    assert result['total_expected'] == 2
    with pytest.raises(RuntimeError, match='already started'):
        evaluate_cohort(root, stage='test', unlock_test=True)
    assert evaluate_cohort(root, stage='test', unlock_test=True, resume_test=True)['total_expected'] == 2
    report = summarize(root, stage='test')
    assert 'test' == report['stage']
    assert len(_collect(root, yaml.safe_load(Path(study_fixture[1]).read_text()),
                        json.loads((root/'protocol.json').read_text()), 'test')) == 2
    with pytest.raises(FileExistsError):
        summarize(root, stage='test')


def test_report_statistics_blocks_pseudoreplication():
    base = {'evaluation_seeds': [5,6,7], 'scenario':'s',
            'methods':{'switch_aware':{'mean_return': 2, 'mean_role_switches':0,
                                     'mean_after_first_disruption_return':1,
                                     'recovery_metrics':{'restoration_rate_among_at_risk':None}},
                       'r_cart':{'mean_return':3, 'mean_role_switches':1,
                                 'mean_after_first_disruption_return':2,
                                 'recovery_metrics':{'restoration_rate_among_at_risk':None}}}}
    a=dict(base, variant='a',training_seed=100)
    b=dict(base, variant='b',training_seed=100)
    summary = report_statistics([a,b],replicates=100)
    assert summary['between_variants']['s']['b_minus_a_r_cart']['n_independent_training_seeds']==1
    assert summary['by_variant_scenario']['a']['s']['paired_contrasts']['r_cart_minus_switch_aware']['mean']==1


def test_rejects_tampered_resolved_training_config(tmp_path):
    spec = _toy_spec(tmp_path)
    root = tmp_path / 'study'
    initialize(spec, root)
    config = root / 'resolved_configs/nominal/seed_17.yaml'
    config.write_text(config.read_text().replace('seed: 17','seed: 19'))
    with pytest.raises(ValueError, match='resolved training YAML changed'):
        train_cohort(root)


def test_pilot_cannot_open_sealed_test(tmp_path):
    spec = yaml.safe_load(_toy_spec(tmp_path).read_text())
    spec['name']='DEVELOPMENT_ONLY_pilot'
    path = tmp_path/'only.yaml';path.write_text(yaml.safe_dump(spec))
    root = tmp_path/'study'
    initialize(path, root)
    with pytest.raises(PermissionError, match='development-only'):
        freeze_development(root)


def test_cft_protocol_requires_training_settings(tmp_path):
    spec=yaml.safe_load(_toy_spec(tmp_path).read_text())
    spec['allocation']['methods'].append('cft_rank')
    with pytest.raises(ValueError, match='per-controller'):
        _assert_spec(spec)
    full=yaml.safe_load(Path('configs/experiments/multiseed_study.yaml').read_text())
    assert _assert_spec(full)['training_seeds'] == [11,22,33,44,55,66,77,88,99,110]
