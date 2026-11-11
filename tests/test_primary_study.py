"""legacy_study primary-hypothesis preregistration, seal, and research statistics."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from dart_marl.research.study import _assert_spec, cohort_status, initialize, summarize
from dart_marl.research.statistics import report_statistics


FULL = Path('configs/experiments/completed_study_protocol.yaml')


def full_spec():
    return yaml.safe_load(FULL.read_text())


def test_primary_study_original_rcart_is_preregistered_primary():
    cfg = _assert_spec(full_spec())
    assert cfg['primary_method'] == 'r_cart'
    assert cfg['confirmatory']['primary_variant'] == 'fault_robust'
    assert cfg['confirmatory']['primary_metric'] == 'mean_return'
    assert cfg['test_scenarios'] == ['configs/env/primary_study_final_test_only.yaml']
    assert cfg['training_seeds'] == [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    assert {'r_cart', 'r_cart_bounded_sync', 'r_cart_no_sync', 'switch_aware',
            'cft', 'cft_rank'}.issubset(set(cfg['allocation']['methods']))


@pytest.mark.parametrize('change,pattern', [
    ({'training_seeds': [11, 101, 202, 303, 404, 505, 606, 707, 808, 909]}, 'consulted'),
    ({'training_seeds': [101, 202]}, 'insufficient'),
    ({'test_scenarios': ['configs/env/historical_fault_stress.yaml',
                         'configs/env/primary_study_final_test_only.yaml']}, 'first sealed test'),
    ({'test_scenarios': ['configs/env/primary_study_final_test_only.yaml',
                         'configs/env/historical_fault_stress.yaml']}, 'historical TEST_ONLY'),
    ({'test_scenarios': ['configs/env/historical_stress_test.yaml',
                         'configs/env/primary_study_final_test_only.yaml']}, 'first sealed test'),
    ({'primary_method': 'r_cart_bounded_sync'}, 'confirmatory'),
])
def test_primary_study_rejects_retroactive_protocol_changes(change, pattern):
    cfg = full_spec()
    cfg.update(deepcopy(change))
    with pytest.raises(ValueError, match=pattern):
        _assert_spec(cfg)


def test_primary_study_status_checks_progress_without_test_access(tmp_path):
    study = tmp_path / 'full_study'
    initialize(FULL, study)
    status = cohort_status(study)
    assert status['progress']['registered_controllers'] == 20
    assert status['progress']['trained_controllers'] == 0
    assert status['progress']['selected_controllers'] == 0
    assert status['progress']['expected_dev_evaluations'] == 60
    assert not status['sealed_test_opened']
    assert not (study / 'sealed_test_started.json').exists()
    assert json.loads((study / 'protocol.json').read_text())['confirmatory']['primary_method'] == 'r_cart'


def _mock_report_rows():
    rows = []
    for variant in ('nominal', 'fault_robust'):
        for seed in [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]:
            methods = {}
            for name, ret in [('switch_aware', 10. + seed/10000), ('r_cart', 12. + seed/10000),
                              ('r_cart_bounded_sync', 11. + seed/10000)]:
                methods[name] = {
                    'mean_return': ret, 'mean_after_first_disruption_return': ret/2,
                    'mean_role_switches': 2., 'mean_poi_service_steps': [3, 6, 9],
                    'recovery_metrics': {
                        'restoration_rate_among_at_risk': .75,
                        'mean_post_event_repeat_service_fraction': .55,
                        'mean_first_restoration_lag_observed_only': 3.,
                    }}
            rows.append({'variant': variant, 'scenario': 'primary_study_preregistered_multifault_TEST_ONLY',
                         'training_seed': seed, 'evaluation_seeds': list(range(8)),
                         'methods': methods})
    return rows


def test_primary_study_reporting_is_training_seed_paired_and_recovers_secondary_metrics():
    summary = report_statistics(_mock_report_rows(), bootstrap_seed=42, replicates=120)
    block = summary['by_variant_scenario']['fault_robust']['primary_study_preregistered_multifault_TEST_ONLY']
    primary = block['paired_contrasts']['r_cart_minus_switch_aware']
    assert primary['n_independent_training_seeds'] == 10
    assert primary['mean'] == pytest.approx(2.)
    assert primary['ci95_training_seed_bootstrap_percentile'] == pytest.approx([2., 2.])
    secondaries = block['secondary_paired_contrasts']['r_cart_minus_switch_aware']
    assert secondaries['post_disruption_return']['mean'] == pytest.approx(1.)
    assert secondaries['poi_service_steps']['mean'] == pytest.approx(0.)
    assert secondaries['restoration_rate_among_at_risk']['mean'] == 0


def test_primary_study_primary_report_is_locked_to_fault_robust_test(tmp_path, monkeypatch):
    from dart_marl.research import study
    spec = full_spec()
    spec['statistics']['bootstrap_replicates'] = 120
    path = tmp_path / 'spec.yaml'
    path.write_text(yaml.safe_dump(spec))
    root = tmp_path / 'study'
    initialize(path, root)
    (root / 'sealed_test_started.json').write_text('{}')
    monkeypatch.setattr(study, '_collect', lambda *args: _mock_report_rows())
    result = summarize(root, stage='test')
    primary = result['predeclared_primary_result']
    assert primary['variant'] == 'fault_robust'
    assert primary['scenario'] == 'primary_study_preregistered_multifault_TEST_ONLY'
    assert primary['contrast'] == 'r_cart_minus_switch_aware'
    assert primary['seed_level_estimate_and_ci']['n_independent_training_seeds'] == 10
    assert result['primary_method'] == 'r_cart'
