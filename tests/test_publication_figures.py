"""Regression guards against publishing synthetic/unpaired 12-rover plots."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.make_publication_figures import _check_report, _per_step, _smooth

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'results/team12/paper_figures'


def test_published_composite_figures_have_original_source_hashes():
    manifest=json.loads((ASSET/'manifest.json').read_text())
    assert len(manifest['figures'])==12
    assert manifest['paired_evaluation_episodes']==8
    assert manifest['demo_event_boundaries']==11
    assert len(manifest['source_sha256'])==5
    for path,sha in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    for name,sha in manifest['artifacts_sha256'].items():
        assert hashlib.sha256((ASSET/name).read_bytes()).hexdigest()==sha
    for name in manifest['figures']:
        assert (ASSET/name).stat().st_size>1500
        if name.endswith('.pdf'):assert (ASSET/name).read_bytes()[:5]==b'%PDF-'
    assert 'No 12-rover training-seed confidence interval' in manifest['limitations']


def test_paired_method_validation_rejects_shuffled_episode_seeds():
    report=json.loads((ROOT/'results/team12/data/allocator_report.json').read_text())
    blocks,seeds=_check_report(report)
    assert len(seeds)==8
    assert blocks['r_cart']['mean_return']<blocks['switch_aware']['mean_return']
    report['methods']['r_cart']['per_episode'][0]['seed']=-999
    with pytest.raises(ValueError, match='unmatched'):
        _check_report(report)


def test_table_equals_observed_episode_means_and_not_inferred_training_ci():
    report=json.loads((ROOT/'results/team12/data/allocator_report.json').read_text())
    rows=list(csv.DictReader((ASSET/'allocator_summary.csv').open()))
    assert len(rows)==9
    for r in rows:
        block=report['methods'][r['method']]
        assert float(r['mean_return'])==pytest.approx(np.mean([x['return'] for x in block['per_episode']]))
        assert float(r['mean_delta_vs_switch_aware'])==pytest.approx(
            block['mean_return']-report['methods']['switch_aware']['mean_return'])


def test_trace_uses_one_controller_and_real_disruption_boundaries():
    trace=json.loads((ROOT/'media/rover_reallocation_vs_fixed_12_trace.json').read_text())
    left,right,events=_per_step(trace)
    assert left['method']=='switch_aware'
    assert right['method']=='fixed'
    assert len(events)==11
    assert len(left['states'])==121
    assert left['final_return']==pytest.approx(786.7)
    assert right['final_return']==pytest.approx(639.5)


def test_smoothing_keeps_input_values_and_missing_intervals_intact():
    data=np.array([2.,np.nan,4.,6.,8.])
    original=data.copy()
    smooth=_smooth(data,3)
    assert np.allclose(original,data,equal_nan=True)
    assert smooth[0]==pytest.approx(2.)
    assert np.isfinite(smooth).all()


def test_regeneration_to_new_folder_is_recorded_data_only(tmp_path):
    pytest.importorskip('matplotlib')
    from scripts.make_publication_figures import build
    m=build(ROOT/'results/team12',tmp_path/'figures')
    assert len(m['figures'])==12
    assert len(list((tmp_path/'figures').glob('*.tex')))==1
    assert m['source_sha256']==json.loads((ASSET/'manifest.json').read_text())['source_sha256']
    with pytest.raises(FileExistsError):
        build(ROOT/'results/team12',tmp_path/'figures')
