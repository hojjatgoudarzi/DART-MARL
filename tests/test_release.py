"""Public release identity, reproducibility, and actual trained-policy demos."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


def test_example_actor_exports_are_safe_and_match_manifest():
    torch = pytest.importorskip('torch')
    root=Path('examples/checkpoints')
    data=json.loads((root/'manifest.json').read_text())
    assert len(data['models'])==3
    assert {item['seed'] for item in data['models']}=={202,404,314159}
    for item in data['models']:
        path=root/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['exported_actor_sha256']
        model=torch.load(path,map_location='cpu',weights_only=True)
        env_name=('rover_team_12.yaml' if item['seed']==314159 else 'dev_roles.yaml')
        sha=hashlib.sha256((Path('configs/env')/env_name).read_bytes()).hexdigest()
        assert model['env_sha256']==sha==item['training_environment_sha256']
        assert model['train_config']['seed']==item['seed']
        assert model['actor'] and all(isinstance(v,torch.Tensor) for v in model['actor'].values())


def test_actual_actor_demo_deterministic_paired_rewards_and_events():
    pytest.importorskip('torch')
    from scripts.make_demo import rollout
    env=Path('configs/env/dev_failure.yaml')
    train=Path('configs/env/dev_roles.yaml')
    ckpt=Path('examples/checkpoints/fault_robust_seed_202.pt')
    a=rollout(env,train,ckpt,'switch_aware',900000)
    b=rollout(env,train,ckpt,'r_cart',900000)
    assert len(a['states'])==len(b['states'])==49
    assert a['final_return']==pytest.approx(120.0,abs=1e-6)
    assert b['final_return']==pytest.approx(118.8,abs=1e-6)
    assert a['controller_sha256']==b['controller_sha256']
    assert a['states'][0]['positions']==b['states'][0]['positions']
    ev=[(st['step'],e['kind']) for st in a['states'] for e in st['events']]
    assert (16,'temporary_disable') in ev
    assert (28,'replacement_agent') in ev
    assert (21,'automatic_reentry') in ev
    assert any((x['roles']!=y['roles']) for x,y in zip(a['states'],b['states']))


def test_published_episode_returns_and_primary_stats():
    rows=list(csv.DictReader(Path('results/heldout_episode_returns.csv').open()))
    assert len(rows)==1600
    models={(r['controller_variant'],r['training_seed']) for r in rows}
    assert len(models)==20
    assert len({r['method'] for r in rows})==10
    manifest=json.loads(Path('results/manifest.json').read_text())
    assert hashlib.sha256(Path('results/heldout_episode_returns.csv').read_bytes()).hexdigest()==manifest['raw_files_sha256']['heldout_episode_returns.csv']
    fault=[r for r in rows if r['controller_variant']=='fault_robust']
    def mean(m):
        vals=[float(r['team_return']) for r in fault if r['method']==m]
        assert len(vals)==80
        return sum(vals)/len(vals)
    assert mean('switch_aware')==pytest.approx(93.54)
    assert mean('r_cart')==pytest.approx(91.4175)
    assert mean('r_cart')-mean('switch_aware')==pytest.approx(-2.1225)
    assert manifest['primary_comparison']['r_cart_minus_switch_aware']==pytest.approx(-2.1225)


def test_twelve_rover_trained_actor_and_recorded_stats():
    pytest.importorskip('torch')
    from scripts.make_demo import rollout
    entry=json.loads(Path('results/team12/provenance.json').read_text())
    checkpoint=Path('examples/checkpoints/team12_seed_314159.pt')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==entry['released_actor_only_checkpoint_sha256']
    assert len(list(csv.DictReader(Path('results/team12/data/episode_results.csv').open())))==72
    eval_data=json.loads(Path('results/team12/data/allocator_report.json').read_text())
    assert eval_data['controller']['checkpoint_sha256']==entry['original_training_checkpoint_sha256']
    assert eval_data['methods']['r_cart']['mean_return'] < eval_data['methods']['switch_aware']['mean_return']
    left=rollout(Path('configs/env/rover_team_12_showcase.yaml'),Path('configs/env/rover_team_12.yaml'),checkpoint,'switch_aware',901110)
    assert left['final_return']==pytest.approx(786.7,abs=1e-6)
    assert len(left['states'])==121
    assert sum(len(x['events']) for x in left['states'])==11


def test_complete_twelve_rover_public_artifact_integrity():
    from scripts.verify_published_results import verify_release
    outcome = verify_release()
    assert outcome['status'] == 'verified'
    assert outcome['evaluated_methods'] == 9
    assert outcome['evaluation_episodes_per_method'] == 8
