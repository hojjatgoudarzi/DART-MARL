"""legacy_study: decision-aligned labels, ranking, calibration, compatibility."""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from dart_marl.cft.decision import (
    decision_metrics,
    select_validation_blend,
)
from dart_marl.cft.rank_training import pairwise_decision_loss, train_rank_predictor
from dart_marl.cft.rollouts import diverse_scenario, generate_data
from dart_marl.cft.training import CFTEstimator, train_predictor
from dart_marl.envs.config import load_config
from dart_marl.envs.rover import RoverEnv
from dart_marl.cft.features import visible_features
from dart_marl.roles.evaluation import evaluate_allocators


def test_pairwise_scores_reward_correct_order():
    actual = torch.tensor([16., 10., 3.])
    best = pairwise_decision_loss(actual, actual)
    wrong = pairwise_decision_loss(-actual, actual)
    assert best.item() < wrong.item()
    assert pairwise_decision_loss(torch.tensor([2.]),torch.tensor([5.])).item() == 0.


def test_decision_penalty_changes_true_best_candidate():
    snap = np.array(['one', 'one', 'two', 'two'])
    ix = np.array([0,1,2,3])
    # Candidate with higher raw reward can lose once switching costs count.
    y=np.array([10.,12.,3.,4.]); analytic=np.array([10.,9.,4.,2.])
    penalty=np.array([0.,-5.,0.,0.])
    pred=np.array([10.,12.,3.,4.])
    metric=decision_metrics(y,analytic,pred,penalty,snap,ix,alpha=1.)
    assert metric['candidate_selection_mean_regret'] == 0.
    assert metric['paired_snapshot_groups'] == 2


def test_calibration_uses_validation_only_and_ties_default_to_analytic():
    snaps=np.array(['validation','validation','test','test'])
    ix=np.array([0,1]); y=np.array([10.,8.,1.,1.])
    analytic=np.array([10.,9.,1.,1.]); pred=np.array([0.,1.,20.,1.])
    penalty=np.zeros(4)
    alpha, regret=select_validation_blend(y,analytic,pred,penalty,snaps,ix)
    assert alpha == 0. and regret == 0.
    y[2:]=1e9
    pred[2:]=-1e9
    assert (alpha,regret) == select_validation_blend(y,analytic,pred,penalty,snaps,ix)


def test_diverse_fault_schedules_do_not_leak_into_input():
    base=load_config('configs/env/dev_roles.yaml')
    event_counts=set()
    kinds=set()
    for seed in range(200,216):
        cfg=diverse_scenario(base,seed)
        assert cfg.events == diverse_scenario(base,seed).events
        event_counts.add(len(cfg.events))
        kinds.update(e.kind for e in cfg.events)
        a=RoverEnv(base); b=RoverEnv(cfg)
        a.reset(seed=seed);b.reset(seed=seed)
        roster=a.reported_active()
        prev=a.roles.copy(); proposed=prev.copy()
        np.testing.assert_array_equal(
            visible_features(a,roster,prev,proposed,horizon=12,discount=.98),
            visible_features(b,roster,prev,proposed,horizon=12,discount=.98))
    assert event_counts == {0,1,2,3}
    assert len(kinds)>=3


@pytest.fixture(scope='module')
def data51(tmp_path_factory):
    out=tmp_path_factory.mktemp('cft51_data')/'dataset'
    generate_data('configs/env/dev_roles.yaml',out,controller='scripted',
                  event_mode='diverse',episodes_train=4,episodes_val=2,episodes_test=2,
                  seed=991232,horizon=12,sample_period=12,candidates=4,beam=60)
    return out


@pytest.fixture(scope='module')
def rank_model(tmp_path_factory,data51):
    out=tmp_path_factory.mktemp('cft51_rank')/'model'
    train_rank_predictor(data51,out,epochs=8,patience=3,hidden=32,batch_size=32)
    return out


def test_dataset_decision_penalties_and_group_splits(data51):
    z=np.load(data51/'counterfactuals.npz',allow_pickle=False)
    assert 'decision_penalty' in z.files
    assert np.isfinite(z['decision_penalty']).all()
    assert len(z['decision_penalty'])==len(z['x'])
    for fold in ('train','val','test'):
        assert len(set(z['episode_id'][z['split']==fold].tolist()))>=2
    for a,b in [('train','test'),('val','test'),('train','val')]:
        assert set(z['episode_id'][z['split']==a]).isdisjoint(
            set(z['episode_id'][z['split']==b]))
    assert any(z['decision_penalty'] < 0.)


def test_rank_checkpoint_provenance_val_only_and_consistency(data51,rank_model):
    report=json.loads((rank_model/'training_report.json').read_text())
    assert report['format']=='dart-cft-rank-training-v2'
    assert 0<=report['selected_val_blend_alpha']<=1
    assert len(report['metrics'])==3
    payload=torch.load(rank_model/'cft_model.pt',map_location='cpu',weights_only=True)
    assert payload['blend_alpha']==report['selected_val_blend_alpha']
    est=CFTEstimator(rank_model/'cft_model.pt',
        expected_config=load_config('configs/env/dev_roles.yaml'),
        horizon=12,discount=.98,strict_controller=True)
    assert est.format=='dart-cft-rank-v2'
    assert est.blend_alpha==payload['blend_alpha']
    with np.load(data51/'counterfactuals.npz',allow_pickle=False) as z:
        predictions=est.residuals(z['x'][:3])
        assert len(predictions)==3 and np.isfinite(predictions).all()


def test_rank_and_regression_same_controller_paired_eval(rank_model,tmp_path,data51):
    regression=tmp_path/'regression'
    train_predictor(data51,regression,epochs=4,patience=2,hidden=32,batch_size=32)
    kw=dict(controller='scripted',capture_trace=False,episodes=2,seed=998800,
            forecast_horizon=12,methods=('fixed','r_cart','cft','cft_rank'),
            cft_candidates=4,cft_beam=60,
            cft_model=regression/'cft_model.pt',
            cft_rank_model=rank_model/'cft_model.pt')
    a=evaluate_allocators('configs/env/analytic_repeated_faults.yaml',**kw)
    b=evaluate_allocators('configs/env/analytic_repeated_faults.yaml',**kw)
    assert a==b
    assert 'cft_rank_minus_cft' in a['paired_differences']
    assert 'cft_rank_minus_r_cart' in a['paired_differences']
    assert a['cft_rank_model_sha256']
    with pytest.raises(ValueError,match='cft-rank-model'):
        evaluate_allocators('configs/env/analytic_repeated_faults.yaml',
                            methods=('cft_rank',),episodes=2)
    with pytest.raises(ValueError,match='cft_rank requires'):
        evaluate_allocators('configs/env/analytic_repeated_faults.yaml',
                            methods=('cft_rank',),episodes=1,forecast_horizon=12,
                            cft_rank_model=regression/'cft_model.pt',controller='scripted')


def test_critic_audit_on_tiny_trusted_training_checkpoint(tmp_path):
    from dart_marl.algorithms.mappo.trainer import train
    from dart_marl.algorithms.mappo.critic_audit import audit_checkpoint
    train('configs/train/mappo_smoke.yaml',tmp_path/'tiny_train')
    checkpoint=tmp_path/'tiny_train/checkpoints/latest.pt'
    audit=audit_checkpoint(checkpoint,'configs/env/learning_sanity.yaml',
        'configs/env/learning_sanity.yaml',episodes=2,seed=43210,device='cpu',
        output=tmp_path/'critic_audit.json')
    assert audit['episodes']==2
    assert audit['valid_rover_timesteps']>0
    assert np.isfinite(audit['critic_rmse'])
    assert np.isfinite(audit['critic_prediction_std'])
    assert (tmp_path/'critic_audit.json').exists()
