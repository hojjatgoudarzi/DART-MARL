"""legacy_study grouped decision-focused counterfactual learning.

Regression is a separate legacy_study baseline. This trainer uses a hybrid
Huber + within-snapshot soft pairwise ranking loss, and selects/checkpoints
only from validation episodes. Evaluation test episodes are never used for
training, early stopping or mixture selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dart_marl.cft.decision import (
    decision_metrics,
    select_validation_blend,
    snapshot_index_groups,
)
from dart_marl.cft.features import SCHEMA


def pairwise_decision_loss(predicted_score, target_score, *, temperature: float = 5.0):
    """Soft binary pairwise likelihood for one shared-snapshot candidate set."""
    import torch
    import torch.nn.functional as F
    if predicted_score.numel() < 2:
        return predicted_score.sum() * 0.
    i, j = torch.triu_indices(predicted_score.numel(), predicted_score.numel(), 1,
                              device=predicted_score.device)
    gap = target_score[i] - target_score[j]
    informative = gap.abs() > 0.05
    if not informative.any():
        return predicted_score.sum() * 0.
    probability = torch.sigmoid(gap[informative] / temperature)
    logits = (predicted_score[i[informative]] - predicted_score[j[informative]]) / temperature
    return F.binary_cross_entropy_with_logits(logits, probability)


def train_rank_predictor(dataset_dir: str | Path, output: str | Path, *,
                         seed: int = 123, epochs: int = 120, batch_size: int = 128,
                         learning_rate: float = .0008, hidden: int = 128,
                         patience: int = 20, device: str = 'cpu',
                         rank_weight: float = .65, rank_temperature: float = 5.) -> dict:
    import torch
    import torch.nn.functional as F
    from dart_marl.cft.model import ResidualMLP
    if not (0 <= rank_weight <= 1 and rank_temperature > 0):
        raise ValueError('invalid ranking loss configuration')
    if min(epochs, batch_size, hidden, patience) < 1 or learning_rate <= 0:
        raise ValueError('invalid training hyperparameters')
    output = Path(output); data_dir = Path(dataset_dir)
    if output.exists():
        raise FileExistsError(f'CFT model destination exists: {output}')
    meta = json.loads((data_dir / 'manifest.json').read_text())
    if meta.get('format') != 'dart-cft-dataset-v1' or meta.get('feature_schema') != SCHEMA:
        raise ValueError('incompatible CFT dataset')
    with np.load(data_dir / 'counterfactuals.npz', allow_pickle=False) as z:
        required = {'x','target','analytic','potential','decision_penalty',
                    'snapshot_id','episode_id','split'}
        if not required.issubset(set(z.files)):
            raise ValueError('ranking objective requires legacy_study data with decision_penalty')
        x = z['x'].astype(np.float32)
        y = z['target'].astype(np.float32)
        analytic = z['analytic'].astype(np.float32)
        potential = z['potential'].astype(np.float32)
        penalty = z['decision_penalty'].astype(np.float32)
        snapshots = z['snapshot_id'].copy()
        episodes = z['episode_id'].copy()
        split = z['split'].copy()
    if x.ndim != 2 or not all(len(v) == len(x) for v in
                               (y, analytic, potential, penalty, snapshots, episodes, split)):
        raise ValueError('CFT shapes incompatible')
    if not all(np.isfinite(v).all() for v in (x,y,analytic,potential,penalty)):
        raise ValueError('nonfinite CFT training data')
    groups = {s:set(episodes[split == s].tolist()) for s in ('train','val','test')}
    if any(not a for a in groups.values()) or any(groups[a] & groups[b] for a,b in
        (('train','val'),('train','test'),('val','test'))):
        raise ValueError('split episodes must be disjoint')
    if any(len(set(episodes[g].tolist())) != 1 for g in
           snapshot_index_groups(snapshots,np.arange(len(x)))):
        raise ValueError('snapshot crosses episode boundaries')
    indices = {s:np.flatnonzero(split == s) for s in ('train','val','test')}
    train_groups = [g for g in snapshot_index_groups(snapshots, indices['train']) if len(g)>=2]
    val_groups = [g for g in snapshot_index_groups(snapshots, indices['val']) if len(g)>=2]
    if not train_groups or not val_groups:
        raise ValueError('ranking needs paired train and validation snapshots')
    torch.manual_seed(seed)
    mean_x = x[indices['train']].mean(axis=0)
    std_x = np.maximum(x[indices['train']].std(axis=0), .25)
    res = y - analytic
    mean_y = float(np.mean(res[indices['train']]))
    std_y = float(max(np.std(res[indices['train']]), 1e-4))
    tx = torch.as_tensor((x-mean_x)/std_x, dtype=torch.float32, device=device)
    ty = torch.as_tensor((res-mean_y)/std_y, dtype=torch.float32, device=device)
    ta = torch.as_tensor(analytic, dtype=torch.float32, device=device)
    tp = torch.as_tensor(penalty, dtype=torch.float32, device=device)
    truth = torch.as_tensor(y, dtype=torch.float32, device=device)
    model = ResidualMLP(x.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed+71)
    best = (float('inf'),float('inf'))
    best_epoch = 0; best_alpha = 0.; best_weights = None; bad = 0
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        epoch_losses = []
        randomized = rng.permutation(len(train_groups))
        # Batch complete counterfactual groups; no snapshot is split across batches.
        for start in range(0,len(randomized),max(1,batch_size//6)):
            these = [train_groups[j] for j in randomized[start:start+max(1,batch_size//6)]]
            all_indices = np.concatenate(these)
            pidx = torch.as_tensor(all_indices, dtype=torch.long, device=device)
            prediction_residual = model(tx[pidx])
            huber = F.smooth_l1_loss(prediction_residual, ty[pidx], beta=1.)
            predicted_objective = ta[pidx] + mean_y + prediction_residual*std_y + tp[pidx]
            true_objective = truth[pidx] + tp[pidx]
            offset = 0; rank_losses = []
            for g in these:
                end = offset+len(g)
                rank_losses.append(pairwise_decision_loss(
                    predicted_objective[offset:end], true_objective[offset:end],
                    temperature=rank_temperature))
                offset=end
            ranking = torch.stack(rank_losses).mean()
            loss = (1.-rank_weight)*huber + rank_weight*ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            ix = indices['val']
            vz = torch.as_tensor(ix,dtype=torch.long,device=device)
            val_prediction = np.clip((model(tx[vz]).cpu().numpy()*std_y +mean_y +analytic[ix]),
                                      0.,potential[ix])
        val_rmse = float(np.sqrt(np.mean((val_prediction-y[ix])**2)))
        val_predictions_full = analytic.copy()
        val_predictions_full[ix] = val_prediction
        alpha, val_regret = select_validation_blend(y, analytic, val_predictions_full,
                                                     penalty, snapshots, ix)
        # Validation episodes alone select the checkpoint and analytic mixture.
        history.append({'epoch':epoch,'train_hybrid_loss':float(np.mean(epoch_losses)),
                        'val_raw_return_rmse':val_rmse,
                        'val_decision_regret':val_regret,'val_blend':alpha})
        criterion = (val_regret,val_rmse)
        if criterion[0] < best[0]-1e-7 or (abs(criterion[0]-best[0])<1e-7 and
                                            criterion[1]<best[1]-1e-6):
            best=criterion; best_epoch=epoch; best_alpha=alpha; bad=0
            best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience:
                break
    assert best_weights is not None
    model.load_state_dict(best_weights)
    model.eval()
    full_predictions = np.empty_like(y)
    with torch.no_grad():
        for fold,ix in indices.items():
            ids=torch.as_tensor(ix,dtype=torch.long,device=device)
            full_predictions[ix] = np.clip(model(tx[ids]).cpu().numpy()*std_y+mean_y+
                                           analytic[ix],0.,potential[ix])
    metrics = {}
    for fold, ix in indices.items():
        predicted=full_predictions[ix]
        metrics[fold]={
            'rows':int(len(ix)), 'episodes':len(groups[fold]),
            'rmse':float(np.sqrt(np.mean((predicted-y[ix])**2))),
            'mae':float(np.mean(np.abs(predicted-y[ix]))),
            'analytic_rmse':float(np.sqrt(np.mean((analytic[ix]-y[ix])**2))),
            'prediction_bias':float(np.mean(predicted-y[ix])),
        }
        metrics[fold].update(decision_metrics(y,analytic,full_predictions,penalty,
                                             snapshots,ix,alpha=best_alpha))
        metrics[fold]['unblended_decision_regret'] = decision_metrics(
            y,analytic,full_predictions,penalty,snapshots,ix,alpha=1.)[
            'candidate_selection_mean_regret']
    output.mkdir(parents=True)
    payload={'format':'dart-cft-rank-v2','schema':SCHEMA,
             'task_problem_hash':meta['task_problem_hash'],
             'controller_checkpoint_sha256':meta['controller_checkpoint_sha256'],
             'horizon':int(meta['horizon']),'discount':float(meta['discount']),
             'features':int(x.shape[1]),'hidden':hidden,'x_mean':mean_x.tolist(),
             'x_std':std_x.tolist(),'residual_mean':mean_y,'residual_std':std_y,
             'weights':best_weights,'best_epoch':best_epoch,
             'blend_alpha':best_alpha,'ranking_weight':rank_weight,
             'rank_temperature':rank_temperature,
             'validation_decision_regret':metrics['val']['candidate_selection_mean_regret']}
    torch.save(payload,output/'cft_model.pt')
    report={'format':'dart-cft-rank-training-v2', 'dataset':str(data_dir),
            'seed':seed,'epochs_run':len(history),'best_epoch':best_epoch,
            'selected_val_blend_alpha':best_alpha,'rank_weight':rank_weight,
            'rank_temperature':rank_temperature,'metrics':metrics,
            'task_problem_hash':meta['task_problem_hash'],
            'controller_checkpoint_sha256':meta['controller_checkpoint_sha256'],
            'warning':'All training and calibration selected on validation episodes only. '
                      'Test metrics are descriptive and NOT an independent tuning source. '
                      'Each snapshot is one paired decision, not independent candidates. '
                      'Blending alpha=0 means analytic fallback and NO learned advantage.'}
    (output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'training_history.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in history))
    return {'model':str(output/'cft_model.pt'),'best_epoch':best_epoch,
            'selected_val_blend_alpha':best_alpha,'metrics':metrics,
            'report':str(output/'training_report.json')}
