"""Supervised CFT with episode-grouped train/val/test and held-out selection.

No inference or tuning on the held-out test split during early stopping.
Torch state-dict checkpoint is stored with only weights and primitive metadata,
so inference can use torch.load(weights_only=True).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dart_marl.cft.features import SCHEMA


def train_predictor(dataset_dir: str | Path, output: str | Path, *,
                    seed: int = 123, epochs: int = 120, batch_size: int = 128,
                    learning_rate: float = .0008, hidden: int = 128,
                    patience: int = 20, device: str = 'cpu') -> dict:
    import torch
    from dart_marl.cft.model import ResidualMLP
    torch.manual_seed(seed)
    data_dir = Path(dataset_dir); output = Path(output)
    if output.exists():
        raise FileExistsError(f'CFT model destination already exists: {output}')
    if min(epochs, batch_size, hidden, patience) < 1 or learning_rate <= 0:
        raise ValueError('invalid training hyperparameters')
    meta = json.loads((data_dir / 'manifest.json').read_text())
    if meta['format'] != 'dart-cft-dataset-v1' or meta['feature_schema'] != SCHEMA:
        raise ValueError('unknown dataset schema')
    with np.load(data_dir / 'counterfactuals.npz', allow_pickle=False) as data:
        x = data['x'].astype(np.float32)
        labels = data['target'].astype(np.float32)
        analytic = data['analytic'].astype(np.float32)
        potential = data['potential'].astype(np.float32)
        penalty = (data['decision_penalty'].astype(np.float32) if
                   'decision_penalty' in data.files else np.zeros_like(labels))
        snapshots = data['snapshot_id'].copy()
        split = data['split'].copy()
        ids = data['episode_id'].copy()
    if not (np.isfinite(x).all() and np.isfinite(labels).all() and
            np.isfinite(analytic).all() and np.isfinite(potential).all()):
        raise ValueError('dataset contains non-finite examples')
    groups = {s:set(ids[split == s].tolist()) for s in ('train','val','test')}
    if any(not v for v in groups.values()) or any(groups[a] & groups[b]
            for a,b in (('train','val'), ('train','test'), ('val','test'))):
        raise ValueError('train/validation/test must have disjoint episode groups')
    if x.ndim != 2 or len(x) != len(labels) or len(x) != len(analytic):
        raise ValueError('bad feature/label shapes')
    normalized_x_mean = x[split == 'train'].mean(axis=0)
    # Input features are ALREADY physically normalized. A feature that is
    # constant in training but nonzero in validation must not be magnified
    # 10,000-fold by near-zero empirical variance. Use an explicit floor.
    normalized_x_std = np.maximum(x[split == 'train'].std(axis=0), 0.25)
    residual = labels - analytic
    mean = float(residual[split == 'train'].mean())
    std = float(max(residual[split == 'train'].std(), 1e-4))
    xn = (x - normalized_x_mean) / normalized_x_std
    yn = (residual - mean) / std
    tensor_x = torch.from_numpy(xn.astype(np.float32))
    tensor_y = torch.from_numpy(yn.astype(np.float32))
    tr = np.flatnonzero(split == 'train'); va = np.flatnonzero(split == 'val')
    model = ResidualMLP(x.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = None; best_val = float('inf'); best_epoch = 0; wait = 0
    history = []
    generator = torch.Generator().manual_seed(seed + 71)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), generator=generator)
        batch_losses = []
        for indices in perm.split(batch_size):
            ix = torch.as_tensor(tr[indices.numpy()], dtype=torch.long)
            xx = tensor_x[ix].to(device)
            yy = tensor_y[ix].to(device)
            loss = torch.nn.functional.smooth_l1_loss(model(xx), yy, beta=1.)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            vx = tensor_x[va].to(device)
            prediction = np.clip(model(vx).cpu().numpy() * std + mean + analytic[va],
                                 0., potential[va])
            validation_mse = float(np.mean((prediction - labels[va]) ** 2))
        history.append({'epoch':epoch, 'train_smooth_l1':float(np.mean(batch_losses)),
                        'val_raw_return_rmse':float(np.sqrt(validation_mse))})
        if validation_mse < best_val - 1e-8:
            best_val = validation_mse
            best_epoch = epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    metrics = {}
    with torch.no_grad():
        for s in ('train','val','test'):
            ix = np.flatnonzero(split == s)
            yhat = np.clip(model(tensor_x[ix].to(device)).cpu().numpy() * std + mean + analytic[ix],
                           0., potential[ix])
            y = labels[ix]
            # All candidates from same episode are correlated. These descriptive
            # metrics do not pretend each row is an independent experiment.
            metrics[s] = {
                'rows':len(ix), 'episodes':len(groups[s]),
                'rmse':float(np.sqrt(np.mean((yhat-y)**2))),
                'mae':float(np.mean(np.abs(yhat-y))),
                'analytic_rmse':float(np.sqrt(np.mean((analytic[ix]-y)**2))),
                'constant_train_mean_rmse':float(np.sqrt(np.mean(
                    (np.mean(labels[split=='train']) - y)**2))),
                'prediction_bias':float(np.mean(yhat-y)),
            }
            # Evaluate the actual choice quality within each saved snapshot's
            # paired counterfactual candidate set, not only pointwise RMSE.
            regrets_learned=[]; regrets_analytic=[]
            choices_better_than_analytic=0
            candidate_groups=0
            for snapshot in np.unique(snapshots[ix]):
                local = np.flatnonzero(snapshots[ix] == snapshot)
                if len(local) < 2:
                    continue
                ground_truth = y[local] + penalty[ix][local]
                model_choice = int(np.argmax(yhat[local] + penalty[ix][local]))
                analytic_choice = int(np.argmax(analytic[ix][local] + penalty[ix][local]))
                best = float(np.max(ground_truth))
                learned_regret = best - float(ground_truth[model_choice])
                analytic_regret = best - float(ground_truth[analytic_choice])
                regrets_learned.append(learned_regret)
                regrets_analytic.append(analytic_regret)
                choices_better_than_analytic += learned_regret < analytic_regret - 1e-6
                candidate_groups += 1
            metrics[s].update({
                'paired_snapshot_groups':candidate_groups,
                'candidate_selection_mean_regret':float(np.mean(regrets_learned)) if candidate_groups else None,
                'analytic_candidate_selection_mean_regret':float(np.mean(regrets_analytic)) if candidate_groups else None,
                'fraction_better_than_analytic':(choices_better_than_analytic / candidate_groups
                                                 if candidate_groups else None),
            })
    output.mkdir(parents=True)
    payload = {'format':'dart-cft-residual-v1', 'schema':SCHEMA,
               'task_problem_hash':meta['task_problem_hash'],
               'controller_checkpoint_sha256':meta['controller_checkpoint_sha256'],
               'horizon':int(meta['horizon']), 'discount':float(meta['discount']),
               'features':int(x.shape[1]), 'hidden':hidden,
               'x_mean':normalized_x_mean.tolist(),
               'x_std':normalized_x_std.tolist(),
               'residual_mean':mean, 'residual_std':std,
               'weights':best_state,
               'best_epoch':best_epoch, 'validation_rmse':metrics['val']['rmse']}
    torch.save(payload, output / 'cft_model.pt')
    report = {'format':'dart-cft-training-v1', 'dataset':str(data_dir),
              'seed':seed, 'epochs_run':len(history), 'best_epoch':best_epoch,
              'metrics':metrics, 'task_problem_hash':meta['task_problem_hash'],
              'controller_checkpoint_sha256':meta['controller_checkpoint_sha256'],
              'decision_score':'target + decision_penalty when available (older datasets use 0)',
              'warning':'Test split is descriptive and only inspected after model selection. '
                        'Not multiple independent training controllers; do not claim publication significance.'}
    (output / 'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output / 'training_history.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in history))
    return {'model':str(output/'cft_model.pt'), 'best_epoch':best_epoch,
            'metrics':metrics, 'report':str(output/'training_report.json')}


class CFTEstimator:
    def __init__(self, path: str | Path, *, expected_config=None,
                 expected_controller_sha: str | None = None,
                 horizon: int | None = None, discount: float | None = None, strict_controller: bool = False):
        import torch
        from dart_marl.cft.model import ResidualMLP
        from dart_marl.cft.features import task_problem_hash
        # weights_only=True: no arbitrary data/code objects needed by this model.
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format') not in ('dart-cft-residual-v1', 'dart-cft-rank-v2') or payload.get('schema') != SCHEMA:
            raise ValueError('unsupported CFT model format/schema')
        if expected_config is not None and payload['task_problem_hash'] != task_problem_hash(expected_config):
            raise ValueError('CFT model was trained on a different task problem')
        if (strict_controller or expected_controller_sha is not None) and (
            payload['controller_checkpoint_sha256'] != expected_controller_sha):
            raise ValueError('CFT model belongs to a different frozen MAPPO controller')
        if horizon is not None and horizon != payload['horizon']:
            raise ValueError('CFT horizon differs from evaluation horizon')
        if discount is not None and abs(discount - payload['discount']) > 1e-12:
            raise ValueError('CFT discount differs from evaluation discount')
        self.schema = payload['schema']
        self.format = payload['format']
        self.blend_alpha = float(payload.get('blend_alpha', 1.))
        self.horizon = payload['horizon']
        self.discount = payload['discount']
        self.controller_sha = payload['controller_checkpoint_sha256']
        self.problem_hash = payload['task_problem_hash']
        self.net = ResidualMLP(payload['features'], payload['hidden'])
        self.net.load_state_dict(payload['weights'])
        self.net.eval()
        self.mean = torch.as_tensor(payload['x_mean'], dtype=torch.float32)
        self.std = torch.as_tensor(payload['x_std'], dtype=torch.float32)
        self.y_mean = payload['residual_mean']
        self.y_std = payload['residual_std']

    def residuals(self, features: np.ndarray) -> np.ndarray:
        import torch
        batch = torch.as_tensor(features, dtype=torch.float32)
        with torch.no_grad():
            y = self.net((batch-self.mean)/self.std).cpu().numpy()
        return y * self.y_std + self.y_mean
