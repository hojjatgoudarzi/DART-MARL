"""Non-invasive held-out Monte Carlo diagnostic of a saved centralized critic.

This does not retrain the critic or establish on-policy explained variance.
Deterministic evaluation differs from the stochastic PPO rollout distribution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dart_marl.algorithms.mappo.roles import assign_initial_roles
from dart_marl.algorithms.mappo.state import critic_input
from dart_marl.envs.config import load_config
from dart_marl.envs.rover import RoverEnv
from dart_marl.roles.evaluation import _load_actor


def audit_checkpoint(checkpoint: str | Path, env_config: str | Path,
                     training_env_config: str | Path, *, episodes: int = 8,
                     seed: int = 770000, device: str = 'cpu',
                     output: str | Path | None = None) -> dict:
    import torch
    from dart_marl.algorithms.mappo.critic import build_critic
    if episodes < 2:
        raise ValueError('audit requires at least two held-out episodes')
    cfg = load_config(env_config)
    actor, dev, _ = _load_actor(checkpoint, training_env_config, cfg, device)
    # Trusted local MAPPO checkpoint with NumPy snapshots; never load third-party
    # checkpoints with weights_only=False.
    payload = torch.load(checkpoint, map_location=dev, weights_only=False)
    train_cfg = payload['train_config']
    gamma = float(train_cfg['ppo']['gamma'])
    if float(train_cfg.get('shaping_coef',0.)) != 0.:
        raise ValueError('raw-return audit currently requires shaping_coef == 0')
    critic = build_critic(architecture=train_cfg.get('critic_arch', 'agent_v1'),
                          hidden=train_cfg['hidden_size'],
                          return_scale=train_cfg['ppo'].get('value_target_scale',1.)).to(dev)
    critic.load_state_dict(payload['critic'])
    critic.eval(); actor.eval()
    estimates, targets, episode_ids = [], [], []
    episode_returns = []
    for ep in range(episodes):
        env = RoverEnv(cfg)
        env.reset(seed=seed+ep*1009)
        assign_initial_roles(env, train_cfg['role_mode'])
        forecast, rewards, rosters = [], [], []
        while not env.is_done:
            state = critic_input(env)
            with torch.no_grad():
                args = [torch.as_tensor(a[None],device=dev) for a in
                        (state.agents,state.pois,state.active,state.task_mask)]
                estimate = critic(*args).squeeze(0).detach().cpu().numpy()
                obs = torch.as_tensor(env.observe()[None],dtype=torch.float32,device=dev)
                actions,_,_ = actor.act(obs,deterministic=True)
            forecast.append(estimate)
            rosters.append(state.active.copy())
            acts=actions.squeeze(0).cpu().numpy()
            acts[~env.active]=0.
            _,reward,terminated,truncated,_ = env.step(acts)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        episode_returns.append(float(np.sum(rewards)))
        mc = 0.; discounted = []
        for r in reversed(rewards):
            mc = float(r)+gamma*mc
            discounted.append(mc)
        discounted = discounted[::-1]
        for t in range(len(rewards)):
            alive = np.asarray(rosters[t],dtype=bool)
            # Report per-rover shared reward target only for physically alive agents.
            estimates.extend(np.asarray(forecast[t])[alive].tolist())
            targets.extend([discounted[t]]*int(alive.sum()))
            episode_ids.extend([ep]*int(alive.sum()))
    y=np.asarray(targets,dtype=float)
    pred=np.asarray(estimates,dtype=float)
    variance=float(np.var(y))
    error=y-pred
    by_episode=[]
    for ep in range(episodes):
        mask=np.asarray(episode_ids)==ep
        by_episode.append({'seed':seed+ep*1009,'episode_return':episode_returns[ep],
                           'rmse':float(np.sqrt(np.mean(error[mask]**2))),
                           'prediction_mean':float(np.mean(pred[mask])),
                           'mc_target_mean':float(np.mean(y[mask]))})
    report={
        'format':'dart-marl-critic-heldout-audit-v1',
        'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'environment':str(env_config),'episodes':episodes,'seed':seed,
        'device':str(dev),'gamma':gamma,'valid_rover_timesteps':len(y),
        'critic_arch':train_cfg.get('critic_arch','agent_v1'),
        'value_target_scale':train_cfg['ppo'].get('value_target_scale',1.),
        'mc_target_mean':float(np.mean(y)),'mc_target_std':float(np.std(y)),
        'critic_prediction_mean':float(np.mean(pred)),
        'critic_prediction_std':float(np.std(pred)),
        'critic_bias':float(np.mean(pred-y)),
        'critic_rmse':float(np.sqrt(np.mean(error**2))),
        'constant_mean_rmse':float(np.sqrt(np.mean((y-np.mean(y))**2))),
        'zero_rmse':float(np.sqrt(np.mean(y**2))),
        'critic_explained_variance':float(1.-np.var(error)/variance) if variance>1e-10 else None,
        'per_episode':by_episode,
        'caveat':'Deterministic held-out trajectories, shared team reward duplicated for active rovers; '
                 'not PPO on-policy target explained variance. This audit does NOT correct the critic.'
    }
    if output is not None:
        target=Path(output)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(report,indent=2)+'\n')
    return report
