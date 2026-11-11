"""Counterfactual branches from cloned real rover states.

Training labels are H-step returns of a *fixed role commitment* executed by the
frozen controller. This is a well-defined surrogate, NOT the deployed policy's
closed-loop return after additional reallocations. All candidates from the
same snapshot receive identical future failure tape and simulator RNG state.
No future information enters the feature vector.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from dart_marl.algorithms.mappo.diagnostics import role_following_actions
from dart_marl.disruptions.events import DisruptionEvent
from dart_marl.disruptions.sampling import diverse_scenario
from dart_marl.envs.config import EnvironmentConfig, load_config
from dart_marl.envs.rover import RoverEnv
from dart_marl.roles.allocator import AllocationConfig
from dart_marl.roles.coordinator import RoleCoordinator
from dart_marl.roles.evaluation import _load_actor
from dart_marl.cft.candidates import assignment_candidates
from dart_marl.cft.features import SCHEMA, task_problem_hash, visible_features


def sampled_scenario(base: EnvironmentConfig, seed: int) -> EnvironmentConfig:
    """Independent randomized hidden fault tapes, reproducible from episode seed.

    Nonoverlapping initial rover targets avoid invalid repeated physical events.
    The last event may be absent when there are not enough rover slots.
    """
    rng = np.random.default_rng(seed ^ 0x5A17A9)
    count = len(base.rovers)
    if count < 2 or base.horizon < 8:
        raise ValueError('sampled faults require >=2 rovers and horizon >=8')
    slots = rng.permutation(count)
    first = int(slots[0]); second = int(slots[1])
    t1 = int(rng.integers(max(1, base.horizon // 6), max(2, base.horizon // 3)))
    t2 = int(rng.integers(max(t1+1, base.horizon // 3), max(t1+2, base.horizon*2//3)))
    kind = ('permanent_failure', 'temporary_disable', 'replacement_agent')[seed % 3]
    if kind == 'replacement_agent' and count == base.max_agents:
        kind = 'permanent_failure'
    kw = {'duration': min(5, base.horizon - t1 - 1)} if kind == 'temporary_disable' else (
        {'new_slot': count, 'spawn': tuple(base.rovers[first].start)}
        if kind == 'replacement_agent' else {})
    events = [DisruptionEvent(step=t1, rover=first, kind=kind,
                              detection_delay=int(rng.integers(1, 4)), **kw)]
    if t2 < base.horizon - 1:
        events.append(DisruptionEvent(step=t2, rover=second, kind='temporary_disable',
                                      duration=min(4, base.horizon - t2 - 1),
                                      detection_delay=int(rng.integers(1, 3))))
    return replace(base, name=f'cft_scenario_{seed}', events=tuple(events))



def frozen_actions(env: RoverEnv, actor=None, device=None) -> np.ndarray:
    if actor is None:
        return role_following_actions(env)
    import torch
    with torch.no_grad():
        obs = torch.as_tensor(env.observe()[None], dtype=torch.float32, device=device)
        actions, _, _ = actor.act(obs, deterministic=True)
    action = actions.squeeze(0).cpu().numpy()
    action[~env.active] = 0.
    return action


def branch_return(env: RoverEnv, proposed: np.ndarray, *, actor=None, device=None,
                  horizon: int, discount: float) -> float:
    """Clone full physical state; commit proposed roles for at most H steps.

    True physical active flags affect only actuator delivery and the environment
    rewards. The planning candidates and neural input cannot see them.
    """
    branch = env.clone()
    physical_roles = np.asarray(proposed, dtype=np.int32).copy()
    physical_roles[~branch.active] = -1
    branch.set_roles(physical_roles)
    discounted = 0.
    for k in range(min(horizon, branch.config.horizon - branch.step_index)):
        action = frozen_actions(branch, actor, device)
        _, r, terminated, truncated, _ = branch.step(action)
        discounted += discount ** k * float(r)
        if terminated or truncated:
            break
    return discounted


def generate_data(base_config: str | Path, output: str | Path, *,
                  controller: str = 'checkpoint', checkpoint: str | Path | None = None,
                  training_env_config: str | Path | None = None,
                  device: str = 'cpu', episodes_train: int = 12,
                  episodes_val: int = 4, episodes_test: int = 4,
                  seed: int = 510000, event_mode: str = 'sampled',
                  horizon: int = 24, discount: float = .98,
                  sample_period: int = 8, candidates: int = 6,
                  beam: int = 200) -> dict:
    if any(k < 1 for k in (episodes_train, episodes_val, episodes_test,
                            horizon, sample_period, candidates, beam)) or candidates > beam:
        raise ValueError('episode counts and sample configuration must be positive')
    if event_mode not in ('sampled', 'configured', 'diverse'):
        raise ValueError('event_mode must be sampled, diverse or configured')
    if not 0 < discount <= 1:
        raise ValueError('discount must be in (0,1]')
    if controller not in ('checkpoint', 'scripted'):
        raise ValueError('unknown controller')
    base = load_config(base_config)
    if controller == 'checkpoint':
        if checkpoint is None or training_env_config is None:
            raise ValueError('checkpoint controller requires checkpoint and training_env_config')
        actor, dev, provenance = _load_actor(checkpoint, training_env_config, base, device)
        ck_hash = provenance['checkpoint_sha256']
    else:
        if checkpoint is not None or training_env_config is not None:
            raise ValueError('scripted controller takes no model checkpoint')
        actor = dev = None
        ck_hash = None
        provenance = {'controller_type':'scripted_privileged_diagnostic_NOT_MAPPO'}
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'dataset destination exists: {output}')
    features, labels, analytical, potentials, penalties, episode_ids, snapshot_ids, role_vectors, splits = [], [], [], [], [], [], [], [], []
    episode_protocols = []
    total = episodes_train + episodes_val + episodes_test
    for episode in range(total):
        # Distinct seeds across splits, never reusing an episode in val/test.
        ep_seed = int(seed + episode * 7919)
        cfg = (sampled_scenario(base, ep_seed) if event_mode == 'sampled' else
               diverse_scenario(base, ep_seed) if event_mode == 'diverse' else base)
        split = ('train' if episode < episodes_train else
                 'val' if episode < episodes_train + episodes_val else 'test')
        protocol = {'seed':ep_seed, 'split':split, 'events': [asdict(e) for e in cfg.events]}
        protocol['fingerprint'] = hashlib.sha256(
            json.dumps(protocol['events'], sort_keys=True).encode()).hexdigest()
        episode_protocols.append(protocol)
        env = RoverEnv(cfg)
        env.reset(seed=ep_seed)
        coordinator = RoleCoordinator(AllocationConfig('r_cart', forecast_horizon=horizon,
                                                       forecast_discount=discount))
        coordinator.reset(env)
        old_roster = coordinator.last_roster.copy()
        while not env.is_done:
            t = env.step_index
            # Regular fixed-grid sample decisions, and every observed roster change.
            roster = env.reported_active().copy()
            if t % sample_period == 0 or not np.array_equal(old_roster, roster):
                prev = coordinator.previous_commanded.copy()
                options = assignment_candidates(
                    env, roster, prev, horizon=horizon, discount=discount,
                    k=candidates, beam=beam)
                snapshot_id = f'{ep_seed}:{t}'
                for roles in options:
                    x = visible_features(env, roster, prev, roles,
                                         horizon=horizon, discount=discount)
                    # Target is discounted RAW team return of actual frozen policy.
                    y = branch_return(env, roles, actor=actor, device=dev,
                                      horizon=horizon, discount=discount)
                    from dart_marl.roles.recovery import score_recovery_assignment
                    score = score_recovery_assignment(
                        env, roster, prev, roles, horizon_steps=horizon,
                        discount=discount, travel_weight=.1,
                        switch_weight=3., sync_weight=.75)
                    assert score.feasible
                    features.append(x)
                    labels.append(y)
                    analytical.append(score.projected_return)
                    potentials.append(score.potential_return)
                    penalties.append(score.objective - score.projected_return)
                    episode_ids.append(ep_seed)
                    snapshot_ids.append(snapshot_id)
                    role_vectors.append(roles.copy())
                    splits.append(split)
            if env.is_done:
                break
            action = frozen_actions(env, actor, dev)
            _, _, terminated, truncated, _ = env.step(action)
            if not (terminated or truncated):
                coordinator.maybe_reassign(env)
            old_roster = roster
    if not features:
        raise RuntimeError('no CFT examples were generated')
    output.mkdir(parents=True)
    np.savez_compressed(output / 'counterfactuals.npz',
                        x=np.stack(features).astype(np.float32),
                        target=np.asarray(labels, dtype=np.float32),
                        analytic=np.asarray(analytical, dtype=np.float32),
                        potential=np.asarray(potentials, dtype=np.float32),
                        decision_penalty=np.asarray(penalties, dtype=np.float32),
                        episode_id=np.asarray(episode_ids, dtype=np.int64),
                        snapshot_id=np.asarray(snapshot_ids, dtype='U32'),
                        roles=np.stack(role_vectors).astype(np.int32),
                        split=np.asarray(splits, dtype='U5'))
    manifest = {
        'format': 'dart-cft-dataset-v1', 'feature_schema': SCHEMA,
        'task_problem_hash': task_problem_hash(base),
        'base_environment': str(base_config), 'controller': provenance,
        'controller_checkpoint_sha256': ck_hash,
        'continuation': 'frozen_actor_fixed_role_commitment_H_steps',
        'target': 'discounted_raw_team_return_not_recovery_time',
        'decision_score': 'target + decision_penalty (analytic travel/switch/sync cost)',
        'horizon': horizon, 'discount': discount,
        'event_mode': event_mode, 'sample_period': sample_period,
        'candidate_limit': candidates, 'beam': beam,
        'episodes': {'train':episodes_train,'val':episodes_val,'test':episodes_test},
        'rows': {s:int(sum(a == s for a in splits)) for s in ('train','val','test')},
        'episode_protocols':episode_protocols,
        'caveat': 'A conditional frozen-role H-step surrogate, not deployed closed-loop Q. '
                  'Splits are by independent episode seeds; all branches share a snapshot. '
                  'No future fault schedule or true physical roster in input. '
                  'Configured event mode shares a fixed fault tape across seeds.'
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return {'out':str(output), 'rows':manifest['rows'], 'controller':provenance,
            'data_file':str(output / 'counterfactuals.npz')}
