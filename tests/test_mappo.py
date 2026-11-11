"""Checks MAPPO math, fault masking, training and checkpoint persistence on CPU."""

# ruff: noqa: E402  (optional torch skip precedes training-only imports)

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dart_marl.algorithms.mappo.actor import SharedActor
from dart_marl.algorithms.mappo.buffer import compute_gae
from dart_marl.algorithms.mappo.critic import CentralizedCritic
from dart_marl.algorithms.mappo.ppo import PPOConfig, PPOLearner
from dart_marl.algorithms.mappo.roles import assign_initial_roles
from dart_marl.algorithms.mappo.state import critic_input
from dart_marl.algorithms.mappo.trainer import (Collector, TrainConfig, evaluate_checkpoint,
                                                load_train_config, train)
from dart_marl.envs import RoverEnv, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_actor_action_range_and_logprob_consistency():
    torch.manual_seed(13)
    a = SharedActor(23, hidden=32)
    obs = torch.randn(7, 12, 23)
    action, latent, old = a.act(obs)
    logp, entropy = a.evaluate(obs, latent)
    assert action.shape == (7, 12, 2)
    assert torch.all(action.abs() <= 1)
    torch.testing.assert_close(logp, old)
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
    assert sum(p.numel() for p in a.parameters()) > 0
    # Gradient propagates through corrected log-prob even with tanh saturation.
    (logp.mean() - 0.01 * entropy.mean()).backward()
    assert a.mean.weight.grad is not None and torch.isfinite(a.mean.weight.grad).all()


def test_large_latent_logprob_finite():
    actor = SharedActor(23, hidden=16)
    logp, _ = actor.evaluate(torch.zeros((2, 23)), torch.tensor([[50.0, -50.0], [0, 0]]))
    assert torch.isfinite(logp).all()


def test_critic_masks_dead_and_padded_rovers():
    torch.manual_seed(2)
    critic = CentralizedCritic(hidden=32)
    agent = torch.randn(2, 12, 9)
    task = torch.randn(2, 6, 10)
    alive = torch.zeros(2, 12, dtype=torch.bool)
    alive[:, :4] = True
    p_mask = torch.zeros(2, 6, dtype=torch.bool)
    p_mask[:, :2] = True
    val = critic(agent, task, alive, p_mask)
    mutated_agent = agent.clone()
    mutated_agent[:, 4:] = 10000
    mutated_pois = task.clone()
    mutated_pois[:, 2:] = -10000
    val2 = critic(mutated_agent, mutated_pois, alive, p_mask)
    torch.testing.assert_close(val, val2, atol=1e-5, rtol=1e-5)
    assert torch.all(val[:, 4:] == 0)


def test_gae_time_limit_bootstrap_without_episode_leakage():
    shape = (3, 1, 1)
    reward = np.ones(shape, dtype=np.float32)
    values = np.zeros(shape, dtype=np.float32)
    future = np.full(shape, 10, dtype=np.float32)
    masks = np.ones(shape, dtype=bool)
    done = np.array([[False], [True], [False]])
    terminated = np.array([[False], [False], [False]])
    advantage, target = compute_gae(reward, values, future, masks, masks, done, terminated,
                                     gamma=0.9, gae_lambda=1.0)
    assert advantage[1, 0, 0] == pytest.approx(10.0)
    assert advantage[0, 0, 0] == pytest.approx(19.0)  # delta_t0=10, plus .9 * A_t1=9
    assert advantage[2, 0, 0] == pytest.approx(10)
    np.testing.assert_allclose(advantage, target)
    terminated[1] = True
    adv2, _ = compute_gae(reward, values, future, masks, masks, done, terminated, 0.9, 1)
    assert adv2[1, 0, 0] == pytest.approx(1.0)


def test_gae_failure_and_reentry_segment_boundary():
    reward = np.ones((4, 1, 1), dtype=np.float32)
    values = np.zeros_like(reward)
    next_val = np.ones_like(reward) * 9
    active = np.array([True, True, False, True]).reshape(4, 1, 1)
    future_active = np.array([True, False, True, True]).reshape(4, 1, 1)
    done = np.zeros((4, 1), dtype=bool)
    adv, _ = compute_gae(reward, values, next_val, active, future_active, done, done,
                         gamma=0.9, gae_lambda=1)
    assert adv[1, 0, 0] == pytest.approx(1.0)  # died at next boundary
    assert adv[0, 0, 0] == pytest.approx(10.0)  # t0 delta includes next V=9
    assert adv[2, 0, 0] == 0.0
    assert adv[3, 0, 0] == pytest.approx(9.1)


def test_state_encoder_and_static_roles():
    env = RoverEnv(load_config(ROOT / "configs/env/dev_roles.yaml"))
    env.reset(seed=55)
    assign_initial_roles(env)
    s = critic_input(env)
    assert s.agents.shape == (12, 9) and s.pois.shape == (6, 10)
    assert s.active.sum() == 6
    assert np.all(s.agents[6:] == 0)
    assert np.all(env.roles[6:] == -1)
    assert sum(env.roles[:6] == 2) >= 2  # most valuable one-shot task
    assigned = env.roles.copy()
    env.step(np.zeros((12, 2)))
    np.testing.assert_array_equal(env.roles, assigned)


def test_masked_ppo_training_parameters_change():
    env_cfg = load_config(ROOT / "configs/env/dev_roles.yaml")
    cfg = TrainConfig(env_config=str(ROOT / "configs/env/dev_roles.yaml"), seed=3,
                      num_envs=2, rollout_steps=8, total_env_steps=16, hidden_size=32,
                      ppo=PPOConfig(ppo_epochs=2, num_minibatches=2))
    torch.manual_seed(3)
    learner = PPOLearner(SharedActor(23, 32), CentralizedCritic(hidden=32),
                         cfg.ppo, torch.device("cpu"))
    collector = Collector(env_cfg, cfg)
    before = copy.deepcopy(learner.actor.state_dict())
    data, stats = collector.collect(learner)
    assert data["obs"].shape == (8, 2, 12, 23)
    assert stats["env_reward_total"] >= 0
    assert int(data["active"].sum()) == 8 * 2 * 6
    outcomes = learner.update(data)
    assert all(np.isfinite(list(outcomes.values())))
    assert any(not torch.equal(before[k], learner.actor.state_dict()[k]) for k in before)


def test_faults_mask_ppo_and_dont_rejoin_through_gae():
    env_cfg = load_config(ROOT / "configs/env/dev_failure.yaml")
    cfg = TrainConfig(env_config=str(ROOT / "configs/env/dev_failure.yaml"),
                      num_envs=1, rollout_steps=45, total_env_steps=45, hidden_size=32)
    learner = PPOLearner(SharedActor(23, 32), CentralizedCritic(hidden=32),
                         cfg.ppo, torch.device("cpu"))
    collector = Collector(env_cfg, cfg)
    data, _ = collector.collect(learner)
    assert data["active"].shape[0] == 45
    assert (~data["active"]).any()
    assert np.all(data["rewards"][~data["active"]] == 0)
    outcomes = learner.update(data)
    assert all(np.isfinite(list(outcomes.values())))


def test_train_save_evaluate_resume(tmp_path):
    from dataclasses import asdict
    import yaml
    original = load_train_config(ROOT / "configs/train/mappo_smoke.yaml")
    base = asdict(original)
    base["env_config"] = str(ROOT / "configs/env/dev_roles.yaml")
    base["total_env_steps"] = 24
    base["rollout_steps"] = 12
    base["num_envs"] = 2
    base["ppo"]["ppo_epochs"] = 1
    base["ppo"]["num_minibatches"] = 1
    cfg = tmp_path / "test.yaml"
    cfg.write_text(yaml.safe_dump({"schema_version": 1, **base}))
    output = tmp_path / "run1"
    report = train(cfg, output)
    assert report["env_steps"] == 24
    checkpoint = output / "checkpoints/latest.pt"
    assert checkpoint.exists()
    eva = evaluate_checkpoint(checkpoint, base["env_config"], episodes=2,
                              seed=5501, device="cpu")
    assert len(eva["returns"]) == 2
    base["total_env_steps"] = 48
    cfg.write_text(yaml.safe_dump({"schema_version": 1, **base}))
    resumed = train(cfg, tmp_path / "run2", resume=checkpoint)
    assert resumed["env_steps"] == 48
    assert len((tmp_path / "run2/train_history.jsonl").read_text().splitlines()) == 1


def test_cpu_resume_matches_uninterrupted_updates_bitwise(tmp_path):
    """Updates, critic/actor weights and optimizer/RNG states must resume exactly on CPU."""
    from dataclasses import asdict

    import yaml

    source = load_train_config(ROOT / "configs/train/mappo_smoke.yaml")
    options = asdict(source)
    options.update(env_config=str(ROOT / "configs/env/learning_sanity.yaml"),
                   device="cpu", num_envs=2, rollout_steps=6, total_env_steps=24,
                   torch_threads=2)
    options["ppo"].update(ppo_epochs=1, num_minibatches=1)
    full_cfg = tmp_path / "full.yaml"
    full_cfg.write_text(yaml.safe_dump({"schema_version": 1, **options}))
    train(full_cfg, tmp_path / "uninterrupted")
    options["total_env_steps"] = 12
    split_cfg = tmp_path / "split.yaml"
    split_cfg.write_text(yaml.safe_dump({"schema_version": 1, **options}))
    train(split_cfg, tmp_path / "first_half")
    options["total_env_steps"] = 24
    split_cfg.write_text(yaml.safe_dump({"schema_version": 1, **options}))
    train(split_cfg, tmp_path / "second_half",
          resume=tmp_path / "first_half/checkpoints/latest.pt")
    full = torch.load(tmp_path / "uninterrupted/checkpoints/latest.pt",
                      map_location="cpu", weights_only=False)
    split = torch.load(tmp_path / "second_half/checkpoints/latest.pt",
                       map_location="cpu", weights_only=False)
    for component in ("actor", "critic"):
        assert full[component].keys() == split[component].keys()
        for key in full[component]:
            assert torch.equal(full[component][key], split[component][key]), key
    assert full["env_steps"] == split["env_steps"] == 24
