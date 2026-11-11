"""Regression checks for per-POI diagnostics and on-policy role-guided MAPPO."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dart_marl.algorithms.mappo.actor import SharedActor
from dart_marl.algorithms.mappo.role_guidance import role_navigation_prior
from dart_marl.algorithms.mappo.roles import assign_initial_roles, role_potential
from dart_marl.algorithms.mappo.trainer import (TrainConfig, evaluate, evaluate_checkpoint,
                                                evaluate_reference, train)
from dart_marl.envs import RoverEnv, load_config

ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "configs/env/dev_roles.yaml"


def test_role_prior_reference_covers_all_three_pois_reproducibly():
    cfg = load_config(ENV_CONFIG)
    report = evaluate_reference(cfg, episodes=8, seed=900000)
    again = evaluate_reference(cfg, episodes=8, seed=900000)
    assert report == again
    assert report["mean_return"] > 120
    assert report["mean_unique_pois_served"] == 3.0
    assert all(x > 0 for x in report["mean_poi_rewards"])
    assert report["mean_poi_completed"][2] == 1.0
    assert report["mean_role_presence_fraction"] > 0.8


def test_no_prior_has_exact_old_actor_behavior_with_same_parameters():
    torch.manual_seed(7)
    original = SharedActor(23, hidden=32)
    legacy_compatible = SharedActor(23, hidden=32, role_guidance_gain=0.0)
    legacy_compatible.load_state_dict(original.state_dict())
    obs = torch.rand(2, 12, 23)
    with torch.no_grad():
        assert torch.equal(original.distribution(obs).mean,
                           legacy_compatible.distribution(obs).mean)


def test_prior_affects_only_valid_roles_and_ppo_logprob_still_matches():
    obs = torch.zeros(4, 23)
    obs[0, -15 + 5] = 0.5
    obs[0, -15 + 11] = 1
    obs[0, -15 + 14] = 1
    obs[1, -15 + 5] = -0.4
    obs[1, -15 + 11] = 1
    obs[1, -15 + 14] = 1
    obs[2, -15 + 5] = 0.5  # invalid (completed/expired role)
    obs[3, -15 + 5] = 0.5  # only one validity flag
    obs[3, -15 + 14] = 1
    p = role_navigation_prior(obs, gain=2.0)
    torch.testing.assert_close(p[0], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(p[1], torch.tensor([-2.0, 0.0]))
    assert torch.equal(p[2:], torch.zeros((2, 2)))
    actor = SharedActor(23, hidden=32, role_guidance_gain=2.0)
    torch.manual_seed(8)
    _, latent, logp = actor.act(obs)
    recomputed, entropy = actor.evaluate(obs, latent)
    torch.testing.assert_close(logp, recomputed)
    assert torch.isfinite(entropy).all()
    with pytest.raises(ValueError):
        role_navigation_prior(obs, gain=-1)


def test_role_potential_ignores_expired_and_completed_targets():
    cfg = load_config(ENV_CONFIG)
    env = RoverEnv(cfg)
    env.reset(seed=7)
    assign_initial_roles(env)
    baseline = role_potential(env)
    assert baseline < 0
    env.poi_completed[:] = True
    assert role_potential(env) == 0.0


def test_guided_actor_can_serve_multiple_tasks_before_training():
    cfg = load_config(ENV_CONFIG)
    torch.manual_seed(42)
    actor = SharedActor(RoverEnv(cfg).observation_size, 128, role_guidance_gain=2.0)
    report = evaluate(actor, cfg, episodes=2, seed=900000, device=torch.device("cpu"),
                      role_mode="initial_coupling")
    assert report["mean_unique_pois_served"] == 3
    assert report["mean_return"] > 120


def test_guided_train_checkpoint_and_evaluation(tmp_path):
    import yaml
    options = asdict(TrainConfig(env_config=str(ENV_CONFIG), device="cpu", seed=45,
                                 num_envs=2, rollout_steps=12, total_env_steps=24,
                                 hidden_size=32, role_guidance_gain=2.0,
                                 eval_episodes=2, eval_every=1, save_every=1))
    options["ppo"].update(ppo_epochs=1, num_minibatches=1)
    config_path = tmp_path / "guided.yaml"
    config_path.write_text(yaml.safe_dump({"schema_version": 1, **options}))
    output = tmp_path / "guided"
    train(config_path, output)
    assert "role_guided" in (output / "manifest.json").read_text()
    report = evaluate_checkpoint(output / "checkpoints/latest.pt", ENV_CONFIG,
                                 episodes=2, seed=900000, device="cpu")
    assert "role_prior" in report["policy"]
    assert len(report["mean_poi_service_steps"]) == 3
    assert len(report["per_episode"]) == 2
    assert report["mean_unique_pois_served"] >= 0


def test_checkpoint_mismatch_rejects_plain_to_guided(tmp_path):
    import yaml
    base = asdict(TrainConfig(env_config=str(ENV_CONFIG), device="cpu", seed=45,
                              num_envs=1, rollout_steps=4, total_env_steps=4,
                              hidden_size=32, eval_episodes=1, eval_every=1, save_every=1))
    base["ppo"].update(ppo_epochs=1, num_minibatches=1)
    config_path = tmp_path / "trial.yaml"
    config_path.write_text(yaml.safe_dump({"schema_version": 1, **base}))
    train(config_path, tmp_path / "plain")
    base["role_guidance_gain"] = 2.0
    base["total_env_steps"] = 8
    config_path.write_text(yaml.safe_dump({"schema_version": 1, **base}))
    with pytest.raises(ValueError, match="role_guidance_gain"):
        train(config_path, tmp_path / "should_fail",
              resume=tmp_path / "plain/checkpoints/latest.pt")
