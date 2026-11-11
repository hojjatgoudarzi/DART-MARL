"""Reproducible MAPPO training/validation, independent evaluation and checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from dart_marl.algorithms.mappo.actor import SharedActor
from dart_marl.algorithms.mappo.buffer import RolloutBuffer
from dart_marl.algorithms.mappo.critic import build_critic
from dart_marl.algorithms.mappo.ppo import PPOConfig, PPOLearner
from dart_marl.algorithms.mappo.roles import assign_initial_roles, role_potential
from dart_marl.algorithms.mappo.diagnostics import EpisodeDiagnostics, role_following_actions
from dart_marl.algorithms.mappo.state import critic_input
from dart_marl.disruptions.sampling import diverse_scenario
from dart_marl.envs import RoverEnv, load_config


@dataclass(frozen=True)
class TrainConfig:
    env_config: str
    seed: int = 42
    device: str = "auto"
    num_envs: int = 8
    rollout_steps: int = 64
    total_env_steps: int = 8192
    hidden_size: int = 128
    role_mode: str = "initial_coupling"
    shaping_coef: float = 0.0
    role_guidance_gain: float = 0.0
    critic_arch: str = "agent_v1"
    treat_horizon_as_terminal: bool = False
    fault_mode: str = "none"
    eval_episodes: int = 5
    eval_every: int = 5
    save_every: int = 5
    torch_threads: int = 2
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if min(self.num_envs, self.rollout_steps, self.total_env_steps, self.hidden_size,
               self.eval_episodes, self.eval_every, self.save_every, self.torch_threads) < 1:
            raise ValueError("train counts/hidden size must be positive")
        if self.role_mode not in {"none", "initial_coupling"}:
            raise ValueError("unsupported role mode")
        if self.shaping_coef < 0 or self.role_guidance_gain < 0:
            raise ValueError("shaping/guidance coefficients cannot be negative")
        if self.fault_mode not in {"none", "diverse"}:
            raise ValueError("fault_mode must be none or diverse")
        if self.critic_arch not in {"agent_v1", "temporal_team_v2"}:
            raise ValueError("unknown critic architecture")
        if self.critic_arch == "temporal_team_v2" and self.ppo.value_target_scale <= 1:
            raise ValueError("v2 critic requires value_target_scale > 1")
        if self.role_guidance_gain and self.role_mode == "none":
            raise ValueError("role guidance requires an initial role assignment")


def load_train_config(path: str | Path) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("train YAML must contain schema_version: 1")
    options = {k: v for k, v in data.items() if k != "schema_version"}
    ppo = PPOConfig(**options.pop("ppo", {}))
    return TrainConfig(**options, ppo=ppo)


def resolve_device(spec: str) -> torch.device:
    device = "cuda" if spec == "auto" and torch.cuda.is_available() else ("cpu" if spec == "auto" else spec)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is not available to PyTorch")
    return torch.device(device)


def set_seeds(seed: int, torch_threads: int = 2) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(torch_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # No promise of full cross-platform CUDA bitwise reproducibility.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, timeout=2).decode().strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _env_sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, device=device)


def _batch_critic_inputs(envs: list[RoverEnv]) -> dict[str, np.ndarray]:
    pack = [critic_input(env) for env in envs]
    return {
        "agents": np.stack([s.agents for s in pack]),
        "pois": np.stack([s.pois for s in pack]),
        "active": np.stack([s.active for s in pack]),
        "task_mask": np.stack([s.task_mask for s in pack]),
    }


class Collector:
    """Sequential environments with batched neural inference, fixed slot masks."""

    def __init__(self, env_cfg: Any, cfg: TrainConfig):
        self.cfg = cfg
        self.base_env_cfg = env_cfg
        if cfg.fault_mode == "diverse":
            if len(env_cfg.rovers) < 3 or env_cfg.horizon < 14:
                raise ValueError("diverse training requires >=3 rovers and horizon >=14")
            if env_cfg.events:
                raise ValueError("fault_mode diverse requires nominal base environment without preset events")
        self.envs = [RoverEnv(env_cfg) for _ in range(cfg.num_envs)]
        self.episode_counts = np.zeros(cfg.num_envs, dtype=np.int64)
        self.current_return = np.zeros(cfg.num_envs, dtype=np.float64)
        self.current_length = np.zeros(cfg.num_envs, dtype=np.int64)
        self.pending_reset = np.zeros(cfg.num_envs, dtype=bool)
        self.completed_returns: list[float] = []
        self.completed_lengths: list[int] = []
        for i in range(cfg.num_envs):
            self._reset(i)

    def _reset(self, idx: int) -> None:
        seed = self.cfg.seed + idx * 100_003 + int(self.episode_counts[idx]) * 1009
        if self.cfg.fault_mode == "diverse":
            self.envs[idx] = RoverEnv(diverse_scenario(self.base_env_cfg, seed))
        self.envs[idx].reset(seed=seed)
        assign_initial_roles(self.envs[idx], self.cfg.role_mode)
        self.episode_counts[idx] += 1
        self.current_return[idx] = 0
        self.current_length[idx] = 0
        self.pending_reset[idx] = False

    def collect(self, learner: PPOLearner) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        device = learner.device
        memory = RolloutBuffer()
        reward_total = 0.0
        recent_returns: list[float] = []
        recent_lengths: list[int] = []
        for _ in range(self.cfg.rollout_steps):
            for i in range(len(self.envs)):
                if self.pending_reset[i]:
                    self._reset(i)
            obs = np.stack([env.observe() for env in self.envs])
            state = _batch_critic_inputs(self.envs)
            with torch.no_grad():
                actions, latent, logp = learner.actor.act(_to_tensor(obs, device))
                values = learner.critic(*[_to_tensor(state[k], device)
                                           for k in ("agents", "pois", "active", "task_mask")])
            actions_np = actions.cpu().numpy()
            actions_np[~state["active"]] = 0
            next_active = np.zeros_like(state["active"])
            done = np.zeros(len(self.envs), dtype=bool)
            terminated = np.zeros(len(self.envs), dtype=bool)
            rewards = np.zeros_like(state["active"], dtype=np.float32)
            for i, env in enumerate(self.envs):
                potential = role_potential(env) if self.cfg.shaping_coef else 0.0
                _, raw_reward, terminal, truncated, _ = env.step(actions_np[i])
                shaping = 0.0
                if self.cfg.shaping_coef:
                    # Terminating OR truncated episodes have zero post-episode potential.
                    # This preserves finite-episode telescoping for PBRS.
                    next_potential = 0.0 if (terminal or truncated) else role_potential(env)
                    shaping = self.cfg.shaping_coef * (learner.config.gamma * next_potential - potential)
                rewards[i, :] = (raw_reward + shaping) * state["active"][i]
                reward_total += raw_reward
                done[i] = terminal or truncated
                # This world has a known, finite-horizon objective. Treat the
                # environment horizon as a true value boundary in v2 training,
                # unlike an external continuing-task TimeLimit wrapper.
                terminated[i] = bool(terminal or (truncated and self.cfg.treat_horizon_as_terminal))
                next_active[i] = env.active
                self.current_return[i] += raw_reward
                self.current_length[i] += 1
                if done[i]:
                    ret = float(self.current_return[i])
                    length = int(self.current_length[i])
                    self.completed_returns.append(ret)
                    self.completed_lengths.append(length)
                    recent_returns.append(ret)
                    recent_lengths.append(length)
                    self.pending_reset[i] = True
            # Bootstrap from s_{t+1} *before* any env reset. Truncation bootstraps,
            # true terminal does not; reactivation cannot connect across inactive steps.
            future = _batch_critic_inputs(self.envs)
            with torch.no_grad():
                future_value = learner.critic(*[_to_tensor(future[k], device)
                                                for k in ("agents", "pois", "active", "task_mask")])
            memory.append(
                obs=obs.astype(np.float32), agents=state["agents"], pois=state["pois"],
                active=state["active"], task_mask=state["task_mask"],
                latent=latent.cpu().numpy().astype(np.float32),
                logp=logp.cpu().numpy().astype(np.float32),
                values=values.cpu().numpy().astype(np.float32),
                next_values=future_value.cpu().numpy().astype(np.float32),
                next_active=next_active, done=done, terminated=terminated,
                rewards=rewards,
            )
        stats = {
            "env_reward_total": reward_total,
            "episodes_completed": len(recent_returns),
            "completed_return_mean": float(np.mean(recent_returns)) if recent_returns else None,
            "completed_episode_length_mean": float(np.mean(recent_lengths)) if recent_lengths else None,
        }
        return memory.stack(), stats

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "env_snapshots": [env.snapshot() for env in self.envs],
            "scenario_configs": [env.config for env in self.envs],
            "episode_counts": self.episode_counts.copy(),
            "current_return": self.current_return.copy(),
            "current_length": self.current_length.copy(),
            "pending_reset": self.pending_reset.copy(),
            "completed_returns": list(self.completed_returns),
            "completed_lengths": list(self.completed_lengths),
        }

    def load_state(self, data: dict[str, Any]) -> None:
        if len(data["env_snapshots"]) != len(self.envs):
            raise ValueError("checkpoint environment count mismatch")
        if "scenario_configs" in data:
            # Critical for exact resume with episode-specific failure tapes.
            self.envs = [RoverEnv(config) for config in data["scenario_configs"]]
        elif self.cfg.fault_mode == "diverse":
            raise ValueError("fault-robust checkpoint missing episode scenario configs")
        for env, snap in zip(self.envs, data["env_snapshots"]):
            env.restore(snap)
        for k in ("episode_counts", "current_return", "current_length", "pending_reset"):
            getattr(self, k)[:] = data[k]
        self.completed_returns = list(data["completed_returns"])
        self.completed_lengths = list(data["completed_lengths"])


def _evaluate_policy(env_cfg: Any, *, episodes: int, seed: int,
                     role_mode: str, action_fn: Any) -> dict[str, Any]:
    """Common measurement path for learned and transparent reference controllers."""
    reports: list[dict[str, Any]] = []
    for episode in range(episodes):
        env = RoverEnv(env_cfg)
        env.reset(seed=seed + episode * 1009)
        assign_initial_roles(env, role_mode)
        diagnostic = EpisodeDiagnostics(env)
        while not env.is_done:
            actions = np.asarray(action_fn(env), dtype=np.float32)
            actions[~env.active] = 0
            _, reward, terminated, truncated, info = env.step(actions)
            diagnostic.record(env, reward, info)
            if terminated or truncated:
                break
        reports.append(diagnostic.summary())
    return EpisodeDiagnostics.aggregate(reports)


def evaluate(actor: SharedActor, env_cfg: Any, *, episodes: int, seed: int,
             device: torch.device, role_mode: str) -> dict[str, Any]:
    """Separate deterministic evaluation seeds, raw world reward (no shaping)."""
    previous_mode = actor.training
    actor.eval()
    try:
        def actor_actions(env: RoverEnv) -> np.ndarray:
            with torch.no_grad():
                action, _, _ = actor.act(_to_tensor(env.observe()[None], device),
                                         deterministic=True)
            return action.squeeze(0).cpu().numpy()
        return _evaluate_policy(env_cfg, episodes=episodes, seed=seed,
                                role_mode=role_mode, action_fn=actor_actions)
    finally:
        actor.train(previous_mode)


def evaluate_reference(env_cfg: Any, *, episodes: int, seed: int,
                       role_mode: str = "initial_coupling") -> dict[str, Any]:
    """Full-position role-following reference; NOT learned, NOT MAPPO/DART."""
    return _evaluate_policy(env_cfg, episodes=episodes, seed=seed,
                            role_mode=role_mode, action_fn=role_following_actions)


def _save_checkpoint(path: Path, learner: PPOLearner, collector: Collector,
                     cfg: TrainConfig, env_hash: str, update: int, env_steps: int) -> None:
    payload = {
        "format": "dart-marl-mappo-v1", "train_config": asdict(cfg), "env_sha256": env_hash,
        "update": update, "env_steps": env_steps,
        "actor": learner.actor.state_dict(), "critic": learner.critic.state_dict(),
        "actor_optimizer": learner.actor_optimizer.state_dict(),
        "critic_optimizer": learner.critic_optimizer.state_dict(),
        "collector": collector.checkpoint_state(),
        "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        "rng_cuda": torch.cuda.get_rng_state_all() if learner.device.type == "cuda" else None,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path: Path, learner: PPOLearner, collector: Collector,
                     cfg: TrainConfig, env_hash: str) -> tuple[int, int]:
    # Contains local numpy RNG data and RoverSnapshot objects; PyTorch pickle
    # loading requires weights_only=False. NEVER load untrusted checkpoints.
    payload = torch.load(path, map_location=learner.device, weights_only=False)
    if payload.get("format") != "dart-marl-mappo-v1" or payload["env_sha256"] != env_hash:
        raise ValueError("checkpoint format/environment differs from current training run")
    previous = payload["train_config"]
    current = asdict(cfg)
    # Old checkpoints may predate these explicitly backwards-compatible flags.
    # Missing keys are defaults, NOT permission to change the algorithm on resume.
    defaults = {"role_guidance_gain": 0.0, "critic_arch": "agent_v1",
                "treat_horizon_as_terminal": False, "fault_mode": "none"}
    previous_ppo = {**asdict(PPOConfig()), **previous["ppo"]}
    # Only stopping budget and eval/save cadence may differ in resumed runs.
    for name in current:
        if name in {"total_env_steps", "eval_every", "save_every", "eval_episodes", "device"}:
            continue
        prior = previous_ppo if name == "ppo" else previous.get(name, defaults.get(name))
        if prior != current[name]:
            raise ValueError(f"checkpoint training setting differs: {name}")
    learner.actor.load_state_dict(payload["actor"])
    learner.critic.load_state_dict(payload["critic"])
    learner.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    learner.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    collector.load_state(payload["collector"])
    torch.set_rng_state(payload["rng_torch"].cpu())
    np.random.set_state(payload["rng_numpy"])
    random.setstate(payload["rng_python"])
    if payload["rng_cuda"] is not None and learner.device.type == "cuda":
        torch.cuda.set_rng_state_all([v.cpu() for v in payload["rng_cuda"]])
    return int(payload["update"]), int(payload["env_steps"])


def train(config_path: str | Path, out_dir: str | Path, *, resume: str | Path | None = None
          ) -> dict[str, Any]:
    cfg = load_train_config(config_path)
    env_cfg = load_config(cfg.env_config)
    if cfg.role_mode != "none" and env_cfg.observation_mode != "research":
        raise ValueError("role-conditioned MAPPO requires research-mode target observations")
    set_seeds(cfg.seed, cfg.torch_threads)
    device = resolve_device(cfg.device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (out / "environment.json").write_text(json.dumps(env_cfg_as_dict(env_cfg), indent=2) + "\n")
    manifest = {
        "experiment": ("mappo_role_guided_not_vanilla_not_dart" if cfg.role_guidance_gain
                       else "mappo_pbrs_not_vanilla_not_dart" if cfg.shaping_coef
                       else "mappo_role_conditioned_baseline_not_dart"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "device": str(device), "numpy": np.__version__, "pyyaml": yaml.__version__, "git_sha": _git_sha(),
        "env_sha256": _env_sha(cfg.env_config), "role_mode": cfg.role_mode,
        "role_guidance_gain": cfg.role_guidance_gain, "shaping_coef": cfg.shaping_coef,
        "evaluation_seeds": "900000+episode*1009 (same across training seeds); validation only",
        "critic_arch": cfg.critic_arch, "treat_horizon_as_terminal": cfg.treat_horizon_as_terminal,
        "value_target_scale": cfg.ppo.value_target_scale, "fault_mode": cfg.fault_mode,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    probe = RoverEnv(env_cfg)
    actor = SharedActor(probe.observation_size, cfg.hidden_size,
                        role_guidance_gain=cfg.role_guidance_gain)
    critic = build_critic(architecture=cfg.critic_arch, hidden=cfg.hidden_size,
                          return_scale=cfg.ppo.value_target_scale)
    learner = PPOLearner(actor, critic, cfg.ppo, device)
    collector = Collector(env_cfg, cfg)
    update_start = 0
    total_steps = 0
    if resume is not None:
        update_start, total_steps = _load_checkpoint(Path(resume), learner, collector,
                                                    cfg, manifest["env_sha256"])
        manifest["resume_source"] = str(resume)
        manifest["resume_update"] = update_start
        manifest["resume_checkpoint_sha256"] = hashlib.sha256(Path(resume).read_bytes()).hexdigest()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Capture the initial policy score to distinguish architectural role-prior
    # effects from actual PPO learning, without consuming training RNG.
    initial_evaluation = evaluate(learner.actor, env_cfg, episodes=cfg.eval_episodes,
                                  seed=900_000, device=device, role_mode=cfg.role_mode)
    (out / "initial_evaluation.json").write_text(json.dumps(initial_evaluation, indent=2) + "\n")
    steps_per_update = cfg.num_envs * cfg.rollout_steps
    updates_target = math.ceil(cfg.total_env_steps / steps_per_update)
    history_path = out / "train_history.jsonl"
    for update in range(update_start + 1, updates_target + 1):
        data, diagnostics = collector.collect(learner)
        stats = learner.update(data)
        total_steps += steps_per_update
        row = {"update": update, "env_steps": total_steps, **diagnostics, **stats}
        if update % cfg.eval_every == 0 or update == updates_target:
            row["evaluation"] = evaluate(learner.actor, env_cfg, episodes=cfg.eval_episodes,
                                         seed=900_000, device=device, role_mode=cfg.role_mode)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        summary = {"update": update, "env_steps": total_steps,
                   "completed_return_mean": row["completed_return_mean"],
                   "policy_loss": round(row["policy_loss"], 5),
                   "value_loss": round(row["value_loss"], 5),
                   "evaluation_mean_return": row.get("evaluation", {}).get("mean_return")}
        print(json.dumps(summary), flush=True)
        if update % cfg.save_every == 0 or update == updates_target:
            _save_checkpoint(out / "checkpoints" / f"update_{update:06d}.pt", learner,
                             collector, cfg, manifest["env_sha256"], update, total_steps)
            _save_checkpoint(out / "checkpoints" / "latest.pt", learner,
                             collector, cfg, manifest["env_sha256"], update, total_steps)
    result = {"status": "training_completed", "updates": updates_target,
              "env_steps": total_steps, "output": str(out),
              "checkpoint": str(out / "checkpoints" / "latest.pt"),
              "note": ("MAPPO with explicit role-navigation prior (not vanilla MAPPO, not DART); "
                       "scientific gain not established" if cfg.role_guidance_gain else
                       "MAPPO baseline; no DART planner, no claimed scientific gain")}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def env_cfg_as_dict(cfg: Any) -> dict[str, Any]:
    return asdict(cfg)


def evaluate_checkpoint(checkpoint: str | Path, env_config: str | Path, *,
                        episodes: int = 10, seed: int = 123456, device: str = "auto",
                        output: str | Path | None = None) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    dev = resolve_device(device)
    # Same trust restriction as resume: checkpoint includes snapshot objects.
    payload = torch.load(checkpoint, map_location=dev, weights_only=False)
    if payload.get("format") != "dart-marl-mappo-v1":
        raise ValueError("invalid MAPPO checkpoint format")
    if payload["env_sha256"] != _env_sha(str(env_config)):
        raise ValueError("evaluation environment is not identical to checkpoint training environment; "
                         "explicit transfer evaluation will require a separate validated protocol")
    cfg = payload["train_config"]
    env_cfg = load_config(env_config)
    probe = RoverEnv(env_cfg)
    actor = SharedActor(probe.observation_size, cfg["hidden_size"],
                        role_guidance_gain=cfg.get("role_guidance_gain", 0.0)).to(dev)
    actor.load_state_dict(payload["actor"])
    report = evaluate(actor, env_cfg, episodes=episodes, seed=seed, device=dev,
                      role_mode=cfg["role_mode"])
    report.update({"checkpoint": str(checkpoint), "environment": str(env_config),
                   "seed": seed, "policy": ("MAPPO_role_prior_deterministic_not_DART"
                                            if cfg.get("role_guidance_gain", 0.0)
                                            else "MAPPO_PBRS_trained_deterministic_not_DART"
                                            if cfg.get("shaping_coef", 0.0)
                                            else "MAPPO_deterministic_not_DART"),
                   "checkpoint_update": payload["update"]})
    if output is not None:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError(f"evaluation result already exists: {p}")
        p.write_text(json.dumps(report, indent=2) + "\n")
    return report
