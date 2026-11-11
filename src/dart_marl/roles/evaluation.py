"""Paired evaluation of allocator baselines using ONE frozen rover controller.

legacy_study adds analytic R-CART and declared ablations. Scripted steering
is explicitly marked a privileged diagnostic, not a learned-policy result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from dart_marl.algorithms.mappo.diagnostics import EpisodeDiagnostics, role_following_actions
from dart_marl.envs.config import EnvironmentConfig, load_config
from dart_marl.envs.rover import RoverEnv
from dart_marl.roles.allocator import AllocationConfig
from dart_marl.roles.coordinator import RoleCoordinator
from dart_marl.roles.metrics import aggregate_recovery, event_recovery_metrics

METHODS = ("fixed", "greedy", "switch_aware", "r_cart",
           "r_cart_no_sync", "r_cart_no_switch", "r_cart_no_hysteresis", "r_cart_bounded_sync", "r_cart_service", "cft", "cft_rank")
DEFAULT_METHODS = ("fixed", "greedy", "switch_aware", "r_cart")


def _same_task_problem(a: EnvironmentConfig, b: EnvironmentConfig) -> bool:
    """Permit transfer from nominal to faulted map ONLY if events/name differ."""
    left, right = asdict(a), asdict(b)
    for d in (left, right):
        d.pop("events", None)
        d.pop("name", None)
    return left == right


def _load_actor(checkpoint: str | Path, training_env_config: str | Path,
                eval_config: EnvironmentConfig, device: str):
    import torch

    from dart_marl.algorithms.mappo.actor import SharedActor
    from dart_marl.algorithms.mappo.trainer import resolve_device

    path = Path(checkpoint)
    trained_config = Path(training_env_config)
    if not _same_task_problem(load_config(trained_config), eval_config):
        raise ValueError("training and evaluation environment must match except name/events")
    dev = resolve_device(device)
    # Local/trusted files only. Training checkpoints contain snapshots and
    # torch.load(weights_only=False) unpickles arbitrary Python objects.
    try:
        payload = torch.load(path, map_location=dev, weights_only=True)
    except Exception as exc:
        # Historical full-training checkpoints contain NumPy RNG/optimizer
        # objects that require legacy deserialization. Use only local files
        # produced by a trusted training run; untrusted pickle is unsafe.
        import pickle
        if not isinstance(exc, pickle.UnpicklingError):
            raise
        payload = torch.load(path, map_location=dev, weights_only=False)
    if payload.get("format") != "dart-marl-mappo-v1":
        raise ValueError("incorrect MAPPO checkpoint format")
    expected_hash = hashlib.sha256(trained_config.read_bytes()).hexdigest()
    if payload.get("env_sha256") != expected_hash:
        raise ValueError("checkpoint training environment SHA256 does not match supplied YAML")
    cfg = payload["train_config"]
    observation_size = RoverEnv(eval_config).observation_size
    actor = SharedActor(observation_size, cfg["hidden_size"],
                        role_guidance_gain=cfg.get("role_guidance_gain", 0.0)).to(dev)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    return actor, dev, {
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trained_env_config": str(trained_config),
        "trained_updates": int(payload["update"]),
        "training_seed": int(cfg["seed"]),
        "role_guidance_gain": float(cfg.get("role_guidance_gain", 0.0)),
        "device": str(dev),
        "controller_type": "frozen_MAPPO_role_prior" if cfg.get("role_guidance_gain", 0.0)
        else "frozen_MAPPO_no_role_prior",
    }


def evaluate_allocators(
    env_config: str | Path, *, episodes: int = 8, seed: int = 700_000,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    controller: str = "scripted", checkpoint: str | Path | None = None,
    training_env_config: str | Path | None = None,
    device: str = "cpu", travel_weight: float = 0.1, switch_weight: float = 3.0,
    periodic_steps: int = 0, capture_trace: bool = True,
    forecast_horizon: int = 24, forecast_discount: float = 0.98,
    sync_weight: float = 0.75, hysteresis_margin: float = 1.0,
    min_dwell_steps: int = 3, profile_latency: bool = False,
    cft_model: str | Path | None = None, cft_candidates: int = 8,
    cft_beam: int = 300, cft_rank_model: str | Path | None = None,
) -> dict[str, Any]:
    if episodes < 1 or not methods or len(set(methods)) != len(methods):
        raise ValueError("episodes >0 and unique nonempty methods required")
    if 'cft' in methods and cft_model is None:
        raise ValueError('CFT evaluation requires a trained --cft-model')
    if 'cft_rank' in methods and cft_rank_model is None:
        raise ValueError('cft_rank evaluation requires --cft-rank-model')
    if any(m not in METHODS for m in methods):
        raise ValueError(f"available methods are {METHODS}")
    if controller not in {"scripted", "checkpoint"}:
        raise ValueError("controller must be scripted or checkpoint")
    if controller == "checkpoint" and (checkpoint is None or training_env_config is None):
        raise ValueError("checkpoint controller requires --checkpoint AND --training-env-config")
    if controller == "scripted" and checkpoint is not None:
        raise ValueError("scripted controller does not accept a checkpoint")
    cfg = load_config(env_config)
    if cfg.observation_mode != "research" and controller == "checkpoint":
        raise ValueError("checkpoint role navigation requires research observation mode")
    actor, dev, provenance = None, None, {
        "controller_type": "scripted_full_position_privileged_diagnostic_NOT_MAPPO"
    }
    if controller == "checkpoint":
        actor, dev, provenance = _load_actor(checkpoint, training_env_config, cfg, device)
    results: dict[str, dict[str, Any]] = {}
    torch = None
    if actor is not None:
        import torch as torch_module
        torch = torch_module
    eval_seeds = [int(seed + 1009 * ep) for ep in range(episodes)]
    for method in methods:
        episode_reports = []
        trace: list[dict[str, Any]] = []
        for ep, episode_seed in enumerate(eval_seeds):
            env = RoverEnv(cfg)
            env.reset(seed=episode_seed)
            coordinator = RoleCoordinator(AllocationConfig(
                method, travel_weight, switch_weight, periodic_steps,
                forecast_horizon, forecast_discount, sync_weight,
                hysteresis_margin, min_dwell_steps,
                str(cft_rank_model if method == 'cft_rank' else cft_model)
                if method in ('cft', 'cft_rank') else None,
                cft_candidates, cft_beam), profile_latency=profile_latency,
                controller_sha=provenance.get('checkpoint_sha256'))
            coordinator.reset(env)
            diag = EpisodeDiagnostics(env)
            fault_step = min((e.step for e in cfg.events), default=None)
            after_fault_reward = 0.0
            post_fault_service = np.zeros(len(cfg.pois), dtype=np.int64)
            service_series: list[list[int]] = []
            reward_series: list[float] = []
            while not env.is_done:
                t = env.step_index
                if actor is None:
                    actions = role_following_actions(env)
                else:
                    assert torch is not None
                    with torch.no_grad():
                        tensor = torch.as_tensor(env.observe()[None], dtype=torch.float32, device=dev)
                        chosen, _, _ = actor.act(tensor, deterministic=True)
                    actions = chosen.squeeze(0).cpu().numpy()
                actions[~env.active] = 0
                before_roles = env.roles.copy()
                before_positions = env.positions.copy()
                _, reward, terminated, truncated, info = env.step(actions)
                diag.record(env, reward, info)
                service_series.append(np.asarray(info["poi_served"][:len(cfg.pois)],
                                                 dtype=np.int64).tolist())
                reward_series.append(float(reward))
                if fault_step is not None and t >= fault_step:
                    after_fault_reward += float(reward)
                    post_fault_service += np.asarray(info["poi_served"][:len(cfg.pois)], dtype=np.int64)
                decision = None
                if not (terminated or truncated):
                    decision = coordinator.maybe_reassign(env)
                if capture_trace and ep == 0:
                    trace.append({
                        "step": t, "reward": float(reward),
                        "positions_before": before_positions.tolist(),
                        "positions_after": np.asarray(info["post_positions"]).tolist(),
                        "active_before": np.asarray(info["pre_active_mask"], dtype=int).tolist(),
                        "roles_before": before_roles.tolist(),
                        "poi_service": np.asarray(info["poi_served"], dtype=int).tolist(),
                        "boundary_events": info["events_next_boundary"],
                        "reassignment": decision,
                    })
                if terminated or truncated:
                    break
            episode_result = diag.summary()
            episode_result.update({
                "seed": episode_seed,
                "role_replans": coordinator.summary()["decision_count"],
                "role_switches": coordinator.summary()["role_switches"],
                "after_first_disruption_return": (
                    after_fault_reward if fault_step is not None else None),
                "post_fault_poi_service_steps": (
                    post_fault_service.tolist() if fault_step is not None else None),
                "role_decisions": coordinator.decisions,
                "recovery_events": event_recovery_metrics(
                    cfg, service_series, reward_series, horizon_steps=forecast_horizon),
            })
            episode_reports.append(episode_result)
        summary = EpisodeDiagnostics.aggregate(episode_reports)
        summary.update({
            "mean_role_replans": float(np.mean([e["role_replans"] for e in episode_reports])),
            "mean_role_switches": float(np.mean([e["role_switches"] for e in episode_reports])),
            "mean_after_first_disruption_return": (
                float(np.mean([e["after_first_disruption_return"] for e in episode_reports]))
                if cfg.events else None),
            "mean_post_fault_poi_service_steps": (
                np.mean([e["post_fault_poi_service_steps"] for e in episode_reports],
                        axis=0).tolist() if cfg.events else None),
            "trace_first_episode": trace if capture_trace else None,
        })
        summary["recovery_metrics"] = aggregate_recovery(episode_reports)
        if profile_latency:
            durations = [d["planner_runtime_ms"] for e in episode_reports
                         for d in e["role_decisions"] if "planner_runtime_ms" in d]
            summary["planner_profile"] = {
                "measured_replans": len(durations),
                "mean_ms": float(np.mean(durations)) if durations else None,
                "median_ms": float(np.median(durations)) if durations else None,
                "max_ms": float(max(durations)) if durations else None,
            }
        results[method] = summary
    reference = "fixed" if "fixed" in results else methods[0]
    paired = {}
    for method in methods:
        if method == reference:
            continue
        differences = np.asarray([
            a["return"] - b["return"]
            for a, b in zip(results[method]["per_episode"], results[reference]["per_episode"])],
            dtype=float)
        paired[f"{method}_minus_{reference}"] = {
            "mean_difference": float(differences.mean()),
            "std_difference": float(differences.std(ddof=1)) if episodes > 1 else None,
            "per_seed_differences": differences.tolist(),
        }
    # Additional paired contrasts distinguish the recovery forecast from the
    # strongest static surrogate, rather than comparing only against fixed.
    if "r_cart" in results:
        for baseline in ("switch_aware", "greedy"):
            if baseline in results:
                differences = [a["return"] - b["return"] for a, b in zip(
                    results["r_cart"]["per_episode"], results[baseline]["per_episode"])]
                paired[f"r_cart_minus_{baseline}"] = {
                    "mean_difference": float(np.mean(differences)),
                    "std_difference": float(np.std(differences, ddof=1)) if episodes > 1 else None,
                    "per_seed_differences": differences,
                }
        for ablation in ("r_cart_no_sync", "r_cart_no_switch", "r_cart_no_hysteresis", "r_cart_bounded_sync", "r_cart_service", "cft", "cft_rank"):
            if ablation in results:
                differences = [a["return"] - b["return"] for a, b in zip(
                    results["r_cart"]["per_episode"], results[ablation]["per_episode"])]
                paired[f"r_cart_minus_{ablation}"] = {
                    "mean_difference": float(np.mean(differences)),
                    "std_difference": float(np.std(differences, ddof=1)) if episodes > 1 else None,
                    "per_seed_differences": differences,
                }
    includes_rcart = any(m.startswith("r_cart") for m in methods)
    includes_cft = any(m in methods for m in ('cft', 'cft_rank'))
    for learned_method in ('cft', 'cft_rank'):
        if learned_method in results and 'r_cart' in results:
            differences = [a['return'] - b['return'] for a,b in zip(
                results[learned_method]['per_episode'], results['r_cart']['per_episode'])]
            paired[f'{learned_method}_minus_r_cart'] = {
                'mean_difference': float(np.mean(differences)),
                'std_difference': float(np.std(differences, ddof=1)) if episodes > 1 else None,
                'per_seed_differences': differences}
    if 'cft_rank' in results and 'cft' in results:
        differences = [a['return'] - b['return'] for a,b in zip(
            results['cft_rank']['per_episode'], results['cft']['per_episode'])]
        paired['cft_rank_minus_cft'] = {
            'mean_difference': float(np.mean(differences)),
            'std_difference': float(np.std(differences, ddof=1)) if episodes > 1 else None,
            'per_seed_differences': differences}
    return {
        "evaluation_kind": ("learned_cft" if includes_cft else
                  "analytic_recovery" if includes_rcart
                  else "allocator_baselines"),
        "research_claim": "engineering diagnostic only; no inference from 1 training seed",
        "environment": str(env_config), "evaluation_seeds": eval_seeds,
        "profile_latency": profile_latency,
        "controller": provenance,
        "cft_model": (str(cft_model) if "cft" in methods else None),
        "cft_rank_model": (str(cft_rank_model) if "cft_rank" in methods else None),
        "cft_model_sha256": (hashlib.sha256(Path(cft_model).read_bytes()).hexdigest()
                               if "cft" in methods else None),
        "cft_rank_model_sha256": (hashlib.sha256(Path(cft_rank_model).read_bytes()).hexdigest()
                                 if "cft_rank" in methods else None),
        "allocator_settings": {"travel_weight": travel_weight,
                               "forecast_horizon": forecast_horizon,
                               "forecast_discount": forecast_discount,
                               "synchronization_weight": sync_weight,
                               "hysteresis_margin": hysteresis_margin,
                               "min_dwell_steps": min_dwell_steps,
                               "switch_weight": switch_weight,
                               "periodic_steps": periodic_steps,
                               "shared_initial_roles": True,
                               "cft_candidates":cft_candidates if includes_cft else None,
                               "cft_beam":cft_beam if includes_cft else None},
        "methods": results, "paired_differences": paired,
        "note": "All methods share identical controller and evaluation seeds. "
                "Only reported-roster and task-change triggers (unless periodic configured); "
                "no oracle fault-time trigger. Analytic R-CART scores H-step service. "
                "CFT (if selected) predicts raw-return residuals from cloned fixed-role "
                "H-step counterfactual training continuations. Learned reward prediction "
                "is a SURROGATE, not a guaranteed optimal or closed-loop Q function. "
                "Recovery metrics are offline and not visible to allocator. "
                "No research claims from one actor training seed.",
    }


def save_report(report: dict[str, Any], output: str | Path) -> Path:
    """Write JSON report in a new directory; never overwrite prior experiments."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "allocator_report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
