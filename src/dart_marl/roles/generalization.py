"""Paired frozen-controller fault-pattern generalization, with training-seed units.

This is a *fault-schedule* transfer test only: the task geometry and dynamics
must match the training environment. Test-only scenarios require an explicit
opt-in so they aren't used while tuning the legacy_study critic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dart_marl.envs.config import load_config
from dart_marl.roles.evaluation import _same_task_problem, evaluate_allocators


def evaluate_fault_suite(*, checkpoints: list[str | Path], training_env_config: str | Path,
                         scenarios: list[str | Path], out: str | Path,
                         methods: tuple[str, ...] = ("fixed", "greedy", "switch_aware", "r_cart"),
                         episodes: int = 8, seed: int = 980000,
                         device: str = "auto", allow_heldout_test: bool = False,
                         forecast_horizon: int = 24) -> dict[str, Any]:
    """Aggregate first within each training seed; never pseudoreplicate episode seeds."""
    import torch

    if not checkpoints or not scenarios or episodes < 1 or len(set(methods)) != len(methods):
        raise ValueError("unique methods and nonempty checkpoints/scenarios/episodes required")
    if any(m not in ("fixed", "greedy", "switch_aware", "r_cart") for m in methods):
        raise ValueError("legacy_study fault suite supports controller-independent allocators only")
    train_cfg = load_config(training_env_config)
    if len({str(Path(x).resolve()) for x in checkpoints}) != len(checkpoints):
        raise ValueError("duplicate checkpoint paths")
    if len({str(Path(x).resolve()) for x in scenarios}) != len(scenarios):
        raise ValueError("duplicate scenario paths")
    scenario_infos = []
    for path in scenarios:
        p = Path(path)
        cfg = load_config(p)
        if not _same_task_problem(train_cfg, cfg):
            raise ValueError(f"scenario {p} changes geometry/dynamics; only fault transfer is valid")
        if "TEST_ONLY" in cfg.name and not allow_heldout_test:
            raise ValueError("predeclared TEST_ONLY scenario requires --allow-heldout-test; do not tune on it")
        scenario_infos.append((p, cfg.name, hashlib.sha256(p.read_bytes()).hexdigest()))
    names = [name for _, name, _ in scenario_infos]
    if len(set(names)) != len(names):
        raise ValueError("scenario names must be unique")
    manifest = []
    config_fingerprints: set[str] = set()
    trained_seeds: set[int] = set()
    for ckpt in checkpoints:
        ckpt = Path(ckpt)
        # Trusted self-generated checkpoint only; never unpickle untrusted data.
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "dart-marl-mappo-v1":
            raise ValueError("incorrect MAPPO checkpoint")
        training_seed = int(checkpoint["train_config"]["seed"])
        if training_seed in trained_seeds:
            raise ValueError("duplicate independent training seed; don't treat checkpoints as replicates")
        trained_seeds.add(training_seed)
        # Never pool controllers from distinct algorithms/configs as replicates.
        resolved = dict(checkpoint["train_config"])
        resolved.pop("seed", None)
        resolved.pop("device", None)
        resolved.setdefault("critic_arch", "agent_v1")
        resolved.setdefault("treat_horizon_as_terminal", False)
        from dataclasses import asdict
        from dart_marl.algorithms.mappo.ppo import PPOConfig
        resolved["ppo"] = {**asdict(PPOConfig()), **resolved["ppo"]}
        config_fingerprints.add(json.dumps(resolved, sort_keys=True))
        manifest.append((ckpt, training_seed, hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                         checkpoint["train_config"].get("critic_arch", "agent_v1")))
    if len(config_fingerprints) > 1:
        raise ValueError("checkpoint training configurations differ; evaluate variants separately")
    results: list[dict[str, Any]] = []
    for ckpt, train_seed, digest, critic_arch in manifest:
        for scenario, name, checksum in scenario_infos:
            report = evaluate_allocators(
                scenario, checkpoint=ckpt, training_env_config=training_env_config,
                controller="checkpoint", methods=methods, episodes=episodes,
                seed=seed, device=device, forecast_horizon=forecast_horizon,
                capture_trace=False,
            )
            method_summaries = {}
            for method in methods:
                record = report["methods"][method]
                method_summaries[method] = {
                    "mean_return": float(record["mean_return"]),
                    "returns": [float(x) for x in record["returns"]],
                    "post_disruption_return": record.get("mean_after_first_disruption_return"),
                    "mean_role_switches": record.get("mean_role_switches"),
                }
            results.append({
                "training_seed": train_seed, "checkpoint_sha256": digest,
                "critic_arch": critic_arch, "scenario": name,
                "scenario_sha256": checksum, "scenario_path": str(scenario),
                "eval_seeds": report["evaluation_seeds"], "methods": method_summaries,
            })
    summary = {}
    for _, name, _ in scenario_infos:
        entries = [r for r in results if r["scenario"] == name]
        summary[name] = {}
        for method in methods:
            by_train = np.array([x["methods"][method]["mean_return"] for x in entries], dtype=float)
            summary[name][method] = {
                "training_seed_count": int(len(by_train)),
                "mean_of_training_seed_means": float(by_train.mean()),
                "training_seed_sample_std": float(by_train.std(ddof=1)) if len(by_train) > 1 else None,
            }
        if "r_cart" in methods and "switch_aware" in methods:
            paired = np.array([x["methods"]["r_cart"]["mean_return"] -
                               x["methods"]["switch_aware"]["mean_return"] for x in entries])
            summary[name]["r_cart_minus_switch_aware"] = {
                "mean_across_training_seeds": float(paired.mean()),
                "sample_std_across_training_seeds": float(paired.std(ddof=1)) if len(paired) > 1 else None,
                "training_seed_count": int(len(paired)),
            }
    outcome = {
        "format": "dart-marl-critic-fault-suite-v1",
        "purpose": "paired fault-pattern generalization, NOT a final scientific claim",
        "training_environment": str(training_env_config),
        "training_seed_count": len(manifest),
        "episodes_per_scenario_per_controller": episodes, "evaluation_seed_base": seed,
        "methods": list(methods), "forecast_horizon": forecast_horizon,
        "sealed_test_used": any("TEST_ONLY" in name for _, name, _ in scenario_infos),
        "results": results, "summary": summary,
        "limitations": "Only name/events change from training environment. Repeated episodes test scenario noise, not independent training runs. "
                       "Controller-dependent CFT checkpoints must be retrained/verified for each controller.",
    }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(outcome, indent=2, allow_nan=False) + "\n")
    return outcome
