"""Select a critic checkpoint using development fault scenarios only.

This is MODEL SELECTION. Never use the predeclared TEST_ONLY scenarios or
report validation numbers here as an unbiased final performance estimate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dart_marl.algorithms.mappo.critic_audit import audit_checkpoint
from dart_marl.envs.config import load_config
from dart_marl.roles.evaluation import _same_task_problem


def select_critic_checkpoint(*, checkpoints: list[str | Path],
                             training_env_config: str | Path,
                             scenarios: list[str | Path], out: str | Path,
                             episodes: int = 4, seed: int = 840000,
                             device: str = "cpu") -> dict[str, Any]:
    import torch

    if not checkpoints or not scenarios or episodes < 2:
        raise ValueError("nonempty checkpoints/scenarios and >=2 validation episodes required")
    if len({str(Path(p).resolve()) for p in checkpoints}) != len(checkpoints):
        raise ValueError("duplicate checkpoint path")
    base = load_config(training_env_config)
    names = []
    for scenario in scenarios:
        cfg = load_config(scenario)
        if not _same_task_problem(base, cfg):
            raise ValueError("candidate validation scenario changes the task geometry/dynamics")
        if "TEST_ONLY" in cfg.name:
            raise ValueError("TEST_ONLY scenario cannot be used for checkpoint selection")
        names.append(cfg.name)
    if len(set(names)) != len(names):
        raise ValueError("validation scenario names must be unique")
    entries = []
    settings = set()
    for ck in checkpoints:
        ck = Path(ck)
        # Trusted self-produced checkpoint only; this is a pickle-containing file.
        payload = torch.load(ck, map_location="cpu", weights_only=False)
        if payload.get("format") != "dart-marl-mappo-v1":
            raise ValueError("invalid MAPPO checkpoint format")
        config = dict(payload["train_config"])
        # A resumed run can have a larger total_env_steps but unchanged algorithm.
        for field in ("total_env_steps", "eval_every", "save_every", "eval_episodes", "device"):
            config.pop(field, None)
        settings.add(json.dumps(config, sort_keys=True))
        entry = {"checkpoint": str(ck),
                 "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
                 "training_step": int(payload["env_steps"]),
                 "training_seed": int(config["seed"]),
                 "scenarios": {}, "normalized_rmse_values": []}
        for scenario, name in zip(scenarios, names):
            report = audit_checkpoint(ck,scenario,training_env_config,
                                      episodes=episodes,seed=seed,device=device)
            norm = report["critic_rmse"] / max(report["constant_mean_rmse"],1e-8)
            entry["scenarios"][name] = {
                "rmse": report["critic_rmse"],
                "constant_mean_rmse": report["constant_mean_rmse"],
                "relative_rmse": float(norm),
                "explained_variance": report["critic_explained_variance"],
            }
            entry["normalized_rmse_values"].append(float(norm))
        entry["mean_relative_rmse"] = float(np.mean(entry["normalized_rmse_values"]))
        entries.append(entry)
    if len(settings) != 1:
        raise ValueError("checkpoint candidates must share identical algorithm, seed, and training settings")
    winner = min(entries,key=lambda v:(v["mean_relative_rmse"],v["training_step"]))
    result = {
        "format": "dart-marl-critic-critic-development-selection-v1",
        "validation_scenario_names": names,
        "validation_seed_base": seed, "episodes_per_scenario": episodes,
        "criterion": "equal-weight mean per-scenario RMSE / constant-mean-target RMSE; lower is better",
        "selected_checkpoint": winner["checkpoint"],
        "selected_checkpoint_sha256": winner["checkpoint_sha256"],
        "selected_training_step": winner["training_step"],
        "entries": entries,
        "warning": "VALIDATION-ONLY selection. Never use test-only fault schedules for choosing checkpoints. "
                   "This may overfit development scenarios; do not quote the selected validation score as a test result.",
    }
    target=Path(out)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result
