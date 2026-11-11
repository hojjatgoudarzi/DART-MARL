#!/usr/bin/env python3
"""Sequential distinct training seeds with immutable config snapshots (no GPU oversubscription).

Example:
python scripts/train_seeds.py --config configs/train/mappo_role_guided.yaml \\
  --seeds 11 22 33 44 55 --out-root outputs/role_prior_guided

Output summaries are VALIDATION diagnostics, never held-out test claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Training seeds must be unique")
    if args.out_root.exists():
        parser.error(f"Output root already exists: {args.out_root}")
    base = yaml.safe_load(args.config.read_text())
    if base.get("schema_version") != 1:
        parser.error("Invalid training YAML schema_version")
    # Training configs assume invocation at repository root.
    root = Path(__file__).resolve().parents[1]
    args.out_root.mkdir(parents=True)
    configs = args.out_root / "resolved_configs"
    configs.mkdir()
    results = []
    try:
        for seed in args.seeds:
            cfg = dict(base)
            cfg["seed"] = int(seed)
            if args.device:
                cfg["device"] = args.device
            path = configs / f"seed_{seed}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            out = args.out_root / f"seed_{seed}"
            print(f"=== Starting independent MAPPO training seed {seed} ===", flush=True)
            subprocess.run([sys.executable, "-m", "dart_marl", "train",
                            "--config", str(path.resolve()), "--out", str(out.resolve())],
                           cwd=root, check=True)
            rows = [json.loads(s) for s in (out / "train_history.jsonl").read_text().splitlines()]
            final = next((r["evaluation"] for r in reversed(rows) if "evaluation" in r), None)
            if final is None:
                raise RuntimeError(f"Training seed {seed} has no validation evaluation")
            results.append({"training_seed": seed,
                            "mean_validation_return": final["mean_return"],
                            "mean_unique_pois_served": final["mean_unique_pois_served"],
                            "poi_service_steps": final["mean_poi_service_steps"],
                            "role_presence": final["mean_role_presence_fraction"],
                            "validation_episodes": final["episodes"]})
            (args.out_root / "validation_per_train_seed.json").write_text(json.dumps(results, indent=2))
    except (subprocess.CalledProcessError, RuntimeError):
        print("Some seeds did not finish; completed runs are preserved. Check logs.", file=sys.stderr)
        raise
    with (args.out_root / "validation_per_train_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    returns = np.asarray([r["mean_validation_return"] for r in results], dtype=np.float64)
    summary = {"independent_training_seeds": len(results),
               "training_seeds": args.seeds,
               "validation_mean_across_training_seeds": float(returns.mean()),
               "validation_sample_std_across_training_seeds": (
                   float(returns.std(ddof=1)) if len(returns) > 1 else None),
               "warning": "These are repeated validation results for debugging/model selection, "
                          "NOT unbiased held-out publication test results."}
    (args.out_root / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
