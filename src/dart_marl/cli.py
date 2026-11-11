"""Thin, reproducible environment inspection / diagnostic rollout CLI.

Scripted diagnostic results are never represented as MARL experiments.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dart_marl.envs import RoverEnv, load_config


def _serialize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(v) for v in value]
    return value


def _scripted_assign(env: RoverEnv) -> None:
    """Diagnostic only: repeat POI IDs by coupling, fill with idle when full."""
    targets = []
    for p, poi in enumerate(env.config.pois):
        targets.extend([p] * poi.coupling)
    targets.extend([-1] * env.config.max_agents)
    roles = np.full(env.config.max_agents, -1, dtype=np.int32)
    for i in np.flatnonzero(env.active):
        roles[i] = targets[i]
    env.set_roles(roles)


def _diagnostic_actions(env: RoverEnv, policy: str, rng: np.random.Generator) -> np.ndarray:
    actions = np.zeros((env.config.max_agents, 2), dtype=np.float32)
    if policy == "random":
        actions[:] = rng.uniform(-1, 1, size=actions.shape)
        return actions
    for i in np.flatnonzero(env.active):
        role = int(env.roles[i])
        if role >= 0:
            delta = env.poi_positions[role] - env.positions[i]
            distance = np.linalg.norm(delta)
            if distance > 0:
                actions[i] = delta / max(float(distance), 1.0)
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(prog="dart-marl")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate config and show observation/state shapes")
    inspect.add_argument("--config", required=True)
    smoke = sub.add_parser("smoke", help="Non-learning scripted/random rollout for debugging")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--seed", type=int, required=True)
    smoke.add_argument("--steps", type=int, default=None)
    smoke.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    smoke.add_argument("--out", type=Path, default=None)
    train = sub.add_parser("train", help="MAPPO on-policy training (legacy_study)")
    train.add_argument("--config", required=True, help="configs/train/*.yaml")
    train.add_argument("--out", required=True, type=Path, help="new output directory")
    train.add_argument("--resume", type=Path, default=None, help="trusted local MAPPO checkpoint")
    eva = sub.add_parser("evaluate", help="Deterministic evaluation of a MAPPO checkpoint")
    eva.add_argument("--checkpoint", required=True, type=Path)
    eva.add_argument("--env-config", required=True)
    eva.add_argument("--episodes", type=int, default=10)
    eva.add_argument("--seed", type=int, default=123456)
    eva.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    eva.add_argument("--out", type=Path, default=None)
    critic_audit = sub.add_parser('audit-critic', help='Held-out Monte Carlo critic quality diagnostic')
    critic_audit.add_argument('--checkpoint', type=Path, required=True)
    critic_audit.add_argument('--env-config', required=True)
    critic_audit.add_argument('--training-env-config', required=True)
    critic_audit.add_argument('--episodes', type=int, default=8)
    critic_audit.add_argument('--seed', type=int, default=770000)
    critic_audit.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    critic_audit.add_argument('--out', type=Path, required=True)
    ref = sub.add_parser("reference-roles", help="Privileged scripted role-following diagnostic; NOT MAPPO")
    ref.add_argument("--env-config", required=True)
    ref.add_argument("--episodes", type=int, default=8)
    ref.add_argument("--seed", type=int, default=900000)
    ref.add_argument("--out", type=Path, default=None)
    plot = sub.add_parser("plot-training", help="Plot a MAPPO training run")
    plot.add_argument("--history", required=True, type=Path)
    plot.add_argument("--out", required=True, type=Path)
    tasks_plot = sub.add_parser("plot-tasks", help="Plot per-POI service across validation points")
    tasks_plot.add_argument("--history", required=True, type=Path)
    tasks_plot.add_argument("--out", required=True, type=Path)
    critic_select = sub.add_parser("select-critic-checkpoint", help="legacy_study DEVELOPMENT-ONLY fault-value model selection")
    critic_select.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    critic_select.add_argument("--training-env-config", required=True)
    critic_select.add_argument("--scenarios", nargs="+", type=Path, required=True)
    critic_select.add_argument("--episodes", type=int, default=4)
    critic_select.add_argument("--seed", type=int, default=840000)
    critic_select.add_argument("--device", choices=["cpu","auto","cuda"], default="cpu")
    critic_select.add_argument("--out", type=Path, required=True)
    critic_plot = sub.add_parser("plot-critic-comparison", help="Compare two matched deterministic critic audits")
    critic_plot.add_argument("--old-audit", type=Path, required=True)
    critic_plot.add_argument("--new-audit", type=Path, required=True)
    critic_plot.add_argument("--out", type=Path, required=True)
    fault_suite = sub.add_parser("evaluate-fault-suite", help="legacy_study paired fault-pattern generalization, grouped by training seed")
    fault_suite.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    fault_suite.add_argument("--training-env-config", required=True)
    fault_suite.add_argument("--scenarios", nargs="+", type=Path, required=True)
    fault_suite.add_argument("--methods", nargs="+", default=["fixed", "greedy", "switch_aware", "r_cart"])
    fault_suite.add_argument("--episodes", type=int, default=8)
    fault_suite.add_argument("--seed", type=int, default=980000)
    fault_suite.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    fault_suite.add_argument("--forecast-horizon", type=int, default=24)
    fault_suite.add_argument("--allow-heldout-test", action="store_true")
    fault_suite.add_argument("--out", type=Path, required=True)
    alloc = sub.add_parser("evaluate-allocators", help="legacy_study/4 paired allocation and analytic recovery evaluation")
    alloc.add_argument("--env-config", required=True)
    alloc.add_argument("--controller", choices=["checkpoint", "scripted"], default="scripted")
    alloc.add_argument("--checkpoint", type=Path, default=None)
    alloc.add_argument("--training-env-config", default=None)
    from dart_marl.roles.evaluation import METHODS as ALLOC_METHODS
    alloc.add_argument("--methods", nargs="+", choices=ALLOC_METHODS,
                       default=["fixed", "greedy", "switch_aware", "r_cart"])
    alloc.add_argument("--episodes", type=int, default=8)
    alloc.add_argument("--seed", type=int, default=700000)
    alloc.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    alloc.add_argument("--travel-weight", type=float, default=0.1)
    alloc.add_argument("--switch-weight", type=float, default=3.0)
    alloc.add_argument("--periodic-steps", type=int, default=0)
    alloc.add_argument("--forecast-horizon", type=int, default=24)
    alloc.add_argument("--forecast-discount", type=float, default=0.98)
    alloc.add_argument("--sync-weight", type=float, default=0.75)
    alloc.add_argument("--hysteresis-margin", type=float, default=1.0)
    alloc.add_argument("--min-dwell-steps", type=int, default=3)
    alloc.add_argument("--profile-latency", action="store_true",
                       help="include nondeterministic wall-clock planner timing")
    alloc.add_argument("--cft-model", type=Path, default=None,
                       help="legacy_study learned residual predictor checkpoint")
    alloc.add_argument("--cft-rank-model", type=Path, default=None,
                       help="legacy_study ranking-trained CFT checkpoint")
    alloc.add_argument("--cft-candidates", type=int, default=8)
    alloc.add_argument("--cft-beam", type=int, default=300)
    alloc.add_argument("--out", type=Path, required=True)
    recovery_plot = sub.add_parser("plot-recovery", help="Plot observed post-event recovery diagnostics")
    recovery_plot.add_argument("--report", type=Path, required=True)
    recovery_plot.add_argument("--out", type=Path, required=True)
    alloc_plot = sub.add_parser("plot-allocators", help="Plot paired legacy_study/4 comparisons")
    alloc_plot.add_argument("--report", type=Path, required=True)
    alloc_plot.add_argument("--out", type=Path, required=True)
    generate_cft = sub.add_parser("generate-cft", help="Paired clone rollouts with frozen controller")
    generate_cft.add_argument("--env-config", required=True)
    generate_cft.add_argument("--controller", choices=["checkpoint", "scripted"], default="checkpoint")
    generate_cft.add_argument("--checkpoint", type=Path, default=None)
    generate_cft.add_argument("--training-env-config", default=None)
    generate_cft.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    generate_cft.add_argument("--train-episodes", type=int, default=12)
    generate_cft.add_argument("--val-episodes", type=int, default=4)
    generate_cft.add_argument("--test-episodes", type=int, default=4)
    generate_cft.add_argument("--seed", type=int, default=510000)
    generate_cft.add_argument("--event-mode", choices=["sampled","configured","diverse"], default="sampled")
    generate_cft.add_argument("--forecast-horizon", type=int, default=24)
    generate_cft.add_argument("--forecast-discount", type=float, default=.98)
    generate_cft.add_argument("--sample-period", type=int, default=8)
    generate_cft.add_argument("--candidates", type=int, default=6)
    generate_cft.add_argument("--beam", type=int, default=200)
    generate_cft.add_argument("--out", type=Path, required=True)
    train_cft = sub.add_parser("train-cft", help="Train supervised CFT residual predictor")
    train_cft.add_argument("--dataset", type=Path, required=True)
    train_cft.add_argument("--out", type=Path, required=True)
    train_cft.add_argument("--objective", choices=["regression","ranking"],
                           default="regression")
    train_cft.add_argument("--rank-weight", type=float, default=0.65)
    train_cft.add_argument("--rank-temperature", type=float, default=5.0)
    train_cft.add_argument("--seed", type=int, default=123)
    train_cft.add_argument("--epochs", type=int, default=120)
    train_cft.add_argument("--batch-size", type=int, default=128)
    train_cft.add_argument("--hidden", type=int, default=128)
    train_cft.add_argument("--patience", type=int, default=20)
    train_cft.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    cft_plot = sub.add_parser("plot-cft", help="Prediction RMSE versus allocation-choice regret")
    cft_plot.add_argument("--report", type=Path, required=True)
    cft_plot.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plot-cft":
        from dart_marl.cft.plots import plot_cft
        print(str(plot_cft(args.report, args.out)))
        return
    if args.command == "generate-cft":
        from dart_marl.cft.rollouts import generate_data
        print(json.dumps(generate_data(
            args.env_config, args.out, controller=args.controller,
            checkpoint=args.checkpoint, training_env_config=args.training_env_config,
            device=args.device, episodes_train=args.train_episodes,
            episodes_val=args.val_episodes, episodes_test=args.test_episodes,
            seed=args.seed, event_mode=args.event_mode, horizon=args.forecast_horizon,
            discount=args.forecast_discount, sample_period=args.sample_period,
            candidates=args.candidates, beam=args.beam), indent=2))
        return
    if args.command == "train-cft":
        from dart_marl.cft.training import train_predictor
        from dart_marl.cft.rank_training import train_rank_predictor
        from dart_marl.algorithms.mappo.trainer import resolve_device
        selected_train = (train_rank_predictor if args.objective == 'ranking' else train_predictor)
        keyword_args = ({'rank_weight':args.rank_weight,
                         'rank_temperature':args.rank_temperature}
                        if args.objective == 'ranking' else {})
        print(json.dumps(selected_train(
            args.dataset, args.out, seed=args.seed, epochs=args.epochs,
            batch_size=args.batch_size, hidden=args.hidden, patience=args.patience,
            device=str(resolve_device(args.device)), **keyword_args), indent=2))
        return
    if args.command == "select-critic-checkpoint":
        from dart_marl.algorithms.mappo.selection import select_critic_checkpoint
        data = select_critic_checkpoint(checkpoints=args.checkpoints,
            training_env_config=args.training_env_config, scenarios=args.scenarios,
            episodes=args.episodes,seed=args.seed,device=args.device,out=args.out)
        print(json.dumps({"selected":data["selected_checkpoint"],
                          "step":data["selected_training_step"],
                          "criterion":data["criterion"]},indent=2))
        return
    if args.command == "plot-critic-comparison":
        from dart_marl.algorithms.mappo.critic_plots import plot_critic_comparison
        print(str(plot_critic_comparison(args.old_audit,args.new_audit,args.out)))
        return
    if args.command == "evaluate-fault-suite":
        from dart_marl.roles.generalization import evaluate_fault_suite
        result = evaluate_fault_suite(checkpoints=args.checkpoints,
            training_env_config=args.training_env_config, scenarios=args.scenarios,
            methods=tuple(args.methods), episodes=args.episodes, seed=args.seed,
            device=args.device, forecast_horizon=args.forecast_horizon,
            allow_heldout_test=args.allow_heldout_test, out=args.out)
        print(json.dumps({"status": "completed", "out": str(args.out),
                          "training_seeds": result["training_seed_count"],
                          "summary": result["summary"]}, indent=2))
        return
    if args.command == "evaluate-allocators":
        from dart_marl.roles.evaluation import evaluate_allocators, save_report
        report = evaluate_allocators(
            args.env_config, episodes=args.episodes, seed=args.seed,
            methods=tuple(args.methods), controller=args.controller,
            checkpoint=args.checkpoint, training_env_config=args.training_env_config,
            device=args.device, travel_weight=args.travel_weight,
            switch_weight=args.switch_weight, periodic_steps=args.periodic_steps,
            forecast_horizon=args.forecast_horizon,
            forecast_discount=args.forecast_discount, sync_weight=args.sync_weight,
            hysteresis_margin=args.hysteresis_margin,
            min_dwell_steps=args.min_dwell_steps,
            profile_latency=args.profile_latency,
            cft_model=args.cft_model, cft_candidates=args.cft_candidates,
            cft_beam=args.cft_beam, cft_rank_model=args.cft_rank_model)
        result = save_report(report, args.out)
        print(json.dumps({"status": "completed", "report": str(result),
                          "means": {m: r["mean_return"] for m, r in report["methods"].items()},
                          "paired_differences": report["paired_differences"],
                          "controller": report["controller"]["controller_type"]}, indent=2))
        return
    if args.command == "plot-recovery":
        from dart_marl.roles.plots import plot_recovery
        print(str(plot_recovery(args.report, args.out)))
        return
    if args.command == "plot-allocators":
        from dart_marl.roles.plots import plot_allocators
        print(str(plot_allocators(args.report, args.out)))
        return
    if args.command == "train":
        from dart_marl.algorithms.mappo.trainer import train as train_mappo
        print(json.dumps(train_mappo(args.config, args.out, resume=args.resume), indent=2))
        return
    if args.command == 'audit-critic':
        from dart_marl.algorithms.mappo.critic_audit import audit_checkpoint
        result = audit_checkpoint(args.checkpoint, args.env_config,
                                  args.training_env_config, episodes=args.episodes,
                                  seed=args.seed, device=args.device, output=args.out)
        print(json.dumps({k:v for k,v in result.items() if k != 'per_episode'},indent=2))
        return
    if args.command == "evaluate":
        from dart_marl.algorithms.mappo.trainer import evaluate_checkpoint
        print(json.dumps(evaluate_checkpoint(args.checkpoint, args.env_config,
                                            episodes=args.episodes, seed=args.seed,
                                            device=args.device, output=args.out), indent=2))
        return
    if args.command == "reference-roles":
        from dart_marl.algorithms.mappo.trainer import evaluate_reference
        report = evaluate_reference(load_config(args.env_config), episodes=args.episodes,
                                    seed=args.seed)
        report.update({"policy": "privileged_scripted_role_following_not_MAPPO",
                       "environment": args.env_config, "seed": args.seed})
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if args.out.exists():
                raise FileExistsError(f"evaluation result already exists: {args.out}")
            args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if args.command == "plot-tasks":
        from dart_marl.algorithms.mappo.plots import plot_task_diagnostics
        print(str(plot_task_diagnostics(args.history, args.out)))
        return
    if args.command == "plot-training":
        from dart_marl.algorithms.mappo.plots import plot_training
        print(str(plot_training(args.history, args.out)))
        return
    env = RoverEnv(load_config(args.config))
    if args.command == "inspect":
        obs, _ = env.reset(seed=0)
        print(json.dumps({"name": env.config.name, "obs_shape": obs.shape,
                          "active_rovers": int(env.active.sum()),
                          "num_pois": len(env.config.pois),
                          "state_keys": list(env.state()),
                          "horizon": env.config.horizon}, indent=2, default=_serialize))
        return
    obs, reset_info = env.reset(seed=args.seed)
    if args.policy == "scripted":
        _scripted_assign(env)
    rng = np.random.default_rng(args.seed + 1)
    trace = [{"initial_state": _serialize(env.state()),
              "initial_observation": _serialize(obs), "reset_info": _serialize(reset_info)}]
    limit = min(args.steps or env.config.horizon, env.config.horizon)
    total = 0.0
    for _ in range(limit):
        if args.policy == "scripted":
            _scripted_assign(env)
        actions = _diagnostic_actions(env, args.policy, rng)
        obs, reward, terminated, truncated, info = env.step(actions)
        total += reward
        trace.append({"actions": _serialize(actions), "reward": reward,
                      "observation": _serialize(obs), "info": _serialize(info),
                      "state": _serialize(env.state()),
                      "terminated": terminated, "truncated": truncated})
        if terminated or truncated:
            break
    summary = {"experiment_type": "diagnostic_not_training", "config": args.config,
               "config_name": env.config.name, "seed": args.seed,
               "policy": args.policy, "steps": len(trace)-1, "team_return": total,
               "utc_created": datetime.now(timezone.utc).isoformat()}
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.out / "resolved_config.json").write_text(
            json.dumps(_serialize(env.resolved_config()), indent=2) + "\n")
        with (args.out / "trajectory.jsonl").open("w") as f:
            for entry in trace:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved diagnostic trace to {args.out}")


if __name__ == "__main__":
    main()
