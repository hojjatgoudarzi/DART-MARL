"""Single-run MAPPO learning and per-task diagnostics (not scientific multi-seed plots)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _read_history(history: str | Path) -> list[dict]:
    lines = [json.loads(line) for line in Path(history).read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty training history")
    return lines


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("Install optional plotting dependencies: pip install -e '.[plots]'") from exc


def _save(fig, output: str | Path) -> Path:
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f"figure already exists: {p}")
    fig.savefig(p, dpi=180)
    _mpl().close(fig)
    return p


def plot_training(history: str | Path, output: str | Path) -> Path:
    plt = _mpl()
    lines = _read_history(history)
    steps = np.asarray([row["env_steps"] for row in lines], dtype=int)
    train_vals = np.asarray([np.nan if row["completed_return_mean"] is None else
                             row["completed_return_mean"] for row in lines], dtype=float)
    eval_pts = [(row["env_steps"], row["evaluation"]["mean_return"])
                for row in lines if "evaluation" in row]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    if np.isfinite(train_vals).any():
        ax.plot(steps, train_vals, alpha=0.55, label="Completed episodes (stochastic)")
    if eval_pts:
        ax.plot([x for x, _ in eval_pts], [y for _, y in eval_pts], "o-",
                label="Validation scenarios (deterministic)")
    ax.set(xlabel="Training environment steps", ylabel="Raw team return",
           title="MAPPO learning diagnostics (single training seed)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return _save(fig, output)


def plot_task_diagnostics(history: str | Path, output: str | Path) -> Path:
    """One line per POI: physical service steps; supports legacy_study histories."""
    plt = _mpl()
    lines = _read_history(history)
    checked = [(r["env_steps"], r["evaluation"]["mean_poi_service_steps"])
               for r in lines if "evaluation" in r and
               "mean_poi_service_steps" in r["evaluation"]]
    if not checked:
        raise ValueError("no legacy_study per-POI evaluation data in this history")
    x = np.asarray([t for t, _ in checked])
    values = np.asarray([v for _, v in checked], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for p in range(values.shape[1]):
        ax.plot(x, values[:, p], marker="o", label=f"POI {p}")
    ax.set(xlabel="Training environment steps", ylabel="Mean steps physically served",
           title="MAPPO per-POI coverage (single training seed)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return _save(fig, output)
