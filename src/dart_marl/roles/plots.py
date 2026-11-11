"""legacy_study paired allocator diagnostics; no misleading statistical claims."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DISPLAY_LABELS = {
    "fixed": "Fixed", "greedy": "Greedy", "switch_aware": "Switch-aware",
    "r_cart": "R-CART", "r_cart_no_sync": "No sync",
    "r_cart_bounded_sync": "Bounded-sync R-CART",
    "r_cart_service": "Service-aware R-CART",
    "r_cart_no_switch": "No switch",
    "r_cart_no_hysteresis": "No hysteresis",
}


def plot_allocators(report_path: str | Path, output: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    methods = list(report["methods"])
    values = [report["methods"][m]["mean_return"] for m in methods]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(np.arange(len(methods)), values, alpha=0.8)
    ax.set_xticks(np.arange(len(methods)),
                  [DISPLAY_LABELS.get(m, m) for m in methods], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.set_ylabel("Mean team return (raw)")
    ax.set_title("legacy_study: analytic R-CART vs frozen-controller baselines"
                 if report.get("evaluation_kind", "").startswith("analytic")
                 else "legacy_study: identical controller, paired environment seeds")
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, f"{v:.2f}", ha="center", fontsize=9)
    fig.text(0.02, 0.01, "Single training seed: paired development diagnostic, not a research result", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_recovery(report_path: str | Path, output: str | Path) -> Path:
    """Observed service recovery diagnostics; censored latencies not imputed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    labels = list(report["methods"])
    coverage = [report["methods"][name]["recovery_metrics"][
        "mean_post_event_repeat_service_fraction"] for name in labels]
    restored = [report["methods"][name]["recovery_metrics"][
        "restoration_rate_among_at_risk"] for name in labels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.7))
    x = np.arange(len(labels))
    for ax, data, ylabel, title in (
        (ax1, coverage, "Fraction", "Post-event repeating-task service"),
        (ax2, restored, "Restored / at-risk", "Observed first restoration (censor-aware)"),
    ):
        values = [float(v) if v is not None else 0.0 for v in data]
        ax.bar(x, values, alpha=0.8)
        ax.set_xticks(x, [DISPLAY_LABELS.get(name, name) for name in labels],
                      rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for i, v in enumerate(data):
            ax.text(i, values[i] + 0.02, "N/A" if v is None else f"{v:.2f}",
                    fontsize=8, ha="center")
    fig.suptitle("R-CART development: paired fault evaluation, one trained actor")
    fig.text(0.02, 0.01, "Recovery denominators differ; no unobserved restoration times imputed",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path
