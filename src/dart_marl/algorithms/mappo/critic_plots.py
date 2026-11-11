"""Plot critic audit comparison; raw value error is a diagnostic, not final proof."""
from __future__ import annotations

import json
from pathlib import Path


def plot_critic_comparison(old_audit: str | Path, new_audit: str | Path,
                           output: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before = json.loads(Path(old_audit).read_text())
    after = json.loads(Path(new_audit).read_text())
    for field in ("episodes", "seed", "gamma", "environment"):
        if before[field] != after[field]:
            raise ValueError(f"critic audits do not share {field}; can't compare")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    vals = [before["critic_rmse"], after["critic_rmse"]]
    bars = ax.bar(["Legacy critic", "legacy_study critic"], vals, width=.53,
                  color=["#64748b", "#2563eb"])
    for bar, value, ev in zip(bars, vals, [before["critic_explained_variance"],
                                          after["critic_explained_variance"]]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.5,
                f"RMSE {value:.2f}\nEV {ev:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Held-out discounted-return RMSE (lower is better)")
    ax.set_title("MAPPO centralized critic: diagnostic comparison")
    ax.set_ylim(0, max(vals) * 1.28 + .5)
    ax.grid(axis="y", alpha=.18)
    fig.text(.12,.01,"Same deterministic seeds; one training seed/model each. Not a policy-return comparison.",
             fontsize=8, color="#52525b")
    fig.tight_layout(rect=(0,.05,1,1))
    out = Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=180)
    plt.close(fig)
    return out
