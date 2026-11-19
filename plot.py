import os
import re
import csv
import math
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from train import BASE

INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6a63", "#e4e3de", "#fcfcfb"

# label, color, marker per baseline name prefix; centralized planners are neutral references,
# color is kept for the compared lines (a validated, colorblind-safe set)
BASELINES = {
    "switch_aware": ("Switch-Aware", INK, "s"),
    "greedy": ("Greedy", "#8f8e86", "h"),
    "fixed": ("Fixed roles", "#1baf7a", "^"),
    "local": ("Local greedy", "#e87ba4", "v"),
    "sticky": ("Sticky local", "#eda100", "D"),
}
# learned policies: one blue, lighter for smaller comm radii
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


"""
parse_args
    Parses which evaluation csv to plot, and where to save.

Returns:
    argparse.Namespace of arguments
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Plot evaluate.py results: python plot.py checkpoints/eval.csv")
    parser.add_argument("eval", type=str, nargs="?", default=f"{BASE}/checkpoints/eval.csv", help="csv from evaluate.py (its _recovery.csv is read too)")
    parser.add_argument("--out-dir", type=str, default=None, help="where the figures go (default: next to the csv)")
    parser.add_argument("--optimal", type=float, default=942.7, help="exact no-failure optimum of the evaluated episodes (942.7 for the default 64 episodes, seed 12345)")
    parser.add_argument("--baseline-radius", type=float, default=15.0, help="comm radius of the local baselines in the failure-sweep and reward figures")
    parser.add_argument("--hide", type=str, nargs="*", default=[], help="baselines left out of the failure-sweep and reward figures (all are in the radius sweep)")
    parser.add_argument("--sweep-failures", type=int, default=3, help="adversarial failures in the radius sweep's second panel")
    return parser.parse_args()


"""
read
    Reads an evaluate.py csv. Each row gets its run's kind
    (a baseline prefix, or 'learned') and comm radius.

Args:
    path: string csv path

Returns:
    list of dict rows
"""
def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['kind'] = next((b for b in BASELINES if row['name'].startswith(b)), "learned")
        if row.get('comm_radius'):
            row['radius'] = float(row['comm_radius'])
        else:   # older csvs: the radius is in the run name (r15_difference, local_r15); centralized runs have none
            match = re.search(r"(?:^|_)r(inf|[0-9.]+)", row['name'])
            row['radius'] = float(match.group(1)) if match and row['kind'] in ("learned", "local", "sticky") else math.inf
    return rows


"""
mean
    Averages a column over the rows that share a key (e.g. the
    seeds of one run name).

Args:
    rows: list of dict rows
    keys: tuple of string columns to group by
    value: string column to average

Returns:
    dict key tuple -> float mean
"""
def mean(rows, keys, value):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(float(row[value]))
    return {k: sum(v) / len(v) for k, v in groups.items()}


"""
interval
    Mean and 95% confidence half-width of a column per key. With
    several runs per key (e.g. seeds), the interval comes from the
    spread of their means; with one run, from that run's standard
    error over episodes (0 if the csv has none).

Args:
    rows: list of dict rows
    keys: tuple of string columns to group by
    value: string column of means
    sem: string column of standard errors

Returns:
    dict key tuple -> (float mean, float half-width)
"""
def interval(rows, keys, value, sem):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = {}
    for key, group in groups.items():
        values = [float(r[value]) for r in group]
        m = sum(values) / len(values)
        if len(values) > 1:
            half = 1.96 * math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1) / len(values))
        else:
            half = 1.96 * float(group[0].get(sem) or 0)
        out[key] = (m, 0.0 if math.isnan(half) else half)
    return out


"""
line
    A mean line with markers and its shaded 95% confidence band.

Args:
    ax: matplotlib Axes
    points: list of (float x, (float mean, float half-width))
    color: string line color
    band: bool shade the confidence band
    **kwargs: passed to Axes.plot
"""
def line(ax, points, color, band=True, **kwargs):
    xs = [x for x, _ in points]
    means = [m for _, (m, _) in points]
    if band:
        ax.fill_between(xs, [m - h for _, (m, h) in points], [m + h for _, (m, h) in points],
                        color=color, alpha=0.15, lw=0, zorder=2)
    ax.plot(xs, means, color=color, markeredgecolor=SURFACE, zorder=3, **kwargs)


"""
radius_label
    Args:
        radius: float comm radius

    Returns:
        string, e.g. '15' or '∞'
"""
def radius_label(radius):
    return "∞" if math.isinf(radius) else f"{radius:g}"


"""
styles
    Label, color, marker and line width of every run: baselines
    by name, learned policies one blue per comm radius (darker for
    larger radii).

Args:
    rows: list of dict rows from read

Returns:
    dict name -> (string label, string color, string marker, float width)
"""
def styles(rows):
    learned = sorted({r['radius'] for r in rows if r['kind'] == "learned"})
    shades = [BLUES[round(i * (len(BLUES) - 1) / (len(learned) - 1))] for i in range(len(learned))] if len(learned) > 1 else [BLUES[2]]
    out = {}
    for row in rows:
        if row['kind'] == "learned":
            out[row['name']] = (f"Learned (r = {radius_label(row['radius'])})", shades[learned.index(row['radius'])], "o", 2.6)
        else:
            label, color, marker = BASELINES[row['kind']]
            if row['kind'] in ("local", "sticky"):
                label = f"{label} (r = {radius_label(row['radius'])})"
            out[row['name']] = (label, color, marker, 2)
    return out


"""
shown
    The runs in the failure-sweep and recovery figures: every
    learned policy, the centralized baselines, and the local
    baselines at one radius, minus the hidden ones.

Args:
    rows: list of dict rows from read
    radius: float comm radius of the local baselines
    hide: list of string baseline kinds to leave out

Returns:
    list of string run names, in a stable order (baselines, then learned by radius)
"""
def shown(rows, radius, hide):
    keep = [r for r in rows if r['kind'] not in hide and (r['kind'] not in ("local", "sticky") or r['radius'] == radius)]
    order = list(BASELINES) + ["learned"]
    keep.sort(key=lambda r: (order.index(r['kind']), r['radius']))
    return list(dict.fromkeys(r['name'] for r in keep))


"""
new_figure
    A grid of panels in the shared style.

Args:
    rows: int panel rows
    cols: int panel columns
    size: (float, float) figure size in inches

Returns:
    matplotlib Figure, 2D array of Axes
"""
def new_figure(rows, cols, size):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(rows, cols, figsize=size, facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig, axes


"""
save
    One legend under each group of panels (every labelled line in
    the group, once), then save as .png and .pdf.

Args:
    fig: matplotlib Figure
    groups: list of (list of Axes, int legend columns)
    path: string output path without extension
"""
def save(fig, groups, path):
    legends = []
    for axes, ncol in [g for g in groups if g[0]]:
        entries = {}
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                entries.setdefault(label, handle)
        legends.append((axes, ncol, entries))
    rows = max(math.ceil(len(entries) / ncol) for _, ncol, entries in legends)
    fig.tight_layout(rect=(0, (0.3 * rows + 0.15) / fig.get_size_inches()[1], 1, 1))   # room for the legend rows
    for axes, ncol, entries in legends:
        left = min(ax.get_position().x0 for ax in axes)
        right = max(ax.get_position().x1 for ax in axes)
        fig.legend(entries.values(), entries.keys(), loc="lower center", bbox_to_anchor=((left + right) / 2, 0),
                   ncol=ncol, frameon=False, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {path}.png, {path}.pdf")


"""
draw_sweep
    Team return against the number of failures (one kind of
    pick), with the no-failure optimum.

Args:
    ax: matplotlib Axes
    rows: list of dict rows from read
    names: list of string run names to draw
    look: dict name -> style (see styles)
    picks: string 'random' or 'adversarial'
    optimal: float no-failure optimum
"""
def draw_sweep(ax, rows, names, look, picks, optimal):
    ret = interval(rows, ("name", "failures", "adversary"), "return", "return_sem")
    ax.axhline(optimal, color=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=1)
    ax.text(0.99, optimal, "optimal", transform=ax.get_yaxis_transform(), color=MUTED, fontsize=8, ha="right", va="bottom")
    for name in names:
        points = sorted((int(k), v) for (n, k, a), v in ret.items() if n == name and (a == picks or k == "0"))
        label, color, marker, width = look[name]
        line(ax, points, color, lw=width, marker=marker, ms=7, markeredgewidth=1.5, label=label)
    ax.set_xticks(sorted({int(k[1]) for k in ret}))
    ax.set_xlabel("Rovers failed per episode")
    ax.set_ylabel("Team return")


"""
draw_reward
    The team reward per step around a single failure (one kind of
    pick), aligned at the failure step and averaged over episodes.

Args:
    ax: matplotlib Axes
    curves: list of dict rows from read (the recovery csv)
    names: list of string run names to draw
    look: dict name -> style (see styles)
    picks: string 'random' or 'adversarial'
"""
def draw_reward(ax, curves, names, look, picks):
    values = interval(curves, ("name", "failures", "adversary", "curve", "step"), "value", "sem")
    for name in names:
        points = sorted((int(s), v) for (n, k, a, c, s), v in values.items() if (n, k, a, c) == (name, "1", picks, "reward"))
        if points:
            label, color, marker, width = look[name]
            line(ax, points, color, lw=width, marker=marker, ms=5, markevery=5, markeredgewidth=1, label=label)
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (5, 4)))
    ax.set_xticks(range(-10, 51, 10))
    ax.set_ylim(bottom=0)
    ax.text(0.5, ax.get_ylim()[1], "failure", color=MUTED, fontsize=8.5, va="top")
    ax.set_xlabel("Steps since the failure")
    ax.set_ylabel("Team reward per step")


"""
draw_radius
    Team return against comm radius in one scenario: the learned
    policies (one blue, mean over seeds) and the local baselines
    at each radius, with fixed roles as a dashed horizontal
    reference. The centralized baselines don't depend on the
    radius, so they are left out.

Args:
    ax: matplotlib Axes
    rows: list of dict rows from read
    look: dict name -> style (see styles)
    failures: int failures per episode
    picks: string 'none', 'random' or 'adversarial'
"""
def draw_radius(ax, rows, look, failures, picks):
    radii = sorted({r['radius'] for r in rows if r['kind'] in ("learned", "local", "sticky")})
    ret = interval(rows, ("kind", "radius", "failures", "adversary"), "return", "return_sem")
    f = str(failures)
    for name in dict.fromkeys(r['name'] for r in rows if r['kind'] == "fixed"):
        kind = next(r['kind'] for r in rows if r['name'] == name)
        if (kind, math.inf, f, picks) in ret:
            label, color, _, _ = look[name]
            ax.axhline(ret[kind, math.inf, f, picks][0], color=color, lw=1.6, ls=(0, (5, 4)), label=label, zorder=1)
    lines = [(kind, *BASELINES[kind], 2) for kind in ("local", "sticky")] + [("learned", "Learned", BLUES[2], "o", 2.6)]
    for kind, label, color, marker, width in lines:
        points = [(i, ret[kind, r, f, picks]) for i, r in enumerate(radii) if (kind, r, f, picks) in ret]
        if points:
            line(ax, points, color, lw=width, marker=marker, ms=7, markeredgewidth=1.5, label=label)
    ax.set_xticks(range(len(radii)), [radius_label(r) for r in radii])
    ax.set_xlabel("Communication radius")
    ax.set_ylabel("Team return")


"""
plot_all
    Draws every figure from an evaluate.py csv into `out`:
        failure_sweep: return against failures, random | adversarial;
        reward_around_failure: reward around one failure, random |
            adversarial (needs the recovery csv);
        radius_sweep: return against comm radius, no failures | k
            adversarial failures (needs more than one radius);
        overview: all six panels as one 2 x 3 figure.

Args:
    path: string eval csv
    out: string output directory
    optimal: float no-failure optimum
    baseline_radius: float comm radius of the local baselines in the failure and reward panels
    hide: list of string baseline kinds to leave out of the failure and reward panels
    sweep_failures: int adversarial failures in the second radius panel
"""
def plot_all(path, out, optimal=942.7, baseline_radius=15.0, hide=(), sweep_failures=3):
    rows = read(path)
    look = styles(rows)
    names = shown(rows, baseline_radius, hide)
    recovery = path.replace(".csv", "_recovery.csv")
    curves = read(recovery) if os.path.exists(recovery) else None
    radius = len({r['radius'] for r in rows if r['kind'] in ("learned", "local", "sticky")}) > 1
    k = sweep_failures

    # each panel: (title, draw function)
    panels = {
        "sweep_random": ("Return vs. random failures", lambda ax: draw_sweep(ax, rows, names, look, "random", optimal)),
        "sweep_adversarial": ("Return vs. adversarial failures", lambda ax: draw_sweep(ax, rows, names, look, "adversarial", optimal)),
        "reward_random": ("Reward recovery after random failure", lambda ax: draw_reward(ax, curves, names, look, "random")),
        "reward_adversarial": ("Reward recovery after adversarial failure", lambda ax: draw_reward(ax, curves, names, look, "adversarial")),
        "radius_none": ("Return vs. radius (no failures)", lambda ax: draw_radius(ax, rows, look, 0, "none")),
        "radius_adversarial": (f"Return vs. radius ({k} adversarial failures)", lambda ax: draw_radius(ax, rows, look, k, "adversarial")),
    }
    # each figure: (file, panel grid); the radius column gets its own legend
    figures = [("failure_sweep", [["sweep_random", "sweep_adversarial"]])]
    if curves:
        figures.append(("reward_around_failure", [["reward_random", "reward_adversarial"]]))
    if radius:
        figures.append(("radius_sweep", [["radius_none", "radius_adversarial"]]))
    grid = [["sweep_random", "sweep_adversarial"] + (["radius_none"] if radius else [])]
    if curves:
        grid.append(["reward_random", "reward_adversarial"] + (["radius_adversarial"] if radius else []))
    figures.append(("overview", grid))

    for file, layout in figures:
        letters = len(layout) * len(layout[0]) > 2
        fig, axes = new_figure(len(layout), len(layout[0]), (5.6 * len(layout[0]), 4.4 * len(layout) + 0.6))
        groups = {"failures": [], "radius": []}
        for r, row in enumerate(layout):
            for c, key in enumerate(row):
                name, draw = panels[key]
                draw(axes[r, c])
                prefix = f"({'ABCDEF'[r * len(row) + c]}) " if letters else ""
                axes[r, c].set_title(prefix + name, color=INK, fontsize=11, loc="left", fontweight="bold" if letters else "normal")
                groups["radius" if key.startswith("radius") else "failures"].append(axes[r, c])
        save(fig, [(groups["failures"], 4 if len(layout[0]) > 2 else 5), (groups["radius"], 2)], f"{out}/{file}")


if __name__ == "__main__":
    args = parse_args()
    plot_all(args.eval, args.out_dir or os.path.dirname(os.path.abspath(args.eval)), args.optimal,
             args.baseline_radius, args.hide, args.sweep_failures)
