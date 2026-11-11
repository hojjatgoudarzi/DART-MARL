"""Transparent research figures based on training-seed aggregates only."""
from __future__ import annotations

import json
from pathlib import Path


def plot_study(summary_file: str | Path, out_dir: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = json.loads(Path(summary_file).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    groups = data['statistics']['by_variant_scenario']
    primary = data.get('primary_method', 'r_cart')
    for variant, scenarios in groups.items():
        for scenario, block in scenarios.items():
            names = list(block['methods'])
            m = [block['methods'][method]['return']['mean'] for method in names]
            lower = []; upper = []
            has_ci = True
            for method in names:
                interval = block['methods'][method]['return']['ci95_training_seed_bootstrap_percentile']
                if interval is None:
                    has_ci = False
                    break
                lower.append(max(m[len(lower)] - interval[0], 0))
                upper.append(max(interval[1] - m[len(upper)], 0))
            fig, ax = plt.subplots(figsize=(max(8, 1.15*len(names)), 5.5))
            x = list(range(len(names)))
            ax.bar(x, m, yerr=[lower,upper] if has_ci else None, capsize=4 if has_ci else 0)
            ax.set_xticks(x, [n.replace('_', '\n') for n in names])
            ax.set_ylabel('Mean team return')
            ax.set_title(f'{variant} — {scenario}\nIndependent training seeds: {len(block["training_seeds"])}')
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            fig.tight_layout()
            target = out / f'{variant}_{scenario}_returns.png'
            fig.savefig(target, dpi=180)
            plt.close(fig)
            paths.append(target)
            contrast = block['paired_contrasts'].get(f'{primary}_minus_switch_aware')
            if contrast:
                fig, ax = plt.subplots(figsize=(7.2, 4.6))
                seed_pairs = sorted((int(k), v) for k,v in contrast['per_training_seed'].items())
                ax.scatter([i for i, _ in seed_pairs], [v for _, v in seed_pairs], s=42)
                ax.axhline(0, color='grey', ls='--', lw=1)
                ax.set_xlabel('Independent training seed')
                ax.set_ylabel(f'Paired return difference: {primary} − Switch-Aware')
                ax.set_title(f'{variant} — {scenario}')
                ax.grid(alpha=.2)
                fig.tight_layout()
                target = out / f'{variant}_{scenario}_paired_delta.png'
                fig.savefig(target, dpi=180)
                plt.close(fig)
                paths.append(target)
    return paths
