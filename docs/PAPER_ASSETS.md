# Publication-ready figures and tables

Run `scripts/make_paper_assets.py` with only **existing** recorded data.
All inputs are optional individually but at least one must be supplied:

```bash
python scripts/make_paper_assets.py \
  --training outputs/team12_training \
  --evaluation outputs/team12_publication/evaluation/allocator_report.json \
  --trace outputs/team12_publication/rover_recovery_12_trace.json \
  --historical-csv results/heldout_episode_returns.csv \
  --out outputs/paper_assets
```

The output directory must not already exist; previous results are never
silently replaced. Every figure is generated as **PNG, PDF, SVG**, with
publication-safe font embedding settings and consistent labels.

## Figures and supporting results

| Data source | Assets produced | Interpretation |
|---|---|---|
| MAPPO training history | `training_return`, `training_diagnostics`, `training_poi_service` | One trained seed: descriptive learning and diagnostics, not multi-seed CIs |
| Matched allocator evaluation | `allocator_returns_descriptive`, `paired_episode_differences`, `allocator_poi_service`, `allocator_recovery_descriptive`, `allocator_switches_descriptive`, `allocator_table.tex`, `allocator_table.csv` | Same frozen policy and episode seeds; episode spread is **not training-seed uncertainty** |
| Real trained-policy trajectory JSON | `rollout_recovery_timeline`, `rollout_service_timeline`, `rollout_events.csv` | One illustrative paired execution; events aligned to simulator decision boundaries |
| Archived multi-seed independent study | `historical_independent_seed_comparison`, `historical_paired_seed_table.tex`, `historical_paired_seed_table.csv` | 10 independently trained controller seeds per variant; seed-level percentile bootstrap intervals |

`manifest.json` contains SHA-256 hashes of every source and produced artifact
and the number of independent units underlying figures. LaTeX tables use
`\\toprule`, `\\midrule`, `\\bottomrule` from `\\usepackage{booktabs}`.

### Scientific caveats

1. **Raw return scales differ** between the historical six-rover map and the
   new twelve-rover 120-step map. Never plot their means on a common y-axis
   implying a matched comparison.
2. The archived final study did **not** demonstrate R-CART superiority over
   Switch-Aware. The displayed seed-level bootstrap interval crosses zero.
3. The trained policy uses a navigation prior; an attractive learning curve
   does not by itself establish novel learned coordination.
4. Observed restoration times can be censored. Recovery-rate denominators and
   restored-only lag should be reported separately.
5. No unobserved points, missing metrics, or unrealized results are estimated
   by this plotting tool.

Additional article-specific composite figures can be assembled from the
saved vector panels without rerunning expensive experiments.

## Published completed results

The repository now includes the **actual** complete twelve-rover figure set in `results/team12/figures/`, LaTeX tables in `results/team12/tables/`, and original source data (training history, initial evaluation, config, nine-method allocator report, per-episode CSV) in `results/team12/data/`. See [twelve-rover interpretation](RESULTS_12_ROVERS.md). The original per-figure hashes and source identities are retained in `results/team12/paper_assets_manifest.json`, with public actor-only export relationship in `results/team12/provenance.json`.

Regenerate from released files by using `--training results/team12/data --evaluation results/team12/data/allocator_report.json --trace media/rover_recovery_12_trace.json --historical-csv results/heldout_episode_returns.csv --out outputs/reproduced_paper_assets`. Output may have different exact byte hashes under other rendering-library versions, although underlying numerical inputs are unchanged.

## Composite paper figures (2×3 publication suite)

For clearly readable six-panel comparisons and independent-controller paired
lines, use [`PUBLICATION_FIGURES.md`](PUBLICATION_FIGURES.md). The source
command is `python scripts/make_publication_figures.py --source results/team12
--out outputs/new_paper_figures`. It produces 4 figures × 3 formats and
paired-episode/summary/event CSV plus a LaTeX table, without rerunning PPO.
