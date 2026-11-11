# Changelog

## 1.3.1 — Public repository polish (no change to experiments)

- Removed one redundant historical preview image from `media/`, leaving all
  twelve-rover GIF/MP4 demos and their raw traces untouched.
- Promoted the blue six-panel PPO/critic figure and the evidence-grounded
  allocator comparison in the README; separated the historical six-rover
  figure and supplementary diagnostics without removing source data.
- Clarified that historical seed-202 demo numbers refer to an old, non-shipped
  six-rover visualization, not the two published twelve-rover videos.
- Redirected the legacy historical plot command to ignored `outputs/` by
  default so it cannot overwrite published assets.
- Matched the package/CITATION/runtime version and added release/provenance
  checks to CI. No training, policy, allocator, or experiment results changed.


## 1.3.0 — Evidence-grounded publication figure suite

- Added three fully labeled 2×3 article figures (training, allocation comparison, and real-policy recovery) and one historical independent-controller paired-line chart.
- Added reproducible PNG/PDF/SVG generation, figure-caption documentation, audited input/output hashes, source-aligned LaTeX and CSV tables, and regression safeguards against false independence claims.
- Preserved the prior 12-rover checkpoint, all demo media, established results, and original scientific limitations; no retraining or new performance claim.

## 1.2.1 — Twelve-rover media clarification

- Removed obsolete six-rover GIF/MP4 and trace from public `media/` to keep only the two verified 90-second twelve-rover comparisons.
- Changed default `scripts/make_demo.py` to the actual twelve-rover actor, environment, and seed, with outputs written to ignored `outputs/` paths.
- Preserved six-rover historical raw results, actor checkpoints, reproduction support, and all scientific limitations.
- Added regression checks preventing the two generations of demo media from being mixed again.

## 1.1.0 — Training and paper-ready showcase

- Added a separate twelve-initial-rover, six-POI, 120-step environment with two replacement slots and an eight-event demo schedule.
- Extended allocator slot masking from 12 to 16 padded positions while preserving legacy methods.
- Added a fully integrated GPU training configuration, short CPU smoke config, provenance-checked approximately 90-second video workflow, and publication figure/table pipeline.
- Kept the original six-rover held-out evaluation and negative scientific conclusion unchanged.

## 1.0.0 — Clean project release

- Unified the cooperative Rover Domain simulator, policy training, allocation baselines, and experiment machinery under descriptive project names.
- Added two verified, compact, actor-only example checkpoints for direct demo playback.
- Added real trained-policy side-by-side GIF and MP4 demonstrations with recorded trajectories and deterministic seeds.
- Exported the completed held-out evaluation into auditable episode-level CSVs, including its negative primary result.
- Added concise architecture, results, reproduction, attribution, security, and contribution documentation.
- Preserved all original baseline allocation mechanisms and regression tests; no algorithmic superiority claims were introduced.
