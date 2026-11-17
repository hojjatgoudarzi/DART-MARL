# Standalone source release — 16 November 2026

This is a **code organization/reproducibility release**, NOT a new trained algorithm or a new empirical result. It extracts the existing HR-MAPPO actor, recurrent centralized critic, physically grounded 12+2 Rover Domain environment, asynchronous rollout/GAE/PPO, seeded tasks/failures, local communication, diagnostics and local-information comparison methods into the independent `hr_mappo` package.

Differences from the previous combined archive:

- **No** legacy `dart_marl` runtime imports, guided-movement MAPPO, R-CART or role coordinator modules.
- New `hr-mappo-research` pip package and `hr-mappo` CLI, short run scripts and three main training configs.
- Standalone GitHub README, methods, prior evidence/limitations and publishing guide.
- Safer train-run output handling; fresh runs cannot silently corrupt existing logs.
- Inherited HR-only regression tests and selected recovery benchmark preflight.
- Source-only Git hygiene: **no trained weights, raw user logs, old figures or generated outputs in the ZIP**.

**What it does not fix:** the previously observed inability to produce *fault-caused* task reassignment to a disrupted coalition in the paired counterfactual study. A smaller controlled learning experiment is the next research activity; it is intentionally not smuggled into this cleanup release.
