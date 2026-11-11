# Deterministic demos from real trained-policy trajectories

The published animation is produced by `scripts/make_demo.py`, which steps a saved **MAPPO shared actor with an engineered role-navigation prior** in the Rover Domain. There are **no scripted rover movements**. An allocator selects roles; the environment, actor and fault schedule determine physical behavior. Frames inserted between simulation steps are display-only interpolation and never affect reward or role decisions.

## Twelve-rover videos

- `media/rover_reallocation_vs_fixed_12.gif` and `.mp4`: **Switch-Aware vs Fixed roles**, seed 901110, mean-free single-episode illustration, actual returns **786.7 / 639.5**. The viewer sees the contrast between adapting role assignments and never replanning.
- `media/rover_recovery_12.gif` and `.mp4`: **Switch-Aware vs original R-CART**, same seed 901110, actual returns **786.7 / 786.7**. Some roles and positions differ but POI service and returns are identical in this episode; do not claim a difference that was not recorded.
- Each demo covers 120 decisions, eight explicit disruptions and three automatic reentries, with **12 initially active rovers and two reserved replacement slots**. Actual active counts change with faults. GIF duration ~89.47 s and MP4 ~91.875 s.
- Full numeric rollouts are stored as `media/rover_reallocation_vs_fixed_12_trace.json` and `media/rover_recovery_12_trace.json`.
- `results/team12/data/showcase_verification.json` is the original verifier report for the R-CART comparison, using the **original full checkpoint SHA-256**. See `results/team12/provenance.json` for the exact actor-only export link.

## Re-create from the bundled actor-only checkpoint

```bash
python scripts/make_demo.py \
  --checkpoint examples/checkpoints/team12_seed_314159.pt \
  --training-env-config configs/env/rover_team_12.yaml \
  --env-config configs/env/rover_team_12_showcase.yaml \
  --compare fixed --seed 901110 \
  --interpolation 4 --event-hold 4 --frame-ms 175 \
  --out outputs/demo_fixed.gif --mp4 outputs/demo_fixed.mp4 \
  --trace outputs/demo_fixed_trace.json
```

Change `--compare fixed` to `--compare r_cart` to reproduce the matched R-CART comparison. Decreasing `--frame-ms` speeds up playback and increasing it slows playback. Changing interpolation or display speed does **not** change rover actions; rerunning with the same checkpoint and seed is what regenerates the trajectories. Requires FFmpeg for MP4; GIF/JSON only need Pillow and PyTorch.

The publicly bundled checkpoint is **actor-only**. Its actor weights are bit-identical to the privately retained full training checkpoint, but its file SHA differs because optimizer, collector and random-state snapshots were removed; the original demo trace still correctly records the SHA of the full model file that produced it.

## Historical six-rover evaluation

The six-rover numerical results, actor exports, configuration, and reproduction code remain in `results/`, `examples/checkpoints/`, and `configs/`. The earlier, shorter GIF/MP4 are **not** shipped in `media/` to avoid mixing the two demos. Regenerate historical visualizations explicitly with the older actor and six-rover configuration if needed; do not present the six-rover results as twelve-rover results.
