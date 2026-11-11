# Twelve-rover training, recovery evaluation, and a 90-second real-policy video

This is the **completed twelve-rover showcase experiment**, distinct from the archived six-rover
20-controller evaluation. The new scenario contains **12 initially active rovers,
6 POIs, and 2 reserved replacement slots (14 fixed slots in total)**. It has a
120-decision horizon. A separately trained **12-rover actor-only checkpoint is now bundled** as `examples/checkpoints/team12_seed_314159.pt`, with its original full-checkpoint identity in `results/team12/provenance.json`.

## Before running

- From the repository root, on Fedora/Linux, activate your existing Conda environment.
- `python -m pip install -e '.[dev,train,plots,demo]'`
- Install FFmpeg (`sudo dnf install ffmpeg` if available through your configured Fedora repositories).
- `python -m pytest -q` verifies all simulator and experiment invariants.
- Optional smoke training: `python -m dart_marl train --config configs/train/smoke_12_rovers.yaml --out outputs/smoke_12`. This is **not** a research model; don't commit its artifacts.

## Training (once, with the full budget)

```bash
python -m dart_marl train \
  --config configs/train/mappo_12_rovers.yaml \
  --out outputs/team12_training
```

This trains a role-conditioned MAPPO **shared actor** and a temporal team-value
critic with an **explicit engineered navigation prior**, fault-diverse episodes,
131,072 environment steps, 8 environment instances, and training seed 314159.
Each update writes metrics to `train_history.jsonl`; model files under
`checkpoints/` include training state. The initial evaluation is archived to
avoid conflating the navigation prior with improvements learned by PPO.

No exact return or runtime is guaranteed. Inspect the actual training history.
The CLI rejects overwriting an existing training directory. If interrupted,
keep the complete output and use the trusted local checkpoint with `--resume`
**and a NEW output directory**; consult the resume/config compatibility checks.
The expected saved checkpoint for the single completed run is:
`outputs/team12_training/checkpoints/latest.pt`.

## One command for evaluation, the long video, and paper assets

After full training completes:

```bash
bash scripts/build_final_artifacts.sh \
  outputs/team12_training outputs/team12_publication
```

This sequential workflow performs:

1. **Matched paired evaluation** using the same trained actor, environment,
   initial episode seeds, and event tape for all methods. Writes
   `evaluation/allocator_report.json` with per-episode physical POI service,
   return, recovery diagnostics, decisions, and optional histories.
2. **Two real trained-policy paired demonstrations**: Switch-Aware versus original R-CART, and Switch-Aware versus Fixed roles, each over **120 simulation steps** with explicit fault and replacement events. Each produces a GIF, MP4, and raw positional JSON trace.
   `interpolation=4`, `frame_ms=175`, `event_hold=4`; the resulting MP4 has
   approximately **92 seconds** of playback (actual GIF duration may differ
   slightly due to identical-frame optimization).
3. **Demo provenance checks** for both videos: tests initial rover count, reserved slots, identical seeds/controller, all expected events and reentries, final return, event sequence, and GIF duration.
4. **Publication outputs** from recorded data only: training learning curve,
   critic/optimization diagnostics, POI coverage, allocator returns and
   ablations, paired episode differences, restoration/service statistics,
   role-switches, raw service timelines, trajectory returns/active rovers,
   historical independent-seed comparison, tables, and hashes.

Every generated figure is saved in **PNG (220 dpi), vector PDF, and SVG**;
LaTeX `booktabs` tables and CSV are written alongside. The script never
fills in missing results with invented data.

### Expected showcase fault sequence

The scenario in `configs/env/rover_team_12_showcase.yaml` contains 8 explicit
fault/degradation/replacement events and 3 automatic reentries. It exercises:

- Temporary failure and delayed detection; automatic reentry
- Speed/sensor degradation
- Permanent failure
- Two distinct replacement-agent entries in previously unused slots 12 and 13
- Another temporary disable and return
- A second degradation and late temporary disable/return

The displayed **number of currently active agents can drop below 12** when
failures occur. The requirement is 12 **initially active** rovers, not 12
perpetually active agents. Event labels remain visible near their appearance.
Movement is sampled from the actor **once per simulation decision**. Extra
frames are only display interpolation; they never affect reward, policy,
roles, event schedules, or statistics.

## Evaluation/reporting boundaries

- The full twelve-rover training uses **one independently initialized
  controller**. Multiple evaluation episodes are repeated environmental
  trials on that controller and do not justify confidence intervals over
  training seeds.
- The archived 10-seed, **six-rover** study is plotted separately and retains
  its negative held-out finding. Do not combine the 6-rover and 12-rover
  returns or imply direct numerical comparability.
- A twelve-rover multi-seed research claim would require independent training
  of further controllers and matched evaluation under a predeclared protocol.
- The new 12-rover work does **not** require training CFT. The historical CFT
  components remain available for the earlier six-rover study, and were not
  validated as twelve-rover methods.
- The integrated artifact builder requires FFmpeg so both GIF and MP4 are created. The standalone `make_demo.py` can still generate only GIF and trace without it.

## Archiving additional output from future training

Keep your entire `outputs/team12_training` folder locally. To share the
important artifacts without uploading huge optimizer snapshots, provide:

- `outputs/team12_training/train_history.jsonl`, `manifest.json`,
  `initial_evaluation.json`, `train_config.json`, and `summary.json`
- `outputs/team12_training/checkpoints/latest.pt` (trusted training checkpoint)
- `outputs/team12_publication/evaluation/allocator_report.json`
- `outputs/team12_publication/showcase_verification.json`
- `outputs/team12_publication/rover_recovery_12_trace.json`
- `outputs/team12_publication/paper_assets/manifest.json`

Both recorded 12-rover GIF/MP4 pairs have already been checked visually and published with their traces. Future independent training runs can supply updated outputs for review.
No new model weights are committed into Git until the training run and its
provenance have been inspected and the GitHub publication policy is decided.

### Long evaluation and restarts

With twelve rovers, the nine-way analytic allocation study is **CPU-heavy**.
The artifact builder evaluates one method at a time and preserves the completed
reports under `evaluation_parts/`. If interrupted, rerun the same command;
completed methods are skipped, and identical controller/seed/configuration
identities are checked before merging the final report. Do **not** change
YAMLs or the checkpoint between partial runs. A partially created method folder
must be examined and removed before that individual method can be retried.

For a quick developer smoke only, `DART_SMOKE_METHODS=1 bash scripts/build_final_artifacts.sh ...`
evaluates two methods in one episode; it **must never** be used as paper evidence.
