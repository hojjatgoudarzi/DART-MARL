# Reproducibility guide

## Minimum environment

Python 3.12+, `numpy`, `PyYAML`; `torch` for MAPPO, `matplotlib` for plots and `Pillow` for GIFs. For MP4 install FFmpeg. Fedora and Ubuntu are supported; no GPU is required for example playback or testing.

```bash
python -m pip install -e '.[dev,train,plots,demo]'
bash scripts/validate.sh
```

This validates the core simulator and software, and checks the example actor identity. It does not claim to reproduce all expensive independent trainings.

## Research evaluation with provided actor

```bash
python -m dart_marl evaluate-allocators \
  --env-config configs/env/dev_failure.yaml \
  --controller checkpoint \
  --checkpoint examples/checkpoints/fault_robust_seed_202.pt \
  --training-env-config configs/env/dev_roles.yaml \
  --methods fixed greedy switch_aware r_cart r_cart_no_sync r_cart_service \
  --episodes 8 --seed 900000 --device cpu \
  --out outputs/compare_seed202
```

Never evaluate CFT against a different actor without retraining and hash-matching the corresponding CFT model.

## Training a new controller

```bash
python -m dart_marl train \
  --config configs/train/mappo_fault_robust.yaml \
  --out outputs/fault_robust_run
```

The environment and training configuration control the seed, rollout length, trained actor, and critic; new runs should use new output directories. `examples/checkpoints/*.pt` are **actor-only** and cannot resume PPO. Full training checkpoints created by `train` contain optimizer state and may require legacy PyTorch deserialization: load those only if they were generated locally or provided by a trusted party.

## Multi-seed study workflow

A restartable CLI remains available:

```bash
python scripts/study.py init --spec configs/experiments/development_pilot.yaml --root outputs/study
python scripts/study.py train --root outputs/study
python scripts/study.py select --root outputs/study
python scripts/study.py evaluate-dev --root outputs/study
python scripts/study.py summarize-dev --root outputs/study
python scripts/study.py plot-dev --root outputs/study
python scripts/study.py status --root outputs/study
```

This is a **new run**, not the previous frozen evaluation. Do not unlock a preregistered test while designing or tuning algorithms. Use independent training seeds for uncertainty, not eight environmental episodes as eight model replicates. The archived held-out episode data are in `results/`.

## Test and artifact provenance

`python -m pytest -q` runs regression tests. `python scripts/make_demo.py` checks the SHA-256 of included example actor exports when using those bundled files. Cross-reference the released CSV SHA-256 with `results/manifest.json`.

## Final 12-rover training and published data

A completed 12-rover actor-only checkpoint, eight-episode nine-allocator experiment, 90-second videos, provenance and figures are bundled. See [12-rover findings](RESULTS_12_ROVERS.md), [the training workflow](TWELVE_ROVER_WORKFLOW.md), and [publication assets](PAPER_ASSETS.md). The full optimizer-bearing checkpoint stays with the owner, while its actor-only export is included for deterministic replay. This is a separate experiment from the six-rover independent-seed study.
