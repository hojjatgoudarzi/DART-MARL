# Publish as a separate clean GitHub branch

**This folder is a complete standalone source tree.** It needs no checkout of the legacy DART-MARL package. The retained physics and event modules are simulator dependencies of HR-MAPPO, not the original guided movement MAPPO / R-CART / Switch-Aware implementations.

## Recommended: push this folder as an independent branch without touching main

Extract the ZIP into its own directory and enter that directory. Initialize a **new Git worktree/history from this clean directory**, then push to a **new branch on the existing GitHub remote**:

```bash
cd HR_MAPPO_Standalone_Research
# This is a separate working directory, NOT your existing main checkout.
git init -b research/hr-mappo-standalone
git add -A
git commit -m "Publish standalone HR-MAPPO research code"
git remote add origin YOUR_EXISTING_GITHUB_REPOSITORY_URL
git push -u origin HEAD:refs/heads/research/hr-mappo-standalone
```

The branch has independent history, intentionally contains only HR-MAPPO source, and **does not delete or modify the original repository's main branch**. GitHub may not offer a normal merge/PR comparison with a branch that has unrelated history. For a reviewable PR with shared history, create a separate branch from main but manually remove all tracked legacy files *in that branch* before committing; check the `git status` diff carefully.

Alternatively, create a completely new repository on GitHub and push the same clean folder there instead of using the old remote.

## Reproducibility checklist

1. Record `git rev-parse HEAD`, `python --version`, `torch.__version__`, CUDA/runtime/driver, and `nvidia-smi`.
2. Save the exact training YAML, `configs/rover_12.yaml` and their SHA-256 hashes; training writes `manifest.json`, `input_hashes.txt` and run-specific logs.
3. Independent **training seeds** are the unit of inference; multiple **environment evaluation seeds** nested under one checkpoint are not independent models. Development and final held-out splits must stay disjoint.
4. Retain `checkpoint.pt` and development-selected `best_checkpoint.pt`; never silently choose an outcome based on held-out evaluation.
5. Run matched fixed-fault scenarios separately from **policy-conditioned cut** stress tests; report actual drops, feasibility, censoring, and service recovery types.
6. Do not change preprocessing, task masks, reward, architecture, or evaluation protocol during resume. Use a fresh output directory when the source or config changes.
7. Archive `outputs/` outside the code branch. Do not commit trained weights or old paper figures by accident.

## Tool output verification

`bash scripts/validate.sh` runs short correctness gates. `python -m pytest -q tests` runs the extended tests and can take much longer because some include training/evaluation. `bash scripts/smoke.sh` performs one short PPO update and checkpoint reload with CPU. These are software checks, **not** a GPU benchmark or evidence of successful learned failure-responsive coordination.
