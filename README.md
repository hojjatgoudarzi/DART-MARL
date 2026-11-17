# HR-MAPPO: Decentralized Rover Task Selection

**Standalone research code** for *Hierarchical Recurrent Multi-Agent PPO* in the 12-rover cooperative task domain. This repository has **no dependency on the legacy DART-MARL allocator or guided movement MAPPO**. It is suitable for a separate GitHub branch/repository.

> **Research status (16 November 2026):** task allocation and coupled physical service are learned, but **failure-triggered, causally demonstrated coalition reassignment has NOT been shown**. Do not claim that our policies have learned robust active failure recovery. See [Results and limitations](docs/RESULTS_AND_LIMITATIONS.md).

## Architecture (what actually runs)

```text
Public task map + own state + sensed neighbors + delayed local reports
                       |
             shared MLP task/entity encoders
                       |
            per-rover GRU hidden state
                       |
            masked categorical task head
                    POI 0..5 / IDLE
                       |
          deterministic local goal controller
                       |
       physical service, coupling, skills, faults
                       |
         on-policy trajectory + macro rewards
                       |
      recurrent PPO + centralized team-value GRU
            (critic used for training only)
```

- **12 initially active rovers + 2 reserve slots**, 6 POIs, 120 physical steps/episode; variable activity masks for agent loss and recurrent state.
- Actor is parameter-shared. Inputs contain **no central allocator output** and **no hidden remote failures**. Each rover has its own GRU state.
- The centralized critic pools masked rover/POI state in **training only**. Shared physical service reward is not multiplied by rover count.
- Synchronous or asynchronous per-rover decisions; correct duration-aware macro GAE, recurrent sequence training with configurable chunking and burn-in.
- `declared` service attribution: a rover contributes only when it **chose** that POI, is active and skill-compatible, and physically lies within its radius. `IDLE` cannot earn service reward.
- Optional delayed, range-limited, droppable local status messages; optional physically grounded shaping/credit and overstaffing penalties for **controlled ablations**.
- The deterministic low-level navigator is shared across learned and local heuristic methods to isolate **task selection**.

## Install

Requires **Python 3.12+** and **PyTorch 2.4+**. Recommended: a separate Python/Conda environment. From this repository's root:

```bash
python -m pip install -e '.[dev,plots]'
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
bash scripts/validate.sh
```

For `device: cuda`, install a CUDA-enabled PyTorch build according to [official PyTorch installation guidance](https://pytorch.org/get-started/locally/) first. A requested CUDA device **fails clearly** if unavailable; there is no silent CPU fallback. The PyTorch actor/critic and PPO use CUDA, while the NumPy physical simulator and task orchestration currently run on CPU. GPU utilization is not guaranteed to remain high because of environment/launch overhead.

### 1. Fast local correctness smoke test

```bash
bash scripts/smoke.sh
```

Runs focused regression tests, **one** CPU PPO update, saves a checkpoint, and evaluates one episode. This is **not evidence of learned coordination**.

### 2. Main 12-rover CUDA pilot

```bash
bash scripts/train.sh configs/pilot_cuda.yaml outputs/pilot_101
```

The configured default is **pure team reward** (the strongest exploratory arm so far). Each PPO update is printed to the terminal and saved in `outputs/pilot_101/terminal.log`. To change the training seed, **copy the YAML** and edit `seed` before training. Changing the YAML between checkpoint/resume steps is not supported. **Use a fresh output directory for each new run**; training refuses to overwrite an existing history/checkpoint without explicit resume.

The main config runs **160 PPO updates** with six environments per update, a fixed five-stage curriculum, and local communication. It can require substantial compute. The demo and pilot retain the same actor/critic as full runs; only training budget and source of evidence differ.

### 3. Matched local baselines and induced-failure stress tests

```bash
bash scripts/evaluate.sh \
    outputs/pilot_101/best_checkpoint.pt \
    configs/pilot_cuda.yaml \
    outputs/pilot_101/evaluation
```

If `best_checkpoint.pt` is absent, investigate the curriculum/development selection; for software debugging only you may explicitly substitute `checkpoint.pt`. Do not silently treat final and best weights as interchangeable.

This evaluates learned Actor alongside untrained, Always-IDLE, random, nearest, static and other local heuristics under the **same environment seeds**. The stress cut is **policy-conditioned** and therefore a diagnostic of repair, **not a fair ranking by return** between policies.

### 4. Optional independent seeds

```bash
bash scripts/multiseed.sh configs/study_cuda.yaml outputs/team_reward_study
```

Starts **three separate trainings** with seeds 101, 202, 303. This can be computationally expensive; use only after a small causal-repair gate. This repo deliberately **does not** claim the next few seeds will demonstrate recovery.

### 5. Plot actual logged learning curves

```bash
python scripts/plot.py --runs \
    outputs/team_reward_study/seed_101/train_history.jsonl \
    outputs/team_reward_study/seed_202/train_history.jsonl \
    outputs/team_reward_study/seed_303/train_history.jsonl \
    --output outputs/team_reward_study/learning.png
```

## Outputs

| File | Description |
|---|---|
| `terminal.log` | Real-time updates, current stage, raw return, entropy, surplus, recovery and throughput |
| `train_history.jsonl` | Every PPO update: losses, entropy/KL, decision counts, curriculum gates, service/recovery metrics |
| `checkpoint.pt` | **Final** resumable actor/critic + optimizer + RNG + configuration |
| `best_checkpoint.pt` | Development-selected checkpoint during final curriculum stage, when eligible |
| `actor_only.pt`, `best_actor_only.pt` | Optional actor exports; do not mistake for resumable training states |
| `manifest.json`, `input_hashes.txt` | Seeds, parameters, platform, config/environment integrity hashes |
| `research_readiness.json`, `best_development.json` | Status and frozen development checkpoint-selection rule |
| `evaluation/matched/*_episodes.csv` | Paired environment-seed physical returns for local selectors |
| `evaluation/matched/*_seed_*_metrics.json` | Coupled service, staffing, one-shots, fault/recovery events |
| `evaluation/stress_cut/intervention_summary.json` | Feasible cuts, **actual drops** and conditional recoveries; null if no opportunity |
| `evaluation/summary.json` | Small human-readable index of actual evaluation means |

All `outputs/` data are ignored by Git. **No private trained checkpoints or prior experimental result artifacts are bundled in this source-only branch.** The code can read prior model state dictionaries with matching shapes/configuration, but the standalone run itself starts with fresh weights.

## Repository layout

```text
configs/                  # only three training presets + standalone 12-rover environment
src/hr_mappo/
  models.py               # parameter-shared task-conditioned MLP-GRU actor; pooled team critic
  observations.py         # local execution-information firewall and training-only critic input
  communication.py        # local status signal constraints
  scheduler.py            # synchronous / asynchronous decision clock
  rollout.py, gae.py      # physical trajectories, macro transitions, discounting and GRU lifecycle
  ppo.py                  # recurrent clipped PPO optimizer
  experiment.py           # seed/config, training, evaluation, checkpoint/resume, console logs
  physics.py, control.py  # declared task service and common deterministic navigation
  scenarios*.py           # seeded environment/task/fault distributions
  interventions.py       # induced stress cuts; diagnostics, never deployed oracle
  recovery_diagnostic.py  # optional centralized feasibility witness (research-only)
  baselines.py, metrics.py, curriculum.py
  envs/, disruptions/    # minimal retained rover physics and event types
scripts/                  # train, evaluate, validate, smoke, multiseed, plotting and stress
  compare.py, stress.py, diagnose.py
  train.sh, evaluate.sh, validate.sh, smoke.sh, multiseed.sh
  plot.py, summarize.py
tests/                    # 74 HR-only regression and training correctness tests
docs/                     # methods, results, extraction validation and next investigation
```

## Scientific boundaries

**Do not confuse:** physical service recovery with policy **reassignment caused by failure**. Our latest paired fault/no-fault analysis found 8 physical restorations among 15 stress cases but **zero effective fault-induced assignments to the damaged POI**. See [Results and limitations](docs/RESULTS_AND_LIMITATIONS.md). We have not opened the final sealed held-out split for model selection.

For the **next research step**, perform a tiny, controlled feasible-repair task first. Keep the architecture fixed and require the Actor to make a different, correct POI choice after a *locally observable* disruption. The next experiment has **not** been implemented or run as part of this cleanup; this repository freezes the current code so you can compare against a future branch.

## License & attribution

MIT License; retains the rover-domain / HR-MAPPO research team's original contributions and simulator lineage. This repository is a **clean extraction**, not a claim of a new algorithm or new benchmark results. Cite the underlying project and baseline papers appropriately in a publication.
