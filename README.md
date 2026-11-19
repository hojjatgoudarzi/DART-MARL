# marl-project

Decentralized multi-rover coordination under rover failure, with limited communication. 12 rovers must serve 6 points of interest (POIs), each of which needs a minimum number of rovers present at once, while teammates fail permanently. Each step, every rover broadcasts one message to teammates within its comm radius, attends to what it hears and to the POIs, and picks a POI (or idles). A fixed controller drives it there. There is no central allocator: the team has to learn who serves which POI, and who fills in after a failure. The rover domain comes from [DART-MARL](https://github.com/hojjatgoudarzi/DART-MARL).

## Project Structure

```
marl-project/
├── configs/rover_team_12.yaml   # map, rovers, POIs
├── envs/RoverEnv.py             # batched environment: controller, rewards, failures, comm graph
├── models/
│   ├── GraphTransformer.py      # decentralized policy
│   ├── ValueNetwork.py          # centralized value network (training only)
│   └── utils.py                 # flatten, CrossAttentionBlock
├── train.py                     # MAPPO with a failure curriculum
├── evaluate.py                  # failure sweep, recovery metrics, baselines; draws the figures
├── plot.py                      # figures from evaluate.py's csvs (also called by evaluate.py)
├── visualize.py                 # record one episode for the Rerun viewer
├── train.sh                     # sbatch wrapper for train.py
├── sweep.sh                     # comm-radius sweep: one training job per radius
└── checkpoints/                 # one folder per run: step{N}.pth
```

## Environment

- **Action:** a POI index, or idle. Closed POIs are masked out.
- **Observation:** each rover sees its own velocity, every other rover and POI only relative to itself, and every POI's mission specification (value, coupling, one-shot or repeating, steps until its window opens and closes). Whether a POI is served or already completed is not observed: rovers infer it from the teammates they hear. A completed one-shot POI stays selectable but pays nothing.
- **Reward:** the team reward `G` is the value of the POIs served this step. Each rover also gets its difference reward `D_i = G − G_{−i}` (what the team would lose without it), which is what it trains on by default.
- **Failures:** permanent, at random steps. The failed rover is picked uniformly, or by a greedy adversary that picks the rover with the largest `D_i` (the one the team needs most right then).
- **Communication:** two living rovers hear each other when they are closer than `--comm-radius`.
- Rovers start at uniformly random positions. POIs are fixed.

## Model

[`GraphTransformer`](models/GraphTransformer.py), shared by every rover: a personal embedding of the rover's own state, one broadcast message, then cross-attention over its own message, the messages it hears and all POIs, each tagged with an MLP of its relative position and distance. A pointer head scores each POI, plus an idle logit. [`ValueNetwork`](models/ValueNetwork.py) is a transformer over all rovers and POIs, with one value per rover.

## Training

MAPPO with per-rover GAE on difference rewards (`--reward team` for the shared reward). The failure curriculum:
1. **Warmup** (`--warmup` updates): no failures.
2. **Failure ramp** (half the remaining updates): the maximum failures per episode rises from 0 to `--max-faults`.
3. **Adversarial ramp** (the other half): the probability that a failure is adversarial rises from 0 to `--adversarial`.

Runs log to wandb (project `marl-project`) and save to `checkpoints/<name>_s<seed>_<timestamp>/`. All settings: `python train.py -h`.

## Evaluation

`evaluate.py` plays policies deterministically on the same episodes with exactly `k` failures (`--failures`), picked at random or adversarially, and reports return, the cost of the failures (vs. the same episode without them), and recovery: the share of lost POIs restored within 10 steps, the median restoration lag, and the excess lag beyond the nearest free rover's travel time.

Baselines (`--baselines`), on the same controller: `greedy` (centralized, replans every step), `switch` (DART-MARL's centralized Switch-Aware: exact coalition search on events, optimal with no failures), `local` (greedy over only the rovers each rover hears), `sticky` (local, with shared commitments), and `fixed` (greedy's first assignment, never replanned).

## Usage

```bash
pip install torch numpy pyyaml tqdm wandb rerun-sdk && wandb login

python train.py                                      # comm radius 15, difference rewards
python train.py --comm-radius inf --seed 1
sbatch train.sh --updates 3000                       # on dgxh
python train.py --checkpoint checkpoints/<run>/step100.pth

python evaluate.py checkpoints/*/step500.pth --baselines greedy switch local sticky fixed
sbatch sweep.sh                                      # job array: radii 0 8 15 inf -> checkpoints/sweep_<job id>/
python plot.py checkpoints/eval.csv                  # redraw: failure_sweep, reward_around_failure, radius_sweep, overview (2 x 3)

python visualize.py checkpoints/<run>/step500.pth --failures 3 --adversarial   # -> checkpoints/episode.rrd
python visualize.py greedy --failures 3 --adversarial --seed 0                  # a baseline on the same episode
rerun checkpoints/episode.rrd                                                    # open (e.g. on your laptop)
```

On a login node, set `OMP_NUM_THREADS=4` to avoid thread-limit errors.
