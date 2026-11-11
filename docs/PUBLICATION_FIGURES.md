# Publication-grade multi-panel figures

Four **reproducible composite figures**, including three 2×3 panels, are stored in
[`results/team12/paper_figures/`](../results/team12/paper_figures/). These supplement,
not replace, the eleven archived publication figures in `results/team12/figures/`.

## Regenerate without training

From the repository root (requires `pip install -e '.[plots]'`):

```bash
python scripts/make_publication_figures.py \
  --source results/team12 \
  --out outputs/new_paper_figures
```

The output directory must not already contain files. Plotting uses only the
existing data in `results/team12/data/`, `results/heldout_episode_returns.csv`,
and the real twelve-rover `media/rover_reallocation_vs_fixed_12_trace.json`.
No GPU, simulation run, or training is performed. All figures save as PNG
(240 dpi), SVG (editable vector text), and vector PDF with embedded-compatible
fonts. CSV and a `booktabs` LaTeX summary table are supplied. SHA-256 hashes
of exact source inputs and output artifacts are written to `manifest.json`.

## Suggested article figure captions

### Fig. A — Training and optimization diagnostics (2×3)

Six logged training metrics over 131,072 environment steps from one 12-rover
MAPPO controller (training seed 314159). Pale traces are raw update values;
blue curves are nine-update moving averages for readability; the first panel
also reports deterministic nominal validation and the pre-training validation
mean. PPO-target value explained variance is not evidence of improved task
reward. The original values are not modified by plotting.

### Fig. B — Role-allocation comparison (2×3)

Evaluation of nine assignment strategies on eight **paired environment
realizations** with the same frozen 12-rover actor. Panels show (A) the mean
and individual episode returns, (B) returns on identical episode indices for
four representative methods, (C) episode-wise differences relative to
Switch-Aware, (D) role-switch counts, (E) descriptive post-event service
fraction, and (F) return after the first disruption. There is **one trained
controller**, so episode spread does not quantify training-seed uncertainty.
Original R-CART does not outperform Switch-Aware in this evaluation.
Panel E pools task instances meeting the *method-specific* pre-event service
eligibility rule; comparisons must not be interpreted as equal denominators.

### Fig. C — Real policy recovery trajectory (2×3)

Switch-Aware versus Fixed role assignment in one selected trained-policy
12-rover rollout, using exactly matched initial conditions and fault schedule.
Panels show cumulative team return, number of currently active rovers,
physically serviced tasks per step, cumulative role replanning events, and
per-POI service matrices for each allocator. Dotted lines mark recorded events
(eight configured disruptions plus three automatic reentries). Intermediate
video frames do not constitute additional policy decisions. One trajectory
must not be generalized as a confidence interval or mean performance.

### Fig. D — Historical 6-rover independent-controller comparison (two panels)

Paired held-out returns for ten **independently trained** fault-robust and
nominal controllers, respectively; each link connects Switch-Aware and
original R-CART under the same controller/episode seeds. These six-rover
results have a different task and return scale and are *not pooled* with the
12-rover study. This test is previously examined, not a fresh blind test.

## Important scientific limits

- No 12-rover confidence intervals across independent trained controllers
  are claimed. One controller and eight environment seeds are descriptive.
- A curve versus failure intensity, agent count, communication uncertainty,
  or training data size requires **actual controlled experiments at each x**.
  The code **does not interpolate, invent, or relabel episodes** to create
  those missing curves.
- The MAPPO actor has an engineered role-navigation prior. Do not equate
  strong task performance with learned navigation from scratch.
- Observed-only recovery lag and censored restoration cases are distinct;
  this suite favors service fraction and episode-conditioned contrasts.
- Single-run illustrative demos are chosen for clarity and should be
  presented alongside the full matched eight-episode evaluation.

For additional stress-regime figures, first run distinct, application-justified
configurations with the *same frozen 12-rover actor*, episode seeds, and
allocation parameters; archive each full evaluator JSON, config SHA-256,
model SHA-256, and scenario label. Only after collecting those measured
results should a severity-axis plot be authored. A new training experiment is
not inherently needed, but multiple independent model seeds are necessary
for training-seed uncertainty claims.
