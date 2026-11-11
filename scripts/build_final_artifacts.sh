#!/usr/bin/env bash
# Resume-safe production artifacts from a *completed* trusted twelve-rover PPO run.
set -euo pipefail
TRAIN_DIR="${1:-outputs/team12_training}"
ARTIFACT_DIR="${2:-outputs/team12_publication}"
CHECKPOINT="$TRAIN_DIR/checkpoints/latest.pt"
[[ -f "$CHECKPOINT" && -f "$TRAIN_DIR/train_history.jsonl" ]] || {
  echo "Missing completed twelve-rover training files: $TRAIN_DIR" >&2; exit 2;
}
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg to produce the final MP4.' >&2; exit 2; }
mkdir -p "$ARTIFACT_DIR/evaluation_parts"
METHODS=(fixed greedy switch_aware r_cart r_cart_no_sync r_cart_no_switch r_cart_no_hysteresis r_cart_bounded_sync r_cart_service)
EPISODES="${DART_EVAL_EPISODES:-8}"
if [[ "${DART_SMOKE_METHODS:-}" == '1' ]]; then
  METHODS=(switch_aware r_cart)
  EPISODES=1
  echo 'SMOKE ONLY: 2 methods x 1 episode; not publication results'
fi
# Lock the controller, scenario, training environment, seeds, and method set.
# In particular a one-episode engineering smoke must never be mixed with the
# production eight-episode study when resuming a directory.
python - "$ARTIFACT_DIR/evaluation_plan.json" "$CHECKPOINT" "$EPISODES" "${METHODS[@]}" <<'PYPLAN'
import hashlib, json, sys
from pathlib import Path
path,checkpoint,episodes,*methods=sys.argv[1:]
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan={'checkpoint_sha256':sha(checkpoint),
      'training_env_sha256':sha('configs/env/rover_team_12.yaml'),
      'evaluation_env_sha256':sha('configs/env/rover_team_12_showcase.yaml'),
      'episodes':int(episodes),'seed':901110,'methods':methods,
      'forecast_horizon':24}
p=Path(path)
if p.exists():
    if json.loads(p.read_text())!=plan:
        raise SystemExit(f'Mismatched restart settings; refuse mixing experiments in {p.parent}')
else:
    p.write_text(json.dumps(plan,indent=2)+'\n')
PYPLAN
# Each independent method report remains on disk, preserving successful work
# when a later CPU-intensive allocator evaluation is interrupted.
for METHOD in "${METHODS[@]}"; do
  PART="$ARTIFACT_DIR/evaluation_parts/$METHOD"
  if [[ -s "$PART/allocator_report.json" ]]; then
    echo "[evaluation] Completed earlier: $METHOD"
    continue
  fi
  if [[ -e "$PART" ]]; then
    echo "Incomplete method folder, inspect and delete that folder to retry: $PART" >&2
    exit 2
  fi
  echo "[evaluation] $METHOD ($EPISODES episodes)..."
  python -m dart_marl evaluate-allocators \
    --env-config configs/env/rover_team_12_showcase.yaml \
    --controller checkpoint --checkpoint "$CHECKPOINT" \
    --training-env-config configs/env/rover_team_12.yaml \
    --methods "$METHOD" --episodes "$EPISODES" --seed 901110 \
    --forecast-horizon 24 --device cpu --out "$PART" >/dev/null
  echo "[evaluation] Saved $METHOD"
done
if [[ ! -s "$ARTIFACT_DIR/evaluation/allocator_report.json" ]]; then
  python scripts/merge_allocator_reports.py \
    --parts "$ARTIFACT_DIR/evaluation_parts" \
    --methods "${METHODS[@]}" \
    --out "$ARTIFACT_DIR/evaluation/allocator_report.json"
fi
# Preserve completed renders across restarts; add the fixed-role comparison
# without invalidating the original R-CART evaluation or its paper manifest.
if [[ -s "$ARTIFACT_DIR/showcase_verification.json" && \
      -s "$ARTIFACT_DIR/fixed_demo_verification.json" && \
      -s "$ARTIFACT_DIR/paper_assets/manifest.json" ]]; then
  echo "Previously completed: $ARTIFACT_DIR"
  exit 0
fi
if [[ ! -s "$ARTIFACT_DIR/rover_recovery_12.gif" || \
      ! -s "$ARTIFACT_DIR/rover_recovery_12.mp4" || \
      ! -s "$ARTIFACT_DIR/rover_recovery_12_trace.json" ]]; then
  rm -f "$ARTIFACT_DIR/rover_recovery_12.gif" \
        "$ARTIFACT_DIR/rover_recovery_12.mp4" \
        "$ARTIFACT_DIR/rover_recovery_12_trace.json" \
        "$ARTIFACT_DIR/showcase_verification.json"
  echo '[demo] Rendering real-policy Switch-Aware versus R-CART...'
  python scripts/make_demo.py \
    --checkpoint "$CHECKPOINT" \
    --training-env-config configs/env/rover_team_12.yaml \
    --env-config configs/env/rover_team_12_showcase.yaml \
    --compare r_cart --seed 901110 --forecast-horizon 24 \
    --interpolation 4 --event-hold 4 --frame-ms 175 \
    --out "$ARTIFACT_DIR/rover_recovery_12.gif" \
    --mp4 "$ARTIFACT_DIR/rover_recovery_12.mp4" \
    --trace "$ARTIFACT_DIR/rover_recovery_12_trace.json"
fi
[[ -s "$ARTIFACT_DIR/rover_recovery_12.mp4" ]] || {
  echo 'MP4 missing: FFmpeg rendering did not complete.' >&2; exit 2;
}
if [[ ! -s "$ARTIFACT_DIR/showcase_verification.json" ]]; then
  python scripts/verify_showcase.py \
    --env-config configs/env/rover_team_12_showcase.yaml \
    --checkpoint "$CHECKPOINT" --gif "$ARTIFACT_DIR/rover_recovery_12.gif" \
    --trace "$ARTIFACT_DIR/rover_recovery_12_trace.json" \
    --out "$ARTIFACT_DIR/showcase_verification.json"
fi
if [[ ! -s "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12.gif" || \
      ! -s "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12.mp4" || \
      ! -s "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12_trace.json" ]]; then
  rm -f "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12.gif" \
        "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12.mp4" \
        "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12_trace.json" \
        "$ARTIFACT_DIR/fixed_demo_verification.json"
  echo '[demo] Rendering real-policy Switch-Aware versus Fixed...'
  python scripts/make_demo.py \
    --checkpoint "$CHECKPOINT" \
    --training-env-config configs/env/rover_team_12.yaml \
    --env-config configs/env/rover_team_12_showcase.yaml \
    --compare fixed --seed 901110 --forecast-horizon 24 \
    --interpolation 4 --event-hold 4 --frame-ms 175 \
    --out "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12.gif" \
    --mp4 "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12.mp4" \
    --trace "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12_trace.json"
fi
[[ -s "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12.mp4" ]] || {
  echo 'Fixed-comparison MP4 missing: FFmpeg did not complete.' >&2; exit 2;
}
if [[ ! -s "$ARTIFACT_DIR/fixed_demo_verification.json" ]]; then
  python scripts/verify_showcase.py \
    --env-config configs/env/rover_team_12_showcase.yaml \
    --checkpoint "$CHECKPOINT" --gif "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12.gif" \
    --trace "$ARTIFACT_DIR/rover_reallocation_vs_fixed_12_trace.json" \
    --compare fixed --out "$ARTIFACT_DIR/fixed_demo_verification.json"
fi
if [[ -d "$ARTIFACT_DIR/paper_assets" && ! -s "$ARTIFACT_DIR/paper_assets/manifest.json" ]]; then
  rm -rf "$ARTIFACT_DIR/paper_assets"
fi
if [[ ! -s "$ARTIFACT_DIR/paper_assets/manifest.json" ]]; then
  echo '[paper] Creating PNG/PDF/SVG figures and tables...'
  python scripts/make_paper_assets.py \
    --training "$TRAIN_DIR" \
    --evaluation "$ARTIFACT_DIR/evaluation/allocator_report.json" \
    --trace "$ARTIFACT_DIR/rover_recovery_12_trace.json" \
    --historical-csv results/heldout_episode_returns.csv \
    --out "$ARTIFACT_DIR/paper_assets"
fi
echo "All completed artifacts: $ARTIFACT_DIR"
