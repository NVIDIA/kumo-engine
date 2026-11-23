#!/usr/bin/env bash
# Blueprint v2 final sweep (benchmarks/results/blueprint_v2_final_sweep.md):
# candidates A, B, C, D at seeds 42, 43, 44, run one after another. C and D
# reuse A's and B's checkpoints and only re-run evaluation with LoRA rank 16.
# Run it from a clean checkout; the record's runs used commit 124ab93.
#
#   export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
#   bash benchmarks/blueprint_v2_final_sweep.sh [A B C D]
#   uv run python benchmarks/blueprint_v2_final_sweep.py outputs/final_sweep
#
# TFM_SWEEP_CHECKOUT (default: this repository) is the checkout to run from;
# TFM_SWEEP_ROOT (default: <checkout>/outputs/final_sweep) holds the runs.
set -euo pipefail

CHECKOUT=${TFM_SWEEP_CHECKOUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${TFM_SWEEP_ROOT:-$CHECKOUT/outputs/final_sweep}
LOG=$ROOT/queue.log
: "${TFM_PRAGMATIQ_ROOT:?set TFM_PRAGMATIQ_ROOT to the synth_100k_gen2_m31_seed0 book}"
export TFM_PRAGMATIQ_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

B_STEPS=(--set training.max_steps=3000 --set training.optimizer.warmup_steps=150)
LORA16=(--set fraud_task.finetune.rank=16 --set fraud_task.finetune.alpha=16.0)

cd "$CHECKOUT"
mkdir -p "$ROOT"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

sweep() {
  local name=$1
  shift
  log "start $name $*"
  uv run --extra baselines python benchmarks/seed_range.py \
    --config recipes/blueprint_v2/config.yaml --seeds 42 43 44 \
    --run-root "$ROOT/$name" --summary "$ROOT/$name.json" --device cuda "$@" \
    >"$ROOT/$name.log" 2>&1
  log "done $name"
}

reuse() {
  local name=$1 source=$2
  shift 2
  if [[ ! -e $ROOT/$name ]]; then
    cp -r "$ROOT/$source" "$ROOT/$name"
  fi
  sweep "$name" --reuse-checkpoint "$@"
}

for candidate in "${@:-A B C D}"; do
  for c in $candidate; do
    case $c in
      A) sweep A ;;
      B) sweep B "${B_STEPS[@]}" ;;
      C) reuse C A "${LORA16[@]}" ;;
      D) reuse D B "${B_STEPS[@]}" "${LORA16[@]}" ;;
      *) echo "unknown candidate $c" >&2; exit 2 ;;
    esac
  done
done
log "queue finished"
