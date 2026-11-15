#!/usr/bin/env bash
# Smoke test of the Stage-2 GT-SFT trainer on one GPU with the real Wan2.2-I2V-A14B weights:
#   1. a run directory from a handful of payloads (scripts/s2sft/v2/make_smoke_run.py): per source the first clip is
#      the validation set, the others the training pool; synthetic payloads unless SMOKE_DATA is given
#   2. 2 optimizer steps from the base model (validation at steps 0 and 2, checkpoints at steps 1 and 2)
#   3. a resume from the newest checkpoint to step 3
#   4. checks: finite losses, checkpoint files, nonzero LoRA B, finite validation, done/paused marker
#
# usage: launch/smoke_sft.sh
#
# environment
#   WAN_MODEL_DIR  (required) Wan2.2-I2V-A14B-Diffusers directory
#   EXPERT         high or low (default high)
#   SMOKE_DATA     folder with p101/ (or p101_final/), ssv2/ (or ssv2_final/) and phyco/ folders of stage2_cond_v2
#                  payloads, at least 2 per source used (default: 3 synthetic payloads per source are written to
#                  $SMOKE_DIR/payloads; they exercise the code path, their losses mean nothing)
#   SMOKE_DIR      default <repo>/work/smoke_sft; $SMOKE_DIR/run is removed and rebuilt on every call
#   GLOBAL_BATCH   default 1 (16 = the real global batch: 16 micro-steps per optimizer step on one GPU)
#   GPU            GPU index (default 0)
#   MASTER_PORT    default 29511
# Needs about 41 GiB of GPU memory (gradient checkpointing, bf16); loading the expert takes a few minutes per run.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${WAN_MODEL_DIR:?set WAN_MODEL_DIR to the Wan2.2-I2V-A14B-Diffusers directory}"
export WAN_MODEL_DIR
EXPERT=${EXPERT:-high}
SMOKE_DIR=${SMOKE_DIR:-$REPO/work/smoke_sft}
GLOBAL_BATCH=${GLOBAL_BATCH:-1}
MASTER_PORT=${MASTER_PORT:-29511}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cd "$REPO"

RUN=$SMOKE_DIR/run
rm -rf "$RUN"
mkdir -p "$RUN/logs"
if [[ -n "${SMOKE_DATA:-}" ]]; then
  python scripts/s2sft/v2/make_smoke_run.py "$SMOKE_DATA" "$RUN" > "$RUN/logs/make_smoke_run.log"
else
  rm -rf "$SMOKE_DIR/payloads"
  python scripts/s2sft/v2/make_smoke_run.py --synthetic 3 "$SMOKE_DIR/payloads" "$RUN" > "$RUN/logs/make_smoke_run.log"
fi

train() {
  torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="$MASTER_PORT" --max_restarts=0 \
    scripts/s2sft/v2/train_s2sft_v2.py --config configs/stage2/sft.yaml --expert "$EXPERT" --output-dir "$RUN" \
    --wan-model-dir "$WAN_MODEL_DIR" --global-batch "$GLOBAL_BATCH" --save-every 1 --keep-every 2 --log-every 1 "$@"
}

echo "[smoke_sft] fresh run to step 2 ($EXPERT)"
train --stop-at-step 2 2>&1 | tee "$RUN/logs/train_fresh.log"
python scripts/s2sft/v2/make_smoke_run.py --check "$RUN" --expert "$EXPERT" --step 2
echo "[smoke_sft] resume to step 3"
train --resume-from latest --stop-at-step 3 2>&1 | tee "$RUN/logs/train_resume.log"
grep -q "resume=.*/checkpoint-000002 " "$RUN/logs/train_resume.log" || { echo "[smoke_sft] the second run did not resume from step 2" >&2; exit 1; }
python scripts/s2sft/v2/make_smoke_run.py --check "$RUN" --expert "$EXPERT" --step 3
echo "[smoke_sft] OK: $RUN/$EXPERT"
