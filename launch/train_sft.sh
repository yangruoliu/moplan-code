#!/usr/bin/env bash
# Stage-2 GT-condition SFT of one Wan2.2-I2V-A14B expert on one node (torchrun), from the base model.
#
# The released generator was fine-tuned (RAFT) from checkpoint-004000 of both experts: step 4000 of an 8000-step
# cosine schedule.  That checkpoint is not distributed: the RAFT round (launch/raft.sh) starts from the one this
# launcher writes.  configs/stage2/sft_overrides.json therefore keeps max_steps 8000, and this launcher stops at
# STOP_AT_STEP=4000.  The reference run used 8 H100 80GB GPUs per expert with global batch 16 (8 GPUs x 2 gradient
# accumulation steps).  Any NPROC that divides 16 trains on the same sequence of samples (clip, sigma, noise); only the
# floating-point reduction order across GPUs differs.  Run the launcher once per expert (EXPERT=high and EXPERT=low);
# the two runs are independent and can run at the same time on different GPUs or nodes.
#
# usage: EXPERT=high NPROC=8 launch/train_sft.sh [extra trainer arguments]
#
# Relaunching the same command resumes from the newest checkpoint (--resume-from latest).  To stop cleanly, create
# $RUN_DIR/STOP_<expert> (or send SIGUSR1 to the worker processes): the run writes a checkpoint after the current
# optimizer step and exits.
#
# environment
#   WAN_MODEL_DIR    (required) Wan2.2-I2V-A14B-Diffusers directory (transformer/ = high, transformer_2/ = low, vae/)
#   MOPLAN_SFT_DATA  (required with the released overrides) data root with pools/{p101,ssv2,phyco}/<sid>.pt and
#                    val/{p101,phyco,ssv2}/<sid>.pt, see docs/DATA_STAGE2.md
#   EXPERT           high or low (default high)
#   NPROC            GPUs on this node (default: every visible GPU); must divide the global batch (16)
#   MOPLAN_WORK_DIR  output root (default <repo>/work)
#   RUN_DIR          run directory (default $MOPLAN_WORK_DIR/runs/stage2_sft); checkpoints go to $RUN_DIR/<expert>
#   STOP_AT_STEP     last optimizer step of this launch (default 4000; 0 = continue to max_steps)
#   CONFIG           default configs/stage2/sft.yaml
#   OVERRIDES        default $RUN_DIR/OVERRIDES.json if it exists (written by make_pools_v2.py), else
#                    configs/stage2/sft_overrides.json
#   MASTER_PORT      rendezvous port (default 29500; give two runs on one node different ports)
#
# Output: $RUN_DIR/<expert>/checkpoint-NNNNNN/{lora_params.pt,motion_adapter.pt,optimizer.pt,scheduler.pt,
# trainer_state.json}; every 500th checkpoint is kept, the others are resume points.  When step STOP_AT_STEP is
# reached, $RUN_DIR/sft_step<STOP_AT_STEP>/<expert> links to its checkpoint (the MOPLAN_SFT_CHECKPOINT layout of the
# RAFT round, docs/TRAINING_SFT.md).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${WAN_MODEL_DIR:?set WAN_MODEL_DIR to the Wan2.2-I2V-A14B-Diffusers directory}"
export WAN_MODEL_DIR
EXPERT=${EXPERT:-high}
case "$EXPERT" in high|low) ;; *) echo "EXPERT must be high or low" >&2; exit 2 ;; esac
export MOPLAN_WORK_DIR=${MOPLAN_WORK_DIR:-$REPO/work}
RUN_DIR=${RUN_DIR:-$MOPLAN_WORK_DIR/runs/stage2_sft}
STOP_AT_STEP=${STOP_AT_STEP:-4000}
CONFIG=${CONFIG:-$REPO/configs/stage2/sft.yaml}
MASTER_PORT=${MASTER_PORT:-29500}
if [[ -z "${OVERRIDES:-}" ]]; then
  if [[ -f "$RUN_DIR/OVERRIDES.json" ]]; then OVERRIDES=$RUN_DIR/OVERRIDES.json; else OVERRIDES=$REPO/configs/stage2/sft_overrides.json; fi
fi
if grep -q '{data_root}' "$OVERRIDES"; then
  : "${MOPLAN_SFT_DATA:?set MOPLAN_SFT_DATA to the GT-SFT data root (pools/ and val/), see docs/DATA_STAGE2.md}"
fi
if [[ -z "${NPROC:-}" ]]; then
  NPROC=$(python -c 'import torch; print(torch.cuda.device_count())')
fi
if (( NPROC < 1 )); then echo "no GPU visible" >&2; exit 2; fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

mkdir -p "$RUN_DIR/logs"
STOP_FILE=$RUN_DIR/STOP_$EXPERT
rm -f "$STOP_FILE"
DATA_ARGS=()
if [[ -n "${MOPLAN_SFT_DATA:-}" ]]; then DATA_ARGS=(--data-root "$MOPLAN_SFT_DATA"); fi

cd "$REPO"
echo "[train_sft] expert=$EXPERT nproc=$NPROC run=$RUN_DIR overrides=$OVERRIDES stop_at_step=$STOP_AT_STEP"
torchrun --nnodes=1 --nproc_per_node="$NPROC" --master_addr=127.0.0.1 --master_port="$MASTER_PORT" --max_restarts=0 \
  scripts/s2sft/v2/train_s2sft_v2.py --config "$CONFIG" --expert "$EXPERT" --output-dir "$RUN_DIR" \
  --overrides "$OVERRIDES" ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} --wan-model-dir "$WAN_MODEL_DIR" \
  --resume-from latest --stop-at-step "$STOP_AT_STEP" --stop-file "$STOP_FILE" "$@" \
  2>&1 | tee -a "$RUN_DIR/logs/train_$EXPERT.log"

if (( STOP_AT_STEP > 0 )); then
  tag=$(printf '%06d' "$STOP_AT_STEP")
  if [[ -f "$RUN_DIR/$EXPERT/checkpoint-$tag/lora_params.pt" ]]; then
    mkdir -p "$RUN_DIR/sft_step$tag"
    ln -sfn "../$EXPERT/checkpoint-$tag" "$RUN_DIR/sft_step$tag/$EXPERT"
    echo "[train_sft] $RUN_DIR/sft_step$tag/$EXPERT -> $EXPERT/checkpoint-$tag"
  fi
fi
