#!/usr/bin/env bash
# Single-node Stage-1 training with torchrun.
#
# The released Stage-1 checkpoint was trained with 8 GPUs (H100 80GB),
# 1 sample per GPU and 4 gradient-accumulation steps (global batch 32) for
# 6000 optimizer steps. NPROC=8 reproduces that setting. NPROC=4 keeps the
# global batch at 32 with 8 accumulation steps; it is a valid run of the same
# recipe, but its data order across ranks differs from the reference run.
#
# usage: launch/train_stage1.sh [extra trainer arguments]
#   e.g. launch/train_stage1.sh --resume-from "$MOPLAN_WORK_DIR/runs/stage1/high/checkpoint-002000"
#
# environment:
#   MOPLAN_DATA      (required) built Stage-1 data root, see docs/DATA_STAGE1.md
#   WAN_MODEL_DIR    (required) Wan2.2-I2V-A14B-Diffusers directory
#   MOPLAN_WORK_DIR  output root (default: <repo>/work); the run is written to
#                    $MOPLAN_WORK_DIR/runs/stage1/high
#   NPROC            GPUs on this node: 8 (default) or 4
#   CONFIG           training config (default: configs/stage1/train.yaml)
#   MASTER_PORT      rendezvous port (default: 29500)
#
# Sending SIGUSR1 to the worker processes (not to torchrun), for example
#   pkill -USR1 -f -- '-u scripts/train_stage1_flowdit.py'
# (torchrun starts the workers as `python -u scripts/...`; a plain `pkill -f scripts/train_stage1_flowdit.py` also
# matches the torchrun agent, which SIGUSR1 kills at once) makes them write an exact-resume checkpoint after the current optimizer step
# and exit. Restart with --resume-from <checkpoint>, the same NPROC and the
# same MOPLAN_DATA / WAN_MODEL_DIR / MOPLAN_WORK_DIR values (the resolved
# config is part of the resume contract).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${MOPLAN_DATA:?set MOPLAN_DATA to the built Stage-1 data root}"
: "${WAN_MODEL_DIR:?set WAN_MODEL_DIR to the Wan2.2-I2V-A14B-Diffusers directory}"
export MOPLAN_DATA WAN_MODEL_DIR
export MOPLAN_WORK_DIR=${MOPLAN_WORK_DIR:-$REPO/work}
NPROC=${NPROC:-8}
CONFIG=${CONFIG:-$REPO/configs/stage1/train.yaml}
MASTER_PORT=${MASTER_PORT:-29500}

# The config pins world_size 8 with grad_accum_steps 4 (global batch 32).
case "$NPROC" in
  8) RANK_OVERRIDES=() ;;
  4) RANK_OVERRIDES=(--set unified_experiment.world_size=4 --set train.grad_accum_steps=8) ;;
  *) echo "NPROC must be 8 or 4; use scripts/stage1_smoke.py for a 1-GPU smoke test" >&2; exit 2 ;;
esac

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

cd "$REPO"
mkdir -p "$MOPLAN_WORK_DIR"
exec torchrun --nnodes=1 --nproc_per_node="$NPROC" --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" --max_restarts=0 \
  scripts/train_stage1_flowdit.py --config "$CONFIG" \
  ${RANK_OVERRIDES[@]+"${RANK_OVERRIDES[@]}"} "$@"
