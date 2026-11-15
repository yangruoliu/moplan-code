# Shared settings of the inference launchers (sourced by launch/infer_*.sh, not run directly).
#
# Environment (defaults in brackets):
#   WAN_MODEL_DIR    Wan2.2-I2V-A14B-Diffusers directory (required)
#   MOPLAN_WEIGHTS   downloaded MoPlan weights: stage1/, stage2/high, stage2/low (required unless BASE=1)
#   SAM2_CHECKPOINT  SAM 2.1 hiera-large checkpoint (only for steps that run SAM 2)
#   PYTHON           python of the MoPlan environment [python]
#   GPUS             space-separated GPU indices [every GPU nvidia-smi lists]
#   OMP_NUM_THREADS  CPU threads [8, as in the reference runs]
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
: "${WAN_MODEL_DIR:?set WAN_MODEL_DIR to the Wan2.2-I2V-A14B-Diffusers directory}"
export WAN_MODEL_DIR
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
S2_CONFIG=${S2_CONFIG:-$REPO/configs/stage2/inference.yaml}
# the 7 Stage-1 query horizons (seconds) of every export
QUERY_HORIZONS=0.20833333333333334,0.4166666666666667,0.625,0.8333333333333334,1.2,2,4

log() { echo "$(date '+%F %T') $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
need() { for f in "$@"; do [ -e "$f" ] || die "missing $f"; done; }

need_weights() {
  : "${MOPLAN_WEIGHTS:?set MOPLAN_WEIGHTS to the downloaded MoPlan weights}"
  STAGE1_CKPT=${STAGE1_CKPT:-$MOPLAN_WEIGHTS/stage1}
  HIGH_CKPT=${HIGH_CKPT:-$MOPLAN_WEIGHTS/stage2/high}
  LOW_CKPT=${LOW_CKPT:-$MOPLAN_WEIGHTS/stage2/low}
}

gpu_list() {
  if [ -n "${GPUS:-}" ]; then echo "$GPUS"; else nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '; fi
}

# run_sharded NAME LOGDIR CMD...: one process per GPU with --shard-index i --num-shards n appended; fails if any fails
run_sharded() {
  local name=$1 logdir=$2; shift 2
  local gpus=($(gpu_list)) n s rc=0 pids=()
  n=${#gpus[@]}
  [ "$n" -gt 0 ] || die "no GPU visible"
  mkdir -p "$logdir"
  for s in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${gpus[$s]} "$@" --shard-index "$s" --num-shards "$n" > "$logdir/${name}_$s.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" = 0 ] || die "$name failed (see $logdir/${name}_*.log)"
}

first_gpu() { gpu_list | awk '{print $1}'; }
