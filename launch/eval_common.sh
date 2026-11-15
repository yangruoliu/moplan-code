# Shared settings of the evaluation launchers (sourced by launch/eval_*.sh, not run directly).
#
# Required environment:
#   WAN_MODEL_DIR    Wan2.2-I2V-A14B-Diffusers directory
#   MOPLAN_WEIGHTS   downloaded MoPlan weights (stage1/, stage2/high, stage2/low)
# Optional environment (defaults in brackets):
#   PYTHON           python of the MoPlan environment [python]
#   EVAL_ROOT        working folder for every evaluation input and output [<repo>/eval_work]
#   STAGE1_CKPT      Stage-1 checkpoint folder [$MOPLAN_WEIGHTS/stage1]
#   STAGE1_CONFIG    Stage-1 config [$STAGE1_CKPT/resolved_config.yaml]
#   PRIOR_LATENT     Stage-1 prior latent [$STAGE1_CKPT/prior_latent.pt]
#   STAGE2_CKPT      generator folder with high/ and low/ (lora_params.pt, motion_adapter.pt) [$MOPLAN_WEIGHTS/stage2]
#   S2_CONFIG        Stage-2 inference config [<repo>/configs/stage2/inference.yaml]
#   GPUS             space-separated GPU indices to use [all visible GPUs]

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
EVAL_ROOT=$(d=${EVAL_ROOT:-$REPO/eval_work}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)    # absolute: the launchers cd
: "${WAN_MODEL_DIR:?set WAN_MODEL_DIR to the Wan2.2-I2V-A14B-Diffusers directory}"
: "${MOPLAN_WEIGHTS:?set MOPLAN_WEIGHTS to the downloaded MoPlan weights}"
STAGE1_CKPT=${STAGE1_CKPT:-$MOPLAN_WEIGHTS/stage1}
STAGE1_CONFIG=${STAGE1_CONFIG:-$STAGE1_CKPT/resolved_config.yaml}
PRIOR_LATENT=${PRIOR_LATENT:-$STAGE1_CKPT/prior_latent.pt}
STAGE2_CKPT=${STAGE2_CKPT:-$MOPLAN_WEIGHTS/stage2}
S2_CONFIG=${S2_CONFIG:-$REPO/configs/stage2/inference.yaml}
export WAN_MODEL_DIR MOPLAN_WEIGHTS
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Stage-1 query horizons (seconds) of every export
QUERY_HORIZONS=0.20833333333333334,0.4166666666666667,0.625,0.8333333333333334,1.2,2,4

log() { echo "$(date '+%F %T') $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
need_file() { for f in "$@"; do [ -e "$f" ] || die "missing $f"; done; }
# count_files DIR GLOB: number of regular files matching GLOB directly in DIR (0 if DIR does not exist)
count_files() { if [ -d "$1" ]; then find "$1" -maxdepth 1 -type f -name "$2" | wc -l | tr -d ' '; else echo 0; fi; }

# GPU list: $GPUS, else every GPU nvidia-smi reports
gpu_list() {
  if [ -n "${GPUS:-}" ]; then echo "$GPUS"; else nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '; fi
}

# run_sharded NAME LOGDIR CMD...: one process per GPU with --shard-index i --num-shards n appended; fails if any fails
run_sharded() {
  local name=$1 logdir=$2; shift 2
  local gpus=($(gpu_list)) n pids=() rc=0 s
  n=${#gpus[@]}
  [ "$n" -gt 0 ] || die "no GPU visible"
  mkdir -p "$logdir"
  for s in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${gpus[$s]} "$@" --shard-index "$s" --num-shards "$n" > "$logdir/${name}_$s.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" = 0 ] || die "$name failed (see $logdir/${name}_*.log)"
}

# Benchmark name -> internal set key (zero-plan cache layout) and clip length at 7 fps
bench_key() {
  case $1 in
    pisabench) echo pisafull ;;
    rigidbench) echo rigido ;;
    morpheus) echo morpho ;;
    vbench) echo vbfull ;;
    moprobe_h) echo miqB ;;
    *) die "unknown benchmark $1 (pisabench | rigidbench | morpheus | vbench | moprobe_h)" ;;
  esac
}
bench_frames() { case $1 in rigidbench) echo 17 ;; *) echo 37 ;; esac; }
