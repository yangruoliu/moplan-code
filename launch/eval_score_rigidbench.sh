#!/usr/bin/env bash
# RigidBench with the OFFICIAL evaluator (swarnim-j/RigidBench @f0b8f29, its own venv, vendor/Video-Depth-Anything @4f5ae23).
#   export   scripts/bench/export_videos.py rigidbench: the 448x256 work frames stretched to 1280x704, all 17 frames @7 fps
#   prepare  rigidbench.eval.prepare.prepare_videos (once per model)
#   metrics  rigidbench.eval.pipeline.EvalPipeline, the samples dealt round-robin over the GPUs (per-sample artifacts)
#   score    `rigidbench evaluate` on all 100 samples (everything cached -> official aggregation, results.json, run.json)
#
# usage: bash launch/eval_score_rigidbench.sh MODEL
# env:   RIGIDBENCH_HOME (checkout with .venv and data/ from `rigidbench download`), GPUS
#        RIGIDBENCH_TORCH_HOME [$TORCH_HOME, else torch's default cache]: TORCH_HOME of the scorer.  The evaluator loads
#        DINOv2 and CoTracker3 through torch.hub and AlexNet (LPIPS) through torchvision; with the pinned checkouts in its
#        hub/ folder torch.hub uses them instead of fetching the current default branches
#        RIGIDBENCH_HF_HOME [$HF_HOME, else the hub's default]: HF_HOME of the scorer (SAM 2.1-L, Video-Depth-Anything-L)
#        The scorer runs offline: the launcher stops before any work when an entry is missing (SKIP_CACHE_CHECK=1 skips
#        this check).  The entries and their reference commits are listed in docs/EVALUATION.md, section 4.
# outputs: $EVAL_ROOT/rigidbench/{predictions/<MODEL>/*.mp4, runs/<MODEL>/{results.json,run.json,metrics/}}
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
EVAL_ROOT=$(d=${EVAL_ROOT:-$REPO/eval_work}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)    # absolute: the steps below cd
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
MODEL=${1:?model label}
: "${RIGIDBENCH_HOME:?set RIGIDBENCH_HOME (RigidBench checkout)}"
RIGIDBENCH_HOME=$(CDPATH= cd -- "$RIGIDBENCH_HOME" && pwd) || { echo "RIGIDBENCH_HOME is not a folder" >&2; exit 1; }
RIGIDBENCH_TORCH_HOME=${RIGIDBENCH_TORCH_HOME:-${TORCH_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/torch}}
RIGIDBENCH_HF_HOME=${RIGIDBENCH_HF_HOME:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}}
R=$RIGIDBENCH_HOME; B=$EVAL_ROOT/rigidbench; G=$B/gen/$MODEL; IN=$B/inputs; L=$G/logs
PR=$B/predictions; RUNS=$B/runs; mkdir -p "$PR" "$RUNS" "$L"
[ -f "$G/GEN_COMPLETE" ] || { echo "generation incomplete: $G" >&2; exit 1; }
case $MODEL in base) ARM=base ;; *) ARM=pred ;; esac
if [ -n "${GPUS:-}" ]; then gpus=($GPUS); else gpus=($(nvidia-smi --query-gpu=index --format=csv,noheader)); fi
N=${#gpus[@]}; [ "$N" -gt 0 ] || { echo "no GPU visible" >&2; exit 1; }
NCPU=$(nproc 2> /dev/null || echo 8)
# the scorer's environment: its own venv, offline, the caches above
renv() { unset PYTHONPATH HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
         export PYTHONPATH=$R/vendor/Video-Depth-Anything TORCH_HOME=$RIGIDBENCH_TORCH_HOME HF_HOME=$RIGIDBENCH_HF_HOME \
                HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1; }
RPY() { ( cd "$R" && renv && export OMP_NUM_THREADS=$(( NCPU / N > 0 ? NCPU / N : 1 )) && .venv/bin/python "$@" ); }

# 0 offline caches: fail before any work instead of in the middle of the scoring
if [ "${SKIP_CACHE_CHECK:-0}" != 1 ]; then
  TH=$RIGIDBENCH_TORCH_HOME/hub; HH=$RIGIDBENCH_HF_HOME/hub; miss=()
  for f in facebookresearch_co-tracker_main/hubconf.py facebookresearch_dinov2_main/hubconf.py checkpoints/scaled_offline.pth \
           checkpoints/dinov2_vitl14_pretrain.pth checkpoints/alexnet-owt-7be5be79.pth; do
    [ -s "$TH/$f" ] || miss+=("$TH/$f")
  done
  for rf in models--facebook--sam2.1-hiera-large:sam2.1_hiera_large.pt \
            models--depth-anything--Video-Depth-Anything-Large:video_depth_anything_vitl.pth; do
    repo=${rf%%:*}; f=${rf#*:}
    [ -s "$HH/$repo/refs/main" ] || miss+=("$HH/$repo/refs/main")
    compgen -G "$HH/$repo/snapshots/*/$f" > /dev/null || miss+=("$HH/$repo/snapshots/<rev>/$f")
  done
  if [ ${#miss[@]} -gt 0 ]; then
    { echo "the RigidBench scorer runs offline, but these files are missing (see docs/EVALUATION.md, section 4):"; printf '  %s\n' "${miss[@]}"; } >&2
    exit 1
  fi
fi

# 1 export + prepare, once per model
if [ "$(cat "$G/RIGID_EXPORTED" 2>/dev/null)" != "$MODEL" ]; then
  rm -f "$PR/$MODEL"/*.tmp.mp4
  (cd "$REPO" && $PYTHON scripts/bench/export_videos.py rigidbench --gen "$G" --manifest "$IN/manifest.jsonl" --arm "$ARM" \
     --model "$MODEL" --predictions "$PR") > "$L/rigid_export.log" 2>&1
  RPY - "$PR/$MODEL" "$MODEL" "$RUNS" > "$L/rigid_prepare.log" 2>&1 <<'EOF'
import sys
from pathlib import Path
from rigidbench.data import find_data_root
from rigidbench.eval.prepare import prepare_videos
from rigidbench.eval.run import _select_samples
pred, model, runs = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
root = find_data_root(Path("data"))
available, selected = _select_samples(root, "eval", None, None, None)
prepared = prepare_videos(pred, model=model, output_dir=runs, expected_sample_ids=[s.id for s in available],
                          selected_sample_ids=[s.id for s in selected])
print(f"prepared {len(prepared.records)} videos at {prepared.fps:g} fps -> {prepared.output_dir}")
EOF
  echo "$MODEL" > "$G/RIGID_EXPORTED"
fi
SEL=$(cd "$PR/$MODEL" && ls *.mp4 | sed 's/\.mp4$//' | sort | paste -sd, -)

# 2 trackers + metrics, the samples dealt round-robin over the GPUs
pids=(); rc=0
for s in $(seq 0 $((N - 1))); do
  ids_s=$(echo "$SEL" | tr ',' '\n' | awk -v s="$s" -v n="$N" 'NR % n == s % n' | paste -sd, -)
  [ -n "$ids_s" ] || continue
  CUDA_VISIBLE_DEVICES=${gpus[$s]} RPY - "$MODEL" "$RUNS" "$ids_s" > "$L/rigid_eval_$s.log" 2>&1 <<'EOF' & pids+=($!)
import sys
from pathlib import Path
from rigidbench.data import find_data_root
from rigidbench.eval.pipeline import EvalPipeline
from rigidbench.eval.run import _select_samples
model, runs, ids = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
root = find_data_root(Path("data"))
_, samples = _select_samples(root, "eval", None, ids, None)
EvalPipeline(model, str(root), runs, None, "eval", None, None).run(samples=samples, force=False).raise_for_failures()
print(f"shard done: {len(samples)} samples")
EOF
done
for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || rc=1; done
[ "$rc" = 0 ] || { echo "evaluator shard failed (see $L/rigid_eval_*.log)" >&2; exit 1; }

# 3 the official CLI over the whole set: nothing left to compute -> aggregation, results.json, run.json
( cd "$R" && renv && .venv/bin/rigidbench evaluate "$PR/$MODEL" --data data --output "$RUNS" --name "$MODEL" ) > "$L/rigid_official.log" 2>&1
grep -E "^IOU=" "$L/rigid_official.log" || true
echo "[rigidbench] $MODEL scored: $RUNS/$MODEL/results.json"
