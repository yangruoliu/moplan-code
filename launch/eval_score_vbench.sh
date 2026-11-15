#!/usr/bin/env bash
# VBench-I2V (all 1,118 suite entries incl. the 763 camera_motion prompts, 5 videos each) with the official VBench code
# (Vchitect/VBench @fd18b3d, vbench2_beta_i2v), crop ratio 7-4 (= the 448x256 work frames).
#   export   scripts/bench/export_videos.py vbench: <prompt>-<k>.mp4, k = 0..4, 448x256 h264
#   shards   scripts/bench/vbench_score_br.py shards: per dimension, symlink folders of at most SIZES videos
#   units    vbench_score_br.py run --dimensions <dim> per shard (VBenchI2V.evaluate, mode vbench_standard), spread over the GPUs
#   merge    vbench_score_br.py merge: one <dim>_eval_results.json per dimension
#   summary  vbench_score_br.py summary: dimension means and Total / I2V / Quality exactly like scripts/cal_i2v_final_score.py
#
# usage: bash launch/eval_score_vbench.sh MODEL
# env:   VBENCH_HOME (checkout with .venv), VBENCH_CACHE_DIR (pre-fetched VBench model weights; the scorer runs offline), GPUS,
#        VBENCH_TORCH_HOME [$VBENCH_CACHE_DIR/torch_home] (TORCH_HOME of the scorer: torch.hub tree with
#        facebookresearch/co-tracker @82e02e8 as hub/facebookresearch_co-tracker_main and hub/checkpoints/cotracker2.pth,
#        which VBench's camera_motion loads through torch.hub)
# outputs: $EVAL_ROOT/vbench/{videos,shards,units,results}/<MODEL>; results/<MODEL>/summary.json
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
EVAL_ROOT=$(d=${EVAL_ROOT:-$REPO/eval_work}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)    # absolute: the steps below cd
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
MODEL=${1:?model label}
: "${VBENCH_HOME:?set VBENCH_HOME (VBench checkout)}"
: "${VBENCH_CACHE_DIR:?set VBENCH_CACHE_DIR (VBench model weights)}"
VBENCH_HOME=$(CDPATH= cd -- "$VBENCH_HOME" && pwd) || { echo "VBENCH_HOME is not a folder" >&2; exit 1; }
VBENCH_CACHE_DIR=$(CDPATH= cd -- "$VBENCH_CACHE_DIR" && pwd) || { echo "VBENCH_CACHE_DIR is not a folder" >&2; exit 1; }
VBENCH_TORCH_HOME=${VBENCH_TORCH_HOME:-$VBENCH_CACHE_DIR/torch_home}
B=$EVAL_ROOT/vbench; G=$B/gen/$MODEL; IN=$B/inputs; L=$G/logs; mkdir -p "$L"
V=$B/videos/$MODEL; S=$B/shards/$MODEL; U=$B/units/$MODEL; OUT=$B/results/$MODEL
SC=$REPO/scripts/bench/vbench_score_br.py
SIZES=${SIZES:-camera_motion=400,i2v_background=220,dynamic_degree=300,motion_smoothness=500,imaging_quality=500,i2v_subject=500,subject_consistency=500,aesthetic_quality=720,background_consistency=440}
[ -f "$G/GEN_COMPLETE" ] || { echo "generation incomplete: $G" >&2; exit 1; }
case $MODEL in base) ARM=base ;; *) ARM=pred ;; esac
if [ -n "${GPUS:-}" ]; then gpus=($GPUS); else gpus=($(nvidia-smi --query-gpu=index --format=csv,noheader)); fi
N=${#gpus[@]}; [ "$N" -gt 0 ] || { echo "no GPU visible" >&2; exit 1; }
vpy() { ( cd "$VBENCH_HOME" && unset PYTHONPATH && export PYTHONPATH=$VBENCH_HOME VBENCH_CACHE_DIR TORCH_HOME=$VBENCH_TORCH_HOME \
          HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=8 && .venv/bin/python -W ignore "$SC" "$@" ); }

# 1 export (5 videos per entry) + shards
(cd "$REPO" && $PYTHON scripts/bench/export_videos.py vbench --gen "$G" --manifest "$IN/manifest.jsonl" --arm "$ARM" --model "$MODEL" \
   --root "$B/videos" --num-seeds 5 --resume) > "$L/vbench_export.log" 2>&1
want=$(( $(wc -l < "$IN/ids_all.txt") * 5 ))
have=$(find "$V" -maxdepth 1 -name "*.mp4" | wc -l | tr -d ' ')
[ "$have" -eq "$want" ] || { echo "$V holds $have videos, want $want" >&2; exit 1; }
rm -rf "$S"; vpy shards --videos "$V" --root "$S" --sizes "$SIZES" > "$L/vbench_shards.log" 2>&1

# 2 units, dealt round-robin over the GPUs (one process per GPU, its units in sequence; finished units are skipped)
mkdir -p "$U"
pids=(); rc=0
for s in $(seq 0 $((N - 1))); do
  (
    k=0
    while read -r dim shard n; do
      [ -n "$dim" ] || continue
      if [ $((k % N)) -eq "$s" ] && [ ! -f "$U/${dim}_$shard/${dim}_eval_results.json" ]; then
        CUDA_VISIBLE_DEVICES=${gpus[$s]} vpy run --videos "$S/$dim/$shard" --out "$U/${dim}_$shard" --ratio 7-4 --dimensions "$dim" \
          > "$U/${dim}_$shard.log" 2>&1
      fi
      k=$((k + 1))
    done < "$S/units.txt"
  ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" = 0 ] || { echo "a VBench unit failed (see $U/*.log)" >&2; exit 1; }
while read -r dim shard n; do
  [ -f "$U/${dim}_$shard/${dim}_eval_results.json" ] || { echo "unit $dim $shard not scored" >&2; exit 1; }
done < "$S/units.txt"

# 3 merge + summary
rm -rf "$OUT"; vpy merge --units "$U" --out "$OUT" > "$L/vbench_merge.log" 2>&1
vpy summary --videos "$V" --out "$OUT" --ratio 7-4 --model "$MODEL" > "$L/vbench_summary.log" 2>&1
tail -n 1 "$L/vbench_summary.log"
echo "[vbench] $MODEL scored: $OUT/summary.json"
