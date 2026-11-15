#!/usr/bin/env bash
# PisaBench real-world evaluation with the OFFICIAL code (vision-x-nyu/pisa-experiments @65866af, unmodified, its own venv),
# all 361 clips, 1 video each.
#   export   scripts/bench/pisa_official_export.py export: the original square (work columns 98..350) of every generated
#            448x256 frame -> 256x256 (PIL bicubic), re-timed nearest to 16 fps (the released pd_fps)
#   GT       data_processing/convert_resolution.py --is_real (1080^2 @120 fps -> 256^2, clicks rescaled), from data/real.zip
#   masks    data_processing/generate_mask.py (SAM 2.1 hiera-large, the GT clicks on frame 0 for every object, propagated) on
#            the GT and on the generated frames, sharded over the GPUs, run from $PISA_HOME/run whose ./sam2 is the upstream
#            SAM 2 checkout at c2ec8e1 (the repo's own sam2/ copy is modified and does not run generate_mask.py)
#   score    eval.py (dropping object: centroid L2, Chamfer distance, mask IoU) -> per_video.jsonl, results.txt, summary.json
#
# usage: bash launch/eval_score_pisabench.sh MODEL            (MODEL = a generation folder of launch/eval_bench_generate.sh)
# env:   PISA_HOME    PisaBench working root: code/ (checkout + .venv), data/real.zip, run/sam2 -> upstream sam2 @c2ec8e1
#        PISA_PYTHON  python of the PisaBench venv [$PISA_HOME/code/.venv/bin/python]
#        HF_HOME      Hugging Face cache holding facebook/sam2.1-hiera-large (the scorer runs offline)
#        GT_WORKERS   CPU workers of the GT conversion [16]
# outputs: $EVAL_ROOT/pisabench/official/<MODEL>/{export.json,summary.json,s0/{eval.jsonl,mask.jsonl,results/...}}
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
EVAL_ROOT=$(d=${EVAL_ROOT:-$REPO/eval_work}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)    # absolute: the steps below cd
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
MODEL=${1:?model label}
: "${PISA_HOME:?set PISA_HOME (PisaBench working root)}"
PISA_HOME=$(CDPATH= cd -- "$PISA_HOME" && pwd) || { echo "PISA_HOME is not a folder" >&2; exit 1; }
PYP=${PISA_PYTHON:-$PISA_HOME/code/.venv/bin/python}
B=$EVAL_ROOT/pisabench; G=$B/gen/$MODEL; IN=$B/inputs; PO=$B/official; L=$G/logs; mkdir -p "$PO" "$L"
X=$REPO/scripts/bench/pisa_official_export.py
[ -f "$G/GEN_COMPLETE" ] || { echo "generation incomplete: $G" >&2; exit 1; }
[ -x "$PYP" ] && [ -e "$PISA_HOME/run/sam2" ] || { echo "PisaBench env incomplete ($PYP, $PISA_HOME/run/sam2)" >&2; exit 1; }
case $MODEL in base) ARM=base ;; *) ARM=pred ;; esac

if [ "$(cat "$G/PISA_EXPORTED" 2>/dev/null)" != "$MODEL" ]; then
  echo "[pisabench] export $MODEL"
  $PYTHON "$X" export --gen "$G" --manifest "$IN/manifest.jsonl" --ids-file "$IN/ids_all.txt" --arm "$ARM" --seeds 0 --model "$MODEL" \
    --fps 16 --out-root "$PO" --gt-root "$PISA_HOME/data/real_256" --zip "$PISA_HOME/data/real.zip" > "$L/pisa_export.log" 2>&1
  echo "$MODEL" > "$G/PISA_EXPORTED"
fi
D=$PO/$MODEL/s0
W=$PO/$MODEL/work; mkdir -p "$W/masks"
if [ -n "${GPUS:-}" ]; then gpus=($GPUS); else gpus=($(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null || true)); fi
N=${#gpus[@]}; [ "$N" -gt 0 ] || { echo "no GPU visible" >&2; exit 1; }
(
  unset PYTHONPATH; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
  # 1 GT frames (CPU)
  "$PYP" "$X" gt --dirs "$D" --zip "$PISA_HOME/data/real.zip" --code "$PISA_HOME/code" --workers "${GT_WORKERS:-16}" \
    --tmp "${TMPDIR:-/tmp}" > "$W/gt.log" 2>&1
  "$PYP" "$X" check "$D" --strict > "$W/check_before.json" 2>&1
  # 2 masks (GPU), longest first onto N shards; generate_mask.py logs and skips a failing clip, so re-check afterwards
  "$PYP" "$X" plan-masks "$D" --shards "$N" --out "$W/masks" > "$W/plan.json" 2>&1
  pids=()
  for i in $(seq 0 $((N - 1))); do
    [ -s "$W/masks/shard_$i.jsonl" ] || continue
    ( cd "$PISA_HOME/run" && CUDA_VISIBLE_DEVICES=${gpus[$i]} exec "$PYP" "$PISA_HOME/code/data_processing/generate_mask.py" \
        --config_file "$W/masks/shard_$i.jsonl" --device cuda ) > "$W/masks/shard_$i.log" 2>&1 & pids+=($!)
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
  "$PYP" "$X" check "$D" --require-masks > "$W/check_after.json" 2>&1
  # 3 official eval.py, then the summary
  rm -rf "$D/results"; mkdir -p "$D/results"
  ( cd "$PISA_HOME/code" && exec "$PYP" eval.py --config_file "$D/eval.jsonl" --output_path "$D/results/per_video.jsonl" \
      --task_name "$MODEL/s0" --results_path "$D/results/results.txt" ) > "$D/results/eval.log" 2>&1
  "$PYP" "$X" summarize "$D"
)
n=$(wc -l < "$D/results/per_video.jsonl" | tr -d ' ')
[ "$n" -ge 361 ] || { echo "only $n of 361 clips scored (eval.py skips clips that raise; see $D/results/eval.log)" >&2; exit 1; }
echo "[pisabench] $MODEL scored: $PO/$MODEL/summary.json"
