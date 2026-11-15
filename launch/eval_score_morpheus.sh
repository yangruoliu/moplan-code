#!/usr/bin/env bash
# Morpheus (single-frame conditioning) with the OFFICIAL scorer (physics-from-video/Morpheus @ee3b6ae, its own venv).
#   export   scripts/bench/export_videos.py morpheus: 1280x1024, 37 frames @7 fps, raw tree
#            raw/single_frame_conditioning/<MM>/<exp>_prompt_<type>/<exp>_prompt_<type>_seed_<k>.mp4
#   resize   scripts/s2eval/morph_worker.py resize (morpheus.resizing.transforms.process_mp4_file)
#   score    scripts/s2eval/morph_worker.py score: morpheus-track-and-score --calculate-scores --use-cache per (experiment,
#            seed) item of the 16 reported experiments (non_holonomic_pendulum excluded), PROCS_PER_GPU workers per GPU,
#            items balanced by their typical cost
#   check    morph_worker.py check: every exported video of the 16 experiments has a score (the scorer can skip a video
#            without writing one; an unscored video would otherwise drop out of the means instead of counting 0)
#   summary  morph_worker.py summary + scripts/bench/morph_aggregate_br.py (discarded videos -> 0, Total = (Dyn + PhysInv) / 2,
#            best of plain / enhanced by Total)
# The Morpheus model name MM = MP-<MODEL> is registered in memory only; it must not contain Morpheus path keywords
# (wan, plain, enhanced, falling, ...).  The physical-invariance score is deterministic; the dynamical score's PINN fits
# are not seeded upstream and can move slightly between runs.
#
# usage: bash launch/eval_score_morpheus.sh MODEL
# env:   MORPHEUS_HOME (checkout with .venv, SAM 2 and data as installed by its README), PROCS_PER_GPU [3], GPUS
#        MORPHEUS_HF_HOME [$HF_HOME, else the hub's default]: HF_HOME of the scorer.  The scorer runs offline, so its hub
#        cache ($MORPHEUS_HF_HOME/hub) must hold nielsr/depth-anything-large (the depth model, at refs/main), and the
#        checkout must hold checkpoints/sam2.1_hiera_large.pt; the launcher stops before any work when they are missing
#        (SKIP_CACHE_CHECK=1 skips this check).  See docs/EVALUATION.md, section 5.
# outputs: $EVAL_ROOT/morpheus/{raw,resized,results}/single_frame_conditioning/MP-<MODEL>/, summary_MP-<MODEL>.json,
#          results/single_frame_conditioning/MP-<MODEL>/{aggregate.json,per_video.jsonl}
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
EVAL_ROOT=$(d=${EVAL_ROOT:-$REPO/eval_work}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)    # absolute: the steps below cd
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
MODEL=${1:?model label}
: "${MORPHEUS_HOME:?set MORPHEUS_HOME (Morpheus checkout)}"
MORPHEUS_HOME=$(CDPATH= cd -- "$MORPHEUS_HOME" && pwd) || { echo "MORPHEUS_HOME is not a folder" >&2; exit 1; }
MORPHEUS_HF_HOME=${MORPHEUS_HF_HOME:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}}
MM=MP-$MODEL; B=$EVAL_ROOT/morpheus; G=$B/gen/$MODEL; IN=$B/inputs; L=$G/logs
RAW=$B/raw; RES=$B/resized; OUTM=$B/results; mkdir -p "$RAW" "$RES" "$OUTM" "$L"
WK=$REPO/scripts/s2eval/morph_worker.py; P=${PROCS_PER_GPU:-3}
[ -f "$G/GEN_COMPLETE" ] || { echo "generation incomplete: $G" >&2; exit 1; }
case $MODEL in base) ARM=base ;; *) ARM=pred ;; esac
if [ -n "${GPUS:-}" ]; then gpus=($GPUS); else gpus=($(nvidia-smi --query-gpu=index --format=csv,noheader)); fi
N=${#gpus[@]}; [ "$N" -gt 0 ] || { echo "no GPU visible" >&2; exit 1; }
NCPU=$(nproc 2> /dev/null || echo 8)
MPY() { ( cd "$MORPHEUS_HOME" && unset PYTHONPATH HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE \
          && export HF_HOME=$MORPHEUS_HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1 \
          OMP_NUM_THREADS=${MOMP:-2} && .venv/bin/python "$@" ); }

# 0 offline caches: fail before any work instead of in the middle of the scoring
if [ "${SKIP_CACHE_CHECK:-0}" != 1 ]; then
  DA=$MORPHEUS_HF_HOME/hub/models--nielsr--depth-anything-large; miss=()
  [ -s "$DA/refs/main" ] || miss+=("$DA/refs/main")
  for f in config.json model.safetensors; do compgen -G "$DA/snapshots/*/$f" > /dev/null || miss+=("$DA/snapshots/<rev>/$f"); done
  [ -s "$MORPHEUS_HOME/checkpoints/sam2.1_hiera_large.pt" ] || miss+=("$MORPHEUS_HOME/checkpoints/sam2.1_hiera_large.pt")
  if [ ${#miss[@]} -gt 0 ]; then
    { echo "the Morpheus scorer runs offline, but these files are missing (see docs/EVALUATION.md, section 5):"; printf '  %s\n' "${miss[@]}"; } >&2
    exit 1
  fi
fi

# 1 export (raw tree) + resize, once per model
if [ "$(cat "$G/MORPH_EXPORTED" 2>/dev/null)" != "$MM" ]; then
  rm -rf "$RAW/single_frame_conditioning/$MM" "$RES/single_frame_conditioning/$MM"
  (cd "$REPO" && $PYTHON scripts/bench/export_videos.py morpheus --gen "$G" --manifest "$IN/manifest.jsonl" --arm "$ARM" --model "$MM" \
     --raw "$RAW" --num-seeds "$(cat "$IN/NUM_SEEDS" 2>/dev/null || echo 1)") > "$L/morph_export.log" 2>&1
  MPY "$WK" resize --raw "$RAW" --resized "$RES" --model "$MM" --workers $(( NCPU > 1 ? NCPU - 1 : 1 )) > "$L/morph_resize.log" 2>&1
  echo "$MM" > "$G/MORPH_EXPORTED"
fi

# 2 (experiment, seed) items -> N x P workers, longest-processing-time first on the typical cost per video
PLAN=$($PYTHON - "$RES/single_frame_conditioning/$MM/RESIZE_MAP.json" $(( N * P )) <<'EOF'
import json, sys
from collections import defaultdict
COST = {"falling_ball": 71, "falling_apple": 43, "falling_marker": 37, "falling_tape": 42, "bouncing_ball": 379, "projectile": 226,
        "holonomic_pendulum": 72, "double_pendulum": 272, "spring": 12, "sliding_book": 444, "rolling_full_can": 430,
        "rolling_empty_can": 430, "rolling_orange": 416, "collision_equal": 31, "collision_big_hits_small": 30, "collision_small_hits_big": 30}
mapping, w = json.load(open(sys.argv[1])), int(sys.argv[2])
videos = defaultdict(int)
for m in mapping:
    if m["experiment"] in COST:
        videos[(m["experiment"], m["seed_index"])] += 1
items = sorted(videos, key=lambda it: (-COST[it[0]] * videos[it], it))
load, out = [0.0] * w, [[] for _ in range(w)]
for it in items:
    b = min(range(w), key=lambda i: (load[i], i))
    load[b] += COST[it[0]] * videos[it]; out[b].append(it)
for i, g in enumerate(out):
    if g:
        print(i, ",".join(f"{e}:{k}" for e, k in g))
EOF
)
pids=(); rc=0
while read -r w items; do
  [ -n "$w" ] || continue
  CUDA_VISIBLE_DEVICES=${gpus[$(( w % N ))]} MOMP=$(( NCPU / (N * P) > 0 ? NCPU / (N * P) : 1 )) \
    MPY "$WK" score --resized "$RES" --results "$OUTM" --model "$MM" --items "$items" >> "$L/morph_score_w$w.log" 2>&1 & pids+=($!)
done <<< "$PLAN"
for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || rc=1; done
[ "$rc" = 0 ] || { echo "scorer worker failed (see $L/morph_score_w*.log)" >&2; exit 1; }

# 3 completeness (every exported video has a score), summary + aggregation (discarded -> 0, best prompt type by Total)
WANT=$(( $(wc -l < "$IN/manifest.jsonl") * $(cat "$IN/NUM_SEEDS" 2>/dev/null || echo 1) ))
MPY "$WK" check --resized "$RES" --results "$OUTM" --model "$MM" --expect "$WANT" > "$L/morph_check.log" 2>&1 \
  || { cat "$L/morph_check.log" >&2; exit 1; }
MPY "$WK" summary --results "$OUTM" --model "$MM" --out "$B/summary_$MM.json" > "$L/morph_summary.log" 2>&1
$PYTHON "$REPO/scripts/bench/morph_aggregate_br.py" "$OUTM/single_frame_conditioning/$MM" > "$L/morph_aggregate.log"
echo "[morpheus] $MODEL scored: $OUTM/single_frame_conditioning/$MM/aggregate.json"
