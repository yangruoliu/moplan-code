#!/usr/bin/env bash
# Score one 198-video Physics-IQ run folder with an UNPATCHED official scorer (CPU only).
#   MODE = verified      c2bca8a physiq/run_physics_iq.py (default = Physics-IQ Verified), descriptions by NAME:
#                        NAME contains -op-  -> descriptions/descriptions_original.csv
#                        NAME contains -bpp- -> descriptions/best_practice/descriptions_base.csv
#          orig_new      c2bca8a physiq/run_physics_iq.py --original_physics_iq, descriptions/descriptions_original.csv
#          orig_ec2c611  ec2c611 code/run_physics_iq.py (original-leaderboard scorer), its descriptions/descriptions.csv
# Steps
#   1. scripts/eval/piq_precheck_run.py: exactly 198 regular *.mp4 (nothing else), IDs 0001..0198 in order, ffprobe
#      duration 5.000+-0.001 s, one FPS (cv2 + ffprobe), >= fps*5 frames
#   2. a private work dir <PIQ_WORK>/<NAME>__<MODE>__<ts>__<pid>/: in/<NAME>/ hard links (copy fallback) of the videos
#      (the scorer renames its inputs), bench/<benchmark>/ symlinks to the shared GT per FPS + a private
#      video-masks/generated/ (the scorers' folder-name mask cache can never collide), out/ (so the "CSV already complete
#      -> skip" shortcut never fires)
#   3. the scorer (optionally under taskset -c $CPUS) with NPROC workers (c2bca8a only; ec2c611 is serial), BLAS threads 1
#   4. CSV / metrics JSON / plots / log -> <PIQ_RESULTS>/<NAME>__<MODE>/ and the summary <PIQ_RESULTS>/<NAME>__<MODE>.json
#      (scripts/eval/piq_summarize_run.py)
#   5. in/ and bench/ of the work dir are deleted unless KEEP_WORK=1
#
# usage: bash launch/eval_piq_score.sh RUN_DIR NAME MODE
# env:   PIQ_PYTHON          python of the scorer environment (numpy, pandas, opencv of the c2bca8a lock file)
#        PIQ_CODE_VERIFIED   physics-IQ-benchmark checkout at c2bca8a
#        PIQ_CODE_ORIGINAL   physics-IQ-benchmark checkout at ec2c611
#        PIQ_DATA            folder holding physics-IQ-benchmark/ and physics-IQ-benchmark-verified/ (switch-frames,
#                            split-videos/testing/<fps>FPS, video-masks/real/<fps>FPS; 8FPS is used by the native profile)
#        PIQ_RESULTS         results folder [$EVAL_ROOT/physics_iq/results]
#        PIQ_WORK            scratch folder [$EVAL_ROOT/physics_iq/score_work]
#        NPROC (1..16, default 8)  CPUS (taskset cpu list, optional)  FORCE=1 (overwrite)  KEEP_WORK=1
#        ALLOW_GT_RESAMPLE=1 (let the scorer build missing GT FPS folders from 30FPS inside the private dir; diagnostic only)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EVAL_ROOT=$(d=${EVAL_ROOT:-$REPO/eval_work}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)    # absolute: the steps below cd
: "${PIQ_PYTHON:?set PIQ_PYTHON (python of the Physics-IQ scorer environment)}"
: "${PIQ_CODE_VERIFIED:?set PIQ_CODE_VERIFIED (physics-IQ-benchmark @c2bca8a)}"
: "${PIQ_CODE_ORIGINAL:?set PIQ_CODE_ORIGINAL (physics-IQ-benchmark @ec2c611)}"
: "${PIQ_DATA:?set PIQ_DATA (folder with physics-IQ-benchmark/ and physics-IQ-benchmark-verified/)}"
PIQ_CODE_VERIFIED=$(CDPATH= cd -- "$PIQ_CODE_VERIFIED" && pwd) || { echo "PIQ_CODE_VERIFIED is not a folder" >&2; exit 1; }
PIQ_CODE_ORIGINAL=$(CDPATH= cd -- "$PIQ_CODE_ORIGINAL" && pwd) || { echo "PIQ_CODE_ORIGINAL is not a folder" >&2; exit 1; }
PIQ_DATA=$(CDPATH= cd -- "$PIQ_DATA" && pwd) || { echo "PIQ_DATA is not a folder" >&2; exit 1; }
RESULTS=$(d=${PIQ_RESULTS:-$EVAL_ROOT/physics_iq/results}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)
WORKROOT=$(d=${PIQ_WORK:-$EVAL_ROOT/physics_iq/score_work}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)   # the scorer runs inside it
NEW=$PIQ_CODE_VERIFIED; OLD=$PIQ_CODE_ORIGINAL; DATA=$PIQ_DATA; PY=$PIQ_PYTHON
NPROC=${NPROC:-8}; CPUS=${CPUS:-}
die() { echo "eval_piq_score.sh: ERROR: $*" >&2; exit 1; }

[[ $# -eq 3 ]] || die "usage: eval_piq_score.sh <RUN_DIR> <NAME> <MODE: verified|orig_new|orig_ec2c611>"
RUN_DIR=$(cd "$1" && pwd) || die "bad RUN_DIR"
NAME=$2
MODE=$3
[[ $NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "NAME must match [A-Za-z0-9][A-Za-z0-9._-]*"
[[ $NPROC =~ ^[0-9]+$ ]] && (( NPROC >= 1 && NPROC <= 16 )) || die "NPROC must be 1..16"

ORIGFLAG=()
case $MODE in
  verified)
    BENCH=physics-IQ-benchmark-verified; CODE=$NEW
    if [[ $NAME == *-op-* ]]; then DESC=$NEW/descriptions/descriptions_original.csv
    elif [[ $NAME == *-bpp-* ]]; then DESC=$NEW/descriptions/best_practice/descriptions_base.csv
    else die "verified mode: NAME must contain -op- or -bpp- to select the descriptions CSV"; fi ;;
  orig_new)
    BENCH=physics-IQ-benchmark; CODE=$NEW; DESC=$NEW/descriptions/descriptions_original.csv; ORIGFLAG=(--original_physics_iq) ;;
  orig_ec2c611)
    BENCH=physics-IQ-benchmark; CODE=$OLD; DESC=$OLD/descriptions/descriptions.csv ;;
  *) die "MODE must be verified | orig_new | orig_ec2c611" ;;
esac

OUTJSON=$RESULTS/${NAME}__${MODE}.json
RES=$RESULTS/${NAME}__${MODE}
if [[ -e $OUTJSON || -e $RES ]]; then
  [[ ${FORCE:-0} == 1 ]] || die "result exists ($OUTJSON); set FORCE=1 to re-score"
  rm -rf "$RES" "$OUTJSON"
fi
mkdir -p "$RESULTS" "$WORKROOT"

TS=$(date +%Y%m%d-%H%M%S)
W=$WORKROOT/${NAME}__${MODE}__${TS}__$$
mkdir -p "$W/in/$NAME" "$W/out" "$W/mpl" || die "cannot create $W"
LOG=$W/scorer.log
echo "[piq score] $(date '+%F %T') name=$NAME mode=$MODE run_dir=$RUN_DIR work=$W nproc=$NPROC cpus=${CPUS:-all}"

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=$W/mpl \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
TS_CMD=()
if [[ -n $CPUS ]] && command -v taskset > /dev/null; then TS_CMD=(taskset -c "$CPUS"); fi

# ---- 1. precheck ---------------------------------------------------------------------------------
PRE_RC=0
${TS_CMD[@]+"${TS_CMD[@]}"} "$PY" -I "$REPO/scripts/eval/piq_precheck_run.py" "$RUN_DIR" --json "$W/precheck.json" > /dev/null || PRE_RC=$?
[[ -s $W/precheck.json ]] || die "precheck produced no output"
if (( PRE_RC != 0 )); then cat "$W/precheck.json" >&2; die "precheck failed for $RUN_DIR (see $W/precheck.json)"; fi
FPS=$("$PY" -I -c 'import json,sys; print(int(round(json.load(open(sys.argv[1]))["fps_cv2"][0])))' "$W/precheck.json")
echo "[piq score] precheck ok: fps=$FPS"

# ---- 2. private benchmark tree + hard-linked inputs ----------------------------------------------
B=$W/bench/$BENCH
mkdir -p "$B/split-videos/testing" "$B/video-masks/real" "$B/video-masks/generated"
ln -s "$DATA/$BENCH/switch-frames" "$B/switch-frames"
for d in "$DATA/$BENCH"/split-videos/testing/*FPS; do ln -s "$d" "$B/split-videos/testing/"; done
for d in "$DATA/$BENCH"/video-masks/real/*FPS; do ln -s "$d" "$B/video-masks/real/"; done
[[ -d $B/split-videos/testing/30FPS ]] || die "GT 30FPS testing folder missing in $DATA/$BENCH (the scorer validates names against it)"
if [[ ! -d $B/split-videos/testing/${FPS}FPS || ! -d $B/video-masks/real/${FPS}FPS ]]; then
  [[ ${ALLOW_GT_RESAMPLE:-0} == 1 ]] || die "GT for ${FPS}FPS missing in $DATA/$BENCH (testing and/or real masks); set ALLOW_GT_RESAMPLE=1 to let the scorer build it privately"
  echo "[piq score] WARNING: GT ${FPS}FPS missing -> the scorer resamples from 30FPS inside $B"
fi
n=0
for f in "$RUN_DIR"/*.mp4; do
  ln "$f" "$W/in/$NAME/" 2> /dev/null || cp -p "$f" "$W/in/$NAME/" || die "cannot link/copy $f"
  n=$((n + 1))
done
(( n == 198 )) || die "linked $n files, expected 198"

# ---- 3. run the official scorer ------------------------------------------------------------------
T0=$(date +%s); T0H=$(date '+%F %T %Z')
RC=0
if [[ $MODE == orig_ec2c611 ]]; then
  # the ec2c611 scorer reads ./physics-IQ-benchmark relative to the cwd and imports its siblings from code/
  (cd "$W/bench" && ${TS_CMD[@]+"${TS_CMD[@]}"} "$PY" "$OLD/code/run_physics_iq.py" \
      --input_folders "$W/in/$NAME" --output_folder "$W/out" --descriptions_file "$DESC") > "$LOG" 2>&1 || RC=$?
  SRC_RES=$W/out/results
else
  (cd "$W" && ${TS_CMD[@]+"${TS_CMD[@]}"} "$PY" "$NEW/physiq/run_physics_iq.py" \
      --input_folders "$W/in/$NAME" --output_folder "$W/out" --descriptions_file "$DESC" \
      --benchmark_base_folder "$W/bench" --n_process "$NPROC" ${ORIGFLAG[@]+"${ORIGFLAG[@]}"}) > "$LOG" 2>&1 || RC=$?
  SRC_RES=$W/out/$BENCH/results
fi
T1=$(date +%s); T1H=$(date '+%F %T %Z')
WALL=$((T1 - T0))
echo "[piq score] scorer exit=$RC wall=${WALL}s"

# ---- 4. collect ----------------------------------------------------------------------------------
mkdir -p "$RES"
cp -p "$LOG" "$W/precheck.json" "$RES/" 2> /dev/null || true
if [[ -d $SRC_RES ]]; then cp -p "$SRC_RES"/* "$RES/" 2> /dev/null || true; fi
SRC=0
PYTHONPATH=$NEW "$PY" "$REPO/scripts/eval/piq_summarize_run.py" --mode "$MODE" --name "$NAME" --work "$W" --log "$LOG" \
  --precheck "$W/precheck.json" --code "$CODE" --descriptions "$DESC" --bench_data "$DATA/$BENCH" \
  --run_dir "$RUN_DIR" --rc "$RC" --t_start "$T0H" --t_end "$T1H" --wall_s "$WALL" --nproc "$NPROC" \
  --cpus "${CPUS:-all}" --results_dir "$RES" --out "$OUTJSON" --gt_source "precomputed_${FPS}FPS" || SRC=$?

# ---- 5. cleanup ----------------------------------------------------------------------------------
if [[ ${KEEP_WORK:-0} != 1 && $RC == 0 && $SRC == 0 ]]; then rm -rf "$W/in" "$W/bench" "$W/mpl"; fi
(( RC == 0 && SRC == 0 )) || die "scoring failed (scorer rc=$RC, summary rc=$SRC); log: $LOG"
echo "[piq score] done -> $OUTJSON"
