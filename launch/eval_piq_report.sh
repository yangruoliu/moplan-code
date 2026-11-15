#!/usr/bin/env bash
# Physics-IQ, all scores of one model (CPU): every finished run folder of the model is scored with every applicable
# official scorer (op runs: verified + orig_new + orig_ec2c611; bpp runs: verified), the runs are aggregated (mean +- std,
# ddof 1, and the official aggregate_runs_from_csvs.py for Verified), and, with BASE, the per-category breakdown with the
# paired scenario bootstrap (scripts/eval/piq_breakdown.py) is written.
#   Reported columns: Verified = bpp runs, verified scorer; Original = op runs, ec2c611 scorer (orig_new gives the same value),
#   the mean of the unrounded per-run scores (from the breakdown; the agg__ file of ec2c611 averages the per-run scores
#   the scorer prints rounded to 2 decimals).  Both are printed at the end, rounded half up from the JSON values
#   (scripts/eval/piq_report_values.py).
#
# usage: bash launch/eval_piq_report.sh LABEL [BASE_LABEL]
#        LABEL / BASE_LABEL = run labels of launch/eval_piq_generate.sh (moplan, wan22-base, moplan-zero, ...)
# env:   the scorer variables of launch/eval_piq_score.sh; RUNS (default "1 2 3 4")
# outputs: $PIQ_RESULTS/<NAME>__<MODE>.json, agg__<LABEL>-<setting>__<MODE>.json, breakdown_<LABEL>_vs_<BASE>.json
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EVAL_ROOT=$(d=${EVAL_ROOT:-$REPO/eval_work}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)    # absolute: the steps below cd
LABEL=${1:?run label, e.g. moplan}; BASE=${2:-}
: "${PIQ_PYTHON:?set PIQ_PYTHON}"
PIQ_RESULTS=$(d=${PIQ_RESULTS:-$EVAL_ROOT/physics_iq/results}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd); export PIQ_RESULTS
GEN=$EVAL_ROOT/physics_iq/gen
RUNS=${RUNS:-1 2 3 4}

score_label() {
  local label=$1 setting run name modes m
  for setting in op bpp; do
    case $setting in op) modes="verified orig_new orig_ec2c611" ;; bpp) modes="verified" ;; esac
    for run in $RUNS; do
      name=$label-$setting-run_0$run
      [ -d "$GEN/$name" ] || { echo "[piq report] skip $name (no run folder)"; continue; }
      for m in $modes; do
        [ -f "$PIQ_RESULTS/${name}__$m.json" ] && continue
        bash "$REPO/launch/eval_piq_score.sh" "$GEN/$name" "$name" "$m"
      done
    done
  done
}

aggregate_label() {
  local label=$1 setting mode names run
  for setting in op bpp; do
    for mode in verified orig_new orig_ec2c611; do
      [ "$setting" = bpp ] && [ "$mode" != verified ] && continue
      names=()
      for run in $RUNS; do
        [ -f "$PIQ_RESULTS/$label-$setting-run_0${run}__$mode.json" ] && names+=("$label-$setting-run_0$run")
      done
      [ ${#names[@]} -gt 0 ] || continue
      "$PIQ_PYTHON" "$REPO/scripts/s2eval_verified/agg_runs_v.py" --results-dir "$PIQ_RESULTS" --mode "$mode" \
        --label "$label-$setting" --piq-python "$PIQ_PYTHON" --piq-code "${PIQ_CODE_VERIFIED:?}" "${names[@]}"
    done
  done
}

score_label "$LABEL"
aggregate_label "$LABEL"
if [ -n "$BASE" ]; then
  score_label "$BASE"
  aggregate_label "$BASE"
  OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 "$PIQ_PYTHON" "$REPO/scripts/eval/piq_breakdown.py" \
    --results-dir "$PIQ_RESULTS" --code-verified "${PIQ_CODE_VERIFIED:?}" --code-original "${PIQ_CODE_ORIGINAL:?}" \
    --final "$LABEL" --base "$BASE" "$PIQ_RESULTS/breakdown_${LABEL}_vs_${BASE}.json"
fi

# the reported columns, read from the JSON files and rounded half up to 2 decimals (scripts/eval/piq_report_values.py)
"$PIQ_PYTHON" "$REPO/scripts/eval/piq_report_values.py" --results-dir "$PIQ_RESULTS" --label "$LABEL" ${BASE:+--base "$BASE"}
