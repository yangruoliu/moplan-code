#!/usr/bin/env bash
# Collect and compare the scored benchmark runs of two models (CPU): the table of every scored benchmark
# (scripts/s2eval/osp_collect.py), paired bootstrap comparisons A - B (osp_paired.py: PisaBench per clip, Morpheus per video
# and per experiment, VBench-I2V over prompts), per-task / per-category breakdowns (osp_breakdown.py: RigidBench, Morpheus)
# and the PisaBench fall-speed analysis (pisa_fall_speed.py).  Benchmarks that are not scored for both models are skipped.
#
# usage: bash launch/eval_bench_compare.sh [A=moplan] [B=base] [EXTRA_LABEL ...]
#        EXTRA_LABEL: further models for the table and the fall-speed analysis (e.g. zero)
# env:   VBENCH_HOME (for the VBench paired test: scripts/constant.py)
# outputs: $EVAL_ROOT/RESULTS.json, $EVAL_ROOT/paired/*.json
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
EVAL_ROOT=$(d=${EVAL_ROOT:-$REPO/eval_work}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)    # absolute: the steps below cd
A=${1:-moplan}; B=${2:-base}; shift $(( $# > 2 ? 2 : $# )); EXTRA=("$@")
cd "$REPO"
X=scripts/s2eval
$PYTHON $X/osp_collect.py --root "$EVAL_ROOT" --models "$A" "$B" ${EXTRA[@]+"${EXTRA[@]}"}
both() { [ -e "$1$A$2" ] && [ -e "$1$B$2" ]; }      # both PREFIX SUFFIX: PREFIX<label>SUFFIX exists for A and B
if both "$EVAL_ROOT/pisabench/official/" "/s0/results/per_video.jsonl"; then
  $PYTHON $X/osp_paired.py pisabench --root "$EVAL_ROOT" --a "$A" --b "$B"
fi
if both "$EVAL_ROOT/pisabench/official/" "/s0/eval.jsonl"; then
  arms=(--arm "ours=$A" --arm "base=$B")
  for m in ${EXTRA[@]+"${EXTRA[@]}"}; do arms+=(--arm "$m=$m"); done
  $PYTHON $X/pisa_fall_speed.py --root "$EVAL_ROOT" "${arms[@]}"
fi
if both "$EVAL_ROOT/morpheus/results/single_frame_conditioning/MP-" "/aggregate.json"; then
  $PYTHON $X/osp_paired.py morpheus --root "$EVAL_ROOT" --a "$A" --b "$B"
  $PYTHON $X/osp_breakdown.py morpheus --root "$EVAL_ROOT" --a "$A" --b "$B"
fi
if both "$EVAL_ROOT/rigidbench/runs/" "/results.json"; then
  $PYTHON $X/osp_breakdown.py rigid --root "$EVAL_ROOT" --a "$A" --b "$B"
fi
if both "$EVAL_ROOT/vbench/results/" "/summary.json"; then
  $PYTHON $X/osp_paired.py vbench --phase all --root "$EVAL_ROOT" --a "$A" --b "$B" ${VBENCH_HOME:+--vbench-code "$VBENCH_HOME"}
fi
