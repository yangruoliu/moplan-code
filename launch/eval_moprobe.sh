#!/usr/bin/env bash
# MoProbe-H for MoPlan and the same-protocol base (see docs/MOPROBE.md).  Generation is the benchmark chain:
#   bash launch/eval_bench_inputs.sh moprobe_h && bash launch/eval_bench_prepare.sh moprobe_h
#   bash launch/eval_bench_generate.sh moprobe_h moplan    (and base, zero)
# then this launcher:
#   export   scripts/s2eval/miq_export.py: generated clips -> $MOPROBE_ROOT/export/<set>/gen/<arm>/s0/<sid>.{mp4,json}
#            arm names: moplan -> moplan, base -> wan_p448, zero -> moplan_zero, <label> -> moplan_<label>
#   measure  scripts/motionbench/measure_clips.py --pin-first-frame (SAM 2.1-L propagation of the item's first-frame instance
#            masks + CoTracker3 background grid, 448x256 work frame) -> $MOPROBE_ROOT/measure/<arm>__<sid>__s0.measure.npz
#   analyze  real-continuation camera verdicts (moprobe_h_rules.py real-camera), MoPlan vs base (moprobe_h_paired.py),
#            the role2 coverage pass (moprobe_h_role2.py) and the ablations (moprobe_h_ablation.py)
#
# usage: bash launch/eval_moprobe.sh export MODEL | measure | analyze
# env:   MOPROBE_ROOT (the MoProbe bundle: item folders; see docs/MOPROBE.md), SAM2_CHECKPOINT, COTRACKER_REPO, COTRACKER_CKPT,
#        GPUS, WORKERS [16] (CPU processes of the analysis), ALLOW_MISSING=1 (analyze: go on when built items of some
#        frames are missing from MOPROBE_ROOT; by default the analysis stops, because such frames would silently drop out
#        of every count)
# analyze recomputes $MOPROBE_ROOT/real_cam.jsonl unless it is current (same frames and item folders, newer than every
# item / masks / ref file it reads, no 'missing' verdict whose inputs exist now; moprobe_h_rules.real_cam_stale).
# outputs: $MOPROBE_ROOT/{export,measure,real_cam.jsonl}, $EVAL_ROOT/moprobe_h/analysis/{paired,role2,ablation}
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
EVAL_ROOT=$(d=${EVAL_ROOT:-$REPO/eval_work}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)    # absolute: the steps below cd
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
: "${MOPROBE_ROOT:?set MOPROBE_ROOT (the MoProbe bundle, see docs/MOPROBE.md)}"
MOPROBE_ROOT=$(CDPATH= cd -- "$MOPROBE_ROOT" && pwd) || { echo "MOPROBE_ROOT is not a folder" >&2; exit 1; }
export MOPROBE_ROOT
CMD=${1:?export MODEL | measure | analyze}
B=$EVAL_ROOT/moprobe_h; MB=$REPO/scripts/motionbench; WORKERS=${WORKERS:-16}
cd "$REPO"
case $CMD in
  export)
    MODEL=${2:?model label}; G=$B/gen/$MODEL
    [ -f "$G/GEN_COMPLETE" ] || { echo "generation incomplete: $G" >&2; exit 1; }
    case $MODEL in moplan) ARM=moplan; SRC=pred ;; base) ARM=wan_p448; SRC=base ;; zero) ARM=moplan_zero; SRC=pred ;; *) ARM=moplan_$MODEL; SRC=pred ;; esac
    $PYTHON scripts/s2eval/miq_export.py --gen "$G" --manifest "$B/inputs/manifest.jsonl" --arm "$ARM" --src-arm "$SRC" \
      --root "$MOPROBE_ROOT/export" --num-seeds 1 --generator "$MODEL" --stage1 stage1
    echo "[moprobe] exported $MODEL as arm $ARM -> $MOPROBE_ROOT/export" ;;
  measure)
    : "${SAM2_CHECKPOINT:?}" "${COTRACKER_REPO:?}" "${COTRACKER_CKPT:?}"
    W=$MOPROBE_ROOT/measure_work; mkdir -p "$W" "$MOPROBE_ROOT/measure"
    for gen in "$MOPROBE_ROOT"/export/*/gen; do
      $PYTHON "$MB/make_measure_manifest.py" --gen-root "$gen" --min-age-s 0 --out "$W/$(basename "$(dirname "$gen")").jsonl"
    done
    # one manifest per item folder (data/moprobe/finding_set_B.json items_dir)
    $PYTHON - "$REPO/data/moprobe/finding_set_B.json" "$W" <<'EOF'
import json, sys
from collections import defaultdict
from pathlib import Path
idir = json.load(open(sys.argv[1]))["items_dir"]; w = Path(sys.argv[2])
by = defaultdict(list)
for m in sorted(w.glob("*.jsonl")):
    if m.name.startswith("items__"):
        continue
    for line in open(m):
        r = json.loads(line)
        d = idir.get(r["item_sid"])
        if d is None:
            print("no item folder for", r["item_sid"]); continue
        by[d].append(line)
for d, rows in by.items():
    (w / f"items__{d}.jsonl").write_text("".join(rows))
    print(d, len(rows), "clips")
EOF
    if [ -n "${GPUS:-}" ]; then gpus=($GPUS); else gpus=($(nvidia-smi --query-gpu=index --format=csv,noheader)); fi
    N=${#gpus[@]}; [ "$N" -gt 0 ] || { echo "no GPU visible" >&2; exit 1; }
    for man in "$W"/items__*.jsonl; do
      d=$(basename "$man" .jsonl); d=${d#items__}
      pids=(); rc=0
      for s in $(seq 0 $((N - 1))); do
        CUDA_VISIBLE_DEVICES=${gpus[$s]} $PYTHON "$MB/measure_clips.py" --manifest "$man" --items-dir "$MOPROBE_ROOT/$d" \
          --out "$MOPROBE_ROOT/measure" --pin-first-frame --sam2-checkpoint "$SAM2_CHECKPOINT" --cotracker-repo "$COTRACKER_REPO" \
          --cotracker-checkpoint "$COTRACKER_CKPT" --shard-index "$s" --num-shards "$N" > "$W/measure_${d}_g$s.log" 2>&1 & pids+=($!)
      done
      for p in "${pids[@]}"; do wait "$p" || rc=1; done
      [ "$rc" = 0 ] || { echo "measurement failed (see $W/measure_${d}_g*.log)" >&2; exit 1; }
      grep -h "MEASURE_DONE" "$W"/measure_${d}_g*.log || true
    done ;;
  analyze)
    OUT=$B/analysis; mkdir -p "$OUT"
    AM=(); if [ "${ALLOW_MISSING:-0}" = 1 ]; then AM=(--allow-missing); fi
    $PYTHON "$MB/moprobe_h_rules.py" real-camera --out "$MOPROBE_ROOT/real_cam.jsonl" --workers "$WORKERS" --skip-if-current ${AM[@]+"${AM[@]}"}
    $PYTHON "$MB/moprobe_h_paired.py" --out "$OUT/paired" --real-camera "$MOPROBE_ROOT/real_cam.jsonl" --workers "$WORKERS" ${AM[@]+"${AM[@]}"}
    $PYTHON "$MB/moprobe_h_role2.py" --records "$OUT/paired" --out "$OUT/role2" ${AM[@]+"${AM[@]}"}
    $PYTHON "$MB/moprobe_h_ablation.py" --out "$OUT/ablation" --workers "$WORKERS" ${AM[@]+"${AM[@]}"}
    echo "[moprobe] analysis -> $OUT" ;;
  *) echo "unknown command $CMD (export | measure | analyze)" >&2; exit 2 ;;
esac
