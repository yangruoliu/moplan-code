#!/usr/bin/env bash
# Benchmark manifests for generation under the MoPlan protocol (448x256 @ 7 fps), following each benchmark's official
# prompts, sample sets and negative prompts (CPU, minutes).
#   pisabench   all 361 real clips; first frame = the 1080^2 clip frame pillarboxed into 1280x720 (scripts/eval/pisa_clips.py);
#               official prompt "<object> falls. A video that conforms to the laws of physics." and negative prompt
#   rigidbench  100 samples (5 tasks x 20), scene prompt + the official suffix, official negative prompt, 17 frames
#   morpheus    16 experiments x {plain, enhanced} x 3 official seeds = 96 rows; Stage-1 reads the plain prompt, the generator
#               the official prompt (enhanced = Wan2.2's prompt extension, shipped in scripts/eval/inputs)
#   vbench      the full VBench-I2V suite (1,118 entries incl. camera_motion), 5 videos each, 7-4 crops
#   moprobe_h   MoProbe-H: every first frame of data/moprobe/finding_set_B.json, conditions ex / cf / rex / rcf
#
# usage: bash launch/eval_bench_inputs.sh pisabench|rigidbench|morpheus|vbench|moprobe_h
# env:   PISA_HOME (pisabench: data/real.zip), RIGIDBENCH_HOME (rigidbench: checkout with data/), MORPHEUS_HOME
#        (morpheus: checkout with prompts/), VBENCH_HOME (vbench: checkout with vbench2_beta_i2v/), MOPROBE_ROOT (moprobe_h:
#        the MoProbe bundle, see docs/MOPROBE.md; MOPROBE_SOURCE_PREFIX: path prefix of first_frame_path in its generation
#        manifests that MOPROBE_ROOT replaces)
# outputs: $EVAL_ROOT/<bench>/inputs/{manifest.jsonl,manifest.csv,ids_all.txt,NUM_SEEDS,...}
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
EVAL_ROOT=$(d=${EVAL_ROOT:-$REPO/eval_work}; mkdir -p "$d" && CDPATH= cd -- "$d" && pwd)    # absolute: the steps below cd
BENCH=${1:?pisabench|rigidbench|morpheus|vbench|moprobe_h}
B=$EVAL_ROOT/$BENCH; IN=$B/inputs; mkdir -p "$B"
cd "$REPO"
X=scripts/bench; I=scripts/eval/inputs
case $BENCH in
  pisabench)
    : "${PISA_HOME:?set PISA_HOME (PisaBench working root with data/real.zip)}"
    PISA_HOME=$(CDPATH= cd -- "$PISA_HOME" && pwd) || { echo "PISA_HOME is not a folder" >&2; exit 1; }
    SRC=$B/pisa_real
    if [ ! -d "$SRC/real" ]; then mkdir -p "$SRC"; unzip -q "$PISA_HOME/data/real.zip" -d "$SRC"; fi
    $PYTHON scripts/eval/pisa_clips.py --real "$SRC/real" --out "$B/pisa_clips"
    $PYTHON $X/official_sp_inputs.py pisafull --src $I/pisabench_real.jsonl --first-frames "$B/pisa_clips/first_frames" --out "$IN" ;;
  rigidbench)
    : "${RIGIDBENCH_HOME:?set RIGIDBENCH_HOME (RigidBench checkout with data/inputs.jsonl)}"
    RIGIDBENCH_HOME=$(CDPATH= cd -- "$RIGIDBENCH_HOME" && pwd) || { echo "RIGIDBENCH_HOME is not a folder" >&2; exit 1; }
    $PYTHON $X/rigidbench_inputs.py --inputs "$RIGIDBENCH_HOME/data/inputs.jsonl" --out "$IN" --frames 17 --exact-prompt
    $PYTHON $X/official_sp_inputs.py rigido --rigidbench-repo "$RIGIDBENCH_HOME" --out "$IN" ;;
  morpheus)
    : "${MORPHEUS_HOME:?set MORPHEUS_HOME (Morpheus checkout with prompts/reference_images)}"
    MORPHEUS_HOME=$(CDPATH= cd -- "$MORPHEUS_HOME" && pwd) || { echo "MORPHEUS_HOME is not a folder" >&2; exit 1; }
    $PYTHON $X/official_sp_inputs.py morpho --src $I/morpheus_plain.jsonl --src $I/morpheus_enhanced.jsonl \
      --morpheus-repo "$MORPHEUS_HOME" --out "$IN" ;;
  vbench)
    : "${VBENCH_HOME:?set VBENCH_HOME (VBench checkout)}"
    VBENCH_HOME=$(CDPATH= cd -- "$VBENCH_HOME" && pwd) || { echo "VBENCH_HOME is not a folder" >&2; exit 1; }
    $PYTHON $X/vbench_inputs.py --code "$VBENCH_HOME" --out "$IN" --frames 37 --ratio 7-4 --exact-prompt --with-camera
    $PYTHON $X/official_sp_inputs.py vbfull --out "$IN" ;;
  moprobe_h)
    : "${MOPROBE_ROOT:?set MOPROBE_ROOT (the MoProbe bundle, see docs/MOPROBE.md)}"
    MOPROBE_ROOT=$(CDPATH= cd -- "$MOPROBE_ROOT" && pwd) || { echo "MOPROBE_ROOT is not a folder" >&2; exit 1; }
    $PYTHON $X/miqB_inputs.py --finding-set data/moprobe/finding_set_B.json --subset all --miq-root "$MOPROBE_ROOT" --out "$IN" \
      ${MOPROBE_SOURCE_PREFIX:+--source-prefix "$MOPROBE_SOURCE_PREFIX"}
    echo 1 > "$IN/NUM_SEEDS" ;;
  *) echo "unknown benchmark $BENCH" >&2; exit 2 ;;
esac
echo "[bench inputs] $BENCH: $(wc -l < "$IN/manifest.jsonl" | tr -d ' ') rows -> $IN"
