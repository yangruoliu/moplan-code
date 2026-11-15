#!/usr/bin/env bash
# Batch path, part 1: a manifest of first frames + captions -> held Stage-2 condition caches (one <sid>.pt per clip),
# exactly the chain of the benchmark runs:
#   SAM 2.1 automatic masks -> k16 filter -> Stage-1 exports (released checkpoint) -> no-video base cache
#   (Wan VAE image latent, cache_stage2_cond_v2.py --no-video) -> refresh_predictions.py (cond_pred/zero/null from the
#   exports) -> hold_known_beyond_horizon.py (cond_pred held past the 4 s Stage-1 horizon).
# Then run launch/infer_batch_stage2.sh on $WORK/cache.
#
# usage: MANIFEST=rows.csv WORK=out/bench FRAMES=37 bash launch/infer_batch_caches.sh
# env:   MANIFEST   CSV with sample_id, first_frame_path, caption (+ optional seed_key: rows sharing one share the generator
#                   noise; the benchmark adapters do not write it, so the seed comes from sample_id)
#        WORK       output folder                                FRAMES  4k+1 clip length [37]      FPS  [7]
#        MASKS_DIR  optional: k16 masks (<sid>.pt) made earlier instead of running SAM 2
#        STAGE1_GPUS GPUs of the Stage-1 export [the first GPU of GPUS: one shard, as in the reference runs]
#        + launch/infer_common.sh (WAN_MODEL_DIR, MOPLAN_WEIGHTS, SAM2_CHECKPOINT, GPUS)
# Stage-1 encodes its captions in batches of 4 per shard, so its exports depend on the row order and on the number
# of Stage-1 shards. The reference exports of every benchmark were made in one shard; keep the default STAGE1_GPUS and
# the same manifest (same row order) to reproduce them bit for bit.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/infer_common.sh"
need_weights
: "${MANIFEST:?set MANIFEST (CSV with sample_id, first_frame_path, caption)}"
: "${WORK:?set WORK (output folder)}"
FRAMES=${FRAMES:-37}; FPS=${FPS:-7}
need "$MANIFEST" "$STAGE1_CKPT/backend_params.pt" "$STAGE1_CKPT/decoder.pt" "$STAGE1_CKPT/resolved_config.yaml" "$STAGE1_CKPT/prior_latent.pt"
mkdir -p "$WORK/logs"; L=$WORK/logs; cd "$REPO"
"$PYTHON" - "$MANIFEST" "$WORK/ids.txt" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
missing = [k for k in ("sample_id", "first_frame_path", "caption") if rows and k not in rows[0]]
if not rows or missing:
    sys.exit(f"{sys.argv[1]}: no rows or missing columns {missing}")
if any(not r["caption"].strip() for r in rows):
    sys.exit("empty captions are not supported (Stage-1 was never trained on one)")
open(sys.argv[2], "w").write("".join(r["sample_id"] + "\n" for r in rows))
PY
N=$(wc -l < "$WORK/ids.txt" | tr -d ' ')
count() { ls "$1" 2>/dev/null | grep -c '\.pt$' || true; }

MASKS=${MASKS_DIR:-$WORK/masks/k16}
if [ -z "${MASKS_DIR:-}" ] && [ "$(count "$MASKS")" -lt "$N" ]; then
  : "${SAM2_CHECKPOINT:?set SAM2_CHECKPOINT (or MASKS_DIR)}"
  log "[caches] SAM 2.1 automatic masks ($N frames)"
  run_sharded sam2 "$L" "$PYTHON" -u scripts/cache_sam2_masks.py --manifest "$MANIFEST" \
    --sam-checkpoint "$SAM2_CHECKPOINT" --output-dir "$WORK/masks/sam2_raw"
  log "[caches] k16 filter"
  "$PYTHON" -u scripts/filter_sam2_masks_v2.py --in-dir "$WORK/masks/sam2_raw" --out-dir "$MASKS" \
    --max-kept 16 --union-cap 1.0 > "$L/k16.log" 2>&1
fi
[ "$(count "$MASKS")" -ge "$N" ] || die "only $(count "$MASKS") of $N k16 masks in $MASKS"

if [ "$(count "$WORK/stage1/flows")" -lt "$N" ]; then
  STAGE1_GPUS=${STAGE1_GPUS:-$(first_gpu)}
  log "[caches] Stage-1 exports (GPUs: $STAGE1_GPUS)"
  GPUS="$STAGE1_GPUS" run_sharded stage1 "$L" "$PYTHON" -u scripts/infer_stage1_physicsiq.py --config "$STAGE1_CKPT/resolved_config.yaml" \
    --checkpoint "$STAGE1_CKPT" --manifest "$MANIFEST" --output-dir "$WORK/stage1" --instance-masks "$MASKS" \
    --prior-latent "$STAGE1_CKPT/prior_latent.pt" --query-horizon-seconds "$QUERY_HORIZONS" --text-batch-size 4 --seed 42 \
    --set "paths.wan_model=$WAN_MODEL_DIR"
fi
[ "$(count "$WORK/stage1/flows")" -ge "$N" ] || die "Stage-1 exports incomplete"

if [ "$(count "$WORK/cache_base")" -lt "$N" ]; then
  log "[caches] base caches (image latent, $FRAMES frames at $FPS fps)"
  run_sharded base_cache "$L" "$PYTHON" -u scripts/cache_stage2_cond_v2.py --manifest "$MANIFEST" --data "$WORK" \
    --exports "$WORK/stage1/flows" --output "$WORK/cache_base" --model-dir "$WAN_MODEL_DIR" --no-video \
    --frames "$FRAMES" --fps "$FPS"
fi
if [ ! -f "$WORK/cache_pred/PREDICTION_RECEIPT.json" ]; then
  rm -rf "$WORK/cache_pred"
  log "[caches] Stage-1 conditions (refresh_predictions)"
  "$PYTHON" -u scripts/clean_r2/refresh_predictions.py --cache "$WORK/cache_base" --exports "$WORK/stage1/flows" \
    --ids "$WORK/ids.txt" --output "$WORK/cache_pred" --stage1-checkpoint "$STAGE1_CKPT" --purpose evaluation \
    --manifest "$MANIFEST" --no-scenes > "$L/refresh.log" 2>&1
fi
if [ ! -f "$WORK/cache/HOLD_KNOWN_RECEIPT.json" ]; then
  log "[caches] hold rule past the Stage-1 horizon"
  "$PYTHON" -u scripts/s2eval/hold_known_beyond_horizon.py "$WORK/cache_pred" "$WORK/cache" --keys cond_pred > "$L/hold.log" 2>&1
fi
log "[caches] done: $WORK/cache ($(count "$WORK/cache") caches), ids $WORK/ids.txt"
