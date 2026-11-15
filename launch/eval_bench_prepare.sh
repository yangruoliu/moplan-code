#!/usr/bin/env bash
# Stage-2 condition caches of one benchmark (GPU): the deployment-only input chain of MoPlan on the benchmark first frames,
# through the batch cache launcher of the inference path (launch/infer_batch_caches.sh):
#   1 SAM 2.1 automatic proposals on the first frame -> k16 instance masks (sharded over the GPUs)
#   2 Stage-1 exports (released Stage-1, the row's caption, 7 query horizons, seed 42) on ONE GPU, in manifest order: the
#     reference exports of every benchmark were made in a single shard (caption batches of 4 depend on the sharding)
#   3 no-video cache, refresh of cond_pred / cond_zero / cond_null from the exports, hold rule past the 4 s Stage-1 horizon
#     (launch/infer_batch_caches.sh)
#   4 zero-plan ablation caches: cond_pred := hold(cond_zero) (scripts/s2eval/make_zero_caches.py)
# Every step skips finished work, so the same command can be run again after an interruption.
#
# usage: bash launch/eval_bench_prepare.sh pisabench|rigidbench|morpheus|vbench|moprobe_h
# env:   SAM2_CHECKPOINT (SAM 2.1 hiera-large .pt), MASKS_DIR (optional: precomputed k16 masks of this benchmark)
#        + the variables of launch/eval_common.sh
# outputs: $EVAL_ROOT/<bench>/{masks/k16,stage1/flows,cache_base,cache_pred,cache,cache_zero}
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"
BENCH=${1:?pisabench|rigidbench|morpheus|vbench|moprobe_h}; KEY=$(bench_key "$BENCH"); FRAMES=$(bench_frames "$BENCH")
B=$EVAL_ROOT/$BENCH; IN=$B/inputs; L=$B/logs; mkdir -p "$L"
need_file "$IN/manifest.jsonl" "$IN/manifest.csv" "$IN/ids_all.txt" "$STAGE1_CONFIG" "$PRIOR_LATENT"
cd "$REPO"
TOTAL=$(wc -l < "$IN/ids_all.txt" | tr -d ' ')

# 1 masks
MASKS=${MASKS_DIR:-$B/masks/k16}
if [ -z "${MASKS_DIR:-}" ] && [ "$(count_files "$MASKS" "*.pt")" -lt "$TOTAL" ]; then
  : "${SAM2_CHECKPOINT:?set SAM2_CHECKPOINT (or MASKS_DIR)}"
  log "[$BENCH prepare] SAM 2.1 proposals"
  run_sharded sam2 "$L" $PYTHON -u scripts/cache_sam2_masks.py --manifest "$IN/manifest.csv" --sam-checkpoint "$SAM2_CHECKPOINT" \
    --output-dir "$B/masks/sam2_raw"
  $PYTHON -u scripts/filter_sam2_masks_v2.py --in-dir "$B/masks/sam2_raw" --out-dir "$MASKS" --max-kept 16 --union-cap 1.0 --manifest "$IN/manifest.csv" \
    > "$L/sam2_filter.log" 2>&1
fi

# 2 Stage-1 exports, one shard
if [ "$(count_files "$B/stage1/flows" "*.pt")" -lt "$TOTAL" ]; then
  log "[$BENCH prepare] Stage-1 exports (one GPU)"
  CUDA_VISIBLE_DEVICES=$(gpu_list | awk '{print $1}') $PYTHON -u scripts/infer_stage1_physicsiq.py --config "$STAGE1_CONFIG" \
    --checkpoint "$STAGE1_CKPT" --manifest "$IN/manifest.csv" --output-dir "$B/stage1" --instance-masks "$MASKS" \
    --prior-latent "$PRIOR_LATENT" --query-horizon-seconds "$QUERY_HORIZONS" --text-batch-size 4 --seed 42 \
    --set "paths.wan_model=$WAN_MODEL_DIR" > "$L/stage1.log" 2>&1
fi

# 3 caches (masks and exports above are reused: the launcher skips complete steps)
log "[$BENCH prepare] condition caches ($FRAMES frames)"
MANIFEST=$IN/manifest.csv WORK=$B FRAMES=$FRAMES FPS=7 MASKS_DIR=$MASKS bash "$REPO/launch/infer_batch_caches.sh"
cp "$IN/ids_all.txt" "$B/cache/ids_eval.txt"

# 4 zero-plan caches (make_zero_caches.py reads <ev>/s1/<src>/caches/<bench>)
Z=$B/zero_work
if [ ! -f "$Z/s1/zeroh/caches/$KEY/ZERO_PLAN_RECEIPT.json" ]; then
  mkdir -p "$Z/s1/moplan/caches"
  [ -e "$Z/s1/moplan/caches/$KEY" ] || ln -s "$B/cache" "$Z/s1/moplan/caches/$KEY"
  log "[$BENCH prepare] zero-plan caches"
  $PYTHON scripts/s2eval/make_zero_caches.py --ev "$Z" --src moplan --dst zeroh --benches "$KEY" > "$L/zero.log" 2>&1
fi
[ -e "$B/cache_zero" ] || ln -s "$Z/s1/zeroh/caches/$KEY" "$B/cache_zero"
log "[$BENCH prepare] done: $(count_files "$B/cache" "*.pt") of $TOTAL caches in $B/cache"
