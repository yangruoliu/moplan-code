#!/usr/bin/env bash
# Physics-IQ (native 448x256 profile), one-time inputs: manifests of the 198 take-1 switch frames for both prompt
# settings (op = descriptions_original.csv, bpp = best_practice/descriptions_base.csv), the k16 SAM 2.1 instance masks
# of the switch frames, the umT5 prompt embeddings, and the Stage-1 exports of each setting (the whole model reads the
# benchmark prompt of the run).
#
# usage: bash launch/eval_piq_prepare.sh
# env:   PIQ_CODE_VERIFIED  physics-IQ-benchmark checkout at commit c2bca8a
#        PIQ_DATA           folder holding physics-IQ-benchmark-verified/ (switch-frames/)
#        SAM2_CHECKPOINT    SAM 2.1 hiera-large checkpoint (not needed with MASKS_DIR)
#        MASKS_DIR          optional: precomputed k16 instance masks (<sid>.pt) instead of running SAM 2
#        + the variables of launch/eval_common.sh
# outputs: $EVAL_ROOT/physics_iq/{inputs,masks,stage1/<op|bpp>/flows}
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"
: "${PIQ_CODE_VERIFIED:?set PIQ_CODE_VERIFIED (physics-IQ-benchmark @c2bca8a)}"
: "${PIQ_DATA:?set PIQ_DATA (folder with physics-IQ-benchmark-verified/)}"
P=$EVAL_ROOT/physics_iq; L=$P/logs; mkdir -p "$P" "$L"
cd "$REPO"

log "[piq prepare] manifests"
$PYTHON scripts/s2eval_verified/prep_verified.py --repo "$PIQ_CODE_VERIFIED" \
  --switch-frames "$PIQ_DATA/physics-IQ-benchmark-verified/switch-frames" --out "$P/inputs" > "$L/prep.log"

MASKS=${MASKS_DIR:-$P/masks/k16}
if [ -z "${MASKS_DIR:-}" ] && [ "$(count_files "$MASKS" "*.pt")" -lt 198 ]; then
  : "${SAM2_CHECKPOINT:?set SAM2_CHECKPOINT (or MASKS_DIR)}"
  log "[piq prepare] SAM 2.1 automatic masks of the switch frames"
  run_sharded sam2 "$L" $PYTHON scripts/cache_sam2_masks.py --manifest "$P/inputs/s1_manifest_op.csv" \
    --sam-checkpoint "$SAM2_CHECKPOINT" --output-dir "$P/masks/sam2_raw"
  $PYTHON scripts/filter_sam2_masks_v2.py --in-dir "$P/masks/sam2_raw" --out-dir "$MASKS" --max-kept 16 --union-cap 1.0 \
    > "$L/sam2_filter.log" 2>&1
fi
n=$(count_files "$MASKS" "*.pt")
[ "$n" -ge 198 ] || die "only $n of 198 k16 masks in $MASKS"

E=$P/inputs/prompt_embeds.pt
if [ ! -s "$E" ]; then
  log "[piq prepare] prompt embeddings (op + bpp + negative)"
  CUDA_VISIBLE_DEVICES=$(gpu_list | awk '{print $1}') $PYTHON -u scripts/s2eval_verified/gen_verified.py --phase embed \
    --manifests "$P/inputs/manifest_op.jsonl" "$P/inputs/manifest_bpp.jsonl" --embeds "$E" --wan "$WAN_MODEL_DIR" > "$L/embed.log" 2>&1
fi

need_file "$STAGE1_CONFIG" "$STAGE1_CKPT" "$PRIOR_LATENT"
for setting in op bpp; do
  # one GPU, all 198 rows in manifest order: the reference exports were made in a single shard (caption batches of 4)
  log "[piq prepare] Stage-1 exports ($setting prompts)"
  CUDA_VISIBLE_DEVICES=$(gpu_list | awk '{print $1}') $PYTHON -u scripts/infer_stage1_physicsiq.py --config "$STAGE1_CONFIG" \
    --checkpoint "$STAGE1_CKPT" --manifest "$P/inputs/s1_manifest_$setting.csv" --output-dir "$P/stage1/$setting" \
    --instance-masks "$MASKS" --prior-latent "$PRIOR_LATENT" --query-horizon-seconds "$QUERY_HORIZONS" \
    --text-batch-size 4 --seed 42 --set "paths.wan_model=$WAN_MODEL_DIR" > "$L/stage1_$setting.log" 2>&1
  n=$(count_files "$P/stage1/$setting/flows" "*.pt")
  [ "$n" -ge 198 ] || die "only $n of 198 Stage-1 exports for $setting"
done
log "[piq prepare] done: $P"
