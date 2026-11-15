#!/usr/bin/env bash
# Stage-2 generation of one benchmark under the MoPlan protocol (GPU), through the batch generation launcher of the
# inference path (launch/infer_batch_stage2.sh -> scripts/infer_stage2_cond_v2.py): UniPC (pipeline schedule, flow shift 3),
# 30 steps, CFG 5, boundary 0.9; 448x256 @ 7 fps; per-clip noise job_seed(42, sample_id, k) = 42 + crc32(sample_id) % 100000
# + 1000003 k, so every model gets the same noise per clip.  The benchmark's own negative prompt (NEGATIVE_PROMPT.txt),
# generator prompts (PROMPT_TABLE.json) and seed order (SEED_MAJOR) are passed when the inputs folder has them.
#   MODEL  moplan   MoPlan on the held Stage-1 caches
#          base     the untrained Wan2.2-I2V-A14B (no LoRA, adapter or condition; same images, prompts, sampler and noise)
#          zero     ablation: the MoPlan generator on the zero-plan caches
#          <label>  like moplan, generator from $STAGE2_CKPT/{high,low}
# Finished clips (and staged high-noise latents) are skipped, so an interrupted run is resumed by the same command.
# The prompt embeddings ($EVAL_ROOT/<bench>/prompt_embeds.pt, shared by every MODEL of the benchmark) always hold every
# prompt of the full id list, also after a LIMIT smoke test (a file that lacks some is encoded again), and GEN_COMPLETE
# (read by the scoring launchers) is written only by a run without LIMIT once every clip of the full id list exists.
#
# usage: bash launch/eval_bench_generate.sh pisabench|rigidbench|morpheus|vbench|moprobe_h MODEL
# env:   LIMIT (first n ids only, smoke test), GPUS + the variables of launch/eval_common.sh
# outputs: $EVAL_ROOT/<bench>/gen/<MODEL>/{<arm>__<sid>[__s<k>].npy,.json}, GEN_COMPLETE; $EVAL_ROOT/<bench>/prompt_embeds.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"
BENCH=${1:?benchmark}; MODEL=${2:?moplan|base|zero|<label>}
bench_key "$BENCH" > /dev/null
B=$EVAL_ROOT/$BENCH; IN=$B/inputs; O=$B/gen/$MODEL; mkdir -p "$O"
case $MODEL in base) KIND=base; CACHE=$B/cache; ARM=base ;; zero) KIND=moplan; CACHE=$B/cache_zero; ARM=pred ;; *) KIND=moplan; CACHE=$B/cache; ARM=pred ;; esac
need_file "$CACHE" "$IN/manifest.jsonl" "$IN/ids_all.txt"
IDS_ALL=$IN/ids_all.txt; IDS=$IDS_ALL; LIMIT=${LIMIT:-0}
if [ "$LIMIT" -gt 0 ]; then head -n "$LIMIT" "$IDS_ALL" > "$O/ids_limit.txt"; IDS=$O/ids_limit.txt; fi
SEEDS=$(cat "$IN/NUM_SEEDS" 2>/dev/null || echo 1)
EXTRA=()
if [ -s "$IN/NEGATIVE_PROMPT.txt" ]; then EXTRA+=(--negative-prompt "$(cat "$IN/NEGATIVE_PROMPT.txt")"); fi
if [ -s "$IN/PROMPT_TABLE.json" ]; then EXTRA+=(--prompt-table "$IN/PROMPT_TABLE.json"); fi
SEED_MAJOR=0; if [ -f "$IN/SEED_MAJOR" ]; then SEED_MAJOR=1; fi

# Prompt embeddings of the FULL id list, also for a LIMIT smoke test: the file is shared by every model of the benchmark,
# and infer_batch_stage2.sh only encodes it when it is absent.  Each prompt is encoded on its own, so the file does not
# depend on which ids are listed.  A file that lacks prompts of the full list is encoded again.
EMBEDS=$B/prompt_embeds.pt
if ! "$PYTHON" "$REPO/scripts/eval/embeds_cover.py" --embeds "$EMBEDS" --cache-dir "$CACHE" --ids "$IDS_ALL" \
     --prompt-table "$IN/PROMPT_TABLE.json" --negative-prompt-file "$IN/NEGATIVE_PROMPT.txt"; then
  log "[$BENCH generate] prompt embeddings of the full id list ($(wc -l < "$IDS_ALL" | tr -d ' ') ids)"
  mkdir -p "$O/logs"; TMP_EMBEDS=$EMBEDS.tmp-$$
  CUDA_VISIBLE_DEVICES=$(gpu_list | awk '{print $1}') "$PYTHON" -u "$REPO/scripts/infer_stage2_cond_v2.py" --config "$S2_CONFIG" \
    --cache-dir "$CACHE" --ids "$IDS_ALL" --output-dir "$O" --prompt-embeds "$TMP_EMBEDS" --wan-model "$WAN_MODEL_DIR" \
    --manifest "$IN/manifest.jsonl" --phase embed ${EXTRA[@]+"${EXTRA[@]}"} > "$O/logs/embed.log" 2>&1 \
    || die "prompt embeddings failed (see $O/logs/embed.log)"
  mv -f "$TMP_EMBEDS" "$EMBEDS"
fi

log "[$BENCH generate] $MODEL: cache $CACHE, seeds $SEEDS, ids $(wc -l < "$IDS" | tr -d ' ')"
CACHE_DIR=$CACHE OUT=$O IDS=$IDS MODEL=$KIND NUM_SEEDS=$SEEDS SEED=42 SEED_MAJOR=$SEED_MAJOR EMBEDS=$EMBEDS \
  MANIFEST=$IN/manifest.jsonl S2_CONFIG=$S2_CONFIG HIGH_CKPT=$STAGE2_CKPT/high LOW_CKPT=$STAGE2_CKPT/low \
  bash "$REPO/launch/infer_batch_stage2.sh" ${EXTRA[@]+"${EXTRA[@]}"}

# GEN_COMPLETE (read by the scoring launchers): written only by a run without LIMIT, when every clip of the full id list
# exists; removed whenever a clip is missing
want=0; have=0
while read -r sid; do
  [ -n "$sid" ] || continue
  for k in $(seq 0 $((SEEDS - 1))); do
    want=$((want + 1))
    if [ "$k" = 0 ]; then f=$O/${ARM}__$sid.npy; else f=$O/${ARM}__${sid}__s$k.npy; fi
    if [ -f "$f" ]; then have=$((have + 1)); fi
  done
done < "$IDS_ALL"
log "[$BENCH generate] $MODEL: $have of $want clips of the full id list"
if [ "$have" -lt "$want" ]; then
  rm -f "$O/GEN_COMPLETE"
elif [ "$LIMIT" -eq 0 ]; then
  touch "$O/GEN_COMPLETE"
fi
if [ "$LIMIT" -gt 0 ]; then log "[$BENCH generate] LIMIT=$LIMIT (smoke test): GEN_COMPLETE is left to a run without LIMIT"; fi
