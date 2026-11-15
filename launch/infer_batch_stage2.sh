#!/usr/bin/env bash
# Batch path, part 2: held Stage-2 condition caches -> videos with scripts/infer_stage2_cond_v2.py, the generation path
# of the MoProbe / PisaBench / RigidBench / Morpheus / VBench results:
#   embed (umT5 prompt embeddings, one GPU) -> high (high-noise expert up to the boundary, every GPU) -> low (every GPU).
# Seeds: job_seed(SEED, sample_id, k) = SEED + crc32(sample_id) % 100000 + 1000003 k for k < NUM_SEEDS.
#
# usage: CACHE_DIR=out/bench/cache OUT=out/bench/gen bash launch/infer_batch_stage2.sh [extra infer_stage2_cond_v2.py flags]
#        (e.g. --negative-prompt "..." or --prompt-table T.json; they are passed to every phase)
# env:   CACHE_DIR   held caches (launch/infer_batch_caches.sh, or restored ones)     OUT   output folder
#        IDS         sample-id list [every <sid>.pt of CACHE_DIR]                     MODEL moplan | base [moplan]
#        NUM_SEEDS   [1]   SEED [42]   SEED_MAJOR=1 orders the jobs seed by seed   EMBEDS prompt-embedding file [$OUT/prompt_embeds.pt]
#                    (EMBEDS is encoded again when it lacks a prompt of IDS: scripts/eval/embeds_cover.py)
#        MANIFEST    optional csv/jsonl with sample_id, first_frame_path (overrides the caches' image paths)
#        MIQ_MANIFEST optional MoProbe manifest (scripts/bench/miqB_inputs.py): also export $OUT/moprobe/<set>/gen/<ARM>/s<k>/<sid>.mp4
#        + launch/infer_common.sh (WAN_MODEL_DIR, MOPLAN_WEIGHTS, GPUS)
# outputs: $OUT/<arm>__<sid>[__s<k>].npy (uint8 [F, 256, 448, 3]) + .json sidecars; arm = pred (MoPlan) or base
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/infer_common.sh"
: "${CACHE_DIR:?set CACHE_DIR}"; : "${OUT:?set OUT}"
MODEL=${MODEL:-moplan}; NUM_SEEDS=${NUM_SEEDS:-1}; SEED=${SEED:-42}; EMBEDS=${EMBEDS:-$OUT/prompt_embeds.pt}
mkdir -p "$OUT/logs"; L=$OUT/logs; cd "$REPO"
if [ -z "${IDS:-}" ]; then IDS=$OUT/ids.txt; ls "$CACHE_DIR" | grep '\.pt$' | sed 's/\.pt$//' | sort > "$IDS"; fi
need "$IDS" "$S2_CONFIG"
ARGS=(--config "$S2_CONFIG" --cache-dir "$CACHE_DIR" --ids "$IDS" --output-dir "$OUT" --prompt-embeds "$EMBEDS"
      --num-seeds "$NUM_SEEDS" --seed "$SEED" --wan-model "$WAN_MODEL_DIR")
case $MODEL in
  base) ARGS+=(--arms base --base); ARM=base ;;
  moplan) need_weights; ARGS+=(--arms pred --high-checkpoint "$HIGH_CKPT" --low-checkpoint "$LOW_CKPT"); ARM=pred ;;
  *) die "MODEL must be moplan or base" ;;
esac
if [ "${SEED_MAJOR:-0}" = 1 ]; then ARGS+=(--seed-major); fi
if [ -n "${MANIFEST:-}" ]; then ARGS+=(--manifest "$MANIFEST"); fi
# Encode the prompts unless EMBEDS already holds every prompt of IDS (with the same negative prompt), so that one file
# shared across id lists is re-encoded instead of failing with a KeyError in the high phase.
COVER=(--embeds "$EMBEDS" --cache-dir "$CACHE_DIR" --ids "$IDS"); prev=""
for x in "$@"; do
  case $prev in
    --prompt-table) COVER+=(--prompt-table "$x") ;;
    --negative-prompt) printf '%s' "$x" > "$L/negative_prompt.txt"; COVER+=(--negative-prompt-file "$L/negative_prompt.txt") ;;
  esac
  case $x in
    --prompt-table=*) COVER+=(--prompt-table "${x#*=}") ;;
    --negative-prompt=*) printf '%s' "${x#*=}" > "$L/negative_prompt.txt"; COVER+=(--negative-prompt-file "$L/negative_prompt.txt") ;;
  esac
  prev=$x
done
if ! "$PYTHON" scripts/eval/embeds_cover.py "${COVER[@]}"; then
  log "[stage2] prompt embeddings"
  CUDA_VISIBLE_DEVICES=$(first_gpu) "$PYTHON" -u scripts/infer_stage2_cond_v2.py "${ARGS[@]}" "$@" --phase embed > "$L/embed.log" 2>&1
fi
for phase in high low; do
  log "[stage2] phase $phase"
  run_sharded "$phase" "$L" "$PYTHON" -u scripts/infer_stage2_cond_v2.py "${ARGS[@]}" "$@" --phase "$phase"
done
n=$(ls "$OUT" | grep -c "^${ARM}__.*\.npy$" || true)
log "[stage2] $n clips in $OUT"
if [ -n "${MIQ_MANIFEST:-}" ]; then
  log "[stage2] MoProbe export"
  "$PYTHON" -u scripts/s2eval/miq_export.py --gen "$OUT" --manifest "$MIQ_MANIFEST" --arm "${MIQ_ARM:-moplan}" --src-arm "$ARM" \
    --root "$OUT/moprobe" --num-seeds "$NUM_SEEDS" > "$L/miq_export.log" 2>&1
fi
