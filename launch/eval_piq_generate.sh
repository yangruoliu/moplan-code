#!/usr/bin/env bash
# Physics-IQ, native profile: one run (198 videos) of one model and prompt setting.
#   sampler   WanImageToVideoPipeline with UniPC (pipeline schedule, flow shift 3), 30 steps, CFG 5 on both experts,
#             boundary 0.9, the official Wan2.2 negative prompt; 448x256, 41 frames written at 8 fps and trimmed to
#             40 frames / 5.000 s (README ffmpeg -t 5); seed 41 + RUN for every video of the run
#   MODEL     moplan  MoPlan: Stage-1 exports of the setting (held as known past the 4 s Stage-1 horizon) + generator
#             base    the untrained Wan2.2-I2V-A14B through the same loop (no LoRA, adapter or condition)
#             zero    ablation: the MoPlan generator fed the zero-motion plan (the plan's objects at rest)
#             other   like moplan, generator from $STAGE2_CKPT, run label = MODEL
# One worker per GPU; workers share the run folder (claims), finished videos are skipped, so the same command can be
# run again (or on several nodes at once) until the folder holds 198 videos.
#
# usage: bash launch/eval_piq_generate.sh MODEL op|bpp RUN(1..4)
# env:   IDS (comma list of 4-digit ids, default all 198), MAXV (max videos per worker), ATTN (default native = torch SDPA)
# outputs: $EVAL_ROOT/physics_iq/gen/<NAME>/*.mp4 (the scorer's run folder), $EVAL_ROOT/physics_iq/work/<NAME>/
#          NAME = <label>-<setting>-run_0<RUN>, label = moplan | wan22-base | moplan-zero | MODEL
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"
MODEL=${1:?MODEL moplan|base|zero|<label>}; SETTING=${2:?op|bpp}; RUN=${3:?RUN 1..4}
case $SETTING in op|bpp) ;; *) die "SETTING must be op or bpp" ;; esac
case $RUN in 1|2|3|4) ;; *) die "RUN must be 1..4" ;; esac
P=$EVAL_ROOT/physics_iq; cd "$REPO"
case $MODEL in base) LABEL=wan22-base ;; zero) LABEL=moplan-zero ;; *) LABEL=$MODEL ;; esac
NAME=$LABEL-$SETTING-run_0$RUN; SEED=$((41 + RUN))
RUN_DIR=$P/gen/$NAME; WORK=$P/work/$NAME; L=$WORK/logs; mkdir -p "$RUN_DIR" "$L"
MAN=$P/inputs/manifest_$SETTING.jsonl; E=$P/inputs/prompt_embeds.pt
need_file "$MAN" "$E" "$S2_CONFIG"
EXTRA=(--profile native --config "$S2_CONFIG")
case $MODEL in
  base) EXTRA+=(--base --tag base) ;;
  *)
    X=$P/stage1/$SETTING/flows
    [ -d "$X" ] || die "no Stage-1 exports $X (run launch/eval_piq_prepare.sh)"
    for x in high low; do need_file "$STAGE2_CKPT/$x/lora_params.pt" "$STAGE2_CKPT/$x/motion_adapter.pt"; done
    EXTRA+=(--tag "$LABEL" --s1tag stage1 --ckpt "$STAGE2_CKPT" --exports "$X" --hold-known)
    if [ "$MODEL" = zero ]; then EXTRA+=(--plan-kind zero); fi ;;
esac
if [ -n "${IDS:-}" ]; then EXTRA+=(--ids "$IDS"); fi
if [ -n "${MAXV:-}" ]; then EXTRA+=(--max-videos "$MAXV"); fi
gpus=($(gpu_list)); [ ${#gpus[@]} -gt 0 ] || die "no GPU visible"
log "[piq generate] $NAME (seed $SEED) on ${#gpus[@]} GPUs, $(count_files "$RUN_DIR" "*.mp4")/198 done"
pids=(); rc=0
for g in "${gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=$g $PYTHON -u scripts/s2eval_verified/gen_verified.py --manifest "$MAN" --embeds "$E" --wan "$WAN_MODEL_DIR" \
    --run-dir "$RUN_DIR" --work-dir "$WORK" --seed "$SEED" --attn "${ATTN:-native}" --offload none "${EXTRA[@]}" \
    >> "$L/gpu$g.log" 2>&1 & pids+=($!)
  sleep 20                                  # stagger: the next worker reads the weights after the previous one started
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
have=$(count_files "$RUN_DIR" "*.mp4")
log "[piq generate] $NAME: $have/198 videos (rc $rc)"
if [ "$have" -ge 198 ]; then touch "$WORK/GEN_COMPLETE"; fi
exit $rc
