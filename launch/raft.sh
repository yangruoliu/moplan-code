#!/usr/bin/env bash
# Light RAFT round of the MoPlan generator on one node.
#
#   sample  G candidates per condition under the cached Stage-1 condition (arm pred) with the GT-SFT generator:
#           scripts/infer_stage2_cond_v2.py, phases embed -> high -> low; candidate k of clip sid has the seed
#           42 + crc32(sid) % 100000 + 1000003 k (job_seed), so the candidates do not depend on the GPU count
#   score   reward v5 of every candidate against the real continuation of its clip:
#           scripts/control/score_generated.py track (CoTracker3 points + SAM 2.1 mask centroids)
#   pick    per condition, the best candidate of a complete group of G whose reward spread is >= MARGIN, VAE-encoded:
#           scripts/dpo/build_pairs.py --winner-only
#   train   per expert, 300 optimizer steps from the GT-SFT checkpoint: flow matching on the picked candidates on a
#           share REPLAY_P of the micro-steps, GT-SFT replay on the rest:
#           scripts/train_s2_v4.py with configs/stage2/raft.yaml and configs/stage2/raft_overrides.json
#
# usage: launch/raft.sh [all | sample | score | pick | train ...]     (default: all; finished work is skipped)
#
# environment
#   WAN_MODEL_DIR          (required) Wan2.2-I2V-A14B-Diffusers directory
#   MOPLAN_SFT_CHECKPOINT  (sample, train) GT-SFT generator the round starts from: high/ and low/, each with
#                          lora_params.pt and motion_adapter.pt.  Not distributed: train it with launch/train_sft.sh,
#                          which stops at step 4000 and links $RUN_DIR/sft_step004000/{high,low} (docs/TRAINING_SFT.md)
#   MOPLAN_RAFT_DATA       (sample, score, train) condition caches: cache_<pool>/<sid>.pt and cache_<pool>/scenes_v5/<sid>.npz;
#                          optional manifest_<pool>.jsonl or .csv (sample_id, first_frame_path) for the first frames
#   MOPLAN_SFT_DATA        (train) GT-SFT replay payloads: pools/{p101,ssv2,phyco}/<sid>.pt
#   COTRACKER_REPO, COTRACKER_CKPT, SAM2_CHECKPOINT     (score) the reward trackers
#   MOPLAN_WORK_DIR        output root (default <repo>/work); the round is written to $MOPLAN_WORK_DIR/raft
#   POOLS                  condition pools (default "phyco phycoB p101 p101B")
#   GPUS                   GPU indices (default: $CUDA_VISIBLE_DEVICES, else every GPU nvidia-smi lists); sampling and
#                          scoring are sharded over them, training runs one expert per GPU
#   GROUP=8 MARGIN=0.2 SEED=42 STEPS=300 REPLAY_P=0.5   round settings (the defaults are the released round);
#                          REPLAY_P = share of micro-steps on the picked candidates (dpo_prob), NOT the replay share
#   EXPERTS                experts to train (default "high low")
#   LIMIT                  only the first LIMIT conditions of every pool (smoke tests; default 0 = all)
#   SAMPLE_STEPS           sampler steps (default 0 = the config's 30; smoke tests only)
#   REPLAY_POOLS           replace the GT-SFT replay pools ("name=dir@ids,..."; smoke tests without the full pools)
#   REPLAY_MIX             replace the replay mix ("pool:real:weight,...")
#   EXTRA_OVERRIDES        JSON object merged into the training overrides last (smoke tests)
#   PYTHON                 interpreter (default python)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=${PYTHON:-python}

: "${WAN_MODEL_DIR:?set WAN_MODEL_DIR to the Wan2.2-I2V-A14B-Diffusers directory}"
export WAN_MODEL_DIR
WORK=${MOPLAN_WORK_DIR:-$REPO/work}
RAFT_DIR=$WORK/raft
POOLS=${POOLS:-"phyco phycoB p101 p101B"}
GROUP=${GROUP:-8}
MARGIN=${MARGIN:-0.2}
SEED=${SEED:-42}
STEPS=${STEPS:-300}
REPLAY_P=${REPLAY_P:-0.5}
EXPERTS=${EXPERTS:-"high low"}
LIMIT=${LIMIT:-0}
SAMPLE_STEPS=${SAMPLE_STEPS:-0}
if [[ -n ${GPUS:-} ]]; then
  read -r -a GPU_LIST <<< "$GPUS"
elif [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
  IFS=, read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
  read -r -a GPU_LIST <<< "$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
fi
(( ${#GPU_LIST[@]} > 0 )) || { echo "[raft] no GPU found; set GPUS" >&2; exit 2; }
LOGS=$RAFT_DIR/logs
mkdir -p "$RAFT_DIR" "$LOGS" "$RAFT_DIR/ids"

need() {   # need VAR...: every variable set (and exported for the Python steps)
  local name
  for name in "$@"; do
    [[ -n ${!name:-} ]] || { echo "[raft] set $name (see the header of launch/raft.sh)" >&2; exit 2; }
    export "${name?}"
  done
}
log() { echo "[raft $(date +%H:%M:%S)] $*"; }

# Sampling and training read the same config; its paths.wan_model is filled in from WAN_MODEL_DIR.
CONFIG=$RAFT_DIR/raft.yaml
"$PY" - "$REPO/configs/stage2/raft.yaml" "$CONFIG" "$WAN_MODEL_DIR" <<'PY'
import json, re, sys
from pathlib import Path
source, target, wan = sys.argv[1:]
text, count = re.subn(r"(?m)^  wan_model: .*$", "  wan_model: " + json.dumps(str(Path(wan).resolve())), Path(source).read_text())
if count != 1:
    raise SystemExit(f"{source}: expected one paths.wan_model line")
Path(target).write_text(text)
PY

# The conditions of every pool (p101 = the 256 RL-train clips of cache_p101).
ids_of() { echo "$RAFT_DIR/ids/$1.txt"; }
for pool in $POOLS; do
  src=$REPO/data/ids/raft/ids_$pool.txt
  if [[ $pool == p101 ]]; then src=$REPO/data/ids/raft/ids_p101_rl.txt; fi
  [[ -f $src ]] || { echo "[raft] unknown pool $pool (no $src)" >&2; exit 2; }
  if (( LIMIT > 0 )); then head -n "$LIMIT" "$src" > "$(ids_of "$pool")"; else cp "$src" "$(ids_of "$pool")"; fi
done

shards() {   # shards NAME CMD...: one process per GPU, --shard-index i --num-shards n appended
  local name=$1; shift
  local n=${#GPU_LIST[@]} s rc=0 pids=()
  for (( s = 0; s < n; s++ )); do
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$s]} "$@" --shard-index "$s" --num-shards "$n" > "$LOGS/${name}_$s.log" 2>&1 &
    pids+=($!)
  done
  for (( s = 0; s < n; s++ )); do wait "${pids[$s]}" || rc=1; done
  if (( rc )); then
    echo "[raft] $name failed; logs: $LOGS/${name}_*.log" >&2
    tail -n 5 "$LOGS/${name}_0.log" >&2 || true
    return 1
  fi
}

manifest_of() {   # the optional first-frame manifest of a pool
  local f
  for f in "$MOPLAN_RAFT_DATA/manifest_$1.jsonl" "$MOPLAN_RAFT_DATA/manifest_$1.csv"; do
    if [[ -f $f ]]; then echo "$f"; return 0; fi
  done
  return 0
}

step_sample() {
  need MOPLAN_SFT_CHECKPOINT MOPLAN_RAFT_DATA
  local pool cache panel embeds manifest extra common
  for pool in $POOLS; do
    cache=$MOPLAN_RAFT_DATA/cache_$pool
    panel=$RAFT_DIR/panels/$pool
    embeds=$panel/prompt_embeds.pt
    mkdir -p "$panel"
    extra=()
    manifest=$(manifest_of "$pool")
    if [[ -n $manifest ]]; then extra+=(--manifest "$manifest"); fi
    if (( SAMPLE_STEPS > 0 )); then extra+=(--steps "$SAMPLE_STEPS"); fi
    common=(scripts/infer_stage2_cond_v2.py --config "$CONFIG" --cache-dir "$cache" --ids "$(ids_of "$pool")" --arms pred
            --output-dir "$panel" --prompt-embeds "$embeds" --num-seeds "$GROUP" --seed "$SEED"
            --high-checkpoint "$MOPLAN_SFT_CHECKPOINT/high" --low-checkpoint "$MOPLAN_SFT_CHECKPOINT/low" ${extra[@]+"${extra[@]}"})
    if [[ ! -f $embeds ]]; then
      log "sample $pool: prompt embeddings"
      CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} "$PY" -u "${common[@]}" --phase embed > "$LOGS/embed_$pool.log" 2>&1 \
        || { echo "[raft] embed $pool failed; see $LOGS/embed_$pool.log" >&2; tail -n 5 "$LOGS/embed_$pool.log" >&2; return 1; }
    fi
    log "sample $pool: $(wc -l < "$(ids_of "$pool")" | tr -d " ") conditions x $GROUP candidates on ${#GPU_LIST[@]} GPU(s)"
    shards "high_$pool" "$PY" -u "${common[@]}" --phase high
    shards "low_$pool" "$PY" -u "${common[@]}" --phase low
  done
}

step_score() {
  need MOPLAN_RAFT_DATA
  local pool panel
  for pool in $POOLS; do
    panel=$RAFT_DIR/panels/$pool
    log "score $pool"
    shards "score_$pool" "$PY" -u scripts/control/score_generated.py track --generated "$panel" \
      --scenes "$MOPLAN_RAFT_DATA/cache_$pool/scenes_v5" --reward-version v5
    "$PY" scripts/control/score_generated.py table --generated "$panel" --reward-version v5 --out "$panel/table.json" > "$panel/table.txt" 2>&1 \
      || log "score $pool: summary table failed (see $panel/table.txt)"
  done
}

step_pick() {
  local pool out f
  for pool in $POOLS; do
    out=$RAFT_DIR/winners/$pool
    if [[ -f $out/selection_receipt.json ]]; then log "pick $pool: done before"; continue; fi
    log "pick $pool"
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} "$PY" -u scripts/dpo/build_pairs.py --generated "$RAFT_DIR/panels/$pool" --model-dir "$WAN_MODEL_DIR" \
      --out "$out" --arm pred --margin "$MARGIN" --suffix .reward_v5.json --ids "$(ids_of "$pool")" --winner-only \
      --expected-group-size "$GROUP" > "$LOGS/pick_$pool.log" 2>&1 \
      || { echo "[raft] pick $pool failed; see $LOGS/pick_$pool.log" >&2; tail -n 5 "$LOGS/pick_$pool.log" >&2; return 1; }
    tail -n 1 "$LOGS/pick_$pool.log"
  done
  mkdir -p "$RAFT_DIR/winners_all"
  shopt -s nullglob
  for pool in $POOLS; do
    for f in "$RAFT_DIR/winners/$pool"/*.pt; do ln -sfn "$f" "$RAFT_DIR/winners_all/$(basename "$f")"; done
  done
  shopt -u nullglob
  log "pick: $(find "$RAFT_DIR/winners_all" -name '*.pt' | wc -l | tr -d " ") selected candidates in $RAFT_DIR/winners_all"
}

write_overrides() {
  local out=$1
  mkdir -p "$out"
  RAFT_DIR=$RAFT_DIR POOLS=$POOLS STEPS=$STEPS REPLAY_P=$REPLAY_P "$PY" - "$REPO" "$out/OVERRIDES.json" <<'PY'
import json, os, sys
from pathlib import Path
repo, target = Path(sys.argv[1]), Path(sys.argv[2])
env = os.environ
template = json.loads((repo / "configs/stage2/raft_overrides.json").read_text())
values = {"{repo_root}": repo, "{sft_data}": env.get("MOPLAN_SFT_DATA", ""), "{raft_data}": env["MOPLAN_RAFT_DATA"],
          "{sft_checkpoint}": env["MOPLAN_SFT_CHECKPOINT"], "{winners}": Path(env["RAFT_DIR"]) / "winners_all"}
values = {key: str(Path(value).resolve()) if str(value) else "" for key, value in values.items()}


def resolve(value):
    if isinstance(value, str):
        for key, path in values.items():
            value = value.replace(key, path)
    return value


cfg = {key: resolve(value) for key, value in template.items() if not key.startswith("_")}
pools = env["POOLS"].split()
items = [item for item in cfg["pools"].split(",") if item]
replay = [item for item in items if not item.startswith("raft_")]
if env.get("REPLAY_POOLS"):
    replay = [item for item in env["REPLAY_POOLS"].split(",") if item]
elif not values["{sft_data}"]:
    raise SystemExit("set MOPLAN_SFT_DATA (GT-SFT replay pools) or REPLAY_POOLS")
raft = [item for item in items if item.startswith("raft_") and item.split("=", 1)[0][len("raft_"):] in pools]
cfg["pools"] = ",".join(replay + raft)
if env.get("REPLAY_MIX"):
    cfg["mix"] = env["REPLAY_MIX"]
cfg["max_steps"] = int(env["STEPS"])
cfg["dpo_prob"] = float(env["REPLAY_P"])
cfg.update(json.loads(env.get("EXTRA_OVERRIDES") or "{}"))
if target.is_file() and json.loads(target.read_text()) != cfg:
    raise SystemExit(f"{target} exists with other settings: remove it or use another MOPLAN_WORK_DIR")
target.write_text(json.dumps(cfg, indent=1) + "\n")
PY
}

step_train() {
  need MOPLAN_SFT_CHECKPOINT MOPLAN_RAFT_DATA
  local out=$RAFT_DIR/train expert latest i=0 rc=0 pids=() names=() experts
  read -r -a experts <<< "$EXPERTS"
  write_overrides "$out"
  local parallel=0
  if (( ${#GPU_LIST[@]} >= ${#experts[@]} )); then parallel=1; fi
  for expert in "${experts[@]}"; do
    if [[ -f $out/$expert/done.json ]]; then log "train $expert: done before"; i=$((i + 1)); continue; fi
    local resume=()
    latest=$(ls -d "$out/$expert"/checkpoint-* 2>/dev/null | sort | tail -n 1 || true)
    if [[ -n $latest && -f $latest/trainer_state.json ]]; then resume=(--resume-from "$latest"); fi
    local gpu=${GPU_LIST[0]}
    if (( parallel )); then gpu=${GPU_LIST[$i]}; fi
    if (( ${#resume[@]} )); then log "train $expert on GPU $gpu, resuming from $(basename "$latest")"; else log "train $expert on GPU $gpu"; fi
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/train_s2_v4.py \
      --config "$CONFIG" --expert "$expert" --output-dir "$out" ${resume[@]+"${resume[@]}"} >> "$LOGS/train_$expert.log" 2>&1 &
    if (( parallel )); then
      pids+=($!); names+=("$expert")
    else
      wait $! || { echo "[raft] train $expert failed; see $LOGS/train_$expert.log" >&2; tail -n 5 "$LOGS/train_$expert.log" >&2; return 1; }
    fi
    i=$((i + 1))
  done
  for (( i = 0; i < ${#pids[@]}; i++ )); do
    wait "${pids[$i]}" || { rc=1; echo "[raft] train ${names[$i]} failed; see $LOGS/train_${names[$i]}.log" >&2; tail -n 5 "$LOGS/train_${names[$i]}.log" >&2; }
  done
  (( rc == 0 )) || return 1
  local final
  final=$(printf "checkpoint-%06d" "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_steps"])' "$out/OVERRIDES.json")")
  mkdir -p "$RAFT_DIR/final"
  for expert in "${experts[@]}"; do
    if [[ -f $out/$expert/$final/lora_params.pt ]]; then ln -sfn "$out/$expert/$final" "$RAFT_DIR/final/$expert"; fi
  done
  log "train: generator in $RAFT_DIR/final/{high,low} (use as --high-checkpoint / --low-checkpoint)"
}

STEPS_TO_RUN=("$@")
(( ${#STEPS_TO_RUN[@]} )) || STEPS_TO_RUN=(all)
for step in "${STEPS_TO_RUN[@]}"; do
  case $step in
    all) step_sample; step_score; step_pick; step_train ;;
    sample) step_sample ;;
    score) step_score ;;
    pick) step_pick ;;
    train) step_train ;;
    *) echo "usage: launch/raft.sh [all | sample | score | pick | train ...]" >&2; exit 2 ;;
  esac
done
