#!/usr/bin/env bash
# Smoke test of the RAFT pipeline (launch/raft.sh) on real weights and data, small enough for one or two GPUs:
# 2 conditions of one pool x 2 candidates (4 sampler steps), reward v5, winner selection without a margin, then
# 2 optimizer steps per expert with checkpoints after every step.  Then checks that every stage wrote its outputs.
#
# usage: launch/smoke_raft.sh
#
# environment: the same as launch/raft.sh (WAN_MODEL_DIR, MOPLAN_SFT_CHECKPOINT, MOPLAN_RAFT_DATA, MOPLAN_SFT_DATA,
# COTRACKER_REPO, COTRACKER_CKPT, SAM2_CHECKPOINT); the smoke settings below can be overridden.
#   MOPLAN_WORK_DIR   default <repo>/work/smoke (the round goes to $MOPLAN_WORK_DIR/raft; remove it to rerun from scratch)
#   POOLS=phyco LIMIT=2 GROUP=2 MARGIN=0 SAMPLE_STEPS=4 STEPS=2
#   REPLAY_POOLS / REPLAY_MIX   to train without the full GT-SFT pools, e.g.
#     REPLAY_POOLS="phyco=/data/sft/pools/phyco@$PWD/data/ids/stage2_sft/train/phyco.txt" REPLAY_MIX="phyco:real:1"
# Memory: sampling loads one Wan2.2 expert in bf16 at a time (about 48 GB); training needs about 41 GiB per expert.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MOPLAN_WORK_DIR=${MOPLAN_WORK_DIR:-$REPO/work/smoke}
export POOLS=${POOLS:-phyco} LIMIT=${LIMIT:-2} GROUP=${GROUP:-2} MARGIN=${MARGIN:-0} SAMPLE_STEPS=${SAMPLE_STEPS:-4} STEPS=${STEPS:-2}
export EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-'{"save_every": 1, "log_every": 1}'}
PY=${PYTHON:-python}

bash "$REPO/launch/raft.sh" all

"$PY" - "$MOPLAN_WORK_DIR/raft" "$POOLS" "$GROUP" "$STEPS" "${EXPERTS:-high low}" <<'PY'
import json, math, sys
from pathlib import Path
raft, pools, group, steps, experts = Path(sys.argv[1]), sys.argv[2].split(), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split()
problems = []
for pool in pools:
    ids = [line.strip() for line in (raft / "ids" / f"{pool}.txt").read_text().splitlines() if line.strip()]
    panel = raft / "panels" / pool
    for sid in ids:
        for k in range(group):
            name = f"pred__{sid}" + (f"__s{k}" if k else "")
            if not (panel / f"{name}.npy").is_file():
                problems.append(f"missing candidate {panel / name}.npy")
                continue
            reward = panel / f"{name}.reward_v5.json"
            if not reward.is_file() or not math.isfinite(json.loads(reward.read_text())["reward"]):
                problems.append(f"missing or nonfinite reward {reward}")
    receipt = raft / "winners" / pool / "selection_receipt.json"
    if not receipt.is_file():
        problems.append(f"missing {receipt}")
winners = sorted((raft / "winners_all").glob("*.pt"))
if not winners:
    problems.append("no selected candidate in winners_all")
for expert in experts:
    run = raft / "train" / expert
    checkpoint = run / f"checkpoint-{steps:06d}"
    for name in ("lora_params.pt", "motion_adapter.pt", "trainer_state.json"):
        if not (checkpoint / name).is_file():
            problems.append(f"missing {checkpoint / name}")
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()] if (run / "metrics.jsonl").is_file() else []
    if not records or not all(math.isfinite(r["loss/last_microbatch"]) for r in records):
        problems.append(f"{run}/metrics.jsonl: missing or nonfinite losses")
    trace = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()] if (run / "steps.jsonl").is_file() else []
    if not any(row["kind"] == "dpo" for row in trace):
        problems.append(f"{run}/steps.jsonl: no micro-step on a selected candidate")
    if not any("loss/winner_fm" in r for r in records):
        problems.append(f"{run}/metrics.jsonl: no winner flow-matching loss")
    if not (run / "done.json").is_file():
        problems.append(f"missing {run}/done.json")
if problems:
    print("SMOKE_RAFT FAILED\n  " + "\n  ".join(problems))
    sys.exit(1)
print(f"SMOKE_RAFT OK: {len(winners)} selected candidates, {steps} steps for {' '.join(experts)} in {raft / 'train'}")
PY
