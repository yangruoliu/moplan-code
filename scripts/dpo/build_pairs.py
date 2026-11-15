#!/usr/bin/env python
"""Winners (RAFT) or preference pairs (Diffusion-DPO) from scored samples of the generator.

Input: clips generated under ONE condition arm with several seeds (infer_stage2_cond_v2.py --num-seeds S) and scored
against the real continuation (score_generated.py track).  For every sample the best and the worst seed by reward form
a pair when they differ by at least --margin (equal rewards: the lower seed index is chosen); both clips are encoded
by the VAE (posterior mode).  --winner-only encodes only the best clip (RAFT); --expected-group-size G skips samples whose
group of G candidates is incomplete.

  <out>/<sid>.pt    {"win", "lose" [16, T, 32, 56] raw latents (half), "reward_win", "reward_lose", "seed_win", "seed_lose",
                     "kind" (the arm), "sample_id", "candidate_count", "winner_only"}   ("lose" only without --winner-only)
  <out>/pairs.jsonl one line per sample: all seed rewards, the reward spread, the chosen (win, lose) seeds or null
  <out>/selection_receipt.json  counts and settings of the selection

Only training clips may be used (the rewards compare with their real continuations).

usage: build_pairs.py --generated DIR --model-dir WAN --out DIR [--arm pred] [--margin 0.3] [--ids train_ids.txt] [--suffix .reward_v5.json]
                      [--winner-only] [--expected-group-size 8] [--drop-frames]
       (the RAFT round: --margin 0.2 --suffix .reward_v5.json --winner-only --expected-group-size 8)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cache_stage2_oracle import preprocess  # noqa: E402
from scripts.control.score_generated import parse_name  # noqa: E402


def choose(rewards: dict[int, float], margin: float) -> tuple[int, int] | None:
    """(winner seed, loser seed) or None when the best and the worst seed are closer than `margin`."""
    if len(rewards) < 2 or not all(math.isfinite(value) for value in rewards.values()):
        return None
    best, worst = max(rewards, key=lambda k: (rewards[k], -k)), min(rewards, key=lambda k: (rewards[k], k))
    return (best, worst) if rewards[best] - rewards[worst] >= margin else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated", required=True)
    p.add_argument("--model-dir", default=os.environ.get("WAN_MODEL_DIR", ""), help="Wan2.2-I2V-A14B-Diffusers directory (default $WAN_MODEL_DIR)")
    p.add_argument("--out", required=True)
    p.add_argument("--arm", default="pred")
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--ids", default="", help="only these samples (the training split)")
    p.add_argument("--suffix", default=".reward.json", help="reward files of the samples (.reward_v5.json for reward v5)")
    p.add_argument("--drop-frames", action="store_true", help="delete each sample's .npy frames once its pair is written (online rounds: disk)")
    p.add_argument("--winner-only", action="store_true", help="encode only the winner for reward-weighted fine-tuning")
    p.add_argument("--expected-group-size", type=int, default=0, help="require a complete candidate group (the RAFT round uses eight)")
    a = p.parse_args()
    if not a.model_dir:
        p.error("--model-dir (or WAN_MODEL_DIR) is required")
    from diffusers import AutoencoderKLWan

    generated, out = Path(a.generated), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allowed = {line.strip() for line in Path(a.ids).read_text().splitlines() if line.strip()} if a.ids else None
    rewards = defaultdict(dict)
    for path in sorted(generated.glob(f"{a.arm}__*{a.suffix}")):
        arm, sid, seed = parse_name(path.name[: -len(a.suffix)])
        if allowed is None or sid in allowed:
            rewards[sid][seed] = float(json.loads(path.read_text())["reward"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKLWan.from_pretrained(Path(a.model_dir) / "vae", torch_dtype=torch.float32).to(device).eval()

    def encode(sid: str, seed: int) -> torch.Tensor:
        path = generated / f"{a.arm}__{sid}__s{seed}.npy"
        if seed == 0 and not path.is_file():
            path = generated / f"{a.arm}__{sid}.npy"
        frames = np.load(path)
        with torch.no_grad():
            return vae.encode(preprocess(frames, frames.shape[1], frames.shape[2]).to(device)).latent_dist.mode()[0].float().cpu().half()

    made, gaps = 0, []
    with open(out / "pairs.jsonl", "w") as log:
        for sid in sorted(rewards):
            complete = not a.expected_group_size or len(rewards[sid]) == a.expected_group_size
            picked = choose(rewards[sid], a.margin) if complete else None
            spread = max(rewards[sid].values()) - min(rewards[sid].values())
            gaps.append(spread)
            log.write(json.dumps({"sample_id": sid, "rewards": rewards[sid], "spread": round(spread, 4), "pair": picked}) + "\n")
            if picked is None:
                continue
            win, lose = picked
            payload = {"win": encode(sid, win), "reward_win": rewards[sid][win], "reward_lose": rewards[sid][lose],
                       "seed_win": win, "seed_lose": lose, "kind": a.arm, "sample_id": sid,
                       "candidate_count": len(rewards[sid]), "winner_only": a.winner_only}
            if not a.winner_only:
                payload["lose"] = encode(sid, lose)
            torch.save(payload, out / f"{sid}.pt")
            made += 1
    if a.drop_frames:
        for path in generated.glob(f"{a.arm}__*.npy"):
            path.unlink()
    gaps = np.array(gaps) if gaps else np.zeros(1)
    (out / "selection_receipt.json").write_text(json.dumps({
        "independent_conditions_scored": len(rewards),
        "candidate_videos_scored": sum(len(group) for group in rewards.values()),
        "conditions_selected": made, "expected_group_size": a.expected_group_size,
        "winner_only": a.winner_only, "margin": a.margin,
        "source_generated": generated.resolve().name, "condition_ids": sorted(rewards)}, indent=2) + "\n")
    print(f"DPO_PAIRS {made} pairs from {len(rewards)} samples (margin {a.margin}); reward spread across seeds: median {np.median(gaps):.3f}, "
          f"share >= 0.1: {np.mean(gaps >= 0.1):.2f}, >= 0.3: {np.mean(gaps >= 0.3):.2f}")


if __name__ == "__main__":
    main()
