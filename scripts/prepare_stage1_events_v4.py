#!/usr/bin/env python
"""Event curriculum v4: v3 relational events + FOT explicit-direction events, all with SUSTAINED motion.

Why: a model trained on v2/v3 predicts motion that stalls after about 0.4-1 s, while real objects keep
rolling / falling / sliding through the 4 s horizon. The FOT push/collide recipes stop at 0.8-3 s
(decelerating), so most v2 push movers are no longer moving at 4 s and the model learns "nudge, then
stop".

v4 regenerates, on the SAME FOT scenes / first frames / SAM2 conditions / split as v2 and v3:
  explicit   FOT v1 planner kinds push / roll / towards / pair / lift / lower / spin / fall with the
             original direction-word captions ("... leftward", "toward the right", ...); every
             decelerating motion is lengthened 2-3.5x in BOTH stop time and planned distance, so the
             initial speed (v0 = 2*plan/T) and the early horizons are unchanged
  relational v3 collide / knock_over / stop_short / topple with the same v0 and contact time as v3
             but 2.5-5 s of deceleration; pushed partners lengthened the same way
Phases are the v3 ones (targets / merge / prompts); merge keeps NO v2 dynamic rows (they are the
short-duration ones), only the v4 rows + the unchanged v2 static source and fixed v2 validation.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.piq_gt_v2 import flyobj_compose as fot  # noqa: E402
from scripts import prepare_stage1_events_v3 as v3  # noqa: E402  (installs the topple displacement field)

SOURCE_TAG = "stage1_event_curriculum_v4"
v3.SOURCE_TAG = SOURCE_TAG

# ----------------------------------------------------------------------------- sustained kinematics

def speed_from_plan(plan_old):
    """Map FOT's adverb-driven distance (gently 40-100 px, default 70-220, hard 150-240) onto a Physics-IQ-like
    initial speed: Physics-IQ ground-truth objects move ~20-100 frame px/s; FOT's implied v0 = 2*plan/T was 50-550 px/s,
    so objects crossed the 640-px frame in ~1-2 s and nothing could be 'sustained' inside the frame."""
    return float(np.clip(20.0 + 80.0 * (plan_old - 40.0) / 200.0, 20.0, 100.0))


def sustain(variants, ctx, rng):
    """Re-time every decelerating actor motion to PIQ-like speed with a 3-8 s deceleration, and recompute
    the geometric cap and the contact time so the labels stay consistent.

    FOT decel: x(t) = plan * (1 - (1 - t/T)^2), v0 = 2*plan/T, stop at min(plan, cap) with cap = frame-edge or
    first-contact distance. v4 draft 1 scaled T only (v0 cut 2-3x); draft 2 scaled plan with T but kept the
    old cap (actual stop median 0.75 s, partners started at the old hit time). Here, per actor (start == 0):
      v0 = speed_from_plan(plan_old);  T' ~ U(3, 8) s;  plan' = v0 T'/2
      no partner:   cap' = min(plan', edge)                (edge recomputed from the bbox)
      with partner: cap' = contact (unchanged);  partner.start = t(contact) on the NEW profile;
                    partner: v0_p = v0 * U(0.5, 1); T_p ~ U(2, 5); plan_p = v0_p T_p / 2; its geometric cap unchanged
    Rolls (FOT T already 3-8 s) get the same speed mapping so their speed matches too.
    """
    for variant in variants:
        motions = variant.get("motions") or []
        if not motions or motions[0].get("profile") != "decel" or motions[0].get("start", 0.0) != 0.0:
            continue
        actor = motions[0]
        if actor["plan"] <= 0:
            continue
        v0 = speed_from_plan(actor["plan"])
        T = rng.uniform(3.0, 8.0)
        actor["stop_time"] = T
        actor["plan"] = v0 * T / 2.0
        u = np.asarray(actor["u"], np.float32)
        obj = ctx["by_id"][actor["id"]]
        partner = next((m for m in motions[1:] if m.get("profile") == "decel" and m.get("start", 0.0) > 0), None)
        if partner is None:
            actor["cap"] = float(min(actor["plan"], fot.frame_distance(obj["bbox"], u)))
        else:
            contact = float(actor["cap"])                                   # unchanged geometry
            if contact >= actor["plan"]:                                    # too slow to reach: make it reach at ~T/2
                actor["plan"] = contact * 1.6
                actor["stop_time"] = 2.0 * actor["plan"] / v0
            x = 1.0 - math.sqrt(max(0.0, 1.0 - min(contact / actor["plan"], 1.0)))
            partner["start"] = float(x * actor["stop_time"])
            v0p = v0 * rng.uniform(0.5, 1.0)
            Tp = rng.uniform(2.0, 5.0)
            partner["stop_time"] = Tp
            partner["plan"] = v0p * Tp / 2.0                                # cap (edge/obstacle) unchanged
    return variants


def approach_motion(obj_id, rng, u, gap, stop_short=False):
    """v3 approach re-timed to PIQ-like speed: v0 ~ U(40,160) px/s, T ~ U(3,8) s (long enough to reach the
    target before stopping), plan = v0 T/2, cap = gap (stops at contact); hit time ~ gap / v0."""
    v0 = rng.uniform(20.0, 100.0)
    if stop_short:
        cap = gap * rng.uniform(0.35, 0.7)
        T = max(rng.uniform(3.0, 8.0), 2.4 * cap / v0)
        plan = max(v0 * T / 2.0, cap * 1.05)
        return {"id": obj_id, "profile": "decel", "u": u.tolist(), "plan": plan, "stop_time": T,
                "cap": cap, "start": 0.0, "spin": 0.0}, None
    T = max(rng.uniform(3.0, 8.0), 2.4 * gap / v0)
    plan = max(v0 * T / 2.0, gap * 1.2)
    hit_time = (1.0 - math.sqrt(max(0.0, 1.0 - gap / plan))) * T
    return {"id": obj_id, "profile": "decel", "u": u.tolist(), "plan": plan, "stop_time": T,
            "cap": gap, "start": 0.0, "spin": 0.0}, hit_time


v3.approach_motion = approach_motion

# FOT v1 kinds for the explicit-direction half (falls kept small: the v2 release curriculum is what
# over-taught "down"; the drop group still needs some)
fot.KIND_WEIGHTS = {"push": .30, "roll": .20, "towards": .15, "pair": .05, "lift": .05, "lower": .05,
                    "spin": .10, "fall": .10}

# ----------------------------------------------------------------------------- planning


def normalize_fot_variant(variant):
    """FOT v1 variants -> the v3 row contract (event / mover / paraphrases)."""
    motions = variant.get("motions") or []
    if not motions:
        return None
    variant = dict(variant)
    variant.setdefault("event", variant.get("kind", "explicit"))
    variant.setdefault("mover", motions[0]["id"])
    paraphrases = variant.get("paraphrases") or variant.get("alternatives") or []
    variant["paraphrases"] = [p for p in paraphrases if isinstance(p, str) and p != variant["caption"]]
    return variant


def plan_scene_v4(scene, labels):
    objects = scene["objects"]
    ctx = {"objects": objects, "by_id": {o["id"]: o for o in objects},
           "contours": {o["id"]: fot.contour_points(labels, o["id"]) for o in objects}}
    rng = random.Random("event-v4-" + scene["scene_id"])
    explicit = [v for v in (normalize_fot_variant(x) for x in fot.plan_scene(ctx, rng)) if v is not None]
    relational = _base_plan_scene_v3(scene, labels)       # own rng, sustained via the patched approach
    variants = sustain(explicit + relational, ctx, rng)
    for index, variant in enumerate(variants):
        variant["sample_id"] = f"event4_{scene['scene_id']}_v{index:02d}"
    return variants


_base_plan_scene_v3 = v3.plan_scene_v3
v3.plan_scene_v3 = plan_scene_v4   # run_targets calls this name


def direction_of(motion):
    if "u" not in motion:
        return {"fall": "down", "lower": "down", "lift": "up"}.get(motion.get("profile"), motion.get("profile", "other"))
    return _base_direction_of(motion)


_base_direction_of = v3.direction_of
v3.direction_of = direction_of
_base_keep_v2_row = v3.keep_v2_row


def keep_v2_row(row, args):
    """merge: no short-duration v2 push/towards/... rows; v2 falls/lowers only via --keep-fall/--keep-lower
    (dropping them entirely removes the only plain-fall supervision and makes predicted falls drift sideways)."""
    if row.get("event") in ("fall", "lower"):
        return _base_keep_v2_row(row, args)
    return False


v3.keep_v2_row = keep_v2_row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["targets", "merge", "prompts"])
    v3.add_input_arguments(p)
    p.add_argument("--output", type=Path, required=True, help="v4-only root (targets, scenes, parts)")
    p.add_argument("--combined", type=Path, default=None, help="combined training root for merge/prompts")
    p.add_argument("--pad-length", type=int, default=64)
    p.add_argument("--keep-fall", type=float, default=0.0)
    p.add_argument("--keep-lower", type=float, default=0.0)
    p.add_argument("--wan-model", default=os.environ.get("WAN_MODEL_DIR", ""),
                   help="Wan2.2-I2V-A14B-Diffusers directory (tokenizer, text encoder); default $WAN_MODEL_DIR")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()
    v3.check_phase_inputs(p, args)
    torch.set_num_threads(1)
    if args.phase == "targets":
        v3.run_targets(args)
    elif args.phase == "merge":
        v3.run_merge(args)
    else:
        v3.run_prompts(args)


if __name__ == "__main__":
    main()
