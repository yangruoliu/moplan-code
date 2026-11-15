#!/usr/bin/env python
"""Event curriculum v4.4 = v4.3 speeds/durations + PER-EVENT motion constraints.

Problem in v4.3: the generic `sustain` pass only recognised delayed decel motions as "partners";
every other actor had its cap replaced by min(plan, frame edge), which discarded the first-obstacle
contact and the event-specific endpoints.  Effects on the exported flow labels:
  stop_short  most actors crossed the contact gap although the caption says they stop short
  knock_over  most topples started at the stale (old-profile) time; some never reached the target
  pair        most pairs travelled past each other (second motion never re-timed)
  collide / towards / push with an obstacle: the actor could slide through the obstacle

v4.4 dispatches by event:
  pair         one coupled contact: both re-timed with the same v0/T, each capped at gap/2
  stop_short   the pre-contact endpoint is kept (<= 0.8 x contact gap, and <= the original short cap)
  knock_over   must reach contact (plan raised so the hit happens by <= 3.2 s); topple start := hit time
  collide      must reach contact the same way; pushed partner starts at the hit time
  towards/push/roll  stop at the first obstacle or the frame edge; a follower that is never reached is
               dropped (nothing gets pushed), otherwise it starts at the recomputed hit time
  topple/spin/fall/lift/lower unchanged.
Every re-timed actor records its geometry (edge, contact, obstacle, rule) for the event audit.
Phases, captions, split and sample ordering are those of v4.3; sample ids use the prefix event44_.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.piq_gt_v2 import flyobj_compose as fot  # noqa: E402
from scripts import prepare_stage1_events_v3 as v3  # noqa: E402
from scripts import prepare_stage1_events_v4 as v4  # noqa: E402  (installs speed mapping, approach_motion, KIND_WEIGHTS, keep_v2_row)

SOURCE_TAG = "stage1_event_curriculum_v4_4"
v3.SOURCE_TAG = SOURCE_TAG
v4.SOURCE_TAG = SOURCE_TAG
ID_PREFIX = "event44"
KNOCK_REACH_BY_S = 3.2          # knock_over / collide: contact must happen by this time (caption promises it)


def hit_time(motion: dict, distance: float) -> float | None:
    """Time at which a decel profile x(t) = plan (1 - (1 - t/T)^2) first reaches `distance`; None if never."""
    plan, period = float(motion["plan"]), float(motion["stop_time"])
    if distance > plan or plan <= 0:
        return None
    return float((1.0 - math.sqrt(max(0.0, 1.0 - distance / plan))) * period)


def travel(motion: dict, t: float) -> float:
    tau = max(0.0, t - float(motion.get("start", 0.0)))
    period = float(motion["stop_time"])
    x = min(tau, period) / period
    return float(min(float(motion["plan"]) * (1.0 - (1.0 - x) ** 2), float(motion["cap"])))


def retime(motion: dict, rng, v0: float | None = None) -> float:
    """PIQ-like initial speed from the planner's distance (unchanged v4.3 mapping), T ~ U(3, 8), plan = v0 T / 2."""
    v0 = v4.speed_from_plan(float(motion["plan"])) if v0 is None else v0
    period = rng.uniform(3.0, 8.0)
    motion["stop_time"] = float(period)
    motion["plan"] = float(v0 * period / 2.0)
    return v0


def reach_by(motion: dict, distance: float, by_seconds: float) -> None:
    """Raise the planned distance so the profile reaches `distance` no later than `by_seconds` (<= T)."""
    period = float(motion["stop_time"])
    t_req = min(by_seconds, period)
    frac = 1.0 - (1.0 - t_req / period) ** 2
    motion["plan"] = float(max(float(motion["plan"]), 1.02 * distance / max(frac, 1e-3)))


def geometry(ctx, actor_id: int, u: np.ndarray, exclude=()):
    obj = ctx["by_id"][actor_id]
    edge = float(fot.frame_distance(obj["bbox"], u))
    contact, other = fot.first_obstacle(ctx, actor_id, u, set(exclude))
    return edge, (float(contact) if math.isfinite(contact) else None), other


def sustain_v44(variants, ctx, rng):
    for variant in variants:
        kind = variant.get("event") or variant.get("kind")
        motions = variant.get("motions") or []
        if not motions:
            continue
        actor = motions[0]
        if actor.get("profile") != "decel" or float(actor.get("start", 0.0)) != 0.0:
            continue                                   # topple / spin / fall / lift / lower: untouched
        u = np.asarray(actor["u"], np.float32)
        edge, contact, other = geometry(ctx, actor["id"], u)
        if kind == "pair":
            partner = motions[1]
            gap = contact if (other == partner["id"] and contact is not None) else 2.0 * min(float(actor["cap"]), float(partner["cap"]))
            v0 = v4.speed_from_plan(float(actor["plan"]))
            period = rng.uniform(3.0, 8.0)
            for motion in (actor, partner):
                m_u = np.asarray(motion["u"], np.float32)
                motion["stop_time"] = float(period)
                motion["plan"] = float(v0 * period / 2.0)
                motion["cap"] = float(min(gap / 2.0, fot.frame_distance(ctx["by_id"][motion["id"]]["bbox"], m_u)))
            actor["geometry"] = {"edge": edge, "contact": gap, "obstacle": partner["id"], "rule": "pair"}
            continue
        old_cap = float(actor["cap"])
        v0 = retime(actor, rng)
        follower = next((m for m in motions[1:] if float(m.get("start", 0.0)) > 0.0), None)   # pushed partner (decel) or topple
        if kind == "stop_short":
            gap = contact if contact is not None else edge
            actor["cap"] = float(min(old_cap, 0.8 * gap, edge))
            actor["geometry"] = {"edge": edge, "contact": gap, "obstacle": other, "rule": "stop_short"}
            continue
        cap = min(edge, contact) if contact is not None else edge
        rule = kind
        if kind in ("knock_over", "collide"):
            if follower is None or other != follower["id"] or contact is None:
                raise ValueError(f"{variant.get('sample_id')}: {kind} without a reachable partner")
            if edge < contact:
                # the frame edge comes before the partner: the caption promises a contact the actor
                # cannot make inside the frame -> drop the variant
                variant["drop"] = "edge_before_contact"
                continue
            reach_by(actor, cap, KNOCK_REACH_BY_S)
        actor["cap"] = float(min(float(actor["plan"]), cap))
        actor["geometry"] = {"edge": edge, "contact": contact, "obstacle": other, "rule": rule}
        if follower is None:
            continue
        t_hit = hit_time(actor, cap) if (other is not None and other == follower["id"] and contact is not None) else None
        if t_hit is None:                                   # never reaches it: nothing is pushed / knocked
            variant["motions"] = [actor]
            variant["dropped_follower"] = follower["id"]
            continue
        follower["start"] = float(t_hit)
        if follower.get("profile") == "decel":
            v0p = v0 * rng.uniform(0.5, 1.0)
            period_p = rng.uniform(2.0, 5.0)
            follower["stop_time"] = float(period_p)
            follower["plan"] = float(v0p * period_p / 2.0)        # its own geometric cap (edge / next obstacle) unchanged
    return [variant for variant in variants if not variant.get("drop")]


v4.sustain = sustain_v44                     # plan_scene_v4 looks the name up in v4's globals at call time


def plan_scene_v44(scene, labels):
    variants = v4.plan_scene_v4(scene, labels)
    for index, variant in enumerate(variants):
        variant["sample_id"] = f"{ID_PREFIX}_{scene['scene_id']}_v{index:02d}"
    return variants


v3.plan_scene_v3 = plan_scene_v44            # run_targets calls this name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["targets", "merge", "prompts"])
    v3.add_input_arguments(p)
    p.add_argument("--output", type=Path, required=True, help="v4.4-only root (targets, scenes, parts)")
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
