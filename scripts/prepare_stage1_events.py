#!/usr/bin/env python
"""Create an isolated, scene-disjoint event curriculum from FOT observations.

No Physics-IQ image, caption, flow, or score is consumed. The analytic renderer
is a controlled kinematic curriculum, not a replacement for real simulation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.piq_gt_v2 import flyobj_compose as fot

TIMES = [.1, .2, .4, .8, 1.2, 2.4, 4.]


def distinct_event_semantics(a, b):
    """Exclude random magnitude differences masquerading as interventions."""
    am, bm = a["motions"], b["motions"]
    if bool(am) != bool(bm):
        return True
    if not am:
        return False
    if {m["id"] for m in am} != {m["id"] for m in bm}:
        return True
    other = {m["id"]: m for m in bm}
    for motion in am:
        paired = other[motion["id"]]
        if motion["profile"] == paired["profile"] == "spin":
            if motion["omega"] * paired["omega"] < 0:
                return True
        elif "u" in motion and "u" in paired:
            if sum(x*y for x,y in zip(motion["u"],paired["u"])) < .8:
                return True
    return False


def plan(scene, labels):
    objects = scene["objects"]
    ctx = {"objects": objects, "by_id": {o["id"]: o for o in objects},
           "contours": {o["id"]: fot.contour_points(labels, o["id"]) for o in objects}}
    rng = random.Random("event-v1-" + scene["scene_id"])
    # Original diverse directions/contact/rotation keep this from becoming a
    # universal downward-motion curriculum. Do not reuse absent-object zero GT.
    variants = [deepcopy(v) for v in scene["variants"] if v.get("contrast") != "absent"][:4]
    variants += fot.extend_scene(ctx, rng, variants, count=2)
    scene_gravity = rng.uniform(3000., 7000.)
    for obj in objects[:2]:
        oid, noun = obj["id"], obj["name"]
        falling = fot.plan_variant(ctx, rng, "fall", oid, None)
        if falling is None:
            continue
        # No outcome verb or screen direction in the base event prompt.
        falling["caption"] = f"The {noun} is held above a surface. Its support is removed. The camera is fixed."
        falling["paraphrases"] = [
            f"The {noun} is released from rest above a surface. The camera is fixed.",
            f"A holder lets go of the {noun}. The camera is fixed.",
            f"The {noun} is dropped. The camera is fixed.",
        ]
        falling["event"] = "release"
        variants.append(falling)
        # The holder stays put in the matched negative. The cue is the action,
        # not presence of 'released', 'held', the noun, or the visual backdrop.
        variants.append({"kind": "static", "mover": oid, "motions": [], "event": "hold",
                         "caption": f"The {noun} is held above a surface. Its support remains in place. The camera is fixed.",
                         "paraphrases": [f"The {noun} is not released. It remains held in place. The camera is fixed.",
                                          f"A holder keeps holding the {noun}. The camera is fixed."]})
    result = []
    for index, variant in enumerate(variants):
        for motion in variant["motions"]:
            if motion["profile"] == "fall":
                motion["g"] = scene_gravity
        variant["sample_id"] = f"event_{scene['scene_id']}_v{index:02d}"
        result.append(variant)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["targets", "finalize"])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--train-scenes", type=int, default=0)
    p.add_argument("--val-scenes", type=int, default=0)
    args = p.parse_args()
    root = args.output
    torch.set_num_threads(1)
    if args.phase == "targets":
        for folder in ("targets", "scenes", "parts"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        paths = []
        for split, limit in (("train", args.train_scenes), ("val", args.val_scenes)):
            selected = sorted((args.source / "scenes").glob(f"fot_{split}_*/scene.json"))
            paths += selected[:limit or None]
        rows, pairs, stats = [], {}, Counter()
        fot.HORIZONS = TIMES
        for path in paths[args.shard_index::args.num_shards]:
            scene = json.loads(path.read_text())
            labels = cv2.imread(str(path.parent / "labels.png"), cv2.IMREAD_UNCHANGED)
            frame = path.parent / "frame.jpg"
            kept = []
            for v in plan(scene, labels):
                target, movers = fot.build_target(scene, v, labels, frame)
                if v["kind"] != "static" and not movers:
                    stats["skipped_no_observable_motion"] += 1
                    continue
                target["event_recipe"] = "stage1_event_curriculum_v2"
                target["source_scene_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                torch.save(target, root / "targets" / (v["sample_id"] + ".pt"))
                row = {"sample_id": v["sample_id"], "scenario": fot.SCENARIO,
                       "split": scene["split"], "video_path": str(frame), "caption": v["caption"],
                       "source": "stage1_event_curriculum_v2", "motion_kind": v["kind"],
                       "scene_id": scene["scene_id"], "event": v.get("event", v["kind"])}
                rows.append(row)
                kept.append((v, target))
                stats[scene["split"] + "/" + row["event"]] += 1
            for v, target in kept:
                options = [{"relation": "invariant", "caption": caption}
                           for caption in v.get("paraphrases", [v["caption"].replace("Static shot with no camera movement.", "The camera stays fixed.")])]
                # Favor same-object action contrasts, then noun binding. Pair
                # only observable differences on shared, trustworthy support.
                opposite = "hold" if v.get("event") == "release" else "release"
                others = sorted(kept, key=lambda x: (x[0]["mover"] != v["mover"],
                    x[0].get("event") != opposite, x[0]["sample_id"]))
                added = []
                for other, ot in others:
                    if other["sample_id"] == v["sample_id"]:
                        continue
                    if not distinct_event_semantics(v, other):
                        continue
                    common = target["visibility"].bool() & ot["visibility"].bool()
                    if not bool((torch.where(common, target["flow"]-ot["flow"], 0).abs() > .01).any()):
                        continue
                    added.append({"relation": "counterfactual", "sample_id": other["sample_id"],
                                  "caption": other["caption"]})
                    if len(added) == len(options):
                        break
                pairs[v["sample_id"]] = options + added
            (root / "scenes" / (scene["scene_id"] + ".json")).write_text(json.dumps(
                {**scene, "variants": [v for v, _ in kept], "source_frame": str(frame), "horizon_seconds": TIMES},
                default=lambda value: value.item()))
        (root / "parts" / f"{args.shard_index:03d}.json").write_text(json.dumps({"rows": rows, "pairs": pairs, "stats": stats}))
        print(json.dumps({"shard": args.shard_index, **stats}), flush=True)
        return
    rows, pairs, stats = [], {}, Counter()
    parts = sorted((root / "parts").glob("*.json"))
    if len(parts) != args.num_shards:
        raise ValueError(f"incomplete target generation: {len(parts)}/{args.num_shards} shards")
    for path in parts:
        part = json.loads(path.read_text()); rows += part["rows"]; pairs.update(part["pairs"]); stats.update(part["stats"])
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate samples across shards")
    (root / "manifests").mkdir(exist_ok=True)
    for split in ("train", "val"):
        selected = sorted((r for r in rows if r["split"] == split), key=lambda r: r["sample_id"])
        (root / "manifests" / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
        members = {r["sample_id"] for r in selected}
        (root / f"event_pairs_{split}.json").write_text(json.dumps({"schema": "stage1_event_pairs_v1",
            "pairs": {sid: specs for sid, specs in pairs.items() if sid in members}}))
    receipt = {"schema": "stage1_event_curriculum_v2", "rows": len(rows), "counts": stats,
               "horizon_seconds": TIMES, "source": str(args.source),
               "benchmark_inputs_used": False, "split_unit": "original FOT scene and source-image split",
               "shared_physics": "one screen-space gravity per scene, shared across event variants",
               "counterfactual_filter": "activity, actor identity, direction or spin sense must differ",
               "limitation": "Analytic 2D kinematics; hidden support is specified by text; no real tool interaction rendering."}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)

if __name__ == "__main__":
    main()
