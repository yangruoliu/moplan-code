#!/usr/bin/env python
"""Label contract v2 of a Stage-2 cache, in place and on the CPU: motion of an object is judged on its POINTS, not on
its centroid, and every anchor of a measured plan carries its provenance.

Per clip (<cache>/<sid>.pt, <cache>/plans_real/<sid>.npz, <cache>/scenes/<sid>.npz):
  plans_real   `fill` is added where the plan predates provenance: a known anchor whose fit residual is 0 can only be a
               mask-centroid fill or an exact few-point translation (both translation-only) -> FILL_MASK; a known anchor
               with a residual -> FILL_POINTS; unknown -> FILL_HOLD.  Plans that already carry `fill` are left as they are.
  labels_real  new fields max_point_px [K, T] (largest corner displacement of the object's first-frame box under the
               measured transform at the latent time; rotation counts only at FILL_POINTS anchors), rot_deg [K, T],
               fill [K, T]; `moving` = centroid-moving OR point motion >= move_px (where the plan knows the object),
               `static` = centroid-static AND NOT point motion > still_px: the mask-centroid instrument (which sees the
               fast falls the point tracker loses) is never overruled, the points add what it cannot see (rotation in
               place).  Centroid readings stay in disp / moving_centroid / static_centroid; labels_version = 2.
  scenes       `moving` / `static` of the reward scene (at its own evaluation times and thresholds, 3 % / 1 % of the
               diagonal) are extended the same way (`moving_centroid` / `static_centroid` kept); the reward formula
               itself does not change.
Objects of the measured scene that are not in the plan (normally none) keep their centroid labels.

usage: relabel_v2.py --cache DIR [--ids F] [--shard-index i --num-shards n] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import motion_plan as mp  # noqa: E402
from scripts.control.reward import MOVE_FRAC, STILL_FRAC  # noqa: E402

VERSION = 2


def provenance_from_residuals(plan: mp.MotionPlan) -> mp.MotionPlan:
    """A plan saved before `fill` existed, with its provenance reconstructed from what was saved (see the module doc)."""
    from dataclasses import replace
    objects = []
    for obj in plan.objects:
        if len(obj.fill) == len(obj.known) and not np.all(np.asarray(obj.fill) == mp.FILL_UNKNOWN):
            objects.append(obj)
            continue
        residual = obj.residual_px if len(obj.residual_px) == len(obj.known) else np.zeros(len(obj.known))
        fill = np.where(obj.known, np.where(residual > 0, mp.FILL_POINTS, mp.FILL_MASK), mp.FILL_HOLD).astype(np.int8)
        objects.append(replace(obj, fill=fill))
    return replace(plan, objects=objects, source=plan.source if "|fill_reconstructed" in plan.source else f"{plan.source}|fill_reconstructed")


def point_readings(plan: mp.MotionPlan, times, slots) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(max_point_px [K, T], rot_deg [K, T], fill [K, T] int8, known [K, T]) of the plan objects listed in `slots` at `times`
    (seconds): the transform at each time (rigid interpolation), rotation trusted only between FILL_POINTS anchors."""
    index = {int(o.slot): o for o in plan.objects}
    count, length = len(slots), len(times)
    moved, rot, fill, known = np.zeros((count, length)), np.zeros((count, length)), np.full((count, length), mp.FILL_HOLD, np.int8), np.zeros((count, length), bool)
    anchor_times = np.concatenate(([0.0], plan.anchors))
    for k, slot in enumerate(slots):
        obj = index.get(int(slot))
        if obj is None:
            fill[k] = mp.FILL_UNKNOWN
            continue
        provenance = np.concatenate(([mp.FILL_POINTS], obj.provenance()))
        flags = np.concatenate(([True], obj.known))
        ys, xs = np.nonzero(obj.mask)
        if len(xs) == 0:
            continue
        corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]], np.float64)
        centre = obj.centroid
        for t, when in enumerate(times):
            if when <= 0:
                known[k, t], fill[k, t] = True, mp.FILL_POINTS
                continue
            affine, ok = mp.affine_at(obj, plan.anchors, float(when))
            upper = int(np.searchsorted(anchor_times, when, side="left"))
            lower = max(upper - 1, 0)
            upper = min(upper, len(anchor_times) - 1)
            segment = [provenance[lower], provenance[upper]] if ok else []
            trust = ok and all(p == mp.FILL_POINTS for p in segment)
            known[k, t] = ok
            fill[k, t] = (max(segment, key=lambda p: {mp.FILL_POINTS: 0, mp.FILL_FEW: 1, mp.FILL_MASK: 2, mp.FILL_UNKNOWN: 3, mp.FILL_HOLD: 4}[int(p)])
                          if ok else mp.FILL_HOLD)
            if not ok:
                continue
            if trust:
                moved[k, t] = np.linalg.norm(mp.apply_affine(affine, corners) - corners, axis=1).max()
                rot[k, t] = np.degrees(mp._polar(affine[:, :2])[0])
            else:
                moved[k, t] = np.linalg.norm(mp.apply_affine(affine, centre[None])[0] - centre)
    return moved, rot, fill, known


def relabel(cache: Path, sid: str, move_px: float, still_px: float, dry_run: bool) -> dict:
    plan_path = cache / "plans_real" / f"{sid}.npz"
    payload_path = cache / f"{sid}.pt"
    if not plan_path.is_file() or not payload_path.is_file():
        return {"sample_id": sid, "skip": "no plan or payload"}
    plan = provenance_from_residuals(mp.load_plan(plan_path))
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    labels = payload["labels_real"]
    slots = [int(v) for v in labels["slots"].tolist()]
    times = [float(v) for v in payload["frame_seconds"]]
    moved, rot, fill, known = point_readings(plan, times, slots)
    centroid_moving = np.asarray(labels.get("moving_centroid", labels["moving"])).astype(bool)
    centroid_static = np.asarray(labels.get("static_centroid", labels["static"])).astype(bool)
    valid = np.asarray(labels["valid"]).astype(bool)
    # the mask-centroid reading (the instrument that sees fast falls the point tracker loses) is never overruled: the point
    # reading ADDS movers it cannot see (rotation in place) and removes "static" where the points say otherwise
    moving = (centroid_moving | (known & (moved >= move_px))) & valid
    static = (centroid_static & ~(known & (moved > still_px))) & valid
    changed = {"moving": int((moving != np.asarray(labels["moving"]).astype(bool)).sum()), "static": int((static != np.asarray(labels["static"]).astype(bool)).sum())}
    labels.update(moving_centroid=torch.from_numpy(centroid_moving), static_centroid=torch.from_numpy(centroid_static),
                  moving=torch.from_numpy(moving), static=torch.from_numpy(static), max_point_px=torch.from_numpy(moved.astype(np.float32)),
                  rot_deg=torch.from_numpy(rot.astype(np.float32)), fill=torch.from_numpy(fill), labels_version=VERSION)
    payload["labels_real"] = labels
    payload["moving_objects"] = int(moving.any(1).sum())
    payload["labels_version"] = VERSION
    scene_path = cache / "scenes" / f"{sid}.npz"
    scene_changed = None
    if scene_path.is_file():
        scene = dict(np.load(scene_path, allow_pickle=True))
        scene_slots = [int(v) for v in scene["slots"]]
        s_moved, _, s_fill, s_known = point_readings(plan, [float(v) for v in scene["times"]], scene_slots)
        c_moving = scene.get("moving_centroid", scene["moving"]).astype(bool)
        c_static = scene.get("static_centroid", scene["static"]).astype(bool)
        ok = scene["real_ok"].astype(bool)
        diagonal = float(scene["diagonal"])                                  # the reward's own thresholds (3 % / 1 % of the frame diagonal)
        s_moving = (c_moving | (s_known & (s_moved >= MOVE_FRAC * diagonal))) & ok
        s_static = (c_static & ~(s_known & (s_moved > STILL_FRAC * diagonal))) & ok
        scene_changed = {"moving": int((s_moving != scene["moving"].astype(bool)).sum()), "static": int((s_static != scene["static"].astype(bool)).sum())}
        scene.update(moving_centroid=c_moving, static_centroid=c_static, moving=s_moving, static=s_static, max_point_px=s_moved.astype(np.float32),
                     fill=s_fill, labels_version=np.int32(VERSION))
        if not dry_run:
            np.savez(scene_path, **scene)
    if not dry_run:
        mp.save_plan(plan_path, plan)
        torch.save(payload, payload_path)
    return {"sample_id": sid, "objects": len(slots), "changed": changed, "scene_changed": scene_changed,
            "fill_unknown_share": float((fill[known] == mp.FILL_UNKNOWN).mean()) if known.any() else 0.0,
            "movers": int(moving.any(1).sum()), "movers_centroid": int(centroid_moving.any(1).sum())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--ids", default="", help="sample ids (default: every <sid>.pt of the cache)")
    p.add_argument("--move-px", type=float, default=8.0)
    p.add_argument("--still-px", type=float, default=4.0)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    cache = Path(a.cache)
    ids = [line.strip() for line in open(a.ids) if line.strip()] if a.ids else sorted(q.stem for q in cache.glob("*.pt") if not q.name.startswith("prompt_embeds"))
    total = {"clips": 0, "skipped": 0, "moving_cells_changed": 0, "static_cells_changed": 0, "movers": 0, "movers_centroid": 0, "fill_unknown_share": []}
    for sid in ids[a.shard_index:: a.num_shards]:
        result = relabel(cache, sid, a.move_px, a.still_px, a.dry_run)
        if "skip" in result:
            total["skipped"] += 1
            continue
        total["clips"] += 1
        total["moving_cells_changed"] += result["changed"]["moving"]
        total["static_cells_changed"] += result["changed"]["static"]
        total["movers"] += result["movers"]
        total["movers_centroid"] += result["movers_centroid"]
        total["fill_unknown_share"].append(result["fill_unknown_share"])
    total["fill_unknown_share"] = float(np.mean(total["fill_unknown_share"])) if total["fill_unknown_share"] else None
    print("RELABEL_V2", json.dumps(total))


if __name__ == "__main__":
    main()
