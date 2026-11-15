#!/usr/bin/env python
"""Measured plans against the mask instrument: anchors whose point fit contradicts the object's mask centroid are
corrected, in place and on the CPU.

The point tracker has one systematic failure on fast motion: its points stay on the background where the object WAS
and are still reported visible, so the fit says "the object is (back) at its start" while its mask is 150 px lower.
complete_with_masks only fills anchors the fit left unknown, so these anchors survived as measured motion.  Per object
with mask-centroid readings (labels_real source 2 at the latent times, the reward scene's readings in mode "mask"):
  exact     a point-fitted anchor (FILL_POINTS / FILL_FEW) at the time of a mask reading m must be within
            tol = max(TOL_PX, TOL_FRAC * |m|) of it;
  bracket   one between two mask readings m0 (earlier, or the start) and m1 must have its centroid displacement
            inside their box widened by tol (of the larger reading);
            otherwise the anchor becomes the mask's reading (linear in time between m0 and m1; the last trusted
            rotation is kept) with provenance FILL_MASK.
  trailing  a point-fitted anchor with no mask reading after it becomes unknown (FILL_HOLD, the earlier transform is
            held) when nothing measured supports it: it puts the object within STILL_PX of its start although an
            earlier known anchor had it >= FAR_PX away ("stuck"), or the object's points were already caught wrong
            (an exact / bracket correction, or a mask fill once the object was >= FAR_PX away) and the anchor is more
            than tol from the last mask reading.
Run relabel_v2.py afterwards (its point readings come from the plan).

usage: reconcile_plans.py --cache DIR [--ids F] [--shard-index i --num-shards n] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import motion_plan as mp  # noqa: E402

TOL_PX, TOL_FRAC, STILL_PX, FAR_PX = 8.0, 0.15, 4.0, 24.0
TAG = "|reconciled_v1"


def mask_series(labels: dict, scene: dict | None, slot: int) -> tuple[np.ndarray, np.ndarray]:
    """(times [n], displacement [n, 2]) of the object's mask centroid, increasing in time, starting with (0, 0)."""
    samples = {0.0: np.zeros(2)}
    slots = [int(v) for v in np.asarray(labels["slots"]).tolist()]
    if slot in slots and "source" in labels:
        k = slots.index(slot)
        source, disp = np.asarray(labels["source"])[k], np.asarray(labels["disp"], np.float64)[k]
        for j, when in enumerate(labels["_times"]):
            if j and source[j] == 2:
                samples[round(float(when), 4)] = disp[j]
    if scene is not None and str(scene.get("mode", "points")) == "mask":
        scene_slots = [int(v) for v in scene["slots"]]
        if slot in scene_slots:
            k = scene_slots.index(slot)
            for j, when in enumerate(scene["times"]):
                if scene["real_ok"][k, j]:
                    samples.setdefault(round(float(when), 4), np.asarray(scene["real"][k, j], np.float64))
    times = np.array(sorted(samples))
    return times, np.stack([samples[t] for t in times])


def reconcile(plan: mp.MotionPlan, labels: dict, scene: dict | None) -> tuple[mp.MotionPlan, dict]:
    stats = {"anchors": 0, "exact": 0, "bracket": 0, "trailing": 0, "objects_changed": 0}
    objects = []
    for obj in plan.objects:
        times, series = mask_series(labels, scene, int(obj.slot))
        affines, known, fill = obj.affine.copy(), obj.known.copy(), obj.provenance().copy()
        centre = obj.centroid
        linear, changed, far, points_wrong = mp.IDENTITY[:, :2].copy(), False, False, False
        for j, when in enumerate(plan.anchors):
            if not known[j]:
                continue
            shift = mp.apply_affine(affines[j], centre[None])[0] - centre
            if fill[j] in (mp.FILL_POINTS, mp.FILL_FEW):
                stats["anchors"] += 1
                upper = int(np.searchsorted(times, when - 1e-4, side="left"))
                if 0 < upper < len(times):                                  # at, or between, mask readings
                    m0, m1 = series[upper - 1], series[upper]
                    tol = max(TOL_PX, TOL_FRAC * max(np.linalg.norm(m0), np.linalg.norm(m1)))
                    if abs(times[upper] - when) <= 1e-3:
                        wrong, kind, target = np.linalg.norm(shift - m1) > max(TOL_PX, TOL_FRAC * np.linalg.norm(m1)), "exact", m1
                    else:
                        weight = (when - times[upper - 1]) / max(times[upper] - times[upper - 1], 1e-9)
                        wrong = bool(((shift < np.minimum(m0, m1) - tol) | (shift > np.maximum(m0, m1) + tol)).any())
                        kind, target = "bracket", (1.0 - weight) * m0 + weight * m1
                    if wrong:
                        shift = target
                        affines[j] = np.concatenate((linear, (centre + shift - linear @ centre)[:, None]), 1)
                        fill[j] = mp.FILL_MASK
                        stats[kind] += 1
                        changed = points_wrong = True
                elif upper >= len(times) and len(times) > 1:                 # after the last mask reading
                    last = series[-1]
                    stuck = far and np.linalg.norm(shift) <= STILL_PX
                    adrift = points_wrong and np.linalg.norm(shift - last) > max(TOL_PX, TOL_FRAC * np.linalg.norm(last))
                    if stuck or adrift:
                        known[j], fill[j] = False, mp.FILL_HOLD
                        affines[j] = affines[j - 1] if j else mp.IDENTITY
                        stats["trailing"] += 1
                        changed = True
                        continue
                if fill[j] == mp.FILL_POINTS:
                    linear = affines[j][:, :2].copy()
            elif fill[j] == mp.FILL_MASK and far:
                points_wrong = True                                         # the points had already lost the moving object here
            far = far or bool(np.linalg.norm(shift) >= FAR_PX)
        stats["objects_changed"] += int(changed)
        objects.append(replace(obj, affine=affines, known=known, fill=fill))
    return replace(plan, objects=objects, source=plan.source if TAG in plan.source else plan.source + TAG), stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--ids", default="")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    cache = Path(a.cache)
    ids = [line.strip() for line in open(a.ids) if line.strip()] if a.ids else sorted(q.stem for q in (cache / "plans_real").glob("*.npz"))
    total = {"clips": 0, "clips_changed": 0, "anchors": 0, "exact": 0, "bracket": 0, "trailing": 0, "objects_changed": 0, "already": 0}
    for sid in ids[a.shard_index:: a.num_shards]:
        plan_path, payload_path = cache / "plans_real" / f"{sid}.npz", cache / f"{sid}.pt"
        if not plan_path.is_file() or not payload_path.is_file():
            continue
        plan = mp.load_plan(plan_path)
        if TAG in plan.source:
            total["already"] += 1
            continue
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        labels = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in payload["labels_real"].items()}
        labels["_times"] = [float(v) for v in payload["frame_seconds"]]
        scene_path = cache / "scenes" / f"{sid}.npz"
        scene = dict(np.load(scene_path, allow_pickle=True)) if scene_path.is_file() else None
        fixed, stats = reconcile(plan, labels, scene)
        total["clips"] += 1
        total["clips_changed"] += int(stats["objects_changed"] > 0)
        for key in ("anchors", "exact", "bracket", "trailing", "objects_changed"):
            total[key] += stats[key]
        if not a.dry_run:
            mp.save_plan(plan_path, fixed)
    print("RECONCILE", json.dumps(total))


if __name__ == "__main__":
    main()
