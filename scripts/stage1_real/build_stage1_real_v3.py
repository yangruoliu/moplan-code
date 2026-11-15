#!/usr/bin/env python
"""Real clips with a measured motion plan -> an admitted unified Stage-1 source (CPU stage).

One clip = one sample with the SAME contract as the synthetic sources of the unified Stage-1
(motion_jepa/flow_dit_data.FlowDiTDataset + valid_partition_v1 runtime supervision + SAM16 conditions):

  targets/<sid>.pt          flow [7,2,160,160] f16: cumulative displacement of every first-frame pixel at each horizon,
                            in pixels of the 160 grid (native frame letterboxed: scale 160/max(H,W), centred), x first,
                            y down.  visibility / flow_valid [7,1,160,160] bool, content_valid [1,160,160],
                            instance_id [1,160,160] int16 (SAM2 first-frame instance k -> k+1, measured extra clusters
                            -> 17, 18, ...), object_mask [1,160,160] (movers), future_mask [7,1,160,160] (movers'
                            positions), horizon_seconds, label_source, caption, video_path, scenario, ...
  sam2_conditions/<sid>.pt  the clip's own first-frame SAM2 masks (the ones Stage-1 is conditioned on at deployment),
                            re-stamped with first_frame_sha256 of the 160 first frame this stage computes with the same
                            code as the flow cache (read_first_frame 256 letterbox -> INTER_AREA 160).  Stored as compact
                            uint8 labels when that is lossless for the loader, otherwise as the original masks+channels.
  manifests/{train,val}.jsonl, metadata/supervision_mover_ids_manifest.json (stage1_supervision_mover_ids_v1),
  metadata/captions.json (every conditioning caption, for the prompt table), metadata/samples.jsonl, summary.json

Labels (work frame = the 448x256 frame the plan was measured in; grid <-> work by motion_plan.inner_box/grid_to_work):
  per object, over the plan anchors that are measured (known) AND inside the real clip (t <= (frames_real-1)/fps,
  padding after a clip at rest is not observation), with the Stage-2 reward thresholds (reward.MOVE_FRAC / STILL_FRAC
  = 3 % / 1 % of the work-frame diagonal = 15.5 / 5.2 px at 448x256):
    mover   largest box-corner motion >= --move-frac * diag (rotation trusted only where fitted from points)
    still   <= --still-frac * diag at every such anchor -> exactly zero where measured
    unknown otherwise, and every object at an anchor where it was not measured
  measured extras: SAM2 proposals (plan slots 100..199) are objects like any other; dense-grid motion clusters (slots
    >= 200, 16-px block masks that often fuse the hand with the object) are never supervised (unknown).
  --border-actor-unknown (Physics-101): a would-be mover whose first-frame mask touches the top / left / right frame
    border is the arm or hand that releases the object (it enters from outside the frame): unknown, not a mover; so is
    a would-be mover that touches such an actor and moves with it as one body (forearm / hand instances of that arm).
  --reconcile-labels/--reconcile-scenes (p101r): its plans predate provenance and the stuck-point repair; both passes
    of the cam1 plans are applied in memory (relabel_v2.provenance_from_residuals, reconcile_plans.reconcile).
  --exclude-scenarios: Physics-101 'spring' is left out (the person pulling the spring is split into several large
    SAM2 instances that move with it; they cannot be told apart from the object, so its movers are mostly arm parts).
  --annotated-only (SSv2): only objects with >= --contain of their first-frame mask inside a non-hand Something-Else
    box may be movers; any other object that is not measured still (hand, arm, sleeve, things dragged along) is unknown.
  background (no instance): zero and valid only in the 16-px cells around dense-grid tracks that stayed within
    --background-px of their start at that anchor (needs a static camera), minus a one-stride zone around every
    dense track that moved more than that (small unsegmented movers); otherwise unknown.
  a clip without a mover is kept only for a declared static template (--static-key/--static-values, e.g. SSv2 family
    'static', Physics-101 ramp / multi where the object may stay put) and when nothing unexplained moved inside the real
    clip (no dense track reached the mover threshold, no unknown object or cluster left the still band, no actor reached
    the mover threshold); --annotated-only lists only annotated still objects.  Otherwise it is skipped.
  horizon after the end of the clip or without a plan anchor: unknown everywhere (flow_valid = visibility = 0).
  non-mover pixels covered by a mover's position at a horizon: visibility 0 (occluded).
  movers (>= 4 grid px) are the mover ids; a clip in which nothing moved lists its still objects (the convention of the
  static event source: listed objects, zero motion).

usage: build_stage1_real_v3.py --source-name p101 --manifest M.jsonl --plans DIR --sam2 DIR --tracks DIR
         --meta META.jsonl --train-ids F --val-ids F --horizons query|phyco --out ROOT [--annotated-only] [--workers 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    import h5py  # noqa: F401
except ImportError:      # h5py is optional; read_first_frame needs it only for Physion HDF5 inputs
    import types
    sys.modules["h5py"] = types.ModuleType("h5py")
from scripts import motion_plan as mp  # noqa: E402
from scripts.cache_stage1_flowdit import read_first_frame  # noqa: E402
from motion_jepa.flow_dit_data import build_instance_inputs, caption_key, first_frame_tensor_sha256  # noqa: E402

SIDE = 160
SLOTS = 16
HORIZON_SETS = {
    # Stage-1's own query horizons (the Stage-1 export QUERY_SECONDS): exact plan anchors, 30 fps tracks
    "query": [0.20833333333333334, 0.4166666666666667, 0.625, 0.8333333333333334, 1.2, 2.0, 4.0],
    # PhyCo/pool horizons = 7-fps latent frame times (4k/7 s): exact plan anchors when the tracks are at 7 fps (SSv2)
    "phyco": [0.5714, 1.1429, 1.7143, 2.2857, 2.8571, 3.4286, 4.0],
}
LABEL_SOURCE = "real_measured_plan_v3"
MOVER_SCHEMA = "stage1_supervision_mover_ids_v1"
CONDITION_MASK_SOURCE = "first_frame_sam2_instance_masks"
ANCHOR_TOL = 1e-3
CLUSTER_BASE = 200        # real_scene.CLUSTER_BASE: plan slots >= 200 are dense-grid motion clusters (100..199: SAM2 proposals)


def first_frame_160(path: str) -> torch.Tensor:
    """uint8 [3,160,160]: exactly the flow-cache first frame (cache_stage1_event_inputs.py)."""
    frame = read_first_frame(path, 256)
    return torch.from_numpy(cv2.resize(frame, (SIDE, SIDE), interpolation=cv2.INTER_AREA)).permute(2, 0, 1).contiguous()


def load_condition(path: Path) -> tuple[dict, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "instance_labels" in payload:
        labels = payload["instance_labels"].long()
        masks = torch.stack([labels == k + 1 for k in range(SLOTS)])
    else:
        masks = payload["instance_masks"] > 0.5
    if tuple(masks.shape) != (SLOTS, SIDE, SIDE) or (masks.sum(0) > 1).any():
        raise ValueError(f"bad SAM2 condition masks {tuple(masks.shape)}")
    return payload, masks.numpy()


def condition_payload(sid: str, original: dict, masks: np.ndarray, first: torch.Tensor, source_path: Path) -> dict:
    """Compact uint8 labels when the loader re-derives the identical masks and channels, else the original tensors."""
    labels = torch.zeros(SIDE, SIDE, dtype=torch.long)
    for k in range(SLOTS):
        labels[torch.from_numpy(masks[k])] = k + 1
    rebuilt, _, channels = build_instance_inputs(labels[None], SLOTS)
    stored = torch.from_numpy(masks).float()
    same = torch.equal(rebuilt, stored) and ("instance_channels" not in original
                                             or torch.equal(channels, original["instance_channels"].float()))
    out = {"sample_id": sid, "num_instances": int(masks.reshape(SLOTS, -1).any(1).sum()),
           "source": str(original.get("source", "")), "condition_mask_source": CONDITION_MASK_SOURCE,
           "first_frame_sha256": first_frame_tensor_sha256(first),
           "provenance": {"original_path": str(source_path), "original_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                          "original_first_frame_sha256": str(original.get("first_frame_sha256", "")),
                          "restamped_for": "first frame of the real Stage-1 flow cache (read_first_frame 256 -> INTER_AREA 160)",
                          "filter": original.get("filter", {})}}
    if same:
        out.update(instance_labels=labels.to(torch.uint8), storage_format="uint8_instance_labels_v1")
    else:
        out.update(instance_masks=original["instance_masks"].half() if "instance_masks" in original else stored.half(),
                   instance_channels=original["instance_channels"].half() if "instance_channels" in original else channels.half(),
                   storage_format="uint8_masks_and_channels_v1")
    return out


def background_evidence(tracks_path: Path | None, anchors: np.ndarray, size: tuple[int, int], background_px: float,
                        max_camera_px: float, end: float, move_thr: float) -> tuple[np.ndarray, float, float]:
    """([T, H, W] bool in the work frame, camera motion px): the 16-px cell around a dense-grid track (owner -2) that is
    visible at 0 and at the anchor and stayed within background_px is static background evidence.  No tracks, a moving
    camera or no tracked frame within half a step of the anchor: no evidence."""
    height, width = size
    out = np.zeros((len(anchors), height, width), bool)
    if tracks_path is None or not tracks_path.is_file():
        return out, float("nan"), float("nan")
    track = np.load(tracks_path)
    tracks, visible, owner, times = (track["tracks"].astype(np.float64), track["visible"].astype(bool), track["owner"],
                                     track["times"].astype(np.float64))
    stride = int(track["dense_stride"]) if "dense_stride" in track.files and int(track["dense_stride"]) > 0 else 16
    dense = owner == -2
    if not dense.any() or len(times) < 2:
        return out, float("nan"), float("nan")
    start = tracks[0, dense]
    camera = []
    for f in range(len(times)):
        ok = visible[f, dense] & visible[0, dense]
        camera.append(float(np.median(np.linalg.norm(tracks[f, dense][ok] - start[ok], axis=1))) if ok.sum() >= 8 else np.nan)
    camera = np.array(camera)
    camera_max = float(np.nanmax(camera)) if np.isfinite(camera).any() else float("inf")
    inside = times <= end + 1e-6
    shift = np.linalg.norm(tracks[:, dense] - start[None], axis=-1)                        # [F, N]
    seen = visible[:, dense] & visible[0, dense][None] & inside[:, None]
    dense_peak = float(np.where(seen, shift, 0.0).max()) if seen.any() else 0.0           # largest dense-grid motion in the clip
    if camera_max > max_camera_px:
        return out, camera_max, dense_peak
    step = float(np.median(np.diff(times)))
    half = stride // 2
    for j, when in enumerate(anchors):
        frame = int(np.abs(times - when).argmin())
        if abs(times[frame] - when) > 0.5 * step + 1e-6:
            continue
        both = visible[frame, dense] & visible[0, dense]
        distance = np.linalg.norm(tracks[frame, dense] - start, axis=1)
        for x, y in start[both & (distance <= background_px)]:
            x0, y0 = int(round(x)), int(round(y))
            out[j, max(0, y0 - half): y0 + half + 1, max(0, x0 - half): x0 + half + 1] = True
        for x, y in start[~both | (distance > background_px)]:                           # a moved or lost grid point vetoes its neighbourhood
            x0, y0 = int(round(x)), int(round(y))
            out[j, max(0, y0 - stride): y0 + stride + 1, max(0, x0 - stride): x0 + stride + 1] = False
    return out, camera_max, dense_peak


def touches_border(mask: np.ndarray, sides: str = "TLR", margin: int = 2) -> bool:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return False
    height, width = mask.shape
    return (("T" in sides and ys.min() <= margin) or ("B" in sides and ys.max() >= height - 1 - margin)
            or ("L" in sides and xs.min() <= margin) or ("R" in sides and xs.max() >= width - 1 - margin))


def same_body(a: np.ndarray, b: np.ndarray, usable: np.ndarray, still_thr: float) -> bool:
    """Two centroid trajectories [T, 2] move as one articulated body: at every usable anchor where either is out of
    the still band, they agree within 8 px or point the same way (< 45 deg) with a magnitude ratio in [0.4, 2.5]."""
    moving = usable & ((np.linalg.norm(a, axis=1) > still_thr) | (np.linalg.norm(b, axis=1) > still_thr))
    if not moving.any():
        return False
    for u, v in zip(a[moving], b[moving]):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if np.linalg.norm(u - v) <= 8.0:
            continue
        if min(nu, nv) < 1e-6 or not (0.4 <= nu / nv <= 2.5) or float(u @ v) / (nu * nv) < math.cos(math.radians(45)):
            return False
    return True


def annotated(mask: np.ndarray, boxes: dict, contain: float) -> bool:
    """>= contain of the work-frame mask inside one non-hand box (Something-Else, canvas = work frame)."""
    area = int(mask.sum())
    if not area:
        return False
    height, width = mask.shape
    for name, (x1, y1, x2, y2) in boxes.items():
        if name == "hand":
            continue
        r0, r1 = int(np.clip(np.floor(y1), 0, height)), int(np.clip(np.ceil(y2), 0, height))
        c0, c1 = int(np.clip(np.floor(x1), 0, width)), int(np.clip(np.ceil(x2), 0, width))
        if mask[r0:r1, c0:c1].sum() >= contain * area:
            return True
    return False


def repair_plan(plan: mp.MotionPlan, sid: str, labels_dir: Path, scenes_dir: Path) -> mp.MotionPlan:
    """In memory, the two passes the cam1 plans went through ('|fill_reconstructed|reconciled_v1'):
    relabel_v2.provenance_from_residuals, then reconcile_plans.reconcile against the clip's mask-centroid readings
    (labels_real of the Stage-2 cache entry, dumped by p101r_labels.py, and the reward scene).  Nothing is written."""
    from scripts.control.reconcile_plans import TAG, reconcile
    from scripts.control.relabel_v2 import provenance_from_residuals
    plan = provenance_from_residuals(plan)
    label_path, scene_path = labels_dir / f"{sid}.npz", scenes_dir / f"{sid}.npz"
    if TAG in plan.source or not label_path.is_file():
        return plan
    data = np.load(label_path, allow_pickle=False)
    labels = {k: data[k] for k in data.files if k != "_times"}
    labels["_times"] = [float(v) for v in data["_times"]]
    scene = dict(np.load(scene_path, allow_pickle=True)) if scene_path.is_file() else None
    return reconcile(plan, labels, scene)[0]


def build_sample(job: dict) -> dict:
    sid, row, meta, a = job["sid"], job["row"], job["meta"], job["args"]
    horizons = HORIZON_SETS[a["horizons"]]
    try:
        plan_path, cond_path = Path(a["plans"]) / f"{sid}.npz", Path(a["sam2"]) / f"{sid}.pt"
        frame_path = row["first_frame_path"]
        for path in (plan_path, cond_path, Path(frame_path)):
            if not path.is_file():
                return {"sample_id": sid, "skip": f"missing {path.parent.name}"}
        native_hw = cv2.imread(frame_path, cv2.IMREAD_COLOR).shape[:2]
        plan = mp.load_plan(plan_path)
        height, width = plan.size
        if a["reconcile_labels"]:
            plan = repair_plan(plan, sid, Path(a["reconcile_labels"]), Path(a["reconcile_scenes"]))
        if any(np.all(obj.provenance() == mp.FILL_UNKNOWN) for obj in plan.objects):
            return {"sample_id": sid, "skip": "plan without anchor provenance"}
        fps = float(meta.get("fps", 7.0))
        real_end = (int(meta["frames_real"]) - 1) / fps
        original, masks = load_condition(cond_path)
        first = first_frame_160(frame_path)

        # grid <-> work geometry
        top, left, grid_h, grid_w = mp.inner_box(SIDE, SIDE, *native_hw)
        gy, gx = np.mgrid[0:SIDE, 0:SIDE]
        xy = mp.grid_to_work(np.stack([gx, gy], -1).astype(np.float64), (top, left, grid_h, grid_w), (height, width))
        content = (gy >= top) & (gy < top + grid_h) & (gx >= left) & (gx < left + grid_w)
        rows = np.clip(np.rint(xy[..., 1]).astype(int), 0, height - 1)
        cols = np.clip(np.rint(xy[..., 0]).astype(int), 0, width - 1)
        sx, sy = grid_w / width, grid_h / height

        # instance map: SAM2 instance k -> k + 1, measured extras (slots >= 16) on uncovered content -> 17, 18, ...
        labels = np.zeros((SIDE, SIDE), np.int16)
        for k in range(SLOTS):
            labels[masks[k] & content & (labels == 0)] = k + 1
        object_ids, mismatched, next_id = {}, 0, SLOTS + 1
        for index, obj in enumerate(plan.objects):
            grid_mask = obj.mask[rows, cols] & content
            if obj.slot < SLOTS:
                sam = masks[obj.slot]
                union = int((grid_mask | sam).sum())
                if not sam.any() or union == 0 or (grid_mask & sam).sum() / union < 0.5:
                    mismatched += 1
                    continue
                object_ids[index] = obj.slot + 1
        for index, obj in enumerate(plan.objects):
            if obj.slot < SLOTS:
                continue
            grid_mask = obj.mask[rows, cols] & content & (labels == 0)
            if grid_mask.sum() >= 4:
                labels[grid_mask] = next_id
                object_ids[index] = next_id
                next_id += 1

        # per-object class over measured anchors inside the real clip
        inside = plan.anchors <= real_end + ANCHOR_TOL
        diagonal = float(np.hypot(height, width))
        move_thr, still_thr = a["move_frac"] * diagonal, a["still_frac"] * diagonal
        boxes = row.get("boxes_t0_canvas") if a["annotated_only"] else None
        kinds, peak, named = {}, {}, {}
        for index, obj in enumerate(plan.objects):
            if index not in object_ids:
                continue
            named[index] = boxes is None or annotated(obj.mask, boxes, a["contain"])
            usable = obj.known & inside
            if not usable.any():
                kinds[index], peak[index] = "unknown", 0.0
                continue
            motion = float(obj.point_motion()[usable].max())
            kind = "mover" if motion >= move_thr else ("still" if motion <= still_thr else "unknown")
            if obj.slot >= CLUSTER_BASE:
                kind = "cluster"          # a dense-grid cluster: 16-px block mask, often hand + object fitted as one body
            if kind == "mover" and a["border_actor_unknown"] and touches_border(obj.mask):
                kind = "border_actor"
            if boxes is not None and kind != "still" and not named[index]:
                kind = "unannotated"
            kinds[index], peak[index] = kind, motion
        if a["border_actor_unknown"]:
            # the arm that enters from a border is often split into several SAM2 instances (upper arm, forearm, hand):
            # a would-be mover touching an actor (5 px) and moving with it as one body is an actor part as well; the
            # released object diverges from the hand (falls / slides), so it never satisfies same_body at every anchor
            actors = {i for i, k in kinds.items() if touches_border(plan.objects[i].mask) and peak[i] > still_thr
                      and k in ("border_actor", "unknown", "cluster")}
            grown = {i: cv2.dilate(plan.objects[i].mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0 for i in actors}
            changed = True
            while changed:
                changed = False
                for i, k in list(kinds.items()):
                    if k != "mover" or i in actors:
                        continue
                    obj = plan.objects[i]
                    for j in list(actors):
                        other = plan.objects[j]
                        if (grown[j] & obj.mask).any() and same_body(obj.displacement(), other.displacement(),
                                                                     obj.known & other.known & inside, still_thr):
                            kinds[i] = "actor_part"
                            actors.add(i)
                            grown[i] = cv2.dilate(obj.mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
                            changed = True
                            break

        background, camera_max, dense_peak = background_evidence(
            Path(a["tracks"]) / f"{sid}.npz" if a["tracks"] else None, plan.anchors, (height, width),
            a["background_px"], a["max_camera_px"], real_end, move_thr)
        count = len(horizons)
        flow = np.zeros((count, 2, SIDE, SIDE), np.float32)
        valid = np.zeros((count, SIDE, SIDE), bool)
        visible = np.zeros((count, SIDE, SIDE), bool)
        observed = np.zeros(count, bool)
        for j, when in enumerate(horizons):
            near = np.flatnonzero(np.abs(plan.anchors - when) < ANCHOR_TOL)
            if when > real_end + ANCHOR_TOL or len(near) == 0:
                continue                                                     # not observed: unknown everywhere
            anchor = int(near[0])
            evidence = background[anchor][rows, cols] & content & (labels == 0)
            valid[j] |= evidence
            visible[j] |= evidence
            for index, iid in object_ids.items():
                kind = kinds.get(index)
                obj = plan.objects[index]
                if kind not in ("mover", "still") or not bool(obj.known[anchor]):
                    continue                                                 # unmeasured here: unknown, whatever the class
                chosen = labels == iid
                if not chosen.any():
                    continue
                valid[j][chosen] = True
                visible[j][chosen] = True
                if kind == "still":
                    continue                                                 # measured still: exactly zero
                points = xy[chosen]
                moved = mp.apply_affine(obj.affine[anchor], points)
                flow[j, 0][chosen] = (moved[:, 0] - points[:, 0]) * sx
                flow[j, 1][chosen] = (moved[:, 1] - points[:, 1]) * sy
                visible[j][chosen] = (moved[:, 0] >= -0.5) & (moved[:, 0] <= width - 0.5) & (moved[:, 1] >= -0.5) & (moved[:, 1] <= height - 0.5)
            observed[j] = valid[j].any()

        movers = sorted(int(object_ids[i]) for i, k in kinds.items() if k == "mover" and int((labels == object_ids[i]).sum()) >= 4)
        stills = sorted(int(object_ids[i]) for i, k in kinds.items()
                        if k == "still" and named[i] and int((labels == object_ids[i]).sum()) >= 4)
        if not observed.any():
            return {"sample_id": sid, "skip": "nothing observed at any horizon"}
        if not movers and not stills:
            return {"sample_id": sid, "skip": "no measured object"}
        if not movers:
            # nothing measured as a mover: admitted only for a declared static template (e.g. SSv2 family 'static'),
            # and only if no unexplained motion happened inside it -- a Physics-101 clip always shows a release, so a
            # clip without a measured mover means the mover was not segmented, not that nothing moved
            if not a["static_values"] or str(row.get(a["static_key"], "")) not in a["static_values"]:
                return {"sample_id": sid, "skip": "no measured mover (not a static template)"}
            unexplained = [i for i, k in kinds.items()
                           if (k in ("unknown", "cluster") and peak[i] > still_thr) or (k in ("border_actor", "actor_part") and peak[i] >= move_thr)]
            dense_moved = np.isfinite(dense_peak) and dense_peak >= move_thr
            if unexplained or (dense_moved and not a["annotated_only"]):
                return {"sample_id": sid, "skip": "no mover but unexplained motion (unsegmented / actor-only motion)"}
        moving = np.isin(labels, movers) if movers else np.zeros((SIDE, SIDE), bool)
        future = np.zeros((count, 1, SIDE, SIDE), bool)
        for j in range(count):
            src = np.argwhere(moving & valid[j] & visible[j])
            if len(src):
                dx = np.clip(np.rint(src[:, 1] + flow[j, 0][src[:, 0], src[:, 1]]), 0, SIDE - 1).astype(int)
                dy = np.clip(np.rint(src[:, 0] + flow[j, 1][src[:, 0], src[:, 1]]), 0, SIDE - 1).astype(int)
                future[j, 0][dy, dx] = True
                future[j, 0] = cv2.morphologyEx(future[j, 0].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
                visible[j] &= ~(future[j, 0] & ~moving)                     # first-frame pixels a mover now covers
        magnitude = np.sqrt((flow ** 2).sum(1))
        payload = {
            "cache_format_version": 1, "kind": "targets", "sample_id": sid, "scenario": job["scenario"],
            "flow": torch.from_numpy(flow).half(), "visibility": torch.from_numpy(visible)[:, None],
            "flow_valid": torch.from_numpy(valid)[:, None], "content_valid": torch.from_numpy(content)[None],
            "object_mask": torch.from_numpy(moving)[None], "future_mask": torch.from_numpy(future),
            "instance_id": torch.from_numpy(labels)[None].short(), "horizon_seconds": list(horizons),
            "source_fps": fps, "flow_source": "measured_plan_rigid_per_anchor_v3",
            "instance_source": "first_frame_sam2_k16_plus_measured_clusters", "label_source": LABEL_SOURCE,
            "caption": row["caption"], "caption_source": a["caption_source"], "video_path": frame_path,
            "real_end_seconds": float(real_end), "observed_horizons": int(observed.sum()),
            "object_slots": [int(plan.objects[i].slot) for i in sorted(object_ids)],
            "object_instance_ids": [int(object_ids[i]) for i in sorted(object_ids)],
            "object_kinds": [kinds.get(i, "unknown") for i in sorted(object_ids)],
            "object_peak_motion_work_px": [float(peak.get(i, 0.0)) for i in sorted(object_ids)],
            "mover_instance_ids": movers, "plan_source": plan.source, "camera_max_px": camera_max,
            "dense_peak_work_px": dense_peak, "move_threshold_work_px": move_thr, "still_threshold_work_px": still_thr,
        }
        out = Path(a["out"])
        torch.save(payload, out / "targets" / f"{sid}.pt")
        torch.save(condition_payload(sid, original, masks, first, cond_path), out / "sam2_conditions" / f"{sid}.pt")
        moving_valid = (valid & moving[None] & (magnitude > 1.0)).sum()
        return {"sample_id": sid, "movers": movers, "stills": stills, "observed": int(observed.sum()),
                "real_end": float(real_end), "kinds": dict(Counter(kinds.values())), "mismatched": mismatched,
                "max_px": float(magnitude.max()), "valid_frac": float(valid[observed].mean()),
                "moving_valid_px": int(moving_valid), "mover_px": int(moving.sum()), "camera_max_px": camera_max,
                "background_frac": float((valid[observed] & (labels == 0)[None]).mean()),
                "first_frame_sha_matches_original": first_frame_tensor_sha256(first) == str(original.get("first_frame_sha256", "")),
                "extras": int(sum(1 for i in object_ids if plan.objects[i].slot >= SLOTS)), "dense_peak": dense_peak}
    except Exception as error:                                                 # noqa: BLE001
        return {"sample_id": sid, "skip": "error: " + repr(error)[:300]}


def read_ids(path: str) -> list[str]:
    return [line.strip() for line in open(path) if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-name", required=True)
    p.add_argument("--manifest", required=True, help="jsonl rows: sample_id, caption, first_frame_path, scenario, ...")
    p.add_argument("--plans", required=True)
    p.add_argument("--sam2", required=True)
    p.add_argument("--tracks", default="")
    p.add_argument("--meta", required=True, help="jsonl: sample_id, frames_real, fps (stage2_cache_meta.py)")
    p.add_argument("--train-ids", required=True)
    p.add_argument("--val-ids", required=True)
    p.add_argument("--horizons", choices=sorted(HORIZON_SETS), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scenario-prefix", default="", help="scenario = prefix + row[--scenario-key]")
    p.add_argument("--scenario-key", default="scenario")
    p.add_argument("--scene-prefix", required=True, help="scene_id = prefix + row[--scene-key] (coverage identity)")
    p.add_argument("--scene-key", default="trial_id")
    p.add_argument("--source-tag", required=True, help="manifest 'source' field")
    p.add_argument("--caption-source", required=True)
    p.add_argument("--annotated-only", action="store_true")
    p.add_argument("--contain", type=float, default=0.5)
    p.add_argument("--move-frac", type=float, default=0.03, help="mover threshold, fraction of the work diagonal (reward.MOVE_FRAC)")
    p.add_argument("--still-frac", type=float, default=0.01, help="still threshold, fraction of the work diagonal (reward.STILL_FRAC)")
    p.add_argument("--border-actor-unknown", action="store_true")
    p.add_argument("--static-key", default="family", help="manifest key of the template family")
    p.add_argument("--static-values", default="", help="comma list of families whose clips may have no mover (default: none)")
    p.add_argument("--background-px", type=float, default=2.0)
    p.add_argument("--max-camera-px", type=float, default=2.0)
    p.add_argument("--reconcile-labels", default="", help="dir of <sid>.npz mask-centroid readings: reconstruct provenance + reconcile plans in memory")
    p.add_argument("--reconcile-scenes", default="", help="dir of the reward scenes (scenes/<sid>.npz) used by the reconcile pass")
    p.add_argument("--exclude-scenarios", default="", help="comma list of row[--scenario-key] values left out entirely")
    p.add_argument("--forbid-ids", default="", help="file of ids that must not appear in any split (e.g. benchmark val)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    out = Path(a.out)
    for name in ("targets", "sam2_conditions", "manifests", "metadata"):
        (out / name).mkdir(parents=True, exist_ok=True)
    rows = {json.loads(l)["sample_id"]: json.loads(l) for l in open(a.manifest) if l.strip()}
    metas = {json.loads(l)["sample_id"]: json.loads(l) for l in open(a.meta) if l.strip()}
    splits = {"train": read_ids(a.train_ids), "val": read_ids(a.val_ids)}
    if a.limit:
        splits = {k: v[: a.limit] for k, v in splits.items()}
    overlap = set(splits["train"]) & set(splits["val"])
    if overlap:
        raise SystemExit(f"train/val overlap: {sorted(overlap)[:3]}")
    scenes = {k: {a.scene_prefix + str(rows[s][a.scene_key]) for s in v if s in rows} for k, v in splits.items()}
    if scenes["train"] & scenes["val"]:
        raise SystemExit(f"train/val share scenes: {sorted(scenes['train'] & scenes['val'])[:3]}")
    if a.forbid_ids:
        forbidden = set(read_ids(a.forbid_ids))
        bad = (set(splits["train"]) | set(splits["val"])) & forbidden
        if bad:
            raise SystemExit(f"forbidden ids present: {sorted(bad)[:3]}")
    args = {k: getattr(a, k) for k in ("plans", "sam2", "tracks", "horizons", "out", "annotated_only", "contain", "move_frac",
                                        "still_frac", "background_px", "max_camera_px", "caption_source", "border_actor_unknown",
                                        "static_key", "reconcile_labels", "reconcile_scenes")}
    args["static_values"] = [v for v in a.static_values.split(",") if v]
    excluded = {v for v in a.exclude_scenarios.split(",") if v}
    if excluded:
        dropped = {k: [s for s in v if s in rows and str(rows[s][a.scenario_key]) in excluded] for k, v in splits.items()}
        splits = {k: [s for s in v if s not in set(dropped[k])] for k, v in splits.items()}
        print("excluded scenarios", sorted(excluded), {k: len(v) for k, v in dropped.items()})
    jobs = []
    for split, ids in splits.items():
        for sid in ids:
            if sid in rows and sid in metas:
                row = rows[sid]
                jobs.append({"sid": sid, "row": row, "meta": metas[sid], "args": args,
                             "scenario": a.scenario_prefix + str(row[a.scenario_key])})
    with ProcessPoolExecutor(a.workers) as pool:
        results = {r["sample_id"]: r for r in pool.map(build_sample, jobs, chunksize=4)}
    mapping, captions, summary = {}, {caption_key(""): ""}, {"skipped": {}, "missing_row_or_meta": 0}
    with open(out / "metadata" / "samples.jsonl", "w") as log:
        for r in results.values():
            log.write(json.dumps(r) + "\n")
    for split, ids in splits.items():
        kept = [sid for sid in ids if sid in results and "skip" not in results[sid]]
        summary["missing_row_or_meta"] += sum(1 for sid in ids if sid not in results)
        with open(out / "manifests" / f"{split}.jsonl", "w") as handle:
            for sid in sorted(kept):
                row = rows[sid]
                scenario = a.scenario_prefix + str(row[a.scenario_key])
                record = {"sample_id": sid, "scenario": scenario, "split": split, "video_path": row["first_frame_path"],
                          "caption": row["caption"], "source": a.source_tag,
                          "scene_id": a.scene_prefix + str(row[a.scene_key]), "trial_id": str(row.get("trial_id", sid)),
                          "camera": str(row.get("camera", "")), "event": "real_motion" if results[sid]["movers"] else "real_static",
                          "original_video_path": row.get("video_path", "")}
                handle.write(json.dumps(record) + "\n")
                captions[caption_key(row["caption"])] = row["caption"]
                mapping[sid] = {"sample_id": sid, "scenario": scenario,
                                "mover_instance_ids": results[sid]["movers"] or results[sid]["stills"],
                                "event": record["event"], "scene_id": record["scene_id"]}
        done = [results[sid] for sid in kept]
        movers = [r for r in done if r["movers"]]
        summary[split] = {
            "requested": len(ids), "clips": len(kept), "with_movers": len(movers), "movers": sum(len(r["movers"]) for r in done),
            "unique_scenes": len({mapping[s]["scene_id"] for s in kept}),
            "observed_horizons_mean": float(np.mean([r["observed"] for r in done])) if done else 0.0,
            "observed_horizons_hist": dict(sorted(Counter(r["observed"] for r in done).items())),
            "real_end_mean_s": float(np.mean([r["real_end"] for r in done])) if done else 0.0,
            "object_kinds": dict(sum((Counter(r["kinds"]) for r in done), Counter())),
            "valid_frac_observed_mean": float(np.mean([r["valid_frac"] for r in done])) if done else 0.0,
            "background_valid_frac_mean": float(np.mean([r["background_frac"] for r in done])) if done else 0.0,
            "moving_valid_px_per_mover_clip_mean": float(np.mean([r["moving_valid_px"] for r in movers])) if movers else 0.0,
            "moving_valid_frac_of_mover_px_horizons": float(np.mean([r["moving_valid_px"] / max(r["mover_px"] * r["observed"], 1) for r in movers])) if movers else 0.0,
            "max_px_median_movers": float(np.median([r["max_px"] for r in movers])) if movers else 0.0,
            "plan_sam_mismatched_objects": sum(r["mismatched"] for r in done),
            "extras_used": sum(r["extras"] for r in done),
            "first_frame_sha_matches_original": sum(bool(r["first_frame_sha_matches_original"]) for r in done),
        }
        for sid in ids:
            if sid in results and "skip" in results[sid]:
                reason = results[sid]["skip"]
                summary["skipped"][reason] = summary["skipped"].get(reason, 0) + 1
    if not a.limit:                                                      # drop files of clips no longer admitted (earlier runs)
        stale = 0
        for name in ("targets", "sam2_conditions", "flowdit"):
            for path in (out / name).glob("*.pt") if (out / name).is_dir() else []:
                if path.stem not in mapping:
                    path.unlink()
                    stale += 1
        summary["stale_files_removed"] = stale
    (out / "metadata" / "supervision_mover_ids_manifest.json").write_text(json.dumps(
        {"schema": MOVER_SCHEMA, "rows": len(mapping), "label_source": LABEL_SOURCE,
         "policy": "measured movers (>= 3% work diagonal inside the real clip); a static clip lists its measured-still (annotated) objects",
         "mapping": mapping}) + "\n")
    (out / "metadata" / "captions.json").write_text(json.dumps({"schema": "stage1_real_captions_v1", "captions": captions}, indent=0) + "\n")
    summary.update(source_name=a.source_name, horizon_seconds=HORIZON_SETS[a.horizons], horizon_set=a.horizons,
                   label_source=LABEL_SOURCE, move_frac=a.move_frac, still_frac=a.still_frac, background_px=a.background_px,
                   border_actor_unknown=a.border_actor_unknown, excluded_scenarios=sorted(excluded),
                   max_camera_px=a.max_camera_px, annotated_only=a.annotated_only, contain=a.contain,
                   inputs={k: getattr(a, k) for k in ("manifest", "plans", "sam2", "tracks", "meta", "train_ids", "val_ids")},
                   captions=len(captions))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("REAL_SOURCE_V3", json.dumps(summary))


if __name__ == "__main__":
    main()
