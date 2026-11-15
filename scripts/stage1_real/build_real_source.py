#!/usr/bin/env python
"""Real clips with measured motion -> a Stage-1 training source (the contract of motion_jepa/flow_dit_data.py).

Per clip the Stage-2 v2 cache holds the measured plan (plans_real/<sid>.npz: every object of the measured scene --
Stage-1's SAM2 instances, the extra proposals and the dense-grid clusters that moved -- with one rigid transform per
anchor, fitted to the real tracks / mask centroids).  Its anchors are Stage-1's own query horizons that fall inside the
clip, so the target is written at exactly the horizons the model is queried at (no interpolation in time).

  targets/<sid>.pt   flow [7, 2, 160, 160] cumulative displacement of the first-frame pixel, in pixels of the 160 grid
                     (the native frame letterboxed, as every Stage-1 source), flow_valid / visibility, instance_id
                     (object k -> k + 1), object_mask (the movers), future_mask, horizon_seconds, caption, ...
                       mover     point motion (corners of the object's box under its transform; rotation counts only
                                 at anchors fitted from points, FILL_POINTS) >= --move-px at some KNOWN anchor
                       still     <= --still-px at every known anchor (and at least one known anchor): exactly zero
                                 where known (tracker jitter is not motion)
                       otherwise unknown (flow_valid 0); every object is unknown at an anchor it was not measured at,
                       whatever its class; horizon after the end of the clip: unknown everywhere
                     Background (pixels on no object) is zero and valid only where the clip's dense tracking grid
                     (real_tracks, --tracks) saw a point stay within --background-px of its start at that anchor; cells
                     without such evidence, and every background pixel of a clip whose camera moved more than
                     --max-camera-px, are unknown.
  manifests/{train,val}.jsonl, task_object_ids.json (schema stage1_supervision_mover_ids_v1: the movers; for a clip in
  which nothing moved, every still object -- the convention of Stage-1's static source: listed objects, zero motion),
  event_pairs_{train,val}.json (empty: no text pairs), summary.json

usage: build_real_source.py --cache DIR --manifest M.jsonl --train-ids F --val-ids F --out ROOT [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import motion_plan as mp  # noqa: E402

SIDE = 160
HORIZONS = [0.20833333333333334, 0.4166666666666667, 0.625, 0.8333333333333334, 1.2, 2.0, 4.0]     # Stage-1's query horizons
LABEL_SOURCE = "real_measured_plan_v2"
MOVER_SCHEMA = "stage1_supervision_mover_ids_v1"


def classify(plan: mp.MotionPlan, move_px: float, still_px: float) -> list[str]:
    """"mover" | "still" | "unknown" per object, from the POINT motion at the anchors it was measured at (corners of
    the object's box under its transform; a rotation in place is motion; the rotation is trusted only at anchors fitted
    from points, elsewhere the centroid translation counts)."""
    out = []
    for obj in plan.objects:
        if not obj.known.any() or not obj.mask.any():
            out.append("unknown")
            continue
        motion = obj.point_motion()[obj.known]
        out.append("mover" if motion.max() >= move_px else ("still" if motion.max() <= still_px else "unknown"))
    return out


def background_evidence(tracks_path, anchors: np.ndarray, height: int, width: int, background_px: float, max_camera_px: float) -> np.ndarray:
    """[T, H, W] bool in the work frame: where the dense tracking grid of the clip (real_tracks/<sid>.npz, owner -2)
    saw a visible point within `background_px` of its start at the anchor, its 16 x 16 px cell is static background
    evidence.  No file, a moving camera, or an anchor after the clip: no evidence anywhere."""
    out = np.zeros((len(anchors), height, width), bool)
    if tracks_path is None or not Path(tracks_path).is_file():
        return out
    track = np.load(tracks_path)
    tracks, visible, owner, times = track["tracks"].astype(np.float64), track["visible"].astype(bool), track["owner"], track["times"].astype(np.float64)
    stride = int(track["dense_stride"]) if "dense_stride" in track.files and int(track["dense_stride"]) > 0 else 16
    dense = owner == -2
    if not dense.any():
        return out
    start = tracks[0, dense]
    camera = np.array([np.median(np.linalg.norm(tracks[f, dense] - start, axis=1)[visible[f, dense]]) if visible[f, dense].sum() >= 8 else np.inf for f in range(len(times))])
    if np.nanmax(camera[np.isfinite(camera)]) > max_camera_px if np.isfinite(camera).any() else True:
        return out
    half = stride // 2
    for j, when in enumerate(anchors):
        frame = int(np.abs(times - when).argmin())
        if abs(times[frame] - when) > 0.5 * float(np.median(np.diff(times))) + 1e-6:
            continue
        ok = visible[frame, dense] & visible[0, dense] & (np.linalg.norm(tracks[frame, dense] - start, axis=1) <= background_px)
        for x, y in start[ok]:
            x0, y0 = int(round(x)), int(round(y))
            out[j, max(0, y0 - half): y0 + half + 1, max(0, x0 - half): x0 + half + 1] = True
    return out


def build_target(plan: mp.MotionPlan, native_hw: tuple[int, int], *, move_px: float = 8.0, still_px: float = 4.0,
                 clip_seconds: float | None = None, background: np.ndarray | None = None) -> tuple[dict, list[int], list[int]]:
    """Tensors of one target, the mover instance ids and the still ones.  clip_seconds: last observed time (default: the
    plan's last anchor).  background [len(plan.anchors), H, W] bool (work frame): where the background is known to be
    static (background_evidence); None = no background pixel is supervised."""
    height, width = plan.size
    top, left, grid_h, grid_w = mp.inner_box(SIDE, SIDE, native_hw[0], native_hw[1])
    gy, gx = np.mgrid[0:SIDE, 0:SIDE]
    xy = mp.grid_to_work(np.stack([gx, gy], -1).astype(np.float64), (top, left, grid_h, grid_w), (height, width))       # [S, S, 2] work pixels
    content = (gy >= top) & (gy < top + grid_h) & (gx >= left) & (gx < left + grid_w)
    work_labels = np.zeros((height, width), np.int16)
    for k, obj in enumerate(plan.objects):                                # earlier objects (Stage-1's instances) keep their pixels
        work_labels[obj.mask & (work_labels == 0)] = k + 1
    rows, cols = np.clip(np.rint(xy[..., 1]).astype(int), 0, height - 1), np.clip(np.rint(xy[..., 0]).astype(int), 0, width - 1)
    labels = np.where(content, work_labels[rows, cols], 0).astype(np.int16)
    kinds = classify(plan, move_px, still_px)
    end = float(plan.anchors[-1]) if clip_seconds is None else float(clip_seconds)
    sx, sy = grid_w / width, grid_h / height
    flow = np.zeros((len(HORIZONS), 2, SIDE, SIDE), np.float32)
    valid = np.zeros((len(HORIZONS), SIDE, SIDE), bool)
    visible = np.zeros((len(HORIZONS), SIDE, SIDE), bool)
    for j, when in enumerate(HORIZONS):
        near = np.flatnonzero(np.abs(plan.anchors - when) < 1e-3)
        if when > end + 1e-3 or len(near) == 0:
            continue                                                       # not observed: unknown everywhere
        anchor = int(near[0])
        if background is not None:                                         # static background only where the tracking grid proved it
            evidence = np.where(content, background[anchor][rows, cols], False)
            valid[j], visible[j] = (labels == 0) & evidence, (labels == 0) & evidence
        for k, obj in enumerate(plan.objects):
            chosen = labels == k + 1
            if not chosen.any():
                continue
            if kinds[k] == "unknown" or not bool(obj.known[anchor]):       # provenance first: an unmeasured anchor is unknown, whatever the class
                continue
            valid[j][chosen], visible[j][chosen] = True, True
            if kinds[k] == "still":
                continue                                                   # measured still here: exactly zero
            points = xy[chosen]
            moved = mp.apply_affine(obj.affine[anchor], points)
            flow[j, 0][chosen], flow[j, 1][chosen] = (moved[:, 0] - points[:, 0]) * sx, (moved[:, 1] - points[:, 1]) * sy
            visible[j][chosen] = (moved[:, 0] >= 0) & (moved[:, 0] <= width - 1) & (moved[:, 1] >= 0) & (moved[:, 1] <= height - 1)
    movers = [k + 1 for k, kind in enumerate(kinds) if kind == "mover" and int((labels == k + 1).sum()) >= 4]
    stills = [k + 1 for k, kind in enumerate(kinds) if kind == "still" and int((labels == k + 1).sum()) >= 4]
    moving = np.isin(labels, movers) if movers else np.zeros((SIDE, SIDE), bool)
    flow_t = torch.from_numpy(flow)
    future = np.zeros((len(HORIZONS), 1, SIDE, SIDE), bool)
    for j in range(len(HORIZONS)):                                        # where the movers are at each horizon (nearest cell)
        src = np.argwhere(moving & valid[j])
        if len(src):
            dest_x = np.clip(np.rint(src[:, 1] + flow[j, 0][src[:, 0], src[:, 1]]), 0, SIDE - 1).astype(int)
            dest_y = np.clip(np.rint(src[:, 0] + flow[j, 1][src[:, 0], src[:, 1]]), 0, SIDE - 1).astype(int)
            future[j, 0][dest_y, dest_x] = True
            future[j, 0] = cv2.morphologyEx(future[j, 0].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    payload = {"cache_format_version": 1, "kind": "targets", "flow": flow_t.half(), "visibility": torch.from_numpy(visible)[:, None],
               "flow_valid": torch.from_numpy(valid)[:, None], "content_valid": torch.from_numpy(content)[None],
               "object_mask": torch.from_numpy(moving)[None], "future_mask": torch.from_numpy(future),
               "instance_id": torch.from_numpy(labels)[None].short(), "horizon_seconds": list(HORIZONS), "source_fps": 7.0,
               "flow_source": "measured_plan_rigid_v2", "instance_source": "sam2_k16_plus_measured_scene", "label_source": LABEL_SOURCE,
               "object_kinds": kinds, "object_fill": [obj.provenance().tolist() for obj in plan.objects],
               "observed_horizons": int(valid.reshape(len(HORIZONS), -1).any(1).sum())}
    return payload, movers, stills


def work(job: tuple) -> dict:
    row, cache, out, move_px, still_px, max_camera, tracks_dir, background_px = job
    sid = row["sample_id"]
    try:
        plan_path = Path(cache) / "plans_real" / f"{sid}.npz"
        if not plan_path.is_file():
            return {"sample_id": sid, "skip": "no measured plan"}
        image = cv2.imread(row["first_frame_path"])
        if image is None:
            return {"sample_id": sid, "skip": "no first frame"}
        plan = mp.load_plan(plan_path)
        if any(np.all(obj.provenance() == mp.FILL_UNKNOWN) for obj in plan.objects):
            return {"sample_id": sid, "skip": "plan without anchor provenance (run relabel_v2.py on the cache)"}
        background = background_evidence(Path(tracks_dir) / f"{sid}.npz" if tracks_dir else None, plan.anchors, *plan.size, background_px, max_camera)
        payload, movers, stills = build_target(plan, image.shape[:2], move_px=move_px, still_px=still_px, background=background)
        if payload["observed_horizons"] == 0:
            return {"sample_id": sid, "skip": "clip shorter than the first horizon"}
        if not movers and not stills:
            return {"sample_id": sid, "skip": "no measured object"}
        payload.update(sample_id=sid, scenario=row["scenario"], caption=row["caption"], video_path=row["first_frame_path"])
        torch.save(payload, Path(out) / "targets" / f"{sid}.pt")
        magnitude = payload["flow"].float().square().sum(1).sqrt()
        return {"sample_id": sid, "movers": movers, "task_ids": movers or stills, "objects": len(plan.objects), "observed": payload["observed_horizons"],
                "kinds": {k: payload["object_kinds"].count(k) for k in ("mover", "still", "unknown")},
                "max_px": float(magnitude.max()), "valid": float(payload["flow_valid"].float().mean())}
    except Exception as error:                                             # noqa: BLE001
        return {"sample_id": sid, "skip": repr(error)[:200]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--train-ids", required=True)
    p.add_argument("--val-ids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--move-px", type=float, default=8.0)
    p.add_argument("--still-px", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tracks", default="", help="real_tracks/<sid>.npz of the clips (the dense grid is the static-background evidence); none = no background supervision")
    p.add_argument("--background-px", type=float, default=2.0)
    p.add_argument("--max-camera-px", type=float, default=2.0)
    a = p.parse_args()
    out = Path(a.out)
    for name in ("targets", "manifests"):
        (out / name).mkdir(parents=True, exist_ok=True)
    rows = {json.loads(line)["sample_id"]: json.loads(line) for line in open(a.manifest) if line.strip()}
    splits = {"train": [l.strip() for l in open(a.train_ids) if l.strip()], "val": [l.strip() for l in open(a.val_ids) if l.strip()]}
    jobs = [(rows[sid], a.cache, str(out), a.move_px, a.still_px, a.max_camera_px, a.tracks, a.background_px) for ids in splits.values() for sid in ids if sid in rows]
    with ProcessPoolExecutor(a.workers) as pool:
        results = {r["sample_id"]: r for r in pool.map(work, jobs, chunksize=8)}
    mapping, summary = {}, {"skipped": {}}
    for split, ids in splits.items():
        kept = [sid for sid in ids if sid in results and "skip" not in results[sid]]
        with open(out / "manifests" / f"{split}.jsonl", "w") as handle:
            for sid in kept:
                row = rows[sid]
                handle.write(json.dumps({"sample_id": sid, "scenario": row["scenario"], "split": split, "video_path": row["first_frame_path"], "caption": row["caption"]}) + "\n")
                mapping[sid] = {"sample_id": sid, "scenario": row["scenario"], "mover_instance_ids": results[sid]["task_ids"],
                                "event": "real_motion" if results[sid]["movers"] else "real_static"}
        (out / f"event_pairs_{split}.json").write_text(json.dumps({"pairs": {}}) + "\n")
        done = [results[sid] for sid in kept]
        summary[split] = {"clips": len(kept), "with_movers": sum(bool(r["movers"]) for r in done), "movers": sum(len(r["movers"]) for r in done),
                          "observed_horizons_mean": float(np.mean([r["observed"] for r in done])) if done else 0.0,
                          "objects": {k: sum(r["kinds"][k] for r in done) for k in ("mover", "still", "unknown")},
                          "max_px_median": float(np.median([r["max_px"] for r in done if r["movers"]])) if any(r["movers"] for r in done) else 0.0}
        for sid in ids:
            if sid in results and "skip" in results[sid]:
                summary["skipped"][results[sid]["skip"]] = summary["skipped"].get(results[sid]["skip"], 0) + 1
    (out / "task_object_ids.json").write_text(json.dumps({"schema": MOVER_SCHEMA, "rows": len(mapping), "label_source": LABEL_SOURCE,
                                                          "policy": "measured movers; every still object when nothing moved", "mapping": mapping}) + "\n")
    summary.update(horizon_seconds=HORIZONS, label_source=LABEL_SOURCE, move_px=a.move_px, still_px=a.still_px, background_px=a.background_px, max_camera_px=a.max_camera_px,
                   tracks=a.tracks)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("REAL_SOURCE", json.dumps(summary))


if __name__ == "__main__":
    main()
