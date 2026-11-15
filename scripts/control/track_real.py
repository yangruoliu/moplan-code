#!/usr/bin/env python
"""Real object motion for a manifest of clips: the objects of the Stage-1 plan, tracked with CoTracker3 at the
source frame rate.

For every sample the frozen Stage-1 export gives the plan (every SAM2 instance, rigid fit of the decoder's own
displacement) in the generator's work frame.  12 points per object (motion_plan.sample_points) and a background grid
are tracked through the real clip.  The source video is tracked at its native rate when the manifest names it
(`source_path`): a fall lasts a few 7-fps frames, far too few for a tracker; the 7-fps positions are read from the
native-rate tracks afterwards.

Besides Stage-1's own instances (owner = slot) three more point sets are tracked, so that what really moved can be
measured even when Stage-1 was not shown it (scripts/control/real_scene.py):
  owner 100 + i   12 points on every raw SAM2 proposal i of --raw-masks that is not one of the plan's objects
  owner -1        a 6 x 10 background grid, clear of the plan's objects
  owner -2        one point per 16 px cell of the work frame (the dense grid, row-major)

The clip is tracked at --track-scale times the work frame (2 by default: the tracker resamples to 384x512 itself,
and small fast objects survive that better from a sharper input); coordinates are stored in work-frame pixels.
The tracker's visibility SCORE is stored: `visible` is score > --vis-threshold (0.5; the tracker's own 0.9 drops
motion-blurred points, i.e. the fastest objects), `score` keeps the raw value for stricter readings.

written per sample
  plans/<sid>.npz        the Stage-1 plan, all objects (what the cache builder reads)
  real_tracks/<sid>.npz  tracks [F, N, 2] float32 (x, y) work px, visible [F, N] bool, score [F, N] uint8 (x255),
                         owner [N] int16, times [F] float32 seconds, points [N, 2], extra_masks (packed bits of
                         [M, H, W]) with extra_ids [M] (raw proposal indices), dense_stride
  summary.jsonl          one line per sample: objects, tracked frames, final displacement per object

usage: track_real.py --manifest M.jsonl --exports DIR --out DIR [--cotracker-repo R] [--checkpoint C]
                     [--size 256x448] [--max-seconds 5.3] [--points 12] [--shard-index i --num-shards n] [--plans-only]
       (--cotracker-repo / --checkpoint default to $COTRACKER_REPO / $COTRACKER_CKPT; the RAFT PhyCo sources were
       tracked with --raw-masks '' --max-seconds 4)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import motion_plan as mp  # noqa: E402
from scripts.control import real_scene as rs  # noqa: E402

cv2.setNumThreads(2)


def read_clip(path: str, size: tuple[int, int], max_seconds: float) -> tuple[np.ndarray, float]:
    """(frames [F, H, W, 3] uint8 resized to the work frame, fps) of the first `max_seconds` of a video."""
    capture = cv2.VideoCapture(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    limit = int(np.floor(max_seconds * fps + 1e-6)) + 1
    frames = []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size[1], size[0]), interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames:
        raise RuntimeError(f"cannot read {path}")
    return np.stack(frames), fps


def extra_proposals(raw_masks: np.ndarray, plan: mp.MotionPlan, grid_box, same_iou: float = 0.5, min_px: int = 12) -> tuple[np.ndarray, list[int]]:
    """(masks [M, H, W] bool in the work frame, raw indices): the raw SAM2 proposals that are not plan objects."""
    kept, ids = [], []
    for index, grid_mask in enumerate(raw_masks):
        if not grid_mask.any():
            continue
        mask = mp.work_mask(grid_mask, grid_box, plan.size)
        if int(mask.sum()) < min_px:
            continue
        same = False
        for obj in plan.objects:
            both = int((mask & obj.mask).sum())
            if both and both / float((mask | obj.mask).sum()) >= same_iou:
                same = True
                break
        if not same:
            kept.append(mask)
            ids.append(index)
    return (np.stack(kept) if kept else np.zeros((0, *plan.size), bool)), ids


def query_points(plan: mp.MotionPlan, per_object: int, rows: int = 6, cols: int = 10, extra_masks: np.ndarray | None = None,
                 extra_ids=(), dense_stride: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(points [N, 2], owner [N]): `per_object` points on every plan object, the same on every extra proposal
    (owner 100 + raw index), a grid of background points that stays clear of the (dilated) plan objects (-1, the
    6 x 10 layout), and the dense grid (-2)."""
    points, owner = [], []
    height, width = plan.size
    union = np.zeros(plan.size, bool)
    for obj in plan.objects:
        sampled = mp.sample_points(obj.mask, per_object)
        points.append(sampled)
        owner += [obj.slot] * len(sampled)
        union |= cv2.dilate(obj.mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    for index, mask in zip(extra_ids, extra_masks if extra_masks is not None else ()):
        sampled = mp.sample_points(mask, per_object)
        points.append(sampled)
        owner += [rs.EXTRA_BASE + int(index)] * len(sampled)
    ys = np.linspace(0.08 * height, 0.92 * height, rows)
    xs = np.linspace(0.06 * width, 0.94 * width, cols)
    grid = np.array([[x, y] for y in ys for x in xs])
    grid = grid[~union[np.rint(grid[:, 1]).astype(int), np.rint(grid[:, 0]).astype(int)]]
    points.append(grid)
    owner += [rs.BACKGROUND_OWNER] * len(grid)
    if dense_stride:
        dense = rs.dense_grid(plan.size, dense_stride)
        points.append(dense)
        owner += [rs.DENSE_OWNER] * len(dense)
    return np.concatenate(points).astype(np.float64), np.array(owner, np.int16)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--exports", required=True, help="Stage-1 export folder (<sid>.pt)")
    p.add_argument("--out", required=True)
    p.add_argument("--size", default="256x448", help="work frame HxW (the generator's resolution)")
    p.add_argument("--max-seconds", type=float, default=36.0 / 7.0, help="37 frames at 7 fps")
    p.add_argument("--points", type=int, default=12)
    p.add_argument("--raw-masks", default="auto", help="raw SAM2 proposals (<sid>.pt, instance_masks on the Stage-1 grid): also track the ones "
                   "Stage-1 was not shown.  auto = <out>/sam2_raw when it exists, '' = none")
    p.add_argument("--dense-stride", type=int, default=16, help="one tracked point per cell of this size (0 = no dense grid)")
    p.add_argument("--max-track-frames", type=int, default=96, help="longer clips are tracked on every n-th source frame")
    p.add_argument("--track-scale", type=float, default=2.0)
    p.add_argument("--vis-threshold", type=float, default=0.5)
    p.add_argument("--cotracker-repo", default=os.environ.get("COTRACKER_REPO", ""), help="co-tracker git checkout (default $COTRACKER_REPO)")
    p.add_argument("--checkpoint", default=os.environ.get("COTRACKER_CKPT", ""), help="CoTracker3 scaled_offline.pth (default $COTRACKER_CKPT)")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--plans-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    import torch

    size = tuple(int(v) for v in a.size.lower().split("x"))
    rows = [json.loads(line) for line in Path(a.manifest).read_text().splitlines() if line.strip()]
    rows = rows[a.shard_index:: a.num_shards]
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out)
    (out / "plans").mkdir(parents=True, exist_ok=True)
    (out / "real_tracks").mkdir(parents=True, exist_ok=True)
    model = None
    if not a.plans_only:
        from scripts.control.cotrack import load_tracker, track_points
        model = load_tracker(a.cotracker_repo, a.checkpoint, torch.device("cuda"))
    if a.raw_masks == "auto":
        a.raw_masks = str(out / "sam2_raw") if (out / "sam2_raw").is_dir() else ""
    print(f"[track_real] raw proposals: {a.raw_masks or 'none'}; dense stride {a.dense_stride}; visibility > {a.vis_threshold}", flush=True)
    track_size = (int(round(size[0] * a.track_scale)), int(round(size[1] * a.track_scale)))
    summary_path = out / f"summary-shard{a.shard_index:02d}of{a.num_shards:02d}.jsonl"
    done = errors = 0
    with open(summary_path, "a") as summary:
        for count, row in enumerate(rows):
            sid = row["sample_id"]
            plan_path, track_path = out / "plans" / f"{sid}.npz", out / "real_tracks" / f"{sid}.npz"
            if track_path.is_file() and plan_path.is_file() and not a.overwrite:
                continue
            try:
                export_path = Path(a.exports) / f"{sid}.pt"
                if not export_path.is_file():
                    raise FileNotFoundError(f"no Stage-1 export for {sid}")
                export = torch.load(export_path, map_location="cpu", weights_only=False)
                native = cv2.imread(row["first_frame_path"])
                plan = mp.from_stage1_export(export, size, native_hw=native.shape[:2])
                plan.sample_id = sid
                mp.save_plan(plan_path, plan)
                if a.plans_only:
                    done += 1
                    continue
                extra_masks, extra_ids = np.zeros((0, *size), bool), []
                if a.raw_masks:
                    raw = torch.load(Path(a.raw_masks) / f"{sid}.pt", map_location="cpu", weights_only=False)["instance_masks"].float().numpy() > 0.5
                    extra_masks, extra_ids = extra_proposals(raw, plan, mp.inner_box(*raw.shape[1:], *native.shape[:2]))
                points, owner = query_points(plan, a.points, extra_masks=extra_masks, extra_ids=extra_ids, dense_stride=a.dense_stride)
                frames, fps = read_clip(row.get("source_path") or row["video_path"], track_size, a.max_seconds)
                every = int(np.ceil(len(frames) / a.max_track_frames))
                frames, fps = frames[::every], fps / every
                factor = np.array([(track_size[1] - 1) / (size[1] - 1), (track_size[0] - 1) / (size[0] - 1)])
                tracks, score = track_points(model, frames, points * factor)
                tracks = tracks / factor
                visible = score > a.vis_threshold
                times = np.arange(len(frames), dtype=np.float32) / np.float32(fps)
                temporary = track_path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, tracks=tracks.astype(np.float32), visible=visible, score=np.round(score * 255).astype(np.uint8),
                                    owner=owner, times=times, points=points.astype(np.float32), fps=np.float32(fps), size=np.array(size),
                                    vis_threshold=np.float32(a.vis_threshold), track_scale=np.float32(a.track_scale),
                                    extra_masks=np.packbits(extra_masks.reshape(-1)), extra_ids=np.array(extra_ids, np.int16),
                                    dense_stride=np.int16(a.dense_stride))
                temporary.replace(track_path)
                last = {}
                for obj in plan.objects:
                    chosen = owner == obj.slot
                    ok = visible[-1, chosen] & visible[0, chosen]
                    delta = (tracks[-1, chosen] - tracks[0, chosen])[ok]
                    last[str(obj.slot)] = [round(float(v), 1) for v in np.median(delta, 0)] if len(delta) >= 3 else None
                found, _, camera = rs.discover(tracks.astype(np.float64), visible, owner, points, np.stack([o.mask for o in plan.objects]) if plan.objects
                                               else np.zeros((0, *size), bool), extra_masks, extra_ids, size, stride=a.dense_stride or 16)
                summary.write(json.dumps({"sample_id": sid, "objects": len(plan.objects), "frames": int(len(frames)), "fps": round(fps, 3),
                                          "seconds": round(float(times[-1]), 3), "points": int(len(points)), "last_displacement": last,
                                          "extras": len(extra_ids), "movers_outside_plan": [[f["kind"], f["slot"], round(f["peak_px"], 1)] for f in found],
                                          "camera_max_px": round(float(np.linalg.norm(camera, axis=-1).max()), 1)}) + "\n")
                summary.flush()
                done += 1
            except Exception as error:  # noqa: BLE001
                errors += 1
                print(f"ERROR {sid}: {error!r}"[:300], flush=True)
            if (count + 1) % 50 == 0:
                print(f"[track_real] {count + 1}/{len(rows)}", flush=True)
    print(f"TRACK_REAL_DONE shard {a.shard_index}/{a.num_shards}: done {done} errors {errors} of {len(rows)}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
