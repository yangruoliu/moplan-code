#!/usr/bin/env python
"""Scene v5 of a Stage-2 v2 cache: the real side of the reward (scripts/control/reward.py) with EVERY frame of the clip
as an evaluation time, plus the relabel_v2 point readings and the real rotation per cell.

The scenes of cache_stage2_cond_v2.py keep 6 evaluation times of a clip (a 37-frame clip: 1.0, 1.86, ... s); a 13-frame
rollout truncated at 1.714 s is then judged at 1.0 s only, and the full-rate SAM2 readings are gone.  Per clip (same
frames as the cache: stretched first frame, read_frames at the cache's fps, the last frame repeated while padded):
  Instrument.measure     CoTracker3 points + SAM2 mask centroids, the settings of score_generated.py track
  build_scene(max_times=None)   every frame k / fps (k = 1 .. F-1) is an evaluation time
  relabel_v2.point_readings     from plans_real/<sid>.npz (plans/ with a warning when absent): moving_centroid /
                         static_centroid (the centroid labels), max_point_px, fill [K, T]; moving / static by the
                         relabel_v2 rule at the reward thresholds (3 % / 1 % of the diagonal); real_rot [K, T] degrees
                         (polar angle of affine_at) and rot_known (fill == FILL_POINTS) for the rotation cells of v5.
Written to --out/<sid>.npz (default <cache>/scenes_v5); the old scenes/ are not touched (v4 stays reproducible).
Finished files are skipped; one receipt line per shard.

usage: rescene_v5.py --cache DIR --manifest M.jsonl [--ids F] [--out DIR/scenes_v5] [--shard-index i --num-shards n] [--limit N]
                     [--cotracker-repo R] [--cotracker-checkpoint C] [--sam2-checkpoint S] [--overwrite]
       (the tracker defaults are $COTRACKER_REPO, $COTRACKER_CKPT and $SAM2_CHECKPOINT)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import motion_plan as mp  # noqa: E402
from scripts.cache_stage2_cond_v2 import read_frames, stretched_image  # noqa: E402
from scripts.control import relabel_v2  # noqa: E402
from scripts.control import reward as rw  # noqa: E402

SCENE_VERSION = 5


def clip_frames(payload: dict, row: dict) -> tuple[np.ndarray, float]:
    """The clip exactly as the cache saw it: (frames [F, H, W, 3] uint8 with F = payload["frames"], fps)."""
    height, width, used, real = int(payload["height"]), int(payload["width"]), int(payload["frames"]), int(payload["frames_real"])
    fps = float(payload["fps"])
    first = stretched_image(row.get("first_frame_path") or payload["first_frame_path"], height, width)[0]
    clip = read_frames(row.get("video_path") or payload["video_path"], real, height, width, fps)
    if len(clip) < real:
        raise ValueError(f"video has {len(clip)} of {real} frames")
    clip[0] = first
    return (np.concatenate((clip, np.repeat(clip[-1:], used - real, 0))) if used > real else clip), fps


def relabel(scene: dict, plan: mp.MotionPlan) -> dict:
    """The relabel_v2 keys and the real rotation on a scene whose `moving` / `static` are the centroid labels."""
    slots = [int(v) for v in scene["slots"]]
    moved, rot, fill, known = relabel_v2.point_readings(plan, [float(v) for v in scene["times"]], slots)
    diagonal, ok = float(scene["diagonal"]), scene["real_ok"].astype(bool)
    c_moving, c_static = scene["moving"].astype(bool), scene["static"].astype(bool)
    moving = (c_moving | (known & (moved >= rw.MOVE_FRAC * diagonal))) & ok
    static = (c_static & ~(known & (moved > rw.STILL_FRAC * diagonal))) & ok
    scene.update(moving_centroid=c_moving, static_centroid=c_static, moving=moving, static=static, max_point_px=moved.astype(np.float32), fill=fill,
                 real_rot=rot.astype(np.float32), rot_known=(fill == mp.FILL_POINTS) & known, labels_version=np.int32(relabel_v2.VERSION),
                 scene_version=np.int32(SCENE_VERSION))
    return scene


def rescene(cache: Path, out: Path, sid: str, row: dict, instrument) -> dict:
    payload = torch.load(cache / f"{sid}.pt", map_location="cpu", weights_only=False)
    old = rw.load_scene(cache / "scenes" / f"{sid}.npz")
    plan_path, warning = cache / "plans_real" / f"{sid}.npz", ""
    if not plan_path.is_file():
        plan_path, warning = cache / "plans" / f"{sid}.npz", "plans_real missing: point readings from plans/"
    plan = relabel_v2.provenance_from_residuals(mp.load_plan(plan_path))
    frames, fps = clip_frames(payload, row)
    times = np.arange(len(frames)) / fps
    masks = rw.scene_masks(old)
    seen = instrument.measure(frames, old["points"], masks)
    scene = rw.build_scene(old["points"], old["owner"], [int(v) for v in old["slots"]], masks, seen["tracks"], seen["visible"], times, float(times[-1]),
                           fps=fps, max_times=None, object_disp=seen["object_disp"], object_ok=seen["object_ok"], object_times=times)
    scene = relabel(scene, plan)
    scene["source_scene"] = str(old.get("version", ""))
    rw.save_scene(out / f"{sid}.npz", scene)
    return {"sample_id": sid, "frames": int(len(frames)), "times": int(len(scene["times"])), "movers": int(scene["moving"].any(1).sum()),
            "rot_cells": int((scene["rot_known"] & (np.abs(scene["real_rot"]) >= rw.ROT_MIN_DEG) & scene["moving"] & ~scene["moving_centroid"]).sum()),
            "warning": warning}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--manifest", required=True, help="jsonl with sample_id, video_path, first_frame_path on this machine")
    p.add_argument("--ids", default="", help="sample ids (default: every <sid>.pt with a scene)")
    p.add_argument("--out", default="", help="default <cache>/scenes_v5")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cotracker-repo", default=os.environ.get("COTRACKER_REPO", ""), help="co-tracker git checkout (default $COTRACKER_REPO)")
    p.add_argument("--cotracker-checkpoint", default=os.environ.get("COTRACKER_CKPT", ""), help="CoTracker3 scaled_offline.pth (default $COTRACKER_CKPT)")
    p.add_argument("--sam2-checkpoint", default=os.environ.get("SAM2_CHECKPOINT", ""), help="SAM 2.1 hiera-large .pt (default $SAM2_CHECKPOINT)")
    p.add_argument("--track-scale", type=float, default=2.0)
    p.add_argument("--vis-threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    from scripts.control.measure import Instrument

    cache = Path(a.cache)
    out = Path(a.out) if a.out else cache / "scenes_v5"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for line in Path(a.manifest).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            manifest[row["sample_id"]] = row
    ids = ([line.strip() for line in Path(a.ids).read_text().splitlines() if line.strip()] if a.ids
           else sorted(q.stem for q in cache.glob("*.pt") if not q.name.startswith("prompt_embeds") and (cache / "scenes" / f"{q.stem}.npz").is_file()))
    if a.limit:
        ids = ids[: a.limit]
    ids = ids[a.shard_index:: a.num_shards]
    masks = {"sam2_checkpoint": a.sam2_checkpoint} if a.sam2_checkpoint else {}
    instrument = Instrument(a.cotracker_repo, a.cotracker_checkpoint, **masks, scale=a.track_scale, vis_threshold=a.vis_threshold, device=a.device)
    done, skipped, missing, errors, warned, times, started = 0, 0, 0, [], 0, [], time.time()
    for count, sid in enumerate(ids):
        if (out / f"{sid}.npz").is_file() and not a.overwrite:
            skipped += 1
            continue
        if not (cache / f"{sid}.pt").is_file() or not (cache / "scenes" / f"{sid}.npz").is_file():
            missing += 1
            continue
        try:
            result = rescene(cache, out, sid, manifest.get(sid, {}), instrument)
            done += 1
            times.append(result["times"])
            warned += bool(result["warning"])
            if result["times"] != result["frames"] - 1:
                errors.append({"sample_id": sid, "error": f"{result['times']} times for {result['frames']} frames"})
        except Exception as error:  # noqa: BLE001
            errors.append({"sample_id": sid, "error": repr(error)[:300]})
            print(f"ERROR {sid}: {error!r}"[:400], flush=True)
        if (count + 1) % 50 == 0:
            print(f"[rescene_v5] {count + 1}/{len(ids)} done {done} errors {len(errors)}", flush=True)
    receipt = {"shard": f"{a.shard_index}/{a.num_shards}", "ids": len(ids), "done": done, "skipped": skipped, "missing": missing, "errors": len(errors),
               "plans_fallback": warned, "mean_times": float(np.mean(times)) if times else None, "seconds_per_clip": (time.time() - started) / max(1, done),
               "out": str(out), "first_errors": errors[:5]}
    (out / f"receipt-shard{a.shard_index:02d}of{a.num_shards:02d}.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print("RESCENE_V5", json.dumps(receipt))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
