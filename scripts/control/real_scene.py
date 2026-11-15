#!/usr/bin/env python
"""What really moved in a clip, independent of which instances Stage-1 happened to be shown.

Stage-1 reads at most 8 (v1 filter) first-frame instances, largest first.  The object that is about to move is often
small (a held cube, a ball) and is then not among them, so a "real plan" built on Stage-1's instances only would call
a clip with a falling cube static: wrong condition, wrong labels, wrong reward.  The measured scene therefore has
three kinds of objects, all tracked from first-frame points by track_real.py:

  A  the Stage-1 instances                      owner = slot (0..15)        always listed, moving or not
  B  SAM2 proposals Stage-1 was not shown       owner = 100 + raw index     listed when they moved
  C  moving regions no proposal covers          owner = 200 + j             clusters of the dense grid (one tracked
                                                                             point per `stride` px cell, owner -2)

"Moved" is measured relative to the camera (median displacement of the dense grid), at least `move_px` in at least
`min_frames_moved` tracked frames.  Everything is in work-frame pixels, points are (x, y).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import motion_plan as mp  # noqa: E402

BACKGROUND_OWNER, DENSE_OWNER, EXTRA_BASE, CLUSTER_BASE = -1, -2, 100, 200


def dense_grid(size: Sequence[int], stride: int = 16) -> np.ndarray:
    """[gh * gw, 2] centres of the stride x stride cells, row-major."""
    gh, gw = size[0] // stride, size[1] // stride
    ys, xs = (np.arange(gh) + 0.5) * stride - 0.5, (np.arange(gw) + 0.5) * stride - 0.5
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)


def camera_motion(tracks: np.ndarray, visible: np.ndarray, pool: np.ndarray, min_points: int = 8) -> np.ndarray:
    """[F, 2] median displacement of the pool points (visible at both times); zero where too few are visible.
    The median is the background as long as moving things cover less than half of the pool."""
    camera = np.zeros((len(tracks), 2))
    for frame in range(1, len(tracks)):
        ok = pool & visible[frame] & visible[0]
        if int(ok.sum()) >= min_points:
            camera[frame] = np.median((tracks[frame] - tracks[0])[ok], 0)
    return camera


def group_motion(tracks: np.ndarray, visible: np.ndarray, chosen: np.ndarray, camera: np.ndarray, min_visible: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """(disp [F, 2] median displacement of the chosen points relative to the camera, valid [F])."""
    disp, valid = np.zeros((len(tracks), 2)), np.zeros(len(tracks), bool)
    valid[0] = True
    for frame in range(1, len(tracks)):
        ok = chosen & visible[frame] & visible[0]
        if int(ok.sum()) >= min_visible:
            disp[frame] = np.median((tracks[frame] - tracks[0])[ok], 0) - camera[frame]
            valid[frame] = True
    return disp, valid


def moved(disp: np.ndarray, valid: np.ndarray, move_px: float, min_frames_moved: int) -> bool:
    return int((valid & (np.linalg.norm(disp, axis=-1) >= move_px)).sum()) >= min(min_frames_moved, max(1, len(disp) - 1))


def at_rest(tracks: np.ndarray, visible: np.ndarray, owner: np.ndarray, track_times: np.ndarray, *, window_s: float = 0.25, rest_px: float = 2.0,
            dense_fraction: float = 0.02) -> bool:
    """True when the scene has stopped by the end of the tracks: over the last `window_s` seconds no tracked object
    (plan instance or extra proposal; median of its points) moved more than `rest_px`, and at most `dense_fraction`
    of the dense grid moved more than twice that (single points jitter; an untracked moving region is many points)."""
    track_times = np.asarray(track_times, np.float64)
    if len(track_times) < 2:
        return False
    end = len(track_times) - 1
    start = min(int(np.searchsorted(track_times, track_times[end] - window_s, side="left")), end - 1)
    ok = visible[end] & visible[start]
    shift = np.linalg.norm(tracks[end] - tracks[start], axis=-1)
    for slot in np.unique(owner[owner >= 0]):
        chosen = ok & (owner == slot)
        if int(chosen.sum()) >= 3 and float(np.median(shift[chosen])) > rest_px:
            return False
    dense = ok & (owner == DENSE_OWNER)
    return not (dense.any() and float((shift[dense] > 2.0 * rest_px).mean()) > dense_fraction)


def resample_clip(tracks: np.ndarray, visible: np.ndarray, track_times: np.ndarray, t_end: float, t_pad_end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The tracks on the timeline of the clip that is actually trained on: the source rate up to `t_end` (the last real
    clip frame), then frozen at that frame until `t_pad_end` (a clip padded with its last frame shows exactly that).
    Returns (tracks, visible, times) with times = arange * source step, covering t_pad_end."""
    track_times = np.asarray(track_times, np.float64)
    step = float(np.median(np.diff(track_times))) if len(track_times) > 1 else 1.0
    count = int(np.ceil(max(t_pad_end, 0.0) / step - 1e-6)) + 1
    times = np.arange(count) * step
    source = np.abs(track_times[None] - np.minimum(times, t_end)[:, None]).argmin(1)
    return tracks[source], visible[source], times


def discover(tracks: np.ndarray, visible: np.ndarray, owner: np.ndarray, points: np.ndarray, plan_masks: np.ndarray, extra_masks: np.ndarray,
             extra_ids: Sequence[int], size: Sequence[int], *, move_px: float = 8.0, min_frames_moved: int = 2, stride: int = 16,
             min_cells: int = 2) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """(found, owner2, camera): the moving objects Stage-1 was not shown.

    found   [{"slot", "mask" [H, W] bool, "kind" "proposal" | "cluster", "peak_px", "points"}]
    owner2  owner with the dense-grid points of every cluster re-labelled to the cluster's slot
    camera  [F, 2]"""
    height, width = size
    owner2 = np.asarray(owner).copy()
    dense = owner2 == DENSE_OWNER
    camera = camera_motion(tracks, visible, dense if dense.any() else owner2 == BACKGROUND_OWNER)
    covered = np.asarray(plan_masks).astype(bool).any(0) if len(plan_masks) else np.zeros((height, width), bool)
    covered = covered.copy()
    found = []
    for index, raw in enumerate(extra_ids):
        chosen = owner2 == EXTRA_BASE + int(raw)
        disp, valid = group_motion(tracks, visible, chosen, camera)
        if moved(disp, valid, move_px, min_frames_moved):
            found.append({"slot": EXTRA_BASE + int(raw), "mask": np.asarray(extra_masks[index]).astype(bool), "kind": "proposal",
                          "peak_px": float(np.linalg.norm(disp, axis=-1).max()), "points": int(chosen.sum())})
            covered |= found[-1]["mask"]
    if dense.any():
        gh, gw = height // stride, width // stride
        index = np.nonzero(dense)[0]
        if len(index) != gh * gw:
            raise ValueError(f"{len(index)} dense points, expected the {gh}x{gw} grid")
        shift = np.linalg.norm(tracks[:, index] - tracks[:1, index] - camera[:, None], axis=-1)            # [F, G]
        seen = visible[:, index] & visible[:1, index]
        far = ((shift >= move_px) & seen).sum(0) >= min(min_frames_moved, max(1, len(tracks) - 1))
        start = np.rint(points[index]).astype(int)
        free = ~covered[np.clip(start[:, 1], 0, height - 1), np.clip(start[:, 0], 0, width - 1)]
        count, labels = cv2.connectedComponents((far & free).reshape(gh, gw).astype(np.uint8), connectivity=8)
        labels = labels.reshape(-1)
        for component in range(1, count):
            cells = np.nonzero(labels == component)[0]
            if len(cells) < min_cells:
                continue
            mask = np.zeros((height, width), bool)
            for cell in cells:
                row, col = divmod(int(cell), gw)
                mask[row * stride:(row + 1) * stride, col * stride:(col + 1) * stride] = True
            mask &= ~covered
            if not mask.any():
                continue
            slot = CLUSTER_BASE + sum(1 for f in found if f["kind"] == "cluster")
            owner2[index[cells]] = slot
            found.append({"slot": slot, "mask": mask, "kind": "cluster", "peak_px": float(np.median(shift[:, cells].max(0))), "points": int(len(cells))})
    return found, owner2, camera


def measured_plan(sample_id: str, plan_pred: mp.MotionPlan, tracks: np.ndarray, visible: np.ndarray, owner: np.ndarray, points: np.ndarray,
                  track_times: np.ndarray, anchors: Sequence[float], *, extra_masks: np.ndarray | None = None, extra_ids: Sequence[int] = (),
                  move_px: float = 8.0, min_frames_moved: int = 2, stride: int = 16, source: str = "cotracker3") -> tuple[mp.MotionPlan, np.ndarray, dict]:
    """(plan_real, owner2, info): Stage-1's instances with their measured motion plus the movers it was not shown.
    Stage-1's selector is carried on its own instances (zero on the others)."""
    size = tuple(plan_pred.size)
    masks = [o.mask for o in plan_pred.objects]
    slots = [o.slot for o in plan_pred.objects]
    found, owner2, camera = discover(tracks, visible, owner, points, np.stack(masks) if masks else np.zeros((0, *size), bool),
                                     extra_masks if extra_masks is not None else np.zeros((0, *size), bool), extra_ids, size,
                                     move_px=move_px, min_frames_moved=min_frames_moved, stride=stride)
    anchors = np.asarray(anchors, np.float64)
    selector = [[float(o.selector[int(np.abs(plan_pred.anchors - when).argmin())]) for when in anchors] for o in plan_pred.objects]
    for item in found:
        masks.append(item["mask"])
        slots.append(item["slot"])
        selector.append([0.0] * len(anchors))
    plan = mp.from_tracks(sample_id, np.stack(masks) if masks else np.zeros((0, *size), bool), slots, tracks, visible, owner2, track_times, anchors,
                          few_points=2, residual_frac=0.12, selector=np.array(selector).reshape(len(slots), len(anchors)), source=source)
    info = {"stage1_objects": len(plan_pred.objects), "added_proposals": sum(f["kind"] == "proposal" for f in found),
            "added_clusters": sum(f["kind"] == "cluster" for f in found), "camera_max_px": float(np.linalg.norm(camera, axis=-1).max()),
            "added": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in f.items() if k != "mask"} for f in found]}
    return plan, owner2, info


def complete_with_masks(plan: mp.MotionPlan, object_disp: np.ndarray, object_ok: np.ndarray, object_times: np.ndarray) -> tuple[mp.MotionPlan, int]:
    """(plan, anchors filled): where the point fit of an object is unknown at an anchor (its first-frame surface points
    are hidden: a tumbling cube, motion blur) but its mask was found, the object is placed by its mask centroid: the
    last known rotation is kept and the centroid displacement is the measured one (linear between clip frames).
    object_disp [F, K, 2] / object_ok [F, K] at object_times [F], K in the order of plan.objects.  Known anchors are
    left untouched; where the mask is lost too the anchor stays unknown."""
    from dataclasses import replace
    object_times = np.asarray(object_times, np.float64)
    filled, objects = 0, []
    for k, obj in enumerate(plan.objects):
        affines, known, fill = obj.affine.copy(), obj.known.copy(), obj.provenance().copy()
        centre = obj.centroid
        linear = mp.IDENTITY[:, :2].copy()
        for index, when in enumerate(plan.anchors):
            if known[index]:
                linear = affines[index][:, :2].copy()
                continue
            if when > object_times[-1] + 1e-6:
                continue
            upper = int(np.searchsorted(object_times, when, side="left"))
            lower = max(upper - 1, 0)
            upper = min(upper, len(object_times) - 1)
            if not (object_ok[lower, k] and object_ok[upper, k]):
                continue
            span = object_times[upper] - object_times[lower]
            weight = 0.0 if span <= 0 else (when - object_times[lower]) / span
            shift = (1.0 - weight) * object_disp[lower, k] + weight * object_disp[upper, k]
            affines[index] = np.concatenate((linear, (centre + shift - linear @ centre)[:, None]), 1)
            known[index] = True
            fill[index] = mp.FILL_MASK                                        # translation measured, rotation carried over
            filled += 1
        objects.append(replace(obj, affine=affines, known=known, fill=fill))
    return replace(plan, objects=objects, source=f"{plan.source}|mask_completed" if filled else plan.source), filled


def dense_field(tracks: np.ndarray, visible: np.ndarray, owner: np.ndarray, track_times: np.ndarray, times: Sequence[float],
                grid_hw: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """(disp [T, gh, gw, 2] float32, valid [T, gh, gw]): the measured displacement of every dense-grid cell at `times`
    (the nearest tracked frame, within half a tracking step).  `owner` is the ORIGINAL owner array (dense points = -2)."""
    index = np.nonzero(np.asarray(owner) == DENSE_OWNER)[0]
    gh, gw = grid_hw
    if len(index) != gh * gw:
        raise ValueError(f"{len(index)} dense points, expected the {gh}x{gw} grid")
    track_times = np.asarray(track_times, np.float64)
    step = float(np.median(np.diff(track_times))) if len(track_times) > 1 else 0.0
    disp, valid = np.zeros((len(times), gh * gw, 2), np.float32), np.zeros((len(times), gh * gw), bool)
    for j, when in enumerate(times):
        frame = int(np.abs(track_times - when).argmin())
        if abs(float(track_times[frame]) - when) > 0.5 * step + 1e-6:
            continue
        disp[j] = tracks[frame, index] - tracks[0, index]
        valid[j] = visible[frame, index] & visible[0, index]
    disp[~valid] = 0.0
    return disp.reshape(len(times), gh, gw, 2), valid.reshape(len(times), gh, gw)


def dense_from_plan(plan: mp.MotionPlan, times: Sequence[float], stride: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """The same dense field read from a plan (simulation ground truth): a cell moves with the object its centre lies on
    (later objects win where masks overlap), every other cell stays; valid where the object's transform is known."""
    height, width = plan.size
    gh, gw = height // stride, width // stride
    centres = dense_grid(plan.size, stride)
    rows, cols = np.rint(centres[:, 1]).astype(int), np.rint(centres[:, 0]).astype(int)
    disp, valid = np.zeros((len(times), gh * gw, 2), np.float32), np.ones((len(times), gh * gw), bool)
    for obj in plan.objects:
        inside = obj.mask[rows, cols]
        if not inside.any():
            continue
        for j, when in enumerate(times):
            affine, known = mp.affine_at(obj, plan.anchors, float(when))
            disp[j, inside] = mp.apply_affine(affine, centres[inside]) - centres[inside]
            valid[j, inside] = known
    disp[~valid] = 0.0
    return disp.reshape(len(times), gh, gw, 2), valid.reshape(len(times), gh, gw)
