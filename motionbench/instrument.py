"""The measuring instrument used on BOTH sides of every comparison (reference and generated clip):
SAM2 mask propagation for object centroids / areas, CoTracker3 for the free background grid and the camera.

GPU-only functions import the scripts.control modules lazily; `background_points` and `reading_from_tracks`
are pure numpy and unit-tested on the CPU."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from . import contract as C
from .containers import Reading

BG_ROWS, BG_COLS, DENSE_STRIDE = 6, 10, 16


def background_points(masks: np.ndarray, height: int = C.WORK_H, width: int = C.WORK_W) -> Tuple[np.ndarray, np.ndarray]:
    """(points [P,2], kind [P]) — the 6x10 background grid clear of the dilated instance masks (kind 0) plus the
    dense 16 px grid over the whole frame (kind 1). Mirrors scripts/control/track_real.query_points and
    real_scene.dense_grid."""
    union = np.zeros((height, width), bool)
    for m in masks:
        union |= cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ys = np.linspace(0.08 * height, 0.92 * height, BG_ROWS)
    xs = np.linspace(0.06 * width, 0.94 * width, BG_COLS)
    grid = np.array([[x, y] for y in ys for x in xs])
    grid = grid[~union[np.rint(grid[:, 1]).astype(int), np.rint(grid[:, 0]).astype(int)]]
    gh, gw = height // DENSE_STRIDE, width // DENSE_STRIDE
    dy, dx = (np.arange(gh) + 0.5) * DENSE_STRIDE - 0.5, (np.arange(gw) + 0.5) * DENSE_STRIDE - 0.5
    dense = np.stack(np.meshgrid(dx, dy), -1).reshape(-1, 2)
    pts = np.concatenate([grid, dense]).astype(np.float64)
    kind = np.concatenate([np.zeros(len(grid), np.int8), np.ones(len(dense), np.int8)])
    return pts, kind


def content_mask(points: np.ndarray, geom: Optional[dict], margin: float = 3.0) -> np.ndarray:
    """[P] bool: points inside the item's letterboxed content box of the work frame (all True without geometry).
    The background grid covers the whole 448x256 work frame, so for non-16:9 sources the static black bars
    biased the camera (dense-grid median and similarity fit) towards zero; camera estimates use only content points."""
    if not geom:
        return np.ones(len(points), bool)
    x0, y0, w, h = geom["x0"], geom["y0"], geom["w"], geom["h"]
    return (points[:, 0] >= x0 + margin) & (points[:, 0] <= x0 + w - 1 - margin) & (points[:, 1] >= y0 + margin) & (points[:, 1] <= y0 + h - 1 - margin)


def object_points(masks: np.ndarray, per_object: int = 12, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(points [K*per_object (or fewer), 2], owner [N]) — points sampled inside each mask (rotation check)."""
    rng = np.random.default_rng(seed)
    pts, owner = [], []
    for k, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        pick = rng.choice(len(xs), size=min(per_object, len(xs)), replace=False)
        pts.append(np.stack([xs[pick], ys[pick]], -1).astype(np.float64))
        owner += [k] * len(pick)
    if not pts:
        return np.zeros((0, 2)), np.zeros(0, np.int16)
    return np.concatenate(pts), np.array(owner, np.int16)


def camera_from_tracks(tracks: np.ndarray, visible: np.ndarray, pool: np.ndarray, min_points: int = 8) -> np.ndarray:
    """[F,2] median displacement of the pool points visible at both times (real_scene.camera_motion)."""
    cam = np.zeros((len(tracks), 2))
    for f in range(1, len(tracks)):
        ok = pool & visible[f] & visible[0]
        if int(ok.sum()) >= min_points:
            cam[f] = np.median((tracks[f] - tracks[0])[ok], 0)
    return cam


def object_point_motion(obj_tracks: np.ndarray, obj_visible: np.ndarray, owner: np.ndarray, n_objects: int,
                        min_visible: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """(pt_disp [K,F,2], pt_ok [K,F]): median displacement of each object's points visible at frame 0 and f."""
    F = obj_tracks.shape[0]
    disp = np.zeros((n_objects, F, 2)); ok = np.zeros((n_objects, F), bool)
    for k in range(n_objects):
        sel = owner == k
        if not sel.any():
            continue
        for f in range(F):
            v = sel & obj_visible[f] & obj_visible[0]
            if int(v.sum()) >= min_visible:
                disp[k, f] = np.median(obj_tracks[f, v] - obj_tracks[0, v], axis=0)
                ok[k, f] = True
    return disp, ok


def reading_from_tracks(times: np.ndarray, centroid: np.ndarray, area: np.ndarray, tracks: np.ndarray,
                        visible: np.ndarray, kind: np.ndarray, fps: float, prompt_iou: Optional[np.ndarray] = None,
                        obj_tracks: Optional[np.ndarray] = None, obj_visible: Optional[np.ndarray] = None,
                        owner: Optional[np.ndarray] = None, content: Optional[np.ndarray] = None) -> Reading:
    """Assemble a Reading from raw instrument outputs.
    centroid [F,K,2] (nan where empty), area [F,K], tracks [F,P,2], visible [F,P], kind [P] (0 bg grid, 1 dense)."""
    F, K = area.shape
    first = centroid[0]
    ratio = area / np.maximum(area[:1], 1.0)
    ok = np.isfinite(centroid).all(-1) & np.isfinite(first).all(-1)[None] & (ratio >= 0.3) & (ratio <= 3.3)
    disp = np.where(ok[..., None], centroid - first[None], 0.0)
    dense = kind == 1
    if content is not None and (dense & content).sum() >= 8:           # letterbox bars out of the camera pool
        dense = dense & content
    camera = camera_from_tracks(tracks, visible, dense if dense.any() else kind == 0)
    bg_disp = (tracks - tracks[:1]).transpose(1, 0, 2)              # [P,F,2]
    bg_ok = (visible & visible[:1]).T                                # [P,F]
    r = Reading(times=np.asarray(times, np.float64), pos0=np.nan_to_num(first), disp=disp.transpose(1, 0, 2).astype(np.float64),
                ok=ok.T, area=area.T.astype(np.float64), camera=camera, bg_disp=bg_disp.astype(np.float64), bg_ok=bg_ok, fps=float(fps))
    if prompt_iou is not None:
        r.meta["prompt_iou"] = np.asarray(prompt_iou, np.float64).tolist()
    if obj_tracks is not None and owner is not None and len(owner):
        r.pt_disp, r.pt_ok = object_point_motion(obj_tracks, obj_visible, owner, K)
    return r.validate()


class MotionInstrument:
    """GPU wrapper around scripts/control/{masktrack,cotrack}: measure a work-frame clip."""

    def __init__(self, cotracker_repo: str, cotracker_checkpoint: str, sam2_checkpoint: str, device: str = "cuda",
                 scale: float = 2.0, vis_threshold: float = 0.5, tmp_root: Optional[str] = None):
        import torch
        from scripts.control import masktrack as mt
        from scripts.control.cotrack import load_tracker
        self.mt, self.scale, self.vis, self.tmp_root = mt, float(scale), float(vis_threshold), tmp_root
        self.points = load_tracker(cotracker_repo, cotracker_checkpoint, torch.device(device))
        self.objects = mt.load_mask_tracker(sam2_checkpoint, device=device)

    def _enlarge(self, frames: np.ndarray):
        h, w = frames.shape[1:3]
        size = (int(round(w * self.scale)), int(round(h * self.scale)))
        big = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_CUBIC) for f in frames])
        return big, np.array([(size[0] - 1) / (w - 1), (size[1] - 1) / (h - 1)])

    def measure(self, frames: np.ndarray, masks: np.ndarray, times: np.ndarray, fps: float,
                extra_points: Optional[np.ndarray] = None, keep_masks: bool = False,
                owner: Optional[np.ndarray] = None, geom: Optional[dict] = None) -> Tuple[Reading, dict]:
        """frames [F,H,W,3] RGB uint8 in the work frame; masks [K,H,W] bool of frame 0.
        Returns (Reading, extras) where extras has the raw tracks of extra_points (object points) and last masks."""
        from scripts.control.cotrack import track_points
        big, factor = self._enlarge(frames)
        bg_pts, kind = background_points(masks, *frames.shape[1:3])
        pts = bg_pts if extra_points is None or len(extra_points) == 0 else np.concatenate([bg_pts, extra_points])
        tracks, score = track_points(self.points, big, pts * factor)
        tracks = tracks / factor
        visible = score > self.vis
        nb = len(bg_pts)
        large = np.stack([cv2.resize(m.astype(np.uint8), (big.shape[2], big.shape[1]), interpolation=cv2.INTER_NEAREST) > 0 for m in masks]) \
            if len(masks) else np.zeros((0, *big.shape[1:3]), bool)
        prop = self.mt.propagate(self.objects, big, large, keep_masks=keep_masks, tmp_root=self.tmp_root)
        centroid = prop["centroid"] / factor[None, None, :]
        area = prop["area"] / float(factor[0] * factor[1])
        reading = reading_from_tracks(times, centroid, area, tracks[:, :nb], visible[:, :nb], kind, fps, prop["prompt_iou"],
                                      obj_tracks=tracks[:, nb:] if owner is not None else None,
                                      obj_visible=visible[:, nb:] if owner is not None else None, owner=owner,
                                      content=content_mask(bg_pts, geom) if geom else None)
        extras = dict(obj_tracks=tracks[:, nb:], obj_visible=visible[:, nb:])
        if keep_masks:
            extras["masks_big"] = prop["masks"]                  # [F,K,H2,W2] bool at the enlarged resolution (contact instrument)
            extras["factor"] = factor
            small = prop["masks"][-1]
            extras["last_masks"] = np.stack([cv2.resize(m.astype(np.uint8), (frames.shape[2], frames.shape[1]),
                                                        interpolation=cv2.INTER_NEAREST) > 0 for m in small]) if len(small) else small
        return reading, extras
