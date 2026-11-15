#!/usr/bin/env python
"""One representation of a coarse, object-level motion plan, shared by every Stage-2 control route.

Stage-1 answers "which first-frame object goes where, roughly, over the next seconds".  This module keeps
that answer intact and converts it -- with one set of coordinate conventions -- into what each control
mechanism consumes:

  to_tracks         point trajectories                              (Wan-Move, ATI)
  to_crude_video    cut-and-drag animation + control mask           (Time-to-Move)
  to_dense_target   flow / masks for the 20-channel adapter input   (cache_stage2_oracle.make_motion_condition)

Conventions
  * work frame: the conditioning image as the generator sees it, W x H pixels, pixel centres at integers.
  * per object: the first-frame mask M0 [H, W] (source coordinates) and one pixel affine per anchor,
        destination = A(t) @ [x, y, 1]                  (A = identity: the object does not move)
    Every position is computed from the SAME first-frame point:  p(t) = A(t) p0  ==  p0 + F_{0->t}(p0).
    A cumulative displacement of 10 px and then 20 px puts the point at 10 and 20, never at 30, and it may
    shrink, reverse or return to zero (bounce, spring): nothing here is made monotone.
  * known[t] False means "the plan says nothing at this anchor".  It is not zero motion.  Anchors are in
    seconds; between anchors the affine is interpolated linearly (identity at t = 0); after the last known
    anchor the position is held and known is False.
  * selector is Stage-1's task-object score.  It is metadata only: it never scales or deletes motion here.
  * three masks are kept apart: the source mask M0 (which first-frame pixels belong to the object), the
    source-point visibility (whether those pixels are still visible at t) and the target mask
    Mt = A(t) o M0 (where the object is at t).  The moved mask is never written into a visibility field.

Only numpy and OpenCV are needed (torch tensors are accepted where a Stage-1 export is read).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
FILL_POINTS, FILL_MASK, FILL_FEW, FILL_HOLD, FILL_UNKNOWN = 0, 1, 2, 3, -1


def _polar(linear: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(angle rad, log-scale, residual S) of a 2x2 linear part L = R(angle) * exp(log-scale) * S (polar decomposition;
    S is symmetric with unit determinant magnitude: the shear / anisotropy left over)."""
    u, singular, vt = np.linalg.svd(linear)
    if np.linalg.det(u @ vt) < 0:
        u = u.copy()
        u[:, -1] *= -1
        singular = singular.copy()
        singular[-1] *= -1
    rotation = u @ vt
    angle = math.atan2(rotation[1, 0], rotation[0, 0])
    scale = float(np.sqrt(max(abs(singular[0] * singular[1]), 1e-12)))
    stretch = rotation.T @ linear / scale
    return angle, math.log(scale), stretch


def _unpolar(angle: float, log_scale: float, stretch: np.ndarray) -> np.ndarray:
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return rotation @ (math.exp(log_scale) * stretch)


# ---------------------------------------------------------------------------------------------- geometry
def inner_box(height: int, width: int, native_h: int, native_w: int) -> tuple[int, int, int, int]:
    """(top, left, inner_h, inner_w) of a native_h x native_w frame letterboxed into height x width."""
    scale = min(height / native_h, width / native_w)
    inner_h, inner_w = round(native_h * scale), round(native_w * scale)
    return (height - inner_h) // 2, (width - inner_w) // 2, inner_h, inner_w


def grid_to_work(xy: np.ndarray, grid_box: Sequence[int], size: Sequence[int]) -> np.ndarray:
    """[..., 2] (x, y) on the Stage-1 grid (pixel centres at integers) -> the same points in the work frame."""
    top, left, grid_h, grid_w = grid_box
    height, width = size
    out = np.empty(xy.shape, np.float64)
    out[..., 0] = (xy[..., 0] - left + 0.5) * (width / grid_w) - 0.5
    out[..., 1] = (xy[..., 1] - top + 0.5) * (height / grid_h) - 0.5
    return out


def fit_affine(src: np.ndarray, dst: np.ndarray, min_points: int = 12, model: str = "affine") -> tuple[np.ndarray, float]:
    """Least-squares pixel transform with dst ~= A @ [src, 1].  Returns (A [2, 3], mean residual in px).

    model 'affine'       all six parameters (rotation, scale, shear, translation)
          'rigid'        rotation about the centroid + translation (Procrustes): the centroid moves by the mean
                         displacement, the object keeps its size and shape
          'translation'  the mean displacement only
    Too few or collinear points fall back to the translation."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    translation = IDENTITY.copy()
    translation[:, 2] = (dst - src).mean(0)
    centred = src - src.mean(0)
    spread = np.linalg.svd(centred, compute_uv=False) if len(src) >= 2 else np.zeros(2)
    if model == "translation" or len(src) < min_points or spread[-1] < 1e-3 * max(spread[0], 1e-9):
        return translation, float(np.linalg.norm(src + translation[:, 2] - dst, axis=1).mean())
    if model == "rigid":
        u, _, vt = np.linalg.svd((dst - dst.mean(0)).T @ centred)
        rotation = u @ np.diag([1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        affine = np.concatenate((rotation, (dst.mean(0) - rotation @ src.mean(0))[:, None]), 1)
        return affine, float(np.linalg.norm(apply_affine(affine, src) - dst, axis=1).mean())
    if model != "affine":
        raise ValueError(f"unknown transform model {model!r}")
    design = np.concatenate((src, np.ones((len(src), 1))), 1)
    solution, *_ = np.linalg.lstsq(design, dst, rcond=None)                 # [3, 2]
    affine = solution.T
    return affine, float(np.linalg.norm(design @ solution - dst, axis=1).mean())


def apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    """A [2, 3] applied to [..., 2] points."""
    return points @ affine[:, :2].T + affine[:, 2]


# -------------------------------------------------------------------------------------------------- plan
@dataclass
class ObjectPlan:
    slot: int
    mask: np.ndarray                 # [H, W] bool, first frame, work frame
    affine: np.ndarray               # [T, 2, 3]
    known: np.ndarray                # [T] bool
    selector: np.ndarray             # [T] float, metadata
    residual_px: np.ndarray = field(default_factory=lambda: np.zeros(0))   # [T] fit residual, diagnostics
    area_scale: np.ndarray = field(default_factory=lambda: np.zeros(0))    # [T] det of the unconstrained affine fitted to the field
    fill: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))  # [T] provenance of each anchor, see FILL_*

    @property
    def centroid(self) -> np.ndarray:
        ys, xs = np.nonzero(self.mask)
        return np.array([xs.mean(), ys.mean()])

    def provenance(self) -> np.ndarray:
        """[T] int8: FILL_POINTS (rotation measured), FILL_MASK (mask centroid: translation only), FILL_FEW (a few points:
        translation only), FILL_HOLD (not measured), FILL_UNKNOWN (a plan saved before provenance was recorded)."""
        if len(self.fill) == len(self.known):
            return np.asarray(self.fill, np.int8)
        return np.full(len(self.known), FILL_UNKNOWN, np.int8)

    def angles(self) -> np.ndarray:
        """[T] rotation angle (degrees) of the linear part of every anchor's transform (polar decomposition)."""
        return np.array([math.degrees(_polar(a[:, :2])[0]) for a in self.affine])

    def point_motion(self, trust_rotation: Optional[np.ndarray] = None) -> np.ndarray:
        """[T] largest displacement (px) of any corner of the mask's bounding box at every anchor.  Where the rotation
        is not trusted (trust_rotation False; default: provenance is FILL_MASK / FILL_FEW / FILL_UNKNOWN) only the
        centroid translation counts, so that a carried-over rotation is not read as motion."""
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return np.zeros(len(self.known))
        corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]], np.float64)
        c0 = self.centroid
        if trust_rotation is None:
            trust_rotation = self.provenance() == FILL_POINTS
        out = np.zeros(len(self.known))
        for t, affine in enumerate(self.affine):
            if trust_rotation[t]:
                out[t] = np.linalg.norm(apply_affine(affine, corners) - corners, axis=1).max()
            else:
                out[t] = np.linalg.norm(apply_affine(affine, c0[None])[0] - c0)
        return out

    def displacement(self) -> np.ndarray:
        """[T, 2] displacement of the first-frame centroid at every anchor (not accumulated across anchors)."""
        c0 = self.centroid
        return np.stack([apply_affine(a, c0) - c0 for a in self.affine])

    def max_point_motion(self) -> float:
        """Largest displacement of any corner of the mask's bounding box over the known anchors (px)."""
        ys, xs = np.nonzero(self.mask)
        corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]], np.float64)
        moved = [np.linalg.norm(apply_affine(a, corners) - corners, axis=1).max() for a, k in zip(self.affine, self.known) if k]
        return float(max(moved)) if moved else 0.0


@dataclass
class MotionPlan:
    sample_id: str
    size: tuple[int, int]            # (H, W) of the work frame
    anchors: np.ndarray              # [T] seconds, increasing, > 0
    objects: list[ObjectPlan]
    source: str = ""

    def object(self, slot: int) -> ObjectPlan:
        return next(o for o in self.objects if o.slot == slot)


def _np(value) -> np.ndarray:
    """numpy view of an array or a torch tensor (half/bf16 tensors are widened first)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        value = value.numpy() if "bool" in str(value.dtype) or "int" in str(value.dtype) else value.float().numpy()
    return np.asarray(value)


def work_mask(grid_mask: np.ndarray, grid_box: Sequence[int], size: Sequence[int]) -> np.ndarray:
    """Content box of a Stage-1 grid mask resampled to the work frame (smooth contour instead of blocks)."""
    top, left, grid_h, grid_w = grid_box
    content = grid_mask[top: top + grid_h, left: left + grid_w].astype(np.float32)
    return cv2.resize(content, (size[1], size[0]), interpolation=cv2.INTER_LINEAR) > 0.5


def from_flow_grid(sample_id: str, flow_px: np.ndarray, instance_masks: np.ndarray, anchors: Sequence[float],
                   size: Sequence[int], grid_box: Sequence[int], selector: Optional[np.ndarray] = None,
                   known: Optional[np.ndarray] = None, min_pixels: int = 12, source: str = "",
                   model: str = "rigid") -> MotionPlan:
    """Plan from a first-to-future displacement field on the Stage-1 grid.

    flow_px [T, 2, S, S]: displacement of every first-frame grid pixel at each anchor, in grid pixels.
    instance_masks [K, S, S]: first-frame instance masks on the same grid (empty slots are skipped).
    One transform per instance and anchor is fitted to the field inside the instance.  The default 'rigid'
    keeps the translation of the centroid and the rotation of the object (a mean translation alone would
    cancel a rotation) and drops scale and shear: a Stage-1 decoder can write part of an object's motion as a
    contraction of the frame about its centre, so its raw field also shrinks the object.  'affine' keeps
    that; `area_scale` records it.
    """
    flow_px, instance_masks = np.asarray(flow_px, np.float64), np.asarray(instance_masks).astype(bool)
    anchors = np.asarray(anchors, np.float64)
    horizons = flow_px.shape[0]
    if len(anchors) != horizons or np.any(np.diff(anchors) <= 0) or anchors[0] <= 0:
        raise ValueError(f"anchors must be {horizons} increasing positive seconds, got {anchors}")
    top, left, grid_h, grid_w = grid_box
    height, width = size
    scale = np.array([width / grid_w, height / grid_h])
    inside = np.zeros(instance_masks.shape[1:], bool)
    inside[top: top + grid_h, left: left + grid_w] = True
    objects = []
    for slot in range(instance_masks.shape[0]):
        grid = instance_masks[slot] & inside
        mask = work_mask(instance_masks[slot], grid_box, size)
        if not grid.any() or not mask.any():
            continue
        ys, xs = np.nonzero(grid)
        source_points = grid_to_work(np.stack((xs, ys), 1).astype(np.float64), grid_box, size)
        affines, residuals, areas = [], [], []
        for t in range(horizons):
            displacement = flow_px[t][:, ys, xs].T * scale                    # [N, 2] work px, from the same p0
            affine, residual = fit_affine(source_points, source_points + displacement, min_pixels, model)
            free = affine if model == "affine" else fit_affine(source_points, source_points + displacement, min_pixels, "affine")[0]
            affines.append(affine)
            residuals.append(residual)
            areas.append(float(np.linalg.det(free[:, :2])))
        flags = np.ones(horizons, bool) if known is None else np.asarray(known[slot], bool).copy()
        objects.append(ObjectPlan(
            slot=slot, mask=mask, affine=np.stack(affines), known=flags,
            selector=np.zeros(horizons) if selector is None else np.asarray(selector[slot], np.float64).copy(),
            residual_px=np.array(residuals), area_scale=np.array(areas), fill=np.where(flags, FILL_POINTS, FILL_HOLD).astype(np.int8)))
    return MotionPlan(sample_id, (int(height), int(width)), anchors, objects, f"{source}|{model}")


def from_stage1_export(export: dict, size: Sequence[int], native_hw: Optional[Sequence[int]] = None,
                       flow_key: str = "flow_ungated", model: str = "rigid") -> MotionPlan:
    """Gate-free plan from a Stage-1 export: every SAM2 instance is kept, with the decoder's own displacement."""
    flow = _np(export[flow_key])
    side = flow.shape[-1]
    geometry = export.get("geometry") or {}
    if "content_top" in geometry:
        grid_box = (int(geometry["content_top"]), int(geometry["content_left"]),
                    int(geometry["content_height"]), int(geometry["content_width"]))
    elif native_hw is not None:
        grid_box = inner_box(side, side, int(native_hw[0]), int(native_hw[1]))
    else:
        raise ValueError("the export has no geometry record: pass native_hw")
    return from_flow_grid(
        str(export.get("sample_id", "")), flow * side, _np(export["activity_input_instance_masks"]),
        [float(v) for v in export["horizon_seconds"]], size, grid_box,
        selector=_np(export["task_object_slot_probability"]) if "task_object_slot_probability" in export else None,
        source=f"stage1_export:{flow_key}", model=model)


def from_instance_affines(sample_id: str, masks: np.ndarray, affines: np.ndarray, anchors: Sequence[float],
                          known: Optional[np.ndarray] = None, selector: Optional[np.ndarray] = None,
                          source: str = "") -> MotionPlan:
    """Plan from masks [K, H, W] and work-pixel affines [K, T, 2, 3] that are already in the work frame."""
    masks, affines = np.asarray(masks).astype(bool), np.asarray(affines, np.float64)
    horizons = affines.shape[1]
    objects = [ObjectPlan(k, masks[k], affines[k].copy(),
                          np.ones(horizons, bool) if known is None else np.asarray(known[k], bool).copy(),
                          np.zeros(horizons) if selector is None else np.asarray(selector[k], np.float64).copy(),
                          np.zeros(horizons),
                          fill=np.where(np.ones(horizons, bool) if known is None else np.asarray(known[k], bool), FILL_POINTS, FILL_HOLD).astype(np.int8))
               for k in range(masks.shape[0]) if masks[k].any()]
    return MotionPlan(sample_id, tuple(int(v) for v in masks.shape[1:]), np.asarray(anchors, np.float64), objects, source)


def from_tracks(sample_id: str, masks: np.ndarray, slots: Sequence[int], tracks: np.ndarray, visible: np.ndarray,
                owner: np.ndarray, track_times: Sequence[float], anchors: Sequence[float], *, min_points: int = 6,
                max_residual_px: float = 6.0, model: str = "rigid", selector: Optional[np.ndarray] = None,
                time_tolerance: Optional[float] = None, few_points: int = 0, residual_frac: float = 0.0,
                source: str = "tracks") -> MotionPlan:
    """Plan measured on a real clip: one transform per object and anchor fitted to point tracks.

    masks [K, H, W] first-frame object masks (work frame) with slot ids `slots`; tracks [F, N, 2] (x, y) work px and
    visible [F, N] for N query points, owner [N] = slot id of each point (other values, e.g. -1 for background
    points, are ignored); track_times [F] seconds of the tracked frames (track_times[0] = 0).
    Every position comes from the same first-frame point: A(t) is fitted to p_n(0) -> p_n(t) over the points that
    are visible at both times.  An anchor is known only when at least `min_points` points are visible and the fit
    explains them to within `max_residual_px`; otherwise the last known transform is held and known is False (the
    plan then says nothing there, which is not "the object stands still").  No object is ever removed.
    few_points > 0: with at least that many but fewer than `min_points` visible points (a small or motion-blurred
    object) the transform is their mean translation, under the same residual test.
    residual_frac > 0: the fit may also miss by that fraction of the object's own displacement (an object that
    tumbles while it falls 120 px is not rigid in the image, yet where it went is known to a latent cell)."""
    masks, tracks, visible = np.asarray(masks).astype(bool), np.asarray(tracks, np.float64), np.asarray(visible).astype(bool)
    owner, track_times, anchors = np.asarray(owner), np.asarray(track_times, np.float64), np.asarray(anchors, np.float64)
    if len(track_times) != len(tracks) or abs(track_times[0]) > 1e-6:
        raise ValueError("track_times must give the seconds of every tracked frame, starting at 0")
    step = float(np.median(np.diff(track_times))) if len(track_times) > 1 else 0.0
    tolerance = 0.5 * step + 1e-6 if time_tolerance is None else float(time_tolerance)
    objects = []
    for index, slot in enumerate(slots):
        if not masks[index].any():
            continue
        chosen = owner == slot
        start = tracks[0, chosen]
        affines, known, residuals, fills, held = [], [], [], [], IDENTITY.copy()
        for when in anchors:
            frame = int(np.abs(track_times - when).argmin())
            ok = visible[frame, chosen] & visible[0, chosen]
            good = False
            enough = int(ok.sum()) >= min_points or (few_points > 0 and int(ok.sum()) >= few_points)
            if abs(track_times[frame] - when) <= tolerance and enough:
                affine, residual = fit_affine(start[ok], tracks[frame, chosen][ok], min_points, model)
                travelled = float(np.linalg.norm((tracks[frame, chosen][ok] - start[ok]).mean(0)))
                good = residual <= max(max_residual_px, residual_frac * travelled)
            if good:
                held = affine
            affines.append(held.copy())
            known.append(good)
            residuals.append(residual if good else 0.0)
            fills.append((FILL_POINTS if int(ok.sum()) >= min_points else FILL_FEW) if good else FILL_HOLD)
        objects.append(ObjectPlan(int(slot), masks[index], np.stack(affines), np.array(known, bool),
                                  np.zeros(len(anchors)) if selector is None else np.asarray(selector[index], np.float64).copy(),
                                  np.array(residuals), fill=np.array(fills, np.int8)))
    return MotionPlan(sample_id, tuple(int(v) for v in masks.shape[1:]), anchors, objects, f"{source}|{model}")


def synthetic_translation(sample_id: str, mask: np.ndarray, anchors: Sequence[float], direction_deg: float,
                          magnitude_px: float, slot: int = 0) -> MotionPlan:
    """One object moving at constant velocity: `magnitude_px` along `direction_deg` (image axes: 0 = right,
    90 = down) by the last anchor.  Used for the direction-following test."""
    anchors = np.asarray(anchors, np.float64)
    unit = np.array([np.cos(np.radians(direction_deg)), np.sin(np.radians(direction_deg))])
    affines = np.repeat(IDENTITY[None], len(anchors), 0)
    affines[:, :, 2] = (anchors / anchors[-1])[:, None] * magnitude_px * unit
    obj = ObjectPlan(slot, np.asarray(mask).astype(bool), affines, np.ones(len(anchors), bool),
                     np.ones(len(anchors)), np.zeros(len(anchors)), fill=np.zeros(len(anchors), np.int8))
    return MotionPlan(sample_id, tuple(int(v) for v in mask.shape), anchors, [obj], f"synthetic:{direction_deg:g}deg:{magnitude_px:g}px")


# ---------------------------------------------------------------------------------------------- time axis
def interpolate_affine(first: np.ndarray, second: np.ndarray, weight: float, centre: np.ndarray) -> np.ndarray:
    """The transform `weight` of the way from `first` to `second` (both 2x3, first-frame -> t), as a rigid motion of
    the object: the rotation angle along the shorter arc, the scale on a log scale, the stretch and the CENTROID
    displacement linearly.  Linear blending of the matrices is not used: half way between the identity and a 180
    degree rotation it has determinant 0 and sends every point of the object to the centroid.  Pure translations
    interpolate exactly as before."""
    a0, s0, k0 = _polar(first[:, :2])
    a1, s1, k1 = _polar(second[:, :2])
    delta = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    if delta <= -math.pi + 1e-12:                                          # a half turn: both arcs are as short; take the positive one
        delta = math.pi
    linear = _unpolar(a0 + weight * delta, (1.0 - weight) * s0 + weight * s1, (1.0 - weight) * k0 + weight * k1)
    d0 = apply_affine(first, centre[None])[0] - centre
    d1 = apply_affine(second, centre[None])[0] - centre
    displacement = (1.0 - weight) * d0 + weight * d1
    return np.concatenate((linear, (centre + displacement - linear @ centre)[:, None]), 1)


def affine_at(obj: ObjectPlan, anchors: np.ndarray, when: float) -> tuple[np.ndarray, bool]:
    """(A, known) at `when` seconds.  Identity at t = 0, rigid (SE(2) + log-scale) interpolation between anchors
    about the object's centroid, held (and unknown) after the last known anchor; a segment is known only when both of
    its ends are."""
    times = np.concatenate(([0.0], anchors))
    series = np.concatenate((IDENTITY[None], obj.affine))
    flags = np.concatenate(([True], obj.known))
    if when <= 0:
        return IDENTITY.copy(), True
    last_known = int(np.nonzero(flags)[0].max())
    if when >= times[last_known] - 1e-9:
        return series[last_known].copy(), bool(abs(when - times[last_known]) <= 1e-6)
    upper = int(np.searchsorted(times, when, side="left"))
    lower = upper - 1
    weight = (when - times[lower]) / (times[upper] - times[lower])
    if np.array_equal(series[lower][:, :2], IDENTITY[:, :2]) and np.array_equal(series[upper][:, :2], IDENTITY[:, :2]):
        return (1.0 - weight) * series[lower] + weight * series[upper], bool(flags[lower] and flags[upper])   # pure translations: bit-identical to the old blend
    return interpolate_affine(series[lower], series[upper], float(weight), obj.centroid), bool(flags[lower] and flags[upper])


def interp_hold(values: np.ndarray, anchors: Sequence[float], when: float) -> np.ndarray:
    """A cumulative first-to-t quantity (e.g. a displacement field [T, ...] given at the anchors) at `when` seconds:
    zero at t = 0, linear between anchors, HELD after the last anchor (nothing is extrapolated)."""
    values, anchors = np.asarray(values), np.asarray(anchors, np.float64)
    if when <= 0:
        return np.zeros_like(values[0])
    if when >= anchors[-1]:
        return values[-1].copy()
    upper = int(np.searchsorted(anchors, when, side="left"))
    if upper == 0:
        return values[0] * (when / anchors[0])
    weight = (when - anchors[upper - 1]) / (anchors[upper] - anchors[upper - 1])
    return (1.0 - weight) * values[upper - 1] + weight * values[upper]


# --------------------------------------------------------------------------------------------- transforms
def _with_centroid_motion(obj: ObjectPlan, change) -> ObjectPlan:
    """Rewrite the centroid displacement of an object with `change(displacement [2])`; its own linear part
    (rotation / scale about itself) is left as it is."""
    c0 = obj.centroid
    affines = obj.affine.copy()
    for t in range(len(affines)):
        linear = affines[t][:, :2]
        displacement = apply_affine(affines[t], c0) - c0
        affines[t][:, 2] = c0 + change(displacement) - linear @ c0
    return replace(obj, affine=affines)


def rotated(plan: MotionPlan, degrees: float, slots: Optional[Iterable[int]] = None) -> MotionPlan:
    """Same objects and magnitudes, every centroid displacement turned by `degrees` (image axes, clockwise
    on screen for positive angles)."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    matrix = np.array([[c, -s], [s, c]])
    chosen = None if slots is None else set(slots)
    objects = [_with_centroid_motion(o, lambda d: matrix @ d) if chosen is None or o.slot in chosen else o for o in plan.objects]
    return replace(plan, objects=objects, source=f"{plan.source}|rot{degrees:g}")


def scaled(plan: MotionPlan, factor: float) -> MotionPlan:
    return replace(plan, objects=[_with_centroid_motion(o, lambda d: factor * d) for o in plan.objects],
                   source=f"{plan.source}|scale{factor:g}")


def zeroed(plan: MotionPlan) -> MotionPlan:
    """Effective zero motion: the same objects, explicitly asked to stay where they are (known stays True)."""
    objects = [replace(o, affine=np.repeat(IDENTITY[None], len(o.affine), 0), known=np.ones(len(o.known), bool)) for o in plan.objects]
    return replace(plan, objects=objects, source=f"{plan.source}|zero")


def unknown(plan: MotionPlan) -> MotionPlan:
    """No motion information: the objects are listed, nothing is said about where they go."""
    return replace(plan, objects=[replace(o, known=np.zeros(len(o.known), bool)) for o in plan.objects],
                   source=f"{plan.source}|unknown")


_FILL_RANK = {FILL_POINTS: 0, FILL_FEW: 1, FILL_MASK: 2, FILL_UNKNOWN: 3, FILL_HOLD: 4}


def provenance_at(obj: ObjectPlan, anchors: np.ndarray, when: float) -> int:
    """Provenance of affine_at(obj, anchors, when): FILL_POINTS at t <= 0, the weaker of the two anchors of the segment
    (the rule of relabel_v2.point_readings), FILL_HOLD once the transform is held."""
    if when <= 0:
        return FILL_POINTS
    if not affine_at(obj, anchors, when)[1]:
        return FILL_HOLD
    times = np.concatenate(([0.0], anchors))
    provenance = np.concatenate(([FILL_POINTS], obj.provenance()))
    upper = min(int(np.searchsorted(times, when, side="left")), len(times) - 1)
    return int(max((provenance[max(upper - 1, 0)], provenance[upper]), key=lambda p: _FILL_RANK[int(p)]))


def time_shifted(plan: MotionPlan, delta_s: float, slots: Optional[Iterable[int]] = None) -> MotionPlan:
    """The same motion `delta_s` seconds later (earlier when negative): the transform at anchor t is the plan's at
    t - delta_s (identity before the motion begins, held and unknown after its last known anchor), `known` and `fill`
    carried from that time; only the objects in `slots` (default: all)."""
    chosen = None if slots is None else set(slots)
    objects = []
    for obj in plan.objects:
        if chosen is not None and obj.slot not in chosen:
            objects.append(obj)
            continue
        affine, known, fill = obj.affine.copy(), obj.known.copy(), obj.provenance().copy()
        for t, when in enumerate(plan.anchors):
            affine[t], known[t] = affine_at(obj, plan.anchors, float(when) - delta_s)
            fill[t] = provenance_at(obj, plan.anchors, float(when) - delta_s)
        objects.append(replace(obj, affine=affine, known=known, fill=fill))
    return replace(plan, objects=objects, source=f"{plan.source}|shift{delta_s:+g}")


def select_objects(plan: MotionPlan, policy: str = "all", min_move_px: float = 0.0) -> list[ObjectPlan]:
    """Which objects a control route constrains.  'all': every object (static ones are pinned);
    'movers': only objects whose plan moves some point by at least `min_move_px` (the rest are left free,
    which removes "stay still" constraints but never removes motion)."""
    if policy == "all":
        return list(plan.objects)
    if policy == "movers":
        return [o for o in plan.objects if o.max_point_motion() >= min_move_px]
    raise ValueError(f"unknown policy {policy!r}")


# ------------------------------------------------------------------------------------------------- tracks
def sample_points(mask: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """`count` points [count, 2] (x, y) spread over the mask: k-means centres snapped to mask pixels; the
    first one is the pixel nearest to the centroid."""
    ys, xs = np.nonzero(mask)
    pixels = np.stack((xs, ys), 1).astype(np.float64)
    centre = pixels[np.linalg.norm(pixels - pixels.mean(0), axis=1).argmin()]
    if count <= 1 or len(pixels) <= 1:
        return centre[None]
    count = min(count, len(pixels))
    chosen = [centre]
    distance = np.linalg.norm(pixels - centre, axis=1)
    for _ in range(count - 1):                                    # farthest-point initialisation (deterministic)
        chosen.append(pixels[distance.argmax()])
        distance = np.minimum(distance, np.linalg.norm(pixels - chosen[-1], axis=1))
    centres = np.stack(chosen)
    for _ in range(10):
        owner = np.linalg.norm(pixels[:, None] - centres[None], axis=2).argmin(1)
        centres = np.stack([pixels[owner == k].mean(0) if np.any(owner == k) else centres[k] for k in range(count)])
    snapped = np.stack([pixels[np.linalg.norm(pixels - c, axis=1).argmin()] for c in centres])
    snapped[0] = centre
    return snapped


def to_tracks(plan: MotionPlan, times: Sequence[float], points_per_object: int = 8,
              objects: Optional[Sequence[ObjectPlan]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tracks [F, N, 2] float32 (x, y) work px, visibility [F, N] bool, owner [N] slot ids).

    Each track is p(t) = A(t) p0 from its own first-frame point.  visibility is False where the plan says
    nothing (so "no trajectory" stays different from "visible and static"); frame 0 is always visible."""
    objects = plan.objects if objects is None else list(objects)
    tracks, visible, owner = [], [], []
    for obj in objects:
        points = sample_points(obj.mask, points_per_object)
        path, flags = [], []
        for when in times:
            affine, known = affine_at(obj, plan.anchors, float(when))
            path.append(apply_affine(affine, points))
            flags.append(known)
        tracks.append(np.stack(path))                               # [F, P, 2]
        visible.append(np.repeat(np.array(flags)[:, None], len(points), 1))
        owner += [obj.slot] * len(points)
    if not tracks:
        return np.zeros((len(times), 0, 2), np.float32), np.zeros((len(times), 0), bool), np.zeros(0, int)
    return np.concatenate(tracks, 1).astype(np.float32), np.concatenate(visible, 1), np.array(owner)


def swept_area(plan: MotionPlan, times: Sequence[float], objects: Optional[Sequence[ObjectPlan]] = None) -> np.ndarray:
    """[H, W] bool: every pixel a controlled object covers at any of `times` (its whole path)."""
    objects = plan.objects if objects is None else list(objects)
    covered = np.zeros(plan.size, bool)
    for obj in objects:
        for when in times:
            covered |= target_mask(obj, affine_at(obj, plan.anchors, float(when))[0])
    return covered


def background_anchors(plan: MotionPlan, times: Sequence[float], rows: int = 5, cols: int = 8, margin_px: int = 12,
                       avoid: Optional[Sequence[np.ndarray]] = None) -> tuple[np.ndarray, np.ndarray]:
    """Static tracks on the background: (tracks [F, N, 2], visibility [F, N], all True).

    A point-track model that only sees tracks on one object may realise them as a camera pan (the whole
    scene shifts with the object).  In a static shot the background does not move, and saying so with a
    sparse grid of zero-displacement tracks removes that ambiguity.  Anchors stay clear of the whole path
    of the plan's objects and of every mask in `avoid` (the other first-frame objects, which are left free
    rather than pinned)."""
    height, width = plan.size
    blocked = swept_area(plan, times)
    for obj in plan.objects:
        blocked |= obj.mask
    for mask in avoid or []:
        blocked |= np.asarray(mask).astype(bool)
    kernel = np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8)
    blocked = cv2.dilate(blocked.astype(np.uint8), kernel) > 0
    ys = np.linspace(0.08 * height, 0.92 * height, rows)
    xs = np.linspace(0.06 * width, 0.94 * width, cols)
    grid = np.array([[x, y] for y in ys for x in xs])
    keep = ~blocked[np.rint(grid[:, 1]).astype(int), np.rint(grid[:, 0]).astype(int)]
    points = grid[keep].astype(np.float32)
    return np.repeat(points[None], len(times), 0), np.ones((len(times), len(points)), bool)


# ------------------------------------------------------------------------------------------- crude video
def target_mask(obj: ObjectPlan, affine: np.ndarray) -> np.ndarray:
    """Mt = A o M0: the first-frame mask moved by the affine, [H, W] bool."""
    height, width = obj.mask.shape
    return cv2.warpAffine(obj.mask.astype(np.uint8), affine.astype(np.float64), (width, height), flags=cv2.INTER_NEAREST) > 0


def to_crude_video(plan: MotionPlan, first_frame: np.ndarray, times: Sequence[float],
                   objects: Optional[Sequence[ObjectPlan]] = None, dilate_px: int = 2,
                   inpaint_radius: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Cut-and-drag animation and its control mask: (frames [F, H, W, 3] uint8, masks [F, H, W] uint8 0/255).

    Every controlled object is cut out of the first frame and moved by its own affine; the area it leaves
    is inpainted.  The control mask is the union of the moved masks at frames where the plan is known; at
    unknown frames the cut-out stays at its last known position and the mask is empty for that object."""
    objects = plan.objects if objects is None else list(objects)
    height, width = plan.size
    if first_frame.shape[:2] != (height, width):
        raise ValueError(f"first frame {first_frame.shape[:2]} is not the work frame {plan.size}")
    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    cutouts = []
    for obj in objects:
        cut = cv2.dilate(obj.mask.astype(np.uint8), kernel) if dilate_px > 0 else obj.mask.astype(np.uint8)
        cutouts.append((obj, cut, obj.max_point_motion()))
    hole = np.zeros((height, width), np.uint8)
    for obj, cut, motion in cutouts:
        if motion > 0.5:
            hole |= cut
    background = cv2.inpaint(first_frame, hole * 255, inpaint_radius, cv2.INPAINT_TELEA) if hole.any() else first_frame.copy()
    order = sorted(cutouts, key=lambda item: item[2])                # small motion first: movers end on top
    frames, masks = [], []
    for when in times:
        canvas = background.copy()
        control = np.zeros((height, width), np.uint8)
        for obj, cut, motion in order:
            affine, known = affine_at(obj, plan.anchors, float(when))
            if motion > 0.5:
                moved_rgb = cv2.warpAffine(first_frame, affine, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                moved_cut = cv2.warpAffine(cut, affine, (width, height), flags=cv2.INTER_NEAREST) > 0
                canvas[moved_cut] = moved_rgb[moved_cut]
            else:
                moved_cut = cut > 0
            if known:
                control[moved_cut] = 255
        frames.append(canvas)
        masks.append(control)
    return np.stack(frames), np.stack(masks)


# -------------------------------------------------------------------------------------------- dense target
def to_dense_target(plan: MotionPlan, times: Sequence[float], objects: Optional[Sequence[ObjectPlan]] = None) -> dict:
    """What cache_stage2_oracle.make_motion_condition reads, built from the plan with one transform per object:

      flow         [F-1, 2, H, W]  first-to-frame displacement at SOURCE pixels (zero outside the objects)
      visibility   [F-1, 1, H, W]  visibility of the SOURCE pixels (ones: the plan does not hide anything)
      object_mask  [1, H, W]       M0, union of the source masks
      future_mask  [F-1, 1, H, W]  Mt, union of the moved masks
      known        [F-1]           whether any object is known at that frame

    times[0] must be 0 (the conditioning frame)."""
    if abs(float(times[0])) > 1e-9:
        raise ValueError("times[0] must be the conditioning frame (0 s)")
    objects = plan.objects if objects is None else list(objects)
    height, width = plan.size
    grid = np.stack(np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)), -1)   # [H, W, 2]
    flow = np.zeros((len(times) - 1, 2, height, width), np.float32)
    future = np.zeros((len(times) - 1, 1, height, width), np.float32)
    known_any = np.zeros(len(times) - 1, bool)
    union = np.zeros((height, width), bool)
    for obj in objects:
        union |= obj.mask
        points = grid[obj.mask]
        for index, when in enumerate(times[1:]):
            affine, known = affine_at(obj, plan.anchors, float(when))
            flow[index][:, obj.mask] = (apply_affine(affine, points) - points).T
            future[index, 0][target_mask(obj, affine)] = 1.0
            known_any[index] |= known
    return {"flow": flow, "visibility": np.ones((len(times) - 1, 1, height, width), np.float32),
            "object_mask": union[None].astype(np.float32), "future_mask": future, "known": known_any}


# ------------------------------------------------------------------------------------------------------ io
def save_plan(path, plan: MotionPlan) -> None:
    meta = {"sample_id": plan.sample_id, "size": list(plan.size), "source": plan.source, "slots": [o.slot for o in plan.objects]}
    np.savez_compressed(
        path, meta=json.dumps(meta), anchors=plan.anchors,
        masks=np.packbits(np.stack([o.mask for o in plan.objects]).reshape(-1)) if plan.objects else np.zeros(0, np.uint8),
        affine=np.stack([o.affine for o in plan.objects]) if plan.objects else np.zeros((0, len(plan.anchors), 2, 3)),
        known=np.stack([o.known for o in plan.objects]) if plan.objects else np.zeros((0, len(plan.anchors)), bool),
        selector=np.stack([o.selector for o in plan.objects]) if plan.objects else np.zeros((0, len(plan.anchors))),
        residual=np.stack([o.residual_px if len(o.residual_px) else np.zeros(len(plan.anchors)) for o in plan.objects]) if plan.objects else np.zeros((0, len(plan.anchors))),
        area_scale=np.stack([o.area_scale if len(o.area_scale) else np.ones(len(plan.anchors)) for o in plan.objects]) if plan.objects else np.zeros((0, len(plan.anchors))),
        fill=np.stack([o.provenance() for o in plan.objects]).astype(np.int8) if plan.objects else np.zeros((0, len(plan.anchors)), np.int8))


def load_plan(path) -> MotionPlan:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    height, width = meta["size"]
    count = len(meta["slots"])
    masks = np.unpackbits(data["masks"])[: count * height * width].reshape(count, height, width).astype(bool) if count else np.zeros((0, height, width), bool)
    area = data["area_scale"] if "area_scale" in data.files else np.ones_like(data["residual"])
    fill = data["fill"] if "fill" in data.files else np.full(data["known"].shape, FILL_UNKNOWN, np.int8)     # older plans: provenance unknown
    objects = [ObjectPlan(int(slot), masks[i], data["affine"][i], data["known"][i], data["selector"][i], data["residual"][i], area[i], fill[i])
               for i, slot in enumerate(meta["slots"])]
    return MotionPlan(meta["sample_id"], (height, width), data["anchors"], objects, meta["source"])


def describe(plan: MotionPlan) -> list[dict]:
    """Per-object summary for logs and audits."""
    rows = []
    for obj in plan.objects:
        displacement = obj.displacement()
        rows.append({"slot": obj.slot, "area_px": int(obj.mask.sum()), "centroid": [round(float(v), 1) for v in obj.centroid],
                     "disp_last_px": [round(float(v), 1) for v in displacement[-1]],
                     "disp_max_px": round(float(np.linalg.norm(displacement, axis=1).max()), 1),
                     "selector_max": round(float(obj.selector.max()), 3), "known": int(obj.known.sum()),
                     "fit_residual_px": round(float(obj.residual_px.max()), 2) if len(obj.residual_px) else 0.0,
                     "field_area_scale_last": round(float(obj.area_scale[-1]), 3) if len(obj.area_scale) else 1.0})
    return rows


def write_mp4(path, frames: np.ndarray, fps: float, crf: int = 10) -> None:
    """[F, H, W, 3] or [F, H, W] uint8 -> H.264 mp4 (yuv420p, near-lossless)."""
    import imageio.v2 as imageio
    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, -1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1, pixelformat="yuv420p",
                            ffmpeg_params=["-crf", str(crf)]) as writer:
        for frame in frames:
            writer.append_data(frame)
