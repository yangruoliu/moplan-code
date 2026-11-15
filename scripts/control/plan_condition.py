#!/usr/bin/env python
"""Motion plan -> Stage-2 condition (contract v2), and the training labels that come with a measured plan.

One place renders a plan for the generator, so the deployed cache, the training cache and anything that edits a plan
(the reader policy) produce the same tensor for the same plan:

  render_base      the 20 channels of the deployed adapter (scripts.cache_stage2_oracle.make_motion_condition)
  render_v2        + the seven extra channels of motion_jepa.cond_v2
  labels_from_tracks / labels_from_plan   per object and latent frame: displacement, valid, moving
  tube_weights     flow-matching weights: moving-object tubes and static objects count more than the background
  clip_window      how many frames of a short real clip are used, and whether a scene at rest may be extended

Times are seconds of the latent frames (t_j = 4 j / fps, t_0 = 0).  The work frame is the generator's H x W.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from motion_jepa import cond_v2  # noqa: E402
from scripts import motion_plan as mp  # noqa: E402
from scripts.cache_stage2_oracle import make_motion_condition  # noqa: E402


def latent_times(frames: int, fps: float) -> list[float]:
    """Seconds of the latent frames of an n = 4k+1 frame clip."""
    if (frames - 1) % 4:
        raise ValueError(f"{frames} frames is not 4k+1")
    return [4.0 * j / fps for j in range((frames - 1) // 4 + 1)]


# ------------------------------------------------------------------------------------------- base channels
def render_base(plan: Optional[mp.MotionPlan], image_latent: torch.Tensor, times: Sequence[float], size: Sequence[int]) -> torch.Tensor:
    """[20, T, h, w]: the deployed adapter's input for this plan (None or no objects = the empty condition)."""
    height, width = size
    steps = len(times) - 1
    if image_latent.shape[1] != len(times):
        raise ValueError(f"image latent has {image_latent.shape[1]} frames, the plan is rendered at {len(times)}")
    if plan is None or not plan.objects:
        target = {"flow": torch.zeros(steps, 2, height, width), "visibility": torch.ones(steps, 1, height, width),
                  "object_mask": torch.zeros(1, height, width), "future_mask": torch.zeros(steps, 1, height, width)}
    else:
        dense = mp.to_dense_target(plan, times)
        target = {key: torch.from_numpy(dense[key]) for key in ("flow", "visibility", "object_mask", "future_mask")}
    return make_motion_condition(image_latent.float(), target, source_height=height, source_width=width,
                                 selected_indices=list(range(len(times))), future_indices=list(range(1, len(times))))


# ------------------------------------------------------------------------------------- Stage-1 dense fields
def _grid_box(export: dict, native_hw: Optional[Sequence[int]]) -> tuple[int, int, int, int]:
    geometry = export.get("geometry") or {}
    side = int(export["flow_ungated"].shape[-1])
    if "content_top" in geometry:
        return (int(geometry["content_top"]), int(geometry["content_left"]), int(geometry["content_height"]), int(geometry["content_width"]))
    if native_hw is None:
        raise ValueError("the export has no geometry record: pass native_hw")
    return mp.inner_box(side, side, int(native_hw[0]), int(native_hw[1]))


def stage1_dense(export: dict, times: Sequence[float], size: Sequence[int], native_hw: Optional[Sequence[int]] = None) -> tuple[np.ndarray, np.ndarray]:
    """(raw_flow [T, 2, H, W] work px at source pixels, motion_prob [T, H, W]) of a Stage-1 export at the latent times.

    The decoder's own field (`flow_ungated`, no fit, no gate): zero at t = 0, linear between its horizons, held after
    the last one.  The motion probability is taken at the nearest horizon (zero at t = 0)."""
    height, width = size
    top, left, grid_h, grid_w = _grid_box(export, native_hw)
    side = int(export["flow_ungated"].shape[-1])
    field = mp._np(export["flow_ungated"]).astype(np.float64) * side                   # [7, 2, S, S] grid px
    horizons = np.array([float(v) for v in export["horizon_seconds"]])
    prob = mp._np(export["motion_mask"]).astype(np.float64)[:, 0] if "motion_mask" in export else None   # [7, S, S]
    flows, probs = [], []
    for when in times:
        value = mp.interp_hold(field, horizons, float(when))[:, top: top + grid_h, left: left + grid_w]
        flow = np.stack([cv2.resize(value[0], (width, height), interpolation=cv2.INTER_LINEAR) * (width / grid_w),
                         cv2.resize(value[1], (width, height), interpolation=cv2.INTER_LINEAR) * (height / grid_h)])
        flows.append(flow)
        if prob is None or when <= 0:
            probs.append(np.zeros((height, width)))
        else:
            nearest = prob[int(np.abs(horizons - when).argmin())][top: top + grid_h, left: left + grid_w]
            probs.append(cv2.resize(nearest, (width, height), interpolation=cv2.INTER_LINEAR))
    return np.stack(flows).astype(np.float32), np.stack(probs).astype(np.float32)


def plan_source_flow(plan: Optional[mp.MotionPlan], times: Sequence[float], size: Sequence[int]) -> np.ndarray:
    """[T, 2, H, W] the plan's own first-to-t displacement at source pixels (zero at t = 0 and outside the objects)."""
    height, width = size
    out = np.zeros((len(times), 2, height, width), np.float32)
    if plan is not None and plan.objects:
        out[1:] = mp.to_dense_target(plan, times)["flow"]
    return out


# ------------------------------------------------------------------------------------------ extra channels
def _area(mask: np.ndarray, latent_hw: Sequence[int]) -> torch.Tensor:
    """[..., H, W] float array -> area-resampled [..., h, w] tensor."""
    tensor = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))
    lead = tensor.shape[:-2]
    return F.interpolate(tensor.reshape(1, -1, *tensor.shape[-2:]), size=tuple(latent_hw), mode="area").reshape(*lead, *latent_hw)


def build_extra(plan: Optional[mp.MotionPlan], times: Sequence[float], size: Sequence[int], latent_hw: Sequence[int], *,
                availability: bool, raw_flow: Optional[np.ndarray] = None, motion_prob: Optional[np.ndarray] = None) -> torch.Tensor:
    """[7, T, h, w]: known, availability, selector, raw_flow x/y, motion_prob, source_mask (motion_jepa.cond_v2)."""
    height, width = size
    frames = len(times)
    lh, lw = latent_hw
    extra = torch.zeros(cond_v2.EXTRA_CHANNELS, frames, lh, lw)
    if not availability:
        return extra
    known = np.zeros((frames, height, width), np.float32)
    weight = np.zeros((frames, height, width), np.float32)
    selector = np.zeros((frames, height, width), np.float32)
    source = np.zeros((height, width), bool)
    for obj in (plan.objects if plan is not None else []):
        source |= obj.mask
        for index, when in enumerate(times):
            affine, is_known = mp.affine_at(obj, plan.anchors, float(when))
            footprint = obj.mask if index == 0 else mp.target_mask(obj, affine)
            if is_known:
                known[index][footprint] = 1.0
            nearest = int(np.abs(plan.anchors - when).argmin())
            weight[index] += footprint
            selector[index] += footprint * float(obj.selector[nearest])
    extra[cond_v2.EXTRA_NAMES.index("known")] = _area(known, latent_hw).clamp(0, 1)
    extra[cond_v2.EXTRA_NAMES.index("availability")] = 1.0
    extra[cond_v2.EXTRA_NAMES.index("selector")] = _area(selector, latent_hw) / _area(weight, latent_hw).clamp_min(1e-6) * (_area(weight, latent_hw) > 0)
    if raw_flow is not None:
        flow = F.interpolate(torch.from_numpy(np.ascontiguousarray(raw_flow, dtype=np.float32)), size=(lh, lw), mode="bilinear", align_corners=False)
        extra[cond_v2.EXTRA_NAMES.index("raw_flow_x")] = flow[:, 0] / width
        extra[cond_v2.EXTRA_NAMES.index("raw_flow_y")] = flow[:, 1] / height
    if motion_prob is not None:
        extra[cond_v2.EXTRA_NAMES.index("motion_prob")] = _area(motion_prob, latent_hw).clamp(0, 1)
    extra[cond_v2.EXTRA_NAMES.index("source_mask")] = _area(source.astype(np.float32), latent_hw).clamp(0, 1)[None].expand(frames, -1, -1)
    return extra


def render_v2(kind: str, plan: Optional[mp.MotionPlan], image_latent: torch.Tensor, times: Sequence[float], size: Sequence[int], *,
              export: Optional[dict] = None, native_hw: Optional[Sequence[int]] = None) -> torch.Tensor:
    """[27, T, h, w] for one kind of condition.

    real   `plan` = the measured plan; raw_flow is the plan's own source flow
    pred   `plan` = the Stage-1 plan and `export` its export; raw_flow / motion_prob are the decoder's dense outputs
    zero   `plan` = any plan with the wanted objects; they are rendered at rest
    null   no hint"""
    if kind not in cond_v2.KINDS:
        raise ValueError(f"unknown condition kind {kind!r}")
    latent_hw = tuple(image_latent.shape[-2:])
    if kind == "null":
        return cond_v2.null_condition(image_latent.float())
    if kind == "zero":
        plan = mp.zeroed(plan)
        raw_flow, prob = None, None
    elif kind == "pred":
        if export is None:
            raise ValueError("the pred condition needs the Stage-1 export")
        raw_flow, prob = stage1_dense(export, times, size, native_hw)
    else:
        raw_flow, prob = plan_source_flow(plan, times, size), None
    base = render_base(plan, image_latent, times, size)
    extra = build_extra(plan, times, size, latent_hw, availability=True, raw_flow=raw_flow, motion_prob=prob)
    return torch.cat((base, extra.to(base.dtype)), 0)


def render_reader(plan: mp.MotionPlan, edited: mp.MotionPlan, use: np.ndarray, scale: np.ndarray, turn: np.ndarray, image_latent: torch.Tensor,
                  times: Sequence[float], size: Sequence[int], *, export: dict, native_hw: Optional[Sequence[int]] = None) -> torch.Tensor:
    """[27, T, h, w] pred condition after the reader policy: the rendered channels come from the edited plan, and
    Stage-1's dense outputs are edited the same way inside each object's first-frame mask (a dropped object's raw
    flow and motion probability are removed, a scaled / turned object's raw flow is scaled / turned), so the two
    halves of the condition never contradict each other.  `plan` is Stage-1's own plan (its masks), `edited` the
    result of reader_policy.apply_action(plan, use, scale, turn)."""
    latent_hw = tuple(image_latent.shape[-2:])
    raw_flow, prob = stage1_dense(export, times, size, native_hw)
    for k, obj in enumerate(plan.objects):
        if not bool(use[k]):
            raw_flow[:, :, obj.mask] = 0.0
            prob[:, obj.mask] = 0.0
            continue
        factor, angle = 2.0 ** float(scale[k]), float(turn[k])
        if factor != 1.0 or angle != 0.0:
            c, sn = np.cos(angle), np.sin(angle)
            x, y = raw_flow[:, 0][:, obj.mask].copy(), raw_flow[:, 1][:, obj.mask].copy()
            raw_flow[:, 0][:, obj.mask] = factor * (c * x - sn * y)
            raw_flow[:, 1][:, obj.mask] = factor * (sn * x + c * y)
    base = render_base(edited, image_latent, times, size)
    extra = build_extra(edited, times, size, latent_hw, availability=True, raw_flow=raw_flow, motion_prob=prob)
    return torch.cat((base, extra.to(base.dtype)), 0)


# -------------------------------------------------------------------------------------------------- labels
def _classify(disp: np.ndarray, valid: np.ndarray, move_px: float, still_px: float) -> tuple[np.ndarray, np.ndarray]:
    size = np.linalg.norm(disp, axis=-1)
    return valid & (size >= move_px), valid & (size <= still_px)


def labels_from_tracks(plan: mp.MotionPlan, tracks: np.ndarray, visible: np.ndarray, owner: np.ndarray, track_times: np.ndarray,
                       times: Sequence[float], *, move_px: float = 8.0, still_px: float = 4.0, min_visible: int = 3) -> dict:
    """Measured motion per plan object and latent frame: disp [K, T, 2] (median over the points visible at both
    times, work px), valid [K, T], moving [K, T] (|disp| >= move_px), static [K, T] (|disp| <= still_px; the band in
    between is neither: tracker jitter and slow drift are not called motion), slots [K]."""
    slots = [o.slot for o in plan.objects]
    disp = np.zeros((len(slots), len(times), 2), np.float32)
    valid = np.zeros((len(slots), len(times)), bool)
    step = float(np.median(np.diff(track_times))) if len(track_times) > 1 else 0.0
    for k, slot in enumerate(slots):
        chosen = owner == slot
        for j, when in enumerate(times):
            frame = int(np.abs(track_times - when).argmin())
            if abs(float(track_times[frame]) - when) > 0.5 * step + 1e-6:
                continue
            ok = visible[frame, chosen] & visible[0, chosen]
            if int(ok.sum()) >= min_visible:
                disp[k, j] = np.median((tracks[frame, chosen] - tracks[0, chosen])[ok], 0)
                valid[k, j] = True
    moving, static = _classify(disp, valid, move_px, still_px)
    return {"disp": disp, "valid": valid, "moving": moving, "static": static, "slots": np.array(slots, np.int16)}


def labels_with_masks(labels: dict, object_disp: np.ndarray, object_ok: np.ndarray, object_times: np.ndarray, times: Sequence[float], *,
                      move_px: float = 8.0, still_px: float = 4.0) -> dict:
    """The same labels with every object's displacement taken from its mask centroid where the mask tracker has it
    (object_disp [F, K, 2], object_ok [F, K] at object_times [F]; K in the order of labels["slots"]); the point reading
    is kept where the mask is lost.  "source" [K, T]: 2 = mask, 1 = points, 0 = none."""
    disp, valid = labels["disp"].copy(), labels["valid"].copy()
    source = valid.astype(np.int8)
    object_times = np.asarray(object_times, np.float64)
    gap = float(np.median(np.diff(object_times))) if len(object_times) > 1 else 0.0
    for j, when in enumerate(times):
        frame = int(np.abs(object_times - when).argmin())
        if abs(float(object_times[frame]) - when) > 0.5 * gap + 1e-6:
            continue
        ok = np.asarray(object_ok[frame]).astype(bool)
        disp[ok, j], valid[ok, j], source[ok, j] = object_disp[frame][ok], True, 2
    disp[:, 0], valid[:, 0] = 0.0, True
    moving, static = _classify(disp, valid, move_px, still_px)
    return {"disp": disp.astype(np.float32), "valid": valid, "moving": moving, "static": static, "slots": labels["slots"], "source": source}


def labels_from_plan(plan: mp.MotionPlan, times: Sequence[float], *, move_px: float = 8.0, still_px: float = 4.0) -> dict:
    """The same labels read from a ground-truth plan (simulation): centroid displacement, valid where the plan is known."""
    slots = [o.slot for o in plan.objects]
    disp = np.zeros((len(slots), len(times), 2), np.float32)
    valid = np.zeros((len(slots), len(times)), bool)
    for k, obj in enumerate(plan.objects):
        centre = obj.centroid
        for j, when in enumerate(times):
            affine, known = mp.affine_at(obj, plan.anchors, float(when))
            disp[k, j] = mp.apply_affine(affine, centre) - centre
            valid[k, j] = known
    moving, static = _classify(disp, valid, move_px, still_px)
    return {"disp": disp, "valid": valid, "moving": moving, "static": static, "slots": np.array(slots, np.int16)}


def object_masks_latent(plan: mp.MotionPlan, latent_hw: Sequence[int]) -> torch.Tensor:
    """[K, h, w] first-frame object masks, area-resampled."""
    if not plan.objects:
        return torch.zeros(0, *latent_hw)
    return _area(np.stack([o.mask for o in plan.objects]).astype(np.float32), latent_hw).clamp(0, 1)


def tube_weights(plan: mp.MotionPlan, labels: dict, times: Sequence[float], latent_hw: Sequence[int], *, w_move: float = 3.0,
                 w_static: float = 1.5, dilate: int = 1, sweep: int = 4) -> torch.Tensor:
    """[T, h, w] flow-matching weights with mean 1.

    At every future latent frame: where a really moving object IS during the four video frames that latent frame
    encodes (its footprint swept from the previous latent time to this one) counts w_move, where a really static
    object sits (its first-frame mask) counts w_static, everything else 1; cells whose label is unknown or in between
    add nothing.  The conditioning frame is uniform.  Static objects can cover half the frame (a table), so their
    weight is kept small: it must not drown the moving tubes again."""
    lh, lw = latent_hw
    frames = len(times)
    moving = np.zeros((frames, *plan.size), np.float32)
    static = np.zeros((frames, *plan.size), np.float32)
    for k, obj in enumerate(plan.objects):
        for j in range(1, frames):
            if not labels["valid"][k, j]:
                continue
            if labels["moving"][k, j]:
                for part in range(1, sweep + 1):
                    when = float(times[j - 1]) + (float(times[j]) - float(times[j - 1])) * part / sweep
                    moving[j][mp.target_mask(obj, mp.affine_at(obj, plan.anchors, when)[0])] = 1.0
            elif labels["static"][k, j]:
                static[j][obj.mask] = 1.0
    move_l, static_l = _area(moving, latent_hw), _area(static, latent_hw)
    for _ in range(dilate):
        move_l = F.max_pool2d(move_l[:, None], 3, stride=1, padding=1)[:, 0]
        static_l = F.max_pool2d(static_l[:, None], 3, stride=1, padding=1)[:, 0]
    weight = 1.0 + torch.maximum((w_move - 1.0) * move_l, (w_static - 1.0) * static_l)
    weight[0] = 1.0
    return weight * (weight.numel() / weight.sum())


# ---------------------------------------------------------------------------------------------- clip length
def largest_4k1(count: int, cap: int = 37) -> int:
    count = min(int(count), cap)
    return 0 if count < 1 else 4 * ((count - 1) // 4) + 1


def scene_at_rest(tracks: np.ndarray, visible: np.ndarray, track_times: np.ndarray, *, window_s: float = 0.25, rest_px: float = 2.0) -> bool:
    """True when no tracked point that is still visible moved more than `rest_px` over the last `window_s` seconds."""
    if len(track_times) < 2:
        return False
    end = len(track_times) - 1
    start = int(np.searchsorted(track_times, track_times[end] - window_s, side="left"))
    start = min(start, end - 1)
    ok = visible[end] & visible[start]
    if not ok.any():
        return True
    return bool(np.linalg.norm(tracks[end] - tracks[start], axis=-1)[ok].max() <= rest_px)


def clip_window(frames_real: int, at_rest: bool, *, min_frames: int = 9, max_frames: int = 37, pad_to: int = 13) -> tuple[int, int]:
    """(frames used, real frames among them).  A clip keeps the largest 4k+1 window it provides.  A scene that has
    come to rest may be extended with its last frame up to `pad_to` frames (and to the next 4k+1): the world stays as
    it is, so this is the true continuation.  A clip that is still moving when it ends is never extended.
    Returns (0, 0) when the clip is too short to use."""
    frames_real = min(int(frames_real), max_frames)
    window = largest_4k1(frames_real, max_frames)
    if at_rest:
        target = max(pad_to, frames_real if (frames_real - 1) % 4 == 0 else 4 * ((frames_real - 1) // 4 + 1) + 1)
        target = min(target, max_frames)
        if target >= min_frames:
            return target, min(frames_real, target)
    if window >= min_frames:
        return window, window
    return 0, 0
