"""Trusted per-object, per-query displacement activity targets and diagnostics.

An active object has moved relative to the *first* frame at this query time.
This is not instantaneous velocity and need not be monotone over time. Labels
are made only from training/evaluation targets, never model conditioning.
Slot k is the actual input instance k; no learned-mask Hungarian assignment is
used. Unknown correspondences cannot become static labels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def _trusted(value: torch.Tensor, shape: tuple[int, ...], name: str,
             device: torch.device) -> torch.Tensor:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    value = value.to(device=device)
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value > 0.5


def _settings(config: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    moving = float(config.get("flow_moving_threshold_px", 1.0))
    static = float(config.get("flow_static_threshold_px", 0.5))
    moving_fraction = float(config.get("activity_moving_fraction", 0.05))
    static_fraction = float(config.get("activity_static_fraction", 0.95))
    min_pixels = float(config.get("activity_min_pixels", 4))
    if not (math.isfinite(moving) and math.isfinite(static) and 0 <= static < moving):
        raise ValueError("activity flow thresholds must satisfy 0 <= static < moving")
    if not (math.isfinite(moving_fraction) and 0 < moving_fraction <= 1):
        raise ValueError("activity_moving_fraction must be finite and in (0, 1]")
    if not (math.isfinite(static_fraction) and 0 < static_fraction <= 1):
        raise ValueError("activity_static_fraction must be finite and in (0, 1]")
    if not (math.isfinite(min_pixels) and min_pixels >= 1 and min_pixels.is_integer()):
        raise ValueError("activity_min_pixels must be a positive integer")
    return moving, static, moving_fraction, static_fraction, int(min_pixels)


@torch.no_grad()
def build_activity_targets(
    batch: Mapping[str, torch.Tensor],
    width: int = 160,
    height: int = 160,
    config: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Make explicit known/unknown object and pixel labels from trusted flow.

    Inputs are ``flow[B,T,2,H,W]`` normalized by width/height,
    ``instance_masks[B,K,H,W]``, ``flow_valid`` and ``static_certified``
    ``[B,T,1,H,W]``, and ``content_valid[B,1,H,W]``. Supports use >0.5.
    Positive objects require moving pixel mass >= the configured fraction of
    their *whole content area*, plus min(4, area) moving pixels by default.
    Negative objects require 95% whole-area certified-static coverage and NO
    trustworthy mover evidence. Gray-band/unknown pixels do not contribute to
    that static coverage. Tiny objects need all pixels to establish motion.

    ``slot_target/slot_valid`` are [B,K,T]. Pixel classifier targets are only
    outside the input instance union; ``dense_valid`` scores the final moving
    mask on ALL trusted moving/certified-static pixels, including the union.
    """
    cfg = {} if config is None else config
    moving_cutoff, static_cutoff, moving_fraction, static_fraction, min_pixels = _settings(cfg)
    if not all(math.isfinite(float(x)) and float(x) > 0 for x in (width, height)):
        raise ValueError("activity width and height must be finite and positive")
    required = {"flow", "instance_masks", "flow_valid", "static_certified", "content_valid"}
    missing = required - batch.keys()
    if missing:
        raise ValueError(f"activity supervision requires batch fields: {sorted(missing)}")
    flow = batch["flow"].detach()
    if flow.ndim != 5 or flow.shape[2] != 2:
        raise ValueError("activity flow must have shape [B,T,2,H,W]")
    b, t, _, h, w = flow.shape
    if min(b, t, h, w) <= 0:
        raise ValueError("activity batch, temporal and spatial dimensions must be positive")
    masks = batch["instance_masks"]
    if masks.ndim != 4 or masks.shape[0] != b or tuple(masks.shape[-2:]) != (h, w):
        raise ValueError("instance_masks must have shape [B,K,H,W] matching flow")
    masks = _trusted(masks, tuple(masks.shape), "instance_masks", flow.device)
    content = _trusted(batch["content_valid"], (b, 1, h, w), "content_valid", flow.device)
    masks = masks & content
    temporal_shape = (b, t, 1, h, w)
    valid = _trusted(batch["flow_valid"], temporal_shape, "flow_valid", flow.device) & content[:, None]
    certified = _trusted(batch["static_certified"], temporal_shape, "static_certified", flow.device)
    # Mask BEFORE arithmetic; NaN placeholder labels outside support are legal.
    clean = torch.where(valid, flow, torch.zeros_like(flow)).float()
    if not bool(torch.isfinite(clean).all()):
        raise ValueError("activity flow is nonfinite on trusted support")
    magnitude = (clean * clean.new_tensor([width, height]).view(1, 1, 2, 1, 1)).square().sum(2, keepdim=True).sqrt()
    moving = valid & (magnitude > moving_cutoff)
    static = valid & certified & (magnitude <= static_cutoff)
    mask_float = masks.float()
    area = mask_float.sum((-1, -2))[:, :, None].expand(-1, -1, t)
    # CUDA/CPU bfloat16 autocast would round counts near the 5%/95% cutoffs.
    # Labels must be independent of the precision used by the model forward.
    with torch.autocast(device_type=flow.device.type, enabled=False):
        moving_count = torch.einsum("bkhw,bthw->bkt", mask_float, moving[:, :, 0].float())
        static_count = torch.einsum("bkhw,bthw->bkt", mask_float, static[:, :, 0].float())
        observed_count = torch.einsum("bkhw,bthw->bkt", mask_float, valid[:, :, 0].float())
    nonempty = area > 0
    positive = nonempty & (moving_count >= area * moving_fraction) & (moving_count >= area.clamp_max(min_pixels))
    negative = nonempty & (static_count >= area * static_fraction) & (moving_count == 0)
    union = masks.any(dim=1, keepdim=True)[:, None].expand(-1, t, -1, -1, -1)
    dense_valid = moving | static
    return {
        "slot_target": positive.float(),
        "slot_valid": positive | negative,
        "slot_nonempty": nonempty,
        "slot_area": area,
        "slot_moving_count": moving_count,
        "slot_static_count": static_count,
        "slot_observed_count": observed_count,
        "moving": moving,
        "static": static,
        "pixel_target": moving.float(),
        "pixel_valid": dense_valid & ~union,
        "dense_valid": dense_valid,
        "input_union": union,
        "pixel_content_valid": content[:, None].expand(-1, t, -1, -1, -1),
    }


def _balanced_cell_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                       sample_weight: torch.Tensor | None) -> torch.Tensor:
    """Inputs [B,T,N]; balance classes within each sample/time then cells.

    Each known object receives equal weight within its class, irrespective of
    area. Absent classes and empty cells are excluded rather than penalized.
    torch.where keeps zero-support loss graph-connected without 0 * NaN.
    """
    if tuple(logits.shape) != tuple(target.shape) or tuple(valid.shape) != tuple(target.shape):
        raise ValueError("activity logits, target and support must have matching shapes")
    clean = torch.where(valid, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    if not bool(torch.isfinite(clean).all()):
        raise ValueError("activity logits are nonfinite on labeled support")
    error = F.binary_cross_entropy_with_logits(clean, target.float(), reduction="none")
    positive = valid & (target > 0.5)
    negative = valid & ~positive
    class_counts = torch.stack([positive.sum(-1), negative.sum(-1)], dim=-1)
    class_sums = torch.stack([(error * positive).sum(-1), (error * negative).sum(-1)], dim=-1)
    class_present = class_counts > 0
    cell_loss = (class_sums / class_counts.clamp_min(1)).sum(-1) / class_present.sum(-1).clamp_min(1)
    cell_weight = class_present.any(-1).to(cell_loss.dtype)
    if sample_weight is not None:
        if tuple(sample_weight.shape) != (logits.shape[0],):
            raise ValueError("activity sample_weight must have shape [B]")
        weights = sample_weight.to(device=logits.device, dtype=cell_loss.dtype)
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("activity sample_weight must be finite and nonnegative")
        cell_weight = cell_weight * weights[:, None]
    return (cell_loss * cell_weight).sum() / cell_weight.sum().clamp_min(1e-8)


def activity_loss_terms(
    prediction: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor],
    width: int = 160, height: int = 160, config: Mapping[str, Any] | None = None,
    sample_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return unweighted balanced BCE terms and label counts for logging."""
    targets = build_activity_targets(batch, width, height, config)
    for key, expected in (("slot_activity_logits", targets["slot_target"].shape),
                          ("pixel_activity_logits", targets["pixel_target"].shape)):
        if key not in prediction or tuple(prediction[key].shape) != tuple(expected):
            raise ValueError(f"activity prediction[{key!r}] must have shape {tuple(expected)}")
    object_loss = _balanced_cell_bce(prediction["slot_activity_logits"].permute(0, 2, 1),
                                    targets["slot_target"].permute(0, 2, 1),
                                    targets["slot_valid"].permute(0, 2, 1), sample_weight)
    pixel_loss = _balanced_cell_bce(prediction["pixel_activity_logits"].flatten(2),
                                   targets["pixel_target"].flatten(2),
                                   targets["pixel_valid"].flatten(2), sample_weight)
    positive = targets["slot_valid"] & (targets["slot_target"] > 0.5)
    return {
        "object_activity": object_loss,
        "pixel_activity": pixel_loss,
        "activity_object_positive_count": positive.sum().detach(),
        "activity_object_static_count": (targets["slot_valid"] & ~positive).sum().detach(),
        "activity_object_unknown_count": (targets["slot_nonempty"] & ~targets["slot_valid"]).sum().detach(),
        "activity_pixel_positive_count": (targets["pixel_valid"] & targets["moving"]).sum().detach(),
        "activity_pixel_static_count": (targets["pixel_valid"] & targets["static"]).sum().detach(),
    }


def _confusion(predicted: torch.Tensor, target: torch.Tensor, support: torch.Tensor,
               prefix: str) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}_tp": (support & predicted & target).sum(),
        f"{prefix}_fp": (support & predicted & ~target).sum(),
        f"{prefix}_fn": (support & ~predicted & target).sum(),
        f"{prefix}_tn": (support & ~predicted & ~target).sum(),
    }


@torch.no_grad()
def activity_sufficient_statistics(
    prediction: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor],
    width: int = 160, height: int = 160, config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Additive object/time and full/inside/outside pixel-mask confusion counts.

    Object probability >=0.5 (logit >=0) and final mask >=0.5 are fixed cutoffs.
    Unknown nonempty objects are counted separately, never scored as static.
    ``motion_mask`` must contain probabilities in [0,1], not a thresholded flow.

    Mask metrics use *pixel displacement activity*, whereas the object target
    is *whether this object has sufficient moving mass*. Selecting an entire
    partially moving/deforming object can therefore be an object true positive
    and incur static-pixel false positives. These pixel metrics diagnose motion
    localization, not consistency with the object classification labels.
    """
    targets = build_activity_targets(batch, width, height, config)
    logits = prediction["slot_activity_logits"].detach()
    if tuple(logits.shape) != tuple(targets["slot_target"].shape):
        raise ValueError("slot_activity_logits shape must match input slots and query times")
    if not bool(torch.isfinite(logits[targets["slot_valid"]]).all()):
        raise ValueError("object activity logits nonfinite on labeled support")
    motion_mask = prediction["motion_mask"].detach()
    if tuple(motion_mask.shape) != tuple(targets["dense_valid"].shape):
        raise ValueError("motion_mask must have shape [B,T,1,H,W]")
    known_probability = motion_mask[targets["dense_valid"]]
    if (not bool(torch.isfinite(known_probability).all()) or
            bool(((known_probability < 0) | (known_probability > 1)).any())):
        raise ValueError("motion_mask must contain finite probabilities in [0,1] on known support")
    values = _confusion(logits >= 0, targets["slot_target"] > 0.5, targets["slot_valid"], "object")
    values.update({
        "object_unknown_count": (targets["slot_nonempty"] & ~targets["slot_valid"]).sum(),
        "object_empty_count": (~targets["slot_nonempty"]).sum(),
    })
    for prefix, support in (
        ("mask_full", targets["dense_valid"]),
        ("mask_inside", targets["dense_valid"] & targets["input_union"]),
        ("mask_outside", targets["dense_valid"] & ~targets["input_union"]),
    ):
        values.update(_confusion(motion_mask >= 0.5, targets["moving"], support, prefix))
    unknown_pixels = targets["pixel_content_valid"] & ~targets["dense_valid"]
    unknown_supports = {
        "mask_full": unknown_pixels,
        "mask_inside": unknown_pixels & targets["input_union"],
        "mask_outside": unknown_pixels & ~targets["input_union"],
    }
    for prefix, unknown in unknown_supports.items():
        values[f"{prefix}_unknown_count"] = unknown.sum()
    # Reuse the targets and collect tensor counters before ONE CPU transfer;
    # never rebuild labels or synchronize separately for each query horizon.
    for time in range(logits.shape[2]):
        namespace = f"horizon_{time}/"
        values.update(_confusion(logits[:, :, time] >= 0,
                                 targets["slot_target"][:, :, time] > 0.5,
                                 targets["slot_valid"][:, :, time], namespace + "object"))
        values[namespace + "object_unknown_count"] = (
            targets["slot_nonempty"][:, :, time] & ~targets["slot_valid"][:, :, time]).sum()
        values[namespace + "object_empty_count"] = (~targets["slot_nonempty"][:, :, time]).sum()
        for prefix, support in (
            ("mask_full", targets["dense_valid"]),
            ("mask_inside", targets["dense_valid"] & targets["input_union"]),
            ("mask_outside", targets["dense_valid"] & ~targets["input_union"]),
        ):
            values.update(_confusion(motion_mask[:, time] >= 0.5, targets["moving"][:, time],
                                     support[:, time], namespace + prefix))
            values[namespace + prefix + "_unknown_count"] = unknown_supports[prefix][:, time].sum()
    packed = torch.stack([value.to(torch.float64) for value in values.values()]).cpu().tolist()
    result = dict(zip(values, packed))
    result["sample_count"] = float(batch["flow"].shape[0])
    return result


def summary_activity_statistics(sums: Mapping[str, float]) -> dict[str, float | None]:
    """Summarize only after accumulating counts; undefined ratios are None."""
    groups = ("object", "mask_full", "mask_inside", "mask_outside")
    namespaces = [""] + sorted({key.split("/")[0] + "/" for key in sums
                                if key.startswith("horizon_") and "/" in key})
    prefixes = [namespace + group for namespace in namespaces for group in groups]
    required = {f"{prefix}_{key}" for prefix in prefixes for key in ("tp", "fp", "fn", "tn")}
    required.update(namespace + key for namespace in namespaces
                    for key in ("object_unknown_count", "object_empty_count",
                                "mask_full_unknown_count", "mask_inside_unknown_count", "mask_outside_unknown_count"))
    required.add("sample_count")
    missing = required - sums.keys()
    if missing:
        raise ValueError(f"missing activity sufficient statistics: {sorted(missing)}")
    values = {key: float(sums[key]) for key in required}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("activity counts must be finite and nonnegative")

    def ratio(a: float, b: float) -> float | None:
        return a / b if b > 0 else None

    result: dict[str, float | None] = {}
    for prefix in prefixes:
        tp, fp, fn, tn = (values[f"{prefix}_{key}"] for key in ("tp", "fp", "fn", "tn"))
        result.update({
            f"{prefix}_precision": ratio(tp, tp + fp),
            f"{prefix}_recall": ratio(tp, tp + fn),
            f"{prefix}_f1": ratio(2 * tp, 2 * tp + fp + fn),
            f"{prefix}_iou": ratio(tp, tp + fp + fn),
            f"{prefix}_static_falsepositive_fraction": ratio(fp, fp + tn),
            f"{prefix}_positive_count": tp + fn,
            f"{prefix}_static_count": fp + tn,
        })
    for namespace in namespaces:
        known = sum(values[f"{namespace}object_{key}"] for key in ("tp", "fp", "fn", "tn"))
        result[namespace + "object_labeled_fraction"] = ratio(known, known + values[namespace + "object_unknown_count"])
        result[namespace + "object_unknown_count"] = values[namespace + "object_unknown_count"]
        result[namespace + "object_empty_count"] = values[namespace + "object_empty_count"]
        for prefix in ("mask_full", "mask_inside", "mask_outside"):
            result[namespace + prefix + "_unknown_count"] = values[namespace + prefix + "_unknown_count"]
    return result
