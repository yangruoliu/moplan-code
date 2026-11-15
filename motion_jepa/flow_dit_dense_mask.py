"""Dense motion-object labels, losses and diagnostics.

Labels live in FIRST-FRAME coordinates at each future query. A positive pixel
has trustworthy cumulative displacement; it is not future occupancy. Predicted
masks never select the ground-truth scoring or regression support. The
historical target is per-pixel displacement. ``whole_object_motion_v1`` instead
marks the complete first-frame instance once trusted flow establishes that the
object moves, while retaining pixel labels outside every instance. The v3 task-
object mode treats the selected first-frame object as a complete semantic mask:
every content pixel outside it is a mask negative even when its flow belongs to
a robot, hand, or another moving object. Flow regression support remains
governed independently by ``flow_valid``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def _support(value: torch.Tensor, shape: tuple[int, ...], name: str,
             device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    value = value.to(device)
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{name} must be finite and in [0,1]")
    return value > .5


@torch.no_grad()
def build_dense_mask_targets(
    batch: Mapping[str, torch.Tensor], width: int = 160, height: int = 160,
    config: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Build [B,T,1,H,W] labels on full trusted content.

    flow is [B,T,2,H,W], normalized independently by width/height. Positive is
    magnitude > moving cutoff (default 1px). Negative requires certified static
    AND magnitude <= static cutoff (default .5px). Both normally require
    flow_valid and content_valid. In authoritative whole-object mode, content
    outside the selected task object is a negative for this mask objective only.
    The whole-object modes consume label-side instance masks; predicted masks
    and visibility never define their targets or support.
    """
    config = {} if config is None else config
    target_mode = str(config.get("dense_motion_mask_target_mode", "pixel_displacement_v1"))
    if target_mode not in {
        "pixel_displacement_v1", "whole_object_motion_v1",
        "whole_object_motion_v2_authoritative",
        "whole_object_motion_v3_task_object",
    }:
        raise ValueError(f"unknown dense_motion_mask_target_mode {target_mode!r}")
    required = {"flow", "flow_valid", "static_certified", "content_valid"}
    if required - batch.keys():
        raise ValueError(f"dense mask supervision requires {sorted(required - batch.keys())}")
    moving_threshold = float(config.get("flow_moving_threshold_px", 1.))
    static_threshold = float(config.get("flow_static_threshold_px", .5))
    if not (math.isfinite(moving_threshold) and math.isfinite(static_threshold)
            and 0 <= static_threshold < moving_threshold):
        raise ValueError("dense mask thresholds must satisfy finite 0 <= static < moving")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in (width, height)):
        raise ValueError("dense mask width and height must be finite and positive")
    flow = batch["flow"].detach()
    if flow.ndim != 5 or flow.shape[2] != 2:
        raise ValueError("dense mask flow must be [B,T,2,H,W]")
    b, t, _, h, w = flow.shape
    if min(b, t, h, w) <= 0:
        raise ValueError("dense mask batch/time/spatial dimensions must be positive")
    shape = (b, t, 1, h, w)
    content = _support(batch["content_valid"], (b, 1, h, w), "content_valid", flow.device)
    content = content[:, None].expand(shape)
    flow_valid = _support(batch["flow_valid"], shape, "flow_valid", flow.device) & content
    certified = _support(batch["static_certified"], shape, "static_certified", flow.device)
    clean = torch.where(flow_valid, flow, torch.zeros_like(flow)).float()
    if not bool(torch.isfinite(clean).all()):
        raise ValueError("dense mask flow is nonfinite on trusted support")
    magnitude = torch.linalg.vector_norm(clean * clean.new_tensor([width, height]).view(1, 1, 2, 1, 1), dim=2, keepdim=True)
    if not bool(torch.isfinite(magnitude).all()):
        raise ValueError("dense mask flow magnitude overflow on trusted support")
    pixel_positive = flow_valid & (magnitude > moving_threshold)
    pixel_static = flow_valid & certified & (magnitude <= static_threshold)
    positive, static = pixel_positive, pixel_static
    if target_mode in {
        "whole_object_motion_v1",
        "whole_object_motion_v2_authoritative",
        "whole_object_motion_v3_task_object",
    }:
        # Use the label-side masks.  PhyCo supplies exact rendered instances;
        # Bridge supplies its audited first-frame segmentation proxy.  The
        # deployable conditioning masks remain a separate ``instance_masks``
        # field and cannot silently replace these labels.
        masks = batch.get("supervision_instance_masks")
        if not isinstance(masks, torch.Tensor) or masks.ndim != 4:
            raise ValueError(
                "whole_object_motion_v1 requires supervision_instance_masks [B,K,H,W]"
            )
        if masks.shape[0] != b or tuple(masks.shape[-2:]) != (h, w):
            raise ValueError(
                "supervision_instance_masks must match dense mask batch/spatial shape"
            )
        masks = _support(masks, tuple(masks.shape), "supervision_instance_masks", flow.device)
        masks = masks & content[:, 0]
        moving_fraction = float(config.get("activity_moving_fraction", .05))
        static_fraction = float(config.get("activity_static_fraction", .95))
        min_pixels_value = float(config.get("activity_min_pixels", 4))
        if not (math.isfinite(moving_fraction) and 0 < moving_fraction <= 1):
            raise ValueError("activity_moving_fraction must be finite and in (0,1]")
        if not (math.isfinite(static_fraction) and 0 < static_fraction <= 1):
            raise ValueError("activity_static_fraction must be finite and in (0,1]")
        if not (math.isfinite(min_pixels_value) and min_pixels_value >= 1
                and min_pixels_value.is_integer()):
            raise ValueError("activity_min_pixels must be a positive integer")
        min_pixels = int(min_pixels_value)
        mask_float = masks.float()
        area = mask_float.sum((-1, -2))[:, :, None].expand(-1, -1, t)
        with torch.autocast(device_type=flow.device.type, enabled=False):
            moving_count = torch.einsum(
                "bkhw,bthw->bkt", mask_float, pixel_positive[:, :, 0].float()
            )
            static_count = torch.einsum(
                "bkhw,bthw->bkt", mask_float, pixel_static[:, :, 0].float()
            )
        nonempty = area > 0
        object_positive = (
            nonempty
            & (moving_count >= area * moving_fraction)
            & (moving_count >= area.clamp_max(min_pixels))
        )
        object_static = (
            nonempty
            & (static_count >= area * static_fraction)
            & (moving_count == 0)
        )
        masks_bt = masks[:, None].expand(-1, t, -1, -1, -1)
        object_positive_pixels = (
            masks_bt & object_positive.permute(0, 2, 1)[:, :, :, None, None]
        ).any(2, keepdim=True)
        object_static_pixels = (
            masks_bt & object_static.permute(0, 2, 1)[:, :, :, None, None]
        ).any(2, keepdim=True)
        union = masks.any(1, keepdim=True)[:, None].expand(shape)
        outside_positive = pixel_positive & ~union
        authoritative_rows = torch.zeros(
            (b, 1, 1, 1, 1), dtype=torch.bool, device=flow.device
        )
        if target_mode in {
            "whole_object_motion_v2_authoritative",
            "whole_object_motion_v3_task_object",
        }:
            authoritative = batch.get("supervision_object_identity_authoritative")
            if (
                not isinstance(authoritative, torch.Tensor)
                or tuple(authoritative.shape) != (b,)
                or not torch.isfinite(authoritative).all()
                or not ((authoritative == 0) | (authoritative == 1)).all()
            ):
                raise ValueError(
                    f"{target_mode} requires a binary "
                    "supervision_object_identity_authoritative tensor [B]"
                )
            # Authoritative rows never fall back to treating outside RAFT
            # motion as a second positive object. The v2 mode retains that area
            # as unknown for historical reproducibility; the v3 mode uses the
            # selected object as a complete task-object label.
            authoritative_rows = authoritative.to(flow.device).bool()[
                :, None, None, None, None
            ]
            outside_positive &= ~authoritative_rows
        positive = object_positive_pixels | outside_positive
        # Positive wins if malformed overlapping instances disagree.  This
        # prevents one object's negative label from suppressing mover evidence.
        task_object_negative = torch.zeros_like(content)
        if target_mode == "whole_object_motion_v3_task_object":
            # Motion from a robot, hand, camera, or another object is negative
            # for the task-object mask head. This does not manufacture static
            # flow supervision: flow losses use independent validity tensors.
            task_object_negative = content & ~union & authoritative_rows
        static = (
            object_static_pixels
            | (pixel_static & ~union)
            | task_object_negative
        ) & ~positive
    valid = positive | static
    return {"target": positive.float(), "valid": valid, "positive": positive,
            "static": static, "unknown": content & ~valid, "content_valid": content,
            "flow_valid": flow_valid, "padding": ~content}


def _sample_weights(sample_weight: torch.Tensor | None, logits: torch.Tensor) -> torch.Tensor:
    if sample_weight is None:
        return torch.ones(logits.shape[0], 1, device=logits.device, dtype=torch.float32)
    if tuple(sample_weight.shape) != (logits.shape[0],):
        raise ValueError("dense mask sample_weight must have shape [B]")
    value = sample_weight.to(device=logits.device, dtype=torch.float32)
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError("dense mask sample_weight must be finite and nonnegative")
    return value[:, None]


def _cell_mean(values: torch.Tensor, nonempty: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights * nonempty.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


@torch.no_grad()
def build_task_object_slot_targets(
    batch: Mapping[str, torch.Tensor], dense_targets: Mapping[str, torch.Tensor],
    config: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate the same semantic pixel contract onto deployable input slots.

    A moving robot can have valid dynamic flow while still being a negative for
    task-object identity.  Deriving these labels from ``dense_targets`` rather
    than raw flow preserves that distinction and keeps the slot selector
    exactly aligned with the full-grid v3 mask objective.
    """
    config = {} if config is None else config
    masks = batch.get("instance_masks")
    positive = dense_targets["positive"]
    static = dense_targets["static"]
    if not isinstance(masks, torch.Tensor) or masks.ndim != 4:
        raise ValueError("task-object slot supervision requires instance_masks [B,K,H,W]")
    b, t, _, h, w = positive.shape
    if masks.shape[0] != b or tuple(masks.shape[-2:]) != (h, w):
        raise ValueError("task-object instance_masks must match dense target geometry")
    masks = _support(masks, tuple(masks.shape), "task-object instance_masks", positive.device)
    content = dense_targets["content_valid"][:, 0, 0]
    masks &= content[:, None]
    moving_fraction = float(config.get("activity_moving_fraction", .05))
    static_fraction = float(config.get("activity_static_fraction", .95))
    min_pixels_value = float(config.get("activity_min_pixels", 4))
    if not (math.isfinite(moving_fraction) and 0 < moving_fraction <= 1):
        raise ValueError("activity_moving_fraction must be finite and in (0,1]")
    if not (math.isfinite(static_fraction) and 0 < static_fraction <= 1):
        raise ValueError("activity_static_fraction must be finite and in (0,1]")
    if not (
        math.isfinite(min_pixels_value)
        and min_pixels_value >= 1
        and min_pixels_value.is_integer()
    ):
        raise ValueError("activity_min_pixels must be a positive integer")
    mask_float = masks.float()
    area = mask_float.sum((-1, -2))[:, :, None].expand(-1, -1, t)
    with torch.autocast(device_type=positive.device.type, enabled=False):
        positive_count = torch.einsum(
            "bkhw,bthw->bkt", mask_float, positive[:, :, 0].float()
        )
        static_count = torch.einsum(
            "bkhw,bthw->bkt", mask_float, static[:, :, 0].float()
        )
    nonempty = area > 0
    positive_slot = (
        nonempty
        & (positive_count >= area * moving_fraction)
        & (positive_count >= area.clamp_max(int(min_pixels_value)))
    )
    negative_slot = (
        nonempty
        & (static_count >= area * static_fraction)
        & (positive_count == 0)
    )
    valid = positive_slot | negative_slot
    return {
        "target": positive_slot.float(),
        "valid": valid,
        "positive": positive_slot,
        "static": negative_slot,
        "unknown": nonempty & ~valid,
        "empty": ~nonempty,
    }


@torch.no_grad()
def build_task_object_identity_targets(
    batch: Mapping[str, torch.Tensor], dense_targets: Mapping[str, torch.Tensor],
    config: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Match deployable proposals to the task object, independent of motion.

    The selected label mask says *which object the caption constrains*.  It is
    deliberately not intersected with flow: a selected object can be static at
    one or every query time.  Ambiguous partial-overlap proposals are ignored
    instead of being forced into either class.
    """
    config = {} if config is None else config
    proposals = batch.get("instance_masks")
    selected = batch.get("supervision_instance_masks")
    positive = dense_targets["positive"]
    if not isinstance(proposals, torch.Tensor) or proposals.ndim != 4:
        raise ValueError("task-object identity requires instance_masks [B,K,H,W]")
    if not isinstance(selected, torch.Tensor) or selected.ndim != 4:
        raise ValueError(
            "task-object identity requires supervision_instance_masks [B,S,H,W]"
        )
    b, _, _, h, w = positive.shape
    if proposals.shape[0] != b or tuple(proposals.shape[-2:]) != (h, w):
        raise ValueError("task-object identity proposals must match dense target geometry")
    if selected.shape[0] != b or tuple(selected.shape[-2:]) != (h, w):
        raise ValueError("task-object identity labels must match dense target geometry")
    authoritative = batch.get("supervision_object_identity_authoritative")
    if (
        not isinstance(authoritative, torch.Tensor)
        or tuple(authoritative.shape) != (b,)
        or not bool(torch.isfinite(authoritative).all())
        or not bool(((authoritative == 0) | (authoritative == 1)).all())
    ):
        raise ValueError(
            "task-object identity requires binary "
            "supervision_object_identity_authoritative [B]"
        )
    device = positive.device
    proposals = _support(
        proposals, tuple(proposals.shape), "task-object identity proposals", device
    )
    selected = _support(
        selected, tuple(selected.shape), "task-object identity labels", device
    )
    content = dense_targets["content_valid"][:, 0, 0]
    proposals &= content[:, None]
    selected_union = selected.any(1) & content
    positive_fraction = float(
        config.get("task_object_identity_positive_fraction", .5)
    )
    negative_fraction = float(
        config.get("task_object_identity_negative_fraction", .05)
    )
    min_pixels_value = float(config.get("task_object_identity_min_pixels", 4))
    if not (
        math.isfinite(positive_fraction)
        and math.isfinite(negative_fraction)
        and 0 <= negative_fraction < positive_fraction <= 1
    ):
        raise ValueError(
            "task-object identity fractions must satisfy 0 <= negative < positive <= 1"
        )
    if not (
        math.isfinite(min_pixels_value)
        and min_pixels_value >= 1
        and min_pixels_value.is_integer()
    ):
        raise ValueError("task_object_identity_min_pixels must be a positive integer")
    proposal_float = proposals.float()
    area = proposal_float.sum((-1, -2))
    overlap = torch.einsum(
        "bkhw,bhw->bk", proposal_float, selected_union.float()
    )
    fraction = overlap / area.clamp_min(1)
    nonempty = area > 0
    trusted = authoritative.to(device).bool()[:, None]
    identity_positive = (
        trusted
        & nonempty
        & (overlap >= int(min_pixels_value))
        & (fraction >= positive_fraction)
    )
    identity_negative = (
        trusted & nonempty & (fraction <= negative_fraction)
    )
    valid = identity_positive | identity_negative
    return {
        "target": identity_positive.float(),
        "valid": valid,
        "positive": identity_positive,
        "static": identity_negative,
        "unknown": nonempty & ~valid,
        "empty": ~nonempty,
    }


def task_object_slot_loss(
    logits: torch.Tensor, targets: Mapping[str, torch.Tensor],
    sample_weight: torch.Tensor | None, prior_bce_fraction: float = 0.0,
) -> torch.Tensor:
    """BCE over known task-object/static slots with an explicit prior blend.

    Pure class balancing makes one positive proposal carry the same total
    weight as all static proposals combined.  That is useful for recall, but
    badly miscalibrates the selector when deployment has one mover among many
    proposals.  ``prior_bce_fraction`` blends back the observed slot frequency
    so a false positive again has a real cost, while retaining a configurable
    rare-mover budget.
    """
    if not math.isfinite(prior_bce_fraction) or not 0 <= prior_bce_fraction <= 1:
        raise ValueError("task_object_slot_prior_bce_fraction must lie in [0,1]")
    target, valid = targets["target"], targets["valid"]
    if tuple(logits.shape) != tuple(target.shape):
        raise ValueError("task_object_slot_logits must have shape [B,K,T]")
    clean = torch.where(valid, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    if not bool(torch.isfinite(clean).all()):
        raise ValueError("task-object slot logits are nonfinite on labeled support")
    error = F.binary_cross_entropy_with_logits(clean, target, reduction="none")
    positive, negative = targets["positive"], targets["static"]
    counts = torch.stack((positive.sum(1), negative.sum(1)), dim=-1)
    sums = torch.stack(((error * positive).sum(1), (error * negative).sum(1)), dim=-1)
    present = counts > 0
    balanced_cells = (
        (sums / counts.clamp_min(1)).sum(-1) / present.sum(-1).clamp_min(1)
    )
    prior_cells = (error * valid).sum(1) / valid.sum(1).clamp_min(1)
    cells = (
        (1.0 - prior_bce_fraction) * balanced_cells
        + prior_bce_fraction * prior_cells
    )
    weights = _sample_weights(sample_weight, clean).to(cells.dtype)
    return _cell_mean(cells, present.any(-1), weights)


def selection_temporal_consistency(logits, targets, sample_weight=None):
    """Smooth only adjacent queries whose known GT selection state is equal.

    This is not monotonicity: release, stop and return transitions may change
    selection. GT states are training labels only, never inference inputs.
    """
    valid = targets["valid"]
    same = valid[..., 1:] & valid[..., :-1] & (targets["target"][..., 1:] == targets["target"][..., :-1])
    clean = torch.where(valid, logits.float(), 0).sigmoid()
    errors = (clean[..., 1:] - clean[..., :-1]).square()
    objects = same.any(-1)
    per_object = (errors * same).sum(-1) / same.sum(-1).clamp_min(1)
    cells = (per_object * objects).sum(-1) / objects.sum(-1).clamp_min(1)
    weights = _sample_weights(sample_weight, clean)
    return _cell_mean(cells[:, None], objects.any(-1)[:, None], weights)


def dense_mask_loss_terms(
    prediction: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor],
    width: int = 160, height: int = 160, config: Mapping[str, Any] | None = None,
    sample_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Balanced BCE + configurable soft Dice, on the complete known support.

    BCE balances positive/static classes within each sample/query, then averages
    nonempty cells. Dice averages ONLY cells with at least one positive; static-
    only cells remain fully supervised by BCE. No target or denominator depends
    on the predicted mask or SAM. Returns unweighted outer loss plus components.
    """
    config = {} if config is None else config
    dice_weight = float(config.get("dense_motion_mask_dice_weight", .5))
    pixel_fraction = float(config.get("dense_motion_mask_pixel_bce_fraction", 0.0))
    if not math.isfinite(pixel_fraction) or not 0 <= pixel_fraction <= 1:
        raise ValueError("dense_motion_mask_pixel_bce_fraction must lie in [0,1]")
    if not math.isfinite(dice_weight) or dice_weight < 0:
        raise ValueError("dense_motion_mask_dice_weight must be finite and nonnegative")
    targets = build_dense_mask_targets(batch, width, height, config)
    logits = prediction.get("dense_motion_mask_logits")
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != tuple(targets["target"].shape):
        raise ValueError("dense_motion_mask_logits must have shape [B,T,1,H,W]")
    valid, target = targets["valid"], targets["target"]
    clean = torch.where(valid, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    if not bool(torch.isfinite(clean).all()):
        raise ValueError("dense mask logits nonfinite on labeled support")
    weights = _sample_weights(sample_weight, clean)
    positive, static = targets["positive"].flatten(2), targets["static"].flatten(2)
    error = F.binary_cross_entropy_with_logits(clean, target, reduction="none").flatten(2)
    counts = torch.stack([positive.sum(-1), static.sum(-1)], -1)
    sums = torch.stack([(error * positive).sum(-1), (error * static).sum(-1)], -1)
    present = counts > 0
    bce_cells = (sums / counts.clamp_min(1)).sum(-1) / present.sum(-1).clamp_min(1)
    balanced_bce = _cell_mean(bce_cells, present.any(-1), weights)
    # Class-balanced BCE discards the true foreground prior; at 2% prevalence
    # its .5 threshold can activate far too many static pixels. Blend in the
    # actual per-query pixel frequency, while retaining a rare-mover budget.
    pixel_cells = sums.sum(-1) / counts.sum(-1).clamp_min(1)
    pixel_bce = _cell_mean(pixel_cells, present.any(-1), weights)
    bce = (1 - pixel_fraction) * balanced_bce + pixel_fraction * pixel_bce
    probability = clean.sigmoid().flatten(2) * valid.flatten(2)
    target_flat = target.flatten(2)
    intersection = (probability * target_flat).sum(-1)
    denominator = probability.sum(-1) + target_flat.sum(-1)
    dice_cells = 1. - (2. * intersection + 1e-6) / (denominator + 1e-6)
    dice = _cell_mean(dice_cells, positive.any(-1), weights)
    result = {
        "dense_motion_mask": bce + dice_weight * dice,
        "dense_motion_mask_bce": bce,
        "dense_motion_mask_balanced_bce": balanced_bce,
        "dense_motion_mask_pixel_bce": pixel_bce,
        "dense_motion_mask_dice": dice,
        "dense_motion_mask_support_positive": targets["positive"].sum().detach(),
        "dense_motion_mask_support_static": targets["static"].sum().detach(),
        "dense_motion_mask_support_unknown": targets["unknown"].sum().detach(),
        "dense_motion_mask_support_padding": targets["padding"].sum().detach(),
        "dense_motion_mask_labeled_cells": present.any(-1).sum().detach(),
        "dense_motion_mask_positive_cells": positive.any(-1).sum().detach(),
    }
    slot_weight = float(config.get("task_object_slot_activity", 0.0))
    if not math.isfinite(slot_weight) or slot_weight < 0:
        raise ValueError("task_object_slot_activity must be finite and nonnegative")
    if slot_weight:
        slot_targets = build_task_object_slot_targets(batch, targets, config)
        slot_logits = prediction.get("task_object_slot_logits")
        if not isinstance(slot_logits, torch.Tensor):
            raise ValueError(
                "task_object_slot_activity requires task_object_slot_logits"
            )
        result.update(
            task_object_slot_activity=task_object_slot_loss(
                slot_logits, slot_targets, sample_weight,
                prior_bce_fraction=float(
                    config.get("task_object_slot_prior_bce_fraction", 0.0)
                ),
            ),
            task_object_slot_positive_count=slot_targets["positive"].sum().detach(),
            task_object_slot_static_count=slot_targets["static"].sum().detach(),
            task_object_slot_unknown_count=slot_targets["unknown"].sum().detach(),
            task_object_slot_empty_count=slot_targets["empty"].sum().detach(),
        )
        consistency_weight = float(config.get("task_object_slot_consistency", 0.))
        if not math.isfinite(consistency_weight) or consistency_weight < 0:
            raise ValueError("task_object_slot_consistency must be finite and nonnegative")
        if consistency_weight:
            consistency = selection_temporal_consistency(slot_logits, slot_targets, sample_weight)
            result["task_object_slot_temporal_consistency"] = consistency
            result["task_object_slot_activity"] = result["task_object_slot_activity"] + consistency_weight * consistency
    identity_weight = float(config.get("task_object_identity", 0.0))
    if not math.isfinite(identity_weight) or identity_weight < 0:
        raise ValueError("task_object_identity must be finite and nonnegative")
    if identity_weight:
        identity_targets = build_task_object_identity_targets(batch, targets, config)
        identity_logits = prediction.get("task_object_identity_logits")
        if not isinstance(identity_logits, torch.Tensor):
            raise ValueError(
                "task_object_identity requires task_object_identity_logits"
            )
        expanded_targets = {
            key: value.unsqueeze(-1) for key, value in identity_targets.items()
        }
        result.update(
            task_object_identity=task_object_slot_loss(
                identity_logits.unsqueeze(-1), expanded_targets, sample_weight,
                prior_bce_fraction=float(
                    config.get("task_object_identity_prior_bce_fraction", .75)
                ),
            ),
            task_object_identity_positive_count=(
                identity_targets["positive"].sum().detach()
            ),
            task_object_identity_negative_count=(
                identity_targets["static"].sum().detach()
            ),
            task_object_identity_unknown_count=(
                identity_targets["unknown"].sum().detach()
            ),
            task_object_identity_empty_count=(
                identity_targets["empty"].sum().detach()
            ),
        )
    return result


def _confusion(pred: torch.Tensor, target: torch.Tensor, support: torch.Tensor,
               unknown: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
    tp, fp = (support & pred & target).sum(), (support & pred & ~target).sum()
    fn, tn = (support & ~pred & target).sum(), (support & ~pred & ~target).sum()
    return {prefix + "tp": tp, prefix + "fp": fp, prefix + "fn": fn, prefix + "tn": tn,
            prefix + "positive_count": tp + fn, prefix + "static_count": fp + tn,
            prefix + "support_count": support.sum(), prefix + "unknown_count": unknown.sum()}


@torch.no_grad()
def dense_mask_sufficient_statistics(
    prediction: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor],
    width: int = 160, height: int = 160, config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Additive full-image and optional SAM partition counts at logit >= 0.

    Always computes full metrics without requiring masks/instance IDs. If
    instance_masks[B,K,H,W] is present, also reports fixed input-mask inside/
    outside diagnostics. Those masks NEVER modify full targets or losses.
    Per-horizon counts use keys ``horizon_0/full/tp`` etc. Metrics use logits to
    avoid exporting low-precision probabilities changing the 0.5 boundary.
    """
    targets = build_dense_mask_targets(batch, width, height, config)
    logits = prediction.get("dense_motion_mask_logits")
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != tuple(targets["target"].shape):
        raise ValueError("dense_motion_mask_logits must have shape [B,T,1,H,W]")
    logits = logits.detach()
    if not bool(torch.isfinite(logits[targets["valid"]]).all()):
        raise ValueError("dense mask logits nonfinite on labeled support")
    predicted = logits >= 0
    b, t, _, h, w = logits.shape
    partitions = {"full": torch.ones_like(targets["valid"])}
    if "instance_masks" in batch:
        masks = batch["instance_masks"]
        if masks.ndim != 4 or masks.shape[0] != b or tuple(masks.shape[-2:]) != (h, w):
            raise ValueError("diagnostic instance_masks must have shape [B,K,H,W]")
        trusted_masks = _support(masks, tuple(masks.shape), "diagnostic instance_masks", logits.device)
        union = trusted_masks.any(dim=1, keepdim=True)[:, None].expand(-1, t, -1, -1, -1)
        partitions.update(inside=union, outside=~union)
    values = {}
    for name, partition in partitions.items():
        support = targets["valid"] & partition
        unknown = targets["unknown"] & partition
        values.update(_confusion(predicted, targets["positive"], support, unknown, name + "/"))
        for time in range(t):
            values.update(_confusion(predicted[:, time], targets["positive"][:, time], support[:, time],
                                     unknown[:, time], f"horizon_{time}/{name}/"))
    slot_logits = prediction.get("task_object_slot_logits")
    if slot_logits is not None:
        if not isinstance(slot_logits, torch.Tensor):
            raise ValueError("task_object_slot_logits must be a tensor")
        slot_targets = build_task_object_slot_targets(batch, targets, config)
        if tuple(slot_logits.shape) != tuple(slot_targets["target"].shape):
            raise ValueError("task_object_slot_logits must have shape [B,K,T]")
        if not bool(torch.isfinite(slot_logits[slot_targets["valid"]]).all()):
            raise ValueError("task-object slot logits nonfinite on labeled support")
        slot_probability = prediction.get("task_object_slot_probability")
        if slot_probability is None:
            slot_predicted = slot_logits.detach() >= 0
        else:
            if (
                not isinstance(slot_probability, torch.Tensor)
                or tuple(slot_probability.shape) != tuple(slot_targets["target"].shape)
                or not bool(torch.isfinite(slot_probability).all())
                or bool(((slot_probability < 0) | (slot_probability > 1)).any())
            ):
                raise ValueError(
                    "task_object_slot_probability must be finite [B,K,T] in [0,1]"
                )
            # Report the deployed identity-times-motion decision.  Legacy
            # checkpoints have no explicit probability and retain logit>=0.
            slot_predicted = slot_probability.detach() >= .5
        values.update(_confusion(
            slot_predicted, slot_targets["target"] > .5, slot_targets["valid"],
            slot_targets["unknown"], "slot/",
        ))
        for time in range(t):
            values.update(_confusion(
                slot_predicted[:, :, time], slot_targets["target"][:, :, time] > .5,
                slot_targets["valid"][:, :, time], slot_targets["unknown"][:, :, time],
                f"horizon_{time}/slot/",
            ))
    identity_logits = prediction.get("task_object_identity_logits")
    if identity_logits is not None:
        if not isinstance(identity_logits, torch.Tensor):
            raise ValueError("task_object_identity_logits must be a tensor")
        identity_targets = build_task_object_identity_targets(batch, targets, config)
        if tuple(identity_logits.shape) != tuple(identity_targets["target"].shape):
            raise ValueError("task_object_identity_logits must have shape [B,K]")
        if not bool(torch.isfinite(identity_logits[identity_targets["valid"]]).all()):
            raise ValueError("task-object identity logits nonfinite on labeled support")
        values.update(_confusion(
            identity_logits.detach() >= 0,
            identity_targets["target"] > .5,
            identity_targets["valid"], identity_targets["unknown"], "identity/",
        ))
    values["padding_count"] = targets["padding"].sum()
    packed = torch.stack([value.to(torch.float64) for value in values.values()]).cpu().tolist()
    result = dict(zip(values, packed))
    result["sample_count"] = float(b)
    result["horizon_cell_count"] = float(b * t)
    return result


def summary_dense_mask_statistics(sums: Mapping[str, float]) -> dict[str, float | None]:
    """Form ratios only after pooling counts. Undefined class metrics are None."""
    required_base = {"sample_count", "horizon_cell_count", "padding_count"}
    fields = ("tp", "fp", "fn", "tn", "positive_count", "static_count", "support_count", "unknown_count")
    scopes = {key.rsplit("/", 1)[0] for key in sums if "/" in key and key.rsplit("/", 1)[1] in fields}
    if "full" not in scopes:
        raise ValueError("dense mask statistics require full-image counters")
    required = required_base | {scope + "/" + field for scope in scopes for field in fields}
    if required - sums.keys():
        raise ValueError(f"missing dense mask statistics: {sorted(required - sums.keys())}")
    values = {key: float(sums[key]) for key in required}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("dense mask counters must be finite and nonnegative")
    result = {key: values[key] for key in required_base}

    def ratio(a: float, b: float) -> float | None:
        return a / b if b > 0 else None

    for scope in sorted(scopes):
        tp, fp, fn, tn = (values[f"{scope}/{key}"] for key in ("tp", "fp", "fn", "tn"))
        if (tp + fn != values[f"{scope}/positive_count"] or fp + tn != values[f"{scope}/static_count"]
                or tp + fp + fn + tn != values[f"{scope}/support_count"]):
            raise ValueError(f"inconsistent dense mask confusion/support counts for {scope}")
        recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
        metrics = {
            "precision": ratio(tp, tp + fp), "recall": recall,
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "iou": ratio(tp, tp + fp + fn),
            "static_falsepositive_fraction": ratio(fp, fp + tn), "static_specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        }
        result.update({scope + "/" + key: value for key, value in metrics.items()})
        result.update({scope + "/" + key: values[scope + "/" + key] for key in fields})
    return result
