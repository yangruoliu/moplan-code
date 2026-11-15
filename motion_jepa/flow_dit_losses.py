"""Flow-space auxiliary losses for Stage-1 WanFlowDiT.

The diffusion objective lives in latent space and is blind to the metric that
decides whether this model is worth anything.  These losses close that gap by
supervising the decoded 160x160 field directly, using exactly the validity
convention that ``motion_jepa/metrics.py:35`` scores:

    valid = object_mask & visibility

That is the default, unchanged ``supervision_mode=legacy`` contract. Opt-in
``valid_partition_v1`` uses explicit trusted correspondence support, separates
the reference instance union from the legacy dynamic object mask, and assigns
independent budgets to moving, certified-static, background and weak flow.

Two asymmetries are deliberate:

* **Static pixels get real weight.**  Most scored pixels do not move, and a
  model that sprinkles a few pixels of jitter over them loses to predicting zero.  The
  moving/static balance is exposed as ``moving_weight`` precisely because it is
  the easiest knob to get wrong in either direction.
* **Auxiliary weight decays with sigma.**  At high noise the x0 estimate is
  mostly a prior sample, so pixel-accurate supervision there is noise.  The
  weight is folded into the per-pixel mask so a single ``masked_mean`` still
  produces a correctly weighted average.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .losses import instance_trajectory_centers, masked_mean


def sigma_weight(sigmas: torch.Tensor, mode: str = "one_minus_sigma_sq") -> torch.Tensor:
    """Per-sample weight for the flow-space losses."""
    if mode == "one_minus_sigma_sq":
        return (1.0 - sigmas.float()).square()
    if mode == "one_minus_sigma":
        return 1.0 - sigmas.float()
    if mode == "uniform":
        return torch.ones_like(sigmas.float())
    raise ValueError(f"unknown aux_sigma_weight mode {mode!r}")


def cosine_ramp(step: int, total: int) -> float:
    """Warm the auxiliary losses in over ``total`` steps: 0 -> 1, cosine shaped.

    VPT anneals its auxiliary terms *down* because they are soft constraints on
    a generative target.  Here the flow term is the actual objective, so the
    ramp goes the other way: let flow-matching establish a sane latent first,
    then bring the pixel supervision up.
    """
    if total <= 0:
        return 1.0
    progress = min(max(step / float(total), 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def _balanced_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Class-balanced BCE, matching ``motion_jepa/losses.py:106-115``."""
    positive_fraction = masked_mean(target, source).detach().clamp(0.05, 0.95)
    balance = torch.where(
        target > 0.5, 0.5 / positive_fraction, 0.5 / (1.0 - positive_fraction)
    )
    return masked_mean(
        F.binary_cross_entropy_with_logits(logits, target, reduction="none"),
        source * balance,
    )


def _unit_support(
    batch: dict[str, torch.Tensor], key: str, shape: tuple[int, ...], reference: torch.Tensor
) -> torch.Tensor:
    """Read an explicit supervision mask; never infer unknown pixels as static."""
    if key not in batch:
        raise ValueError(f"valid_partition_v1 requires batch[{key!r}]")
    value = batch[key].to(device=reference.device, dtype=reference.dtype)
    if tuple(value.shape) != shape:
        raise ValueError(f"{key} must have shape {shape}, got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{key} must contain finite weights in [0, 1]")
    return value


def _partition_mean(
    error: torch.Tensor,
    support: torch.Tensor,
    sample_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Average each nonempty sample/horizon first, then average those cells.

    A large static region cannot dilute a moving region's separate budget, and
    a large image support cannot dominate another sample in the same group.
    Empty cells are excluded, including samples with zero sigma weight.
    """
    weight = support.expand_as(error)
    denominator = weight.sum(dim=(2, 3, 4))
    per_cell = (error * weight).sum(dim=(2, 3, 4)) / denominator.clamp_min(1e-6)
    cell_weight = (denominator > 0).to(error.dtype)
    if sample_weight is not None:
        cell_weight = cell_weight * sample_weight.to(error).view(-1, 1)
    return masked_mean(per_cell, cell_weight)


def _direction_and_parallel_magnitude(
    predicted: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    sample_weight: torch.Tensor | None,
    *,
    width: int,
    height: int,
    epsilon_px: float,
    magnitude_scale_px: float,
    magnitude_beta_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate mover direction from signed displacement magnitude.

    Endpoint loss alone has a strong zero/conditional-mean solution on
    ambiguous data.  A smoothed cosine supplies a nonzero directional gradient
    even when the prediction is exactly zero.  Magnitude is measured by the
    signed projection onto the true direction rather than ``norm(pred)``;
    unlike a vector norm, its gradient is also nonzero at the origin and an
    opposite vector cannot receive credit for having the right length.
    """
    scale = predicted.new_tensor([width, height]).view(1, 1, 2, 1, 1)
    predicted_px = predicted * scale
    target_px = target * scale
    target_magnitude = target_px.square().sum(dim=2, keepdim=True).sqrt().clamp_min(1e-6)
    predicted_smooth_norm = (
        predicted_px.square().sum(dim=2, keepdim=True) + epsilon_px * epsilon_px
    ).sqrt()
    dot = (predicted_px * target_px).sum(dim=2, keepdim=True)
    direction_error = 1.0 - dot / (predicted_smooth_norm * target_magnitude)

    target_unit = target_px / target_magnitude
    signed_parallel_px = (predicted_px * target_unit).sum(dim=2, keepdim=True)
    magnitude_error = F.smooth_l1_loss(
        signed_parallel_px / magnitude_scale_px,
        target_magnitude / magnitude_scale_px,
        reduction="none",
        beta=magnitude_beta_px / magnitude_scale_px,
    )
    return (
        _partition_mean(direction_error, support, sample_weight),
        _partition_mean(magnitude_error, support, sample_weight),
    )


def trajectory_segment_loss(predicted, target, valid, horizons, sample_weight=None,
                            *, width=160, height=160, scale_px_per_second=32.,
                            moving_threshold_px=1., epsilon_px_per_second=.5):
    """Supervise changes along the same first-frame point's cumulative path.

    Both endpoints must have valid correspondence. No constant-velocity or
    smoothness assumption is imposed: bounces and stops follow the labeled
    interval velocity. A newly visible point is not linked across occlusion.
    Masks predicted by the model never determine this regression support.
    """
    b, t = predicted.shape[:2]
    times = torch.as_tensor(horizons, device=predicted.device, dtype=torch.float32)
    if times.ndim == 1:
        times = times[None].expand(b, -1)
    if tuple(times.shape) != (b, t) or not bool(torch.isfinite(times).all()):
        raise ValueError("trajectory_segment requires finite horizon_seconds [T] or [B,T]")
    dt = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
    if bool((dt <= 0).any()):
        raise ValueError("trajectory_segment horizons must be positive and increasing")
    if any(not math.isfinite(x) or x <= 0 for x in (scale_px_per_second, epsilon_px_per_second)):
        raise ValueError("trajectory_segment scales must be finite and positive")
    known = valid > .5
    pred = torch.where(known, predicted, 0).float()
    truth = torch.where(known, target, 0).float()
    pixel_scale = pred.new_tensor([width, height]).view(1, 1, 2, 1, 1)
    pd = torch.diff(pred, dim=1, prepend=torch.zeros_like(pred[:, :1])) * pixel_scale
    gd = torch.diff(truth, dim=1, prepend=torch.zeros_like(truth[:, :1])) * pixel_scale
    previous_known = torch.cat([torch.ones_like(known[:, :1]), known[:, :-1]], dim=1)
    displaced = (truth * pixel_scale).square().sum(2, keepdim=True).sqrt() > moving_threshold_px
    previously_displaced = torch.cat([torch.zeros_like(displaced[:, :1]), displaced[:, :-1]], dim=1)
    # Include a mover that has stopped: its labeled interval velocity is zero.
    moving = displaced | previously_displaced
    support = known & previous_known & moving
    velocity_error = (pd - gd) / dt[:, :, None, None, None]
    error = (velocity_error.square().sum(2, keepdim=True)
             + epsilon_px_per_second ** 2).sqrt() - epsilon_px_per_second
    return _partition_mean(error / scale_px_per_second, support, sample_weight), support.sum().detach()


def _object_direction_and_parallel_magnitude(
    predicted_displacement: torch.Tensor,
    target_displacement: torch.Tensor,
    support: torch.Tensor,
    sample_weight: torch.Tensor | None,
    *,
    width: int,
    height: int,
    epsilon_px: float,
    magnitude_scale_px: float,
    magnitude_beta_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Instance-balanced counterpart of the pixel mover-vector losses."""
    scale = predicted_displacement.new_tensor([width, height])
    predicted_px = predicted_displacement * scale
    target_px = target_displacement * scale
    target_magnitude = target_px.square().sum(dim=-1).sqrt().clamp_min(1e-6)
    predicted_smooth_norm = (
        predicted_px.square().sum(dim=-1) + epsilon_px * epsilon_px
    ).sqrt()
    dot = (predicted_px * target_px).sum(dim=-1)
    direction_error = 1.0 - dot / (predicted_smooth_norm * target_magnitude)
    target_unit = target_px / target_magnitude[..., None]
    signed_parallel_px = (predicted_px * target_unit).sum(dim=-1)
    magnitude_error = F.smooth_l1_loss(
        signed_parallel_px / magnitude_scale_px,
        target_magnitude / magnitude_scale_px,
        reduction="none",
        beta=magnitude_beta_px / magnitude_scale_px,
    )
    weight = support.to(direction_error.dtype)
    if sample_weight is not None:
        weight = weight * sample_weight.to(weight).view(-1, 1, 1)
    return masked_mean(direction_error, weight), masked_mean(magnitude_error, weight)


def _task_object_affine_motion_losses(
    slot_params: torch.Tensor,
    batch: dict[str, torch.Tensor],
    target_flow: torch.Tensor,
    flow_valid: torch.Tensor,
    config: dict[str, Any],
    sample_weight: torch.Tensor | None,
    *,
    width: int,
    height: int,
) -> dict[str, torch.Tensor]:
    """Directly supervise the affine trajectory of caption-selected input slots.

    The older object-vector terms average the final flow inside a label mask.
    Their gradient is consequently shared by the slot transform, mask overlap and
    dense residual. This loss maps the authoritative task-object label onto the
    deployable first-frame proposals, then scores each matched slot's affine
    displacement directly. Label masks choose loss support only; they are never
    passed to the model or used at inference.
    """
    if slot_params.ndim != 4 or slot_params.shape[-1] != 6:
        raise ValueError("task-object affine supervision requires slot_params [B,K,T,6]")
    proposals = batch.get("instance_masks")
    selected = batch.get("supervision_instance_masks")
    authoritative = batch.get("supervision_object_identity_authoritative")
    if not isinstance(proposals, torch.Tensor) or proposals.ndim != 4:
        raise ValueError("task-object affine supervision requires instance_masks [B,K,H,W]")
    if not isinstance(selected, torch.Tensor) or selected.ndim != 4:
        raise ValueError(
            "task-object affine supervision requires supervision_instance_masks [B,S,H,W]"
        )
    b, k, t, _ = slot_params.shape
    if tuple(proposals.shape) != (b, k, height, width):
        raise ValueError("instance_masks do not match task-object affine slot geometry")
    if selected.shape[0] != b or tuple(selected.shape[-2:]) != (height, width):
        raise ValueError("supervision_instance_masks do not match task-object affine geometry")
    if (
        not isinstance(authoritative, torch.Tensor)
        or tuple(authoritative.shape) != (b,)
        or not bool(torch.isfinite(authoritative).all())
        or not bool(((authoritative == 0) | (authoritative == 1)).all())
    ):
        raise ValueError(
            "task-object affine supervision requires binary "
            "supervision_object_identity_authoritative [B]"
        )
    if tuple(target_flow.shape) != (b, t, 2, height, width):
        raise ValueError("target flow does not match task-object affine geometry")
    if tuple(flow_valid.shape) != (b, t, 1, height, width):
        raise ValueError("flow_valid does not match task-object affine geometry")

    proposal = proposals.to(device=slot_params.device, dtype=torch.float32) > .5
    selected_union = (
        selected.to(device=slot_params.device, dtype=torch.float32) > .5
    ).any(1)
    proposal_area = proposal.sum((-1, -2))
    overlap_mask = proposal & selected_union[:, None]
    overlap = overlap_mask.sum((-1, -2))
    positive_fraction = float(
        config.get("task_object_identity_positive_fraction", .5)
    )
    min_pixels_value = float(config.get("task_object_identity_min_pixels", 4))
    if not math.isfinite(positive_fraction) or not 0 < positive_fraction <= 1:
        raise ValueError("task-object affine positive fraction must lie in (0,1]")
    if (
        not math.isfinite(min_pixels_value)
        or min_pixels_value < 1
        or not min_pixels_value.is_integer()
    ):
        raise ValueError("task-object affine minimum pixels must be a positive integer")
    min_pixels = int(min_pixels_value)
    positive_slot = (
        authoritative.to(slot_params.device).bool()[:, None]
        & (proposal_area > 0)
        & (overlap >= min_pixels)
        & (overlap.float() / proposal_area.clamp_min(1).float() >= positive_fraction)
    )

    # Score the transform over the label/proposal intersection so a SAM mask's
    # background spill does not become a false moving target.
    regression_mask = overlap_mask.float()
    valid = flow_valid[:, :, 0].to(slot_params.device, dtype=torch.float32) > .5
    support_pixels = regression_mask[:, :, None] * valid[:, None].float()
    area = support_pixels.sum((-1, -2))
    valid_cell = positive_slot[:, :, None] & (area > 0)

    ys = torch.linspace(-1., 1., height, device=slot_params.device, dtype=torch.float32)
    xs = torch.linspace(-1., 1., width, device=slot_params.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    denominator = area.clamp_min(1.)
    mean_x = (support_pixels * grid_x).sum((-1, -2)) / denominator
    mean_y = (support_pixels * grid_y).sum((-1, -2)) / denominator
    params = slot_params.float()
    predicted = torch.stack((
        .5 * (params[..., 0] * mean_x + params[..., 1] * mean_y + params[..., 4]),
        .5 * (params[..., 2] * mean_x + params[..., 3] * mean_y + params[..., 5]),
    ), dim=-1)
    target = torch.einsum(
        "bkthw,btchw->bktc", support_pixels, target_flow.float()
    ) / denominator[..., None]

    cell_weight = valid_cell.to(predicted.dtype)
    if sample_weight is not None:
        cell_weight = cell_weight * sample_weight.to(cell_weight).view(-1, 1, 1)
    displacement = masked_mean(
        (predicted - target).square().sum(-1).add(1e-8).sqrt(), cell_weight
    )
    target_px = target * target.new_tensor([width, height])
    moving = target_px.square().sum(-1).sqrt() > float(
        config.get("flow_moving_threshold_px", 1.)
    )
    moving_support = valid_cell & moving
    direction, magnitude = _object_direction_and_parallel_magnitude(
        predicted, target, moving_support, sample_weight,
        width=width, height=height,
        epsilon_px=float(config.get("motion_direction_epsilon_px", .5)),
        magnitude_scale_px=float(config.get("motion_magnitude_scale_px", 16.)),
        magnitude_beta_px=float(config.get("motion_magnitude_beta_px", 1.)),
    )

    horizons = torch.as_tensor(
        batch.get("horizon_seconds"), device=slot_params.device, dtype=torch.float32
    )
    if horizons.ndim == 1:
        horizons = horizons[None].expand(b, -1)
    if tuple(horizons.shape) != (b, t) or not bool(torch.isfinite(horizons).all()):
        raise ValueError(
            "task-object affine trajectory requires finite horizon_seconds [T] or [B,T]"
        )
    dt = torch.diff(horizons, dim=1, prepend=torch.zeros_like(horizons[:, :1]))
    if bool((dt <= 0).any()):
        raise ValueError("task-object affine trajectory horizons must increase")
    scale = predicted.new_tensor([width, height])
    predicted_delta = torch.diff(
        predicted * scale, dim=2, prepend=torch.zeros_like(predicted[:, :, :1])
    )
    target_delta = torch.diff(
        target * scale, dim=2, prepend=torch.zeros_like(target[:, :, :1])
    )
    previous_valid = torch.cat(
        (torch.ones_like(valid_cell[:, :, :1]), valid_cell[:, :, :-1]), dim=2
    )
    previously_moving = torch.cat(
        (torch.zeros_like(moving[:, :, :1]), moving[:, :, :-1]), dim=2
    )
    trajectory_support = valid_cell & previous_valid & (moving | previously_moving)
    velocity_error = (predicted_delta - target_delta) / dt[:, None, :, None]
    epsilon_velocity = float(config.get("trajectory_velocity_epsilon_px", .5))
    velocity_scale = float(config.get("trajectory_velocity_scale_px", 32.))
    if (
        not math.isfinite(epsilon_velocity)
        or epsilon_velocity <= 0
        or not math.isfinite(velocity_scale)
        or velocity_scale <= 0
    ):
        raise ValueError("task-object affine trajectory scales must be finite and positive")
    trajectory_error = (
        velocity_error.square().sum(-1).add(epsilon_velocity ** 2).sqrt()
        - epsilon_velocity
    ) / velocity_scale
    trajectory_weight = trajectory_support.to(trajectory_error.dtype)
    if sample_weight is not None:
        trajectory_weight = trajectory_weight * sample_weight.to(
            trajectory_weight
        ).view(-1, 1, 1)
    trajectory = masked_mean(trajectory_error, trajectory_weight)
    return {
        "task_object_affine_displacement": displacement,
        "task_object_affine_direction": direction,
        "task_object_affine_parallel_magnitude": magnitude,
        "task_object_affine_trajectory": trajectory,
        "task_object_affine_support": valid_cell.sum().detach(),
        "task_object_affine_moving_support": moving_support.sum().detach(),
        "task_object_affine_trajectory_support": trajectory_support.sum().detach(),
    }


def _task_object_flow_motion_losses(
    predicted_flow: torch.Tensor,
    batch: dict[str, torch.Tensor],
    target_flow: torch.Tensor,
    flow_valid: torch.Tensor,
    config: dict[str, Any],
    sample_weight: torch.Tensor | None,
    *,
    width: int,
    height: int,
) -> dict[str, torch.Tensor]:
    """Supervise final flow on the authoritative caption-selected object.

    This is deliberately downstream of the slot-affine plus residual
    decomposition.  The loss therefore measures the exact field exported to
    Stage-2 and cannot improve an internal affine parameter while a residual
    silently cancels (or amplifies) it.  The label mask only defines training
    support and is never an inference input.
    """
    if predicted_flow.ndim != 5 or predicted_flow.shape[2] != 2:
        raise ValueError("task-object flow supervision requires flow [B,T,2,H,W]")
    if target_flow.shape != predicted_flow.shape:
        raise ValueError("task-object flow target must match predicted flow")
    b, t, _, h, w = predicted_flow.shape
    if (h, w) != (height, width):
        raise ValueError("task-object flow geometry does not match configured size")
    if tuple(flow_valid.shape) != (b, t, 1, height, width):
        raise ValueError("task-object flow supervision requires flow_valid [B,T,1,H,W]")
    selected = batch.get("supervision_instance_masks")
    authoritative = batch.get("supervision_object_identity_authoritative")
    if not isinstance(selected, torch.Tensor) or selected.ndim != 4:
        raise ValueError(
            "task-object flow supervision requires supervision_instance_masks [B,S,H,W]"
        )
    if selected.shape[0] != b or tuple(selected.shape[-2:]) != (height, width):
        raise ValueError("supervision_instance_masks do not match task-object flow geometry")
    if (
        not isinstance(authoritative, torch.Tensor)
        or tuple(authoritative.shape) != (b,)
        or not bool(torch.isfinite(authoritative).all())
        or not bool(((authoritative == 0) | (authoritative == 1)).all())
    ):
        raise ValueError(
            "task-object flow supervision requires binary "
            "supervision_object_identity_authoritative [B]"
        )

    selected_union = (
        selected.to(device=predicted_flow.device, dtype=torch.float32) > .5
    ).any(1)
    valid = flow_valid.to(device=predicted_flow.device, dtype=torch.float32) > .5
    support_pixels = (
        selected_union[:, None, None]
        * valid.float()
        * authoritative.to(predicted_flow).view(b, 1, 1, 1, 1)
    )
    # Keep the complete field here.  Averaging the object to one vector before
    # scoring lets opposite pixel errors cancel, so the scalar objective can
    # improve while the exact field exported to Stage-2 gets worse.  Each
    # nonempty sample/horizon is still balanced by ``_partition_mean``.
    support = support_pixels > .5
    displacement_error = (
        (predicted_flow.float() - target_flow.float())
        .square().sum(2, keepdim=True).add(1e-8).sqrt()
    )
    displacement = _partition_mean(
        displacement_error, support.to(displacement_error), sample_weight
    )

    pixel_scale = target_flow.new_tensor([width, height]).view(1, 1, 2, 1, 1)
    moving = (
        (target_flow.float() * pixel_scale).square().sum(2, keepdim=True).sqrt()
        > float(config.get("flow_moving_threshold_px", 1.))
    )
    moving_support = support & moving
    direction, magnitude = _direction_and_parallel_magnitude(
        predicted_flow.float(),
        target_flow.float(),
        moving_support.to(predicted_flow),
        sample_weight,
        width=width,
        height=height,
        epsilon_px=float(config.get("motion_direction_epsilon_px", .5)),
        magnitude_scale_px=float(config.get("motion_magnitude_scale_px", 16.)),
        magnitude_beta_px=float(config.get("motion_magnitude_beta_px", 1.)),
    )

    horizons = torch.as_tensor(
        batch.get("horizon_seconds"), device=predicted_flow.device, dtype=torch.float32
    )
    if horizons.ndim == 1:
        horizons = horizons[None].expand(b, -1)
    if tuple(horizons.shape) != (b, t) or not bool(torch.isfinite(horizons).all()):
        raise ValueError(
            "task-object flow trajectory requires finite horizon_seconds [T] or [B,T]"
        )
    trajectory, trajectory_support = trajectory_segment_loss(
        predicted_flow.float(), target_flow.float(), support.to(predicted_flow),
        horizons, sample_weight,
        width=width, height=height,
        scale_px_per_second=float(config.get("trajectory_velocity_scale_px", 32.)),
        moving_threshold_px=float(config.get("flow_moving_threshold_px", 1.)),
        epsilon_px_per_second=float(config.get("trajectory_velocity_epsilon_px", .5)),
    )
    return {
        "task_object_flow_displacement": displacement,
        "task_object_flow_direction": direction,
        "task_object_flow_parallel_magnitude": magnitude,
        "task_object_flow_trajectory": trajectory,
        "task_object_flow_support": support.sum().detach(),
        "task_object_flow_moving_support": moving_support.sum().detach(),
        "task_object_flow_trajectory_support": trajectory_support.sum().detach(),
    }


def _valid_partition_supports(
    batch: dict[str, torch.Tensor], target: torch.Tensor, config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Validate the v1 sidecar contract and keep unknown labels out of losses.

    ``flow_valid`` and ``visibility_valid`` are independent [B,T,1,H,W]
    supports. ``static_certified`` is supplied by the label producer; it is
    never replaced by the complement of a motion threshold. The instance
    union [B,1,H,W] is a loss-only reference, not a SAM slot-ID correspondence.
    """
    if target.ndim != 5 or target.shape[2] != 2:
        raise ValueError("flow must have shape [B,T,2,H,W]")
    batch_size, horizons, _, height, width = target.shape
    temporal_shape = (batch_size, horizons, 1, height, width)
    spatial_shape = (batch_size, 1, height, width)
    result = {
        key: _unit_support(batch, key, temporal_shape, target)
        for key in (
            "flow_valid", "visibility_valid", "static_certified",
            "motion_target", "motion_valid",
        )
    }
    result["union"] = _unit_support(batch, "supervision_object_mask", spatial_shape, target)
    if "content_valid" in batch:
        content = _unit_support(batch, "content_valid", spatial_shape, target)
        result["flow_valid"] = result["flow_valid"] * content[:, None]
        result["visibility_valid"] = result["visibility_valid"] * content[:, None]
        result["content_valid"] = content
    else:
        result["content_valid"] = torch.ones_like(result["union"])
    # These older losses do not implement this validity contract. Refuse an
    # accidental mixture rather than allowing gradients from unknown labels.
    for key, default in (("object_occupancy", 0.0), ("position", 0.2),
                         ("velocity", 0.05), ("acceleration", 0.01)):
        if float(config.get(key, default)) != 0.0:
            raise ValueError(f"valid_partition_v1 requires {key}=0; its loss is not validity-aware")
    return result


def flow_space_losses(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    sample_weight: torch.Tensor | None = None,
    width: int = 160,
    height: int = 160,
) -> dict[str, torch.Tensor]:
    """Supervise the decoded flow field, gate, visibility and object mask.

    Args:
        prediction: output of :class:`motion_jepa.flow_dit.LatentFlowDecoder`.
        batch: Stage-1 targets with ``flow`` already normalised by width/height.
        sample_weight: ``[B]`` per-sample weight (typically the sigma weight).

    ``supervision_mode=valid_partition_v1`` additionally requires explicit
    ``flow_valid``, ``visibility_valid``, ``static_certified`` and
    ``supervision_object_mask``. Static/moving cutoffs are configured through
    ``flow_static_threshold_px`` (default 0.5) and
    ``flow_moving_threshold_px`` (default 1.0). Valid gray-band or uncertified
    low-motion labels retain regression gradients in a separate ``weak``
    group, but do not become negative motion-gate labels. Group budgets live
    in ``flow_partition_weights`` (moving/static/weak=1, background=0.02).
    """
    # A separate dispatch is intentional: zeroing legacy weights still leaves
    # implicit defaults and unnecessarily computes unrelated auxiliary heads.
    profile = config.get("profile", "legacy")
    if profile == "unified_trajectory_v1":
        from .flow_dit_unified import unified_trajectory_losses
        return unified_trajectory_losses(prediction, batch, config, sample_weight,
                                         width=width, height=height)
    if profile == "clean_trajectory_v1":
        from .flow_dit_clean import clean_trajectory_losses
        return clean_trajectory_losses(prediction, batch, config, sample_weight,
                                       width=width, height=height)
    if profile != "legacy":
        from .flow_dit_minimal import PROFILE, minimal5_losses
        if profile != PROFILE:
            raise ValueError(f"unknown loss profile {profile!r}")
        return minimal5_losses(prediction, batch, config, sample_weight)
    predicted_flow = prediction["flow"]
    target_flow = batch["flow"]
    object_mask = batch["object_mask"]
    visibility = batch["visibility"]
    mode = str(config.get("supervision_mode", "legacy"))
    if mode not in {"legacy", "valid_partition_v1"}:
        raise ValueError(f"unknown supervision_mode {mode!r}")
    reliable = mode == "valid_partition_v1"
    supports = _valid_partition_supports(batch, target_flow, config) if reliable else None
    loss_prediction = predicted_flow
    if reliable:
        # Mask before arithmetic: an unknown NaN placeholder must not turn a
        # valid loss into NaN through 0 * NaN, nor receive a regression gradient.
        known = supports["flow_valid"] > 0
        target_flow = torch.where(known, target_flow, torch.zeros_like(target_flow))
        loss_prediction = torch.where(known, predicted_flow, torch.zeros_like(predicted_flow))
        if not bool(torch.isfinite(target_flow).all()):
            raise ValueError("flow has nonfinite values on flow_valid support")

    scored = object_mask[:, None] * visibility
    background = (1.0 - object_mask)[:, None].expand_as(visibility)
    if sample_weight is not None:
        shape = (-1,) + (1,) * (scored.ndim - 1)
        weight = sample_weight.view(shape).to(scored.dtype)
        scored = scored * weight
        background = background * weight

    # The 1px moving threshold matches the evaluator's motion statistics, so
    # the loss and the reported motion precision/recall agree on what
    # "moving" means.
    magnitude_pixels = (
        target_flow.detach() * target_flow.new_tensor([width, height]).view(1, 1, 2, 1, 1)
    ).square().sum(dim=2, keepdim=True).sqrt()
    moving = (magnitude_pixels > 1.0).float()
    partition_support: dict[str, torch.Tensor] = {}
    partition_losses: dict[str, torch.Tensor] = {}
    if reliable:
        static_threshold = float(config.get("flow_static_threshold_px", 0.5))
        moving_threshold = float(config.get("flow_moving_threshold_px", 1.0))
        if not (math.isfinite(static_threshold) and math.isfinite(moving_threshold)
                and 0 <= static_threshold < moving_threshold):
            raise ValueError("flow thresholds must satisfy 0 <= static < moving and be finite")
        # Use the explicit label contract.  Re-thresholding flow here used to
        # resurrect robot/occluder motion that an authoritative object label had
        # intentionally marked unknown.
        moving = supports["motion_target"]
        if bool((moving > supports["flow_valid"]).any()):
            raise ValueError("motion_target must be a subset of flow_valid")
        if bool((supports["motion_valid"] < moving).any()):
            raise ValueError("motion_target must be a subset of motion_valid")
        static = supports["static_certified"] * (magnitude_pixels <= static_threshold)
        union = supports["union"][:, None]
        valid = supports["flow_valid"]
        partition_support = {
            "moving": valid * moving,
            "static": valid * static * union,
            "background": valid * static * (1.0 - union),
            # Includes the threshold gray band and trustworthy small flow
            # without a certified activity label. The target is NOT set to 0.
            "weak": valid * (1.0 - moving - static).clamp_min(0.0),
        }

    moving_weight = float(config.get("moving_weight", 2.0))
    background_weight = float(config.get("background_motion_weight", 0.02))
    flow_weight = scored * (1.0 + moving_weight * moving) + background_weight * background

    # Vector Charbonnier on the 2-norm, matching the endpoint error the
    # evaluator reports.
    #
    # The predecessor was ``smooth_l1_loss(..., beta=0.02)`` applied
    # per-component. 0.02 in normalised units is 3.2 px on a 160 px grid, above
    # most target magnitudes, so essentially the whole objective sat in the
    # quadratic branch: a weighted MSE on u and v independently. Two
    # consequences. Its minimiser is the conditional *mean*, and a
    # scenario-mean lookup table with zero parameters can beat the trained
    # model -- the model is rewarded for regressing toward exactly that. And penalising
    # u and v separately is not the metric: it prefers halving both components
    # over getting the direction right.
    #
    # Charbonnier ``sqrt(|d|^2 + eps^2)`` is the smooth surrogate for the vector
    # norm, so its minimiser is the conditional geometric median. eps is small
    # relative to real displacements (0.5 px) and only rounds off the kink at
    # zero, where a true L1 would otherwise fight the optimiser on the static
    # pixels that are most of the support.
    #
    # Kept switchable because the swap is not a free win: holding moving_weight
    # fixed, Charbonnier plus direct affine supervision can do worse than the
    # SmoothL1 predecessor on val (relative_epe_moving).
    # Without the old path reachable, moving_weight and the loss form cannot be
    # told apart in any later comparison.
    if str(config.get("flow_loss", "charbonnier")) == "smooth_l1":
        flow_error = F.smooth_l1_loss(
            loss_prediction, target_flow, reduction="none", beta=0.02
        )
    else:
        epsilon = float(config.get("flow_epsilon_px", 0.5)) / max(width, height)
        difference = loss_prediction - target_flow
        flow_error = (
            difference.square().sum(dim=2, keepdim=True) + epsilon * epsilon
        ).sqrt()
    if reliable:
        budgets = {"moving": 1.0, "static": 1.0, "background": 0.02, "weak": 1.0}
        configured = config.get("flow_partition_weights", {})
        if not isinstance(configured, dict) or set(configured) - set(budgets):
            raise ValueError("flow_partition_weights accepts moving/static/background/weak only")
        budgets.update({key: float(value) for key, value in configured.items()})
        if any(not math.isfinite(value) or value < 0 for value in budgets.values()):
            raise ValueError("flow_partition_weights must be finite and nonnegative")
        partition_losses = {
            key: _partition_mean(flow_error, support, sample_weight)
            for key, support in partition_support.items()
        }
        flow_loss = sum(budgets[key] * value for key, value in partition_losses.items())
    else:
        flow_loss = masked_mean(flow_error, flow_weight)

    vector_weights = {
        key: float(config.get(key, 0.0))
        for key in (
            "moving_direction", "moving_parallel_magnitude",
            "object_motion_direction", "object_motion_parallel_magnitude",
        )
    }
    if any(not math.isfinite(value) or value < 0 for value in vector_weights.values()):
        raise ValueError("motion direction/magnitude weights must be finite and nonnegative")
    if any(vector_weights.values()) and not reliable:
        raise ValueError("motion direction/magnitude losses require valid_partition_v1")
    direction_epsilon = float(config.get("motion_direction_epsilon_px", 0.5))
    magnitude_scale = float(config.get("motion_magnitude_scale_px", 16.0))
    magnitude_beta = float(config.get("motion_magnitude_beta_px", 1.0))
    if any(vector_weights.values()) and (
        not math.isfinite(direction_epsilon) or direction_epsilon <= 0
        or not math.isfinite(magnitude_scale) or magnitude_scale <= 0
        or not math.isfinite(magnitude_beta) or magnitude_beta <= 0
    ):
        raise ValueError("motion vector loss scales must be finite and positive")
    moving_direction = predicted_flow.new_zeros(())
    moving_parallel_magnitude = predicted_flow.new_zeros(())
    if reliable and (vector_weights["moving_direction"] or vector_weights["moving_parallel_magnitude"]):
        moving_direction, moving_parallel_magnitude = _direction_and_parallel_magnitude(
            loss_prediction, target_flow, partition_support["moving"], sample_weight,
            width=width, height=height, epsilon_px=direction_epsilon,
            magnitude_scale_px=magnitude_scale, magnitude_beta_px=magnitude_beta,
        )

    trajectory_weight = float(config.get("trajectory_segment", 0.0))
    if not math.isfinite(trajectory_weight) or trajectory_weight < 0:
        raise ValueError("trajectory_segment weight must be finite and nonnegative")
    trajectory_segment = predicted_flow.new_zeros(())
    trajectory_support = predicted_flow.new_zeros(())
    if trajectory_weight:
        if not reliable or "horizon_seconds" not in batch:
            raise ValueError("trajectory_segment requires valid_partition_v1 and batch horizon_seconds")
        trajectory_segment, trajectory_support = trajectory_segment_loss(
            loss_prediction, target_flow, supports["flow_valid"], batch["horizon_seconds"],
            sample_weight, width=width, height=height,
            scale_px_per_second=float(config.get("trajectory_velocity_scale_px", 32.0)),
            moving_threshold_px=float(config.get("flow_moving_threshold_px", 1.0)),
        )

    # The gate is what lets the dense decoder say "this pixel does not move"
    # without having to cancel a regression head against itself.  The slot
    # decoder gets that for free -- a static slot has the identity transform and
    # therefore exactly zero flow -- so it emits a constant gate.  Scoring a
    # constant would add a fixed offset to the total and
    # make the two decoders' loss curves incomparable, which is precisely what
    # the head-to-head is meant to measure.
    gate_source = scored + background_weight * background
    if reliable:
        # Activity labels cover only definite movers and certified static
        # regions. Unknown/weak labels still train flow, not this classifier.
        gate_source = (partition_support["moving"] + partition_support["static"]
                       + partition_support["background"])
        if sample_weight is not None:
            gate_source = gate_source * sample_weight.to(gate_source).view(-1, 1, 1, 1, 1)
    gate_logits = prediction["gate_logits"]
    gate_loss = (
        _balanced_bce(gate_logits, moving, gate_source)
        if gate_logits.requires_grad
        else predicted_flow.new_zeros(())
    )

    visibility_source = object_mask[:, None].expand_as(visibility)
    if reliable:
        visibility_source = supports["visibility_valid"]
        visibility = torch.where(visibility_source > 0, visibility, torch.zeros_like(visibility))
        if not bool(torch.isfinite(visibility).all()):
            raise ValueError("visibility has nonfinite values on visibility_valid support")
    if sample_weight is not None:
        visibility_source = visibility_source * sample_weight.view(
            (-1,) + (1,) * (visibility_source.ndim - 1)
        ).to(visibility_source.dtype)
    visibility_loss = _balanced_bce(
        prediction["visibility_logits"], visibility, visibility_source
    )

    object_mask_loss = _balanced_bce(
        prediction["object_mask_logits"],
        supports["union"] if reliable else object_mask,
        supports["content_valid"] if reliable else torch.ones_like(object_mask),
    )

    if reliable:
        # Disabled legacy terms must not even evaluate unknown center labels:
        # multiplying a resulting NaN by a zero coefficient would still fail.
        position = velocity = acceleration = predicted_flow.new_zeros(())
    else:
        position, velocity, acceleration = _kinematic_losses(
            predicted_flow, batch, config, sample_weight=sample_weight
        )

    # Object-level supervision: what the pixel loss cannot see.
    #
    # The pixel Charbonnier treats every pixel independently -- it has no
    # concept of "this object, as a whole, moved 12px down-left" or "the object
    # must END UP over there". The instance masks were requested precisely so
    # motion could be supervised at the object level; these two terms close
    # that gap using nothing but the masks and the ground-truth flow, so they
    # work identically on Physion and PhyCo (no reliance on the Physion-only
    # projected 3D centers).
    object_displacement = predicted_flow.new_zeros(())
    object_occupancy = predicted_flow.new_zeros(())
    object_motion_direction = predicted_flow.new_zeros(())
    object_motion_parallel_magnitude = predicted_flow.new_zeros(())
    label_instance_masks = batch.get("supervision_instance_masks", batch.get("instance_masks"))
    if label_instance_masks is not None:
        masks = label_instance_masks.to(predicted_flow.dtype)         # [B,K,H,W]
        area = masks.sum(dim=(-1, -2)).clamp_min(1.0)                 # [B,K]
        occupied = (masks.sum(dim=(-1, -2)) > 0).to(predicted_flow.dtype)
        if float(config.get("object_displacement", 0.0)) > 0:
            # Per-instance centroid displacement, every horizon: the L1 between
            # the mean predicted and mean true flow over the instance's own
            # pixels. This is exactly the quantity instance_ade_px scores, made
            # differentiable and dataset-agnostic.
            if reliable:
                valid = supports["flow_valid"]
                valid_area = torch.einsum("bkhw,bthw->bkt", masks, valid[:, :, 0])
                denominator = valid_area.clamp_min(1e-6)[..., None]
                pred_disp = torch.einsum(
                    "bkhw,btchw->bktc", masks, loss_prediction * valid
                ) / denominator
                true_disp = torch.einsum(
                    "bkhw,btchw->bktc", masks, target_flow * valid
                ) / denominator
                weight_kt = (valid_area > 0).to(predicted_flow.dtype)
            else:
                pred_disp = torch.einsum("bkhw,btchw->bktc", masks, predicted_flow) / area[..., None, None]
                true_disp = torch.einsum("bkhw,btchw->bktc", masks, target_flow) / area[..., None, None]
                weight_kt = occupied[:, :, None].expand(pred_disp.shape[:3])
            if sample_weight is not None:
                weight_kt = weight_kt * sample_weight.view(-1, 1, 1).to(weight_kt.dtype)
            error = (pred_disp - true_disp).square().sum(-1).add(1e-8).sqrt()
            object_displacement = masked_mean(error, weight_kt)
        if reliable and (
            vector_weights["object_motion_direction"]
            or vector_weights["object_motion_parallel_magnitude"]
        ):
            valid = supports["flow_valid"]
            valid_area = torch.einsum("bkhw,bthw->bkt", masks, valid[:, :, 0])
            denominator = valid_area.clamp_min(1e-6)[..., None]
            pred_disp = torch.einsum(
                "bkhw,btchw->bktc", masks, loss_prediction * valid
            ) / denominator
            true_disp = torch.einsum(
                "bkhw,btchw->bktc", masks, target_flow * valid
            ) / denominator
            object_target_px = true_disp * true_disp.new_tensor([width, height])
            moving_object = (
                object_target_px.square().sum(-1).sqrt()
                > float(config.get("flow_moving_threshold_px", 1.0))
            )
            object_support = (valid_area > 0) & moving_object
            object_motion_direction, object_motion_parallel_magnitude = (
                _object_direction_and_parallel_magnitude(
                    pred_disp, true_disp, object_support, sample_weight,
                    width=width, height=height, epsilon_px=direction_epsilon,
                    magnitude_scale_px=magnitude_scale, magnitude_beta_px=magnitude_beta,
                )
            )
        if float(config.get("object_occupancy", 0.0)) > 0:
            # Where does the object END UP: splat the union mask through the
            # predicted vs the true flow and penalise the occupancy gap. This
            # is the differentiable twin of warped_mask_iou, and it is the only
            # term here that cares about the destination rather than the
            # per-pixel vector.
            from .video import forward_splat_mask

            scale_hw = predicted_flow.new_tensor([width, height]).view(1, 1, 2, 1, 1)
            union = batch["object_mask"].to(predicted_flow.dtype)
            predicted_occ = forward_splat_mask(union, predicted_flow * scale_hw)
            true_occ = forward_splat_mask(union, target_flow * scale_hw)
            gap = (predicted_occ - true_occ).abs()
            occ_weight = (
                sample_weight.view(-1, 1, 1, 1, 1).to(gap.dtype).expand_as(gap)
                if sample_weight is not None
                else torch.ones_like(gap)
            )
            object_occupancy = masked_mean(gap, occ_weight)

    # Slot decoder only: with identity-initialised transforms the flow loss
    # cannot reach the mask branch on step 0 (d flow / d mask = params = 0), so
    # the masks need their own permutation-invariant supervision.
    slot_mask = predicted_flow.new_zeros(())
    slot_affine = predicted_flow.new_zeros(())
    if "slot_masks" in prediction and "instance_id" in batch:
        from .flow_dit_slots import hungarian_mask_loss

        want_affine = (
            "slot_params" in prediction
            and "affine_by_instance" in batch
            and "affine_valid" in batch
            and float(config.get("slot_affine", 0.0)) > 0
        )
        if want_affine and "instance_mask_ids" in batch:
            # Oracle-mask arm: slot k pooled from and rendered with instance
            # mask k, so the correspondence is fixed by construction, not by
            # whatever the learned masks happen to overlap. Supervising through
            # a Hungarian solve on the learned masks here would score slot k's
            # params against a *different* object than the one it rendered.
            slot_mask = hungarian_mask_loss(
                prediction["slot_masks"], batch["instance_id"]
            )
            mask_ids = batch["instance_mask_ids"]
            assignment = []
            for index in range(mask_ids.shape[0]):
                slots = [k for k in range(mask_ids.shape[1]) if int(mask_ids[index, k]) > 0]
                assignment.append(
                    (index, slots, [int(mask_ids[index, k]) for k in slots])
                )
            slot_affine = _slot_affine_loss(prediction["slot_params"], batch, assignment)
        else:
            result = hungarian_mask_loss(
                prediction["slot_masks"], batch["instance_id"], return_assignment=want_affine
            )
            if want_affine:
                slot_mask, assignment = result
                slot_affine = _slot_affine_loss(prediction["slot_params"], batch, assignment)
            else:
                slot_mask = result

    # Opt-in activity labels follow the actual input instances at every query
    # time. They are independent of learned-mask matching and the legacy gate.
    # Leave the complete historical loss/return contract untouched when this
    # branch is absent (including checkpoints trained before activity heads).
    activity_terms: dict[str, torch.Tensor] = {}
    activity_weights = {key: float(config.get(key, 0.0))
                        for key in ("object_activity", "pixel_activity")}
    if any(not math.isfinite(value) or value < 0 for value in activity_weights.values()):
        raise ValueError("object_activity and pixel_activity weights must be finite and nonnegative")
    if any(activity_weights.values()) or "slot_activity_logits" in prediction:
        if not reliable:
            raise ValueError("activity supervision requires supervision_mode=valid_partition_v1")
        from .flow_dit_activity import activity_loss_terms

        activity_terms = activity_loss_terms(
            prediction, batch, width, height, config, sample_weight=sample_weight
        )

    dense_mask_terms: dict[str, torch.Tensor] = {}
    dense_mask_weight = float(config.get("dense_motion_mask", 0.0))
    task_object_slot_weight = float(config.get("task_object_slot_activity", 0.0))
    task_object_identity_weight = float(config.get("task_object_identity", 0.0))
    if not math.isfinite(dense_mask_weight) or dense_mask_weight < 0:
        raise ValueError("dense_motion_mask weight must be finite and nonnegative")
    if not math.isfinite(task_object_slot_weight) or task_object_slot_weight < 0:
        raise ValueError("task_object_slot_activity weight must be finite and nonnegative")
    if not math.isfinite(task_object_identity_weight) or task_object_identity_weight < 0:
        raise ValueError("task_object_identity weight must be finite and nonnegative")
    if dense_mask_weight > 0:
        if not reliable:
            raise ValueError("dense mask supervision requires supervision_mode=valid_partition_v1")
        from .flow_dit_dense_mask import dense_mask_loss_terms

        dense_mask_terms = dense_mask_loss_terms(
            prediction, batch, width, height, config, sample_weight=sample_weight
        )

    task_affine_weights = {
        key: float(config.get(key, 0.0))
        for key in (
            "task_object_affine_displacement",
            "task_object_affine_direction",
            "task_object_affine_parallel_magnitude",
            "task_object_affine_trajectory",
        )
    }
    if any(not math.isfinite(value) or value < 0 for value in task_affine_weights.values()):
        raise ValueError("task-object affine loss weights must be finite and nonnegative")
    task_affine_terms: dict[str, torch.Tensor] = {}
    if any(task_affine_weights.values()):
        if not reliable:
            raise ValueError("task-object affine supervision requires valid_partition_v1")
        slot_params = prediction.get("slot_params")
        if not isinstance(slot_params, torch.Tensor):
            raise ValueError("task-object affine supervision requires a slot decoder")
        task_affine_terms = _task_object_affine_motion_losses(
            slot_params, batch, target_flow, supports["flow_valid"], config,
            sample_weight, width=width, height=height,
        )

    task_flow_weights = {
        key: float(config.get(key, 0.0))
        for key in (
            "task_object_flow_displacement",
            "task_object_flow_direction",
            "task_object_flow_parallel_magnitude",
            "task_object_flow_trajectory",
        )
    }
    if any(not math.isfinite(value) or value < 0 for value in task_flow_weights.values()):
        raise ValueError("task-object final-flow loss weights must be finite and nonnegative")
    task_flow_terms: dict[str, torch.Tensor] = {}
    if any(task_flow_weights.values()):
        if not reliable:
            raise ValueError("task-object final-flow supervision requires valid_partition_v1")
        task_flow_terms = _task_object_flow_motion_losses(
            loss_prediction, batch, target_flow, supports["flow_valid"], config,
            sample_weight, width=width, height=height,
        )

    # A separately weighted consistency term makes the predicted mask matter to
    # motion learning without allowing it to choose easier GT support.  It is
    # evaluated on the exact same fixed moving/static/background/weak
    # partitions as raw flow, so predicting zero mask cannot hide a mistake.
    gated_flow_weight = float(config.get("gated_flow_charbonnier", 0.0))
    if not math.isfinite(gated_flow_weight) or gated_flow_weight < 0:
        raise ValueError("gated_flow_charbonnier weight must be finite and nonnegative")
    gated_flow_charbonnier = predicted_flow.new_zeros(())
    if gated_flow_weight:
        if not reliable or not dense_mask_terms:
            raise ValueError(
                "gated_flow_charbonnier requires dense mask and valid_partition_v1"
            )
        gated_field = str(config.get("gated_flow_field", "flow_activity_gated"))
        if gated_field not in {"flow_activity_gated", "flow_task_object_gated"}:
            raise ValueError("gated_flow_field must name a supported gated flow")
        gated_prediction = prediction.get(gated_field)
        if not isinstance(gated_prediction, torch.Tensor) or gated_prediction.shape != predicted_flow.shape:
            raise ValueError(f"{gated_field} must match raw flow shape")
        known = supports["flow_valid"] > 0
        gated_prediction = torch.where(
            known, gated_prediction, torch.zeros_like(gated_prediction)
        )
        if not bool(torch.isfinite(gated_prediction).all()):
            raise ValueError(f"{gated_field} is nonfinite on flow_valid support")
        epsilon = float(config.get("flow_epsilon_px", 0.5)) / max(width, height)
        gated_error = (
            (gated_prediction - target_flow).square().sum(dim=2, keepdim=True)
            + epsilon * epsilon
        ).sqrt()
        configured = config.get("flow_partition_weights", {})
        gated_budgets = {"moving": 1.0, "static": 1.0, "background": .02, "weak": 1.0}
        gated_budgets.update({key: float(value) for key, value in configured.items()})
        gated_flow_charbonnier = sum(
            gated_budgets[key] * _partition_mean(gated_error, support, sample_weight)
            for key, support in partition_support.items()
        )

    from .flow_dit_coarse import coarse_motion_loss_terms
    coarse_batch = {**batch, "flow_valid": supports["flow_valid"]} if reliable else batch
    coarse_total, coarse_terms = coarse_motion_loss_terms(prediction, coarse_batch, config, sample_weight)
    total = coarse_total + (
        float(config.get("flow_charbonnier", 1.0)) * flow_loss
        + float(config.get("gate_bce", 0.3)) * gate_loss
        + float(config.get("visibility_bce", 0.1)) * visibility_loss
        + float(config.get("object_mask_bce", 0.1)) * object_mask_loss
        + float(config.get("position", 0.2)) * position
        + float(config.get("velocity", 0.05)) * velocity
        + float(config.get("acceleration", 0.01)) * acceleration
        + float(config.get("slot_mask", 0.5)) * slot_mask
        + float(config.get("slot_affine", 0.0)) * slot_affine
        + float(config.get("object_displacement", 0.0)) * object_displacement
        + float(config.get("object_occupancy", 0.0)) * object_occupancy
        + vector_weights["moving_direction"] * moving_direction
        + vector_weights["moving_parallel_magnitude"] * moving_parallel_magnitude
        + vector_weights["object_motion_direction"] * object_motion_direction
        + vector_weights["object_motion_parallel_magnitude"] * object_motion_parallel_magnitude
        + trajectory_weight * trajectory_segment
        + gated_flow_weight * gated_flow_charbonnier
    )
    for key, coefficient in activity_weights.items():
        if key in activity_terms:
            total = total + coefficient * activity_terms[key]
    for key, coefficient in task_affine_weights.items():
        if coefficient:
            total = total + coefficient * task_affine_terms[key]
    for key, coefficient in task_flow_weights.items():
        if coefficient:
            total = total + coefficient * task_flow_terms[key]
    if dense_mask_terms:
        total = total + dense_mask_weight * dense_mask_terms["dense_motion_mask"]
    if task_object_slot_weight:
        if "task_object_slot_activity" not in dense_mask_terms:
            raise ValueError(
                "task_object_slot_activity requires dense task-object supervision"
            )
        total = total + task_object_slot_weight * dense_mask_terms[
            "task_object_slot_activity"
        ]
    if task_object_identity_weight:
        if "task_object_identity" not in dense_mask_terms:
            raise ValueError(
                "task_object_identity requires dense task-object supervision"
            )
        total = total + task_object_identity_weight * dense_mask_terms[
            "task_object_identity"
        ]
    result = {
        "aux_total": total,
        "slot_mask": slot_mask,
        "slot_affine": slot_affine,
        "object_displacement": object_displacement,
        "object_occupancy": object_occupancy,
        "flow": flow_loss,
        "gate": gate_loss,
        "visibility": visibility_loss,
        "object_mask": object_mask_loss,
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "trajectory_segment": trajectory_segment,
        "trajectory_segment_support": trajectory_support,
    }
    if gated_flow_weight:
        result["gated_flow_charbonnier"] = gated_flow_charbonnier
    if any(vector_weights.values()):
        result.update(
            moving_direction=moving_direction,
            moving_parallel_magnitude=moving_parallel_magnitude,
            object_motion_direction=object_motion_direction,
            object_motion_parallel_magnitude=object_motion_parallel_magnitude,
        )
    result.update(activity_terms)
    result.update(dense_mask_terms)
    result.update(task_affine_terms)
    result.update(task_flow_terms)
    result.update(coarse_terms)
    if reliable:
        for key, value in partition_losses.items():
            result[f"partition_{key}"] = value
            result[f"support_{key}"] = partition_support[key].detach().sum()
    return result


_CONTROL_POINTS = (
    (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
    (0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0),
)


def _affine_control_displacement(params: torch.Tensor) -> torch.Tensor:
    """Displacement of a fixed control-point set under ``[da,db,dc,dd,tx,ty]``.

    Mirrors ``affine_flow`` exactly, including its ``* 0.5`` conversion from the
    ``[-1,1]`` grid span to fraction-of-frame, so the returned values are in the
    same units as the flow the rest of this module scores.

    Control points rather than the raw six numbers because the parameters are
    not commensurable: ``da..dd`` are dimensionless shape terms while ``tx,ty``
    are displacements, so an L1 on the vector silently weights a shear against a
    translation at whatever ratio the parameterisation happens to imply.
    Transformed points are all displacements, and their error is the quantity
    the evaluator actually measures.

    Args:
        params: ``[..., 6]``.
    Returns:
        ``[..., P, 2]`` displacement per control point.
    """
    grid = params.new_tensor(_CONTROL_POINTS)                    # [P,2]
    grid_x, grid_y = grid[:, 0], grid[:, 1]
    da, db, dc, dd, tx, ty = (params[..., index, None] for index in range(6))
    x = da * grid_x + db * grid_y + tx
    y = dc * grid_x + dd * grid_y + ty
    return torch.stack((x, y), dim=-1) * 0.5


def _slot_affine_loss(
    slot_params: torch.Tensor,
    batch: dict[str, torch.Tensor],
    assignment: list[tuple[int, list[int], list[int]]],
) -> torch.Tensor:
    """Direct supervision for the per-slot affine transforms.

    Those 1,542 parameters currently learn only through the rendered flow field,
    behind a soft mask that is itself still being learned -- the longest possible
    path for the quantity that decides the whole prediction. The ground-truth
    affine is a least-squares fit of the true flow inside each instance, so this
    is the same information the pixel loss carries, delivered without the
    intermediary.

    Supervision follows the mask branch's Hungarian assignment, so slot k is
    scored against whichever instance it was already claiming. Unmatched slots
    are left alone: nothing says how many objects the scene has, and forcing
    spare slots toward identity would fight the mask loss.
    """
    reference = batch["affine_by_instance"]
    valid = batch["affine_valid"]
    if valid.ndim != 3:
        raise ValueError(
            "affine_valid must be [batch, instance, horizon]; regenerate v1 "
            "affine sidecars with scripts/cache_stage1_affine.py"
        )
    errors = []
    for index, slots, instances in assignment:
        for slot, instance in zip(slots, instances):
            if instance >= valid.shape[1]:
                continue
            horizon_valid = valid[index, instance].bool()
            if not bool(horizon_valid.any()):
                continue
            predicted = _affine_control_displacement(slot_params[index, slot])
            target = _affine_control_displacement(
                reference[index, instance].to(slot_params.dtype)
            )
            per_horizon = (
                (predicted - target).square().sum(-1).add(1e-8).sqrt().mean(-1)
            )
            errors.append(per_horizon[horizon_valid].mean())
    if not errors:
        return slot_params.new_zeros(())
    return torch.stack(errors).mean()


def _kinematic_losses(
    predicted_flow: torch.Tensor,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    sample_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-instance position / velocity / acceleration, mirroring ``losses.py:117-168``.

    These are the terms that move ``instance_ade_px`` / ``instance_fde_px``,
    which are separate gate conditions from dense EPE -- a model can be dense-
    accurate and still drift on object centroids.
    """
    zero = predicted_flow.new_zeros(())
    if not all(key in batch for key in ("instance_id", "centers", "center_valid")):
        return zero, zero, zero

    max_instances = batch["centers"].shape[1]
    predicted_surface, source_present = instance_trajectory_centers(
        predicted_flow, batch["instance_id"], max_instances
    )
    source_surface, _ = instance_trajectory_centers(
        torch.zeros_like(predicted_flow), batch["instance_id"], max_instances
    )
    source_object = batch["centers"][:, :, 0].to(predicted_surface.dtype)[:, None]
    predicted_centers = source_object + (predicted_surface - source_surface)
    target_centers = batch["centers"][:, :, 1:].permute(0, 2, 1, 3).to(predicted_centers.dtype)
    target_valid = batch["center_valid"][:, :, 1:].permute(0, 2, 1)
    position_valid = (target_valid & source_present[:, None]).float()
    if sample_weight is not None:
        position_valid = position_valid * sample_weight.view(-1, 1, 1).to(position_valid.dtype)

    position = masked_mean(
        F.smooth_l1_loss(predicted_centers, target_centers, reduction="none", beta=0.02),
        position_valid,
    )
    delta_time = float(config.get("delta_time", 0.2))
    predicted_velocity = (predicted_centers[:, 1:] - predicted_centers[:, :-1]) / delta_time
    target_velocity = (target_centers[:, 1:] - target_centers[:, :-1]) / delta_time
    velocity_valid = position_valid[:, 1:] * position_valid[:, :-1]
    velocity = masked_mean(
        F.smooth_l1_loss(predicted_velocity, target_velocity, reduction="none", beta=0.05),
        velocity_valid,
    )
    predicted_acceleration = (predicted_velocity[:, 1:] - predicted_velocity[:, :-1]) / delta_time
    target_acceleration = (target_velocity[:, 1:] - target_velocity[:, :-1]) / delta_time
    acceleration_valid = velocity_valid[:, 1:] * velocity_valid[:, :-1]
    acceleration = masked_mean(
        F.smooth_l1_loss(
            predicted_acceleration, target_acceleration, reduction="none", beta=0.1
        ),
        acceleration_valid,
    )
    return position, velocity, acceleration
