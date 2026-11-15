"""Object-balanced coarse motion objectives on fixed, trusted GT support.

The affine term compares the best affine components of rendered displacement
fields, not raw matrix coefficients or a learned mask. Temporal supervision
uses the SAME t0 pixels at both endpoints and physical, possibly unequal dt.
"""
from __future__ import annotations

import math
import torch


COARSE_KEYS = ("displacement", "relative_displacement", "direction", "magnitude", "affine", "velocity",
               "direction_margin", "segment_direction", "field_direction")


def validate_coarse_config(config):
    weights = {k: float(config.get("coarse_motion_" + k, 0.)) for k in COARSE_KEYS}
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("coarse motion weights must be finite and nonnegative")
    if not any(weights.values()):
        return weights
    if config.get("supervision_mode") != "valid_partition_v1":
        raise ValueError("coarse motion requires valid_partition_v1")
    for key, default in [("coarse_motion_scale_px", 16.), ("coarse_velocity_scale_px_s", 32.),
                         ("coarse_direction_epsilon_px", .5), ("coarse_min_pixels", 4.),
                         ("coarse_relative_floor_px", 4.)]:
        value = float(config.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    margin = float(config.get("coarse_direction_min_projection", .25))
    if not math.isfinite(margin) or not 0 < margin <= 1:
        raise ValueError("coarse_direction_min_projection must lie in (0,1]")
    objective = config.get("coarse_direction_objective", "smooth_cosine")
    if objective not in ("smooth_cosine", "angular_hinge"):
        raise ValueError("coarse_direction_objective must be smooth_cosine or angular_hinge")
    angle = float(config.get("coarse_direction_tolerance_degrees", 30.))
    if not math.isfinite(angle) or not 0 < angle < 90:
        raise ValueError("coarse_direction_tolerance_degrees must lie in (0,90)")
    rw = float(config.get("coarse_raw_weight", .5))
    gw = float(config.get("coarse_gated_weight", 1.))
    if any(not math.isfinite(v) or v < 0 for v in [rw, gw]) or rw + gw <= 0:
        raise ValueError("coarse raw/gated weights must be nonnegative with positive sum")
    if not isinstance(config.get("coarse_gate_only", False), bool):
        raise ValueError("coarse_gate_only must be an explicit boolean")
    return weights


def _mean(values, support, sample_weight):
    # Average horizons, then objects, then samples: size/duration cannot buy
    # one object more budget. Empty cells and zero-weight samples are excluded.
    weights = support.to(values.dtype)
    per_object = (values * weights).sum(-1) / weights.sum(-1).clamp_min(1)
    objects = weights.sum(-1) > 0
    per_sample = (per_object * objects).sum(-1) / objects.sum(-1).clamp_min(1)
    samples = objects.any(-1).to(values.dtype)
    if sample_weight is not None:
        samples = samples * sample_weight.to(samples)
    return (per_sample * samples).sum() / samples.sum().clamp_min(1e-8)


def _direction_cost(projection, vector, dim, config):
    """Optional angle tolerance without rewarding excess speed on a correct ray.

    Legacy smooth cosine is retained for reproducibility. Its epsilon inside
    sqrt leaves a speed-dependent loss even for exactly aligned vectors.
    The hinge uses an exact norm above the numerical floor. Projection margin
    remains a separate term to provide a gradient at zero/exact reversal.
    """
    eps = float(config.get("coarse_direction_epsilon_px", .5))
    objective = config.get("coarse_direction_objective", "smooth_cosine")
    if objective == "smooth_cosine":
        return 1 - projection / (vector.square().sum(dim) + eps**2).sqrt()
    if objective != "angular_hinge":
        raise ValueError("unknown coarse direction objective")
    # vector_norm has a finite derivative at zero; sqrt(sum(square)) does not.
    cosine = projection / torch.linalg.vector_norm(vector, dim=dim).clamp_min(eps)
    tolerance = math.cos(math.radians(float(config.get("coarse_direction_tolerance_degrees", 30.))))
    return torch.relu(tolerance - cosine.clamp(-1., 1.))


def object_coarse_terms(predicted, target, masks, valid, horizons, config, sample_weight=None):
    """Inputs: normalized flow [B,T,2,H,W], t0 labels [B,K,H,W]."""
    if predicted.shape != target.shape or target.ndim != 5 or target.shape[2] != 2:
        raise ValueError("coarse flow must have matching [B,T,2,H,W] shapes")
    b, t, _, h, w = target.shape
    if masks.ndim != 4 or masks.shape[0] != b or tuple(masks.shape[-2:]) != (h, w):
        raise ValueError("coarse labels must have shape [B,K,H,W]")
    if tuple(valid.shape) != (b, t, 1, h, w):
        raise ValueError("coarse validity must have shape [B,T,1,H,W]")
    if not bool(torch.isfinite(masks).all()) or not bool(((masks >= 0) & (masks <= 1)).all()):
        raise ValueError("coarse labels must be finite weights in [0,1]")
    if not bool(torch.isfinite(valid).all()) or not bool(((valid >= 0) & (valid <= 1)).all()):
        raise ValueError("coarse validity must be finite weights in [0,1]")
    times = torch.as_tensor(horizons, dtype=torch.float32, device=target.device)
    if times.ndim == 1:
        times = times[None].expand(b, -1)
    if tuple(times.shape) != (b, t) or not bool(torch.isfinite(times).all()):
        raise ValueError("coarse horizons must be finite [T] or [B,T]")
    dt = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
    if bool((dt <= 0).any()):
        raise ValueError("coarse horizons must be positive and increasing")
    known = valid > .5
    scale = target.new_tensor([w, h]).view(1, 1, 2, 1, 1)
    p = torch.where(known, predicted, 0).float() * scale
    g = torch.where(known, target, 0).float() * scale
    if not bool(torch.isfinite(p).all() & torch.isfinite(g).all()):
        raise ValueError("coarse flow is nonfinite on valid support")
    mask = masks.detach().float().flatten(2)
    support = mask[:, :, None] * known[:, None, :, 0].flatten(-2).float()
    area = support.sum(-1)
    good = area >= float(config.get("coarse_min_pixels", 4))
    denom = area.clamp_min(1.)
    pm = torch.einsum("bktn,btcn->bktc", support, p.flatten(-2)) / denom[..., None]
    gm = torch.einsum("bktn,btcn->bktc", support, g.flatten(-2)) / denom[..., None]
    eps = float(config.get("coarse_direction_epsilon_px", .5))
    scale_px = float(config.get("coarse_motion_scale_px", 16.))
    norm = gm.norm(dim=-1)
    moving = good & (norm > float(config.get("flow_moving_threshold_px", 1.)))
    unit = gm / norm.clamp_min(1e-6)[..., None]
    projected = (pm * unit).sum(-1)
    robust = lambda x: (x.square().sum(-1) + eps**2).sqrt() - eps
    displacement = _mean(robust(pm - gm) / scale_px, good, sample_weight)
    # A moving object gets its own budget, regardless of the number of static
    # distractors or its travel distance. Keep the SIGNED vector error: speed
    # alone would reward a large displacement in the wrong direction. The
    # floor prevents subpixel annotation noise from dominating early queries.
    relative_displacement = _mean(
        robust(pm - gm) / norm.clamp_min(float(config.get("coarse_relative_floor_px", 4.))),
        moving, sample_weight)
    direction = _mean(_direction_cost(projected, pm, -1, config), moving, sample_weight)
    # Unlike cosine alone, a signed projection margin has a useful gradient
    # at exact reversal AND at zero motion. It demands forward progress, not
    # the full GT speed; displacement/magnitude retain their separate role.
    floor = float(config.get("coarse_relative_floor_px", 4.))
    margin = float(config.get("coarse_direction_min_projection", .25))
    direction_margin = _mean(torch.relu(margin - projected / norm.clamp_min(floor)),
                             moving, sample_weight)
    magnitude = _mean(torch.nn.functional.smooth_l1_loss(projected / scale_px,
                     norm / scale_px, reduction="none", beta=1 / scale_px), moving, sample_weight)

    # Solve centered affine slopes from known pixels. The covariance metric
    # evaluates their spatial displacement, in pixels, over the object's own
    # extent. It remains defined for a thin/degenerate mask via pseudoinverse.
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=p.device),
                          torch.linspace(-1, 1, w, device=p.device), indexing="ij")
    xy = torch.stack((x, y), -1).reshape(-1, 2)
    center = torch.einsum("bktn,nc->bktc", support, xy) / denom[..., None]
    second = torch.einsum("bktn,ni,nj->bktij", support, xy, xy) / denom[..., None, None]
    covariance = (second - center[..., :, None] * center[..., None, :]).detach()
    covariance = .5 * (covariance + covariance.transpose(-1, -2))
    error = (p - g).flatten(-2)
    cross = torch.einsum("bktn,ni,btcn->bktic", support, xy, error) / denom[..., None, None]
    cross = cross - center[..., :, None] * (pm - gm)[..., None, :]
    slope = torch.linalg.pinv(covariance, hermitian=True, rtol=1e-4) @ cross
    affine_mse = torch.einsum("bktic,bktij,bktjc->bkt", slope, covariance, slope).clamp_min(0)
    affine = _mean(((affine_mse + eps**2).sqrt() - eps) / scale_px, good, sample_weight)

    previous_known = torch.cat((torch.ones_like(known[:, :1]), known[:, :-1]), dim=1)
    common = mask[:, :, None] * (known & previous_known)[:, None, :, 0].flatten(-2)
    common_area = common.sum(-1)
    pg = torch.diff(p, dim=1, prepend=torch.zeros_like(p[:, :1]))
    gg = torch.diff(g, dim=1, prepend=torch.zeros_like(g[:, :1]))
    velocity_error = torch.einsum("bktn,btcn->bktc", common, (pg-gg).flatten(-2))
    velocity_error = velocity_error / common_area.clamp_min(1)[..., None] / dt[:, None, :, None]
    velocity = _mean(robust(velocity_error) / float(config.get("coarse_velocity_scale_px_s", 32.)),
                     common_area >= float(config.get("coarse_min_pixels", 4)), sample_weight)
    # Consecutive displacement differences follow instantaneous travel better
    # than cumulative displacement: a returning object can still lie right of
    # t0 while travelling left. Compare exactly the same points at both times.
    psegment = torch.einsum("bktn,btcn->bktc", common, pg.flatten(-2)) / common_area.clamp_min(1)[..., None]
    gsegment = torch.einsum("bktn,btcn->bktc", common, gg.flatten(-2)) / common_area.clamp_min(1)[..., None]
    segment_norm = gsegment.norm(dim=-1)
    segment_projection = (psegment * gsegment / segment_norm.clamp_min(1e-6)[..., None]).sum(-1)
    segment_good = ((common_area >= float(config.get("coarse_min_pixels", 4)))
                    & (segment_norm > float(config.get("flow_moving_threshold_px", 1.))))
    segment_penalty = (_direction_cost(segment_projection, psegment, -1, config)
                       + torch.relu(margin - segment_projection / segment_norm.clamp_min(floor)))
    segment_direction = _mean(segment_penalty, segment_good, sample_weight)

    # Rotation can cancel the mean translation. Give each object's moving
    # vector field its own directional budget, independent of area/background.
    pixel_norm = g.norm(dim=2)
    pixel_projection = (p * g / pixel_norm.clamp_min(1e-6).unsqueeze(2)).sum(2)
    pixel_penalty = (_direction_cost(pixel_projection, p, 2, config)
                     + torch.relu(margin - pixel_projection / pixel_norm.clamp_min(floor)))
    directional_support = support * (pixel_norm > float(config.get("flow_moving_threshold_px", 1.)))[:, None].flatten(-2)
    directional_area = directional_support.sum(-1)
    field_direction = _mean((directional_support * pixel_penalty[:, None].flatten(-2)).sum(-1)
                            / directional_area.clamp_min(1),
                            directional_area >= float(config.get("coarse_min_pixels", 4)), sample_weight)
    return dict(displacement=displacement, relative_displacement=relative_displacement,
                direction=direction, magnitude=magnitude,
                affine=affine, velocity=velocity, object_cells=good.sum().detach(),
                moving_cells=moving.sum().detach(), direction_margin=direction_margin,
                segment_direction=segment_direction, field_direction=field_direction)


def coarse_motion_loss_terms(prediction, batch, config, sample_weight=None):
    weights = validate_coarse_config(config)
    zero = prediction["flow"].new_zeros(())
    if not any(weights.values()):
        return zero, {}
    masks = batch.get("supervision_instance_masks")
    if not isinstance(masks, torch.Tensor):
        raise ValueError("coarse motion requires supervision_instance_masks")
    terms, total = {}, zero
    for name, field, default in [("raw", "flow", .5), ("gated", "flow_task_object_gated", 1.)]:
        weight = float(config.get(f"coarse_{name}_weight", default))
        if not weight:
            continue
        if field not in prediction:
            raise ValueError(f"coarse motion requires {field}")
        value = prediction[field]
        if name == "gated" and config.get("coarse_gate_only", False):
            mask = prediction.get("task_object_slot_mask")
            expected = (value.shape[0], value.shape[1], 1, *value.shape[-2:])
            if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != expected:
                raise ValueError("coarse_gate_only requires task_object_slot_mask")
            value = prediction["flow"].detach() * mask
        # Small-object covariance differences and pseudoinverses require FP32
        # even when the surrounding model is run under mixed precision.
        with torch.autocast(device_type=prediction[field].device.type, enabled=False):
            branch = object_coarse_terms(value, batch["flow"], masks,
                        batch["flow_valid"], batch["horizon_seconds"], config, sample_weight)
        for key, value in branch.items():
            terms[f"coarse_{name}_{key}"] = value
            if key in weights:
                total = total + weight * weights[key] * value
    terms["coarse_motion_total"] = total
    return total, terms
