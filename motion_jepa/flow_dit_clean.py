"""One object-balanced material-point objective for fresh Stage1 training.

Instance labels are supervision only. Neither predicted masks nor SAM coverage
gate this loss, so rotations and valid points outside the input proposals train
the affine/residual predictor. Unknown correspondence is never a zero target.
"""
from __future__ import annotations

import math

import torch

PROFILE = "clean_trajectory_v1"


def validate_clean_config(config):
    loss, model, train = config["loss"], config["model"], config["train"]
    allowed = {"profile", "supervision_mode", "flow_matching", "aux_sigma_weight",
               "aux_warmup_steps", "flow_charbonnier", "flow_epsilon_px",
               "trajectory_scale_px", "trajectory_segment", "trajectory_velocity_scale_px",
               "trajectory_epsilon_px_per_second"}
    if set(loss) - allowed:
        raise ValueError(f"clean trajectory rejects extra loss settings: {sorted(set(loss) - allowed)}")
    if loss.get("supervision_mode") != "valid_partition_v1":
        raise ValueError("clean trajectory requires explicit valid_partition_v1 labels")
    if train.get("objective") != "direct" or float(loss.get("flow_matching", 0)) != 0:
        raise ValueError("clean trajectory requires direct prediction without latent FM")
    if train.get("refine_from") or train.get("activity_heads_only") or train.get("mask_heads_only"):
        raise ValueError("clean trajectory trains one fresh predictor, without a refine partner or head-only phase")
    if float(loss.get("flow_charbonnier", 1)) != 1:
        raise ValueError("clean trajectory main loss has fixed weight one")
    if loss.get("aux_sigma_weight", "uniform") != "uniform" or int(loss.get("aux_warmup_steps", 0)) != 0:
        raise ValueError("clean trajectory does not use sigma weighting or auxiliary warmup")
    for key, default in (("flow_epsilon_px", .5), ("trajectory_scale_px", 16.),
                         ("trajectory_velocity_scale_px", 32.), ("trajectory_epsilon_px_per_second", .5)):
        value = float(loss.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    segment = float(loss.get("trajectory_segment", 0.))
    if not math.isfinite(segment) or segment < 0:
        raise ValueError("trajectory_segment must be finite and nonnegative")
    decoder = model["decoder"]
    if (decoder.get("type") != "slots" or not decoder.get("oracle_instance_masks")
            or not decoder.get("residual_branch")):
        raise ValueError("clean trajectory requires input SAM slots and affine plus residual output")
    for key in ("dense_motion_mask", "motion_activity", "task_object_slot_activity",
                "task_object_identity", "text_slot_binding", "text_slot_fusion"):
        if decoder.get(key, False):
            raise ValueError(f"clean trajectory requires decoder.{key}=false")
    if not train.get("freeze_visibility_heads", False):
        raise ValueError("clean trajectory requires diagnostic visibility heads frozen")
    if train.get("sampler") != "source_cycle_v1" or not train.get("exact_resume", False):
        raise ValueError("clean trajectory requires source_cycle_v1 and exact_resume")
    minimum = train.get("min_unique_scenes", 20000)
    if type(minimum) is not int or minimum < 0:
        raise ValueError("min_unique_scenes must be a nonnegative integer")


def freeze_clean_unused_heads(decoder):
    """Leave legacy diagnostic state loadable, but never optimize unused heads."""
    prefixes = ("image_stem.", "mask_fuse.", "mask_up.", "mask_head.",
                "visibility_fuse.", "visibility_up.", "visibility_head.")
    frozen = []
    for name, parameter in decoder.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen


def object_balanced_mean(error, valid, labels, sample_weight=None):
    """Mean points/time within each instance, then instances, then samples.

Label zero is one background region only where correspondence is explicitly
valid; it is never filled with static labels. All labeled objects are included,
even if SAM omitted them. Empty/padded objects do not affect the denominator.
"""
    b, _, _, h, w = error.shape
    if labels.shape == (b, 1, h, w):
        labels = labels[:, 0]
    if labels.shape != (b, h, w) or not bool(torch.isfinite(labels).all()):
        raise ValueError("instance_id must contain finite [B,1,H,W] labels")
    if bool((labels < 0).any()) or not torch.equal(labels, labels.long()):
        raise ValueError("instance_id must contain nonnegative integer labels")
    if sample_weight is not None:
        if sample_weight.shape != (b,) or not bool(torch.isfinite(sample_weight).all()) or bool((sample_weight < 0).any()):
            raise ValueError("sample_weight must be finite nonnegative [B]")
    samples, object_count = [], 0
    for index in range(b):
        regions = []
        active = valid[index].sum(dim=(0, 1)) > 0
        for label in torch.unique(labels[index][active]):
            support = valid[index] * (labels[index] == label)[None, None]
            regions.append((error[index] * support).sum() / support.sum().clamp_min(1e-12))
        if regions:
            value = torch.stack(regions).mean()
            if sample_weight is not None:
                value = value * sample_weight[index]
            samples.append(value)
            object_count += len(regions)
    result = torch.stack(samples).mean() if samples else error.sum() * 0
    return result, error.new_tensor(object_count)


def clean_trajectory_losses(prediction, batch, config, sample_weight=None, *, width=160, height=160):
    predicted, target = prediction["flow"].float(), batch["flow"].float()
    if predicted.shape != target.shape or target.ndim != 5 or target.shape[2] != 2:
        raise ValueError("clean trajectory needs matching [B,T,2,H,W] fields")
    b, t, _, h, w = target.shape
    valid = batch["flow_valid"].to(predicted)
    if valid.shape != (b, t, 1, h, w) or not bool(torch.isfinite(valid).all()) or bool(((valid < 0) | (valid > 1)).any()):
        raise ValueError("flow_valid must contain [B,T,1,H,W] weights in [0,1]")
    if "content_valid" in batch:
        content = batch["content_valid"].to(predicted)
        if content.shape != (b, 1, h, w) or not bool(torch.isfinite(content).all()) or bool(((content < 0) | (content > 1)).any()):
            raise ValueError("content_valid must contain [B,1,H,W] weights in [0,1]")
        valid = valid * content[:, None]
    # Mask BEFORE subtraction: unavailable labels may be NaN placeholders.
    predicted = torch.where(valid > 0, predicted, 0)
    target = torch.where(valid > 0, target, 0)
    if not bool(torch.isfinite(predicted).all() & torch.isfinite(target).all()):
        raise ValueError("nonfinite trajectory on known correspondence")
    labels = batch["instance_id"].to(predicted.device)
    pixels = predicted.new_tensor([width, height]).view(1, 1, 2, 1, 1)
    epsilon = float(config.get("flow_epsilon_px", .5))
    error = (((predicted - target) * pixels).square().sum(2, keepdim=True) + epsilon**2).sqrt() - epsilon
    flow, objects = object_balanced_mean(error / float(config.get("trajectory_scale_px", 16.)), valid, labels, sample_weight)
    segment, support = flow * 0, valid.new_zeros(())
    coefficient = float(config.get("trajectory_segment", 0.))
    if coefficient:
        times = torch.as_tensor(batch["horizon_seconds"], device=predicted.device, dtype=torch.float32)
        if times.ndim == 1:
            times = times[None].expand(b, -1)
        if times.shape != (b, t) or not bool(torch.isfinite(times).all()):
            raise ValueError("horizon_seconds must be finite [B,T] or [T]")
        dt = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
        if bool((dt <= 0).any()):
            raise ValueError("trajectory horizons must be positive and strictly increasing")
        velocity_error = torch.diff(predicted - target, dim=1,
                                    prepend=torch.zeros_like(predicted[:, :1])) * pixels / dt[:, :, None, None, None]
        pair_valid = torch.minimum(valid, torch.cat([torch.ones_like(valid[:, :1]), valid[:, :-1]], dim=1))
        eps = float(config.get("trajectory_epsilon_px_per_second", .5))
        error = (velocity_error.square().sum(2, keepdim=True) + eps**2).sqrt() - eps
        segment, _ = object_balanced_mean(error / float(config.get("trajectory_velocity_scale_px", 32.)), pair_valid, labels, sample_weight)
        support = pair_valid.sum().detach()
    zero = flow.detach().new_zeros(())
    # Existing logging ABI; none of these diagnostic losses is evaluated.
    result = {key: zero for key in ("gate", "visibility", "object_mask", "slot_mask", "slot_affine",
                                   "object_displacement", "object_occupancy", "position", "velocity", "acceleration")}
    return {**result, "aux_total": flow + coefficient * segment, "flow": flow,
            "trajectory_segment": segment, "trajectory_segment_support": support,
            "support_objects": objects, "support_valid_points": valid.sum().detach()}
