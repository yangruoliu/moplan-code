from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def masked_mean(value: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.expand_as(value).sum().clamp_min(epsilon)


def trajectory_centers(
    flow: torch.Tensor,
    mask: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Return normalized flow-derived centers [B,T,2]."""
    batch, horizons, _, height, width = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype) / height,
        torch.arange(width, device=flow.device, dtype=flow.dtype) / width,
        indexing="ij",
    )
    coordinates = torch.stack((xx, yy), dim=0).view(1, 1, 2, height, width)
    weight = mask[:, None] * visibility
    numerator = ((coordinates + flow) * weight).sum(dim=(-1, -2))
    denominator = weight.sum(dim=(-1, -2)).clamp_min(1e-6)
    return numerator / denominator


def instance_trajectory_centers(
    flow: torch.Tensor,
    instance_id: torch.Tensor,
    max_instances: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-instance normalized centers [B,T,M,2]."""
    batch, horizons, _, height, width = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype) / height,
        torch.arange(width, device=flow.device, dtype=flow.dtype) / width,
        indexing="ij",
    )
    coordinates = torch.stack((xx, yy), dim=0)[None, None]
    destinations = (coordinates + flow).permute(0, 1, 3, 4, 2).reshape(
        batch, horizons, height * width, 2
    )
    ids = instance_id[:, 0].long().reshape(batch, height * width) - 1
    centers = flow.new_zeros((batch, horizons, max_instances, 2))
    counts = flow.new_zeros((batch, horizons, max_instances, 1))
    for batch_index in range(batch):
        valid = (ids[batch_index] >= 0) & (ids[batch_index] < max_instances)
        if not valid.any():
            continue
        slots = ids[batch_index, valid]
        centers[batch_index].scatter_add_(
            1,
            slots[None, :, None].expand(horizons, -1, 2),
            destinations[batch_index, :, valid],
        )
        counts[batch_index].scatter_add_(
            1,
            slots[None, :, None].expand(horizons, -1, 1),
            torch.ones_like(destinations[batch_index, :, valid, :1]),
        )
    present = counts[:, 0, :, 0] > 0
    return centers / counts.clamp_min(1), present


def compute_losses(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    z_pred = F.layer_norm(prediction["z_pred"], (prediction["z_pred"].shape[-1],))
    z_target = F.layer_norm(batch["z_future"], (batch["z_future"].shape[-1],))
    batch_size, horizons, _, height, width = batch["future_mask"].shape
    grid_height, grid_width = z_pred.shape[2:4]
    future_grid = F.interpolate(
        batch["future_mask"].reshape(batch_size * horizons, 1, height, width),
        size=(grid_height, grid_width),
        mode="area",
    ).reshape(batch_size, horizons, grid_height, grid_width)
    latent_weight = float(config["background_weight"]) + (
        float(config["object_weight"]) - float(config["background_weight"])
    ) * future_grid
    latent = masked_mean((z_pred - z_target).abs(), latent_weight)

    object_valid = batch["object_mask"][:, None] * batch["visibility"]
    if bool(config.get("full_frame_supervision", False)):
        background = 1 - batch["object_mask"][:, None]
        valid = object_valid + float(config.get("background_motion_weight", 0.02)) * background
    else:
        valid = object_valid
    flow_error = F.smooth_l1_loss(prediction["flow"], batch["flow"], reduction="none", beta=0.02)
    flow = masked_mean(flow_error, valid)

    logits = prediction["visibility_logits"]
    target_visibility = batch["visibility"]
    if bool(config.get("full_frame_supervision", False)):
        source = torch.ones_like(target_visibility)
    else:
        source = batch["object_mask"][:, None].expand_as(target_visibility)
    positive_fraction = masked_mean(target_visibility, source).detach().clamp(0.05, 0.95)
    balance = torch.where(
        target_visibility > 0.5,
        0.5 / positive_fraction,
        0.5 / (1 - positive_fraction),
    )
    visibility = masked_mean(
        F.binary_cross_entropy_with_logits(logits, target_visibility, reduction="none"),
        source * balance,
    )

    if "instance_id" in batch and "centers" in batch and "center_valid" in batch:
        max_instances = batch["centers"].shape[1]
        predicted_surface_centers, source_present = instance_trajectory_centers(
            prediction["flow"], batch["instance_id"], max_instances
        )
        source_surface_centers, _ = instance_trajectory_centers(
            torch.zeros_like(prediction["flow"]), batch["instance_id"], max_instances
        )
        source_object_centers = batch["centers"][:, :, 0].to(
            predicted_surface_centers.dtype
        )[:, None]
        predicted_centers = source_object_centers + (
            predicted_surface_centers - source_surface_centers
        )
        target_centers = batch["centers"][:, :, 1:].permute(0, 2, 1, 3).to(
            predicted_centers.dtype
        )
        target_valid = batch["center_valid"][:, :, 1:].permute(0, 2, 1)
        position_valid = target_valid & source_present[:, None]
        position = masked_mean(
            F.smooth_l1_loss(
                predicted_centers, target_centers, reduction="none", beta=0.02
            ),
            position_valid,
        )
        delta_time = float(config.get("delta_time", 1.0))
        predicted_velocity = (
            predicted_centers[:, 1:] - predicted_centers[:, :-1]
        ) / delta_time
        target_velocity = (
            target_centers[:, 1:] - target_centers[:, :-1]
        ) / delta_time
        velocity_valid = position_valid[:, 1:] & position_valid[:, :-1]
        velocity = masked_mean(
            F.smooth_l1_loss(
                predicted_velocity, target_velocity, reduction="none", beta=0.05
            ),
            velocity_valid,
        )
        predicted_acceleration = (
            predicted_velocity[:, 1:] - predicted_velocity[:, :-1]
        ) / delta_time
        target_acceleration = (
            target_velocity[:, 1:] - target_velocity[:, :-1]
        ) / delta_time
        acceleration_valid = velocity_valid[:, 1:] & velocity_valid[:, :-1]
        acceleration = masked_mean(
            F.smooth_l1_loss(
                predicted_acceleration, target_acceleration, reduction="none", beta=0.1
            ),
            acceleration_valid,
        )
    else:
        predicted_centers = trajectory_centers(
            prediction["flow"], batch["object_mask"], logits.sigmoid()
        )
        target_centers = trajectory_centers(
            batch["flow"], batch["object_mask"], target_visibility
        )
        position = F.smooth_l1_loss(predicted_centers, target_centers, beta=0.02)
        predicted_velocity = predicted_centers[:, 1:] - predicted_centers[:, :-1]
        target_velocity = target_centers[:, 1:] - target_centers[:, :-1]
        velocity = F.smooth_l1_loss(predicted_velocity, target_velocity, beta=0.01)
        predicted_acceleration = predicted_velocity[:, 1:] - predicted_velocity[:, :-1]
        target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
        acceleration = F.smooth_l1_loss(
            predicted_acceleration, target_acceleration, beta=0.01
        )
    explicit_kinematic_weights = all(
        key in config for key in ("position", "velocity", "acceleration")
    )
    if explicit_kinematic_weights:
        # New configs expose every physical term directly so that the effective
        # weights cannot be hidden behind a nested trajectory multiplier.
        trajectory = position + velocity + acceleration
        total = (
            float(config["latent"]) * latent
            + float(config["flow"]) * flow
            + float(config["visibility"]) * visibility
            + float(config["position"]) * position
            + float(config["velocity"]) * velocity
            + float(config["acceleration"]) * acceleration
        )
    else:
        # Backward-compatible path for the original Stage-1 checkpoint/config.
        trajectory = (
            position
            + float(config["velocity_weight"]) * velocity
            + float(config["acceleration_weight"]) * acceleration
        )
        total = (
            float(config["latent"]) * latent
            + float(config["flow"]) * flow
            + float(config["visibility"]) * visibility
            + float(config["trajectory"]) * trajectory
        )
    return {
        "total": total,
        "latent": latent,
        "flow": flow,
        "visibility": visibility,
        "trajectory": trajectory,
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
    }
