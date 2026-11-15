from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch

from .losses import instance_trajectory_centers, trajectory_centers
from .video import forward_splat_mask


@torch.no_grad()
def sample_metrics(
    predicted_flow: torch.Tensor,
    visibility_logits: torch.Tensor,
    target_flow: torch.Tensor,
    target_visibility: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    height: int,
    width: int,
    instance_id: torch.Tensor | None = None,
    centers: torch.Tensor | None = None,
    center_valid: torch.Tensor | None = None,
    delta_time: float = 1.0,
    moving_threshold_px: float = 2.0,
) -> dict[str, float]:
    """Compute metrics for one sample; flows are normalized by width/height."""
    scale = torch.tensor([width, height], device=predicted_flow.device, dtype=predicted_flow.dtype)
    difference_px = (predicted_flow - target_flow) * scale[None, :, None, None]
    target_px = target_flow * scale[None, :, None, None]
    predicted_px = predicted_flow * scale[None, :, None, None]
    epe = difference_px.square().sum(dim=1).sqrt()
    magnitude = target_px.square().sum(dim=1).sqrt()
    valid = (object_mask[None, 0] > 0.5) & (target_visibility[:, 0] > 0.5)
    if not valid.any():
        return {}
    diagonal = math.hypot(height, width)
    object_epe = epe[valid].mean()
    fl_all = ((epe > 3.0) & (epe > 0.05 * magnitude))[valid].float().mean()

    pred_vector = torch.cat((predicted_px, torch.ones_like(predicted_px[:, :1])), dim=1)
    target_vector = torch.cat((target_px, torch.ones_like(target_px[:, :1])), dim=1)
    cosine = (pred_vector * target_vector).sum(dim=1) / (
        pred_vector.norm(dim=1) * target_vector.norm(dim=1)
    ).clamp_min(1e-6)
    angular = cosine.clamp(-1, 1).acos().mul(180 / math.pi)[valid].mean()

    pred_visibility = visibility_logits.sigmoid()
    predicted_centers = trajectory_centers(
        predicted_flow[None], object_mask[None], pred_visibility[None]
    )[0]
    target_centers = trajectory_centers(
        target_flow[None], object_mask[None], target_visibility[None]
    )[0]
    center_error = (predicted_centers - target_centers).square().sum(dim=-1).sqrt()
    velocity_error = (
        (predicted_centers[1:] - predicted_centers[:-1])
        - (target_centers[1:] - target_centers[:-1])
    ).square().sum(dim=-1).sqrt().mean()
    pred_velocity = predicted_centers[1:] - predicted_centers[:-1]
    target_velocity = target_centers[1:] - target_centers[:-1]
    acceleration_error = (
        (pred_velocity[1:] - pred_velocity[:-1])
        - (target_velocity[1:] - target_velocity[:-1])
    ).square().sum(dim=-1).sqrt().mean()

    predicted_occupancy = forward_splat_mask(
        object_mask[None], predicted_px[None]
    )[0, :, 0] > 0.25
    target_occupancy = forward_splat_mask(object_mask[None], target_px[None])[0, :, 0] > 0.25
    intersection = (predicted_occupancy & target_occupancy).flatten(1).sum(1).float()
    union = (predicted_occupancy | target_occupancy).flatten(1).sum(1).float().clamp_min(1)
    warped_iou = (intersection / union).mean()

    source = object_mask[None].expand_as(target_visibility) > 0.5
    truth = target_visibility > 0.5
    guess = visibility_logits > 0
    positive = source & truth
    negative = source & ~truth
    true_positive = (guess & positive).sum().float()
    false_positive = (guess & negative).sum().float()
    false_negative = (~guess & positive).sum().float()
    tpr = true_positive / positive.sum().clamp_min(1)
    tnr = ((~guess) & negative).sum().float() / negative.sum().clamp_min(1)
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1)
    result = {
        "valid_flow_pixels": float(valid.sum()),
        "object_epe_px": float(object_epe),
        "object_epe_norm": float(object_epe / diagonal),
        "fl_all": float(fl_all),
        "angular_error_deg": float(angular),
        "ade_norm": float(center_error.mean()),
        "fde_norm": float(center_error[-1]),
        "velocity_error_norm": float(velocity_error),
        "acceleration_error_norm": float(acceleration_error),
        "warped_mask_iou": float(warped_iou),
        "visibility_balanced_accuracy": float(0.5 * (tpr + tnr)),
        "visibility_f1": float(f1),
    }

    # Moving-instance support. On the union support above, many instances never
    # move, so much of the score is credit for correctly predicting "static" --
    # which the zero-flow baseline earns for free. Restricting to instances whose
    # mean ground-truth displacement clears the threshold keeps the object whole
    # (so a rotating body is scored on the parts near its axis too) and leaves
    # only the part anyone cares about. Diagnostic by construction: the support is
    # chosen with ground truth, so a model can be measured this way, not run.
    if instance_id is not None:
        moving = torch.zeros_like(valid)
        identifiers = instance_id[0]
        for value in torch.unique(identifiers):
            if int(value) <= 0:
                continue
            selected = valid & (identifiers == value)[None]
            if selected.any() and float(magnitude[selected].mean()) >= moving_threshold_px:
                moving |= selected
        if moving.any():
            result["moving_flow_pixels"] = float(moving.sum())
            result["object_epe_px_moving"] = float(epe[moving].mean())
            # Carried so the accumulator can form a true micro relative error:
            # sum(error)/sum(magnitude), not the mean of per-sample ratios.
            result["target_magnitude_px_moving"] = float(magnitude[moving].mean())
        # The other half of the moving-only support, and the reason it cannot be
        # used alone: a model that sprinkles motion everywhere scores well on
        # moving instances while destroying the static ones, and downstream a
        # confidently wrong displacement can cost more than none at all. This is
        # the hallucination monitor that has to be read alongside
        # relative_epe_moving.
        static = valid & ~moving
        if static.any():
            result["static_flow_pixels"] = float(static.sum())
            result["static_predicted_magnitude_px"] = float(
                predicted_px.square().sum(dim=1).sqrt()[static].mean()
            )
    if instance_id is not None and centers is not None and center_valid is not None:
        max_instances = centers.shape[0]
        surface_centers, instance_present = instance_trajectory_centers(
            predicted_flow[None], instance_id[None], max_instances
        )
        source_surface, _ = instance_trajectory_centers(
            torch.zeros_like(predicted_flow[None]), instance_id[None], max_instances
        )
        source_object = centers[:, 0].to(surface_centers.dtype)[None]
        predicted_instance = source_object + (
            surface_centers[0] - source_surface[0]
        )
        target_instance = centers[:, 1:].permute(1, 0, 2).to(predicted_instance.dtype)
        instance_valid = center_valid[:, 1:].permute(1, 0) & instance_present[0][None]
        pixel_scale = scale.view(1, 1, 2)
        position_error_px = (
            (predicted_instance - target_instance) * pixel_scale
        ).square().sum(dim=-1).sqrt()
        if instance_valid.any():
            result["instance_ade_px"] = float(position_error_px[instance_valid].mean())
            result["valid_instances"] = float(instance_present[0].sum())
        final_valid = instance_valid[-1]
        if final_valid.any():
            result["instance_fde_px"] = float(position_error_px[-1, final_valid].mean())
        velocity_valid = instance_valid[1:] & instance_valid[:-1]
        predicted_velocity_px_s = (
            (predicted_instance[1:] - predicted_instance[:-1]) * pixel_scale
        ) / delta_time
        target_velocity_px_s = (
            (target_instance[1:] - target_instance[:-1]) * pixel_scale
        ) / delta_time
        velocity_error_px_s = (
            predicted_velocity_px_s - target_velocity_px_s
        ).square().sum(dim=-1).sqrt()
        if velocity_valid.any():
            result["instance_velocity_error_px_s"] = float(
                velocity_error_px_s[velocity_valid].mean()
            )
        acceleration_valid = velocity_valid[1:] & velocity_valid[:-1]
        predicted_acceleration_px_s2 = (
            predicted_velocity_px_s[1:] - predicted_velocity_px_s[:-1]
        ) / delta_time
        target_acceleration_px_s2 = (
            target_velocity_px_s[1:] - target_velocity_px_s[:-1]
        ) / delta_time
        acceleration_error_px_s2 = (
            predicted_acceleration_px_s2 - target_acceleration_px_s2
        ).square().sum(dim=-1).sqrt()
        if acceleration_valid.any():
            result["instance_acceleration_error_px_s2"] = float(
                acceleration_error_px_s2[acceleration_valid].mean()
            )
    for horizon in range(predicted_flow.shape[0]):
        horizon_valid = valid[horizon]
        if horizon_valid.any():
            horizon_epe = epe[horizon][horizon_valid].mean()
            result[f"object_epe_px_h{horizon + 1}"] = float(horizon_epe)
            result[f"object_epe_norm_h{horizon + 1}"] = float(horizon_epe / diagonal)
            result[f"trajectory_error_norm_h{horizon + 1}"] = float(center_error[horizon])
    return result


class MetricAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            if math.isfinite(value):
                self.values[key].append(float(value))

    def summary(self) -> dict[str, float]:
        counts = ("valid_flow_pixels", "moving_flow_pixels", "static_flow_pixels")
        result = {
            key: sum(values) / len(values)
            for key, values in self.values.items()
            if values and key not in counts
        }

        def micro(key: str, weight_key: str) -> float | None:
            weights = self.values.get(weight_key, [])
            values = self.values.get(key, [])
            if not weights or len(values) != len(weights) or sum(weights) <= 0:
                return None
            return sum(v * w for v, w in zip(values, weights)) / sum(weights)

        weights = self.values.get("valid_flow_pixels", [])
        if weights:
            result["valid_flow_pixels"] = sum(weights)
            for key in ("object_epe_px", "object_epe_norm"):
                value = micro(key, "valid_flow_pixels")
                if value is not None:
                    result[f"{key}_micro"] = value
        moving_weights = self.values.get("moving_flow_pixels", [])
        if moving_weights:
            result["moving_flow_pixels"] = sum(moving_weights)
            epe_micro = micro("object_epe_px_moving", "moving_flow_pixels")
            magnitude_micro = micro("target_magnitude_px_moving", "moving_flow_pixels")
            if epe_micro is not None:
                result["object_epe_px_moving_micro"] = epe_micro
            if epe_micro is not None and magnitude_micro:
                # Main moving-pixel metric: what fraction of the real displacement survives as
                # error. 1.0 means the model is worth exactly as much as predicting
                # zero on the pixels that actually move.
                result["relative_epe_moving"] = epe_micro / magnitude_micro
        static_weights = self.values.get("static_flow_pixels", [])
        if static_weights:
            result["static_flow_pixels"] = sum(static_weights)
            value = micro("static_predicted_magnitude_px", "static_flow_pixels")
            if value is not None:
                result["static_predicted_magnitude_px_micro"] = value
        result["num_samples"] = float(max((len(values) for values in self.values.values()), default=0))
        return result
