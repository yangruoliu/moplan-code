"""Additive, full-trusted-support motion diagnostics for Stage-1.

These do not use the legacy dynamic ``object_mask`` and do not change model
selection. Inputs are cumulative source-to-future flow, normalized separately
by image width and height. Counts are pixel-horizons, not objects or videos.
Every returned sufficient statistic can be summed across samples and ranks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def _support_mask(value: torch.Tensor, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    # Evaluation uses hard, trusted support, not confidence-weighted counts.
    return value > 0.5


@torch.no_grad()
def motion_sufficient_statistics(
    pred: torch.Tensor,
    target: torch.Tensor,
    flow_valid: torch.Tensor,
    static_certified: torch.Tensor | None = None,
    *,
    width: int,
    height: int,
    moving_threshold_px: float = 2.0,
    static_threshold_px: float = 0.5,
) -> dict[str, float]:
    """Return finite sums/counts for one ``[T,2,H,W]`` flow sample.

    ``flow_valid`` is [T,1,H,W], with >0.5 defining trusted support. NaNs or
    infinities outside that support are ignored; nonfinite valid predictions
    or targets raise. No model-predicted visibility or input mask restricts
    the scoring support.

    GT moving means magnitude > ``moving_threshold_px``. Active predictions
    use the same cutoff; near-zero predictions use <= ``static_threshold_px``.
    Static GT support uses <= the latter cutoff and additionally intersects
    ``static_certified`` when supplied. Without that optional field, "static"
    means trusted low-flow pseudo-GT, not certified physical immobility.

    "Magnitude sums" are L1 sums of vector lengths in pixels (not squared
    physical energy). Signed projection is predicted displacement along the
    GT unit direction, summed only over GT-moving pixels. Direction errors
    are counted on active predictions within GT-moving support.
    """
    if pred.ndim != 4 or pred.shape[1] != 2 or tuple(target.shape) != tuple(pred.shape):
        raise ValueError("pred and target must have matching shape [T,2,H,W]")
    if any(size <= 0 for size in (pred.shape[0], pred.shape[2], pred.shape[3])):
        raise ValueError("flow temporal and spatial dimensions must be positive")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in (width, height)):
        raise ValueError("width and height must be finite and positive")
    if not (math.isfinite(moving_threshold_px) and math.isfinite(static_threshold_px)
            and 0 <= static_threshold_px < moving_threshold_px):
        raise ValueError("thresholds must satisfy 0 <= static < moving and be finite")

    shape = (pred.shape[0], 1, pred.shape[2], pred.shape[3])
    valid = _support_mask(flow_valid.to(pred.device), shape, "flow_valid")[:, 0]
    certified = None
    if static_certified is not None:
        certified = _support_mask(
            static_certified.to(pred.device), shape, "static_certified"
        )[:, 0][valid]

    # Index before any arithmetic, preventing unknown NaNs from contaminating
    # sums. Float64 accumulation avoids fp16 overflow on large support.
    predicted = pred.detach().to(dtype=torch.float64).permute(0, 2, 3, 1)[valid]
    reference = target.detach().to(device=pred.device, dtype=torch.float64).permute(0, 2, 3, 1)[valid]
    if not bool(torch.isfinite(predicted).all()):
        raise ValueError("pred contains nonfinite flow on valid support")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("target contains nonfinite flow on valid support")

    scale = predicted.new_tensor([width, height])
    predicted = predicted * scale
    reference = reference * scale
    pred_mag = torch.linalg.vector_norm(predicted, dim=-1)
    gt_mag = torch.linalg.vector_norm(reference, dim=-1)
    error = torch.linalg.vector_norm(predicted - reference, dim=-1)
    moving = gt_mag > moving_threshold_px
    static = gt_mag <= static_threshold_px
    if certified is not None:
        static = static & certified
    pred_active = pred_mag > moving_threshold_px
    active_moving = moving & pred_active
    dot = (predicted * reference).sum(dim=-1)
    # Both denominators below are strictly positive on active-moving support.
    cosine = (dot[active_moving] / (pred_mag[active_moving] * gt_mag[active_moving])).clamp(-1., 1.)

    values = {
        "valid_count": valid.sum(),
        "moving_count": moving.sum(),
        "static_count": static.sum(),
        "moving_active_count": active_moving.sum(),
        "moving_nearzero_count": (moving & (pred_mag <= static_threshold_px)).sum(),
        "moving_reverse_count": (cosine < 0).sum(),
        "moving_angle_gt60_count": (cosine < 0.5).sum(),
        "moving_direction30_success_count": (cosine >= math.cos(math.pi / 6)).sum(),
        "static_falseactive_count": (static & pred_active).sum(),
        "valid_epe_sum_px": error.sum(),
        "moving_epe_sum_px": error[moving].sum(),
        "moving_zero_epe_sum_px": gt_mag[moving].sum(),
        "static_pred_magnitude_sum_px": pred_mag[static].sum(),
        "valid_gt_magnitude_sum_px": gt_mag.sum(),
        "valid_pred_magnitude_sum_px": pred_mag.sum(),
        "moving_gt_magnitude_sum_px": gt_mag[moving].sum(),
        "moving_pred_magnitude_sum_px": pred_mag[moving].sum(),
        "moving_aligned_projection_sum_px": (dot[moving] / gt_mag[moving]).sum(),
    }
    # A single transfer also works for CUDA callers. Counters stay exact at
    # practical dataset sizes while sharing a common distributed sum dtype.
    packed = torch.stack([value.to(torch.float64) for value in values.values()]).cpu().tolist()
    result = dict(zip(values, packed))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("motion statistics overflowed; inspect flow magnitudes")
    result["sample_count"] = 1.0
    result["sample_with_motion_count"] = float(result["moving_count"] > 0)
    result["sample_moving_better_than_zero_count"] = float(
        result["moving_count"] > 0
        and result["moving_epe_sum_px"] < result["moving_zero_epe_sum_px"]
    )
    return result


def summary_statistics(sums: Mapping[str, float]) -> dict[str, float | None]:
    """Derive micro metrics after summing sufficient statistics.

    Undefined ratios are ``None``, never a success-looking zero. In particular
    an all-zero predictor has moving relative EPE=1 and recall=0, but direction
    error fractions on *active* moving predictions are undefined. These are
    pixel-horizon pooled metrics; sample win rate is a separate denominator.
    """
    required = {
        "valid_count", "moving_count", "static_count", "moving_active_count",
        "moving_nearzero_count", "moving_reverse_count", "moving_angle_gt60_count",
        "static_falseactive_count", "valid_epe_sum_px", "moving_epe_sum_px",
        "moving_zero_epe_sum_px", "static_pred_magnitude_sum_px",
        "valid_gt_magnitude_sum_px", "valid_pred_magnitude_sum_px",
        "moving_gt_magnitude_sum_px", "moving_pred_magnitude_sum_px",
        "moving_aligned_projection_sum_px", "sample_count", "sample_with_motion_count",
        "sample_moving_better_than_zero_count",
    }
    missing = required - sums.keys()
    if missing:
        raise ValueError(f"missing motion sufficient statistics: {sorted(missing)}")
    values = {key: float(sums[key]) for key in required}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("sufficient statistics must be finite")
    if any(value < 0 for key, value in values.items() if key != "moving_aligned_projection_sum_px"):
        raise ValueError("counts and magnitude/error sums must be nonnegative")

    def ratio(numerator: str, denominator: str) -> float | None:
        return values[numerator] / values[denominator] if values[denominator] > 0 else None

    direction_extra = {}
    if "moving_direction30_success_count" in sums:
        success = float(sums["moving_direction30_success_count"])
        if not math.isfinite(success) or not 0 <= success <= values["moving_active_count"]:
            raise ValueError("invalid direction success count")
        fraction = success / values["moving_count"] if values["moving_count"] else None
        direction_extra = {"moving_direction30_success_fraction": fraction,
                           "moving_direction30_failure_fraction": 1-fraction if fraction is not None else None}
    return {
        **direction_extra,
        "full_valid_epe_px": ratio("valid_epe_sum_px", "valid_count"),
        "moving_epe_px": ratio("moving_epe_sum_px", "moving_count"),
        "moving_zero_epe_px": ratio("moving_zero_epe_sum_px", "moving_count"),
        "moving_relative_epe": ratio("moving_epe_sum_px", "moving_zero_epe_sum_px"),
        "moving_recall": ratio("moving_active_count", "moving_count"),
        "moving_nearzero_fraction": ratio("moving_nearzero_count", "moving_count"),
        "moving_reverse_fraction_all": ratio("moving_reverse_count", "moving_count"),
        "moving_reverse_fraction_active": ratio("moving_reverse_count", "moving_active_count"),
        "moving_angle_gt60_fraction_all": ratio("moving_angle_gt60_count", "moving_count"),
        "moving_angle_gt60_fraction_active": ratio("moving_angle_gt60_count", "moving_active_count"),
        "static_falseactive_fraction": ratio("static_falseactive_count", "static_count"),
        "static_pred_magnitude_px": ratio("static_pred_magnitude_sum_px", "static_count"),
        "full_valid_gt_magnitude_px": ratio("valid_gt_magnitude_sum_px", "valid_count"),
        "full_valid_pred_magnitude_px": ratio("valid_pred_magnitude_sum_px", "valid_count"),
        "moving_gt_magnitude_px": ratio("moving_gt_magnitude_sum_px", "moving_count"),
        "moving_pred_magnitude_px": ratio("moving_pred_magnitude_sum_px", "moving_count"),
        "moving_pred_to_gt_magnitude": ratio("moving_pred_magnitude_sum_px", "moving_gt_magnitude_sum_px"),
        "moving_signed_projection_px": ratio("moving_aligned_projection_sum_px", "moving_count"),
        "moving_signed_projection_to_gt": ratio("moving_aligned_projection_sum_px", "moving_gt_magnitude_sum_px"),
        "sample_motion_win_rate": ratio("sample_moving_better_than_zero_count", "sample_with_motion_count"),
    }
