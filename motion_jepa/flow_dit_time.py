"""Shared physical-time contract for the Stage-1 backbone and decoders."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


HorizonSeconds = Sequence[float] | torch.Tensor


def validate_time_scale(time_scale: float | None) -> float | None:
    """Keep the legacy relative-time mode only when no scale is configured."""
    if time_scale is None:
        return None
    scale = float(time_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"time_scale must be finite and positive, got {time_scale!r}")
    return scale


def normalized_horizon_times(
    horizon_seconds: HorizonSeconds | None,
    *,
    batch_size: int,
    horizons: int,
    time_scale: float | None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``[B, T]`` times with identical list/1D/2D tensor semantics.

    A fixed scale always means ``seconds / time_scale``. Without a configured
    scale, retain the old per-sample maximum convention for legacy checkpoints.
    Omitted seconds are supported only in that legacy mode, where their exact
    physical units never entered the model. Normalize in float32 before casting
    to the consumer's dtype so bf16 inputs do not lose physical-time precision.
    """
    scale = validate_time_scale(time_scale)
    if batch_size < 1 or horizons < 1:
        raise ValueError("batch_size and horizons must be positive")
    if horizon_seconds is None:
        if scale is not None:
            raise ValueError("horizon_seconds is required when time_scale is configured")
        horizon_seconds = list(range(1, horizons + 1))

    seconds = torch.as_tensor(horizon_seconds, dtype=torch.float32, device=device)
    if seconds.ndim == 1:
        if seconds.shape[0] != horizons:
            raise ValueError(f"expected {horizons} horizons, got {seconds.shape[0]}")
        seconds = seconds.unsqueeze(0).expand(batch_size, -1)
    elif seconds.ndim != 2 or seconds.shape != (batch_size, horizons):
        raise ValueError(
            f"expected horizons [{horizons}] or [{batch_size}, {horizons}], "
            f"got {tuple(seconds.shape)}"
        )
    if not torch.isfinite(seconds).all():
        raise ValueError("horizon_seconds must contain only finite values")
    if not (seconds > 0).all():
        raise ValueError("horizon_seconds must be positive future times")
    if not (seconds[:, 1:] > seconds[:, :-1]).all():
        raise ValueError("horizon_seconds must be strictly increasing per sample")

    denominator = scale if scale is not None else seconds.amax(dim=1, keepdim=True)
    return (seconds / denominator).to(dtype=dtype)
