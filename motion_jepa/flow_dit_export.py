"""Canonical Stage1 bridge fields, independent of optional diagnostic heads."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from motion_jepa.flow_dit_sampler import reduce_samples


def checkpoint_provenance(checkpoint: str | Path) -> dict:
    checkpoint = Path(checkpoint).resolve()
    marker = checkpoint / "checkpoint_complete.json"
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_state_sha256": hashlib.sha256((checkpoint / "trainer_state.json").read_bytes()).hexdigest(),
        "checkpoint_complete_sha256": hashlib.sha256(marker.read_bytes()).hexdigest() if marker.is_file() else None,
    }


def bridge_fields(decoded: dict, reduced: dict, *, reduction: str,
                  eval_side: int, instance_masks: torch.Tensor | None) -> dict:
    """Keep raw source flow and actual input SAM support; never use predicted gates."""
    raw = reduced.get("flow_ungated")
    if raw is None:
        raw = reduce_samples(decoded, reduce=reduction, shrinkage_px=0.0,
                             eval_side=eval_side, gain=1.0)["flow"]
    if not isinstance(raw, torch.Tensor) or raw.ndim != 4 or raw.shape[1] != 2 or not torch.isfinite(raw).all():
        raise ValueError("bridge requires finite ungated [horizon,2,height,width] flow")
    result = {"flow_ungated": raw.detach().cpu().half()}
    if instance_masks is not None:
        if instance_masks.ndim != 3 or tuple(instance_masks.shape[-2:]) != tuple(raw.shape[-2:]):
            raise ValueError("bridge input SAM masks must share the source flow grid")
        result["activity_input_instance_masks"] = instance_masks.detach().cpu().bool()
    return result
