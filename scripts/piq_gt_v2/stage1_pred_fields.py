#!/usr/bin/env python
"""One definition of the Stage-1 readouts that Stage-2 is conditioned on.

Every chain that turns a Stage-1 export into a Stage-2 condition (PhyCo training/validation cache,
Physics-IQ cache) resolves the flow through ``resolve_field`` so the field that was selected on the
validation set is, byte for byte in its definition, the field that is trained on and deployed:

  raw          flow_ungated                        the decoder output, nothing selected
  instances    raw x union_k mask_k                raw inside the input SAM2 instances (no selection: only the
                                                   full-frame residual outside every instance is dropped)
  soft         flow_task_object_gated              raw x sum_k p[k, t] * mask_k  (per-time soft gate)
  binary_<tau> raw x sum_k g_k * mask_k            g_k = 1[max_t p[k, t] >= tau]  -- constant in time

The binary gate is deliberately time-constant: a per-time gate that closes at one horizon would turn an
accumulated displacement of 20 px into 0 px at the next anchor, i.e. a teleport in the condition.

``rotate_objects`` builds the direction-shuffled control: every flow vector of an instance is rotated by
one angle per instance (the same at every horizon), so per-pixel magnitude, the spatial variation inside
the object, the support and everything outside the instances are unchanged -- only direction differs.
"""
from __future__ import annotations

import math
import random

import torch

CONTRACT_VERSION = 1
FIELDS = ("raw", "instances", "soft", "binary_0.5", "binary_0.7")
LEGACY_NAMES = {"flow_ungated": "raw", "flow_task_object_gated": "soft"}


def canonical(name: str) -> str:
    name = LEGACY_NAMES.get(name, name)
    if name in ("raw", "soft", "instances"):
        return name
    if name.startswith("binary_"):
        tau = float(name.split("_", 1)[1])
        if not 0.0 < tau < 1.0:
            raise ValueError(f"binary threshold must lie in (0, 1): {name}")
        return f"binary_{tau:g}"
    raise ValueError(f"unknown Stage-1 readout field {name!r} (known: raw, instances, soft, binary_<tau>)")


def slot_gate(probability: torch.Tensor, tau: float) -> torch.Tensor:
    """[K] bool: a slot is selected when its task-object probability reaches tau at any horizon."""
    if probability.ndim != 2:
        raise ValueError(f"expected slot probabilities [K, T], got {tuple(probability.shape)}")
    return probability.float().amax(dim=1) >= tau


def selected_support(export: dict, tau: float) -> torch.Tensor:
    """[S, S] bool union of the instance masks of the selected slots."""
    masks = export["activity_input_instance_masks"].bool()                      # [K, S, S], disjoint
    gate = slot_gate(export["task_object_slot_probability"], tau)               # [K]
    if masks.shape[0] != gate.shape[0]:
        raise ValueError("slot probabilities and instance masks disagree on the slot count")
    return (masks & gate[:, None, None]).any(dim=0)


def resolve_field(export: dict, name: str) -> tuple[torch.Tensor, dict]:
    """(flow [T, 2, S, S] float32 in the export's own normalised units, contract dict)."""
    field = canonical(name)
    contract = {"version": CONTRACT_VERSION, "field": field}
    if field == "raw":
        flow = export["flow_ungated"].float()
        contract.update(source="flow_ungated", gate=None)
    elif field == "soft":
        flow = export["flow_task_object_gated"].float()
        contract.update(source="flow_task_object_gated", gate="soft_per_time_slot_probability")
    elif field == "instances":
        support = export["activity_input_instance_masks"].bool().any(dim=0)
        flow = export["flow_ungated"].float() * support[None, None].float()
        contract.update(source="flow_ungated", gate="all_input_instances")
    else:
        tau = float(field.split("_", 1)[1])
        support = selected_support(export, tau)
        flow = export["flow_ungated"].float() * support[None, None].float()
        contract.update(source="flow_ungated", gate="binary_time_constant_max_over_horizons", tau=tau,
                        selected_slots=[int(i) for i in torch.nonzero(
                            slot_gate(export["task_object_slot_probability"], tau)).flatten()])
    if not torch.isfinite(flow).all():
        raise ValueError(f"non-finite Stage-1 flow in field {field}")
    return flow, contract


def rotate_objects(flow: torch.Tensor, instance_masks: torch.Tensor, key: str, *, seed: int = 20261011,
                   low_deg: float = 120.0, high_deg: float = 240.0) -> tuple[torch.Tensor, dict[int, float]]:
    """Rotate every flow vector inside each instance by one angle per instance.

    flow [T, 2, H, W] (any units), instance_masks [K, H, W] bool (disjoint).  The angle of instance k is
    drawn from U(low, high) degrees with Random(seed:key:k) -- identical at every horizon, so the motion
    stays coherent in time and within the object while its direction is wrong by at least 120 degrees.
    Pixels outside all instances are returned unchanged.
    """
    if flow.ndim != 4 or flow.shape[1] != 2 or instance_masks.shape[-2:] != flow.shape[-2:]:
        raise ValueError(f"shape mismatch: flow {tuple(flow.shape)} masks {tuple(instance_masks.shape)}")
    rotated, angles = flow.clone(), {}
    for slot in range(instance_masks.shape[0]):
        mask = instance_masks[slot].bool()
        if not bool(mask.any()):
            continue
        degrees = random.Random(f"{seed}:{key}:{slot}").uniform(low_deg, high_deg)
        cosine, sine = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        x, y = flow[:, 0][:, mask], flow[:, 1][:, mask]
        rotated[:, 0][:, mask] = cosine * x - sine * y
        rotated[:, 1][:, mask] = sine * x + cosine * y
        angles[slot] = degrees
    return rotated, angles


def displacement_jump(flow: torch.Tensor, support: torch.Tensor | None = None) -> float:
    """Largest drop of the cumulative displacement between consecutive horizons, relative to its peak.

    A value near 1 means some pixel goes from (almost) its largest displacement to (almost) none at the
    next anchor -- the teleport a per-time gate can cause."""
    magnitude = flow.float().square().sum(1).sqrt()                              # [T, H, W]
    if support is not None:
        magnitude = magnitude * support.float()[None]
    peak = magnitude.amax(dim=0)
    if float(peak.max()) <= 0:
        return 0.0
    drop = (magnitude[:-1] - magnitude[1:]).clamp_min(0).amax(dim=0)             # [H, W]
    moving = peak > 0.25 * float(peak.max())
    return float((drop[moving] / peak[moving]).max()) if bool(moving.any()) else 0.0
