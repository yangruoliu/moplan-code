"""Explicit Stage-2 input interfaces; legacy checkpoints keep their exact layout.

The native five channels are source-coordinate cumulative u, v, M0,
availability, known.  They are not a slice of the 27-channel rendered input:
its known map lives at destination pixels and its raw flow can differ from the
object plan used by the first twenty channels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NATIVE5_NAMES = ("source_u", "source_v", "source_mask", "availability", "source_known")
INTERFACES = {"legacy20": (20, 0), "native5": (5, 0), "legacy27": (20, 7)}


def interface_from_config(model: dict) -> str:
    shape = (int(model.get("motion_condition_channels", 20)), int(model.get("motion_condition_extra_channels", 0)))
    inferred = next((name for name, channels in INTERFACES.items() if channels == shape), None)
    name = model.get("condition_interface", inferred)
    if name not in INTERFACES or INTERFACES[name] != shape:
        raise ValueError(f"condition_interface {name!r} disagrees with channel configuration {shape}")
    return name


def read_cached_condition(payload: dict, kind: str, interface: str, source: str = "cached") -> torch.Tensor:
    if interface not in INTERFACES or source not in ("cached", "matched_plan"):
        raise ValueError(f"unsupported condition interface/source: {interface}/{source}")
    key = f"cond_{kind}"
    if interface == "native5":
        key = f"cond_native5_{kind}"
        if tuple(payload.get("native5_channels", ())) != NATIVE5_NAMES:
            raise ValueError("native5 requires an explicitly prepared source-coordinate condition cache")
    elif interface == "legacy20" and source == "matched_plan":
        key = f"cond_matched20_{kind}"
    if key not in payload:
        raise KeyError(f"missing {key}; prepare matched plan conditions before using this interface")
    value = payload[key].float()
    channels = sum(INTERFACES[interface])
    if interface == "legacy20" and key == f"cond_{kind}":
        value = value[:20]
    if value.ndim != 4 or value.shape[0] != channels or not torch.isfinite(value).all():
        raise ValueError(f"invalid {key}: expected finite [{channels},T,H,W], got {tuple(value.shape)}")
    return value


def render_native5(plan, image_latent: torch.Tensor, times, size, *, kind: str = "real") -> torch.Tensor:
    """Render the SAME object plan as the legacy20 renderer, at source pixels.

This controlled comparison uses the plan's dense affine field, not the extra
ungated decoder residual in an old cond_pred cache. Known-weighted pooling
avoids attenuating a tiny object by the surrounding unknown background.
An unknown time is explicitly flagged, never labelled as known zero motion.
"""
    import numpy as np
    from scripts import motion_plan as mp

    if kind not in ("real", "pred", "zero", "null"):
        raise ValueError(kind)
    height, width = map(int, size)
    frames, latent_hw = len(times), tuple(image_latent.shape[-2:])
    if image_latent.shape[1] != frames:
        raise ValueError("condition and latent time axes disagree")
    result = image_latent.new_zeros(5, frames, *latent_hw, dtype=torch.float32)
    if kind == "null":
        return result
    result[3] = 1.0
    if plan is None:
        raise ValueError("a supplied native5 condition needs an explicit plan (possibly with no objects)")
    if tuple(plan.size) != (height, width):
        raise ValueError("plan and video work frames disagree")
    if kind == "zero":
        plan = mp.zeroed(plan)
    dense = mp.to_dense_target(plan, times)
    flow = torch.zeros(frames, 2, height, width)
    flow[1:] = torch.from_numpy(dense["flow"])
    known = np.zeros((frames, height, width), np.float32)
    # Match to_dense_target's last-object ownership in overlapping source masks.
    for obj in plan.objects:
        for j, when in enumerate(times):
            known[j, obj.mask] = float(mp.affine_at(obj, plan.anchors, float(when))[1])
    valid = torch.from_numpy(known)[:, None]
    pooled_valid = F.interpolate(valid, size=latent_hw, mode="area")
    pooled_flow = F.interpolate(flow * valid, size=latent_hw, mode="area") / pooled_valid.clamp_min(1e-8)
    pooled_flow *= pooled_valid > 0
    result[0], result[1] = pooled_flow[:, 0] / width, pooled_flow[:, 1] / height
    source = torch.from_numpy(dense["object_mask"])[None]
    result[2] = F.interpolate(source, size=latent_hw, mode="area")[0].expand(frames, -1, -1)
    result[4] = pooled_valid[:, 0]
    return result


def load_adapter_state(adapter, state: dict, *, reset_input_stem: bool = False) -> dict:
    """Strict load, or explicit 20/27 -> 5 transfer of every shared body tensor.

Only the entire new stem (weight AND bias) is freshly initialized. No silent
missing-layer tolerance is permitted. Resume/inference always use strict load.
"""
    from motion_jepa.stage2 import ExtendedOnlyFlowAdapter

    if not reset_input_stem:
        if isinstance(adapter, ExtendedOnlyFlowAdapter) and "stem_extra.weight" not in state:
            adapter.load_deployed(state)
        else:
            adapter.load_state_dict(state, strict=True)
        return {"mode": "strict", "reset": []}
    if adapter.stem.in_channels != 5 or state.get("stem.weight", torch.empty(0)).ndim != 5:
        raise ValueError("stem reset is only an explicit legacy20/27 -> native5 initialization")
    if state["stem.weight"].shape[1] != 20:
        raise ValueError("stem reset expects a legacy twenty-channel input stem")
    ignored = {"stem.weight", "stem.bias", "stem_extra.weight"}
    shared = {name: value for name, value in state.items() if name not in ignored}
    expected = set(adapter.state_dict()) - {"stem.weight", "stem.bias"}
    if set(shared) != expected:
        raise ValueError(f"adapter body mismatch: missing={sorted(expected - set(shared))}, unexpected={sorted(set(shared) - expected)}")
    incompatible = adapter.load_state_dict(shared, strict=False)
    if set(incompatible.missing_keys) != {"stem.weight", "stem.bias"} or incompatible.unexpected_keys:
        raise RuntimeError("unexpected adapter transfer mismatch")
    return {"mode": "legacy_body_new_native5_stem", "reset": ["stem.weight", "stem.bias"], "shared_tensors": len(shared)}
