"""Audited Bridge teacher/input contract, isolated from simulator policies.

SAM is a first-frame condition and segmentation proxy, never correspondence
validity or motion truth. Future occupancy is unknown; its zero tensor exists
only for the old dataset ABI and must never enter an occupancy loss.
"""
from __future__ import annotations

import math
from typing import Any

import torch

BRIDGE_LABEL_SOURCE = "bridge_real_raft_fb_bounded_v2"
BRIDGE_MASK_SOURCE = "first_frame_sam2_instance_masks"
BRIDGE_FRAME_OFFSETS = [3, 6, 9, 11, 14, 17, 20]
BRIDGE_HORIZONS = [v / 5.0 for v in BRIDGE_FRAME_OFFSETS]


def validate_bridge_times(times: list[float]) -> None:
    if len(times) != 7 or any(
        not math.isfinite(float(a)) or abs(float(a) - b) > 1e-6
        for a, b in zip(times, BRIDGE_HORIZONS)
    ):
        raise ValueError("Bridge horizons must match actual frame offsets / 5 fps")


def _binary(value: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
            or not ((value == 0) | (value == 1)).all()):
        raise ValueError(f"Bridge invalid {name}")
    return value.bool()


def validate_bridge_target(target: dict[str, Any]) -> None:
    """Reject unpromoted candidates or silent changes to reviewed semantics."""
    if (target.get("label_source") != BRIDGE_LABEL_SOURCE
            or target.get("kind") != "stage1_bridge_target"
            or target.get("bridge_promotion_version") != 1
            or target.get("training_ready") is not True):
        raise ValueError("Bridge requires explicitly promoted target schema")
    if (target.get("input_mask_source") != BRIDGE_MASK_SOURCE
            or target.get("SAM_is_motion_GT") is not False):
        raise ValueError("Bridge requires first-frame SAM input provenance")
    conditioning = target.get("conditioning_provenance", {})
    if (conditioning.get("first_frame_only") is not True
            or conditioning.get("motion_supervision_support") is not False
            or target.get("conditioning_variant") not in {"raw", "filtered"}):
        raise ValueError("Bridge condition may not depend on future motion")
    validate_bridge_times(target.get("horizon_seconds", []))
    if target.get("frame_offsets") != BRIDGE_FRAME_OFFSETS or target.get("source_fps") != 5.0:
        raise ValueError("Bridge frame/fps contract mismatch")
    provenance = target.get("provenance", {})
    for key in ("source_candidate_sha256", "first_frame_sha256", "source_video_sha256"):
        digest = provenance.get(key, "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Bridge missing/invalid {key}")
    if (not target.get("caption") or target.get("caption_source") != "original_episode_task_instruction"
            or target.get("video_path") != provenance.get("first_frame_path")):
        raise ValueError("Bridge caption/retained first-frame input contract mismatch")
    pts = provenance.get("actual_pts_seconds", [])
    start = float(target.get("source_start_frame", -1)) / 5.0
    if len(pts) != 21 or any(
        not math.isfinite(float(t)) or abs(float(t) - (start + i / 5.0)) > 0.01
        for i, t in enumerate(pts)
    ):
        raise ValueError("Bridge actual PTS do not match episode frame indices")
    if target.get("source_stop_frame_exclusive", -1) < target.get("source_start_frame", 0) + 21:
        raise ValueError("Bridge clip crosses episode boundary")
    flow = target.get("flow")
    if (not isinstance(flow, torch.Tensor) or flow.ndim != 4
            or flow.shape[:2] != (7, 2) or flow.shape[-2] != flow.shape[-1]
            or not torch.isfinite(flow).all()):
        raise ValueError("Bridge invalid finite stored flow")
    h, w = flow.shape[-2:]
    valid = _binary(target.get("flow_valid"), (7, 1, h, w), "flow_valid")
    content = _binary(target.get("content_valid"), (1, h, w), "content_valid")
    visible = _binary(target.get("visibility"), tuple(valid.shape), "visibility")
    if not torch.equal(visible, valid) or (valid & ~content[None]).any():
        raise ValueError("Bridge visibility must equal bounded full-content correspondence support")
    if flow.masked_select(~valid.expand_as(flow)).count_nonzero():
        raise ValueError("Bridge unknown flow placeholders must be zero")
    ids = target.get("instance_id")
    if (not isinstance(ids, torch.Tensor) or tuple(ids.shape) != (1, h, w)
            or not torch.isfinite(ids).all() or not torch.equal(ids, ids.long())
            or (ids < 0).any() or int(ids.max()) > 64 or ((ids > 0) & ~content).any()):
        raise ValueError("Bridge invalid first-frame SAM instance IDs")
    union = _binary(target.get("object_mask"), (1, h, w), "object_mask")
    if not torch.equal(union, ids > 0):
        raise ValueError("Bridge legacy object_mask must be the first-frame SAM union")
    future = _binary(target.get("future_mask"), tuple(valid.shape), "future_mask")
    future_valid = _binary(target.get("future_mask_valid"), tuple(valid.shape), "future_mask_valid")
    if future.any() or future_valid.any() or target.get("future_mask_semantics") != "unknown_ABI_placeholder_never_supervise":
        raise ValueError("Bridge future occupancy is unknown, not a zero target")


def validate_bridge_config(config: dict[str, Any], sources: list[dict[str, Any]]) -> None:
    """Bridge's partial labels may only use the explicitly masked direct loss."""
    bridge = [s for s in sources if s.get("label_source") == BRIDGE_LABEL_SOURCE]
    if not bridge:
        return
    loss, data, train = config["loss"], config["data"], config["train"]
    if loss.get("supervision_mode", "legacy") != "valid_partition_v1" or data.get(
        "supervision_mode", loss.get("supervision_mode")
    ) != "valid_partition_v1":
        raise ValueError("Bridge requires valid_partition_v1 data and loss")
    if train.get("objective", "flow_matching") != "direct":
        raise ValueError("Bridge requires direct objective; no future latent may enter the input")
    for name, default in (("flow_matching", 1), ("visibility_bce", .1),
                          ("object_occupancy", 0), ("position", .2),
                          ("velocity", .05), ("acceleration", .01)):
        if float(loss.get(name, default)) != 0:
            raise ValueError(f"Bridge requires loss.{name}=0; unavailable labels may not be supervised")
    for source in bridge:
        validate_bridge_times(source.get("horizon_seconds", []))

