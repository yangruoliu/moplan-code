"""Config loading for Stage-1 WanFlowDiT.

Kept separate from :mod:`motion_jepa.config` because that validator hard-codes
the old MotionJEPA refinement head (``height == grid_height * 16``), which the
new architecture does not have.  Everything derived -- latent grid size, number
of latent frames, horizon timestamps -- is computed in one place so the cache,
the trainer and the sampler cannot silently disagree.
"""

from __future__ import annotations

import json
import math
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .flow_dit_bridge import BRIDGE_LABEL_SOURCE, BRIDGE_MASK_SOURCE, validate_bridge_config
from .bridge_mover_selection import CONDITION_SOURCE as BRIDGE_TASK_OBJECT_MASK_SOURCE


VAE_SPATIAL_RATIO = 8
LEGACY_STATIC_GUARD = "static_predicted_magnitude_px"


def validate_dense_motion_mask_config(config: dict[str, Any]) -> None:
    """Jointly predict a full-grid displacement mask and flow from shared features."""
    decoder = config.get("model", {}).get("decoder", {})
    loss, train, data = config.get("loss", {}), config.get("train", {}), config.get("data", {})
    minimal5 = loss.get("profile") == "minimal5_direction_v1"
    if minimal5:
        from .flow_dit_minimal import validate_minimal_config
        validate_minimal_config(config)
    enabled = decoder.get("dense_motion_mask", False)
    if not isinstance(enabled, bool):
        raise ValueError("decoder.dense_motion_mask must be an explicit boolean")
    slot_activity = decoder.get("task_object_slot_activity", False)
    if not isinstance(slot_activity, bool):
        raise ValueError("decoder.task_object_slot_activity must be an explicit boolean")
    from .flow_dit_selection import validate_selection
    gate_mode = decoder.get("task_object_gate_mode", "soft")
    validate_selection(gate_mode, float(decoder.get("task_object_gate_threshold", .5)), slot_activity)
    consistency = float(loss.get("task_object_slot_consistency", 0.))
    if not math.isfinite(consistency) or consistency < 0 or (consistency and not slot_activity):
        raise ValueError("task_object_slot_consistency requires a nonnegative weight and slot activity")
    if gate_mode == "hard":
        if loss.get("coarse_gate_only", False):
            raise ValueError("hard selection cannot use coarse_gate_only")
        if float(loss.get("coarse_raw_weight", .5)) <= 0 or float(loss.get("event_raw_weight", .5)) <= 0:
            raise ValueError("hard selection requires raw supervision for missed movers")
    if train.get("selection_guard_limits"):
        limits = train["selection_guard_limits"]
        if not isinstance(limits, dict) or any(not math.isfinite(float(v)) or float(v) <= 0 for v in limits.values()):
            raise ValueError("selection_guard_limits must contain finite positive budgets")
        if train.get("static_guard_metric") != "worst_domain_task_object_gated_motion/selection_guard" or float(train.get("static_magnitude_budget_px", 1.)) != 1.:
            raise ValueError("normalized selection guards require the matching metric and budget 1")
    task_identity = decoder.get("task_object_identity", False)
    if not isinstance(task_identity, bool):
        raise ValueError("decoder.task_object_identity must be an explicit boolean")
    gate_source = decoder.get("task_object_gate_source", "activity")
    if gate_source not in {"activity", "identity"}:
        raise ValueError("task_object_gate_source must be activity or identity")
    if gate_source == "identity" and (not task_identity or gate_mode != "hard"):
        raise ValueError("identity selection requires identity head and hard readout")
    rescue = decoder.get("task_object_dense_rescue", False)
    if not isinstance(rescue, bool) or (rescue and (gate_mode != "hard" or not enabled)):
        raise ValueError("task_object_dense_rescue requires explicit boolean and hard dense readout")
    validate_selection("hard", float(decoder.get("task_object_dense_rescue_threshold", .9)))
    semantic_identity = decoder.get("task_object_identity_semantic", False)
    if not isinstance(semantic_identity, bool):
        raise ValueError("decoder.task_object_identity_semantic must be an explicit boolean")
    if semantic_identity and not task_identity:
        raise ValueError("task_object_identity_semantic requires task_object_identity")
    token_attention = decoder.get("task_object_identity_token_attention", False)
    if not isinstance(token_attention, bool) or (token_attention and not semantic_identity):
        raise ValueError("task_object_identity_token_attention requires explicit boolean and semantic identity")
    if semantic_identity and int(decoder.get("prompt_dim", 4096)) <= 0:
        raise ValueError("decoder.prompt_dim must be positive")
    detach_task_shared = decoder.get("task_object_heads_detach_shared", False)
    if not isinstance(detach_task_shared, bool):
        raise ValueError("decoder.task_object_heads_detach_shared must be an explicit boolean")
    if detach_task_shared and not slot_activity:
        raise ValueError("task_object_heads_detach_shared requires task_object_slot_activity")
    mask_heads_only = train.get("mask_heads_only", False)
    if not isinstance(mask_heads_only, bool):
        raise ValueError("train.mask_heads_only must be an explicit boolean")
    if mask_heads_only and train.get("activity_heads_only", False):
        raise ValueError("mask_heads_only and activity_heads_only are mutually exclusive")
    separate_mask_clip = train.get("separate_mask_gradient_clip", False)
    separate_task_clip = train.get("separate_task_object_gradient_clip", False)
    if not isinstance(separate_mask_clip, bool):
        raise ValueError("train.separate_mask_gradient_clip must be an explicit boolean")
    if not isinstance(separate_task_clip, bool):
        raise ValueError(
            "train.separate_task_object_gradient_clip must be an explicit boolean"
        )
    if separate_task_clip and (not separate_mask_clip or not slot_activity):
        raise ValueError(
            "separate task-object clipping requires separate mask clipping and the task-object head"
        )
    detach_shared = decoder.get("dense_motion_mask_detach_shared", False)
    if not isinstance(detach_shared, bool):
        raise ValueError("decoder.dense_motion_mask_detach_shared must be an explicit boolean")
    if detach_shared and not enabled:
        raise ValueError("dense_motion_mask_detach_shared requires dense_motion_mask")
    shared_gradient_scale = float(decoder.get("dense_motion_mask_shared_gradient_scale", 1.0))
    if not math.isfinite(shared_gradient_scale) or not 0 <= shared_gradient_scale <= 1:
        raise ValueError("dense_motion_mask_shared_gradient_scale must lie in [0, 1]")
    if shared_gradient_scale != 1.0 and not enabled:
        raise ValueError("dense_motion_mask_shared_gradient_scale requires dense_motion_mask")
    weight = float(loss.get("dense_motion_mask", 0.0))
    dice_weight = float(loss.get("dense_motion_mask_dice_weight", 0.5))
    pixel_fraction = float(loss.get("dense_motion_mask_pixel_bce_fraction", 0.0))
    target_mode = str(loss.get("dense_motion_mask_target_mode", "pixel_displacement_v1"))
    gated_flow_weight = float(loss.get("gated_flow_charbonnier", 0.0))
    from .flow_dit_coarse import validate_coarse_config
    coarse_weights = validate_coarse_config(loss)
    if any(coarse_weights.values()) and float(loss.get("coarse_gated_weight", 1.)) > 0 and not slot_activity:
        raise ValueError("coarse gated supervision requires task_object_slot_activity")
    gated_field = loss.get("gated_flow_field", "flow_activity_gated")
    if gated_field not in {"flow_activity_gated", "flow_task_object_gated"}:
        raise ValueError("gated_flow_field must name a supported gated flow")
    if gated_flow_weight and gated_field == "flow_task_object_gated" and not slot_activity:
        raise ValueError("task-object gated flow loss requires task_object_slot_activity")
    slot_activity_weight = float(loss.get("task_object_slot_activity", 0.0))
    task_identity_weight = float(loss.get("task_object_identity", 0.0))
    slot_prior_fraction = float(
        loss.get("task_object_slot_prior_bce_fraction", 0.0)
    )
    identity_prior_fraction = float(
        loss.get("task_object_identity_prior_bce_fraction", 0.0)
    )
    if not math.isfinite(pixel_fraction) or not 0 <= pixel_fraction <= 1:
        raise ValueError("dense_motion_mask_pixel_bce_fraction must lie in [0,1]")
    if not math.isfinite(slot_prior_fraction) or not 0 <= slot_prior_fraction <= 1:
        raise ValueError("task_object_slot_prior_bce_fraction must lie in [0,1]")
    if not math.isfinite(identity_prior_fraction) or not 0 <= identity_prior_fraction <= 1:
        raise ValueError("task_object_identity_prior_bce_fraction must lie in [0,1]")
    if (not math.isfinite(weight) or weight < 0
            or not math.isfinite(dice_weight) or dice_weight < 0
            or not math.isfinite(gated_flow_weight) or gated_flow_weight < 0
            or not math.isfinite(slot_activity_weight) or slot_activity_weight < 0
            or not math.isfinite(task_identity_weight) or task_identity_weight < 0):
        raise ValueError("dense_motion_mask loss weights must be finite and nonnegative")
    if target_mode not in {
        "pixel_displacement_v1", "whole_object_motion_v1",
        "whole_object_motion_v2_authoritative",
        "whole_object_motion_v3_task_object",
    }:
        raise ValueError("unknown dense_motion_mask_target_mode")
    if not enabled:
        if (weight or gated_flow_weight or slot_activity or slot_activity_weight
                or slot_prior_fraction or task_identity or task_identity_weight
                or identity_prior_fraction or mask_heads_only or detach_task_shared
                or semantic_identity):
            raise ValueError("dense_motion_mask loss requires the dense_motion_mask decoder branch")
        return
    if decoder.get("type", "dense") != "slots" or decoder.get("temporal_slots", True) is not True:
        raise ValueError("dense_motion_mask requires the temporal slots decoder")
    if (decoder.get("motion_activity", False) or decoder.get("motion_activity_gate", False)
            or decoder.get("dense_correction", False) or train.get("activity_heads_only", False)):
        raise ValueError("dense_motion_mask is mutually exclusive with legacy activity, gating, correction and heads-only modes")
    if train.get("objective") != "direct" or float(loss.get("flow_matching", 1)) != 0:
        raise ValueError("dense_motion_mask requires direct objective and flow_matching=0")
    if train.get("refine_from"):
        raise ValueError("dense_motion_mask requires one shared predictor, without a refine partner")
    if (loss.get("supervision_mode") != "valid_partition_v1"
            or data.get("supervision_mode", loss.get("supervision_mode")) != "valid_partition_v1"):
        raise ValueError("dense_motion_mask requires full-valid valid_partition_v1 supervision")
    if weight <= 0 and not minimal5:
        raise ValueError("dense_motion_mask requires explicit positive dense_motion_mask loss weight")
    if any(float(loss.get(key, 0)) != 0 for key in ("object_activity", "pixel_activity")):
        raise ValueError("dense_motion_mask uses its full-grid loss, not legacy slot activity losses")
    if target_mode.startswith("whole_object_motion_") and int(data.get("emit_instance_masks", 0)) <= 0:
        raise ValueError("whole-object motion targets require data.emit_instance_masks > 0")
    if slot_activity:
        if decoder.get("oracle_instance_masks", False) is not True:
            raise ValueError("task_object_slot_activity requires deployable input instance masks")
        if target_mode != "whole_object_motion_v3_task_object":
            raise ValueError("task_object_slot_activity requires the v3 task-object target")
        if slot_activity_weight <= 0 and not minimal5:
            raise ValueError("task_object_slot_activity requires an explicit positive loss weight")
    elif slot_activity_weight:
        raise ValueError(
            "task_object_slot_activity loss requires decoder.task_object_slot_activity=True"
        )
    elif slot_prior_fraction:
        raise ValueError(
            "task_object_slot_prior_bce_fraction requires decoder.task_object_slot_activity=True"
        )
    if task_identity:
        if not slot_activity:
            raise ValueError(
                "task_object_identity requires decoder.task_object_slot_activity=True"
            )
        if task_identity_weight <= 0 and not minimal5:
            raise ValueError(
                "task_object_identity requires an explicit positive loss weight"
            )
    elif task_identity_weight:
        raise ValueError(
            "task_object_identity loss requires decoder.task_object_identity=True"
        )
    elif identity_prior_fraction:
        raise ValueError(
            "task_object_identity_prior_bce_fraction requires "
            "decoder.task_object_identity=True"
        )
    if mask_heads_only:
        if not task_identity:
            raise ValueError("mask_heads_only requires task_object_identity=True")


def validate_motion_vector_loss_config(config: dict[str, Any]) -> None:
    """Validate explicit direction/magnitude terms for true movers.

    These terms are opt-in so historical checkpoints and configs retain their
    exact objective.  They require the trusted valid-partition sidecars: an
    unknown low-flow pixel must never be silently treated as a static negative.
    """
    loss = config.get("loss", {})
    weights = {
        key: float(loss.get(key, 0.0))
        for key in (
            "moving_direction", "moving_parallel_magnitude",
            "object_motion_direction", "object_motion_parallel_magnitude",
        )
    }
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("motion direction/magnitude loss weights must be finite and nonnegative")
    affine_weights = {
        key: float(loss.get(key, 0.0))
        for key in (
            "task_object_affine_displacement",
            "task_object_affine_direction",
            "task_object_affine_parallel_magnitude",
            "task_object_affine_trajectory",
        )
    }
    if any(not math.isfinite(value) or value < 0 for value in affine_weights.values()):
        raise ValueError("task-object affine loss weights must be finite and nonnegative")
    task_flow_weights = {
        key: float(loss.get(key, 0.0))
        for key in (
            "task_object_flow_displacement",
            "task_object_flow_direction",
            "task_object_flow_parallel_magnitude",
            "task_object_flow_trajectory",
        )
    }
    if any(not math.isfinite(value) or value < 0 for value in task_flow_weights.values()):
        raise ValueError("task-object final-flow loss weights must be finite and nonnegative")
    if (
        not any(weights.values())
        and not any(affine_weights.values())
        and not any(task_flow_weights.values())
    ):
        return
    if loss.get("supervision_mode") != "valid_partition_v1":
        raise ValueError("motion direction/magnitude losses require valid_partition_v1")
    for key, default in (
        ("motion_direction_epsilon_px", 0.5),
        ("motion_magnitude_scale_px", 16.0),
        ("motion_magnitude_beta_px", 1.0),
    ):
        value = float(loss.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"loss.{key} must be finite and positive")
    if (
        weights["object_motion_direction"]
        or weights["object_motion_parallel_magnitude"]
    ) and int(config.get("data", {}).get("emit_instance_masks", 0)) <= 0:
        raise ValueError("object motion vector losses require data.emit_instance_masks > 0")
    if any(affine_weights.values()):
        decoder = config.get("model", {}).get("decoder", {})
        if loss.get("supervision_mode") != "valid_partition_v1":
            raise ValueError("task-object affine supervision requires valid_partition_v1")
        if decoder.get("type") != "slots" or decoder.get("oracle_instance_masks") is not True:
            raise ValueError(
                "task-object affine supervision requires the oracle-input slot decoder"
            )
        if int(config.get("data", {}).get("emit_instance_masks", 0)) <= 0:
            raise ValueError("task-object affine supervision requires data.emit_instance_masks > 0")
        positive_fraction = float(loss.get("task_object_identity_positive_fraction", .5))
        min_pixels = float(loss.get("task_object_identity_min_pixels", 4))
        if not math.isfinite(positive_fraction) or not 0 < positive_fraction <= 1:
            raise ValueError("task-object affine positive fraction must lie in (0,1]")
        if not math.isfinite(min_pixels) or min_pixels < 1 or not min_pixels.is_integer():
            raise ValueError("task-object affine minimum pixels must be a positive integer")
    if any(task_flow_weights.values()):
        if int(config.get("data", {}).get("emit_instance_masks", 0)) <= 0:
            raise ValueError(
                "task-object final-flow supervision requires data.emit_instance_masks > 0"
            )
        for key, default in (
            ("trajectory_velocity_epsilon_px", .5),
            ("trajectory_velocity_scale_px", 32.),
        ):
            value = float(loss.get(key, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"loss.{key} must be finite and positive")


def validate_motion_activity_config(config: dict[str, Any]) -> None:
    """Require explicit, independently supervised per-object activity heads."""
    decoder = config.get("model", {}).get("decoder", {})
    loss, train, data = config.get("loss", {}), config.get("train", {}), config.get("data", {})
    enabled = decoder.get("motion_activity", False)
    gated = decoder.get("motion_activity_gate", False)
    heads_only = train.get("activity_heads_only", False)
    if not isinstance(heads_only, bool):
        raise ValueError("train.activity_heads_only must be an explicit boolean")
    if heads_only and (not enabled or gated):
        raise ValueError("activity_heads_only requires motion_activity=True and motion_activity_gate=False")
    if heads_only and train.get("refine_from"):
        raise ValueError("activity_heads_only does not support a second refine partner")
    if not isinstance(enabled, bool) or not isinstance(gated, bool):
        raise ValueError("decoder.motion_activity and motion_activity_gate must be explicit booleans")
    if gated and not enabled:
        raise ValueError("motion_activity_gate requires motion_activity")
    weights = {name: float(loss.get(name, 0)) for name in ("object_activity", "pixel_activity")}
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("object_activity and pixel_activity weights must be finite and nonnegative")
    if not enabled:
        if any(weights.values()):
            raise ValueError("activity loss weights require decoder.motion_activity=True")
        return
    if decoder.get("type", "dense") != "slots":
        raise ValueError("motion_activity requires the slots decoder")
    if (decoder.get("temporal_slots", True) is not True
            or decoder.get("oracle_instance_masks", False) is not True):
        raise ValueError("motion_activity requires temporal_slots=True and oracle_instance_masks=True")
    if decoder.get("dense_correction", False):
        raise ValueError("motion_activity pilot requires dense_correction=False")
    if train.get("objective") != "direct" or float(loss.get("flow_matching", 1)) != 0:
        raise ValueError("motion_activity requires direct objective and flow_matching=0")
    if (loss.get("supervision_mode") != "valid_partition_v1"
            or data.get("supervision_mode", loss.get("supervision_mode")) != "valid_partition_v1"):
        raise ValueError("motion_activity requires full-valid valid_partition_v1 supervision")
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("motion_activity requires explicit positive object_activity and pixel_activity weights")


def validate_dense_correction_config(config: dict[str, Any]) -> None:
    """The optional capacity candidate needs full-support, decoded supervision."""
    decoder = config.get("model", {}).get("decoder", {})
    enabled = decoder.get("dense_correction", False)
    if not isinstance(enabled, bool):
        raise ValueError("decoder.dense_correction must be an explicit boolean")
    if not enabled:
        return
    if decoder.get("type", "dense") != "slots":
        raise ValueError("dense_correction is a slots-decoder capacity candidate")
    if float(decoder.get("dense_correction_gain", .05)) != .05:
        raise ValueError("dense_correction v1 requires dense_correction_gain=0.05")
    loss, train, data = config.get("loss", {}), config.get("train", {}), config.get("data", {})
    if train.get("objective") != "direct" or float(loss.get("flow_matching", 1)) != 0:
        raise ValueError("dense_correction requires direct objective and flow_matching=0")
    if (loss.get("supervision_mode") != "valid_partition_v1"
            or data.get("supervision_mode", loss.get("supervision_mode")) != "valid_partition_v1"):
        raise ValueError("dense_correction requires full-valid valid_partition_v1 supervision")
    if (train.get("selection_metric") != "raw_motion/moving_relative_epe"
            or train.get("static_guard_metric") != "raw_motion/static_pred_magnitude_px"):
        raise ValueError("dense_correction requires raw full-valid motion selection and static guard")
    budget = float(train.get("static_magnitude_budget_px", float("nan")))
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("dense_correction requires an explicit finite static_magnitude_budget_px")


def validation_selection(
    summary: dict[str, Any], *, best: float, selection_metric: str,
    static_budget: float, min_delta: float,
    static_guard_metric: str = LEGACY_STATIC_GUARD,
    strict_selection: bool = True,
) -> tuple[float, float, bool, bool]:
    """One criterion shared by live checkpoint selection and log recovery.

    Legacy runs allowed absent static support. Explicit new guards fail closed
    when undefined, so an unscored background cannot silently pass the budget.
    """
    current = float(summary.get(selection_metric, float("nan")))
    if strict_selection and not math.isfinite(current):
        raise KeyError(f"selection_metric {selection_metric!r} is missing or non-finite")
    guard = float(summary.get(static_guard_metric, float("nan")))
    if static_guard_metric != LEGACY_STATIC_GUARD and not math.isfinite(guard):
        raise KeyError(f"static_guard_metric {static_guard_metric!r} is missing or non-finite")
    within_budget = not math.isfinite(guard) or guard <= static_budget
    improved = math.isfinite(current) and current < best * (1.0 - min_delta) and within_budget
    return current, guard, within_budget, improved


def validate_selection_resume(
    state: dict[str, Any], *, selection_metric: str, static_guard_metric: str,
    static_budget: float, min_delta: float,
) -> None:
    """Do not carry best/stale state across a change in selection semantics."""
    recorded = state.get("selection_metric")
    if recorded and recorded != selection_metric:
        raise ValueError(f"resume checkpoint used a different selection metric: {recorded!r} vs {selection_metric!r}")
    recorded_guard = state.get("static_guard_metric", LEGACY_STATIC_GUARD)
    if recorded_guard != static_guard_metric:
        raise ValueError(f"resume checkpoint used a different static guard metric: {recorded_guard!r} vs {static_guard_metric!r}")
    for key, expected in (("static_magnitude_budget_px", static_budget), ("early_stop_min_delta", min_delta)):
        if key in state and float(state[key]) != expected:
            raise ValueError(f"resume checkpoint used a different {key}")


def recover_validation_state(
    path: str | Path,
    *,
    selection_metric: str,
    static_budget: float,
    min_delta: float,
    static_guard_metric: str = LEGACY_STATIC_GUARD,
) -> tuple[float, float, int]:
    """Recover early-stopping state from an older checkpoint's metrics log.

    Checkpoints written before the chunked-run protocol did not persist the
    best/stale values. Replaying only real validation records preserves their
    selection semantics without treating the zero-flow initial record as a
    trained checkpoint.
    """
    path = Path(path)
    best_selection = float("inf")
    best_object_epe = float("inf")
    stale = 0
    if not path.is_file():
        return best_selection, best_object_epe, stale
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") != "validation":
                continue
            if "static_guard_metric" in record:
                validate_selection_resume(record, selection_metric=selection_metric,
                                          static_guard_metric=static_guard_metric,
                                          static_budget=static_budget, min_delta=min_delta)
            object_epe = float(
                record.get("metric/object_epe_norm_micro", float("nan"))
            )
            if math.isfinite(object_epe):
                best_object_epe = min(best_object_epe, object_epe)
            metrics = {key[len("metric/"):]: value for key, value in record.items() if key.startswith("metric/")}
            current, _, _, improved = validation_selection(
                metrics, best=best_selection, selection_metric=selection_metric,
                static_guard_metric=static_guard_metric, static_budget=static_budget,
                min_delta=min_delta, strict_selection=False,
            )
            if improved:
                best_selection = current
                stale = 0
            else:
                stale += 1
    return best_selection, best_object_epe, stale


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    config = deepcopy(config)
    validate_config(config)
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` CLI overrides using YAML scalar parsing."""
    result = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value: {override}")
        dotted, raw = override.split("=", 1)
        keys = dotted.split(".")
        target: dict[str, Any] = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise ValueError(f"unknown override path: {dotted}")
            target = target[key]
        optional_caption_path = dotted in {"paths.caption_overrides", "data.secondary.caption_overrides"}
        if keys[-1] not in target and not optional_caption_path:
            raise ValueError(f"unknown override key: {dotted}")
        target[keys[-1]] = yaml.safe_load(raw)
    validate_config(result)
    return result


def validate_config(config: dict[str, Any]) -> None:
    for section in ("paths", "data", "codec", "model", "loss", "train", "evaluation"):
        if section not in config:
            raise ValueError(f"missing config section: {section}")
    profile = config["loss"].get("profile", "legacy")
    if profile not in {"legacy", "minimal5_direction_v1", "clean_trajectory_v1", "unified_trajectory_v1"}:
        raise ValueError(f"unknown loss profile {profile!r}")
    from .flow_dit_unified import validate_unified_campaign, validate_unified_config
    validate_unified_campaign(config)
    if profile == "unified_trajectory_v1":
        validate_unified_config(config)
    if profile == "clean_trajectory_v1":
        from .flow_dit_clean import validate_clean_config
        validate_clean_config(config)
    decoder = config["model"].get("decoder", {})
    if decoder.get("slot_pooling_mode", "bilinear") not in {"bilinear", "area"}:
        raise ValueError("slot_pooling_mode must be bilinear or area")
    if not isinstance(decoder.get("text_slot_fusion", False), bool):
        raise ValueError("text_slot_fusion must be an explicit boolean")
    if decoder.get("text_slot_fusion", False) and not decoder.get("text_slot_binding", False):
        raise ValueError("text_slot_fusion requires text_slot_binding")
    validate_dense_correction_config(config)
    validate_motion_activity_config(config)
    validate_dense_motion_mask_config(config)
    validate_motion_vector_loss_config(config)
    data = config["data"]
    additional = data.get("additional_sources", [])
    if additional is None:
        additional = []
    if not isinstance(additional, list) or any(not isinstance(item, dict) for item in additional):
        raise ValueError("data.additional_sources must be a list of source mappings")
    named_blocks = [("paths", config["paths"]), ("data.secondary", data.get("secondary"))]
    named_blocks.extend(
        (f"data.additional_sources[{index}]", block)
        for index, block in enumerate(additional)
    )
    for name, block in named_blocks:
        if block is not None and "caption_overrides" in block:
            value = block["caption_overrides"]
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name}.caption_overrides must be a train-only file path or null")
            if value:
                pad = data.get("prompt_pad_length")
                if type(pad) is not int or pad < 1:
                    raise ValueError("caption overrides require positive integer data.prompt_pad_length")
    future = list(data["future_indices"])
    if len(future) != len(data["horizon_seconds"]):
        raise ValueError("future_indices and horizon_seconds disagree")
    if any(index <= 0 for index in future):
        raise ValueError("future_indices must exclude frame zero")
    if config["codec"]["backend"] == "vae" and int(data["flow_resolution"]) % VAE_SPATIAL_RATIO:
        raise ValueError(
            f"flow_resolution {data['flow_resolution']} is not divisible by {VAE_SPATIAL_RATIO}"
        )
    if int(data["height"]) != int(data["width"]):
        raise ValueError("Stage-1 metrics assume a square evaluation grid")
    expert = config["model"]["expert"]
    if expert not in ("low", "high", "single"):
        raise ValueError(f"model.expert must be low, high or single, got {expert!r}")

    # Exact instance masks are labels, not deployable first-frame inputs.  Two
    # switches can put them in the conditioning path: union/boundary channels
    # into the Wan backbone, and per-instance masks into the slot decoder.  An
    # oracle-mask configuration must never be labelled as a first-frame+caption
    # model, so make that mismatch impossible to load.
    decoder = config["model"].get("decoder", {})
    emitted_masks = int(data.get("emit_instance_masks", 0))
    use_instance_channels = bool(config["model"].get("use_instance_channels", False))
    oracle_instance_masks = bool(decoder.get("oracle_instance_masks", False))
    if (use_instance_channels or oracle_instance_masks) and emitted_masks <= 0:
        raise ValueError(
            "instance-mask conditioning is enabled but data.emit_instance_masks is zero"
        )
    if oracle_instance_masks and emitted_masks != int(decoder.get("num_slots", 0)):
        raise ValueError(
            "oracle_instance_masks requires data.emit_instance_masks == "
            "model.decoder.num_slots"
        )
    scope = str(config.get("provenance", {}).get("condition_scope", ""))
    scope_declares_ground_truth_masks = "ground_truth_instance_masks" in scope
    sources = [{
        "label_source": config.get("provenance", {}).get("label_source"),
        "horizon_seconds": data["horizon_seconds"],
        "condition_instance_cache_dir": config.get("paths", {}).get(
            "condition_instance_cache_dir", ""
        ),
        "condition_mask_source": config.get("paths", {}).get("condition_mask_source", ""),
        "supervision_instance_masks_from_condition": config.get("paths", {}).get(
            "supervision_instance_masks_from_condition", False
        ),
        "supervision_instance_cache_dir": config.get("paths", {}).get(
            "supervision_instance_cache_dir", ""
        ),
        "supervision_instance_mask_source": config.get("paths", {}).get(
            "supervision_instance_mask_source", ""
        ),
    }]
    if data.get("secondary") is not None:
        sources.append(data["secondary"])
    sources.extend(additional)
    uses_masks = emitted_masks > 0 and (use_instance_channels or oracle_instance_masks)
    uses_ground_truth_masks = uses_masks and any(
        source.get("label_source") != BRIDGE_LABEL_SOURCE
        and source.get("condition_mask_source") != BRIDGE_MASK_SOURCE
        for source in sources
    )
    if uses_ground_truth_masks != scope_declares_ground_truth_masks:
        raise ValueError(
            "provenance.condition_scope must declare ground_truth_instance_masks "
            "if and only if exact instance masks enter the conditioning path"
        )
    uses_sam_masks = uses_masks and any(
        source.get("label_source") == BRIDGE_LABEL_SOURCE
        or source.get("condition_mask_source") == BRIDGE_MASK_SOURCE
        for source in sources
    )
    if uses_sam_masks != (BRIDGE_MASK_SOURCE in scope):
        raise ValueError(
            "provenance.condition_scope must declare first_frame_sam2_instance_masks "
            "if and only if the audited Bridge source supplies mask conditions"
        )
    validate_bridge_config(config, sources)
    for source in sources:
        condition_dir = source.get("condition_instance_cache_dir")
        condition_source = source.get("condition_mask_source")
        supervision_from_condition = source.get(
            "supervision_instance_masks_from_condition", False
        )
        supervision_dir = source.get("supervision_instance_cache_dir", "")
        supervision_source = source.get("supervision_instance_mask_source", "")
        if not isinstance(supervision_from_condition, bool):
            raise ValueError(
                "supervision_instance_masks_from_condition must be an explicit boolean"
            )
        if bool(condition_dir) != bool(condition_source):
            raise ValueError(
                "condition_instance_cache_dir and condition_mask_source must be configured together"
            )
        if condition_source and condition_source != BRIDGE_MASK_SOURCE:
            raise ValueError("only audited first-frame SAM2 condition caches are supported")
        if supervision_from_condition and (
            not condition_dir or source.get("label_source") != BRIDGE_LABEL_SOURCE
        ):
            raise ValueError(
                "condition-mask supervision requires a Bridge source with an external condition cache"
            )
        if bool(supervision_dir) != bool(supervision_source):
            raise ValueError(
                "supervision_instance_cache_dir and supervision_instance_mask_source "
                "must be configured together"
            )
        if supervision_dir and source.get("label_source") != BRIDGE_LABEL_SOURCE:
            raise ValueError(
                "external task-object supervision masks are only valid for Bridge"
            )
        if supervision_source and supervision_source != BRIDGE_TASK_OBJECT_MASK_SOURCE:
            raise ValueError("unsupported Bridge task-object supervision mask source")
        if supervision_dir and supervision_from_condition:
            raise ValueError(
                "choose either condition-mask or separate task-object supervision"
            )
    if any(source.get("condition_instance_cache_dir") for source in sources) and float(
        config.get("loss", {}).get("slot_affine", 0.0)
    ) != 0:
        raise ValueError(
            "external condition masks have no GT slot correspondence; set loss.slot_affine=0"
        )

    mixture_sources = []
    if data.get("secondary") is not None:
        mixture_sources.append(("data.secondary", data["secondary"]))
    mixture_sources.extend(
        (f"data.additional_sources[{index}]", source)
        for index, source in enumerate(additional)
    )
    mixture_fraction = 0.0
    names = set()
    for source_name, secondary in mixture_sources:
        fraction = float(secondary.get("mix_fraction", -1.0))
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                f"{source_name}.mix_fraction must explicitly lie between 0 and 1; "
                "dataset-length-proportional mixing is not a controlled comparison"
            )
        mixture_fraction += fraction
        metric_name = secondary.get(
            "name", "phyco" if source_name == "data.secondary" else ""
        )
        if (
            not isinstance(metric_name, str) or not metric_name
            or not metric_name.replace("_", "").isalnum()
            or metric_name in names
        ):
            raise ValueError(f"{source_name}.name must be a unique alphanumeric identifier")
        names.add(metric_name)
        temporal = secondary.get("temporal_subsample_indices")
        if temporal is not None and (
            not isinstance(temporal, list)
            or any(type(value) is not int or value < 0 for value in temporal)
            or temporal != sorted(set(temporal))
            or len(temporal) != len(secondary.get("horizon_seconds", []))
        ):
            raise ValueError(
                "data.secondary.temporal_subsample_indices must be sorted unique "
                "nonnegative integers matching its horizon count"
            )
    if mixture_sources and not 0.0 < mixture_fraction < 1.0:
        raise ValueError(
            "the sum of secondary/additional source mix_fraction values must lie between 0 and 1"
        )

    supervision_mode = config["loss"].get("supervision_mode", "legacy")
    minimal5 = config["loss"].get("profile") in {"minimal5_direction_v1", "clean_trajectory_v1", "unified_trajectory_v1"}
    if (config["train"].get("freeze_visibility_heads", False)
            and not minimal5 and float(config["loss"].get("visibility_bce", 0.1)) != 0):
        raise ValueError("freeze_visibility_heads requires loss.visibility_bce=0")
    if supervision_mode not in {"legacy", "valid_partition_v1"}:
        raise ValueError(f"unknown loss.supervision_mode {supervision_mode!r}")
    if supervision_mode == "valid_partition_v1":
        if "moving_weight" in config["loss"]:
            warnings.warn(
                "loss.moving_weight is ignored by valid_partition_v1; use "
                "loss.flow_partition_weights.moving (effective value: "
                f"{config['loss'].get('flow_partition_weights', {}).get('moving', 1.0)}).",
                UserWarning,
                stacklevel=2,
            )
        for key, default in (("object_occupancy", 0), ("position", 0.2),
                             ("velocity", 0.05), ("acceleration", 0.01)):
            if not minimal5 and float(config["loss"].get(key, default)) != 0:
                raise ValueError(f"valid_partition_v1 requires loss.{key}=0")
        if (float(config["loss"].get("flow_matching", 1)) > 0
                and not config["loss"].get("allow_unmasked_latent_supervision", False)):
            raise ValueError(
                "valid_partition_v1 does not mask cached latent targets: set "
                "loss.flow_matching=0, or explicitly acknowledge legacy targets "
                "with loss.allow_unmasked_latent_supervision=true"
            )


def horizons(config: dict[str, Any]) -> int:
    return len(config["data"]["future_indices"])


def latent_frames(config: dict[str, Any]) -> int:
    """Flow-video length: the 7 horizons plus the known zero-flow slot 0."""
    return horizons(config) + 1


def latent_size(config: dict[str, Any]) -> int:
    if config["codec"]["backend"] == "vae":
        return int(config["data"]["flow_resolution"]) // VAE_SPATIAL_RATIO
    return int(config["data"].get("direct_latent_size", config["data"]["flow_resolution"] // 4))


def latent_channels(config: dict[str, Any]) -> int:
    return 16 if config["codec"]["backend"] == "vae" else 3


def horizon_seconds(config: dict[str, Any]) -> list[float]:
    return [float(value) for value in config["data"]["horizon_seconds"]]


def resolve_derived(config: dict[str, Any]) -> dict[str, Any]:
    """Everything downstream code needs, computed once."""
    return {
        "horizons": horizons(config),
        "latent_frames": latent_frames(config),
        "latent_size": latent_size(config),
        "latent_channels": latent_channels(config),
        "horizon_seconds": horizon_seconds(config),
        "latent_shape": (
            latent_channels(config),
            latent_frames(config),
            latent_size(config),
            latent_size(config),
        ),
    }
