"""Weight-only initialization for a new, independently scheduled S1 run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


# These fields only control how an auxiliary loss enters the shared trunk.
# They do not add, remove, or reshape checkpoint parameters and are expected to
# change when a weight-only initialization starts a new optimization objective.
DECODER_OPTIMIZATION_ONLY_FIELDS = frozenset({
    "dense_motion_mask_detach_shared",
    "dense_motion_mask_shared_gradient_scale",
    "task_object_heads_detach_shared",
})
DECODER_READOUT_DEFAULTS = {"task_object_gate_mode": "soft", "task_object_gate_threshold": .5,
    "task_object_gate_source": "activity", "task_object_dense_rescue": False,
    "task_object_dense_rescue_threshold": .9}
DECODER_READOUT_ONLY_FIELDS = frozenset(DECODER_READOUT_DEFAULTS)
TASK_OBJECT_SLOT_FIELD = "task_object_slot_activity"
TASK_OBJECT_SLOT_STATE_PREFIX = "task_object_slot_head."
TASK_OBJECT_IDENTITY_FIELD = "task_object_identity"
TASK_OBJECT_IDENTITY_SEMANTIC_FIELD = "task_object_identity_semantic"
TASK_OBJECT_IDENTITY_STATE_PREFIX = "task_object_identity_head."
TASK_OBJECT_IDENTITY_SEMANTIC_STATE_PREFIXES = (
    "task_identity_visual_project.", "task_identity_prompt_project.",
    "task_identity_attention.",
)
TEXT_SLOT_BINDING_FIELDS = frozenset({
    "text_slot_binding", "text_slot_binding_heads", "text_slot_binding_max_gain",
})
TEXT_SLOT_BINDING_STATE_PREFIXES = (
    "tsb_token_project.", "tsb_query_project.", "tsb_attention.", "tsb_score.",
)


def decoder_architecture_contract(value: dict) -> dict:
    contract = {
        key: item for key, item in value.items()
        if key not in DECODER_OPTIMIZATION_ONLY_FIELDS | DECODER_READOUT_ONLY_FIELDS
    }
    contract.setdefault("slot_pooling_mode", "bilinear")
    contract.setdefault("text_slot_fusion", False)
    contract.setdefault("task_object_identity_token_attention", False)
    return contract


def initialize_from_checkpoint(
    backend: Any,
    decoder: torch.nn.Module,
    checkpoint: str | Path,
    *,
    config: dict,
    expert: str,
) -> dict:
    """Load complete model weights, without optimizer or selection state.

    A finished run cannot be resumed with a fresh cosine schedule. Transfer
    learning is an explicit new run, with step zero and a
    new validation baseline. Validate both parameter sets before mutation so
    a partial/incompatible checkpoint cannot silently initialize half a model.
    """
    checkpoint = Path(checkpoint)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state.get("expert") != expert:
        raise ValueError(
            f"initialization expert {state.get('expert')!r} != requested {expert!r}"
        )
    if state.get("codec") != config["codec"]:
        raise ValueError("initialization checkpoint has a different flow codec")
    if int(state.get("flow_resolution", -1)) != int(config["data"]["flow_resolution"]):
        raise ValueError("initialization checkpoint has a different flow resolution")
    recorded_decoder = state.get("decoder_config")
    if recorded_decoder is None:
        raise ValueError("initialization checkpoint lacks decoder_config provenance")
    target_decoder = config["model"]["decoder"]
    recorded_architecture = decoder_architecture_contract(recorded_decoder)
    target_architecture = decoder_architecture_contract(target_decoder)
    slot_activity_upgrade = (
        not bool(recorded_architecture.get(TASK_OBJECT_SLOT_FIELD, False))
        and bool(target_architecture.get(TASK_OBJECT_SLOT_FIELD, False))
        and {key: value for key, value in recorded_architecture.items()
             if key != TASK_OBJECT_SLOT_FIELD}
        == {key: value for key, value in target_architecture.items()
            if key != TASK_OBJECT_SLOT_FIELD}
    )
    identity_ignored_fields = {
        TASK_OBJECT_IDENTITY_FIELD,
        TASK_OBJECT_IDENTITY_SEMANTIC_FIELD,
        "task_object_identity_token_attention",
        "prompt_dim",
    }
    identity_upgrade = (
        bool(recorded_architecture.get(TASK_OBJECT_SLOT_FIELD, False))
        and bool(target_architecture.get(TASK_OBJECT_SLOT_FIELD, False))
        and not bool(recorded_architecture.get(TASK_OBJECT_IDENTITY_FIELD, False))
        and bool(target_architecture.get(TASK_OBJECT_IDENTITY_FIELD, False))
        and {key: value for key, value in recorded_architecture.items()
             if key not in identity_ignored_fields}
        == {key: value for key, value in target_architecture.items()
            if key not in identity_ignored_fields}
    )
    # Adding the zero-initialized text-to-slot binding gain to a checkpoint that
    # lacks it: every other architecture field must match exactly.
    binding_upgrade = (
        not bool(recorded_architecture.get("text_slot_binding", False))
        and bool(target_architecture.get("text_slot_binding", False))
        and {key: value for key, value in recorded_architecture.items()
             if key not in TEXT_SLOT_BINDING_FIELDS}
        == {key: value for key, value in target_architecture.items()
            if key not in TEXT_SLOT_BINDING_FIELDS}
    )
    # Explicit, weight-only pooling/fusion repair. Pooling changes semantics without
    # changing tensor shapes; fusion adds exactly two zero-initialized tensors.
    # Exact resume still checks the complete config in flow_dit_resume.py.
    repair_fields = {"slot_pooling_mode", "text_slot_fusion"}
    pooling_upgrade = (
        recorded_architecture["slot_pooling_mode"] == "bilinear"
        and target_architecture["slot_pooling_mode"] == "area"
    )
    fusion_upgrade = (
        recorded_architecture["text_slot_fusion"] is False
        and target_architecture["text_slot_fusion"] is True
        and bool(recorded_architecture.get("text_slot_binding", False))
    )
    repair_upgrade = (
        (pooling_upgrade or fusion_upgrade)
        and (pooling_upgrade or recorded_architecture["slot_pooling_mode"] == target_architecture["slot_pooling_mode"])
        and (fusion_upgrade or recorded_architecture["text_slot_fusion"] == target_architecture["text_slot_fusion"])
        and {k: v for k, v in recorded_architecture.items() if k not in repair_fields}
        == {k: v for k, v in target_architecture.items() if k not in repair_fields}
    )
    if (recorded_architecture != target_architecture
            and not slot_activity_upgrade and not identity_upgrade and not binding_upgrade
            and not repair_upgrade):
        raise ValueError("initialization checkpoint has a different decoder configuration")

    backend_state = torch.load(
        checkpoint / "backend_params.pt", map_location="cpu", weights_only=True
    )
    decoder_state = torch.load(
        checkpoint / "decoder.pt", map_location="cpu", weights_only=True
    )
    expected_backend = {
        name: parameter
        for name, parameter in backend.transformer.named_parameters()
        if parameter.requires_grad
    }
    allowed_decoder_missing = set()
    if repair_upgrade and fusion_upgrade:
        allowed_decoder_missing |= {"tsb_fusion.weight", "tsb_fusion.bias"}
        for key in allowed_decoder_missing:
            value = decoder.state_dict().get(key)
            if value is None or value.count_nonzero():
                raise ValueError("text fusion upgrade requires a zero-initialized residual")
    if slot_activity_upgrade:
        allowed_decoder_missing |= {
            key for key in decoder.state_dict()
            if key.startswith(TASK_OBJECT_SLOT_STATE_PREFIX)
        }
    if identity_upgrade:
        allowed_decoder_missing |= {
            key for key in decoder.state_dict()
            if key.startswith(
                (TASK_OBJECT_IDENTITY_STATE_PREFIX,)
                + TASK_OBJECT_IDENTITY_SEMANTIC_STATE_PREFIXES
            )
        }
    if binding_upgrade:
        allowed_decoder_missing |= {
            key for key in decoder.state_dict()
            if key.startswith(TEXT_SLOT_BINDING_STATE_PREFIXES)
        }
    for name, actual, expected in (
        ("backend", backend_state, expected_backend),
        ("decoder", decoder_state, decoder.state_dict()),
    ):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        unexpected_missing = (
            sorted(set(missing) - allowed_decoder_missing)
            if name == "decoder" else missing
        )
        if unexpected_missing or extra:
            raise ValueError(
                f"incomplete {name} initialization: "
                f"missing={unexpected_missing[:5]}, extra={extra[:5]}"
            )
        incompatible = [
            key for key in expected
            if key in actual and actual[key].shape != expected[key].shape
        ]
        if incompatible:
            raise ValueError(f"incompatible {name} parameter shapes: {incompatible[:5]}")
        nonfinite = [key for key, value in actual.items() if not bool(torch.isfinite(value).all())]
        if nonfinite:
            raise ValueError(f"nonfinite {name} initialization parameters: {nonfinite[:5]}")
    backend.load_state_dict(backend_state)
    decoder_upgrade = slot_activity_upgrade or identity_upgrade or binding_upgrade or (repair_upgrade and fusion_upgrade)
    incompatible = decoder.load_state_dict(decoder_state, strict=not decoder_upgrade)
    if decoder_upgrade and (
        set(incompatible.missing_keys) != allowed_decoder_missing
        or incompatible.unexpected_keys
    ):
        raise ValueError("task-object slot-head migration changed unexpected decoder keys")
    return {
        "kind": "weights_only",
        "checkpoint": str(checkpoint.resolve()),
        "source_step": int(state["step"]),
        "source_expert": state["expert"],
        "source_label_source": state.get("label_source"),
        "source_condition_scope": state.get("condition_scope"),
        "decoder_readout_change": {
            "from": {key: recorded_decoder.get(key, default) for key, default in DECODER_READOUT_DEFAULTS.items()},
            "to": {key: target_decoder.get(key, default) for key, default in DECODER_READOUT_DEFAULTS.items()},
        },
        "source_decoder_optimization_fields": {
            key: recorded_decoder.get(key) for key in DECODER_OPTIMIZATION_ONLY_FIELDS
            if key in recorded_decoder
        },
        "target_decoder_optimization_fields": {
            key: target_decoder.get(key) for key in DECODER_OPTIMIZATION_ONLY_FIELDS
            if key in target_decoder
        },
        "optimizer_reinitialized": True,
        "schedule_reinitialized": True,
        "decoder_pooling_change": (
            {"from": recorded_architecture["slot_pooling_mode"], "to": target_architecture["slot_pooling_mode"]}
            if repair_upgrade and pooling_upgrade else None
        ),
        "decoder_migration": "add_zero_initialized_text_slot_fusion_v1" if repair_upgrade and fusion_upgrade else (
            (
                "add_zero_initialized_semantic_task_object_identity_head_v2"
                if bool(target_architecture.get(TASK_OBJECT_IDENTITY_SEMANTIC_FIELD, False))
                else "add_zero_initialized_task_object_identity_head_v1"
            )
            if identity_upgrade else (
                "add_zero_initialized_task_object_slot_head_v1"
                if slot_activity_upgrade else "none"
            )
        ),
    }
