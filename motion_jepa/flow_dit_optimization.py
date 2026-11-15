"""Keep an auxiliary mask head from consuming the motion gradient clip budget."""
from __future__ import annotations

import math
import torch


DENSE_MASK_HEAD_PREFIXES = (
    "dense_motion_mask_fuse.", "dense_motion_mask_up.", "dense_motion_mask_head.",
)
TASK_OBJECT_HEAD_PREFIXES = (
    "task_object_slot_head.", "task_object_identity_head.",
    "task_identity_visual_project.", "task_identity_prompt_project.",
    "task_identity_attention.",
)
MASK_HEAD_PREFIXES = DENSE_MASK_HEAD_PREFIXES + TASK_OBJECT_HEAD_PREFIXES


def split_gradient_groups(
    trainable,
    named_decoder,
    *,
    separate_task_object: bool = False,
    allow_empty_motion: bool = False,
):
    named_decoder = list(named_decoder)
    dense_ids = {
        id(p) for name, p in named_decoder if name.startswith(DENSE_MASK_HEAD_PREFIXES)
    }
    task_ids = {
        id(p) for name, p in named_decoder if name.startswith(TASK_OBJECT_HEAD_PREFIXES)
    }
    mask_ids = dense_ids | task_ids
    motion, dense_mask, task_object = [], [], []
    seen = set()
    for p in trainable:
        if id(p) in seen:
            raise ValueError("duplicate trainable parameter in gradient clip groups")
        seen.add(id(p))
        if separate_task_object and id(p) in task_ids:
            task_object.append(p)
        elif id(p) in mask_ids:
            dense_mask.append(p)
        else:
            motion.append(p)
    if not dense_mask or (not motion and not allow_empty_motion):
        raise ValueError(
            "separate mask clipping requires mask parameters and, unless explicitly "
            "running heads-only, motion parameters"
        )
    if separate_task_object:
        if not task_object:
            raise ValueError("separate task-object clipping requires task-object parameters")
        return motion, dense_mask, task_object
    return motion, dense_mask


def clip_gradient_groups(motion, mask, max_norm: float, *, task_object=None):
    """Call AFTER Accelerate unscales gradients, once per optimizer step.

    This separates clip budgets, not the gradients entering shared features.
    Shared-feature mask gradients are controlled separately by the decoder.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    groups = [("motion", motion), ("mask", mask)]
    if task_object is not None:
        groups.append(("task_object", task_object))
    groups = [(name, group) for name, group in groups if group]
    if not groups:
        raise ValueError("at least one non-empty gradient clip group is required")
    norms = [
        torch.nn.utils.clip_grad_norm_(group, max_norm, error_if_nonfinite=True)
        for _, group in groups
    ]
    total = torch.stack([n.float() for n in norms]).square().sum().sqrt()
    diagnostics = {}
    for (name, _), norm in zip(groups, norms):
        diagnostics[f"grad_clip/{name}_norm"] = float(norm)
        diagnostics[f"grad_clip/{name}_multiplier"] = min(
            1.0, max_norm / (float(norm) + 1e-6)
        )
    return total, diagnostics
