"""Kinematic losses on object displacements read from a generated (or predicted-clean) video.

`pred` and `truth` are displacements from the first frame, [B, K, T, 2] in work-frame pixels (index 0 is the
conditioning frame: zero).  `truth` comes from the real continuation, never from a Stage-1 hint.  `valid` [B, K, T]
is the annotation's validity (tracked / known); it is never a predicted gate.

  direction     over GT-moving segments: 1 - cos(v_pred, v_true), with |v_pred| clamped from below, so a prediction that
                does not move costs exactly 1 and still has a gradient that pushes it along the true direction
                (reverse = 2, correct = 0).  Undefined on GT-static segments, which are left out.
  displacement  Huber on (d_pred - d_true) / disp_scale over every valid cell, static objects included with the SAME
                scale, so "stop everything" is not a cheap answer and invented motion on a static object is paid for.

Velocities use the physical seconds of the frames.  Nothing here asks for smooth acceleration or constant speed
(collisions, rebounds and springs are neither).  Both terms are balanced over objects: every object with at least one
scorable cell weighs the same, whatever its size.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _object_balanced(values: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """values, weights [B, K, N] -> (per-sample mean over objects of the per-object weighted mean [B], objects counted [B])."""
    cells = weights.sum(-1)                                             # [B, K]
    per_object = (values * weights).sum(-1) / cells.clamp_min(1.0)
    has = (cells > 0).to(values.dtype)
    return (per_object * has).sum(-1) / has.sum(-1).clamp_min(1.0), has.sum(-1)


def kinematic_losses(pred: torch.Tensor, truth: torch.Tensor, seconds: torch.Tensor, valid: torch.Tensor, *,
                     disp_scale: float = 32.0, epsilon_speed: float = 4.0) -> dict:
    """Per-sample {"direction" [B], "displacement" [B], "moving_objects" [B], "objects" [B]}.

    seconds [B, T] strictly increasing; disp_scale in px; epsilon_speed in px/s (a GT segment slower than this is not
    "moving", and it is the floor of the predicted speed in the cosine)."""
    if disp_scale <= 0 or epsilon_speed <= 0:
        raise ValueError("scales must be positive")
    dt = seconds[:, 1:] - seconds[:, :-1]                               # [B, T-1]
    if bool((dt <= 0).any()):
        raise ValueError("frame times must strictly increase")
    valid = valid.bool()
    pair = valid[:, :, 1:] & valid[:, :, :-1]
    v_pred = (pred[:, :, 1:] - pred[:, :, :-1]) / dt[:, None, :, None]
    v_true = (truth[:, :, 1:] - truth[:, :, :-1]) / dt[:, None, :, None]
    n_pred, n_true = v_pred.norm(dim=-1), v_true.norm(dim=-1)
    moving = pair & (n_true >= epsilon_speed)
    cosine = (v_pred * v_true).sum(-1) / (n_pred.clamp_min(epsilon_speed) * n_true.clamp_min(epsilon_speed))
    direction, moving_objects = _object_balanced(1.0 - cosine.clamp(-1.0, 1.0), moving.to(pred.dtype))
    anchored = valid[:, :, 1:] & valid[:, :, :1]
    huber = F.smooth_l1_loss(pred[:, :, 1:] / disp_scale, truth[:, :, 1:] / disp_scale, reduction="none").sum(-1)
    displacement, objects = _object_balanced(huber, anchored.to(pred.dtype))
    return {"direction": direction, "displacement": displacement, "moving_objects": moving_objects, "objects": objects}
