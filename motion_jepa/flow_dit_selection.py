"""Object selection is a decision, not a displacement magnitude.

Hard readouts use input proposals and predicted probabilities only. Selection
heads learn from classification targets; vector losses cannot change their
confidence to conceal vector errors. Raw-flow supervision covers missed movers.
"""
import math
import torch


def validate_selection(mode, threshold, enabled=True):
    if mode not in {"soft", "hard"}:
        raise ValueError("task_object_gate_mode must be soft or hard")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("task_object_gate_threshold must lie in (0,1)")
    if not enabled and (mode != "soft" or threshold != .5):
        raise ValueError("object selection requires task_object_slot_activity")


def hard_object_selection(probability, masks, threshold=.5):
    """[B,K,T] x [B,K,H,W] -> binary [B,T,1,H,W], detached.

    Disjoint proposal probabilities are thresholded per OBJECT, not per pixel.
    The final support threshold keeps fractional mask edges from rescaling
    vectors. No forced top-1: a static instruction must allow zero selections.
    """
    validate_selection("hard", threshold)
    if probability.ndim != 3 or masks.ndim != 4 or probability.shape[:2] != masks.shape[:2]:
        raise ValueError("selection expects probabilities [B,K,T] and masks [B,K,H,W]")
    for value in (probability, masks):
        if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
            raise ValueError("selection inputs must be finite in [0,1]")
    if bool((masks.sum(1) > 1.00001).any()):
        raise ValueError("selection requires disjoint input proposals")
    selected = probability.detach() >= threshold
    support = torch.einsum("bkt,bkhw->bthw", selected.float(), masks.detach().float())
    return (support > .5).float().unsqueeze(2)


def selection_readout(activity_probability, masks, threshold=.5, *,
                      source="activity", identity_probability=None,
                      dense_probability=None, dense_threshold=.9):
    """Binary object decision, optionally rescued outside every input proposal.

    Identity is constant over query times and is never multiplied by activity.
    The optional learned dense task-motion mask can rescue SAM omissions, but
    cannot reopen a proposal rejected by the identity classifier. This is a
    separately calibrated readout; it does not establish that raw motion on
    the rescued pixels is correct. No labels enter this function.
    """
    if source not in ("activity", "identity"):
        raise ValueError("selection source must be activity or identity")
    probability = activity_probability
    if source == "identity":
        if identity_probability is None or identity_probability.shape != probability.shape[:2]:
            raise ValueError("identity selection requires identity probabilities [B,K]")
        probability = identity_probability[..., None].expand_as(probability)
    selected = hard_object_selection(probability, masks, threshold)
    if dense_probability is not None:
        validate_selection("hard", dense_threshold)
        if dense_probability.shape != selected.shape:
            raise ValueError("dense rescue requires probabilities [B,T,1,H,W]")
        if not torch.isfinite(dense_probability).all() or ((dense_probability < 0) | (dense_probability > 1)).any():
            raise ValueError("dense rescue probabilities must be finite in [0,1]")
        uncovered = (masks.sum(1) <= .5)[:, None, None]
        rescue = uncovered & (dense_probability.detach() >= dense_threshold)
        selected = (selected.bool() | rescue).float()
    return selected


def add_selection_guard(summary, limits):
    """Dimensionless max of explicit error/false-motion budgets, <=1 passes."""
    ratios = []
    for metric, budget in limits.items():
        budget = float(budget)
        value = float(summary.get(metric, float("nan")))
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("selection guard budgets must be finite and positive")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"selection guard missing/invalid metric: {metric}")
        ratios.append(value / budget)
    if not ratios:
        raise ValueError("selection guard requires at least one budget")
    summary["task_object_gated_motion/selection_guard"] = max(ratios)
