"""Evaluation-only identity threshold calibration on a separate scene split.

The training objective and the historical 0.5 readout never enter this module.
The caller must supply a manifest with independently audited scene membership.
"""
from __future__ import annotations

from collections import defaultdict
import math

import torch

from .flow_dit_dense_mask import build_task_object_identity_targets
from .flow_dit_selection import hard_object_selection


@torch.no_grad()
def identity_calibration_record(sample_id, domain, scene_id, prediction, reference,
                                config=None):
    """Compact one unbatched validation prediction to known identity labels."""
    probability = prediction["task_object_identity_probability"].detach().cpu().float()
    if probability.ndim != 1 or not torch.isfinite(probability).all():
        raise ValueError("identity calibration requires finite slot probabilities")
    if bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("identity probabilities must lie in [0,1]")
    keys = ("instance_masks", "supervision_instance_masks",
            "supervision_object_identity_authoritative")
    batch = {key: reference[key].detach().cpu().unsqueeze(0) for key in keys}
    h, w = batch["instance_masks"].shape[-2:]
    content = reference["content_valid"].detach().cpu().bool().reshape(1, 1, 1, h, w)
    dense = {"positive": torch.zeros_like(content), "content_valid": content}
    labels = build_task_object_identity_targets(batch, dense, config)
    if probability.shape != labels["target"][0].shape:
        raise ValueError("identity probabilities and proposals disagree")
    return dict(sample_id=str(sample_id), domain=str(domain), scene_id=str(scene_id),
                probability=probability.tolist(), target=labels["target"][0].tolist(),
                valid=labels["valid"][0].tolist())


def calibrate_identity_threshold(records, manifest, thresholds=None):
    """Maximize domain-balanced identity balanced accuracy on calibration only.

    The fixed grid is declared before evaluation. Each domain must contain
    positives and negatives. Ties favor recall, then the smaller threshold.
    Missing, duplicate, extra, or incorrectly assigned scene rows fail closed.
    """
    if manifest.get("schema") != "stage1_gate_calibration_v1":
        raise ValueError("unknown calibration manifest")
    if not manifest.get("audit", {}).get("scene_disjoint_verified"):
        raise ValueError("scene-disjoint calibration audit is required")
    allowed = {row["sample_id"]: row for row in manifest["rows"]}
    observed = [r["sample_id"] for r in records]
    if len(allowed) != len(manifest["rows"]) or len(set(observed)) != len(observed):
        raise ValueError("duplicate calibration sample")
    if set(observed) != set(allowed):
        raise ValueError("calibration observations must exactly match the frozen manifest")
    if not records:
        raise ValueError("empty calibration split")
    groups = defaultdict(list)
    for record in records:
        expected = allowed[record["sample_id"]]
        if any(record[k] != expected[k] for k in ("domain", "scene_id")):
            raise ValueError("calibration scene/domain membership mismatch")
        p, y, valid = (torch.as_tensor(record[k]) for k in ("probability", "target", "valid"))
        if p.ndim != 1 or p.shape != y.shape or p.shape != valid.shape:
            raise ValueError("calibration record shape mismatch")
        if not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("invalid calibration probability")
        if not ((y == 0) | (y == 1)).all() or not ((valid == 0) | (valid == 1)).all():
            raise ValueError("calibration target/known flags must be binary")
        groups[record["domain"]].append((p[valid.bool()], y[valid.bool()].bool()))
    groups = {name: (torch.cat([x[0] for x in rows]), torch.cat([x[1] for x in rows]))
              for name, rows in groups.items()}
    for name, (_, target) in groups.items():
        if not target.any() or target.all():
            raise ValueError(f"calibration domain needs both identity classes: {name}")
    grid = list(thresholds) if thresholds is not None else [i / 100 for i in range(1, 100)]
    if not grid or any(not math.isfinite(x) or not 0 < x < 1 for x in grid):
        raise ValueError("threshold grid must be finite and inside (0,1)")
    table = []
    for threshold in grid:
        domains = {}
        for name, (probability, target) in groups.items():
            selected = probability >= threshold
            recall = float(selected[target].float().mean())
            falsepositive = float(selected[~target].float().mean())
            domains[name] = dict(identity_recall=recall, identity_falsepositive=falsepositive,
                                 balanced_accuracy=(recall + 1 - falsepositive) / 2,
                                 positives=int(target.sum()), negatives=int((~target).sum()))
        table.append(dict(threshold=float(threshold), domains=domains,
                          balanced_accuracy=sum(x["balanced_accuracy"] for x in domains.values()) / len(domains),
                          recall=sum(x["identity_recall"] for x in domains.values()) / len(domains)))
    chosen = max(table, key=lambda r: (r["balanced_accuracy"], r["recall"], -r["threshold"]))
    return dict(schema="stage1_gate_threshold_v1", threshold=chosen["threshold"],
                selection_rule="domain_macro_identity_balanced_accuracy_then_recall_then_lower_threshold",
                scope="evaluation_readout_only", samples=len(records), selected=chosen,
                grid=table, manifest_sha256=manifest.get("manifest_sha256"))


@torch.no_grad()
def calibrated_identity_readout(prediction, reference, threshold):
    """Apply a chosen threshold to unbatched raw flow without scaling vectors."""
    flow = prediction.get("flow_ungated", prediction["flow"])
    probability = prediction["task_object_identity_probability"]
    selection = hard_object_selection(probability[None, :, None].expand(1, -1, flow.shape[0]),
                                      reference["instance_masks"].to(flow.device)[None], threshold)[0]
    return flow * selection.to(flow), selection
