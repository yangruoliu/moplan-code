"""Matched release/hold diagnostics on recipe actors and common valid pixels.

Records contain only the first-frame task object's pixels, allowing gathering
across validation shards without retaining dense predictions for every scene.
No caption parsing, outcome-derived actor selection, or training loss is used.
"""
from __future__ import annotations

import hashlib
import math

import torch


@torch.no_grad()
def release_hold_observation(sample_id, prediction, reference, horizon_seconds, recipe,
                             calibrated_flow=None):
    """Compress an unbatched prediction for one recipe-certified pair member."""
    if recipe["sample_id"] != sample_id or recipe["event"] not in ("release", "hold"):
        raise ValueError("invalid release/hold recipe membership")
    if not bool(reference["supervision_object_identity_authoritative"]):
        raise ValueError("release/hold diagnostic requires authoritative actor masks")
    masks = reference["supervision_instance_masks"].detach().cpu() > .5
    nonempty = masks.flatten(1).any(1)
    if int(nonempty.sum()) != 1:
        raise ValueError("release/hold diagnostic expects exactly one recipe actor")
    mask_ids = reference.get("supervision_instance_mask_ids")
    if mask_ids is not None and int(mask_ids.detach().cpu()[nonempty][0]) != int(recipe["actor_id"]):
        raise ValueError("supervision actor does not match the event recipe")
    mask = masks.any(0) & (reference["content_valid"].detach().cpu()[0] > .5)
    indices = mask.flatten().nonzero().flatten()
    if len(indices) < 4:
        raise ValueError("event actor has insufficient first-frame support")
    gt = reference["flow"].detach().cpu().float()
    t, _, h, w = gt.shape
    times = torch.as_tensor(horizon_seconds, dtype=torch.float32).cpu()
    if times.shape != (t,) or not torch.isfinite(times).all() or times[0] <= 0 or not (times[1:] > times[:-1]).all():
        raise ValueError("invalid event horizon times")
    scale = torch.tensor([w, h]).reshape(1, 2, 1)
    def compact(flow):
        if tuple(flow.shape) != tuple(gt.shape):
            raise ValueError("event flow geometry mismatch")
        return flow.detach().cpu().float().reshape(t, 2, -1)[..., indices] * scale
    fields = {"raw": compact(prediction.get("flow_ungated", prediction["flow"])),
              "selected": compact(prediction["flow_task_object_gated"]), "gt": compact(gt)}
    if calibrated_flow is not None:
        fields["calibrated_selected"] = compact(calibrated_flow)
    return dict(sample_id=sample_id, scene_id=recipe["scene_id"], actor_id=int(recipe["actor_id"]),
                pair_id=recipe["pair_id"], event=recipe["event"], indices=indices,
                horizon_seconds=times, fields=fields,
                frame_sha256=hashlib.sha256(reference["first_frame"].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                valid=(reference["flow_valid"].detach().cpu().reshape(t, -1)[:, indices] > .5))


def summarize_release_hold(observations, pair_manifest, epsilon_px=.5, min_pixels=4):
    """Compare paired captions on the same actor, times and valid t0 points.

    Ratios are epsilon stabilized and accompanied by absolute and GT-normalized
    deltas. Both-zero predictions thus cannot masquerade as discriminative.
    Summary averages time within pair, then pairs equally; full rows retain
    horizon detail and GT comparisons. All expected pairs must be present.
    """
    if not math.isfinite(epsilon_px) or epsilon_px <= 0:
        raise ValueError("event ratio epsilon must be positive")
    if pair_manifest.get("schema") != "stage1_release_hold_diagnostic_v1":
        raise ValueError("unknown event diagnostic manifest")
    lookup = {r["sample_id"]: r for r in observations}
    if len(lookup) != len(observations):
        raise ValueError("duplicate event observation")
    expected = {p[key] for p in pair_manifest["pairs"] for key in ("release_id", "hold_id")}
    if set(lookup) != expected:
        raise ValueError("release/hold observations do not exactly match the manifest")
    pairs = []
    for pair in pair_manifest["pairs"]:
        release, hold = lookup[pair["release_id"]], lookup[pair["hold_id"]]
        for record, event in ((release, "release"), (hold, "hold")):
            if any(record[k] != pair[k] for k in ("scene_id", "actor_id", "pair_id")) or record["event"] != event:
                raise ValueError("event pairing disagrees with recipe")
        for key in ("indices", "horizon_seconds"):
            if not torch.equal(release[key], hold[key]):
                raise ValueError("event pairs must share task actor pixels and times")
        if release["frame_sha256"] != hold["frame_sha256"]:
            raise ValueError("event pairs must share the same Stage1 input frame")
        if set(release["fields"]) != set(hold["fields"]):
            raise ValueError("event pair readouts differ")
        common = release["valid"] & hold["valid"]
        eligible = common.sum(-1) >= min_pixels
        if not eligible.any():
            raise ValueError("event pair has no common trustworthy pixels")
        vectors = {}
        for field in release["fields"]:
            vectors[field] = []
            for record in (release, hold):
                value = record["fields"][field]
                if not torch.isfinite(value.masked_select(common[:, None].expand_as(value))).all():
                    raise ValueError("nonfinite event displacement on valid support")
                vector = torch.where(common[:, None], value, 0).sum(-1) / common.sum(-1).clamp_min(1)[:, None]
                vectors[field].append(vector)
        gt_r, gt_h = (v.norm(dim=-1) for v in vectors["gt"])
        if not ((gt_r - gt_h)[eligible] > 0).any():
            raise ValueError("release/hold labels have no observable displacement contrast")
        fields = {}
        for field, (r, h) in vectors.items():
            rm, hm = r.norm(dim=-1), h.norm(dim=-1)
            values = dict(release_magnitude_px=rm, hold_magnitude_px=hm,
                          release_hold_ratio=rm / (hm + epsilon_px),
                          release_minus_hold_px=rm - hm,
                          contrast=(rm - hm) / (rm + hm + epsilon_px),
                          gt_normalized_delta=(rm - hm) / (gt_r - gt_h).clamp_min(epsilon_px))
            fields[field] = {name: float(value[eligible].mean()) for name, value in values.items()}
            fields[field]["by_horizon"] = [dict(time=float(release["horizon_seconds"][i]),
                known_pixels=int(common[i].sum()), **{name: float(value[i]) for name, value in values.items()})
                for i in eligible.nonzero().flatten().tolist()]
        pairs.append({**pair, "fields": fields})
    if not pairs:
        raise ValueError("empty release/hold diagnostic")
    names = set(pairs[0]["fields"])
    if any(set(p["fields"]) != names for p in pairs):
        raise ValueError("incomplete readout fields across pairs")
    metrics = {}
    for field in sorted(names):
        metrics[field] = {name: sum(p["fields"][field][name] for p in pairs) / len(pairs)
                          for name in pairs[0]["fields"][field] if name != "by_horizon"}
    return dict(schema="stage1_release_hold_results_v1", pairs=len(pairs), summary=metrics,
                epsilon_px=epsilon_px, reduction="equal_pair_mean_of_common_valid_horizon_means",
                ratio_note="release/(hold+epsilon); assess absolute and GT-normalized delta too",
                rows=pairs)
