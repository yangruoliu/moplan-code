"""Opt-in five-term Stage1 objective; legacy objectives are never evaluated.

Direction is the distance to a forward, angle-tolerant cone, evaluated on
consecutive displacements of the SAME first-frame material points. It has a
useful gradient at zero and exact reversal without a second margin loss.
All regression is upstream of object selection. Selection has only BCE.
"""
from __future__ import annotations

import math

import torch

from .flow_dit_coarse import _mean
from .flow_dit_dense_mask import build_task_object_identity_targets, task_object_slot_loss


PROFILE = "minimal5_direction_v1"
KEYS = ("displacement", "direction", "text", "selection", "pixel")
DEFAULT_WEIGHTS = dict(displacement=1., direction=4., text=1., selection=1., pixel=.1)
SETTINGS = {
    "profile", "weights", "supervision_mode", "flow_matching", "aux_sigma_weight",
    "aux_warmup_steps", "direction_tolerance_degrees", "direction_min_progress_px",
    "direction_min_target_px", "direction_scale_floor_px", "robust_epsilon_px",
    "displacement_scale_floor_px", "displacement_scale_mode", "displacement_scale_px",
    "pixel_scale_px", "min_pixels",
    "task_object_identity_positive_fraction", "task_object_identity_negative_fraction",
    "task_object_identity_min_pixels", "selection_prior_bce_fraction",
    # Retained solely for the existing decoder/validator label contract.
    "dense_motion_mask_target_mode",
    "direction_objective",
    "counterfactual_text_support",
}


def freeze_minimal5_unused_heads(decoder):
    """Exclude legacy diagnostic-only branches from the optimizer and DDP.

    Retain their state_dict entries for strict checkpoint compatibility. The
    image stem only serves those branches and the detached semantic selector
    in this profile; do not freeze it if selector features are trainable.
    """
    prefixes = ("mask_fuse.", "mask_up.", "mask_head.",
                "visibility_fuse.", "visibility_up.", "visibility_head.",
                "dense_motion_mask_fuse.", "dense_motion_mask_up.", "dense_motion_mask_head.",
                "task_object_slot_head.")
    if decoder.task_object_heads_detach_shared:
        prefixes += ("image_stem.",)
    frozen = []
    for name, parameter in decoder.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen


def validate_minimal_loss(config):
    if config.get("profile") != PROFILE:
        raise ValueError(f"minimal loss requires profile={PROFILE}")
    extra = set(config) - SETTINGS
    if extra:
        raise ValueError(f"minimal5 rejects legacy/unknown loss settings: {sorted(extra)}")
    weights = config.get("weights", {})
    if set(weights) != set(KEYS):
        raise ValueError("minimal5 requires exactly five explicit weights")
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in weights.values()):
        raise ValueError("minimal5 weights must be finite and positive")
    if config.get("supervision_mode") != "valid_partition_v1":
        raise ValueError("minimal5 requires valid_partition_v1")
    if float(config.get("flow_matching", 0.)) != 0:
        raise ValueError("minimal5 requires flow_matching=0")
    angle = float(config.get("direction_tolerance_degrees", 20.))
    if not math.isfinite(angle) or not 0 < angle < 90:
        raise ValueError("direction_tolerance_degrees must be finite and in (0,90)")
    if config.get("direction_objective", "cone_distance") not in ("cone_distance", "angular_hinge"):
        raise ValueError("minimal5 direction_objective must be cone_distance or angular_hinge")
    if config.get("displacement_scale_mode", "target_magnitude") not in ("target_magnitude", "fixed"):
        raise ValueError("displacement_scale_mode must be target_magnitude or fixed")
    if config.get("counterfactual_text_support", "input_regions") not in (
            "input_regions", "semantic_actor_balanced"):
        raise ValueError("counterfactual_text_support must be input_regions or semantic_actor_balanced")
    defaults = dict(direction_min_progress_px=.5, direction_min_target_px=1.,
                    direction_scale_floor_px=4., robust_epsilon_px=.1,
                    displacement_scale_floor_px=4., displacement_scale_px=16.,
                    pixel_scale_px=16., min_pixels=4.)
    for key, default in defaults.items():
        value = float(config.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if float(config.get("direction_min_progress_px", .5)) > float(config.get("direction_min_target_px", 1.)):
        raise ValueError("direction minimum progress must not exceed the trusted target cutoff")
    fraction = float(config.get("selection_prior_bce_fraction", .25))
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("selection_prior_bce_fraction must be in [0,1]")
    return {key: float(weights[key]) for key in KEYS}


def validate_minimal_config(config):
    """Keep the opt-in profile explicit and reject silent extra objectives."""
    loss = config.get("loss", {})
    profile = loss.get("profile", "legacy")
    if profile == "legacy":
        return
    if profile != PROFILE:
        raise ValueError(f"unknown loss profile {profile!r}")
    validate_minimal_loss(loss)
    decoder, train = config["model"]["decoder"], config["train"]
    if train.get("raw_motion_guard", False):
        if train.get("early_stop_patience", 0) != 0 or not train.get("selection_guard_record_only", False):
            raise ValueError("raw motion guard requires selector record-only and no patience stop")
        if train.get("selection_metric") != "balanced_minimal5_raw_direction/direction30_failure_fraction":
            raise ValueError("raw motion guard requires raw-only checkpoint selection")
    required = {"type": "slots", "temporal_slots": True, "oracle_instance_masks": True,
                "dense_motion_mask": True, "task_object_slot_activity": True,
                "task_object_identity": True, "task_object_identity_semantic": True,
                "task_object_identity_token_attention": True,
                "task_object_gate_mode": "hard", "task_object_gate_source": "identity",
                "text_slot_binding": True, "text_slot_fusion": True}
    for key, expected in required.items():
        if decoder.get(key) != expected:
            raise ValueError(f"minimal5 requires decoder.{key}={expected!r}")
    if decoder.get("task_object_dense_rescue", False):
        raise ValueError("minimal5 cannot read an unsupervised dense rescue head")
    if train.get("objective") != "direct" or train.get("refine_from"):
        raise ValueError("minimal5 requires a single direct predictor")
    if train.get("mask_heads_only", False) or train.get("activity_heads_only", False):
        raise ValueError("minimal5 must train the motion predictor")
    if train.get("separate_task_object_gradient_clip", False):
        raise ValueError("minimal5 freezes dense auxiliary heads; use the two-group separate_mask_gradient_clip")
    if float(train.get("caption_dropout", 0.)) != 0:
        raise ValueError("minimal5 paired text supervision requires caption_dropout=0")
    if not config.get("paths", {}).get("event_pairs_manifest"):
        raise ValueError("minimal5 text supervision requires an event-pair manifest")


def direction_cone_cost(predicted, target, config):
    """Per-vector cost, [...,2] pixel displacements -> [...].

    Let a=p.u, b=|p-a.u|, u=g/|g|. The feasible set is
    C={p: a>=delta, b<=a*tan(theta)}. Compute Euclidean distance to C exactly:
    the closest boundary is either its bottom segment or either oblique ray.
    Robustify that distance and scale by max(|g|, floor), never by |p|.
    Thus zero/reversed predictions cannot hide behind normalization; all
    sufficiently forward vectors inside the cone have zero direction cost,
    regardless of speed. Speed remains the displacement objective's job.
    """
    norm = torch.linalg.vector_norm(target, dim=-1)
    unit = target / norm.clamp_min(1e-8)[..., None]
    parallel = (predicted * unit).sum(-1)
    perpendicular = torch.linalg.vector_norm(predicted - parallel[..., None] * unit, dim=-1)
    slope = math.tan(math.radians(float(config.get("direction_tolerance_degrees", 20.))))
    delta = float(config.get("direction_min_progress_px", .5))
    # Vertical base of the truncated cone.
    base_distance_sq = (parallel - delta).square() + torch.relu(perpendicular - delta * slope).square()
    # Orthogonal projection onto the cone's side, clamped to the base vertex.
    side_parallel = ((parallel + slope * perpendicular) / (1 + slope**2)).clamp_min(delta)
    side_distance_sq = (parallel - side_parallel).square() + (perpendicular - slope * side_parallel).square()
    inside = (parallel >= delta) & (perpendicular <= parallel * slope)
    distance_sq = torch.where(inside, torch.zeros_like(parallel), torch.minimum(base_distance_sq, side_distance_sq))
    eps = float(config.get("robust_epsilon_px", .1))
    return ((distance_sq + eps**2).sqrt() - eps) / norm.clamp_min(float(config.get("direction_scale_floor_px", 4.)))


def direction_cost(predicted, target, config):
    """Opt-in angle cost that cannot reward shrinking a forward wrong vector.

    c = (p.u) / sqrt(|p|^2 + epsilon^2); L = relu(cos(theta)-c).
    epsilon = min_progress*tan(theta), so an aligned vector of exactly
    min_progress reaches zero loss, with no pressure to inflate it further.
    This is an actual smooth normalization (no stop-gradient denominator).
    At zero or exact reversal it still has a forward gradient. GT magnitude
    does not dilute the supervision of large motions. Static/unknown target
    segments are excluded by _segments, as for the historical cone cost.
    """
    if config.get("direction_objective", "cone_distance") == "cone_distance":
        return direction_cone_cost(predicted, target, config)
    theta = math.radians(float(config.get("direction_tolerance_degrees", 20.)))
    epsilon = float(config.get("direction_min_progress_px", .5)) * math.tan(theta)
    unit = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1e-8)
    cosine = (predicted*unit).sum(-1) / (predicted.square().sum(-1)+epsilon**2).sqrt()
    return torch.relu(math.cos(theta)-cosine)


def _binary(value, shape, name, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    value = value.detach().to(device)
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{name} must be finite and in [0,1]")
    return value > .5


def _inputs(prediction, batch, sample_weight):
    flow, target = prediction["flow"], batch["flow"]
    if flow.ndim != 5 or flow.shape[2] != 2 or flow.shape != target.shape:
        raise ValueError("minimal5 flow must be matching [B,T,2,H,W]")
    b, t, _, h, w = flow.shape
    known = _binary(batch["flow_valid"], (b,t,1,h,w), "flow_valid", flow.device)
    content = _binary(batch["content_valid"], (b,1,h,w), "content_valid", flow.device)
    known = known & content[:, None]
    labels = batch["supervision_instance_masks"]
    if labels.ndim != 4 or labels.shape[0] != b or tuple(labels.shape[-2:]) != (h,w):
        raise ValueError("supervision_instance_masks must have shape [B,K,H,W]")
    masks = _binary(labels, labels.shape, "supervision_instance_masks", flow.device) & content
    scale = flow.new_tensor([w,h], dtype=torch.float32).view(1,1,2,1,1)
    p = torch.where(known, flow.float(), 0) * scale
    g = torch.where(known, target.detach().float(), 0) * scale
    if not bool(torch.isfinite(p).all() & torch.isfinite(g).all()):
        raise ValueError("minimal5 flow is nonfinite on trusted support")
    times = torch.as_tensor(batch["horizon_seconds"], device=flow.device, dtype=torch.float32)
    if times.ndim == 1:
        times = times[None].expand(b,-1)
    if times.shape != (b,t) or not bool(torch.isfinite(times).all()):
        raise ValueError("horizon_seconds must be finite [B,T] or [T]")
    if bool((torch.diff(times, dim=1, prepend=torch.zeros_like(times[:,:1])) <= 0).any()):
        raise ValueError("horizon_seconds must be positive and increasing")
    if sample_weight is not None:
        if sample_weight.shape != (b,) or not bool(torch.isfinite(sample_weight).all()) or bool((sample_weight < 0).any()):
            raise ValueError("sample_weight must be finite nonnegative [B]")
    return p, g, masks.float(), known, content, times


def _pool(field, support):
    area = support.sum((-2,-1))
    value = torch.einsum("bkrthw,btchw->bkrtc", support, field) / area.clamp_min(1)[...,None]
    return value, area


def _direction_regions(masks):
    """Whole object plus four fixed t0 quadrants; avoid cancelling rotation.

    Every region averages vectors before the loss, rather than chasing noisy
    pixel angles. Regions average within an object, so area and region count
    cannot give a large object extra budget. No predicted masks are involved.
    """
    h,w = masks.shape[-2:]
    y,x = torch.meshgrid(torch.arange(h,device=masks.device), torch.arange(w,device=masks.device), indexing="ij")
    area = masks.sum((-2,-1)).clamp_min(1)
    cx = (masks*x).sum((-2,-1)) / area
    cy = (masks*y).sum((-2,-1)) / area
    right, lower = x >= cx[...,None,None], y >= cy[...,None,None]
    quadrants = torch.stack((~right & ~lower, right & ~lower, ~right & lower, right & lower),dim=2)
    return torch.cat((masks[:,:,None], masks[:,:,None]*quadrants),dim=2)


def _field_mean(error, masks, known, config, sample_weight=None):
    # Background is one fixed region, not millions of individual votes.
    background = (~masks.bool().any(1,keepdim=True)).float()
    regions = torch.cat((masks,background),dim=1)
    support = regions[:,:,None] * known[:,None,:,0]
    area = support.sum((-2,-1))
    cells = (support*error[:,None]).sum((-2,-1))/area.clamp_min(1)
    return _mean(cells, area >= float(config.get("min_pixels",4)), sample_weight)


def _segments(p, g, masks, known, config):
    common = known & torch.cat((torch.ones_like(known[:,:1]),known[:,:-1]),dim=1)
    support = _direction_regions(masks)[:,:,:,None] * common[:,None,None,:,0]
    dp = torch.diff(p,dim=1,prepend=torch.zeros_like(p[:,:1]))
    dg = torch.diff(g,dim=1,prepend=torch.zeros_like(g[:,:1]))
    ps,area = _pool(dp,support)
    gs,_ = _pool(dg,support)
    moving = ((area>=float(config.get("min_pixels",4)))
              & (torch.linalg.vector_norm(gs,dim=-1)>float(config.get("direction_min_target_px",1.))))
    return ps, gs, moving


@torch.no_grad()
def segment_direction_statistics(predicted, batch, config):
    """Additive case-macro validation statistics, independent of the selector.

    Empty/static cases carry an explicit count, never a fabricated 100% score.
    Call once for raw and once for selected flow, using identical GT support.
    """
    with torch.autocast(device_type=predicted.device.type, enabled=False):
        p,g,masks,known,_,_ = _inputs({"flow":predicted},batch,None)
        ps,gs,moving = _segments(p,g,masks,known,config)
        pn,gn = torch.linalg.vector_norm(ps,dim=-1),torch.linalg.vector_norm(gs,dim=-1)
        cosine = (ps*gs).sum(-1)/(pn*gn).clamp_min(1e-8)
        object_times = moving.any(2)
        objects = object_times.any(-1)
        eligible = objects.any(-1)
        label_area = (masks[:,:,None]*known[:,None,:,0]).sum((-2,-1))
        result = {"eligible_cases":float(eligible.sum()),
                  "known_cases":float((label_area>=float(config.get("min_pixels",4))).flatten(1).any(-1).sum())}
        for angle in (15,30):
            correct = (pn>=float(config.get("direction_min_progress_px",.5))) & (cosine>=math.cos(math.radians(angle)))
            cells = (correct*moving).sum(2)/moving.sum(2).clamp_min(1)
            per_object = (cells*object_times).sum(-1)/object_times.sum(-1).clamp_min(1)
            per_case = (per_object*objects).sum(-1)/objects.sum(-1).clamp_min(1)
            result[f"direction{angle}_correct_case_sum"] = float((per_case*eligible).sum())
        return result


def semantic_text_group_values(error, actor, common, config):
    """Per-pair actor/nonactor means on common trustworthy support.

    ``error`` is the existing pixel-vector difference cost, already divided
    by pixel_scale_px. ``actor`` is the UNION of both captions' semantic task
    masks [P,H,W], never a prediction or a movement-threshold-derived mask.
    Within each group, average pixels then eligible horizons. An empty group
    returns zero and has no gradient; its fixed half-budget is not transferred
    to the other group. Callers reject active counterfactuals lacking actor
    support rather than silently teaching only their surroundings.
    """
    if error.ndim != 4 or actor.shape != error.shape[:1] + error.shape[-2:]:
        raise ValueError("semantic text support geometry mismatch")
    if common.shape != error.shape[:2] + (1,) + error.shape[-2:]:
        raise ValueError("semantic text common-valid geometry mismatch")
    values, present = [], []
    for region in (actor.bool(), ~actor.bool()):
        support = common[:, :, 0].bool() & region[:, None]
        area = support.sum((-1, -2))
        valid = area >= float(config.get("min_pixels", 4))
        cell = torch.where(support, error, 0).sum((-1, -2)) / area.clamp_min(1)
        values.append((cell * valid).sum(-1) / valid.sum(-1).clamp_min(1))
        present.append(valid.any(-1))
    return values[0], values[1], present[0], present[1]


def _legacy_text_pair_values(error, masks, common, config):
    """Expose unchanged input-region means for invariant members of a batch."""
    background = (~masks.bool().any(1, keepdim=True)).float()
    regions = torch.cat((masks, background), dim=1)
    support = regions[:, :, None] * common[:, None, :, 0]
    area = support.sum((-2, -1))
    cells = (support * error[:, None]).sum((-2, -1)) / area.clamp_min(1)
    valid = area >= float(config.get("min_pixels", 4))
    per_region = (cells * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    regions_present = valid.any(-1)
    return ((per_region * regions_present).sum(-1) / regions_present.sum(-1).clamp_min(1),
            regions_present.any(-1))


def _text_loss(p, g, batch, known, times, config, sample_weight):
    indices = batch.get("event_pair_indices")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 2 or indices.shape[1] != 2 or indices.dtype != torch.long:
        raise ValueError("minimal5 text supervision requires event_pair_indices [P,2] int64")
    indices = indices.to(p.device)
    zero = p.reshape(-1)[:0].sum()
    if indices.numel() == 0:
        return zero, zero.detach()
    if bool((indices < 0).any() | (indices >= len(p)).any()):
        raise ValueError("event pair index out of range")
    left,right = indices.unbind(1)
    if not torch.equal(times[left], times[right]):
        raise ValueError("paired physical times disagree")
    # Dataset also verifies identical RGB/latent observations before collation.
    proposals = batch["instance_masks"]
    if not torch.equal(proposals[left], proposals[right]):
        raise ValueError("paired first-frame masks disagree")
    masks = _binary(proposals[left], proposals[left].shape, "paired input masks",p.device).float()
    common = known[left] & known[right]
    delta_g = torch.where(common, g[left]-g[right], 0)
    relations = batch.get("event_pair_relations", [])
    if len(relations) != len(left) or any(x not in ("invariant","counterfactual") for x in relations):
        raise ValueError("event relations do not match pairs")
    invariant = torch.tensor([x == "invariant" for x in relations],device=p.device)
    if bool((delta_g[invariant].abs() > 1e-4).any()):
        raise ValueError("invariance labels disagree")
    observed = (delta_g.abs() > 1e-4).flatten(1).any(-1)
    if bool((~invariant & ~observed).any()):
        raise ValueError("counterfactual has no observable target difference")
    delta_p = torch.where(common,p[left]-p[right],0)
    eps = float(config.get("robust_epsilon_px",.1))
    error = ((delta_p-delta_g).square().sum(2)+eps**2).sqrt()-eps
    weight = None if sample_weight is None else (sample_weight[left]+sample_weight[right])*.5
    error = error / float(config.get("pixel_scale_px",16.))
    mode = config.get("counterfactual_text_support", "input_regions")
    if mode not in ("input_regions", "semantic_actor_balanced"):
        raise ValueError("unknown counterfactual_text_support")
    if mode == "input_regions" or bool(invariant.all()):
        # Preserve the historical path exactly for default configs and all
        # invariant batches; this opt-in changes counterfactual support only.
        loss = _field_mean(error, masks, common, config, weight)
    elif mode == "semantic_actor_balanced":
        labels = batch.get("supervision_instance_masks")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 4 or labels.shape[0] != len(p):
            raise ValueError("semantic counterfactual text loss requires task actor masks")
        labels = _binary(labels, (len(p), labels.shape[1], *p.shape[-2:]),
                         "semantic counterfactual actor masks", p.device)
        authoritative = _binary(batch.get("supervision_object_identity_authoritative"),
                                (len(p),), "semantic actor authority", p.device)
        counterfactual = ~invariant
        active = counterfactual if weight is None else counterfactual & (weight > 0)
        if bool((active & ~(authoritative[left] & authoritative[right])).any()):
            raise ValueError("counterfactual text loss requires authoritative actors on both sides")
        actor = labels[left].any(1) | labels[right].any(1)
        actor_value, nonactor_value, actor_present, _ = semantic_text_group_values(
            error, actor, common, config)
        if bool((active & ~actor_present).any()):
            raise ValueError("counterfactual semantic actor has no common valid support")
        balanced = .5 * actor_value + .5 * nonactor_value
        legacy, legacy_present = _legacy_text_pair_values(error, masks, common, config)
        values = torch.where(counterfactual, balanced, legacy)
        present = torch.where(counterfactual, actor_present, legacy_present).to(values.dtype)
        pair_weight = present if weight is None else present * weight
        loss = (values * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8)
    else:
        raise ValueError("unknown counterfactual_text_support")
    return loss, p.new_tensor(len(left))


def pooled_displacement_loss(predicted, target, area, config, sample_weight=None):
    """Object/time-balanced displacement loss on pooled [B,K,T,2] vectors.

    Fixed scaling gives release and hold exactly the same cost per pixel.
    The legacy target-dependent scale remains the default for reproducibility.
    This function is also used to replay native validation exports offline.
    """
    eps = float(config.get("robust_epsilon_px", .1))
    distance = ((predicted-target).square().sum(-1)+eps**2).sqrt()-eps
    mode = config.get("displacement_scale_mode", "target_magnitude")
    if mode == "fixed":
        scale = float(config.get("displacement_scale_px", 16.))
    elif mode == "target_magnitude":
        scale = torch.linalg.vector_norm(target, dim=-1).clamp_min(
            float(config.get("displacement_scale_floor_px", 4.)))
    else:
        raise ValueError("unknown displacement_scale_mode")
    return _mean(distance/scale, area >= float(config.get("min_pixels", 4.)), sample_weight)


def minimal5_losses(prediction, batch, config, sample_weight=None):
    weights = validate_minimal_loss(config)
    # FP32 reductions are necessary for small objects and near-zero vectors.
    with torch.autocast(device_type=prediction["flow"].device.type, enabled=False):
        p,g,masks,known,content,times = _inputs(prediction,batch,sample_weight)
        support = masks[:,:,None,None] * known[:,None,None,:,0]
        pm,area = _pool(p,support)
        gm,_ = _pool(g,support)
        eps = float(config.get("robust_epsilon_px",.1))
        displacement = pooled_displacement_loss(
            pm.squeeze(2), gm.squeeze(2), area.squeeze(2), config, sample_weight)

        ps,gs,moving = _segments(p,g,masks,known,config)
        penalty = direction_cost(ps,gs,config)
        # Average regions first, then horizons, objects, samples.
        cells = (penalty*moving).sum(2)/moving.sum(2).clamp_min(1)
        direction = _mean(cells,moving.any(2),sample_weight)
        pixel_error = ((p-g).square().sum(2)+eps**2).sqrt()-eps
        pixel = _field_mean(pixel_error/float(config.get("pixel_scale_px",16.)), masks,known,config,sample_weight)
        text_loss,pairs = _text_loss(p,g,batch,known,times,config,sample_weight)

        # Identity labels are semantic, time independent, and include a named
        # stationary object. No movement threshold enters this BCE target.
        targets = build_task_object_identity_targets(batch,{
            "positive": torch.zeros_like(known),
            "content_valid": content[:,None].expand_as(known)},config)
        logits = prediction.get("task_object_identity_logits")
        if not isinstance(logits,torch.Tensor) or logits.shape != targets["target"].shape:
            raise ValueError("minimal5 requires task_object_identity_logits [B,K]")
        selection = task_object_slot_loss(logits[...,None], {k:v[...,None] for k,v in targets.items()},
                                         sample_weight,float(config.get("selection_prior_bce_fraction",.25)))
        terms = dict(displacement=displacement,direction=direction,text=text_loss,selection=selection,pixel=pixel)
        result = {f"minimal5_{k}":v for k,v in terms.items()}
        result.update({f"minimal5_weighted_{k}":weights[k]*v for k,v in terms.items()})
        result["aux_total"] = sum(weights[k]*v for k,v in terms.items())
        # Compatibility for legacy trainer logging only; these are NOT losses.
        zero = p.reshape(-1)[:0].sum()
        result.update({k:zero for k in ("gate","visibility","object_mask","slot_mask","position")})
        result["flow"] = pixel
        with torch.no_grad():
            pn,gn = torch.linalg.vector_norm(ps,dim=-1),torch.linalg.vector_norm(gs,dim=-1)
            cosine = (ps*gs).sum(-1)/(pn*gn).clamp_min(1e-8)
            for angle in (15,30):
                success = (pn>=float(config.get("direction_min_progress_px",.5))) & (cosine>=math.cos(math.radians(angle)))
                rate = (success*moving).sum(2)/moving.sum(2).clamp_min(1)
                result[f"minimal5_segment_direction{angle}_accuracy"] = _mean(rate,moving.any(2),sample_weight)
            result["minimal5_direction_objects"] = moving.any(2).any(-1).sum()
            result["minimal5_direction_regions"] = moving.sum()
            result["minimal5_text_pairs"] = pairs
            result["minimal5_selection_known"] = targets["valid"].sum()
        return result


def minimal5_gradient_diagnostics(terms, prediction):
    """Diagnostic only: weighted loss gradients at the actual loss input flow.

    Run on log steps before backward. This is NOT a learned balancing rule or
    an additional loss; the norm share is not a fraction of the final update
    (gradients can align or cancel, and residual/backbone paths differ). The
    Accelerate AMP wrapper casts returned slot_params and flow independently:
    the returned FP32 slot_params is then a sibling, not an ancestor, of flow.
    Differentiating with respect to that sibling silently reports fake zeros.
    Measure at flow itself; the trainer separately logs true parameter grads.
    """
    flow = prediction.get("flow")
    if not isinstance(flow, torch.Tensor) or not flow.requires_grad:
        raise ValueError("minimal5 gradient diagnostics require differentiable flow")
    gradients, result = {}, {}
    for key in ("displacement", "direction", "text", "pixel"):
        value = terms["minimal5_weighted_"+key]
        gradient, = torch.autograd.grad(value,flow,retain_graph=True,allow_unused=False)
        gradients[key] = gradient.detach().float()
        result["minimal5_gradient/flow_"+key+"_norm"] = float(gradient.detach().float().norm())
    d, m = gradients["direction"].flatten(), gradients["displacement"].flatten()
    result["minimal5_gradient/direction_displacement_cosine"] = float((d*m).sum()/(d.norm()*m.norm()).clamp_min(1e-12))
    total = sum(g.norm() for g in gradients.values())
    result["minimal5_gradient/direction_norm_share"] = float(d.norm()/total.clamp_min(1e-12))
    return result
