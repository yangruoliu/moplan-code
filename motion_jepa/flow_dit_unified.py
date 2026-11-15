"""Fresh unified Stage1: balanced material-point regression and signed direction.

The two groups are trusted movers and all remaining valid correspondences. The
latter keep their measured vectors, including gray-band motion; they are not
static pseudo-labels. Neither SAM coverage nor a predicted gate sets support.
"""
from __future__ import annotations

import math

import torch

from .flow_dit_clean import freeze_clean_unused_heads

PROFILE = "unified_trajectory_v1"


def unified_world_size(config, *, required=False):
    """Read a frozen 4/8-rank contract without changing older config ABIs."""
    experiment = config.get("unified_experiment")
    if experiment is None and not required:
        return None
    if not isinstance(experiment, dict):
        raise ValueError("unified experiment metadata is required for its world_size")
    world_size = experiment.get("world_size")
    if type(world_size) is not int or world_size not in (4, 8):
        raise ValueError("unified experiment world_size must be 4 or 8")
    train = config["train"]
    for key, expected in (("global_batch", 32), ("per_gpu_batch", 1), ("grad_accum_steps", 32 // world_size)):
        if type(train.get(key)) is not int or train[key] != expected:
            raise ValueError(f"unified {world_size}-rank contract requires train.{key}={expected}")
    return world_size


def validate_unified_runtime(config, actual_world_size):
    expected = unified_world_size(config)
    if expected is not None and actual_world_size != expected:
        raise ValueError(f"unified runtime world_size {actual_world_size} differs from frozen config {expected}; "
                         "changing rank count is not an exact resume")


def validate_unified_campaign(config):
    """Common fresh-base/runtime contract, including the legacy-objective arm."""
    train = config["train"]
    if type(train.get("unified_campaign", False)) is not bool:
        raise ValueError("train.unified_campaign must be an explicit boolean")
    if not (train.get("unified_campaign", False) or config["loss"].get("profile") == PROFILE):
        return
    unified_world_size(config)
    decoder, data = config["model"]["decoder"], config["data"]
    if train.get("objective") != "direct" or float(config["loss"].get("flow_matching", 0)) != 0:
        raise ValueError("unified campaign requires direct prediction without latent FM")
    if any(train.get(key) for key in ("refine_from", "activity_heads_only", "mask_heads_only")):
        raise ValueError("unified campaign forbids a refinement partner or head-only training")
    if train.get("sampler") != "source_cycle_v1" or not train.get("exact_resume", False):
        raise ValueError("unified campaign requires source_cycle_v1 and exact_resume")
    if float(train.get("caption_dropout", 0)) != 0:
        raise ValueError("unified campaign requires caption_dropout=0")
    if decoder.get("type") != "slots" or int(decoder.get("num_slots", 0)) != 16 or int(data.get("emit_instance_masks", 0)) != 16:
        raise ValueError("unified campaign requires SAM16 slot conditioning")
    for key in ("oracle_instance_masks", "residual_branch", "text_slot_binding", "zero_init_flow_head", "temporal_slots", "causal_slots"):
        if decoder.get(key) is not True:
            raise ValueError(f"unified campaign requires decoder.{key}=true")
    if decoder.get("slot_pooling_mode") != "area":
        raise ValueError("unified campaign requires area slot pooling")
    if float(decoder.get("residual_scale", .05)) != .05:
        raise ValueError("unified campaign requires residual_scale=0.05")
    if decoder.get("incremental_params", False):
        raise ValueError("unified campaign retains cumulative affine parameters")
    for key, expected in (("num_steps", 1), ("num_samples", 1), ("guidance_scale", 1.), ("gain", 1.), ("shrinkage_px", 0.)):
        if float(config["evaluation"].get(key, float("nan"))) != expected:
            raise ValueError(f"unified campaign requires raw single-step evaluation.{key}={expected}")
    if config["evaluation"].get("reduce") != "first":
        raise ValueError("unified campaign requires evaluation.reduce=first")
    minimum = train.get("min_unique_scenes", 20000)
    if type(minimum) is not int or minimum < 0:
        raise ValueError("min_unique_scenes must be a nonnegative integer")
    minima = train.get("min_unique_scenes_by_source", {})
    if not isinstance(minima, dict) or any(not isinstance(k, str) or not k or type(v) is not int or v < 0 for k, v in minima.items()):
        raise ValueError("min_unique_scenes_by_source must map source names to nonnegative integers")


def validate_unified_config(config):
    validate_unified_campaign(config)
    loss, decoder, train = config["loss"], config["model"]["decoder"], config["train"]
    fixed = {"flow_matching": 0., "flow_charbonnier": 1., "flow_epsilon_px": .5,
             "trajectory_scale_px": 16., "motion_direction_epsilon_px": .5,
             "aux_warmup_steps": 0.}
    allowed = set(fixed) | {"profile", "supervision_mode", "aux_sigma_weight", "moving_direction"}
    if set(loss) - allowed:
        raise ValueError(f"unified trajectory rejects extra loss settings: {sorted(set(loss) - allowed)}")
    for key, value in fixed.items():
        if float(loss.get(key, value)) != value:
            raise ValueError(f"unified trajectory fixes loss.{key}={value}")
    if float(loss.get("moving_direction", .1)) not in {0., .1}:
        raise ValueError("unified trajectory moving_direction must be 0 or 0.1")
    if loss.get("supervision_mode") != "valid_partition_v1" or loss.get("aux_sigma_weight", "uniform") != "uniform":
        raise ValueError("unified trajectory requires valid_partition_v1 and uniform weighting")
    for key in ("dense_motion_mask", "motion_activity", "task_object_slot_activity", "task_object_identity", "dense_correction"):
        if decoder.get(key, False):
            raise ValueError(f"unified trajectory requires decoder.{key}=false")
    if not train.get("freeze_visibility_heads", False):
        raise ValueError("unified trajectory freezes diagnostic visibility heads")


def validate_fresh_start(config, init_from="", resume_state=None):
    """Do not relabel an old checkpoint as a fresh unified training run."""
    if not (config["train"].get("unified_campaign", False) or config["loss"].get("profile") == PROFILE):
        return
    if init_from:
        raise ValueError("unified Stage1 starts from original Wan; init-from is forbidden")
    if resume_state is not None and resume_state.get("initialization", {}).get("kind") != "pretrained_backbone_new_heads":
        raise ValueError("unified resume requires its own fresh-base initialization provenance")


def freeze_unified_unused_heads(decoder):
    return freeze_clean_unused_heads(decoder)


def _support(batch, key, shape, reference):
    value = batch[key].to(reference)
    if value.shape != shape or not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{key} must contain finite {shape} weights in [0,1]")
    return value


def _object_means(error, support, labels):
    """Return one value per sample, including differentiable zero for empties."""
    values, present, counts = [], [], []
    for index in range(error.shape[0]):
        active = support[index].sum(dim=(0, 1)) > 0
        regions = []
        for label in torch.unique(labels[index][active]):
            weights = support[index] * (labels[index] == label)[None, None]
            regions.append((error[index] * weights).sum() / weights.sum())
        values.append(torch.stack(regions).mean() if regions else error[index].sum() * 0)
        present.append(bool(regions))
        counts.append(len(regions))
    return torch.stack(values), error.new_tensor(present), error.new_tensor(counts)


def unified_trajectory_losses(prediction, batch, config, sample_weight=None, *, width=160, height=160):
    predicted, target = prediction["flow"].float(), batch["flow"].float()
    if predicted.shape != target.shape or target.ndim != 5 or target.shape[2] != 2 or target.shape[0] < 1:
        raise ValueError("unified trajectory needs matching nonempty [B,T,2,H,W] fields")
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        raise ValueError("trajectory pixel dimensions must be positive and finite")
    b, t, _, h, w = target.shape
    shape = (b, t, 1, h, w)
    valid = _support(batch, "flow_valid", shape, predicted)
    moving = _support(batch, "motion_target", shape, predicted)
    motion_valid = _support(batch, "motion_valid", shape, predicted)
    if bool((moving > valid).any() | (moving > motion_valid).any()):
        raise ValueError("motion_target must be a subset of flow_valid and motion_valid")
    if "content_valid" in batch:
        valid = valid * _support(batch, "content_valid", (b, 1, h, w), predicted)[:, None]
    # Unknown predictions AND labels are removed before any arithmetic/norm.
    predicted = torch.where(valid > 0, predicted, 0)
    target = torch.where(valid > 0, target, 0)
    if not bool(torch.isfinite(predicted).all() & torch.isfinite(target).all()):
        raise ValueError("nonfinite trajectory on known correspondence")
    labels = batch["instance_id"].to(predicted.device)
    if labels.shape == (b, 1, h, w):
        labels = labels[:, 0]
    if labels.shape != (b, h, w) or not bool(torch.isfinite(labels).all()) or bool((labels < 0).any()) or not torch.equal(labels, labels.long()):
        raise ValueError("instance_id must contain finite nonnegative integer [B,1,H,W] labels")
    if sample_weight is not None:
        sample_weight = sample_weight.to(predicted)
        if sample_weight.shape != (b,) or not bool(torch.isfinite(sample_weight).all()) or bool((sample_weight < 0).any()):
            raise ValueError("sample_weight must be finite nonnegative [B]")
    else:
        sample_weight = predicted.new_ones(b)
    pixels = predicted.new_tensor([width, height]).view(1, 1, 2, 1, 1)
    p, g = predicted * pixels, target * pixels
    error = ((p - g).square().sum(2, keepdim=True) + .5**2).sqrt().sub(.5) / 16.
    moving_support, rest_support = valid * moving, valid * (1. - moving)
    mover, mover_present, mover_count = _object_means(error, moving_support, labels)
    rest, rest_present, rest_count = _object_means(error, rest_support, labels)
    # Renormalize missing groups within each sample, never across a batch/rank.
    point_per_sample = (mover + rest) / (mover_present + rest_present).clamp_min(1)
    point = (point_per_sample * sample_weight).mean()
    magnitude = g.square().sum(2, keepdim=True).sqrt()
    direction_support = moving_support * (magnitude > 1.).to(moving_support)
    target_unit = g / magnitude.clamp_min(1e-6)
    signed_cosine = (p * target_unit).sum(2, keepdim=True) / (p.square().sum(2, keepdim=True) + .5**2).sqrt()
    direction_values, _, direction_count = _object_means(1. - signed_cosine, direction_support, labels)
    direction = (direction_values * sample_weight).mean()
    coefficient = float(config.get("moving_direction", .1))
    if coefficient not in {0., .1}:
        raise ValueError("unified trajectory moving_direction must be 0 or 0.1")
    zero = point.detach().new_zeros(())
    logging_abi = {key: zero for key in ("gate", "visibility", "object_mask", "slot_mask", "slot_affine", "object_displacement", "object_occupancy", "position", "velocity", "acceleration")}
    return {**logging_abi, "aux_total": point + coefficient * direction, "flow": point,
            "moving_direction": direction, "unified_point": point, "unified_direction": direction,
            "unified_direction_weighted": coefficient * direction,
            "unified_moving_point": (mover * sample_weight).mean(),
            "unified_rest_point": (rest * sample_weight).mean(),
            "unified_moving_objects": mover_count.sum().detach(),
            "unified_rest_objects": rest_count.sum().detach(),
            "unified_direction_objects": direction_count.sum().detach(),
            "unified_valid_points": valid.sum().detach(),
            "unified_moving_points": moving_support.sum().detach(),
            "unified_rest_points": rest_support.sum().detach()}


def gradient_diagnostics(decoder, lora_parameters=(), *, gradient_scale=1.):
    """Read accumulated parameter gradients; no retained graph/extra backward."""
    if not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError("gradient_scale must be positive and finite")
    named = list(decoder.named_parameters())
    prefixes = {
        "affine": ("slot_params.",), "residual": ("residual_head.",),
        "shared_motion": ("latent_stem.", "latent_blocks.", "slot_project.", "slot_temporal.", "horizon_embedding"),
        "text_gain": ("tsb_score.",),
        "text_attention": ("tsb_token_project.", "tsb_query_project.", "tsb_attention."),
        "text_binding": ("tsb_score.", "tsb_token_project.", "tsb_query_project.", "tsb_attention."),
        "text_fusion": ("tsb_fusion.",),
    }
    groups = {key: [p for name, p in named if name.startswith(value) and p.requires_grad] for key, value in prefixes.items()}
    groups["lora"] = [p for p in lora_parameters if p.requires_grad]
    result = {}
    for name, parameters in groups.items():
        gradients = [p.grad for p in parameters if p.grad is not None]
        norm = float(torch.stack([g.detach().float().square().sum() for g in gradients]).sum().sqrt()) / gradient_scale if gradients else 0.
        result[f"grad_preclip/{name}"] = norm
        result[f"grad_tensors/{name}"] = len(gradients)
        result[f"grad_missing/{name}"] = len(parameters) - len(gradients)
    for output, source in (("decoder_shared", "shared_motion"), ("lora", "lora"),
                           ("text_binding", "text_binding"), ("text_fusion", "text_fusion")):
        result[f"unified_grad/{output}"] = result[f"grad_preclip/{source}"]
    return result


def aggregate_gradient_diagnostics(rows):
    """Expose every rank's preclip norm and a conservative minimum summary."""
    if not rows:
        raise ValueError("gradient diagnostics require at least one rank")
    result = dict(rows[0])
    for key in ("decoder_shared", "lora", "text_binding", "text_fusion"):
        values = [float(row[f"unified_grad/{key}"]) for row in rows]
        # A NaN in any rank must remain visible rather than Python min hiding it.
        low = min(values) if all(math.isfinite(value) for value in values) else float("nan")
        high = max(values) if all(math.isfinite(value) for value in values) else float("nan")
        result[f"unified_grad/{key}"] = low
        result[f"unified_grad/{key}_rank_min"] = low
        result[f"unified_grad/{key}_rank_max"] = high
        result[f"unified_grad/{key}_by_rank"] = values
    result["unified_grad/world_size"] = len(rows)
    return result


class UpdateLossAccumulator:
    """Detached scalar sums for a true global optimizer-update sample mean.

Ranks gather the key lists before packing, then all-reduce equal-sized vectors.
Missing optional diagnostics contribute zero; all samples stay in the divisor.
"""
    def __init__(self):
        self.sums = {}
        self.samples = 0

    def add(self, values, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("update loss logging requires a positive batch size")
        for key, value in values.items():
            if not torch.is_tensor(value) or value.numel() != 1:
                raise ValueError(f"update loss logging requires scalar tensors: {key}")
            weighted = value.detach().float().reshape(()) * batch_size
            self.sums[key] = self.sums[key] + weighted if key in self.sums else weighted
        self.samples += batch_size

    def vector(self, keys):
        if not self.sums or self.samples < 1 or set(self.sums) - set(keys):
            raise ValueError("update loss logging needs accumulated samples and the complete key union")
        reference = next(iter(self.sums.values()))
        return torch.stack([self.sums.get(key, reference.new_zeros(())) for key in keys]
                           + [reference.new_tensor(self.samples)])

    def reset(self):
        self.sums.clear()
        self.samples = 0
