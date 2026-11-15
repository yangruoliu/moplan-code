"""Sampling and decoding for Stage-1 WanFlowDiT.

The awkward part of a generative Stage-1 is that the evaluation metric is a
squared error, whose Bayes-optimal predictor is the conditional *mean*, while
Stage-2 wants a single self-consistent *sample*.  Both are produced here and
kept explicitly separate:

* ``reduce="mean"`` averages K decoded fields -- this is what the flow
  evaluation scores, and it is the only reduction that can be expected to beat
  the zero-flow baseline.
* ``reduce="first"`` returns one sample -- this is what feeds Stage-2.
* ``min_ade`` over the K samples is reported alongside as a multimodality
  diagnostic. If it never beats the mean, the model has collapsed to a
  deterministic map and the generative framing bought nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from .flow_dit import WanFlowDiTBackend, apply_shrinkage


LEGACY_ACTIVITY_FIELDS = ("slot_activity_logits", "pixel_activity_logits", "motion_mask",
                          "flow_ungated", "flow_activity_gated")
DENSE_MASK_FIELDS = ("dense_motion_mask_logits", "motion_mask", "flow_ungated", "flow_activity_gated")
TASK_OBJECT_SLOT_FIELDS = (
    "task_object_slot_logits", "task_object_slot_probability",
    "task_object_slot_mask", "flow_task_object_gated", "slot_params",
)
TASK_OBJECT_SELECTION_FIELDS = ("task_object_selection_mask", "flow_task_object_soft_gated")
TASK_OBJECT_DECOMPOSITION_FIELDS = (
    "task_object_identity_logits", "task_object_identity_probability",
    "task_object_motion_probability",
)


def motion_output_fields(output: dict[str, torch.Tensor]) -> tuple[str, ...]:
    """Keep legacy slot activity and direct full-grid segmentation unambiguous."""
    if "dense_motion_mask_logits" in output:
        if any(key in output for key in ("slot_activity_logits", "pixel_activity_logits")):
            raise ValueError("mixed dense-mask and legacy motion-activity output")
        if not all(key in output for key in DENSE_MASK_FIELDS):
            raise ValueError("incomplete dense motion-mask output")
        has_slot = any(key in output for key in TASK_OBJECT_SLOT_FIELDS)
        if has_slot and not all(key in output for key in TASK_OBJECT_SLOT_FIELDS):
            raise ValueError("incomplete task-object slot output")
        has_selection = any(key in output for key in TASK_OBJECT_SELECTION_FIELDS)
        if has_selection and (not has_slot or not all(key in output for key in TASK_OBJECT_SELECTION_FIELDS)):
            raise ValueError("incomplete hard task-object selection output")
        has_decomposition = any(
            key in output for key in TASK_OBJECT_DECOMPOSITION_FIELDS
        )
        if has_decomposition and not all(
            key in output for key in TASK_OBJECT_DECOMPOSITION_FIELDS
        ):
            raise ValueError("incomplete task-object identity/activity output")
        if has_decomposition and not has_slot:
            raise ValueError("task-object identity/activity output requires slot output")
        return (
            DENSE_MASK_FIELDS
            + (TASK_OBJECT_SLOT_FIELDS if has_slot else ())
            + (TASK_OBJECT_SELECTION_FIELDS if has_selection else ())
            + (TASK_OBJECT_DECOMPOSITION_FIELDS if has_decomposition else ())
        )
    if any(key in output for key in LEGACY_ACTIVITY_FIELDS):
        if not all(key in output for key in LEGACY_ACTIVITY_FIELDS):
            raise ValueError("incomplete motion-activity decoder output")
        return LEGACY_ACTIVITY_FIELDS
    return ()


@dataclass
class SamplerConfig:
    num_steps: int = 4
    guidance_scale: float = 1.5
    flow_shift: float = 5.0
    boundary_sigma: float = 0.9
    num_samples: int = 8
    shrinkage_px: float = 0.5
    # Averaging K samples pulls magnitude toward the conditional mean and the
    # soft threshold then removes another tau pixels from what survives; both
    # are strictly contractive and neither is compensated. This is the only
    # operator in the pipeline that can expand. Tuned on val like the rest.
    gain: float = 1.0
    reduce: Literal["mean", "first"] = "mean"
    seed: int = 42
    eval_side: float = 160.0

    @classmethod
    def from_config(cls, config: dict) -> "SamplerConfig":
        evaluation = config.get("evaluation", {})
        return cls(
            num_steps=int(evaluation.get("num_steps", 4)),
            guidance_scale=float(evaluation.get("guidance_scale", 1.5)),
            flow_shift=float(config["train"]["flow_shift"]),
            boundary_sigma=float(config["model"].get("boundary_sigma", 0.9)),
            num_samples=int(evaluation.get("num_samples", 8)),
            shrinkage_px=float(evaluation.get("shrinkage_px", 0.5)),
            gain=float(evaluation.get("gain", 1.0)),
            reduce=str(evaluation.get("reduce", "mean")),
            seed=int(evaluation.get("seed", config.get("seed", 42))),
        )


def sigma_schedule(num_steps: int, shift: float, device: torch.device) -> torch.Tensor:
    """Flow-matching sigma grid from 1 (noise) down to 0 (clean).

    Uses the same ``sigma = shift*u / (1 + (shift-1)*u)`` warp as training
    (``motion_jepa.stage2.sample_expert_sigmas``).  Training and inference must
    agree on ``shift``.
    """
    uniform = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return shift * uniform / (1.0 + (shift - 1.0) * uniform)


def _select_backend(
    backends: dict[str, WanFlowDiTBackend], sigma: float, boundary_sigma: float
) -> WanFlowDiTBackend:
    if "single" in backends:
        return backends["single"]
    if sigma >= boundary_sigma and "high" in backends:
        return backends["high"]
    return backends["low"]


@torch.inference_mode()
def sample_flow_latents(
    backends: dict[str, WanFlowDiTBackend],
    *,
    image_cond_latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    latent_shape: tuple[int, int, int, int],
    horizon_seconds: list[float],
    config: SamplerConfig,
    generator: torch.Generator | None = None,
    instance_channels: torch.Tensor | None = None,
    init_latent: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw ``config.num_samples`` flow latents for one conditioning pair.

    Args:
        image_cond_latent: ``[C, h, w]`` or ``[C, 1, h, w]`` raw VAE latent.
        prompt_embeds: ``[L, 4096]``.
        latent_shape: ``(C, T, h, w)`` of the flow latent to generate.
    Returns:
        ``[K, C, T, h, w]`` normalised latents (still in the backend's
        normalised space; hand them straight to the decoder).
    """
    reference = next(iter(backends.values()))
    device = reference.device
    channels, frames, height, width = latent_shape
    samples = config.num_samples

    condition = image_cond_latent
    if condition.ndim == 4:
        condition = condition[:, 0]
    condition = reference.normalize_latents(condition[None, :, None])[:, :, 0]
    condition = condition.expand(samples, -1, -1, -1)
    condition = condition[:, :, None].expand(-1, -1, frames, -1, -1)

    prompt = prompt_embeds[None].expand(samples, -1, -1).to(device, reference.train_dtype)
    use_guidance = (
        negative_prompt_embeds is not None and abs(config.guidance_scale - 1.0) > 1e-6
    )
    if use_guidance:
        negative = (
            negative_prompt_embeds[None]
            .expand(samples, -1, -1)
            .to(device, reference.train_dtype)
        )

    if init_latent is not None:
        # Deterministic (direct-objective) inference: start from the zero-flow
        # prior instead of noise. One Euler step from sigma=1 yields exactly
        # init - v = the regressed clean latent; further steps would query the
        # model at sigmas it never trained on, so they are refused rather than
        # silently produce garbage.
        # Multi-step is legitimate here when the checkpoint was trained as a
        # deterministic interpolant across sigma (dual-expert direct): each
        # Euler step queries the expert owning that sigma band. A single-expert
        # sigma=1-pinned checkpoint should still use num_steps=1; more steps
        # would query sigmas it never trained on.
        prior = reference.normalize_latents(init_latent[None, :, None].float())[:, :, 0]
        latent = (
            prior[:, :, None]
            .expand(samples, -1, frames, -1, -1)
            .contiguous()
            .to(device)
        )
    else:
        latent = torch.randn(
            (samples, channels, frames, height, width),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    schedule = sigma_schedule(config.num_steps, config.flow_shift, device)
    for index in range(config.num_steps):
        sigma = float(schedule[index])
        next_sigma = float(schedule[index + 1])
        backend = _select_backend(backends, sigma, config.boundary_sigma)
        sigmas = torch.full((samples,), sigma, device=device, dtype=torch.float32)
        latent_input = backend.build_input(
            latent,
            condition,
            horizon_seconds,
            instance_channels=(
                instance_channels[None].expand(samples, -1, -1, -1)
                if instance_channels is not None
                else None
            ),
        )
        velocity = backend.predict_velocity(latent_input, sigmas, prompt).float()
        if use_guidance:
            unconditional = backend.predict_velocity(latent_input, sigmas, negative).float()
            velocity = unconditional + config.guidance_scale * (velocity - unconditional)
        latent = latent + (next_sigma - sigma) * velocity
    return latent


@torch.inference_mode()
def decode_samples(
    decoder: torch.nn.Module,
    latents: torch.Tensor,
    *,
    image_cond_latent: torch.Tensor,
    first_frame: torch.Tensor,
    horizon_seconds: list[float],
    backend: WanFlowDiTBackend,
    chunk_size: int = 4,
    instance_masks: torch.Tensor | None = None,
    prompt_embeds: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Decode ``[K, C, T, h, w]`` latents into ``[K, horizons, ...]`` fields."""
    samples = latents.shape[0]
    condition = image_cond_latent
    if condition.ndim == 4:
        condition = condition[:, 0]
    condition = backend.normalize_latents(condition[None, :, None])[:, :, 0]

    # The semantic task-object head consumes the exact same caption embedding
    # that conditioned the flow backbone. A dataset item is unbatched here;
    # accept a leading singleton only for cached [1,L,D] tensors, then
    # replicate it across sampled latents/chunks.
    prompt_condition = None
    if prompt_embeds is not None:
        if prompt_embeds.ndim == 2:
            prompt_condition = prompt_embeds[None]
        elif prompt_embeds.ndim == 3 and prompt_embeds.shape[0] == 1:
            prompt_condition = prompt_embeds
        else:
            raise ValueError(
                "decode_samples prompt_embeds must have shape [L,D] or [1,L,D]"
            )

    flows, gates, visibility, masks = [], [], [], []
    activity: dict[str, list[torch.Tensor]] = {}
    activity_fields: tuple[str, ...] | None = None
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        count = stop - start
        decoder_kwargs = {}
        if prompt_condition is not None:
            decoder_kwargs["prompt_embeds"] = prompt_condition.expand(count, -1, -1)
        output = decoder(
            latents[start:stop],
            condition.expand(count, -1, -1, -1),
            first_frame[None].expand(count, -1, -1, -1),
            horizon_seconds=horizon_seconds,
            instance_masks=(
                instance_masks[None].expand(count, -1, -1, -1)
                if instance_masks is not None
                else None
            ),
            **decoder_kwargs,
        )
        flows.append(output["flow"].float())
        gates.append(output["gate_logits"].float())
        visibility.append(output["visibility_logits"].float())
        masks.append(output["object_mask_logits"].float())
        present_fields = motion_output_fields(output)
        if activity_fields is None:
            activity_fields = present_fields
            activity = {key: [] for key in present_fields}
        elif present_fields != activity_fields:
            raise ValueError("motion-activity output changed across decode chunks")
        for key in activity:
            activity[key].append(output[key].float())
    result = {
        "flow": torch.cat(flows),
        "gate_logits": torch.cat(gates),
        "visibility_logits": torch.cat(visibility),
        "object_mask_logits": torch.cat(masks),
    }
    for key, values in activity.items():
        if values:
            if len(values) != len(flows):
                raise ValueError("motion-activity output changed across decode chunks")
            result[key] = torch.cat(values)
    return result


def _mean_logits(logits: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Average in probability space, then return to logits.

    Averaging logits directly would be a geometric mean of odds, which is not
    what "the ensemble thinks this pixel is visible" means.
    """
    probability = logits.sigmoid().mean(dim=dim).clamp(1e-5, 1.0 - 1e-5)
    return torch.logit(probability)


def reduce_samples(
    decoded: dict[str, torch.Tensor],
    *,
    reduce: str,
    shrinkage_px: float,
    eval_side: float = 160.0,
    gain: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Collapse K decoded samples into the tensors the evaluator expects."""
    if reduce == "mean":
        flow = decoded["flow"].mean(dim=0)
        visibility_logits = _mean_logits(decoded["visibility_logits"])
        object_mask_logits = _mean_logits(decoded["object_mask_logits"])
    elif reduce == "first":
        flow = decoded["flow"][0]
        visibility_logits = decoded["visibility_logits"][0]
        object_mask_logits = decoded["object_mask_logits"][0]
    else:
        raise ValueError(f"unknown reduce {reduce!r}; expected 'mean' or 'first'")
    # Gain before shrinkage, so tau keeps meaning "displacements below this
    # many pixels are noise" in the units the flow is finally read in.
    flow = apply_shrinkage(flow * gain, shrinkage_px / eval_side)
    result = {
        "flow": flow,
        "visibility_logits": visibility_logits,
        "object_mask_logits": object_mask_logits,
    }
    # Keep the activity outputs and BOTH flow alternatives from exactly the
    # same forward/sample reduction. Extra flow fields deliberately remain
    # pre-gain/pre-shrinkage, so gating is never confused with postprocessing.
    extra_keys = motion_output_fields(decoded)
    if extra_keys:
        for key in extra_keys:
            if reduce == "first":
                result[key] = decoded[key][0]
            elif key.endswith("_logits"):
                result[key] = _mean_logits(decoded[key])
            else:
                # mean(p * flow) must not become mean(p) * mean(flow).
                result[key] = decoded[key].mean(dim=0)
    return result


def min_ade_over_samples(
    decoded_flow: torch.Tensor,
    target_flow: torch.Tensor,
    object_mask: torch.Tensor,
    visibility: torch.Tensor,
    *,
    width: int = 160,
    height: int = 160,
) -> dict[str, float]:
    """Best-of-K trajectory error -- the multimodality diagnostic.

    Reported separately from the main flow metrics; ``mean-of-K`` is the number
    that gets compared against the baselines.
    """
    from .losses import trajectory_centers

    samples = decoded_flow.shape[0]
    scale = decoded_flow.new_tensor([width, height]).view(1, 1, 2)
    target_centers = trajectory_centers(
        target_flow[None], object_mask[None], visibility[None]
    )[0]
    errors = []
    for index in range(samples):
        centers = trajectory_centers(
            decoded_flow[index][None], object_mask[None], visibility[None]
        )[0]
        distance = ((centers - target_centers) * scale[0]).square().sum(dim=-1).sqrt()
        errors.append(distance)
    stacked = torch.stack(errors)
    return {
        "min_ade_px": float(stacked.mean(dim=-1).min()),
        "min_fde_px": float(stacked[:, -1].min()),
        "mean_ade_px": float(stacked.mean(dim=-1).mean()),
        "sample_ade_spread_px": float(stacked.mean(dim=-1).std(unbiased=False)),
    }
