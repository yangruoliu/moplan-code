"""Stage-1 WanFlowDiT: predict first-to-future optical flow with a Wan expert.

The flow sequence is treated as an 8-frame "video" (slot 0 is the known
zero-flow frame) encoded by :mod:`motion_jepa.flow_codec`.  A single Wan2.2
I2V-A14B expert denoises it under the model's *native* image-to-video channel
layout, so the frozen backbone keeps seeing what it was trained on:

    [ noisy flow latent (C) | mask (4) | image condition (C) | time (1) ]

Only three things are new -- the two projection layers that bracket the frozen
stack (``flow_patch_embedding`` / ``flow_proj_out``, both initialised from the
pretrained ones) and :class:`LatentFlowDecoder`, which turns the denoised latent
into a 160x160 flow field under direct supervision.

Two facts from the data drive the design and are worth restating here, because
they are easy to violate by accident:

* Only ``object_mask & visibility`` pixels are scored, and most of them are
  static.  Predicting zero everywhere is already a strong baseline, so the flow
  head is **zero-initialised**: an untrained model reproduces the zero-flow
  baseline exactly and training can only move away from it deliberately.
* The latent grid imposes a hard resampling floor at 32x32.  The
  decoder therefore fuses full-resolution first-frame features rather than
  upsampling the latent on its own.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2 import DEFAULT_LORA_TARGETS, Residual3DBlock, _u_for_sigma
from .flow_dit_time import HorizonSeconds, normalized_horizon_times, validate_time_scale


# Channel layout of Wan's pretrained i2v patch embedding.
PRETRAINED_LATENT_CHANNELS = 16
PRETRAINED_MASK_SLICE = slice(16, 20)
PRETRAINED_IMAGE_SLICE = slice(20, 36)
MASK_CHANNELS = 4


class _CastConv3d(nn.Conv3d):
    """Conv3d holding fp32 weights inside an otherwise bf16 stack.

    PEFT's LoRA layers cast their input to the adapter dtype themselves, which
    is why fp32 LoRA works alongside a bf16 backbone; plain ``nn.Conv3d`` does
    not, so the cast is done here.  The two new projections bracket the frozen
    stack and are cheap (1.1M parameters together), so fp32 master weights are
    worth the two casts.
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value.to(self.weight.dtype)).to(value.dtype)


class _CastLinear(nn.Linear):
    """Linear counterpart of :class:`_CastConv3d`."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value.to(self.weight.dtype)).to(value.dtype)


def expert_subfolder(expert: str) -> str:
    """Map an expert name to its diffusers subfolder.

    ``transformer_2`` is the low-noise expert and ``transformer`` the high-noise
    one -- the same convention as ``motion_jepa/stage2.py:146``.  ``single``
    trains one model over the whole sigma range for cheap pilots.
    """
    if expert in ("low", "single"):
        return "transformer_2"
    if expert == "high":
        return "transformer"
    raise ValueError(f"expert must be low, high or single, got {expert!r}")


def sample_flow_sigmas(
    batch_size: int,
    *,
    expert: str,
    boundary_sigma: float,
    shift: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sigma sampler that additionally understands the ``single`` expert.

    Kept here rather than patched into ``stage2.sample_expert_sigmas`` so the
    Stage-2 training path kept its exact behaviour.
    """
    boundary_u = _u_for_sigma(boundary_sigma, shift)
    lower, upper = {
        "low": (0.0, boundary_u),
        "high": (boundary_u, 1.0),
        "single": (0.0, 1.0),
    }[expert]
    uniform = lower + torch.rand(batch_size, device=device, generator=generator) * (upper - lower)
    sigma = shift * uniform / (1.0 + (shift - 1.0) * uniform)
    return sigma.clamp_(0.001, 0.999)


def _copy_patch_embedding(
    source: nn.Conv3d,
    target: nn.Conv3d,
    latent_channels: int,
) -> None:
    """Initialise the new patch embedding from the pretrained i2v one.

    The new layout interleaves the same semantic groups, so each group is copied
    from its pretrained counterpart and the extra time channel starts at zero
    (it must not perturb the pretrained activation statistics at step 0).
    """
    with torch.no_grad():
        target.weight.zero_()
        usable = min(latent_channels, PRETRAINED_LATENT_CHANNELS)
        noisy = slice(0, usable)
        mask = slice(latent_channels, latent_channels + MASK_CHANNELS)
        image = slice(latent_channels + MASK_CHANNELS, latent_channels + MASK_CHANNELS + usable)
        target.weight[:, noisy] = source.weight[:, :usable]
        target.weight[:, mask] = source.weight[:, PRETRAINED_MASK_SLICE]
        target.weight[:, image] = source.weight[:, PRETRAINED_IMAGE_SLICE][:, :usable]
        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias)


def _copy_proj_out(source: nn.Linear, target: nn.Linear, latent_channels: int, patch_elements: int) -> None:
    """Initialise the new output head from the pretrained one.

    ``WanTransformer3DModel`` unpatchifies with channel as the *fastest* axis
    (``transformer_wan.py:699-702`` reshapes to ``(..., p_t, p_h, p_w, C)``), so
    a channel subset is a strided row selection, not a prefix.
    """
    with torch.no_grad():
        target.weight.zero_()
        if target.bias is not None:
            target.bias.zero_()
        usable = min(latent_channels, PRETRAINED_LATENT_CHANNELS)
        rows = torch.arange(patch_elements)[:, None] * PRETRAINED_LATENT_CHANNELS + torch.arange(usable)[None, :]
        destination = torch.arange(patch_elements)[:, None] * latent_channels + torch.arange(usable)[None, :]
        target.weight[destination.reshape(-1)] = source.weight[rows.reshape(-1)]
        if source.bias is not None and target.bias is not None:
            target.bias[destination.reshape(-1)] = source.bias[rows.reshape(-1)]


class WanFlowDiTBackend:
    """One Wan expert re-purposed to denoise flow latents."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        expert: str,
        latent_channels: int = PRETRAINED_LATENT_CHANNELS,
        lora_rank: int = 32,
        lora_alpha: int = 32,
        latents_mean: torch.Tensor | None = None,
        latents_std: torch.Tensor | None = None,
        device: torch.device,
        flow_shift: float = 5.0,
        boundary_sigma: float = 0.9,
        gradient_checkpointing: bool = True,
        target_modules: list[str] | None = None,
        train_dtype: torch.dtype = torch.bfloat16,
        use_time_channel: bool = True,
        use_instance_channels: bool = False,
        objective: str = "flow_matching",
        time_scale: float | None = None,
    ) -> None:
        from diffusers import WanTransformer3DModel
        from peft import LoraConfig

        self.expert = expert
        self.device = device
        self.train_dtype = train_dtype
        self.flow_shift = float(flow_shift)
        self.boundary_sigma = float(boundary_sigma)
        self.latent_channels = int(latent_channels)
        self.use_time_channel = bool(use_time_channel)
        # Union mask + instance boundaries, zero-initialised like the time
        # channel, so step 0 is bit-identical to an arm without them. The
        # boundary map is the informative half: the union says "something is
        # here", the boundaries say "these are separate bodies".
        self.use_instance_channels = bool(use_instance_channels)
        # Fixed denominator for the time channel. Per-sample normalisation maps
        # every uniform horizon grid onto the same 1/T..1 ladder, which erases
        # the one thing a joint Physion(1.4s)+PhyCo(4.0s) model must know: how
        # much time it is being asked to span. None keeps the per-sample
        # convention for single-domain checkpoints.
        self.time_scale = validate_time_scale(time_scale)
        if objective not in ("flow_matching", "direct"):
            raise ValueError(f"unknown objective {objective!r}")
        # "direct" is flow matching pinned at sigma=1 with the Gaussian noise
        # replaced by the deterministic zero-flow prior latent (slot 0's
        # content, which is sample-independent: it encodes "no motion, all
        # visible"). The velocity target becomes prior - clean, i.e. plain
        # regression of the clean latent, one forward pass, no sampling. An
        # oracle-decoder probe motivated it: handed the clean latent, the
        # decoder recovers clearly more motion than the diffusion path, so much
        # of the achievable signal was being lost between the supervision and
        # the weights.
        self.objective = objective
        # Refinement training for the low expert: a frozen high-expert backend
        # whose sigma=1 estimate supplies the *input* distribution the low
        # expert actually sees at inference. Set externally by the trainer.
        self.refine_partner = None
        self.refine_sigma = 5.0 * 0.5 / (1.0 + 4.0 * 0.5)  # midpoint of the 2-step schedule, sigma = 0.8333

        self.transformer = WanTransformer3DModel.from_pretrained(
            Path(model_dir) / expert_subfolder(expert), torch_dtype=train_dtype
        )
        self.transformer.requires_grad_(False)

        inner_dim = (
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )
        self.hidden_dim = inner_dim
        patch_size = tuple(self.transformer.config.patch_size)
        patch_elements = int(patch_size[0] * patch_size[1] * patch_size[2])
        in_channels = (
            2 * self.latent_channels
            + MASK_CHANNELS
            + int(self.use_time_channel)
            + 2 * int(self.use_instance_channels)
        )

        # Swapping the modules outright is cleaner than hooking forward: the
        # stock forward already calls self.patch_embedding / self.proj_out.
        pretrained_embedding = self.transformer.patch_embedding
        flow_embedding = _CastConv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        ).to(pretrained_embedding.weight.dtype)
        _copy_patch_embedding(pretrained_embedding, flow_embedding, self.latent_channels)

        pretrained_proj = self.transformer.proj_out
        flow_proj = _CastLinear(inner_dim, self.latent_channels * patch_elements).to(
            pretrained_proj.weight.dtype
        )
        _copy_proj_out(pretrained_proj, flow_proj, self.latent_channels, patch_elements)

        self.transformer.patch_embedding = flow_embedding
        self.transformer.proj_out = flow_proj
        self.transformer.config.in_channels = in_channels
        self.transformer.config.out_channels = self.latent_channels
        self.transformer.register_to_config(
            in_channels=in_channels, out_channels=self.latent_channels
        )

        lora = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules or DEFAULT_LORA_TARGETS),
            init_lora_weights=True,
        )
        self.transformer.add_adapter(lora)
        for name, parameter in self.transformer.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
                parameter.requires_grad_(True)
        # The two new projections are trained in fp32 for stability; they are
        # tiny (0.8M + 0.3M) next to the 153M of LoRA.
        for module in (self.transformer.patch_embedding, self.transformer.proj_out):
            module.float()
            module.requires_grad_(True)

        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.transformer.train().to(device)

        if latents_mean is None or latents_std is None:
            self.latents_mean = torch.zeros(self.latent_channels, device=device).view(1, -1, 1, 1, 1)
            self.latents_std = torch.ones(self.latent_channels, device=device).view(1, -1, 1, 1, 1)
        else:
            self.latents_mean = latents_mean.to(device)[: self.latent_channels].view(1, -1, 1, 1, 1)
            self.latents_std = latents_std.to(device)[: self.latent_channels].view(1, -1, 1, 1, 1)

    # ------------------------------------------------------------------ utils

    def normalize_latents(self, value: torch.Tensor) -> torch.Tensor:
        return (value.float() - self.latents_mean) / self.latents_std

    def denormalize_latents(self, value: torch.Tensor) -> torch.Tensor:
        return value.float() * self.latents_std + self.latents_mean

    def time_channel(
        self, shape: tuple[int, ...], horizon_seconds: HorizonSeconds
    ) -> torch.Tensor:
        """Normalised absolute time per latent frame, broadcast to the grid.

        Slot 0 is t = 0 (the known zero-flow frame); the remaining slots carry
        ``t / time_scale`` when configured (otherwise legacy ``t / t_max``),
        independent of whether seconds came from training tensors or a list.

        Accepts a global list or a per-sample ``[B, T]`` tensor of seconds --
        the joint Physion+PhyCo dataset shuffles 1.4s rows next to 4.0s rows in
        the same batch, so the span has to ride with the sample.
        """
        batch_size, _, frames, height, width = shape
        times = normalized_horizon_times(
            horizon_seconds, batch_size=batch_size, horizons=frames - 1,
            time_scale=self.time_scale, device=self.device,
        )
        times = torch.cat((times.new_zeros(batch_size, 1), times), dim=1)
        return times.view(batch_size, 1, frames, 1, 1).expand(
            batch_size, 1, frames, height, width
        )

    def build_input(
        self,
        noisy: torch.Tensor,
        image_condition: torch.Tensor,
        horizon_seconds: HorizonSeconds,
        instance_channels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble Wan's i2v channel stack for a flow video."""
        batch_size, channels, frames, height, width = noisy.shape
        mask = noisy.new_zeros(batch_size, MASK_CHANNELS, frames, height, width)
        mask[:, :, 0] = 1.0
        parts = [noisy, mask, image_condition]
        if self.use_time_channel:
            parts.append(self.time_channel(noisy.shape, horizon_seconds))
        if self.use_instance_channels:
            if instance_channels is None:
                raise ValueError(
                    "use_instance_channels is on but no instance_channels were "
                    "passed; refusing to fall back to zeros silently"
                )
            resized = F.interpolate(
                instance_channels.to(noisy.device, torch.float32),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            parts.append(resized[:, :, None].expand(-1, -1, frames, -1, -1))
        return torch.cat(parts, dim=1).to(self.train_dtype)

    def predict_velocity(
        self,
        latent_input: torch.Tensor,
        sigmas: torch.Tensor,
        prompt: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=latent_input,
            timestep=sigmas * 1000.0,
            encoder_hidden_states=prompt,
            return_dict=False,
        )[0]

    @staticmethod
    def x0_from_velocity(
        noisy: torch.Tensor, velocity: torch.Tensor, sigmas: torch.Tensor
    ) -> torch.Tensor:
        """Recover the clean latent: ``x0 = noisy - sigma * v`` (exact for v = noise - clean)."""
        sigma = sigmas.view(-1, 1, 1, 1, 1)
        return noisy.float() - sigma * velocity.float()

    def expand_image_condition(
        self, image_cond_latent: torch.Tensor, frames: int
    ) -> torch.Tensor:
        """Broadcast the single-frame image latent across the flow timeline."""
        if image_cond_latent.ndim == 5:
            if image_cond_latent.shape[2] == frames:
                return image_cond_latent
            image_cond_latent = image_cond_latent[:, :, 0]
        return image_cond_latent[:, :, None].expand(-1, -1, frames, -1, -1)

    # ------------------------------------------------------------------ train

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        *,
        horizon_seconds: HorizonSeconds,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        clean = self.normalize_latents(batch["flow_latent"].to(self.device))
        image_condition = self.normalize_latents(batch["image_cond_latent"].to(self.device))
        prompt = batch["prompt_embeds"].to(self.device, self.train_dtype)
        batch_size, _, frames, _, _ = clean.shape
        image_condition = self.expand_image_condition(image_condition, frames)

        if self.objective == "direct":
            prior = clean[:, :, :1].expand_as(clean).contiguous()
            if self.expert == "low" and self.refine_partner is not None:
                # The low expert's real job in the 2-step schedule is to REFINE
                # the high expert's estimate, so its training input must be the
                # high expert's output, not the true interpolant. On the true
                # interpolant the task is algebraic (the input contains z
                # linearly) and the model learns an identity shortcut: sigma-band
                # training improves early and then worsens, the signature of an
                # answer-leak shortcut breaking at eval.
                sigma_2 = float(self.refine_sigma)
                with torch.no_grad():
                    partner_input = self.refine_partner.build_input(
                        prior,
                        image_condition,
                        batch.get("horizon_seconds", horizon_seconds),
                        instance_channels=batch.get("instance_channels"),
                    )
                    partner_v = self.refine_partner.predict_velocity(
                        partner_input,
                        torch.ones(batch_size, device=self.device),
                        prompt,
                    ).float()
                estimate = prior - partner_v
                sigmas = torch.full((batch_size,), sigma_2, device=self.device)
                noisy = (1.0 - sigma_2) * estimate + sigma_2 * prior
                latent_input = self.build_input(
                    noisy,
                    image_condition,
                    batch.get("horizon_seconds", horizon_seconds),
                    instance_channels=batch.get("instance_channels"),
                )
                velocity = self.predict_velocity(latent_input, sigmas, prompt)
                # One Euler step from sigma_2 must land on the clean latent:
                # z = noisy - sigma_2 * v  =>  v target = (noisy - z)/sigma_2.
                target = (noisy - clean) / sigma_2
                flow_matching = F.mse_loss(velocity.float(), target)
                return {
                    "flow_matching": flow_matching,
                    "x0": self.x0_from_velocity(noisy, velocity, sigmas),
                    "sigmas": sigmas,
                }
            # single AND high both pin sigma=1: on a deterministic path the
            # informative input is the pure prior; any sigma<1 leaks a fraction
            # of the answer into the input and teaches a copy shortcut.
            sigmas = torch.ones(batch_size, device=self.device)
            noise = prior
        else:
            sigmas = sample_flow_sigmas(
                batch_size,
                expert=self.expert,
                boundary_sigma=self.boundary_sigma,
                shift=self.flow_shift,
                device=self.device,
                generator=generator,
            )
            noise = torch.randn(
                clean.shape, device=self.device, dtype=torch.float32, generator=generator
            )
        sigma = sigmas.view(batch_size, 1, 1, 1, 1)
        noisy = (1.0 - sigma) * clean + sigma * noise
        latent_input = self.build_input(
            noisy,
            image_condition,
            batch.get("horizon_seconds", horizon_seconds),
            instance_channels=batch.get("instance_channels"),
        )
        velocity = self.predict_velocity(latent_input, sigmas, prompt)
        flow_matching = F.mse_loss(velocity.float(), noise - clean)
        return {
            "flow_matching": flow_matching,
            "x0": self.x0_from_velocity(noisy, velocity, sigmas),
            "sigmas": sigmas,
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.transformer.named_parameters()
            if parameter.requires_grad
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        parameters = dict(self.transformer.named_parameters())
        missing = [name for name in state if name not in parameters]
        if missing:
            raise KeyError(f"checkpoint parameter not found in model: {missing[0]}")
        with torch.no_grad():
            for name, tensor in state.items():
                parameters[name].copy_(tensor.to(parameters[name].dtype))


class _ImageStem(nn.Module):
    """Multi-scale features from the first frame at 160, 80 and 40."""

    def __init__(self, widths: tuple[int, int, int] = (16, 32, 64)) -> None:
        super().__init__()
        full, half, quarter = widths
        self.level0 = nn.Sequential(
            nn.Conv2d(3, full, 3, padding=1), nn.GroupNorm(min(8, full), full), nn.SiLU()
        )
        self.level1 = nn.Sequential(
            nn.Conv2d(full, half, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, half), half),
            nn.SiLU(),
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(half, quarter, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, quarter), quarter),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        level0 = self.level0(image)
        level1 = self.level1(level0)
        level2 = self.level2(level1)
        return [level0, level1, level2]


class LatentFlowDecoder(nn.Module):
    """Decode denoised flow latents into a supervised 160x160 flow field.

    Deliberately *not* the VAE decoder: bilinear or transposed-conv upsampling
    from a 32x32 grid carries a sizeable resampling error floor.  Fusing
    first-frame features at full resolution is what
    buys that back, because flow discontinuities coincide with object edges that
    are perfectly visible in the conditioning frame.

    All randomness lives upstream in the latent; this map is deterministic and
    trained directly against the ground-truth flow, which is what keeps a
    generative Stage-1 from losing to the zero-flow baseline on EPE.
    """

    def __init__(
        self,
        *,
        latent_channels: int = PRETRAINED_LATENT_CHANNELS,
        hidden: int = 128,
        horizons: int = 7,
        out_size: int = 160,
        depth: int = 2,
        image_widths: tuple[int, int, int] = (16, 32, 64),
        zero_init_flow_head: bool = True,
        time_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.horizons = horizons
        self.out_size = out_size
        self.time_scale = validate_time_scale(time_scale)
        full, half, quarter = image_widths

        self.image_stem = _ImageStem(image_widths)
        # +1 for the normalised-time channel appended per horizon.
        self.latent_stem = nn.Conv3d(2 * latent_channels + 1, hidden, 3, padding=1)
        self.latent_blocks = nn.Sequential(*[Residual3DBlock(hidden) for _ in range(depth)])

        self.fuse_latent = nn.Sequential(
            nn.Conv2d(hidden + quarter, hidden, 3, padding=1),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.SiLU(),
        )
        self.up_half = nn.Sequential(
            nn.Conv2d(hidden + half, hidden // 2, 3, padding=1),
            nn.GroupNorm(min(32, hidden // 2), hidden // 2),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, hidden // 2, 3, padding=1),
            nn.GroupNorm(min(32, hidden // 2), hidden // 2),
            nn.SiLU(),
        )
        self.up_full = nn.Sequential(
            nn.Conv2d(hidden // 2 + full, hidden // 4, 3, padding=1),
            nn.GroupNorm(min(16, hidden // 4), hidden // 4),
            nn.SiLU(),
            nn.Conv2d(hidden // 4, hidden // 4, 3, padding=1),
            nn.GroupNorm(min(16, hidden // 4), hidden // 4),
            nn.SiLU(),
        )

        self.flow_head = nn.Conv2d(hidden // 4, 2, 3, padding=1)
        self.gate_head = nn.Conv2d(hidden // 4, 1, 3, padding=1)
        self.visibility_head = nn.Conv2d(hidden // 4, 1, 3, padding=1)
        self.object_mask_head = nn.Sequential(
            nn.Conv2d(latent_channels + full, full, 3, padding=1),
            nn.GroupNorm(min(8, full), full),
            nn.SiLU(),
            nn.Conv2d(full, 1, 3, padding=1),
        )

        if zero_init_flow_head:
            # Step 0 must reproduce the zero-flow baseline exactly, so that any
            # movement away from it is something the model learned.
            nn.init.zeros_(self.flow_head.weight)
            nn.init.zeros_(self.flow_head.bias)

    def forward(
        self,
        x0_latent: torch.Tensor,
        image_cond_latent: torch.Tensor,
        first_frame: torch.Tensor,
        horizon_seconds: HorizonSeconds | None = None,
        instance_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del instance_masks  # dense decoder has no slots to bind them to
        """Args:
            x0_latent: ``[B, C, T, h, w]`` with ``T = horizons + 1`` (slot 0 is t=0).
            image_cond_latent: ``[B, C, h, w]`` or ``[B, C, T, h, w]``.
            first_frame: ``[B, 3, out_size, out_size]`` in ``[-1, 1]``.
        """
        batch_size, channels, frames, height, width = x0_latent.shape
        if frames != self.horizons + 1:
            raise ValueError(
                f"expected {self.horizons + 1} latent frames, got {frames}"
            )
        if image_cond_latent.ndim == 5:
            image_cond_latent = image_cond_latent[:, :, 0]

        future = x0_latent[:, :, 1:]
        condition = image_cond_latent[:, :, None].expand(-1, -1, self.horizons, -1, -1)
        times = normalized_horizon_times(
            horizon_seconds, batch_size=batch_size, horizons=self.horizons,
            time_scale=self.time_scale, device=x0_latent.device, dtype=x0_latent.dtype,
        )
        time_channel = times.view(batch_size, 1, self.horizons, 1, 1).expand(
            batch_size, 1, self.horizons, height, width
        )

        latent = self.latent_stem(torch.cat((future, condition, time_channel), dim=1))
        latent = self.latent_blocks(latent)
        latent = latent.permute(0, 2, 1, 3, 4).reshape(batch_size * self.horizons, -1, height, width)

        levels = self.image_stem(first_frame)
        level_full, level_half, level_quarter = levels

        def repeat(value: torch.Tensor) -> torch.Tensor:
            return value[:, None].expand(-1, self.horizons, -1, -1, -1).reshape(
                batch_size * self.horizons, *value.shape[1:]
            )

        guide = F.adaptive_avg_pool2d(level_quarter, (height, width))
        feature = self.fuse_latent(torch.cat((latent, repeat(guide)), dim=1))

        half_size = self.out_size // 2
        feature = F.interpolate(feature, size=(half_size, half_size), mode="bilinear", align_corners=False)
        feature = self.up_half(torch.cat((feature, repeat(level_half)), dim=1))

        feature = F.interpolate(
            feature, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
        )
        feature = self.up_full(torch.cat((feature, repeat(level_full)), dim=1))

        def unflatten(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch_size, self.horizons, *value.shape[1:])

        flow_raw = unflatten(self.flow_head(feature))
        gate_logits = unflatten(self.gate_head(feature))
        visibility_logits = unflatten(self.visibility_head(feature))

        mask_input = torch.cat(
            (
                F.interpolate(
                    image_cond_latent,
                    size=(self.out_size, self.out_size),
                    mode="bilinear",
                    align_corners=False,
                ),
                level_full,
            ),
            dim=1,
        )
        object_mask_logits = self.object_mask_head(mask_input)

        flow = flow_raw * torch.sigmoid(gate_logits)
        return {
            "flow": flow,
            "flow_raw": flow_raw,
            "gate_logits": gate_logits,
            "visibility_logits": visibility_logits,
            "object_mask_logits": object_mask_logits,
        }


def apply_shrinkage(flow: torch.Tensor, tau_normalized: float) -> torch.Tensor:
    """Soft-threshold the predicted flow toward zero.

    A statistically free win on this metric: where the truth is zero (most
    scored pixels) it strictly reduces error, and where the displacement is well
    above ``tau`` it is nearly lossless.  ``tau`` is tuned on val only.
    """
    if tau_normalized <= 0:
        return flow
    magnitude = flow.square().sum(dim=-3, keepdim=True).sqrt()
    scale = ((magnitude - tau_normalized) / magnitude.clamp_min(1e-8)).clamp(0.0, 1.0)
    return flow * scale


def build_decoder(config: dict, derived: dict) -> nn.Module:
    """Instantiate the decoder named by ``model.decoder.type``.

    ``dense`` is :class:`LatentFlowDecoder`; ``slots`` is the structured
    per-object affine decoder.  Both expose the same forward signature and the
    same output keys, so every other component -- backend, sampler, losses,
    evaluator, Stage-2 bridge -- is shared and the two can be compared with
    nothing else changed.
    """
    decoder_config = config["model"]["decoder"]
    # Also validate callers that construct directly without load_config (e.g.
    # inference after restoring the checkpoint's complete decoder_config).
    from .flow_dit_config import (
        validate_dense_correction_config, validate_motion_activity_config,
        validate_dense_motion_mask_config, validate_motion_vector_loss_config,
    )

    validate_dense_correction_config(config)
    validate_motion_activity_config(config)
    validate_dense_motion_mask_config(config)
    validate_motion_vector_loss_config(config)
    kind = str(decoder_config.get("type", "dense"))
    common = dict(
        latent_channels=derived["latent_channels"],
        hidden=int(decoder_config["hidden"]),
        horizons=derived["horizons"],
        out_size=int(config["data"]["height"]),
        depth=int(decoder_config["depth"]),
        image_widths=tuple(decoder_config["image_widths"]),
        zero_init_flow_head=bool(decoder_config["zero_init_flow_head"]),
        time_scale=config["data"].get("time_scale_seconds"),
    )
    if kind == "dense":
        return LatentFlowDecoder(**common)
    if kind == "slots":
        from .flow_dit_slots import SlotTransformDecoder

        return SlotTransformDecoder(
            **common,
            num_slots=int(decoder_config.get("num_slots", 16)),
            slot_hidden=int(decoder_config.get("slot_hidden", 256)),
            residual_branch=bool(decoder_config.get("residual_branch", True)),
            residual_scale=float(decoder_config.get("residual_scale", 0.05)),
            dense_correction=bool(decoder_config.get("dense_correction", False)),
            dense_correction_gain=float(decoder_config.get("dense_correction_gain", 0.05)),
            motion_activity=bool(decoder_config.get("motion_activity", False)),
            motion_activity_gate=bool(decoder_config.get("motion_activity_gate", False)),
            dense_motion_mask=bool(decoder_config.get("dense_motion_mask", False)),
            task_object_slot_activity=bool(
                decoder_config.get("task_object_slot_activity", False)
            ),
            task_object_gate_mode=str(decoder_config.get("task_object_gate_mode", "soft")),
            task_object_gate_threshold=float(decoder_config.get("task_object_gate_threshold", .5)),
            task_object_gate_source=str(decoder_config.get("task_object_gate_source", "activity")),
            task_object_dense_rescue=bool(decoder_config.get("task_object_dense_rescue", False)),
            task_object_dense_rescue_threshold=float(decoder_config.get("task_object_dense_rescue_threshold", .9)),
            task_object_identity=bool(
                decoder_config.get("task_object_identity", False)
            ),
            task_object_identity_semantic=bool(
                decoder_config.get("task_object_identity_semantic", False)
            ),
            task_object_identity_token_attention=bool(decoder_config.get("task_object_identity_token_attention", False)),
            prompt_dim=int(decoder_config.get("prompt_dim", 4096)),
            text_slot_binding=bool(decoder_config.get("text_slot_binding", False)),
            text_slot_binding_heads=int(decoder_config.get("text_slot_binding_heads", 4)),
            text_slot_binding_max_gain=float(decoder_config.get("text_slot_binding_max_gain", 2.0)),
            text_slot_fusion=bool(decoder_config.get("text_slot_fusion", False)),
            slot_pooling_mode=str(decoder_config.get("slot_pooling_mode", "bilinear")),
            task_object_heads_detach_shared=bool(
                decoder_config.get("task_object_heads_detach_shared", False)
            ),
            dense_motion_mask_detach_shared=bool(
                decoder_config.get("dense_motion_mask_detach_shared", False)
            ),
            dense_motion_mask_shared_gradient_scale=float(
                decoder_config.get("dense_motion_mask_shared_gradient_scale", 1.0)
            ),
            visibility_from_motion=bool(decoder_config.get("visibility_from_motion", True)),
            temporal_slots=bool(decoder_config.get("temporal_slots", True)),
            causal_slots=bool(decoder_config.get("causal_slots", False)),
            incremental_params=bool(decoder_config.get("incremental_params", False)),
            oracle_instance_masks=bool(decoder_config.get("oracle_instance_masks", False)),
        )
    raise ValueError(f"unknown decoder type {kind!r}; expected 'dense' or 'slots'")


def count_trainable(module_or_backend: Any) -> int:
    if isinstance(module_or_backend, WanFlowDiTBackend):
        return sum(p.numel() for p in module_or_backend.trainable_parameters())
    return sum(p.numel() for p in module_or_backend.parameters() if p.requires_grad)
