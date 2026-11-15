from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LORA_TARGETS = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
]


SCENARIO_PROMPTS = {
    "collide": "Two objects collide and respond naturally to the impact. Static camera.",
    "contain": "An object moves into or interacts with a container. Static camera.",
    "dominoes": "A chain of domino-like objects falls through contact. Static camera.",
    "drape": "A flexible object falls and drapes over another object. Static camera.",
    "drop": "An object drops under gravity and interacts with the surface below. Static camera.",
    "link": "Linked objects move while respecting their physical connection. Static camera.",
    "roll": "An object rolls naturally along a surface. Static camera.",
    "support": "Objects move while respecting support and gravity. Static camera.",
}


@dataclass
class Stage2Batch:
    video_latent: torch.Tensor
    image_cond_latent: torch.Tensor
    prompt_embeds: torch.Tensor
    motion_condition: torch.Tensor
    fm_weight: torch.Tensor | None = None        # [B, T, h, w] flow-matching weights (train_step_v2), mean 1


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(32, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.silu(self.norm1(value)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return value + residual


class MotionAdapter(nn.Module):
    """Map warped-first-frame and oracle-flow maps to Wan token residuals.

    Two injection styles, selected by ``interaction``:

    * ``False`` -- ``h' = h + c``. The residual is a fixed offset: it does not
      depend on what the transformer is currently holding, so the same nudge is
      applied at every denoising step regardless of the content.
    * ``True``  -- ``h' = h + Linear(c + h)``, the form OnlyFlow uses
      (arXiv:2411.10501). Letting the control signal see the hidden state lets
      the injection modulate itself against the content instead of overriding it.

    ``False`` is the default so checkpoints trained before this existed keep
    loading and behaving identically.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        transformer_dim: int,
        injection_blocks: list[int],
        depth: int = 3,
        interaction: bool = False,
    ) -> None:
        super().__init__()
        self.injection_blocks = list(injection_blocks)
        self.interaction = interaction
        self.stem = nn.Conv3d(in_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(*[Residual3DBlock(hidden_channels) for _ in range(depth)])
        self.projections = nn.ModuleList(
            nn.Conv3d(hidden_channels, transformer_dim, 1) for _ in self.injection_blocks
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        if interaction:
            # Zero-init here too, so step 0 is exactly the unconditioned model
            # and the adapter can only earn influence.
            self.mixers = nn.ModuleList(
                nn.Linear(transformer_dim, transformer_dim) for _ in self.injection_blocks
            )
            for mixer in self.mixers:
                nn.init.zeros_(mixer.weight)
                nn.init.zeros_(mixer.bias)

    def forward(self, condition: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.blocks(self.stem(condition))
        # Wan's patch embedding downsamples the VAE latent grid by 2x spatially.
        hidden = F.avg_pool3d(hidden, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        residuals = []
        for projection in self.projections:
            value = projection(hidden).permute(0, 2, 3, 4, 1).contiguous()
            residuals.append(value.flatten(1, 3))
        return residuals

    def mix(self, index: int, control: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """The residual actually added at injection point ``index``."""
        if not self.interaction:
            return control
        return self.mixers[index](control + hidden)


class _TemporalAttention(nn.Module):
    """Attention along the horizon axis at each spatial position.

    OnlyFlow interleaves temporal attention with the ResNet blocks of its flow
    encoder so the control signal is aware of how the motion evolves, not just
    what it looks like at one instant. With eight horizons this is cheap.
    """

    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = value.shape
        normed = self.norm(value)
        sequence = normed.permute(0, 3, 4, 2, 1).reshape(batch * height * width, frames, channels)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        attended = attended.reshape(batch, height, width, frames, channels).permute(0, 4, 3, 1, 2)
        return value + attended


class OnlyFlowMotionAdapter(nn.Module):
    """Motion conditioning in the style of OnlyFlow (arXiv:2411.10501).

    Three things it does that the original MotionAdapter did not:

    * a multi-scale encoder -- ResNet blocks interleaved with temporal attention
      at three spatial scales, fused back to the token grid -- instead of a
      single-scale stack;
    * injection at every transformer block rather than three of them, which is
      the translation of OnlyFlow's "all temporal attention layers" to a DiT
      where every block attends over space and time together;
    * ``h + gamma * up(SiLU(down(c + h)))`` per block.

    The per-block transform is low rank by necessity. OnlyFlow's UNet is
    320-1280 channels wide, so a full Linear per layer costs at most 1.6M; Wan
    is 5120 wide, where the same thing costs 26M per layer and 1.05B over 40
    layers -- seven times the LoRA, trained from scratch on far less data.
    Rank 128 keeps the per-layer structure at 52M total.

    ``up`` is zero-initialised, so step 0 is exactly the unconditioned model.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        transformer_dim: int,
        injection_blocks: list[int],
        depth: int = 2,
        levels: int = 3,
        heads: int = 8,
        mixer_rank: int = 128,
    ) -> None:
        super().__init__()
        self.injection_blocks = list(injection_blocks)
        self.interaction = True
        self.stem = nn.Conv3d(in_channels, hidden_channels, 3, padding=1)

        self.stages = nn.ModuleList()
        for _ in range(levels):
            stage = nn.ModuleList(
                [Residual3DBlock(hidden_channels) for _ in range(depth)]
                + [_TemporalAttention(hidden_channels, heads)]
            )
            self.stages.append(stage)
        self.fuse = nn.Conv3d(hidden_channels * levels, hidden_channels, 1)
        self.to_token = nn.Conv3d(hidden_channels, transformer_dim, 1)

        self.down = nn.ModuleList(
            nn.Linear(transformer_dim, mixer_rank) for _ in self.injection_blocks
        )
        self.up = nn.ModuleList(
            nn.Linear(mixer_rank, transformer_dim) for _ in self.injection_blocks
        )
        for layer in self.up:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def embed(self, condition: torch.Tensor) -> torch.Tensor:
        """Condition -> stem features; subclasses that read more channels override this."""
        return self.stem(condition)

    def forward(self, condition: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.embed(condition)
        _, _, _, height, width = hidden.shape
        # Wan's patch embedding halves the spatial grid, so the control features
        # have to live on that grid for the residual to line up token for token.
        token_size = (height // 2, width // 2)

        features, current = [], hidden
        for index, stage in enumerate(self.stages):
            for layer in stage:
                current = layer(current)
            features.append(F.interpolate(
                current, size=(current.shape[2], *token_size), mode="nearest",
            ))
            if index + 1 < len(self.stages):
                current = F.avg_pool3d(current, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        control = self.to_token(self.fuse(torch.cat(features, dim=1)))
        flat = control.permute(0, 2, 3, 4, 1).contiguous().flatten(1, 3)
        return [flat for _ in self.injection_blocks]

    def mix(self, index: int, control: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.up[index](F.silu(self.down[index](control + hidden)))


class ExtendedOnlyFlowAdapter(OnlyFlowMotionAdapter):
    """OnlyFlow adapter that also reads the extra channels of the v2 condition (motion_jepa.cond_v2).

    The deployed adapter's stem keeps reading the first `base_channels` channels; a second, zero-initialised stem
    without bias reads the extra ones and is added to it.  Loaded from a deployed 20-channel checkpoint
    (`load_deployed`), the model is therefore bit-identical to the deployed one at step 0 whatever the extra channels
    contain, and the new rows receive gradient at once because the rest of the adapter is already trained.
    A condition with only the base channels is accepted as well.
    """

    def __init__(self, base_channels: int, extra_channels: int, hidden_channels: int, transformer_dim: int,
                 injection_blocks: list[int], **kwargs) -> None:
        super().__init__(base_channels, hidden_channels, transformer_dim, injection_blocks, **kwargs)
        self.base_channels, self.extra_channels = int(base_channels), int(extra_channels)
        self.stem_extra = nn.Conv3d(self.extra_channels, hidden_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.stem_extra.weight)

    def embed(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.shape[1] == self.base_channels:
            return self.stem(condition)
        if condition.shape[1] != self.base_channels + self.extra_channels:
            raise ValueError(f"condition has {condition.shape[1]} channels, expected {self.base_channels} or "
                             f"{self.base_channels + self.extra_channels}")
        return self.stem(condition[:, : self.base_channels]) + self.stem_extra(condition[:, self.base_channels:])

    def load_deployed(self, state_dict: dict) -> None:
        """Weights of a deployed (base-channels only) adapter; the extra stem stays at zero."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if set(missing) - {"stem_extra.weight"} or unexpected:
            raise RuntimeError(f"not a deployed adapter checkpoint: missing {sorted(missing)}, unexpected {sorted(unexpected)}")


def motion_agreement_loss(latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    """Penalise a predicted latent whose motion disagrees with the condition.

    ``condition`` is ``[B, 20, T, h, w]`` laid out as
    ``[warped_first_frame_latent(16), flow_xy(2), mask(1), visibility(1)]``, with
    the flow already normalised by the latent grid. Warping frame 0 of the
    predicted latent along that flow should land on the predicted frame t, so the
    residual between them measures how far the generated content drifted from the
    motion it was handed.

    Restricted to the conditioned region: outside the mask the condition asserts
    nothing, and scoring there would penalise the background for existing.
    """
    flow = condition[:, 16:18]
    mask = condition[:, 18:19]
    batch, _, frames, height, width = flow.shape
    first = latent[:, :, :1].expand(-1, -1, frames, -1, -1)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=latent.device, dtype=latent.dtype),
        torch.linspace(-1.0, 1.0, width, device=latent.device, dtype=latent.dtype),
        indexing="ij",
    )
    # make_motion_condition splats the flow onto the *target* grid, so the value
    # at a destination pixel is the displacement that brought content there.
    # Reconstructing frame t therefore samples frame 0 at ``p - flow(p)``; using
    # ``+`` moves the content the wrong way and makes a correctly-followed
    # condition score worse than a static video.
    scale_x = 2.0 * width / max(width - 1, 1)
    scale_y = 2.0 * height / max(height - 1, 1)
    grid_x = xx[None, None] - scale_x * flow[:, 0]
    grid_y = yy[None, None] - scale_y * flow[:, 1]
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(batch * frames, height, width, 2)

    warped = F.grid_sample(
        first.permute(0, 2, 1, 3, 4).reshape(batch * frames, -1, height, width),
        grid.to(first.dtype), mode="bilinear", padding_mode="border", align_corners=True,
    ).reshape(batch, frames, -1, height, width).permute(0, 2, 1, 3, 4)

    weight = mask.expand(-1, latent.shape[1], -1, -1, -1)
    return ((warped - latent).square() * weight).sum() / weight.sum().clamp_min(1.0)


def _u_for_sigma(sigma: float, shift: float) -> float:
    return sigma / (shift - (shift - 1.0) * sigma)


def sample_expert_sigmas(
    batch_size: int,
    *,
    expert: str,
    boundary_sigma: float,
    shift: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    boundary_u = _u_for_sigma(boundary_sigma, shift)
    lower, upper = {
        "low": (0.0, boundary_u),
        "high": (boundary_u, 1.0),
    }[expert]
    uniform = lower + torch.rand(batch_size, device=device, generator=generator) * (upper - lower)
    sigma = shift * uniform / (1.0 + (shift - 1.0) * uniform)
    return sigma.clamp_(0.001, 0.999)


class OracleFlowWanBackend:
    """One A14B diffusion expert with LoRA and block-level motion injection."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        expert: str,
        injection_blocks: list[int],
        lora_rank: int,
        lora_alpha: int,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        device: torch.device,
        flow_shift: float = 5.0,
        boundary_sigma: float = 0.9,
        gradient_checkpointing: bool = True,
        target_modules: list[str] | None = None,
        train_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from diffusers import WanTransformer3DModel
        from peft import LoraConfig

        if expert not in ("low", "high"):
            raise ValueError(f"expert must be low or high, got {expert}")
        self.expert = expert
        self.device = device
        self.train_dtype = train_dtype
        self.flow_shift = float(flow_shift)
        self.boundary_sigma = float(boundary_sigma)
        subfolder = "transformer_2" if expert == "low" else "transformer"
        self.transformer = WanTransformer3DModel.from_pretrained(
            Path(model_dir) / subfolder, torch_dtype=train_dtype
        )
        self.transformer.requires_grad_(False)
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
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.transformer.train().to(device)
        self.hidden_dim = (
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )
        self.injection_blocks = list(map(int, injection_blocks))
        num_blocks = len(self.transformer.blocks)
        if len(set(self.injection_blocks)) != len(self.injection_blocks):
            raise ValueError("motion injection blocks must be unique")
        if any(block < 0 or block >= num_blocks for block in self.injection_blocks):
            raise ValueError(f"invalid injection blocks {self.injection_blocks}; model has {num_blocks}")
        self._residuals: dict[int, torch.Tensor] = {}
        # Set when the adapter mixes control with hidden state; the residual is
        # then a function of what the block receives and cannot be precomputed.
        self._mixer = None
        self._hook_handles = [
            self.transformer.blocks[block].register_forward_pre_hook(self._make_injector(block))
            for block in self.injection_blocks
        ]
        self.latents_mean = latents_mean.to(device).view(1, -1, 1, 1, 1)
        self.latents_std = latents_std.to(device).view(1, -1, 1, 1, 1)

    def _make_injector(self, block: int):
        def inject(_module, inputs):
            if block not in self._residuals:
                raise RuntimeError(f"motion residual for block {block} was not set")
            hidden = inputs[0]
            residual = self._residuals[block]
            # a mixer may carry several concatenated controls (DualAdapter); the token layout must still match
            if hidden.shape[:-1] != residual.shape[:-1] or (self._mixer is None and hidden.shape != residual.shape):
                raise RuntimeError(
                    f"motion residual shape {tuple(residual.shape)} does not match "
                    f"block-{block} hidden shape {tuple(hidden.shape)}"
                )
            residual = residual.to(hidden.dtype)
            if self._mixer is not None:
                residual = self._mixer(self.injection_blocks.index(block), residual, hidden)
            return (hidden + residual, *inputs[1:])

        return inject

    def set_motion_residuals(self, residuals: list[torch.Tensor], mixer=None) -> None:
        """``mixer(index, control, hidden) -> residual``; None keeps plain addition.

        Passing the adapter's ``mix`` keeps the mixing Linear inside the autograd
        graph, so it trains along with the rest of the adapter.
        """
        if len(residuals) != len(self.injection_blocks):
            raise ValueError("adapter residual count and injection block count differ")
        self._residuals = dict(zip(self.injection_blocks, residuals))
        self._mixer = mixer

    def normalize_latents(self, value: torch.Tensor) -> torch.Tensor:
        return (value.float() - self.latents_mean) / self.latents_std

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for name, parameter in self.transformer.named_parameters()
                if parameter.requires_grad and ".reference." not in name]

    # ------------------------------------------------------------------ frozen reference policy (L_keep, DPO)
    def add_reference(self, lora_params_path: str | Path, *, lora_rank: int, lora_alpha: int,
                      target_modules: list[str] | None = None) -> None:
        """A second, frozen LoRA ("reference") holding the weights of a starting checkpoint.

        The trained LoRA stays the "default" adapter.  `reference_velocity` switches every LoRA layer to the reference
        for one no-grad forward and back, without touching requires_grad (peft's own set_adapter would flip it)."""
        from peft import LoraConfig
        from peft.tuners.tuners_utils import BaseTunerLayer

        was_trainable = {name: parameter.requires_grad for name, parameter in self.transformer.named_parameters() if "lora_" in name}
        self.transformer.add_adapter(LoraConfig(r=lora_rank, lora_alpha=lora_alpha, init_lora_weights=True,
                                                target_modules=list(target_modules or DEFAULT_LORA_TARGETS)), adapter_name="reference")
        state = torch.load(lora_params_path, map_location="cpu", weights_only=True)
        parameters = dict(self.transformer.named_parameters())
        copied = 0
        with torch.no_grad():
            for name, tensor in state.items():
                target = name.replace(".default.", ".reference.")
                if target == name or target not in parameters:
                    raise KeyError(f"reference LoRA has no parameter for {name}")
                parameters[target].data = tensor.to(self.device, torch.float32)      # same dtype as the trained LoRA: step 0 is exact
                copied += 1
        for name, parameter in self.transformer.named_parameters():
            if ".reference." in name:
                parameter.requires_grad_(False)
            elif name in was_trainable:
                parameter.requires_grad_(was_trainable[name])     # adding an adapter activates it and freezes the others
        self._tuner_layers = [m for m in self.transformer.modules() if isinstance(m, BaseTunerLayer)]
        self._activate("default")
        self.reference_parameters = copied

    def _activate(self, name: str) -> None:
        for layer in self._tuner_layers:
            layer._active_adapter = [name]

    def active_adapters(self) -> set:
        return {tuple(layer.active_adapters) for layer in self._tuner_layers}

    @torch.no_grad()
    def reference_velocity(self, latent_input: torch.Tensor, sigmas: torch.Tensor, prompt: torch.Tensor,
                           residuals: list[torch.Tensor], mixer=None) -> torch.Tensor:
        """Velocity of the frozen reference policy on exactly this input (its own adapter residuals are passed in).

        Call it BEFORE the forward that carries gradients: the block hooks read the residuals that are set at call
        time, and under gradient checkpointing they run again during backward."""
        self._activate("reference")
        try:
            self.set_motion_residuals(residuals, mixer)
            output = self.transformer(hidden_states=latent_input, timestep=sigmas * 1000.0,
                                      encoder_hidden_states=prompt, return_dict=False)[0]
        finally:
            self._activate("default")
        return output.float()

    # ------------------------------------------------------------------ preference pairs (Diffusion-DPO, motion_jepa/dpo.py)
    def _pair_input(self, latent: torch.Tensor, image_cond_latent: torch.Tensor, sigmas: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clean = self.normalize_latents(latent.to(self.device))
        image_condition = self.normalize_latents(image_cond_latent.to(self.device))
        sigma = sigmas.view(-1, 1, 1, 1, 1)
        noisy = (1.0 - sigma) * clean + sigma * noise
        mask = clean.new_zeros(clean.shape[0], 4, *clean.shape[2:])
        mask[:, :, 0] = 1.0
        return torch.cat((noisy, mask, image_condition), dim=1).to(self.train_dtype), noise - clean

    def pair_error(self, latent: torch.Tensor, image_cond_latent: torch.Tensor, prompt_embeds: torch.Tensor, residuals: list[torch.Tensor], *,
                   mixer=None, sigmas: torch.Tensor, noise: torch.Tensor, reference: bool = False) -> torch.Tensor:
        """Per-sample flow-matching error [B] of the trained model (or, reference=True, of the frozen reference: no
        gradients) on one clip of a preference pair, at the given sigma and noise."""
        latent_input, target = self._pair_input(latent, image_cond_latent, sigmas.to(self.device), noise.to(self.device))
        prompt = prompt_embeds.to(self.device, self.train_dtype)
        if reference:
            output = self.reference_velocity(latent_input, sigmas.to(self.device), prompt, residuals, mixer)
        else:
            self.set_motion_residuals(residuals, mixer)
            output = self.transformer(hidden_states=latent_input, timestep=sigmas.to(self.device) * 1000.0,
                                      encoder_hidden_states=prompt, return_dict=False)[0].float()
        return (output - target).square().mean((1, 2, 3, 4))

    def pair_noise(self, latent: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(sigmas [B], noise) for a preference pair: both clips of the pair must use the same ones."""
        sigmas = sample_expert_sigmas(latent.shape[0], expert=self.expert, boundary_sigma=self.boundary_sigma, shift=self.flow_shift,
                                      device=self.device, generator=generator)
        return sigmas, torch.randn(latent.shape, device=self.device, dtype=torch.float32, generator=generator)

    def train_step_v2(self, batch: Stage2Batch, residuals: list[torch.Tensor], *, mixer=None,
                      sigmas: torch.Tensor | None = None, noise: torch.Tensor | None = None,
                      generator: torch.Generator | None = None, keep: dict | None = None, kin: dict | None = None) -> dict:
        """Flow matching on the real continuation with optional tube weights, frozen-reference keeping and a kinematic term.

        batch.fm_weight [B, T, h, w]   per-latent-pixel weights with mean 1 (None = plain mean)
        keep  {"weight", "mask" [B] bool (samples fed the no-hint condition), "residuals", "mixer"}:
              on those samples the velocity is also pulled to the frozen reference policy's velocity for the same input
        kin   {"weight", "alpha_disp", "sigma_min", "sigma_max", "readout" (frozen module, gradients flow through its input),
               "disp" [B, K, T, 2], "valid" [B, K, T], "masks" [B, K, h, w], "seconds" [B, T], optional scales}:
              direction / displacement of the one-step clean estimate, only for sigmas inside the validated window and
              weighted by sigma / sigma_max (d z_hat / d v = -sigma)
        sigmas / noise may be supplied (preference pairs must share them).
        Returns {"loss", "sigmas", "per_sample" [B] flow-matching error, "model_output", "target"}; terms in last_terms."""
        clean = self.normalize_latents(batch.video_latent.to(self.device))
        image_condition = self.normalize_latents(batch.image_cond_latent.to(self.device))
        prompt = batch.prompt_embeds.to(self.device, self.train_dtype)
        batch_size, _, latent_frames, height, width = clean.shape
        if sigmas is None:
            sigmas = sample_expert_sigmas(batch_size, expert=self.expert, boundary_sigma=self.boundary_sigma,
                                          shift=self.flow_shift, device=self.device, generator=generator)
        sigmas = sigmas.to(self.device)
        sigma = sigmas.view(batch_size, 1, 1, 1, 1)
        if noise is None:
            noise = torch.randn(clean.shape, device=self.device, dtype=torch.float32, generator=generator)
        noise = noise.to(self.device)
        noisy = (1.0 - sigma) * clean + sigma * noise
        mask = clean.new_zeros(batch_size, 4, latent_frames, height, width)
        mask[:, :, 0] = 1.0
        latent_input = torch.cat((noisy, mask, image_condition), dim=1).to(self.train_dtype)
        reference = None
        keep_mask = None
        if keep is not None and keep["weight"] > 0:
            keep_mask = keep["mask"].to(self.device).bool()
            if bool(keep_mask.any()):
                reference = self.reference_velocity(latent_input, sigmas, prompt, keep["residuals"], keep.get("mixer"))
        self.set_motion_residuals(residuals, mixer)
        model_output = self.transformer(hidden_states=latent_input, timestep=sigmas * 1000.0,
                                        encoder_hidden_states=prompt, return_dict=False)[0].float()
        target = noise - clean
        per_pixel = (model_output - target).square().mean(1)                 # [B, T, h, w]
        plain = per_pixel.mean((1, 2, 3))
        if batch.fm_weight is not None:
            weight = batch.fm_weight.to(self.device, per_pixel.dtype)
            per_sample = (per_pixel * weight).sum((1, 2, 3)) / weight.sum((1, 2, 3))
        else:
            per_sample = plain
        flow_matching = per_sample.mean()
        total = flow_matching
        self.last_terms = {"flow_matching": float(flow_matching.detach()), "flow_matching_plain": float(plain.mean().detach())}
        if reference is not None:
            difference = (model_output - reference).square().mean((1, 2, 3, 4))
            keeping = (difference * keep_mask).sum() / keep_mask.sum()
            self.last_terms["keep"] = float(keeping.detach())
            total = total + float(keep["weight"]) * keeping
        if kin is not None and kin["weight"] > 0:
            window = ((sigmas >= kin["sigma_min"]) & (sigmas <= kin["sigma_max"])).float() * sigmas / kin["sigma_max"]
            self.last_terms["kin_active"] = float((window > 0).float().mean())
            if bool((window > 0).any()):
                from motion_jepa.kinematics import kinematic_losses
                predicted = noisy - sigma * model_output                         # one-step clean estimate (normalised latent)
                seconds = kin["seconds"].to(self.device)
                read = kin["readout"](z0=clean[:, :, :1], z=predicted, seconds=seconds, masks=kin["masks"].to(self.device), sigma=sigmas)
                terms = kinematic_losses(read, kin["disp"].to(self.device), seconds, kin["valid"].to(self.device),
                                         disp_scale=float(kin.get("disp_scale", 32.0)), epsilon_speed=float(kin.get("epsilon_speed", 4.0)))
                kinematic = (window * (terms["direction"] + float(kin["alpha_disp"]) * terms["displacement"])).mean()
                self.last_terms.update(kin_direction=float(terms["direction"].mean().detach()),
                                       kin_displacement=float(terms["displacement"].mean().detach()), kin=float(kinematic.detach()))
                total = total + float(kin["weight"]) * kinematic
        return {"loss": total, "sigmas": sigmas, "per_sample": per_sample, "model_output": model_output, "target": target}

    def train_step(
        self,
        batch: Stage2Batch,
        residuals: list[torch.Tensor],
        *,
        generator: torch.Generator | None = None,
        mixer=None,
        condition_weight: float = 0.0,
        agreement_condition: torch.Tensor | None = None,
        fed_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``agreement_condition`` lets the agreement term be measured against a different
        condition than the one the adapter was fed: when the model is conditioned on a
        Stage-1 prediction, the ground-truth condition is the right reference, otherwise
        the term would teach the model to reproduce the prediction's mistakes."""
        self.set_motion_residuals(residuals, mixer)
        clean = self.normalize_latents(batch.video_latent.to(self.device))
        image_condition = self.normalize_latents(batch.image_cond_latent.to(self.device))
        prompt = batch.prompt_embeds.to(self.device, self.train_dtype)
        batch_size, _, latent_frames, height, width = clean.shape
        sigmas = sample_expert_sigmas(
            batch_size,
            expert=self.expert,
            boundary_sigma=self.boundary_sigma,
            shift=self.flow_shift,
            device=self.device,
            generator=generator,
        )
        sigma = sigmas.view(batch_size, 1, 1, 1, 1)
        noise = torch.randn(clean.shape, device=self.device, dtype=torch.float32, generator=generator)
        noisy = (1.0 - sigma) * clean + sigma * noise
        mask = clean.new_zeros(batch_size, 4, latent_frames, height, width)
        mask[:, :, 0] = 1.0
        latent_input = torch.cat((noisy, mask, image_condition), dim=1).to(self.train_dtype)
        model_output = self.transformer(
            hidden_states=latent_input,
            timestep=sigmas * 1000.0,
            encoder_hidden_states=prompt,
            return_dict=False,
        )[0]
        target = noise - clean
        flow_matching = F.mse_loss(model_output.float(), target)
        self.last_terms = {"flow_matching": float(flow_matching.detach())}
        if condition_weight <= 0.0 and fed_weight <= 0.0:
            return flow_matching, sigmas

        # Nothing in the flow-matching objective asks the model to *use* the
        # motion condition: it reaches the same MSE by predicting a plausible
        # video from the first frame and the caption while ignoring the
        # condition. Without this term, the correct, zero and shuffled
        # conditions give nearly indistinguishable errors, which is what "the
        # loss never asked" looks like.
        #
        # x0 = noisy - sigma*v recovers the predicted clean latent; warping its
        # first frame by the conditioned displacement should reproduce the later
        # frames, so the residual depends on the condition's *content*, not
        # merely its presence.
        predicted = noisy - sigma * model_output.float()
        reference = batch.motion_condition if agreement_condition is None else agreement_condition
        total = flow_matching
        if condition_weight > 0.0:
            agreement = motion_agreement_loss(predicted, reference.to(self.device))
            # Logged separately: a combined scalar cannot show whether the agreement
            # term is contributing at all, and a term that is silently always zero
            # looks exactly like a term that is working.
            self.last_terms["agreement"] = float(agreement.detach())
            total = total + condition_weight * agreement
        if fed_weight > 0.0:
            # Agreement with the condition that was actually fed: with a Stage-1 prediction as
            # condition this is what keeps the model *sensitive* to its input flow, while the
            # GT term above keeps the motion correct when the prediction is wrong.
            fed = motion_agreement_loss(predicted, batch.motion_condition.to(self.device))
            self.last_terms["agreement_fed"] = float(fed.detach())
            total = total + fed_weight * fed
        return total, sigmas

    def load_lora_parameters(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        parameters = dict(self.transformer.named_parameters())
        missing = [name for name in state if name not in parameters]
        if missing:
            raise KeyError(f"LoRA parameter not found: {missing[0]}")
        with torch.no_grad():
            for name, tensor in state.items():
                parameters[name].copy_(tensor.to(parameters[name].dtype))

    def save_lora(self, output_dir: str | Path, transformer: nn.Module | None = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model = transformer if transformer is not None else self.transformer
        model.save_lora_adapter(output_dir / "lora")
        state = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if "lora_" in name and ".reference." not in name   # by name, not requires_grad: a frozen LoRA (adapter-only arm) is still part of the model
        }
        torch.save(state, output_dir / "lora_params.pt")

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()


def load_latent_statistics(model_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    from diffusers import AutoencoderKLWan

    config = AutoencoderKLWan.load_config(Path(model_dir) / "vae")
    return torch.tensor(config["latents_mean"]), torch.tensor(config["latents_std"])
