"""Latent motion readout (Phi_lat): where did each first-frame object go, read from a video LATENT.

It is the measuring instrument of the kinematic loss: applied to the one-step clean estimate of the generator, it turns
"the object should have moved 80 px to the right by now" into a gradient on the generated latent, without decoding
video or running a tracker inside the training step.

Inputs (all in the generator's normalised latent space, 8x spatial / 4x temporal compression)
  z0       [B, 16, 1, h, w]  the OBSERVED first-frame latent (never the generator's own first frame: a readout that
                             compared two predicted frames could be satisfied by a blank latent)
  z        [B, 16, T, h, w]  the video latent to measure (real, or a clean estimate)
  seconds  [B, T]            physical time of every latent frame
  masks    [B, K, h, w]      first-frame object masks (area fractions)
  sigma    [B]               noise level the estimate was made at (0 for a real latent)
Output    [B, K, T, 2]       displacement of every object from its first-frame position, work-frame pixels; index 0 is 0.

It never sees the motion hint, the caption or the ground truth.  A dense field "displacement of the content that sat
at this first-frame location" is predicted on the token grid and pooled over each object mask.

What keeps it honest is how it is trained (by a separate readout trainer): the same first frame appears with its
real continuation, frozen, played backwards and subsampled, each with the matching labels, so the motion cannot be
guessed from the first frame ("a held cube will fall") and has to be read from the latent frames themselves.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PIXEL_SCALE = 64.0          # the head predicts displacement / 64 px


def _sinusoid(value: torch.Tensor, dim: int, max_period: float = 100.0) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=value.device, dtype=torch.float32) / half)
    angles = value.float()[..., None] * frequencies * 2 * math.pi
    return torch.cat((angles.sin(), angles.cos()), -1)


class _Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(8, channels), nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))
        self.norm2, self.conv2 = nn.GroupNorm(8, channels), nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(y)))


class LatentMotionReadout(nn.Module):
    def __init__(self, latent_channels: int = 16, width: int = 128, dim: int = 256, depth: int = 4, heads: int = 8,
                 token_grid: tuple[int, int] = (16, 28)) -> None:
        super().__init__()
        self.config = dict(latent_channels=latent_channels, width=width, dim=dim, depth=depth, heads=heads, token_grid=list(token_grid))
        self.token_grid = tuple(token_grid)
        self.stem = nn.Conv3d(3 * latent_channels, width, (1, 3, 3), padding=(0, 1, 1))
        self.blocks = nn.Sequential(_Block(width), _Block(width))
        self.to_token = nn.Conv3d(width, dim, 1)
        self.position = nn.Parameter(torch.zeros(1, token_grid[0] * token_grid[1], dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.noise = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def field(self, z0: torch.Tensor, z: torch.Tensor, seconds: torch.Tensor, sigma: torch.Tensor | None = None,
              frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Dense displacement [B, T, gh, gw, 2] (px) of the content at every first-frame token location.

        frame_mask [B, T] True for real frames (padded frames are ignored by the attention)."""
        batch, _, frames, height, width = z.shape
        first = z0.expand(-1, -1, frames, -1, -1)
        features = self.blocks(self.stem(torch.cat((z, first, z - first), 1)))
        gh, gw = self.token_grid
        if (height, width) != (2 * gh, 2 * gw):
            raise ValueError(f"latent grid {(height, width)} does not match the token grid {self.token_grid}")
        tokens = self.to_token(F.avg_pool3d(features, (1, 2, 2)))                       # [B, D, T, gh, gw]
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(batch, frames, gh * gw, -1)
        dim = tokens.shape[-1]
        tokens = tokens + self.position[:, None] + self.time(_sinusoid(seconds, dim))[:, :, None]
        if sigma is not None:
            tokens = tokens + self.noise(_sinusoid(sigma, dim, max_period=10.0))[:, None, None]
        padding = None
        if frame_mask is not None:
            padding = (~frame_mask.bool())[:, :, None].expand(-1, -1, gh * gw).reshape(batch, -1)
        encoded = self.encoder(tokens.reshape(batch, frames * gh * gw, dim), src_key_padding_mask=padding)
        return self.head(self.norm(encoded)).reshape(batch, frames, gh, gw, 2) * PIXEL_SCALE

    def pool(self, field: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Object displacements [B, K, T, 2] from a dense field [B, T, gh, gw, 2] and first-frame masks [B, K, h, w]
        (mean of the field over each mask; index 0 is zero by definition)."""
        weights = F.adaptive_avg_pool2d(masks.float(), self.token_grid)                 # [B, K, gh, gw]
        pooled = torch.einsum("bkhw,bthwc->bktc", weights, field) / weights.sum((-1, -2)).clamp_min(1e-6)[:, :, None, None]
        moving = (torch.arange(field.shape[1], device=field.device) > 0).to(pooled.dtype)
        return pooled * moving[None, None, :, None]

    def forward(self, z0: torch.Tensor, z: torch.Tensor, seconds: torch.Tensor, masks: torch.Tensor,
                sigma: torch.Tensor | None = None, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.pool(self.field(z0.float(), z.float(), seconds, sigma, frame_mask), masks)


class LatentObjectMatcher(nn.Module):
    """Readout v2: FIND each first-frame object in every later latent frame instead of regressing a field.

    features   F_t = Enc(z_t; time, sigma)                       [B, D, T, h, w]  per latent frame, at latent resolution
    query      q_k = mean of the frame-0 features over the object's mask (frame 0 is the observed image latent)
    match      p_k(t) = softmax_x( <q_k, F_t(x)> / tau )          a distribution over the h x w cells
    position   expected cell centre under p_k(t), in work pixels;  displacement = position(t) - position(0)

    A static continuation gives zero by construction of the comparison (the same query matched in both frames), a
    large displacement costs nothing extra (the match is global), and an object that is not where the first frame
    suggests has to be found by its appearance.  The match distribution is also returned so that training can ask it
    to peak on the right cell (cross-entropy), which keeps the expectation from averaging two candidate places.
    Same call signature and output as LatentMotionReadout: [B, K, T, 2] work pixels, index 0 exactly zero."""

    def __init__(self, latent_channels: int = 16, width: int = 160, dim: int = 128, blocks: int = 4, cell_px: int = 8) -> None:
        super().__init__()
        self.config = dict(kind="matcher", latent_channels=latent_channels, width=width, dim=dim, blocks=blocks, cell_px=cell_px)
        self.cell_px, self.width = cell_px, width
        self.stem = nn.Conv3d(latent_channels, width, (1, 3, 3), padding=(0, 1, 1))
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.noise = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.first = nn.Parameter(torch.zeros(width))                      # marks the observed frame
        self.blocks = nn.ModuleList([_Block(width) for _ in range(blocks)])
        self.temporal = nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0))      # a frame also sees its neighbours (motion smear spans latent frames)
        nn.init.zeros_(self.temporal.weight)
        nn.init.zeros_(self.temporal.bias)
        self.to_query, self.to_key = nn.Conv3d(width, dim, 1), nn.Conv3d(width, dim, 1)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))

    def features(self, z: torch.Tensor, seconds: torch.Tensor, sigma: torch.Tensor | None) -> torch.Tensor:
        x = self.stem(z)
        x = x + self.time(_sinusoid(seconds, self.width)).permute(0, 2, 1)[:, :, :, None, None]
        if sigma is not None:
            x = x + self.noise(_sinusoid(sigma, self.width, max_period=10.0))[:, :, None, None, None]
        marker = torch.zeros(1, 1, x.shape[2], 1, 1, device=x.device, dtype=x.dtype)
        marker[:, :, 0] = 1.0
        x = x + marker * self.first[None, :, None, None, None]
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == len(self.blocks) // 2 - 1:
                x = x + self.temporal(x)
        return x

    def match(self, z0: torch.Tensor, z: torch.Tensor, seconds: torch.Tensor, masks: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        """log-probabilities [B, K, T, h * w] of where each object is in every latent frame."""
        z = torch.cat((z0.float(), z.float()[:, :, 1:]), 2)                 # frame 0 is always the observed one
        feats = self.features(z, seconds.float(), sigma)
        keys = F.normalize(self.to_key(feats), dim=1)                       # [B, D, T, h, w]
        queries = self.to_query(feats[:, :, :1])[:, :, 0]                   # [B, D, h, w]
        weights = masks.float() / masks.float().sum((-1, -2), keepdim=True).clamp_min(1e-6)        # [B, K, h, w]
        q = F.normalize(torch.einsum("bkhw,bdhw->bkd", weights, queries), dim=-1)
        corr = torch.einsum("bkd,bdthw->bkthw", q, keys) / self.log_tau.exp().clamp(0.01, 1.0)
        return F.log_softmax(corr.flatten(3), dim=-1)

    def positions(self, log_prob: torch.Tensor, latent_hw: tuple[int, int]) -> torch.Tensor:
        """Expected position [B, K, T, 2] (x, y) in work pixels."""
        height, width = latent_hw
        ys, xs = torch.meshgrid(torch.arange(height, device=log_prob.device), torch.arange(width, device=log_prob.device), indexing="ij")
        centres = torch.stack(((xs + 0.5) * self.cell_px - 0.5, (ys + 0.5) * self.cell_px - 0.5), -1).reshape(-1, 2).to(log_prob.dtype)
        return log_prob.exp() @ centres

    def forward(self, z0: torch.Tensor, z: torch.Tensor, seconds: torch.Tensor, masks: torch.Tensor,
                sigma: torch.Tensor | None = None, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        log_prob = self.match(z0, z, seconds, masks, sigma)
        position = self.positions(log_prob, tuple(z.shape[-2:]))
        return position - position[:, :, :1]


def save_readout(path: str | Path, model: nn.Module, latents_mean: torch.Tensor, latents_std: torch.Tensor, extra: dict | None = None) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), "latents_mean": latents_mean.flatten().cpu(),
                "latents_std": latents_std.flatten().cpu(), **(extra or {})}, path)


def load_readout(path: str | Path, latents_mean: torch.Tensor | None = None, latents_std: torch.Tensor | None = None) -> nn.Module:
    """The readout of a checkpoint.  When the caller's latent statistics are given they must be the ones it was trained with."""
    stored = torch.load(path, map_location="cpu", weights_only=True)
    if latents_mean is not None and not torch.allclose(stored["latents_mean"], latents_mean.flatten().cpu().float(), atol=1e-5):
        raise ValueError("the readout was trained with different latent statistics")
    if latents_std is not None and not torch.allclose(stored["latents_std"], latents_std.flatten().cpu().float(), atol=1e-5):
        raise ValueError("the readout was trained with different latent statistics")
    config = dict(stored["config"])
    if config.pop("kind", "field") == "matcher":
        model = LatentObjectMatcher(**config)
    else:
        config["token_grid"] = tuple(config["token_grid"])
        model = LatentMotionReadout(**config)
    model.load_state_dict(stored["state_dict"])
    return model
