"""Invertible optical-flow <-> image codecs for Stage-1 flow diffusion.

Stage-1 treats the first-to-future flow sequence as a short "video" so that the
frozen Wan video prior sees in-distribution activations.  To do that the flow
field has to become a 3-channel image in ``[-1, 1]``.

Design notes (each choice is meant to be checked by a codec calibration run):

* Compression is **radial** by default: only the magnitude is warped, the
  direction is carried verbatim in Cartesian ``(x, y)``.  A polar/HSV encoding
  would wrap hue at +-pi, and a reconstruction error across that seam turns into
  a ~180 degree angular error.
* The warp is a power law with ``gamma = 0.5``.  Physion flow is dominated by
  static pixels (most scored pixels move less than 1px), so the codec must
  spend its resolution near zero: at 1px the round-trip sensitivity is
  ``d|f| = 2 * M * s * ds`` for a code step ``ds``, several times finer than a
  linear code with the same clip.  ``tanh`` fails the opposite way -- it
  saturates and destroys the large displacements.
* Everything is expressed in **normalized** units (fraction of frame width or
  height) relative to the 160x160 evaluation grid, so the codec is independent
  of whatever spatial resolution the VAE happens to run at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F


EVAL_SIDE = 160.0

RADIAL_POWER = "radial_power"
SIGNED_POWER = "signed_power"
RADIAL_LOG = "radial_log"
LINEAR = "linear"

CODEC_KINDS = (RADIAL_POWER, SIGNED_POWER, RADIAL_LOG, LINEAR)

_EPSILON = 1e-8


def _check_kind(kind: str) -> None:
    if kind not in CODEC_KINDS:
        raise ValueError(f"unknown codec kind {kind!r}; expected one of {CODEC_KINDS}")


def compress(
    flow_normalized: torch.Tensor,
    *,
    max_normalized: float,
    gamma: float,
    kind: str = RADIAL_POWER,
) -> torch.Tensor:
    """Map normalized flow to ``[-1, 1]``.

    Args:
        flow_normalized: ``[..., 2, H, W]`` flow as a fraction of frame size.
        max_normalized: clip radius in the same normalized units.
        gamma: power-law exponent; 0.5 gives sub-pixel resolution near zero.
        kind: one of :data:`CODEC_KINDS`.
    """
    _check_kind(kind)
    value = flow_normalized.float()
    if kind == SIGNED_POWER:
        scaled = (value.abs() / max_normalized).clamp(0.0, 1.0)
        return torch.sign(value) * scaled.pow(gamma)
    magnitude = value.square().sum(dim=-3, keepdim=True).sqrt()
    scaled = (magnitude / max_normalized).clamp(0.0, 1.0)
    if kind == RADIAL_POWER:
        warped = scaled.pow(gamma)
    elif kind == RADIAL_LOG:
        warped = torch.log1p(scaled / gamma) / torch.log1p(torch.tensor(1.0 / gamma))
    else:  # LINEAR
        warped = scaled
    # Direction is preserved exactly; only the radius is warped.
    return value / magnitude.clamp_min(_EPSILON) * warped


def decompress(
    code: torch.Tensor,
    *,
    max_normalized: float,
    gamma: float,
    kind: str = RADIAL_POWER,
) -> torch.Tensor:
    """Exact inverse of :func:`compress` inside the clip radius."""
    _check_kind(kind)
    value = code.float()
    if kind == SIGNED_POWER:
        return torch.sign(value) * value.abs().clamp(0.0, 1.0).pow(1.0 / gamma) * max_normalized
    warped = value.square().sum(dim=-3, keepdim=True).sqrt().clamp(0.0, 1.0)
    if kind == RADIAL_POWER:
        scaled = warped.pow(1.0 / gamma)
    elif kind == RADIAL_LOG:
        scaled = torch.expm1(warped * torch.log1p(torch.tensor(1.0 / gamma))) * gamma
    else:  # LINEAR
        scaled = warped
    return value / warped.clamp_min(_EPSILON) * scaled * max_normalized


def clip_fraction(
    flow_pixels: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_pixels: float,
) -> float:
    """Fraction of scored pixels whose displacement exceeds the clip radius.

    The calibration gate rejects a ``max_pixels`` that clips more than 1% of the
    pixels that :func:`motion_jepa.metrics.sample_metrics` actually scores.
    """
    clipped, total = clip_counts(flow_pixels, valid, max_pixels=max_pixels)
    return clipped / total if total else 0.0


def clip_counts(
    flow_pixels: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_pixels: float,
) -> tuple[float, float]:
    """``(clipped_pixels, scored_pixels)`` so callers can aggregate correctly.

    Averaging per-sample fractions over-weights samples with few scored pixels;
    the honest number is pixel-weighted across the whole set.
    """
    magnitude = flow_pixels.float().square().sum(dim=-3).sqrt()
    # ``valid`` follows the metrics convention of keeping a singleton channel.
    if valid.ndim == magnitude.ndim + 1:
        valid = valid[..., 0, :, :]
    selected = magnitude[valid.expand_as(magnitude)]
    if selected.numel() == 0:
        return 0.0, 0.0
    return float((selected > max_pixels).sum()), float(selected.numel())


def prepare_visibility(
    visibility: torch.Tensor,
    object_mask: torch.Tensor,
) -> torch.Tensor:
    """Force background visibility to 1 so the encoded image has a flat backdrop.

    Mirrors ``source_valid`` in ``scripts/cache_stage2_oracle.py``: outside
    the object the flow is exactly zero and the pixel is trivially "visible".
    A constant ``(0, 0, +1)`` background is the easiest thing for a VAE to
    reconstruct; leaving noise there would spend reconstruction budget on
    pixels that are never scored.

    Args:
        visibility: ``[T, 1, H, W]``
        object_mask: ``[1, H, W]``
    """
    inside = object_mask.float().clamp(0.0, 1.0)[None]
    return (1.0 - inside) + inside * visibility.float()


def flow_to_image(
    flow_pixels: torch.Tensor,
    visibility: torch.Tensor,
    *,
    size: int,
    max_pixels: float,
    gamma: float,
    kind: str = RADIAL_POWER,
    eval_side: float = EVAL_SIDE,
) -> torch.Tensor:
    """Encode a flow sequence as a ``[-1, 1]`` image sequence.

    Args:
        flow_pixels: ``[T, 2, H, W]`` first-to-future flow in pixels on the
            ``H x W`` grid (the Stage-1 cache stores 160x160).
        visibility: ``[T, 1, H, W]`` in ``[0, 1]``; pass the output of
            :func:`prepare_visibility`.
        size: side length of the encoded image (the VAE input resolution).
    Returns:
        ``[T, 3, size, size]`` with channels ``(code_x, code_y, 2*vis - 1)``.
    """
    if flow_pixels.ndim != 4 or flow_pixels.shape[1] != 2:
        raise ValueError(f"flow must be [T,2,H,W], got {tuple(flow_pixels.shape)}")
    if visibility.shape[0] != flow_pixels.shape[0] or visibility.shape[1] != 1:
        raise ValueError(f"visibility must be [T,1,H,W], got {tuple(visibility.shape)}")
    height, width = flow_pixels.shape[-2:]
    scale = flow_pixels.new_tensor([width, height]).view(1, 2, 1, 1)
    normalized = flow_pixels.float() / scale
    if (height, width) != (size, size):
        normalized = F.interpolate(
            normalized, size=(size, size), mode="bilinear", align_corners=False
        )
        visibility = F.interpolate(
            visibility.float(), size=(size, size), mode="bilinear", align_corners=False
        )
    max_normalized = max_pixels / eval_side
    code = compress(normalized, max_normalized=max_normalized, gamma=gamma, kind=kind)
    return torch.cat((code, visibility.float() * 2.0 - 1.0), dim=1).clamp(-1.0, 1.0)


def image_to_flow(
    image: torch.Tensor,
    *,
    out_size: int,
    max_pixels: float,
    gamma: float,
    kind: str = RADIAL_POWER,
    eval_side: float = EVAL_SIDE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`flow_to_image`.

    Returns:
        ``(flow_pixels [T,2,out_size,out_size], visibility_prob [T,1,...])``
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must be [T,3,S,S], got {tuple(image.shape)}")
    code = image[:, :2].float().clamp(-1.0, 1.0)
    visibility = ((image[:, 2:3].float() + 1.0) * 0.5).clamp(0.0, 1.0)
    max_normalized = max_pixels / eval_side
    normalized = decompress(code, max_normalized=max_normalized, gamma=gamma, kind=kind)
    if normalized.shape[-1] != out_size or normalized.shape[-2] != out_size:
        normalized = F.interpolate(
            normalized, size=(out_size, out_size), mode="bilinear", align_corners=False
        )
        visibility = F.interpolate(
            visibility, size=(out_size, out_size), mode="bilinear", align_corners=False
        )
    scale = normalized.new_tensor([out_size, out_size]).view(1, 2, 1, 1)
    return normalized * scale, visibility


def prepend_zero_slot(flow_pixels: torch.Tensor, visibility: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepend the known ``t = 0`` slot (zero flow, fully visible).

    Wan's image-to-video layout marks latent frame 0 as given
    (``mask[:, :, 0] = 1`` in ``motion_jepa/stage2.py:233``).  For a flow video
    that slot is not a hack: the first-to-first flow *is* identically zero, so
    the conditioning semantics stay honest.
    """
    zero_flow = flow_pixels.new_zeros(1, *flow_pixels.shape[1:])
    ones = visibility.new_ones(1, *visibility.shape[1:])
    return torch.cat((zero_flow, flow_pixels), dim=0), torch.cat((ones, visibility), dim=0)


class WanVaeCodec:
    """Track A: encode the flow image with the frozen Wan video VAE.

    The point is not compression -- it is that a flow image pushed through the
    RGB VAE lands on the RGB latent manifold, so the 40 frozen DiT blocks see
    in-distribution activations.  Frames are encoded independently (``T = 1``),
    which is the same path Wan uses for its i2v conditioning image, and which
    keeps the 7 horizons from being blended by the VAE's 4x temporal
    compression.
    """

    latent_channels = 16
    spatial_ratio = 8

    def __init__(self, vae, *, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.vae = vae
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "WanVaeCodec":
        from diffusers import AutoencoderKLWan

        vae = (
            AutoencoderKLWan.from_pretrained(Path(model_dir) / "vae", torch_dtype=dtype)
            .to(device)
            .eval()
        )
        vae.requires_grad_(False)
        return cls(vae, device=device, dtype=dtype)

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """``[T, 3, S, S]`` in ``[-1, 1]`` -> ``[16, T, S/8, S/8]`` raw VAE latents."""
        frames = image.to(self.device, self.dtype).unsqueeze(2)  # [T,3,1,S,S]
        latent = self.vae.encode(frames).latent_dist.mode()  # [T,16,1,h,w]
        return latent.squeeze(2).permute(1, 0, 2, 3).contiguous()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """``[16, T, h, w]`` -> ``[T, 3, S, S]``."""
        frames = latent.to(self.device, self.dtype).permute(1, 0, 2, 3).unsqueeze(2)
        decoded = self.vae.decode(frames).sample  # [T,3,1,S,S]
        return decoded.squeeze(2).clamp(-1.0, 1.0)


class DirectCodec:
    """Track B fallback: no VAE at all -- diffuse the flow image itself.

    Chosen only if codec calibration shows the VAE round-trip eats too much of
    the error budget.  It removes the round-trip floor entirely
    but hands the frozen backbone out-of-distribution activations, so the
    per-channel standardisation below (to roughly unit variance) matters: flow
    is mostly exact zeros and would otherwise sit far from the unit-variance
    assumption baked into the flow-matching sigma schedule.
    """

    def __init__(
        self,
        *,
        latent_size: int,
        channel_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        device: torch.device | None = None,
    ) -> None:
        self.latent_size = int(latent_size)
        self.latent_channels = 3
        self.device = device or torch.device("cpu")
        self.channel_scale = torch.tensor(channel_scale).view(3, 1, 1, 1)

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        frames = image.to(self.device).float()
        if frames.shape[-1] != self.latent_size:
            frames = F.interpolate(
                frames, size=(self.latent_size, self.latent_size), mode="area"
            )
        latent = frames.permute(1, 0, 2, 3).contiguous()
        return latent * self.channel_scale.to(latent.device)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        value = latent.to(self.device).float() / self.channel_scale.to(latent.device)
        return value.permute(1, 0, 2, 3).contiguous().clamp(-1.0, 1.0)


def build_codec(
    config: dict,
    *,
    model_dir: str | Path,
    device: torch.device,
    latent_size: int,
) -> WanVaeCodec | DirectCodec:
    """Instantiate the codec named by ``config['backend']`` (``vae`` or ``direct``)."""
    backend = str(config.get("backend", "vae"))
    if backend == "vae":
        return WanVaeCodec.from_model_dir(model_dir, device=device)
    if backend == "direct":
        return DirectCodec(
            latent_size=latent_size,
            channel_scale=tuple(config.get("channel_scale", (1.0, 1.0, 1.0))),
            device=device,
        )
    raise ValueError(f"unknown codec backend {backend!r}; expected 'vae' or 'direct'")


def codec_params(config: dict) -> dict:
    """Extract the compression knobs so callers cannot silently disagree."""
    return {
        "max_pixels": float(config.get("max_px", 64.0)),
        "gamma": float(config.get("gamma", 0.5)),
        "kind": str(config.get("kind", RADIAL_POWER)),
    }
