"""Condition contract v2 of the Stage-2 motion adapter.

The adapter keeps its 20 rendered channels and gets seven more that it may learn to read:

  0-15   warped first-frame latent        |
  16-17  flow at target pixels (/W, /H)   |  rendered from the motion plan by cache_stage2_oracle.make_motion_condition,
  18     target mask Mt                   |  bit-identical to what the deployed model reads
  19     visibility weight                |
  20     known          the plan says where the object is at this time (painted on its target footprint)
  21     availability   1 = a motion plan is supplied, 0 = no hint at all (the strict-null input)
  22     selector       Stage-1 task-object score on the target footprint (metadata: it never scales anything)
  23-24  raw_flow       first-to-t displacement at SOURCE pixels (/W, /H); for Stage-1 inputs this is the decoder's own
                        dense field, not fitted and not gated (it keeps non-rigid and background motion)
  25     motion_prob    Stage-1 dense motion probability (first-frame coordinates)
  26     source_mask    M0, union of the first-frame object masks

Four conditions exist per clip ("kinds"):
  real   plan measured on the real continuation (training only)
  pred   plan of the frozen Stage-1 model from (first frame, caption): what deployment supplies
  zero   the same objects explicitly at rest (availability 1, known 1, zero flow)
  null   no hint: channels 0-15 hold the first-frame latent, everything else is 0

"unknown" (known = 0: nothing is said, the position shown in channels 0-18 is only held) is different from "static"
(known = 1 with zero displacement) and from "null" (availability = 0).
"""
from __future__ import annotations

import torch

BASE_CHANNELS = 20
EXTRA_NAMES = ("known", "availability", "selector", "raw_flow_x", "raw_flow_y", "motion_prob", "source_mask")
EXTRA_CHANNELS = len(EXTRA_NAMES)
V2_CHANNELS = BASE_CHANNELS + EXTRA_CHANNELS
CHANNEL = {name: BASE_CHANNELS + index for index, name in enumerate(EXTRA_NAMES)}
KINDS = ("real", "pred", "zero", "null")
CACHE_FORMAT = "stage2_cond_v2"


def null_condition(image_latent: torch.Tensor) -> torch.Tensor:
    """[27, T, h, w] for "no hint": the first-frame latent at every frame, all other channels zero.

    image_latent [16, T, h, w] is the image-conditioning latent (only its first frame is read)."""
    channels, frames, height, width = image_latent.shape
    condition = image_latent.new_zeros(V2_CHANNELS, frames, height, width)
    condition[:channels] = image_latent[:, :1].expand(-1, frames, -1, -1)
    return condition


def extra_channel_index(names) -> list[int]:
    """Positions (inside the 7 extra channels) of the named extras; unknown names raise."""
    unknown = [name for name in names if name not in EXTRA_NAMES]
    if unknown:
        raise ValueError(f"unknown extra condition channels {unknown}; choose from {EXTRA_NAMES}")
    return [EXTRA_NAMES.index(name) for name in names]


def mask_extras(condition: torch.Tensor, keep) -> torch.Tensor:
    """Copy of a v2 condition with the extra channels that are NOT in `keep` zeroed (ablations: e.g. no selector
    metadata).  The 20 base channels are never touched."""
    kept = set(extra_channel_index(keep))
    out = condition.clone()
    for index in range(EXTRA_CHANNELS):
        if index not in kept:
            out[..., BASE_CHANNELS + index, :, :, :] = 0
    return out
