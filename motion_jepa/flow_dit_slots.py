"""Slot-based per-object 2D affine decoder -- the structured alternative.

Motivation: on Physion, fitting one 2D affine per object per horizon
reproduces the ground-truth dense flow far more closely than the round-trip
floor of the dense flow-through-VAE path.  Six numbers per object per horizon
replace 160x160x2 numbers per horizon -- a large reduction in what the model
has to output.

The affine residual is small for most scenarios; ``drape`` (cloth) is the
exception because a deformable sheet is not one rigid transform.  Hence the
optional dense residual branch.

An identity transform gives exactly zero affine flow for a static object.
The optional residual can still introduce motion there. Optional heads predict
either object activity or a freely decoded full-grid motion mask. The latter
shares motion/image features with flow and never gates the training flow.

Interface is deliberately identical to
:class:`motion_jepa.flow_dit.LatentFlowDecoder` so the two can be swapped behind
a config flag and compared with everything else held fixed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_dit import _ImageStem
from .flow_dit_time import HorizonSeconds, normalized_horizon_times, validate_time_scale
from .stage2 import Residual3DBlock


def resize_slot_masks(masks: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    """Downsample ROI coverage without dropping sub-cell objects in area mode.

    Keep bilinear as the checkpoint default. Point-sampled bilinear reduction
    can entirely miss a nonempty instance; area integrates its coverage before
    normalized pooling. Rendering still uses the original full-size masks.
    """
    if mode == "area":
        return F.interpolate(masks, size=size, mode="area")
    if mode == "bilinear":
        return F.interpolate(masks, size=size, mode="bilinear", align_corners=False)
    raise ValueError("slot_pooling_mode must be bilinear or area")


def affine_flow(
    params: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Render per-slot affine parameters into per-slot dense displacement.

    ``params`` holds the *deviation* from identity, so all-zero parameters give
    exactly zero flow -- this is what keeps an untrained model sitting on the
    zero-flow baseline.

    Args:
        params: ``[B, K, T, 6]`` as ``(da, db, dc, dd, tx, ty)`` where the
            destination is ``x' = (1+da)x + db*y + tx``, ``y' = dc*x + (1+dd)y + ty``.
            Linear terms act on **normalised** coordinates in ``[-1, 1]`` so that
            a rotation of a given angle has the same parameter magnitude
            regardless of resolution; translations are in normalised units too.
    Returns:
        ``[B, K, T, 2, H, W]`` displacement in normalised (fraction-of-frame) units.
    """
    batch, slots, horizons, _ = params.shape
    device, dtype = params.device, params.dtype
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    da, db, dc, dd, tx, ty = [params[..., index] for index in range(6)]
    shape = (batch, slots, horizons, 1, 1)
    # Displacement, not destination: identity parameters -> exactly zero.
    flow_x = da.view(shape) * grid_x + db.view(shape) * grid_y + tx.view(shape)
    flow_y = dc.view(shape) * grid_x + dd.view(shape) * grid_y + ty.view(shape)
    # Normalised coords span 2 units across the frame; convert to fraction-of-frame.
    return torch.stack((flow_x, flow_y), dim=3) * 0.5


def render_blended_affine(
    slot_masks: torch.Tensor,
    params: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Blend per-slot affine parameters by mask, then evaluate once.

    Args:
        slot_masks: ``[B, K, H, W]``, a soft partition (softmax over slots).
        params: ``[B, K, T, 6]`` deviation-from-identity parameters.
    Returns:
        ``[B, T, 2, H, W]`` displacement in fraction-of-frame units.
    """
    device, dtype = params.device, params.dtype
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    blended = torch.einsum("bkhw,bktj->bthwj", slot_masks.to(dtype), params)
    flow_x = blended[..., 0] * grid_x + blended[..., 1] * grid_y + blended[..., 4]
    flow_y = blended[..., 2] * grid_x + blended[..., 3] * grid_y + blended[..., 5]
    # Normalised coordinates span 2 units across the frame.
    return torch.stack((flow_x, flow_y), dim=2) * 0.5


class SlotTransformDecoder(nn.Module):
    """Decode flow latents into per-object affine motion, then render.

    Outputs match :class:`LatentFlowDecoder` exactly (``flow``, ``flow_raw``,
    ``gate_logits``, ``visibility_logits``, ``object_mask_logits``) plus the
    structured internals (``slot_masks``, ``slot_params``) for diagnostics and
    for the optional Hungarian mask loss. With ``motion_activity`` enabled,
    additional object/pixel activity logits, a motion mask and both ungated and
    activity-gated flows are exposed for paired evaluation. The alternative
    ``dense_motion_mask`` mode exposes independent full-grid logits and the
    same flow alternatives, without object activity logits or SAM blending.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        hidden: int = 128,
        horizons: int = 7,
        out_size: int = 160,
        depth: int = 2,
        num_slots: int = 16,
        image_widths: tuple[int, int, int] = (16, 32, 64),
        slot_hidden: int = 256,
        residual_branch: bool = True,
        residual_scale: float = 0.05,
        dense_correction: bool = False,
        dense_correction_gain: float = 0.05,
        motion_activity: bool = False,
        motion_activity_gate: bool = False,
        dense_motion_mask: bool = False,
        task_object_slot_activity: bool = False,
        task_object_gate_mode: str = "soft",
        task_object_gate_threshold: float = .5,
        task_object_gate_source: str = "activity",
        task_object_dense_rescue: bool = False,
        task_object_dense_rescue_threshold: float = .9,
        task_object_identity: bool = False,
        task_object_identity_semantic: bool = False,
        task_object_identity_token_attention: bool = False,
        prompt_dim: int = 4096,
        text_slot_binding: bool = False,
        text_slot_binding_heads: int = 4,
        text_slot_binding_max_gain: float = 2.0,
        text_slot_fusion: bool = False,
        slot_pooling_mode: str = "bilinear",
        task_object_heads_detach_shared: bool = False,
        dense_motion_mask_detach_shared: bool = False,
        dense_motion_mask_shared_gradient_scale: float = 1.0,
        visibility_from_motion: bool = True,
        zero_init_flow_head: bool = True,
        temporal_slots: bool = True,
        causal_slots: bool = False,
        incremental_params: bool = False,
        oracle_instance_masks: bool = False,
        time_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.horizons = horizons
        self.out_size = out_size
        self.num_slots = num_slots
        self.residual_branch = residual_branch
        self.residual_scale = residual_scale
        self.dense_correction = bool(dense_correction)
        self.dense_correction_gain = float(dense_correction_gain)
        if self.dense_correction and (
            not math.isfinite(self.dense_correction_gain) or self.dense_correction_gain <= 0
        ):
            raise ValueError("dense_correction_gain must be finite and positive")
        # With temporal_slots=False the slot feature is the motion latent averaged
        # over horizons, so all 7 transforms are a linear readout of one static
        # vector and the head cannot tell "moves then stops" from "stops then
        # moves": reversing the motion features' time order leaves the
        # parameters unchanged up to float noise, so collision-like dynamics
        # cannot be learned. Set false only to reload checkpoints trained with
        # the older flat readout.
        self.temporal_slots = temporal_slots
        # Horizons are conditionally independent given the latent unless a
        # temporal model links them. A collision is a dependency *between*
        # horizons -- post-contact motion is a function of the state at contact
        # -- which an independent per-horizon readout cannot represent.
        # The mask is causal so horizon t sees only horizons <= t, matching how
        # the trajectory is actually generated. This runs on pooled features,
        # not on sampled outputs, so it adds no exposure bias.
        self.causal_slots = causal_slots and temporal_slots
        # Oracle-mask arm: pooling and rendering use the exact instance masks
        # handed in at forward time, unloading the binding problem entirely so
        # the capacity goes to dynamics. The learned mask branch still runs and
        # is still supervised -- it keeps every parameter in the DDP graph and
        # keeps a deployable mask path warm -- but its output no longer steers
        # the motion. Reported as a separate arm: the masks are ground truth.
        self.oracle_instance_masks = bool(oracle_instance_masks)
        self.motion_activity = bool(motion_activity)
        self.motion_activity_gate = bool(motion_activity_gate)
        self.dense_motion_mask = bool(dense_motion_mask)
        self.task_object_slot_activity = bool(task_object_slot_activity)
        from .flow_dit_selection import validate_selection
        validate_selection(task_object_gate_mode, task_object_gate_threshold, self.task_object_slot_activity)
        self.task_object_gate_mode = task_object_gate_mode
        self.task_object_gate_threshold = task_object_gate_threshold
        if task_object_gate_source not in {"activity", "identity"}:
            raise ValueError("task_object_gate_source must be activity or identity")
        if task_object_gate_source == "identity" and (not task_object_identity or task_object_gate_mode != "hard"):
            raise ValueError("identity selection requires identity head and hard readout")
        if task_object_dense_rescue and (task_object_gate_mode != "hard" or not dense_motion_mask):
            raise ValueError("dense rescue requires hard readout and dense task mask")
        validate_selection("hard", task_object_dense_rescue_threshold)
        self.task_object_gate_source = task_object_gate_source
        self.task_object_dense_rescue = bool(task_object_dense_rescue)
        self.task_object_dense_rescue_threshold = task_object_dense_rescue_threshold
        self.task_object_identity = bool(task_object_identity)
        self.task_object_identity_semantic = bool(task_object_identity_semantic)
        self.task_object_identity_token_attention = bool(task_object_identity_token_attention)
        if self.task_object_identity_token_attention and not self.task_object_identity_semantic:
            raise ValueError("identity token attention requires semantic identity")
        self.prompt_dim = int(prompt_dim)
        self.text_slot_binding = bool(text_slot_binding)
        self.text_slot_fusion = bool(text_slot_fusion)
        self.slot_pooling_mode = str(slot_pooling_mode)
        if self.slot_pooling_mode not in {"bilinear", "area"}:
            raise ValueError("slot_pooling_mode must be bilinear or area")
        if self.text_slot_fusion and not self.text_slot_binding:
            raise ValueError("text_slot_fusion requires text_slot_binding")
        self.text_slot_binding_max_gain = float(text_slot_binding_max_gain)
        if self.text_slot_binding and (self.prompt_dim <= 0 or not self.text_slot_binding_max_gain > 1.0):
            raise ValueError("text_slot_binding needs prompt_dim > 0 and max_gain > 1")
        self.task_object_heads_detach_shared = bool(task_object_heads_detach_shared)
        self.dense_motion_mask_detach_shared = bool(dense_motion_mask_detach_shared)
        self.dense_motion_mask_shared_gradient_scale = float(
            dense_motion_mask_shared_gradient_scale
        )
        if self.dense_motion_mask_detach_shared and not self.dense_motion_mask:
            raise ValueError("dense_motion_mask_detach_shared requires dense_motion_mask")
        if self.task_object_slot_activity and not self.dense_motion_mask:
            raise ValueError("task_object_slot_activity requires dense_motion_mask")
        if self.task_object_identity and not self.task_object_slot_activity:
            raise ValueError("task_object_identity requires task_object_slot_activity")
        if self.task_object_identity_semantic and not self.task_object_identity:
            raise ValueError("task_object_identity_semantic requires task_object_identity")
        if self.task_object_identity_semantic and self.prompt_dim <= 0:
            raise ValueError("prompt_dim must be positive for semantic task identity")
        if self.task_object_heads_detach_shared and not self.task_object_slot_activity:
            raise ValueError("task_object_heads_detach_shared requires task_object_slot_activity")
        if (
            not math.isfinite(self.dense_motion_mask_shared_gradient_scale)
            or not 0 <= self.dense_motion_mask_shared_gradient_scale <= 1
        ):
            raise ValueError("dense_motion_mask_shared_gradient_scale must lie in [0, 1]")
        if self.dense_motion_mask and (self.motion_activity or self.dense_correction):
            raise ValueError("dense_motion_mask is mutually exclusive with motion_activity and dense_correction")
        if self.task_object_slot_activity and (
            not self.temporal_slots or not self.oracle_instance_masks
        ):
            raise ValueError(
                "task_object_slot_activity requires temporal_slots=True and "
                "oracle_instance_masks=True"
            )
        if self.motion_activity_gate and not self.motion_activity:
            raise ValueError("motion_activity_gate requires motion_activity")
        if self.motion_activity and (
            not self.temporal_slots or not self.oracle_instance_masks or self.dense_correction
        ):
            raise ValueError(
                "motion_activity requires temporal_slots=True, oracle_instance_masks=True "
                "and dense_correction=False"
            )
        self.time_scale = validate_time_scale(time_scale)
        # Predict each horizon's step transform and compose. Affine composition
        # is an exact 3x3 matrix product, so this costs no approximation, and
        # the regression target becomes a step displacement (~6px) instead of a
        # cumulative one that grows to ~40px by 1.4s.
        self.incremental_params = incremental_params
        full, half, quarter = image_widths

        self.image_stem = _ImageStem(image_widths)
        self.latent_stem = nn.Conv3d(2 * latent_channels + 1, hidden, 3, padding=1)
        self.latent_blocks = nn.Sequential(*[Residual3DBlock(hidden) for _ in range(depth)])

        # --- slot masks: a first-frame property, so driven by the image branch
        # with the (motion-carrying) latent as context.
        self.mask_fuse = nn.Sequential(
            nn.Conv2d(latent_channels + quarter, hidden, 3, padding=1),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.SiLU(),
        )
        self.mask_up = nn.Sequential(
            nn.Conv2d(hidden + half, hidden // 2, 3, padding=1),
            nn.GroupNorm(min(32, hidden // 2), hidden // 2),
            nn.SiLU(),
        )
        # +1 channel for an explicit background slot; softmax makes the
        # assignment a partition, so overlapping slots cannot double-count flow.
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden // 2 + full, hidden // 4, 3, padding=1),
            nn.GroupNorm(min(16, hidden // 4), hidden // 4),
            nn.SiLU(),
            nn.Conv2d(hidden // 4, num_slots + 1, 3, padding=1),
        )

        # --- per-slot motion: mask-weighted pooling of the motion features.
        self.slot_project = nn.Sequential(
            nn.Linear(hidden + latent_channels, slot_hidden),
            nn.SiLU(),
            nn.Linear(slot_hidden, slot_hidden),
            nn.SiLU(),
        )
        # Temporal slots read out 6 numbers per horizon from that horizon's own
        # pooled feature; the flat variant expands one static vector into all of
        # them at once.
        if self.causal_slots:
            layer = nn.TransformerEncoderLayer(
                d_model=slot_hidden, nhead=4, dim_feedforward=slot_hidden * 2,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            self.slot_temporal = nn.TransformerEncoder(layer, num_layers=2)
            self.horizon_embedding = nn.Parameter(torch.zeros(1, horizons, slot_hidden))
            nn.init.normal_(self.horizon_embedding, std=0.02)
        self.slot_params = nn.Linear(slot_hidden, 6 if temporal_slots else horizons * 6)

        # Visibility is a per-horizon, motion-dependent quantity: a pixel stops
        # being visible because its object moved out of frame or behind another.
        # The original wiring read it off ``mask_feature``, which is built purely
        # from the first frame and therefore carries no motion at all -- an
        # information-theoretically impossible ask: visibility F1 can fall
        # below the trivial all-visible baseline, which in turn drags down
        # ade_norm because trajectory_centers weights predicted centroids by it.
        self.visibility_from_motion = visibility_from_motion
        if visibility_from_motion:
            self.visibility_fuse = nn.Sequential(
                nn.Conv2d(hidden + quarter, hidden // 2, 3, padding=1),
                nn.GroupNorm(min(32, hidden // 2), hidden // 2),
                nn.SiLU(),
            )
            self.visibility_up = nn.Sequential(
                nn.Conv2d(hidden // 2 + half, hidden // 4, 3, padding=1),
                nn.GroupNorm(min(16, hidden // 4), hidden // 4),
                nn.SiLU(),
            )
            self.visibility_head = nn.Sequential(
                nn.Conv2d(hidden // 4 + full, hidden // 4, 3, padding=1),
                nn.GroupNorm(min(16, hidden // 4), hidden // 4),
                nn.SiLU(),
                nn.Conv2d(hidden // 4, 1, 3, padding=1),
            )
        else:
            # Kept for backward compatibility: older checkpoints use this head
            # layout and stay loadable with it.
            self.visibility_head = nn.Sequential(
                nn.Conv2d(hidden // 2 + full, hidden // 4, 3, padding=1),
                nn.GroupNorm(min(16, hidden // 4), hidden // 4),
                nn.SiLU(),
                nn.Conv2d(hidden // 4, horizons, 3, padding=1),
            )
        if residual_branch:
            self.residual_head = nn.Conv3d(hidden, 2, 3, padding=1)
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

        if zero_init_flow_head:
            # Identity transform at step 0 => flow exactly zero => the untrained
            # model reproduces the zero-flow baseline, same guarantee as the
            # dense decoder's zero-initialised flow head.
            nn.init.zeros_(self.slot_params.weight)
            nn.init.zeros_(self.slot_params.bias)

        if self.dense_correction:
            # Capacity candidate, unmeasured. Construct last and restore the CPU
            # RNG: enabling a zero head must not perturb any old parameter or
            # downstream initialization/sampling. All constructors here are CPU.
            with torch.random.fork_rng(devices=[]):
                self.dense_correction_head = nn.Conv3d(hidden, 2, 3, padding=1)
                nn.init.zeros_(self.dense_correction_head.weight)
                nn.init.zeros_(self.dense_correction_head.bias)

        if self.motion_activity:
            # Keep all historical weights and the caller's RNG stream exact.
            # Neutral probabilities are explicit (.5), not an always-open gate;
            # ungated training/inference is retained unless separately enabled.
            with torch.random.fork_rng(devices=[]):
                self.slot_activity_head = nn.Linear(slot_hidden, 1)
                self.pixel_activity_head = nn.Conv3d(hidden, 1, 3, padding=1)
                for head in (self.slot_activity_head, self.pixel_activity_head):
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)

        if self.dense_motion_mask:
            # A freely predicted full-image mask, independent of input-instance
            # blending/pooling. Build last and restore CPU RNG so every old
            # weight and the caller's next random operation remain unchanged.
            with torch.random.fork_rng(devices=[]):
                self.dense_motion_mask_fuse = nn.Sequential(
                    nn.Conv2d(hidden + quarter, hidden // 2, 3, padding=1),
                    nn.GroupNorm(min(32, hidden // 2), hidden // 2),
                    nn.SiLU(),
                )
                self.dense_motion_mask_up = nn.Sequential(
                    nn.Conv2d(hidden // 2 + half, hidden // 4, 3, padding=1),
                    nn.GroupNorm(min(16, hidden // 4), hidden // 4),
                    nn.SiLU(),
                )
                self.dense_motion_mask_head = nn.Sequential(
                    nn.Conv2d(hidden // 4 + full, hidden // 4, 3, padding=1),
                    nn.GroupNorm(min(16, hidden // 4), hidden // 4),
                    nn.SiLU(),
                    nn.Conv2d(hidden // 4, 1, 3, padding=1),
                )
                # Small, NONZERO output weights let mask-only supervision
                # update shared image/motion features on the very first step.
                # The old flow stays exact because training never gates it.
                nn.init.normal_(self.dense_motion_mask_head[-1].weight, std=1e-3)
                nn.init.zeros_(self.dense_motion_mask_head[-1].bias)
        if self.text_slot_binding:
            # Text-to-slot binding: each object slot reads the caption tokens with
            # cross-attention and gets one gain on its whole motion (affine params and
            # the residual under its mask).  The pooled-prompt identity head cannot do
            # this: the scene sentence names every object, so only token-level
            # attention can tell the subject of the action from a listed bystander.
            # gain = max_gain * sigmoid(score) with a zero-initialized score layer, so
            # gain == 1 exactly at transfer and every historical output is unchanged.
            with torch.random.fork_rng(devices=[]):
                self.tsb_token_project = nn.Sequential(
                    nn.Linear(self.prompt_dim, slot_hidden), nn.LayerNorm(slot_hidden)
                )
                self.tsb_query_project = nn.Sequential(
                    nn.LayerNorm(slot_hidden), nn.Linear(slot_hidden, slot_hidden)
                )
                self.tsb_attention = nn.MultiheadAttention(
                    slot_hidden, int(text_slot_binding_heads), batch_first=True
                )
                self.tsb_score = nn.Sequential(
                    nn.Linear(slot_hidden * 2, slot_hidden), nn.SiLU(), nn.Linear(slot_hidden, 1)
                )
                nn.init.zeros_(self.tsb_score[-1].weight)
                nn.init.zeros_(self.tsb_score[-1].bias)
        if self.text_slot_fusion:
            # A positive scalar gain cannot rotate a wrong motion vector. This
            # zero-initialized residual lets caption tokens condition the
            # vector readout AND the task-object head. No old tensor/RNG changes.
            with torch.random.fork_rng(devices=[]):
                self.tsb_fusion = nn.Linear(slot_hidden * 2, slot_hidden)
                nn.init.zeros_(self.tsb_fusion.weight)
                nn.init.zeros_(self.tsb_fusion.bias)
        if self.task_object_slot_activity:
            # This head answers a different question from the dense mask: which
            # first-frame SAM proposal is the caption's task object at each
            # query time?  Keeping it separate exposes whether object binding
            # or within-object motion is failing.  Neutral initialization keeps
            # every historical raw-flow output exact at transfer step zero.
            with torch.random.fork_rng(devices=[]):
                self.task_object_slot_head = nn.Linear(slot_hidden, 1)
                nn.init.zeros_(self.task_object_slot_head.weight)
                nn.init.zeros_(self.task_object_slot_head.bias)
        if self.task_object_identity:
            # Keep caption-conditioned object identity separate from temporal
            # motion activity.  A prompt can identify a stationary object; in
            # that case identity should stay high while every activity query is
            # low.  This head is time invariant by construction.
            with torch.random.fork_rng(devices=[]):
                if self.task_object_identity_semantic:
                    self.task_identity_visual_project = nn.Sequential(
                        nn.Linear(latent_channels + quarter, slot_hidden),
                        nn.SiLU(),
                        nn.LayerNorm(slot_hidden),
                    )
                    self.task_identity_prompt_project = nn.Sequential(
                        nn.Linear(self.prompt_dim, slot_hidden),
                        nn.SiLU(),
                        nn.LayerNorm(slot_hidden),
                    )
                    if self.task_object_identity_token_attention:
                        self.task_identity_attention = nn.MultiheadAttention(slot_hidden, 4, batch_first=True)
                    self.task_object_identity_head = nn.Sequential(
                        nn.Linear(slot_hidden * 3, slot_hidden),
                        nn.SiLU(),
                        nn.Linear(slot_hidden, 1),
                    )
                    nn.init.zeros_(self.task_object_identity_head[-1].weight)
                    nn.init.zeros_(self.task_object_identity_head[-1].bias)
                else:
                    self.task_object_identity_head = nn.Linear(slot_hidden, 1)
                    nn.init.zeros_(self.task_object_identity_head.weight)
                    nn.init.zeros_(self.task_object_identity_head.bias)

    def text_slot_context(
        self, slot_feature: torch.Tensor, prompt_embeds: torch.Tensor | None, batch: int
    ) -> torch.Tensor:
        """Return concatenated slot queries and attended caption context."""
        if prompt_embeds is None:
            raise ValueError("text_slot_binding requires prompt_embeds [B,L,D]")
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds[None]
        if prompt_embeds.shape[0] == 1 and batch > 1:
            prompt_embeds = prompt_embeds.expand(batch, -1, -1)
        if prompt_embeds.shape[0] != batch or prompt_embeds.shape[-1] != self.prompt_dim:
            raise ValueError("text_slot_binding received incompatible prompt_embeds")
        source = slot_feature.mean(dim=2) if slot_feature.ndim == 4 else slot_feature   # [B, K, C]
        query = self.tsb_query_project(source)
        prompt = prompt_embeds.to(device=query.device, dtype=query.dtype)
        padding = prompt.float().abs().amax(dim=-1) == 0                    # [B, L] True = pad
        padding[:, 0] = False                                               # never mask a whole row
        tokens = self.tsb_token_project(prompt)
        context, _ = self.tsb_attention(query, tokens, tokens, key_padding_mask=padding, need_weights=False)
        return torch.cat((query, context), dim=-1)

    def text_slot_gain(
        self, slot_feature: torch.Tensor, prompt_embeds: torch.Tensor | None, batch: int
    ) -> torch.Tensor:
        """Legacy per-slot positive scalar gain; alone it cannot change direction."""
        score = self.tsb_score(self.text_slot_context(slot_feature, prompt_embeds, batch)).squeeze(-1)
        return self.text_slot_binding_max_gain * torch.sigmoid(score.float())

    def semantic_identity_logits_from_pooled(
        self, appearance_pooled: torch.Tensor, guide_pooled: torch.Tensor,
        prompt_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """One canonical semantic selector implementation, also used by full forward."""
        batch = appearance_pooled.shape[0]
        if prompt_embeds is None:
            raise ValueError(
                "semantic task identity requires prompt_embeds [B,L,D]"
            )
        if (
            prompt_embeds.ndim != 3
            or prompt_embeds.shape[0] != batch
            or prompt_embeds.shape[-1] != self.prompt_dim
        ):
            raise ValueError(
                "semantic task identity received incompatible prompt_embeds"
            )
        prompt = prompt_embeds.to(
            device=appearance_pooled.device,
            dtype=appearance_pooled.dtype,
        )
        prompt_valid = prompt.float().abs().amax(dim=-1) > 0
        prompt_denominator = prompt_valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        prompt_pooled = (
            prompt * prompt_valid[..., None].to(prompt.dtype)
        ).sum(dim=1) / prompt_denominator.to(prompt.dtype)
        identity_visual = torch.cat(
            (appearance_pooled, guide_pooled), dim=-1
        )
        if self.task_object_heads_detach_shared:
            identity_visual = identity_visual.detach()
        visual_feature = self.task_identity_visual_project(
            identity_visual
        )
        if self.task_object_identity_token_attention:
            # Object-specific token attention avoids reducing
            # the subject and distractor nouns to one mean.
            tokens = self.task_identity_prompt_project(prompt)
            padding = ~prompt_valid
            padding[:, 0] = False
            prompt_feature, _ = self.task_identity_attention(
                visual_feature, tokens, tokens,
                key_padding_mask=padding, need_weights=False)
        else:
            prompt_feature = self.task_identity_prompt_project(prompt_pooled)
            prompt_feature = prompt_feature[:, None].expand_as(visual_feature)
        identity_feature = torch.cat(
            (
                visual_feature,
                prompt_feature,
                visual_feature * prompt_feature,
            ),
            dim=-1,
        )
        return self.task_object_identity_head(
            identity_feature
        ).squeeze(-1)

    def forward_semantic_identity(
        self, image_cond_latent: torch.Tensor, first_frame: torch.Tensor,
        instance_masks: torch.Tensor, prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Selector-only execution with the exact full-decoder input-feature operations.

        No future/Wan latent enters semantic identity in this architecture. The
        helper changes no detach behavior; callers freeze all non-selector
        parameters explicitly for an isolated classifier experiment. As in
        full forward, image_cond_latent must already use Wan mean/std
        normalization; this method must not receive the raw cached VAE code.
        """
        if not (self.oracle_instance_masks and self.task_object_identity_semantic):
            raise ValueError("independent semantic identity requires input masks and semantic head")
        if image_cond_latent.ndim == 5:
            image_cond_latent = image_cond_latent[:, :, 0]
        height, width = image_cond_latent.shape[-2:]
        guide = F.adaptive_avg_pool2d(self.image_stem(first_frame)[-1], (height, width))
        # In full forward render masks inherit the autocast mask-head dtype,
        # which equals the image-stem guide dtype (FP32 or BF16).
        render_masks = instance_masks.to(guide.dtype)
        if render_masks.shape != (image_cond_latent.shape[0], self.num_slots, self.out_size, self.out_size):
            raise ValueError("semantic identity mask geometry mismatch")
        small_masks = resize_slot_masks(render_masks, (height, width), self.slot_pooling_mode)
        weights = small_masks.flatten(2)
        denominator = weights.sum(-1, keepdim=True).clamp_min(1e-4)
        appearance_pooled = torch.einsum(
            "bkn,bcn->bkc", weights, image_cond_latent.flatten(2)
        ) / denominator
        guide_pooled = torch.einsum(
            "bkn,bcn->bkc", weights, guide.flatten(2)
        ) / denominator
        return self.semantic_identity_logits_from_pooled(
            appearance_pooled, guide_pooled, prompt_embeds)

    def forward(
        self,
        x0_latent: torch.Tensor,
        image_cond_latent: torch.Tensor,
        first_frame: torch.Tensor,
        horizon_seconds: HorizonSeconds | None = None,
        instance_masks: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, channels, frames, height, width = x0_latent.shape
        if frames != self.horizons + 1:
            raise ValueError(f"expected {self.horizons + 1} latent frames, got {frames}")
        if image_cond_latent.ndim == 5:
            image_cond_latent = image_cond_latent[:, :, 0]

        future = x0_latent[:, :, 1:]
        condition = image_cond_latent[:, :, None].expand(-1, -1, self.horizons, -1, -1)
        times = normalized_horizon_times(
            horizon_seconds, batch_size=batch, horizons=self.horizons,
            time_scale=self.time_scale, device=x0_latent.device, dtype=x0_latent.dtype,
        )
        time_channel = times.view(batch, 1, self.horizons, 1, 1).expand(
            batch, 1, self.horizons, height, width
        )

        motion = self.latent_blocks(
            self.latent_stem(torch.cat((future, condition, time_channel), dim=1))
        )  # [B, hidden, T, h, w]

        levels = self.image_stem(first_frame)
        level_full, level_half, level_quarter = levels

        # --- slot masks at full resolution
        guide = F.adaptive_avg_pool2d(level_quarter, (height, width))
        mask_feature = self.mask_fuse(torch.cat((image_cond_latent, guide), dim=1))
        half_size = self.out_size // 2
        mask_feature = F.interpolate(
            mask_feature, size=(half_size, half_size), mode="bilinear", align_corners=False
        )
        mask_feature = self.mask_up(torch.cat((mask_feature, level_half), dim=1))
        mask_feature = F.interpolate(
            mask_feature, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
        )
        mask_feature = torch.cat((mask_feature, level_full), dim=1)
        slot_logits = self.mask_head(mask_feature)                    # [B, K+1, H, W]
        assignment = slot_logits.softmax(dim=1)
        slot_masks = assignment[:, : self.num_slots]                  # [B, K, H, W]
        background = assignment[:, self.num_slots]

        if self.oracle_instance_masks:
            if instance_masks is None:
                raise ValueError(
                    "oracle_instance_masks is on but forward received no "
                    "instance_masks; refusing to fall back to the learned "
                    "masks silently"
                )
            if instance_masks.shape[1] != self.num_slots:
                raise ValueError(
                    f"expected {self.num_slots} instance mask slots, got "
                    f"{instance_masks.shape[1]}"
                )
            render_masks = instance_masks.to(slot_masks.dtype)
        else:
            render_masks = slot_masks

        if self.motion_activity:
            expected = (batch, self.num_slots, self.out_size, self.out_size)
            if render_masks.shape != expected:
                raise ValueError(f"motion_activity requires native instance_masks shape {expected}")
            if (not torch.isfinite(render_masks).all()
                    or (render_masks < 0).any() or (render_masks > 1).any()
                    or (render_masks.sum(dim=1) > 1.00001).any()):
                raise ValueError("motion_activity requires finite disjoint instance_masks in [0, 1]")

        # --- per-slot motion features by mask-weighted pooling at latent scale
        small_masks = resize_slot_masks(
            render_masks, (height, width), self.slot_pooling_mode
        )                                                             # [B, K, h, w]
        weights = small_masks.flatten(2)                              # [B, K, hw]
        denominator = weights.sum(-1, keepdim=True).clamp_min(1e-4)
        appearance_pooled = torch.einsum(
            "bkn,bcn->bkc", weights, image_cond_latent.flatten(2)
        ) / denominator
        if self.temporal_slots:
            # Keep the horizon axis: [B, K, T, C]. Averaging it away first is what
            # made the trajectory head blind to *when* anything happens.
            motion_pooled = torch.einsum(
                "bkn,bctn->bktc", weights, motion.flatten(3)
            ) / denominator[..., None]
            appearance = appearance_pooled[:, :, None].expand(-1, -1, self.horizons, -1)
            slot_feature = self.slot_project(
                torch.cat((motion_pooled, appearance), dim=-1)
            )
            if self.causal_slots:
                sequence = slot_feature.reshape(batch * self.num_slots, self.horizons, -1)
                sequence = sequence + self.horizon_embedding
                causal = torch.triu(
                    torch.full((self.horizons, self.horizons), float("-inf"),
                               device=sequence.device, dtype=sequence.dtype),
                    diagonal=1,
                )
                sequence = self.slot_temporal(sequence, mask=causal)
                slot_feature = sequence.reshape(batch, self.num_slots, self.horizons, -1)
        else:
            motion_pooled = torch.einsum(
                "bkn,bcn->bkc", weights, motion.mean(dim=2).flatten(2)
            ) / denominator
            slot_feature = self.slot_project(
                torch.cat((motion_pooled, appearance_pooled), dim=-1)
            )
        text_context = None
        if self.text_slot_fusion:
            text_context = self.text_slot_context(slot_feature, prompt_embeds, batch)
            update = self.tsb_fusion(text_context)
            slot_feature = slot_feature + (update[:, :, None] if self.temporal_slots else update)
        params = self.slot_params(slot_feature)
        params = params.reshape(batch, self.num_slots, self.horizons, 6)
        if self.incremental_params:
            params = compose_affine_sequence(params)
        slot_gain = None
        if self.text_slot_binding:
            if text_context is None:
                slot_gain = self.text_slot_gain(slot_feature, prompt_embeds, batch)
            else:
                slot_gain = self.text_slot_binding_max_gain * self.tsb_score(text_context).squeeze(-1).float().sigmoid()
            params = params * slot_gain[:, :, None, None].to(params.dtype)

        # --- render: blend the affine *parameters* per pixel, then apply once.
        # Identical to summing per-slot flow fields (affine is linear in the
        # coordinates) but never materialises [B,K,T,2,H,W], which at K=16 is
        # 5x the memory of the blended parameter maps.
        flow = render_blended_affine(render_masks, params, self.out_size, self.out_size)

        if self.residual_branch:
            residual = self.residual_head(motion)                     # [B,2,T,h,w]
            residual = F.interpolate(
                residual.permute(0, 2, 1, 3, 4).reshape(batch * self.horizons, 2, height, width),
                size=(self.out_size, self.out_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, self.horizons, 2, self.out_size, self.out_size)
            # Bounded so the residual can model cloth without silently taking
            # over the rigid part the slots are supposed to explain.
            residual = self.residual_scale * residual.tanh()
            if slot_gain is not None:
                covered = render_masks.sum(dim=1).clamp(0, 1)                 # [B, H, W]
                gain_map = torch.einsum("bkhw,bk->bhw", render_masks.to(slot_gain.dtype), slot_gain)
                gain_map = gain_map + (1.0 - covered.to(slot_gain.dtype))
                residual = residual * gain_map[:, None, None].to(residual.dtype)
            flow = flow + residual

        if self.dense_correction:
            correction = self.dense_correction_head(motion)
            correction = F.interpolate(
                correction.permute(0, 2, 1, 3, 4).reshape(batch * self.horizons, 2, height, width),
                size=(self.out_size, self.out_size), mode="bilinear", align_corners=False,
            ).reshape(batch, self.horizons, 2, self.out_size, self.out_size)
            # Gain is an optimization scale, not a cap. Never multiply by SAM:
            # the missing-SAM support is exactly what this branch must reach.
            # Supervision validity belongs in the loss, never this input path.
            flow = flow + self.dense_correction_gain * correction

        if self.visibility_from_motion:
            # Same up-path as the flow branch, but fed by the motion latent so
            # the head can actually see where things went.
            per_frame = motion.permute(0, 2, 1, 3, 4).reshape(
                batch * self.horizons, -1, height, width
            )

            def repeat_image(value: torch.Tensor) -> torch.Tensor:
                return value[:, None].expand(-1, self.horizons, -1, -1, -1).reshape(
                    batch * self.horizons, *value.shape[1:]
                )

            feature = self.visibility_fuse(
                torch.cat((per_frame, repeat_image(guide)), dim=1)
            )
            feature = F.interpolate(
                feature, size=(half_size, half_size), mode="bilinear", align_corners=False
            )
            feature = self.visibility_up(
                torch.cat((feature, repeat_image(level_half)), dim=1)
            )
            feature = F.interpolate(
                feature, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
            )
            feature = torch.cat((feature, repeat_image(level_full)), dim=1)
            visibility_logits = self.visibility_head(feature).reshape(
                batch, self.horizons, 1, self.out_size, self.out_size
            )
        else:
            visibility_logits = self.visibility_head(mask_feature).unsqueeze(2)
        object_mask_logits = torch.logit(
            (1.0 - background).clamp(1e-5, 1.0 - 1e-5)
        ).unsqueeze(1)
        # The gate is implicit here (a static slot yields exact zeros), but the
        # key is kept so the loss and the evaluator need no special-casing.
        gate_logits = torch.full_like(visibility_logits, 10.0)
        result = {
            "flow": flow,
            "flow_raw": flow,
            "gate_logits": gate_logits,
            "visibility_logits": visibility_logits,
            "object_mask_logits": object_mask_logits,
            "slot_masks": slot_masks,
            "slot_params": params,
        }
        if self.dense_motion_mask:
            # The mask head never reads render_masks, input instance masks,
            # ROI-pooled features or affine parameters.  In the isolated mode
            # it may still learn from the same feature *values*, but its much
            # larger classification loss cannot overwrite the flow trunk.
            # This is useful for the high-weight mask arm: measured weight-5
            # supervision increased false motion and worsened vectors when its
            # gradient was allowed to dominate the shared representation.
            shared_scale = (
                0.0 if self.dense_motion_mask_detach_shared
                else self.dense_motion_mask_shared_gradient_scale
            )

            def mask_shared(value: torch.Tensor) -> torch.Tensor:
                if shared_scale == 1.0:
                    return value
                # Same forward value; only the gradient entering the common
                # motion/image trunk is reduced. Mask-head gradients stay full.
                detached = value.detach()
                return detached + shared_scale * (value - detached)

            shared_motion = mask_shared(motion)
            shared_guide = mask_shared(guide)
            shared_half = mask_shared(level_half)
            shared_full = mask_shared(level_full)
            mask_motion = shared_motion.permute(0, 2, 1, 3, 4).reshape(
                batch * self.horizons, -1, height, width
            )

            def mask_image_level(value: torch.Tensor) -> torch.Tensor:
                return value[:, None].expand(-1, self.horizons, -1, -1, -1).reshape(
                    batch * self.horizons, *value.shape[1:]
                )

            mask_prediction = self.dense_motion_mask_fuse(
                torch.cat((mask_motion, mask_image_level(shared_guide)), dim=1)
            )
            mask_prediction = F.interpolate(
                mask_prediction, size=(half_size, half_size), mode="bilinear", align_corners=False,
            )
            mask_prediction = self.dense_motion_mask_up(
                torch.cat((mask_prediction, mask_image_level(shared_half)), dim=1)
            )
            mask_prediction = F.interpolate(
                mask_prediction, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False,
            )
            dense_motion_mask_logits = self.dense_motion_mask_head(
                torch.cat((mask_prediction, mask_image_level(shared_full)), dim=1)
            ).reshape(batch, self.horizons, 1, self.out_size, self.out_size)
            # Keep probabilities in FP32: BF16 sigmoid rounds small negative
            # logits to exactly .5, contradicting the logits>=0 mask metric.
            motion_mask = dense_motion_mask_logits.float().sigmoid()
            result.update(
                dense_motion_mask_logits=dense_motion_mask_logits,
                motion_mask=motion_mask,
                flow_ungated=flow,
                # A diagnostic alternative only: flow/flow_raw stay ungated.
                flow_activity_gated=flow * motion_mask,
            )
            if self.task_object_slot_activity:
                expected = (batch, self.num_slots, self.out_size, self.out_size)
                if render_masks.shape != expected:
                    raise ValueError(
                        f"task_object_slot_activity requires native instance_masks shape {expected}"
                    )
                if (
                    not torch.isfinite(render_masks).all()
                    or (render_masks < 0).any()
                    or (render_masks > 1).any()
                    or (render_masks.sum(dim=1) > 1.00001).any()
                ):
                    raise ValueError(
                        "task_object_slot_activity requires finite disjoint instance_masks in [0,1]"
                    )
                task_features = (
                    slot_feature.detach()
                    if self.task_object_heads_detach_shared else slot_feature
                )
                task_object_slot_logits = self.task_object_slot_head(task_features).squeeze(-1)
                task_object_slot_probability = task_object_slot_logits.float().sigmoid()
                identity_values = {}
                if self.task_object_identity:
                    if self.task_object_identity_semantic:
                        guide_pooled = torch.einsum(
                            "bkn,bcn->bkc", weights, guide.flatten(2)
                        ) / denominator
                        task_object_identity_logits = self.semantic_identity_logits_from_pooled(
                            appearance_pooled, guide_pooled, prompt_embeds)
                    else:
                        task_object_identity_logits = self.task_object_identity_head(
                            task_features.mean(dim=2)
                        ).squeeze(-1)
                    task_object_identity_probability = (
                        task_object_identity_logits.float().sigmoid()
                    )
                    task_object_slot_probability = (
                        task_object_slot_probability
                        * task_object_identity_probability.unsqueeze(-1)
                    )
                    identity_values = {
                        "task_object_identity_logits": task_object_identity_logits,
                        "task_object_identity_probability": task_object_identity_probability,
                        "task_object_motion_probability": (
                            task_object_slot_logits.float().sigmoid()
                        ),
                    }
                task_object_slot_mask = torch.einsum(
                    "bkt,bkhw->bthw",
                    task_object_slot_probability,
                    render_masks.float(),
                ).unsqueeze(2)
                result.update(
                    task_object_slot_logits=task_object_slot_logits,
                    task_object_slot_probability=task_object_slot_probability,
                    task_object_slot_mask=task_object_slot_mask,
                    flow_task_object_gated=flow * task_object_slot_mask,
                    **identity_values,
                )
                if self.task_object_gate_mode == "hard":
                    from .flow_dit_selection import selection_readout
                    selection = selection_readout(task_object_slot_probability, render_masks,
                        self.task_object_gate_threshold, source=self.task_object_gate_source,
                        identity_probability=identity_values.get("task_object_identity_probability"),
                        dense_probability=motion_mask if self.task_object_dense_rescue else None,
                        dense_threshold=self.task_object_dense_rescue_threshold)
                    # Preserve the calibrated-score field for diagnostics/BCE.
                    # Only the selected readout becomes binary, in train AND eval.
                    result.update(task_object_selection_mask=selection,
                                  flow_task_object_soft_gated=flow * task_object_slot_mask,
                                  flow_task_object_gated=flow * selection)
        if self.motion_activity:
            slot_activity_logits = self.slot_activity_head(slot_feature).squeeze(-1)
            pixel_activity_logits = self.pixel_activity_head(motion)
            pixel_activity_logits = F.interpolate(
                pixel_activity_logits.permute(0, 2, 1, 3, 4).reshape(
                    batch * self.horizons, 1, height, width
                ),
                size=(self.out_size, self.out_size), mode="bilinear", align_corners=False,
            ).reshape(batch, self.horizons, 1, self.out_size, self.out_size)
            # Cumulative activity at each horizon in FIRST-FRAME coordinates,
            # matching flow support. This is not future-frame occupancy.
            object_activity = torch.einsum(
                "bkt,bkhw->bthw", slot_activity_logits.sigmoid(), render_masks
            ).unsqueeze(2)
            union = render_masks.sum(dim=1).clamp(0, 1)[:, None, None]
            motion_mask = object_activity + (1 - union) * pixel_activity_logits.sigmoid()
            result.update(
                slot_activity_logits=slot_activity_logits,
                pixel_activity_logits=pixel_activity_logits,
                motion_mask=motion_mask,
                flow_ungated=flow,
                flow_activity_gated=flow * motion_mask,
            )
            if self.motion_activity_gate:
                # Gate the complete field: a residual cannot bypass object
                # activity, and an empty SAM still has the dense pixel path.
                result["flow"] = result["flow_activity_gated"]
                result["flow_raw"] = result["flow_activity_gated"]
        return result


def compose_affine_sequence(increments: torch.Tensor) -> torch.Tensor:
    """Turn per-horizon *increments* into cumulative transforms.

    Params are the deviation from identity: the affine acting on normalised
    coordinates is ``[[1+da, db, tx], [dc, 1+dd, ty]]``. Composing two of those
    is an exact 3x3 matrix product, so an incremental parameterisation costs no
    approximation here -- unlike a dense flow field, which would have to be
    warped and resampled to compose.

    Args:
        increments: ``[B, K, T, 6]`` step transforms, each relative to the
            previous horizon.
    Returns:
        ``[B, K, T, 6]`` cumulative transforms, each relative to frame 0.
    """
    batch, slots, horizons, _ = increments.shape
    matrix = _params_to_matrix(increments)                       # [B,K,T,3,3]
    cumulative = matrix[:, :, 0]
    out = [cumulative]
    for horizon in range(1, horizons):
        cumulative = matrix[:, :, horizon] @ cumulative
        out.append(cumulative)
    return _matrix_to_params(torch.stack(out, dim=2))


def _params_to_matrix(params: torch.Tensor) -> torch.Tensor:
    da, db, dc, dd, tx, ty = params.unbind(dim=-1)
    one = torch.ones_like(da)
    zero = torch.zeros_like(da)
    rows = [
        torch.stack((one + da, db, tx), dim=-1),
        torch.stack((dc, one + dd, ty), dim=-1),
        torch.stack((zero, zero, one), dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def _matrix_to_params(matrix: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            matrix[..., 0, 0] - 1.0,
            matrix[..., 0, 1],
            matrix[..., 1, 0],
            matrix[..., 1, 1] - 1.0,
            matrix[..., 0, 2],
            matrix[..., 1, 2],
        ),
        dim=-1,
    )


def hungarian_mask_loss(
    slot_masks: torch.Tensor,
    instance_id: torch.Tensor,
    *,
    max_instances: int = 32,
    return_assignment: bool = False,
):
    """Permutation-invariant supervision for the slot masks.

    Slot ordering is arbitrary, so each ground-truth instance is matched to the
    slot that already overlaps it best (maximum soft IoU, solved exactly with
    the Hungarian algorithm) and only then supervised.  Without this the flow
    loss still shapes the slots, but far more slowly.

    With ``return_assignment`` the solved matching is handed back as
    ``[(batch_index, slot_indices, instance_values)]`` so a caller can supervise
    the *same* pairing on another quantity -- the per-slot affine parameters --
    without paying for a second Hungarian solve or, worse, solving it
    differently and supervising a different correspondence.
    """
    from scipy.optimize import linear_sum_assignment

    batch = slot_masks.shape[0]
    total = slot_masks.new_zeros(())
    counted = 0
    assignment: list[tuple[int, list[int], list[int]]] = []
    for index in range(batch):
        ids = instance_id[index, 0].long()
        present = [int(value) for value in torch.unique(ids) if int(value) > 0]
        present = present[:max_instances]
        if not present:
            continue
        truth = torch.stack([(ids == value).float() for value in present])  # [G,H,W]
        predicted = slot_masks[index]                                        # [K,H,W]
        intersection = torch.einsum("khw,ghw->kg", predicted, truth)
        union = (
            predicted.sum((-1, -2))[:, None] + truth.sum((-1, -2))[None, :] - intersection
        ).clamp_min(1e-6)
        iou = intersection / union
        rows, columns = linear_sum_assignment(-iou.detach().cpu().numpy())
        matched = iou[rows, columns]
        total = total + (1.0 - matched).mean()
        counted += 1
        if return_assignment:
            assignment.append(
                (index, [int(r) for r in rows], [present[int(c)] for c in columns])
            )
    loss = total / max(counted, 1)
    return (loss, assignment) if return_assignment else loss


def normalized_grid(height: int, width: int, device=None, dtype=torch.float32):
    """Coordinate grid in ``[-1, 1]``, the convention the affine params act in."""
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    return torch.meshgrid(ys, xs, indexing="ij")


def fit_instance_affine(
    flow_pixels: torch.Tensor,
    instance_id: torch.Tensor,
    valid: torch.Tensor,
    *,
    min_pixels: int = 12,
    eval_side: float = 160.0,
    return_valid: bool = False,
) -> tuple[torch.Tensor, list[int]] | tuple[torch.Tensor, list[int], torch.Tensor]:
    """Least-squares fit one 2D affine per instance per horizon to ground-truth flow.

    Produces the *oracle* version of what :class:`SlotTransformDecoder` predicts,
    in exactly the same parameterisation, so the two are directly comparable and
    the fitted params can serve as an upper bound for the structured Stage-1.

    Args:
        flow_pixels: ``[T, 2, H, W]`` first-to-future flow in pixels.
        instance_id: ``[1, H, W]`` with 0 = background, 1..K = instances.
        valid: ``[T, 1, H, W]`` the pixels the metric scores.
    Returns:
        ``(params [K, T, 6], instance_ids)`` by default.  With
        ``return_valid=True`` a boolean ``[K,T]`` tensor is appended.  Instances
        with too few pixels at a horizon keep the identity transform, but that
        identity is only a placeholder and the returned validity bit is false.
    """
    horizons = flow_pixels.shape[0]
    height, width = flow_pixels.shape[-2:]
    grid_y, grid_x = normalized_grid(height, width, flow_pixels.device, torch.float32)
    ids = [int(value) for value in torch.unique(instance_id) if int(value) > 0]
    params = flow_pixels.new_zeros(len(ids), horizons, 6)
    affine_valid = torch.zeros(
        len(ids), horizons, dtype=torch.bool, device=flow_pixels.device
    )
    scale = eval_side * 0.5  # render multiplies by eval_side/2

    for slot, identifier in enumerate(ids):
        belongs = instance_id[0] == identifier
        for horizon in range(horizons):
            selected = belongs & (valid[horizon, 0] > 0.5)
            count = int(selected.sum())
            if count < min_pixels:
                continue
            affine_valid[slot, horizon] = True
            x = grid_x[selected]
            y = grid_y[selected]
            design = torch.stack((x, y, torch.ones_like(x)), dim=1).float()
            target = flow_pixels[horizon][:, selected].float() / scale  # [2, N]
            solution = torch.linalg.lstsq(design, target.T).solution      # [3, 2]
            params[slot, horizon, 0] = solution[0, 0]   # da
            params[slot, horizon, 1] = solution[1, 0]   # db
            params[slot, horizon, 4] = solution[2, 0]   # tx
            params[slot, horizon, 2] = solution[0, 1]   # dc
            params[slot, horizon, 3] = solution[1, 1]   # dd
            params[slot, horizon, 5] = solution[2, 1]   # ty
    if return_valid:
        return params, ids, affine_valid
    return params, ids


def render_instance_affine(
    params: torch.Tensor,
    instance_id: torch.Tensor,
    ids: list[int],
    *,
    eval_side: float = 160.0,
) -> torch.Tensor:
    """Render fitted per-instance affine params back to a dense pixel flow field.

    Background stays exactly zero, and every pixel of an instance moves under one
    consistent transform -- the two structural properties that make this family
    attractive as a Stage-2 condition.
    """
    height, width = instance_id.shape[-2:]
    masks = torch.stack(
        [(instance_id[0] == identifier).float() for identifier in ids]
    ) if ids else params.new_zeros(0, height, width)
    if masks.shape[0] == 0:
        return params.new_zeros(params.shape[1], 2, height, width)
    flow = render_blended_affine(masks[None], params[None], height, width)[0]
    return flow * eval_side
