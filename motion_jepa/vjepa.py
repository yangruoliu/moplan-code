from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch


@dataclass(frozen=True)
class TokenGrid:
    time: int
    height: int
    width: int

    @property
    def tokens_per_frame(self) -> int:
        return self.height * self.width

    @property
    def num_tokens(self) -> int:
        return self.time * self.tokens_per_frame


def _runtime_position_ids(self, value: torch.Tensor, masks: torch.Tensor | None = None):
    grid: TokenGrid | None = getattr(self, "_moplan_runtime_grid", None)
    if grid is None:
        raise RuntimeError("V-JEPA runtime token grid was not configured")
    if masks is None:
        if value.shape[1] != grid.num_tokens:
            raise RuntimeError(
                f"V-JEPA emitted {value.shape[1]} tokens, expected {grid.num_tokens} for {grid}"
            )
        ids = torch.arange(grid.num_tokens, device=value.device)
    else:
        ids = masks.to(value.device).unsqueeze(1).repeat(1, self.num_attention_heads, 1)
    frame = ids // grid.tokens_per_frame
    within = ids.remainder(grid.tokens_per_frame)
    row = within // grid.width
    column = within.remainder(grid.width)
    return frame, row, column


def configure_runtime_rope(
    model: torch.nn.Module,
    *,
    frames: int,
    height: int,
    width: int,
) -> TokenGrid:
    tubelet = int(model.config.tubelet_size)
    patch = int(model.config.patch_size)
    if frames % tubelet or height % patch or width % patch:
        raise ValueError(
            f"input {(frames, height, width)} is incompatible with tubelet={tubelet}, patch={patch}"
        )
    grid = TokenGrid(frames // tubelet, height // patch, width // patch)
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "VJEPA2RopeAttention":
            continue
        if not hasattr(module, "_moplan_original_position_ids"):
            module._moplan_original_position_ids = module.get_position_ids
            module.get_position_ids = MethodType(_runtime_position_ids, module)
        module._moplan_runtime_grid = grid
        patched += 1
    if not patched:
        raise RuntimeError("loaded model contains no VJEPA2RopeAttention blocks")
    return grid
