from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_2d(height: int, width: int, dimension: int) -> torch.Tensor:
    if dimension % 4:
        raise ValueError("2D sinusoidal embedding dimension must be divisible by four")
    quarter = dimension // 4
    omega = torch.arange(quarter, dtype=torch.float32) / max(1, quarter - 1)
    omega = 1.0 / (10000**omega)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    values = []
    for coordinate in (xx.reshape(-1), yy.reshape(-1)):
        angles = coordinate[:, None] * omega[None]
        values.extend((angles.sin(), angles.cos()))
    return torch.cat(values, dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(16, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class RefinementHead(nn.Module):
    def __init__(self, input_channels: int = 5, base_channels: int = 128) -> None:
        super().__init__()
        widths = (base_channels, 128, 96, 64, 32)
        self.input = nn.Conv2d(input_channels, widths[0], 3, padding=1)
        stages: list[nn.Module] = []
        for before, after in zip(widths[:-1], widths[1:]):
            stages.extend(
                [ResidualBlock(before), nn.ConvTranspose2d(before, after, 4, stride=2, padding=1)]
            )
        self.stages = nn.Sequential(*stages)
        self.output = nn.Sequential(ResidualBlock(widths[-1]), nn.SiLU(), nn.Conv2d(widths[-1], 3, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        coarse_flow: torch.Tensor,
        confidence: torch.Tensor,
        object_mask_grid: torch.Tensor,
        progress_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, horizons, _, grid_height, grid_width = coarse_flow.shape
        if progress_override is None:
            progress = torch.linspace(
                1 / horizons,
                1,
                horizons,
                device=coarse_flow.device,
                dtype=coarse_flow.dtype,
            ).view(1, horizons, 1, 1, 1)
        else:
            progress = progress_override.to(
                device=coarse_flow.device, dtype=coarse_flow.dtype
            ).reshape(1, horizons, 1, 1, 1)
        progress = progress.expand(batch, -1, -1, grid_height, grid_width)
        mask = object_mask_grid[:, None].expand(-1, horizons, -1, -1, -1)
        value = torch.cat((coarse_flow, confidence, mask, progress), dim=2).reshape(
            batch * horizons, 5, grid_height, grid_width
        )
        value = self.output(self.stages(self.input(value)))
        height, width = value.shape[-2:]
        value = value.reshape(batch, horizons, 3, height, width)
        coarse_up = F.interpolate(
            coarse_flow.reshape(batch * horizons, 2, grid_height, grid_width),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, horizons, 2, height, width)
        flow = coarse_up + 0.25 * value[:, :, :2].tanh()
        visibility_logits = value[:, :, 2:3]
        return flow, visibility_logits


class MotionJEPA(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        latent_dim = int(config["latent_dim"])
        hidden = int(config["hidden_dim"])
        match_dim = int(config["match_dim"])
        heads = int(config["heads"])
        expansion = int(round(hidden * float(config["mlp_ratio"])))
        dropout = float(config["dropout"])
        self.grid_height = int(config["grid_height"])
        self.grid_width = int(config["grid_width"])
        self.horizons = int(config["horizons"])
        self.temperature = float(config["match_temperature"])
        self.latent_dim = latent_dim
        self.condition_type = str(config.get("condition_type", "scenario"))

        self.input_norm = nn.LayerNorm(latent_dim)
        self.input_projection = nn.Linear(latent_dim, hidden)
        self.mask_projection = nn.Linear(1, hidden)
        if self.condition_type == "scenario":
            self.scenario_embedding = nn.Embedding(int(config["scenarios"]), hidden)
            self.text_norm = None
            self.text_projection = None
        elif self.condition_type == "text":
            text_dim = int(config["text_dim"])
            self.scenario_embedding = None
            self.text_norm = nn.LayerNorm(text_dim)
            self.text_projection = nn.Linear(text_dim, hidden)
        else:
            raise ValueError(
                f"condition_type must be 'scenario' or 'text', got {self.condition_type!r}"
            )
        self.time_embedding = nn.Embedding(self.horizons, hidden)
        self.register_buffer(
            "spatial_position",
            sinusoidal_2d(self.grid_height, self.grid_width, hidden),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=expansion,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(config["encoder_depth"]), norm=nn.LayerNorm(hidden)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=expansion,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.future_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=int(config["decoder_depth"]), norm=nn.LayerNorm(hidden)
        )
        self.latent_delta = nn.Linear(hidden, latent_dim)
        self.match_source = nn.Linear(latent_dim, match_dim)
        self.match_future = nn.Linear(latent_dim, match_dim)
        self.refinement = RefinementHead()

    def forward(
        self,
        z0: torch.Tensor,
        object_mask: torch.Tensor,
        scenario: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, grid_height, grid_width, channels = z0.shape
        if (grid_height, grid_width, channels) != (
            self.grid_height,
            self.grid_width,
            self.latent_dim,
        ):
            raise ValueError(
                f"unexpected z0 shape {tuple(z0.shape)}; expected "
                f"[B,{self.grid_height},{self.grid_width},{self.latent_dim}]"
            )
        tokens = z0.reshape(batch, grid_height * grid_width, channels)
        mask_grid = F.interpolate(
            object_mask.float(), size=(grid_height, grid_width), mode="area"
        )
        if self.condition_type == "scenario":
            if scenario is None or self.scenario_embedding is None:
                raise ValueError("scenario conditioning requires scenario ids")
            condition_token = self.scenario_embedding(scenario)[:, None]
        else:
            if text_embedding is None or self.text_norm is None or self.text_projection is None:
                raise ValueError("text conditioning requires pooled text_embedding")
            condition_token = self.text_projection(self.text_norm(text_embedding))[:, None]
        memory = self.input_projection(self.input_norm(tokens))
        memory = (
            memory
            + self.spatial_position[None].to(memory.dtype)
            + self.mask_projection(mask_grid.flatten(2).transpose(1, 2))
            + condition_token
        )
        memory = self.spatial_encoder(memory)

        times = self.time_embedding.weight[None, :, None]
        queries = memory[:, None] + times + condition_token[:, None]
        queries = queries.reshape(batch, self.horizons * grid_height * grid_width, -1)
        decoded = self.future_decoder(queries, memory)
        decoded = decoded.reshape(batch, self.horizons, grid_height * grid_width, -1)
        z_pred = tokens[:, None] + self.latent_delta(decoded)

        source = F.normalize(self.match_source(tokens), dim=-1)
        future = F.normalize(self.match_future(z_pred), dim=-1)
        similarity = torch.einsum("bnd,btmd->btnm", source, future).float()
        attention = (similarity / self.temperature).softmax(dim=-1).to(z_pred.dtype)
        confidence = attention.amax(dim=-1)
        coordinates = self._coordinates(z_pred.device, z_pred.dtype)
        matched = torch.einsum("btnm,md->btnd", attention, coordinates)
        coarse = matched - coordinates[None, None]
        coarse = coarse.reshape(batch, self.horizons, grid_height, grid_width, 2).permute(
            0, 1, 4, 2, 3
        )
        confidence = confidence.reshape(batch, self.horizons, 1, grid_height, grid_width)
        flow, visibility_logits = self.refinement(coarse, confidence, mask_grid)
        full_mask = object_mask[:, None].to(flow.dtype)
        flow = flow * full_mask
        visibility_logits = visibility_logits * full_mask + (-10.0) * (1 - full_mask)
        return {
            "z_pred": z_pred.reshape(
                batch, self.horizons, grid_height, grid_width, self.latent_dim
            ),
            "flow": flow,
            "visibility_logits": visibility_logits,
            "coarse_flow": coarse,
            "match_confidence": confidence,
        }

    def _coordinates(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, self.grid_height, device=device, dtype=dtype),
            torch.linspace(0, 1, self.grid_width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
