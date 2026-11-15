#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from motion_jepa.config import apply_overrides, load_config
from motion_jepa.data import ManifestRow, cache_path, read_jsonl
from motion_jepa.video import forward_splat_mask, read_official_agent_mask, read_sampled_video
from motion_jepa.vjepa import configure_runtime_rope


CACHE_FORMAT_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache V-JEPA features and RAFT pseudo targets")
    parser.add_argument("--config", required=True, help="config YAML (this legacy entry point ships no config)")
    parser.add_argument(
        "--split", action="append", choices=("train", "val", "internal_test", "test_core")
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target-only", action="store_true")
    parser.add_argument("--feature-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_raft(device: torch.device):
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms()


def load_vjepa(model_id: str, device: torch.device):
    from transformers import AutoModel, AutoVideoProcessor

    processor = AutoVideoProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


@torch.inference_mode()
def raft_flow(
    model,
    transform,
    frames: np.ndarray,
    future_indices: list[int],
    updates: int,
    device: torch.device,
    absolute_threshold: float,
    relative_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255).to(device)
    maximum = max(future_indices)
    earlier, later = transform(video[:maximum], video[1 : maximum + 1])
    step_forward = model(earlier, later, num_flow_updates=updates)[-1]
    step_backward = model(later, earlier, num_flow_updates=updates)[-1]
    height, width = step_forward.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=step_forward.dtype),
        torch.arange(width, device=device, dtype=step_forward.dtype),
        indexing="ij",
    )
    step_destination_x = xx[None] + step_forward[:, 0]
    step_destination_y = yy[None] + step_forward[:, 1]
    step_grid = torch.stack(
        (
            2 * step_destination_x / max(1, width - 1) - 1,
            2 * step_destination_y / max(1, height - 1) - 1,
        ),
        dim=-1,
    )
    backward_at_destination = F.grid_sample(
        step_backward, step_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    residual = (step_forward + backward_at_destination).square().sum(dim=1).sqrt()
    scale = (
        step_forward.square().sum(dim=1).sqrt()
        + backward_at_destination.square().sum(dim=1).sqrt()
    )
    step_in_bounds = (
        (step_destination_x >= 0)
        & (step_destination_x <= width - 1)
        & (step_destination_y >= 0)
        & (step_destination_y <= height - 1)
    )
    step_visible = step_in_bounds & (
        residual <= absolute_threshold + relative_threshold * scale
    )

    cumulative = torch.zeros_like(step_forward[:1])
    alive = torch.ones(1, 1, height, width, device=device, dtype=torch.bool)
    selected_flow: list[torch.Tensor] = []
    selected_visibility: list[torch.Tensor] = []
    future_set = set(future_indices)
    for step in range(maximum):
        current_x = xx[None] + cumulative[:, 0]
        current_y = yy[None] + cumulative[:, 1]
        current_grid = torch.stack(
            (
                2 * current_x / max(1, width - 1) - 1,
                2 * current_y / max(1, height - 1) - 1,
            ),
            dim=-1,
        )
        increment = F.grid_sample(
            step_forward[step : step + 1],
            current_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        increment_visible = F.grid_sample(
            step_visible[step : step + 1, None].float(),
            current_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ) > 0.5
        cumulative = cumulative + increment
        final_x = xx[None] + cumulative[:, 0]
        final_y = yy[None] + cumulative[:, 1]
        alive = alive & increment_visible & (
            (final_x[:, None] >= 0)
            & (final_x[:, None] <= width - 1)
            & (final_y[:, None] >= 0)
            & (final_y[:, None] <= height - 1)
        )
        if step + 1 in future_set:
            selected_flow.append(cumulative[0].clone())
            selected_visibility.append(alive[0].clone())
    return torch.stack(selected_flow), torch.stack(selected_visibility)


def moving_object_mask(flow: torch.Tensor, visibility: torch.Tensor, config: dict[str, Any]) -> np.ndarray:
    magnitude = flow.square().sum(dim=1).sqrt()
    score = (magnitude * visibility[:, 0]).amax(dim=0).float().cpu().numpy()
    positive = score[score >= float(config["motion_threshold_px"])]
    if positive.size:
        threshold = max(float(config["motion_threshold_px"]), float(np.percentile(positive, 35)))
    else:
        threshold = max(0.05, float(np.percentile(score, 98)))
    binary = (score >= threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum = int(config.get("min_mask_pixels", 32))
    candidates: list[tuple[float, int]] = []
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area < minimum:
            continue
        component_score = float(score[labels == label].sum()) * np.sqrt(area)
        candidates.append((component_score, label))
    if candidates:
        selected = labels == max(candidates)[1]
    else:
        y, x = np.unravel_index(int(score.argmax()), score.shape)
        selected = np.zeros_like(binary, dtype=bool)
        selected[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4] = True
    selected = cv2.dilate(selected.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    return selected.astype(bool)


@torch.inference_mode()
def encode_vjepa(model, processor, frames: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = frames.shape[1:3]

    def encode(clip: np.ndarray) -> torch.Tensor:
        inputs = processor(
            clip,
            size={"height": height, "width": width},
            do_center_crop=False,
            return_tensors="pt",
        ).to(device)
        pixels = inputs["pixel_values_videos"].to(torch.bfloat16)
        grid = configure_runtime_rope(
            model, frames=int(pixels.shape[1]), height=int(pixels.shape[-2]), width=int(pixels.shape[-1])
        )
        tokens = model.get_vision_features(pixel_values_videos=pixels).squeeze(0)
        if tokens.shape[0] != grid.num_tokens:
            raise RuntimeError(f"V-JEPA token count {tokens.shape[0]} != runtime grid {grid}")
        return tokens.reshape(grid.time, grid.height, grid.width, -1)

    repeated_first = np.repeat(frames[:1], 2, axis=0)
    z0 = encode(repeated_first)[0]
    full = encode(frames)
    if full.shape[0] != 8:
        raise RuntimeError(f"expected eight V-JEPA tubelets, got {full.shape[0]}")
    return z0.cpu().half(), full[1:].cpu().half()


def valid_cache(path: Path, sample_id: str, kind: str) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return (
            payload.get("cache_format_version") == CACHE_FORMAT_VERSION
            and payload.get("sample_id") == sample_id
            and payload.get("kind") == kind
        )
    except (OSError, RuntimeError, EOFError, KeyError):
        return False


def main() -> None:
    args = parse_args()
    if args.target_only and args.feature_only:
        raise ValueError("--target-only and --feature-only are mutually exclusive")
    config = apply_overrides(load_config(args.config), args.overrides)
    paths, data_config, cache_config = config["paths"], config["data"], config["cache"]
    splits = args.split or ["train", "val", "internal_test", "test_core"]
    rows: list[ManifestRow] = []
    for split in splits:
        rows.extend(read_jsonl(Path(paths["manifest_dir"]) / f"{split}.jsonl"))
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit:
        rows = rows[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("real Stage 1 caching requires a CUDA GPU")
    raft_model = raft_transform = None
    vjepa_model = vjepa_processor = None
    if not args.feature_only:
        raft_model, raft_transform = load_raft(device)
    if not args.target_only:
        vjepa_model, vjepa_processor = load_vjepa(paths["vjepa_model"], device)

    completed = {"targets": 0, "features": 0, "skipped": 0, "errors": []}
    for row in tqdm(rows, desc=f"cache {args.shard_index}/{args.num_shards}"):
        target_path = cache_path(paths["cache_dir"], "targets", row.sample_id)
        feature_path = cache_path(paths["cache_dir"], "features", row.sample_id)
        need_target = not args.feature_only and (
            args.overwrite or not valid_cache(target_path, row.sample_id, "targets")
        )
        need_feature = not args.target_only and (
            args.overwrite or not valid_cache(feature_path, row.sample_id, "features")
        )
        if not need_target and not need_feature:
            completed["skipped"] += 1
            continue
        try:
            frames, frame_indices, source_info = read_sampled_video(
                row.video_path,
                num_frames=int(data_config["input_frames"]),
                target_fps=float(data_config["target_fps"]),
                start_sec=float(data_config["start_sec"]),
                height=int(data_config["height"]),
                width=int(data_config["width"]),
            )
            if need_target:
                forward, visibility = raft_flow(
                    raft_model,
                    raft_transform,
                    frames,
                    list(map(int, data_config["future_indices"])),
                    int(cache_config["raft_updates"]),
                    device,
                    float(cache_config["fb_absolute_threshold_px"]),
                    float(cache_config["fb_relative_threshold"]),
                )
                if row.map_path:
                    mask_np = read_official_agent_mask(
                        row.map_path, int(data_config["height"]), int(data_config["width"])
                    )
                else:
                    mask_np = moving_object_mask(
                        forward, visibility,
                        {**cache_config, "min_mask_pixels": int(data_config["min_mask_pixels"])},
                    )
                mask = torch.from_numpy(mask_np).to(device=device, dtype=forward.dtype)[None, None]
                future_mask = forward_splat_mask(mask, forward[None])[0]
                atomic_torch_save(
                    {
                        "cache_format_version": CACHE_FORMAT_VERSION,
                        "kind": "targets",
                        "sample_id": row.sample_id,
                        "flow": forward.cpu().half(),
                        "visibility": visibility.cpu(),
                        "object_mask": mask[0].cpu().bool(),
                        "future_mask": future_mask.cpu().half(),
                        "frame_indices": frame_indices,
                        "source_info": source_info,
                        "flow_estimator": "torchvision_raft_large_C_T_SKHT_V2_adjacent_composed",
                        "raft_updates": int(cache_config["raft_updates"]),
                    },
                    target_path,
                )
                completed["targets"] += 1
            if need_feature:
                z0, z_future = encode_vjepa(vjepa_model, vjepa_processor, frames, device)
                atomic_torch_save(
                    {
                        "cache_format_version": CACHE_FORMAT_VERSION,
                        "kind": "features",
                        "sample_id": row.sample_id,
                        "z0": z0,
                        "z_future": z_future,
                        "vjepa_model": str(paths["vjepa_model"]),
                        "input_policy": "two_repeated_first_frames_no_future_leakage",
                        "frame_indices": frame_indices,
                    },
                    feature_path,
                )
                completed["features"] += 1
        except Exception as error:
            completed["errors"].append({"sample_id": row.sample_id, "error": repr(error)})
            print(f"ERROR {row.sample_id}: {error!r}", flush=True)
    summary = Path(paths["cache_dir"]) / "logs" / f"shard-{args.shard_index:03d}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: len(value) if key == "errors" else value for key, value in completed.items()}))
    if completed["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
