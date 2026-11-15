#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from motion_jepa.data import cache_path, read_jsonl
from motion_jepa.stage2 import SCENARIO_PROMPTS
from motion_jepa.video import read_sampled_video


CACHE_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Wan A14B latents and oracle-flow conditions")
    parser.add_argument("--config", required=True, help="config YAML (this legacy entry point ships no config)")
    parser.add_argument("--split", action="append", choices=("train", "val", "internal_test", "test_core"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def preprocess(frames, height: int, width: int) -> torch.Tensor:
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255)
    if video.shape[-2:] != (height, width):
        video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return (video * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0)


def forward_splat_values(
    values: torch.Tensor,
    flow: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-splat [T,C,H,W] values with source-to-target pixel flow."""
    horizons, channels, height, width = values.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    destination_x = xx[None] + flow[:, 0]
    destination_y = yy[None] + flow[:, 1]
    x0, y0 = destination_x.floor(), destination_y.floor()
    output = values.new_zeros(horizons, channels, height * width)
    normalizer = values.new_zeros(horizons, 1, height * width)
    source = values.flatten(2)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xi, yi = x0 + dx, y0 + dy
        valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        bilinear = (1 - (destination_x - xi).abs()) * (1 - (destination_y - yi).abs())
        contribution = bilinear[:, None] * valid[:, None] * weight
        index = (yi.clamp(0, height - 1) * width + xi.clamp(0, width - 1)).long().flatten(1)
        output.scatter_add_(2, index[:, None].expand(-1, channels, -1), source * contribution.flatten(2))
        normalizer.scatter_add_(2, index[:, None], contribution.flatten(2))
    output = output / normalizer.clamp_min(1e-6)
    return output.reshape(horizons, channels, height, width), normalizer.reshape(horizons, 1, height, width)


def make_motion_condition(
    image_latent: torch.Tensor,
    target: dict,
    *,
    source_height: int,
    source_width: int,
    selected_indices: list[int],
    future_indices: list[int],
) -> torch.Tensor:
    """Construct [warped z0, target-grid flow, mask, visibility] at VAE resolution."""
    _, latent_frames, latent_height, latent_width = image_latent.shape
    if latent_frames != len(selected_indices):
        raise ValueError(
            f"image condition has {latent_frames} latent frames, expected {len(selected_indices)}"
        )
    future_to_slot = {int(frame): slot for slot, frame in enumerate(future_indices)}
    flows = [torch.zeros(2, source_height, source_width)]
    visibility = [torch.ones(1, source_height, source_width)]
    future_masks = [target["object_mask"].float()]
    for frame_index in selected_indices[1:]:
        if frame_index not in future_to_slot:
            raise KeyError(f"Stage-1 target cache has no flow for sampled frame {frame_index}")
        slot = future_to_slot[frame_index]
        flows.append(target["flow"][slot].float())
        visibility.append(target["visibility"][slot].float())
        future_masks.append(target["future_mask"][slot].float())
    flow = torch.stack(flows).to(image_latent.device)
    visible = torch.stack(visibility).to(image_latent.device)
    target_mask = torch.stack(future_masks).to(image_latent.device)
    source_mask = target["object_mask"].float().to(image_latent.device)

    flow = F.interpolate(flow, size=(latent_height, latent_width), mode="bilinear", align_corners=False)
    flow[:, 0].mul_(latent_width / source_width)
    flow[:, 1].mul_(latent_height / source_height)
    visible = F.interpolate(visible, size=(latent_height, latent_width), mode="nearest")
    target_mask = F.interpolate(target_mask, size=(latent_height, latent_width), mode="area").clamp_(0, 1)
    source_mask = F.interpolate(source_mask[None], size=(latent_height, latent_width), mode="area")[0].clamp_(0, 1)

    # Only the selected object moves; static background flow is exactly zero.
    object_flow = flow * source_mask[None]
    source_valid = (1 - source_mask[None]) + source_mask[None] * visible
    first_latent = image_latent[:, :1].permute(1, 0, 2, 3).expand(latent_frames, -1, -1, -1)
    warped, _ = forward_splat_values(first_latent, object_flow, source_valid)
    normalized_flow = object_flow.clone()
    normalized_flow[:, 0].div_(latent_width)
    normalized_flow[:, 1].div_(latent_height)
    flow_at_target, visible_weight = forward_splat_values(
        normalized_flow, object_flow, source_mask[None] * visible
    )
    visible_target = visible_weight.clamp(0, 1)
    condition = torch.cat(
        (
            warped.permute(1, 0, 2, 3),
            flow_at_target.permute(1, 0, 2, 3),
            target_mask.permute(1, 0, 2, 3),
            visible_target.permute(1, 0, 2, 3),
        ),
        dim=0,
    )
    return condition


def valid_cache(path: Path, sample_id: str) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return (
            payload.get("cache_format_version") == CACHE_FORMAT_VERSION
            and payload.get("sample_id") == sample_id
            and payload.get("label_source") == "torchvision RAFT-large pseudo-GT"
        )
    except (OSError, RuntimeError, EOFError):
        return False


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths, data = config["paths"], config["data"]
    splits = args.split or ["train", "val", "internal_test", "test_core"]
    rows = []
    for split in splits:
        rows.extend(read_jsonl(Path(paths["manifest_dir"]) / f"{split}.jsonl"))
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit:
        rows = rows[: args.limit]
    if not torch.cuda.is_available():
        raise RuntimeError("real Stage-2 caching requires a CUDA GPU")

    from diffusers import AutoencoderKLWan
    from transformers import T5TokenizerFast, UMT5EncoderModel

    device = torch.device("cuda")
    model_dir = Path(paths["wan_model"])
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32).to(device).eval()
    tokenizer = T5TokenizerFast.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_dir / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()
    prompt_cache: dict[str, torch.Tensor] = {}

    def encode_prompt(prompt: str) -> torch.Tensor:
        if prompt not in prompt_cache:
            tokens = tokenizer(
                [prompt], padding="max_length", max_length=512, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device)
            attention = tokens.attention_mask.to(device)
            length = int(attention.sum())
            prompt_cache[prompt] = text_encoder(
                input_ids=input_ids, attention_mask=attention
            ).last_hidden_state[0, :length].cpu().to(torch.bfloat16)
        return prompt_cache[prompt]

    output_dir = Path(paths["stage2_cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    completed, skipped, errors = 0, 0, []
    selected_indices = list(map(int, data["latent_frame_indices"]))
    future_indices = list(map(int, data["stage1_future_indices"]))
    for row in tqdm(rows, desc=f"stage2-cache {args.shard_index}/{args.num_shards}"):
        output = output_dir / f"{row.sample_id}.pt"
        if not args.overwrite and valid_cache(output, row.sample_id):
            skipped += 1
            continue
        try:
            frames, frame_indices, source_info = read_sampled_video(
                row.video_path,
                num_frames=int(data["frames"]),
                target_fps=float(data["fps"]),
                start_sec=float(data.get("start_sec", 0)),
                height=int(data["height"]),
                width=int(data["width"]),
            )
            video = preprocess(frames, int(data["height"]), int(data["width"])).to(device)
            video_latent = vae.encode(video).latent_dist.mode()[0]
            zeros = video.new_zeros(video.shape[0], video.shape[1], video.shape[2] - 1, video.shape[3], video.shape[4])
            image_video = torch.cat((video[:, :, :1], zeros), dim=2)
            image_cond_latent = vae.encode(image_video).latent_dist.mode()[0]
            target_path = cache_path(paths["stage1_cache_dir"], "targets", row.sample_id)
            target = torch.load(target_path, map_location="cpu", weights_only=True)
            motion_condition = make_motion_condition(
                image_cond_latent,
                target,
                source_height=int(config["stage1_data"]["height"]),
                source_width=int(config["stage1_data"]["width"]),
                selected_indices=selected_indices,
                future_indices=future_indices,
            )
            payload = {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "sample_id": row.sample_id,
                "scenario": row.scenario,
                "video_latent": video_latent.cpu().half(),
                "image_cond_latent": image_cond_latent.cpu().half(),
                "motion_condition": motion_condition.cpu().half(),
                "prompt_embeds": encode_prompt(SCENARIO_PROMPTS[row.scenario]),
                "prompt": SCENARIO_PROMPTS[row.scenario],
                "frames": int(data["frames"]),
                "fps": float(data["fps"]),
                "height": int(data["height"]),
                "width": int(data["width"]),
                "frame_indices": frame_indices,
                "source_info": source_info,
                "latent_frame_indices": selected_indices,
                "label_source": "torchvision RAFT-large pseudo-GT",
                "condition_channels": ["warped_first_frame_latent[16]", "flow_xy[2]", "mask[1]", "visibility[1]"],
            }
            temporary = output.with_suffix(f".tmp-{os.getpid()}")
            torch.save(payload, temporary)
            temporary.replace(output)
            completed += 1
        except Exception as error:
            errors.append({"sample_id": row.sample_id, "error": repr(error)})
            print(f"ERROR {row.sample_id}: {error!r}", flush=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {"completed": completed, "skipped": skipped, "errors": errors}
    (log_dir / f"shard-{args.shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"completed": completed, "skipped": skipped, "errors": len(errors)}))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
