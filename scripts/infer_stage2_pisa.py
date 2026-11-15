#!/usr/bin/env python
"""Stage-2 renderer driven by PisaBench oracle trajectories (RAFT flow of the reference clip), and the shared
temporal_upsample() helper that scripts/infer_stage2_cond_v2.py --piq-mp4 uses to resample a clip to 120 frames."""
from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from scripts.cache_stage1 import load_raft, raft_flow
from scripts.cache_stage2_oracle import make_motion_condition
from scripts.infer_stage2_oracle import (
    NEGATIVE_PROMPT,
    ResidualController,
    attach_lora,
    load_adapter,
    write_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage-2 renderer with PISABench oracle trajectories")
    parser.add_argument("--config", required=True, help="config YAML (this legacy entry point ships no config)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--low-checkpoint", required=True)
    parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize_mask(mask: np.ndarray, height: int, width: int) -> torch.Tensor:
    return torch.from_numpy(
        cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    ).float()


def centroid(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError("empty dropping-object mask")
    return float(xx.mean() / mask.shape[1]), float(yy.mean() / mask.shape[0])


def temporal_upsample(frames: list[np.ndarray], count: int) -> list[np.ndarray]:
    positions = np.linspace(0, len(frames) - 1, count)
    output = []
    for position in positions:
        lower = int(np.floor(position))
        upper = min(len(frames) - 1, lower + 1)
        weight = float(position - lower)
        blended = cv2.addWeighted(frames[lower], 1 - weight, frames[upper], weight, 0)
        output.append(blended)
    return output


def main() -> None:
    args = parse_args()
    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    paths, model_cfg, eval_cfg = config["paths"], config["model"], config["evaluation"]
    rows = list(csv.DictReader(Path(args.manifest).open()))
    if args.limit:
        rows = rows[: args.limit]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    model_dir = Path(paths["wan_model"])
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32).to("cuda").eval()
    raft, raft_transform = load_raft(torch.device("cuda"))
    prepared = []
    for row in rows:
        metadata = Path(row["metadata_path"])
        clip_dir = metadata.parent
        rgb_paths = sorted(clip_dir.glob("rgba_*.jpg"))
        first = np.asarray(Image.open(rgb_paths[0]).convert("RGB").resize((256, 256), Image.LANCZOS))
        image_tensor = torch.from_numpy(first).permute(2, 0, 1).float().div(255).mul(2).sub(1)
        image_video = torch.zeros(1, 3, 9, 256, 256, device="cuda")
        image_video[:, :, 0] = image_tensor.to("cuda")
        with torch.inference_mode():
            image_latent = vae.encode(image_video).latent_dist.mode()[0].cpu()
        masks = np.load(metadata)["mask"][:, -1]
        raw_indices = [0, 16, len(masks) - 1]
        latent_frames, latent_height, latent_width = image_latent.shape[1:]
        if latent_frames != len(raw_indices):
            raise RuntimeError(f"PISA image latent has {latent_frames} time steps")
        reference_frames = np.stack(
            [
                np.asarray(Image.open(rgb_paths[raw_index]).convert("RGB").resize((256, 256), Image.LANCZOS))
                for raw_index in raw_indices
            ]
        )
        flow, visibility = raft_flow(
            raft,
            raft_transform,
            reference_frames,
            [1, 2],
            12,
            torch.device("cuda"),
            1.0,
            0.05,
        )
        source_mask = resize_mask(masks[0], 256, 256)[None]
        future_mask = torch.stack([resize_mask(masks[index], 256, 256) for index in raw_indices[1:]])[:, None]
        target = {
            "flow": flow.cpu(),
            "visibility": visibility.float().cpu(),
            "object_mask": source_mask,
            "future_mask": future_mask,
        }
        condition = make_motion_condition(
            image_latent,
            target,
            source_height=256,
            source_width=256,
            selected_indices=[0, 1, 2],
            future_indices=[1, 2],
        )
        prepared.append((row, first, condition.cpu()))
    del raft, raft_transform, flow, visibility
    vae.to("cpu")
    torch.cuda.empty_cache()

    pipeline = WanImageToVideoPipeline.from_pretrained(
        model_dir, vae=vae, image_encoder=None, torch_dtype=torch.bfloat16
    )
    low_checkpoint, high_checkpoint = Path(args.low_checkpoint), Path(args.high_checkpoint)
    attach_lora(pipeline.transformer_2, low_checkpoint)
    attach_lora(pipeline.transformer, high_checkpoint)
    transformer_dim = (
        pipeline.transformer.config.num_attention_heads
        * pipeline.transformer.config.attention_head_dim
    )
    low_adapter = load_adapter(low_checkpoint, model_cfg, transformer_dim)
    high_adapter = load_adapter(high_checkpoint, model_cfg, transformer_dim)
    blocks = list(model_cfg["injection_blocks"])
    low_controller = ResidualController(pipeline.transformer_2, blocks)
    high_controller = ResidualController(pipeline.transformer, blocks)
    pipeline.enable_model_cpu_offload()

    generated = []
    for index, (row, first, condition) in enumerate(prepared):
        output = output_dir / f"{row['sample_id']}.mp4"
        if output.exists() and not args.overwrite:
            continue
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            condition_gpu = condition[None].to("cuda", torch.bfloat16)
            low_controller.set(low_adapter(condition_gpu))
            high_controller.set(high_adapter(condition_gpu))
        seed = args.seed + zlib.crc32(row["sample_id"].encode()) % 100000
        result = pipeline(
            image=Image.fromarray(first),
            prompt=row["caption"],
            negative_prompt=NEGATIVE_PROMPT,
            height=256,
            width=256,
            num_frames=9,
            num_inference_steps=int(args.steps or eval_cfg["inference_steps"]),
            guidance_scale=float(eval_cfg["guidance_scale"]),
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        frames = temporal_upsample([np.asarray(frame) for frame in result.frames[0]], 32)
        write_video(frames, output, 16)
        generated.append({
            "sample_id": row["sample_id"],
            "video": str(output.resolve()),
            "seed": seed,
            "condition_source": "PISABench real future RGB, RAFT-large pseudo-GT, official object mask",
        })
        print(f"[{index + 1}/{len(prepared)}] {output}", flush=True)
    (output_dir / f"generated_manifest_shard{args.shard_index:02d}.json").write_text(
        json.dumps({"videos": generated}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
