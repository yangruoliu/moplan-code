#!/usr/bin/env python
"""Build Stage-2 training payloads from the PhyCo target cache.

Follows the HDF5 oracle-flow cache, differing where PhyCo differs:

* clips are mp4 at 768x432, not HDF5 at 512x512, so frames are letterboxed the
  same way the targets were -- a condition built on a squashed frame would
  disagree with a flow field measured on an unsquashed one;
* the caption comes from the target payload, which records whether it is the
  dataset's own text or a sibling scenario's;
* the span is 4.0s rather than 1.2s, because every benchmark wants 5.0s and
  training at 1.2s makes that a 4x extrapolation.

Writes ``video_latent`` / ``image_cond_latent`` / ``motion_condition`` /
``prompt_embeds`` -- exactly the fields the Stage-2 trainer reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cache_stage2_oracle import make_motion_condition

CACHE_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache PhyCo Stage-2 payloads")
    parser.add_argument("--targets", required=True, help="PhyCo Stage-1 target cache (<sample_id>.pt)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default=os.environ.get("WAN_MODEL_DIR", ""), required=not os.environ.get("WAN_MODEL_DIR"),
                        help="Wan2.2-I2V-A14B-Diffusers directory (default: $WAN_MODEL_DIR)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument(
        "--frames", type=int, default=29,
        help="29 frames over 4.0s gives 8 latent frames landing on 0 and all "
             "seven target horizons exactly, covering the full clip. Every "
             "benchmark wants 5.0s, so training on the whole 4.08s keeps the "
             "extrapolation at 1.22x; stopping at 3.43s (15 frames) would make "
             "it 1.46x to save half the tokens, which is the wrong trade. "
             "Frame counts whose latent frames miss the horizons -- 17 puts them "
             "on 1.0/2.0/3.0s, 0.143s from the nearest target -- are rejected by "
             "the alignment check below rather than silently misaligned.",
    )
    parser.add_argument("--span-seconds", type=float, default=4.0)
    parser.add_argument("--source-fps", type=float, default=24.0)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_letterboxed(path: Path, height: int, width: int, seconds: list[float],
                     fps: float) -> np.ndarray:
    """Sample at the requested times and letterbox into height x width.

    The targets were letterboxed from the same 768x432 source, so the condition
    and the flow describe the same pixels. Stretching to fill instead would put
    the object somewhere the flow does not claim it is.
    """
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    source = np.stack(frames)

    native_h, native_w = source.shape[1:3]
    scale = min(height / native_h, width / native_w)
    inner_h, inner_w = round(native_h * scale), round(native_w * scale)
    top, left = (height - inner_h) // 2, (width - inner_w) // 2

    out = np.zeros((len(seconds), height, width, 3), np.uint8)
    for slot, second in enumerate(seconds):
        index = int(np.clip(round(second * fps), 0, len(source) - 1))
        resized = cv2.resize(source[index], (inner_w, inner_h), interpolation=cv2.INTER_AREA)
        out[slot, top: top + inner_h, left: left + inner_w] = resized
    return out


def preprocess(frames: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(frames).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
    return value[None]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 caching requires CUDA")

    targets = sorted(Path(args.targets).glob("*.pt"))
    if args.limit:
        targets = targets[: args.limit]
    targets = targets[args.shard_index :: args.num_shards]
    if not targets:
        raise SystemExit(f"no targets under {args.targets}")
    print(f"{len(targets)} targets on shard {args.shard_index}/{args.num_shards}", flush=True)

    from diffusers import AutoencoderKLWan
    from transformers import T5TokenizerFast, UMT5EncoderModel

    device = torch.device("cuda")
    model_dir = Path(args.model_dir)
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32).to(device).eval()
    tokenizer = T5TokenizerFast.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_dir / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()

    prompt_cache: dict[str, torch.Tensor] = {}

    def encode_prompt(prompt: str) -> torch.Tensor:
        # PhyCo has one caption per scenario, so this cache holds a handful of
        # entries and the text encoder runs a handful of times in total.
        if prompt not in prompt_cache:
            tokens = tokenizer(
                [prompt], padding="max_length", max_length=args.prompt_length, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
            )
            attention = tokens.attention_mask.to(device)
            length = max(int(attention.sum()), 1)
            with torch.no_grad():
                prompt_cache[prompt] = text_encoder(
                    input_ids=tokens.input_ids.to(device), attention_mask=attention
                ).last_hidden_state[0, :length].cpu().to(torch.bfloat16)
        return prompt_cache[prompt]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The video is sampled uniformly across the span; the latent frames Stage-2
    # conditions on are every 4th, matching Wan's 4x temporal compression.
    seconds = [args.span_seconds * i / (args.frames - 1) for i in range(args.frames)]
    selected = list(range(0, args.frames, 4))
    completed = skipped = failed = 0
    errors: list[dict] = []

    for path in targets:
        target = torch.load(path, map_location="cpu", weights_only=True)
        sample_id = target["sample_id"]
        output = output_dir / f"{sample_id}.pt"
        if output.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            horizons = list(target["horizon_seconds"])
            # selected[1:] are video frames; map each to the target horizon at
            # the same time, and fail loudly rather than silently misaligning.
            future_indices = []
            for index in selected[1:]:
                when = seconds[index]
                nearest = min(range(len(horizons)), key=lambda h: abs(horizons[h] - when))
                if abs(horizons[nearest] - when) > 0.05:
                    raise ValueError(
                        f"no target horizon near {when:.3f}s (have {horizons})"
                    )
                future_indices.append(nearest)

            frames = read_letterboxed(Path(target["video_path"]), args.height, args.width,
                                      seconds, args.source_fps)
            video = preprocess(frames).to(device)
            with torch.no_grad():
                video_latent = vae.encode(video).latent_dist.mode()[0]
                image_video = torch.cat(
                    (video[:, :, :1], video.new_zeros(*video.shape[:2], video.shape[2] - 1,
                                                      *video.shape[3:])), dim=2)
                image_cond_latent = vae.encode(image_video).latent_dist.mode()[0]

            side = target["flow"].shape[-1]
            reordered = {
                "flow": target["flow"].float()[future_indices],
                "visibility": target["visibility"].float()[future_indices],
                "object_mask": target["object_mask"].float(),
                "future_mask": target["future_mask"].float()[future_indices],
            }
            # make_motion_condition builds its zero first-frame entries on CPU
            # and moves the stack to image_latent's device, so the target has to
            # arrive on CPU or the stack mixes devices.
            motion_condition = make_motion_condition(
                image_cond_latent,
                reordered,
                source_height=side, source_width=side,
                selected_indices=list(range(len(selected))),
                future_indices=list(range(1, len(selected))),
            )

            prompt = target.get("caption", "") or f"A physical simulation of {target['scenario']}."
            payload = {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "sample_id": sample_id,
                "scenario": target["scenario"],
                "video_latent": video_latent.cpu().half(),
                "image_cond_latent": image_cond_latent.cpu().half(),
                "motion_condition": motion_condition.cpu().half(),
                "prompt_embeds": encode_prompt(prompt),
                "prompt": prompt,
                "caption_source": target.get("caption_source", ""),
                "frames": args.frames,
                "span_seconds": args.span_seconds,
                "height": args.height,
                "width": args.width,
                "latent_frame_indices": selected,
                "horizon_seconds": [seconds[i] for i in selected[1:]],
                "label_source": target["label_source"],
                "flow_source": target.get("flow_source", ""),
                "instance_source": target.get("instance_source", ""),
                "condition_scope": "all_moving_instances",
                "condition_channels": [
                    "warped_first_frame_latent[16]", "flow_xy[2]", "mask[1]", "visibility[1]"
                ],
            }
            if "affine_3d" in target:
                payload["affine_3d"] = target["affine_3d"]

            temporary = output.with_suffix(f".tmp-{os.getpid()}")
            torch.save(payload, temporary)
            temporary.replace(output)
            completed += 1
            if completed % 200 == 0:
                print(f"  {completed} written, {skipped} skipped, {failed} failed", flush=True)
        except Exception as error:                                    # noqa: BLE001
            failed += 1
            if failed <= 5:
                print(f"[fail] {sample_id}: {error!r}", flush=True)
            errors.append({"sample_id": sample_id, "error": repr(error)})

    summary = {"completed": completed, "skipped": skipped, "failed": failed,
               "shard": args.shard_index, "num_shards": args.num_shards,
               "latent_frames": len(selected), "span_seconds": args.span_seconds,
               "errors": errors[:20]}
    (output_dir / f"summary-shard{args.shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}))


if __name__ == "__main__":
    main()
