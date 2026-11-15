#!/usr/bin/env python
"""Cache Stage-1 WanFlowDiT inputs: flow latents, image latents, first frames.

Reuses the existing target cache verbatim -- no geometry is recomputed, so the
supervision stays byte-identical.  Only the first RGB frame is read from the
source clip (HDF5, mp4 or image).

Per sample at ``flow_resolution=256``:
    flow_latent        (16, 8, 32, 32) fp16   262 KB
    image_cond_latent  (16, 1, 32, 32) fp16    33 KB
    first_frame        (160,160,3)     uint8   77 KB
                                             ~374 KB  ->  6.5 GB for 17.4k
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_jepa.data import cache_path, read_jsonl
from motion_jepa.flow_codec import (
    WanVaeCodec,
    build_codec,
    codec_params,
    clip_fraction,
    flow_to_image,
    prepare_visibility,
    prepend_zero_slot,
)
from motion_jepa.flow_dit_config import apply_overrides, load_config, resolve_derived
from motion_jepa.flow_dit_data import CACHE_FORMAT_VERSION, sha256_file
from scripts.stage1_paths import placeholder_values, resolve_placeholders


SPLITS = ("train", "val", "internal_test", "test_core")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Stage-1 flow-diffusion inputs")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "stage1" / "train.yaml"),
                        help="config; ${...} path placeholders are filled from MOPLAN_DATA / WAN_MODEL_DIR")
    parser.add_argument("--split", action="append", choices=SPLITS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--from-targets",
        action="store_true",
        help="treat the target cache as the corpus (PhyCo) instead of reading a "
             "manifest; each target carries its own video path and caption",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--video-override-manifest",
        default="",
        help="jsonl whose rows map sample_id -> video_path (png/jpg/mp4/hdf5); when a "
             "sample is listed, its first frame is read from there instead of the "
             "corpus path. Targets are unchanged. Works with and without --from-targets.",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def read_first_frame(path: str | Path, size: int) -> np.ndarray:
    """Decode frame 0 and fit it into ``size`` x ``size``.

    Physion clips are square HDF5, so a plain resize is exact. PhyCo clips are
    768x432 mp4 and their flow targets were letterboxed -- the frame must go
    through the identical transform (scale by size/max(h,w), centre, black bars)
    or the decoder's image guidance describes different pixels than the flow.
    """
    path = Path(path)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot decode first frame image {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif path.suffix.lower() in (".mp4", ".avi", ".mkv", ".webm"):
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"cannot decode first frame of {path}")
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        import h5py  # only Physion HDF5 inputs need h5py
        from motion_jepa.physion_hdf5 import decode_image

        with h5py.File(path, "r") as handle:
            keys = sorted(handle["frames"])
            image = decode_image(handle["frames"][keys[0]]["images/_img"])
            image = np.asarray(image)[..., :3]

    height, width = image.shape[:2]
    if height == width:
        if height != size:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        return image.astype(np.uint8, copy=False)
    scale = size / max(height, width)
    inner_h, inner_w = round(height * scale), round(width * scale)
    resized = cv2.resize(image, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), np.uint8)
    top, left = (size - inner_h) // 2, (size - inner_w) // 2
    out[top: top + inner_h, left: left + inner_w] = resized
    return out


def to_model_range(image: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> ``[-1, 1]`` CHW float."""
    # copy(): cv2.resize can hand back a non-writable view, which torch warns about.
    # HDF5/Pillow decoders can return a contiguous but read-only array.
    # ``ascontiguousarray`` then preserves the read-only backing store and
    # PyTorch warns that later in-place operations are undefined.  The tensor
    # is tiny relative to the VAE input, so take an explicit writable copy.
    tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True, order="C"))
    tensor = tensor.permute(2, 0, 1).float().div_(255.0)
    return tensor * 2.0 - 1.0


def valid_cache(path: Path, sample_id: str, expected: dict) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError):
        return False
    return (
        payload.get("cache_format_version") == CACHE_FORMAT_VERSION
        and payload.get("sample_id") == sample_id
        and payload.get("codec") == expected
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    config = resolve_placeholders(config, placeholder_values(), strict=False)
    paths, data = config["paths"], config["data"]
    derived = resolve_derived(config)
    params = codec_params(config["codec"])
    codec_signature = {
        **params,
        "backend": config["codec"]["backend"],
        "flow_resolution": int(data["flow_resolution"]),
        "latent_frames": derived["latent_frames"],
    }

    if args.from_targets:
        # PhyCo has no manifest; the target cache is the corpus, and each target
        # carries its own video path and caption. Rows are synthesised so the
        # rest of this script is unchanged.
        rows = []
        for path in sorted(Path(paths["target_cache_dir"]).glob("*.pt")):
            if path.name.startswith("summary-"):
                continue
            payload = torch.load(path, map_location="cpu", weights_only=True)
            rows.append(SimpleNamespace(
                sample_id=payload["sample_id"],
                scenario=payload.get("scenario", ""),
                video_path=payload.get("video_path", ""),
                caption=payload.get("caption", ""),
            ))
        print(f"[data] {len(rows)} clips from {paths['target_cache_dir']}", flush=True)
    else:
        splits = args.split or list(SPLITS)
        rows = []
        for split in splits:
            rows.extend(read_jsonl(Path(paths["manifest_dir"]) / f"{split}.jsonl"))
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit:
        rows = rows[: args.limit]

    video_override: dict[str, str] = {}
    if args.video_override_manifest:
        with Path(args.video_override_manifest).open() as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    video_override[record["sample_id"]] = record["video_path"]
        overridden = sum(1 for row in rows if row.sample_id in video_override)
        print(f"[override] {overridden}/{len(rows)} first frames come from "
              f"{args.video_override_manifest}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 flow caching requires a CUDA GPU for the Wan VAE")
    device = torch.device("cuda")
    codec = build_codec(
        config["codec"],
        model_dir=paths["wan_model"],
        device=device,
        latent_size=derived["latent_size"],
    )

    output_dir = Path(paths["flow_cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = int(data["flow_resolution"])
    evaluation_side = int(data["height"])

    completed, skipped, missing_targets, errors = 0, 0, 0, []
    clipped: list[float] = []
    for row in tqdm(rows, desc=f"flowdit-cache {args.shard_index}/{args.num_shards}"):
        output = output_dir / f"{row.sample_id}.pt"
        if not args.overwrite and valid_cache(output, row.sample_id, codec_signature):
            skipped += 1
            continue
        target_path = (Path(paths["target_cache_dir"]) / f"{row.sample_id}.pt"
                       if args.from_targets
                       else cache_path(paths["target_cache_dir"], "targets", row.sample_id))
        if not target_path.is_file():
            # Either the sample failed the coverage gate, or the target cache is
            # still being rebuilt. Neither is a failure of this script, and
            # counting it as one would mask real errors -- rerun to fill in.
            missing_targets += 1
            continue
        try:
            target = torch.load(target_path, map_location="cpu", weights_only=True)
            flow = target["flow"].float()
            visibility = prepare_visibility(
                target["visibility"].float(), target["object_mask"].float()
            )
            scored = (target["object_mask"].float()[None] > 0.5) & (
                target["visibility"].float() > 0.5
            )
            clipped.append(clip_fraction(flow, scored, max_pixels=params["max_pixels"]))

            padded_flow, padded_visibility = prepend_zero_slot(flow, visibility)
            image = flow_to_image(
                padded_flow,
                padded_visibility,
                size=resolution,
                max_pixels=params["max_pixels"],
                gamma=params["gamma"],
                kind=params["kind"],
                eval_side=float(evaluation_side),
            )
            flow_latent = codec.encode(image).cpu().half()

            frame_source = video_override.get(row.sample_id, row.video_path)
            frame = read_first_frame(frame_source, resolution)
            frame_tensor = to_model_range(frame)[None].to(device)
            if isinstance(codec, WanVaeCodec):
                image_cond_latent = codec.vae.encode(
                    frame_tensor.unsqueeze(2)
                ).latent_dist.mode()[0].cpu().half()
            else:
                image_cond_latent = (
                    F.interpolate(
                        frame_tensor,
                        size=(derived["latent_size"], derived["latent_size"]),
                        mode="area",
                    )[0][:, None]
                    .cpu()
                    .half()
                )
            first_frame = cv2.resize(
                frame, (evaluation_side, evaluation_side), interpolation=cv2.INTER_AREA
            )
            first_frame_sha256 = sha256_file(frame_source)
            source_target_sha256 = sha256_file(target_path)
            target_first_frame_sha256 = target.get("provenance", {}).get(
                "first_frame_sha256"
            )
            if (
                target.get("label_source") == "bridge_real_raft_fb_bounded_v2"
                and target_first_frame_sha256 != first_frame_sha256
            ):
                raise ValueError(
                    f"{row.sample_id}: source first-frame hash no longer matches target provenance"
                )
            wan_input_identity_sha256 = hashlib.sha256(
                json.dumps(
                    {
                        "first_frame_sha256": first_frame_sha256,
                        "source_target_sha256": source_target_sha256,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()

            payload = {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "sample_id": row.sample_id,
                "scenario": row.scenario,
                "flow_latent": flow_latent,
                "image_cond_latent": image_cond_latent,
                "first_frame": torch.from_numpy(first_frame).permute(2, 0, 1).contiguous(),
                "codec": codec_signature,
                "horizon_seconds": derived["horizon_seconds"],
                "label_source": target.get("label_source", "unknown"),
                "reconstruction_coverage": float(target.get("reconstruction_coverage", 0.0)),
                # provenance of the RGB actually encoded; the dataset's _validate
                # cannot tell a substituted first frame from the render, this field can
                "first_frame_source": frame_source,
                "provenance": {
                    "source_target_sha256": source_target_sha256,
                    "first_frame_sha256": first_frame_sha256,
                    "wan_input_identity_sha256": wan_input_identity_sha256,
                    "teacher_flow_latent_semantics":
                        "ABI placeholder only; latent/FM loss must remain zero",
                    "codec_visibility_semantics":
                        "flow_valid inside content; flat visible backdrop only in padding",
                    "conditioning_sources": [
                        "saved first RGB frame",
                        "original task-language caption",
                        "first-frame SAM",
                    ],
                },
            }
            temporary = output.with_suffix(f".tmp-{os.getpid()}")
            torch.save(payload, temporary)
            temporary.replace(output)
            completed += 1
        except Exception as error:  # noqa: BLE001 - shard must survive a bad sample
            errors.append({"sample_id": row.sample_id, "error": repr(error)})
            print(f"ERROR {row.sample_id}: {error!r}", flush=True)

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "completed": completed,
        "skipped": skipped,
        "missing_targets": missing_targets,
        "errors": errors,
        "clip_fraction_mean": float(np.mean(clipped)) if clipped else 0.0,
        "clip_fraction_max": float(np.max(clipped)) if clipped else 0.0,
        "codec": codec_signature,
    }
    (log_dir / f"shard-{args.shard_index:04d}-of-{args.num_shards:04d}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {key: (len(value) if key == "errors" else value) for key, value in summary.items()},
            sort_keys=True,
        )
    )
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
