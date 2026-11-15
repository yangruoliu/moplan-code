#!/usr/bin/env python
"""SAM2 automatic instance masks for real footage, in the Stage-1 input format.

The joint Stage-1 was trained with exact simulator instance masks steering its
slot pooling and rendering. Benchmarks provide those; real footage does not, so
deployment substitutes SAM2's automatic mask generator. Output matches
``build_instance_inputs`` exactly: ``instance_masks [K, side, side]`` (binary,
largest first, spare slots empty), ``instance_channels [2, side, side]`` (union
+ boundary), letterboxed to the square grid the model saw in training.

Deployment cleanup keeps masks non-overlapping (higher-confidence mask wins a
contested pixel), drops tiny fragments, and removes near-full-frame proposals.
The area cutoff is a deployment heuristic, not a universal training prior:
PhyCo oracle instance labels also include large environment domes/platforms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_jepa.flow_dit_data import build_instance_inputs
from motion_jepa.flow_dit_bridge import BRIDGE_MASK_SOURCE


def letterbox_square(image: np.ndarray, side: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    height, width = image.shape[:2]
    scale = side / max(height, width)
    resized = cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=interpolation
    )
    # cv2.resize drops a trailing singleton channel; restore it so the canvas
    # assignment below broadcasts.
    if image.ndim == 3 and resized.ndim == 2:
        resized = resized[..., None]
    canvas = np.zeros((side, side, image.shape[2]) if image.ndim == 3 else (side, side),
                      dtype=image.dtype)
    top = (side - resized.shape[0]) // 2
    left = (side - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def conditioning_from_labels(ids: np.ndarray, side: int, slots: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize categorical labels once, then use the exact training input builder.

    Interpolating label numbers with AREA can invent instance IDs and produce a
    boundary channel inconsistent with separately resized binary slot masks.
    Nearest labels preserve instance identity; deriving both inputs from this
    partition also sorts slots by their actual area on the final grid.
    """
    boxed = letterbox_square(ids[..., None], side, interpolation=cv2.INTER_NEAREST)[..., 0]
    ids_t = torch.from_numpy(boxed.astype(np.int64))
    masks, _, channels = build_instance_inputs(ids_t[None], slots)
    return masks, channels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 masks for Stage-1 deployment")
    parser.add_argument("--manifest", required=True, help="CSV (sample_id, first_frame_path) or JSONL manifest")
    parser.add_argument("--sam-checkpoint", default=os.environ.get("SAM2_CHECKPOINT", ""),
                        help="SAM 2.1 hiera-large checkpoint (default: $SAM2_CHECKPOINT)")
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--side", type=int, default=160)
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--min-area-fraction", type=float, default=0.001)
    parser.add_argument("--max-area-fraction", type=float, default=0.6,
                        help="drop background-plate masks covering most of the frame")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--flow-cache-dir", default="",
        help="optional FlowDiT cache containing first_frame tensors; enables JSONL manifests",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compact-labels",
        action="store_true",
        help="store one uint8 label map and rebuild masks/channels in the loader",
    )
    parser.add_argument("--receipt", default="")
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_first_frame(row: dict, flow_cache_dir: Path | None) -> tuple[np.ndarray, str]:
    """Read the exact first-frame tensor used by Stage1 when available."""
    if flow_cache_dir is None:
        path = Path(row["first_frame_path"])
        encoded = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode first frame: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), hashlib.sha256(encoded).hexdigest()
    path = flow_cache_dir / f"{row['sample_id']}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    frame = payload.get("first_frame")
    if (
        not isinstance(frame, torch.Tensor)
        or frame.ndim != 3
        or frame.shape[0] != 3
        or not torch.isfinite(frame).all()
        or frame.min() < 0
        or frame.max() > 255
    ):
        raise ValueError(f"{path}: invalid cached first_frame")
    frame = frame.detach().cpu().to(torch.uint8).contiguous()
    digest = hashlib.sha256(frame.numpy().tobytes()).hexdigest()
    return frame.permute(1, 2, 0).numpy(), digest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.sam_checkpoint:
        raise ValueError("pass --sam-checkpoint or set SAM2_CHECKPOINT")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.points_per_side < 1 or args.points_per_batch < 1 or args.crop_n_layers < 0:
        raise ValueError("invalid SAM2 automatic-mask sampling arguments")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    model = build_sam2(args.sam_config, args.sam_checkpoint, device=device)
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        crop_n_layers=args.crop_n_layers,
    )

    manifest = Path(args.manifest)
    rows = read_rows(manifest)
    total_rows = len(rows)
    rows = rows[args.shard_index :: args.num_shards]
    if args.limit:
        rows = rows[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_cache_dir = Path(args.flow_cache_dir) if args.flow_cache_dir else None

    side, slots = args.side, args.num_slots
    started = time.monotonic()
    done = skipped = 0
    instance_counts: list[int] = []
    for row in rows:
        output = output_dir / f"{row['sample_id']}.pt"
        if output.is_file() and not args.overwrite:
            skipped += 1
            continue
        image, first_frame_sha256 = read_first_frame(row, flow_cache_dir)
        proposals = generator.generate(image)
        area = image.shape[0] * image.shape[1]
        proposals = [
            p for p in proposals
            if args.min_area_fraction * area <= p["area"] <= args.max_area_fraction * area
        ]
        # Higher predicted IoU wins contested pixels; then keep the largest K.
        proposals.sort(key=lambda p: -float(p.get("predicted_iou", 0.0)))
        claimed = np.zeros(image.shape[:2], dtype=bool)
        cleaned = []
        for p in proposals:
            mask = np.asarray(p["segmentation"], dtype=bool) & ~claimed
            if mask.sum() < args.min_area_fraction * area:
                continue
            claimed |= mask
            cleaned.append(mask)
        cleaned.sort(key=lambda m: -int(m.sum()))
        cleaned = cleaned[:slots]

        ids = np.zeros(image.shape[:2], dtype=np.int16)
        for index, mask in enumerate(cleaned):
            ids[mask] = index + 1
        masks, channels = conditioning_from_labels(ids, side, slots)

        temporary = output.with_suffix(f".tmp-{os.getpid()}")
        payload = {
            "sample_id": row["sample_id"],
            "num_instances": int(masks.flatten(1).any(1).sum()),
            "source": "sam2.1_hiera_large_automatic+nearest_labels_v2",
            "condition_mask_source": BRIDGE_MASK_SOURCE,
            "first_frame_sha256": first_frame_sha256,
        }
        if args.compact_labels:
            foreground = masks.bool().any(0)
            labels = masks.argmax(0).add(1).to(torch.uint8)
            labels.masked_fill_(~foreground, 0)
            payload["instance_labels"] = labels
            payload["storage_format"] = "uint8_instance_labels_v1"
        else:
            payload["instance_masks"] = masks.to(torch.uint8)
            payload["instance_channels"] = channels.to(torch.uint8)
            payload["storage_format"] = "uint8_masks_and_channels_v1"
        torch.save(payload, temporary)
        temporary.replace(output)
        done += 1
        instance_counts.append(int(masks.flatten(1).any(1).sum()))
        if done % 20 == 0:
            print(f"  {done} done", flush=True)
    elapsed = time.monotonic() - started
    receipt = {
        "schema": "stage1_sam2_condition_shard_v1",
        "manifest": manifest.name,                                      # file names plus sha256, no machine paths
        "manifest_sha256": file_sha256(manifest),
        "total_manifest_rows": total_rows,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_rows": len(rows),
        "done": done,
        "skipped": skipped,
        "elapsed_seconds": elapsed,
        "seconds_per_generated_sample": elapsed / done if done else None,
        "sam_checkpoint": Path(args.sam_checkpoint).name,
        "sam_checkpoint_sha256": file_sha256(Path(args.sam_checkpoint)),
        "sam_config": args.sam_config,
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "crop_n_layers": args.crop_n_layers,
        "side": side,
        "num_slots": slots,
        "min_area_fraction": args.min_area_fraction,
        "max_area_fraction": args.max_area_fraction,
        "instance_count_min": min(instance_counts) if instance_counts else None,
        "instance_count_max": max(instance_counts) if instance_counts else None,
        "instance_count_mean": sum(instance_counts) / len(instance_counts) if instance_counts else None,
        "storage_format": (
            "uint8_instance_labels_v1" if args.compact_labels
            else "uint8_masks_and_channels_v1"
        ),
    }
    receipt_path = Path(args.receipt) if args.receipt else output_dir / f"receipt_shard_{args.shard_index:04d}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt_path)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
