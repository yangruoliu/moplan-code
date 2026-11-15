#!/usr/bin/env python
"""Build Stage-1/Stage-2 motion targets from PhyCo Kubric clips.

PhyCo ships rendered instance segmentation and per-frame object poses but no
optical flow: the generator ran Kubric with ``layers=image,segmentation,depth``
and left ``forward_flow`` off. So the flow here is RAFT pseudo-ground-truth,
while the *decomposition* is exact -- which is the half that matters most for a
per-object affine (connected components are a poor substitute).

Emitted per clip, matching the Physion target cache so downstream code is
unchanged:

  flow          [T, 2, S, S] pixels, first frame to each horizon, cumulative
  visibility    [T, 1, S, S] RAFT forward-backward consistency
  object_mask   [1, S, S]    union of instances that move
  future_mask   [T, 1, S, S] object_mask splatted along the flow
  instance_id   [1, S, S]    exact, decoded from segmentation.mp4
  affine_2d     [K, T, 6]    least squares per instance per horizon
  affine_3d     [T, 3, 4]    rigid transform of the tracked object, when poses
                             are present -- rotation and translation directly
                             from animation_data.pkl, no estimation involved

``affine_3d`` covers only the object the pkl tracks (one per clip), because that
is all the dataset records. Clips without poses simply omit it rather than
substituting an estimate that would look like ground truth downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_jepa.flow_dit_slots import fit_instance_affine
from motion_jepa.video import forward_splat_mask
from scripts.cache_stage1 import load_raft

SIDE = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache PhyCo motion targets")
    parser.add_argument("--root", required=True,
                        help="extracted PhyCo clips: <root>/<scenario>/<date>/<hash>/{metadata.json,rgba.mp4,segmentation.mp4}")
    parser.add_argument("--output", required=True, help="target cache directory (<sample_id>.pt)")
    parser.add_argument("--horizons", type=int, default=7)
    parser.add_argument("--span-seconds", type=float, default=4.0)
    parser.add_argument("--source-fps", type=float, default=24.0)
    parser.add_argument("--raft-side", type=int, default=384,
                        help="RAFT runs here then the flow is resampled to --side; "
                             "measured on Physion, 160px doubles the error vs 512px")
    parser.add_argument("--side", type=int, default=SIDE)
    parser.add_argument("--moving-threshold-px", type=float, default=2.0)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--per-scenario", type=int, default=0,
        help="cap clips per scenario before sharding. friction_slide_flat_v2 has "
             "30k clips against pool_table's 15k and ball_drop_v3's 10k; without "
             "a cap the corpus is dominated by one kind of physics",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--clip-list",
        default="",
        help="path to a pre-built newline-separated list of metadata.json paths "
             "(from `find <root> -mindepth 4 -maxdepth 4 -name metadata.json`). "
             "Skips the slow per-clip Lustre discovery; each shard reads it.",
    )
    parser.add_argument("--caption-root", required=True,
                        help="raw PhyCo download with <scenario>/common_caption_cosmos.txt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--ids-file", default="",
        help="optional newline-separated sample_id list; only matching clips are "
             "cached (applied before --limit/sharding). Needed because --limit "
             "takes a scenario-sorted prefix and cannot target specific clips.",
    )
    return parser.parse_args()


def read_video(path: Path, side: int = 0) -> np.ndarray:
    """Decode a clip; ``side=0`` keeps the native size.

    PhyCo renders 768x432. Squashing that into a square compresses horizontal
    detail 1.78x more than vertical, which measurably hurt RAFT on the small
    fast pool-table balls. Aspect ratio is preserved everywhere now.
    """
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if side:
            height, width = frame.shape[:2]
            scale = side / max(height, width)
            frame = cv2.resize(frame, (round(width * scale), round(height * scale)),
                               interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return np.stack(frames)


def letterbox(value: torch.Tensor, side: int) -> torch.Tensor:
    """Resize [N,C,H,W] to fit inside side x side, centred, zero padded.

    Keeps the square target grid the rest of the codebase assumes while leaving
    the motion field undistorted, so a pixel of flow means the same thing on
    both axes.
    """
    _, _, height, width = value.shape
    scale = side / max(height, width)
    inner_h, inner_w = round(height * scale), round(width * scale)
    resized = F.interpolate(value, size=(inner_h, inner_w), mode="bilinear", align_corners=False)
    top, left = (side - inner_h) // 2, (side - inner_w) // 2
    out = value.new_zeros(value.shape[0], value.shape[1], side, side)
    out[:, :, top: top + inner_h, left: left + inner_w] = resized
    return out


def letterbox_nearest(value: torch.Tensor, side: int) -> torch.Tensor:
    _, _, height, width = value.shape
    scale = side / max(height, width)
    inner_h, inner_w = round(height * scale), round(width * scale)
    resized = F.interpolate(value.float(), size=(inner_h, inner_w), mode="nearest")
    top, left = (side - inner_h) // 2, (side - inner_w) // 2
    out = value.new_zeros(value.shape[0], value.shape[1], side, side).float()
    out[:, :, top: top + inner_h, left: left + inner_w] = resized
    return out


def decode_instances(segmentation: np.ndarray, colour_map: dict) -> np.ndarray:
    """Nearest-palette assignment; mp4 compression bleeds edges by ~1% of pixels."""
    identifiers = [0] + sorted(int(k) for k in colour_map)
    palette = np.stack([np.zeros(3)] + [np.asarray(colour_map[str(i)]) for i in identifiers[1:]])
    flat = segmentation.reshape(-1, 3).astype(np.int16)
    distance = np.linalg.norm(flat[:, None, :] - palette[None], axis=2)
    nearest = distance.argmin(1)
    return np.asarray(identifiers)[nearest].reshape(segmentation.shape[:2]).astype(np.int16)


def cumulative_flow(raft, transform, frames: np.ndarray, marks: list[int],
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255).to(device)
    height, width = video.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    accumulated = torch.zeros(2, height, width, device=device)
    consistent = torch.ones(1, height, width, device=device)
    flows, visibility = {}, {}
    for index in range(1, max(marks) + 1):
        earlier, later = transform(video[index - 1: index], video[index: index + 1])
        with torch.no_grad():
            forward = raft(earlier, later, num_flow_updates=12)[-1][0]
            backward = raft(later, earlier, num_flow_updates=12)[-1][0]
        step_grid = torch.stack(
            (2 * (xx + forward[0]) / max(1, width - 1) - 1,
             2 * (yy + forward[1]) / max(1, height - 1) - 1), dim=-1)
        returned = F.grid_sample(backward[None], step_grid[None], mode="bilinear",
                                 padding_mode="border", align_corners=True)[0]
        error = (forward + returned).square().sum(0).sqrt()
        step_ok = (error <= 1.0 + 0.05 * forward.square().sum(0).sqrt()).float()[None]

        warp = torch.stack(
            (2 * (xx + accumulated[0]) / max(1, width - 1) - 1,
             2 * (yy + accumulated[1]) / max(1, height - 1) - 1), dim=-1)
        accumulated = accumulated + F.grid_sample(
            forward[None], warp[None], mode="bilinear", padding_mode="border", align_corners=True)[0]
        consistent = consistent * F.grid_sample(
            step_ok[None], warp[None], mode="nearest", padding_mode="border", align_corners=True)[0]
        if index in marks:
            flows[index] = accumulated.clone()
            visibility[index] = consistent.clone()
    return (torch.stack([flows[i] for i in marks]),
            torch.stack([visibility[i] for i in marks]))


def rigid_transforms(track: dict, marks_seconds: list[float], fps: float) -> torch.Tensor | None:
    """Frame-0-relative rigid transform per horizon, from the recorded poses.

    Returned as ``[T, 3, 4]``: rows are ``[R | t]`` in world coordinates. This is
    the dataset's own simulation state, so unlike the 2D affine it involves no
    fitting and no estimator.
    """
    positions = np.asarray(track["position"], dtype=np.float64)
    quaternions = np.asarray(track["quaternion"], dtype=np.float64)
    if len(positions) < 2:
        return None

    def matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    rotation0 = matrix(quaternions[0])
    out = []
    for seconds in marks_seconds:
        index = int(np.clip(round(seconds * fps), 0, len(positions) - 1))
        rotation = matrix(quaternions[index]) @ rotation0.T
        translation = positions[index] - rotation @ positions[0]
        out.append(np.concatenate([rotation, translation[:, None]], axis=1))
    return torch.from_numpy(np.stack(out)).float()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("RAFT needs CUDA")
    raft, transform = load_raft(device)

    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Skip hidden dirs (e.g. a stray .cache/ left by hf download) so they are
    # not mistaken for scenarios.
    scenarios = args.scenario or sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    clips = []
    if args.clip_list:
        # Pre-built discovery: one C-level `find <root> -mindepth 4 -maxdepth 4
        # -name metadata.json | sort` lists 95k clips in ~100s, versus Python
        # per-clip stat globbing that never finishes on Lustre. Each shard reads
        # the same list instead of re-walking the tree.
        scenario_set = set(scenarios) if args.scenario else None
        by_scenario: dict[str, list] = {}
        for line in Path(args.clip_list).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            meta = Path(line)
            scenario = meta.relative_to(root).parts[0]
            if scenario_set is not None and scenario not in scenario_set:
                continue
            by_scenario.setdefault(scenario, []).append((scenario, meta.parent))
        for scenario in sorted(by_scenario):
            found = by_scenario[scenario]
            if args.per_scenario and len(found) > args.per_scenario:
                found = found[:: len(found) // args.per_scenario][: args.per_scenario]
            clips.extend(found)
            print(f"  {scenario}: {len(found)} clips", flush=True)
    else:
        for scenario in scenarios:
            found = []
            # Clips live at a fixed depth <scenario>/<date>/<hash>/metadata.json.
            # A fixed-depth glob stats only that level; rglob walks the whole
            # subtree and is pathologically slow on Lustre. Fall back to rglob for
            # any scenario whose layout differs. Prefer --clip-list at scale.
            metas = sorted((root / scenario).glob("*/*/metadata.json"))
            if not metas:
                metas = sorted((root / scenario).rglob("metadata.json"))
            for meta in metas:
                if (meta.parent / "rgba.mp4").is_file() and (meta.parent / "segmentation.mp4").is_file():
                    found.append((scenario, meta.parent))
            if args.per_scenario and len(found) > args.per_scenario:
                # Stride rather than truncate: clips are grouped by render date,
                # and a prefix would sample one day's parameter sweep only.
                found = found[:: len(found) // args.per_scenario][: args.per_scenario]
            clips.extend(found)
            print(f"  {scenario}: {len(found)} clips", flush=True)
    if args.ids_file:
        wanted = {line.strip() for line in open(args.ids_file) if line.strip()}
        clips = [
            (scenario, clip)
            for scenario, clip in clips
            if f"phyco_{scenario}_{clip.parent.name}_{clip.name}" in wanted
        ]
        print(f"[ids-file] kept {len(clips)}/{len(wanted)} requested clips", flush=True)
    if args.limit:
        clips = clips[: args.limit]
    clips = clips[args.shard_index :: args.num_shards]
    print(f"{len(clips)} clips on shard {args.shard_index}/{args.num_shards}", flush=True)

    # Captions are per scenario, not per clip: the newer archives dropped the
    # per-clip caption.txt and the text lives in common_caption_cosmos.txt
    # beside the archives. Reading it once avoids writing 100k empty prompts.
    captions: dict[str, str] = {}
    borrowed: set[str] = set()
    caption_root = Path(args.caption_root)
    for scenario in scenarios:
        shared = caption_root / scenario / "common_caption_cosmos.txt"
        if shared.is_file():
            captions[scenario] = shared.read_text().strip()
        elif sorted((root / scenario).glob("*/*/caption.txt"))[:1]:
            captions[scenario] = ""          # read per clip below
        else:
            # ball_drop_v3 ships no caption of its own; it is a re-render of the
            # same scenario family, so borrow the sibling's text rather than
            # training on an empty prompt. Recorded in caption_source so the
            # substitution is not mistaken for the dataset's own label.
            family = scenario.rsplit("_v", 1)[0]
            sibling = next((caption_root / n / "common_caption_cosmos.txt"
                            for n in sorted(p.name for p in caption_root.iterdir() if p.is_dir())
                            if n.rsplit("_v", 1)[0] == family
                            and (caption_root / n / "common_caption_cosmos.txt").is_file()), None)
            if sibling is not None:
                captions[scenario] = sibling.read_text().strip()
                borrowed.add(scenario)
                print(f"[caption] {scenario}: borrowed from {sibling.parent.name}", flush=True)
            else:
                print(f"[warn] {scenario}: no caption source", flush=True)
                captions[scenario] = ""

    step = args.span_seconds / args.horizons
    marks_seconds = [round((i + 1) * step, 4) for i in range(args.horizons)]
    written = skipped = failed = 0

    for scenario, clip in clips:
        sample_id = f"phyco_{scenario}_{clip.parent.name}_{clip.name}"
        destination = output / f"{sample_id}.pt"
        if destination.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            caption = captions.get(scenario, "")
            if not caption and (clip / "caption.txt").is_file():
                caption = (clip / "caption.txt").read_text().strip()
            meta = json.loads((clip / "metadata.json").read_text())
            colour_map = meta["segmentation_color_map"]
            frames = read_video(clip / "rgba.mp4", args.raft_side)  # aspect preserved
            marks = [int(np.clip(round(s * args.source_fps), 1, len(frames) - 1))
                     for s in marks_seconds]
            if len(set(marks)) < args.horizons or max(marks) >= len(frames):
                skipped += 1
                continue

            flow_big, visible_big = cumulative_flow(raft, transform, frames, marks, device)
            # Displacements are in pixels of the grid they were measured on, so
            # the letterbox rescale has to be applied to the values too.
            scale = args.side / max(frames.shape[1], frames.shape[2])
            flow = (letterbox(flow_big, args.side) * scale).cpu()
            visible = (letterbox_nearest(visible_big.float(), args.side) > 0.5).cpu()

            segmentation = read_video(clip / "segmentation.mp4", args.raft_side)
            decoded = decode_instances(segmentation[0], colour_map)
            instance = letterbox_nearest(
                torch.from_numpy(decoded).float()[None, None], args.side
            )[0].round().short()

            magnitude = flow.square().sum(1).sqrt()
            moving = torch.zeros(args.side, args.side, dtype=torch.bool)
            for identifier in [int(v) for v in instance.unique() if int(v) > 0]:
                belongs = instance[0] == identifier
                if belongs.sum() < 12:
                    continue
                if float(magnitude[:, belongs].amax(0).mean()) >= args.moving_threshold_px:
                    moving |= belongs
            if not moving.any():
                skipped += 1
                continue

            object_mask = moving[None]
            valid = object_mask[None].float() * visible.float()
            future_mask = forward_splat_mask(object_mask[None].float(), flow[None])[0] > 0.25
            affine_2d, ids = fit_instance_affine(
                flow, instance * moving[None].long(), valid, eval_side=float(args.side))

            payload = {
                "cache_format_version": 1,
                "kind": "targets",
                "sample_id": sample_id,
                "scenario": scenario,
                "flow": flow.half(),
                "visibility": visible,
                "object_mask": object_mask,
                "future_mask": future_mask,
                "instance_id": instance,
                "affine_2d": affine_2d.half(),
                "affine_2d_instance_ids": ids,
                "horizon_seconds": marks_seconds,
                "source_fps": args.source_fps,
                "flow_source": f"raft_large_at_{args.raft_side}px",
                "instance_source": "kubric_rendered_segmentation_exact",
                "label_source": "phyco_kubric_raft_flow_plus_rendered_instance_segmentation",
                "caption": caption,
                "caption_source": ("borrowed_sibling_scenario" if scenario in borrowed
                                   else "phyco_common_caption_cosmos"),
                "video_path": str(clip / "rgba.mp4"),
            }

            animation = clip / "animation_data.pkl"
            if animation.is_file():
                tracks = pickle.load(animation.open("rb"))
                if tracks:
                    rigid = rigid_transforms(next(iter(tracks.values())), marks_seconds,
                                             args.source_fps)
                    if rigid is not None:
                        payload["affine_3d"] = rigid
                        payload["affine_3d_source"] = "animation_data_pkl_exact_single_object"

            temporary = destination.with_suffix(f".tmp-{os.getpid()}")
            torch.save(payload, temporary)
            temporary.replace(destination)
            written += 1
            if written % 100 == 0:
                print(f"  {written} written, {skipped} skipped, {failed} failed", flush=True)
        except Exception as error:                                  # noqa: BLE001
            failed += 1
            if failed <= 5:
                print(f"[fail] {sample_id}: {error!r}", flush=True)

    summary = {"written": written, "skipped": skipped, "failed": failed,
               "shard": args.shard_index, "num_shards": args.num_shards,
               "horizon_seconds": marks_seconds}
    (output / f"summary-shard{args.shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
