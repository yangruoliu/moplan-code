#!/usr/bin/env python
"""Pool-table direction data: Physics-IQ-style causal captions that name the struck ball and
its strike direction, exact mover IDs, and a caption-flip evaluation set.

Why: on PhyCo slides the motion direction is predictable from the layout (objects always
move toward the frame centre), so Stage-1 never had to read direction words.  Pool balls
are symmetric and 60% of strikes have a free direction, so the caption is the main cue.

Caption (no outcomes: no collisions, bounces, stops or final positions):
  "An overhead view of a pool table with 3 balls. The red ball is struck toward the upper
   left. Static shot with no camera movement."
  "... The yellow striped ball is struck toward the white cue ball. ..."
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

# Rendered appearance of the Kubric pool textures, verified from frame-0 pixels of 200
# samples (pool_ball_colors.py): standard numbering except that the 6 and 9 textures are
# swapped (6 renders as the yellow striped ball, 9 as the solid green ball).
BALL_DESC = {1: "yellow ball", 2: "blue ball", 3: "red ball", 4: "purple ball", 5: "orange ball",
             6: "yellow striped ball", 7: "brown ball", 8: "black ball", 9: "green ball",
             10: "blue striped ball", 11: "red striped ball", 12: "purple striped ball",
             13: "orange striped ball", 14: "green striped ball", 15: "brown striped ball"}
DIRS = ["right", "upper right", "top", "upper left", "left", "lower left", "bottom", "lower right"]
OPPOSITE = {d: DIRS[(i + 4) % 8] for i, d in enumerate(DIRS)}


def ball_desc(name: str) -> str:
    if name.startswith("white"):
        return "white cue ball"
    return BALL_DESC[int(name.split("_")[0])]


def screen_direction(dx: float, dy: float) -> str:
    angle = math.degrees(math.atan2(-dy, dx)) % 360  # image y points down
    return DIRS[int(((angle + 22.5) % 360) // 45)]


def caption_for(n_balls: int, struck: str, direction_phrase: str) -> str:
    return (f"An overhead view of a pool table with {n_balls} balls. The {struck} is struck toward "
            f"{direction_phrase}. Static shot with no camera movement.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", required=True,
                        help="PhyCo manifest directory with train.jsonl, val.jsonl and test_core.jsonl")
    parser.add_argument("--targets", required=True, help="PhyCo target cache (cache_phyco_targets.py output)")
    parser.add_argument("--flowdit", required=True, help="PhyCo Stage-1 flow cache (cache_stage1_flowdit.py output)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=6000)
    parser.add_argument("--mover-px", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    out = Path(args.output_dir)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "flip_eval" / "frames").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    movers, stats = {}, Counter()
    flip_rows, flip_gt = [], {}
    for split in ("train", "val", "test_core"):
        rows = [json.loads(l) for l in (Path(args.manifests) / f"{split}.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r["scenario"] == "pool_table_force"
                and (Path(args.targets) / f"{r['sample_id']}.pt").exists()
                and (Path(args.flowdit) / f"{r['sample_id']}.pt").exists()]
        if split == "train" and args.train_limit and len(rows) > args.train_limit:
            rows = rng.sample(rows, args.train_limit)
        kept = []
        for row in rows:
            sid = row["sample_id"]
            video = Path(row["video_path"])
            meta = json.loads((video.parent / "metadata.json").read_text())
            names = meta["object_data"]["object_name"]
            seg_ids = meta["object_data"]["segmentation_id"]
            kinds = meta["object_data"]["type"]
            ball_ids = {n: i for n, i, k in zip(names, seg_ids, kinds) if k == "pool_ball"}
            struck = meta.get("force_applied_ball")
            force = (meta.get("applied_forces_image") or [None])[0]
            if struck not in ball_ids or not force:
                stats["skip_no_force"] += 1
                continue
            dx = force["force_end_image_coordinates"][0] - force["image_coordinates"][0]
            dy = force["force_end_image_coordinates"][1] - force["image_coordinates"][1]
            if math.hypot(dx, dy) < 1e-3:
                stats["skip_zero_force"] += 1
                continue
            mode = meta.get("force_application_mode")
            target = meta.get("force_target_ball")
            if mode == "towards_ball" and target in ball_ids:
                phrase = f"the {ball_desc(target)}"
            else:
                mode = "free_direction"
                phrase = f"the {screen_direction(dx, dy)}"
            caption = caption_for(len(ball_ids), ball_desc(struck), phrase)
            payload = torch.load(Path(args.targets) / f"{sid}.pt", map_location="cpu", weights_only=True)
            instance = payload["instance_id"][0].long()
            flow = payload["flow"].float()  # [7,2,160,160] grid px
            magnitude = flow.norm(dim=1)  # [7,160,160]
            moving_ids = []
            for name, identifier in ball_ids.items():
                region = instance == identifier
                if region.sum() < 1:
                    continue
                q75 = torch.quantile(magnitude[:, region], 0.75, dim=1).max().item()
                if q75 >= args.mover_px:
                    moving_ids.append(int(identifier))
            if not moving_ids:
                stats["skip_no_visible_mover"] += 1
                continue
            movers[sid] = {"sample_id": sid, "scenario": row["scenario"], "mover_instance_ids": sorted(moving_ids),
                           "struck_ball_id": int(ball_ids[struck]), "strike_mode": mode}
            kept.append({**row, "caption": caption, "split": split, "source": "phyco_kubric_pool_direction_v1"})
            stats[f"{split}_{mode}"] += 1
            if split in ("val", "test_core") and mode == "free_direction":
                struck_region = instance == ball_ids[struck]
                if struck_region.sum() >= 1:
                    capture = cv2.VideoCapture(str(video))
                    ok, frame = capture.read()
                    capture.release()
                    if ok:
                        frame_path = out / "flip_eval" / "frames" / f"{sid}.png"
                        cv2.imwrite(str(frame_path), frame)
                        direction = screen_direction(dx, dy)
                        variants = {"orig": caption,
                                    "flip": caption_for(len(ball_ids), ball_desc(struck), f"the {OPPOSITE[direction]}"),
                                    "nodir": (f"An overhead view of a pool table with {len(ball_ids)} balls. The "
                                              f"{ball_desc(struck)} is struck. Static shot with no camera movement.")}
                        for name, text in variants.items():
                            flip_rows.append({"sample_id": f"{sid}__{name}", "first_frame_path": str(frame_path), "caption": text})
                        ys, xs = torch.nonzero(struck_region, as_tuple=True)
                        true = flow[:, :, ys, xs].mean(-1).tolist()  # [7,2] grid px, struck ball
                        flip_gt[sid] = {"split": split, "caption": caption, "direction": direction,
                                        "force_image_vector": [dx, dy], "true_disp_grid_px": true,
                                        "brick_pixels_grid": [xs.tolist(), ys.tolist()],
                                        "scenario": "pool_table_force"}
        with (out / "manifests" / f"{split}.jsonl").open("w") as handle:
            for row in kept:
                handle.write(json.dumps(row) + "\n")
    (out / "mover_ids.json").write_text(json.dumps({"schema": "stage1_supervision_mover_ids_v1",
                                                    "rows": len(movers), "mapping": movers,
                                                    "source": "rendered_instance_id_plus_target_flow_q75_ge_1px"}))
    with (out / "flip_eval" / "infer_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "first_frame_path", "caption"])
        writer.writeheader()
        writer.writerows(flip_rows)
    base = {}
    for row in flip_rows:
        base.setdefault(row["sample_id"].rsplit("__", 1)[0], row["first_frame_path"])
    with (out / "flip_eval" / "base_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "first_frame_path", "caption"])
        writer.writeheader()
        for sid, path in base.items():
            writer.writerow({"sample_id": sid, "first_frame_path": path, "caption": flip_gt[sid]["caption"]})
    (out / "flip_eval" / "gt.json").write_text(json.dumps(flip_gt))
    print(json.dumps(dict(stats), indent=1))
    print("flip-eval samples", len(flip_gt))
    for line in (out / "manifests" / "val.jsonl").read_text().splitlines()[:4]:
        print(json.loads(line)["caption"])


if __name__ == "__main__":
    main()
