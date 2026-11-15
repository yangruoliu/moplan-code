#!/usr/bin/env python
"""Name each pool ball by its rendered colour (not by its number).

The Kubric pool textures do not follow standard numbering for every ball (e.g. 6 renders
pale yellow, 9 green, 11 orange), so captions built from the number named the wrong
ball for ~3/16 of the balls.  This reads frame 0 of rgba.mp4 and segmentation.mp4 at native
resolution, takes each ball's exact mask, and names it from its pixels:
  white fraction > 0.75            -> "white cue ball"
  dark (v < 0.22) majority          -> "black ball"
  else hue of saturated pixels     -> red/orange/yellow/green/blue/purple/maroon
  white fraction in [0.2, 0.75]    -> "<colour> striped ball"
Output: {sample_id: {ball_name: description}}
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def hue_name(hue: float, value: float) -> str:
    if hue >= 345 or hue < 15:
        return "maroon" if value < 0.45 else "red"
    if hue < 40:
        return "maroon" if value < 0.42 else "orange"
    if hue < 70:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 260:
        return "blue"
    if hue < 320:
        return "purple"
    return "red"


def describe(pixels_rgb: np.ndarray) -> str:
    hsv = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV_FULL).reshape(-1, 3) / 255.0
    hue, sat, val = hsv[:, 0] * 360, hsv[:, 1], hsv[:, 2]
    white = (sat < 0.22) & (val > 0.72)
    dark = val < 0.22
    white_fraction = float(white.mean())
    if white_fraction > 0.75:
        return "white cue ball"
    if float(dark.mean()) > 0.5:
        return "black ball"
    coloured = (sat > 0.35) & (val > 0.22)
    if coloured.sum() < 3:
        return "white cue ball" if white_fraction > 0.5 else "black ball"
    # circular mean of hue
    angle = np.deg2rad(hue[coloured])
    mean_hue = float(np.rad2deg(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())) % 360)
    name = hue_name(mean_hue, float(np.median(val[coloured])))
    return f"{name} striped ball" if white_fraction >= 0.2 else f"{name} ball"


def one(row: dict) -> tuple[str, dict]:
    video = Path(row["video_path"])
    meta = json.loads((video.parent / "metadata.json").read_text())
    objects = meta["object_data"]
    rgb = cv2.VideoCapture(str(video)).read()[1]
    seg = cv2.VideoCapture(str(video.parent / "segmentation.mp4")).read()[1]
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB).astype(np.int16)
    out = {}
    for name, kind, colour in zip(objects["object_name"], objects["type"], objects["segmentation_color"]):
        if kind != "pool_ball":
            continue
        mask = np.abs(seg - np.asarray(colour, np.int16)).sum(-1) < 40
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if mask.sum() < 8:
            continue
        out[name] = describe(rgb[mask])
    return row["sample_id"], out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    rows = []
    for split in ("train", "val", "test_core"):
        rows += [json.loads(l) for l in (Path(args.manifest_dir) / f"{split}.jsonl").read_text().splitlines() if l.strip()]
    with ProcessPoolExecutor(args.workers) as pool:
        result = dict(pool.map(one, rows, chunksize=16))
    Path(args.output).write_text(json.dumps(result))
    from collections import Counter
    by_number = {}
    for balls in result.values():
        for name, desc in balls.items():
            by_number.setdefault(name, Counter())[desc] += 1
    for name in sorted(by_number, key=lambda n: (len(n), n)):
        print(name, by_number[name].most_common(3))


if __name__ == "__main__":
    main()
