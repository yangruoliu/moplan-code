#!/usr/bin/env python
"""Clip timing of real Stage-2 v2 cache entries (<cache>/<sid>.pt) as one JSON line per clip.

frames_real is the number of REAL frames of the clip at `fps` (a clip at rest is padded after it with its last frame;
padding is not observation, so the Stage-1 builder ends the labels at (frames_real - 1) / fps).
usage: stage2_cache_meta.py --cache DIR --ids F [F ...] --out META.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--ids", required=True, nargs="+")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    ids = []
    for path in a.ids:
        ids += [l.strip() for l in open(path) if l.strip()]
    missing = 0
    with open(a.out, "w") as handle:
        for sid in dict.fromkeys(ids):
            path = a.cache / f"{sid}.pt"
            if not path.is_file():
                missing += 1
                continue
            x = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            scene = json.loads(x.get("real_scene") or "{}")
            handle.write(json.dumps({"sample_id": sid, "frames": int(x["frames"]), "frames_real": int(x["frames_real"]),
                                     "at_rest": bool(x.get("at_rest", False)), "fps": float(x["fps"]),
                                     "frame_seconds": list(x["frame_seconds"]), "label_source": x.get("label_source", ""),
                                     "camera_max_px": float(scene.get("camera_max_px", float("nan")))}) + "\n")
    print(json.dumps({"ids": len(dict.fromkeys(ids)), "missing": missing, "out": str(a.out)}))


if __name__ == "__main__":
    main()
