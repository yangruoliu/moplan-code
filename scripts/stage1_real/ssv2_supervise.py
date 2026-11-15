#!/usr/bin/env python
"""Which instances of a Something-Something v2 scene record may be supervised on.

SSv2 clips are hand-held and hand-driven: besides the objects named in the label, SAM2 instances of the arm, the
sleeve, the table edge or the background drift with the hand and the camera, and the measured plan calls them movers.
Only the objects that Something-Else annotates at the conditioning frame (its placeholder boxes, not the hand) are
trusted: an instance is supervised when at least --contain of its first-frame mask lies inside one of those boxes.
Every other instance becomes unknown (data.SceneSet sets known = False where `supervise` is False), neither mover nor
static.  The masks are the reward scene's work-frame masks (<cache>/scenes/<sid>.npz, 448 x 256 = the clip canvas),
matched to the record by slot id; the boxes are the manifest's boxes_t0_canvas.

writes `supervise` [N] bool into every <physplan>/<sid>.npz (other keys unchanged) and prints one SSV2_SUPERVISE line.
usage: ssv2_supervise.py --physplan DIR(<cache>/ssv2) --scenes DIR --manifest M.jsonl [--contain 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.control import reward as rw  # noqa: E402


def annotated(masks: np.ndarray, boxes: dict, contain: float) -> np.ndarray:
    """[K] bool: the share of each mask inside one of the (non-hand) boxes reaches `contain`."""
    out = np.zeros(len(masks), bool)
    height, width = masks.shape[1:] if len(masks) else (0, 0)
    for k, mask in enumerate(masks):
        area = int(mask.sum())
        for name, (x1, y1, x2, y2) in boxes.items():
            if name == "hand" or not area:
                continue
            r0, r1 = int(np.clip(np.floor(y1), 0, height)), int(np.clip(np.ceil(y2), 0, height))
            c0, c1 = int(np.clip(np.floor(x1), 0, width)), int(np.clip(np.ceil(x2), 0, width))
            if mask[r0:r1, c0:c1].sum() >= contain * area:
                out[k] = True
                break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--physplan", required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--contain", type=float, default=0.5)
    a = p.parse_args()
    rows = {json.loads(line)["sample_id"]: json.loads(line) for line in open(a.manifest) if line.strip()}
    stats = {"records": 0, "no_scene": 0, "no_boxes": 0, "instances": 0, "supervised": 0, "clips_without_supervised": 0}
    for path in sorted(Path(a.physplan).glob("*.npz")):
        if ".feat" in path.name:
            continue
        sid = path.name[: -len(".npz")]
        record = dict(np.load(path, allow_pickle=False))
        slots = [int(v) for v in record["slots"]]
        supervise = np.zeros(len(slots), bool)
        scene_path = Path(a.scenes) / f"{sid}.npz"
        boxes = (rows.get(sid) or {}).get("boxes_t0_canvas") or {}
        if not scene_path.is_file():
            stats["no_scene"] += 1
        elif not any(k != "hand" for k in boxes):
            stats["no_boxes"] += 1
        else:
            scene = rw.load_scene(scene_path)
            inside = dict(zip((int(v) for v in scene["slots"]), annotated(rw.scene_masks(scene), boxes, a.contain)))
            supervise = np.array([inside.get(slot, False) for slot in slots], bool)
        record["supervise"] = supervise
        np.savez_compressed(path, **record)
        stats["records"] += 1
        stats["instances"] += len(slots)
        stats["supervised"] += int(supervise.sum())
        stats["clips_without_supervised"] += int(not supervise.any())
    print("SSV2_SUPERVISE " + json.dumps(stats))


if __name__ == "__main__":
    main()
