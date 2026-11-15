#!/usr/bin/env python
"""Text-conditioned FlyingObjects, step 1: real object cutouts from ImageNet val.

ImageNet localisation provides a box per object; SAM2.1-large segments the object inside
that box (box prompts are far more reliable than automatic masks).  Kept cutouts:
one dominant box of a whitelisted movable/tabletop class, box 6-70% of the image and not
touching the border, mask/box fill ratio 0.35-0.98, one dominant connected component.
Output: cutouts/<id>.png (RGBA, cropped to the mask box) and cutouts.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

CLASSES = {
    "n02799071": ("baseball", "ball"), "n02802426": ("basketball", "ball"), "n03134739": ("croquet ball", "ball"),
    "n03445777": ("golf ball", "ball"), "n03942813": ("ping-pong ball", "ball"), "n04118538": ("rugby ball", "ball"),
    "n04254680": ("soccer ball", "ball"), "n04409515": ("tennis ball", "ball"), "n04540053": ("volleyball", "ball"),
    "n07747607": ("orange", "ball"), "n07749582": ("lemon", "ball"), "n07742313": ("green apple", "ball"),
    "n07753592": ("banana", "slide"), "n03063599": ("coffee mug", "slide"), "n02823428": ("beer bottle", "slide"),
    "n03983396": ("soda bottle", "slide"), "n04557648": ("water bottle", "slide"), "n04591713": ("wine bottle", "slide"),
    "n03937543": ("pill bottle", "slide"), "n03733805": ("measuring cup", "slide"), "n04398044": ("teapot", "slide"),
    "n03950228": ("pitcher", "slide"), "n04522168": ("vase", "slide"), "n04399382": ("teddy bear", "slide"),
    "n02708093": ("clock", "slide"), "n03935335": ("piggy bank", "slide"), "n03793489": ("computer mouse", "slide"),
    "n04074963": ("remote control", "slide"), "n02948072": ("candle", "slide"), "n04548362": ("wallet", "slide"),
    "n15075141": ("toilet paper roll", "slide"), "n03063689": ("coffeepot", "slide"), "n04254120": ("soap dispenser", "slide"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet", required=True,
                        help="ImageNet ILSVRC-2012 root with LOC_val_solution.csv and ILSVRC/Data/CLS-LOC/val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam-checkpoint", default=os.environ.get("SAM2_CHECKPOINT", ""),
                        help="SAM 2.1 hiera-large checkpoint (default: $SAM2_CHECKPOINT)")
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not args.sam_checkpoint:
        parser.error("set --sam-checkpoint or SAM2_CHECKPOINT")
    root = Path(args.imagenet)
    out = Path(args.output_dir)
    (out / "cutouts").mkdir(parents=True, exist_ok=True)
    boxes = defaultdict(list)
    with (root / "LOC_val_solution.csv").open() as handle:
        for row in csv.DictReader(handle):
            parts = row["PredictionString"].split()
            for i in range(0, len(parts), 5):
                boxes[row["ImageId"]].append((parts[i], *map(int, parts[i + 1:i + 5])))
    items = []
    for image_id, entries in sorted(boxes.items()):
        synsets = {e[0] for e in entries}
        if len(synsets) != 1 or entries[0][0] not in CLASSES:
            continue
        # largest box of that class
        items.append((image_id, max(entries, key=lambda e: (e[3] - e[1]) * (e[4] - e[2]))))
    items = items[args.shard_index::args.num_shards]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(args.sam_config, args.sam_checkpoint, device="cuda"))
    kept, stats = [], defaultdict(int)
    for image_id, (synset, x0, y0, x1, y1) in items:
        image = cv2.imread(str(root / "ILSVRC/Data/CLS-LOC/val" / f"{image_id}.JPEG"), cv2.IMREAD_COLOR)
        if image is None:
            stats["unreadable"] += 1
            continue
        h, w = image.shape[:2]
        area = (x1 - x0) * (y1 - y0) / float(h * w)
        if not 0.06 <= area <= 0.70 or x0 < 3 or y0 < 3 or x1 > w - 4 or y1 > h - 4:
            stats["box_filter"] += 1
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(box=np.array([x0, y0, x1, y1]), multimask_output=False)
        mask = masks[0] > 0.5
        fill = mask.sum() / float((x1 - x0) * (y1 - y0))
        count, labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if count < 2:
            stats["empty_mask"] += 1
            continue
        largest = 1 + int(np.argmax(cc_stats[1:, cv2.CC_STAT_AREA]))
        dominant = cc_stats[largest, cv2.CC_STAT_AREA] / max(mask.sum(), 1)
        if not 0.35 <= fill <= 0.98 or dominant < 0.95 or float(scores[0]) < 0.8:
            stats["mask_filter"] += 1
            continue
        mask = labels == largest
        ys, xs = np.nonzero(mask)
        crop = np.zeros((ys.max() - ys.min() + 1, xs.max() - xs.min() + 1, 4), np.uint8)
        crop[..., :3] = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        crop[..., 3] = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
        path = out / "cutouts" / f"{image_id}.png"
        cv2.imwrite(str(path), crop)
        name, motion = CLASSES[synset]
        kept.append({"id": image_id, "synset": synset, "name": name, "motion": motion, "path": str(path),
                     "score": float(scores[0]), "fill": float(fill)})
        stats["kept"] += 1
    with (out / f"cutouts_shard{args.shard_index}.jsonl").open("w") as handle:
        for item in kept:
            handle.write(json.dumps(item) + "\n")
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main()
