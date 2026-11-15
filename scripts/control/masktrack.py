#!/usr/bin/env python
"""Where did each first-frame object go?  SAM2 video propagation of the objects' masks through a clip.

Point tracking answers this badly exactly where it matters: at the generator's 7 fps a falling object jumps tens of
pixels per frame and CoTracker3 can leave its points on the background (a real fall is then read as "did not move"),
and a tumbling object hides its first-frame surface points.  A mask does not have these
problems: SAM2 re-finds the object in every frame from its appearance memory, without a locality prior, and when the
object is gone the mask is empty.

propagate(predictor, frames, masks) -> {"centroid" [F, K, 2] (x, y; nan where the mask is empty), "area" [F, K] px,
                                        "box" [F, K, 4] (x0, y0, x1, y1), "prompt_iou" [K],
                                        "masks" [F, K, H, W] bool when keep_masks}
Coordinates are in the pixels of `frames`; `masks` [K, H, W] are the objects in frames[0].

The first frame is shown to SAM2 twice.  On the prompted frame SAM2 returns the prompt (refined); from the next frame
on it returns what its memory makes of the object, which for an ambiguous prompt (a part of an object, a blocky
region) can be a different region.  Measured from the prompted frame, that re-interpretation is a jump that looks
like motion (a frozen video can then appear to contain a moving object).
Index 0 of the result is therefore SAM2's memory-based reading of the repeated first frame: every later reading is
compared with a reading of the same kind.  `prompt_iou` says how much of the prompt that reading still is.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")          # SAM2 prints two progress bars per clip
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "")      # path of sam2.1_hiera_large.pt


def load_mask_tracker(checkpoint: str = SAM2_CHECKPOINT, config: str = SAM2_CONFIG, device: str = "cuda"):
    checkpoint = checkpoint or os.environ.get("SAM2_CHECKPOINT", "")
    if not checkpoint:
        raise RuntimeError("no SAM 2.1 checkpoint: set SAM2_CHECKPOINT to the path of sam2.1_hiera_large.pt")
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(config, checkpoint, device=device)


@torch.no_grad()
def propagate(predictor, frames: np.ndarray, masks: np.ndarray, *, keep_masks: bool = False, tmp_root: str | None = None, min_area: int = 4) -> dict:
    """frames [F, H, W, 3] uint8 RGB, masks [K, H, W] bool."""
    frames = np.concatenate((frames[:1], frames))                     # the first frame twice: see the module text
    count, height, width = frames.shape[:3]
    objects = len(masks)
    out = {"centroid": np.full((count, objects, 2), np.nan, np.float32), "area": np.zeros((count, objects), np.float32),
           "box": np.zeros((count, objects, 4), np.float32), "prompt_iou": np.zeros(objects, np.float32)}
    if keep_masks:
        out["masks"] = np.zeros((count, objects, height, width), bool)
    if not objects:
        return {key: (value[1:] if key != "prompt_iou" else value) for key, value in out.items()}
    folder = Path(tempfile.mkdtemp(prefix="masktrack_", dir=tmp_root))
    try:
        for index, frame in enumerate(frames):
            cv2.imwrite(str(folder / f"{index:05d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 97])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(folder))
            for k, mask in enumerate(masks):
                if mask.any():
                    predictor.add_new_mask(state, frame_idx=0, obj_id=k, mask=torch.from_numpy(np.ascontiguousarray(mask)))
            ys, xs = np.mgrid[0:height, 0:width]
            for frame_index, object_ids, logits in predictor.propagate_in_video(state):
                found = (logits[:, 0] > 0).cpu().numpy()                      # [objects present, H, W]
                for row, k in enumerate(object_ids):
                    area = int(found[row].sum())
                    out["area"][frame_index, k] = area
                    if area >= min_area:
                        out["centroid"][frame_index, k] = (xs[found[row]].mean(), ys[found[row]].mean())
                        out["box"][frame_index, k] = (xs[found[row]].min(), ys[found[row]].min(), xs[found[row]].max(), ys[found[row]].max())
                    if keep_masks:
                        out["masks"][frame_index, k] = found[row]
                    if frame_index == 1:
                        union = int((found[row] | masks[k]).sum())
                        out["prompt_iou"][k] = float((found[row] & masks[k]).sum()) / union if union else 0.0
            predictor.reset_state(state)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return {key: (value[1:] if key != "prompt_iou" else value) for key, value in out.items()}


def displacement(result: dict, start: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(disp [F, K, 2] centroid displacement from frame 0, ok [F, K]): ok needs the object in both frames with at least
    30 % and at most 3.3 x of its first-frame area (a mask that collapsed or leaked onto the background is not a position)."""
    centroid, area = result["centroid"], result["area"]
    first = centroid[0] if start is None else start
    ratio = area / np.maximum(area[:1], 1.0)
    ok = np.isfinite(centroid).all(-1) & np.isfinite(first).all(-1)[None] & (ratio >= 0.3) & (ratio <= 3.3)
    disp = np.where(ok[..., None], centroid - first[None], 0.0)
    return disp.astype(np.float32), ok
