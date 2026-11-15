#!/usr/bin/env python
"""One measuring instrument for every clip that is compared with a real one (the trained real clip, a generated clip,
a decoded one-step estimate): the SAME settings on both sides of every comparison.

  points   CoTracker3 (visibility score > 0.5, 2x input) from the scene's first-frame points -> background, camera
  objects  SAM2 mask propagation from the scene's first-frame masks -> centroid displacement of every object

measure(frames [F, H, W, 3] at the work resolution, points [N, 2], masks [K, H, W]) ->
  {"tracks" [F, N, 2], "visible" [F, N], "object_disp" [F, K, 2], "object_ok" [F, K]}   (work-frame pixels)
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from scripts.control import masktrack as mt
from scripts.control.cotrack import load_tracker, track_points


class Instrument:
    def __init__(self, cotracker_repo: str, cotracker_checkpoint: str, sam2_checkpoint: str = mt.SAM2_CHECKPOINT, *, scale: float = 2.0,
                 vis_threshold: float = 0.5, device: str = "cuda", tmp_root: str | None = None) -> None:
        self.scale, self.vis_threshold, self.tmp_root = float(scale), float(vis_threshold), tmp_root
        self.points = load_tracker(cotracker_repo, cotracker_checkpoint, torch.device(device))
        self.objects = mt.load_mask_tracker(sam2_checkpoint, device=device)

    def _enlarge(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = frames.shape[1:3]
        size = (int(round(width * self.scale)), int(round(height * self.scale)))
        big = np.stack([cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC) for frame in frames])
        return big, np.array([(size[0] - 1) / (width - 1), (size[1] - 1) / (height - 1)])

    def measure(self, frames: np.ndarray, points: np.ndarray, masks: np.ndarray, *, with_points: bool = True) -> dict:
        big, factor = self._enlarge(frames)
        out = {}
        if with_points:
            tracks, score = track_points(self.points, big, np.asarray(points, np.float64) * factor)
            out["tracks"], out["visible"] = (tracks / factor).astype(np.float32), score > self.vis_threshold
        large = np.stack([cv2.resize(mask.astype(np.uint8), (big.shape[2], big.shape[1]), interpolation=cv2.INTER_NEAREST) > 0 for mask in masks]) \
            if len(masks) else np.zeros((0, *big.shape[1:3]), bool)
        disp, ok = mt.displacement(mt.propagate(self.objects, big, large, tmp_root=self.tmp_root))
        out["object_disp"], out["object_ok"] = (disp / factor).astype(np.float32), ok
        return out
