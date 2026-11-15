#!/usr/bin/env python
"""CoTracker3 (offline) as a point tracker that returns its visibility SCORE instead of a hard decision.

The stock predictor thresholds visibility at 0.9.  A falling or rolling object is motion-blurred, its points drop
below that for a few frames, and a reader that keeps "visible" points only then loses exactly the objects that move
fastest.  Here the score is returned; callers choose the threshold (0.5 by default in the plan builders) and keep 0.9
available as the strict reading.

track_points(predictor, frames, points)  ->  tracks [F, N, 2] float32 (x, y) in the frames' pixels, score [F, N] in [0, 1]
The call mirrors CoTrackerPredictor._compute_sparse_tracks (queries at frame 0, 6x6 support grid, 6 iterations).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def load_tracker(repo: str, checkpoint: str, device: torch.device, window_len: int = 60):
    """CoTracker3 offline predictor.  `repo` is a co-tracker git checkout (default: $COTRACKER_REPO; empty when the
    package is installed), `checkpoint` the scaled_offline.pth weights (default: $COTRACKER_CKPT)."""
    repo = repo or os.environ.get("COTRACKER_REPO", "")
    checkpoint = checkpoint or os.environ.get("COTRACKER_CKPT", "")
    if not checkpoint:
        raise RuntimeError("no CoTracker3 checkpoint: pass it or set COTRACKER_CKPT to scaled_offline.pth")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    from cotracker.predictor import CoTrackerPredictor
    return CoTrackerPredictor(checkpoint=checkpoint, offline=True, window_len=window_len).to(device).eval()


@torch.no_grad()
def track_points(predictor, frames: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """frames [F, H, W, 3] uint8, points [N, 2] (x, y) pixels of frame 0."""
    from cotracker.models.core.model_utils import get_points_on_a_grid

    device = next(predictor.parameters()).device
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)        # [1, F, 3, H, W]
    _, count, _, height, width = video.shape
    shape = tuple(predictor.interp_shape)
    video = F.interpolate(video[0], shape, mode="bilinear", align_corners=True)[None]
    scale = video.new_tensor([(shape[1] - 1) / (width - 1), (shape[0] - 1) / (height - 1)])
    queries = torch.cat([torch.zeros(len(points), 1), torch.from_numpy(np.asarray(points, np.float32))], 1)[None].to(device)
    queries[:, :, 1:] *= scale
    support = get_points_on_a_grid(predictor.support_grid_size, shape, device=device)
    support = torch.cat([torch.zeros_like(support[:, :, :1]), support], 2)
    tracks, score, *_ = predictor.model.forward(video=video, queries=torch.cat([queries, support], 1), iters=6)
    tracks, score = tracks[:, :, : len(points)], score[:, :, : len(points)].float()
    tracks[0, 0] = queries[0, :, 1:]                                                        # the query points themselves
    score[0, 0] = 1.0
    return (tracks[0] / scale).float().cpu().numpy(), score[0].clamp(0, 1).cpu().numpy()
