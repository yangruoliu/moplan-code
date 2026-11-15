from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def video_info(path: str | Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {path}")
    result = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return result


def sampling_indices(
    frame_count: int,
    source_fps: float,
    num_frames: int,
    target_fps: float,
    start_sec: float,
) -> list[int]:
    if frame_count <= 0 or source_fps <= 0 or target_fps <= 0:
        raise ValueError(
            f"invalid sampling metadata frames={frame_count}, source_fps={source_fps}, "
            f"target_fps={target_fps}"
        )
    stride = source_fps / target_fps
    requested_start = int(round(start_sec * source_fps))
    maximum_start = max(0, int(np.floor((frame_count - 1) - (num_frames - 1) * stride)))
    start = min(requested_start, maximum_start)
    return [min(frame_count - 1, int(round(start + index * stride))) for index in range(num_frames)]


def read_sampled_video(
    path: str | Path,
    *,
    num_frames: int,
    target_fps: float,
    start_sec: float,
    height: int,
    width: int,
) -> tuple[np.ndarray, list[int], dict[str, float | int]]:
    info = video_info(path)
    indices = sampling_indices(
        int(info["frames"]), float(info["fps"]), num_frames, target_fps, start_sec
    )
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise OSError(f"failed to decode frame {index} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()
    return np.stack(frames), indices, info


def color_object_mask(image_rgb: np.ndarray, *, include_patient: bool = False) -> np.ndarray:
    """Extract Physion's red agent (and optionally yellow patient) mask."""
    image = image_rgb.astype(np.int16)
    red = (image[..., 0] > 140) & (image[..., 0] > image[..., 1] * 1.35) & (
        image[..., 0] > image[..., 2] * 1.35
    )
    if not include_patient:
        return red
    yellow = (
        (image[..., 0] > 135)
        & (image[..., 1] > 115)
        & (image[..., 0] > image[..., 2] * 1.35)
        & (image[..., 1] > image[..., 2] * 1.25)
    )
    return red | yellow


def read_official_agent_mask(path: str | Path, height: int, width: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read map: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = color_object_mask(image)
    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def forward_splat_mask(mask: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Bilinearly splat source masks through first-to-future flow.

    Args:
        mask: [B, 1, H, W]
        flow: [B, T, 2, H, W] in pixels
    Returns:
        Occupancy [B, T, 1, H, W] clipped to [0, 1].
    """
    batch, horizons, _, height, width = flow.shape
    device = flow.device
    dtype = flow.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x = xx.view(1, 1, height, width) + flow[:, :, 0]
    y = yy.view(1, 1, height, width) + flow[:, :, 1]
    x0, y0 = x.floor(), y.floor()
    source = mask[:, None].expand(batch, horizons, 1, height, width).reshape(
        batch * horizons, -1
    )
    output = torch.zeros(batch * horizons, height * width, device=device, dtype=dtype)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xi = x0 + dx
        yi = y0 + dy
        valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        weight = (1 - (x - xi).abs()) * (1 - (y - yi).abs()) * valid
        index = (yi.clamp(0, height - 1) * width + xi.clamp(0, width - 1)).long()
        output.scatter_add_(1, index.reshape(batch * horizons, -1), source * weight.reshape(batch * horizons, -1))
    return output.reshape(batch, horizons, 1, height, width).clamp_(0, 1)


def resize_mask_to_grid(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(mask.float(), size=(height, width), mode="area").clamp(0, 1)
