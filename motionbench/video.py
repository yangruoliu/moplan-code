"""Reading clips into the work frame: letterbox (never stretch), frame subsampling, first-frame pinning."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from . import contract as C


def letterbox(frame: np.ndarray, height: int = C.WORK_H, width: int = C.WORK_W,
              interpolation: int = cv2.INTER_AREA) -> Tuple[np.ndarray, dict]:
    """Resize keeping the aspect ratio and pad with black to (height, width).
    Returns (canvas, geometry) with geometry = {scale, x0, y0, w, h} of the content box."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=interpolation)
    if frame.ndim == 3 and resized.ndim == 2:
        resized = resized[..., None]
    canvas = np.zeros((height, width) + frame.shape[2:], dtype=frame.dtype)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, dict(scale=scale, x0=x0, y0=y0, w=nw, h=nh, native_h=h, native_w=w)


def native_to_work(xy: np.ndarray, geom: dict) -> np.ndarray:
    """Map native-pixel coordinates [..., 2] into the letterboxed work frame."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([xy[..., 0] * geom["scale"] + geom["x0"], xy[..., 1] * geom["scale"] + geom["y0"]], axis=-1)


def read_video(path: str, start_frame: int = 0, max_seconds: Optional[float] = None,
               fps_override: Optional[float] = None, step: int = 1, max_side: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """(frames [F,H,W,3] RGB uint8, effective fps). Frames before start_frame are skipped; with step > 1 every
    step-th frame is kept (fps / step); max_side downsizes large sources on read."""
    cap = cv2.VideoCapture(str(path))
    fps = fps_override or float(cap.get(cv2.CAP_PROP_FPS)) or 12.0
    fps_eff = fps / max(1, step)
    limit = None if max_seconds is None else int(np.floor(max_seconds * fps_eff + 1e-6)) + 1
    frames, index = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if index >= start_frame and (index - start_frame) % max(1, step) == 0:
            if max_side and max(fr.shape[:2]) > max_side:
                sc = max_side / max(fr.shape[:2])
                fr = cv2.resize(fr, (int(round(fr.shape[1] * sc)), int(round(fr.shape[0] * sc))), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if limit is not None and len(frames) >= limit:
                break
        index += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"cannot read frames from {path} (start {start_frame})")
    return np.stack(frames), fps_eff


def read_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"{path}: expected [F,H,W,3], got {arr.shape}")
    return arr.astype(np.uint8)


def to_work(frames: np.ndarray) -> Tuple[np.ndarray, dict]:
    out, geom = [], None
    for fr in frames:
        canvas, geom = letterbox(fr)
        out.append(canvas)
    return np.stack(out), geom


def subsample(frames: np.ndarray, fps: float, max_frames: int = 96) -> Tuple[np.ndarray, float, np.ndarray]:
    """Keep at most max_frames frames (every n-th). Returns (frames, fps, times)."""
    every = int(np.ceil(len(frames) / max_frames))
    frames = frames[::every]
    fps = fps / every
    times = np.arange(len(frames), dtype=np.float64) / fps
    return frames, fps, times


def looks_crossfaded(frames: np.ndarray, declared_fps: float, native_fps: float) -> bool:
    """A 24 fps submission made by blending 7 fps frames shows near-duplicate triplets."""
    if len(frames) < 8 or declared_fps <= native_fps * 1.5:
        return False
    d = [float(np.mean(np.abs(frames[i + 1].astype(np.int16) - frames[i].astype(np.int16)))) for i in range(min(24, len(frames) - 1))]
    d = np.array(d)
    return bool(np.mean(d < 0.25 * (d.max() + 1e-6)) > 0.4)


CANVAS_W, CANVAS_H = 1280, 720


def canvas_box(native_hw, canvas_w: int = CANVAS_W, canvas_h: int = CANVAS_H) -> Tuple[int, int, int, int]:
    """(x0, y0, w, h) of the native frame inside the 16:9 generation canvas — the same rounding as
    scripts/bench/bench_common.letterbox_16x9_array (scale = min(W/w, H/h), centred)."""
    h, w = int(native_hw[0]), int(native_hw[1])
    scale = min(canvas_w / w, canvas_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return (canvas_w - nw) // 2, (canvas_h - nh) // 2, nw, nh


# Region of the 16:9 canvas that a generator actually renders, normalised (x0, y0, x1, y1), measured by aligning
# frame 0 of each model's clips with its input canvas: HunyuanVideo-1.5 (480p) centre-crops the 832-wide input to 816.
MODEL_CANVAS_CROP = {
    "hunyuan15_480p": (8 / 832, 0.0, 824 / 832, 1.0),
    # LTX-2.3 (official CLI, 768x448): the 1280x720 canvas is resized to cover 796x448 and centre-cropped by 14 px a side
    "ltx23": (14 / 796, 0.0, 782 / 796, 1.0),
}


def canvas_to_work(frames: np.ndarray, native_hw, geom: dict, canvas_w: int = CANVAS_W, canvas_h: int = CANVAS_H,
                   crop=None) -> np.ndarray:
    """Generated frames [F,Hg,Wg,3] that depict the whole 16:9 canvas (the model resized the canvas to its own
    resolution) → the item's work frame: crop the native-content box of the canvas (in generated pixels), resize it to
    the item's letterbox content size and paste it at the item's offset. Reproduces the geometry used to build the
    item's masks, so the pinned first frame and the generated frames line up exactly."""
    F, Hg, Wg = frames.shape[:3]
    x0, y0, w, h = canvas_box(native_hw, canvas_w, canvas_h)
    cx0, cy0, cx1, cy1 = crop if crop is not None else (0.0, 0.0, 1.0, 1.0)
    # canvas pixel X -> generated pixel ((X / canvas_w - cx0) / (cx1 - cx0)) * Wg  (same for y)
    gx = lambda X: ((X / canvas_w - cx0) / (cx1 - cx0)) * Wg
    gy = lambda Y: ((Y / canvas_h - cy0) / (cy1 - cy0)) * Hg
    gx0, gy0 = int(round(gx(x0))), int(round(gy(y0)))
    gx1, gy1 = int(round(gx(x0 + w))), int(round(gy(y0 + h)))
    if gx0 < 0 or gy0 < 0 or gx1 > Wg or gy1 > Hg:
        # the native content reaches into a part of the canvas the model cropped away: pad with black
        pad_l, pad_t = max(0, -gx0), max(0, -gy0)
        pad_r, pad_b = max(0, gx1 - Wg), max(0, gy1 - Hg)
        frames = np.pad(frames, ((0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)))
        gx0, gx1, gy0, gy1 = gx0 + pad_l, gx1 + pad_l, gy0 + pad_t, gy1 + pad_t
    out = np.zeros((F, C.WORK_H, C.WORK_W, 3), np.uint8)
    for i, fr in enumerate(frames):
        crop = fr[gy0:gy1, gx0:gx1]
        out[i, geom["y0"]:geom["y0"] + geom["h"], geom["x0"]:geom["x0"] + geom["w"]] = cv2.resize(crop, (geom["w"], geom["h"]), interpolation=cv2.INTER_AREA)
    return out


def base_sid(sid: str) -> str:
    """Item sid of a generation row (`<sid>-ex` explicit variant, `<sid>-cf` counterfactual variant)."""
    for suf in ("-cf", "-ex"):
        if sid.endswith(suf):
            return sid[: -len(suf)]
    return sid
