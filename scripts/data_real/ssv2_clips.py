#!/usr/bin/env python
"""SSv2 train real pool -> the Stage-2 v2 protocol, mirroring scripts/eval/pisa_clips.py.

Source: Something-Something v2 train clips (webm, 12 fps, height 240, width 320 / 360 / 427), each with the selection
of scripts/stage1_real/ssv2_train_select.py (items_selected.jsonl: t0_frame, onset, roles, boxes at t0).

  timing   real time at 7 fps from the selected t0: output frame k is source frame t0 + round(k * 12 / 7); the clip
           keeps the largest 4m+1 frames available, at most 37 (5.14 s).  Nothing is slowed down.
  frame    the 240-high frame is letterboxed into a 448 x 256 canvas (aspect kept; the inner image is upscaled by
           1.05-1.07).  The bands are the first frame's blurred border columns / rows, as in pisa_clips.py: one colour
           per row (side bands) or per column (top / bottom bands), the same in every frame -- a still backdrop with no
           copy of the moving object.  4:3 and 3:2 sources get side bands (53 / 32 px); 16:9 (427 x 240) sources get
           2 px top and bottom bands.
  caption  the concrete SSv2 label (placeholders filled, lower case kept as it is); prompt = caption.
  split    hold = 10 % by a hash of (template, first placeholder noun): every clip of an (event, object) group is on the
           same side.  The rest is train.

writes <out>/clips/<sid>.mp4 (libx264 crf 12, 7 fps), <out>/first_frames/<sid>.png (lossless, output frame 0),
       <out>/manifest.jsonl (every clip, with split), <out>/manifest.csv (the columns cache_sam2_masks.py and
       cache_stage2_cond_v2.py read), <out>/splits/{train,hold}.jsonl, <out>/errors.jsonl, <out>/summary.json
       sid = ssv2_<id>; manifest rows also carry source_id, template, family, t0_frame, onset_frame, hand_at_t0,
       boxes_t0 (native px) and boxes_t0_canvas (canvas px), source_frame_indices (source frame of every output frame).
usage: ssv2_clips.py --items items_selected.jsonl --video-dir DIR(with <id>.webm) --out DIR [--workers 8]
                     [--ffmpeg PATH] [--hold-share 0.1] [--limit N]
ffmpeg: --ffmpeg, else $FFMPEG, else imageio_ffmpeg's binary, else the first `ffmpeg` on PATH that runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

FPS, WIDTH, HEIGHT, MAX_FRAMES, MIN_FRAMES = 7.0, 448, 256, 37, 5
BORDER = 12          # border columns / rows averaged into the bands
BLUR_SIGMA = 9.0
MANIFEST_CSV_FIELDS = ["sample_id", "scenario", "split", "video_path", "first_frame_path", "caption", "fps", "frames", "width", "height"]


def find_ffmpeg(explicit: str | None = None) -> str:
    """The first ffmpeg that actually runs (a PATH entry built for another CPU is skipped)."""
    candidates = [explicit, os.environ.get("FFMPEG")]
    try:
        import imageio_ffmpeg  # noqa: WPS433
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001
        pass
    candidates.append(shutil.which("ffmpeg"))
    for c in candidates:
        if not c:
            continue
        try:
            if subprocess.run([c, "-version"], capture_output=True, timeout=30).returncode == 0:
                return c
        except (OSError, subprocess.SubprocessError):
            continue
    raise RuntimeError("no working ffmpeg (pass --ffmpeg)")


def largest_4m1(n: int, cap: int = MAX_FRAMES) -> int:
    n = min(n, cap)
    return 0 if n < 1 else 4 * ((n - 1) // 4) + 1


def source_picks(t0: int, n_source: int, source_fps: float, fps: float = FPS, max_frames: int = MAX_FRAMES) -> list[int]:
    """Source frame index of every output frame: t0 + round(k * source_fps / fps), the largest 4m+1 count that fits."""
    if n_source - 1 < t0:
        return []
    available = int((n_source - 1 - t0) * fps / source_fps) + 1
    count = largest_4m1(available, max_frames)
    return [min(n_source - 1, t0 + round(k * source_fps / fps)) for k in range(count)]


def letterbox_geometry(w: int, h: int) -> dict:
    scale = min(WIDTH / w, HEIGHT / h)
    iw, ih = int(round(w * scale)), int(round(h * scale))
    return {"scale": scale, "inner_w": iw, "inner_h": ih, "left": (WIDTH - iw) // 2, "top": (HEIGHT - ih) // 2}


def resize_inner(frame: np.ndarray, g: dict) -> np.ndarray:
    interpolation = cv2.INTER_AREA if g["scale"] < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (g["inner_w"], g["inner_h"]), interpolation=interpolation)


def _band(columns: np.ndarray, length: int, axis: int) -> np.ndarray:
    """axis 1: `columns` [H, b, 3] -> [H, length, 3] (one colour per row); axis 0: [b, W, 3] -> [length, W, 3]."""
    colour = columns.mean(axis, keepdims=True).astype(np.float32)
    band = np.repeat(colour, length, axis)
    if BLUR_SIGMA > 0:
        band = cv2.GaussianBlur(band, (0, 0), BLUR_SIGMA)
    return band.clip(0, 255).astype(np.uint8)


def backdrop(first: np.ndarray, g: dict) -> np.ndarray:
    """The still canvas every frame is pasted into: bands from the first frame's border columns / rows."""
    inner = resize_inner(first, g)
    iw, ih, left, top = g["inner_w"], g["inner_h"], g["left"], g["top"]
    right, bottom = WIDTH - left - iw, HEIGHT - top - ih
    out = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    out[top:top + ih, left:left + iw] = inner
    if left > 0:
        out[top:top + ih, :left] = _band(inner[:, :BORDER], left, 1)
    if right > 0:
        out[top:top + ih, left + iw:] = _band(inner[:, -BORDER:], right, 1)
    if top > 0:
        out[:top, left:left + iw] = _band(inner[:BORDER], top, 0)
        out[:top, :left] = out[top, :left]
        out[:top, left + iw:] = out[top, left + iw:]
    if bottom > 0:
        out[top + ih:, left:left + iw] = _band(inner[-BORDER:], bottom, 0)
        out[top + ih:, :left] = out[top + ih - 1, :left]
        out[top + ih:, left + iw:] = out[top + ih - 1, left + iw:]
    return out


def canvas(frame: np.ndarray, back: np.ndarray, g: dict) -> np.ndarray:
    out = back.copy()
    out[g["top"]:g["top"] + g["inner_h"], g["left"]:g["left"] + g["inner_w"]] = resize_inner(frame, g)
    return out


def box_to_canvas(box: list[float], g: dict) -> list[float]:
    s, left, top = g["scale"], g["left"], g["top"]
    return [round(box[0] * s + left, 1), round(box[1] * s + top, 1), round(box[2] * s + left, 1), round(box[3] * s + top, 1)]


def read_source(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, float(fps)


def hold_split(template: str, noun: str, hold_share: float) -> str:
    key = hashlib.sha256(f"ssv2-hold:{template}|{noun}".encode()).hexdigest()
    return "hold" if int(key, 16) % 1000 < hold_share * 1000 else "train"


def _init_worker() -> None:
    cv2.setNumThreads(1)


def work(job: tuple[dict, str, str, str]) -> dict:
    item, video_dir, out_dir, ffmpeg = job
    out = Path(out_dir)
    sid = f"ssv2_{item['id']}"
    source = Path(video_dir) / f"{item['id']}.webm"
    if not source.is_file():
        return {"sample_id": sid, "error": "missing source"}
    frames, source_fps = read_source(source)
    t0 = int(item["t0_frame"])
    picks = source_picks(t0, len(frames), source_fps)
    if len(picks) < MIN_FRAMES:
        return {"sample_id": sid, "error": f"short: {len(frames)} source frames, t0 {t0}"}
    h, w = frames[0].shape[:2]
    g = letterbox_geometry(w, h)
    back = backdrop(frames[picks[0]], g)
    clip, frame_path = out / "clips" / f"{sid}.mp4", out / "first_frames" / f"{sid}.png"
    if not frame_path.is_file():
        cv2.imwrite(str(frame_path), canvas(frames[picks[0]], back, g))
    if not clip.is_file():
        tmp = clip.with_suffix(".tmp.mp4")
        encoder = subprocess.Popen([ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
                                    "-r", f"{FPS:g}", "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-threads", "2",
                                    "-an", str(tmp)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for i in picks:
            encoder.stdin.write(canvas(frames[i], back, g).tobytes())
        encoder.stdin.close()
        err = encoder.stderr.read().decode(errors="replace")
        if encoder.wait() != 0:
            tmp.unlink(missing_ok=True)
            return {"sample_id": sid, "error": "ffmpeg: " + err[-200:]}
        os.replace(tmp, clip)
    caption = str(item.get("caption") or item.get("label") or "").strip()
    return {"sample_id": sid, "trial_id": sid, "camera": "ssv2", "scenario": item["family"], "caption": caption, "prompt": caption,
            "video_path": str(clip.resolve()), "first_frame_path": str(frame_path.resolve()), "source_path": str(source.resolve()),
            "raw_path": str(source.resolve()), "fps": FPS, "frames": len(picks), "width": WIDTH, "height": HEIGHT, "domain": "ssv2",
            "recorded_fps": source_fps, "source_id": str(item["id"]), "template_id": int(item["template_id"]), "template": item["template"],
            "family": item["family"], "event": item.get("event"), "static_caption": bool(item.get("static_caption", False)),
            "placeholders": item.get("placeholders", []), "noun": item.get("noun", ""), "pool": item.get("pool"),
            "t0_frame": t0, "onset_frame": item.get("onset_frame"), "sec_after_t0": item.get("sec_after_t0"),
            "hand_at_t0": bool(item.get("hand_at_t0", False)), "boxes_t0": item.get("boxes_t0", {}),
            "boxes_t0_canvas": {c: box_to_canvas(b, g) for c, b in item.get("boxes_t0", {}).items()},
            "mover_cat": item.get("mover_cat"), "actor_cat": item.get("actor_cat"), "patient_cat": item.get("patient_cat"),
            "direction_expected": item.get("direction_expected"), "dx_pct_w": item.get("dx_pct_w"), "dy_pct_h": item.get("dy_pct_h"),
            "cam_tier": item.get("cam_tier"), "flags": item.get("flags", []), "source_frames": len(frames), "source_width": w,
            "source_height": h, "letterbox": g, "source_frame_indices": picks}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--video-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--hold-share", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    ffmpeg = find_ffmpeg(a.ffmpeg)
    for name in ("clips", "first_frames", "splits"):
        (a.out / name).mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in a.items.read_text().splitlines() if line.strip()]
    if a.limit:
        items = items[: a.limit]
    print(f"{len(items)} items, ffmpeg {ffmpeg}", flush=True)
    jobs = [(it, str(a.video_dir), str(a.out), ffmpeg) for it in items]
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init_worker) as pool:
        for n, row in enumerate(pool.map(work, jobs, chunksize=4), 1):
            rows.append(row)
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    bad = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    for row in rows:
        row["split"] = hold_split(row["template"], row["noun"], a.hold_share)
    (a.out / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (a.out / "errors.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bad))
    for name in ("train", "hold"):
        part = [r for r in rows if r["split"] == name]
        (a.out / "splits" / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
    with open(a.out / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_CSV_FIELDS)
        writer.writeheader()
        writer.writerows({k: r[k] for k in MANIFEST_CSV_FIELDS} for r in rows)
    hist, by_family, by_split = {}, {}, {}
    for r in rows:
        hist[r["frames"]] = hist.get(r["frames"], 0) + 1
        by_family[r["scenario"]] = by_family.get(r["scenario"], 0) + 1
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    summary = {"clips": len(rows), "failed": len(bad), "frames_histogram": {str(k): v for k, v in sorted(hist.items())},
               "by_family": by_family, "by_split": by_split, "hold_share": a.hold_share,
               "hold_groups": len({(r["template"], r["noun"]) for r in rows if r["split"] == "hold"}),
               "train_groups": len({(r["template"], r["noun"]) for r in rows if r["split"] == "train"}),
               "errors": [r["error"][:80] for r in bad[:10]]}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("SSV2_CLIPS_DONE", json.dumps(summary))


if __name__ == "__main__":
    main()
