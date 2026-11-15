#!/usr/bin/env python
"""PisaBench real drop clips (nyu-visionx/pisa-experiments, pisabench/real: square frames at 120 fps, static phone cameras,
an object released from an invisible wire at the first frame) -> 1280x720 first frames (and 7 fps clips) for the MoPlan
generation protocol.  The PisaBench evaluation only uses first_frames/<sid>.png (scripts/bench/official_sp_inputs.py
pisafull); scripts/bench/pisa_official_export.py crops the square back out of the generated work frames.

  timing   real time at 7 fps: output frame k is source frame round(k * 120 / 7); the clip keeps the largest 4m+1
           frames it has (3 s -> 21 frames).  Nothing is slowed down.
  frame    the fall spans the whole height (median 996 of 1080 rows), so a 16:9 crop would cut it.  The square frame
           is kept whole in the middle of a 16:9 canvas; the two side bands are the first frame's border columns,
           blurred, and stay the same in every frame (a still backdrop, no mirrored copy of the falling object).
  split    by the caption (the object): every clip of an object goes to the same side (`--dev-share` of the objects are
           "dev", the rest "reserve"); the split does not affect the first frames.

  encoder  the clips are written by the ffmpeg binary bundled with imageio-ffmpeg (`--ffmpeg` / $FFMPEG pick another
           one); it has libx264, which the ffmpeg of the conda lock file lacks.  Only the clips depend on it.

writes <out>/clips/<sid>.mp4, <out>/first_frames/<sid>.png, <out>/manifest.{jsonl,csv} (dev clips only),
       <out>/manifest_all.jsonl, <out>/splits/dev.jsonl
usage: pisa_clips.py --real DIR(with 0000/ 0001/ ...) --out DIR [--dev-share 0.3] [--workers 16] [--ffmpeg EXE]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

FPS, SOURCE_FPS, WIDTH, HEIGHT = 7.0, 120.0, 1280, 720


def ffmpeg_exe(explicit: str = "") -> str:
    """--ffmpeg, else $FFMPEG, else the ffmpeg bundled with imageio-ffmpeg (pinned in requirements.txt)."""
    if explicit or os.environ.get("FFMPEG"):
        return explicit or os.environ["FFMPEG"]
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def canvas(frame: np.ndarray, bands: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    side = HEIGHT
    inner = cv2.resize(frame, (side, side), interpolation=cv2.INTER_AREA)
    left = (WIDTH - side) // 2
    out = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    out[:, :left], out[:, left:left + side], out[:, left + side:] = bands[0], inner, bands[1]
    return out


def side_bands(first: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inner = cv2.resize(first, (HEIGHT, HEIGHT), interpolation=cv2.INTER_AREA)
    left = (WIDTH - HEIGHT) // 2
    bands = []
    for columns in (inner[:, :12], inner[:, -12:]):
        colour = columns.mean(1, keepdims=True).astype(np.float32)             # [H, 1, 3]: one colour per row
        band = np.repeat(colour, left, 1)
        bands.append(cv2.GaussianBlur(band, (0, 0), 9).clip(0, 255).astype(np.uint8))
    right = WIDTH - HEIGHT - left
    return bands[0], cv2.resize(bands[1], (right, HEIGHT))


def work(job: tuple[str, str, str]) -> dict:
    source, out, ffmpeg = Path(job[0]), Path(job[1]), job[2]
    sid = f"pisa_real_{source.name}"
    files = sorted(source.glob("*.jpg"))
    info = json.loads((source / "clip_info.json").read_text())
    available = int((len(files) - 1) * FPS / SOURCE_FPS) + 1
    count = min(37, (available - 1) // 4 * 4 + 1)
    if count < 5:
        return {"sample_id": sid, "error": f"{len(files)} source frames"}
    picks = [files[min(len(files) - 1, round(k * SOURCE_FPS / FPS))] for k in range(count)]
    clip, frame_path = out / "clips" / f"{sid}.mp4", out / "first_frames" / f"{sid}.png"
    first = cv2.imread(str(picks[0]))
    bands = side_bands(first)
    if not frame_path.is_file():
        cv2.imwrite(str(frame_path), canvas(first, bands))
    if not clip.is_file():
        part = clip.with_name(f"{clip.stem}.tmp{os.getpid()}.mp4")
        encoder = subprocess.Popen([ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", f"{FPS:g}", "-i", "-",
                                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-an", str(part)],
                                   stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for path in picks:
                encoder.stdin.write(canvas(cv2.imread(str(path)), bands).tobytes())
            encoder.stdin.close()
        except BrokenPipeError:                                       # the encoder exited early; its stderr says why
            pass
        error = encoder.stderr.read().decode(errors="replace").strip()
        if encoder.wait() != 0:
            part.unlink(missing_ok=True)
            return {"sample_id": sid, "error": f"ffmpeg: {error[-300:]}"}
        part.replace(clip)
    what = str(info.get("prompt", "An object falls.")).strip().rstrip(".")
    return {"sample_id": sid, "trial_id": sid, "camera": "pisa", "scenario": "pisa/drop", "object": what.lower(),
            "caption": f"{what}. Static shot with no camera movement.", "video_path": str(clip.resolve()), "first_frame_path": str(frame_path.resolve()),
            "source_path": str(clip.resolve()), "raw_path": str(source.resolve()), "fps": FPS, "frames": count, "width": WIDTH, "height": HEIGHT, "domain": "pisa", "recorded_fps": SOURCE_FPS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dev-share", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--ffmpeg", default="", help="ffmpeg binary with libx264 (default: $FFMPEG, else the one bundled with imageio-ffmpeg)")
    a = p.parse_args()
    ffmpeg = ffmpeg_exe(a.ffmpeg)
    try:
        check = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
    except OSError as error:
        raise SystemExit(f"{ffmpeg} does not start ({error}); pass --ffmpeg") from error
    if check.returncode != 0 or " libx264 " not in check.stdout:
        raise SystemExit(f"{ffmpeg} does not run or has no libx264 encoder; pass --ffmpeg")
    for name in ("clips", "first_frames", "splits"):
        (a.out / name).mkdir(parents=True, exist_ok=True)
    sources = sorted(d for d in a.real.iterdir() if d.is_dir() and d.name.isdigit())
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(work, [(str(d), str(a.out), ffmpeg) for d in sources], chunksize=4))
    bad = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    for row in rows:
        row["split"] = "dev" if int(hashlib.sha256(row["object"].encode()).hexdigest(), 16) % 1000 < a.dev_share * 1000 else "reserve"
    (a.out / "manifest_all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    dev = [r for r in rows if r["split"] == "dev"]
    (a.out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dev))
    (a.out / "splits" / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dev))
    with open(a.out / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "scenario", "video_path", "first_frame_path", "caption", "fps", "frames", "width", "height"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in dev)
    lengths = sorted(r["frames"] for r in rows) or [0]
    print(f"PISA_CLIPS {len(rows)} clips ({len(bad)} failed), dev {len(dev)} of {len({r['object'] for r in dev})} objects, reserve {len(rows) - len(dev)}; "
          f"frames min/median/max {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}; errors {bad[:3]}")


if __name__ == "__main__":
    main()
