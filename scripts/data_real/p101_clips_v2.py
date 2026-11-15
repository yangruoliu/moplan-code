#!/usr/bin/env python
"""Physics 101 -> 7 fps clips for the Stage-2 v2 protocol, from the first native frame of every usable camera.

Views (all 1920x1080, checked by eye):
  cam1  Camera_1    front / side view, every scenario
  kin   Kinect_RGB  second upright view from another angle, every scenario
  top   Camera_2    bird's-eye view, ramp and multi only (for fall / spring / liquid Camera_2 is rotated by 90 degrees:
                    gravity points sideways in the image, so it is not used)
For every trial and view: a 1280x720 clip at 7 fps starting at the native first frame, at most 37 frames (the Stage-2
deployment length), plus a lossless PNG of the native first frame.  `frames_extracted` is what the source provides
(most fall / ramp trials last 0.9-1.4 s, i.e. 6-9 frames); the cache builder decides the 4k+1 window and whether a
clip whose objects are at rest may be extended.  Clips with fewer than --min-frames frames go to short.jsonl.
Splits are assigned later, by object (data_real/p101_splits.py).
Cutting, encoding (libx264) and frame counting use one ffmpeg binary: --ffmpeg, else $FFMPEG, else the one bundled with
imageio-ffmpeg (the ffmpeg / ffprobe of the conda lock file do not start without an extra step, see docs/INSTALL.md).
Re-encoded clips and the scaled first frames depend on the ffmpeg / x264 build.

usage: p101_clips_v2.py --catalog catalog.jsonl --out DIR [--workers 8] [--min-frames 9] [--max-frames 37] [--limit N]
                        [--ffmpeg EXE]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FPS = 7.0
CAMERAS = {"cam1": "video_path", "kin": "kinect_rgb_path", "top": "side_video_path"}
TOP_SCENARIOS = {"ramp", "multi"}


def ffmpeg_exe(explicit: str = "") -> str:
    """--ffmpeg, else $FFMPEG, else the ffmpeg bundled with imageio-ffmpeg (pinned in requirements.txt)."""
    if explicit or os.environ.get("FFMPEG"):
        return explicit or os.environ["FFMPEG"]
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_frames(path: Path, ffmpeg: str) -> int:
    """Number of decoded frames of the first video stream (what `ffprobe -count_frames` reports), from a full decode
    with ffmpeg.  Raises when ffmpeg fails or reports no frame count, so a broken tool cannot pass as a short clip."""
    r = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null",
                        "-progress", "pipe:1", "-nostats", "-"], capture_output=True, text=True)
    counts = [line.split("=", 1)[1] for line in r.stdout.splitlines() if line.startswith("frame=")]
    if r.returncode != 0 or not counts:
        raise RuntimeError(f"frame count of {path} failed: {r.stderr.strip()[-200:]}")
    return int(counts[-1])


def largest_4k1(n: int, cap: int) -> int:
    n = min(n, cap)
    return 0 if n < 1 else 4 * ((n - 1) // 4) + 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-frames", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=37)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--crf", type=int, default=12)
    p.add_argument("--ffmpeg", default="", help="ffmpeg binary with libx264 (default: $FFMPEG, else the one bundled with imageio-ffmpeg)")
    a = p.parse_args()
    ffmpeg = ffmpeg_exe(a.ffmpeg)
    try:
        check = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
    except OSError as error:
        raise SystemExit(f"{ffmpeg} does not start ({error}); pass --ffmpeg") from error
    if check.returncode != 0 or " libx264 " not in check.stdout:
        raise SystemExit(f"{ffmpeg} does not run or has no libx264 encoder; pass --ffmpeg")
    rows = [json.loads(line) for line in a.catalog.read_text().splitlines() if line.strip()]
    if a.limit:
        rows = rows[: a.limit]
    clips, frames_dir = a.out / "clips", a.out / "first_frames"
    clips.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    def work(item):
        row, camera = item
        source = row.get(CAMERAS[camera])
        sid = f"{row['sample_id']}__{camera}"
        if not source or not Path(source).is_file():
            return {"sample_id": sid, "error": "missing source"}
        clip, frame = clips / f"{sid}.mp4", frames_dir / f"{sid}.png"
        try:
            if not frame.is_file():
                x = subprocess.run(["nice", "-n", "10", ffmpeg, "-y", "-v", "error", "-i", source, "-vf", "scale=1280:720", "-frames:v", "1",
                                    str(frame)], capture_output=True, text=True)
                if x.returncode != 0:
                    return {"sample_id": sid, "error": "frame: " + x.stderr[-200:]}
            if not clip.is_file():
                x = subprocess.run(["nice", "-n", "10", ffmpeg, "-y", "-v", "error", "-i", source, "-vf", f"fps={FPS:g},scale=1280:720",
                                    "-frames:v", str(a.max_frames), "-c:v", "libx264", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-an", str(clip)],
                                   capture_output=True, text=True)
                if x.returncode != 0:
                    return {"sample_id": sid, "error": "clip: " + x.stderr[-200:]}
            n_raw = probe_frames(clip, ffmpeg)
        except Exception as error:  # noqa: BLE001
            return {"sample_id": sid, "error": repr(error)[:200]}
        n = largest_4k1(n_raw, a.max_frames)
        record = {"sample_id": sid, "trial_id": row["sample_id"], "camera": camera, "scenario": f"phys101/{row['scenario']}",
                  "caption": row["caption"], "video_path": str(clip.resolve()), "first_frame_path": str(frame.resolve()),
                  "source_path": source,
                  "fps": FPS, "frames": n, "frames_extracted": n_raw, "width": 1280, "height": 720, "domain": "phys101",
                  "object": row["object"], "material": row.get("material"), "setting": row["setting"], "trial": row.get("trial"),
                  "source_duration": row.get("duration"), "mass": row.get("mass"), "vol": row.get("vol")}
        if n < a.min_frames:
            record["error"] = f"short: {n_raw} frames extracted"
        return record

    items = [(row, camera) for row in rows for camera in CAMERAS if camera != "top" or row["scenario"] in TOP_SCENARIOS]
    print(f"{len(rows)} trials -> {len(items)} clips", flush=True)
    results = []
    with ThreadPoolExecutor(a.workers) as pool:
        for count, record in enumerate(pool.map(work, items), 1):
            results.append(record)
            if count % 200 == 0:
                print(f"{count}/{len(items)}", flush=True)
    good = [r for r in results if "error" not in r]
    short = [r for r in results if r.get("error", "").startswith("short")]
    failed = [r for r in results if "error" in r and not r["error"].startswith("short")]
    with open(a.out / "manifest.jsonl", "w") as handle:
        for r in good:
            handle.write(json.dumps(r) + "\n")
    with open(a.out / "short.jsonl", "w") as handle:
        for r in short:
            handle.write(json.dumps(r) + "\n")
    with open(a.out / "errors.jsonl", "w") as handle:
        for r in failed:
            handle.write(json.dumps(r) + "\n")
    hist = {}
    for r in good:
        key = (r["scenario"], r["frames"])
        hist[key] = hist.get(key, 0) + 1
    summary = {"clips": len(good), "short": len(short), "failed": len(failed),
               "frames_histogram": {f"{s}:{f}": c for (s, f), c in sorted(hist.items())}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("P101_CLIPS_DONE", json.dumps(summary))


if __name__ == "__main__":
    main()
