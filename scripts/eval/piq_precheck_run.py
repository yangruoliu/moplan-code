#!/usr/bin/env python3
"""Pre-flight check of a Physics-IQ run folder (mirrors and tightens the official scorers' input rules).

usage: piq_precheck_run.py RUN_DIR [--n 198] [--duration 5.0] [--json OUT.json]

Checks (any failure -> exit 1):
  * the folder contains exactly N entries, all regular *.mp4 files (no subfolders, no hidden/other files)
  * sorted names start with 0001_, 0002_, ..., N (4-digit zero-padded IDs, in order)
  * ffprobe format=duration of every file is within 0.001 s of DURATION (the scorers' math.isclose(abs_tol=0.001))
  * a single FPS for the whole folder, both as ffprobe r_frame_rate and as OpenCV CAP_PROP_FPS
    (the scorers read FPS with cv2 and require one value per folder)
  * every file decodes to >= fps*DURATION frames with OpenCV frame count (the scorers use the first fps*5 frames)
Prints a JSON summary (fps, frame counts, resolutions, duration range) to stdout.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys

import cv2


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,pix_fmt:format=duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    # same call the official scorers use for the duration check
    dur_txt = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    cap = cv2.VideoCapture(path)
    cv_fps = cap.get(cv2.CAP_PROP_FPS)
    cv_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "duration": float(dur_txt), "width": s.get("width"), "height": s.get("height"),
        "r_frame_rate": s.get("r_frame_rate"), "avg_frame_rate": s.get("avg_frame_rate"),
        "nb_frames": s.get("nb_frames"), "codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt"),
        "cv_fps": cv_fps, "cv_frames": cv_n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=198)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    errors = []
    entries = sorted(os.listdir(a.run_dir))
    bad = [e for e in entries if not (e.endswith(".mp4") and not e.startswith(".")
                                      and os.path.isfile(os.path.join(a.run_dir, e)))]
    if bad:
        errors.append(f"non-mp4 / hidden / non-regular entries present: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    mp4 = [e for e in entries if e not in bad]
    if len(entries) != a.n:
        errors.append(f"folder has {len(entries)} entries ({len(mp4)} mp4), expected exactly {a.n}")
    for i, f in enumerate(mp4, 1):
        if not f.startswith(f"{i:04d}_"):
            errors.append(f"file #{i} '{f}' does not start with expected ID {i:04d}_")
            break

    with cf.ThreadPoolExecutor(a.threads) as ex:
        info = dict(zip(mp4, ex.map(lambda f: ffprobe(os.path.join(a.run_dir, f)), mp4)))

    durs = [v["duration"] for v in info.values()]
    bad_dur = {f: v["duration"] for f, v in info.items() if abs(v["duration"] - a.duration) > 0.001}
    if bad_dur:
        errors.append(f"{len(bad_dur)} files not {a.duration}s (+-0.001): {dict(list(bad_dur.items())[:5])}")
    r_fps = sorted({v["r_frame_rate"] for v in info.values()})
    cv_fps = sorted({v["cv_fps"] for v in info.values()})
    if len(cv_fps) != 1:
        errors.append(f"inconsistent OpenCV FPS values: {cv_fps}")
    if len(r_fps) != 1:
        errors.append(f"inconsistent ffprobe r_frame_rate values: {r_fps}")
    fps = cv_fps[0] if len(cv_fps) == 1 else None
    if fps is not None:
        if abs(fps - round(fps)) > 1e-6:
            errors.append(f"non-integer FPS {fps}: the scorers use int(fps) for GT folder names")
        need = int(round(fps * a.duration))
        short = {f: v["cv_frames"] for f, v in info.items() if v["cv_frames"] < need}
        if short:
            errors.append(f"{len(short)} files have fewer than fps*{a.duration}={need} frames: {dict(list(short.items())[:5])}")
    res = sorted({f"{v['width']}x{v['height']}" for v in info.values()})
    summary = {
        "run_dir": os.path.basename(os.path.abspath(a.run_dir)), "n_files": len(mp4), "n_entries": len(entries),
        "fps_cv2": cv_fps, "r_frame_rate": r_fps,
        "avg_frame_rate": sorted({v["avg_frame_rate"] for v in info.values()}),
        "resolutions": res, "codecs": sorted({str(v["codec"]) for v in info.values()}),
        "pix_fmts": sorted({str(v["pix_fmt"]) for v in info.values()}),
        "cv_frame_counts": sorted({v["cv_frames"] for v in info.values()}),
        "duration_min": min(durs) if durs else None, "duration_max": max(durs) if durs else None,
        "ok": not errors, "errors": errors,
    }
    txt = json.dumps(summary, indent=1)
    print(txt)
    if a.json:
        with open(a.json, "w") as f:
            f.write(txt + "\n")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
