#!/usr/bin/env python3
"""Step 3 of the MoProbe SSv2 build: per-clip video stats + a cheap camera-motion estimate.

For every candidate clip: decode (cv2), downscale to 160 px wide grayscale, estimate the
global translation between consecutive frames with phase correlation and an ECC affine
(scale / rotation) on a coarser grid. Writes one CSV row per clip:
  id, n_frames, fps, w, h, dur_s, cam_med_pct_w_per_s, cam_peak_pct_w, cam_cum_pct_w,
  scale_dev_pct, rot_deg_max, motion_energy, decode_ok
Thresholds are applied later (ssv2_select.py); this script only measures.
"""
import argparse, csv, math, multiprocessing as mp, os, sys
from pathlib import Path

import cv2
import numpy as np

W_SMALL = 160


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, fps


def analyse(args):
    vid, path = args
    row = dict(id=vid, n_frames=0, fps=0.0, w=0, h=0, dur_s=0.0, cam_med_pct_w_per_s=np.nan,
               cam_peak_pct_w=np.nan, cam_cum_pct_w=np.nan, bcam_med_pct_w_per_s=np.nan, bcam_peak_pct_w=np.nan,
               bcam_cum_pct_w=np.nan, scale_dev_pct=np.nan, rot_deg_max=np.nan, motion_energy=np.nan, decode_ok=0)
    try:
        frames, fps = read_frames(path)
    except Exception:
        return row
    if len(frames) < 4:
        return row
    h, w = frames[0].shape[:2]
    row.update(n_frames=len(frames), fps=float(fps), w=int(w), h=int(h), dur_s=len(frames) / fps, decode_ok=1)
    scale = W_SMALL / w
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (W_SMALL, int(round(h * scale))))
             for f in frames]
    win = cv2.createHanningWindow((small[0].shape[1], small[0].shape[0]), cv2.CV_32F)
    sh, sw = small[0].shape
    border = np.ones((sh, sw), np.float32)
    border[int(0.25 * sh):int(0.75 * sh), int(0.25 * sw):int(0.75 * sw)] = 0.0   # ignore the centre: hands / objects
    shifts, bshifts, energies = [], [], []
    for a, b in zip(small[:-1], small[1:]):
        fa, fb = a.astype(np.float32), b.astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(fa, fb, win)
        shifts.append((dx, dy))
        (bx, by), _ = cv2.phaseCorrelate(fa * border, fb * border, win)
        bshifts.append((bx, by))
        energies.append(float(np.mean(np.abs(fa - fb))))
    shifts = np.array(shifts) / W_SMALL * 100.0                     # percent of width per frame
    step = np.linalg.norm(shifts, axis=1)
    cum = np.cumsum(shifts, axis=0)
    row["cam_med_pct_w_per_s"] = float(np.median(step) * fps)
    row["cam_peak_pct_w"] = float(step.max())
    row["cam_cum_pct_w"] = float(np.linalg.norm(cum, axis=1).max())
    bshifts = np.array(bshifts) / W_SMALL * 100.0
    bstep = np.linalg.norm(bshifts, axis=1)
    bcum = np.cumsum(bshifts, axis=0)
    row["bcam_med_pct_w_per_s"] = float(np.median(bstep) * fps)
    row["bcam_peak_pct_w"] = float(bstep.max())
    row["bcam_cum_pct_w"] = float(np.linalg.norm(bcum, axis=1).max())
    row["motion_energy"] = float(np.mean(energies))
    # ECC affine on a sparse set of frame pairs (first→middle→last) for scale / rotation
    try:
        idx = [0, len(small) // 2, len(small) - 1]
        scales, rots = [], []
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
        for i, j in zip(idx[:-1], idx[1:]):
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                cv2.findTransformECC(small[i].astype(np.float32) / 255.0, small[j].astype(np.float32) / 255.0,
                                     warp, cv2.MOTION_AFFINE, crit, None, 5)
            except cv2.error:
                continue
            a, b = warp[0, 0], warp[0, 1]
            scales.append(math.hypot(a, b)); rots.append(math.degrees(math.atan2(b, a)))
        if scales:
            row["scale_dev_pct"] = float(max(abs(s - 1.0) for s in scales) * 100.0)
            row["rot_deg_max"] = float(max(abs(r) for r in rots))
    except Exception:
        pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--split", default="official_val")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    with open(args.candidates) as f:
        ids = [r["id"] for r in csv.DictReader(f) if r["split"] == args.split]
    if args.limit:
        ids = ids[:args.limit]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {r["id"] for r in csv.DictReader(f)}
    todo = [(i, os.path.join(args.video_dir, f"{i}.webm")) for i in ids if i not in done]
    print(f"{len(ids)} clips, {len(done)} done, {len(todo)} to do", flush=True)
    fields = ["id", "n_frames", "fps", "w", "h", "dur_s", "cam_med_pct_w_per_s", "cam_peak_pct_w", "cam_cum_pct_w",
              "bcam_med_pct_w_per_s", "bcam_peak_pct_w", "bcam_cum_pct_w", "scale_dev_pct", "rot_deg_max",
              "motion_energy", "decode_ok"]
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f, mp.Pool(args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for n, row in enumerate(pool.imap_unordered(analyse, todo, chunksize=8), 1):
            w.writerow(row)
            if n % 500 == 0:
                f.flush(); print(f"  {n}/{len(todo)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
