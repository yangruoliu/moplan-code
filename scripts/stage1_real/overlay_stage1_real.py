#!/usr/bin/env python
"""Visual check of real Stage-1 targets: does the cumulative flow at horizon t, applied to the first-frame mover
pixels, land on the object in the real frame at t?

Per clip one PNG row: [first frame + instance map + mover masks (red)] then, for each observed horizon, the real video
frame nearest to t (letterboxed into the same 160 grid, x3) with the movers' first-frame outline (red) and their
flow-displaced pixels (green dots = first-frame mover pixels moved by the target flow; |t_frame - t| is printed).
Valid static support is shaded blue.  usage: overlay_stage1_real.py --root ROOT --ids a b c --out DIR [--fps 7]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

SIDE, ZOOM = 160, 3


def letterbox(image: np.ndarray, side: int = SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = side / max(h, w)
    ih, iw = round(h * scale), round(w * scale)
    out = np.zeros((side, side, 3), np.uint8)
    top, left = (side - ih) // 2, (side - iw) // 2
    out[top:top + ih, left:left + iw] = cv2.resize(image, (iw, ih), interpolation=cv2.INTER_AREA)
    return out


def frames_of(path: str) -> list[np.ndarray]:
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out


def contour(canvas: np.ndarray, mask: np.ndarray, color) -> None:
    big = cv2.resize(mask.astype(np.uint8), (SIDE * ZOOM, SIDE * ZOOM), interpolation=cv2.INTER_NEAREST)
    found, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, found, -1, color, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--fps", type=float, default=7.0)
    p.add_argument("--video-dir", default="", help="read <video-dir>/<sid>.mp4 instead of the manifest's original_video_path")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for split in ("train", "val"):
        for line in (a.root / "manifests" / f"{split}.jsonl").read_text().splitlines():
            r = json.loads(line)
            rows[r["sample_id"]] = r
    report = []
    for sid in a.ids:
        t = torch.load(a.root / "targets" / f"{sid}.pt", map_location="cpu", weights_only=True)
        row = rows[sid]
        first = letterbox(cv2.imread(row["video_path"]))
        video = frames_of(str(Path(a.video_dir) / f"{sid}.mp4") if a.video_dir else row["original_video_path"])
        ids = t["instance_id"][0].numpy()
        movers = t["mover_instance_ids"]
        moving = np.isin(ids, movers)
        panels = []
        base = cv2.resize(first, (SIDE * ZOOM, SIDE * ZOOM), interpolation=cv2.INTER_NEAREST)
        rng = np.random.default_rng(0)
        colors = rng.integers(60, 255, (int(ids.max()) + 1, 3))
        tint = np.where(ids[..., None] > 0, colors[ids], 0).astype(np.uint8)
        base = cv2.addWeighted(base, 0.6, cv2.resize(tint, base.shape[1::-1], interpolation=cv2.INTER_NEAREST), 0.4, 0)
        contour(base, moving, (0, 0, 255))
        cv2.putText(base, f"{sid[:40]}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(base, f"movers {movers}", (3, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        panels.append(base)
        flow, valid, vis = t["flow"].float().numpy(), t["flow_valid"][:, 0].numpy(), t["visibility"][:, 0].numpy()
        clip_report = {"sample_id": sid, "movers": movers, "horizons": []}
        for j, when in enumerate(t["horizon_seconds"]):
            if not valid[j].any():
                continue
            index = int(round(when * a.fps))
            if index >= len(video):
                continue
            frame = letterbox(video[index])
            canvas = cv2.resize(frame, (SIDE * ZOOM, SIDE * ZOOM), interpolation=cv2.INTER_NEAREST)
            static = valid[j] & vis[j] & ~moving
            shade = np.zeros_like(canvas)
            shade[..., 0] = cv2.resize(static.astype(np.uint8) * 120, canvas.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            canvas = cv2.add(canvas, shade)
            contour(canvas, moving, (0, 0, 255))
            ys, xs = np.nonzero(moving & valid[j])
            for y, x in zip(ys[::3], xs[::3]):
                dx, dy = flow[j, 0, y, x], flow[j, 1, y, x]
                cv2.circle(canvas, (int((x + dx + 0.5) * ZOOM), int((y + dy + 0.5) * ZOOM)), 1, (0, 255, 0), -1)
            mean = [float(flow[j, c][moving & valid[j]].mean()) if (moving & valid[j]).any() else 0.0 for c in (0, 1)]
            cv2.putText(canvas, f"t={when:.3f}s frame {index / a.fps:.3f}s", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(canvas, f"mover mean d=({mean[0]:.1f},{mean[1]:.1f}) grid px", (3, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            panels.append(canvas)
            clip_report["horizons"].append({"t": when, "frame_t": index / a.fps, "mover_mean_flow_grid_px": mean,
                                            "valid_px": int(valid[j].sum()), "static_px": int(static.sum())})
        cv2.imwrite(str(a.out / f"{sid}.png"), np.concatenate(panels, 1))
        report.append(clip_report)
    (a.out / "overlay_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report)[:3000])


if __name__ == "__main__":
    main()
