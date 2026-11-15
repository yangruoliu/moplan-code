#!/usr/bin/env python3
"""Generator-agnostic TRACK step (GPU): measure clips with the benchmark instrument and store Readings.

Clip manifest (jsonl): {"sid": ..., "arm": ..., "seed": 0, "path": "<mp4|webm|npy>", "fps": 16.0 (required for npy),
                        "start_frame": 0, "max_seconds": 4.0}
For every row: frames → work frame (letterbox; frame 0 is replaced by the item's stored t0 work frame when
--pin-first-frame), SAM2 propagation of the item's instance masks + CoTracker background grid →
<out>/<arm>__<sid>__s<seed>.measure.npz (a Reading). Special arm "real": re-measure the real continuation
(ceiling). Scoring is a separate CPU step (the motionbench/metrics rules), so thresholds never trigger re-tracking.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from motionbench import video as V                     # noqa: E402
from motionbench import instrument as I                # noqa: E402
from motionbench.io import load_built                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--items-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sam2-checkpoint", default=os.environ.get("SAM2_CHECKPOINT", ""), help="SAM 2.1 hiera-large .pt (default: $SAM2_CHECKPOINT)")
    ap.add_argument("--cotracker-repo", default=os.environ.get("COTRACKER_REPO", ""), help="co-tracker git checkout (default: $COTRACKER_REPO)")
    ap.add_argument("--cotracker-checkpoint", default=os.environ.get("COTRACKER_CKPT", ""), help="CoTracker3 scaled_offline.pth (default: $COTRACKER_CKPT)")
    ap.add_argument("--max-seconds", type=float, default=4.0)
    ap.add_argument("--max-frames", type=int, default=96)
    ap.add_argument("--pin-first-frame", action="store_true", help="replace frame 0 by the item's stored t0 work frame")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for key in ("sam2_checkpoint", "cotracker_repo", "cotracker_checkpoint"):
        if not getattr(a, key):
            ap.error(f"--{key.replace('_', '-')} (or its environment variable) is required")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if not a.overwrite:
        # shard the clips still to measure (not the whole manifest): in incremental runs the measured ones are spread
        # unevenly, which left one GPU idle while the other still had most of the work
        rows = [r for r in rows if not (out / f"{r['arm']}__{r['sid']}__s{int(r.get('seed', 0))}.measure.npz").is_file()]
    rows = rows[a.shard_index::a.num_shards]
    if a.limit:
        rows = rows[:a.limit]
    inst = I.MotionInstrument(a.cotracker_repo, a.cotracker_checkpoint, a.sam2_checkpoint, tmp_root=os.environ.get("TMPDIR"))
    log = open(out / f"measure-shard{a.shard_index:02d}of{a.num_shards:02d}.jsonl", "a")
    done = errors = 0
    for n, r in enumerate(rows):
        name = f"{r['arm']}__{r['sid']}__s{int(r.get('seed', 0))}"
        target = out / f"{name}.measure.npz"
        if target.is_file() and not a.overwrite:
            continue
        try:
            t0 = time.time()
            d, item, masks, _ = load_built(a.items_dir, r.get("item_sid") or V.base_sid(r["sid"]))
            path = str(r["path"])
            if path.endswith(".npy"):
                frames = V.read_npy(path)
                fps = float(r["fps"])
                frames = frames[: int(np.floor(a.max_seconds * fps + 1e-6)) + 1]
            else:
                frames, fps = V.read_video(path, start_frame=int(r.get("start_frame", 0)), max_seconds=r.get("max_seconds", a.max_seconds),
                                           fps_override=r.get("fps"), step=int(r.get("step", 1)), max_side=r.get("max_side"))
            if r.get("simulate_canvas", False):
                # the real future sent through the generated-clip path: native -> 16:9 canvas -> model resolution
                # (832x480, stretch) -> mp4-like JPEG artefacts -> back to the work frame (an honest instrument ceiling)
                x0, y0, cw, ch = V.canvas_box(d["native_hw"])
                sim = []
                for fr in frames:
                    canvas = np.zeros((V.CANVAS_H, V.CANVAS_W, 3), np.uint8)
                    canvas[y0:y0 + ch, x0:x0 + cw] = cv2.resize(fr, (cw, ch), interpolation=cv2.INTER_CUBIC)
                    sw, sh = r.get("simulate_size", (832, 480))           # the generator's own resolution (MoPlan arm: 448x256)
                    small = cv2.resize(canvas, (int(sw), int(sh)), interpolation=cv2.INTER_AREA)
                    ok_, enc = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    sim.append(cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                work = V.canvas_to_work(np.stack(sim), d["native_hw"], d["geometry"])
            elif r.get("canvas", False):
                crop = r.get("canvas_crop") or V.MODEL_CANVAS_CROP.get(r.get("arm"))
                work = V.canvas_to_work(frames, d["native_hw"], d["geometry"], crop=crop)   # generated: the 16:9 canvas
            else:
                work, geom = V.to_work(frames)                                      # real: the native frames
            if a.pin_first_frame:
                pin = cv2.imread(str(Path(a.items_dir) / "frames" / f"{(r.get('item_sid') or V.base_sid(r['sid']))}_t0_work.png"))
                if pin is not None:
                    work[0] = cv2.cvtColor(pin, cv2.COLOR_BGR2RGB)
            work, fps_eff, times = V.subsample(work, fps, a.max_frames)
            obj_pts, owner = I.object_points(masks)
            reading, _ = inst.measure(work, masks, times, fps_eff, extra_points=obj_pts, owner=owner, geom=d.get("geometry"))
            reading.meta.update(arm=r["arm"], seed=int(r.get("seed", 0)), source_fps=float(fps), path=path)
            reading.to_npz(target)
            log.write(json.dumps(dict(name=name, ok=True, frames=int(len(work)), fps=fps_eff, seconds=round(time.time() - t0, 1))) + "\n"); log.flush()
            done += 1
        except Exception as e:  # noqa: BLE001
            errors += 1
            log.write(json.dumps(dict(name=name, ok=False, error=repr(e)[:300])) + "\n"); log.flush()
            print(f"ERROR {name}: {e!r}"[:300], flush=True)
        if (n + 1) % 20 == 0:
            print(f"[measure] {n + 1}/{len(rows)} done {done} errors {errors}", flush=True)
    print(f"MEASURE_DONE shard {a.shard_index}/{a.num_shards}: done {done} errors {errors} of {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
