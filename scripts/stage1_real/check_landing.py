#!/usr/bin/env python
"""Quantitative landing check of real Stage-1 targets against an independent instrument: the SAM2 mask-centroid
readings of the Stage-2 reward scene (scenes_v5/<sid>.npz: real [K, T, 2] work px, real_ok, times = every clip frame).

For a rigid / affine target the mean flow over an object's first-frame pixels is the displacement of its centroid, so
per mover and observed horizon t with a scene reading within --tol seconds:
  error = | mean target flow over the mover's valid pixels  -  scene centroid displacement (work -> grid scale) |
reported in 160-grid px, also relative to the displacement magnitude.
usage: check_landing.py --root ROOT --scenes DIR [--tol 0.025] [--native-hw 720 1280] --out J.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--scenes", required=True, type=Path)
    p.add_argument("--tol", type=float, default=0.025)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    errors, rel, per_h = [], [], {}
    used = 0
    for split in ("train", "val"):
        for line in (a.root / "manifests" / f"{split}.jsonl").read_text().splitlines():
            sid = json.loads(line)["sample_id"]
            scene_path = a.scenes / f"{sid}.npz"
            if not scene_path.is_file():
                continue
            t = torch.load(a.root / "targets" / f"{sid}.pt", map_location="cpu", weights_only=True)
            scene = np.load(scene_path)
            slots = [int(v) for v in scene["slots"]]
            content = t["content_valid"][0].numpy()
            rows, cols = np.nonzero(content)
            grid_h, grid_w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
            height, width = int(scene["mask_shape"][1]), int(scene["mask_shape"][2])
            scale = np.array([grid_w / width, grid_h / height])
            ids = t["instance_id"][0].numpy()
            flow, valid = t["flow"].float().numpy(), (t["flow_valid"][:, 0] & t["visibility"][:, 0]).numpy()
            slot_of = dict(zip(t["object_instance_ids"], t["object_slots"]))
            for iid in t["mover_instance_ids"]:
                slot = slot_of.get(iid)
                if slot not in slots:
                    continue
                k = slots.index(slot)
                for j, when in enumerate(t["horizon_seconds"]):
                    chosen = (ids == iid) & valid[j]
                    if chosen.sum() < 4:
                        continue
                    near = np.abs(scene["times"] - when)
                    f = int(near.argmin())
                    if near[f] > a.tol or not bool(scene["real_ok"][k, f]):
                        continue
                    mine = np.array([flow[j, 0][chosen].mean(), flow[j, 1][chosen].mean()])
                    ref = scene["real"][k, f].astype(np.float64) * scale
                    err = float(np.linalg.norm(mine - ref))
                    errors.append(err)
                    rel.append(err / max(float(np.linalg.norm(ref)), 1.0))
                    per_h.setdefault(round(float(when), 3), []).append(err)
                    used += 1
    errors, rel = np.array(errors), np.array(rel)
    report = {"root": str(a.root), "comparisons": int(used),
              "median_err_grid_px": float(np.median(errors)) if used else None,
              "p90_err_grid_px": float(np.percentile(errors, 90)) if used else None,
              "frac_within_2px_or_20pct": float(np.mean((errors <= 2.0) | (rel <= 0.2))) if used else None,
              "per_horizon_median": {str(h): [len(v), float(np.median(v))] for h, v in sorted(per_h.items())}}
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
