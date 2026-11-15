#!/usr/bin/env python
"""Physics-101 other-camera (p101r), step 2: dump the mask-centroid readings that reconcile_plans.py needs (labels_real of the Stage-2 cache
entry + its frame_seconds) as small npz files, so the Stage-1 builder can reconstruct the plan provenance and
reconcile the plan exactly like the cam1 cache (relabel_v2.provenance_from_residuals -> reconcile_plans.reconcile)
without touching the Stage-2 cache.  usage: p101r_labels.py --cache DIR --meta meta.jsonl --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--meta", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    count = 0
    for line in a.meta.read_text().splitlines():
        sid = json.loads(line)["sample_id"]
        x = torch.load(a.cache / f"{sid}.pt", map_location="cpu", weights_only=False, mmap=True)
        labels = {k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in x["labels_real"].items()}
        np.savez_compressed(a.out / f"{sid}.npz", _times=np.asarray(x["frame_seconds"], np.float64), **labels)
        count += 1
    print(json.dumps({"labels": count}))


if __name__ == "__main__":
    main()
