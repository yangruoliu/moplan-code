#!/usr/bin/env python
"""Parallel form of `scripts/control/rerender_real.py --apply FIELDS --cache WORK --out FINAL` (same result per clip:
payload.update(fields); plan_source = plans_real_reconciled_v1; a clip without fields is copied unchanged).
usage: apply_fields.py --cache WORK --fields FIELDS --out FINAL [--workers 16]
"""
import argparse, json, os, shutil
from multiprocessing import Pool
from pathlib import Path

import torch

A = None


def one(name):
    cache, fields, out = Path(A.cache), Path(A.fields), Path(A.out)
    target = out / name
    if target.is_file():
        return "skip"
    new = fields / name
    if new.is_file():
        payload = torch.load(cache / name, map_location="cpu", weights_only=False)
        payload.update(torch.load(new, map_location="cpu", weights_only=False))
        payload["plan_source"] = "plans_real_reconciled_v1"
        tmp = out / f"{name}.tmp-{os.getpid()}"
        torch.save(payload, tmp)
        tmp.replace(target)
        return "replaced"
    shutil.copy(cache / name, target)
    return "copied"


def main():
    global A
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--fields", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    A = p.parse_args()
    torch.set_num_threads(1)
    Path(A.out).mkdir(parents=True, exist_ok=True)
    names = sorted(q.name for q in Path(A.cache).glob("*.pt") if not q.name.startswith("prompt_embeds"))
    with Pool(A.workers) as pool:
        res = pool.map(one, names, chunksize=8)
    summary = {k: res.count(k) for k in ("replaced", "copied", "skip")}
    (Path(A.out) / "APPLY_SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("RERENDER_APPLY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
