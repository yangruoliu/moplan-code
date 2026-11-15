#!/usr/bin/env python
"""The fields of a Stage-2 cache that are rendered from the measured plan, again from the current plans_real (e.g. after
reconcile_plans.py), on the CPU: cond_real, fm_weight (tube weights), known_frames_real, plus the current labels_real.

Writes <out>/<sid>.pt with only these fields; apply_fields() merges them into a copy of the full payloads elsewhere
(the full cache is large, the fields travel).  --check: report, for clips whose plan did not change, whether the
re-rendered fields equal the cached ones bit for bit (the renderer must reproduce the cache).
usage: rerender_real.py --cache DIR --ids F --out DIR [--check] [--shard-index i --num-shards n]
       rerender_real.py --apply FIELDS_DIR --cache DIR --out NEW_CACHE_DIR     (copies DIR's payloads with the fields replaced)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import motion_plan as mp  # noqa: E402
from scripts.control import plan_condition as pc  # noqa: E402

FIELDS = ("cond_real", "fm_weight", "known_frames_real", "labels_real")


def render(payload: dict, plan: mp.MotionPlan) -> dict:
    times = [float(t) for t in payload["frame_seconds"]]
    size = (int(payload["height"]), int(payload["width"]))
    image_latent = payload["image_cond_latent"].float()
    latent_hw = tuple(image_latent.shape[-2:])
    labels = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in payload["labels_real"].items()}
    return {"cond_real": pc.render_v2("real", plan, image_latent, times, size).half(),
            "fm_weight": pc.tube_weights(plan, labels, times, latent_hw).half(),
            "known_frames_real": [bool(any(mp.affine_at(o, plan.anchors, t)[1] for o in plan.objects)) for t in times],
            "labels_real": payload["labels_real"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--ids", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--check", action="store_true")
    p.add_argument("--apply", default="")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.apply:
        fields = Path(a.apply)
        done = copied = 0
        for path in sorted(cache.glob("*.pt")):
            if path.name.startswith("prompt_embeds"):
                continue
            target = out / path.name
            new = fields / path.name
            if new.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                payload.update(torch.load(new, map_location="cpu", weights_only=False))
                payload["plan_source"] = "plans_real_reconciled_v1"
                torch.save(payload, target)
                done += 1
            else:
                shutil.copy(path, target)
                copied += 1
        for extra in cache.glob("ids_*.txt"):
            shutil.copy(extra, out / extra.name)
        print("RERENDER_APPLY", json.dumps({"replaced": done, "copied_unchanged": copied}))
        return
    ids = [line.strip() for line in open(a.ids) if line.strip()] if a.ids else sorted(q.stem for q in cache.glob("*.pt"))
    stats = {"clips": 0, "changed_plans": 0, "unchanged_equal": 0, "unchanged_differ": 0}
    for sid in ids[a.shard_index:: a.num_shards]:
        payload = torch.load(cache / f"{sid}.pt", map_location="cpu", weights_only=False)
        plan = mp.load_plan(cache / "plans_real" / f"{sid}.npz")
        fields = render(payload, plan)
        stats["clips"] += 1
        old_plan = cache / "plans_real_pre_reconcile" / f"{sid}.npz"
        changed = True
        if old_plan.is_file():
            before = mp.load_plan(old_plan)
            changed = any(not np.array_equal(o.affine, n.affine) or not np.array_equal(o.known, n.known) for o, n in zip(before.objects, plan.objects))
        stats["changed_plans"] += int(changed)
        if a.check and not changed:
            same = torch.equal(fields["cond_real"], payload["cond_real"]) and fields["known_frames_real"] == list(payload["known_frames_real"])
            stats["unchanged_equal" if same else "unchanged_differ"] += 1
        torch.save(fields, out / f"{sid}.pt")
    print("RERENDER", json.dumps(stats))


if __name__ == "__main__":
    main()
