#!/usr/bin/env python3
"""29-frame training pools + mix of the single-run GT SFT -> <run>/OVERRIDES.json and POOLS_RECEIPT.json.

Provenance tool: it wrote the frozen training id lists <root>/pools/ids/*.txt that the 41-frame run reuses unchanged
(data/ids/stage2_sft/train in the release); v2/make_pools_v2.py builds the pools of the released run.

Pools (symlink folders under <root>/pools, frozen id lists under <root>/pools/ids):
  p_<scenario>   Physics 101, camera 1 (--p101-cam1-cache) + the two other views (<cache-root>/p101r_v2, same
                 reconciled-plan contract), split by scenario
  ssv2           Something-Something v2 strict train (<cache-root>/ssv2_v2: cond_real re-rendered from reconciled plans)
  phyco          PhyCo simulation, ground-truth plan (--phyco-existing-cache folders in order, then <cache-root>/phyco_new)
Mix (default): real 60 % = Physics 101 50 % (scenarios ~ clip count^0.75: fixed per-scenario shares of 30/25/20/15/10 %
would replay the 298 spring clips ~40 times in 128k draws) + SSv2 10 %; PhyCo 40 %; kind real (GT plan) everywhere;
tube FM weights.
Validation pools (held out, never trained on): <root>/val/{p101 (dev split minus dev96 / panels), phyco (Stage-1 val), ssv2 (strict hold)}.
Every id is checked against <root>/inventory/exclusion_lists.json (P101 by trial: no camera of an evaluation trial).
--validate loads every payload (multiprocessing) and drops unreadable / nonconforming files.

usage: make_pools.py --root S2SFT --run RUN_DIR --p101-cam1-cache DIR --phyco-existing-cache DIR [...] [--cache-root DIR]
         [--validate] [--workers 16] [--max-steps 8000] [--phyco-share 0.4 --ssv2-share 0.1]
       <root>/inventory (inventory.py output), <root>/phyco_new/ids.txt (select_phyco_new.py output)
"""
from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path

SCEN = ("fall", "ramp", "spring", "multi", "liquid")


def read(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).is_file() else []


def check(path: str):
    import torch
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
        assert x.get("cache_format_version") == "stage2_cond_v2"
        assert list(x["condition_channels_v2"]) == ["known", "availability", "selector", "raw_flow_x", "raw_flow_y", "motion_prob", "source_mask"]
        t = x["video_latent"].shape[1]
        assert x["cond_real"].shape[0] == 27 and x["cond_real"].shape[1] == t and x["image_cond_latent"].shape[1] == t
        assert x["fm_weight"].shape[0] == t and bool(torch.isfinite(x["cond_real"].float()).all()) and bool(torch.isfinite(x["video_latent"].float()).all())
        return path, True, int(t)
    except Exception as error:  # noqa: BLE001
        return path, False, repr(error)[:200]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--p101-cam1-cache", required=True, help="29-frame Physics 101 camera-1 payloads")
    p.add_argument("--phyco-existing-cache", action="append", default=[], help="29-frame PhyCo payload folders, searched in order")
    p.add_argument("--cache-root", default="", help="folder with p101r_v2, ssv2_v2, phyco_new (default <root>/cache)")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--phyco-share", type=float, default=0.40)
    p.add_argument("--ssv2-share", type=float, default=0.10)
    p.add_argument("--scenario-power", type=float, default=0.75, help="P101 scenario weight ~ count^power")
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--global-batch", type=int, default=16)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--keep-every", type=int, default=500)
    p.add_argument("--limit", type=int, default=0, help="smoke: first N ids of every pool")
    a = p.parse_args()
    root, run = Path(a.root), Path(a.run)
    inv = root / "inventory"
    pdir = root / "pools"
    (pdir / "ids").mkdir(parents=True, exist_ok=True)
    cache_root = Path(a.cache_root) if a.cache_root else root / "cache"
    sources = {
        "p101": [(Path(a.p101_cam1_cache), read(inv / "ids/p101_cam1.txt")), (cache_root / "p101r_v2", read(inv / "ids/p101_otherviews.txt"))],
        "ssv2": [(cache_root / "ssv2_v2", read(inv / "ids/ssv2.txt"))],
        "phyco": [(Path(d), read(inv / "ids/phyco_existing.txt")) for d in a.phyco_existing_cache]
                 + [(cache_root / "phyco_new", read(root / "phyco_new/ids.txt"))],
    }
    chosen: dict[str, dict[str, Path]] = {}
    for pool, entries in sources.items():
        chosen[pool] = {}
        for folder, ids in entries:
            for sid in ids:
                f = folder / f"{sid}.pt"
                if sid not in chosen[pool] and f.is_file():
                    chosen[pool][sid] = f
    bad = {}
    if a.validate:
        files = [str(f) for pool in chosen.values() for f in pool.values()]
        with Pool(a.workers) as workers:
            for path, ok, info in workers.imap_unordered(check, files, chunksize=32):
                if not ok:
                    bad[path] = info
        for pool in chosen.values():
            for sid in [s for s, f in pool.items() if str(f) in bad]:
                del pool[sid]
    excl = json.loads((inv / "exclusion_lists.json").read_text())
    eval_ids = set().union(*map(set, excl.values()))
    eval_trials = {s.split("__")[0] for s in eval_ids if s.startswith("p101_")}
    leaks = [s for pool in chosen.values() for s in pool if s in eval_ids or (s.startswith("p101_") and s.split("__")[0] in eval_trials)]
    if leaks:
        raise SystemExit(f"evaluation clips in the pools: {leaks[:5]} ...")
    for pool, files in chosen.items():
        folder = pdir / pool
        folder.mkdir(exist_ok=True)
        for sid, f in files.items():
            link = folder / f"{sid}.pt"
            if link.is_symlink() and os.readlink(link) != str(f):
                link.unlink()
            if not link.exists():
                link.symlink_to(f)
    suffix = f"_limit{a.limit}" if a.limit else ""
    lists = {}
    for scen in SCEN:
        ids = sorted(set(read(inv / f"ids/p101_{scen}.txt")) & set(chosen["p101"]))
        lists[f"p_{scen}"] = ("p101", ids[: a.limit] if a.limit else ids)
    for pool in ("ssv2", "phyco"):
        ids = sorted(chosen[pool])
        lists[pool] = (pool, ids[: a.limit] if a.limit else ids)
    for name, (_, ids) in lists.items():
        (pdir / "ids" / f"{name}{suffix}.txt").write_text("".join(s + "\n" for s in ids))
    real_p101 = 1.0 - a.phyco_share - a.ssv2_share
    raw = {s: len(lists[f"p_{s}"][1]) ** a.scenario_power for s in SCEN}
    weights = {f"p_{s}": real_p101 * raw[s] / sum(raw.values()) for s in SCEN}
    weights.update(ssv2=a.ssv2_share, phyco=a.phyco_share)
    draws = a.max_steps * a.global_batch
    overrides = {
        "pools": ",".join(f"{name}={pdir / folder}@{pdir / 'ids' / f'{name}{suffix}.txt'}" for name, (folder, _) in lists.items()),
        "mix": ",".join(f"{name}:real:{weights[name]:.5f}" for name in lists),
        "fm_weight_mode": "tube", "max_latent_frames": 8, "init_from": "", "max_steps": a.max_steps, "save_every": a.save_every,
        "keep_every": a.keep_every, "global_batch": a.global_batch, "log_every": 10, "seed": 42,
        "val_pools": ",".join(f"{n}={root / 'val' / n}@{root / 'val' / f'ids_{n}.txt'}" for n in ("p101", "phyco", "ssv2")), "val_sigmas": 2}
    run.mkdir(parents=True, exist_ok=True)
    (run / "OVERRIDES.json").write_text(json.dumps(overrides, indent=1) + "\n")
    unique = sum(len(ids) for _, ids in lists.values())
    receipt = {"pools": {name: {"clips": len(ids), "weight": round(weights[name], 5), "expected_draws": round(weights[name] * draws),
                                "expected_passes": round(weights[name] * draws / max(1, len(ids)), 2)} for name, (_, ids) in lists.items()},
               "unique_clips": unique, "sources": {pool: {str(folder): sum(1 for f in chosen[pool].values() if f.parent == folder) for folder, _ in entries}
                                                   for pool, entries in sources.items()},
               "validated": a.validate, "dropped_invalid": bad, "eval_leaks": 0, "draws": draws, "mean_passes": round(draws / max(1, unique), 2)}
    (run / "POOLS_RECEIPT.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
