#!/usr/bin/env python3
"""Stage-2 GT-SFT training pools + mix -> <run>/OVERRIDES.json (read by v2/train_s2sft_v2.py) and POOLS_RECEIPT.json.

Pools (frozen id lists, data/ids/stage2_sft/train in the release), with the 41-frame payloads of the cache builders:
  p_<scenario>  Physics 101 (camera 1 + the two other views): <cache-dir>/p101
  ssv2          SSv2 strict train:                            <cache-dir>/ssv2
  phyco         PhyCo ground-truth plan:                      <cache-dir>/phyco
A clip without a 41-frame payload (a PhyCo clip whose raw video is unavailable, or a failed build) uses its 29-frame
payload from <fallback-dir>/pools/<pool> (listed in the receipt).  Validation: the ids_<name>.txt lists of
<val-ids-dir>, payloads from <cache-dir>/{p101,ssv2,phyco_val} (29-frame payload from <fallback-dir>/val/<name> where
none was built).  Every id is checked against the exclusion lists (P101 by trial: no camera of an evaluation trial).
--validate loads every payload (multiprocessing): drops unreadable / nonconforming files and reports the latent-frame
statistics per pool (latents per clip, share of clips with all 11, and of the late latents 8-10 how many show real
footage / the held last frame / the junction).
Writes symlink folders <out-dir>/pools/{p101,ssv2,phyco}, <out-dir>/val/{p101,phyco,ssv2} and the kept id lists
<out-dir>/pools/ids/<name>.txt, <out-dir>/val/ids_<name>.txt; OVERRIDES.json points at them.

usage (release layout):
  make_pools_v2.py --ids-dir data/ids/stage2_sft/train --val-ids-dir data/ids/stage2_sft/val
      --exclusions data/ids/stage2_sft/exclusion_lists.json --cache-dir CACHE --fallback-dir FALLBACK
      --out-dir DATA_ROOT --run RUN_DIR [--validate] [--workers 32]
usage (one root): make_pools_v2.py --root ROOT --run RUN_DIR   (ROOT/pools/ids, ROOT/val, ROOT/inventory, ROOT/v2/cache,
  fallbacks ROOT/pools/* and ROOT/val/*, output ROOT/v2)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from multiprocessing import Pool
from pathlib import Path

SCEN = ("fall", "ramp", "spring", "multi", "liquid")
LATE0, NLAT = 8, 11


def read(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).is_file() else []


def latent_kind(j: int, frames_real: int) -> str:
    last = frames_real - 1
    return "real" if 4 * j <= last else ("held" if 4 * j - 3 > last else "mixed")


def check(path: str):
    import torch
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
        assert x.get("cache_format_version") == "stage2_cond_v2"
        assert list(x["condition_channels_v2"]) == ["known", "availability", "selector", "raw_flow_x", "raw_flow_y", "motion_prob", "source_mask"]
        t = x["video_latent"].shape[1]
        assert x["cond_real"].shape[0] == 27 and x["cond_real"].shape[1] == t and x["image_cond_latent"].shape[1] == t
        assert x["fm_weight"].shape[0] == t and bool(torch.isfinite(x["cond_real"].float()).all()) and bool(torch.isfinite(x["video_latent"].float()).all())
        assert t <= NLAT
        return path, True, {"t": int(t), "frames_real": int(x.get("frames_real", x.get("frames", 4 * t - 3))), "frames": int(x.get("frames", 4 * t - 3)),
                            "at_rest": bool(x.get("at_rest", False))}
    except Exception as error:  # noqa: BLE001
        return path, False, repr(error)[:200]


def stats(infos: list[dict]) -> dict:
    lat = collections.Counter(i["t"] for i in infos)
    late = collections.Counter()
    for i in infos:
        for j in range(LATE0, i["t"]):
            late[latent_kind(j, i["frames_real"])] += 1
    n = max(1, len(infos))
    total_late = sum(late.values())
    return {"clips": len(infos), "latents_hist": dict(sorted(lat.items())), "mean_latents": round(sum(i["t"] for i in infos) / n, 2),
            "share_full11": round(lat.get(NLAT, 0) / n, 3), "padded_clips": sum(i["frames"] > i["frames_real"] for i in infos),
            "late_latents": dict(late), "late_share": {k: round(v / max(1, total_late), 3) for k, v in late.items()}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="", help="one data root holding every input below in its default place")
    p.add_argument("--ids-dir", default="", help="training id lists <name>.txt (default <root>/pools/ids)")
    p.add_argument("--val-ids-dir", default="", help="validation id lists ids_<name>.txt (default <root>/val)")
    p.add_argument("--exclusions", default="", help="exclusion_lists.json (default <root>/inventory/exclusion_lists.json)")
    p.add_argument("--cache-dir", default="", help="41-frame payloads <dir>/{p101,ssv2,phyco,phyco_val} (default <root>/v2/cache)")
    p.add_argument("--fallback-dir", default="", help="29-frame payloads <dir>/pools/<pool>, <dir>/val/<name> (default <root>)")
    p.add_argument("--out-dir", default="", help="symlink pools and kept id lists are written here (default <root>/v2)")
    p.add_argument("--run", required=True)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--phyco-share", type=float, default=0.40)
    p.add_argument("--ssv2-share", type=float, default=0.10)
    p.add_argument("--scenario-power", type=float, default=0.75)
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--global-batch", type=int, default=16)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--keep-every", type=int, default=500)
    p.add_argument("--limit", type=int, default=0, help="smoke: first N ids of every pool")
    p.add_argument("--allow-missing", action="store_true", help="write the pools even if some payloads exist nowhere "
                   "(their links dangle and training stops when it reaches one); default: stop with a list of them")
    a = p.parse_args()
    run = Path(a.run)
    root = Path(a.root) if a.root else None

    def place(value: str, default: str) -> Path:
        if value:
            return Path(value)
        if root is None:
            raise SystemExit(f"give --root or every path option ({default} is missing)")
        return root / default

    ids_dir, val_ids_dir = place(a.ids_dir, "pools/ids"), place(a.val_ids_dir, "val")
    exclusions, cache, fallback_root = place(a.exclusions, "inventory/exclusion_lists.json"), place(a.cache_dir, "v2/cache"), place(a.fallback_dir, ".")
    v2 = place(a.out_dir, "v2")
    pdir = v2 / "pools"
    (pdir / "ids").mkdir(parents=True, exist_ok=True)
    v1_ids = {name: read(ids_dir / f"{name}.txt") for name in [f"p_{s}" for s in SCEN] + ["ssv2", "phyco"]}
    folder_of = {"p101": cache / "p101", "ssv2": cache / "ssv2", "phyco": cache / "phyco"}
    v1_folder = {"p101": fallback_root / "pools/p101", "ssv2": fallback_root / "pools/ssv2", "phyco": fallback_root / "pools/phyco"}
    pool_of = {name: ("p101" if name.startswith("p_") else name) for name in v1_ids}
    chosen: dict[str, dict[str, Path]] = {"p101": {}, "ssv2": {}, "phyco": {}}
    fallback: dict[str, list[str]] = {"p101": [], "ssv2": [], "phyco": []}
    for name, ids in v1_ids.items():
        pool = pool_of[name]
        for sid in ids:
            f = folder_of[pool] / f"{sid}.pt"
            if not f.is_file():
                f = (v1_folder[pool] / f"{sid}.pt").resolve()
                fallback[pool].append(sid)
            chosen[pool][sid] = f
    val = {}
    val_src = {"p101": cache / "p101", "ssv2": cache / "ssv2", "phyco": cache / "phyco_val"}
    for name, (src, v1dir) in {"p101": (val_src["p101"], fallback_root / "val/p101"), "ssv2": (val_src["ssv2"], fallback_root / "val/ssv2"),
                               "phyco": (val_src["phyco"], fallback_root / "val/phyco")}.items():
        val[name] = {}
        for sid in read(val_ids_dir / f"ids_{name}.txt"):
            f = src / f"{sid}.pt"
            val[name][sid] = f if f.is_file() else (v1dir / f"{sid}.pt").resolve()
    missing = sorted(sid for pool in list(chosen.values()) + list(val.values()) for sid, f in pool.items() if not f.is_file())
    if missing and not a.validate and not a.allow_missing:
        raise SystemExit(f"{len(missing)} payloads exist neither in {cache} nor in the fallback folders, e.g. {missing[:5]}")
    bad, infos = {}, {}
    if a.validate:
        files = [str(f) for pool in list(chosen.values()) + list(val.values()) for f in pool.values()]
        with Pool(a.workers) as workers:
            for path, ok, info in workers.imap_unordered(check, files, chunksize=16):
                if ok:
                    infos[path] = info
                else:
                    bad[path] = info
        for pool in chosen.values():
            for sid in [s for s, f in pool.items() if str(f) in bad]:
                del pool[sid]
    excl = json.loads(exclusions.read_text())
    eval_ids = set().union(*map(set, excl.values()))
    eval_trials = {s.split("__")[0] for s in eval_ids if s.startswith("p101_")}
    leaks = [s for pool in chosen.values() for s in pool if s in eval_ids or (s.startswith("p101_") and s.split("__")[0] in eval_trials)]
    if leaks:
        raise SystemExit(f"evaluation clips in the pools: {leaks[:5]} ...")
    train_ids = set().union(*map(set, chosen.values()))
    val_overlap = [s for v in val.values() for s in v if s in train_ids]
    if val_overlap:
        raise SystemExit(f"validation clips in the training pools: {val_overlap[:5]}")
    for pool, files in list(chosen.items()) + [(f"../val/{n}", v) for n, v in val.items()]:
        folder = (pdir / pool).resolve() if not pool.startswith("../") else (v2 / "val" / pool[7:])
        folder.mkdir(parents=True, exist_ok=True)
        for sid, f in files.items():
            link = folder / f"{sid}.pt"
            if link.is_symlink() and os.readlink(link) != str(f):
                link.unlink()
            if not link.exists():
                link.symlink_to(f)
    for n, v in val.items():
        (v2 / "val" / f"ids_{n}.txt").write_text("".join(s + "\n" for s in sorted(v)))
    suffix = f"_limit{a.limit}" if a.limit else ""
    lists = {}
    for name, ids in v1_ids.items():
        kept = [s for s in ids if s in chosen[pool_of[name]]]
        lists[name] = (pool_of[name], kept[: a.limit] if a.limit else kept)
        (pdir / "ids" / f"{name}{suffix}.txt").write_text("".join(s + "\n" for s in lists[name][1]))
    real_p101 = 1.0 - a.phyco_share - a.ssv2_share
    raw = {s: len(lists[f"p_{s}"][1]) ** a.scenario_power for s in SCEN}
    weights = {f"p_{s}": real_p101 * raw[s] / sum(raw.values()) for s in SCEN}
    weights.update(ssv2=a.ssv2_share, phyco=a.phyco_share)
    draws = a.max_steps * a.global_batch
    overrides = {
        "pools": ",".join(f"{name}={pdir / folder}@{pdir / 'ids' / f'{name}{suffix}.txt'}" for name, (folder, _) in lists.items()),
        "mix": ",".join(f"{name}:real:{weights[name]:.5f}" for name in lists),
        "fm_weight_mode": "tube", "max_latent_frames": NLAT, "init_from": "", "max_steps": a.max_steps, "save_every": a.save_every,
        "keep_every": a.keep_every, "global_batch": a.global_batch, "log_every": 10, "seed": 42,
        "val_pools": ",".join(f"{n}={v2 / 'val' / n}@{v2 / 'val' / f'ids_{n}.txt'}" for n in ("p101", "phyco", "ssv2")), "val_sigmas": 2}
    run.mkdir(parents=True, exist_ok=True)
    (run / "OVERRIDES.json").write_text(json.dumps(overrides, indent=1) + "\n")
    unique = sum(len(ids) for _, ids in lists.values())
    receipt = {"pools": {name: {"clips": len(ids), "weight": round(weights[name], 5), "expected_draws": round(weights[name] * draws),
                                "expected_passes": round(weights[name] * draws / max(1, len(ids)), 2),
                                **({"latents": stats([infos[str(chosen[pool][s])] for s in ids if str(chosen[pool][s]) in infos])} if infos else {})}
                         for name, (pool, ids) in lists.items()},
               "unique_clips": unique, "v1_payload_fallback": {k: {"n": len(v), "ids": v} for k, v in fallback.items()},
               "validation": {n: {"clips": len(v), "v1_payloads": sum(f.parent != val_src[n] for f in v.values()),
                                  **({"latents": stats([infos[str(f)] for f in v.values() if str(f) in infos])} if infos else {})} for n, v in val.items()},
               "validated": a.validate, "dropped_invalid": bad, "eval_leaks": 0, "draws": draws, "mean_passes": round(draws / max(1, unique), 2)}
    if infos:
        allinfo = [infos[str(f)] for pool in chosen.values() for f in pool.values() if str(f) in infos]
        receipt["all_training_clips_latents"] = stats(allinfo)
        mixed = collections.Counter()
        for name, (pool, ids) in lists.items():
            st = stats([infos[str(chosen[pool][s])] for s in ids if str(chosen[pool][s]) in infos])
            for k in ("mean_latents", "share_full11"):
                mixed[k] += weights[name] * st[k]
        receipt["mix_weighted"] = {k: round(v, 3) for k, v in mixed.items()}
    (run / "POOLS_RECEIPT.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "v1_payload_fallback"} | {"v1_payload_fallback": {k: v["n"] for k, v in receipt["v1_payload_fallback"].items()}}, indent=1))


if __name__ == "__main__":
    main()
