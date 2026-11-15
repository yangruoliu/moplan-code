#!/usr/bin/env python3
"""Per-task / per-category breakdowns of model A vs model B (default: MoPlan vs the untrained Wan2.2 under the same protocol,
448x256), paired bootstrap 95% CIs (10,000 replicates, seed 20261024), each cell marked favourable / unfavourable for A
(* = CI excludes 0).
  rigid     RigidBench per task (5 x 20 samples) and overall x the 10 official metrics, per-sample pairs (results.json samples);
            the overall means are checked against results.json "aggregated"
  morpheus  Morpheus per Fig. 8 category (9 buckets of morph_aggregate_br.py) x Total / PhysInv / Dyn, each model's reported prompt type,
            discarded -> 0, pairs by (experiment, seed)
Model labels are the launchers' labels (moplan, base, zero, ...): RigidBench <root>/rigidbench/runs/<label>/results.json,
Morpheus <root>/morpheus/results/single_frame_conditioning/MP-<label>/{aggregate.json,per_video.jsonl} (morph_aggregate_br.py).
usage: osp_breakdown.py rigid|morpheus --root EVAL_ROOT [--a moplan] [--b base]
       -> stdout table + <root>/paired/breakdown_<bench>_<a>_vs_<b>.json"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")                                                # set from --root in __main__
HIGHER = {"iou": True, "ssim": True, "l2": False, "chamfer": False, "ate": False, "ate3d": False, "si_mse": False, "lpips": False,
          "iddrift": False, "bgdrift": False, "total": True, "physinv": True, "dyn": True}
BUCKET = {"falling_ball": "Falling", "falling_apple": "Falling", "falling_marker": "Falling", "falling_tape": "Falling",
          "bouncing_ball": "Bouncing", "projectile": "Projectile", "holonomic_pendulum": "Holonomic", "double_pendulum": "Double Pendulum",
          "spring": "Spring", "sliding_book": "Sliding Book", "rolling_full_can": "Rolling", "rolling_empty_can": "Rolling",
          "rolling_orange": "Rolling", "collision_equal": "Collision", "collision_big_hits_small": "Collision", "collision_small_hits_big": "Collision"}
rng = np.random.default_rng(20261024)


def cell(a: np.ndarray, b: np.ndarray, higher: bool) -> dict:
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    idx = rng.integers(0, len(d), (10000, len(d)))
    bm = d[idx].mean(1)
    lo, hi = float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
    fav = bool((d.mean() > 0) == higher) if d.mean() != 0 else None
    sig = bool(lo > 0 or hi < 0)
    return {"n": int(len(d)), "a": float(a.mean()), "b": float(b.mean()), "diff": float(d.mean()), "ci": [lo, hi], "favourable": fav, "significant": sig}


def fmt(c: dict, prec: int = 3) -> str:
    tag = "fav" if c["favourable"] else ("unfav" if c["favourable"] is False else "tie")
    return f"{c['a']:.{prec}f}/{c['b']:.{prec}f} {c['diff']:+.{prec}f}[{c['ci'][0]:+.{prec}f},{c['ci'][1]:+.{prec}f}] {tag}{'*' if c['significant'] else ''}"


def rigid(a_name: str, b_name: str) -> dict:
    load = lambda m: json.loads((ROOT / f"rigidbench/runs/{m}/results.json").read_text())
    ra, rb = load(a_name), load(b_name)
    sa = {s["sample_id"]: s for s in ra["samples"]}
    sb = {s["sample_id"]: s for s in rb["samples"]}
    ids = sorted(set(sa) & set(sb))
    metrics = ["iou", "l2", "chamfer", "ate", "ate3d", "si_mse", "ssim", "lpips", "iddrift", "bgdrift"]
    val = lambda s, m: float("nan") if s.get(m) is None or (isinstance(s.get(m), float) and math.isnan(s[m])) else float(s[m])
    check = {m: (float(np.nanmean([val(sa[i], m) for i in ids])), ra["aggregated"].get(m)) for m in metrics}
    tasks = sorted({sa[i]["task_type"] for i in ids})
    out = {"bench": "rigidbench", "a": a_name, "b": b_name, "n": len(ids), "check_overall_mean_vs_aggregated_a": check, "rows": {}}
    for t in ["overall"] + tasks:
        sel = [i for i in ids if t == "overall" or sa[i]["task_type"] == t]
        out["rows"][t] = {m: cell(np.array([val(sa[i], m) for i in sel]), np.array([val(sb[i], m) for i in sel]), HIGHER[m]) for m in metrics}
    print(f"RigidBench {a_name} vs {b_name}: cells = a/b diff[CI] fav|unfav (*: CI excludes 0); n={len(ids)}")
    print("check (per-sample mean vs results.json aggregated, model a):", {m: (round(x, 4), None if y is None else round(y, 4)) for m, (x, y) in check.items()})
    for t, row in out["rows"].items():
        print(f"\n[{t}] n={row['iou']['n']}")
        for m in metrics:
            print(f"  {m:8s} {fmt(row[m], 4 if m in ('si_mse', 'bgdrift') else 3)}")
    return out


def morpheus(a_name: str, b_name: str) -> dict:
    def rows(model: str):
        d = ROOT / "morpheus/results/single_frame_conditioning" / model
        agg = json.loads((d / "aggregate.json").read_text())[model]
        best = agg["best_prompt_by_total"]
        r = {}
        for line in (d / "per_video.jsonl").read_text().splitlines():
            x = json.loads(line)
            if x["pt"] != best:
                continue
            phys = 0.0 if x["disc"] else (x["phys"] or 0.0)
            dyn = 0.0 if x["disc"] else (x["dyn"] or 0.0)
            r[(x["exp"], int(Path(x["video"]).name))] = {"total": (phys + dyn) / 2, "physinv": phys, "dyn": dyn}
        return best, r
    (pa, ra), (pb, rb) = rows(f"MP-{a_name}"), rows(f"MP-{b_name}")
    keys = sorted(set(ra) & set(rb))
    out = {"bench": "morpheus", "a": a_name, "b": b_name, "prompt_a": pa, "prompt_b": pb, "n": len(keys), "rows": {}}
    for cat in ["overall"] + sorted({BUCKET[k[0]] for k in keys}):
        sel = [k for k in keys if cat == "overall" or BUCKET[k[0]] == cat]
        out["rows"][cat] = {m: cell(np.array([ra[k][m] for k in sel]), np.array([rb[k][m] for k in sel]), True) for m in ("total", "physinv", "dyn")}
    print(f"Morpheus {a_name} ({pa}) vs {b_name} ({pb}), per-video zeroed, pairs by (experiment, seed); cells = a/b diff[CI] fav|unfav")
    for cat, row in out["rows"].items():
        print(f"  {cat:16s} n={row['total']['n']:2d}  total {fmt(row['total'])}  physinv {fmt(row['physinv'])}  dyn {fmt(row['dyn'])}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bench", choices=("rigid", "morpheus"))
    p.add_argument("--root", type=Path, required=True, help="evaluation root ($EVAL_ROOT)")
    p.add_argument("--a", default="moplan"); p.add_argument("--b", default="base")
    a = p.parse_args()
    ROOT = a.root
    res = {"rigid": rigid, "morpheus": morpheus}[a.bench](a.a, a.b)
    path = ROOT / f"paired/breakdown_{a.bench}_{a.a}_vs_{a.b}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(path)
