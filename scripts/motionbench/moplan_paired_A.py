#!/usr/bin/env python3
"""MoPlan vs the untrained Wan2.2-A14B under the SAME
generation protocol (arm wan_p448: 448x256, 37 frames, 7 fps, same frames / captions), paired on the same first frames of
test set A, from the per-clip records the finding analyses already wrote (same rules, nothing re-measured):
  F1 (H)  notext_records.jsonl, ex / cf only: same5 (same outcome under real vs opposite instruction, five scored dims),
          dboth5 (both followed minus sum_d w_d p_ex,d p_cf,d), dir (lr + updown, real instruction, right | moved on axis)
  F3 (H)  camera_deep_records.jsonl, real-instruction clips (cond ex) on the first frames whose real continuation has a
          still camera: share with camera motion; camera_patterns_records.jsonl: pans >= 16 px with a moving mover,
          share whose direction follows the mover (cos >= 0.5, chance 1/3)
Differences are A - B over the frames both arms have; 95% intervals by resampling first frames (2000).  The open models at
their own default protocol are printed for context only (not the comparison).  F2 (P) comes from f2_merged.json later.
usage: python3 moplan_paired_A.py --dir <folder with the records + finding_set_A.json> [--out paired.json]
"""
import argparse, collections, json
from pathlib import Path

import numpy as np

OPEN = ["wan22_a14b", "wan22_ti2v5b", "hunyuan15_480p", "cogvideox15_5b", "ltx23"]
SCORED = ["relation", "noun", "lr", "updown", "nothing"]
T_DIR = 16.0
PAIRS = [("moplan", "wan_p448"), ("moplan_norl", "wan_p448"), ("moplan", "moplan_norl"), ("moplan_zero", "wan_p448"), ("moplan_nosft", "wan_p448"), ("moplan_zero", "moplan"), ("moplan_nosft", "moplan")]


def jl(p):
    return [json.loads(l) for l in open(p)]


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def paired(groups_a, groups_b, stat, n=2000, seed=0):
    """groups_*: {frame: [records]}; stat(list) -> float.  Point estimates on the shared frames, CI of A - B."""
    keys = sorted(set(groups_a) & set(groups_b))
    pool = lambda g, ks: [r for k in ks for r in g[k]]
    a, b = stat(pool(groups_a, keys)), stat(pool(groups_b, keys))
    rng = np.random.default_rng(seed); ds = []
    for _ in range(n):
        ks = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        ds.append(stat(pool(groups_a, ks)) - stat(pool(groups_b, ks)))
    return dict(frames=len(keys), a=a, b=b, diff=a - b, ci=[float(np.nanpercentile(ds, 2.5)), float(np.nanpercentile(ds, 97.5))])


def same5(rs):
    return mean([r["same_excf"] for r in rs])


def dboth5(rs):
    if not rs:
        return float("nan")
    byd = collections.defaultdict(list)
    for r in rs:
        byd[r["type"]].append(r)
    base = sum(len(v) / len(rs) * mean([r["ex_real"] for r in v]) * mean([r["cf_opp"] for r in v]) for v in byd.values())
    return mean([r["ex_real"] and r["cf_opp"] for r in rs]) - base


def dirright(rs):
    xs = [r for r in rs if r["type"] in ("lr", "updown") and (r["ex_real"] or r["ex_opp"])]
    return mean([r["ex_real"] for r in xs])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", default="")
    a = ap.parse_args(); D = Path(a.dir)
    H = json.load(open(D / "finding_set_A.json"))["H"]
    res = {"F1": {}, "F3": {}}
    # ---- F1
    recs = [r for r in jl(D / "notext_records.jsonl") if r["item"] in H and r["type"] == H[r["item"]] and r["type"] in SCORED]
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        g[r["arm"]][r["item"]].append(r)
    print("## F1 on H (five scored dims; ex / cf; paired on shared first frames)")
    for A, B in PAIRS:
        for name, st in (("same5", same5), ("dboth5", dboth5), ("dir", dirright)):
            p = paired(g[A], g[B], st); res["F1"][f"{A}-{B}/{name}"] = p
            print(f"  {A:12s} vs {B:12s} {name:7s} frames {p['frames']:3d}  {p['a']:+.3f} vs {p['b']:+.3f}  diff {p['diff']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]")
    for m in OPEN:
        res["F1"][f"{m}/same5"] = same5([r for v in g[m].values() for r in v])
    print("  context (own default protocol): " + ", ".join(f"{m} same5 {res['F1'][m + '/same5']:.2f}" for m in OPEN))
    # ---- F3 camera on still-camera frames
    real = {}
    for fn in ("camera_finding_records.jsonl", "camera_finding_unfit.jsonl"):
        for x in jl(D / fn):
            if x.get("arm") == "real":
                real[x["item"]] = x.get("verdict") or ("moved" if x.get("any") else "still")
    still = {i for i in H if real.get(i) == "still"}
    cam = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in jl(D / "camera_deep_records.jsonl"):
        it = x["sid"].rsplit("-", 1)[0]
        if x.get("cond") == "ex" and x.get("seed", 0) == 0 and it in still and x["verdict"] in ("moved", "still"):
            cam[x["arm"]][it].append(x)
    camrate = lambda xs: mean([x["verdict"] == "moved" for x in xs])
    print(f"\n## F3 on H: camera motion on the {len(still)} still-camera first frames (real instruction)")
    for A, B in PAIRS:
        p = paired(cam[A], cam[B], camrate); res["F3"][f"{A}-{B}/cam"] = p
        print(f"  {A:12s} vs {B:12s} cam    frames {p['frames']:3d}  {p['a']:.3f} vs {p['b']:.3f}  diff {p['diff']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]")
    # ---- F3 follow (pans >= 16 px with a moving mover; all H conditions of the arm, as in camera_patterns.py)
    pans = collections.defaultdict(list)
    for x in jl(D / "camera_patterns_records.jsonl"):
        if x.get("item") in H and x.get("trans", 0) >= T_DIR and "obj_world_at_campk" in x and np.linalg.norm(x["obj_world_at_campk"]) >= 8:
            pans[x["arm"]].append(cos(x["cam_vec"], x["obj_world_at_campk"]) >= 0.5)
    print("\n## F3 on H: pans that follow the mover (chance 0.33)")
    for m in ["moplan", "moplan_norl", "wan_p448", "moplan_zero", "moplan_nosft"] + OPEN:
        xs = pans.get(m, [])
        res["F3"][f"{m}/follow"] = dict(n=len(xs), follow=mean(xs))
        print(f"  {m:14s} pans {len(xs):3d} follow {mean(xs):.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
