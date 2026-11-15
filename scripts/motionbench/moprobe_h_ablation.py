#!/usr/bin/env python3
"""MoProbe-H ablation numbers.
  SFT only vs MoPlan: Delta_both over the five scored dims (dboth5 of moplan_paired_A.py), MoPlan - moplan_norl, paired on
      the scored frames, first-frame bootstrap 2000 seed 0 -- records exactly as moprobe_h_paired.py
  zero-motion plan vs MoPlan: every MoProbe-H ex clip of moplan and moplan_zero; obj_peak = camera_patterns.per_clip's peak
      camera-compensated world displacement of the main object (work-frame px); share with obj_peak >= 16 px (+ 8 / 20 px
      sensitivity) and the median obj_peak, all clips and the frames both arms have (paired difference, first-frame bootstrap)
Stops when built items are missing (moprobe_h_rules.require_items) unless --allow-missing.
usage: MOPROBE_ROOT=DIR python3 moprobe_h_ablation.py --out OUT_DIR [--workers 12] [--allow-missing]
       -> OUT_DIR/{patterns_moplan_zero_ex.jsonl, moprobe_h_ablation.json}"""
import argparse, collections, json, os, random, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2]))
import moprobe_h_rules as V                                    # noqa: E402
import moprobe_h_paired as MP                                  # noqa: E402
from moplan_paired_A import paired, dboth5, same5               # noqa: E402

M = V.M; THR = (8.0, 16.0, 20.0)


def boot_diff(a, b, stat, n=2000, seed=0):
    """a, b: {frame: [values]} -> stat(A) - stat(B) on shared frames with a first-frame bootstrap"""
    ks = sorted(set(a) & set(b)); rng = random.Random(seed); out = []
    f = lambda g, keys: stat([v for k in keys for v in g[k]])
    d0 = f(a, ks) - f(b, ks)
    for _ in range(n):
        s = [ks[rng.randrange(len(ks))] for _ in ks]; out.append(f(a, s) - f(b, s))
    out.sort(); return dict(frames=len(ks), a=f(a, ks), b=f(b, ks), diff=d0, ci=[out[int(0.025 * n)], out[int(0.975 * n) - 1]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--allow-missing", action="store_true", help="go on when built items are missing (default: stop)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    V.require_items(V.b_rows(), a.allow_missing)
    fs = json.load(open(V.DATA / "finding_set_B.json")); F1 = fs["f1_frames"]; diag = fs.get("f1_diag_frames", {})
    byf = collections.defaultdict(dict)
    for r in V.b_rows():
        byf[r["frame"]][r["cond"]] = r
    res = {}
    # ---- Delta_both, MoPlan - SFT-only
    MP.ARMS = ["moplan", "moplan_norl"]
    tasks = []
    for sid, dim in sorted(set(F1.items()) | set(diag.items())):
        c = byf.get(sid, {}); ce, cc = V.PAIR_CONDS.get(dim, ("ex", "cf"))
        if ce in c and cc in c:
            tasks.append((sid, dim, c[ce]["items_dir"], c[ce]["instruction"], c[cc]["instruction"]))
    with Pool(a.workers) as pool:
        recs = [r for g in pool.map(MP.f1_one, tasks, chunksize=4) for r in g]
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        if r["type"] in V.SCORED and F1.get(r["item"]) == r["type"]:
            g[r["arm"]][r["item"]].append(r)
    for name, st in (("dboth5", dboth5), ("same5", same5)):
        res[f"moplan-moplan_norl/{name}"] = paired(g["moplan"], g["moplan_norl"], st)
    res["coverage_f1"] = {arm: dict(frames=len(g[arm]), records=sum(len(v) for v in g[arm].values())) for arm in MP.ARMS}
    # ---- main object moves, MoPlan vs zero plan, every H ex clip (b_rows: H rows minus the rejected frames)
    ct = []
    for r in V.b_rows():
        if r["cond"] != "ex":
            continue
        for arm in ("moplan", "moplan_zero"):
            sid = f"{r['frame']}-ex"; mp = str(V.MEASURE / f"{arm}__{sid}__s0.measure.npz")
            if os.path.isfile(mp):
                ct.append(dict(arm=arm, base=r["frame"], cond="ex", sid=sid, items_dir=r["items_dir"], mp=mp))
    ct = list({(t["arm"], t["sid"]): t for t in ct}.values())
    with Pool(a.workers) as pool:
        pats = [p for p, _ in pool.map(V.pattern_task_safe, ct, chunksize=8) if p is not None and "err" not in p]
    with open(out / "patterns_moplan_zero_ex.jsonl", "w") as f:
        for p in pats:
            f.write(json.dumps(p, default=float) + "\n")
    pk = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in pats:
        if p.get("obj_peak") is not None:
            pk[p["arm"]][p["item"]].append(float(p["obj_peak"]))
    for arm in ("moplan", "moplan_zero"):
        v = [x for xs in pk[arm].values() for x in xs]
        res[f"{arm}/moves"] = dict(clips=len(v), frames=len(pk[arm]), median_px=float(np.median(v)) if v else None,
                                   **{f"share_ge{int(t)}": float(np.mean([x >= t for x in v])) if v else None for t in THR})
    for t in THR:
        res[f"paired_moves_ge{int(t)}"] = boot_diff(pk["moplan_zero"], pk["moplan"], lambda xs, t=t: float(np.mean([x >= t for x in xs])))
    res["paired_median_px"] = boot_diff(pk["moplan_zero"], pk["moplan"], lambda xs: float(np.median(xs)))
    json.dump(res, open(out / "moprobe_h_ablation.json", "w"), indent=1, default=float)
    p = res["moplan-moplan_norl/dboth5"]
    print(f"dboth5 MoPlan - SFT-only: frames {p['frames']}  {100 * p['a']:+.1f} vs {100 * p['b']:+.1f}  diff {100 * p['diff']:+.1f} [{100 * p['ci'][0]:+.1f}, {100 * p['ci'][1]:+.1f}]  coverage {res['coverage_f1']}")
    for arm in ("moplan", "moplan_zero"):
        r = res[f"{arm}/moves"]
        if r["clips"]:
            print(f"{arm:12s} clips {r['clips']:3d} frames {r['frames']:3d}  moves>=8 {100 * r['share_ge8']:.1f}  >=16 {100 * r['share_ge16']:.1f}  >=20 {100 * r['share_ge20']:.1f}  median {r['median_px']:.0f} px")
    for k in ["paired_moves_ge8", "paired_moves_ge16", "paired_moves_ge20", "paired_median_px"]:
        r = res[k]
        if r["frames"]:
            sc = 1 if "median" in k else 100
            print(f"    {k:18s} frames {r['frames']:3d}  zero {sc * r['a']:.1f}  moplan {sc * r['b']:.1f}  zero-moplan {sc * r['diff']:+.1f} [{sc * r['ci'][0]:+.1f}, {sc * r['ci'][1]:+.1f}]")


if __name__ == "__main__":
    main()
