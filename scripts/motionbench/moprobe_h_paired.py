#!/usr/bin/env python3
"""MoProbe-H (finding_set_B.json): MoPlan vs the untrained Wan2.2-A14B under the same protocol (arm wan_p448), paired on the same
first frames, with the definitions of moplan_paired_A.py / moplan_subsets_A.py (same5, dboth5, dir; per dimension same /
real / opp / dboth; Cam on still-camera first frames, real instruction; Follow on pans >= 16 px with a moving mover).
Records come from the measure files with the rules of moprobe_h_rules.py:
  F1   moprobe_h_rules.derive / record on the scored frames (data/moprobe/finding_set_B.json f1_frames + f1_diag_frames),
       conditions ex / cf (rex / rcf for role2), measures $MOPROBE_ROOT/measure/<arm>__<sid>__s0.measure.npz
  F3   moprobe_h_rules.cam_clip per clip (camera_deep verdict + camera_patterns per_clip); still-camera frames = the
       real-continuation verdict 'still' (--real-camera, written by moprobe_h_rules.py real-camera); Cam uses ex clips,
       Follow ex + cf (+ rex / rcf)
  context  optional: the five open models' records ($MOPROBE_ROOT/context/{f1_records_B.jsonl,cam_records.jsonl})
95% intervals of A - B by resampling first frames (2,000, seed 0).
Stops when built items of MoProbe-H frames are missing in MOPROBE_ROOT (moprobe_h_rules.require_items); --allow-missing
analyses without those frames.
usage: MOPROBE_ROOT=DIR python3 moprobe_h_paired.py --out OUT_DIR [--real-camera $MOPROBE_ROOT/real_cam.jsonl] [--workers 16]
           [--allow-missing]
       -> OUT_DIR/{f1_records_moplan_B.jsonl, cam_records_moplan_B.jsonl, moprobe_h_paired.json, moprobe_h_paired.md}"""
import argparse, collections, json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2]))
import moprobe_h_rules as V                                    # noqa: E402
from moplan_paired_A import paired, same5, dboth5, dirright, mean, cos   # noqa: E402

M = V.M; CONTEXT = V.CONTEXT
ARMS = ["moplan", "wan_p448"]; OPEN = V.OPEN; SCORED = V.SCORED; T_DIR = 16.0
PAIRS = [("moplan", "wan_p448")]


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def jl_optional(p):
    """context files of the open models are optional (they only add context lines)"""
    return jl(p) if Path(p).is_file() else []


def mp(arm, sid):
    return V.mp(arm, sid)


def f1_one(t):
    sid, dim, idir, ext, cft = t
    b = V.built(idir, sid)
    if b is None:
        return []
    P, _ = V.derive(dim, b[0], ext, cft)
    if P is None:
        return []
    ce, cc = V.PAIR_CONDS.get(dim, ("ex", "cf")); out = []
    for arm in ARMS:
        r, _ = V.record(dim, P, sid, arm, V.rd(mp(arm, f"{sid}-{ce}")), V.rd(mp(arm, f"{sid}-{cc}")), None, False)
        if r:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--real-camera", default=f"{M}/real_cam.jsonl", help="real-continuation camera verdicts (moprobe_h_rules.py real-camera)")
    ap.add_argument("--allow-missing", action="store_true", help="go on when built items are missing (default: stop)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fs = json.load(open(V.DATA / "finding_set_B.json")); F1 = fs["f1_frames"]; diag = fs.get("f1_diag_frames", {})
    rows = V.b_rows(); V.require_items(rows, a.allow_missing); byf = collections.defaultdict(dict)
    for r in rows:
        byf[r["frame"]][r["cond"]] = r
    tasks = []
    # one task per (frame, dimension): a frame can be both a scored item and a role / role2 diagnostic item
    for sid, dim in sorted(set(F1.items()) | set(diag.items())):
        c = byf.get(sid, {})
        ce, cc = V.PAIR_CONDS.get(dim, ("ex", "cf"))
        if ce in c and cc in c:
            tasks.append((sid, dim, c[ce]["items_dir"], c[ce]["instruction"], c[cc]["instruction"]))
    with Pool(a.workers) as pool:
        recs = [r for g in pool.map(f1_one, tasks, chunksize=4) for r in g]
    with open(out / "f1_records_moplan_B.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    recs_open = [r for r in jl_optional(CONTEXT / "f1_records_B.jsonl") if r["arm"] in OPEN]
    allrec = recs + recs_open
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in allrec:
        if r["type"] in SCORED and F1.get(r["item"]) == r["type"]:
            g[r["arm"]][r["item"]].append(r)
    res = {"F1": {}, "F1_dim": {}, "F3": {}, "coverage": {}}
    L = ["## MoProbe-H: MoPlan vs wan_p448 (same protocol; paired on shared first frames; 95% CI of the difference)", ""]
    for A, B in PAIRS:
        for name, st in (("same5", same5), ("dboth5", dboth5), ("dir", dirright)):
            p = paired(g[A], g[B], st); res["F1"][f"{A}-{B}/{name}"] = p
            L.append(f"F1 {name:7s} frames {p['frames']:3d}  {A} {100 * p['a']:.1f}  {B} {100 * p['b']:.1f}  diff {100 * p['diff']:+.1f} [{100 * p['ci'][0]:+.1f}, {100 * p['ci'][1]:+.1f}]")
        # per dimension (as moplan_subsets_A): same / real / opp / dboth (+ dir for lr / updown)
        stats = dict(same=lambda rs: mean([r["same_excf"] for r in rs]), real=lambda rs: mean([r["ex_real"] for r in rs]),
                     opp=lambda rs: mean([r["cf_opp"] for r in rs]),
                     dboth=lambda rs: mean([r["ex_real"] and r["cf_opp"] for r in rs]) - mean([r["ex_real"] for r in rs]) * mean([r["cf_opp"] for r in rs]) if rs else float("nan"))
        for dim in SCORED + ["role", "role2"]:
            ga, gb = collections.defaultdict(list), collections.defaultdict(list)
            for r in recs:
                if r["type"] != dim:
                    continue
                if r["arm"] == A:
                    ga[r["item"]].append(r)
                elif r["arm"] == B:
                    gb[r["item"]].append(r)
            if not ga or not gb:
                continue
            for name, st in list(stats.items()) + ([("dir", dirright)] if dim in ("lr", "updown") else []):
                p = paired(ga, gb, st); res["F1_dim"][f"{dim}/{name}"] = p
                L.append(f"  {dim:8s} {name:5s} frames {p['frames']:3d}  {100 * p['a']:.1f} vs {100 * p['b']:.1f}  diff {100 * p['diff']:+.1f} [{100 * p['ci'][0]:+.1f}, {100 * p['ci'][1]:+.1f}]")
    for m in OPEN:
        res["F1"][f"{m}/same5"] = same5([r for v in g[m].values() for r in v]); res["F1"][f"{m}/dboth5"] = dboth5([r for v in g[m].values() for r in v])
    L.append("context (own protocol): " + ", ".join(f"{m} same5 {100 * res['F1'][m + '/same5']:.0f}" for m in OPEN))
    # ---- F3
    real = {x["base"]: x["verdict"] for x in jl(a.real_camera) if x.get("ver") == "B"}
    still = {s for s in byf if real.get(s) == "still"}
    ctasks = []
    for sid, conds in byf.items():
        for cond, r in conds.items():
            if cond not in ("ex", "cf", "rex", "rcf"):
                continue
            for arm in ARMS:
                p = mp(arm, f"{sid}-{cond}")
                if p:
                    ctasks.append(dict(ver="B", arm=arm, base=sid, cond=cond, set="bench3", items_dir=r["items_dir"], mp=p))
    with Pool(a.workers) as pool:
        cam = pool.map(V.cam_clip, ctasks, chunksize=4)
    with open(out / "cam_records_moplan_B.jsonl", "w") as f:
        for x in cam:
            f.write(json.dumps(x, default=float) + "\n")
    cg = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in cam:
        if x.get("cond") == "ex" and x["base"] in still and x.get("verdict") in ("moved", "still"):
            cg[x["arm"]][x["base"]].append(x)
    for x in jl_optional(CONTEXT / "cam_records.jsonl"):          # context: the five models (ex) on the same still frames
        if x.get("ver") == "B" and x["arm"] in OPEN and x.get("cond") == "ex" and x["base"] in still and x.get("verdict") in ("moved", "still"):
            cg[x["arm"]][x["base"]].append(x)
    camrate = lambda xs: mean([x["verdict"] == "moved" for x in xs])
    for A, B in PAIRS:
        p = paired(cg[A], cg[B], camrate); res["F3"][f"{A}-{B}/cam"] = p
        L.append(f"F3 cam     frames {p['frames']:3d}  {A} {100 * p['a']:.1f}  {B} {100 * p['b']:.1f}  diff {100 * p['diff']:+.1f} [{100 * p['ci'][0]:+.1f}, {100 * p['ci'][1]:+.1f}]  (still-camera frames {len(still)})")
    for m in OPEN:
        res["F3"][f"{m}/cam"] = camrate([x for v in cg[m].values() for x in v])
    pans = collections.defaultdict(list)
    for x in cam:
        pt = x.get("pat")
        if pt and pt.get("trans", 0) >= T_DIR and pt.get("obj_world_at_campk") is not None and np.linalg.norm(pt["obj_world_at_campk"]) >= 8:
            pans[x["arm"]].append(cos(pt["cam_vec"], pt["obj_world_at_campk"]) >= 0.5)
    for m in ARMS:
        res["F3"][f"{m}/follow"] = dict(n=len(pans[m]), follow=mean(pans[m]))
        L.append(f"F3 follow  {m:9s} pans {len(pans[m]):3d} follow {100 * mean(pans[m]):.1f}")
    res["coverage"] = dict(f1_records={arm: sum(r["arm"] == arm for r in recs) for arm in ARMS}, cam_clips={arm: sum(x["arm"] == arm and "verdict" in x for x in cam) for arm in ARMS},
                           cam_errors={arm: sum(x["arm"] == arm and "err" in x for x in cam) for arm in ARMS}, still_frames=len(still), f1_frames=len(F1))
    L.append(f"coverage {json.dumps(res['coverage'])}")
    json.dump(res, open(out / "moprobe_h_paired.json", "w"), indent=1, default=float); (out / "moprobe_h_paired.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
