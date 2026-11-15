#!/usr/bin/env python3
"""MoPlan vs the untrained Wan2.2-A14B under the same generation protocol (arm wan_p448), per subset of MoProbe test set A,
from the same per-clip records as moplan_paired_A.py (same rules, nothing re-measured):
  F1 (MoProbe-H, per dimension relation / noun / lr / updown / nothing, + role as diagnostic):
     same  = same outcome under the real vs the opposite instruction (same_excf)             lower is better
     real  = the real instruction is carried out (ex_real)                                  higher is better
     opp   = the opposite instruction is carried out (cf_opp)                               higher is better
     dboth = both carried out minus the independence baseline p_real * p_opp               higher is better
     dir   = lr / updown only: right direction given that the object moved on the axis      higher is better
  F2 (MoProbe-P, per source ps / piq / mo / cc / rb; f2_merged_1023.json after the blind check, first frames without people):
     fab   = anything absent from the first frame appears (hand or person, or an object)    lower is better
     Needs the blind manual labels (--blind-key / --blind-labels); they are not distributed with this repository.
     Without both arguments the F2 section is skipped and F1 / F3 are still computed.
  F3 camera:
     camH  = MoProbe-H first frames whose real camera is still, real instruction            lower is better
     camP  = MoProbe-P (all fixed rigs) per source, camera_bench_records variant pd          lower is better
Paired on the shared first frames of each subset; 95% CI of A - B by resampling first frames (2,000, seed 0).
--dir holds finding_set_A.json, notext_records.jsonl, f2_merged_1023.json, camera_bench_records.jsonl,
camera_finding_records.jsonl, camera_finding_unfit.jsonl and camera_deep_records.jsonl (see docs/MOPROBE.md).
usage: python3 moplan_subsets_A.py --dir RECORDS_DIR [--blind-key KEY.json --blind-labels LABEL_DIR] --out subsets_A.json
"""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np

from moplan_paired_A import jl, mean, paired

A, B = "moplan", "wan_p448"
DIMS = ["relation", "noun", "lr", "updown", "nothing", "role"]
SETS = ["ps", "piq", "mo", "cc", "rb"]


def dboth(rs):
    if not rs:
        return float("nan")
    return mean([r["ex_real"] and r["cf_opp"] for r in rs]) - mean([r["ex_real"] for r in rs]) * mean([r["cf_opp"] for r in rs])


def dirright(rs):
    xs = [r for r in rs if r["ex_real"] or r["ex_opp"]]
    return mean([r["ex_real"] for r in xs])


def row(ga, gb, stat, better):
    p = paired(ga, gb, stat)
    d = p["diff"]
    p["better"] = better
    p["verdict"] = "nan" if d != d else ("better" if (d < 0 if better == "lower" else d > 0) else ("tie" if abs(d) < 1e-12 else "worse"))
    p["sig"] = bool(p["ci"][0] > 0 or p["ci"][1] < 0)
    p["n_a"] = sum(len(v) for v in ga.values())
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", default="")
    ap.add_argument("--blind-key", default="", help="blind check key: label id -> clip (required for F2; not distributed)")
    ap.add_argument("--blind-labels", default="", help="folder of <label id>.json with the answer (required for F2; not distributed)")
    a = ap.parse_args(); D = Path(a.dir)
    SA = json.load(open(D / "finding_set_A.json"))
    H, P = SA["H"], SA["P"]
    res = {"F1": {}, "F2": {}, "F3": {}}
    # ---- F1 per dimension
    g = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for r in jl(D / "notext_records.jsonl"):
        if r["item"] in H and r["type"] == H[r["item"]] and r["arm"] in (A, B):
            g[r["type"]][r["arm"]][r["item"]].append(r)
    for d in DIMS:
        ga, gb = g[d][A], g[d][B]
        res["F1"][d] = {
            "same": row(ga, gb, lambda rs: mean([r["same_excf"] for r in rs]), "lower"),
            "real": row(ga, gb, lambda rs: mean([r["ex_real"] for r in rs]), "higher"),
            "opp": row(ga, gb, lambda rs: mean([r["cf_opp"] for r in rs]), "higher"),
            "dboth": row(ga, gb, dboth, "higher"),
        }
        if d in ("lr", "updown"):
            res["F1"][d]["dir"] = row(ga, gb, dirright, "higher")
    # ---- F2 per source: the clips of both arms in f2_merged_1023.json (first frames without people), with the blind manual
    # labels (--blind-key / --blind-labels): object = answer new / replaced / agent on an object
    # flag; agent = a hand touching the moving object (answer touch); unlabelled flags count as no fabrication
    if bool(a.blind_key) != bool(a.blind_labels):
        raise SystemExit("--blind-key and --blind-labels go together")
    if not a.blind_key:
        print("[moplan_subsets_A] no --blind-key / --blind-labels: F2 skipped (the manual labels are not distributed)", file=sys.stderr)
    else:
        f2 = json.load(open(D / "f2_merged_1023.json"))["clips"]
        key = json.load(open(a.blind_key))
        ans = {p.stem: json.load(open(p)).get("answer") for p in Path(a.blind_labels).glob("*.json")}
        lab = collections.defaultdict(lambda: {"object": False, "agent": False})
        for k, v in key.items():
            if v["type"] == "obj" and ans.get(k) in ("new", "replaced", "agent"):
                lab[(v["arm"], v["item"])]["object"] = True
            if v["type"] == "hand" and ans.get(k) == "touch":
                lab[(v["arm"], v["item"])]["agent"] = True
        fg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        for c in f2:
            if c["arm"] in (A, B) and c["item"] in P.get(c["set"], []):
                L = lab[(c["arm"], c["item"])]
                x = {"object": L["object"], "agent": L["agent"], "any": L["object"] or L["agent"]}
                fg[c["set"]][c["arm"]][c["item"]].append(x)
                fg["ALL"][c["arm"]][c["item"]].append(x)
        for s in SETS + ["ALL"]:
            ga, gb = fg[s][A], fg[s][B]
            res["F2"][s] = {k: row(ga, gb, (lambda k: lambda cs: mean([c[k] for c in cs]))(k), "lower") for k in ("any", "agent", "object")}
    # ---- F3 camera, P per source
    cg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for x in jl(D / "camera_bench_records.jsonl"):
        if x["arm"] in (A, B) and x.get("variant") == "pd" and x["verdict"] in ("moved", "still"):
            cg[x["set"]][x["arm"]][x["item"]].append(x)
            cg["ALL"][x["arm"]][x["item"]].append(x)
    for s in SETS + ["ALL"]:
        res["F3"]["camP/" + s] = row(cg[s][A], cg[s][B], lambda xs: mean([x["verdict"] == "moved" for x in xs]), "lower")
    # ---- F3 camera, H still frames (as moplan_paired_A.py)
    real = {}
    for fn in ("camera_finding_records.jsonl", "camera_finding_unfit.jsonl"):
        for x in jl(D / fn):
            if x.get("arm") == "real":
                real[x["item"]] = x.get("verdict") or ("moved" if x.get("any") else "still")
    still = {i for i in H if real.get(i) == "still"}
    ch = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in jl(D / "camera_deep_records.jsonl"):
        it = x["sid"].rsplit("-", 1)[0]
        if x.get("cond") == "ex" and x.get("seed", 0) == 0 and it in still and x["verdict"] in ("moved", "still") and x["arm"] in (A, B):
            ch[x["arm"]][it].append(x)
    res["F3"]["camH"] = row(ch[A], ch[B], lambda xs: mean([x["verdict"] == "moved" for x in xs]), "lower")
    # ---- print
    f = lambda v: f"{100 * v:5.1f}"
    for sec, rows in res.items():
        print(f"## {sec}")
        for name, r in rows.items():
            items = r.items() if "a" not in r else [("", r)]
            for k, p in items:
                print(f"  {name:10s} {k:6s} n={p['frames']:3d}  MoPlan {f(p['a'])}  base {f(p['b'])}  diff {100 * p['diff']:+6.1f} "
                      f"[{100 * p['ci'][0]:+6.1f}, {100 * p['ci'][1]:+6.1f}]  {p['verdict']}{' *' if p['sig'] else ''}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
