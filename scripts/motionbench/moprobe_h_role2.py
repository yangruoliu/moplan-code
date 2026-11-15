#!/usr/bin/env python3
"""Coverage check of the MoProbe-H Inertia rows after moprobe_h_paired.py.
1. finding_set_B.json f1_diag_frames is a dict sid -> dim, so a frame that is both a role and a role2 diagnostic item is
   listed only as role, and its role2 (new role, rex / rcf) item is not scored by moprobe_h_paired.py.  Score it for both
   arms and recompute the per-dimension rows.
2. For every (frame, dim) without records on both arms, print why (record() reason).
CPU only, reads the same measure files as moprobe_h_paired.py.
Stops when built items are missing (moprobe_h_rules.require_items) unless --allow-missing.
usage: MOPROBE_ROOT=DIR python3 moprobe_h_role2.py --records PAIRED_OUT_DIR --out OUT_DIR [--allow-missing]
       -> OUT_DIR/{f1_records_moplan_B.jsonl, moprobe_h_role2.json, moprobe_h_role2.md}"""
import argparse, collections, json, sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2]))
import moprobe_h_rules as V                                    # noqa: E402
from moplan_paired_A import paired, dirright, mean              # noqa: E402
from moprobe_h_paired import ARMS, mp                          # noqa: E402

M = V.M


def why(sid, dim, byf):
    c = byf.get(sid, {})
    ce, cc = V.PAIR_CONDS.get(dim, ("ex", "cf"))
    if ce not in c or cc not in c:
        return {a: f"no {ce}/{cc} rows" for a in ARMS}, []
    b = V.built(c[ce]["items_dir"], sid)
    if b is None:
        return {a: "item not built" for a in ARMS}, []
    P, _ = V.derive(dim, b[0], c[ce]["instruction"], c[cc]["instruction"])
    if P is None:
        return {a: "derive failed" for a in ARMS}, []
    out, recs = {}, []
    for arm in ARMS:
        r, reason = V.record(dim, P, sid, arm, V.rd(mp(arm, f"{sid}-{ce}")), V.rd(mp(arm, f"{sid}-{cc}")), None, False)
        out[arm] = "ok" if r else reason
        if r:
            recs.append(r)
    return out, recs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--records", required=True, help="output folder of moprobe_h_paired.py (f1_records_moplan_B.jsonl)")
    ap.add_argument("--allow-missing", action="store_true", help="go on when built items are missing (default: stop)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fs = json.load(open(V.DATA / "finding_set_B.json"))
    F1, diag, role2 = fs["f1_frames"], fs["f1_diag_frames"], fs["role2"]
    rows = V.b_rows(); V.require_items(rows, a.allow_missing); byf = collections.defaultdict(dict)
    for r in rows:
        byf[r["frame"]][r["cond"]] = r
    recs = [json.loads(l) for l in open(Path(a.records) / "f1_records_moplan_B.jsonl") if l.strip()]
    pairs = set(F1.items()) | set(diag.items()) | {(s, "role2") for s in role2}
    have = collections.defaultdict(set)
    for r in recs:
        have[(r["item"], r["type"])].add(r["arm"])
    L = [f"(frame, dim) pairs: {len(pairs)} (f1 {len(F1)}, diag dict {len(diag)}, role2 list {len(role2)})"]
    new = []
    for sid, dim in sorted(pairs):
        if have.get((sid, dim), set()) == set(ARMS):
            continue
        reasons, rs = why(sid, dim, byf)
        L.append(f"  {sid} {dim}: in records {sorted(have.get((sid, dim), set()))}; now {reasons}")
        new += [r for r in rs if r["arm"] not in have.get((sid, dim), set())]
    allr = recs + new
    with open(out / "f1_records_moplan_B.jsonl", "w") as f:
        for r in allr:
            f.write(json.dumps(r) + "\n")
    stats = dict(same=lambda rs: mean([r["same_excf"] for r in rs]), real=lambda rs: mean([r["ex_real"] for r in rs]),
                 opp=lambda rs: mean([r["cf_opp"] for r in rs]))
    res = {"coverage": {}, "F1_dim": {}}
    for dim in V.SCORED + ["role", "role2"]:
        ga, gb = collections.defaultdict(list), collections.defaultdict(list)
        for r in allr:
            if r["type"] == dim:
                (ga if r["arm"] == "moplan" else gb)[r["item"]].append(r)
        n_set = sum(1 for p in pairs if p[1] == dim)
        res["coverage"][dim] = dict(frames_in_set=n_set, both=len(set(ga) & set(gb)))
        L.append(f"{dim:8s} frames in set {n_set:3d}  paired {len(set(ga) & set(gb)):3d}")
        for name, st in list(stats.items()) + ([("dir", dirright)] if dim in ("lr", "updown") else []):
            p = paired(ga, gb, st); res["F1_dim"][f"{dim}/{name}"] = p
            L.append(f"  {name:5s} frames {p['frames']:3d}  {100 * p['a']:.1f} vs {100 * p['b']:.1f}  diff {100 * p['diff']:+.1f} "
                     f"[{100 * p['ci'][0]:+.1f}, {100 * p['ci'][1]:+.1f}]")
    json.dump(res, open(out / "moprobe_h_role2.json", "w"), indent=1, default=float)
    (out / "moprobe_h_role2.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
