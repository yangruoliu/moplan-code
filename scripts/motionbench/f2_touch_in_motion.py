#!/usr/bin/env python3
"""Confabulation, stricter agent measure.  The plain 'hand touches a moving object' flag of the agent records
(agent_eval/records.jsonl) only asks that the touched object moves >= 8 px SOMEWHERE in the clip, which also counts
real hands -- the experimenter picking the object up after it landed, or the dropping hand at the top edge -- so the
fair question is whether a hand acts on the object WHILE it is still in its first motion:
  touch_in_motion  a hand-only box within 2 px of the mask of object k at a frame f >= the agent's appearance, with
                   t[f] < t_rest(k) and t[f] <= T_CAP (both arms compared on the same window: the real continuations
                   are ~3 s), where k moves >= 8 px and t_rest(k) is the first time after k starts moving that its
                   speed stays below REST_V px/s for REST_S seconds (end of clip if it never rests)
Population: as the agent records (first frame without hand / arm / person, judged on the real clip).  Reads
<set>_eval/contact/*.contact.npz and agent_eval/records.jsonl; cluster bootstrap over first frames.
Writes $MOPROBE_ROOT/agent_eval/touch_in_motion.jsonl (one row per clip), touch_in_motion_summary.json and prints the table.
usage: MOPROBE_ROOT=DIR python3 f2_touch_in_motion.py --sets ps piq mo cc rb
"""
import argparse, collections, json, os
from pathlib import Path

import numpy as np

M = os.environ.get("MOPROBE_ROOT", "moprobe")                 # MoProbe working root (see docs/MOPROBE.md)
ARMS = ["real", "wan22_a14b", "wan22_ti2v5b", "hunyuan15_480p", "cogvideox15_5b", "ltx23", "moplan", "moplan_norl", "wan_p448", "moplan_zero", "moplan_nosft"]
T_CAP, REST_V, REST_S, MOVE_PX, GAP_PX = 3.0, 15.0, 0.3, 8.0, 2.0


def speed(t, d, ok):
    """|velocity| of one object's displacement track (px/s), NaN where not tracked"""
    x = np.where(ok[:, None], d, np.nan)
    if len(x) >= 3:                                      # 3-frame moving average: tracker jitter is not motion
        x = np.vstack([x[:1], (x[:-2] + x[1:-1] + x[2:]) / 3.0, x[-1:]])
    v = np.gradient(x, t, axis=0)
    return np.linalg.norm(v, axis=-1)


def t_rest(t, d, ok):
    s = speed(t, d, ok)
    mv = np.where(np.nan_to_num(s) >= REST_V)[0]
    if not mv.size:
        return None                                      # never moves fast enough to call a motion
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.1
    need = max(1, int(round(REST_S / dt)))
    for f in range(mv[0] + 1, len(t) - need + 1):
        w = s[f:f + need]
        if np.all(np.isfinite(w)) and np.all(w < REST_V):
            return float(t[f])
    return float(t[-1]) + 1e-6


def boot(per, n=2000, seed=7):
    keys = sorted(per); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        s = rng.choice(len(keys), len(keys)); v = [x for i in s for x in per[keys[i]]]
        out.append(np.mean(v) if v else np.nan)
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sets", nargs="+", required=True); a = ap.parse_args()
    agent = {}
    for l in open(f"{M}/agent_eval/records.jsonl"):
        r = json.loads(l); agent[(r["set"], r["arm"], r["sid"])] = r
    rows = []
    for (st, arm, sid), r in sorted(agent.items()):
        if st not in a.sets or arm not in ARMS or not r["absent_start"]:
            continue
        rec = dict(set=st, arm=arm, sid=sid, item=r["item"], appears=bool(r["appears"]), touches_mover=bool(r.get("touches_mover")),
                   appears_cap=bool(r["appears"] and r["t_appear"] <= T_CAP), touch_in_motion=False, t_touch=None, t_rest=None)
        if r["appears"] and r["t_appear"] <= T_CAP:
            p = Path(f"{M}/{st}_eval/contact/{arm}__{sid}__s0.contact.npz")
            if p.is_file():
                z = np.load(p); t = z["times"]; gh = z["gap_hand_only"] if "gap_hand_only" in z else z["gap_hand"]
                disp = z["disp"]; okm = z["ok"].astype(bool)
                f0 = int(np.searchsorted(t, r["t_appear"]))
                for k in range(gh.shape[0]):
                    pk = np.linalg.norm(disp[k][okm[k]], axis=-1).max() if okm[k].any() else 0.0
                    if pk < MOVE_PX:
                        continue
                    tr = t_rest(t, disp[k], okm[k])
                    if tr is None:
                        continue
                    hit = [f for f in range(f0, min(gh.shape[1], len(t))) if np.isfinite(gh[k, f]) and gh[k, f] <= GAP_PX and t[f] < tr and t[f] <= T_CAP]
                    if hit:
                        rec.update(touch_in_motion=True, t_touch=float(t[hit[0]]), t_rest=tr, obj=k); break
        rows.append(rec)
    with open(f"{M}/agent_eval/touch_in_motion.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = {}
    for st in a.sets:
        for arm in ARMS:
            xs = [r for r in rows if r["set"] == st and r["arm"] == arm]
            if not xs:
                continue
            e = dict(n=len(xs))
            for k in ("appears", "appears_cap", "touches_mover", "touch_in_motion"):
                per = collections.defaultdict(list)
                for r in xs:
                    per[r["item"]].append(float(r[k]))
                e[k] = float(np.mean([r[k] for r in xs])); e[k + "_ci"] = boot(per); e[k + "_n"] = int(sum(r[k] for r in xs))
            summ[f"{st}/{arm}"] = e
    json.dump(summ, open(f"{M}/agent_eval/touch_in_motion_summary.json", "w"), indent=1)
    print(f"| set | arm | clips (no agent at t0) | hand appears (any time) | hand appears <= {T_CAP:.0f} s | hand touches an object that moves | **hand touches the object while it is still in its first motion, <= {T_CAP:.0f} s** |")
    print("|---|---|---|---|---|---|---|")
    for st in a.sets:
        for arm in ARMS:
            xs = [r for r in rows if r["set"] == st and r["arm"] == arm]
            if not xs:
                continue
            cells = []
            for k in ("appears", "appears_cap", "touches_mover", "touch_in_motion"):
                per = collections.defaultdict(list)
                for r in xs:
                    per[r["item"]].append(float(r[k]))
                m = np.mean([r[k] for r in xs]); ci = boot(per)
                cells.append(f"{100 * m:.0f}% [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}] ({sum(r[k] for r in xs)})")
            print(f"| {st} | {arm} | {len(xs)} | " + " | ".join(cells[:3]) + f" | **{cells[3]}** |")


if __name__ == "__main__":
    main()
