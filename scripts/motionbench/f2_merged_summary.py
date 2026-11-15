#!/usr/bin/env python3
"""Confabulation in one table: something that is not in the first frame appears out of nowhere, whether a person, a hand
or an object (the hand / person / object split is reported separately).  Population: clips whose first frame has no
hand / arm / person (agent_eval records, judged on the real clip) AND that the new-object scan has processed.  Per clip:
  agent   a hand / person appears and touches a first-frame object while that object is still in its first motion
          (f2_touch_in_motion.py: before it first rests, within 3 s); the real videos' experimenter picking the object up
          after it landed does not count
  object  the class-agnostic new-object scan flagged the clip and the flag was confirmed as invented (new object,
          replaced object, an object that comes in and hits, or a hand / person the scan caught): automatic audit labels
          (--auto-labels: {clip name: {"label": ...}}), overridden by the blind human check where it covered the flag
  any     agent OR object
The blind human check also overrides the agent flag where it covered it ('touch' keeps it, 'appear only' / 'none' drops
it).  Unaudited object flags count as not invented, so 'object' is a lower bound there.
Separate split of 'agent': person if the detector's best box is >= 50 % of the frame height, else hand / arm (approximate).
usage: MOPROBE_ROOT=DIR python3 f2_merged_summary.py --auto-labels newobj_labels.json --blind-key key.json --blind-labels <dir> \
  [--blind2-key key2.json --blind2-labels <dir2>] [--blind3-key key3.json --blind3-labels <dir3>] [--blind4-key key4.json --blind4-labels <dir4>] \
  --out $MOPROBE_ROOT/f2_merged.json
"""
import argparse, collections, glob, json, os
from pathlib import Path

import numpy as np

M = os.environ.get("MOPROBE_ROOT", "moprobe")                 # MoProbe working root (see docs/MOPROBE.md)
ARMS = ["real", "wan22_a14b", "wan22_ti2v5b", "hunyuan15_480p", "cogvideox15_5b", "ltx23", "moplan", "moplan_norl", "wan_p448", "moplan_zero", "moplan_nosft"]
SETS = ["ps", "piq", "mo", "cc", "rb"]
TRUE = {"agent", "replaced", "new_object", "agent_object"}
H_YES = {"new": "new_object", "replaced": "replaced", "agent": "agent"}
WORK_H = 512.0


def boot(per, n=2000, seed=7):
    keys = sorted(per); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        s = rng.choice(len(keys), len(keys)); v = [x for i in s for x in per[keys[i]]]
        out.append(np.mean(v) if v else np.nan)
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--auto-labels", required=True); ap.add_argument("--blind-key", required=True)
    ap.add_argument("--blind-labels", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--blind2-key", default=""); ap.add_argument("--blind2-labels", default="")   # additional label set: object flags only
    ap.add_argument("--blind3-key", default=""); ap.add_argument("--blind3-labels", default="")   # additional label set: object and touch flags
    ap.add_argument("--blind4-key", default=""); ap.add_argument("--blind4-labels", default="")   # additional label set: object and touch flags
    ap.add_argument("--blind-extra", nargs="*", default=[], help="further annotation rounds: KEY.json:LABELDIR pairs")
    a = ap.parse_args()
    agent = {}
    for fn in ("records.jsonl", "records_cc.jsonl"):
        p = Path(f"{M}/agent_eval/{fn}")
        if p.is_file():
            for l in open(p):
                r = json.loads(l); agent.setdefault((r["set"], r["arm"], r["sid"]), r)
    tim = {(r["set"], r["arm"], r["sid"]): r for r in (json.loads(l) for l in open(f"{M}/agent_eval/touch_in_motion.jsonl"))}
    auto = json.load(open(a.auto_labels))
    key = json.load(open(a.blind_key))
    blind = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(a.blind_labels, "*.json"))}
    h_obj, h_hand = {}, {}
    for bid, k in key.items():
        ans = (blind.get(bid) or {}).get("answer")
        if not ans or ans == "unclear":
            continue
        (h_obj if k["type"] == "obj" else h_hand)[k["name"]] = ans
    if a.blind2_key:
        key2 = json.load(open(a.blind2_key))
        blind2 = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(a.blind2_labels, "*.json"))}
        for bid, k in key2.items():
            ans = (blind2.get(bid) or {}).get("answer")
            if ans and ans != "unclear":
                h_obj[k["name"]] = ans
    if a.blind3_key:
        key3 = json.load(open(a.blind3_key))
        blind3 = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(a.blind3_labels, "*.json"))}
        for bid, k in key3.items():
            ans = (blind3.get(bid) or {}).get("answer")
            if ans and ans != "unclear":
                (h_obj if k["type"] == "obj" else h_hand)[k["name"]] = ans
    for pair in a.blind_extra:                                    # same answer vocabulary as round 3
        kp, ld = pair.split(":", 1)
        keyx = json.load(open(kp))
        labx = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(ld, "*.json"))}
        for bid, k in keyx.items():
            ans = (labx.get(bid) or {}).get("answer")
            if ans and ans != "unclear":
                (h_obj if k["type"] == "obj" else h_hand)[k["name"]] = ans
    if a.blind4_key:
        key4 = json.load(open(a.blind4_key))
        blind4 = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(a.blind4_labels, "*.json"))}
        for bid, k in key4.items():
            ans = (blind4.get(bid) or {}).get("answer")
            if ans and ans != "unclear":
                (h_obj if k["type"] == "obj" else h_hand)[k["name"]] = ans
    clips = []
    for st in SETS:
        flags = {}
        for p in Path(f"{M}/newobj_eval/{st}").glob("*.newobj.json"):
            r = json.load(open(p)); flags[(r["arm"], r["sid"])] = (bool(r["candidates"]), p.name.replace(".newobj.json", ""))
        for (s, arm, sid), ag in agent.items():
            if s != st or arm not in ARMS or not ag["absent_start"] or (arm, sid) not in flags:
                continue                                     # only clips the object scan has seen (else 'object' is silently 0)
            name = f"{arm}__{sid}__s0"
            t = tim.get((st, arm, sid)) or {}
            ag_flag = bool(t.get("touch_in_motion")); ag_src = "auto"
            if name in h_hand:
                ag_flag = h_hand[name] == "touch"; ag_src = "human"
            flagged, nm = flags.get((arm, sid), (False, name))
            lab = (auto.get(nm) or {}).get("label") if flagged else None
            obj_flag = lab in TRUE; obj_kind = lab if obj_flag else None; obj_src = "auto" if lab is not None else ("unaudited" if flagged else "none")
            if nm in h_obj:
                obj_flag = h_obj[nm] in H_YES; obj_kind = H_YES.get(h_obj[nm]); obj_src = "human"
            best = ag.get("best")
            person = bool(ag_flag and best and (best[1][3] - best[1][1]) >= 0.5 * WORK_H)
            clips.append(dict(set=st, arm=arm, sid=sid, item=ag.get("item") or sid.split("-")[0], agent=ag_flag, agent_src=ag_src,
                              person=person, hand=ag_flag and not person, object=obj_flag, obj_kind=obj_kind, obj_src=obj_src,
                              obj_flagged=flagged, any=ag_flag or obj_flag))
    res = {}
    print("| set | arm | clips | any invented [95% CI] | agent (hand / person) | object | object flags: human / automatic / unaudited |")
    print("|---|---|---|---|---|---|---|")
    for st in SETS:
        for arm in ARMS:
            xs = [c for c in clips if c["set"] == st and c["arm"] == arm]
            if not xs:
                continue
            e = dict(n=len(xs))
            for k in ("any", "agent", "hand", "person", "object"):
                per = collections.defaultdict(list)
                for c in xs:
                    per[c["item"]].append(float(c[k]))
                e[k] = float(np.mean([c[k] for c in xs])); e[k + "_ci"] = boot(per); e[k + "_n"] = int(sum(c[k] for c in xs))
            fl = [c for c in xs if c["obj_flagged"]]
            e["obj_flags"] = len(fl); e["obj_human"] = sum(c["obj_src"] == "human" for c in fl); e["obj_auto"] = sum(c["obj_src"] == "auto" for c in fl)
            e["obj_unaudited"] = sum(c["obj_src"] == "unaudited" for c in fl); e["agent_human"] = sum(c["agent_src"] == "human" for c in xs)
            e["obj_kinds"] = dict(collections.Counter(c["obj_kind"] for c in xs if c["object"]))
            res[f"{st}/{arm}"] = e
            print(f"| {st} | {arm} | {e['n']} | **{100 * e['any']:.0f}%** [{100 * e['any_ci'][0]:.0f}, {100 * e['any_ci'][1]:.0f}] | "
                  f"{100 * e['agent']:.0f}% ({e['hand_n']} / {e['person_n']}) | {100 * e['object']:.0f}% | {e['obj_human']} / {e['obj_auto']} / {e['obj_unaudited']} |")
    json.dump(dict(res=res, clips=clips), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
