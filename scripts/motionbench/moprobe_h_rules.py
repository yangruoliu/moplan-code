#!/usr/bin/env python3
"""MoProbe-H (finding_set_B.json) record rules: one code path from measure files to per-frame Inertia records and per-clip camera
verdicts, shared by moprobe_h_paired.py, moprobe_h_role2.py and moprobe_h_ablation.py.

F1 (Inertia) rule parameters are DERIVED from the item json + instruction texts with one function (derive):
  relation  A/B = matched[actor_cat] / matched[patient_cat]; real = apart for template 36, closer for 37
  noun      named object of the real instruction = placeholder 0000, of the opposite = 0001
  role      instrument/agent = matched[actor_cat] (real) vs matched[patient_cat] (opposite); first_mover (diagnostic)
  role2     'A hand hits P with I': instrument parsed from each text; role_outcome v2 == follows (diagnostic)
  lr        mover = the actor (else patient) object; real side from the real text ('from left to right' -> right)
  updown    mover as lr; the opposite text always ends in 'up' / 'down' -> real = the other one
  nothing   mover as lr; event_happens(real text) / nothing_holds(opposite text) (instruction_rules)
Outcome rules = motionbench/metrics/instruction_rules.py.

Camera verdict per clip (cam_clip) = camera_deep.py's (similarity fit over the whole measured clip, witness = real-static
non-hand objects, camera_rule.verdict); real-camera verdict (real_cam) = the same fit on the real continuation (4 s window,
camera_finding_analyse, else camera_patterns.camera_summary); camera patterns per clip (pattern_task) = camera_patterns.per_clip.

Paths: MOPROBE_ROOT (env) holds the built items (<items_dir>/...) and the measure files (measure/<arm>__<sid>__s0.measure.npz);
the probe manifests come from data/moprobe (env MOPROBE_DATA overrides).
Missing built items (require_items): every analysis stops when a MoProbe-H frame has no item / masks in MOPROBE_ROOT,
because a frame whose item cannot be loaded drops out of every count; --allow-missing analyses without those frames.
usage: MOPROBE_ROOT=DIR python3 moprobe_h_rules.py real-camera --out $MOPROBE_ROOT/real_cam.jsonl [--workers 8]
           [--skip-if-current] [--allow-missing]
       (real-continuation camera verdicts of the MoProbe-H frames; moprobe_h_paired.py reads them.  --skip-if-current
       keeps an existing --out unless real_cam_stale finds it out of date)
"""
import argparse, collections, functools, json, os, re, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2])); sys.path.insert(0, str(HERE.parent))
from motionbench.containers import Reading                      # noqa: E402
from motionbench.io import load_built                           # noqa: E402
from motionbench import instrument as I                         # noqa: E402
from motionbench.metrics.instruction_rules import (relation_outcome, top_mover, first_mover, single_mover, lr_outcome,  # noqa: E402
                                                   ud_outcome, event_happens, nothing_holds, role_outcome)
from motionbench.metrics.camera_rule import verdict as cam_verdict, CAM_DIR_MIN_PX  # noqa: E402

M = os.environ.get("MOPROBE_ROOT", "moprobe")                 # MoProbe working root (see docs/MOPROBE.md)
DATA = Path(os.environ.get("MOPROBE_DATA", str(HERE.parents[2] / "data" / "moprobe")))
MEASURE = Path(M) / "measure"
CONTEXT = Path(M) / "context"                                  # optional: records of the open models (context lines only)
OPEN = ["wan22_a14b", "wan22_ti2v5b", "hunyuan15_480p", "cogvideox15_5b", "ltx23"]
SCORED = ["relation", "noun", "lr", "updown", "nothing"]
DIAG = ["role", "role2"]
PAIR_CONDS = {"role2": ("rex", "rcf")}


def jl(p):
    return [json.loads(l) for l in open(p)]


@functools.lru_cache(maxsize=256)
def built(items_dir, sid):
    try:
        return load_built(items_dir, sid)
    except Exception:
        return None


def mover_index(d):                                              # actor first, else patient
    for want in ("actor", "patient"):
        for k, role in enumerate(d["roles"]):
            if role == want:
                return k
    return None


def clean(n):
    n = (n or "").strip().lower()
    for a in ("the ", "a ", "an "):
        if n.startswith(a):
            n = n[len(a):]
    return n.strip(" .")


def last_dir(text, words):
    hits = [(m.start(), m.group(1)) for m in re.finditer(r"\b(" + "|".join(words) + r")\b", (text or "").lower())]
    return hits[-1][1] if hits else None


def derive(dim, d, ex_text, cf_text):
    """rule parameters of one frame x dimension from the item json + the two instruction texts; 'issues' lists anything that
    does not parse / disagree (the frame is still scored unless a required parameter is missing)"""
    mt = d.get("matched") or {}; tid = int(d.get("template_id", -1)); iss = []
    ph = [clean(x) for x in (d.get("placeholders") or [])]
    if dim == "relation":
        P = dict(A=mt.get(d.get("actor_cat")), B=mt.get(d.get("patient_cat")))
        P["want_ex"] = "apart" if tid in (36, 40) else "closer"
        txt = "apart" if "away from each other" in (ex_text or "") else ("closer" if "closer" in (ex_text or "") else None)
        if txt != P["want_ex"]:
            iss.append(f"relation text {txt} vs template {P['want_ex']}")
        P["want_cf"] = "closer" if P["want_ex"] == "apart" else "apart"
        if P["A"] is None or P["B"] is None or P["A"] == P["B"]:
            iss.append("relation objects unmatched"); P = None
    elif dim == "noun":
        P = dict(te=mt.get("0000"), tc=mt.get("0001"), roles=list(d.get("roles") or []))
        m = re.match(r"A hand moves (?:the |a |an )?(.+?) (?:closer to|away from) ", ex_text or "")
        if not m or not ph or not (clean(m.group(1)) in ph[0] or ph[0] in clean(m.group(1))):
            iss.append(f"noun: real text object {m.group(1) if m else None!r} vs placeholder0 {ph[:1]}")
        if P["te"] is None:
            iss.append("noun target unmatched")
    elif dim == "role":
        P = dict(te=mt.get(d.get("actor_cat")), tc=mt.get(d.get("patient_cat")))
    elif dim == "role2":
        P = None
        m1 = re.match(r"A hand hits (?:the |a |an )?(.+?) with (?:the |a |an )?(.+?)\.$", ex_text or "")
        if m1 and len(ph) >= 2:
            pat, ins = clean(m1.group(1)), clean(m1.group(2))
            cat = {ph[0]: "0000", ph[1]: "0001"} if ph[0] != ph[1] else {}
            if ins in cat and pat in cat and cat[ins] != cat[pat]:
                P = dict(te=mt.get(cat[ins]), tc=mt.get(cat[pat]))
        if P is None:
            iss.append("role2 text unparsed / ambiguous -> actor_cat"); P = dict(te=mt.get(d.get("actor_cat")), tc=mt.get(d.get("patient_cat")))
        if P["te"] is None or P["tc"] is None:
            iss.append("role2 objects unmatched"); P = None
    elif dim == "lr":
        k = mover_index(d)
        s = "right" if "left to right" in (ex_text or "") else ("left" if "right to left" in (ex_text or "") else None)
        if s is None:
            c = "right" if "left to right" in (cf_text or "") else ("left" if "right to left" in (cf_text or "") else None)
            s = None if c is None else ("left" if c == "right" else "right")
        P = None if k is None or s is None else dict(k=k, want_ex=s, want_cf="left" if s == "right" else "right")
        if P is None:
            iss.append("lr mover / side missing")
    elif dim == "updown":
        k = mover_index(d); c = last_dir(cf_text, ["up", "down"])
        P = None if k is None or c is None else dict(k=k, want_ex="down" if c == "up" else "up", want_cf=c)
        if P is None:
            iss.append("updown mover / direction missing")
    elif dim == "nothing":
        k = mover_index(d)
        P = None if k is None else dict(k=k, ext=ex_text or "", cft=cf_text or "")
        if P is None:
            iss.append("nothing mover missing")
    else:
        P = None
    return P, iss


def judge(dim, P, g):
    """(real outcome?, opposite outcome?, outcome label) of one clip, or None when not measurable"""
    if dim == "relation":
        o = relation_outcome(g, P["A"], P["B"])
        return None if o is None else (o == P["want_ex"], o == P["want_cf"], o)
    if dim == "noun":
        return (single_mover(g, P["te"], P["tc"])[0], single_mover(g, P["tc"], P["te"])[0], top_mover(g, P["roles"]))
    if dim == "role":
        f = first_mover(g, [P["te"], P["tc"]]); return (f == P["te"], f == P["tc"], f)
    if dim == "role2":
        oe, oc = role_outcome(g, P["te"], P["tc"]), role_outcome(g, P["tc"], P["te"])
        return None if oe is None or oc is None else (oe == "follows", oc == "follows", f"{oe}|{oc}")
    if dim == "lr":
        o = lr_outcome(g, P["k"]); return (o == P["want_ex"], o == P["want_cf"], o)
    if dim == "updown":
        o = ud_outcome(g, P["k"]); return (o == P["want_ex"], o == P["want_cf"], o)
    if dim == "nothing":
        e, h = event_happens(g, P["k"], P["ext"]), nothing_holds(g, P["k"], P["cft"])
        return None if e is None or h is None else (bool(e), bool(h), bool(e))
    raise ValueError(dim)


def rd(p):
    if not p or not Path(p).is_file():
        return None
    try:
        return Reading.from_npz(p)
    except Exception:                                            # noqa: BLE001
        return None


def record(dim, P, item, arm, ex, cf, nx, require_nx_judgeable):
    if ex is None or cf is None:
        return None, "missing ex/cf measure"
    je, jc = judge(dim, P, ex), judge(dim, P, cf)
    jn = judge(dim, P, nx) if nx is not None else None
    if je is None or jc is None:
        return None, "ex/cf not measurable"
    if require_nx_judgeable and nx is not None and jn is None:
        return None, "nx not measurable (A convention)"
    return dict(type=dim, arm=arm, item=item, ex_real=bool(je[0]), ex_opp=bool(je[1]), cf_real=bool(jc[0]), cf_opp=bool(jc[1]),
                same_excf=bool(je[2] == jc[2]), nx_real=None if jn is None else bool(jn[0]), nx_opp=None if jn is None else bool(jn[1])), None


# ------------------------------------------------------------------ camera, per clip
CX, CY = 224.0, 128.0


def cam_clip(task):
    """task: dict(mp=measure path, items_dir, base, key...) -> camera_deep verdict + camera_patterns per_clip fields"""
    from camera_deep import fit_camera
    from camera_patterns import series, per_clip
    out = dict(task)
    b = built(task["items_dir"], task["base"])
    if b is None or b[3] is None:
        out["err"] = "no item/ref"; return out
    d, it, masks, ref = b
    try:
        r = Reading.from_npz(task["mp"])
    except Exception as e:                                       # noqa: BLE001
        out["err"] = f"read {e}"; return out
    if r.bg_disp is None:
        out["err"] = "no bg"; return out
    p0, kind = I.background_points(masks, *masks.shape[1:3])
    if len(p0) != r.bg_disp.shape[0]:
        out["err"] = "grid mismatch"; return out
    fit, Rs, ts = fit_camera(p0, r.bg_disp, r.bg_ok, geom=d.get("geometry"))
    good = np.isfinite(fit[:, 0])
    zoom = float(np.nanmax(np.abs(np.log(fit[good, 0])))) if good.any() else np.nan
    rot = float(np.nanmax(np.abs(fit[good, 1]))) if good.any() else np.nan
    tr = float(np.nanmax(fit[good, 2])) if good.any() else np.nan
    roles = d.get("roles") or []
    statics = [k for k in range(min(r.n_objects, ref.reading.n_objects)) if ref.static[k] and (k >= len(roles) or roles[k] != "hand")]
    raw_pk = [float(np.linalg.norm(r.disp[k][r.ok[k]], axis=-1).max()) for k in statics if r.ok[k].any()]
    shift = float(np.median(raw_pk)) if len(raw_pk) >= 2 else np.nan
    lost = float(np.mean(~good[1:])) if len(good) > 1 else 1.0
    out.update(trans=tr, zoom=zoom, rot=rot, shift=shift, lost=lost, verdict=cam_verdict(tr, zoom, rot, lost, None if not np.isfinite(shift) else shift))
    S = series(r.times, r.disp, r.ok, r.pos0, r.bg_disp, r.bg_ok, masks, roles, geom=d.get("geometry"))
    if S is not None:
        pc = per_clip(S)
        out["pat"] = dict(trans=pc["trans"], cam_vec=pc["cam_vec"], obj_world_at_campk=pc.get("obj_world_at_campk"))
    return out


def real_cam(task):
    """real-continuation camera verdict (4 s window; analyse, else camera_summary)"""
    from camera_patterns import camera_summary
    analyse = camera_finding_analyse
    out = dict(task)
    b = built(task["items_dir"], task["base"])
    p = Path(task["items_dir"]) / "ref" / f"{task['base']}.npz"
    if b is None or not p.is_file():
        out["verdict"] = "missing"; return out
    d, it, masks, ref = b; z = np.load(p)
    if "bg_disp" not in z:
        out["verdict"] = "no_bg"; return out
    w = z["times"] <= 4.0 + 1e-6
    rec = analyse(z["times"][w], z["disp"][:, w], z["ok"].astype(bool)[:, w], z["area"][:, w], z["pos0"], z["bg_disp"][:, w],
                  z["bg_ok"].astype(bool)[:, w], masks, d.get("roles") or [], geom=d.get("geometry"))
    if rec is None:
        rec = camera_summary(z["times"][w], z["disp"][:, w], z["ok"].astype(bool)[:, w], z["bg_disp"][:, w], z["bg_ok"].astype(bool)[:, w],
                             masks, d.get("roles") or [], geom=d.get("geometry")) or dict(verdict="unscorable")
    out.update(verdict=rec["verdict"], trans=rec.get("trans"))
    return out


def b_rows():
    """MoProbe-H rows (frame x condition: instruction, items dir) minus the frames rejected by the item check
    (data/moprobe/rejected_frames.json); items_dir is resolved against MOPROBE_ROOT"""
    rej = set(json.load(open(DATA / "rejected_frames.json"))["rejected"])
    out = []
    for r in jl(DATA / "moprobe_h_rows.jsonl"):
        if r["part"] == "H" and r["frame"] not in rej:
            out.append(dict(r, items_dir=os.path.join(M, r["items_dir"])))
    return out


def missing_items(rows):
    """(items_dir, frame) of the rows whose built item (items/<frame>.json + masks/<frame>.npz) is not in MOPROBE_ROOT"""
    out = set()
    for r in rows:
        d = Path(r["items_dir"])
        if not ((d / "items" / f"{r['frame']}.json").is_file() and (d / "masks" / f"{r['frame']}.npz").is_file()):
            out.add((r["items_dir"], r["frame"]))
    return sorted(out)


def require_items(rows, allow_missing=False):
    """Stop when built items of the MoProbe-H rows are missing: the analyses skip a frame whose item cannot be loaded,
    so an incomplete bundle would silently lower every count.  With allow_missing, warn and go on without them."""
    miss = missing_items(rows)
    if miss:
        folders = dict(sorted(collections.Counter(os.path.basename(d) for d, _ in miss).items()))
        msg = (f"{len(miss)} MoProbe-H frames have no built item under {M} (per item folder: {folders}; e.g. "
               f"{miss[0][0]}/items/{miss[0][1]}.json)")
        if not allow_missing:
            raise SystemExit(f"{msg}.  Complete the bundle, or pass --allow-missing to analyse without these frames.")
        print(f"WARNING: {msg}; analysed without these frames (--allow-missing)", file=sys.stderr)
    return miss


def real_cam_stale(path, tasks):
    """Why the stored real-camera verdicts at path cannot be reused for tasks (None: they can).  Stale when the file is
    absent or unreadable, covers other frames or item folders, is older than an item / masks / ref file it was computed
    from, or holds a 'missing' verdict whose item and ref exist now."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return "no stored verdicts"
    try:
        stored = {r["base"]: r for r in jl(p) if r.get("ver") == "B"}
    except (ValueError, KeyError):
        return "unreadable"
    if set(stored) != {t["base"] for t in tasks}:
        return "other frames"
    written = p.stat().st_mtime
    for t in tasks:
        r, d = stored[t["base"]], Path(t["items_dir"])
        if r.get("items_dir") != t["items_dir"]:
            return f"{t['base']}: other item folder"
        inputs = [d / "items" / f"{t['base']}.json", d / "masks" / f"{t['base']}.npz", d / "ref" / f"{t['base']}.npz"]
        if any(f.is_file() and f.stat().st_mtime > written for f in inputs):
            return f"{t['base']}: inputs newer than {p.name}"
        if r.get("verdict") == "missing" and all(f.is_file() for f in inputs):
            return f"{t['base']}: stored as missing, inputs present now"
    return None


def mp(arm, sid):
    """measure file of one clip (seed 0), or None"""
    p = MEASURE / f"{arm}__{sid}__s0.measure.npz"
    return str(p) if p.is_file() else None


_CACHE = {}


def _built_plain(idir, base):
    if (idir, base) not in _CACHE:
        try:
            _CACHE[(idir, base)] = load_built(idir, base)
        except Exception:
            _CACHE[(idir, base)] = None
    return _CACHE[(idir, base)]


def pattern_task(t):
    """t: dict(arm, base, cond, sid, items_dir, mp) -> (patterns record, edge record or None)"""
    from camera_patterns import series, per_clip, cos, headroom, T_DIR
    b = _built_plain(t["items_dir"], t["base"])
    if b is None:
        return None, None
    d, it, masks, ref = b
    r = Reading.from_npz(t["mp"])
    if r.bg_disp is None:
        return None, None
    S = series(r.times, r.disp, r.ok, r.pos0, r.bg_disp, r.bg_ok, masks, d.get("roles") or [], geom=d.get("geometry"))
    if S is None:
        return None, None
    c = per_clip(S)
    cond = "nx" if t["cond"] == "nt" else t["cond"]
    pat = dict(c, set="bench3", arm=t["arm"], sid=t["sid"], seed=0, item=t["base"], cond=cond, cond_raw=t["cond"])
    edge = None
    if S["mover"] is not None and "obj_world_pk" in c:
        u = np.asarray(c["obj_world_pk"], float); mag = float(np.linalg.norm(u))
        if mag >= 16:
            u = u / mag; hr = headroom(np.asarray(c["obj_pos0"], float), u)
            pan = c["trans"] >= T_DIR and "obj_world_at_campk" in c
            cs = cos(c["cam_vec"], c["obj_world_at_campk"]) if pan else np.nan
            edge = dict(set="bench3", arm=t["arm"], item=t["base"], sid=t["sid"], cond=cond, headroom=float(max(hr, 1.0)), mover_px=mag,
                        towards=bool(pan and np.isfinite(cs) and cs >= 0.5), away=bool(pan and np.isfinite(cs) and cs <= -0.5),
                        pan=bool(pan), cos=float(cs) if np.isfinite(cs) else None)
    return pat, edge


def pattern_task_safe(t):
    try:
        return pattern_task(t)
    except Exception as e:                                       # noqa: BLE001
        return dict(err=repr(e)[:200], **{k: t[k] for k in ("arm", "sid")}), None


def camera_finding_analyse(times, disp, ok, area, pos0, bg_disp, bg_ok, masks, roles, geom=None):
    """camera verdict of a real continuation from the similarity fit of its background grid (None: the fit failed on most frames)"""
    from camera_deep import fit_camera
    from motionbench.metrics.camera_rule import CAM_TRANS_PX as T_TR, CAM_ZOOM as T_ZOOM, verdict, witness_from
    p0, kind = I.background_points(masks, *masks.shape[1:3])
    if len(p0) != bg_disp.shape[0]:
        return None
    fit, Rs, ts = fit_camera(p0, bg_disp, bg_ok, geom=geom)
    good = np.isfinite(fit[:, 0])
    if good[1:].mean() < 0.5:
        return None                                    # failed fits: scored separately by camera_summary (main loop)
    tr = np.where(good, fit[:, 2], 0.0); zm = np.where(good, np.abs(np.log(np.where(good, fit[:, 0], 1.0))), 0.0); ro = np.where(good, np.abs(fit[:, 1]), 0.0)
    lost = float(np.mean(~good[1:])); wit = witness_from(disp, ok, [k for k in range(min(len(roles), disp.shape[0])) if roles[k] == "static"])
    rec = dict(trans=float(tr.max()), zoom=float(zm.max()), rot=float(ro.max()), lost=lost, witness=wit, fitted=True)
    rec["verdict"] = verdict(rec["trans"], rec["zoom"], rec["rot"], lost, wit)
    rec["any"] = rec["verdict"] == "moved"
    rec["type"] = ("zoom" if rec["zoom"] >= T_ZOOM and rec["trans"] < T_TR else "pan" if rec["trans"] >= T_TR and rec["zoom"] < T_ZOOM else "mixed") if rec["any"] else "none"
    cam_on = np.nonzero((tr >= 8) | (zm >= 0.05))[0]
    rec["cam_onset"] = float(times[cam_on[0]]) if cam_on.size else None
    # main mover: largest camera-free peak over non-hand objects
    K = disp.shape[0]; best = (0.0, None, None)
    for k in range(K):
        if k < len(roles) and roles[k] == "hand":
            continue
        okk = ok[k]
        if okk.sum() < 2:
            continue
        world = np.zeros((len(times), 2))
        for f in range(len(times)):
            if okk[f] and good[f]:
                pred = fit[f, 0] * (Rs[f] @ pos0[k]) + ts[f]
                world[f] = pos0[k] + disp[k, f] - pred
        mag = np.linalg.norm(world, axis=-1) * (okk & good)
        if mag.max() > best[0]:
            best = (float(mag.max()), k, mag)
    rec["obj_peak"] = best[0]
    if best[1] is not None:
        mag = best[2]; on = [i for i in range(len(mag)) if mag[i] >= 4 and (i + 1 >= len(mag) or mag[i + 1] >= 4)]
        rec["obj_onset"] = float(times[on[0]]) if on else None
    else:
        rec["obj_onset"] = None
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rc = sub.add_parser("real-camera")
    rc.add_argument("--out", required=True); rc.add_argument("--workers", type=int, default=8)
    rc.add_argument("--allow-missing", action="store_true", help="go on when built items are missing (their frames get the "
                    "verdict 'missing'); default: stop")
    rc.add_argument("--skip-if-current", action="store_true", help="keep --out when it is current (see real_cam_stale); "
                    "otherwise recompute it")
    a = ap.parse_args()
    rows = b_rows()
    require_items(rows, a.allow_missing)
    byf = collections.defaultdict(dict)
    for r in rows:
        byf[r["frame"]][r["cond"]] = r
    tasks = [dict(ver="B", base=s, items_dir=byf[s][next(iter(byf[s]))]["items_dir"]) for s in byf if byf[s]]
    if a.skip_if_current:
        why = real_cam_stale(a.out, tasks)
        if why is None:
            print(f"real verdicts: {a.out} is current, kept"); return
        print(f"real verdicts: recomputing {a.out} ({why})")
    with Pool(a.workers) as pool:
        reals = pool.map(real_cam, tasks, chunksize=4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for x in reals:
            f.write(json.dumps(x, default=float) + "\n")
    print("real verdicts", dict(collections.Counter(x["verdict"] for x in reals)))


if __name__ == "__main__":
    main()
