#!/usr/bin/env python3
"""Camera vs object motion: what are the regularities?
Builds on camera_deep.fit_camera (per-frame similarity fit of the background grid).  For every generated clip of the
finding sets (cfset / dirbal / fx, 5 open models) and every real continuation:
  1. direction   a pan's direction vs the main mover's world displacement (cos; 'follows' >= 0.5) and vs the hand's;
                 null = the same pans paired with movers of other clips (permutation)
  2. zoom        zoom-in vs zoom-out; zoom centre (fixed point of the similarity) vs the mover / hand / image centre
  3. framing     pan towards the mover's heading vs how close the mover starts to the frame border along its heading
  4. timing      camera onset / stop vs mover onset / stop; camera speed while the mover moves vs while it is still
  5. propensity  how much of the camera variance belongs to the first frame (ICC-like), cross-model agreement, and
                 first-frame features (hand size, mover size / border distance, blur, real camera shake, real motion size)
  6. signatures  per model: pan / zoom / rotation shares, zoom-in share, camera onset time
  7. compensation  apparent-only motion (image >= 8 px, world < 8 px) and cancelled motion (world >= 8, image < 8)
  8. real cue    generated pan direction vs the real clip's own small camera drift (does the model continue it?)
usage: MOPROBE_ROOT=DIR python scripts/motionbench/camera_patterns.py --out $MOPROBE_ROOT/camera_eval/patterns
"""
import argparse, collections, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench import instrument as I                         # noqa: E402
from motionbench.containers import Reading                      # noqa: E402
from motionbench.io import load_built                           # noqa: E402
from camera_deep import fit_camera                              # noqa: E402

M = os.environ.get("MOPROBE_ROOT", "moprobe")                 # MoProbe working root (see docs/MOPROBE.md)
OPEN = ["wan22_a14b", "wan22_ti2v5b", "hunyuan15_480p", "cogvideox15_5b", "ltx23"]
OURS = ["moplan", "moplan_r900", "moplan_norl", "wan_p448", "moplan_zero", "moplan_nosft"]    # MoPlan + ablation arms + base under the MoPlan protocol, never pooled into OPEN5
# moplan_r900 (a longer RAFT round, not released) stays in the list: section 1 draws the permutations of every arm from one
# shared rng, so dropping an arm with records would shift the draws of all later arms; an arm without records is skipped
MODELS = OPEN + OURS
NICE = {"wan22_a14b": "Wan2.2-I2V-A14B", "wan22_ti2v5b": "Wan2.2-TI2V-5B", "hunyuan15_480p": "HunyuanVideo-1.5",
        "cogvideox15_5b": "CogVideoX1.5-5B", "ltx23": "LTX-2.3", "moplan": "MoPlan (final)", "moplan_r900": "MoPlan RAFT-900", "moplan_norl": "MoPlan no RL", "wan_p448": "Wan-A14B (our protocol)", "moplan_zero": "MoPlan zero plan", "moplan_nosft": "MoPlan RAFT w/o SFT", "OPEN5": "5 models pooled", "real": "real videos"}
ITEMS = {"cfset": "items_cfset", "dirbal": "items_dirbal", "fx": "items_fx"}
W, H = 448.0, 256.0
CEN = np.array([W / 2, H / 2])
from motionbench.metrics.camera_rule import CAM_TRANS_PX as T_TR, CAM_ZOOM as T_ZOOM, CAM_ROT_DEG as T_ROT, CAM_DIR_MIN_PX as T_DIR, verdict, witness_from   # noqa: E402  one camera rule (8 px)


def smooth(x, k=3):
    if len(x) < k:
        return x
    pad = np.pad(x, ((k // 2, k // 2),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    return np.stack([pad[i:i + len(x)] for i in range(k)]).mean(0)


def _fit(times, disp, ok, bg_disp, bg_ok, masks, roles, tmax, geom):
    w = times <= tmax + 1e-6
    times, disp, ok, bg_disp, bg_ok = times[w], disp[:, w], ok[:, w], bg_disp[:, w], bg_ok[:, w]
    p0, _ = I.background_points(masks, *masks.shape[1:3])
    if len(p0) != bg_disp.shape[0]:
        return None
    fit, Rs, ts = fit_camera(p0, bg_disp, bg_ok, geom=geom)
    good = np.isfinite(fit[:, 0])
    lost = float(np.mean(~good[1:])) if len(good) > 1 else 1.0
    statics = [k for k in range(min(len(roles), disp.shape[0])) if roles[k] == "static"]
    return times, disp, ok, fit, Rs, ts, good, lost, witness_from(disp, ok, statics)


def camera_summary(times, disp, ok, bg_disp, bg_ok, masks, roles, tmax=4.0, geom=None):
    """Camera verdict for ANY clip, also when the grid fit fails on most frames (camera_rule.verdict); None only
    when the grid does not match the masks"""
    f = _fit(times, disp, ok, bg_disp, bg_ok, masks, roles, tmax, geom)
    if f is None:
        return None
    times, disp, ok, fit, Rs, ts, good, lost, wit = f
    g = good.copy(); g[0] = False
    tr = float(np.nanmax(fit[g, 2])) if g.any() else float("nan")
    zm = float(np.nanmax(np.abs(np.log(fit[g, 0])))) if g.any() else float("nan")
    ro = float(np.nanmax(np.abs(fit[g, 1]))) if g.any() else float("nan")
    return dict(trans=tr, zoom=zm, rot=ro, lost=lost, witness=wit, verdict=verdict(tr, zm, ro, lost, wit))


def series(times, disp, ok, pos0, bg_disp, bg_ok, masks, roles, tmax=4.0, geom=None):
    """per-frame camera + world motion of every object; None if the background grid cannot be fitted (the pattern
    analyses need the per-frame fit; the camera RATE uses camera_summary, which also scores failed fits)"""
    f = _fit(times, disp, ok, bg_disp, bg_ok, masks, roles, tmax, geom)
    if f is None:
        return None
    times, disp, ok, fit, Rs, ts, good, lost, wit = f
    if good[1:].mean() < 0.5:
        return None
    F, K = len(times), disp.shape[0]
    tc = np.zeros((F, 2)); logs = np.zeros(F); rot = np.zeros(F); zc = np.full((F, 2), np.nan)
    for f in range(F):
        if not good[f]:
            continue
        A = fit[f, 0] * Rs[f]
        tc[f] = A @ CEN + ts[f] - CEN; logs[f] = np.log(fit[f, 0]); rot[f] = fit[f, 1]
        if abs(logs[f]) >= 0.05:
            try:
                x = np.linalg.solve(np.eye(2) - A, ts[f])
                if np.all(np.abs(x - CEN) < 2 * np.array([W, H])):
                    zc[f] = x
            except np.linalg.LinAlgError:
                pass
    world = np.zeros((K, F, 2)); wok = np.zeros((K, F), bool)
    for k in range(K):
        for f in range(F):
            if ok[k, f] and good[f]:
                pred = fit[f, 0] * (Rs[f] @ pos0[k]) + ts[f]
                world[k, f] = pos0[k] + disp[k, f] - pred; wok[k, f] = True
    hands = [k for k in range(K) if k < len(roles) and roles[k] == "hand"]
    best, kbest = 0.0, None
    for k in range(K):
        if k in hands or wok[k].sum() < 2:
            continue
        pk = float(np.linalg.norm(world[k][wok[k]], axis=-1).max())
        if pk > best:
            best, kbest = pk, k
    trm = np.where(good, np.linalg.norm(tc, axis=-1), 0.0); zm = np.where(good, np.abs(logs), 0.0); rm = np.where(good, np.abs(rot), 0.0)
    rec = dict(trans=float(trm.max()), zoom=float(zm.max()), rot=float(rm.max()), obj_peak=best, lost=lost, witness=wit)
    rec["verdict"] = verdict(rec["trans"], rec["zoom"], rec["rot"], lost, wit)
    rec["any"] = rec["verdict"] == "moved"
    rec["type"] = ("zoom" if rec["zoom"] >= T_ZOOM and rec["trans"] < T_TR else "pan" if rec["trans"] >= T_TR and rec["zoom"] < T_ZOOM else "mixed") if rec["any"] else "none"
    return dict(rec=rec, times=times, good=good, tc=tc, logs=logs, rot=rot, zc=zc, world=world, wok=wok, disp=disp, ok=ok, pos0=pos0,
                mover=kbest, hands=hands)


def onset(mag, times, thr):
    on = [i for i in range(len(mag)) if mag[i] >= thr and (i + 1 >= len(mag) or mag[i + 1] >= thr)]
    return float(times[on[0]]) if on else None


def per_clip(S):
    """scalar summaries for the pattern analyses"""
    rec = dict(S["rec"]); t = S["times"]; good = S["good"]; k = S["mover"]
    trm = np.where(good, np.linalg.norm(S["tc"], axis=-1), 0.0)
    fpk = int(np.argmax(trm))
    cam_vec = -S["tc"][fpk]                                     # camera motion direction in image coords (content moves the other way)
    rec["cam_vec"] = cam_vec.tolist()
    zpk = int(np.argmax(np.where(good, np.abs(S["logs"]), 0.0)))
    rec["zoom_in"] = bool(S["logs"][zpk] > 0)
    rec["zoom_centre"] = None if not np.all(np.isfinite(S["zc"][zpk])) else S["zc"][zpk].tolist()
    cam_mag = np.where(good, np.linalg.norm(S["tc"], axis=-1) + 100 * np.abs(S["logs"]), 0.0)   # 10 % zoom ~ 10 px
    rec["cam_onset"] = onset(cam_mag, t, 8.0)
    if k is not None:
        wk = np.where(S["wok"][k][:, None], S["world"][k], np.nan)
        mag = np.nan_to_num(np.linalg.norm(wk, axis=-1))
        rec["obj_onset"] = onset(mag, t, 4.0)
        fpk_valid = fpk if S["wok"][k][fpk] else (int(np.nonzero(S["wok"][k])[0][-1]) if S["wok"][k].any() else None)
        if fpk_valid is not None:
            rec["obj_world_at_campk"] = S["world"][k][fpk_valid].tolist()
            rec["obj_img_at_campk"] = S["disp"][k][fpk_valid].tolist()
        fo = int(np.argmax(mag))
        rec["obj_world_pk"] = S["world"][k][fo].tolist(); rec["obj_img_pk"] = S["disp"][k][fo].tolist()
        rec["obj_pos0"] = S["pos0"][k].tolist()
        if rec["zoom_centre"] is not None and S["ok"][k][zpk]:
            rec["obj_pos_at_zoompk"] = (S["pos0"][k] + S["disp"][k][zpk]).tolist()
        # speeds (smoothed), px/s
        dt = np.gradient(t)
        vobj = np.linalg.norm(np.gradient(smooth(np.nan_to_num(wk)), axis=0), axis=-1) / np.maximum(dt, 1e-3)
        vcam = (np.linalg.norm(np.gradient(smooth(S["tc"]), axis=0), axis=-1) + 100 * np.abs(np.gradient(smooth(S["logs"])))) / np.maximum(dt, 1e-3)
        moving = (vobj >= 20) & S["wok"][k] & good
        still = (vobj < 5) & S["wok"][k] & good
        rec["vcam_obj_moving"] = float(np.mean(vcam[moving])) if moving.sum() >= 3 else None
        rec["vcam_obj_still"] = float(np.mean(vcam[still])) if still.sum() >= 3 else None
        act = np.nonzero(mag >= 4)[0]
        rec["obj_stop"] = float(t[act[-1]]) if act.size and act[-1] < len(t) - 3 else None
        cact = np.nonzero(cam_mag >= 8)[0]
        rec["cam_stop"] = float(t[cact[-1]]) if cact.size and cact[-1] < len(t) - 3 else None
        # cross-correlation of speeds (lag > 0: camera lags the mover)
        if rec["any"] and moving.sum() >= 3:
            a = vobj - vobj.mean(); b = vcam - vcam.mean(); fps = 1.0 / float(np.median(dt))
            lags = range(-int(fps), int(fps) + 1); xc = []
            for L in lags:
                if L >= 0:
                    x, y = a[:len(a) - L], b[L:]
                else:
                    x, y = a[-L:], b[:len(b) + L]
                xc.append(float(np.corrcoef(x, y)[0, 1]) if len(x) > 4 and x.std() > 0 and y.std() > 0 else np.nan)
            xc = np.array(xc)
            if np.isfinite(xc).any():
                rec["xcorr_lag_s"] = float(list(lags)[int(np.nanargmax(xc))] / fps); rec["xcorr_max"] = float(np.nanmax(xc))
    for h in S["hands"]:
        if S["wok"][h].sum() >= 2:
            fh = fpk if S["wok"][h][fpk] else int(np.nonzero(S["wok"][h])[0][-1])
            rec["hand_world_at_campk"] = S["world"][h][fh].tolist()
            rec["hand_pos0"] = S["pos0"][h].tolist()
            if rec["zoom_centre"] is not None and S["ok"][h][zpk]:
                rec["hand_pos_at_zoompk"] = (S["pos0"][h] + S["disp"][h][zpk]).tolist()
            break
    return rec


def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-6 and nb > 1e-6 else np.nan


def headroom(p, u):
    """distance from p to the frame border along unit direction u"""
    ds = []
    for i, lim in ((0, W), (1, H)):
        if u[i] > 1e-6:
            ds.append((lim - p[i]) / u[i])
        elif u[i] < -1e-6:
            ds.append(-p[i] / u[i])
    return float(min(ds)) if ds else np.inf


def frame_feats(idir, sid, masks, roles, ref):
    """first-frame features of an item"""
    f = {}
    try:
        import cv2
        img = cv2.imread(f"{M}/{idir}/frames/{sid}_t0_work.png", cv2.IMREAD_GRAYSCALE)
        if img is not None:
            f["sharpness"] = float(cv2.Laplacian(img, cv2.CV_64F).var()); f["brightness"] = float(img.mean())
    except Exception:
        pass
    K = masks.shape[0]
    hand = [k for k in range(K) if k < len(roles) and roles[k] == "hand"]
    f["hand_area"] = float(sum(masks[k].mean() for k in hand)) if hand else 0.0
    act = [k for k in range(K) if k < len(roles) and roles[k] == "actor"] or [k for k in range(K) if k not in hand][:1]
    if act:
        a = act[0]; ys, xs = np.nonzero(masks[a])
        if len(xs):
            f["actor_area"] = float(masks[a].mean())
            cx, cy = xs.mean(), ys.mean(); f["actor_border"] = float(min(cx, W - cx, cy, H - cy))
            f["actor_offcentre"] = float(np.hypot(cx - W / 2, cy - H / 2))
    if ref is not None:
        rr = ref.reading
        try:
            f["real_motion"] = float(max(np.linalg.norm(rr.compensated()[k][rr.ok[k]], axis=-1).max() for k in range(rr.n_objects) if rr.ok[k].any() and (k >= len(roles) or roles[k] != "hand")))
        except ValueError:
            pass
    return f


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else np.nan


def boot_ci(vals, groups, fn=np.mean, n=1000, seed=0):
    by = collections.defaultdict(list)
    for v, g in zip(vals, groups):
        by[g].append(v)
    keys = list(by); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        v = [x for g in rng.choice(keys, len(keys)) for x in by[g]]
        out.append(fn(v))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--finding-set", default="", help="only the H frames of this finding set (finding_set_A.json = MoProbe-H)")
    a = ap.parse_args()
    FS = set(json.load(open(a.finding_set))["H"]) if a.finding_set else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = {}
    def built(idir, base):
        if (idir, base) not in cache:
            try:
                cache[(idir, base)] = load_built(f"{M}/{idir}", base)
            except Exception:
                cache[(idir, base)] = None
        return cache[(idir, base)]
    recs, feats = [], {}
    for st, idir in ITEMS.items():
        files = sorted(Path(f"{M}/{st}_eval/measure").glob("*.measure.npz"))
        for n, p in enumerate(files):
            if a.limit and n >= a.limit:
                break
            arm, sid, seed = p.name[:-len(".measure.npz")].split("__")
            if arm not in MODELS:
                continue
            base = sid.split("-")[0]
            if FS is not None and base not in FS:
                continue
            b = built(idir, base)
            if b is None:
                continue
            d, it, masks, ref = b; r = Reading.from_npz(p)
            if r.bg_disp is None:
                continue
            roles = d.get("roles") or []
            S = series(r.times, r.disp, r.ok, r.pos0, r.bg_disp, r.bg_ok, masks, roles, geom=d.get("geometry"))
            if S is None:
                continue
            rec = per_clip(S)
            cond = sid.split("-")[1] if "-" in sid else "plain"
            rec.update(set=st, arm=arm, sid=sid, seed=int(seed[1:]), item=base, cond=cond if cond in ("nx", "ex", "cf") else "other", template=d.get("template_id"))
            recs.append(rec)
            if base not in feats:
                feats[base] = frame_feats(idir, base, masks, roles, ref)
    # real continuations (same window), only items in use
    used = set(json.load(open(f"{M}/cfset/sel/pair_meta.json")))
    used |= {json.loads(l)["item_sid"] for l in open(f"{M}/dirbal/sel/gen_manifest.jsonl")}
    used |= {json.loads(l)["item_sid"] for l in open(f"{M}/fx/sel/gen_manifest_s0.jsonl")}
    reals, seen = {}, set()
    for st, idir in ITEMS.items():
        for p in sorted(Path(f"{M}/{idir}/ref").glob("*.npz")):
            sid = p.stem
            if sid not in used or sid in seen or (FS is not None and sid not in FS):
                continue
            seen.add(sid); b = built(idir, sid)
            if b is None:
                continue
            d, it, masks, ref = b; z = np.load(p)
            if "bg_disp" not in z:
                continue
            S = series(z["times"], z["disp"], z["ok"].astype(bool), z["pos0"], z["bg_disp"], z["bg_ok"].astype(bool), masks, d.get("roles") or [], geom=d.get("geometry"))
            if S is None:
                continue
            rec = per_clip(S); rec.update(arm="real", item=sid); reals[sid] = rec
    print(f"generated clips {len(recs)}, real {len(reals)}, frames with features {len(feats)}")
    open_recs = [x for x in recs if x["arm"] in OPEN]
    def rows_for(arm):
        if arm == "real":
            return list(reals.values())
        if arm == "OPEN5":
            return open_recs
        return [x for x in recs if x["arm"] == arm]
    res, L = {}, ["# Camera vs object motion: regularities", ""]
    grp = lambda xs: [x["item"] for x in xs]

    # 1. direction of pans vs mover / hand
    L += ["## 1. Pan direction vs the main mover's world motion (pans: translation >= 16 px; mover world >= 8 px at the pan peak)", "",
          "| model | pans with a moving mover | cos median | follows (cos >= 0.5) | opposes (cos <= -0.5) | null follows (permuted) | pans with a hand | follows the hand | null (hand) |",
          "|---|---|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for arm in MODELS + ["OPEN5", "real"]:
        xs = rows_for(arm)
        pans = [x for x in xs if x["trans"] >= T_DIR and "obj_world_at_campk" in x and np.linalg.norm(x["obj_world_at_campk"]) >= 8]
        cs = np.array([cos(x["cam_vec"], x["obj_world_at_campk"]) for x in pans])
        hp = [x for x in xs if x["trans"] >= T_DIR and "hand_world_at_campk" in x and np.linalg.norm(x["hand_world_at_campk"]) >= 8]
        ch = np.array([cos(x["cam_vec"], x["hand_world_at_campk"]) for x in hp])
        null, nullh = [], []
        if len(pans) >= 5:
            for _ in range(200):
                perm = rng.permutation(len(pans)); null.append(np.mean([cos(pans[i]["cam_vec"], pans[j]["obj_world_at_campk"]) >= 0.5 for i, j in enumerate(perm)]))
        if len(hp) >= 5:
            for _ in range(200):
                perm = rng.permutation(len(hp)); nullh.append(np.mean([cos(hp[i]["cam_vec"], hp[j]["hand_world_at_campk"]) >= 0.5 for i, j in enumerate(perm)]))
        row = dict(n=len(pans), cos_med=float(np.nanmedian(cs)) if len(cs) else None, follows=float(np.mean(cs >= 0.5)) if len(cs) else None,
                   opposes=float(np.mean(cs <= -0.5)) if len(cs) else None, null=float(np.mean(null)) if null else None,
                   n_hand=len(hp), follows_hand=float(np.mean(ch >= 0.5)) if len(ch) else None, null_hand=float(np.mean(nullh)) if nullh else None)
        if len(pans) >= 10 and arm != "real":
            row["follows_ci"] = boot_ci([c >= 0.5 for c in cs], grp(pans))
        res[f"dir/{arm}"] = row
        f2 = lambda v: "—" if v is None else f"{v:.2f}"
        L.append(f"| {NICE[arm]} | {len(pans)} | {f2(row['cos_med'])} | **{f2(row['follows'])}**{(' [%.2f, %.2f]' % row['follows_ci']) if 'follows_ci' in row else ''} | {f2(row['opposes'])} | {f2(row['null'])} | {len(hp)} | {f2(row['follows_hand'])} | {f2(row['null_hand'])} |")

    # 1b. follow share by pan size -- why direction analyses use pans >= T_DIR (camera_rule.CAM_DIR_MIN_PX)
    res["dir_bins"] = {}
    for grp_name, pool in (("generated", open_recs), ("real", list(reals.values()))):
        out_b = []
        for lo, hi in ((T_TR, T_DIR), (T_DIR, 32.0), (32.0, 64.0), (64.0, 1e9)):
            xs = [x for x in pool if lo <= x["trans"] < hi and "obj_world_at_campk" in x and np.linalg.norm(x["obj_world_at_campk"]) >= 8]
            cs = np.array([cos(x["cam_vec"], x["obj_world_at_campk"]) for x in xs]); cs = cs[np.isfinite(cs)] if len(cs) else cs
            out_b.append(dict(lo=lo, hi=hi, n=int(len(cs)), follows=float(np.mean(cs >= 0.5)) if len(cs) else None, opposes=float(np.mean(cs <= -0.5)) if len(cs) else None))
        res["dir_bins"][grp_name] = out_b
    L += ["", "Follow share by pan size (generated / real): " + "; ".join(f"{b['lo']:.0f}-{b['hi']:.0f} px n{b['n']} follows {b['follows']}" for g in ("generated", "real") for b in res["dir_bins"][g])]

    # 2. zoom
    L += ["", "## 2. Zoom: direction and centre (zoom clips: |log scale| >= 0.10)", "",
          "| model | zoom clips | zoom-in share | centre fitted | centre nearer the mover than the image centre is | median distance centre→mover (px) | median distance image centre→mover | centre nearer the hand |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in MODELS + ["OPEN5", "real"]:
        xs = rows_for(arm)
        zs = [x for x in xs if x["zoom"] >= T_ZOOM]
        zc = [x for x in zs if x.get("zoom_centre") is not None and "obj_pos_at_zoompk" in x]
        dzc = np.array([np.linalg.norm(np.subtract(x["zoom_centre"], x["obj_pos_at_zoompk"])) for x in zc])
        dic = np.array([np.linalg.norm(np.subtract(CEN, x["obj_pos_at_zoompk"])) for x in zc])
        zh = [x for x in zs if x.get("zoom_centre") is not None and "hand_pos_at_zoompk" in x]
        nh = np.mean([np.linalg.norm(np.subtract(x["zoom_centre"], x["hand_pos_at_zoompk"])) < np.linalg.norm(np.subtract(CEN, x["hand_pos_at_zoompk"])) for x in zh]) if zh else None
        row = dict(n=len(zs), zoom_in=float(np.mean([x["zoom_in"] for x in zs])) if zs else None, n_centre=len(zc),
                   nearer_mover=float(np.mean(dzc < dic)) if len(zc) else None, med_d_centre=float(np.median(dzc)) if len(zc) else None,
                   med_d_image=float(np.median(dic)) if len(zc) else None, nearer_hand=None if nh is None else float(nh), n_hand=len(zh))
        res[f"zoom/{arm}"] = row
        f2 = lambda v: "—" if v is None else f"{v:.2f}"
        L.append(f"| {NICE[arm]} | {len(zs)} | {f2(row['zoom_in'])} | {len(zc)} | **{f2(row['nearer_mover'])}** | {f2(row['med_d_centre'])} | {f2(row['med_d_image'])} | {f2(row['nearer_hand'])} (n {len(zh)}) |")

    # 3. framing: pan towards the mover's heading vs headroom along the heading
    L += ["", "## 3. Framing: does the camera pan after a mover heading out of frame? (mover world peak >= 16 px)", "",
          "| headroom along the heading (px) | clips | any camera | pan towards the heading (cos >= 0.5, translation >= 16) | pan away | real videos: pan towards |",
          "|---|---|---|---|---|---|"]
    bins = [(0, 40), (40, 100), (100, 200), (200, 1e9)]
    def hrows(xs):
        out = []
        for x in xs:
            if "obj_world_pk" not in x or np.linalg.norm(x["obj_world_pk"]) < 16:
                continue
            u = np.asarray(x["obj_world_pk"]) / np.linalg.norm(x["obj_world_pk"])
            hr = headroom(np.asarray(x["obj_pos0"]), u)
            c = cos(x["cam_vec"], u) if x["trans"] >= T_DIR else np.nan
            out.append((hr, x["any"], bool(c >= 0.5), bool(c <= -0.5), x["item"]))
        return out
    hg, hr_real = hrows(open_recs), hrows(list(reals.values()))
    res["framing"] = []
    for lo, hi in bins:
        g = [h for h in hg if lo <= h[0] < hi]; rr = [h for h in hr_real if lo <= h[0] < hi]
        if not g:
            continue
        row = dict(lo=lo, hi=hi, n=len(g), any=float(np.mean([h[1] for h in g])), towards=float(np.mean([h[2] for h in g])), away=float(np.mean([h[3] for h in g])),
                   real_towards=float(np.mean([h[2] for h in rr])) if rr else None, n_real=len(rr))
        row["towards_ci"] = boot_ci([h[2] for h in g], [h[4] for h in g])
        res["framing"].append(row)
        rt = "—" if row["real_towards"] is None else "%.2f" % row["real_towards"]
        L.append(f"| {lo:.0f}–{hi if hi < 1e8 else '∞'} | {len(g)} | {row['any']:.2f} | **{row['towards']:.2f}** [{row['towards_ci'][0]:.2f}, {row['towards_ci'][1]:.2f}] | {row['away']:.2f} | "
                 f"{rt} (n {len(rr)}) |")

    # 4. timing
    L += ["", "## 4. Timing (clips with camera motion and a moving mover)", ""]
    for arm in MODELS + ["OPEN5"]:
        xs = [x for x in rows_for(arm) if x["any"]]
        on = np.array([x["cam_onset"] - x["obj_onset"] for x in xs if x.get("cam_onset") is not None and x.get("obj_onset") is not None])
        stp = np.array([x["cam_stop"] - x["obj_stop"] for x in xs if x.get("cam_stop") is not None and x.get("obj_stop") is not None])
        lag = np.array([x["xcorr_lag_s"] for x in xs if x.get("xcorr_lag_s") is not None and x.get("xcorr_max", 0) >= 0.3])
        vm = [x["vcam_obj_moving"] for x in xs if x.get("vcam_obj_moving") is not None and x.get("vcam_obj_still") is not None]
        vs = [x["vcam_obj_still"] for x in xs if x.get("vcam_obj_moving") is not None and x.get("vcam_obj_still") is not None]
        abs_on = np.array([x["cam_onset"] for x in xs if x.get("cam_onset") is not None])
        res[f"timing/{arm}"] = dict(n_on=len(on), onset_med=float(np.median(on)) if len(on) else None, obj_first=float(np.mean(on > 0.25)) if len(on) else None,
                                    cam_first=float(np.mean(on < -0.25)) if len(on) else None, n_stop=len(stp), stop_med=float(np.median(stp)) if len(stp) else None,
                                    n_lag=len(lag), lag_med=float(np.median(lag)) if len(lag) else None, n_speed=len(vm),
                                    vcam_moving=float(np.median(vm)) if vm else None, vcam_still=float(np.median(vs)) if vs else None,
                                    cam_onset_abs_med=float(np.median(abs_on)) if len(abs_on) else None, cam_onset_before_05s=float(np.mean(abs_on < 0.5)) if len(abs_on) else None)
        t_ = res[f"timing/{arm}"]
        f2 = lambda v, fmt="+.2f": "—" if v is None else format(v, fmt)
        L.append(f"- {NICE[arm]}: camera onset − mover onset median {f2(t_['onset_med'])} s (n {t_['n_on']}; mover first {f2(t_['obj_first'], '.2f')}, camera first {f2(t_['cam_first'], '.2f')}); "
                 f"camera stop − mover stop median {f2(t_['stop_med'])} s (n {t_['n_stop']}); speed cross-correlation peak lag median {f2(t_['lag_med'])} s (n {t_['n_lag']}); "
                 f"camera speed while the mover moves {f2(t_['vcam_moving'], '.1f')} vs while it is still {f2(t_['vcam_still'], '.1f')} px/s (n {t_['n_speed']}); "
                 f"camera onset (absolute) median {f2(t_['cam_onset_abs_med'], '.2f')} s, before 0.5 s {f2(t_['cam_onset_before_05s'], '.2f')}")

    # 5. propensity: first frame vs model vs text
    L += ["", "## 5. Is camera motion a property of the first frame?", ""]
    by_item = collections.defaultdict(list)
    for x in open_recs:
        by_item[x["item"]].append(float(x["any"]))
    items = [i for i, v in by_item.items() if len(v) >= 4] or list(by_item)
    allv = np.array([v for i in items for v in by_item[i]]); grand = allv.mean()
    between = np.mean([(np.mean(by_item[i]) - grand) ** 2 for i in items]); total = allv.var()
    perm_between = []
    flat = [(i, v) for i in items for v in by_item[i]]
    for _ in range(100):
        vals = rng.permutation([v for _, v in flat]); pb = collections.defaultdict(list)
        for (i, _), v in zip(flat, vals):
            pb[i].append(v)
        perm_between.append(np.mean([(np.mean(pb[i]) - grand) ** 2 for i in items]))
    res["propensity"] = dict(items=len(items), clips=len(allv), share_between=float(between / total), share_between_null=float(np.mean(perm_between) / total))
    L.append(f"Share of the variance of 'camera moved' that lies between first frames: **{between / total:.2f}** (same clips with frames shuffled: {np.mean(perm_between) / total:.2f}); {len(items)} frames with >= 4 clips.")
    # cross-model agreement on per-frame rates
    pm = collections.defaultdict(dict)
    for x in recs:
        pm[x["arm"]].setdefault(x["item"], []).append(float(x["any"]))
    L += ["", "Per-frame camera rate, Spearman correlation between models (frames both models generated):", "", "| | " + " | ".join(NICE[m] for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    res["agree"] = {}
    for m1 in MODELS:
        cells = []
        for m2 in MODELS:
            sh = [i for i in pm[m1] if i in pm[m2]]
            if m1 == m2 or len(sh) < 15:
                cells.append("—"); continue
            rho = spearman(np.array([np.mean(pm[m1][i]) for i in sh]), np.array([np.mean(pm[m2][i]) for i in sh]))
            res["agree"][f"{m1}|{m2}"] = dict(rho=float(rho), n=len(sh)); cells.append(f"{rho:+.2f} (n {len(sh)})")
        L.append(f"| {NICE[m1]} | " + " | ".join(cells) + " |")
    # first-frame features
    L += ["", "First-frame features vs the frame's camera rate (5 models pooled; Spearman over frames, bootstrap CI over frames; tertile rates):", "",
          "| feature | frames | Spearman ρ [95% CI] | low tertile | mid | high |", "|---|---|---|---|---|---|"]
    res["features"] = {}
    for fname in ("hand_area", "actor_area", "actor_border", "actor_offcentre", "sharpness", "brightness", "real_motion", "real_cam"):
        pairs = []
        for i in items:
            fv = feats.get(i, {}).get(fname) if fname != "real_cam" else (reals[i]["trans"] if i in reals else None)
            if fv is not None and np.isfinite(fv):
                pairs.append((fv, np.mean(by_item[i])))
        if len(pairs) < 20:
            continue
        fv = np.array([p[0] for p in pairs]); cr = np.array([p[1] for p in pairs])
        rho = spearman(fv, cr)
        bs = []
        for _ in range(500):
            idx = rng.integers(0, len(pairs), len(pairs)); bs.append(spearman(fv[idx], cr[idx]))
        q = np.quantile(fv, [1 / 3, 2 / 3]); tert = [cr[fv <= q[0]].mean(), cr[(fv > q[0]) & (fv <= q[1])].mean(), cr[fv > q[1]].mean()]
        res["features"][fname] = dict(n=len(pairs), rho=float(rho), ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], tertiles=[float(t) for t in tert])
        L.append(f"| {fname} | {len(pairs)} | **{rho:+.2f}** [{np.nanpercentile(bs, 2.5):+.2f}, {np.nanpercentile(bs, 97.5):+.2f}] | {tert[0]:.2f} | {tert[1]:.2f} | {tert[2]:.2f} |")
    # templates
    L += ["", "Camera rate by SSv2 template (>= 30 clips):", "", "| template | clips | frames | camera rate |", "|---|---|---|---|"]
    tl = collections.defaultdict(list); tf = collections.defaultdict(set)
    for x in open_recs:
        if x.get("template") is not None:
            tl[x["template"]].append(x["any"]); tf[x["template"]].add(x["item"])
    res["templates"] = {}
    for tid, v in sorted(tl.items(), key=lambda kv: -np.mean(kv[1])):
        if len(v) >= 30:
            res["templates"][str(tid)] = dict(n=len(v), frames=len(tf[tid]), rate=float(np.mean(v)))
            L.append(f"| {tid} | {len(v)} | {len(tf[tid])} | {np.mean(v):.2f} |")

    # 6. signatures
    L += ["", "## 6. Model signatures (clips with camera motion)", "", "| model | camera clips | pan only | zoom only | pan + zoom | rotation >= 3° | zoom-in share of zooms | median max translation | median max zoom |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in MODELS + ["real"]:
        xs = rows_for(arm)
        cx = [x for x in xs if x["any"]]
        if not cx:
            continue
        ty = collections.Counter(x["type"] for x in cx); zs = [x for x in cx if x["zoom"] >= T_ZOOM]
        res[f"sig/{arm}"] = dict(n=len(cx), pan=ty["pan"] / len(cx), zoom=ty["zoom"] / len(cx), mixed=ty["mixed"] / len(cx), rot=float(np.mean([x["rot"] >= T_ROT for x in cx])),
                                 zoom_in=float(np.mean([x["zoom_in"] for x in zs])) if zs else None, med_trans=float(np.median([x["trans"] for x in cx])), med_zoom=float(np.median([x["zoom"] for x in cx])))
        s_ = res[f"sig/{arm}"]
        zi = "—" if s_["zoom_in"] is None else "%.2f" % s_["zoom_in"]
        L.append(f"| {NICE[arm]} | {len(cx)} | {s_['pan']:.2f} | {s_['zoom']:.2f} | {s_['mixed']:.2f} | {s_['rot']:.2f} | {zi} | {s_['med_trans']:.1f} | {s_['med_zoom']:.3f} |")

    # 7. compensation
    L += ["", "## 7. Apparent vs world motion of the main mover (at its peak)", "", "| model | clips | camera moved | apparent only (image >= 8, world < 8) | cancelled (world >= 8, image < 8) | |image| / |world| median when both >= 8 (camera clips) | same, no-camera clips |", "|---|---|---|---|---|---|---|"]
    for arm in MODELS + ["OPEN5"]:
        xs = [x for x in rows_for(arm) if "obj_img_pk" in x]
        if not xs:
            continue
        img = np.array([np.linalg.norm(x["obj_img_pk"]) for x in xs], float); wor = np.array([np.linalg.norm(x["obj_world_pk"]) for x in xs], float)
        cam = np.array([bool(x["any"]) for x in xs], dtype=bool)
        both = (img >= 8) & (wor >= 8)
        res[f"comp/{arm}"] = dict(n=len(xs), cam=float(cam.mean()), apparent=float(np.mean((img >= 8) & (wor < 8))), cancelled=float(np.mean((wor >= 8) & (img < 8))),
                                  ratio_cam=float(np.median(img[both & cam] / wor[both & cam])) if (both & cam).any() else None,
                                  ratio_nocam=float(np.median(img[both & ~cam] / wor[both & ~cam])) if (both & ~cam).any() else None)
        c_ = res[f"comp/{arm}"]
        L.append(f"| {NICE[arm]} | {len(xs)} | {c_['cam']:.2f} | {c_['apparent']:.3f} | {c_['cancelled']:.3f} | {c_['ratio_cam'] or float('nan'):.2f} | {c_['ratio_nocam'] or float('nan'):.2f} |")

    # 8. real cue
    L += ["", "## 8. Does the generated pan continue the real clip's own small camera drift?", ""]
    pr = [(cos(x["cam_vec"], reals[x["item"]]["cam_vec"])) for x in open_recs if x["trans"] >= T_DIR and x["item"] in reals and reals[x["item"]]["trans"] >= 4]
    pr = np.array([c for c in pr if np.isfinite(c)])
    if len(pr):
        res["realcue"] = dict(n=len(pr), same=float(np.mean(pr >= 0.5)), opposite=float(np.mean(pr <= -0.5)))
        L.append(f"Generated pans (>= 16 px) on frames whose real continuation drifts >= 4 px: n {len(pr)}; same direction (cos >= 0.5) {np.mean(pr >= 0.5):.2f}, opposite {np.mean(pr <= -0.5):.2f} (uniform directions: 0.33 / 0.33).")
    json.dump(res, open(out / "camera_patterns.json", "w"), indent=1, default=float)
    with open(out / "camera_patterns_records.jsonl", "w") as fh:
        for x in recs + list(reals.values()):
            fh.write(json.dumps(x, default=float) + "\n")
    json.dump(feats, open(out / "frame_features.json", "w"), indent=1, default=float)
    (out / "camera_patterns.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
