#!/usr/bin/env python3
"""Camera motion, second look.  The stored camera estimate is a pure
TRANSLATION (median background displacement), so a dolly / zoom (radial flow, median ~ 0) and a rotation are invisible
to it, and zoom-induced radial motion of an off-centre object is mis-read as object motion.  Here every clip's background
grid (bg_disp / bg_ok of the Reading, positions regenerated with instrument.background_points) is fitted per frame with a
SIMILARITY transform (scale s, rotation r, translation t; robust refit after dropping outliers):
  camera (any)   translation >= 16 px (at the image centre), or |log s| >= 0.10 (zoom >= 10 %), or |r| >= 3 deg
  scene shift    the real-static objects (>= 2, not hands) all move together (median peak >= 12 px) -- an independent witness
For movers with a known expected direction (dirbal real direction, fx exp2 instructed side):
  image          displacement at 2 s in image space, along the expected direction (>= 8 px within 60 deg = yes)
  world          displacement after removing the fitted camera (object position minus where a static point would be)
  substitution   image yes, world no                       (the camera alone produces the apparent motion)
  camera share   1 - world.u / image.u on 'image yes' clips (fraction of the apparent motion due to the camera; < 0 = tracking shot)
Also: the clips listed as camera motion in $MOPROBE_ROOT/audit/camera_flagged_ids.json (a manual check), with all metrics.
usage: MOPROBE_ROOT=DIR python scripts/motionbench/camera_deep.py --out $MOPROBE_ROOT/camera_eval
"""
import argparse, collections, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from motionbench import instrument as I                         # noqa: E402
from motionbench.containers import Reading                      # noqa: E402
from motionbench.io import load_built                           # noqa: E402

M = os.environ.get("MOPROBE_ROOT", "moprobe")                 # MoProbe working root (see docs/MOPROBE.md)
OPEN = ["wan22_a14b", "wan22_ti2v5b", "hunyuan15_480p", "cogvideox15_5b", "ltx23"]
OURS = ["moplan", "moplan_norl", "wan_p448", "moplan_zero", "moplan_nosft"]    # MoPlan + ablation arms + base under the MoPlan protocol, never pooled into OPEN5
MODELS = OPEN + OURS
NICE = {"wan22_a14b": "Wan2.2-I2V-A14B", "wan22_ti2v5b": "Wan2.2-TI2V-5B", "hunyuan15_480p": "HunyuanVideo-1.5",
        "cogvideox15_5b": "CogVideoX1.5-5B", "ltx23": "LTX-2.3", "moplan": "MoPlan (final)", "moplan_norl": "MoPlan no RL", "wan_p448": "Wan-A14B (our protocol)", "moplan_zero": "MoPlan zero plan", "moplan_nosft": "MoPlan RAFT w/o SFT", "OPEN5": "5 models pooled"}
DIRV = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
CX, CY = 224.0, 128.0
from motionbench.metrics.camera_rule import CAM_TRANS_PX as T_TR, CAM_ZOOM as T_ZOOM, CAM_ROT_DEG as T_ROT, verdict   # noqa: E402  one camera rule for every analysis


def similarity(p, q):
    """least-squares s R p + t = q (Umeyama); returns (s, rot_deg, t[2], residuals)"""
    mp, mq = p.mean(0), q.mean(0); P, Q = p - mp, q - mq
    U, S, Vt = np.linalg.svd(Q.T @ P)
    d = np.sign(np.linalg.det(U @ Vt)); D = np.diag([1, d])
    R = U @ D @ Vt; s = (S * [1, d]).sum() / max((P ** 2).sum(), 1e-9)
    t = mq - s * R @ mp
    res = np.linalg.norm((s * (R @ p.T)).T + t - q, axis=-1)
    return s, float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), t, res, R


def content_box(p0, geom, margin=3.0):
    """[P] bool: grid points inside the item's letterboxed content box (work frame).  Needed because the background grid
    covers the whole 448x256 work frame, so for non-16:9 sources the static black bars (24 % of the points at 4:3,
    43 % square, 57-68 % portrait) dragged the fit to 'no camera motion' or made it unfittable."""
    return I.content_mask(p0, geom, margin)


def fit_camera(p0, disp, ok, iters=4, geom=None):
    """per frame: (scale, rot_deg, translation at the image centre [2], n_inliers) with iterative outlier trimming
    (hands / objects sweeping over background points are dropped: residual > max(3 px, 1.5 x median)); a frame whose
    inliers are < 60 % of the visible points or < 8 is marked unreliable (nan).  geom: the item's content box -- grid
    points on the letterbox bars are ignored (pass it for every non-16:9 source)"""
    ok = ok & content_box(p0, geom)[:, None]
    F = disp.shape[1]; out = np.zeros((F, 4)); out[:, 0] = 1.0; Rs = [np.eye(2)] * F; ts = [np.zeros(2)] * F
    for f in range(1, F):
        sel = ok[:, f] & ok[:, 0]
        if sel.sum() < 8:
            out[f] = [np.nan, np.nan, np.nan, int(sel.sum())]; continue
        p, q = p0[sel], p0[sel] + disp[sel, f]
        keep = np.ones(len(p), bool); s = r = t = R = None
        for _ in range(iters):
            s, r, t, res, R = similarity(p[keep], q[keep])
            res_all = np.linalg.norm((s * (R @ p.T)).T + t - q, axis=-1)
            new = res_all <= max(3.0, 1.5 * float(np.median(res_all[keep])))
            if new.sum() < 8 or np.array_equal(new, keep):
                break
            keep = new
        if keep.sum() < 8 or keep.mean() < 0.6:
            out[f] = [np.nan, np.nan, np.nan, int(keep.sum())]; continue
        c = np.array([CX, CY]); tc = s * (R @ c) + t - c
        out[f] = [s, r, float(np.linalg.norm(tc)), int(keep.sum())]; Rs[f] = R; ts[f] = t
    return out, Rs, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # expected directions (the dirbal real direction and the fx instructed side)
    feats = {f["sid"]: f for f in (json.loads(l) for l in open(f"{M}/dirbal/sel/features.jsonl")) if f.get("ok")}
    expect = {}
    for r in (json.loads(l) for l in open(f"{M}/dirbal/sel/gen_manifest.jsonl")):
        f = feats.get(r["item_sid"])
        if f and r["variant"] == "explicit":
            g = f["g"]; expect[("dirbal", r["sid"])] = (f["mover"], ("right" if g[0] > 0 else "left") if abs(g[0]) >= abs(g[1]) else ("down" if g[1] > 0 else "up"), f["cell"])
    picks = {p["sid"]: p for p in json.load(open(f"{M}/fx/pick2x2.json"))}
    for r in (json.loads(l) for l in open(f"{M}/fx/sel/gen_manifest_s0.jsonl")):
        if r.get("exp") == "exp2" and r.get("side"):
            d = json.load(open(f"{picks[r['item_sid']]['items_dir']}/items/{r['item_sid']}.json"))
            expect[("fx", r["sid"])] = (d["matched"][r["target_cat"]], r["side"], "exp2")
    # audit-flagged clips
    flagged = {}
    try:
        ids = set(json.load(open(f"{M}/audit/camera_flagged_ids.json")))
        for s in json.load(open(f"{M}/audit/spec.json")):
            if s["id"] in ids:
                for p in s["panels"]:
                    if p.get("kind") == "gen":
                        flagged[(p["set"], p["arm"], p["sid"], int(p.get("seed", 0)))] = s["id"]
    except Exception as e:
        print("audit mapping unavailable:", e)
    cache = {}
    def built(st, base):
        idir = {"cfset": "items_cfset", "dirbal": "items_dirbal", "fx": "items_fx"}[st]
        if (idir, base) not in cache:
            try:
                cache[(idir, base)] = load_built(f"{M}/{idir}", base)
            except Exception:
                cache[(idir, base)] = None
        return cache[(idir, base)]
    recs = []; check = []
    for st in ("cfset", "dirbal", "fx"):
        for p in sorted(Path(f"{M}/{st}_eval/measure").glob("*.measure.npz")):
            arm, sid, seed = p.name[:-len(".measure.npz")].split("__")
            if arm not in MODELS:
                continue
            base = sid.split("-")[0]; b = built(st, base)
            if b is None or b[3] is None:
                continue
            d, it, masks, ref = b
            r = Reading.from_npz(p)
            if r.bg_disp is None:
                continue
            p0, kind = I.background_points(masks, *masks.shape[1:3])
            if len(p0) != r.bg_disp.shape[0]:
                continue
            fit, Rs, ts = fit_camera(p0, r.bg_disp, r.bg_ok, geom=d.get("geometry"))
            if len(check) < 200:                                     # self-check: dense-grid median == stored camera
                f = len(r.times) - 1; sel = (kind == 1) & r.bg_ok[:, f] & r.bg_ok[:, 0]
                if sel.sum() >= 8:
                    check.append(float(np.linalg.norm(np.median(r.bg_disp[sel, f], 0) - r.camera[f])))
            good = np.isfinite(fit[:, 0])
            zoom = float(np.nanmax(np.abs(np.log(fit[good, 0])))) if good.any() else np.nan
            rot = float(np.nanmax(np.abs(fit[good, 1]))) if good.any() else np.nan
            tr = float(np.nanmax(fit[good, 2])) if good.any() else np.nan
            cam_old = float(np.linalg.norm(r.camera, axis=-1).max())
            roles = d.get("roles") or []
            statics = [k for k in range(min(r.n_objects, ref.reading.n_objects)) if ref.static[k] and (k >= len(roles) or roles[k] != "hand")]
            raw_pk = [float(np.linalg.norm(r.disp[k][r.ok[k]], axis=-1).max()) for k in statics if r.ok[k].any()]
            shift = float(np.median(raw_pk)) if len(raw_pk) >= 2 else np.nan
            cond = sid.split("-")[1] if "-" in sid else "plain"; cond = cond if cond in ("nx", "ex", "cf") else "other"
            rec = dict(set=st, arm=arm, sid=sid, seed=int(seed[1:]), cond=cond, cam_old=cam_old, trans=tr, zoom=zoom, rot=rot, shift=shift,
                       lost=float(np.mean(~good[1:])) if len(good) > 1 else 1.0, flagged=flagged.get((st, arm, sid, int(seed[1:]))))
            rec["verdict"] = verdict(tr, zoom, rot, rec["lost"], None if not np.isfinite(shift) else shift)   # camera_rule.verdict
            # expected-direction cases: image vs world displacement at 2 s
            e = expect.get((st, sid))
            if e and int(seed[1:]) == 0:
                k, dname, label = e
                if k < r.n_objects and r.ok[k].sum() >= 2:
                    tt = min(2.0, float(r.times[r.ok[k]][-1])); f2 = int(np.argmin(np.abs(r.times - tt)))
                    if r.ok[k][f2] and np.isfinite(fit[f2, 0]):
                        img = r.disp[k][f2]; pos0 = r.pos0[k]
                        pred = fit[f2, 0] * (Rs[f2] @ pos0) + ts[f2]                      # where a static point at pos0 would be
                        world = (pos0 + img) - pred
                        u = np.array(DIRV[dname], float)
                        iy, wy = float(img @ u), float(world @ u)
                        ang = lambda v: float(np.degrees(np.arccos(np.clip(v @ u / (np.linalg.norm(v) + 1e-9), -1, 1))))
                        rec.update(label=label, expected=dname, img_u=iy, world_u=wy, img_yes=bool(np.linalg.norm(img) >= 8 and ang(img) <= 60),
                                   world_yes=bool(np.linalg.norm(world) >= 8 and ang(world) <= 60), cam_share=(1 - wy / iy) if iy >= 8 else None)
            recs.append(rec)
    print(f"self-check (dense median vs stored camera, px): median {np.median(check):.2f}, max {np.max(check):.2f} over {len(check)} clips")
    L = ["# Camera motion, second look (similarity fit of the background grid)", "",
         f"Self-check: dense-grid median vs stored camera, median {np.median(check):.2f} px (n {len(check)}).", "",
         "## How often the camera moves in generated clips (all measured clips)", "",
         "| model | clips | translation >= 16 px | zoom >= 10 % | rotation >= 3° | any of the three | scene shift (static objects move together) | shift but translation < 16 (missed before) | grid lost (> 50 % frames unfit) |",
         "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for arm in MODELS + ["OPEN5"]:
        xs = [x for x in recs if (x["arm"] in OPEN if arm == "OPEN5" else x["arm"] == arm)]
        if not xs:
            continue
        tr = np.array([x["trans"] for x in xs]); zm = np.array([x["zoom"] for x in xs]); ro = np.array([x["rot"] for x in xs])
        anyc = (tr >= T_TR) | (zm >= T_ZOOM) | (ro >= T_ROT)
        sh = [x for x in xs if np.isfinite(x["shift"])]; shift = [x["shift"] >= 12 for x in sh]
        missed = [x["shift"] >= 12 and x["cam_old"] < 16 for x in sh]
        res[f"any/{arm}"] = dict(n=len(xs), trans=float(np.nanmean(tr >= T_TR)), zoom=float(np.nanmean(zm >= T_ZOOM)), rot=float(np.nanmean(ro >= T_ROT)), any=float(np.nanmean(anyc)),
                                shift=float(np.mean(shift)) if shift else None, shift_n=len(sh), missed=float(np.mean(missed)) if sh else None, lost=float(np.mean([x["lost"] > 0.5 for x in xs])))
        L.append(f"| {NICE[arm]} | {len(xs)} | {np.nanmean(tr >= T_TR):.2f} | {np.nanmean(zm >= T_ZOOM):.2f} | {np.nanmean(ro >= T_ROT):.2f} | **{np.nanmean(anyc):.2f}** | "
                 f"{np.mean(shift) if shift else float('nan'):.2f} (n {len(sh)}) | {np.mean(missed) if sh else float('nan'):.2f} | {np.mean([x['lost'] > 0.5 for x in xs]):.2f} |")
    L += ["", "## By text condition (original 5 open models pooled)", "", "| condition | clips | translation >= 16 | zoom >= 10 % | any | scene shift |", "|---|---|---|---|---|---|"]
    for cond in ("nx", "ex", "cf", "other"):
        xs = [x for x in recs if x["arm"] in OPEN and x["cond"] == cond]
        if not xs:
            continue
        tr = np.array([x["trans"] for x in xs]); zm = np.array([x["zoom"] for x in xs]); ro = np.array([x["rot"] for x in xs])
        sh = [x["shift"] >= 12 for x in xs if np.isfinite(x["shift"])]
        L.append(f"| {cond} | {len(xs)} | {np.nanmean(tr >= T_TR):.2f} | {np.nanmean(zm >= T_ZOOM):.2f} | {np.nanmean((tr >= T_TR) | (zm >= T_ZOOM) | (ro >= T_ROT)):.2f} | {np.mean(sh) if sh else float('nan'):.2f} |")
    L += ["", "## Expected-direction movers: is the apparent motion made by the camera?", "",
          "| model | clips | image-space yes | world yes | substitution (image yes, world no) | camera share median on image-yes | share >= 0.5 (camera makes most of it) | share <= -0.5 (tracking shot) |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in MODELS + ["OPEN5"]:
        xs = [x for x in recs if "img_yes" in x and (x["arm"] in OPEN if arm == "OPEN5" else x["arm"] == arm)]
        if not xs:
            continue
        sh = [x["cam_share"] for x in xs if x["cam_share"] is not None]
        sub = [x["img_yes"] and not x["world_yes"] for x in xs]
        res[f"dir/{arm}"] = dict(n=len(xs), img=float(np.mean([x["img_yes"] for x in xs])), world=float(np.mean([x["world_yes"] for x in xs])), substitution=float(np.mean(sub)),
                                share_med=float(np.median(sh)) if sh else None, share_ge05=float(np.mean(np.array(sh) >= 0.5)) if sh else None, share_le05=float(np.mean(np.array(sh) <= -0.5)) if sh else None)
        L.append(f"| {NICE[arm]} | {len(xs)} | {np.mean([x['img_yes'] for x in xs]):.2f} | {np.mean([x['world_yes'] for x in xs]):.2f} | **{np.mean(sub):.2f}** | "
                 f"{np.median(sh) if sh else float('nan'):+.2f} | {np.mean(np.array(sh) >= 0.5) if sh else float('nan'):.2f} | {np.mean(np.array(sh) <= -0.5) if sh else float('nan'):.2f} |")
    L += ["", "## By scene kind (original 5 open models pooled)", "", "| kind | clips | image yes | world yes | substitution | camera share median |", "|---|---|---|---|---|---|"]
    for lab in sorted({x["label"] for x in recs if "label" in x}):
        xs = [x for x in recs if x["arm"] in OPEN and x.get("label") == lab]; sh = [x["cam_share"] for x in xs if x["cam_share"] is not None]
        L.append(f"| {lab} | {len(xs)} | {np.mean([x['img_yes'] for x in xs]):.2f} | {np.mean([x['world_yes'] for x in xs]):.2f} | {np.mean([x['img_yes'] and not x['world_yes'] for x in xs]):.2f} | {np.median(sh) if sh else float('nan'):+.2f} |")
    L += ["", "## The clips the human audit flagged as camera motion", "", "| audit item | clip | old camera max | translation | zoom | rotation | static-object shift | grid lost |", "|---|---|---|---|---|---|---|---|"]
    for x in sorted([x for x in recs if x["flagged"]], key=lambda x: x["flagged"]):
        L.append(f"| {x['flagged']} | {x['arm']} {x['sid']} | {x['cam_old']:.0f} | {x['trans']:.0f} | {x['zoom']:.2f} | {x['rot']:.1f} | {x['shift'] if np.isfinite(x['shift']) else float('nan'):.0f} | {x['lost']:.2f} |")
    json.dump(res, open(out / "camera_deep.json", "w"), indent=1)
    with open(out / "camera_deep_records.jsonl", "w") as fh:
        for x in recs:
            fh.write(json.dumps(x, default=float) + "\n")
    (out / "camera_deep.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
