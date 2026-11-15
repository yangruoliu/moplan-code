#!/usr/bin/env python3
"""Confabulation, object part (the invented cause need not be a hand -- a model may invent other objects): a
class-agnostic scan for things that are NOT in the first frame and appear later, and whether they touch the moving
object.  Per clip (rows of <set>_eval/contact_manifest.jsonl, the same frames / work frame / times as the contact pass):
  tracked first-frame instances  masks_packed of the contact pass (every k16 instance, propagated by SAM2)
  camera                         similarity fit of the contact pass's background grid (camera_deep.fit_camera)
  candidates                     SAM2 automatic masks (2x work frame) at 0.5 s steps; a candidate must be
    (a) not a tracked instance: < 30 % of it inside the tracked masks (dilated 4 px) at that frame
    (b) not revealed by the camera: >= 70 % of it maps back (inverse camera) inside the first frame's content
    (c) not a dis-occlusion: < 60 % of the mapped region lay on a first-frame instance (an arm reaching for the object
        crosses the object's first-frame place, revealed background lies almost entirely inside it)
    (d) not there before: DINOv2 patch features under the mask differ from the first frame's at the mapped place
        (mean cosine < --tau); a wall patch that SAM2 merely segments differently later keeps its appearance
    (e) area >= 0.4 % of the frame and persistent: another candidate within 1 s (another sampled time) whose centre in
        first-frame coordinates is within 1/4 of the frame diagonal and whose area is within 4x (agents move)
  touches_mover  a kept candidate within 3 px of a first-frame instance that moves >= 8 px in this clip
Writes <out>/<arm>__<sid>__s0.newobj.json (+ <...>_c<k>.jpg crops: first frame | frame t, mask outlined) for the audit.
usage: MOPROBE_ROOT=DIR CUDA_VISIBLE_DEVICES=0 python new_object_scan.py --set ps --out $MOPROBE_ROOT/newobj_eval/ps [--arms real wan22_a14b ...]
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench import video as V                              # noqa: E402
from motionbench import instrument as I                         # noqa: E402
from motionbench.io import load_built                           # noqa: E402
from camera_deep import fit_camera                              # noqa: E402

M = os.environ.get("MOPROBE_ROOT", "moprobe")                 # MoProbe working root (see docs/MOPROBE.md)
SAM2_CKPT = os.environ.get("SAM2_CHECKPOINT", "sam2.1_hiera_large.pt")   # SAM 2.1 hiera-large checkpoint
DINO = os.environ.get("DINOV2_MODEL", "facebook/dinov2-base")             # DINOv2-base (local dir or hub id)
STEP_S, MIN_AREA, MAX_AREA = 0.5, 0.004, 0.35


class Dino:
    def __init__(self):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.m = AutoModel.from_pretrained(DINO).cuda().eval().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def patches(self, img):
        """img [H,W,3] uint8 work frame -> unit features [h,w,C] on a 14 px grid of the 2x frame"""
        t = self.torch
        x = cv2.resize(img, (896, 504))
        x = t.from_numpy(x).cuda().permute(2, 0, 1)[None].float() / 255.0
        x = ((x - self.mean) / self.std).half()
        with t.inference_mode():
            h = self.m(pixel_values=x).last_hidden_state[0, 1:]          # [64*36, C]
        f = t.nn.functional.normalize(h.float(), dim=-1).view(36, 64, -1)
        return f.cpu().numpy()


def to_first(pts, s, R, tr):
    """work-frame points at frame f -> first-frame coordinates (inverse similarity)"""
    A = s * R
    return np.linalg.solve(A, (pts - tr).T).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="*", default=None); ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--debug", action="store_true"); ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(f"{M}/{a.set}_eval/contact_manifest.jsonl")]
    rows = [r for r in rows if (a.arms is None or r["arm"] in a.arms) and int(r.get("seed", 0)) == 0 and (a.only is None or r["sid"] in a.only)][a.shard_index::a.num_shards]
    from motionbench.sam2auto import Sam2Masks
    sam = Sam2Masks(SAM2_CKPT, points_per_side=24); dino = Dino()
    for r in rows:
        name = f"{r['arm']}__{r['sid']}__s0"
        cp = Path(f"{M}/{a.set}_eval/contact/{name}.contact.npz")
        if (out / f"{name}.newobj.json").is_file() and not a.debug or not cp.is_file():
            continue
        t0 = time.time()
        try:
            z = np.load(cp)
            if "masks_packed" not in z:
                continue
            d, item, masks, _ = load_built(r["items_dir"], r["item_sid"]); g = d.get("geometry")
            frames, fps = V.read_video(str(r["path"]), start_frame=int(r.get("start_frame", 0)), max_seconds=r.get("max_seconds", 4.0),
                                       fps_override=r.get("fps"), step=int(r.get("step", 1)), max_side=r.get("max_side"))
            if r.get("canvas", False):
                work = V.canvas_to_work(frames, d["native_hw"], g, crop=r.get("canvas_crop") or V.MODEL_CANVAS_CROP.get(r.get("arm")))
            else:
                work, _ = V.to_work(frames)
            pin = cv2.imread(str(Path(r["items_dir"]) / "frames" / f"{r['item_sid']}_t0_work.png"))
            if pin is not None:
                work[0] = cv2.cvtColor(pin, cv2.COLOR_BGR2RGB)
            work, fps_eff, times = V.subsample(work, fps, 96)
            tz = z["times"]; F = min(len(tz), len(work))
            shp = tuple(int(v) for v in z["masks_shape"])
            mk = np.unpackbits(z["masks_packed"], axis=-1)[..., :shp[-1]].astype(bool)          # [F,K,H,W]
            H, W = mk.shape[2:]
            p0, _ = I.background_points(masks, H, W)
            fit, Rs, ts = fit_camera(p0, z["bg_disp"], z["bg_ok"].astype(bool), geom=g)
            content = I.content_mask(np.stack(np.meshgrid(np.arange(W), np.arange(H)), -1).reshape(-1, 2).astype(float), g).reshape(H, W) \
                if g else np.ones((H, W), bool)
            disp, okm = z["disp"], z["ok"].astype(bool)
            moving = [k for k in range(disp.shape[0]) if okm[k].any() and np.linalg.norm(disp[k][okm[k]], axis=-1).max() >= 8]
            first_inst = mk[0].any(0)
            f0_feat = dino.patches(work[0])
            # re-identification: mean DINOv2 feature of every first-frame instance (a candidate that looks like one of them
            # is that object relocated -- typically a fast faller the mask tracker lost -- not a new thing)
            inst_feat = []
            for k in range(mk.shape[1]):
                m0 = mk[0, k]
                if m0.sum() < 4:
                    inst_feat.append(None); continue
                yy, xx = np.nonzero(m0)
                v = f0_feat[np.clip((yy * 36 / H).astype(int), 0, 35), np.clip((xx * 64 / W).astype(int), 0, 63)].mean(0)
                inst_feat.append(v / max(np.linalg.norm(v), 1e-9))
            sample = [int(np.argmin(np.abs(tz[:F] - s))) for s in np.arange(STEP_S, tz[F - 1] + 1e-6, STEP_S)]
            sample = sorted(set(f for f in sample if f > 0))
            cands = []; trace = []
            for f in sample:
                if not np.isfinite(fit[f, 0]):
                    continue
                img2 = cv2.resize(work[f], (W * 2, H * 2))
                am = sam.automatic(img2, min_area_fraction=MIN_AREA, max_area_fraction=MAX_AREA)
                tracked = cv2.dilate(mk[f].any(0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                feat = dino.patches(work[f])
                for m2 in am:
                    m = cv2.resize(m2.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                    area = m.sum(); ys_, xs_ = np.nonzero(m)
                    tr = dict(f=int(f), area=float(area / (H * W)), tracked=float((m & tracked).sum() / max(area, 1)), box=[int(xs_.min()), int(ys_.min()), int(xs_.max()), int(ys_.max())])
                    trace.append(tr)
                    if area < MIN_AREA * H * W or (m & tracked).sum() >= 0.3 * area:
                        tr["drop"] = "tracked"; continue
                    ys, xs = np.nonzero(m)
                    q = to_first(np.stack([xs, ys], -1).astype(float), fit[f, 0], Rs[f], ts[f])
                    qi = np.round(q).astype(int); inside = (qi[:, 0] >= 0) & (qi[:, 0] < W) & (qi[:, 1] >= 0) & (qi[:, 1] < H)
                    tr["inside"] = float(inside.mean())
                    if inside.mean() < 0.7:
                        tr["drop"] = "revealed"; continue
                    qi_in = qi[inside]
                    tr["first_inst"] = float(first_inst[qi_in[:, 1], qi_in[:, 0]].mean())
                    if content[qi_in[:, 1], qi_in[:, 0]].mean() < 0.7 or first_inst[qi_in[:, 1], qi_in[:, 0]].mean() >= 0.6:
                        tr["drop"] = "disocclusion"; continue
                    # DINOv2: features under the mask at frame f vs the first frame at the mapped places (patch grid 14 px on 2x)
                    pf = np.clip((ys * 2 * 36 / (2 * H)).astype(int), 0, 35), np.clip((xs * 2 * 64 / (2 * W)).astype(int), 0, 63)
                    pq = np.clip((q[:, 1] * 36 / H).astype(int), 0, 35), np.clip((q[:, 0] * 64 / W).astype(int), 0, 63)
                    cos = float(np.mean(np.sum(feat[pf] * f0_feat[pq], -1)))
                    tr["cos"] = cos
                    if cos >= a.tau:
                        tr["drop"] = "same_look"; continue
                    tr["drop"] = "cand"
                    vc = feat[pf].mean(0); vc = vc / max(np.linalg.norm(vc), 1e-9)
                    sims = [float(vc @ v) if v is not None else -1.0 for v in inst_feat]
                    reid_k = int(np.argmax(sims)) if sims else -1; reid = float(max(sims)) if sims else -1.0
                    touch = False
                    if moving:
                        dmap = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
                        touch = any(mk[f, k].any() and float(dmap[mk[f, k]].min()) <= 3.0 for k in moving)
                    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    cands.append(dict(f=int(f), t=float(tz[f]), area=float(area / (H * W)), cos=cos, touch=bool(touch), box=box, reid=reid, reid_k=reid_k,
                                      reid_moving=bool(reid_k in moving),
                                      q=[float(v) for v in q.mean(0)], mask_rle=None, _m=m, _q=q))
            # persistence: another candidate at a different sampled time within 1 s, near in first-frame coordinates
            diag = float(np.hypot(H, W)); kept = []
            for c in cands:
                for o in cands:
                    if o is c or o["f"] == c["f"] or abs(o["t"] - c["t"]) > 1.0 + 1e-6:
                        continue
                    if np.hypot(*(np.array(o["q"]) - np.array(c["q"]))) <= 0.25 * diag and 0.25 <= o["area"] / c["area"] <= 4.0:
                        kept.append(c); break
            res = dict(set=a.set, arm=r["arm"], sid=r["sid"], n_sampled=len(sample), candidates=[])
            if a.debug:
                res["trace"] = trace
            for k, c in enumerate(kept[:12]):
                m = c["_m"]; f = c["f"]
                x0, y0, x1, y1 = c["box"]; pad = 12
                x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
                cur = cv2.cvtColor(work[f], cv2.COLOR_RGB2BGR).copy()
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(cur, cnts, -1, (40, 40, 255), 1)
                first = cv2.cvtColor(work[0], cv2.COLOR_RGB2BGR)
                strip = np.hstack([cv2.resize(first, (W, H)), cv2.resize(cur, (W, H))])
                cv2.rectangle(strip, (W + x0, y0), (W + x1, y1), (0, 255, 255), 1)
                cv2.imwrite(str(out / f"{name}_c{k}.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
                res["candidates"].append({kk: v for kk, v in c.items() if not kk.startswith("_") and kk != "mask_rle"})
            res["seconds"] = round(time.time() - t0, 1)
            json.dump(res, open(out / f"{name}.newobj.json", "w"))
            print(f"{name}: {len(cands)} raw, {len(kept)} kept, {res['seconds']} s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {name}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
