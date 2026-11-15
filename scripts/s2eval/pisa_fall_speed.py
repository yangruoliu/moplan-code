#!/usr/bin/env python3
"""Fall-speed (effective gravity) analysis on the official PisaBench real-world masks: do generated objects fall as fast
as the real ones?  Reuses the SAM 2.1 masks of the official PisaBench scoring (<root>/pisabench/official/<model>/s0/eval.jsonl);
no GPU, no new generation.  Definitions fixed before looking at any result:
  dropping object  the mask channel whose positive click is highest (min y), as pisa_official_export.py / the official eval
  centroid y(t)    of that mask per frame; GT at 120 fps (t = i/120); generated clips were exported at 16 fps by nearest-frame
                   re-timing of our 7 fps frames (source frame floor(k*7/16 + 0.5)), so each source frame j is read once at
                   t = j/7 (its first 16-fps copy)
  D                real drop = max_t (y_gt(t) - y_gt(0)) in pixels (same 256x256 geometry for GT and generated clips);
                   clips with D < 10 px are skipped (counted)
  t_half           first time y(t) - y(0) >= D/2, linear interpolation between frames; generated clips that never get there
                   are "not fallen" (censored at the clip end + one source frame, i.e. their g_rel is an upper bound)
  g_rel            (t_half_gt / t_half_gen)^2 = generated / real effective gravity (constant acceleration from rest)
  |log2 g_rel|     deviation from real gravity (0 = real speed; 1 = 2x too slow or too fast); lower is better
Paired bootstrap over clips (10,000 replicates, seed 20261024) of model a - model b for: mean |log2 g_rel| (lower better),
fraction fallen (higher better), median g_rel (reported, not paired-tested), median log2 g_rel.
Arms: --arm KEY=MODEL (repeatable, in this order; KEY "ours" is the reference arm every other arm is compared with), default
ours=moplan base=base; arms whose eval.jsonl does not exist are skipped.  All paired bootstraps draw from ONE generator in
arm order, so the CIs of an arm depend on the arms listed before it: ours vs base (listed first) is always the same, an
extra arm's CIs repeat a run only with the same preceding arms in the same order; point estimates do not depend on it.
usage: pisa_fall_speed.py --root EVAL_ROOT [--arm ours=moplan --arm base=base ...] [--workers 8]
       -> <root>/paired/pisa_fall_speed.json"""
import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ARMS = {"ours": "moplan", "base": "base"}                     # replaced by --arm KEY=MODEL
GT_FPS, SRC_FPS, OUT_FPS, MIN_DROP = 120.0, 7.0, 16.0, 10.0


def centroid_y(masks: np.ndarray) -> np.ndarray:
    """masks [T, H, W] bool -> centroid row per frame (nan where empty)."""
    rows = np.arange(masks.shape[1], dtype=np.float64)[None, :, None]
    area = masks.sum(axis=(1, 2)).astype(np.float64)
    ys = (masks * rows).sum(axis=(1, 2)) / np.where(area > 0, area, np.nan)
    return ys


def crossing(t: np.ndarray, d: np.ndarray, level: float):
    """first time d(t) >= level, linear interpolation; None if never (nan frames are skipped)."""
    ok = ~np.isnan(d)
    t, d = t[ok], d[ok]
    hit = np.nonzero(d >= level)[0]
    if len(hit) == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return float(t[0])
    w = (level - d[i - 1]) / (d[i] - d[i - 1]) if d[i] != d[i - 1] else 1.0
    return float(t[i - 1] + w * (t[i] - t[i - 1]))


def drop_index(annotation: str) -> int:
    pts = json.loads(Path(annotation).read_text())["points"]
    return int(np.argmin([p["positive"][0][1] for p in pts]))


def gt_job(row: dict):
    k = drop_index(row["gt_annotation"])
    arr = np.load(row["gt_mask"])["mask"]
    m = arr if arr.ndim == 3 else arr[:, k]   # 16 GT clips store a single-object [T, H, W] array
    y = centroid_y(m)
    t = np.arange(len(y)) / GT_FPS
    d = y - y[0] if not np.isnan(y[0]) else y - np.nanmin(y[:5])
    D = float(np.nanmax(d)) if np.any(~np.isnan(d)) else float("nan")
    th = crossing(t, d, D / 2) if D >= MIN_DROP else None
    return row["sample_id"], {"k": k, "D": D, "t_half": th, "gt_seconds": float(t[-1])}


def gen_job(args):
    arm, row, D, k = args
    m = np.load(row["pd_mask"])["mask"]
    m = m[:, None] if m.ndim == 3 else m
    n_out = m.shape[0]
    src = [int(math.floor(j * SRC_FPS / OUT_FPS + 0.5)) for j in range(n_out)]
    first = {}
    for j, s in enumerate(src):
        first.setdefault(s, j)
    order = sorted(first)
    y = centroid_y(m[[first[s] for s in order], k])
    t = np.array(order, dtype=np.float64) / SRC_FPS
    d = y - y[0] if not np.isnan(y[0]) else y - np.nanmin(y[:2])
    th = crossing(t, d, D / 2)
    return arm, row["sample_id"], {"t_half": th, "end": float(t[-1]), "max_drop": float(np.nanmax(d)) if np.any(~np.isnan(d)) else None}


def boot(a: np.ndarray, b: np.ndarray, rng, n: int):
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    bm = d[idx].mean(1)
    return float(d.mean()), [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N clips only, result to <root>/paired/pisa_fall_speed_smoke.json")
    p.add_argument("--root", type=Path, required=True, help="evaluation root ($EVAL_ROOT)")
    p.add_argument("--arm", action="append", default=[], help="KEY=MODEL (MODEL = folder under <root>/pisabench/official)")
    a = p.parse_args()
    if a.arm:
        ARMS.clear()
        ARMS.update(dict(x.split("=", 1) for x in a.arm))
    ROOT = a.root / "pisabench/official"
    rows = {arm: {r["sample_id"]: r for r in map(json.loads, (ROOT / m / "s0/eval.jsonl").read_text().splitlines())}
            for arm, m in ARMS.items() if (ROOT / m / "s0/eval.jsonl").is_file()}
    ids = sorted(set.intersection(*(set(v) for v in rows.values())))
    ids = ids[:a.limit] if a.limit else ids
    ref = rows["ours"]
    with Pool(a.workers) as pool:
        gt = dict(pool.map(gt_job, [ref[i] for i in ids], chunksize=4))
        use = [i for i in ids if gt[i]["t_half"] is not None]
        jobs = [(arm, rows[arm][i], gt[i]["D"], gt[i]["k"]) for arm in rows for i in use]
        gen = {}
        for arm, sid, rec in pool.imap_unordered(gen_job, jobs, chunksize=4):
            gen.setdefault(arm, {})[sid] = rec
    out = {"description": __doc__.split("usage:")[0].strip(), "arms": {k: ARMS[k] for k in rows}, "n_clips": len(ids),
           "n_used": len(use), "n_skipped_small_drop": len(ids) - len(use),
           "gt": {"median_D_px": float(np.median([gt[i]["D"] for i in use])), "median_t_half_s": float(np.median([gt[i]["t_half"] for i in use]))},
           "per_arm": {}, "paired": {}, "per_clip": {i: {"gt": gt[i], **{arm: gen[arm][i] for arm in gen}} for i in use}}
    metrics = {}
    for arm in gen:
        th = np.array([gen[arm][i]["t_half"] if gen[arm][i]["t_half"] is not None else gen[arm][i]["end"] + 1 / SRC_FPS for i in use])
        fallen = np.array([gen[arm][i]["t_half"] is not None for i in use], dtype=float)
        th = np.maximum(th, 1e-3)
        lg = 2 * np.log2(np.array([gt[i]["t_half"] for i in use]) / th)
        metrics[arm] = {"abs_log2": np.abs(lg), "fallen": fallen, "log2": lg}
        f = fallen.astype(bool)
        out["per_arm"][arm] = {"fallen_frac": float(fallen.mean()), "median_g_rel_all": float(2 ** np.median(lg)),
                               "median_g_rel_fallen": float(2 ** np.median(lg[f])) if f.any() else None,
                               "mean_abs_log2_g_rel": float(np.abs(lg).mean()),
                               "frac_within_2x": float(((np.abs(lg) <= 1) & f).mean()),
                               "median_t_half_fallen_s": float(np.median(th[f])) if f.any() else None}
    rng = np.random.default_rng(20261024)
    for b in [x for x in metrics if x != "ours"]:
        res = {}
        for key, higher in (("abs_log2", False), ("fallen", True)):
            diff, ci = boot(metrics["ours"][key], metrics[b][key], rng, a.n)
            res[key] = {"a": float(metrics["ours"][key].mean()), "b": float(metrics[b][key].mean()), "diff": diff, "ci": ci,
                        "favourable": bool((diff > 0) == higher) if diff != 0 else None, "significant": bool(ci[0] > 0 or ci[1] < 0)}
        within = lambda arm: ((metrics[arm]["abs_log2"] <= 1) & (metrics[arm]["fallen"] > 0)).astype(float)
        diff, ci = boot(within("ours"), within(b), rng, a.n)
        res["within_2x"] = {"a": float(within("ours").mean()), "b": float(within(b).mean()), "diff": diff, "ci": ci,
                            "favourable": bool(diff > 0) if diff != 0 else None, "significant": bool(ci[0] > 0 or ci[1] < 0)}
        out["paired"][f"ours_vs_{b}"] = res
    path = a.root / "paired" / ("pisa_fall_speed_smoke.json" if a.limit else "pisa_fall_speed.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n_clips", "n_used", "n_skipped_small_drop", "gt", "per_arm", "paired")}, indent=1))
    print(path)


if __name__ == "__main__":
    main()
