#!/usr/bin/env python3
"""Paired comparison A - B (default: MoPlan - the untrained Wan2.2 under the same protocol) with bootstrap 95% CIs.
  morpheus   per video of each model's best prompt type (the one its Total is reported for), paired by (experiment, seed index):
             Total_v = (Dyn + PhysInv) / 2 with discarded videos -> 0 (morph_aggregate_br.py rule); also PhysInv and Dyn.  CIs over
             videos and over the 16 experiments (cluster bootstrap)
  pisabench  per clip (361), paired by clip id: l2, chamfer_distance (lower = better), iou (higher = better)
  vbench     per dimension: per-video values paired by video file name (<prompt>-<k>.mp4); Total / I2V / Quality: bootstrap over
             the suite's prompts, each replicate re-aggregated exactly like VBench scripts/cal_i2v_final_score.py (dimension
             means as the compute_<dim> functions: background_consistency frame-pair weighted, imaging_quality / 100; min-max
             normalisation with DIM_WEIGHT_I2V).  --phase s0 (seed 0, provisional) or all (5 videos)
Model labels are the launchers' labels (moplan, base, zero, ...): Morpheus <root>/morpheus/results/single_frame_conditioning/
MP-<label>, PisaBench <root>/pisabench/official/<label>, VBench <root>/vbench/results/<label>[-s0].
usage: osp_paired.py morpheus|pisabench|vbench --root EVAL_ROOT [--a LABEL] [--b LABEL] [--n 10000] [--phase s0|all]
       [--vbench-code VBENCH_HOME]   (python3 + numpy)"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(".")                                                # set from --root in main()
VBCODE = Path(os.environ.get("VBENCH_HOME", "."))              # VBench checkout (scripts/constant.py); --vbench-code
VDIMS = ["camera_motion", "i2v_subject", "i2v_background", "subject_consistency", "background_consistency", "motion_smoothness",
         "dynamic_degree", "aesthetic_quality", "imaging_quality"]
VKEY = {"camera_motion": "Video-Text Camera Motion", "i2v_subject": "Video-Image Subject Consistency",
        "i2v_background": "Video-Image Background Consistency", "subject_consistency": "Subject Consistency",
        "background_consistency": "Background Consistency", "motion_smoothness": "Motion Smoothness",
        "dynamic_degree": "Dynamic Degree", "aesthetic_quality": "Aesthetic Quality", "imaging_quality": "Imaging Quality"}


def boot(diff: np.ndarray, n: int, rng, clusters=None) -> tuple[float, float]:
    if clusters is None:
        idx = rng.integers(0, len(diff), (n, len(diff)))
        means = diff[idx].mean(1)
    else:
        keys = sorted(set(clusters)); groups = [np.where(np.array(clusters) == k)[0] for k in keys]
        means = np.empty(n)
        for i in range(n):
            pick = rng.integers(0, len(groups), len(groups))
            means[i] = diff[np.concatenate([groups[j] for j in pick])].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def morph_rows(model: str) -> tuple[str, dict]:
    d = ROOT / "morpheus/results/single_frame_conditioning" / model
    agg = json.loads((d / "aggregate.json").read_text())[model]
    best = agg["best_prompt_by_total"]
    rows = {}
    for line in (d / "per_video.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["pt"] != best:
            continue
        seed = int(Path(r["video"]).name)
        z = 0.0 if r["disc"] else None
        phys = z if r["disc"] else (r["phys"] or 0.0)
        dyn = z if r["disc"] else (r["dyn"] or 0.0)
        rows[(r["exp"], seed)] = {"total": (phys + dyn) / 2, "physinv": phys, "dyn": dyn}
    return best, rows


def pisa_rows(model: str) -> dict:
    rows = {}
    for line in (ROOT / "pisabench/official" / model / "s0/results/per_video.jsonl").read_text().splitlines():
        r = json.loads(line)
        cid = re.search(r"/real/([^/]+)/", r["config"]["pd_mask"]).group(1)
        rows[cid] = {m: r[m] for m in ("l2", "chamfer_distance", "iou")}
    return rows


def vb_items(model: str, phase: str) -> dict:
    """dim -> {video file name: (value, weight)}; dim mean = sum(value * weight) / sum(weight) (= the compute_<dim> rule)"""
    d = ROOT / "vbench/results" / (model + ("-s0" if phase == "s0" else ""))
    out = {}
    for dim in VDIMS:
        res = json.loads((d / f"{dim}_eval_results.json").read_text())[dim]
        items = res[2] if dim == "camera_motion" else res[1]
        if dim == "background_consistency":
            out[dim] = {Path(x["video_path"]).name: (x["video_sim"] / x["cnt_per_video"], x["cnt_per_video"]) for x in items}
        else:
            scale = 0.01 if dim == "imaging_quality" else 1.0
            out[dim] = {Path(x["video_path"]).name: (float(x["video_results"]) * scale, 1.0) for x in items}
    return out


def vbench(a, rng) -> dict:
    import sys
    sys.path.insert(0, str(VBCODE / "scripts"))
    import constant as C  # VBench scripts/constant.py
    A, B = vb_items(a.a, a.phase), vb_items(a.b, a.phase)
    prompt_of = lambda name: re.sub(r"-\d+\.mp4$", "", name)
    prompts = sorted({prompt_of(v) for dim in VDIMS for v in A[dim]})
    pid = {p: i for i, p in enumerate(prompts)}
    arr = {}
    for dim in VDIMS:
        names = sorted(set(A[dim]) & set(B[dim]))
        arr[dim] = (np.array([pid[prompt_of(n)] for n in names]), np.array([A[dim][n][0] for n in names]),
                    np.array([B[dim][n][0] for n in names]), np.array([A[dim][n][1] for n in names]))

    def agg(means: dict) -> tuple[float, float, float]:
        up = {k: means.get(d, 0.0) for d, k in VKEY.items()}
        up = {k: up.get(k, 0.0) for k in C.TASK_INFO_I2V}
        norm = {k: (up[k] - C.NORMALIZE_DIC_I2V[k]["Min"]) / (C.NORMALIZE_DIC_I2V[k]["Max"] - C.NORMALIZE_DIC_I2V[k]["Min"]) * C.DIM_WEIGHT_I2V[k]
                for k in C.TASK_INFO_I2V}
        q = sum(norm[k] for k in C.I2V_QUALITY_LIST) / sum(C.DIM_WEIGHT_I2V[k] for k in C.I2V_QUALITY_LIST)
        i = sum(norm[k] for k in C.I2V_LIST) / sum(C.DIM_WEIGHT_I2V[k] for k in C.I2V_LIST)
        return (q * C.I2V_QUALITY_WEIGHT + i * C.I2V_WEIGHT) / (C.I2V_QUALITY_WEIGHT + C.I2V_WEIGHT), i, q

    def means(which: int, pw: np.ndarray | None) -> dict:
        m = {}
        for dim, (p, va, vb, w) in arr.items():
            ww = w if pw is None else w * pw[p]
            m[dim] = float(np.sum((va if which == 0 else vb) * ww) / np.sum(ww))
        return m

    ma, mb = means(0, None), means(1, None)
    ta, tb = agg({d: ma[d] for d in VDIMS}), agg({d: mb[d] for d in VDIMS})
    reps = {"total": [], "i2v": [], "quality": []}; dreps = {d: [] for d in VDIMS}
    for _ in range(a.n):
        pw = np.bincount(rng.integers(0, len(prompts), len(prompts)), minlength=len(prompts)).astype(float)
        xa, xb = means(0, pw), means(1, pw)
        ra, rb = agg(xa), agg(xb)
        for j, k in enumerate(("total", "i2v", "quality")):
            reps[k].append(ra[j] - rb[j])
        for d in VDIMS:
            dreps[d].append(xa[d] - xb[d])
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    out = {"phase": a.phase, "n_prompts": len(prompts), "metrics": {}}
    for j, k in enumerate(("total", "i2v", "quality")):
        out["metrics"][k] = {"a": ta[j], "b": tb[j], "diff": ta[j] - tb[j], "ci_prompts": ci(reps[k]), "higher_is_better": True}
    for d in VDIMS:
        out["metrics"][d] = {"a": ma[d], "b": mb[d], "diff": ma[d] - mb[d], "n": len(arr[d][0]), "ci_prompts": ci(dreps[d]), "higher_is_better": True}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bench", choices=("morpheus", "pisabench", "vbench"))
    p.add_argument("--phase", choices=("s0", "all"), default="s0")
    p.add_argument("--root", type=Path, required=True, help="evaluation root ($EVAL_ROOT)")
    p.add_argument("--vbench-code", type=Path, default=None, help="VBench checkout (default: $VBENCH_HOME)")
    p.add_argument("--a", default="moplan"); p.add_argument("--b", default="base"); p.add_argument("--n", type=int, default=10000)
    a = p.parse_args()
    global ROOT, VBCODE
    ROOT = a.root
    if a.vbench_code is not None:
        VBCODE = a.vbench_code
    rng = np.random.default_rng(0)
    out = {"bench": a.bench, "a": a.a, "b": a.b, "metrics": {}}
    if a.bench == "vbench":
        out.update(vbench(a, rng))
    elif a.bench == "morpheus":
        (pa, ra), (pb, rb) = morph_rows(f"MP-{a.a}"), morph_rows(f"MP-{a.b}")
        keys = sorted(set(ra) & set(rb)); out.update(prompt_a=pa, prompt_b=pb, n=len(keys))
        for m in ("total", "physinv", "dyn"):
            x, y = np.array([ra[k][m] for k in keys]), np.array([rb[k][m] for k in keys]); diff = x - y
            out["metrics"][m] = {"a": x.mean(), "b": y.mean(), "diff": diff.mean(), "ci_videos": boot(diff, a.n, rng),
                                 "ci_experiments": boot(diff, a.n, rng, [k[0] for k in keys]), "higher_is_better": True}
    else:
        ra, rb = pisa_rows(a.a), pisa_rows(a.b)
        keys = sorted(set(ra) & set(rb)); out["n"] = len(keys)
        for m, hib in (("l2", False), ("chamfer_distance", False), ("iou", True)):
            ok = [k for k in keys if ra[k][m] is not None and rb[k][m] is not None and not np.isnan(ra[k][m]) and not np.isnan(rb[k][m])]
            x, y = np.array([ra[k][m] for k in ok]), np.array([rb[k][m] for k in ok]); diff = x - y
            out["metrics"][m] = {"a": x.mean(), "b": y.mean(), "diff": diff.mean(), "n": len(ok), "ci_videos": boot(diff, a.n, rng), "higher_is_better": hib}
    for m, v in out["metrics"].items():
        lo, hi = v.get("ci_videos") or v["ci_prompts"]; sig = "*" if lo > 0 or hi < 0 else ""
        extra = f"; experiment-cluster CI [{v['ci_experiments'][0]:+.4f}, {v['ci_experiments'][1]:+.4f}]" if "ci_experiments" in v else ""
        print(f"{a.bench} {m}: A={v['a']:.4f} B={v['b']:.4f} A-B={v['diff']:+.4f} CI [{lo:+.4f}, {hi:+.4f}]{sig}{extra}")
    path = ROOT / f"paired/paired_{a.bench}{'_' + a.phase if a.bench == 'vbench' else ''}_{a.a}_vs_{a.b}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(path)


if __name__ == "__main__":
    main()
