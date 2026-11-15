#!/usr/bin/env python
"""VBench-I2V scoring (VBench @ fd18b3d, vbench2_beta_i2v), run INSIDE the VBench environment with the VBench
checkout as cwd (its I2V loader reads vbench2_beta_i2v/data/crop/<ratio>/ relative to the cwd) and VBENCH_CACHE_DIR /
TORCH_HOME pointing at the pre-fetched weights, offline (launch/eval_score_vbench.sh).

  run      VBenchI2V(...).evaluate(videos_path, name=<dim>, dimension_list=[dim], local=True, read_frame=False,
           resolution=<ratio>, mode='vbench_standard', imaging_quality_preprocessing_mode='longer') once per dimension
           (= evaluate_i2v.py's call, one dimension per call so dimensions can be spread over processes/GPUs).
           Writes <out>/<dim>_eval_results.json ({dim: [mean, per-video list]}) and <dim>_full_info.json.
           A finished dimension is skipped unless --overwrite.
  summary  per-dimension means and the leaderboard aggregates computed exactly like scripts/cal_i2v_final_score.py
           (min-max normalisation, DIM_WEIGHT_I2V, I2V_QUALITY_LIST / I2V_LIST of scripts/constant.py; a missing
           dimension counts 0 like the official script) + comparison with a reference row. Writes <out>/summary.json.

usage: vbench_score_br.py run --videos V --out O --dimensions d1 [d2 ...] --ratio 16-9
       vbench_score_br.py shards --videos V --root SHARDS --sizes camera_motion=240,motion_smoothness=160,...
       vbench_score_br.py merge --units UNITS --out O          (per-video lists of all shards -> one result per dim)
       vbench_score_br.py summary --videos V --out O --model M [--ref reference.json]
"""
import argparse
import collections
import json
import os
import re
import sys
import time
from pathlib import Path

DIMS = ["camera_motion", "i2v_subject", "i2v_background", "subject_consistency", "background_consistency", "motion_smoothness",
        "dynamic_degree", "aesthetic_quality", "imaging_quality"]
KEY = {"camera_motion": "Video-Text Camera Motion", "i2v_subject": "Video-Image Subject Consistency",
       "i2v_background": "Video-Image Background Consistency", "subject_consistency": "Subject Consistency",
       "background_consistency": "Background Consistency", "motion_smoothness": "Motion Smoothness",
       "dynamic_degree": "Dynamic Degree", "aesthetic_quality": "Aesthetic Quality", "imaging_quality": "Imaging Quality"}


def per_video(dim, result):
    """per-video list of a <dim>_eval_results.json entry: camera_motion returns (mean, per-type dict, videos)"""
    return result[2] if dim == "camera_motion" else result[1]


def dim_mean(dim, items):
    """the dimension score from per-video results, exactly as the VBench compute_<dim> functions aggregate"""
    if dim == "background_consistency":  # frame-pair weighted: sum(video_sim) / sum(cnt_per_video)
        return sum(d["video_sim"] for d in items) / sum(d["cnt_per_video"] for d in items)
    m = sum(float(d["video_results"]) for d in items) / len(items)  # per-video mean (subject_consistency: frames equal)
    return m / 100. if dim == "imaging_quality" else m  # MUSIQ per-video values are 0-100, VBench divides the mean by 100


def shards(a):
    """symlink shard folders: <root>/<dim>/<s>/ holding the videos of that dimension (sizes in --sizes dim=n)"""
    full = json.loads(Path(a.full_info).read_text())
    have = set(os.listdir(a.videos))
    sizes = dict(x.split("=") for x in a.sizes.split(",") if x)
    root = Path(a.root); units = []
    for d in DIMS:
        vids = [f"{e['prompt_en']}-{i}.mp4" for e in full if d in e["dimension"] for i in range(5)]
        missing = [v for v in vids if v not in have]
        if missing:
            print(f"WARNING {d}: {len(missing)} expected videos missing, e.g. {missing[:2]}")
        vids = [v for v in vids if v in have]
        n = int(sizes.get(d, 10 ** 9)); k = 0
        for s0 in range(0, len(vids), n):
            sd = root / d / f"{k:02d}"; sd.mkdir(parents=True, exist_ok=True)
            for v in vids[s0:s0 + n]:
                if not (sd / v).exists():
                    os.symlink(os.path.join(os.path.abspath(a.videos), v), sd / v)
            units.append(f"{d} {k:02d} {len(vids[s0:s0 + n])}"); k += 1
    (root / "units.txt").write_text("\n".join(units) + "\n")
    print(f"{len(units)} units -> {root}/units.txt")


def merge(a):
    """unit results <units...>/<dim>_<s>/<dim>_eval_results.json -> <out>/<dim>_eval_results.json (VBench layout)"""
    full = json.loads(Path(a.full_info).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {}
    for d in DIMS:
        files = sorted(f for u in a.units for f in Path(u).glob(f"{d}_[0-9][0-9]/{d}_eval_results.json"))
        if not files:
            continue
        items, unit_check = [], []
        for f in files:
            res = json.loads(f.read_text())[d]
            pv = per_video(d, res)
            unit_check.append(abs(dim_mean(d, pv) - res[0]))
            items += pv
        names = [os.path.basename(x["video_path"]) for x in items]
        expected = 5 * sum(d in e["dimension"] for e in full)
        mean = dim_mean(d, items)
        res = [mean, items]
        if d == "camera_motion":
            by = collections.defaultdict(list)
            for x in items:
                by[x["prompt_type"]].append(x["video_results"])
            res = [mean, {k: sum(v) / len(v) for k, v in by.items()}, items]
        (out / f"{d}_eval_results.json").write_text(json.dumps({d: res}, indent=1))
        report[d] = {"units": len(files), "videos": len(items), "unique": len(set(names)), "expected": expected,
                     "max_unit_recompute_err": max(unit_check), "mean": mean}
        print(f"MERGE {d}: {len(files)} units, {len(items)} videos ({len(set(names))} unique, expected {expected}), "
              f"unit-mean recompute err {max(unit_check):.2e}, score {mean:.6f}")
    (out / "merge_report.json").write_text(json.dumps(report, indent=1))


def run(a):
    import torch
    from vbench2_beta_i2v import VBenchI2V
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = os.listdir(a.videos)
    bad = [n for n in names if not n.endswith(".mp4")]
    if not names or bad:
        raise SystemExit(f"{a.videos}: empty or holds non-mp4 files {bad[:3]}")
    if not Path("vbench2_beta_i2v/data/crop", a.ratio).is_dir():
        raise SystemExit(f"run from the VBench checkout: vbench2_beta_i2v/data/crop/{a.ratio} not found under {os.getcwd()}")
    bench = VBenchI2V(torch.device("cuda"), a.full_info, str(out))
    for dim in a.dimensions:
        target = out / f"{dim}_eval_results.json"
        if target.is_file() and not a.overwrite:
            print(f"[skip] {dim}: {target} exists", flush=True); continue
        t0 = time.time()
        bench.evaluate(videos_path=a.videos, name=dim, dimension_list=[dim], local=True, read_frame=False, resolution=a.ratio,
                       mode="vbench_standard", imaging_quality_preprocessing_mode="longer")
        result = json.loads(target.read_text())[dim]
        print(f"VBENCH_DIM {dim} {result[0]:.6f} n={len(per_video(dim, result))} in {time.time() - t0:.0f}s", flush=True)


def summary(a):
    sys.path.insert(0, str(Path(a.full_info).parents[1] / "scripts"))
    import constant as C  # VBench scripts/constant.py
    out = Path(a.out)
    full = json.loads(Path(a.full_info).read_text())
    expected = {d: 5 * sum(d in e["dimension"] for e in full) for d in DIMS}
    videos = sorted(p.name for p in Path(a.videos).glob("*.mp4") if not p.name.startswith("."))
    per_prompt = collections.Counter(re.sub(r"-\d+\.mp4$", "", v) for v in videos)
    raw, dims = {}, {}
    for d in DIMS:
        path = out / f"{d}_eval_results.json"
        if not path.is_file():
            continue
        res = json.loads(path.read_text())[d]
        mean, items = res[0], per_video(d, res)
        raw[KEY[d]] = mean
        dims[d] = {"mean": mean, "videos_scored": len(items) if isinstance(items, list) else None, "videos_expected": expected[d]}
    upload = {k: raw.get(k, 0) for k in C.TASK_INFO_I2V}  # official: missing dimension -> 0
    norm = {k: (upload[k] - C.NORMALIZE_DIC_I2V[k]["Min"]) / (C.NORMALIZE_DIC_I2V[k]["Max"] - C.NORMALIZE_DIC_I2V[k]["Min"]) * C.DIM_WEIGHT_I2V[k]
            for k in C.TASK_INFO_I2V}
    quality = sum(norm[k] for k in C.I2V_QUALITY_LIST) / sum(C.DIM_WEIGHT_I2V[k] for k in C.I2V_QUALITY_LIST)
    i2v = sum(norm[k] for k in C.I2V_LIST) / sum(C.DIM_WEIGHT_I2V[k] for k in C.I2V_LIST)
    total = (quality * C.I2V_QUALITY_WEIGHT + i2v * C.I2V_WEIGHT) / (C.I2V_QUALITY_WEIGHT + C.I2V_WEIGHT)
    rec = {"model": a.model, "videos_dir": a.videos, "videos": len(videos), "prompts": len(per_prompt),
           "videos_per_prompt": dict(collections.Counter(per_prompt.values())), "ratio": a.ratio, "dimensions": dims,
           "total_score": total, "i2v_score": i2v, "quality_score": quality, "all_9_dims_present": len(raw) == 9,
           "note": "aggregation = VBench scripts/cal_i2v_final_score.py (missing dims count as 0)"}
    if a.ref:
        ref = json.loads(Path(a.ref).read_text())
        rec["reference"] = ref
        rec["delta_pts"] = {d: round(100 * dims[d]["mean"] - ref[d], 2) for d in dims if d in ref}
        for k, v in (("total_score", total), ("i2v_score", i2v), ("quality_score", quality)):
            if k in ref:
                rec["delta_pts"][k] = round(100 * v - ref[k], 2)
    (out / "summary.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{a.model}: {len(videos)} videos / {len(per_prompt)} prompts | " + " ".join(f"{d} {100 * v['mean']:.2f}" for d, v in dims.items())
          + f" | Total {100 * total:.2f} I2V {100 * i2v:.2f} Quality {100 * quality:.2f}" + ("" if len(raw) == 9 else f" (only {len(raw)}/9 dims)"))
    if a.ref:
        print("delta vs ref (pts):", json.dumps(rec["delta_pts"]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run"); s = sub.add_parser("summary"); sh = sub.add_parser("shards"); m = sub.add_parser("merge")
    for q in (sh, m):
        q.add_argument("--full-info", default="vbench2_beta_i2v/vbench2_i2v_full_info.json")
    sh.add_argument("--videos", required=True); sh.add_argument("--root", required=True); sh.add_argument("--sizes", default="")
    m.add_argument("--units", nargs="+", required=True); m.add_argument("--out", required=True)
    for q in (r, s):
        q.add_argument("--videos", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--ratio", default="16-9")
        q.add_argument("--full-info", default="vbench2_beta_i2v/vbench2_i2v_full_info.json")
    r.add_argument("--dimensions", nargs="+", required=True, choices=DIMS)
    r.add_argument("--overwrite", action="store_true")
    s.add_argument("--model", required=True)
    s.add_argument("--ref", default="")
    a = p.parse_args()
    a.full_info = str(Path(a.full_info).resolve())
    {"run": run, "summary": summary, "shards": shards, "merge": merge}[a.command](a)


if __name__ == "__main__":
    main()
