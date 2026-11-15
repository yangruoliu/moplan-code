#!/usr/bin/env python3
"""Collect the official numbers of one launch/eval_piq_score.sh invocation into a compact JSON.

Called by launch/eval_piq_score.sh (run with the scoring venv's python, NOT with -I, because it imports physiq for the
supplementary IQTable numbers of the ec2c611 CSV).

Official numbers per mode:
  verified      c2bca8a default mode: metrics JSON written by the scorer (final_score_view = verified score,
                score_*_view sub-scores, final_score_orig, final_score_origround) + the two printed lines.
  orig_new      c2bca8a --original_physics_iq: same JSON; the official original Physics-IQ score is
                final_score_origround (README "Physics-IQ Original Workflow" F).
  orig_ec2c611  old scorer: the printed line "Physics-IQ score for <name>: <x>" (calculate_iq_score()[0]).

Provenance is kept free of machine details: no host name, folders by their last component only, and the result files
by their path relative to the folder of --out (scripts/s2eval_verified/agg_runs_v.py resolves them from there).
"""
import argparse
import json
import os
import re
import subprocess
import sys


def git_info(path):
    def g(*a):
        return subprocess.run(["git", "-C", path, *a], capture_output=True, text=True).stdout.strip()
    return {"path": name_of(path), "commit": g("rev-parse", "HEAD"), "dirty_files": g("status", "--porcelain").splitlines()}


def name_of(path):
    """Last component of a folder or file path (no absolute paths in the summary)."""
    return os.path.basename(os.path.normpath(path)) if path else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--precheck", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--descriptions", required=True)
    ap.add_argument("--bench_data", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--rc", type=int, required=True)
    ap.add_argument("--t_start", required=True)
    ap.add_argument("--t_end", required=True)
    ap.add_argument("--wall_s", type=float, required=True)
    ap.add_argument("--nproc", required=True)
    ap.add_argument("--cpus", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt_source", default="precomputed")
    a = ap.parse_args()

    log = open(a.log, errors="replace").read()
    pre = json.load(open(a.precheck))
    name = re.escape(a.name)
    out_dir = os.path.dirname(os.path.abspath(a.out))

    def rel(path):  # result files relative to the folder of the summary JSON
        return os.path.relpath(os.path.abspath(path), out_dir)

    out = {
        "name": a.name, "mode": a.mode, "status": "ok" if a.rc == 0 else f"scorer_exit_{a.rc}",
        "run_dir": name_of(a.run_dir), "work_dir": name_of(a.work), "results_dir": rel(a.results_dir),
        "scorer_code": git_info(a.code), "descriptions_file": name_of(a.descriptions),
        "benchmark_data": name_of(a.bench_data),
        "gt_source": a.gt_source,
        "input": {k: pre.get(k) for k in ["n_files", "fps_cv2", "r_frame_rate", "resolutions", "cv_frame_counts",
                                          "duration_min", "duration_max", "codecs", "pix_fmts"]},
        "timing": {"start": a.t_start, "end": a.t_end, "wall_s": round(a.wall_s, 1),
                   "n_process": a.nproc if a.mode != "orig_ec2c611" else "serial (ec2c611 has no --n_process)",
                   "taskset_cpus": a.cpus},
        "env": {"python": sys.version.split()[0]},
    }
    try:
        import cv2, numpy, pandas
        out["env"].update(cv2=cv2.__version__, numpy=numpy.__version__, pandas=pandas.__version__)
    except Exception as e:  # pragma: no cover
        out["env"]["import_error"] = repr(e)
    try:
        ff = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True).stdout.splitlines()
    except OSError:                                              # no ffprobe on PATH, or one that does not start
        ff = []
    out["env"]["ffprobe"] = ff[0] if ff else None

    printed = {}
    if a.mode in ("verified", "orig_new"):
        m1 = re.findall(rf"Physics-IQ score \(original\) for {name}: ([0-9.]+)", log)
        m2 = re.findall(rf"Physics-IQ score \(verified\) for {name}: ([0-9.]+)", log)
        printed = {"printed_original_x100": float(m1[-1]) if m1 else None,
                   "printed_verified_x100": float(m2[-1]) if m2 else None}
        bench = "physics-IQ-benchmark-verified" if a.mode == "verified" else "physics-IQ-benchmark"
        mpath = os.path.join(a.results_dir, f"{a.name}_metrics.json")
        cpath = os.path.join(a.results_dir, f"{a.name}.csv")
        metrics = json.load(open(mpath)) if os.path.exists(mpath) else None
        out["files"] = {"csv": rel(cpath) if os.path.exists(cpath) else None, "metrics_json": rel(mpath) if metrics else None}
        out["metrics_json"] = metrics
        if metrics:
            if a.mode == "verified":
                out["official"] = {
                    "benchmark": "Physics-IQ Verified (c2bca8a default mode)",
                    "verified_score": metrics["final_score_view"] * 100,
                    "final_score_view": metrics["final_score_view"],
                    "score_spatiotemporal_view": metrics["score_spatiotemporal_view"],
                    "score_spatial_view": metrics["score_spatial_view"],
                    "score_weighted_spatial_view": metrics["score_weighted_spatial_view"],
                    "score_mse_view": metrics["score_mse_view"],
                    "final_score_orig": metrics["final_score_orig"],
                    "final_score_origround": metrics["final_score_origround"],
                    **printed,
                }
            else:
                out["official"] = {
                    "benchmark": "Physics-IQ original (c2bca8a --original_physics_iq)",
                    "original_physics_iq_score": metrics["final_score_origround"],
                    "final_score_origround": metrics["final_score_origround"],
                    "final_score_orig": metrics["final_score_orig"],
                    "score_spatiotemporal": metrics["score_spatiotemporal"],
                    "score_spatial": metrics["score_spatial"],
                    "score_weighted_spatial": metrics["score_weighted_spatial"],
                    "score_mse": metrics["score_mse"],
                    "final_score_view_on_original_data": metrics["final_score_view"],
                    **printed,
                }
    else:
        m = re.findall(rf"Physics-IQ score for {name}: ([0-9.]+)", log)
        cpath = os.path.join(a.results_dir, f"{a.name}.csv")
        out["files"] = {"csv": rel(cpath) if os.path.exists(cpath) else None}
        if m:
            out["official"] = {"benchmark": "Physics-IQ original (ec2c611 code/run_physics_iq.py)",
                               "original_physics_iq_score": float(m[-1]), "printed": float(m[-1])}
        if os.path.exists(cpath):
            try:  # supplementary only: newer IQTable decomposition of the same CSV
                from physiq.calculate_iq_score_stable import IQTable
                d = IQTable.from_csv(cpath).get_output_dict()
                out["supplementary_c2bca8a_IQTable_on_this_csv"] = d
            except Exception as e:
                out["supplementary_error"] = repr(e)
    if "official" not in out:
        out["status"] = "FAILED_no_score" if a.rc == 0 else out["status"]
    tail = log.strip().splitlines()[-15:]
    out["log_tail"] = tail
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out.get(k) for k in ["name", "mode", "status", "official", "timing"]}, indent=1))
    sys.exit(0 if "official" in out else 2)


if __name__ == "__main__":
    main()
