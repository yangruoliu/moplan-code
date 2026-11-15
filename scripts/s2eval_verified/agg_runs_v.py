#!/usr/bin/env python3
"""Aggregate several Physics-IQ run scores (same MODE, written by launch/eval_piq_score.sh) into mean +- std (ddof=1).

usage: agg_runs_v.py --results-dir DIR --mode MODE --label LABEL NAME [NAME ...]
  reads  <results-dir>/<NAME>__<MODE>.json
  writes <results-dir>/agg__<LABEL>__<MODE>.json and prints a short table
For MODE=verified it also runs the official c2bca8a physiq/aggregate_runs_from_csvs.py --score-type verified on the
per-run CSVs (official leaderboard aggregation; <=4 CSVs) and stores its printed table.
For orig_* modes the official per-run number is final_score_origround / the ec2c611 printed score; the official
aggregator has no rounded-original mode, so mean/std are computed here (ddof=1, as in the aggregator).
Environment defaults: PIQ_RESULTS (results dir), PIQ_PYTHON (scorer venv python), PIQ_CODE_VERIFIED (c2bca8a checkout).
The per-run CSV paths of the run summaries are resolved from the folder of each summary (absolute paths stay as they
are). For orig_* modes the printed mean is the mean of the rounded per-run scores (for orig_ec2c611 the scores as the
scorer prints them, 2 decimals); the reported Original value comes from launch/eval_piq_report.sh
(scripts/eval/piq_report_values.py).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["verified", "orig_new", "orig_ec2c611"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--results-dir", default=os.environ.get("PIQ_RESULTS", ""))
    ap.add_argument("--piq-python", default=os.environ.get("PIQ_PYTHON", sys.executable),
                    help="python of the scorer environment (runs the official aggregator)")
    ap.add_argument("--piq-code", default=os.environ.get("PIQ_CODE_VERIFIED", ""),
                    help="physics-IQ-benchmark checkout at c2bca8a (for physiq/aggregate_runs_from_csvs.py)")
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    if not a.results_dir:
        sys.exit("--results-dir (or PIQ_RESULTS) is required")

    runs, csvs = [], []
    for n in a.names:
        p = os.path.join(a.results_dir, f"{n}__{a.mode}.json")
        d = json.load(open(p))
        if d.get("status") != "ok" or "official" not in d:
            sys.exit(f"run {n} not ok: {d.get('status')}")
        runs.append(d)
        csv = (d.get("files") or {}).get("csv")
        csvs.append(os.path.join(os.path.dirname(p), csv) if csv else csv)   # relative to the summary's folder

    key = "verified_score" if a.mode == "verified" else "original_physics_iq_score"
    vals = [r["official"][key] for r in runs]
    out = {"label": a.label, "mode": a.mode, "score_key": key, "names": a.names, "per_run": vals,
           "n": len(vals), "mean": statistics.mean(vals),
           "std_ddof1": statistics.stdev(vals) if len(vals) > 1 else None}
    if a.mode == "verified":
        for sub in ["score_spatiotemporal_view", "score_spatial_view", "score_weighted_spatial_view", "score_mse_view"]:
            xs = [100 * r["official"][sub] for r in runs]
            out[sub + "_x100"] = {"mean": statistics.mean(xs), "std_ddof1": statistics.stdev(xs) if len(xs) > 1 else None}
        if len(csvs) <= 4 and a.piq_code:
            cmd = [a.piq_python, os.path.join(a.piq_code, "physiq/aggregate_runs_from_csvs.py"), *csvs,
                   "--score-type", "verified"]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
            res = subprocess.run(cmd, capture_output=True, text=True, env=env)
            shown = ["python", "physiq/aggregate_runs_from_csvs.py",                 # recorded without machine paths
                     *[os.path.relpath(c, a.results_dir) for c in csvs], "--score-type", "verified"]
            out["official_aggregator"] = {"cmd": shown, "rc": res.returncode, "stdout": res.stdout, "stderr": res.stderr[-2000:]}
        elif not a.piq_code:
            out["official_aggregator"] = "skipped (no --piq-code given)"
        else:
            out["official_aggregator"] = "skipped (>4 CSVs not supported by the official script)"
    else:
        alt = [100 * r["official"]["final_score_orig"] for r in runs if "final_score_orig" in r["official"]]
        if len(alt) == len(runs):
            out["unrounded_final_score_orig_x100"] = {"per_run": alt, "mean": statistics.mean(alt),
                                                      "std_ddof1": statistics.stdev(alt) if len(alt) > 1 else None}
    dst = os.path.join(a.results_dir, f"agg__{a.label}__{a.mode}.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    sd = f"{out['std_ddof1']:.2f}" if out["std_ddof1"] is not None else "n/a"
    what = key if a.mode == "verified" else f"{key} (mean of the rounded per-run scores)"
    print(f"{a.label} [{a.mode}] n={out['n']} {what}: {out['mean']:.2f} +- {sd}  per-run={['%.2f' % v for v in vals]}")
    if a.mode == "verified" and isinstance(out.get("official_aggregator"), dict):
        print(out["official_aggregator"]["stdout"])
    print("->", dst)


if __name__ == "__main__":
    main()
