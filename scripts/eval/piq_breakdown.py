#!/usr/bin/env python3
"""Physics-IQ per-category / per-component breakdown, model "final" vs model "base" (e.g. MoPlan vs the untrained
Wan2.2 under the same protocol).

Reads ONLY existing scored per-scenario CSVs (no generation, no re-scoring), written by launch/eval_piq_score.sh:
  (A) Verified scorer (piq_c2bca8a IQTable, final_score_view), bpp prompts, `<label>-bpp-run_0k__verified` dirs
      (also op prompts, `<label>-op-run_0k__verified`)
  (B) Original scorer (piq_ec2c611 calculate_iq_score), op prompts, `<label>-op-run_0k__orig_ec2c611` dirs
The official aggregation is re-implemented as a function of scenario weights (subset or bootstrap multiplicities),
verified on the full set against the official outputs, and cross-checked on every category subset by running the
official code itself on that subset (in memory, nothing written).

Run with the Physics-IQ scorer environment (numpy, pandas; the c2bca8a checkout is put on sys.path):
  OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $PIQ_PYTHON scripts/eval/piq_breakdown.py \
      --results-dir $PIQ_RESULTS --code-verified $PIQ_CODE_VERIFIED --code-original $PIQ_CODE_ORIGINAL \
      --final moplan --base wan22-base OUT.json
"""

import ast
import datetime
import glob
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd

import argparse as _argparse

_ap = _argparse.ArgumentParser(description=__doc__, formatter_class=_argparse.RawDescriptionHelpFormatter)
_ap.add_argument("out", nargs="?", default="piq_breakdown.json")
_ap.add_argument("--results-dir", default=os.environ.get("PIQ_RESULTS", ""), help="folder of <NAME>__<MODE> results")
_ap.add_argument("--code-verified", default=os.environ.get("PIQ_CODE_VERIFIED", ""), help="physics-IQ-benchmark @c2bca8a")
_ap.add_argument("--code-original", default=os.environ.get("PIQ_CODE_ORIGINAL", ""), help="physics-IQ-benchmark @ec2c611")
_ap.add_argument("--final", default="moplan", help='run label of arm "final" (<label>-<op|bpp>-run_0k)')
_ap.add_argument("--base", default="wan22-base", help='run label of arm "base" (may also be an ablation, e.g. moplan-zero)')
_args = _ap.parse_args()
if not (_args.results_dir and _args.code_verified and _args.code_original):
    _ap.error("--results-dir, --code-verified and --code-original (or PIQ_RESULTS / PIQ_CODE_VERIFIED / PIQ_CODE_ORIGINAL) are required")

R = _args.results_dir
CODE_NEW = _args.code_verified
CODE_OLD = _args.code_original
DESC_NEW = CODE_NEW + "/descriptions/best_practice/descriptions_base.csv"
DESC_NEW_ORIG = CODE_NEW + "/descriptions/descriptions_original.csv"
DESC_OLD = CODE_OLD + "/descriptions/descriptions.csv"


def shown(path, checkout):
    """a file of a scorer checkout as recorded in the output: <checkout folder name>/<path inside it>, no machine path"""
    return os.path.join(os.path.basename(os.path.normpath(checkout)), os.path.relpath(path, checkout))


ARMS = {
    "final": _args.final,
    "base": _args.base,
}
VIEWS = ["perspective-left", "perspective-center", "perspective-right"]
EPS = 1e-8
SEED = 20261024
B = 10000
OUT = _args.out

# ---- official code -----------------------------------------------------------
sys.path.insert(0, CODE_NEW)
from physiq.calculate_iq_score import parse_list_of_floats  # noqa: E402  (rounds to 4 dp)
from physiq.calculate_iq_score_stable import IQTable  # noqa: E402


def load_ec2c611_function():
    """Exec the verbatim ec2c611 calculate_iq_score() source without importing matplotlib."""
    src = open(CODE_OLD + "/code/calculate_iq_score.py").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("parse_list_of_floats", "calculate_iq_score")]
    ns = {"pd": pd, "np": np, "re": re, "VIEWS": VIEWS}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "ec2c611_calculate_iq_score", "exec"), ns)
    return ns["calculate_iq_score"]


calc_iq_ec2c611 = load_ec2c611_function()

LIST_KEYS = ["spatiotemporal_iou_v1", "v1_mse", "variance_spatiotemporal_iou", "variance_mse"]
SCALAR_KEYS = ["spatial_iou_v1", "weighted_spatial_iou_v1", "variance_spatial", "variance_weighted_spatial"]


def read_parsed(path):
    df = pd.read_csv(path)
    raw = df.copy()
    for k in LIST_KEYS:
        for v in VIEWS:
            df[f"{k}_{v}"] = df[f"{k}_{v}"].apply(parse_list_of_floats)
    return raw, df


# ---- per-scenario arrays -------------------------------------------------------
def scenario_arrays(df):
    """Per-scenario, per-view primitives from a parsed CSV (rows sorted by scenario)."""
    n = len(df)
    a = {}
    for k in LIST_KEYS:
        mean = np.zeros((n, 3)); ssum = np.zeros((n, 3)); cnt = np.zeros((n, 3))
        for j, v in enumerate(VIEWS):
            for i, lst in enumerate(df[f"{k}_{v}"]):
                arr = np.array(lst, dtype=float)
                mean[i, j] = arr.mean(); ssum[i, j] = arr.sum(); cnt[i, j] = arr.size
        a[k + "_mean"] = mean; a[k + "_sum"] = ssum; a[k + "_cnt"] = cnt
        # row mean over the three views concatenated (ec2c611 / c2bca8a list aggregation)
        a[k + "_rowcat"] = np.array([
            np.mean(np.concatenate([np.array(df[f"{k}_{v}"].iloc[i], dtype=float) for v in VIEWS]))
            for i in range(n)])
    for k in SCALAR_KEYS:
        a[k] = df[[f"{k}_{v}" for v in VIEWS]].to_numpy(dtype=float)
    for key, val in a.items():
        assert np.all(np.isfinite(val)), f"non-finite values in {key}"
    return a


def verified_contrib(a):
    """(n_scen, 5) per-scenario contribution = mean over views of the clipped per-(scenario,view)
    scores, columns [final, spatiotemporal, spatial, weighted_spatial, mse]. The official
    final_score_view is the plain mean over all (scenario, view) rows, so a subset/bootstrap
    aggregate is the (weighted) mean of these contributions -- exact, no approximation."""
    sp = a["spatial_iou_v1"] / (a["variance_spatial"] + EPS)
    ws = a["weighted_spatial_iou_v1"] / (a["variance_weighted_spatial"] + EPS)
    st = a["spatiotemporal_iou_v1_mean"] / (a["variance_spatiotemporal_iou_mean"] + EPS)
    with np.errstate(divide="ignore"):
        mse = (a["v1_mse_mean"] / (a["variance_mse_mean"] + EPS)) ** -1
    cl = [np.clip(x, 0.0, 1.0) for x in (sp, ws, st, mse)]  # official metric_keys order
    final = np.stack(cl, axis=-1).mean(axis=-1)
    per_view = np.stack([final, cl[2], cl[0], cl[1], cl[3]], axis=-1)  # (n, 3, 5)
    return per_view.mean(axis=1)


VER_COLS = ["final_score_view", "score_spatiotemporal_view", "score_spatial_view",
            "score_weighted_spatial_view", "score_mse_view"]


def orig_ec2c611(a, w):
    """ec2c611 original Physics-IQ score (x100, clipped, NOT rounded to 2 dp) and its components,
    for scenario weight matrix w (B, n). Mirrors calculate_iq_score() incl. variance rounding."""
    w = np.atleast_2d(w).astype(float)
    W = w.sum(axis=1)
    st = w @ a["spatiotemporal_iou_v1_rowcat"] / W
    mse = w @ a["v1_mse_rowcat"] / W
    sp = ((w @ a["spatial_iou_v1"]) / W[:, None]).mean(axis=1)
    ws = ((w @ a["weighted_spatial_iou_v1"]) / W[:, None]).mean(axis=1)
    # variance lists: per view, explode().mean() over all frames of all (weighted) rows
    v_mse = np.round(((w @ a["variance_mse_sum"]) / (w @ a["variance_mse_cnt"])).mean(axis=1), 4)
    v_st = np.round(((w @ a["variance_spatiotemporal_iou_sum"]) /
                     (w @ a["variance_spatiotemporal_iou_cnt"])).mean(axis=1), 4)
    v_sp = np.round(((w @ a["variance_spatial"]) / W[:, None]).mean(axis=1), 5)
    v_ws = np.round(((w @ a["variance_weighted_spatial"]) / W[:, None]).mean(axis=1), 4)
    c_st, c_sp, c_ws, c_mse = st / v_st, sp / v_sp, ws / v_ws, mse - v_mse
    raw = ((c_st + c_sp + c_ws) / 3 - c_mse) * 100
    score = np.clip(raw, 0.0, 100.0)
    return np.stack([score, c_st * 100, c_sp * 100, c_ws * 100, c_mse * 100], axis=-1), raw


ORIG_COLS = ["final_score_orig_ec2c611", "score_spatiotemporal", "score_spatial",
             "score_weighted_spatial", "score_mse"]


def orig_c2bca8a_full(a):
    """c2bca8a IQTable 'orig' quantities on the full set (unrounded variances) -- used only to
    check the component machinery against the stored supplementary numbers to 1e-6."""
    st = a["spatiotemporal_iou_v1_rowcat"].mean(); mse = a["v1_mse_rowcat"].mean()
    v_st = a["variance_spatiotemporal_iou_rowcat"].mean(); v_mse = a["variance_mse_rowcat"].mean()
    sp = a["spatial_iou_v1"].mean(axis=1).mean(); ws = a["weighted_spatial_iou_v1"].mean(axis=1).mean()
    v_sp = a["variance_spatial"].mean(axis=1).mean(); v_ws = a["variance_weighted_spatial"].mean(axis=1).mean()
    d = {"score_spatiotemporal": st / v_st, "score_spatial": sp / v_sp,
         "score_weighted_spatial": ws / v_ws, "score_mse": mse - v_mse}
    d["final_score_orig"] = float(np.clip((d["score_spatiotemporal"] + d["score_spatial"]
                                           + d["score_weighted_spatial"]) / 3 - d["score_mse"], 0, 1))
    return d


# ---- load ------------------------------------------------------------------------
def run_ids(label, proto, mode):
    out = []
    for d in sorted(glob.glob(f"{R}/{label}-{proto}-run_0[1-4]__{mode}")):
        if os.path.isdir(d):
            out.append(re.search(r"run_(\d\d)__", d).group(1))
    return out


def load_mode(proto, mode):
    data = {}
    for arm, label in ARMS.items():
        runs = []
        for rid in run_ids(label, proto, mode):
            name = f"{label}-{proto}-run_{rid}"
            d = f"{R}/{name}__{mode}"
            raw, df = read_parsed(f"{d}/{name}.csv")
            order = np.argsort(df["scenario"].to_numpy())
            raw = raw.iloc[order].reset_index(drop=True); df = df.iloc[order].reset_index(drop=True)
            if mode == "verified":
                official = json.load(open(f"{d}/{name}_metrics.json"))
            else:
                j = json.load(open(f"{d}.json"))
                official = {"printed": j["official"]["original_physics_iq_score"],
                            "supp": j.get("supplementary_c2bca8a_IQTable_on_this_csv", {})}
            runs.append({"run": rid, "name": name, "raw": raw, "df": df,
                         "a": scenario_arrays(df), "official": official})
        data[arm] = runs
    scen = data["final"][0]["df"]["scenario"].tolist()
    for arm in data:
        for r in data[arm]:
            assert r["df"]["scenario"].tolist() == scen, "scenario sets differ"
    return data, scen


def categories(scen):
    maps = {}
    for f in (DESC_NEW, DESC_NEW_ORIG, DESC_OLD):
        x = pd.read_csv(f)
        x["scn"] = x["scenario"].str.replace(r"^\d+_perspective-[a-z]+_take-\d+_", "", regex=True)
        assert x.groupby("scn")["category"].nunique().max() == 1
        maps[f] = x.groupby("scn")["category"].first().to_dict()
    ref = maps[DESC_NEW]
    for f, m in maps.items():
        assert all(m[s] == ref[s] for s in scen), f"category mapping differs in {f}"
    return [ref[s] for s in scen]


# ---- main -------------------------------------------------------------------------
def main():
    np.seterr(over="ignore")
    modes = {"verified_bpp": ("bpp", "verified"), "verified_op": ("op", "verified"), "original_op_ec2c611": ("op", "orig_ec2c611")}
    loaded = {k: load_mode(*v) for k, v in modes.items()}
    scen = loaded["verified_bpp"][1]
    assert scen == loaded["original_op_ec2c611"][1]
    cats = np.array(categories(scen))
    n = len(scen)
    cat_order = pd.Series(cats).value_counts().index.tolist()  # by size
    rows = ["Overall"] + cat_order
    masks = {"Overall": np.ones(n, bool)}
    masks.update({c: cats == c for c in cat_order})

    # bootstrap weight matrices (shared by both scorers): Overall unstratified, category rows stratified
    rng = np.random.default_rng(SEED)
    boot_w = {}
    for row in rows:
        members = np.flatnonzero(masks[row])
        idx = rng.integers(0, members.size, size=(B, members.size))
        w = np.zeros((B, n))
        np.add.at(w, (np.repeat(np.arange(B), members.size), members[idx.ravel()]), 1.0)
        boot_w[row] = w

    verification = {}
    results = {}
    for mkey, (data, _) in loaded.items():
        is_ver = mkey.startswith("verified")
        cols = VER_COLS if is_ver else ORIG_COLS
        ver = {"full_set": [], "subset_vs_official_code": []}
        # ---- verification on the full set
        for arm in data:
            for r in data[arm]:
                a = r["a"]
                if is_ver:
                    mine = verified_contrib(a).mean(axis=0)
                    off = [r["official"][c] for c in VER_COLS]
                    tab = IQTable(r["df"]).get_output_dict()
                    ver["full_set"].append({
                        "run": r["name"],
                        "max_abs_diff_vs_metrics_json": float(np.max(np.abs(mine - np.array(off)))),
                        "max_abs_diff_vs_IQTable_now": float(np.max(np.abs(mine - np.array([tab[c] for c in VER_COLS])))),
                        "mine": dict(zip(VER_COLS, mine.tolist()))})
                else:
                    mine, raw = orig_ec2c611(a, np.ones(n))
                    off_fn = calc_iq_ec2c611(io.StringIO(r["raw"].to_csv(index=False)))[0]
                    supp = r["official"]["supp"]
                    c2 = orig_c2bca8a_full(a)
                    ver["full_set"].append({
                        "run": r["name"],
                        "mine_unrounded_x100": float(mine[0, 0]),
                        "official_printed_2dp": r["official"]["printed"],
                        "official_fn_rerun_2dp": off_fn,
                        "round2_matches_printed": bool(round(float(mine[0, 0]), 2) == r["official"]["printed"] == off_fn),
                        "abs_diff_vs_printed": float(abs(mine[0, 0] - r["official"]["printed"])),
                        "components_c2bca8a_max_abs_diff_vs_supplementary_json": float(max(
                            abs(c2[k] - supp[k]) for k in c2)) if supp else None,
                        "mine_components_ec2c611_x100": dict(zip(ORIG_COLS[1:], mine[0, 1:].tolist()))})
        # ---- subset checks vs official code run on each category subset
        for arm in data:
            for r in data[arm]:
                for row in rows[1:]:
                    m = masks[row]
                    if is_ver:
                        mine = verified_contrib(r["a"])[m].mean(axis=0)
                        tab = IQTable(r["df"][m].reset_index(drop=True)).get_output_dict()
                        ver["subset_vs_official_code"].append({
                            "run": r["name"], "category": row,
                            "max_abs_diff": float(np.max(np.abs(mine - np.array([tab[c] for c in VER_COLS]))))})
                    else:
                        mine, _ = orig_ec2c611(r["a"], m.astype(float))
                        off = calc_iq_ec2c611(io.StringIO(r["raw"][m].to_csv(index=False)))[0]
                        ver["subset_vs_official_code"].append({
                            "run": r["name"], "category": row, "mine_x100": float(mine[0, 0]),
                            "official_fn_2dp": off, "round2_match": bool(round(float(mine[0, 0]), 2) == off)})
        verification[mkey] = ver

        # ---- point estimates + paired bootstrap
        res = {"columns": cols, "runs": {arm: [r["run"] for r in data[arm]] for arm in data},
               "direction": {c: ("lower_better" if (not is_ver and c == "score_mse") else "higher_better")
                             for c in cols},
               "rows": {}}
        for row in rows:
            m = masks[row].astype(float)
            arm_point, arm_runs, arm_boot, clipped = {}, {}, {}, {}
            for arm in data:
                per_run, boots, clip_flags = [], [], []
                for r in data[arm]:
                    if is_ver:
                        c = verified_contrib(r["a"])
                        per_run.append((m @ c / m.sum()) * 100)
                        boots.append((boot_w[row] @ c / boot_w[row].sum(axis=1, keepdims=True)) * 100)
                    else:
                        v, raw = orig_ec2c611(r["a"], m)
                        per_run.append(v[0]); clip_flags.append(bool(raw[0] < 0 or raw[0] > 100))
                        bv, braw = orig_ec2c611(r["a"], boot_w[row])
                        boots.append(bv)
                per_run = np.array(per_run)
                arm_runs[arm] = per_run
                arm_point[arm] = per_run.mean(axis=0)
                arm_boot[arm] = np.mean(boots, axis=0)  # (B, 5): run-mean on each replicate
                clipped[arm] = clip_flags
            diff = arm_point["final"] - arm_point["base"]
            bdiff = arm_boot["final"] - arm_boot["base"]
            lo, hi = np.percentile(bdiff, [2.5, 97.5], axis=0)
            entry = {"n_scenarios": int(masks[row].sum()), "n_scenario_view_rows": int(3 * masks[row].sum()),
                     "metrics": {}}
            if not is_ver:
                entry["orig_score_clipped_runs"] = clipped
            for j, c in enumerate(cols):
                higher = res["direction"][c] == "higher_better"
                excl = bool(lo[j] > 0 or hi[j] < 0)
                fav = bool(diff[j] > 0) if higher else bool(diff[j] < 0)
                entry["metrics"][c] = {
                    "final_mean": float(arm_point["final"][j]),
                    "final_runs": arm_runs["final"][:, j].tolist(),
                    "final_std_ddof1": float(np.std(arm_runs["final"][:, j], ddof=1)),
                    "base_mean": float(arm_point["base"][j]),
                    "base_runs": arm_runs["base"][:, j].tolist(),
                    "base_std_ddof1": float(np.std(arm_runs["base"][:, j], ddof=1)),
                    "diff_final_minus_base": float(diff[j]),
                    "ci95_lo": float(lo[j]), "ci95_hi": float(hi[j]),
                    "boot_mean_diff": float(bdiff[:, j].mean()),
                    "ci_excludes_0": excl, "favourable_to_final": fav,
                    "frac_boot_favourable": float(np.mean(bdiff[:, j] > 0) if higher else np.mean(bdiff[:, j] < 0)),
                }
            res["rows"][row] = entry
        results[mkey] = res

    out = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "description": "Physics-IQ breakdown, arm final vs arm base; "
                       "existing scored CSVs only. All scores x100.",
        "arms": ARMS,
        "results_root": os.path.basename(os.path.normpath(R)),          # folder names only, no machine paths
        "scorers": {
            "verified_bpp": {"code": os.path.basename(os.path.normpath(CODE_NEW)), "fn": "physiq.calculate_iq_score_stable.IQTable.get_output_dict "
                             "(final_score_view, score_*_view)", "dirs": "<label>-bpp-run_XX__verified"},
            "original_op_ec2c611": {"code": os.path.basename(os.path.normpath(CODE_OLD)), "fn": "code/calculate_iq_score.py::calculate_iq_score "
                                    "(variance rounding kept; final x100 clipped [0,100], not rounded to 2 dp)",
                                    "dirs": "<label>-op-run_XX__orig_ec2c611",
                                    "components": "score_st/sp/ws = metric_mean / rounded physical variance; "
                                                  "score_mse = mse_mean - rounded variance_mse (subtracted: lower is better)"}},
        "categories": {"source": shown(DESC_NEW, CODE_NEW), "also_checked_identical": [shown(DESC_NEW_ORIG, CODE_NEW),
                                                                                   shown(DESC_OLD, CODE_OLD)],
                       "n_scenarios": {c: int(masks[c].sum()) for c in rows},
                       "scenario_to_category": dict(zip(scen, cats.tolist()))},
        "bootstrap": {"B": B, "seed": SEED, "rng": "numpy.default_rng",
                      "scheme": "paired: identical scenario multiplicities for both arms and all runs; "
                                "Overall resamples all 66 scenarios (unstratified); category rows resample "
                                "within the category; each arm = mean over its runs of the official aggregate "
                                "on the replicate; CI = 2.5/97.5 percentiles of final-base; replicates shared "
                                "between the two scorers"},
        "verification": verification,
        "results": results,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    # ---- console summary
    for mkey, v in verification.items():
        if mkey.startswith("verified"):
            print(mkey, "full-set max|diff| vs metrics.json:",
                  max(x["max_abs_diff_vs_metrics_json"] for x in v["full_set"]),
                  " subset max|diff| vs IQTable:", max(x["max_abs_diff"] for x in v["subset_vs_official_code"]))
        else:
            print(mkey, "full-set round2 matches:", all(x["round2_matches_printed"] for x in v["full_set"]),
                  " max|diff| vs printed:", max(x["abs_diff_vs_printed"] for x in v["full_set"]),
                  " comp vs supp:", max(x["components_c2bca8a_max_abs_diff_vs_supplementary_json"] or 0 for x in v["full_set"]),
                  " subset round2 matches:", all(x["round2_match"] for x in v["subset_vs_official_code"]),
                  sum(x["round2_match"] for x in v["subset_vs_official_code"]), "/", len(v["subset_vs_official_code"]))
    for mkey, res in results.items():
        print("\n==", mkey, res["runs"])
        for row, e in res["rows"].items():
            cells = []
            for c in res["columns"]:
                x = e["metrics"][c]
                cells.append(f"{x['final_mean']:.2f}/{x['base_mean']:.2f}/{x['diff_final_minus_base']:+.2f}"
                             f"[{x['ci95_lo']:+.2f},{x['ci95_hi']:+.2f}]{'fav' if x['favourable_to_final'] else 'unfav'}"
                             f"{'*' if x['ci_excludes_0'] else ''}")
            print(f"{row} (n={e['n_scenarios']}):", " | ".join(cells),
                  ("clipped:" + json.dumps(e.get("orig_score_clipped_runs"))) if "orig_score_clipped_runs" in e else "")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
