#!/usr/bin/env python3
"""The reported Physics-IQ columns of a model, read from the JSON files of launch/eval_piq_report.sh and rounded half up
to 2 decimals from the stored values (the convention of the reported tables; plain python, no third-party packages).

  Verified  mean over the bpp runs of the Verified score (agg__<label>-bpp__verified.json "mean": the scorer's unrounded
            final_score_view x 100 of each run)
  Original  mean over the op runs of the UNROUNDED ec2c611 score.  The ec2c611 scorer prints each run's score rounded to
            2 decimals, and agg__<label>-op__orig_ec2c611.json averages these printed values, so its mean (rounded a
            second time, and printed by agg_runs_v.py with %.2f) can differ from the unrounded mean in the last digit.
            The unrounded per-run scores are recomputed from the scorer CSVs by scripts/eval/piq_breakdown.py
            (checked against the printed values), so the Original column is taken from breakdown_<label>_vs_<base>.json
            when it exists (--base); without it, the mean of the printed per-run scores is shown and marked as such.

usage: piq_report_values.py --results-dir PIQ_RESULTS --label LABEL [--base BASE_LABEL]
"""
from __future__ import annotations

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


def round2(x: float) -> str:
    """half up at 2 decimals, from the shortest decimal form of the stored float (2.675 -> 2.68, where %.2f gives 2.67)"""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def columns(results: Path, label: str, arm: str | None, breakdown: dict | None) -> dict:
    """{column: (value, runs, source)} of one label; arm = 'final' / 'base' in the breakdown (None: no breakdown)"""
    out = {}
    ver = read(results / f"agg__{label}-bpp__verified.json")
    if ver:
        out["Verified"] = (ver["mean"], ver["n"], "bpp runs, verified scorer")
    rows = (((breakdown or {}).get("results") or {}).get("original_op_ec2c611") or {}).get("rows") or {}
    cell = ((rows.get("Overall") or {}).get("metrics") or {}).get("final_score_orig_ec2c611")
    if arm and cell:
        out["Original"] = (cell[f"{arm}_mean"], len(cell[f"{arm}_runs"]), "op runs, ec2c611 scorer, unrounded per-run scores")
    else:
        orig = read(results / f"agg__{label}-op__orig_ec2c611.json")
        if orig:
            out["Original"] = (orig["mean"], orig["n"], "op runs, ec2c611 scorer, mean of the per-run scores as printed "
                               "(2 decimals; run with a base label for the unrounded mean)")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--base", default="", help="base label: the Original column of both comes from their breakdown")
    a = p.parse_args(argv)
    results = Path(a.results_dir)
    breakdown = read(results / f"breakdown_{a.label}_vs_{a.base}.json") if a.base else None
    print("[piq report] reported columns (2 decimals, rounded half up from the JSON values):")
    for label, arm in [(a.label, "final")] + ([(a.base, "base")] if a.base else []):
        cols = columns(results, label, arm if breakdown else None, breakdown)
        if not cols:
            print(f"  {label}: no aggregated runs in {results}")
        for name, (value, runs, source) in cols.items():
            print(f"  {label:24s} {name:8s} {round2(value):>6s}  ({runs} runs; {source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
