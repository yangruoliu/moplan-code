#!/usr/bin/env python3
"""Collect whatever is scored so far for a list of models on the benchmark sets generated under the MoPlan protocol
(448x256 @ 7 fps) -> <root>/RESULTS.json + a markdown table on stdout.
  RigidBench  <root>/rigidbench/runs/<M>/results.json (official `rigidbench evaluate`, 100 samples)
  Morpheus    morph_aggregate_br.py over <root>/morpheus/results/single_frame_conditioning/MP-<M>: discarded videos -> 0,
              Total = (Dyn + PhysInv) / 2, best of plain / enhanced by Total (Morpheus paper rule)
  PisaBench   <root>/pisabench/official/<M>/summary.json (official eval.py, 361 clips, 1 video each)
  VBench-I2V  <root>/vbench/results/<M>/summary.json (5 videos x 1,118 entries), else <root>/vbench/results/<M>-s0/summary.json
              (provisional, seed 0)
M is the model label used by the launchers (e.g. moplan, base, zero).
usage: osp_collect.py --root EVAL_ROOT [--models moplan base]   (plain python3)"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

AGG = Path(__file__).resolve().parents[1] / "bench" / "morph_aggregate_br.py"


def load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def rigid(root: Path, m: str):
    r = load(root / f"rigidbench/runs/{m}/results.json")
    if not r:
        return None
    a = r["aggregated"]
    return {"n": a["n_samples"], "iou": a["iou"], "l2": a["l2"], "chamfer": a["chamfer"], "ate": a["ate"], "ate3d": a["ate3d"],
            "official": a.get("official"), "path": str(root / f"rigidbench/runs/{m}/results.json")}


def morph(root: Path, m: str):
    d = root / f"morpheus/results/single_frame_conditioning/MP-{m}"
    if not (root / f"morpheus/summary_MP-{m}.json").is_file():
        return None
    subprocess.run([sys.executable, str(AGG), str(d)], check=True, capture_output=True)
    agg = load(d / "aggregate.json")
    (model, views), = agg.items()
    best = views.get("best_prompt_by_total")
    out = {pt: {"n": v["n"], "discard_rate": v["discard_rate"], **v["per_video_zeroed"]} for pt, v in views.items() if pt != "best_prompt_by_total"}
    return {"best_prompt": best, "best": out.get(best), "by_prompt": out, "path": str(d / "aggregate.json")}


def pisa(root: Path, m: str):
    s = load(root / f"pisabench/official/{m}/summary.json")
    if not s:
        return None
    r = s["per_seed"]["s0"]
    return {"n": r["scored"], "videos": r["videos"], "failed": r["failed"], "l2": r["l2"], "chamfer": r["chamfer_distance"], "iou": r["iou"],
            "path": str(root / f"pisabench/official/{m}/summary.json")}


def vbench(root: Path, m: str):
    for p, kind in ((root / f"vbench/results/{m}/summary.json", "official 5 videos"),
                    (root / f"vbench/results/{m}-s0/summary.json", "PROVISIONAL seed 0 only")):
        s = load(p)
        if s:
            return {"kind": kind, "videos": s["videos"], "total": s["total_score"], "i2v": s["i2v_score"], "quality": s["quality_score"],
                    "all_9_dims": s["all_9_dims_present"], "dims": {d: v["mean"] for d, v in s["dimensions"].items()}, "path": str(p)}
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="evaluation root ($EVAL_ROOT)")
    p.add_argument("--models", nargs="+", default=["moplan", "base"])
    a = p.parse_args()
    res = {m: {"rigidbench": rigid(a.root, m), "morpheus": morph(a.root, m), "pisabench": pisa(a.root, m),
               "vbench_i2v": vbench(a.root, m)} for m in a.models}
    (a.root / "RESULTS.json").write_text(json.dumps(res, indent=1) + "\n")
    f = lambda x, k, fmt="{:.3f}": fmt.format(x[k]) if x else "-"
    print("| model | RigidBench IoU / L2 / CD / ATE / ATE-3D | Morpheus Total / PhysInv / Dyn (best prompt) | PisaBench L2 / CD / IoU | VBench-I2V Total / I2V / Quality |")
    print("|---|---|---|---|---|")
    for name, r in res.items():
        rg, mo, pi, vb = r["rigidbench"], r["morpheus"], r["pisabench"], r["vbench_i2v"]
        rgs = " / ".join(f(rg, k) for k in ("iou", "l2", "chamfer", "ate")) + " / " + f(rg, "ate3d", "{:.2f}") if rg else "-"
        mos = (" / ".join(f(mo["best"], k) for k in ("total", "physinv", "dyn")) + f" ({mo['best_prompt']})") if mo else "-"
        pis = " / ".join(f(pi, k) for k in ("l2", "chamfer", "iou")) + (f" (n={pi['n']})" if pi else "") if pi else "-"
        vbs = (" / ".join(f"{100 * vb[k]:.2f}" for k in ("total", "i2v", "quality")) + f" ({vb['kind']})") if vb else "-"
        print(f"| {name} | {rgs} | {mos} | {pis} | {vbs} |")
    print(f"\n{a.root / 'RESULTS.json'}")


if __name__ == "__main__":
    main()
