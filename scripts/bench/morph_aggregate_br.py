#!/usr/bin/env python3
"""Aggregate Morpheus per-video outputs (combined_scores.json + filtering_info.json) into the paper's views.
Paper (v3 §4.2 / App. D.5): discarded videos get PhysInv = 0 and Dyn = 0; Total = (Dyn + PhysInv) / 2; best of plain/enhanced
prompts is reported. The released code only implements the disappearance filter, so 'discarded' here = that filter.
Views printed per (model, prompt_type) and best-of-prompt:
  per_video_zeroed : mean over videos, discarded -> 0 (paper rule, released filter)
  per_video_kept   : mean over kept videos only
  per_category9    : unweighted mean over the 9 Fig. 8 buckets of per-video_zeroed means
usage: morph_aggregate_br.py OUT_DIR [--model NAME]   (plain python3; OUT_DIR = a Morpheus results folder, e.g.
       <results>/single_frame_conditioning/<model>)"""
import json, sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1]); want = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None
# Fig. 8 buckets (rolling and sliding book separate)
BUCKET = {"falling_ball": "Falling", "falling_apple": "Falling", "falling_marker": "Falling", "falling_tape": "Falling",
          "bouncing_ball": "Bouncing", "projectile": "Projectile", "holonomic_pendulum": "Holonomic", "double_pendulum": "Double Pendulum",
          "spring": "Spring", "sliding_book": "Sliding Book", "rolling_full_can": "Rolling", "rolling_empty_can": "Rolling",
          "rolling_orange": "Rolling", "collision_equal": "Collision", "collision_big_hits_small": "Collision", "collision_small_hits_big": "Collision"}
rows = []
for p in out.rglob("combined_scores.json"):
    d = json.loads(p.read_text()); f = p.with_name("filtering_info.json")
    disc = json.loads(f.read_text()).get("trajectory_filtering", {}).get("discard_details", {}) if f.is_file() else {}
    rows.append(dict(model=d.get("model"), exp=d.get("experiment"), pt=d.get("prompt_type") or "-", video=str(p.parent.relative_to(out)),
                     phys=d.get("physical_score"), dyn=d.get("statistical_score"), disc=any(v.get("is_discarded") for v in disc.values())))
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")


def view(rs):
    z = [(0.0 if r["disc"] else (r["dyn"] or 0.0), 0.0 if r["disc"] else (r["phys"] or 0.0)) for r in rs]
    kept = [r for r in rs if not r["disc"]]
    by = defaultdict(list)
    for r, (dz, pz) in zip(rs, z):
        by[BUCKET.get(r["exp"], r["exp"])].append((dz, pz))
    cat = {k: (mean([a for a, _ in v]), mean([b for _, b in v])) for k, v in by.items()}
    dyn_z, phy_z = mean([a for a, _ in z]), mean([b for _, b in z])
    dyn_c, phy_c = mean([v[0] for v in cat.values()]), mean([v[1] for v in cat.values()])
    return {"n": len(rs), "discard_rate": round(1 - len(kept) / max(len(rs), 1), 4),
            "per_video_zeroed": {"dyn": round(dyn_z, 4), "physinv": round(phy_z, 4), "total": round((dyn_z + phy_z) / 2, 4)},
            "per_video_kept": {"dyn": round(mean([r["dyn"] or 0 for r in kept]), 4), "physinv": round(mean([r["phys"] or 0 for r in kept]), 4)},
            "per_category9": {"dyn": round(dyn_c, 4), "physinv": round(phy_c, 4), "total": round((dyn_c + phy_c) / 2, 4),
                              "by_cat_total": {k: round((v[0] + v[1]) / 2, 4) for k, v in sorted(cat.items())}}}


res = {}
for m in sorted({r["model"] for r in rows if r["model"]} | ({None} if any(r["model"] is None for r in rows) else set())):
    if want and m != want:
        continue
    mr = [r for r in rows if r["model"] == m]
    res[str(m)] = {pt: view([r for r in mr if r["pt"] == pt]) for pt in sorted({r["pt"] for r in mr})}
    if len(res[str(m)]) > 1:
        best = max(res[str(m)].items(), key=lambda kv: kv[1]["per_video_zeroed"]["total"])
        res[str(m)]["best_prompt_by_total"] = best[0]
(out / "aggregate.json").write_text(json.dumps(res, indent=1))
(out / "per_video.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(json.dumps(res, indent=1))
