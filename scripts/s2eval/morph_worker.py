#!/usr/bin/env python3
"""Morpheus steps for launch/eval_score_morpheus.sh.  Runs in the BENCHMARK's venv ($MORPHEUS_HOME/.venv/bin/python,
cwd = the Morpheus checkout), never the MoPlan env.  Our model name is registered in memory, with exactly the effect of
`export_videos.py morpheus-register` (taxonomy.MODEL_NAMES -> --methods choice + "VGM_trajectories" scoring category;
transforms.MODEL_KEYWORDS -> path detection, ours first; default transform = plain cv2 resize to 1280x1024), so the
shared repo is never edited and concurrent jobs cannot race on taxonomy.py.

  resize  --raw RAW --resized RES --model MM [--workers n]
          morpheus.resizing.transforms.process_mp4_file over RAW/single_frame_conditioning/MM/<exp>_prompt_<type>/
          <exp>_prompt_<type>_seed_<k>.mp4 -> RES/single_frame_conditioning/MM/<type>/<exp>/<k>/ (frames_for_tracking,
          output_video.mp4, info.json).  Same function as morpheus-resize; the clip folder number is the seed index
          (upstream's resize_tree numbers clips in unsorted directory order).  Writes RES/.../MM/RESIZE_MAP.json.
  score   --resized RES --results OUT --model MM --items exp:k[,exp:k...]
          the official scoring call (morpheus-track-and-score --conditioning single_frame_conditioning
          --calculate-scores --use-cache, default SAM2 / labels), one (experiment, video number) at a time, one SAM2
          tracker per process.  Finished videos (combined_scores.json) are skipped, so a killed job just reruns.
  check   --resized RES --results OUT --model MM [--expect N]
          completeness check before the aggregation: every resized clip of the 16 reported experiments (RESIZE_MAP.json)
          has its combined_scores.json, and with --expect there are N such clips.  The scorer skips a video without
          writing a score in several cases (missing labels or first frame, a tracking failure, a killed worker), and the
          aggregation averages the scores it finds, so an unscored video would silently drop out of the means instead of
          counting 0 like a discarded one.  Exits 1 and lists the missing videos.
  summary --results OUT --model MM --out JSON
          kept / all means of physical invariance and dynamical score (per
          phenomenon), for MM only, + per-video rows.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# the keywords Morpheus matches in paths (export_videos.py MORPHEUS_RESERVED): a model name holding one would be detected
# as another model / prompt type / experiment
RESERVED = ["kling-turbo", "wan", "cosmos", "ltx", "pyramid", "cogvideo", "veo3", "plain", "regular", "enhanced", "glm4",
            "single_frame_conditioning", "multi_frame_conditioning", "keyframe_interpolation", "real-world",
            "falling", "projectile", "pendulum", "bouncing", "rolling", "sliding", "spring", "collision"]
COND = "single_frame_conditioning"
# the 16 reported experiments (non_holonomic_pendulum excluded, as in the paper)
EXPERIMENTS = ["falling_ball", "falling_apple", "falling_marker", "falling_tape", "bouncing_ball", "projectile",
               "holonomic_pendulum", "double_pendulum", "spring", "sliding_book", "rolling_full_can", "rolling_empty_can",
               "rolling_orange", "collision_equal", "collision_big_hits_small", "collision_small_hits_big"]


def register(name: str, with_pipeline: bool) -> None:
    low = name.lower()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or any(w in low for w in RESERVED):
        raise SystemExit(f"model name {name!r} clashes with Morpheus path keywords (or has odd characters)")
    from morpheus import taxonomy
    from morpheus.resizing import transforms
    clash = [k for k in transforms.MODEL_KEYWORDS if k != low and (k in low or low in k)]
    if clash:
        raise SystemExit(f"model name {name!r} overlaps registered keywords {clash}")
    if name not in taxonomy.MODEL_NAMES:
        taxonomy.MODEL_NAMES.insert(0, name)          # in place: combined.py / pipeline.py hold this very list
    if low not in transforms.MODEL_KEYWORDS:
        merged = {low: name, **transforms.MODEL_KEYWORDS}
        transforms.MODEL_KEYWORDS.clear()
        transforms.MODEL_KEYWORDS.update(merged)
    if hasattr(taxonomy, "VIDEO_SOURCE") and name not in taxonomy.VIDEO_SOURCE:
        taxonomy.VIDEO_SOURCE.append(name)
    if with_pipeline:                                 # sets/lists derived from MODEL_NAMES at import time
        from morpheus import pipeline
        from morpheus.cli import track_and_score
        pipeline.ALL_METHODS.add(name)
        if name not in track_and_score._ALL_METHODS:
            track_and_score._ALL_METHODS.append(name)


def _resize_one(job):
    from morpheus.resizing.transforms import process_mp4_file
    mp4, rel, out, k = job
    process_mp4_file(mp4, rel, out, k, False)
    return job


def cmd_resize(a) -> None:
    register(a.model, with_pipeline=False)
    from morpheus.resizing.transforms import extract_metadata_from_path
    raw, out = Path(a.raw), Path(a.resized)
    src = raw / COND / a.model
    jobs, mapping = [], []
    for leaf in sorted(p for p in src.iterdir() if p.is_dir()):
        for mp4 in sorted(leaf.glob("*.mp4")):
            m = re.fullmatch(rf"{re.escape(leaf.name)}_seed_(\d+)", mp4.stem)
            if not m:
                raise SystemExit(f"unexpected clip name {mp4}")
            k = int(m.group(1))
            rel = os.path.relpath(str(mp4.with_suffix("")), str(raw))
            cond, model, ptype, exp = extract_metadata_from_path(rel)
            if (cond, model) != (COND, a.model) or exp is None or not leaf.name.startswith(exp):
                raise SystemExit(f"{rel}: Morpheus reads conditioning={cond} model={model} type={ptype} experiment={exp}")
            jobs.append((str(mp4), rel, str(out), k))
            mapping.append({"source": str(mp4), "experiment": exp, "prompt_type": ptype, "seed_index": k,
                            "resized": str(out / COND / a.model / ptype / exp / str(k))})
    if not jobs:
        raise SystemExit(f"no clips under {src}")
    from multiprocessing import Pool
    with Pool(max(1, a.workers)) as pool:
        for _ in pool.imap_unordered(_resize_one, jobs):
            pass
    missing = [m["resized"] for m in mapping if not (Path(m["resized"]) / "info.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} clips not resized, e.g. {missing[:2]}")
    target = out / COND / a.model / "RESIZE_MAP.json"
    target.write_text(json.dumps(mapping, indent=1) + "\n")
    print(f"resized {len(jobs)} clips of {a.model} -> {out / COND / a.model}")


def cmd_score(a) -> None:
    register(a.model, with_pipeline=True)
    from morpheus.cli.track_and_score import build_parser
    from morpheus.pipeline import process_all_videos
    from morpheus.tracking.sam2_tracker import LazyTracker
    items = []
    for it in a.items.split(","):
        exp, k = it.rsplit(":", 1)
        if exp not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {exp}")
        items.append((exp, int(k)))
    args = build_parser().parse_args(["--input-dir", a.resized, "--output-dir", a.results, "--methods", a.model,
                                      "--conditioning", COND, "--calculate-scores", "--use-cache"])
    args.depth_processor_v2 = None
    os.makedirs(args.output_dir, exist_ok=True)
    tracker = LazyTracker(checkpoint=args.sam2_checkpoint, config_dir=args.sam2_config_dir, model_cfg=args.sam2_model_cfg,
                          device=args.device)
    for exp, k in items:
        args.video_number = k
        print(f"[morph_worker] item {exp}:{k}", flush=True)
        process_all_videos(tracker, args, conditioning=(COND,), methods=(a.model,), experiments=(exp,))
    print(f"[morph_worker] DONE {len(items)} items", flush=True)


def cmd_check(a) -> None:
    resized, results = Path(a.resized), Path(a.results)
    mapping = json.loads((resized / COND / a.model / "RESIZE_MAP.json").read_text())
    clips = [m for m in mapping if m["experiment"] in EXPERIMENTS]
    # the scorer mirrors the resized tree: RES/<cond>/MM/<type>/<exp>/<k> -> OUT/<cond>/MM/<type>/<exp>/<k>/combined_scores.json
    missing = [m for m in clips if not (results / COND / a.model / m["prompt_type"] / m["experiment"] / str(m["seed_index"])
                                        / "combined_scores.json").is_file()]
    report = {"model": a.model, "clips": len(clips), "per_prompt_type": dict(sorted(Counter(m["prompt_type"] for m in clips).items())),
              "scored": len(clips) - len(missing),
              "missing": [f"{m['prompt_type']}/{m['experiment']}/{m['seed_index']}" for m in missing]}
    print(json.dumps(report))
    if a.expect and len(clips) != a.expect:
        raise SystemExit(f"{len(clips)} resized clips of the reported experiments in RESIZE_MAP.json, expected {a.expect}")
    if missing:
        raise SystemExit(f"{len(missing)} of {len(clips)} videos have no combined_scores.json (see the scorer logs), e.g. "
                         f"{report['missing'][:3]}; rerun the scoring step (finished videos are skipped)")


def cmd_summary(a) -> None:
    register(a.model, with_pipeline=False)
    from morpheus.taxonomy import PHENOMENON_MAP
    out = Path(a.results)
    rows = []
    for path in sorted((out / COND / a.model).rglob("combined_scores.json")):
        d = json.loads(path.read_text())
        f = path.with_name("filtering_info.json")
        disc = json.loads(f.read_text()).get("trajectory_filtering", {}).get("discard_details", {}) if f.is_file() else {}
        rows.append({"model": d.get("model"), "experiment": d.get("experiment"), "prompt_type": d.get("prompt_type"),
                     "video": str(path.parent.relative_to(out)), "seed_index": int(path.parent.name),
                     "physical": d.get("physical_score"), "dynamical": d.get("statistical_score"),
                     "discarded": any(v.get("is_discarded") for v in disc.values())})
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    kept = [r for r in rows if not r["discarded"]]
    by = defaultdict(list)
    for r in rows:
        by[PHENOMENON_MAP.get(r["experiment"], r["experiment"])].append(r)
    summary = {"model": a.model, "videos": len(rows), "discarded": len(rows) - len(kept),
               "physical_invariance_kept": mean([r["physical"] for r in kept if r["physical"] is not None]),
               "dynamical_kept": mean([r["dynamical"] for r in kept if r["dynamical"] is not None]),
               "physical_invariance_all": mean([r["physical"] or 0.0 for r in rows]),
               "dynamical_all": mean([r["dynamical"] or 0.0 for r in rows]),
               "by_phenomenon": {k: {"n": len(v), "discarded": sum(r["discarded"] for r in v),
                                     "physical_kept": mean([r["physical"] for r in v if not r["discarded"] and r["physical"] is not None]),
                                     "dynamical_kept": mean([r["dynamical"] for r in v if not r["discarded"] and r["dynamical"] is not None])}
                                 for k, v in sorted(by.items())},
               "rows": rows}
    Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_phenomenon", "rows")}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resize")
    r.add_argument("--raw", required=True)
    r.add_argument("--resized", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("--resized", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--items", required=True)
    c = sub.add_parser("check")
    c.add_argument("--resized", required=True)
    c.add_argument("--results", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--expect", type=int, default=0, help="number of videos of the reported experiments (0: not checked)")
    m = sub.add_parser("summary")
    m.add_argument("--results", required=True)
    m.add_argument("--model", required=True)
    m.add_argument("--out", required=True)
    a = p.parse_args()
    {"resize": cmd_resize, "score": cmd_score, "check": cmd_check, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
