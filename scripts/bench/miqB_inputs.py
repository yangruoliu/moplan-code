#!/usr/bin/env python
"""MoProbe-H first frames (finding_set_B.json) -> our deployment manifest, built from the bench3 generation manifests
of the other open models, so MoPlan is measured on exactly their inputs (the same recipe as miq_inputs.py).
The letters A / B in file names, keys and manifest values (finding_set_A / _B, new_vs_A, the bench / domain values,
miqB) are legacy identifiers of the probe-set build, kept so that the outputs stay identical.

  items      --finding-set finding_set_B.json: the "new_vs_A" frames (in B, not in A; --subset all for every H frame),
             dim = its "H" value (first dimension; camera = camera / direction only)
  rows       per item, its "conds" (B's available conditions) restricted to ex / cf / rex / rcf (scored frames ex + cf, role
             frames rex + rcf, camera-only frames ex); nx / nt are empty captions (Stage-1 never saw one, the chain refuses
             it, as in miq_inputs.py).  The text is the other models' exact instruction from the bench3 generation manifests
             (--miq-root/bench3/sel/{gen_manifest,gen_manifest_role,gen_manifest_camB}.jsonl, all three required; the first
             one with a sid wins, and the camB rows carry only the camera-fixed exc / cx conditions, which are not taken
             here); a listed condition with no manifest row is reported, not invented.  --sids: explicit rows (jsonl with
             sid / frame / cond) instead of a subset.
  sample_id  = the MoProbe generation sid (<item>-<cond>); miq_set bench3; miq_gen bench3/gen/{arm}/s0/<sid>.mp4
  caption    --caption-mode ours (default, as miq_inputs.py): bench_common.caption_for(the text); exact: verbatim
  fps/frames 7 / 37 (5.14 s; MoProbe readers stop at 4.0 s), canvas 1280x720 (frames16x9), first frames copied byte-exact
usage (CPU, seconds):
  python scripts/bench/miqB_inputs.py --finding-set data/moprobe/finding_set_B.json --miq-root MOPROBE_ROOT
      --out WORK/miqB/inputs [--subset new|all] [--source-prefix PREFIX] [--limit N (smoke: first N rows per condition)]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402

FEEDS = {"ex": ["F1 Same/Delta_both", "F3 Cam (H, still-camera frames)", "D Dir"], "cf": ["F1 Same/Delta_both"],
         "nt": ["F1 nothing (text says nothing happens)"], "rex": ["F1 role (diagnostic)"], "rcf": ["F1 role (diagnostic)"],
         "exc": ["F3 camera-fixed (ex -> exc)"], "cx": ["F3 camera-fixed (camera sentence only)"]}


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--finding-set", required=True); p.add_argument("--out", required=True)
    p.add_argument("--subset", choices=("new", "all"), default="new"); p.add_argument("--miq-root", required=True)
    p.add_argument("--source-prefix", default="", help="path prefix of first_frame_path in the source manifests that --miq-root replaces")
    p.add_argument("--sids", default="", help="jsonl with sid/frame/cond: explicit rows (e.g. shared frames whose B text differs from A)")
    p.add_argument("--frames", type=int, default=37); p.add_argument("--caption-mode", choices=("ours", "exact"), default="ours")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if (a.frames - 1) % 4 or (a.frames - 1) / bc.OUR_FPS < 4.0:
        raise SystemExit(f"--frames {a.frames}: must be 4k+1 and cover 4.0 s at {bc.OUR_FPS:g} fps")
    root, out = Path(a.miq_root), Path(a.out)
    B = json.loads(Path(a.finding_set).read_text())
    keys = B["new_vs_A"] if a.subset == "new" else sorted(B["H"])
    want = {k: [c for c in B["conds"][k] if c in ("ex", "cf", "rex", "rcf")] for k in keys}
    if a.sids:                                                 # explicit rows instead of a subset of frames
        rows_in = [json.loads(x) for x in Path(a.sids).read_text().splitlines() if x.strip()]
        keys = sorted({r["frame"] for r in rows_in})
        want = {k: [r["cond"] for r in rows_in if r["frame"] == k] for k in keys}
    items = {k: B["H"][k] for k in keys}
    conds = ["ex", "cf", "rex", "rcf", "exc", "cx"]
    man = {}
    for f in ("gen_manifest.jsonl", "gen_manifest_role.jsonl", "gen_manifest_camB.jsonl"):   # camB: B camera-fixed cx / exc
        for r in jl(root / "bench3" / "sel" / f):
            man.setdefault(r["sid"], r)
    frames_dir, frame_of = out / "first_frames", {}

    def frame(item: str, src: Path):
        if item in frame_of:
            path, w, h, digest = frame_of[item]
            if md5(src) != digest:
                raise SystemExit(f"{item}: {src} differs from the item's first frame already copied")
            return path, w, h
        path, w, h = bc.copy_first_frame(src, frames_dir, item)
        frame_of[item] = (str(path), w, h, md5(path))
        return str(path), w, h

    rows, missing = [], collections.Counter()
    for cond in conds:                                        # condition-major: ex first (most metrics read it)
        for item, dim in sorted(items.items()):
            if cond not in want[item]:
                continue
            sid = f"{item}-{cond}"
            r = man.get(sid)
            if r is None or not r.get("instruction", "").strip():
                missing[cond] += 1
                continue
            prompt = r["instruction"]
            caption = bc.caption_for(prompt) if a.caption_mode == "ours" else prompt
            rows.append({"sample_id": sid, "scenario": f"miqB/H/{dim}", "video_path": "", "first_frame_path": None, "caption": caption,
                         "fps": bc.OUR_FPS, "frames": a.frames, "width": None, "height": None, "bench": "motioniq_B", "bench_id": sid,
                         "miq_sid": sid, "item_sid": item, "cond": cond, "miq_set": "bench3",
                         "miq_gen": f"bench3/gen/{{arm}}/s0/{sid}.mp4", "benchmark_prompt": prompt, "caption_mode": a.caption_mode,
                         "says_static": bc.says_static(prompt), "source_image": r["first_frame_path"], "canvas": "16:9 letterbox 1280x720 (frames16x9)",
                         "feeds": FEEDS.get(cond, []), "domain": "motioniq", "part": "H", "dim": dim, "role_diagnostic": dim.startswith("role") or cond in ("rex", "rcf"),
                         "miq_exp": r.get("exp")})
    if a.limit:
        keep, seen = [], collections.Counter()
        for r in rows:
            if seen[r["cond"]] < a.limit:
                keep.append(r); seen[r["cond"]] += 1
        rows = keep
    def local(path: str) -> Path:                              # manifests may store another machine's root
        if a.source_prefix and path.startswith(a.source_prefix):
            return Path(path.replace(a.source_prefix, str(root), 1))
        return Path(path) if Path(path).is_absolute() else root / path
    for r in rows:
        r["first_frame_path"], r["width"], r["height"] = frame(r["item_sid"], local(r["source_image"]))
    bc.write_manifest(rows, out)
    report = bc.check_manifest(out, len(rows))
    report.update(items=len(items), conds=dict(collections.Counter(r["cond"] for r in rows)), missing_rows=dict(missing),
                  dims=dict(collections.Counter(items.values())), first_frames=len(frame_of), frames=a.frames,
                  caption_mode=a.caption_mode, finding_set=str(a.finding_set), finding_set_md5=md5(Path(a.finding_set)), subset=a.subset,
                  no_row_frames=sum(1 for k in keys if not want[k]), sids_file=a.sids or None, limit=a.limit, out=str(out))
    (out / "manifest_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    if report["n_problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
