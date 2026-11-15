#!/usr/bin/env python
"""MoProbe test set A -> our deployment manifest (first frame + caption only), so MoPlan is generated from exactly the
inputs the other I2V models of the MoProbe study received and is scored with the same measurement code.

Not used by the released launchers: launch/eval_bench_inputs.sh moprobe_h builds the MoProbe-H manifest with
miqB_inputs.py.  This adapter also needs a "clips" map that the shipped data/moprobe/finding_set_A.json does not have;
it is kept for reference.

Inputs (the MoProbe bundle under --miq-root; not distributed, see docs/MOPROBE.md):
  finding_set/finding_set_A.json   H: SSv2-val first frames by dimension (relation / noun / role / nothing / lr / updown)
                                   + P: frames without people (ps, piq, mo, cc, rb); "clips" maps model|sid -> clip path
  <set>/sel/gen_manifest*.jsonl    the generation manifests of the other models (instruction, first_frame_path), seed 42
  <set>/sel/frames16x9/<item>.png  the 16:9 canvas every model received: the native frame letterboxed into 1280x720
                                   (generate.py --canvas 16x9 builds the same canvas for a new image)
Every row here copies that row's canvas and text.

  rows (default 436)  the generation items the MoProbe metrics read:
    H  <item>-ex   real-event instruction   (cfset or dirbal, whichever set the other models' A clip came from)   142
       <item>-cf   opposite instruction     (cfset or dirbal)                                                    142
       <item>-exc  real instruction + " The camera is fixed and does not move."  (fx camfix, the frames in A's H)  42
       <item>-cx   (--camfix both only) "The camera is fixed and does not move."                                 (42)
    P  <item>-pd   the source benchmark's prompt as the MoProbe manifest gave it                                  110
  not written: <item>-nx (empty prompt): Stage-1 was trained with caption_dropout = 0, i.e. never on an empty caption,
    and the chain (bench_common.write_manifest) refuses empty captions on purpose.
  sample_id   = the MoProbe generation sid (<item>-<cond>), 1:1; jsonl miq_set / miq_sid / miq_gen give the layout the
              exported clip must take under the MoProbe root: <miq_set>/gen/<arm>/s0/<miq_sid>.mp4 + .json
  scenario    miqA/H/<dim> | miqA/P/<set>
  caption     --caption-mode ours (default): bench_common.caption_for(the exact text the other models got), i.e. our
              training phrase " Static shot with no camera movement." is appended iff the text itself states a static
              camera; whitespace is normalised.  --caption-mode exact: the text verbatim.
              The verbatim text is always kept in benchmark_prompt.
  fps/frames  7 / --frames (default 37 = 5.14 s; the MoProbe readers stop at 4.0 s of clip time; the hold rule is
              applied beyond the 4 s Stage-1 horizon)
  width/height  1280x720, the canvas (the generator stretches it to the 448x256 work frame)
First frames: one byte-exact copy per item, <out>/first_frames/<item>.png (rows of the same item share it; a source that
differs byte-wise from the item's copy fails the run).

usage (CPU, seconds):
  python scripts/bench/miq_inputs.py --miq-root MOPROBE_ROOT --out WORK/miqA/inputs
      [--finding-set FILE] [--source-prefix PREFIX] [--frames 37] [--camfix exc|both|none] [--caption-mode ours|exact]
      [--limit N (smoke: first N rows of every part)]
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

CAM_FIXED = "The camera is fixed and does not move."
P_MANIFEST = {"ps": "gen_manifest_ps.jsonl", "piq": "gen_manifest_piq.jsonl", "mo": "gen_manifest_mo.jsonl",
              "cc": "gen_manifest_cc.jsonl", "rb": "gen_manifest_rb.jsonl"}
FEEDS = {"ex": ["F1 Same/Delta_both", "F3 Cam (H, still-camera frames)", "F3 Follow", "F3 edge", "F3 camera-fixed (base of exc)", "D Dir (candidate)"],
         "cf": ["F1 Same/Delta_both"], "exc": ["F3 camera-fixed (ex -> exc)"], "cx": ["F3 camera-fixed (nx -> cx; needs nx)"],
         "pd": ["F2 Fab", "F2 Touch", "F3 Cam (P)"]}


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--miq-root", required=True, help="MoProbe root with finding_set/ and <set>/sel/{gen_manifest*.jsonl,frames16x9}")
    p.add_argument("--source-prefix", default="", help="path prefix of first_frame_path in the source manifests that --miq-root "
                   "replaces (default: absolute paths are used as they are, relative ones are read under --miq-root)")
    p.add_argument("--finding-set", default="", help="finding_set_A.json (default <miq-root>/finding_set/finding_set_A.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=37)
    p.add_argument("--camfix", choices=("exc", "both", "none"), default="exc")
    p.add_argument("--caption-mode", choices=("ours", "exact"), default="ours")
    p.add_argument("--limit", type=int, default=0, help="smoke: keep only the first N rows of every part (H ex, H cf, camfix, each P set)")
    a = p.parse_args()
    if (a.frames - 1) % 4 or (a.frames - 1) / bc.OUR_FPS < 4.0:
        raise SystemExit(f"--frames {a.frames}: must be 4k+1 and cover the 4.0 s the MoProbe readers score at {bc.OUR_FPS:g} fps")
    root, out = Path(a.miq_root), Path(a.out)
    fs_path = Path(a.finding_set) if a.finding_set else root / "finding_set" / "finding_set_A.json"
    fs = json.loads(fs_path.read_text())
    H, P, clips = fs["H"], fs["P"], fs["clips"]

    def local(path: str) -> Path:                              # manifests may store another machine's root
        if a.source_prefix and path.startswith(a.source_prefix):
            return Path(path.replace(a.source_prefix, str(root), 1))
        return Path(path) if Path(path).is_absolute() else root / path

    man = {st: {r["sid"]: r for r in jl(root / st / "sel" / "gen_manifest.jsonl")} for st in ("cfset", "dirbal")}
    frames_dir, frame_of = out / "first_frames", {}

    def frame(item: str, src: Path) -> tuple[str, int, int]:
        if item in frame_of:
            path, w, h, digest = frame_of[item]
            if md5(src) != digest:
                raise SystemExit(f"{item}: {src} differs from the item's first frame already copied")
            return path, w, h
        path, w, h = bc.copy_first_frame(src, frames_dir, item)
        frame_of[item] = (str(path), w, h, md5(path))
        return str(path), w, h

    def row(sid: str, item: str, cond: str, st: str, scenario: str, prompt: str, src: str, **extra) -> dict:
        caption = bc.caption_for(prompt) if a.caption_mode == "ours" else prompt
        # first_frame_path / width / height are filled after --limit, so a smoke run copies only its own frames
        return {"sample_id": sid, "scenario": scenario, "video_path": "", "first_frame_path": None, "caption": caption,
                "fps": bc.OUR_FPS, "frames": a.frames, "width": None, "height": None,
                "bench": "motioniq_A", "bench_id": sid, "miq_sid": sid, "item_sid": item, "cond": cond, "miq_set": st,
                "miq_gen": f"{st}/gen/{{arm}}/s0/{sid}.mp4", "benchmark_prompt": prompt, "caption_mode": a.caption_mode,
                "says_static": bc.says_static(prompt), "source_image": src, "canvas": "16:9 letterbox 1280x720 (frames16x9)",
                "feeds": FEEDS[cond], "domain": "motioniq", **extra}

    parts: dict[str, list[dict]] = collections.defaultdict(list)
    ex_text = {}
    for item, dim in sorted(H.items()):
        for cond in ("ex", "cf"):
            sid = f"{item}-{cond}"
            sets = {path.split("/motioniq/", 1)[1].split("/", 1)[0] for key, path in clips.items() if key.split("|", 1)[1] == sid}
            if len(sets) != 1 or not sets <= {"cfset", "dirbal"}:
                raise SystemExit(f"{sid}: the five models' clips come from {sets}")
            st = sets.pop()
            r = man[st].get(sid)
            if r is None or not r.get("instruction", "").strip():
                raise SystemExit(f"{sid}: no row / empty instruction in {st}/sel/gen_manifest.jsonl")
            if cond == "ex":
                ex_text[item] = r["instruction"]
            parts[f"H_{cond}"].append(row(sid, item, cond, st, f"miqA/H/{dim}", r["instruction"], r["first_frame_path"], part="H", dim=dim,
                                          role_diagnostic=dim == "role", miq_variant=r.get("variant"), cf_kind=r.get("cf_kind")))
    if a.camfix != "none":
        keep = {"exc": {"real_plus_camera"}, "both": {"real_plus_camera", "camera_only"}}[a.camfix]
        for r in jl(root / "fx" / "sel" / "gen_manifest_camfix.jsonl"):
            item = r["item_sid"]
            if item not in H or r["variant"] not in keep:
                continue
            cond = r["sid"].rsplit("-", 1)[1]
            expect = f"{ex_text[item]} {CAM_FIXED}" if cond == "exc" else CAM_FIXED
            if r["instruction"] != expect:
                raise SystemExit(f"{r['sid']}: camfix text is not the A ex text + the camera sentence: {r['instruction']!r}")
            parts["H_camfix"].append(row(r["sid"], item, cond, "fx", f"miqA/H/{H[item]}", r["instruction"], r["first_frame_path"], part="H",
                                         dim=H[item], role_diagnostic=H[item] == "role", miq_variant=r["variant"], cf_kind=None))
    for st in sorted(P):
        m = {r["sid"]: r for r in jl(root / st / "sel" / P_MANIFEST[st])}
        for item in P[st]:
            sid = f"{item}-pd"
            r = m.get(sid)
            if r is None or not r.get("instruction", "").strip():
                raise SystemExit(f"{sid}: no row / empty instruction in {st}/sel/{P_MANIFEST[st]}")
            parts[f"P_{st}"].append(row(sid, item, "pd", st, f"miqA/P/{st}", r["instruction"], r["first_frame_path"], part="P",
                                        source=st, source_scenario=r.get("scenario"), miq_variant=r.get("variant") or "pd"))
    if a.limit:
        parts = {k: v[:a.limit] for k, v in parts.items()}
    order = ["H_ex", "H_cf", "H_camfix"] + [f"P_{st}" for st in P_MANIFEST]       # ex first: most metrics read it
    rows = [r for k in order + sorted(set(parts) - set(order)) for r in parts.get(k, [])]
    for r in rows:
        r["first_frame_path"], r["width"], r["height"] = frame(r["item_sid"], local(r["source_image"]))
    bc.write_manifest(rows, out)
    report = bc.check_manifest(out, len(rows))
    report.update(parts={k: len(v) for k, v in sorted(parts.items())}, conds=dict(collections.Counter(r["cond"] for r in rows)),
                  sets=dict(collections.Counter(r["miq_set"] for r in rows)), first_frames=len(frame_of),
                  first_frames_mb=round(sum(Path(v[0]).stat().st_size for v in frame_of.values()) / 2 ** 20, 1),
                  H_frames=len({r["item_sid"] for r in rows if r["part"] == "H"}), P_frames=len({r["item_sid"] for r in rows if r["part"] == "P"}),
                  caption_changed_by_cond=dict(collections.Counter(r["cond"] if r["cond"] != "pd" else f"pd/{r['miq_set']}" for r in rows
                                                                 if r["caption"] != " ".join(r["benchmark_prompt"].split()))),
                  frames=a.frames, seconds=round((a.frames - 1) / bc.OUR_FPS, 3), caption_mode=a.caption_mode, camfix=a.camfix, limit=a.limit,
                  caption_words_max=max(len(r["caption"].split()) for r in rows), finding_set=str(fs_path),
                  finding_set_md5=md5(fs_path), finding_set_rule=fs.get("rule"), out=str(out))
    (out / "manifest_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    if report["n_problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
