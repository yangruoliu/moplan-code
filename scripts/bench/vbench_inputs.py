#!/usr/bin/env python
"""VBench-I2V (VBench++, vbench2_beta_i2v) image suite -> our deployment manifest (first frame + caption only).

The suite (VBench commit fd18b3d0, vbench2_beta_i2v/vbench2_i2v_full_info.json) has 1118 entries: 355 image-prompt pairs
(246 "subject" images -> i2v_subject, subject_consistency, motion_smoothness, dynamic_degree; 109 "background" images ->
i2v_background, background_consistency; all 355 -> aesthetic_quality, imaging_quality, temporal_flickering) + 763
camera_motion prompts ("<caption>, camera pans left" ... on the 109 background images).  Only the 355 pairs are written:
unless --with-camera is given (the full 1118-entry suite).
Images: vbench2_beta_i2v/data/crop/<ratio>/<image_name> (official crops; 7-4 = 7:4, the aspect of our 448x256 work frame,
so the stretch to the work frame is a pure downscale).

  sample_id   vb_<k> (k = 000 ... 354, the order of the 355 pairs in vbench2_i2v_full_info.json); bench_id = the VBench
              prompt, which is also the stem of the video file the evaluator looks for (<prompt>-<index>.mp4)
  scenario    the image category (single-human, animal, architecture, ...); group = subject | background
  caption     the VBench prompt (bench_common.caption_for: " Static shot with no camera movement." is appended only when
              the prompt itself states a static camera)
  fps/frames  7 / --frames (default 37 = 5.14 s at 7 fps, like VBench's Wan2.2-I2V-A14B row: 81 frames at 16 fps = 5.06 s,
              and our Physics-IQ / Morpheus length)
  width/height  native size of the crop (export_videos.py vbench keeps the 448x256 work frame; the evaluator reads the crop
              itself for i2v_subject / i2v_background)

usage: vbench_inputs.py --code VBENCH_HOME --out DIR [--frames 37] [--ratio 7-4] [--per-group N (smoke: first N of each group)] [--max-width 1344]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402

DEFAULT_CODE = os.environ.get("VBENCH_HOME", "")
SUBJECT_DIMS = ["i2v_subject", "subject_consistency", "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"]
BACKGROUND_DIMS = ["i2v_background", "background_consistency", "aesthetic_quality", "imaging_quality"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code", default=DEFAULT_CODE, required=not DEFAULT_CODE, help="VBench checkout (full info json + data/crop/<ratio>; default: $VBENCH_HOME)")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=37)
    p.add_argument("--ratio", default="7-4")
    p.add_argument("--exact-prompt", action="store_true", help="caption = the benchmark prompt verbatim (whitespace normalised), no appended camera phrase (official-prompt runs)")
    p.add_argument("--with-camera", action="store_true", help="also write the 763 camera_motion entries (sample ids vbc_<k>): the full 1118-entry suite")
    p.add_argument("--per-group", type=int, default=0, help="smoke: keep only the first N subject and the first N background pairs")
    p.add_argument("--max-width", type=int, default=1344, help="downscale wider first frames to this width (aspect kept; 0 = native crop)")
    a = p.parse_args()
    if (a.frames - 1) % 4:
        raise SystemExit(f"--frames {a.frames} is not 4k+1")
    code, out = Path(a.code), Path(a.out)
    full = json.loads((code / "vbench2_beta_i2v" / "vbench2_i2v_full_info.json").read_text())
    pairs = [e for e in full if "camera_motion" not in e["dimension"]]
    entries = full if a.with_camera else pairs                  # --with-camera = the whole suite, file order kept
    meta_path = code / "vbench2_beta_i2v" / "data" / "i2v-bench-info.json"
    meta = {e["file_name"]: e for e in json.loads(meta_path.read_text())} if meta_path.is_file() else {}
    crop_dir = code / "vbench2_beta_i2v" / "data" / "crop" / a.ratio
    rows, taken = [], collections.Counter()
    k_pair = k_cam = -1
    for entry in entries:
        camera = "camera_motion" in entry["dimension"]
        if camera:
            k_cam += 1; k = k_cam
        else:
            k_pair += 1; k = k_pair
        group = "camera" if camera else ("subject" if "i2v_subject" in entry["dimension"] else "background")
        if a.per_group and taken[group] >= a.per_group:
            continue
        taken[group] += 1
        sid = f"vbc_{k:03d}" if camera else f"vb_{k:03d}"
        prompt = entry["prompt_en"]
        first, width, height = bc.copy_first_frame((crop_dir / entry["image_name"]).resolve(), out / "first_frames", sid)
        info = meta.get(entry["image_name"], {})
        box = info.get("diff_ratio_crop", {}).get(a.ratio)
        if box and (int(box[2]), int(box[3])) != (width, height):
            raise SystemExit(f"{entry['image_name']}: {width}x{height} but the official {a.ratio} crop box is {box[2]}x{box[3]}")
        native = (width, height)
        if a.max_width and width > a.max_width:               # crops reach 5901x3372: SAM2's mask RLE overflows INT_MAX; the work frame is 448x256 anyway
            small = first.with_name(f"{sid}_w{a.max_width}.png")
            if not small.is_file():
                with Image.open(first) as image:
                    image.convert("RGB").resize((a.max_width, round(a.max_width * height / width)), Image.LANCZOS).save(small)
            first, width, height = small.resolve(), a.max_width, round(a.max_width * height / width)
        rows.append({"sample_id": sid, "scenario": entry["image_type"], "video_path": "", "first_frame_path": str(first),
                     "caption": " ".join(prompt.split()) if a.exact_prompt else bc.caption_for(prompt), "fps": bc.OUR_FPS, "frames": a.frames, "width": width, "height": height,
                     "bench": "vbench_i2v", "bench_id": prompt, "image_name": entry["image_name"], "image_type": entry["image_type"],
                     "group": group, "dimensions": [d for d in entry["dimension"] if d != "temporal_flickering"], "ratio": a.ratio,
                     "source_image": str((crop_dir / entry["image_name"]).resolve()), "benchmark_prompt": prompt,
                     "says_static": bc.says_static(prompt), "pair_index": k, "domain": "vbench_i2v",
                     "native_width": native[0], "native_height": native[1]})
    bc.write_manifest(rows, out)
    report = bc.check_manifest(out, len(rows))
    aspect = [round(r["width"] / r["height"], 3) for r in rows]
    report.update(pairs_in_suite=len(pairs), camera_motion_prompts=len(full) - len(pairs), groups=dict(taken),
                  categories=dict(collections.Counter(r["image_type"] for r in rows)), ratio=a.ratio,
                  aspect_min=min(aspect), aspect_max=max(aspect), work_frame_aspect=round(448 / 256, 3), frames=a.frames,
                  seconds=round((a.frames - 1) / bc.OUR_FPS, 3), caption_words_max=max(len(r["caption"].split()) for r in rows),
                  official_crop_boxes_checked=bool(meta), out=str(out))
    report.pop("sizes", None)
    (out / "manifest_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    if report["n_problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
