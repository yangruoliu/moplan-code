#!/usr/bin/env python
"""RigidBench -> our deployment manifest (first frame + caption only; nothing of the reference rollout is read).

Reads RigidBench's data/inputs.jsonl (written by `uv run rigidbench download`; fields id, image, scene_prompt, prompt,
negative_prompt, duration_seconds=2.0, reference_fps=24, reference_frames=49, reference_resolution=[1280, 704], task,
protocol) and writes <out>/manifest.{jsonl,csv} + ids_all.txt + first_frames/ (byte-exact copies).

  sample_id   rigid_<id> (RigidBench ids are 00000 ... 00099, so rigid_00000 ... rigid_00099); bench_id keeps the id
  scenario    the RigidBench task
  caption     --caption-mode full (default, what the RigidBench protocol prescribes): the COMPLETE prompt
              (scene prompt + RigidBench's fixed suffix, which already says "Static shot, locked-off camera ... No camera
              movement"), plus " Static shot with no camera movement." because the prompt states a static camera
              (bench_common.caption_for).  --caption-mode scene: scene prompt + the same rule (non-protocol ablation).
              Stage-1 reads only the first 64 T5 tokens (prompt_pad_length), Stage-2 up to 512.
  fps/frames  7 / --frames (default 17: 4k+1 at 7 fps covering t = 0 ... 2.0 s needs (F - 1) / 7 >= 2.0 -> F = 17, 2.29 s)
  width/height  native size of the conditioning image (export_videos.py stretches the 448x256 work frame back to it)
RigidBench's negative_prompt is kept in the jsonl.  The official negative prompt reaches the sampler through
NEGATIVE_PROMPT.txt, which official_sp_inputs.py rigido writes (rigidbench.core.constants) and
launch/eval_bench_generate.sh passes on; without that file the sampler uses Wan2.2's default negative prompt.

usage: rigidbench_inputs.py --inputs RIGIDBENCH_HOME/data/inputs.jsonl --out DIR [--frames 17]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402

DEFAULT_INPUTS = str(Path(os.environ["RIGIDBENCH_HOME"]) / "data/inputs.jsonl") if os.environ.get("RIGIDBENCH_HOME") else ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", default=DEFAULT_INPUTS, required=not DEFAULT_INPUTS, help="RigidBench data/inputs.jsonl (default: $RIGIDBENCH_HOME/data/inputs.jsonl)")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--caption-mode", choices=("full", "scene"), default="full")
    p.add_argument("--expected", type=int, default=100)
    p.add_argument("--exact-prompt", action="store_true", help="caption = the benchmark prompt verbatim (whitespace normalised), no appended camera phrase (official-prompt runs)")
    a = p.parse_args()
    if (a.frames - 1) % 4 or (a.frames - 1) / bc.OUR_FPS < 2.0:
        raise SystemExit(f"--frames {a.frames}: must be 4k+1 and cover 2.0 s at {bc.OUR_FPS:g} fps")
    inputs, out = Path(a.inputs), Path(a.out)
    examples = [json.loads(line) for line in inputs.read_text().splitlines() if line.strip()]
    rows = []
    for example in examples:
        bench_id = str(example["id"])
        sid = f"rigid_{bench_id}" if bench_id.isdigit() else f"rigid_{len(rows):05d}"
        image = Path(example["image"])
        image = image if image.is_absolute() else (inputs.parent / image)
        first, width, height = bc.copy_first_frame(image.resolve(), out / "first_frames", sid)
        prompt = example["prompt"] if a.caption_mode == "full" else example["scene_prompt"]
        rows.append({"sample_id": sid, "scenario": example.get("task", ""), "video_path": "", "first_frame_path": str(first),
                     "caption": " ".join(prompt.split()) if a.exact_prompt else bc.caption_for(prompt), "fps": bc.OUR_FPS, "frames": a.frames, "width": width, "height": height,
                     "bench": "rigidbench", "bench_id": bench_id, "source_image": str(image.resolve()), "task": example.get("task", ""),
                     "caption_mode": a.caption_mode, "benchmark_prompt": example["prompt"], "scene_prompt": example.get("scene_prompt", ""),
                     "benchmark_negative_prompt": example.get("negative_prompt", ""), "says_static": bc.says_static(prompt),
                     "reference_fps": example.get("reference_fps"), "reference_resolution": example.get("reference_resolution"),
                     "duration_seconds": example.get("duration_seconds"), "domain": "rigidbench"})
    bc.write_manifest(rows, out)
    report = bc.check_manifest(out, a.expected)
    tasks: dict = {}
    for row in rows:
        tasks[row["task"]] = tasks.get(row["task"], 0) + 1
    report.update(tasks=tasks, frames=a.frames, seconds=round((a.frames - 1) / bc.OUR_FPS, 3), caption_mode=a.caption_mode,
                  caption_words_max=max(len(r["caption"].split()) for r in rows), out=str(out))
    (out / "manifest_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    if report["n_problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
