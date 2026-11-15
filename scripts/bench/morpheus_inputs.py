#!/usr/bin/env python
"""Morpheus -> our deployment manifest for single-frame conditioning (image-to-video), from the repo's prompts/ tree.

Not used by the released evaluation: launch/eval_bench_inputs.sh builds the Morpheus manifest with
official_sp_inputs.py morpho (Wan2.2's own extension of the plain prompt, sample ids morph_<exp>_<plain|enhanced>_<seed>;
see docs/EVALUATION.md).  This adapter is kept for reference.

Morpheus prompt files prompts/{plain,enhanced}/<experiment>.txt read "<prompt>@@<path>,<path>,...": the first path is
the conditioning frame prompts/reference_images/<experiment>/video_0/00000.jpg (video_5 for holonomic_pendulum),
1280x1024, the real-world camera.  The released Wan-2.1 split covers all 17 experiments x {plain, enhanced} x 3 seeds
(video_0..2), 81 frames each (Wan 2.1 default frame_num at 16 fps = 5.06 s).

  sample_id   morph_<experiment>_<k>, k = prompt type index: 0 = plain, 1 = enhanced (prompt_type keeps the name);
              seeds are generator seeds (NUM_SEEDS=3 -> video_0..2 like the Wan split)
  scenario    the Morpheus phenomenon bucket (freefall, projectile, collision, ...)
  caption     the prompt text; every Morpheus prompt ends with "Fixed camera view, no camera movement.", so
              " Static shot with no camera movement." is appended (bench_common.caption_for).  Note: Morpheus' Wan
              reference IGNORES the enhanced files and re-extends the plain prompt with its own Qwen2.5-VL extender; we
              feed the released enhanced text as is.  Stage-1 reads only the first 64 T5 tokens (prompt_pad_length).
  fps/frames  7 / --frames (default 37: Wan's 81 frames at 16 fps last 5.06 s -> 35.4 frames at 7 fps -> nearest 4k+1
              = 37 (5.14 s), also our Physics-IQ length; Morpheus scores on the frame index, so only the count matters)
  width/height  1280x1024 (export_videos.py + morpheus-resize stretch the 448x256 work frame back to it)

usage: morpheus_inputs.py --repo MORPHEUS_HOME --out DIR [--frames 37] [--experiments ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402

DEFAULT_REPO = os.environ.get("MORPHEUS_HOME", "")
# morpheus/taxonomy.py CANONICAL_EXPERIMENTS and PHENOMENON_MAP (copied: the taxonomy module is not importable from our env)
# from Morpheus (https://github.com/physics-from-video/Morpheus @ee3b6ae), Copyright (c) 2026 Antonios Tragoudaras, MIT License
CANONICAL = ["falling_ball", "falling_apple", "falling_marker", "falling_tape", "bouncing_ball", "projectile", "holonomic_pendulum",
             "non_holonomic_pendulum", "double_pendulum", "spring", "sliding_book", "rolling_full_can", "rolling_empty_can",
             "rolling_orange", "collision_equal", "collision_big_hits_small", "collision_small_hits_big"]
PHENOMENON = {"falling_ball": "freefall", "falling_apple": "freefall", "falling_tape": "freefall", "falling_marker": "freefall",
              "rolling_orange": "sliding_object", "rolling_empty_can": "sliding_object", "rolling_full_can": "sliding_object",
              "projectile": "projectile", "sliding_book": "sliding_object", "holonomic_pendulum": "pendulum", "bouncing_ball": "bouncingball",
              "non_holonomic_pendulum": "pendulum", "double_pendulum": "doublependulum", "collision_equal": "collision",
              "collision_small_hits_big": "collision", "collision_big_hits_small": "collision", "spring": "spring"}
PROMPT_TYPES = ["plain", "enhanced"]


def parse_prompt_file(path: Path, repo: Path) -> tuple[str, Path]:
    content = path.read_text().strip()
    if "@@" not in content:
        raise SystemExit(f"{path}: not 'prompt@@paths'")
    text, paths = content.split("@@", 1)
    first = [item.strip() for item in paths.split(",") if item.strip()][0]
    marker = "/prompts/"
    if marker not in first:
        raise SystemExit(f"{path}: cannot map {first} into the repo")
    return text.strip(), repo / "prompts" / first.split(marker, 1)[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO, required=not DEFAULT_REPO, help="Morpheus checkout (default: $MORPHEUS_HOME)")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=37)
    p.add_argument("--experiments", nargs="+", default=CANONICAL)
    p.add_argument("--prompt-types", nargs="+", default=PROMPT_TYPES, choices=PROMPT_TYPES)
    p.add_argument("--exact-prompt", action="store_true", help="caption = the benchmark prompt verbatim (whitespace normalised), no appended camera phrase (official-prompt runs)")
    a = p.parse_args()
    if (a.frames - 1) % 4:
        raise SystemExit(f"--frames {a.frames} is not 4k+1")
    repo, out = Path(a.repo), Path(a.out)
    rows = []
    for experiment in a.experiments:
        for prompt_type in a.prompt_types:
            k = PROMPT_TYPES.index(prompt_type)
            sid = f"morph_{experiment}_{k}"
            text, image = parse_prompt_file(repo / "prompts" / prompt_type / f"{experiment}.txt", repo)
            first, width, height = bc.copy_first_frame(image, out / "first_frames", sid)
            rows.append({"sample_id": sid, "scenario": PHENOMENON[experiment], "video_path": "", "first_frame_path": str(first),
                         "caption": " ".join(text.split()) if a.exact_prompt else bc.caption_for(text), "fps": bc.OUR_FPS, "frames": a.frames, "width": width, "height": height,
                         "bench": "morpheus", "bench_id": f"{experiment}/{prompt_type}", "experiment": experiment, "prompt_type": prompt_type,
                         "source_image": str(image.resolve()), "benchmark_prompt": text, "says_static": bc.says_static(text),
                         "conditioning": "single_frame_conditioning", "domain": "morpheus"})
    bc.write_manifest(rows, out)
    report = bc.check_manifest(out, len(a.experiments) * len(a.prompt_types))
    report.update(experiments=len(a.experiments), prompt_types=a.prompt_types, frames=a.frames, seconds=round((a.frames - 1) / bc.OUR_FPS, 3),
                  caption_words_max=max(len(r["caption"].split()) for r in rows), out=str(out))
    (out / "manifest_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    if report["n_problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
