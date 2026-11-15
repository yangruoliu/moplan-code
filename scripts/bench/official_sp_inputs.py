#!/usr/bin/env python
"""Benchmark manifests that follow each benchmark's official prompts, sample sets and negative prompts, for generation
under the MoPlan protocol (448x256 @ 7 fps).  Writes <out>/manifest.{jsonl,csv} + ids_all.txt (pisafull, morpho; rigido
and vbfull reuse the manifest written by rigidbench_inputs.py / vbench_inputs.py into <out>) and, also in <out>:
  NUM_SEEDS            videos per manifest row
  NEGATIVE_PROMPT.txt  the benchmark's official negative prompt (absent = Wan2.2's default negative prompt)
  PROMPT_TABLE.json    sample_id -> the prompt given to the VIDEO GENERATOR when it differs from the manifest caption
                       (the caption is what Stage-1 reads; Stage-1 was trained on English motion descriptions)
  SEED_MAJOR           (vbfull) generate seed 0 of every sample first

  pisafull  PisaBench, all 361 real clips (official App. H.2): caption "<object> falls. A video that conforms to the laws
            of physics.", negative "no camera motion, no slow-motion", 1 video per clip; first frame = the 1080^2 clip frame
            pillarboxed into 1280x720 (scripts/eval/pisa_clips.py; pisa_official_export.py crops it back)
  morpho    Morpheus single-frame split, 16 experiments x {plain, enhanced} x 3 seeds = 96: plain = prompts/plain/<exp>.txt
            verbatim; enhanced = Wan2.2's own Qwen2.5-VL prompt extension of it (Chinese, the official Wan route; the released
            enhanced/*.txt are not used); Wan2.2 default negative prompt.  One row per (experiment, type, official seed);
            Stage-1 reads the plain English prompt, the generator the official prompt.
  rigido    RigidBench: <out>/manifest.jsonl written by rigidbench_inputs.py --exact-prompt (scene prompt + official
            PROMPT_SUFFIX); this tool adds the official NEGATIVE_PROMPT (rigidbench.core.constants) and NUM_SEEDS 1
  vbfull    VBench-I2V: <out>/manifest.jsonl written by vbench_inputs.py --exact-prompt --with-camera (1118 entries); this
            tool adds NUM_SEEDS 5 and SEED_MAJOR
usage: official_sp_inputs.py pisafull --src pisabench_real.jsonl --first-frames DIR --out DIR
       official_sp_inputs.py morpho --src morpheus_plain.jsonl --src morpheus_enhanced.jsonl --morpheus-repo DIR --out DIR
       official_sp_inputs.py rigido --rigidbench-repo DIR --out DIR
       official_sp_inputs.py vbfull --out DIR
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402

PHENOMENON = {"falling_ball": "freefall", "falling_apple": "freefall", "falling_tape": "freefall", "falling_marker": "freefall",
              "rolling_orange": "sliding_object", "rolling_empty_can": "sliding_object", "rolling_full_can": "sliding_object",
              "projectile": "projectile", "sliding_book": "sliding_object", "holonomic_pendulum": "pendulum", "bouncing_ball": "bouncingball",
              "non_holonomic_pendulum": "pendulum", "double_pendulum": "doublependulum", "collision_equal": "collision",
              "collision_big_hits_small": "collision", "collision_small_hits_big": "collision", "spring": "spring"}


def work(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    return out


def pisafull(src: Path, first_frames: Path, out: Path) -> None:
    rows = []
    for rec in (json.loads(line) for line in src.read_text().splitlines() if line.strip()):
        sid = f"pisa_real_{int(rec['id']):04d}"
        first = first_frames / f"{sid}.png"
        rows.append({"sample_id": sid, "scenario": "pisa/drop", "video_path": "", "first_frame_path": str(first),
                     "caption": " ".join(rec["prompt"].split()), "fps": bc.OUR_FPS, "frames": 37, "width": 1280, "height": 720,
                     "bench": "pisabench", "bench_id": rec["id"], "domain": "pisa", "official_negative": rec["negative_prompt"]})
    negatives = {r.pop("official_negative") for r in rows}
    assert len(rows) == 361 and len(negatives) == 1, (len(rows), negatives)
    bc.write_manifest(rows, out)
    w = work(out)
    (w / "NUM_SEEDS").write_text("1\n")
    (w / "NEGATIVE_PROMPT.txt").write_text(negatives.pop())
    print(f"pisafull: {len(rows)} rows; negative {(w / 'NEGATIVE_PROMPT.txt').read_text()!r}")


def morpho(srcs: list[Path], morph_repo: Path, out: Path) -> None:
    rows, table = [], {}
    for src in srcs:
        for rec in (json.loads(line) for line in src.read_text().splitlines() if line.strip()):
            kind = "enhanced" if "_enhanced_" in rec["id"] else "plain"
            exp, seed = rec["exp"], int(rec["seed"])
            image = morph_repo / re.sub(r"^.*?/prompts/", "prompts/", rec["image"])
            sid = f"morph_{exp}_{kind}_{seed}"
            rows.append({"sample_id": sid, "scenario": PHENOMENON[exp], "video_path": "", "first_frame_path": str(image),
                         "caption": " ".join(rec.get("plain_prompt", rec["prompt"]).split()), "fps": bc.OUR_FPS, "frames": 37, "width": 1280, "height": 1024,
                         "bench": "morpheus", "bench_id": f"{exp}_prompt_{kind}", "experiment": exp, "prompt_type": kind,
                         "official_seed": seed, "generator_prompt": rec["prompt"], "extend_status": rec.get("extend_status")})
            table[sid] = rec["prompt"]
    assert len(rows) == 96, len(rows)
    bc.write_manifest(rows, out)
    w = work(out)
    (w / "NUM_SEEDS").write_text("1\n")
    (w / "PROMPT_TABLE.json").write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n")
    print(f"morpho: {len(rows)} rows ({sum(r['prompt_type'] == 'enhanced' for r in rows)} enhanced)")


def rigido(rigid_repo: Path, out: Path) -> None:
    text = (rigid_repo / "src/rigidbench/core/constants.py").read_text()
    scope: dict = {}
    exec(compile(text, "constants.py", "exec"), scope)            # plain string constants, no imports needed
    w = work(out)
    (w / "NUM_SEEDS").write_text("1\n")
    (w / "NEGATIVE_PROMPT.txt").write_text(scope["NEGATIVE_PROMPT"])
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines() if line.strip()]
    bad = [r["sample_id"] for r in rows if not r["caption"].endswith(scope["PROMPT_SUFFIX"].strip())]
    assert len(rows) == 100 and not bad, (len(rows), bad[:3])
    print(f"rigido: {len(rows)} rows end with the official suffix; negative {len(scope['NEGATIVE_PROMPT'])} chars")


def vbfull(out: Path) -> None:
    w = work(out)
    (w / "NUM_SEEDS").write_text("5\n")
    (w / "SEED_MAJOR").write_text("")
    n = sum(1 for _ in (out / "manifest.jsonl").open())
    assert n == 1118, n
    print(f"vbfull: {n} rows x 5 videos")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bench", choices=("pisafull", "morpho", "rigido", "vbfull"))
    p.add_argument("--src", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True, help="benchmark input folder (manifest + generation settings)")
    p.add_argument("--first-frames", type=Path, default=None, help="pisafull: folder of pisa_real_<id>.png (pisa_clips.py)")
    p.add_argument("--morpheus-repo", type=Path, default=None, help="morpho: Morpheus checkout (prompts/reference_images)")
    p.add_argument("--rigidbench-repo", type=Path, default=None, help="rigido: RigidBench checkout (src/rigidbench/core/constants.py)")
    a = p.parse_args()
    need = {"pisafull": ("first_frames",), "morpho": ("morpheus_repo",), "rigido": ("rigidbench_repo",), "vbfull": ()}[a.bench]
    for key in need:
        if getattr(a, key) is None:
            p.error(f"{a.bench} needs --{key.replace('_', '-')}")
    {"pisafull": lambda: pisafull(a.src[0], a.first_frames, a.out), "morpho": lambda: morpho(a.src, a.morpheus_repo, a.out),
     "rigido": lambda: rigido(a.rigidbench_repo, a.out), "vbfull": lambda: vbfull(a.out)}[a.bench]()
