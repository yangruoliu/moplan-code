#!/usr/bin/env python
"""Our generated clips -> the layouts the external benchmarks score (CPU only).

Input: a generation folder of infer_stage2_cond_v2.py: <gen>/<arm>__<sid>[__s<k>].npy, uint8
[F, 256, 448, 3] at 7 fps, + <same>.json (fps, frames, seed).  The work frame is the conditioning image STRETCHED to
448x256 (cache_stage2_cond_v2.py geometry "stretch": PIL LANCZOS, no crop, no letterbox), so the inverse is a plain
stretch back to the benchmark image's native size (manifest width/height) -- same field of view, frame 0 = the VAE
reconstruction of the conditioning image (--first-frame conditioning pastes the exact image instead).

  rigidbench   predictions/<model>/<bench_id>.mp4 (00000.mp4 ... 00099.mp4), native size (1280x704), native 7 fps, all
               F frames (17 frames = 2.29 s >= the scored 2.0 s); RigidBench itself resizes to 1280x704 and aligns fps.
               Writes predictions/<model>.export.json beside the folder (not inside: the folder must hold only videos).
  morpheus     the raw tree morpheus-resize walks (cf. generation/Wan/run_sample_video.sh):
               <raw>/single_frame_conditioning/<model>/<experiment>_prompt_<plain|enhanced>/<experiment>_prompt_<type>_seed_<k>.mp4
               at 1280x1024 (already the real-world resolution; Morpheus' default branch of transform_frame is a plain
               cv2.resize to 1280x1024, i.e. the identity here) and 7 fps.  morpheus-resize detects the model by a
               keyword in the path, so the model must be registered first (morpheus-register); it then writes
               <resized>/single_frame_conditioning/<model>/<type>/<experiment>/<n>/{frames_for_tracking,output_video.mp4,info.json}.
               Morpheus scores on the frame index (not seconds); --target-frames N linearly re-times the clip to N
               frames (e.g. 81 = Wan's count) -- a cross-fade, off by default.
  morpheus-register   add model names to Morpheus' taxonomy.MODEL_NAMES (needed: --methods choices, and scoring
               treats any source not in MODEL_NAMES as REAL-WORLD footage with tighter windows) and to
               resizing/transforms.MODEL_KEYWORDS (path keyword -> name).  Idempotent text patch; see `git diff` there.
  fake-gen     a fake generation folder (the stretched first frame repeated F times) for CPU checks of the chain.
  vbench       the folder VBench-I2V's evaluator reads (vbench_standard mode): <root>/<model>/<prompt>-<k>.mp4, k = seed
               index 0..4 (clip <arm>__<sid>[__s<k>].npy), prompt = the manifest's bench_id (the VBench prompt).  Kept at the
               448x256 work frame (= the 7-4 crop downscaled; VBench resizes every frame itself and the 7-4 crops are ~4-6K)
               and at the native 7 fps with all F frames (VBench has no fixed fps: dynamic_degree samples every
               round(fps / 8)-th frame = every frame at 7 fps, motion_smoothness works on the frame index).  Encoded like
               VBench's sampling code: h264 (libx264, PyAV), crf 10, yuv420p.  The folder must hold only the videos (the
               evaluator takes the file suffix from the first listed file), so the record goes to <root>/<model>.export.json.

usage (WORK = your benchmark work root, MODEL = a model name, e.g. moplan-pred):
  export_videos.py rigidbench --gen G --manifest WORK/rigidbench/manifest.jsonl --arm pred --model MODEL \
      --predictions RIGIDBENCH_CHECKOUT/predictions
  export_videos.py morpheus-register --repo MORPHEUS_CHECKOUT --model MoPlan-pred MoPlan-null
  export_videos.py morpheus --gen G --manifest WORK/morpheus/manifest.jsonl --arm pred --model MoPlan-pred \
      --raw WORK/morpheus/gen_raw --num-seeds 3
  export_videos.py fake-gen --manifest M --gen G --arms pred,null [--num-seeds 1]
  export_videos.py vbench --gen G --manifest WORK/vbench/manifest.jsonl --arm pred --model MODEL \
      --root WORK/vbench/videos [--num-seeds 1]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

WORK_HW = (256, 448)
# keywords Morpheus already matches in a path (resizing/transforms.py) or uses as path levels: a model name containing
# one of them would be detected as another model / prompt type / experiment
MORPHEUS_RESERVED = ["kling-turbo", "wan", "cosmos", "ltx", "pyramid", "cogvideo", "veo3", "plain", "regular", "enhanced", "glm4",
                     "single_frame_conditioning", "multi_frame_conditioning", "keyframe_interpolation", "real-world",
                     "falling", "projectile", "pendulum", "bouncing", "rolling", "sliding", "spring", "collision"]


def read_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def clip_path(gen: Path, arm: str, sid: str, seed_index: int) -> Path:
    return gen / (f"{arm}__{sid}" + (f"__s{seed_index}" if seed_index else "") + ".npy")


def load_clip(gen: Path, arm: str, sid: str, seed_index: int) -> tuple[np.ndarray, float]:
    path = clip_path(gen, arm, sid, seed_index)
    frames = np.load(path)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (*WORK_HW, 3):
        raise ValueError(f"{path}: expected uint8 [F, 256, 448, 3], got {frames.dtype} {frames.shape}")
    meta = path.with_suffix(".json")
    fps = float(json.loads(meta.read_text())["fps"]) if meta.is_file() else 7.0
    return frames, fps


def to_native(frames: np.ndarray, width: int, height: int, first_image: str | None) -> list[np.ndarray]:
    """Undo the stretch: every work frame -> (width, height); optionally frame 0 := the exact conditioning image."""
    out = [np.asarray(Image.fromarray(frame).resize((width, height), Image.LANCZOS)) for frame in frames]
    if first_image:
        image = Image.open(first_image).convert("RGB")
        out[0] = np.asarray(image if image.size == (width, height) else image.resize((width, height), Image.LANCZOS))
    return out


def retime(frames: list[np.ndarray], count: int) -> list[np.ndarray]:
    positions = np.linspace(0, len(frames) - 1, count)
    result = []
    for position in positions:
        lower = int(np.floor(position)); upper = min(len(frames) - 1, lower + 1); weight = float(position - lower)
        result.append(cv2.addWeighted(frames[lower], 1 - weight, frames[upper], weight, 0))
    return result


def write_mp4(frames: list[np.ndarray], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    temporary = path.with_name(path.stem + ".tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"cannot create {temporary}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    temporary.replace(path)


def probe(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS)); count = 0; size = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        size = (frame.shape[1], frame.shape[0]); count += 1
    capture.release()
    return {"fps": fps, "frames": count, "size": size, "seconds": (count - 1) / fps if fps > 0 and count else 0.0}


def export_rigidbench(a) -> None:
    gen, rows = Path(a.gen), read_rows(a.manifest)
    folder = Path(a.predictions) / a.model
    records, missing = [], []
    for row in rows:
        try:
            frames, fps = load_clip(gen, a.arm, row["sample_id"], a.seed_index)
        except FileNotFoundError:
            missing.append(row["sample_id"]); continue
        if (len(frames) - 1) / fps + 1e-6 < 2.0:
            raise SystemExit(f"{row['sample_id']}: {len(frames)} frames at {fps:g} fps cover less than RigidBench's 2.0 s")
        native = to_native(frames, int(row["width"]), int(row["height"]), row["first_frame_path"] if a.first_frame == "conditioning" else None)
        target = folder / f"{row['bench_id']}.mp4"
        write_mp4(native, target, fps)
        info = probe(target)
        if info["frames"] != len(frames) or tuple(info["size"]) != (int(row["width"]), int(row["height"])):
            raise SystemExit(f"{target}: wrote {len(frames)} frames {row['width']}x{row['height']}, read back {info}")
        records.append({"sample_id": row["sample_id"], "bench_id": row["bench_id"], "video": str(target), **info})
    if missing and not a.allow_missing:
        raise SystemExit(f"{len(missing)} clips missing in {gen} for arm {a.arm} (e.g. {missing[:3]}); --allow-missing to export a partial set")
    record = {"bench": "rigidbench", "model": a.model, "gen": str(gen), "arm": a.arm, "seed_index": a.seed_index, "first_frame": a.first_frame,
              "geometry": "448x256 stretch -> native stretch (PIL LANCZOS)", "videos": len(records), "missing": missing, "clips": records}
    (folder.parent / f"{a.model}.export.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "clips"} | {"example": records[:1]}, indent=1))


def export_morpheus(a) -> None:
    gen, rows = Path(a.gen), read_rows(a.manifest)
    root = Path(a.raw) / "single_frame_conditioning" / a.model
    written, missing = [], []
    # official-rule morpho manifest (official_sp_inputs.py): one row per (experiment, prompt type, official seed), 1 clip each;
    # the Morpheus seed index of that clip = the rank of its official seed within the (experiment, prompt type) group
    rank = {}
    for row in rows:
        if "official_seed" in row:
            group = sorted(r["official_seed"] for r in rows if (r["experiment"], r["prompt_type"]) == (row["experiment"], row["prompt_type"]))
            rank[row["sample_id"]] = group.index(row["official_seed"])
    for row in rows:
        for k in range(a.num_seeds):
            try:
                frames, fps = load_clip(gen, a.arm, row["sample_id"], k)
            except FileNotFoundError:
                missing.append(f"{row['sample_id']}__s{k}"); continue
            native = to_native(frames, int(row["width"]), int(row["height"]), row["first_frame_path"] if a.first_frame == "conditioning" else None)
            if a.target_frames:
                native = retime(native, a.target_frames)
            leaf = f"{row['experiment']}_prompt_{row['prompt_type']}"
            target = root / leaf / f"{leaf}_seed_{rank.get(row['sample_id'], k)}.mp4"
            write_mp4(native, target, fps * (len(native) - 1) / max(len(frames) - 1, 1))
            written.append(str(target))
    if missing and not a.allow_missing:
        raise SystemExit(f"{len(missing)} clips missing in {gen} for arm {a.arm} (e.g. {missing[:3]}); --allow-missing to export a partial set")
    record = {"bench": "morpheus", "model": a.model, "gen": str(gen), "arm": a.arm, "num_seeds": a.num_seeds, "first_frame": a.first_frame,
              "target_frames": a.target_frames or "native", "videos": len(written), "missing": missing, "raw_root": str(root),
              "next": f"morpheus-resize --input-dir {a.raw} --output-dir <resized> --use-cache  (model {a.model} must be registered)"}
    (Path(a.raw) / f"{a.model}.export.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


def morpheus_register(a) -> None:
    repo = Path(a.repo)
    taxonomy, transforms = repo / "morpheus" / "taxonomy.py", repo / "morpheus" / "resizing" / "transforms.py"
    tax_text, tr_text = taxonomy.read_text(), transforms.read_text()
    current = re.findall(r'^    "([^"]+)",  # our model \(export_videos\.py\)$', tax_text, flags=re.M)
    for name in a.model:
        low = name.lower()
        clash = [w for w in MORPHEUS_RESERVED if w in low] + [c for c in current if c != name and (c.lower() in low or low in c.lower())]
        if clash or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise SystemExit(f"model name {name!r} clashes with Morpheus path keywords / registered names {clash} (or has odd characters)")
        if name not in current:
            tax_text = tax_text.replace("MODEL_NAMES = [\n", f'MODEL_NAMES = [\n    "{name}",  # our model (export_videos.py)\n', 1)
            current.append(name)
        line = f'    "{low}": "{name}",  # our model (export_videos.py); default transform = plain resize, the inverse of our stretch\n'
        if line not in tr_text:
            tr_text = tr_text.replace("MODEL_KEYWORDS = {\n", "MODEL_KEYWORDS = {\n" + line, 1)
    if "MODEL_NAMES = [\n" not in taxonomy.read_text() or "MODEL_KEYWORDS = {\n" not in transforms.read_text():
        raise SystemExit("Morpheus source changed: anchors MODEL_NAMES / MODEL_KEYWORDS not found")
    taxonomy.write_text(tax_text); transforms.write_text(tr_text)
    print(json.dumps({"registered": current, "taxonomy": str(taxonomy), "transforms": str(transforms)}, indent=1))


def write_mp4_h264(frames: list[np.ndarray], path: Path, fps: float, crf: int = 10) -> None:
    """VBench's sampling encoder (torchvision.io.write_video(..., video_codec='h264', options={'crf': '10'})), via PyAV."""
    import av
    from fractions import Fraction
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    temporary = path.with_name(path.stem + ".tmp.mp4")
    with av.open(str(temporary), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": str(crf)}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    temporary.replace(path)


def export_vbench(a) -> None:
    gen, rows = Path(a.gen), read_rows(a.manifest)
    folder = Path(a.root) / a.model
    folder.mkdir(parents=True, exist_ok=True)
    if a.resume:   # a job cut at its wall limit leaves only finished videos (written as .tmp.mp4, then renamed) + one .tmp
        for t in folder.glob("*.tmp.mp4"):
            t.unlink()
    records, missing = [], []
    for row in rows:
        for k in range(a.num_seeds):
            prompt = row["bench_id"]
            if "/" in prompt or not prompt.strip():
                raise SystemExit(f"{row['sample_id']}: prompt {prompt!r} cannot be a file name")
            target = folder / f"{prompt}-{k}.mp4"
            if a.resume and target.is_file() and clip_path(gen, a.arm, row["sample_id"], k).is_file():
                info = probe(target)   # written and checked by an earlier (cut) run of this export
                if info["frames"] != int(row["frames"]) or tuple(info["size"]) != (WORK_HW[1], WORK_HW[0]):
                    raise SystemExit(f"{target}: resumed video reads back {info}, expected {row['frames']} frames {WORK_HW[1]}x{WORK_HW[0]}")
                records.append({"sample_id": row["sample_id"], "seed_index": k, "video": str(target), "resumed": True, **info})
                continue
            try:
                frames, fps = load_clip(gen, a.arm, row["sample_id"], k)
            except FileNotFoundError:
                missing.append(f"{row['sample_id']}__s{k}"); continue
            native = to_native(frames, WORK_HW[1], WORK_HW[0], row["first_frame_path"] if a.first_frame == "conditioning" else None)
            write_mp4_h264(native, target, fps)
            info = probe(target)
            if info["frames"] != len(frames) or tuple(info["size"]) != (WORK_HW[1], WORK_HW[0]) or abs(info["fps"] - fps) > 0.01:
                raise SystemExit(f"{target}: wrote {len(frames)} frames {WORK_HW[1]}x{WORK_HW[0]} at {fps:g} fps, read back {info}")
            records.append({"sample_id": row["sample_id"], "seed_index": k, "video": str(target), **info})
    if missing and not a.allow_missing:
        raise SystemExit(f"{len(missing)} clips missing in {gen} for arm {a.arm} (e.g. {missing[:3]}); --allow-missing to export a partial set")
    stray = sorted(p.name for p in folder.iterdir() if p.suffix != ".mp4" or p.name not in {Path(r["video"]).name for r in records})
    record = {"bench": "vbench_i2v", "model": a.model, "gen": str(gen), "arm": a.arm, "num_seeds": a.num_seeds, "first_frame": a.first_frame,
              "geometry": "448x256 work frame kept (7:4 = the 7-4 crop downscaled)", "codec": "h264 libx264 crf 10 yuv420p (PyAV)",
              "videos": len(records), "missing": missing, "stray_files_in_folder": stray[:20], "folder": str(folder), "clips": records}
    (folder.parent / f"{a.model}.export.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "clips"} | {"example": records[:1]}, indent=1))
    if stray:
        raise SystemExit(f"{folder} holds {len(stray)} files that are not this export's videos (e.g. {stray[:3]}): the evaluator would read them")


def fake_gen(a) -> None:
    gen = Path(a.gen); gen.mkdir(parents=True, exist_ok=True)
    rows = read_rows(a.manifest)
    for row in rows:
        first = np.asarray(Image.open(row["first_frame_path"]).convert("RGB").resize((WORK_HW[1], WORK_HW[0]), Image.LANCZOS))
        clip = np.repeat(first[None], int(row["frames"]), 0)
        for arm in a.arms.split(","):
            for k in range(a.num_seeds):
                path = clip_path(gen, arm, row["sample_id"], k)
                np.save(path, clip)
                path.with_suffix(".json").write_text(json.dumps({"sample_id": row["sample_id"], "arm": arm, "seed_index": k, "frames": len(clip),
                                                                 "fps": float(row["fps"]), "fake": "repeated first frame"}) + "\n")
    print(json.dumps({"fake_gen": str(gen), "rows": len(rows), "arms": a.arms, "num_seeds": a.num_seeds}))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("rigidbench")
    m = sub.add_parser("morpheus")
    for q in (r, m):
        q.add_argument("--gen", required=True, help="generation folder (<arm>__<sid>[__s<k>].npy)")
        q.add_argument("--manifest", required=True)
        q.add_argument("--arm", default="pred")
        q.add_argument("--model", required=True)
        q.add_argument("--first-frame", choices=("generated", "conditioning"), default="generated")
        q.add_argument("--allow-missing", action="store_true")
    r.add_argument("--predictions", required=True, help="RigidBench predictions root; videos go to <root>/<model>/")
    r.add_argument("--seed-index", type=int, default=0)
    m.add_argument("--raw", required=True, help="root of the raw tree that morpheus-resize reads")
    m.add_argument("--num-seeds", type=int, default=3)
    m.add_argument("--target-frames", type=int, default=0)
    g = sub.add_parser("morpheus-register")
    g.add_argument("--repo", required=True)
    g.add_argument("--model", nargs="+", required=True)
    f = sub.add_parser("fake-gen")
    f.add_argument("--manifest", required=True)
    f.add_argument("--gen", required=True)
    f.add_argument("--arms", default="pred")
    f.add_argument("--num-seeds", type=int, default=1)
    v = sub.add_parser("vbench")
    v.add_argument("--gen", required=True, help="generation folder (<arm>__<sid>[__s<k>].npy)")
    v.add_argument("--manifest", required=True)
    v.add_argument("--arm", default="pred")
    v.add_argument("--model", required=True)
    v.add_argument("--root", required=True, help="videos go to <root>/<model>/<prompt>-<k>.mp4")
    v.add_argument("--num-seeds", type=int, default=1, help="seed indices 0..N-1 -> VBench video indices (official protocol: 5)")
    v.add_argument("--resume", action="store_true", help="keep videos already in the folder (still read back and checked)")
    v.add_argument("--first-frame", choices=("generated", "conditioning"), default="generated")
    v.add_argument("--allow-missing", action="store_true")
    a = p.parse_args(argv)
    {"rigidbench": export_rigidbench, "morpheus": export_morpheus, "morpheus-register": morpheus_register, "fake-gen": fake_gen,
     "vbench": export_vbench}[a.command](a)


if __name__ == "__main__":
    main()
