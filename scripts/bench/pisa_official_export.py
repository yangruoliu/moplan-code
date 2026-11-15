#!/usr/bin/env python
"""Our generated PISA clips -> the OFFICIAL PisaBench real-world evaluation (CPU only; numpy + PIL, no cv2).

Official code: github.com/vision-x-nyu/pisa-experiments @65866af (clone + venv: $PISA_HOME/code), data: HF dataset
nyu-visionx/pisa-experiments pisabench/real.zip (sha256 8006c913...).  README "PisaBench / Evaluation": convert the real
videos to the model's (square) resolution -> SAM 2 masks -> eval.py.
  GT      real/<cid>/{00000.jpg.., clip_info.json}: 1080x1080 frames at 120 fps, 118-362 frames (0.98-3.02 s), frame 0 = the
          wire release; clip_info.points = positive/negative clicks on frame 0 for EVERY object in view (1-19).
          data_processing/convert_resolution.py --is_real: frames -> RxR (PIL default resample), clicks int(p * R / 1080);
          the released configs use R = 256 (real_256).
  masks   data_processing/generate_mask.py: SAM 2.1 hiera-large (HF facebook/sam2.1-hiera-large, fp32), clicks on frame 0
          propagated over a %05d.jpg folder -> mask.npz [F, N_obj, R, R].  Run on the GT folder AND on the generated folder
          with the same (GT) clicks.
  pred    a folder of RxR %05d.jpg frames starting at the conditioning frame; process_results.py makes it from a generated
          mp4 with `ffmpeg -vf fps=16`, hence pd_fps 16 in configs/eval/real_example.jsonl (gt_fps 120).
  score   eval.py: pred frame k <-> GT frame int(k * 120 / pd_fps) while that index < len(GT) (= the overlap of the two
          durations); only the dropping object (the one whose positive click is highest, min y) is scored: centroid L2,
          Chamfer distance (both / R / sqrt 2) and mask IoU per clip, then nanmean over clips (clips that raise are skipped).
Our side (scripts/eval/pisa_clips.py -> Stage-2 work frame):
  space   the 1080x1080 frame sits at x 280..1000 of the 1280x720 canvas; the work frame stretches the canvas to 448x256
          (x * 0.35, y * 256/720), so the original square is exactly work columns 98..350 (252 x 256).  We crop it and resize
          to RxR with PIL BICUBIC (PIL's default for RGB = what convert_resolution.py applies to the GT): same field of view
          as the GT, one resampling.  The blurred side bands are dropped.
  time    the evaluation clips are 37 frames at 7 fps (5.14 s; any 4m+1 length is accepted).  --fps 16 (default, the
          released pd_fps): output frame j (t = j/16 s) is our frame nearest in time, floor(j*7/16 + 0.5), for
          j = 0..floor((F-1)*16/7) (37 -> 83 frames): no frame past our last one (ffmpeg's fps filter in process_results.py
          would hold the last frame 1/7 s longer).  --resample blend cross-fades the two neighbours instead; --fps 7 writes
          our frames unchanged (eval.py takes any pd_fps; GT frame int(k * 120/7)).
  horizon eval.py scores only the overlap, so a clip is scored on min(our seconds, GT seconds).  A 37-frame clip outlasts
          every GT clip (<= 3.02 s), so the overlap is the GT clip up to its last frame on the 16 fps grid, less than 1/16 s
          short of the GT end: coverage (scored seconds / GT seconds) >= 1 - (1/16) / GT seconds, below 1 only through
          this rounding.  export.json lists the per-clip coverage.  The PisaBench paper's baselines made 3-6 s clips and
          are scored on the whole GT clip.

layout ($PISA_HOME = the PisaBench working root, cid = official clip id = pisa_real_<cid>):
  $PISA_HOME/data/real_256/<cid>/{%05d.jpg, clip_info.json, mask.npz}       official GT (gt; masks by launch/eval_score_pisabench.sh)
  <out-root>/<model>/s<k>/{real/<cid>/%05d.jpg (+mask.npz), mask.jsonl, eval.jsonl, gt_ids.txt}   one "MODEL_DIR" per seed
  <out-root>/<model>/export.json, <out-root>/<model>/summary.json (mean over seeds)

usage:
  pisa_official_export.py gt --dirs MODEL_DIR...|--ids-file F|--manifest M [--zip $PISA_HOME/data/real.zip] [--out $PISA_HOME/data/real_256] [--workers 8]
  pisa_official_export.py export --gen GEN_DIR --manifest MANIFEST.jsonl \
      --arm pred --seeds 0 --model MODEL [--out-root $PISA_HOME/gen] [--fps 16] [--resample nearest|blend] [--ids-file F]
  pisa_official_export.py check MODEL_DIR...            preflight (read-only): frames, configs, GT folders, existing masks
  pisa_official_export.py plan-masks MODEL_DIR... --shards N --out DIR    GT + predicted mask jobs still missing (stale masks
                                                        deleted), longest-first onto N shards
  pisa_official_export.py summarize MODEL_DIR...       eval.py outputs -> MODEL_DIR/results/summary.json + <model>/summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

PISA_HOME = Path(os.environ.get("PISA_HOME", "pisa"))           # PisaBench working root: code/, data/real.zip, data/real_256, gen/
OFFICIAL_COMMIT = "65866af0f698fba7b3f38bc2e15bb044584dd08a"
REAL_ZIP_SHA256 = "8006c91383cf8c6c1283b727042081731897b0ca9ceda9f65aa65659cc7bc3ef"
GT_FPS, GT_SIDE = 120.0, 1080
WORK_HW = (256, 448)
CANVAS_W, CANVAS_H = 1280, 720                                         # pisa_clips.py
_LEFT = (CANVAS_W - CANVAS_H) // 2
SQUARE_COLUMNS = (_LEFT * WORK_HW[1] // CANVAS_W, (_LEFT + CANVAS_H) * WORK_HW[1] // CANVAS_W)     # (98, 350)
assert all(c * CANVAS_W % WORK_HW[1] == 0 for c in SQUARE_COLUMNS), "square edges must fall on work-frame columns"


def read_rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def cid_of(row: dict) -> str:
    """official clip id: the source folder name (raw_path .../real/<cid>), else the sample id's suffix (pisa_real_<cid>)."""
    cid = Path(row["raw_path"]).name if row.get("raw_path") else row["sample_id"].rsplit("_", 1)[1]
    if not (cid.isdigit() and len(cid) == 4):
        raise ValueError(f"{row.get('sample_id')}: no official 4-digit clip id ({cid!r})")
    return cid


def jpgs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.jpg"))


def gt_frame_counts(zip_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            parts = name.split("/")
            if len(parts) == 3 and parts[0] == "real" and parts[2].endswith(".jpg"):
                counts[parts[1]] = counts.get(parts[1], 0) + 1
    return counts


_MODULES: dict = {}


def official_module(code: Path, relative: str, name: str):
    if name in _MODULES:
        return _MODULES[name]
    spec = importlib.util.spec_from_file_location(name, code / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULES[name] = module
    return module


# ---------------------------------------------------------------- gt: official convert_resolution.py, clip by clip from the zip
def gt_one(job: tuple) -> dict:
    zip_path, code, cid, out, resolution, tmp_root = job
    dst = Path(out) / cid
    if (dst / "clip_info.json").is_file():                     # the official converter writes clip_info.json after all frames
        return {"cid": cid, "status": "cached", "frames": len(jpgs(dst))}
    convert = official_module(Path(code), "data_processing/convert_resolution.py", "pisa_convert_resolution")
    with zipfile.ZipFile(zip_path) as archive, tempfile.TemporaryDirectory(prefix=f"pisa_gt_{cid}_", dir=tmp_root) as tmp:
        members = [n for n in archive.namelist() if n.startswith(f"real/{cid}/") and (n.endswith(".jpg") or n.endswith("/clip_info.json"))]
        for member in members:
            archive.extract(member, tmp)
        source = Path(tmp) / "real" / cid
        if dst.exists():
            shutil.rmtree(dst)
        convert.process_single_video({"src_video": str(source), "dst_video": str(dst), "resolution": resolution}, True)
        expected = len(jpgs(source))
    got = len(jpgs(dst))
    if got != expected or not (dst / "clip_info.json").is_file():
        raise RuntimeError(f"{cid}: converted {got} of {expected} frames")
    return {"cid": cid, "status": "converted", "frames": got}


def run_gt(a) -> None:
    """GT clips -> <root>/<cid> with the official converter; roots come from --out, or from the MODEL_DIRs' mask.jsonl (--dirs)."""
    wanted: dict[str, set[str]] = {}
    if a.dirs:
        for d in seed_dirs(a.dirs):
            for row in read_rows(d / "mask.jsonl"):
                ann = Path(row["annotation"])
                wanted.setdefault(str(ann.parent.parent), set()).add(ann.parent.name)
    elif a.ids_file:
        wanted[a.out] = {line.strip().rsplit("_", 1)[-1] for line in Path(a.ids_file).read_text().split() if line.strip()}
    elif a.manifest:
        wanted[a.out] = {cid_of(r) for r in read_rows(a.manifest)}
    else:
        wanted[a.out] = set(gt_frame_counts(Path(a.zip)))
    jobs = []
    for root, cids in wanted.items():
        Path(root).mkdir(parents=True, exist_ok=True)
        jobs += [(str(a.zip), str(a.code), cid, root, a.resolution, a.tmp) for cid in sorted(cids)]
    with ProcessPoolExecutor(max(1, a.workers)) as pool:
        done = list(pool.map(gt_one, jobs))
    converted = [d for d in done if d["status"] == "converted"]
    print(json.dumps({"gt_roots": sorted(wanted), "clips": len(done), "converted": len(converted), "cached": len(done) - len(converted),
                      "frames": sum(d["frames"] for d in done), "resolution": a.resolution}))


# ---------------------------------------------------------------- export: our [F, 256, 448, 3] @ 7 fps -> official pred folders
def to_square(frame: np.ndarray, resolution: int) -> np.ndarray:
    left, right = SQUARE_COLUMNS
    return np.asarray(Image.fromarray(np.ascontiguousarray(frame[:, left:right])).resize((resolution, resolution), Image.BICUBIC))


def retime(frames: list[np.ndarray], fps_in: float, fps_out: float, mode: str) -> list[np.ndarray]:
    if abs(fps_in - fps_out) < 1e-9:
        return list(frames)
    count = int(math.floor((len(frames) - 1) * fps_out / fps_in + 1e-9)) + 1
    out = []
    for j in range(count):
        x = j * fps_in / fps_out
        if mode == "nearest":
            out.append(frames[min(len(frames) - 1, int(math.floor(x + 0.5)))])
        else:
            lower = int(math.floor(x)); upper = min(len(frames) - 1, lower + 1); w = x - lower
            out.append(((1 - w) * frames[lower].astype(np.float32) + w * frames[upper].astype(np.float32)).round().clip(0, 255).astype(np.uint8))
    return out


def scored_frames(frames_out: int, fps_out: float, gt_frames: int) -> int:
    """how many predicted frames eval.py keeps (target index int(k * 120 / pd_fps) < len(GT))."""
    return sum(1 for k in range(frames_out) if int(k * (GT_FPS / fps_out)) < gt_frames)


def export(a) -> None:
    gen, gt_root, out_root = Path(a.gen), Path(a.gt_root), Path(a.out_root)
    rows = read_rows(a.manifest)
    if a.ids_file:
        keep = {line.strip() for line in Path(a.ids_file).read_text().split() if line.strip()}
        rows = [r for r in rows if r["sample_id"] in keep]
    gt_counts = gt_frame_counts(Path(a.zip))
    seeds = [int(s) for s in str(a.seeds).split(",") if s.strip()]
    record = {"bench": "pisabench-real (official eval.py)", "official_commit": OFFICIAL_COMMIT, "model": a.model, "gen": str(gen), "arm": a.arm,
              "seeds": seeds, "resolution": a.resolution, "fps": a.fps, "resample": a.resample, "first_frame": a.first_frame, "jpeg_quality": a.jpeg_quality,
              "geometry": f"work columns {SQUARE_COLUMNS[0]}..{SQUARE_COLUMNS[1]} (the original square) -> {a.resolution}x{a.resolution} PIL BICUBIC",
              "gt_root": str(gt_root), "seed_dirs": {}}
    for k in seeds:
        folder = out_root / a.model / f"s{k}"
        clips, mask_rows, eval_rows = [], [], []
        path_of = {r["sample_id"]: gen / (f"{a.arm}__{r['sample_id']}" + (f"__s{k}" if k else "") + ".npy") for r in rows}
        missing = [sid for sid, path in path_of.items() if not path.is_file()]
        if missing and not a.allow_missing:
            raise SystemExit(f"{len(missing)} of {len(rows)} clips missing in {gen} for arm {a.arm} seed {k} (e.g. {missing[:3]}); "
                             "--ids-file to pick the panel's clips, --allow-missing to export a partial set")
        for row in rows:
            sid = row["sample_id"]
            path = path_of[sid]
            if not path.is_file():
                continue
            frames = np.load(path)
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (*WORK_HW, 3):
                raise SystemExit(f"{path}: expected uint8 [F, 256, 448, 3], got {frames.dtype} {frames.shape}")
            meta = path.with_suffix(".json")
            fps_in = float(json.loads(meta.read_text()).get("fps", 7.0)) if meta.is_file() else 7.0
            square = [to_square(f, a.resolution) for f in frames]
            if a.first_frame == "conditioning":
                canvas = Image.open(row["first_frame_path"]).convert("RGB")
                side, left = canvas.height, (canvas.width - canvas.height) // 2
                square[0] = np.asarray(canvas.crop((left, 0, left + side, side)).resize((a.resolution, a.resolution), Image.BICUBIC))
            out = retime(square, fps_in, a.fps, a.resample)
            cid = cid_of(row)
            video = folder / "real" / cid
            if video.exists():
                shutil.rmtree(video)
            video.mkdir(parents=True)
            for j, frame in enumerate(out):
                Image.fromarray(frame).save(video / f"{j:05d}.jpg", quality=a.jpeg_quality)
            gt_n = gt_counts.get(cid)
            if gt_n is None:
                raise SystemExit(f"{sid}: clip {cid} is not in {a.zip}")
            n_scored = scored_frames(len(out), a.fps, gt_n)
            ann = gt_root / cid / "clip_info.json"
            mask_rows.append({"video": str(video), "annotation": str(ann), "mask": str(video / "mask.npz")})
            eval_rows.append({"gt_mask": str(gt_root / cid / "mask.npz"), "gt_fps": int(GT_FPS), "gt_annotation": str(ann),
                              "pd_mask": str(video / "mask.npz"), "pd_fps": a.fps, "sample_id": sid})
            clips.append({"sample_id": sid, "cid": cid, "frames_in": len(frames), "fps_in": fps_in, "frames_out": len(out), "scored_frames": n_scored,
                          "gt_frames": gt_n, "seconds_scored": round((n_scored - 1) / a.fps, 4), "gt_seconds": round((gt_n - 1) / GT_FPS, 4),
                          "coverage": round(((n_scored - 1) / a.fps) / ((gt_n - 1) / GT_FPS), 4)})
        if not clips:
            raise SystemExit(f"nothing exported for seed {k}")
        write_rows(folder / "mask.jsonl", mask_rows)
        write_rows(folder / "eval.jsonl", eval_rows)
        (folder / "gt_ids.txt").write_text("".join(c["cid"] + "\n" for c in clips))
        cov = sorted(c["coverage"] for c in clips)
        record["seed_dirs"][f"s{k}"] = {"dir": str(folder), "videos": len(clips), "missing": missing,
                                        "coverage_min_median_mean": [cov[0], cov[len(cov) // 2], round(float(np.mean(cov)), 4)], "clips": clips}
    (out_root / a.model / "export.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "seed_dirs"} | {s: {kk: vv for kk, vv in d.items() if kk != "clips"} for s, d in record["seed_dirs"].items()}, indent=1))


# ---------------------------------------------------------------- check / plan-masks: preflight and mask job planning
def seed_dirs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "eval.jsonl").is_file():
            out.append(p)
        else:
            subs = sorted(d for d in p.glob("s*") if (d / "eval.jsonl").is_file())
            if not subs:
                raise SystemExit(f"{p}: neither a MODEL_DIR (eval.jsonl) nor a model folder with s<k>/eval.jsonl")
            out.extend(subs)
    return out


def npz_shape(path: Path) -> tuple:
    """shape of mask.npz["mask"] from the .npy header alone (a GT mask can be ~0.5 GB unpacked); a npz cut short by a
    killed job has no zip directory and raises."""
    with zipfile.ZipFile(path) as archive, archive.open("mask.npy") as handle:
        version = np.lib.format.read_magic(handle)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        return tuple(reader(handle)[0])


def mask_ok(mask: Path, video: Path, annotation: Path) -> str | None:
    """None if mask.npz matches its frame folder (frames, objects, square size), else the reason."""
    try:
        shape = npz_shape(mask)
    except Exception as error:                                   # noqa: BLE001  (truncated / corrupt npz)
        return f"unreadable ({type(error).__name__})"
    n_obj = len(json.loads(annotation.read_text())["points"])
    shape = shape if len(shape) == 4 else (shape[0], 1, *shape[1:])     # generate_mask.py squeezes a single object away
    frames = jpgs(video)
    side = Image.open(frames[0]).size[0] if frames else -1
    if shape[0] != len(frames) or shape[1] != n_obj or shape[2:] != (side, side):
        return f"shape {shape} vs {len(frames)} frames, {n_obj} objects, {side}px"
    return None


def mask_jobs(dirs: list[Path], force_pred: bool = False, delete: bool = False) -> tuple[list[dict], list[dict], list[str], list[str]]:
    """(GT jobs, predicted-mask jobs, fatal problems, notes).  A mask that does not match its folder counts as missing
    (and is deleted when `delete`, so a half-written npz of a killed job is redone)."""
    gt, pred, fatal, notes, seen = [], [], [], [], set()

    def stale(mask: Path, reason: str) -> None:
        if mask.is_file() and delete:
            mask.unlink()
        notes.append(f"{'deleted ' if delete else ''}{mask}: {reason}")

    for d in dirs:
        for row in read_rows(d / "mask.jsonl"):
            video, ann, mask = Path(row["video"]), Path(row["annotation"]), Path(row["mask"])
            gt_dir = ann.parent
            if not ann.is_file():
                fatal.append(f"GT not prepared: {ann}"); continue
            n = len(jpgs(video))
            if n < 2:
                fatal.append(f"{video}: {n} frames"); continue
            if str(gt_dir) not in seen:
                seen.add(str(gt_dir))
                gm = gt_dir / "mask.npz"
                reason = mask_ok(gm, gt_dir, ann) if gm.is_file() else "missing"
                if reason:
                    if reason != "missing":
                        stale(gm, reason)
                    gt.append({"video": str(gt_dir), "annotation": str(ann), "mask": str(gm), "frames": len(jpgs(gt_dir))})
            reason = ("forced" if force_pred else mask_ok(mask, video, ann)) if mask.is_file() else "missing"
            if reason:
                if reason != "missing":
                    stale(mask, reason)
                pred.append({"video": str(video), "annotation": str(ann), "mask": str(mask), "frames": n})
    return gt, pred, fatal, notes


def check(a) -> None:
    dirs = seed_dirs(a.dirs)
    report = {}
    for d in dirs:
        rows, evals = read_rows(d / "mask.jsonl"), read_rows(d / "eval.jsonl")
        sizes = {Image.open(jpgs(Path(r["video"]))[0]).size for r in rows if jpgs(Path(r["video"]))}
        report[str(d)] = {"videos": len(rows), "eval_rows": len(evals), "frame_sizes": sorted(sizes),
                          "gt_prepared": sum(Path(r["annotation"]).is_file() for r in rows),
                          "pred_masks": sum(Path(r["mask"]).is_file() for r in rows), "gt_masks": sum(Path(r["gt_mask"]).is_file() for r in evals)}
        if len(rows) != len(evals) or len(sizes) != 1:
            raise SystemExit(f"{d}: inconsistent export {report[str(d)]}")
    gt, pred, fatal, notes = mask_jobs(dirs)                     # read-only: nothing is deleted here
    print(json.dumps({"dirs": report, "gt_masks_todo": len(gt), "pred_masks_todo": len(pred), "fatal": fatal[:20], "n_fatal": len(fatal),
                      "stale_masks": notes[:20]}, indent=1))
    if a.strict and fatal:
        raise SystemExit(3)
    if a.require_masks and (gt or pred):
        raise SystemExit(4)


def plan_masks(a) -> None:
    gt, pred, fatal, notes = mask_jobs(seed_dirs(a.dirs), a.force_pred, delete=True)
    if fatal:
        raise SystemExit(f"cannot plan ({len(fatal)} problems): " + "; ".join(fatal[:5]))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("shard_*.jsonl"):
        old.unlink()
    jobs = sorted(gt + pred, key=lambda j: -j["frames"])
    loads, shards = [0] * a.shards, [[] for _ in range(a.shards)]
    for job in jobs:                                              # longest first onto the least loaded shard
        i = int(np.argmin(loads)); loads[i] += job["frames"]; shards[i].append({k: job[k] for k in ("video", "annotation", "mask")})
    for i, shard in enumerate(shards):
        if shard:
            write_rows(out / f"shard_{i}.jsonl", shard)
    print(json.dumps({"gt_jobs": len(gt), "pred_jobs": len(pred), "frames": int(sum(loads)), "shard_frames": loads, "notes": notes[:10]}))


# ---------------------------------------------------------------- summarize: eval.py outputs -> summary.json
def nanmean(values: list) -> float | None:
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    return None if arr.size == 0 or np.all(np.isnan(arr)) else round(float(np.nanmean(arr)), 5)


def summarize(a) -> None:
    by_model: dict[Path, list[dict]] = {}
    for d in seed_dirs(a.dirs):
        expected = read_rows(d / "eval.jsonl")
        out = d / "results" / "per_video.jsonl"
        got = read_rows(out) if out.is_file() else []
        scored = {r["config"]["pd_mask"]: r for r in got}
        failed = [e["sample_id"] for e in expected if e["pd_mask"] not in scored]
        rows = list(scored.values())
        s = {"dir": str(d), "videos": len(expected), "scored": len(rows), "failed": failed,
             "nan": {m: sum(1 for r in rows if r[m] is None or (isinstance(r[m], float) and math.isnan(r[m]))) for m in ("l2", "chamfer_distance", "iou")},
             "l2": nanmean([r["l2"] for r in rows]), "chamfer_distance": nanmean([r["chamfer_distance"] for r in rows]), "iou": nanmean([r["iou"] for r in rows]),
             "official_results_txt": (d / "results" / "results.txt").read_text() if (d / "results" / "results.txt").is_file() else None}
        (d / "results").mkdir(exist_ok=True)
        (d / "results" / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
        by_model.setdefault(d.parent, []).append(s)
    lines = []
    for model_dir, seeds in by_model.items():
        export_json = model_dir / "export.json"
        coverage = None
        if export_json.is_file():
            e = json.loads(export_json.read_text())
            coverage = {k: v["coverage_min_median_mean"] for k, v in e["seed_dirs"].items()}
        total = {"model": model_dir.name, "seeds": [Path(s["dir"]).name for s in seeds], "per_seed": {Path(s["dir"]).name: {m: s[m] for m in ("l2", "chamfer_distance", "iou", "scored", "videos")} | {"failed": len(s["failed"])} for s in seeds},
                 "mean_over_seeds": {m: nanmean([s[m] for s in seeds]) for m in ("l2", "chamfer_distance", "iou")},
                 "videos_per_seed": [s["videos"] for s in seeds], "failed_total": sum(len(s["failed"]) for s in seeds), "coverage_min_median_mean": coverage,
                 "protocol": "official eval.py (nanmean over clips; clips that raise are skipped -> failed); lower L2/Chamfer, higher IoU is better"}
        (model_dir / "summary.json").write_text(json.dumps(total, indent=1) + "\n")
        m = total["mean_over_seeds"]
        lines.append(f"{model_dir.name}: L2 {m['l2']} CD {m['chamfer_distance']} IoU {m['iou']} ({len(seeds)} seeds x {seeds[0]['videos']} clips, failed {total['failed_total']})")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gt")
    g.add_argument("--zip", default=str(PISA_HOME / "data/real.zip"))
    g.add_argument("--code", default=str(PISA_HOME / "code"))
    g.add_argument("--out", default=str(PISA_HOME / "data/real_256"))
    g.add_argument("--resolution", type=int, default=256)
    g.add_argument("--ids-file", help="sample ids (pisa_real_<cid>) or cids, whitespace separated")
    g.add_argument("--manifest")
    g.add_argument("--dirs", nargs="+", help="MODEL_DIRs: convert exactly the GT clips their mask.jsonl point to")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--tmp", default=os.environ.get("TMPDIR") or None)
    e = sub.add_parser("export")
    e.add_argument("--gen", required=True, help="generation folder (<arm>__<sid>[__s<k>].npy, uint8 [F, 256, 448, 3])")
    e.add_argument("--manifest", required=True, help="PISA manifest (sample_id, raw_path, first_frame_path)")
    e.add_argument("--arm", default="pred")
    e.add_argument("--seeds", default="0", help="seed indices, e.g. 0,1")
    e.add_argument("--model", required=True)
    e.add_argument("--ids-file")
    e.add_argument("--out-root", default=str(PISA_HOME / "gen"))
    e.add_argument("--gt-root", default=str(PISA_HOME / "data/real_256"))
    e.add_argument("--zip", default=str(PISA_HOME / "data/real.zip"))
    e.add_argument("--resolution", type=int, default=256)
    e.add_argument("--fps", type=float, default=16.0)
    e.add_argument("--resample", choices=("nearest", "blend"), default="nearest")
    e.add_argument("--first-frame", choices=("generated", "conditioning"), default="generated")
    e.add_argument("--jpeg-quality", type=int, default=95)
    e.add_argument("--allow-missing", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--strict", action="store_true", help="exit 3 on fatal problems (GT not prepared, empty folders)")
    c.add_argument("--require-masks", action="store_true", help="exit 4 if any GT / predicted mask is missing or stale")
    m = sub.add_parser("plan-masks")
    m.add_argument("dirs", nargs="+")
    m.add_argument("--shards", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--force-pred", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("dirs", nargs="+")
    a = p.parse_args()
    if a.command == "export" and a.fps == int(a.fps):
        a.fps = int(a.fps)
    {"gt": run_gt, "export": export, "check": check, "plan-masks": plan_masks, "summarize": summarize}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
