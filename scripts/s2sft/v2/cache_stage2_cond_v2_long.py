#!/usr/bin/env python
"""Stage-2 training / evaluation cache in the v2 condition contract (motion_jepa.cond_v2), one file per clip.

Evaluation-length form of scripts/cache_stage2_cond_v2.py, run with --max-frames 41 --pad-to 41 for the GT SFT (an
at-rest scene is extended with its last frame to the full 41-frame window).  One addition: --no-pred skips the frozen
Stage-1 export (cond_pred / known_frames_pred are not written; cond_zero and the objects still come from
plans/<sid>.npz): the GT SFT never reads cond_pred and the RAFT preparation re-renders pred / zero / null from its own
export.

For a real clip (manifest row: sample_id, video_path (any frame rate: frame k is read at k / fps seconds), first_frame_path, caption) with a
frozen Stage-1 export, its plan (plans/<sid>.npz) and real CoTracker tracks (real_tracks/<sid>.npz, track_real.py):

  video_latent / image_cond_latent  [16, T, 32, 56]   the real continuation and the first frame (Wan VAE, stretched to
                                                       448x256 exactly as the deployed sampler resizes its input image)
  cond_real   [27, T, 32, 56]   plan measured on the real clip (rigid fit to the tracks; training only)
  cond_pred   [27, T, 32, 56]   plan of the frozen Stage-1 model, every SAM2 instance, nothing removed
  cond_zero   [27, T, 32, 56]   the same objects explicitly at rest
  cond_null   [27, T, 32, 56]   no hint
  fm_weight   [T, 32, 56]       flow-matching weights (moving tubes x3, static objects x1.5, mean 1)
  labels_real {disp [K, T, 2], valid [K, T], moving [K, T], slots [K]}  measured motion of every plan object
  object_masks_latent [K, 32, 56], frame_seconds [T], prompt_embeds

Clip length: the largest 4k+1 window of the clip at 7 fps, at most --max-frames.  Most Physics 101 trials end within
about a second; a scene that has come to rest is extended with its last frame to --pad-to frames
(plan_condition.clip_window), a clip that is still moving when it ends is not.  Clips shorter than --min-frames are skipped and listed.

With --no-video (Physics-IQ and other deployment inputs) only cond_pred / cond_zero / cond_null and the image latent
are written, for --frames frames; nothing of a future video is read.

usage: cache_stage2_cond_v2_long.py --manifest M.jsonl --data DIR --output DIR [--model-dir WAN] [--prompt-table T.pt]
         [--exports DIR] [--no-pred]
       DIR holds plans/<sid>.npz, real_tracks/<sid>.npz and (without --no-pred) stage1/flows/<sid>.pt (the Stage-1
       exports; another folder with --exports).  --model-dir defaults to $WAN_MODEL_DIR; --cotracker-repo /
       --cotracker-checkpoint default to $COTRACKER_REPO / $COTRACKER_CKPT; the SAM 2.1 checkpoint is read from
       $SAM2_CHECKPOINT.
GT-SFT build of a real-clip source (P101 or SSv2):
       cache_stage2_cond_v2_long.py --manifest M.jsonl --data DIR --output WORK --max-frames 41 --pad-to 41 --no-pred
       followed by the label chain of docs/DATA_STAGE2.md (relabel, reconcile, relabel, re-render, apply_fields.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from motion_jepa import cond_v2  # noqa: E402
from scripts import motion_plan as mp  # noqa: E402
from scripts.cache_stage2_oracle import preprocess  # noqa: E402
from scripts.cache_stage2_phyco_v2 import PromptEncoder  # noqa: E402
from scripts.control import plan_condition as pc  # noqa: E402
from scripts.control import real_scene as rs  # noqa: E402
from scripts.control import reward as rw  # noqa: E402


def read_rows(path: str) -> list[dict]:
    if path.endswith(".csv"):
        import csv
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stretched_image(path: str, height: int, width: int) -> tuple[np.ndarray, tuple[int, int]]:
    """The conditioning image as the deployed sampler makes it (PIL LANCZOS resize to the work frame), and its native size."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image.resize((width, height), Image.LANCZOS), dtype=np.uint8), (image.height, image.width)


def video_rate(path: str) -> tuple[float, int]:
    capture = cv2.VideoCapture(path)
    rate, count = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return rate, count


def frames_available(path: str, fps: float) -> int:
    """How many frames at `fps` the video holds (frame k is the source frame nearest to k / fps seconds)."""
    rate, count = video_rate(path)
    if rate <= 0 or count <= 0:
        return 0
    return count if abs(rate - fps) < 0.01 else int(np.floor((count - 1) / rate * fps + 1e-6)) + 1


def read_frames(path: str, count: int, height: int, width: int, fps: float | None = None) -> np.ndarray:
    """The first `count` frames at `fps` (None or the file's own rate: its frames as they are; otherwise frame k is
    the source frame nearest to k / fps seconds), resized to the work frame."""
    rate, _ = video_rate(path)
    step = 1.0 if fps is None or rate <= 0 or abs(rate - fps) < 0.01 else rate / fps
    wanted = [int(np.floor(k * step + 0.5)) for k in range(count)]
    capture = cv2.VideoCapture(path)
    frames, index = [], 0
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        while len(frames) < count and wanted[len(frames)] == index:
            frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (width, height), interpolation=cv2.INTER_AREA))
        index += 1
    capture.release()
    if len(frames) < count:
        raise RuntimeError(f"{path}: {len(frames)} frames at {fps} fps, {count} needed")
    return np.stack(frames)


def real_anchors(horizons, times: list[float]) -> np.ndarray:
    """Anchors of the measured plan: the Stage-1 horizons inside the clip and every latent frame time (so the condition
    is exact at the latent frames)."""
    merged = sorted([float(h) for h in horizons if h <= times[-1] + 1e-6] + [float(t) for t in times[1:]])
    kept = []
    for value in merged:
        if not kept or value - kept[-1] > 1e-3:
            kept.append(value)
    return np.array(kept)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data", required=True, help="folder with plans, real_tracks (and stage1/flows without --no-pred)")
    p.add_argument("--exports", default="", help="folder with the Stage-1 exports <sid>.pt (default: DATA/stage1/flows)")
    p.add_argument("--output", required=True)
    p.add_argument("--model-dir", default=os.environ.get("WAN_MODEL_DIR", ""), required=not os.environ.get("WAN_MODEL_DIR"),
                   help="Wan2.2-I2V-A14B-Diffusers directory (default: $WAN_MODEL_DIR)")
    p.add_argument("--prompt-table", default="")
    p.add_argument("--prompt-length", type=int, default=512)
    p.add_argument("--fps", type=float, default=7.0)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--min-frames", type=int, default=5, help="5 frames = 2 latent frames: most Physics 101 falls are over in 0.6 s")
    p.add_argument("--max-frames", type=int, default=41)
    p.add_argument("--pad-to", type=int, default=41, help="a scene at rest is extended to at least this many frames (0 = never extend)")
    p.add_argument("--move-px", type=float, default=8.0)
    p.add_argument("--still-px", type=float, default=4.0)
    p.add_argument("--no-video", action="store_true", help="deployment inputs: pred / zero / null only")
    p.add_argument("--no-masktrack", action="store_true", help="object labels and the reward scene from point tracks only (no SAM2 pass)")
    p.add_argument("--no-pred", action="store_true", help="do not read the Stage-1 export: no cond_pred / known_frames_pred")
    p.add_argument("--cotracker-repo", default=os.environ.get("COTRACKER_REPO", ""), help="co-tracker checkout (default: $COTRACKER_REPO)")
    p.add_argument("--cotracker-checkpoint", default=os.environ.get("COTRACKER_CKPT", ""),
                   help="CoTracker3 scaled_offline.pth (default: $COTRACKER_CKPT)")
    p.add_argument("--frames", type=int, default=37, help="--no-video: frames to generate")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--claim", action="store_true", help="walk the whole manifest (rotated by --shard-index) and claim a clip with an "
                   "exclusive claims/<sid>.claim file (claims older than 20 min are taken over): any number of jobs may run")
    p.add_argument("--deadline", type=float, default=0, help="unix time after which no new clip is started")
    a = p.parse_args()
    from diffusers import AutoencoderKLWan

    rows = read_rows(a.manifest)
    if a.claim:
        start = (len(rows) * a.shard_index) // max(1, a.num_shards)
        rows = rows[start:] + rows[:start]
    else:
        rows = rows[a.shard_index:: a.num_shards]
    if a.limit:
        rows = rows[: a.limit]
    data, out = Path(a.data), Path(a.output)
    exports = Path(a.exports) if a.exports else data / "stage1" / "flows"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plans_real").mkdir(exist_ok=True)
    (out / "scenes").mkdir(exist_ok=True)
    (out / "claims").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKLWan.from_pretrained(Path(a.model_dir) / "vae", torch_dtype=torch.float32).to(device).eval()
    encoder = PromptEncoder(a.model_dir, a.prompt_length, a.prompt_table)
    instrument = None
    if not a.no_video and not a.no_masktrack:
        from scripts.control.measure import Instrument
        instrument = Instrument(a.cotracker_repo, a.cotracker_checkpoint)
    height, width = a.height, a.width
    size, latent_hw = (height, width), (height // 8, width // 8)
    done, skipped, short, errors = 0, 0, [], []
    for count, row in enumerate(rows):
        sid = row["sample_id"]
        target = out / f"{sid}.pt"
        if target.is_file() and not a.overwrite:
            skipped += 1
            continue
        if a.deadline and time.time() > a.deadline:
            print("[cond_v2] deadline reached", flush=True)
            break
        claim = out / "claims" / f"{sid}.claim"
        if a.claim:
            try:
                os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                try:
                    stale = time.time() - claim.stat().st_mtime > 1200
                except FileNotFoundError:
                    stale = False
                if not stale:
                    continue
                claim.touch()
            if target.is_file():
                continue
        try:
            export = None if a.no_pred else torch.load(exports / f"{sid}.pt", map_location="cpu", weights_only=False)
            first, native_hw = stretched_image(row["first_frame_path"], height, width)
            plan_path = data / "plans" / f"{sid}.npz"
            if export is None and not plan_path.is_file():
                raise FileNotFoundError(f"{plan_path} (needed with --no-pred)")
            plan_pred = mp.load_plan(plan_path) if plan_path.is_file() else mp.from_stage1_export(export, size, native_hw=native_hw)
            if tuple(plan_pred.size) != size:
                raise ValueError(f"plan is at {plan_pred.size}, the cache at {size}")
            payload = {"cache_format_version": cond_v2.CACHE_FORMAT, "sample_id": sid, "scenario": row.get("scenario", ""),
                       "source": row.get("domain", ""), "group_id": row.get("trial_id", sid), "split": row.get("split", ""),
                       "prompt": row["caption"], "fps": a.fps, "height": height, "width": width, "geometry": "stretch",
                       "condition_channels_v2": list(cond_v2.EXTRA_NAMES), "base_channels": cond_v2.BASE_CHANNELS,
                       "first_frame_path": row["first_frame_path"], "stage1_export": None if a.no_pred else str(exports / f"{sid}.pt"),
                       "plan_pred_summary": json.dumps(mp.describe(plan_pred)), "slots": [o.slot for o in plan_pred.objects]}
            if row.get("seed_key"):                     # optional: rows sharing a seed_key share the generator noise (infer_stage2_cond_v2.job_seed)
                payload["seed_key"] = str(row["seed_key"])
            if a.no_video:
                frames_used = frames_real = int(a.frames)
                times = pc.latent_times(frames_used, a.fps)
                frames = np.zeros((frames_used, height, width, 3), np.uint8)
                frames[0] = first
            else:
                track = np.load(data / "real_tracks" / f"{sid}.npz")
                tracks, visible, owner, track_times = track["tracks"].astype(np.float64), track["visible"], track["owner"], track["times"].astype(np.float64)
                at_rest = rs.at_rest(tracks, visible, owner, track_times)
                available = min(frames_available(row["video_path"], a.fps), int(np.floor(track_times[-1] * a.fps + 0.5)) + 1)
                frames_used, frames_real = pc.clip_window(available, at_rest and a.pad_to > 0, min_frames=a.min_frames, max_frames=a.max_frames, pad_to=a.pad_to)
                if frames_used == 0:
                    short.append({"sample_id": sid, "frames": available, "at_rest": bool(at_rest)})
                    continue
                times = pc.latent_times(frames_used, a.fps)
                clip = read_frames(row["video_path"], frames_real, height, width, a.fps)
                clip[0] = first
                frames = np.concatenate((clip, np.repeat(clip[-1:], frames_used - frames_real, 0))) if frames_used > frames_real else clip
            video = preprocess(frames, height, width).to(device)                       # [1, 3, F, H, W] in [-1, 1]
            image_video = torch.zeros_like(video)
            image_video[:, :, :1] = video[:, :, :1]
            with torch.no_grad():
                image_latent = vae.encode(image_video).latent_dist.mode()[0].float().cpu()
                video_latent = None if a.no_video else vae.encode(video).latent_dist.mode()[0].float().cpu()
            payload.update({"frames": frames_used, "frames_real": frames_real, "latent_frames": len(times), "frame_seconds": [round(t, 6) for t in times],
                            "image_cond_latent": image_latent.half(), "prompt_embeds": encoder(row["caption"]).cpu(),
                            "cond_zero": pc.render_v2("zero", plan_pred, image_latent, times, size).half(),
                            "cond_null": pc.render_v2("null", None, image_latent, times, size).half(),
                            "object_masks_pred_latent": pc.object_masks_latent(plan_pred, latent_hw).half()})
            if export is not None:
                payload.update({"cond_pred": pc.render_v2("pred", plan_pred, image_latent, times, size, export=export, native_hw=native_hw).half(),
                                "known_frames_pred": [bool(any(mp.affine_at(o, plan_pred.anchors, t)[1] for o in plan_pred.objects)) for t in times]})
            if not a.no_video:
                # the timeline of the clip that is trained on: real up to its last real frame, frozen while padded
                tracks, visible, track_times = rs.resample_clip(tracks, visible, track_times, (frames_real - 1) / a.fps, times[-1])
                stride = int(track["dense_stride"]) if "dense_stride" in track.files else 0
                count = len(track["extra_ids"]) if "extra_ids" in track.files else 0
                extra_masks = np.unpackbits(track["extra_masks"])[: count * height * width].reshape(count, height, width).astype(bool) if count else None
                plan_real, owner_real, info = rs.measured_plan(sid, plan_pred, tracks, visible, owner, track["points"].astype(np.float64), track_times,
                                                               real_anchors(plan_pred.anchors, times), extra_masks=extra_masks,
                                                               extra_ids=track["extra_ids"] if count else (), move_px=a.move_px, stride=stride or 16)
                labels = pc.labels_from_tracks(plan_real, tracks, visible, owner_real, track_times, times, move_px=a.move_px, still_px=a.still_px)
                masks_real = np.stack([o.mask for o in plan_real.objects]) if plan_real.objects else np.zeros((0, *size), bool)
                reading = {}
                if instrument is not None and len(masks_real):
                    # where every object is in the frames the generator is trained on (and a generated clip is measured the same way)
                    seen = instrument.measure(frames, track["points"], masks_real, with_points=False)
                    clip_times = np.arange(frames_used) / a.fps
                    reading = {"object_disp": seen["object_disp"], "object_ok": seen["object_ok"], "object_times": clip_times}
                    labels = pc.labels_with_masks(labels, seen["object_disp"], seen["object_ok"], clip_times, times, move_px=a.move_px, still_px=a.still_px)
                    plan_real, filled = rs.complete_with_masks(plan_real, seen["object_disp"], seen["object_ok"], clip_times)
                    info["anchors_filled_by_masks"] = int(filled)
                mp.save_plan(out / "plans_real" / f"{sid}.npz", plan_real)
                # the real side of the reward: what a generated clip of this length is compared with
                rw.save_scene(out / "scenes" / f"{sid}.npz", rw.build_scene(
                    track["points"], owner_real, [o.slot for o in plan_real.objects], masks_real, tracks, visible, track_times, times[-1], fps=a.fps, **reading))
                payload.update({"video_latent": video_latent.half(), "at_rest": bool(at_rest),
                                "cond_real": pc.render_v2("real", plan_real, image_latent, times, size).half(),
                                "fm_weight": pc.tube_weights(plan_real, labels, times, latent_hw).half(),
                                "object_masks_latent": pc.object_masks_latent(plan_real, latent_hw).half(),
                                "labels_real": {k: torch.from_numpy(v) for k, v in labels.items()},
                                "plan_real_summary": json.dumps(mp.describe(plan_real)), "real_scene": json.dumps(info),
                                "known_frames_real": [bool(any(mp.affine_at(o, plan_real.anchors, t)[1] for o in plan_real.objects)) for t in times],
                                "moving_objects": int(labels["moving"].any(1).sum()),
                                "label_source": "real_sam2_centroid+cotracker_v3" if reading else "real_cotracker_v2", "video_path": row["video_path"]})
                if stride:
                    disp, valid = rs.dense_field(tracks, visible, owner, track_times, times, (height // stride, width // stride))
                    payload["dense_real"] = {"disp": torch.from_numpy(disp).half(), "valid": torch.from_numpy(valid), "stride": stride}
            temporary = out / f"{sid}.tmp-{os.getpid()}"
            torch.save(payload, temporary)
            temporary.replace(target)
            claim.unlink(missing_ok=True)
            done += 1
        except Exception as error:  # noqa: BLE001
            errors.append({"sample_id": sid, "error": repr(error)[:300]})
            print(f"ERROR {sid}: {error!r}"[:400], flush=True)
        if (count + 1) % 100 == 0:
            print(f"[cond_v2] {count + 1}/{len(rows)} done {done} short {len(short)} errors {len(errors)}", flush=True)
    summary = {"done": done, "skipped": skipped, "short": len(short), "errors": errors[:20], "n_errors": len(errors)}
    (out / f"summary-shard{a.shard_index:03d}of{a.num_shards:03d}-{os.getpid()}.json").write_text(json.dumps({**summary, "short_clips": short}, indent=1) + "\n")
    print("COND_V2_DONE", json.dumps(summary))
    if errors and not a.claim:
        sys.exit(1)


if __name__ == "__main__":
    main()
