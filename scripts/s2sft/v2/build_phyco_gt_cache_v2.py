#!/usr/bin/env python
"""PhyCo train clips -> Stage-2 cond_v2 (27-channel, legacy27) cache with the ground-truth plan, in ONE GPU pass,
at evaluation length: a 41-frame window at 7 fps (11 latent frames, 5.71 s).

The renders hold 97-98 frames at 24 fps (4.04 s), i.e. 29 real frames at 7 fps.  As in the real-clip builder
(plan_condition.clip_window, pad_to = --frames): a clip whose scene has come to rest by its end is extended with its
last real frame to --frames frames (the world stays as it is: the true continuation), and its ground-truth plan is
HELD -- the first-to-future field of the last horizon is repeated at every later latent time, so those anchors are
known.  At rest = every instance's median motion between the last two ground-truth horizons (0.29 s) is below
--rest-px work pixels (the real-clip rule: 2 px over the last 0.25 s).  A clip still moving at its end keeps its 29
real frames (8 latents) -- then the payload equals the 29-frame build.  A manifest row may carry `v1_payload` (a
29-frame payload of the same clip): its prompt is used, exactly.

Definitions (the same as the 29-frame PhyCo payloads, which were made in two steps):
  video_latent / image_cond_latent / prompt_embeds   frames at k / 7 s from the 24 fps source (read_letterboxed:
                                768x432 letterboxed into 448x256), Wan2.2 VAE mode, umT5 prompt
  cond_real = the ground-truth plan (one rigid transform per rendered instance fitted to the dataset's first-to-future
                                field at each target horizon: motion_plan.from_flow_grid), cond_null, tube weights,
                                object masks, labels -- rendered through plan_condition.render_v2 from the
                                HALF-precision image latent
Here both steps run per clip.  cond_pred is NOT written (there is no Stage-1 export for these clips and GT SFT never
reads it); cond_zero is rendered from the ground-truth objects only with --with-zero.

usage: build_phyco_gt_cache_v2.py --manifest M.jsonl --output DIR [--model-dir WAN] [--frames 41]
         [--shard-index i --num-shards n | --claim]
       manifest rows (make_manifests.py): sample_id, caption, video_path, stage1_target_path [, scenario, v1_payload]
       --model-dir defaults to $WAN_MODEL_DIR (only the VAE and the umT5 text encoder are loaded).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from motion_jepa import cond_v2  # noqa: E402
from scripts import motion_plan as mp  # noqa: E402
from scripts.control import plan_condition as pc  # noqa: E402

NATIVE_HW = (432, 768)
BUILDER = "s2sft/v2/build_phyco_gt_cache_v2.py v2"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-dir", default=os.environ.get("WAN_MODEL_DIR", ""), required=not os.environ.get("WAN_MODEL_DIR"),
                   help="Wan2.2-I2V-A14B-Diffusers directory (default: $WAN_MODEL_DIR)")
    p.add_argument("--prompt-length", type=int, default=512)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--frames", type=int, default=41, help="window (4k+1) at --fps; an at-rest clip is padded to it")
    p.add_argument("--rest-px", type=float, default=2.0)
    p.add_argument("--no-pad", action="store_true", help="never extend (the 29-frame behaviour at the real length)")
    p.add_argument("--source-fps", type=float, default=24.0)
    p.add_argument("--fps", type=float, default=7.0)
    p.add_argument("--move-px", type=float, default=8.0)
    p.add_argument("--still-px", type=float, default=4.0)
    p.add_argument("--with-zero", action="store_true")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--claim", action="store_true", help="every process walks the whole manifest (rotated by --shard-index) and "
                   "claims a clip with an exclusive <sid>.claim file; claims older than 20 min are taken over")
    p.add_argument("--deadline", type=float, default=0, help="unix time after which no new clip is started")
    a = p.parse_args()
    import cv2
    from diffusers import AutoencoderKLWan
    from scripts.cache_stage2_phyco import preprocess, read_letterboxed
    from scripts.cache_stage2_phyco_v2 import PromptEncoder

    rows = [json.loads(line) for line in Path(a.manifest).read_text().splitlines() if line.strip()]
    if a.claim:
        start = (len(rows) * a.shard_index) // max(1, a.num_shards)
        rows = rows[start:] + rows[:start]
    else:
        rows = rows[a.shard_index:: a.num_shards]
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.output)
    (out / "plans_real").mkdir(parents=True, exist_ok=True)
    claims = out / "claims"
    claims.mkdir(exist_ok=True)
    for stale in out.glob(f"*.tmp-*"):
        if stale.stat().st_mtime < time.time() - 600:
            stale.unlink(missing_ok=True)
    device = torch.device("cuda")
    vae = AutoencoderKLWan.from_pretrained(Path(a.model_dir) / "vae", torch_dtype=torch.float32).to(device).eval()
    encode_prompt = PromptEncoder(a.model_dir, a.prompt_length)
    size, latent_hw = (a.height, a.width), (a.height // 8, a.width // 8)
    done = skipped = 0
    errors, started = [], time.time()
    print(f"[phyco gt] shard {a.shard_index}/{a.num_shards}: {len(rows)} clips -> {out}", flush=True)
    for count, row in enumerate(rows):
        sid = row["sample_id"]
        target_path = out / f"{sid}.pt"
        if target_path.is_file() and not a.overwrite:
            skipped += 1
            continue
        if a.deadline and time.time() > a.deadline:
            print("[phyco gt] deadline reached", flush=True)
            break
        claim = claims / f"{sid}.claim"
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
            if target_path.is_file():
                continue
        try:
            capture = cv2.VideoCapture(str(row["video_path"]))
            native = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            capture.release()
            if native != NATIVE_HW:
                raise ValueError(f"native size {native} != {NATIVE_HW}")
            n_video = int(cv2.VideoCapture(str(row["video_path"])).get(cv2.CAP_PROP_FRAME_COUNT))
            available = int(np.floor((n_video - 1) / a.source_fps * a.fps + 1e-6)) + 1
            frames_real = pc.largest_4k1(available, a.frames)
            truth = torch.load(row["stage1_target_path"], map_location="cpu", weights_only=False)
            side = int(truth["flow"].shape[-1])
            grid_box = mp.inner_box(side, side, *NATIVE_HW)
            ids_map = truth["instance_id"][0].numpy()
            instance_ids = [int(v) for v in np.unique(ids_map) if v > 0]
            flow = truth["flow"].float().numpy()
            horizons = [float(v) for v in truth["horizon_seconds"]]
            top, left, grid_h, grid_w = grid_box
            scale = np.array([a.width / grid_w, a.height / grid_h])
            inside = np.zeros(ids_map.shape, bool)
            inside[top: top + grid_h, left: left + grid_w] = True
            rest_motion = 0.0                                         # largest per-instance median motion over the last horizon step
            for v in instance_ids:
                sel = (ids_map == v) & inside
                if sel.sum() >= 12:
                    step = (flow[-1][:, sel] - flow[-2][:, sel]).T * scale
                    rest_motion = max(rest_motion, float(np.median(np.hypot(step[:, 0], step[:, 1]))))
            at_rest = bool(rest_motion < a.rest_px)
            frames_used = a.frames if (at_rest and not a.no_pad and frames_real < a.frames) else frames_real
            seconds = [k / a.fps for k in range(frames_real)]
            frames = read_letterboxed(Path(row["video_path"]), a.height, a.width, seconds, a.source_fps)
            if frames_used > frames_real:
                frames = np.concatenate((frames, np.repeat(frames[-1:], frames_used - frames_real, 0)))
            times = pc.latent_times(frames_used, a.fps)
            if times[-1] > horizons[-1] + 1e-6 and frames_used > frames_real:      # held plan beyond the last horizon
                extra = [t for t in times if t > horizons[-1] + 1e-6]
                flow = np.concatenate((flow, np.repeat(flow[-1:], len(extra), 0)))
                horizons = horizons + extra
            decode = {"frames_sha1": hashlib.sha1(frames.tobytes()).hexdigest(), "opencv": cv2.__version__,
                      "mean_rgb": [round(float(v), 4) for v in frames.reshape(-1, 3).mean(0)]}
            with torch.no_grad():
                video = preprocess(frames).to(device)
                video_latent = vae.encode(video).latent_dist.mode()[0].cpu().half()
                image_video = torch.cat((video[:, :, :1], video.new_zeros(*video.shape[:2], video.shape[2] - 1, *video.shape[3:])), dim=2)
                image_cond_latent = vae.encode(image_video).latent_dist.mode()[0].cpu().half()
            if image_cond_latent.shape[1] != len(times):
                raise ValueError(f"{image_cond_latent.shape[1]} latent frames, expected {len(times)}")
            prompt = row["caption"]
            if row.get("v1_payload"):
                prompt = torch.load(row["v1_payload"], map_location="cpu", weights_only=False)["prompt"]
            prompt_embeds = encode_prompt(prompt)
            image_latent = image_cond_latent.float()                      # the half-precision latent, read back as float
            masks = np.stack([ids_map == v for v in instance_ids]) if instance_ids else np.zeros((0, side, side), bool)
            plan_real = mp.from_flow_grid(sid, flow, masks, horizons, size, grid_box,
                                          source="phyco_ground_truth")
            for obj in plan_real.objects:
                obj.slot = instance_ids[obj.slot]
            labels = pc.labels_from_plan(plan_real, times, move_px=a.move_px, still_px=a.still_px)
            scenario = row.get("scenario") or truth.get("scenario", "")
            payload = {"cache_format_version": cond_v2.CACHE_FORMAT, "sample_id": sid, "scenario": scenario, "source": "phyco",
                       "group_id": sid, "split": "train", "prompt": prompt, "prompt_embeds": prompt_embeds,
                       "fps": a.fps, "height": a.height, "width": a.width, "geometry": "content_box_v2+stretch_condition",
                       "condition_channels_v2": list(cond_v2.EXTRA_NAMES), "base_channels": cond_v2.BASE_CHANNELS,
                       "frames": frames_used, "frames_real": frames_real, "latent_frames": len(times), "frame_seconds": [round(t, 6) for t in times],
                       "video_latent": video_latent, "image_cond_latent": image_cond_latent,
                       "cond_real": pc.render_v2("real", plan_real, image_latent, times, size).half(),
                       "cond_null": pc.render_v2("null", None, image_latent, times, size).half(),
                       "fm_weight": pc.tube_weights(plan_real, labels, times, latent_hw).half(),
                       "object_masks_latent": pc.object_masks_latent(plan_real, latent_hw).half(),
                       "labels_real": {k: torch.from_numpy(v) for k, v in labels.items()},
                       "slots": [o.slot for o in plan_real.objects], "moving_objects": int(labels["moving"].any(1).sum()),
                       "plan_real_summary": json.dumps(mp.describe(plan_real)),
                       "label_source": "phyco_ground_truth_flow_rigid_v1", "at_rest": at_rest, "rest_motion_px": round(rest_motion, 4),
                       "pad_rule": f"at_rest_hold_last(pad_to={a.frames}, rest_px={a.rest_px})", "plan_anchors": [round(float(h), 6) for h in horizons],
                       "known_frames_real": [True] * len(times), "video_path": row["video_path"], "target_path": Path(row["stage1_target_path"]).name,
                       "decode": decode, "builder": BUILDER, "conditions_present": ["real", "null"] + (["zero"] if a.with_zero else [])}
            if a.with_zero:
                payload["cond_zero"] = pc.render_v2("zero", plan_real, image_latent, times, size).half()
            for key in ("video_latent", "image_cond_latent", "cond_real", "cond_null"):
                if not bool(torch.isfinite(payload[key].float()).all()):
                    raise FloatingPointError(f"nonfinite {key}")
            mp.save_plan(out / "plans_real" / f"{sid}.npz", plan_real)
            temporary = out / f"{sid}.tmp-{os.getpid()}"
            torch.save(payload, temporary)
            temporary.replace(target_path)
            claim.unlink(missing_ok=True)
            done += 1
        except Exception as error:  # noqa: BLE001
            errors.append({"sample_id": sid, "error": repr(error)[:300]})
            print(f"ERROR {sid}: {error!r}"[:400], flush=True)
        if (count + 1) % 25 == 0:
            rate = (time.time() - started) / max(done, 1)
            print(f"[phyco gt] {count + 1}/{len(rows)} done {done} skipped {skipped} errors {len(errors)} {rate:.2f}s/clip", flush=True)
    summary = {"shard": a.shard_index, "num_shards": a.num_shards, "done": done, "skipped": skipped, "n_errors": len(errors),
               "errors": errors[:50], "seconds": round(time.time() - started, 1), "prompts_encoded": encode_prompt.encoded}
    (out / f"summary-shard{a.shard_index:03d}of{a.num_shards:03d}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("PHYCO_GT_DONE", json.dumps({k: v for k, v in summary.items() if k != "errors"}), flush=True)


if __name__ == "__main__":
    main()
