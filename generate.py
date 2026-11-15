#!/usr/bin/env python
"""MoPlan end to end: first image + prompt -> video.

Pipeline (every step calls the same code as the batch path, see docs/INFERENCE.md):
  1. input      optional 16:9 canvas (--canvas 16x9, the MoProbe input rule: bench_common.letterbox_16x9);
                caption rule (bench_common.caption_for for --protocol moprobe, the text verbatim for physics_iq)
  2. masks      SAM 2.1 automatic masks (scripts/cache_sam2_masks.py) -> k16 filter (scripts/filter_sam2_masks_v2.py),
                or --masks FILE (a k16 mask file made earlier)
  3. Stage-1    scripts/infer_stage1_physicsiq.py on a one-row manifest (released Stage-1 checkpoint, prior latent,
                7 query horizons up to 4 s, text batch 4, seed 42), or --stage1-export FILE
  4. Stage-2    --protocol moprobe     the benchmark path of scripts/infer_stage2_cond_v2.py: a stage2_cond_v2 cache item
                                       (Wan VAE latent of the stretched 448x256 frame, stored fp16; cond_pred/zero/null
                                       re-rendered by refresh_predictions.refresh_payload; hold rule beyond the Stage-1
                                       horizon), then the phases embed -> high -> low; 37 frames @ 7 fps by default;
                                       seed job_seed(--seed, --sample-id, k) when --sample-id is given, else --seed
                --protocol physics_iq  the native profile of scripts/s2eval_verified/gen_verified.py (NativeGenerator):
                                       41 frames, condition at 8 fps latent times, both experts resident, flat --seed,
                                       written at 8 fps and trimmed to 5.000 s (40 frames)
                --base                 the untrained Wan2.2-I2V-A14B (no LoRA, no adapter, no condition), same sampler,
                                       image, prompt and seed (the comparison panels)
  5. output     --output OUT.mp4, OUT.npy (the exact uint8 frames: 37 for moprobe, all 41 generated for physics_iq,
                whose mp4 keeps the first 5 s = 40 frames) and OUT.json (settings, seed, frame hash)

Defaults read the environment: WAN_MODEL_DIR (Wan2.2-I2V-A14B-Diffusers), MOPLAN_WEIGHTS (stage1/, stage2/high,
stage2/low), SAM2_CHECKPOINT (sam2.1_hiera_large.pt).  Needs one 80 GB GPU (the physics_iq protocol keeps both 14B
experts resident, about 57 GiB).  The intermediate files live in a work directory (default OUT.work/), removed at the
end unless --save-intermediates is given.

usage:
  python generate.py --image FRAME.png --prompt "A hand pushes the mug from right to left." --output out.mp4
                     [--protocol moprobe|physics_iq] [--sample-id SID] [--seed 42] [--canvas 16x9] [--base]
                     [--masks K16.pt] [--stage1-export EXPORT.pt] [--save-intermediates] [--no-npy]
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                     # heavy imports stay lazy so --help and the CPU tests need no GPU stack
    import numpy as np
    import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# The 7 query horizons of the released Stage-1 model (seconds); identical to the batch export's grid.
QUERY_SECONDS = "0.20833333333333334,0.4166666666666667,0.625,0.8333333333333334,1.2,2,4"
STAGE1_TEXT_BATCH = 4
STAGE1_SEED = 42
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"          # Hydra name inside the sam2 package
K16 = dict(max_kept=16, union_cap=1.0, area_max=0.20, border_max=0.35)
WORK_H, WORK_W = 256, 448
PROTOCOLS = {
    # frames / fps of the generated clip; seed rule; caption rule; writer
    "moprobe": dict(frames=37, fps=7.0, caption_rule="static-suffix"),
    "physics_iq": dict(frames=41, fps=8.0, caption_rule="exact"),
}
TORCH_FLAGS = ("num_threads", "matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark", "cudnn_deterministic",
               "deterministic_algorithms", "float32_matmul_precision", "flash_sdp", "mem_efficient_sdp", "math_sdp")


def job_seed(base: int, sid: str, seed_index: int) -> int:
    """The benchmark seed rule of infer_stage2_cond_v2.py (kept identical; tested against it)."""
    return base + zlib.crc32(sid.encode()) % 100000 + 1000003 * seed_index


def safe_id(text: str) -> str:
    sid = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    if not sid:
        raise ValueError(f"cannot make a sample id from {text!r}")
    return sid


def log(message: str) -> None:
    print(f"[generate {time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="first frame (any size; Physics-IQ switch frames are 3840x2160)")
    p.add_argument("--prompt", default=None, help="text prompt (required unless --prompt-file)")
    p.add_argument("--prompt-file", default="", help="read the prompt from this UTF-8 text file")
    p.add_argument("--output", "--out", dest="output", required=True, help="output .mp4 (OUT.npy and OUT.json are written next to it)")
    p.add_argument("--protocol", choices=tuple(PROTOCOLS), default="moprobe")
    p.add_argument("--sample-id", default="", help="sample id; with --protocol moprobe it selects the benchmark seed "
                   "job_seed(--seed, sample_id, --seed-index), e.g. ssv2_9578-ex")
    p.add_argument("--seed", type=int, default=42, help="base seed (moprobe with --sample-id) or the flat seed")
    p.add_argument("--seed-index", type=int, default=0, help="k of job_seed (moprobe with --sample-id); flat seeds add k")
    p.add_argument("--frames", type=int, default=0, help="moprobe only: clip length 4k+1 (default 37)")
    p.add_argument("--fps", type=float, default=0.0, help="moprobe only: frame rate of the clip (default 7)")
    p.add_argument("--base", action="store_true", help="the untrained Wan2.2-I2V-A14B: no LoRA, no adapter, no condition")
    p.add_argument("--canvas", choices=("none", "16x9"), default="none",
                   help="16x9: letterbox the image into a black 1280x720 canvas first (the MoProbe input rule)")
    p.add_argument("--caption-rule", choices=("auto", "exact", "static-suffix"), default="auto",
                   help="static-suffix: append ' Static shot with no camera movement.' iff the prompt states a static "
                   "camera (bench_common.caption_for; MoProbe); exact: verbatim (Physics-IQ); auto: per protocol")
    p.add_argument("--negative-prompt", default="", help="moprobe only: negative prompt (default: Wan2.2's official one)")
    p.add_argument("--masks", default="", help="k16 instance-mask .pt to use instead of running SAM2")
    p.add_argument("--stage1-export", default="", help="Stage-1 export .pt to use instead of running Stage-1")
    p.add_argument("--stage1-text-manifest", default="", help="reproduce the Stage-1 text batches of a batch run over this "
                   "manifest CSV (see scripts/infer_stage1_physicsiq.py --text-manifest)")
    p.add_argument("--stage1-text-shard", default="0/1", help="i/n shard of --stage1-text-manifest used by that batch run")
    p.add_argument("--weights", default=os.environ.get("MOPLAN_WEIGHTS", ""), help="MoPlan weights root (default $MOPLAN_WEIGHTS)")
    p.add_argument("--stage1-checkpoint", default="", help="default <weights>/stage1")
    p.add_argument("--high-checkpoint", default="", help="default <weights>/stage2/high")
    p.add_argument("--low-checkpoint", default="", help="default <weights>/stage2/low")
    p.add_argument("--wan-model", default=os.environ.get("WAN_MODEL_DIR", ""), help="default $WAN_MODEL_DIR")
    p.add_argument("--sam2-checkpoint", default=os.environ.get("SAM2_CHECKPOINT", ""), help="default $SAM2_CHECKPOINT")
    p.add_argument("--sam2-config", default=SAM2_CONFIG)
    p.add_argument("--config", default=str(ROOT / "configs" / "stage2" / "inference.yaml"))
    p.add_argument("--work-dir", default="", help="intermediate files (default OUT.work/, removed unless --save-intermediates)")
    p.add_argument("--save-intermediates", action="store_true", help="keep masks, Stage-1 export, cache item / condition, "
                   "prompt embeddings and raw frames in the work directory")
    p.add_argument("--no-npy", action="store_true", help="do not write OUT.npy (uint8 [F, 256, 448, 3], the frames before encoding)")
    p.add_argument("--verify-weights", action="store_true", help="check the sha256 of every weight file against tools/weights_manifest.json")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    resolve(a, p)
    return a


def resolve(a: argparse.Namespace, p: argparse.ArgumentParser) -> None:
    """Fill the protocol defaults and the weight paths; reject combinations the paper pipelines never ran."""
    proto = PROTOCOLS[a.protocol]
    if a.prompt is None and not a.prompt_file:
        p.error("pass --prompt or --prompt-file")
    if a.prompt_file:
        a.prompt = Path(a.prompt_file).read_text(encoding="utf-8").rstrip("\n")
    if a.protocol == "physics_iq":
        if a.frames or a.fps:
            p.error("--frames / --fps are fixed by --protocol physics_iq (41 frames at 8 fps, trimmed to 5 s)")
        if a.negative_prompt:
            p.error("--protocol physics_iq always uses Wan2.2's official negative prompt")
    a.frames = int(a.frames or proto["frames"])
    a.fps = float(a.fps or proto["fps"])
    if a.frames < 5 or (a.frames - 1) % 4:
        p.error(f"--frames {a.frames} must be 4k+1 and at least 5")
    if a.fps <= 0:
        p.error("--fps must be positive")
    if a.seed_index < 0:
        p.error("--seed-index must be non-negative")
    a.caption_rule = proto["caption_rule"] if a.caption_rule == "auto" else a.caption_rule
    weights = Path(a.weights) if a.weights else None
    if not a.stage1_checkpoint and weights is not None:
        a.stage1_checkpoint = str(weights / "stage1")
    if not a.high_checkpoint and weights is not None:
        a.high_checkpoint = str(weights / "stage2" / "high")
    if not a.low_checkpoint and weights is not None:
        a.low_checkpoint = str(weights / "stage2" / "low")
    try:
        index, count = (int(v) for v in a.stage1_text_shard.split("/"))
    except ValueError:
        p.error("--stage1-text-shard must be i/n")
    if count < 1 or not 0 <= index < count:
        p.error("--stage1-text-shard must be i/n with 0 <= i < n")
    a.stage1_text_shard_index, a.stage1_text_num_shards = index, count
    a.output = str(Path(a.output))
    if not a.output.endswith(".mp4"):
        p.error("--output must end with .mp4")


def caption_of(prompt: str, rule: str) -> str:
    from scripts.bench import bench_common as bc
    return bc.caption_for(prompt) if rule == "static-suffix" else prompt


def generation_seed(a: argparse.Namespace, sid: str) -> int:
    """moprobe + --sample-id: the benchmark rule job_seed(seed, sid, k); otherwise flat: seed + k."""
    if a.protocol == "moprobe" and a.sample_id:
        return job_seed(a.seed, sid, a.seed_index)
    return a.seed + a.seed_index


# ------------------------------------------------------------------------------------------------ torch hygiene
def torch_flags() -> dict:
    import torch
    return {"num_threads": torch.get_num_threads(), "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(), "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp": torch.backends.cuda.math_sdp_enabled()}


def restore_flags(flags: dict) -> None:
    """Undo any process-global torch setting a stage changed: every batch-path stage starts in a fresh process."""
    import torch
    now = torch_flags()
    changed = [k for k in TORCH_FLAGS if now[k] != flags[k]]
    if not changed:
        return
    log(f"restoring torch settings changed by the previous stage: {changed}")
    torch.set_num_threads(flags["num_threads"])
    torch.backends.cuda.matmul.allow_tf32 = flags["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = flags["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
    torch.use_deterministic_algorithms(flags["deterministic_algorithms"])
    torch.set_float32_matmul_precision(flags["float32_matmul_precision"])
    torch.backends.cuda.enable_flash_sdp(flags["flash_sdp"])
    torch.backends.cuda.enable_mem_efficient_sdp(flags["mem_efficient_sdp"])
    torch.backends.cuda.enable_math_sdp(flags["math_sdp"])


def release_gpu() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ inputs
def prepare_image(a: argparse.Namespace, sid: str, work: Path) -> Path:
    """The image every later step reads: the original file, or the 16:9 canvas PNG made from it."""
    source = Path(a.image).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if a.canvas == "none":
        return source
    from scripts.bench import bench_common as bc
    target = work / "inputs" / f"{sid}.png"
    bc.letterbox_16x9(source, target)
    return target.resolve()


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as image:
        return image.height, image.width


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------------------------------------ masks + Stage-1
def run_masks(a: argparse.Namespace, sid: str, image: Path, work: Path) -> Path:
    """k16 instance masks of the first frame -> the directory holding <sid>.pt (Stage-1 --instance-masks)."""
    import torch
    mask_dir = work / "masks" / "k16"
    mask_dir.mkdir(parents=True, exist_ok=True)
    target = mask_dir / f"{sid}.pt"
    if a.masks:
        payload = torch.load(a.masks, map_location="cpu", weights_only=True)
        if tuple(payload["instance_masks"].shape) != (16, 160, 160) or tuple(payload["instance_channels"].shape) != (2, 160, 160):
            raise ValueError(f"{a.masks}: expected instance_masks [16,160,160] and instance_channels [2,160,160]")
        shutil.copyfile(a.masks, target)
        log(f"masks: {a.masks}")
        return mask_dir
    if not a.sam2_checkpoint:
        raise SystemExit("SAM2 masks need --sam2-checkpoint or $SAM2_CHECKPOINT (or pass --masks)")
    from scripts import cache_sam2_masks
    from scripts import filter_sam2_masks_v2 as k16
    raw_dir = work / "masks" / "sam2_raw"
    manifest = work / "masks" / "manifest.csv"
    write_csv(manifest, ["sample_id", "first_frame_path"], [{"sample_id": sid, "first_frame_path": str(image)}])
    log("masks: SAM 2.1 automatic mask generator")
    cache_sam2_masks.main(["--manifest", str(manifest), "--sam-checkpoint", a.sam2_checkpoint, "--sam-config", a.sam2_config,
                           "--output-dir", str(raw_dir), "--overwrite"])
    release_gpu()
    raw = torch.load(raw_dir / f"{sid}.pt", map_location="cpu")
    k16.configure(**K16)
    from PIL import Image
    with Image.open(image) as im:                      # measure on the content box of the image SAM 2 saw
        box = k16.filter_box(im.height, im.width)
    out, stats = k16.filter_payload(raw, box)
    torch.save(out, target)
    log(f"masks: k16 filter kept {out['num_instances']} instances (raw {stats['n_raw']}, fallback {stats['fallback']})")
    return mask_dir


def run_stage1(a: argparse.Namespace, sid: str, image: Path, caption: str, mask_dir: Path, work: Path) -> Path:
    """Stage-1 export of (first frame, caption, k16 masks) with the released checkpoint."""
    if a.stage1_export:
        log(f"Stage-1: {a.stage1_export}")
        return Path(a.stage1_export)
    checkpoint = Path(a.stage1_checkpoint)
    for name in ("backend_params.pt", "decoder.pt", "trainer_state.json", "resolved_config.yaml", "prior_latent.pt"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Stage-1 checkpoint file missing: {checkpoint / name}")
    from scripts import infer_stage1_physicsiq as s1
    out_dir = work / "stage1"
    manifest = out_dir / "manifest.csv"
    write_csv(manifest, ["sample_id", "first_frame_path", "caption"], [{"sample_id": sid, "first_frame_path": str(image), "caption": caption}])
    argv = ["--config", str(checkpoint / "resolved_config.yaml"), "--checkpoint", str(checkpoint), "--manifest", str(manifest),
            "--output-dir", str(out_dir), "--instance-masks", str(mask_dir), "--prior-latent", str(checkpoint / "prior_latent.pt"),
            "--query-horizon-seconds", QUERY_SECONDS, "--text-batch-size", str(STAGE1_TEXT_BATCH), "--seed", str(STAGE1_SEED),
            "--set", f"paths.wan_model={a.wan_model}", "--overwrite"]
    if a.stage1_text_manifest:
        argv += ["--text-manifest", a.stage1_text_manifest, "--text-shard-index", str(a.stage1_text_shard_index),
                 "--text-num-shards", str(a.stage1_text_num_shards)]
    log("Stage-1: motion plan")
    s1.main(argv)
    release_gpu()
    return out_dir / "flows" / f"{sid}.pt"


# ------------------------------------------------------------------------------------------------ moprobe protocol
def encode_image_latent(vae, first, frames: int, height: int, width: int, device) -> "torch.Tensor":
    """Wan VAE posterior mode of (first frame ++ zero frames), [16, L, h/8, w/8] fp32 on CPU: the image latent of the
    no-video branch of scripts/cache_stage2_cond_v2.py, line for line."""
    import numpy as np
    import torch
    from scripts.cache_stage2_oracle import preprocess
    clip = np.zeros((frames, height, width, 3), np.uint8)
    clip[0] = first
    video = preprocess(clip, height, width).to(device)                       # [1, 3, F, H, W] in [-1, 1]
    image_video = torch.zeros_like(video)
    image_video[:, :, :1] = video[:, :, :1]
    with torch.no_grad():
        return vae.encode(image_video).latent_dist.mode()[0].float().cpu()


def base_payload(sid: str, caption: str, image: Path, image_latent, frames: int, fps: float,
                 height: int = WORK_H, width: int = WORK_W) -> dict:
    """The deployment cache item written by scripts/cache_stage2_cond_v2.py --no-video (the keys Stage-2 reads):
    fp16 image latent, frame times rounded to 1e-6 s, the null condition."""
    from motion_jepa import cond_v2
    from scripts.control import plan_condition as pc
    times = pc.latent_times(frames, fps)
    if image_latent.shape[1] != len(times):
        raise ValueError(f"image latent has {image_latent.shape[1]} latent frames, {frames} frames need {len(times)}")
    return {"cache_format_version": cond_v2.CACHE_FORMAT, "sample_id": sid, "scenario": "", "source": "", "group_id": sid,
            "split": "", "prompt": caption, "fps": float(fps), "height": height, "width": width, "geometry": "stretch",
            "condition_channels_v2": list(cond_v2.EXTRA_NAMES), "base_channels": cond_v2.BASE_CHANNELS,
            "first_frame_path": str(image), "frames": frames, "frames_real": frames, "latent_frames": len(times),
            "frame_seconds": [round(t, 6) for t in times], "image_cond_latent": image_latent.half(),
            "cond_null": pc.render_v2("null", None, image_latent, times, (height, width)).half()}


def moprobe_cache_item(payload: dict, export: dict, native_hw: tuple[int, int]) -> dict:
    """Stage-1 conditions of the evaluation caches: refresh_predictions.refresh_payload (cond_pred / cond_zero /
    cond_null from the cache's fp16 image latent and rounded frame times), then the hold rule on cond_pred exactly as
    scripts/s2eval/hold_known_beyond_horizon.py applies it to a cache file."""
    from scripts.clean_r2.refresh_predictions import refresh_payload
    from scripts.s2eval.hold_known_beyond_horizon import hold
    item, _ = refresh_payload(payload, export, native_hw)
    held, _, _ = hold(item["cond_pred"].float())
    item["cond_pred"] = held.to(item["cond_pred"].dtype)
    item["hold_known_beyond_horizon"] = True
    return item


def run_moprobe(a: argparse.Namespace, sid: str, image: Path, caption: str, export_path: Path | None, work: Path,
                seed: int) -> "np.ndarray":
    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan
    from scripts import infer_stage2_cond_v2 as s2
    from scripts.cache_stage2_cond_v2 import stretched_image
    cache_dir = work / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first, native_hw = stretched_image(str(image), WORK_H, WORK_W)
    vae = AutoencoderKLWan.from_pretrained(Path(a.wan_model) / "vae", torch_dtype=torch.float32).to(device).eval()
    latent = encode_image_latent(vae, first, a.frames, WORK_H, WORK_W, device)
    del vae
    release_gpu()
    payload = base_payload(sid, caption, image, latent, a.frames, a.fps)
    if not a.base:
        export = torch.load(export_path, map_location="cpu", weights_only=False)
        if export.get("caption") not in (None, caption):
            log("warning: the Stage-1 export was made with another caption than the generator prompt")
        payload = moprobe_cache_item(payload, export, native_hw)
    torch.save(payload, cache_dir / f"{sid}.pt")
    ids = work / "ids.txt"
    ids.write_text(sid + "\n")
    gen_dir, embeds = work / "gen", work / "prompt_embeds.pt"
    arm = "base" if a.base else "pred"
    common = ["--config", a.config, "--cache-dir", str(cache_dir), "--ids", str(ids), "--arms", arm,
              "--output-dir", str(gen_dir), "--prompt-embeds", str(embeds), "--wan-model", a.wan_model,
              "--seed", str(seed), "--seed-mode", "flat", "--overwrite"]
    if a.base:
        common.append("--base")
    else:
        common += ["--high-checkpoint", a.high_checkpoint, "--low-checkpoint", a.low_checkpoint]
    if a.negative_prompt:
        common += ["--negative-prompt", a.negative_prompt]
    for phase in ("embed", "high", "low"):
        log(f"Stage-2: phase {phase}")
        s2.main(common + ["--phase", phase])
        release_gpu()
    frames = np.load(gen_dir / f"{arm}__{sid}.npy")
    return frames


def write_moprobe_mp4(frames, output: Path, fps: float) -> dict:
    """The MoProbe export format (scripts/s2eval/miq_export.py): h264 crf 10 yuv420p at the clip fps, 448x256."""
    from scripts.bench import export_videos as ev
    imgs = [f for f in frames]
    ev.write_mp4_h264(imgs, output, fps)
    info = ev.probe(output)
    if info["frames"] != len(imgs) or abs(info["fps"] - fps) > 0.01:
        raise SystemExit(f"{output}: wrote {len(imgs)} frames at {fps}, read back {info}")
    return info


# ------------------------------------------------------------------------------------------------ physics_iq protocol
def run_physics_iq(a: argparse.Namespace, sid: str, image: Path, caption: str, export_path: Path | None, work: Path,
                   seed: int, output: Path) -> tuple["np.ndarray", dict]:
    import numpy as np
    import torch
    from PIL import Image
    from scripts.s2eval_verified import gen_verified as gv
    manifest = work / "native_manifest.jsonl"
    manifest.write_text(json.dumps({"id": sid, "sid": sid, "image": str(image), "prompt": caption}, ensure_ascii=False) + "\n")
    embeds = work / "prompt_embeds_native.pt"
    log("Stage-2: prompt embeddings (native profile)")
    gv.phase_embed(argparse.Namespace(manifests=[str(manifest)], wan=a.wan_model, embeds=str(embeds)))
    release_gpu()
    exports = None
    if not a.base:
        exports = work / "native_exports"
        exports.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(export_path, exports / f"{sid}.pt")
    ckpt = work / "native_ckpt"
    if not a.base:
        ckpt.mkdir(parents=True, exist_ok=True)
        for name, source in (("high", a.high_checkpoint), ("low", a.low_checkpoint)):
            link = ckpt / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path(source).resolve(), target_is_directory=True)
    ns = argparse.Namespace(config=a.config, wan=a.wan_model, attn="native", base=a.base, ckpt=str(ckpt),
                            embeds=str(embeds), exports=str(exports) if exports else "", hold_known=True, plan_kind="pred",
                            stage1_only=False, native_hw=image_size(image))
    log("Stage-2: loading both experts (native profile)")
    gen = gv.NativeGenerator(ns, {"sha": ""})
    item = {"id": sid, "sid": sid, "prompt": caption, "image": str(image)}
    claims = work / "claims"
    (claims / sid).mkdir(parents=True, exist_ok=True)
    if a.save_intermediates and not a.base:
        with Image.open(image) as img:
            resized = img.convert("RGB").resize((gv.N["width"], gv.N["height"]), Image.LANCZOS)
        torch.save({"cond_pred": gen.condition(item, resized)}, work / "condition.pt")
    log(f"Stage-2: sampling (seed {seed})")
    result = gen.run(item, seed, math.inf, work / "partial.pt", claims)
    frames = result["frames"]
    del gen
    release_gpu()
    raw = work / "raw" / output.name
    tmp = work / f"trimmed_{output.name}"
    gv.write_official(frames, raw, tmp, gv.N["fps"])
    info_raw, info = gv.probe(raw), gv.probe(tmp)
    if info["frames"] != 5 * gv.N["fps"] or info_raw["frames"] != gv.N["frames"] or (info["width"], info["height"]) != (gv.N["width"], gv.N["height"]) \
            or abs(info["duration"] - 5.0) > 1e-3:
        raise SystemExit(f"unexpected Physics-IQ output: raw {info_raw} trimmed {info}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(output))
    return np.ascontiguousarray(frames), {"raw": info_raw, "trimmed": info, "stats": result["stats"]}


# ------------------------------------------------------------------------------------------------ weights
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_weights(a: argparse.Namespace) -> dict:
    """sha256 of the MoPlan weight files in use against tools/weights_manifest.json (the base model needs none)."""
    if a.base:
        return {}
    manifest = json.loads((ROOT / "tools" / "weights_manifest.json").read_text())
    roots = {"stage2/high": a.high_checkpoint, "stage2/low": a.low_checkpoint}
    if not a.stage1_export:
        roots["stage1"] = a.stage1_checkpoint
    checks = {}
    for entry in manifest["files"]:
        target, digest = entry["target"], entry.get("sha256")
        prefix = next((k for k in roots if target.startswith(k + "/")), None)
        if not digest or prefix is None:
            continue
        path = Path(roots[prefix]) / target[len(prefix) + 1:]
        if not path.is_file():
            if entry.get("required", True):
                raise SystemExit(f"missing weight file {path}")
            continue
        got = sha256(path)
        if got != digest:
            raise SystemExit(f"{path}: sha256 {got} differs from the released {digest}")
        checks[target] = "ok"
    log(f"weights verified: {len(checks)} files")
    return checks


# ------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    output = Path(a.output).resolve()
    if output.exists() and not a.overwrite:
        raise SystemExit(f"{output} exists (pass --overwrite)")
    if not a.wan_model:
        raise SystemExit("pass --wan-model or set WAN_MODEL_DIR")
    if not a.base and not (a.high_checkpoint and a.low_checkpoint):
        raise SystemExit("pass --weights / $MOPLAN_WEIGHTS or --high-checkpoint and --low-checkpoint")
    if not a.base and not a.stage1_export and not a.stage1_checkpoint:
        raise SystemExit("pass --weights / $MOPLAN_WEIGHTS, --stage1-checkpoint or --stage1-export")
    if os.environ.get("OMP_NUM_THREADS") != "8":
        log("note: the reference runs used OMP_NUM_THREADS=8 (CPU-side condition rendering); launch/infer_*.sh sets it")
    sid = safe_id(a.sample_id or Path(a.image).stem)
    caption = caption_of(a.prompt, a.caption_rule)
    if not caption.strip() and not a.base:
        raise SystemExit("an empty prompt is not supported (Stage-1 was never trained on an empty caption)")
    auto_work = not a.work_dir
    work = Path(a.work_dir).resolve() if a.work_dir else output.with_name(output.stem + ".work")
    work.mkdir(parents=True, exist_ok=True)
    seed = generation_seed(a, sid)
    started = time.time()
    import torch
    flags = torch_flags()
    checks = verify_weights(a) if a.verify_weights else {}
    image = prepare_image(a, sid, work)
    log(f"sample {sid}: protocol {a.protocol}, {'base' if a.base else 'MoPlan'}, seed {seed}, caption {caption!r}")
    export_path = None
    if not a.base:
        mask_dir = None if a.stage1_export else run_masks(a, sid, image, work)
        restore_flags(flags)
        export_path = run_stage1(a, sid, image, caption, mask_dir, work)
        restore_flags(flags)
    extra = {}
    if a.protocol == "moprobe":
        frames = run_moprobe(a, sid, image, caption, export_path, work, seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        extra["video"] = write_moprobe_mp4(frames, output, a.fps)
    else:
        frames, extra = run_physics_iq(a, sid, image, caption, export_path, work, seed, output)
    if not a.no_npy:
        import numpy as np
        np.save(output.with_suffix(".npy"), frames)
    sidecar = {"sample_id": sid, "protocol": a.protocol, "model": "base" if a.base else "moplan", "seed": seed,
               "seed_rule": "job_seed(seed, sample_id, k)" if (a.protocol == "moprobe" and a.sample_id) else "flat",
               "base_seed": a.seed, "seed_index": a.seed_index, "frames_generated": int(frames.shape[0]),
               "fps": a.fps if a.protocol == "moprobe" else 8.0, "size": [WORK_W, WORK_H], "prompt": a.prompt, "caption": caption,
               "caption_rule": a.caption_rule, "canvas": a.canvas, "image": Path(a.image).name, "image_sha256": sha256(Path(a.image)),
               "masks": "given" if a.masks else ("sam2_k16" if not a.base and not a.stage1_export else None),
               "stage1_export": "given" if a.stage1_export else (None if a.base else "computed"),
               "negative_prompt": "official" if not a.negative_prompt else a.negative_prompt,
               "frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(), "weights_verified": checks,
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "seconds": round(time.time() - started, 1), **extra}
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=1, ensure_ascii=False, default=str) + "\n")
    if auto_work and not a.save_intermediates:
        shutil.rmtree(work, ignore_errors=True)
    log(f"done: {output} ({frames.shape[0]} frames, {sidecar['seconds']} s)")


if __name__ == "__main__":
    main()
