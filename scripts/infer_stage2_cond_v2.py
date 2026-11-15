#!/usr/bin/env python
"""Batch Stage-2 generation from v2 condition caches (one <sid>.pt per clip, see generate.py / cache_stage2_cond_v2.py).

One job = (arm, sample, seed).  Arms read the cached condition of the same name:
  pred   Stage-1's plan and raw field          null        no plan at all (availability 0)
  real   the measured / ground-truth plan      zero        Stage-1's objects, pinned still
  realrot180  the measured plan with every displacement reversed (needs plans_real/)
A 20-channel checkpoint reads channels 0-19 only (config without motion_condition_extra_channels).

Sampler: the stock WanImageToVideoPipeline call (UniPC, text CFG, the two experts switched at the boundary), run in
single-GPU phases: embed (text encoder) -> high (high-noise expert, stores the boundary latent and scheduler state) ->
low (low-noise expert resumes and decodes); `full` does all in one call with CPU offload (bit-identical to high + low).
All arms of a sample share the seed, i.e. the initial noise.  Seeds: --seed-mode job (default) gives every sample
job_seed(--seed, sample_id, k); --seed-mode flat uses --seed + k for every sample.

Written per job: <output>/<arm>__<sid>[__s<k>].npy  uint8 [F, 256, 448, 3] at the cache's frame rate, + a .json sidecar.
--piq-mp4 additionally writes a Physics-IQ submission of the clip (cut to 5.0 s of generated frames, resampled to
120 frames at 24 fps without the conditioning frame, 1280x720): <output>/piq[_s<k>]/<arm>/<sid>.mp4.

usage: infer_stage2_cond_v2.py --config configs/stage2/inference.yaml --cache-dir D [--ids F] --arms pred
                               --low-checkpoint L --high-checkpoint H --output-dir O --phase embed|high|low|full
                               --prompt-embeds P [--num-seeds 1] [--manifest M] [--wan-model DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_jepa import cond_v2  # noqa: E402
from motion_jepa.stage2 import ExtendedOnlyFlowAdapter  # noqa: E402
from motion_jepa.stage2_condition import interface_from_config, read_cached_condition, render_native5  # noqa: E402
from scripts.infer_stage2_oracle import (WAN_BENCHMARK_NEGATIVE_PROMPT, ResidualController, ResumedScheduler, attach_lora,  # noqa: E402
                                         boundary_callback, load_adapter)

ARMS = ("pred", "real", "zero", "null", "realrot180")
BASE_ARM = "base"            # --base: the untrained Wan2.2-I2V-A14B (no LoRA, no motion adapter), same sampler / seeds / prompts
PLAN_DIRS: dict = {}       # extra arms: name -> folder of <sid>.npz plans rendered as plan-only conditions (--plans NAME=DIR)


def job_name(arm: str, sid: str, seed_index: int) -> str:
    return f"{arm}__{sid}" + (f"__s{seed_index}" if seed_index else "")


def job_seed(base: int, sid: str, seed_index: int) -> int:
    """The seed of a sample: shared by all arms, so arms differ in the condition only."""
    return base + zlib.crc32(sid.encode()) % 100000 + 1000003 * seed_index


def clip_generation_payload(payload: dict, max_frames: int) -> dict:
    """Clip a cache consistently to a Wan 4k+1 frame prefix without mutation."""
    if max_frames < 0 or 0 < max_frames < 5:
        raise ValueError("max_frames must be zero or at least five")
    if max_frames == 0 or int(payload["frames"]) <= max_frames:
        return payload
    frames = 4 * ((min(int(payload["frames"]), max_frames) - 1) // 4) + 1
    count = (frames - 1) // 4 + 1
    result = dict(payload)
    result["frames"] = frames
    result["frame_seconds"] = payload["frame_seconds"][:count]
    for key, value in payload.items():
        if (key in ("video_latent", "image_cond_latent") or key.startswith("cond_")) and isinstance(value, torch.Tensor):
            if value.ndim != 4 or value.shape[1] < count:
                raise ValueError(f"{key}: cannot clip invalid latent time axis")
            result[key] = value[:, :count]
    return result


def read_condition(payload: dict, arm: str, cache_dir: Path, interface: str = "legacy27", source: str = "cached") -> torch.Tensor:
    if arm in ("pred", "real", "zero", "null"):
        return read_cached_condition(payload, arm, interface, source)
    from scripts import motion_plan as mp
    from scripts.control import plan_condition as pc
    if arm == "realrot180":
        plan = mp.rotated(mp.load_plan(cache_dir / "plans_real" / f"{payload['sample_id']}.npz"), 180.0)
    else:
        plan = mp.load_plan(Path(PLAN_DIRS[arm]) / f"{payload['sample_id']}.npz")          # any plan in the cache's work frame, plan-only rendering
    latent, times, size = payload["image_cond_latent"].float(), payload["frame_seconds"], (int(payload["height"]), int(payload["width"]))
    if interface == "native5":
        return render_native5(plan, latent, times, size)
    rendered = pc.render_v2("real", plan, latent, times, size)
    return rendered[:20] if interface == "legacy20" else rendered


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--ids", default="", help="sample ids, one per line (default: every <sid>.pt of the cache)")
    p.add_argument("--manifest", default="", help="csv/jsonl with sample_id, first_frame_path: for caches that do not store the image path")
    p.add_argument("--arms", default="pred,null")
    p.add_argument("--plans", action="append", default=[], help="NAME=DIR: an extra arm NAME whose condition is the plan DIR/<sid>.npz rendered like the measured plan")
    weights = os.environ.get("MOPLAN_WEIGHTS", "")
    p.add_argument("--low-checkpoint", default=str(Path(weights) / "stage2" / "low") if weights else "",
                   help="low-noise expert folder (lora_params.pt, motion_adapter.pt; default $MOPLAN_WEIGHTS/stage2/low)")
    p.add_argument("--high-checkpoint", default=str(Path(weights) / "stage2" / "high") if weights else "",
                   help="high-noise expert folder (default $MOPLAN_WEIGHTS/stage2/high)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--phase", choices=("full", "embed", "high", "low"), default="full")
    p.add_argument("--stage-dir", default="")
    p.add_argument("--prompt-embeds", default="")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-mode", choices=("job", "flat"), default="job",
                   help="job: job_seed(--seed, seed_key or sample_id, k) per sample (benchmark convention); "
                   "flat: --seed + k for every sample")
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--condition-scale", type=float, default=1.0)
    p.add_argument("--guidance-scale", type=float, default=0.0, help="text guidance of the sampler (0 = the config's evaluation.guidance_scale)")
    p.add_argument("--flow-shift", type=float, default=0.0, help="timestep shift of the sampler (0 = the pipeline's own scheduler, shift 3)")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0, help="optional 4k+1 frame prefix cap; zero preserves cached length")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--piq-mp4", action="store_true", help="also write the official Physics-IQ mp4 of every clip")
    p.add_argument("--base", action="store_true", help="arm base = the untrained base model: no LoRA, no adapter, no condition (checkpoints ignored)")
    p.add_argument("--prompt-table", default="", help="json sample_id -> prompt for the video generator, overriding the cache's prompt "
                   "(e.g. a benchmark's extended prompts, while Stage-1 read the plain one)")
    p.add_argument("--seed-major", action="store_true", help="order the jobs seed by seed (all samples at seed 0 first): a multi-seed "
                   "benchmark can be scored on seed 0 while the other seeds are still generating")
    p.add_argument("--negative-prompt", default="", help="negative prompt (default: Wan2.2's); a benchmark's official one, e.g. RigidBench")
    p.add_argument("--wan-model", default="", help="Wan2.2-I2V-A14B-Diffusers directory (default: $WAN_MODEL_DIR, else the config's paths.wan_model)")
    a = p.parse_args(argv)
    table_prompts = json.loads(Path(a.prompt_table).read_text()) if a.prompt_table else {}
    negative_prompt = a.negative_prompt or WAN_BENCHMARK_NEGATIVE_PROMPT
    gen_prompt = lambda sid, payload: table_prompts.get(sid, payload["prompt"])   # noqa: E731
    if a.max_frames < 0 or 0 < a.max_frames < 5:
        p.error("--max-frames must be zero or at least five")
    config = yaml.safe_load(Path(a.config).read_text())
    paths, model_cfg, eval_cfg = config.get("paths") or {}, config["model"], config["evaluation"]
    condition_interface = interface_from_config(model_cfg)
    condition_source = str(model_cfg.get("condition_source", "cached"))
    cache_dir, output_dir = Path(a.cache_dir), Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in a.plans:
        name, folder = item.split("=", 1)
        PLAN_DIRS[name] = folder
    arms = [v for v in a.arms.split(",") if v]
    if a.base and arms != [BASE_ARM]:
        raise SystemExit(f"--base generates the single arm {BASE_ARM}")
    if not a.base and any(arm not in ARMS and arm not in PLAN_DIRS for arm in arms):
        raise SystemExit(f"arms must be among {ARMS}")
    ids = ([line.strip() for line in Path(a.ids).read_text().splitlines() if line.strip()] if a.ids
           else sorted(path.stem for path in cache_dir.glob("*.pt")))
    if a.limit:
        ids = ids[: a.limit]
    images = {}
    if a.manifest:
        rows = (list(csv.DictReader(open(a.manifest, newline=""))) if a.manifest.endswith(".csv")
                else [json.loads(line) for line in Path(a.manifest).read_text().splitlines() if line.strip()])
        images = {row["sample_id"]: row["first_frame_path"] for row in rows}
    jobs = ([(arm, sid, k) for k in range(a.num_seeds) for sid in ids for arm in arms] if a.seed_major
            else [(arm, sid, k) for sid in ids for k in range(a.num_seeds) for arm in arms])
    all_ids = ids
    jobs = [job for index, job in enumerate(jobs) if index % a.num_shards == a.shard_index]

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
    from transformers import UMT5EncoderModel

    wan_model = a.wan_model or os.environ.get("WAN_MODEL_DIR", "") or str(paths.get("wan_model") or "")
    if not wan_model:
        raise SystemExit("no Wan2.2 model directory: pass --wan-model or set WAN_MODEL_DIR")
    model_dir = Path(wan_model)
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32)
    phase = a.phase
    stage_dir = Path(a.stage_dir or output_dir / "_stage")
    wanted = {"full": ("transformer", "transformer_2", "text_encoder"), "embed": ("text_encoder",), "high": ("transformer",), "low": ("transformer_2",)}[phase]
    if phase == "full":
        pipeline = WanImageToVideoPipeline.from_pretrained(model_dir, vae=vae, image_encoder=None, torch_dtype=torch.bfloat16)
    else:
        parts = {name: None for name in ("transformer", "transformer_2", "text_encoder") if name not in wanted}
        if "text_encoder" not in wanted:
            parts["tokenizer"] = None
        for name in wanted:
            loader = UMT5EncoderModel if name == "text_encoder" else WanTransformer3DModel
            parts[name] = loader.from_pretrained(model_dir, subfolder=name, torch_dtype=torch.bfloat16, device_map="cuda")
        pipeline = WanImageToVideoPipeline.from_pretrained(model_dir, vae=vae, image_encoder=None, torch_dtype=torch.bfloat16, **parts)
    if phase == "embed":
        if not a.prompt_embeds:
            raise SystemExit("--phase embed needs --prompt-embeds (output file)")
        prompts = sorted({gen_prompt(sid, torch.load(cache_dir / f"{sid}.pt", map_location="cpu", weights_only=False)) for sid in all_ids})
        table, negative = {}, None
        with torch.inference_mode():
            for prompt in prompts:
                positive, negative = pipeline.encode_prompt(prompt=prompt, negative_prompt=negative_prompt, do_classifier_free_guidance=True,
                                                            num_videos_per_prompt=1, max_sequence_length=512, device=torch.device("cuda"))
                table[prompt] = positive[0].cpu()
        Path(a.prompt_embeds).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"prompts": table, "negative": negative[0].cpu(), "negative_prompt": negative_prompt, "max_sequence_length": 512}, a.prompt_embeds)
        print(json.dumps({"prompt_embeds": a.prompt_embeds, "prompts": len(table)}))
        return
    embeds = None
    if phase in ("high", "low") or a.prompt_embeds:
        if not a.prompt_embeds or not Path(a.prompt_embeds).is_file():
            raise SystemExit("--phase high/low needs --prompt-embeds (run --phase embed first)")
        embeds = torch.load(a.prompt_embeds, map_location="cpu", weights_only=True)
        if a.negative_prompt and embeds.get("negative_prompt") != a.negative_prompt:      # only official-negative runs check (older files differ)
            raise SystemExit(f"{a.prompt_embeds} was encoded with another negative prompt than this run's")
        stage_dir.mkdir(parents=True, exist_ok=True)
    blocks = list(model_cfg["injection_blocks"])
    adapters, controllers = {}, {}
    for name, checkpoint in (("transformer_2", a.low_checkpoint), ("transformer", a.high_checkpoint)):
        expert = getattr(pipeline, name, None)
        if expert is None or a.base:
            continue
        if not checkpoint:
            raise SystemExit(f"phase {phase} needs the checkpoint of {name}")
        attach_lora(expert, Path(checkpoint))
        adapters[name] = load_adapter(Path(checkpoint), model_cfg, expert.config.num_attention_heads * expert.config.attention_head_dim)
        controllers[name] = ResidualController(expert, blocks, a.condition_scale, adapters[name])
    if a.flow_shift > 0:
        pipeline.scheduler = type(pipeline.scheduler).from_config(pipeline.scheduler.config, flow_shift=float(a.flow_shift))
    boundary_timestep = float(pipeline.config.boundary_ratio) * pipeline.scheduler.config.num_train_timesteps
    base_scheduler = pipeline.scheduler
    if phase == "full":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")

    written = 0
    for index, (arm, sid, seed_index) in enumerate(jobs):
        name = job_name(arm, sid, seed_index)
        output, stage_path = output_dir / f"{name}.npy", stage_dir / f"{name}.pt"
        if output.is_file() and not a.overwrite:
            continue
        if phase == "high" and stage_path.is_file() and not a.overwrite:
            continue
        if phase == "low" and not stage_path.is_file():
            print(f"[skip] no high-noise stage for {name}", flush=True)
            continue
        generation_started = time.monotonic()
        payload = torch.load(cache_dir / f"{sid}.pt", map_location="cpu", weights_only=False)
        if payload.get("cache_format_version") != cond_v2.CACHE_FORMAT:
            raise SystemExit(f"{sid}: not a {cond_v2.CACHE_FORMAT} cache")
        source_frames = int(payload["frames"])
        payload = clip_generation_payload(payload, a.max_frames)
        frames = int(payload["frames"])
        condition_cpu = read_condition(payload, "null" if a.base else arm, cache_dir, condition_interface, condition_source)
        if condition_cpu.shape[1] != (frames - 1) // 4 + 1:
            raise SystemExit(f"{sid}: {frames} frames but a condition of {condition_cpu.shape[1]} latent frames")
        for expert_name, adapter in adapters.items():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                controllers[expert_name].set(adapter(condition_cpu[None].to("cuda", torch.bfloat16)))
        image_path = images.get(sid) or payload.get("first_frame_path")
        if not image_path:
            raise SystemExit(f"{sid}: the cache has no first_frame_path; pass --manifest")
        image = Image.open(image_path).convert("RGB").resize((int(payload["width"]), int(payload["height"])), Image.LANCZOS)
        if a.seed_mode == "flat":
            seed = a.seed + seed_index
        else:
            seed = job_seed(a.seed, str(payload.get("seed_key", sid)), seed_index)   # rollouts of one condition share their noise through seed_key
        call = dict(image=image, height=int(payload["height"]), width=int(payload["width"]), num_frames=frames,
                    num_inference_steps=int(a.steps or eval_cfg["inference_steps"]), guidance_scale=float(a.guidance_scale or eval_cfg["guidance_scale"]))
        if embeds is None:
            call.update(prompt=gen_prompt(sid, payload), negative_prompt=negative_prompt)
        else:
            call.update(prompt_embeds=embeds["prompts"][gen_prompt(sid, payload)][None].to("cuda"), negative_prompt_embeds=embeds["negative"][None].to("cuda"))
        signature = [seed, arm, a.condition_scale, str(cache_dir), frames]
        if condition_interface != "legacy27" or condition_source != "cached":
            signature += [condition_interface, condition_source]
        if a.guidance_scale > 0 or a.flow_shift > 0:                       # non-default sampler: its staged latents are its own
            signature += [float(a.guidance_scale), float(a.flow_shift)]
        if phase == "high":
            store: dict = {}
            pipeline(**call, generator=torch.Generator(device="cuda").manual_seed(seed), output_type="latent",
                     callback_on_step_end=boundary_callback(boundary_timestep, store))
            if "latents" not in store:
                raise RuntimeError(f"{name}: the schedule has no low-noise step after the boundary")
            temporary = stage_path.with_suffix(f".tmp-{os.getpid()}")
            torch.save({**store, "name": name, "signature": signature}, temporary)
            temporary.replace(stage_path)
            print(f"[{index + 1}/{len(jobs)}] stage {stage_path.name} seconds={time.monotonic() - generation_started:.3f}", flush=True)
            continue
        if phase == "low":
            stage = torch.load(stage_path, weights_only=False)
            if stage["signature"] != signature:
                raise RuntimeError(f"{name}: the stored high-noise stage belongs to a different arm/seed/cache")
            object.__setattr__(pipeline, "scheduler", ResumedScheduler(stage["scheduler"], stage["scheduler"].timesteps[stage["next_index"]:]))
            try:
                result = pipeline(**call, latents=stage["latents"].to("cuda"))
            finally:
                object.__setattr__(pipeline, "scheduler", base_scheduler)
        else:
            result = pipeline(**call, generator=torch.Generator(device="cuda").manual_seed(seed))
        native = np.stack([np.asarray(frame) for frame in result.frames[0]])
        if native.dtype != np.uint8:
            native = np.clip(native * (255.0 if float(native.max()) <= 1.5 else 1.0), 0, 255).round().astype(np.uint8)
        if a.piq_mp4:
            import cv2

            from scripts.infer_stage2_oracle import write_video
            from scripts.infer_stage2_pisa import temporal_upsample
            keep = int(round(5.0 * float(payload["fps"]))) + 1
            official = temporal_upsample([frame for frame in native[:keep]], 121)[1:]
            official = [cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LANCZOS4) for frame in official]
            folder = output_dir / ("piq" + (f"_s{seed_index}" if seed_index else "")) / arm
            folder.mkdir(parents=True, exist_ok=True)
            write_video(official, folder / f"{sid}.mp4", 24)
        temporary = output_dir / f"{name}.tmp-{os.getpid()}.npy"
        np.save(temporary, native)
        temporary.replace(output)
        (output_dir / f"{name}.json").write_text(json.dumps({"sample_id": sid, "arm": arm, "seed": seed, "seed_index": seed_index, "frames": frames,
                                                              "fps": float(payload["fps"]), "low": a.low_checkpoint, "high": a.high_checkpoint,
                                                              "cache": str(cache_dir), "condition_scale": a.condition_scale,
                                                              "condition_interface": condition_interface, "condition_source": condition_source,
                                                              "steps": call["num_inference_steps"], "guidance_scale": call["guidance_scale"],
                                                              "flow_shift": a.flow_shift, "seed_mode": a.seed_mode,
                                                              "source_frames": source_frames, "max_frames": a.max_frames,
                                                              "frame_seconds": [float(t) for t in payload["frame_seconds"]],
                                                              "phase_seconds": time.monotonic() - generation_started}) + "\n")
        if phase == "low":
            stage_path.unlink()
        written += 1
        print(f"[{index + 1}/{len(jobs)}] {output.name} seconds={time.monotonic() - generation_started:.3f}", flush=True)
    print(f"INFER_V2_DONE phase {phase} shard {a.shard_index}/{a.num_shards}: wrote {written} of {len(jobs)} jobs")


if __name__ == "__main__":
    main()
