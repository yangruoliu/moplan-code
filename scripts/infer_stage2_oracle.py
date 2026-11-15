#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from motion_jepa.data import read_jsonl
from motion_jepa.stage2 import ExtendedOnlyFlowAdapter, MotionAdapter, OnlyFlowMotionAdapter
from motion_jepa.video import read_sampled_video


NEGATIVE_PROMPT = (
    "static, frozen, blurry, low quality, distorted object, duplicated object, "
    "camera shake, text, watermark"
)

WAN_BENCHMARK_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate videos with the Stage-2 oracle-flow renderer")
    parser.add_argument("--config", required=True, help="config YAML (this legacy entry point ships no config)")
    parser.add_argument("--low-checkpoint", required=True)
    parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument("--split", default="test_core", choices=("val", "internal_test", "test_core"))
    parser.add_argument("--output-dir", required=True)
    # "base" renders the same pipeline with no motion training at all -- no LoRA,
    # no adapter, no injection hooks -- i.e. stock Wan2.2-I2V. Everything else
    # (first frame, prompt, seed, frames, fps, guidance) is held identical, so it
    # isolates what the Stage-2 training bought us. "zero" keeps the trained
    # weights but feeds an all-zero motion condition; the two are different
    # baselines and are not interchangeable.
    parser.add_argument(
        "--condition-arm",
        choices=("correct", "zero", "shuffled", "base", "pred", "zero_cache", "pred_zero", "pred_shuffled"), default="correct",
        help="pred / zero_cache read the v2 cache fields motion_condition_pred[__<field>] / motion_condition_zero "
        "(cache_stage2_phyco_v2.py); zero rebuilds a zero-displacement condition from the GT one (mask kept); "
        "pred_zero does the same from the PREDICTED condition (same predicted support, no displacement: what the "
        "motion itself is worth); pred_shuffled reads motion_condition_pred_shuffled__<field> (same support and "
        "magnitudes, every instance's direction rotated: what the direction is worth). The legacy 'shuffled' arm "
        "translates the whole first frame along another clip's GT centroid path and is not a shuffled prediction.",
    )
    parser.add_argument("--pred-field", default="", help="Stage-1 readout field for pred / pred_zero / pred_shuffled "
                        "(raw, soft, binary_<tau>); empty = the cache's legacy motion_condition_pred")
    parser.add_argument("--injection-scale", type=float, default=1.0,
                        help="multiplier on the adapter residual at every injection block; 0 keeps the trained LoRA "
                        "but switches the control path off (unlike --condition-arm base, which also drops the LoRA)")
    parser.add_argument("--samples-per-scenario", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--cache-dir",
        default="",
        help="override paths.stage2_cache_dir, e.g. an affine-conditioned or "
        "Stage-1-predicted condition cache",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--phase", choices=("full", "embed", "high", "low"), default="full",
        help="single-48GB-GPU generation in passes that each hold one large model: 'embed' encodes the prompts "
        "(text encoder only) into --prompt-embeds; 'high' runs the high-noise expert up to the boundary and stores "
        "the latents + scheduler state under --stage-dir; 'low' resumes from there with the low-noise expert and "
        "writes the video. Both passes go through the stock pipeline call, so high+low reproduces 'full'.",
    )
    parser.add_argument("--stage-dir", default="", help="boundary states of --phase high/low (default <output-dir>/_stage)")
    parser.add_argument("--prompt-embeds", default="", help="prompt embedding file written by --phase embed")
    parser.add_argument("--save-frames-npy", action="store_true", help="also store the decoded frames losslessly (tests)")
    parser.add_argument(
        "--offload", choices=("model", "none"), default="model",
        help="model: enable_model_cpu_offload (needs ~70 GB host RAM, swaps the experts through PCIe -- fine on "
        "GH200, 10x slower on PCIe cards); none: stream every component straight onto the GPU and keep it there "
        "(needs ~72 GB of VRAM, e.g. A100/H100 80 GB, and only a few GB of host RAM)",
    )
    return parser.parse_args()


def attach_lora(transformer, checkpoint: Path) -> None:
    # diffusers' load_lora_adapter has to infer a LoraConfig from the bare
    # safetensors (older checkpoints carry no adapter_config.json), and the
    # kwargs it infers no longer instantiate under peft>=0.18. The checkpoint
    # also stores lora_params.pt, whose key set is identical to the safetensors,
    # so build the adapter with the same LoraConfig the trainer used and load
    # those tensors directly.
    params = checkpoint / "lora_params.pt"
    if params.is_file():
        from peft import LoraConfig

        from motion_jepa.stage2 import DEFAULT_LORA_TARGETS

        state = torch.load(params, map_location="cpu", weights_only=True)
        rank = next(t.shape[0] for name, t in state.items() if name.endswith("lora_A.default.weight"))
        transformer.add_adapter(
            LoraConfig(
                r=rank,
                lora_alpha=rank,
                target_modules=list(DEFAULT_LORA_TARGETS),
                init_lora_weights=True,
            )
        )
        named = dict(transformer.named_parameters())
        missing = [name for name in state if name not in named]
        if missing:
            raise KeyError(f"LoRA parameter not found in transformer: {missing[0]}")
        with torch.no_grad():
            for name, tensor in state.items():
                named[name].copy_(tensor.to(named[name].dtype))
    else:
        lora_dir = checkpoint / "lora" if (checkpoint / "lora").is_dir() else checkpoint
        transformer.load_lora_adapter(
            lora_dir,
            prefix=None,
            use_safetensors=True,
            weight_name="pytorch_lora_weights.safetensors",
        )
    if not any("lora_" in name for name, _ in transformer.named_parameters()):
        raise RuntimeError(f"failed to attach LoRA from {checkpoint}")


def translate_no_wrap(value: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate the final two dimensions and fill exposed pixels with zeros."""
    height, width = value.shape[-2:]
    result = torch.zeros_like(value)
    if abs(dy) >= height or abs(dx) >= width:
        return result
    source_y = slice(max(0, -dy), min(height, height - dy))
    target_y = slice(max(0, dy), min(height, height + dy))
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_x = slice(max(0, dx), min(width, width + dx))
    result[..., target_y, target_x] = value[..., source_y, source_x]
    return result


def make_manual_condition(base: torch.Tensor, path_source: torch.Tensor | None) -> torch.Tensor:
    """Warp the current first-frame condition along a zero or shuffled centroid path."""
    channels, frames, height, width = base.shape
    if channels != 20:
        raise ValueError(f"expected 20 condition channels, got {channels}")
    first_latent = base[:16, 0]
    source_mask = base[18:19, 0]
    if path_source is None:
        displacement = torch.zeros(frames, 2, dtype=base.dtype)
    else:
        other_mask = path_source[18:19]
        weights = other_mask.flatten(2).sum(-1).clamp_min(1e-6)
        displacement = (
            (path_source[16:18] * other_mask).flatten(2).sum(-1) / weights
        ).transpose(0, 1)
    result = torch.zeros_like(base)
    for time in range(frames):
        dx = int(round(float(displacement[time, 0] * width)))
        dy = int(round(float(displacement[time, 1] * height)))
        shifted_latent = translate_no_wrap(first_latent, dy, dx)
        shifted_mask = translate_no_wrap(source_mask, dy, dx)
        result[:16, time] = shifted_latent
        result[16, time] = shifted_mask[0] * displacement[time, 0]
        result[17, time] = shifted_mask[0] * displacement[time, 1]
        result[18:19, time] = shifted_mask
        result[19:20, time] = shifted_mask
    return result


class ResidualController:
    """Adds the adapter's residual into selected transformer blocks.

    ``scale`` exists to answer, on an already-trained model, whether the block
    is used at all: if motion fidelity rises monotonically with it, the
    condition is being consumed; if the curve is flat, the injection path is
    not doing what it claims. OnlyFlow exposes the same knob (their gamma) and
    trains at 1.0, so scale=1.0 reproduces the trained behaviour exactly.
    """

    def __init__(self, transformer, blocks: list[int], scale: float = 1.0, adapter=None) -> None:
        self.blocks = blocks
        self.scale = float(scale)
        # When the adapter mixes the control signal with the hidden state, the
        # residual cannot be precomputed: it depends on what the block receives.
        self.adapter = adapter
        self.residuals: dict[int, torch.Tensor] = {}
        self.handles = [
            transformer.blocks[block].register_forward_pre_hook(self._hook(block)) for block in blocks
        ]

    def _hook(self, block: int):
        def inject(_module, inputs):
            hidden = inputs[0]
            residual = self.residuals[block]
            if hidden.shape[0] != residual.shape[0]:
                if hidden.shape[0] % residual.shape[0]:
                    raise RuntimeError(f"cannot expand residual {residual.shape} to {hidden.shape}")
                residual = residual.repeat_interleave(hidden.shape[0] // residual.shape[0], dim=0)
            mixing = self.adapter is not None and getattr(self.adapter, "interaction", False)
            # a mixing adapter may carry several concatenated controls (DualAdapter); the token count must match
            if hidden.shape[1:-1] != residual.shape[1:-1] or (not mixing and hidden.shape[1:] != residual.shape[1:]):
                raise RuntimeError(f"residual {residual.shape} does not match hidden {hidden.shape}")
            residual = residual.to(device=hidden.device, dtype=hidden.dtype)
            if self.adapter is not None and getattr(self.adapter, "interaction", False):
                residual = self.adapter.mix(self.blocks.index(block), residual, hidden)
            return (hidden + self.scale * residual, *inputs[1:])

        return inject

    def set(self, residuals: list[torch.Tensor]) -> None:
        self.residuals = dict(zip(self.blocks, residuals))


class ResumedScheduler:
    """The scheduler of an interrupted pipeline call, presented so the next call continues it.

    The stock call resets the scheduler and walks ``scheduler.timesteps`` from the start; here
    ``set_timesteps`` is a no-op and ``timesteps`` is the unfinished tail, while ``step`` and every other
    attribute go to the saved multistep scheduler with its history and step index intact."""

    def __init__(self, inner, remaining: torch.Tensor) -> None:
        self._inner, self.timesteps = inner, remaining

    def set_timesteps(self, *args, **kwargs) -> None:
        return None

    def __getattr__(self, name):
        return getattr(self._inner, name)


def boundary_callback(boundary_timestep: float, store: dict):
    """Stop the pipeline after its last high-noise step and keep what the low-noise pass needs."""

    def callback(pipe, index: int, timestep, kwargs: dict) -> dict:
        timesteps = pipe.scheduler.timesteps
        if "latents" not in store and index + 1 < len(timesteps) and float(timesteps[index + 1]) < boundary_timestep:
            store["latents"] = kwargs["latents"].detach().float().cpu().clone()
            store["scheduler"] = copy.deepcopy(pipe.scheduler)
            store["next_index"] = index + 1
            pipe._interrupt = True
        return {}

    return callback


def build_adapter(model_cfg: dict, transformer_dim: int):
    """Construct the adapter the config asks for.

    ``onlyflow`` follows arXiv:2411.10501: multi-scale encoder, injection at
    every block, h + up(SiLU(down(c + h))). ``residual`` is the original single
    scale, three-block, plain-addition form kept so old checkpoints still load.
    """
    kind = str(model_cfg.get("adapter_architecture", "residual"))
    if kind == "onlyflow":
        common = dict(
            hidden_channels=int(model_cfg["adapter_hidden_channels"]),
            transformer_dim=transformer_dim,
            injection_blocks=list(model_cfg["injection_blocks"]),
            depth=int(model_cfg.get("adapter_depth", 2)),
            levels=int(model_cfg.get("adapter_levels", 3)),
            heads=int(model_cfg.get("adapter_heads", 8)),
            mixer_rank=int(model_cfg.get("adapter_mixer_rank", 128)),
        )
        extra = int(model_cfg.get("motion_condition_extra_channels", 0))
        if extra > 0:                       # condition contract v2: the deployed 20 channels + extra ones (zero-initialised stem)
            return ExtendedOnlyFlowAdapter(base_channels=int(model_cfg["motion_condition_channels"]), extra_channels=extra, **common)
        return OnlyFlowMotionAdapter(in_channels=int(model_cfg["motion_condition_channels"]), **common)
    if kind != "residual":
        raise ValueError(f"unknown adapter_architecture {kind!r}")
    return MotionAdapter(
        in_channels=int(model_cfg["motion_condition_channels"]),
        hidden_channels=int(model_cfg["adapter_hidden_channels"]),
        transformer_dim=transformer_dim,
        injection_blocks=list(model_cfg["injection_blocks"]),
        depth=int(model_cfg["adapter_depth"]),
        interaction=bool(model_cfg.get("adapter_interaction", False)),
    )


def load_adapter(checkpoint: Path, model_cfg: dict, transformer_dim: int):
    adapter = build_adapter(model_cfg, transformer_dim)
    state = torch.load(checkpoint / "motion_adapter.pt", map_location="cpu", weights_only=True)
    if isinstance(adapter, ExtendedOnlyFlowAdapter) and "stem_extra.weight" not in state:
        adapter.load_deployed(state)        # a deployed 20-channel checkpoint: the extra stem stays at zero
    else:
        adapter.load_state_dict(state)
    return adapter.to("cuda", torch.bfloat16).eval()


def write_video(frames: list[np.ndarray], output: Path, fps: float) -> None:
    height, width = frames[0].shape[:2]
    temporary = output.with_name(output.stem + ".tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"cannot create {temporary}")
    for frame in frames:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(frame)) <= 1.5 else 1.0
            frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
            frame = np.clip(frame * scale, 0, 255).round().astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    paths, data_cfg, model_cfg, eval_cfg = (
        config["paths"], config["data"], config["model"], config["evaluation"]
    )
    rows = read_jsonl(Path(paths["manifest_dir"]) / f"{args.split}.jsonl")
    selected = []
    counts: dict[str, int] = {}
    for row in rows:
        count = counts.get(row.scenario, 0)
        if count < args.samples_per_scenario:
            selected.append(row)
            counts[row.scenario] = count + 1
    if not selected:
        raise RuntimeError("no evaluation rows selected")
    selected = [
        row for index, row in enumerate(selected) if index % args.num_shards == args.shard_index
    ]

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    model_dir = Path(paths["wan_model"])
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32)
    phase = args.phase
    stage_dir = Path(args.stage_dir or Path(args.output_dir) / "_stage")
    embeds = None
    if phase in ("high", "low"):
        if not args.prompt_embeds or not Path(args.prompt_embeds).is_file():
            raise SystemExit("--phase high/low needs --prompt-embeds (run --phase embed first)")
        if args.condition_arm == "base":
            raise SystemExit("--condition-arm base is only implemented for --phase full")
        embeds = torch.load(args.prompt_embeds, map_location="cpu", weights_only=True)
        stage_dir.mkdir(parents=True, exist_ok=True)
    # which large models this pass holds; everything else is passed as None so it is never read from disk
    wanted = {"full": ("transformer", "transformer_2", "text_encoder"), "embed": ("text_encoder",),
              "high": ("transformer",), "low": ("transformer_2",)}[phase]
    absent = {name: None for name in ("transformer", "transformer_2", "text_encoder") if name not in wanted}
    if "text_encoder" not in wanted:
        absent["tokenizer"] = None
    if args.offload == "none" or phase != "full":
        # Load the big components directly onto the GPU: the host never holds a full copy.
        from diffusers import WanTransformer3DModel
        from transformers import UMT5EncoderModel
        preloaded = {}
        for name in wanted:
            loader = UMT5EncoderModel if name == "text_encoder" else WanTransformer3DModel
            preloaded[name] = loader.from_pretrained(model_dir, subfolder=name, torch_dtype=torch.bfloat16, device_map="cuda")
        pipeline = WanImageToVideoPipeline.from_pretrained(
            model_dir, vae=vae, image_encoder=None, torch_dtype=torch.bfloat16, **preloaded, **absent
        )
    else:
        pipeline = WanImageToVideoPipeline.from_pretrained(
            model_dir, vae=vae, image_encoder=None, torch_dtype=torch.bfloat16
        )
    if phase == "embed":
        # the pipeline's own encoder call (512 tokens, as in its __call__), so later passes that receive
        # these tensors are indistinguishable from passes that received the text
        if not args.prompt_embeds:
            raise SystemExit("--phase embed needs --prompt-embeds (output file)")
        cache_dir = Path(args.cache_dir or paths["stage2_cache_dir"])
        prompts = sorted({torch.load(cache_dir / f"{row.sample_id}.pt", map_location="cpu", weights_only=True)["prompt"]
                          for row in selected if (cache_dir / f"{row.sample_id}.pt").is_file()})
        table, negative = {}, None
        with torch.inference_mode():
            for prompt in prompts:
                positive, negative = pipeline.encode_prompt(
                    prompt=prompt, negative_prompt=NEGATIVE_PROMPT, do_classifier_free_guidance=True,
                    num_videos_per_prompt=1, max_sequence_length=512, device=torch.device("cuda"))
                table[prompt] = positive[0].cpu()
        Path(args.prompt_embeds).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"prompts": table, "negative": negative[0].cpu(), "negative_prompt": NEGATIVE_PROMPT,
                    "max_sequence_length": 512}, args.prompt_embeds)
        print(json.dumps({"prompt_embeds": args.prompt_embeds, "prompts": len(table),
                          "shape": list(negative[0].shape), "dtype": str(negative.dtype)}))
        return
    low_checkpoint, high_checkpoint = Path(args.low_checkpoint), Path(args.high_checkpoint)
    is_base = args.condition_arm == "base"
    if is_base:
        # Stock Wan2.2-I2V: no LoRA, no adapter, no hooks. The checkpoint paths are
        # still required by the CLI but deliberately unused, so the arm cannot
        # silently inherit any Stage-2 training.
        print("[base] stock Wan2.2-I2V -- skipping LoRA, adapter and injection hooks", flush=True)
        low_adapter = high_adapter = None
        low_controller = high_controller = None
    else:
        low_adapter = high_adapter = low_controller = high_controller = None
        blocks = list(model_cfg["injection_blocks"])
        # a single-expert pass (--phase high/low) only carries that expert's LoRA, adapter and hooks
        for name, checkpoint in (("transformer_2", low_checkpoint), ("transformer", high_checkpoint)):
            expert = getattr(pipeline, name, None)
            if expert is None:
                continue
            attach_lora(expert, checkpoint)
            adapter = load_adapter(checkpoint, model_cfg,
                                   expert.config.num_attention_heads * expert.config.attention_head_dim)
            controller = ResidualController(expert, blocks, scale=args.injection_scale, adapter=adapter)
            if name == "transformer_2":
                low_adapter, low_controller = adapter, controller
            else:
                high_adapter, high_controller = adapter, controller
        if args.injection_scale != 1.0:
            print(f"[injection] adapter residual scaled by {args.injection_scale}", flush=True)
    boundary_timestep = float(pipeline.config.boundary_ratio) * pipeline.scheduler.config.num_train_timesteps
    base_scheduler = pipeline.scheduler
    if args.offload == "none" or phase != "full":
        pipeline.to("cuda")
        print(f"[offload] none: everything resident on the GPU ({torch.cuda.memory_allocated() / 1e9:.1f} GB allocated)", flush=True)
    else:
        pipeline.enable_model_cpu_offload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir or paths["stage2_cache_dir"])
    generated = []
    for index, row in enumerate(selected):
        output = output_dir / f"{row.sample_id}.mp4"
        seed = args.seed + zlib.crc32(row.sample_id.encode()) % 100000
        if output.is_file() and not args.overwrite:
            generated.append(
                {
                    "sample_id": row.sample_id,
                    "scenario": row.scenario,
                    "video": str(output.resolve()),
                    "condition_arm": args.condition_arm,
                    "condition_source": row.sample_id if args.condition_arm == "correct" else "existing",
                    "seed": seed,
                }
            )
            continue
        condition_path = cache_dir / f"{row.sample_id}.pt"
        if not condition_path.is_file():
            # A prediction-derived condition cache only covers clips Stage-1
            # produced flow for; an upstream coverage gate can drop some clips.
            # Skipping keeps the shard alive instead of losing every remaining
            # clip behind the missing one.
            print(f"[skip] no condition cached for {row.sample_id}", flush=True)
            continue
        payload = torch.load(condition_path, map_location="cpu", weights_only=True)
        condition = payload["motion_condition"].float()
        condition_source = row.sample_id
        if is_base:
            # The cache is still read, but only for payload["prompt"] -- keeping the
            # text conditioning byte-identical to the other arms is what makes the
            # comparison fair. The motion condition itself is never consumed.
            condition_source = "none"
        elif args.condition_arm == "zero":
            condition = make_manual_condition(condition, None)
        elif args.condition_arm in ("pred", "zero_cache", "pred_zero", "pred_shuffled"):
            if args.condition_arm == "zero_cache":
                field = "motion_condition_zero"
            else:
                stem = "motion_condition_pred_shuffled" if args.condition_arm == "pred_shuffled" else "motion_condition_pred"
                field = f"{stem}__{args.pred_field}" if args.pred_field else stem
                if field not in payload and args.pred_field and payload.get("pred_field") == args.pred_field \
                        and args.condition_arm != "pred_shuffled":
                    field = "motion_condition_pred"                      # single-field legacy cache
            if field not in payload:
                raise KeyError(f"{condition_path} has no {field}; build the cache with cache_stage2_phyco_v2.py")
            condition = payload[field].float()
            if args.condition_arm == "pred_zero":
                # same predicted support (t0 mask), zero displacement, everything rebuilt from it
                condition = make_manual_condition(condition, None)
            condition_source = f"{row.sample_id}:{field}" + (":zero_displacement" if args.condition_arm == "pred_zero" else "")
        elif args.condition_arm == "shuffled":
            donor = next(
                candidate
                for candidate in rows
                if candidate.scenario == row.scenario
                and candidate.sample_id != row.sample_id
                and (cache_dir / f"{candidate.sample_id}.pt").is_file()
            )
            donor_payload = torch.load(cache_dir / f"{donor.sample_id}.pt", map_location="cpu", weights_only=True)
            condition = make_manual_condition(condition, donor_payload["motion_condition"].float())
            condition_source = donor.sample_id
        stage_path = stage_dir / f"{row.sample_id}.pt"
        if phase == "high" and stage_path.is_file() and not args.overwrite:
            continue
        if phase == "low" and not stage_path.is_file():
            print(f"[skip] no high-noise stage for {row.sample_id} (run --phase high first)", flush=True)
            continue
        if not is_base:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                condition_gpu = condition[None].to("cuda", torch.bfloat16)
                if low_controller is not None:
                    low_controller.set(low_adapter(condition_gpu))
                if high_controller is not None:
                    high_controller.set(high_adapter(condition_gpu))
        frames, _, _ = read_sampled_video(
            row.video_path,
            num_frames=int(data_cfg["frames"]),
            target_fps=float(data_cfg["fps"]),
            start_sec=float(data_cfg["start_sec"]),
            height=int(data_cfg["height"]),
            width=int(data_cfg["width"]),
        )
        call = dict(
            image=Image.fromarray(frames[0]),
            height=int(data_cfg["height"]),
            width=int(data_cfg["width"]),
            num_frames=int(data_cfg["frames"]),
            num_inference_steps=int(args.steps or eval_cfg["inference_steps"]),
            guidance_scale=float(eval_cfg["guidance_scale"]),
        )
        if embeds is None:
            call.update(prompt=payload["prompt"], negative_prompt=NEGATIVE_PROMPT)
        else:
            call.update(prompt_embeds=embeds["prompts"][payload["prompt"]][None].to("cuda"),
                        negative_prompt_embeds=embeds["negative"][None].to("cuda"))
        if phase == "high":
            store: dict = {}
            pipeline(**call, generator=torch.Generator(device="cuda").manual_seed(seed), output_type="latent",
                     callback_on_step_end=boundary_callback(boundary_timestep, store))
            if "latents" not in store:
                raise RuntimeError(f"{row.sample_id}: the schedule has no low-noise step after the boundary")
            temporary = stage_path.with_suffix(f".tmp-{os.getpid()}")
            torch.save({**store, "sample_id": row.sample_id, "seed": seed, "condition_arm": args.condition_arm,
                        "condition_source": condition_source, "pred_field": args.pred_field,
                        "injection_scale": args.injection_scale, "high_checkpoint": str(high_checkpoint)}, temporary)
            temporary.replace(stage_path)
            print(f"[{index + 1}/{len(selected)}] stage {stage_path.name} (resume at step {store['next_index']})", flush=True)
            continue
        if phase == "low":
            # no map_location: every scheduler tensor returns to the device it lived on (sigmas on the CPU, history
            # on the GPU).  Moving the sigmas to the GPU changes the solver coefficients by an ulp, and after the
            # remaining bf16 steps the frames are no longer identical to the single-call run.
            stage = torch.load(stage_path, weights_only=False)
            if (stage["seed"], stage["condition_arm"], stage["condition_source"], stage["pred_field"], stage["injection_scale"]) != \
                    (seed, args.condition_arm, condition_source, args.pred_field, args.injection_scale):
                raise RuntimeError(f"{row.sample_id}: the stored high-noise stage belongs to a different arm/seed")
            # bypass the pipeline's module registration: this is a view of the saved scheduler, not a component
            object.__setattr__(pipeline, "scheduler", ResumedScheduler(
                stage["scheduler"], stage["scheduler"].timesteps[stage["next_index"]:]))
            try:
                result = pipeline(**call, latents=stage["latents"].to("cuda"))
            finally:
                object.__setattr__(pipeline, "scheduler", base_scheduler)
        else:
            probe: dict = {}
            if args.save_frames_npy:
                # tests: record the latents at the expert boundary of the single-call run (no interruption)
                def record(pipe, step, timestep, kwargs, _probe=probe):
                    steps = pipe.scheduler.timesteps
                    if "latents" not in _probe and step + 1 < len(steps) and float(steps[step + 1]) < boundary_timestep:
                        _probe["latents"], _probe["next_index"] = kwargs["latents"].detach().float().cpu().clone(), step + 1
                    return {}
                call["callback_on_step_end"] = record
            result = pipeline(**call, generator=torch.Generator(device="cuda").manual_seed(seed))
            if probe:
                torch.save(probe, output.with_suffix(".boundary.pt"))
        output_frames = [np.asarray(frame) for frame in result.frames[0]]
        write_video(output_frames, output, float(data_cfg["fps"]))
        if args.save_frames_npy:
            np.save(output.with_suffix(".npy"), np.stack(output_frames))
        generated.append(
            {
                "sample_id": row.sample_id,
                "scenario": row.scenario,
                "video": str(output.resolve()),
                "condition_arm": args.condition_arm,
                "condition_source": condition_source,
                "pred_field": args.pred_field,
                "injection_scale": args.injection_scale,
                "seed": seed,
            }
        )
        print(f"[{index + 1}/{len(selected)}] {output}", flush=True)
    if phase == "high":
        print(f"[phase high] stages under {stage_dir}: {len(list(stage_dir.glob('*.pt')))}", flush=True)
        return
    (output_dir / f"generated_manifest_shard{args.shard_index:02d}.json").write_text(
        json.dumps({"videos": generated}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
