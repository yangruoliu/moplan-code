#!/usr/bin/env python
# Portions adapted from Wan2.2 (https://github.com/Wan-Video/Wan2.2 @1ea34ff, wan/image2video.py and wan/utils),
# Copyright 2024-2025 The Alibaba Wan Team Authors, Apache License 2.0; modified (diffusers modules, MoPlan condition).
"""Physics-IQ Verified generation of a Stage-2 generator (or the untrained base) under the OFFICIAL Wan2.2 i2v-A14B
protocol (profile official720) or under our deployed Stage-2 sampler (profile native, the one used for MoPlan).

The sampling loop is a line-by-line transcription of wan/image2video.py WanI2V.generate (Wan-Video/Wan2.2 @1ea34ff)
run on the diffusers modules of the same weights (the LoRA / motion adapter of our generators live in diffusers
module space):
  image   TF.to_tensor(switch frame) -> [-1, 1] -> bicubic interpolate on CPU to 720x1280 (official size rule:
          max_area 1280*720, 16:9 -> exactly 720x1280), no uint8 round trip
  noise   torch.randn(16, 21, 90, 160, float32, generator=cuda Generator(seed)) -> bit-identical to the official noise
  y       mask (first 4 frames = 1) ++ VAE(first frame ++ 80 zero frames) (fp32 VAE, posterior mean, normalized)
  text    umT5 (bf16, 512 tokens, trimmed + zero padded), negative = the official Chinese sample_neg_prompt
  loop    UniPC (flow sigmas, shift 5.0, 40 steps; timesteps / sigmas asserted equal to FlowUniPCMultistepScheduler),
          expert = high-noise if t >= 900 (boundary 0.9) else low-noise, guidance 3.5 / 3.5, CFG in fp32 on the fp32
          model outputs (official: WanModel returns float outputs), latent kept in fp32
  decode  fp32 VAE -> clamp(-1, 1) -> official save_video ((x+1)/2*255 truncated to uint8, imageio libx264 quality 8,
          16 fps, 81 frames) -> raw; README step E `ffmpeg -y -i raw -t 5 out` -> 80 frames @16 fps = 5.000 s
Our extra input (not --base): the Stage-1 plan of the switch frame and the SAME prompt (export in --exports), rendered
by plan_condition.render_v2("pred") at the generation work frame 720x1280 and its latent grid 21 x 90 x 160 (times
4j/16 s = 0 ... 5 s; past the last Stage-1 horizon (4 s) the plan is held and flagged unknown, exactly as in the
37-frame path), stored like the caches (fp16), read by the motion adapter of each expert (forward pre-hooks on all
40 blocks, scale 1) with the LoRA attached.

Resumable at the level of single denoising steps: before its time limit (--deadline) a worker stores the unfinished
video (fp32 latent, scheduler with its multistep history, y, condition) in work/partial/<id>.pt and releases the claim;
the next worker continues it bit-for-bit.  Finished mp4s are skipped; claims (work/claims/<id>) keep concurrent workers
from doing the same video; a claim whose owner process is gone (or that is older than 30 min) is taken over.
Phase embed: umT5 embeddings of every prompt of the given manifests + the negative prompt (once, 1 GPU).

Profile `native` (--profile native, class NativeGenerator): our deployed Stage-2 sampler (448x256, 41 frames, UniPC 30
steps with the pipeline's own schedule / flow shift 3, CFG 5 for both experts) on the same Verified inputs, written at
8 fps and trimmed with the same README command to 40 frames = 5.000 s.  One video is ~half a minute, so it is resumable
per video (no step states).

Manifest rows (jsonl): id, sid, image (switch frame), prompt, setting, out_name.  Stage-1 exports: <exports>/<sid>.pt
made from the same switch frame and the SAME prompt (scripts/infer_stage1_physicsiq.py).  --ckpt holds high/ and low/
(lora_params.pt, motion_adapter.pt).  Wan2.2 directory: --wan or $WAN_MODEL_DIR.

usage: gen_verified.py --phase embed --manifests M_op M_bpp --embeds E.pt
       gen_verified.py --manifest M.jsonl --embeds E.pt --run-dir RUN --work-dir WORK --seed 42 --profile native
                       --config configs/stage2/inference.yaml (--base | --ckpt CK --exports DIR --hold-known)
                       [--ids 0001,0002] [--deadline EPOCH] [--attn native|cudnn]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.infer_stage2_oracle import WAN_BENCHMARK_NEGATIVE_PROMPT  # noqa: E402

OFFICIAL_NEG = WAN_BENCHMARK_NEGATIVE_PROMPT          # Wan2.2's official sample_neg_prompt (byte-identical)
S = dict(max_area=1280 * 720, frames=81, fps=16, steps=40, shift=5.0, guide_low=3.5, guide_high=3.5, boundary=0.9,
         text_len=512, num_train_timesteps=1000, vae_stride=(4, 8, 8), patch=(1, 2, 2), solver="unipc")
NATIVE_HW = (2160, 3840)
# native = the deployed Stage-2 sampler of infer_stage2_cond_v2.py (WanImageToVideoPipeline call, 448x256, UniPC with
# the pipeline's own schedule (flow_shift 3 of the diffusers config),
# config evaluation.inference_steps 30 / guidance_scale 5 for both experts, the same Chinese negative prompt), on the
# Verified inputs, 41 frames written at 8 fps and trimmed to 5.000 s (40 frames)
N = dict(height=256, width=448, frames=41, fps=8, steps=30, guidance=5.0, flow_shift=3.0, boundary=0.9, text_len=512)
PROFILES = {"official720": dict(width=1280, height=720, raw_frames=81, fps=16),
            "native": dict(width=N["width"], height=N["height"], raw_frames=N["frames"], fps=N["fps"])}


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def official_hw(h: int, w: int) -> tuple[int, int]:
    """wan/image2video.py: latent grid from max_area and the aspect ratio of the input image."""
    ar = h / w
    lat_h = round(np.sqrt(S["max_area"] * ar) // S["vae_stride"][1] // S["patch"][1] * S["patch"][1])
    lat_w = round(np.sqrt(S["max_area"] / ar) // S["vae_stride"][2] // S["patch"][2] * S["patch"][2])
    return lat_h * S["vae_stride"][1], lat_w * S["vae_stride"][2]


def official_image(path: str) -> torch.Tensor:
    """[1, 3, h, w] float32 CPU, exactly the tensor the official pipeline feeds to the VAE as frame 0."""
    import torchvision.transforms.functional as TF
    from PIL import Image
    img = TF.to_tensor(Image.open(path).convert("RGB")).sub_(0.5).div_(0.5)
    h, w = official_hw(*img.shape[1:])
    return torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic")


def official_timesteps(steps: int, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """FlowUniPCMultistepScheduler(shift=1).set_timesteps(steps, shift=shift) of the official repo, in numpy."""
    n = S["num_train_timesteps"]
    alphas = np.linspace(1, 1 / n, n)[::-1].copy()
    train = torch.from_numpy(1.0 - alphas).to(dtype=torch.float32)
    smax, smin = train[0].item(), train[-1].item()
    sig = np.linspace(smax, smin, steps + 1).copy()[:-1]
    sig = shift * sig / (1 + (shift - 1) * sig)
    timesteps = torch.from_numpy(sig * n).to(dtype=torch.int64).numpy()
    return timesteps, np.concatenate([sig, [0.0]]).astype(np.float32)


def official_mask(frames: int, lat_h: int, lat_w: int) -> torch.Tensor:
    msk = torch.ones(1, frames, lat_h, lat_w)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    return msk.transpose(1, 2)[0]                                    # [4, 21, lat_h, lat_w]


def official_frames(video: torch.Tensor) -> np.ndarray:
    """wan.utils.utils.save_video preprocessing for one video [3, F, H, W] in [-1, 1] -> uint8 [F, H, W, 3]."""
    x = video.float().clamp(-1, 1)
    x = x.add(1).div(2)                                              # make_grid(normalize, value_range=(-1, 1)), 1 image
    return (x.permute(1, 2, 3, 0) * 255).type(torch.uint8).cpu().numpy()


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def container_duration(path: Path) -> float:
    """Container duration in seconds (libavformat's format duration): from `ffprobe -show_entries format=duration`
    when an ffprobe on PATH runs, otherwise the same field read through PyAV (pinned in requirements.txt). Raises
    when neither works, so the 5.000 s check can never be skipped."""
    tool = shutil.which("ffprobe")
    if tool:
        try:
            r = subprocess.run([tool, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                               capture_output=True, text=True)
            return float(r.stdout.strip())
        except (OSError, ValueError):
            pass                                                     # an ffprobe that does not start or prints nothing
    try:
        import av
        with av.open(str(path)) as container:
            if container.duration is not None:
                return container.duration / av.time_base
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cannot measure the duration of {path}: no working ffprobe on PATH and PyAV failed ({exc})") from exc
    raise RuntimeError(f"cannot measure the duration of {path}: no working ffprobe on PATH and PyAV reports none")


def duration_tool() -> str:
    """The tool container_duration() uses: "ffprobe" (a working ffprobe on PATH) or "pyav"; raises when neither is
    usable. Called once before any model is loaded, so a missing tool stops the run before the first video."""
    tool = shutil.which("ffprobe")
    if tool:
        try:
            if subprocess.run([tool, "-version"], capture_output=True).returncode == 0:
                return "ffprobe"
        except OSError:
            pass
    try:
        import av  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("no way to measure output durations: no working ffprobe on PATH and PyAV is not installed") from exc
    return "pyav"


def probe(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return {"frames": n, "fps": fps, "width": w, "height": h, "duration": container_duration(path)}


def write_official(frames: np.ndarray, raw: Path, tmp: Path, fps: int) -> None:
    import imageio
    raw.parent.mkdir(parents=True, exist_ok=True)
    part = raw.with_name(raw.stem + f".tmp{os.getpid()}.mp4")
    writer = imageio.get_writer(str(part), fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    part.replace(raw)
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw), "-t", "5", str(tmp)], check=True)


# ------------------------------------------------------------------------------------------------------- claims
def owner_alive(info: dict) -> bool:
    """The claiming process still runs (only checkable on this host; other hosts count as alive)."""
    if info.get("host") != socket.gethostname():
        return True
    try:
        os.kill(int(info.get("pid", "0")), 0)
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return int(info.get("pid", "0")) > 0


def claim(claims: Path, key: str) -> bool:
    path = claims / key
    try:
        path.mkdir()
    except FileExistsError:
        owner = (path / "owner").read_text().split() if (path / "owner").is_file() else []
        info = dict(x.split("=", 1) for x in owner if "=" in x)
        age = time.time() - path.stat().st_mtime
        mine = info.get("host") == socket.gethostname() and info.get("pid") == str(os.getpid())
        if mine:
            return True
        if age < 120 or (owner_alive(info) and age < 1800):
            return False
        shutil.rmtree(path, ignore_errors=True)
        try:
            path.mkdir()
        except FileExistsError:
            return False
    (path / "owner").write_text(f"host={socket.gethostname()} pid={os.getpid()} "
                                f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} t={time.time():.0f}\n")
    return True


def release(claims: Path, key: str) -> None:
    shutil.rmtree(claims / key, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------- phases
def phase_embed(a) -> None:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import UMT5EncoderModel
    prompts = sorted({json.loads(l)["prompt"] for m in a.manifests for l in Path(m).read_text().splitlines() if l.strip()})
    wan = Path(a.wan)
    vae = AutoencoderKLWan.from_pretrained(wan / "vae", torch_dtype=torch.float32)
    encoder = UMT5EncoderModel.from_pretrained(wan, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cuda")
    pipe = WanImageToVideoPipeline.from_pretrained(wan, vae=vae, image_encoder=None, transformer=None, transformer_2=None,
                                                   text_encoder=encoder, torch_dtype=torch.bfloat16)
    table, negative = {}, None
    with torch.inference_mode():
        for prompt in prompts:
            pos, neg = pipe.encode_prompt(prompt=prompt, negative_prompt=OFFICIAL_NEG, do_classifier_free_guidance=True,
                                          num_videos_per_prompt=1, max_sequence_length=S["text_len"], device=torch.device("cuda"))
            table[prompt] = trim(pos[0])
            negative = trim(neg[0])
    Path(a.embeds).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(a.embeds) + f".tmp{os.getpid()}")
    torch.save({"prompts": table, "negative": negative, "negative_prompt": OFFICIAL_NEG, "text_len": S["text_len"],
                "manifests": [str(m) for m in a.manifests]}, tmp)
    tmp.replace(a.embeds)
    log(f"EMBED_DONE {len(table)} prompts -> {a.embeds} (token counts {min(v.shape[0] for v in table.values())}.."
        f"{max(v.shape[0] for v in table.values())}, negative {negative.shape[0]})")


def trim(embed: torch.Tensor) -> torch.Tensor:
    """[512, 4096] zero-padded -> [n, 4096] (the pipeline pads trimmed umT5 states with exact zeros)."""
    nonzero = (embed != 0).any(dim=1).nonzero()
    n = int(nonzero.max()) + 1 if len(nonzero) else 0
    if n == 0 or (embed[n:] != 0).any():
        raise RuntimeError("unexpected prompt embedding padding")
    return embed[:n].cpu().clone()


def pad(embed: torch.Tensor) -> torch.Tensor:
    out = embed.new_zeros(S["text_len"], embed.shape[1])
    out[: embed.shape[0]] = embed
    return out[None].to("cuda", torch.bfloat16)


def hold_known_beyond_horizon(cond):
    """--hold-known: latents after the last one whose `known` channel (20) is set copy that latent's full 27-channel
    condition, i.e. the Stage-1 end state is held as KNOWN, the convention the generator was trained with (same rule as
    scripts/s2eval/hold_known_beyond_horizon.py, including its edge case: the last known latent comes from the footprint
    of channel 20, so when every object has left the frame before 4 s the hold starts earlier).
    cond: [27, L, H, W] or [B, 27, L, H, W]."""
    x = cond if cond.dim() == 5 else cond[None]
    k = x[:, 20].flatten(2).abs().sum(-1).sum(0)                 # (L,)
    nz = (k > 0).nonzero().flatten()
    if len(nz) == 0 or int(nz[-1]) == x.shape[2] - 1:
        return cond
    last = int(nz[-1]); y = x.clone(); y[:, :, last + 1:] = x[:, :, last:last + 1]
    return y if cond.dim() == 5 else y[0]


class Generator:
    def __init__(self, a, contract: dict) -> None:
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
        self.a, self.contract = a, contract
        wan = Path(a.wan)
        t0 = time.time()
        self.vae = AutoencoderKLWan.from_pretrained(wan / "vae", torch_dtype=torch.float32).to("cuda").eval()
        cfg = self.vae.config
        self.lat_mean = torch.tensor(cfg.latents_mean).view(1, cfg.z_dim, 1, 1, 1).to("cuda", torch.float32)
        self.lat_inv_std = (1.0 / torch.tensor(cfg.latents_std)).view(1, cfg.z_dim, 1, 1, 1).to("cuda", torch.float32)
        where = {} if a.offload == "swap" else {"device_map": "cuda"}
        self.experts = {}
        for name in ("transformer", "transformer_2"):            # transformer = high-noise expert, transformer_2 = low-noise
            self.experts[name] = WanTransformer3DModel.from_pretrained(wan, subfolder=name, torch_dtype=torch.bfloat16, **where).eval()
            if a.attn != "native":
                self.experts[name].set_attention_backend(a.attn)
        self.adapters, self.controllers, self.prefetched = {}, {}, {}
        from scripts.control import plan_condition as pc
        self._stage1_dense = pc.stage1_dense
        if not a.base:
            import yaml
            from scripts.infer_stage2_oracle import ResidualController, attach_lora, load_adapter
            from motion_jepa.stage2_condition import interface_from_config
            model_cfg = yaml.safe_load(Path(a.config).read_text())["model"]
            if interface_from_config(model_cfg) != "legacy27":
                raise SystemExit("only legacy27 generators are supported here")
            for name, sub in (("transformer", "high"), ("transformer_2", "low")):
                expert = self.experts[name]
                if a.offload == "swap":
                    expert.to("cuda")
                attach_lora(expert, Path(a.ckpt) / sub)
                self.adapters[name] = load_adapter(Path(a.ckpt) / sub, model_cfg, expert.config.num_attention_heads * expert.config.attention_head_dim)
                self.controllers[name] = ResidualController(expert, list(model_cfg["injection_blocks"]), 1.0, self.adapters[name])
                if a.offload == "swap":
                    expert.to("cpu")
        if a.offload == "swap":
            for expert in self.experts.values():
                expert.to("cpu")
            self.on_gpu = None
        self.scheduler0 = UniPCMultistepScheduler.from_pretrained(wan, subfolder="scheduler")
        self.scheduler0 = UniPCMultistepScheduler.from_config(self.scheduler0.config, flow_shift=S["shift"])
        probe_s = copy.deepcopy(self.scheduler0)
        probe_s.set_timesteps(S["steps"], device="cuda")
        self.official_t, self.official_s = official_timesteps(S["steps"], S["shift"])
        # diffusers 0.38 spaces the unshifted sigmas as linspace(1, 1/1000) (first sigma 1 - 1e-6, last timestep 117);
        # the official FlowUniPC uses linspace(0.999, 0) (first 0.9998, last 113): the official schedule is imposed
        self.diffusers_t = probe_s.timesteps.cpu().numpy().tolist()
        got = self.new_scheduler()
        if not (np.array_equal(got.timesteps.cpu().numpy(), self.official_t) and np.array_equal(got.sigmas.numpy(), self.official_s)):
            raise SystemExit("could not impose the official FlowUniPC schedule")
        self.timesteps = [int(t) for t in self.official_t]
        self.boundary_t = S["boundary"] * S["num_train_timesteps"]
        self.n_high = sum(t >= self.boundary_t for t in self.timesteps)
        embeds = torch.load(a.embeds, map_location="cpu", weights_only=True)
        if embeds["negative_prompt"] != OFFICIAL_NEG:
            raise SystemExit("prompt embeddings were made with another negative prompt")
        self.embeds, self.negative = embeds["prompts"], pad(embeds["negative"])
        log(f"models ready in {time.time() - t0:.0f}s: base={a.base} attn={a.attn} offload={a.offload} official timesteps {self.timesteps} "
            f"(high-noise steps {self.n_high}; diffusers default would be {self.diffusers_t}) "
            f"gpu_alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    def new_scheduler(self):
        """A fresh UniPC (flow sigmas, bh2, order 2, x0, lower-order final) carrying the official sigmas / timesteps."""
        sched = copy.deepcopy(self.scheduler0)
        sched.set_timesteps(S["steps"], device="cuda")             # resets the multistep state
        sched.sigmas = torch.from_numpy(self.official_s.copy())     # float32, CPU (as both schedulers keep them)
        sched.timesteps = torch.from_numpy(self.official_t.copy()).to("cuda", torch.int64)
        return sched

    def expert(self, name: str):
        if self.a.offload == "swap" and self.on_gpu != name:
            for other, model in self.experts.items():
                if other != name:
                    model.to("cpu")
            torch.cuda.empty_cache()
            self.experts[name].to("cuda")
            self.on_gpu = name
        return self.experts[name]

    def cpu_part(self, item: dict):
        """Everything of the condition that does not need the image latent: export, plan, Stage-1 dense fields (the
        latter is the slow part at 720p; it runs in a background thread for the next video)."""
        from scripts import motion_plan as mp
        from scripts.control import plan_condition as pc
        export = torch.load(Path(self.a.exports) / f"{item['sid']}.pt", map_location="cpu", weights_only=False)
        if export.get("caption") != item["prompt"]:
            raise RuntimeError(f"{item['sid']}: the Stage-1 export was made with another prompt")
        size, times = (720, 1280), pc.latent_times(S["frames"], float(S["fps"]))
        plan = mp.from_stage1_export(export, size, native_hw=NATIVE_HW)
        return export, plan, self._stage1_dense(export, times, size, NATIVE_HW)

    def prefetch(self, items: list) -> None:
        """Keep the CPU part of exactly these videos (the current and the next one) computed or in flight."""
        if self.a.base:
            return
        keep = {it["sid"] for it in items if it is not None}
        for sid in [s for s in self.prefetched if s not in keep]:
            self.prefetched.pop(sid).cancel()
        if not hasattr(self, "pool"):
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(1)
        for it in items:
            if it is not None and it["sid"] not in self.prefetched:
                self.prefetched[it["sid"]] = self.pool.submit(self.cpu_part, it)

    def condition(self, item: dict, raw_latent: torch.Tensor) -> torch.Tensor:
        """plan_condition.render_v2("pred") at 720x1280 / 21 x 90 x 160, identical to calling it directly (its
        stage1_dense step is only computed earlier, by the same function)."""
        from scripts.control import plan_condition as pc
        future = self.prefetched.pop(item["sid"], None)
        export, plan, dense = future.result() if future is not None else self.cpu_part(item)
        size, times = (720, 1280), pc.latent_times(S["frames"], float(S["fps"]))
        if tuple(raw_latent.shape[-2:]) != (size[0] // 8, size[1] // 8):
            raise RuntimeError("latent grid is not 90 x 160")
        pc.stage1_dense = lambda *args, **kwargs: dense
        try:
            cond = pc.render_v2(getattr(self.a, "plan_kind", "pred"), plan, raw_latent[0].float().cpu(), times, size, export=export, native_hw=NATIVE_HW)
        finally:
            pc.stage1_dense = self._stage1_dense
        if getattr(self.a, "hold_known", False):
            cond = hold_known_beyond_horizon(cond)
        return cond.half()                                         # the cache precision every earlier run read

    def run(self, item: dict, seed: int, deadline: float, partial: Path, claims: Path) -> dict | None:
        a, stats = self.a, {"segments": 1}
        t_item = time.time()
        torch.cuda.reset_peak_memory_stats()
        signature = {"sid": item["sid"], "seed": seed, "prompt": item["prompt"], "contract": self.contract["sha"]}
        state = None
        if partial.is_file():
            state = torch.load(partial, weights_only=False)
            if state["signature"] != signature:
                raise RuntimeError(f"{partial} belongs to another contract")
            stats = state["stats"]
            stats["segments"] += 1
            cond_y, cond27, latents, sched, start = (state["cond_y"].to("cuda"), state["cond27"], state["latents"].to("cuda"),
                                                     state["scheduler"], int(state["next_index"]))
            log(f"{item['id']} resume at step {start} (segment {stats['segments']})")
        else:
            image = official_image(item["image"])
            h, w = image.shape[-2:]
            if (h, w) != (720, 1280):
                raise RuntimeError(f"{item['id']}: official size rule gives {h}x{w}")
            lat_h, lat_w, lat_t = h // 8, w // 8, (S["frames"] - 1) // 4 + 1
            g = torch.Generator(device="cuda").manual_seed(seed)
            latents = torch.randn(16, lat_t, lat_h, lat_w, dtype=torch.float32, generator=g, device="cuda")[None]
            t0 = time.time()
            video = torch.concat([image.transpose(0, 1)[None], torch.zeros(1, 3, S["frames"] - 1, h, w)], dim=2).to("cuda")
            with torch.no_grad():
                raw = self.vae.encode(video).latent_dist.mode()     # [1, 16, 21, 90, 160] posterior mean
            del video
            cond_y = torch.concat([official_mask(S["frames"], lat_h, lat_w)[None].to("cuda"), (raw - self.lat_mean) * self.lat_inv_std], dim=1)
            stats["sec_vae_encode"] = round(time.time() - t0, 2)
            t0 = time.time()
            cond27 = None if a.base else self.condition(item, raw)
            stats["sec_condition"] = round(time.time() - t0, 2)
            del raw
            sched = self.new_scheduler()
            start = 0
            stats.update(sec_high=0.0, sec_low=0.0, steps_high=0, steps_low=0)
        if cond27 is not None:
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for name, adapter in self.adapters.items():
                    self.controllers[name].set(adapter(cond27.float()[None].to("cuda", torch.bfloat16)))
            stats["sec_adapter"] = round(stats.get("sec_adapter", 0) + time.time() - t0, 2)
        context, negative = pad(self.embeds[item["prompt"]]), self.negative
        step_times = []
        for i in range(start, S["steps"]):
            t = sched.timesteps[i]
            high = float(t) >= self.boundary_t
            if step_times:                                           # leave room for one more step + saving
                need = max(step_times) * 1.15 + 120
                if time.time() + need > deadline:
                    self.save_partial(partial, signature, stats, cond_y, cond27, latents, sched, i)
                    return None
            ts = time.time()
            model = self.expert("transformer" if high else "transformer_2")
            gs = S["guide_high"] if high else S["guide_low"]
            x_in = torch.concat([latents, cond_y], dim=1).to(torch.bfloat16)
            tt = t.expand(1)
            with torch.no_grad():
                cond = model(hidden_states=x_in, timestep=tt, encoder_hidden_states=context, return_dict=False)[0].float()
                uncond = model(hidden_states=x_in, timestep=tt, encoder_hidden_states=negative, return_dict=False)[0].float()
                noise_pred = uncond + gs * (cond - uncond)
                latents = sched.step(noise_pred, t, latents, return_dict=False)[0]
            del cond, uncond, noise_pred, x_in
            torch.cuda.synchronize()
            dt = time.time() - ts
            step_times.append(dt)
            key = "high" if high else "low"
            stats[f"sec_{key}"] = round(stats[f"sec_{key}"] + dt, 2)
            stats[f"steps_{key}"] += 1
            os.utime(claims / item["id"], None)
        if time.time() + 150 > deadline:                            # no time to decode + write: keep the finished latent
            self.save_partial(partial, signature, stats, cond_y, cond27, latents, sched, S["steps"])
            return None
        t0 = time.time()
        with torch.no_grad():
            lat = latents.to(self.vae.dtype) / self.lat_inv_std + self.lat_mean
            video = self.vae.decode(lat, return_dict=False)[0][0]     # [3, 81, 720, 1280]
        frames = official_frames(video)
        del video, lat
        stats["sec_decode"] = round(time.time() - t0, 2)
        stats["peak_gpu_gib_alloc_reserved"] = [round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                                round(torch.cuda.max_memory_reserved() / 2**30, 2)]
        stats["sec_this_segment"] = round(time.time() - t_item, 1)
        stats["frames"] = list(frames.shape)
        return {"frames": frames, "stats": stats}

    def save_partial(self, path, signature, stats, cond_y, cond27, latents, sched, next_index) -> None:
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        torch.save({"signature": signature, "stats": stats, "cond_y": cond_y.cpu(), "cond27": cond27, "latents": latents.cpu(),
                    "scheduler": copy.deepcopy(sched), "next_index": next_index}, tmp)
        tmp.replace(path)
        log(f"deadline: stored {path.name} at step {next_index}")


class NativeGenerator:
    """Profile `native`: exactly the deployed infer_stage2_cond_v2.py generation call (phase `full` without CPU offload:
    both experts resident, numerically the same pipeline call), on the Verified switch frame and prompt.
      image      PIL LANCZOS resize of the switch frame to 448x256 (the deployed sampler and the caches do the same)
      condition  plan_condition.render_v2("pred") of the Stage-1 plan at the 448x256 work frame, latent grid 11x32x56,
                 image latent = Wan VAE posterior mean of (stretched first frame ++ zero frames) as cache_stage2_cond_v2.py
                 makes it, stored fp16 like the caches, read by each expert's adapter (bf16, autocast) with the LoRA
      sampler    pipeline(image, height 256, width 448, num_frames 41, num_inference_steps 30, guidance_scale 5,
                 prompt / negative prompt embeddings, generator = cuda Generator(seed)); scheduler untouched
      frames     result.frames[0] (float [0, 1]) -> clip(x * 255).round() uint8, as the deployed path
    The native size of the switch frame is NATIVE_HW (Physics-IQ, 3840x2160); a caller may set `a.native_hw` for other
    images (the Stage-1 export's letterbox geometry is used whenever the export records it)."""

    def __init__(self, a, contract: dict) -> None:
        import yaml
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
        if WAN_BENCHMARK_NEGATIVE_PROMPT != OFFICIAL_NEG:
            raise SystemExit("the deployed negative prompt differs from the official one")
        self.a, self.contract = a, contract
        self.native_hw = tuple(getattr(a, "native_hw", None) or NATIVE_HW)
        if getattr(a, "stage1_only", False):
            self.durations = []
            log("stage-1 alone: no video model; motion_plan.to_crude_video of each Stage-1 plan")
            return
        t0 = time.time()
        cfg = yaml.safe_load(Path(a.config).read_text())
        ev = cfg["evaluation"]
        if int(ev["inference_steps"]) != N["steps"] or float(ev["guidance_scale"]) != N["guidance"]:
            raise SystemExit(f"config evaluation {ev} differs from the native profile {N}")
        wan = Path(a.wan)
        vae = AutoencoderKLWan.from_pretrained(wan / "vae", torch_dtype=torch.float32)
        experts = {}
        for name in ("transformer", "transformer_2"):
            experts[name] = WanTransformer3DModel.from_pretrained(wan, subfolder=name, torch_dtype=torch.bfloat16, device_map="cuda")
            if a.attn != "native":
                experts[name].set_attention_backend(a.attn)
        self.pipe = WanImageToVideoPipeline.from_pretrained(wan, vae=vae, image_encoder=None, transformer=experts["transformer"],
                                                            transformer_2=experts["transformer_2"], text_encoder=None, tokenizer=None,
                                                            torch_dtype=torch.bfloat16)
        self.adapters, self.controllers = {}, {}
        if not a.base:
            from scripts.infer_stage2_oracle import ResidualController, attach_lora, load_adapter
            from motion_jepa.stage2_condition import interface_from_config
            model_cfg = cfg["model"]
            if interface_from_config(model_cfg) != "legacy27":
                raise SystemExit("only legacy27 generators are supported here")
            for name, sub in (("transformer_2", "low"), ("transformer", "high")):     # the deployed order
                expert = getattr(self.pipe, name)
                attach_lora(expert, Path(a.ckpt) / sub)
                self.adapters[name] = load_adapter(Path(a.ckpt) / sub, model_cfg, expert.config.num_attention_heads * expert.config.attention_head_dim)
                self.controllers[name] = ResidualController(expert, list(model_cfg["injection_blocks"]), 1.0, self.adapters[name])
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        sc = self.pipe.scheduler.config
        if abs(float(sc.flow_shift) - N["flow_shift"]) > 1e-9 or abs(float(self.pipe.config.boundary_ratio) - N["boundary"]) > 1e-9:
            raise SystemExit(f"pipeline scheduler flow_shift {sc.flow_shift} / boundary {self.pipe.config.boundary_ratio} differ from {N}")
        embeds = torch.load(a.embeds, map_location="cpu", weights_only=True)
        if embeds["negative_prompt"] != OFFICIAL_NEG:
            raise SystemExit("prompt embeddings were made with another negative prompt")
        self.embeds, self.negative = embeds["prompts"], pad(embeds["negative"])
        self.durations: list[float] = []
        probe_s = copy.deepcopy(self.pipe.scheduler)
        probe_s.set_timesteps(N["steps"], device="cuda")
        log(f"native models ready in {time.time() - t0:.0f}s: base={a.base} attn={a.attn} scheduler {type(probe_s).__name__} "
            f"flow_shift {sc.flow_shift} timesteps {probe_s.timesteps.tolist()} gpu_alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    def time_needed(self) -> float:
        return (max(self.durations) * 1.3 if self.durations else 150.0) + 60

    def prefetch(self, items) -> None:
        return None

    def condition(self, item: dict, image) -> torch.Tensor:
        from scripts import motion_plan as mp
        from scripts.cache_stage2_oracle import preprocess
        from scripts.control import plan_condition as pc
        export = torch.load(Path(self.a.exports) / f"{item['sid']}.pt", map_location="cpu", weights_only=False)
        if export.get("caption") != item["prompt"]:
            raise RuntimeError(f"{item['sid']}: the Stage-1 export was made with another prompt")
        size, times = (N["height"], N["width"]), pc.latent_times(N["frames"], float(N["fps"]))
        frames = np.zeros((N["frames"], N["height"], N["width"], 3), np.uint8)
        frames[0] = np.asarray(image, dtype=np.uint8)
        video = preprocess(frames, N["height"], N["width"]).to("cuda")                 # [1, 3, F, H, W] in [-1, 1]
        image_video = torch.zeros_like(video)
        image_video[:, :, :1] = video[:, :, :1]
        with torch.no_grad():
            latent = self.pipe.vae.encode(image_video).latent_dist.mode()[0].float().cpu()
        plan = mp.from_stage1_export(export, size, native_hw=self.native_hw)
        cond = pc.render_v2(getattr(self.a, "plan_kind", "pred"), plan, latent, times, size, export=export, native_hw=self.native_hw)
        if getattr(self.a, "hold_known", False):
            cond = hold_known_beyond_horizon(cond)
        return cond.half()

    def run(self, item: dict, seed: int, deadline: float, partial: Path, claims: Path) -> dict | None:
        from PIL import Image
        t_item, stats = time.time(), {"segments": 1}
        with Image.open(item["image"]) as img:
            if (img.height, img.width) != self.native_hw:
                raise RuntimeError(f"{item['id']}: switch frame is {img.size}")
            image = img.convert("RGB").resize((N["width"], N["height"]), Image.LANCZOS)
        if getattr(self.a, "stage1_only", False):
            return self.run_stage1_only(item, image, t_item, stats, claims)
        torch.cuda.reset_peak_memory_stats()
        if self.controllers:
            t0 = time.time()
            cond = self.condition(item, image)
            stats["sec_condition"] = round(time.time() - t0, 2)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for name, adapter in self.adapters.items():
                    self.controllers[name].set(adapter(cond.float()[None].to("cuda", torch.bfloat16)))
        call = dict(image=image, height=N["height"], width=N["width"], num_frames=N["frames"], num_inference_steps=N["steps"],
                    guidance_scale=N["guidance"], prompt_embeds=pad(self.embeds[item["prompt"]]), negative_prompt_embeds=self.negative)
        t0 = time.time()
        result = self.pipe(**call, generator=torch.Generator(device="cuda").manual_seed(seed))
        torch.cuda.synchronize()
        stats["sec_pipeline"] = round(time.time() - t0, 2)
        native = np.stack([np.asarray(frame) for frame in result.frames[0]])
        if native.dtype != np.uint8:
            native = np.clip(native * (255.0 if float(native.max()) <= 1.5 else 1.0), 0, 255).round().astype(np.uint8)
        stats["peak_gpu_gib_alloc_reserved"] = [round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                                round(torch.cuda.max_memory_reserved() / 2**30, 2)]
        stats["sec_this_segment"] = round(time.time() - t_item, 1)
        stats["frames"] = list(native.shape)
        self.durations.append(time.time() - t_item)
        os.utime(claims / item["id"], None)
        return {"frames": native, "stats": stats}

    def run_stage1_only(self, item: dict, image, t_item: float, stats: dict, claims: Path) -> dict:
        """--stage1-only (ablation): the same Stage-1 plan condition() renders, as a video without any generator:
        every planned object cut out of the 448x256 switch frame and dragged along its predicted affine (area left
        behind inpainted), held at its last known position past the Stage-1 horizon; same 41 frames @8 fps."""
        from scripts import motion_plan as mp
        export = torch.load(Path(self.a.exports) / f"{item['sid']}.pt", map_location="cpu", weights_only=False)
        if export.get("caption") != item["prompt"]:
            raise RuntimeError(f"{item['sid']}: the Stage-1 export was made with another prompt")
        plan = mp.from_stage1_export(export, (N["height"], N["width"]), native_hw=self.native_hw)
        t0 = time.time()
        native, _ = mp.to_crude_video(plan, np.asarray(image, dtype=np.uint8), [k / float(N["fps"]) for k in range(N["frames"])])
        stats.update(sec_pipeline=round(time.time() - t0, 2), peak_gpu_gib_alloc_reserved=[0.0, 0.0],
                     sec_this_segment=round(time.time() - t_item, 1), frames=list(native.shape),
                     plan_objects=len(plan.objects), plan_objects_moved=sum(1 for o in plan.objects if o.max_point_motion() > 0.5))
        self.durations.append(time.time() - t_item)
        os.utime(claims / item["id"], None)
        return {"frames": np.ascontiguousarray(native.astype(np.uint8)), "stats": stats}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("embed", "gen"), default="gen")
    p.add_argument("--manifests", nargs="*", default=[])
    p.add_argument("--manifest", default="")
    p.add_argument("--embeds", required=True)
    p.add_argument("--wan", default=os.environ.get("WAN_MODEL_DIR", ""), help="Wan2.2-I2V-A14B-Diffusers directory (default: $WAN_MODEL_DIR)")
    p.add_argument("--run-dir", default="")
    p.add_argument("--work-dir", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base", action="store_true")
    p.add_argument("--ckpt", default="")
    p.add_argument("--config", default="")
    p.add_argument("--exports", default="", help="Stage-1 exports flows/ folder made with this setting's prompts")
    p.add_argument("--tag", default="")
    p.add_argument("--s1tag", default="")
    p.add_argument("--plan-kind", choices=("pred", "zero"), default="pred",
                   help="zero: the Stage-1 plan's objects rendered at rest (zero-flow ablation)")
    p.add_argument("--stage1-only", action="store_true",
                   help="no video model: the Stage-1 plan's objects dragged across the switch frame (ablation; native profile, needs --exports)")
    p.add_argument("--hold-known", action="store_true",
                   help="hold the Stage-1 end state as KNOWN past the last Stage-1 horizon (the GT-SFT training convention)")
    p.add_argument("--ids", default="", help="comma list of 4-digit ids (in this order); default all 198")
    p.add_argument("--deadline", type=float, default=0.0)
    p.add_argument("--max-videos", type=int, default=0)
    p.add_argument("--attn", default="native", help="native (torch SDPA) or a diffusers attention backend, e.g. _native_cudnn")
    p.add_argument("--offload", choices=("none", "swap"), default="none")
    p.add_argument("--profile", choices=tuple(PROFILES), default="official720",
                   help="official720 = Wan2.2 official i2v defaults (720p, 81f @16 fps); native = our deployed sampler (448x256, 41f @8 fps)")
    p.add_argument("--shard", default="", help="i/n: only the ids at positions i, i+n, ... of the 198 (fixed shard of a run)")
    a = p.parse_args(argv)
    if not a.wan:
        p.error("pass --wan or set WAN_MODEL_DIR")
    if a.phase == "embed":
        return phase_embed(a)
    if a.stage1_only and (a.base or not a.exports or a.profile != "native"):
        p.error("--stage1-only needs --exports and the native profile (no --base)")
    if not a.base and not a.stage1_only and not (a.ckpt and a.config and a.exports):
        p.error("a generator needs --ckpt, --config and --exports (or --base)")
    if a.profile == "native" and not a.config:
        p.error("the native profile reads its sampler settings from --config (also for --base)")
    prof = PROFILES[a.profile]
    deadline = a.deadline or time.time() + 365 * 86400
    run_dir, work = Path(a.run_dir), Path(a.work_dir)
    for sub in ("raw", "meta", "claims", "partial"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in Path(a.manifest).read_text().splitlines() if l.strip()]
    if len(items) != 198:
        raise SystemExit("manifest must hold the 198 take-1 rows")
    by_id = {it["id"]: it for it in items}
    order = [i for i in a.ids.split(",") if i] if a.ids else [it["id"] for it in items]
    if a.shard:
        si, sn = (int(v) for v in a.shard.split("/"))
        order = [i for k, i in enumerate(order) if k % sn == si]

    # run contract: everything that defines the videos of this folder; workers of later jobs must agree
    contract = {"profile": a.profile, "settings": S if a.profile == "official720" else N, "negative_prompt": OFFICIAL_NEG, "seed": a.seed, "manifest": a.manifest, "manifest_sha256": sha256(Path(a.manifest)),
                "embeds_sha256": sha256(Path(a.embeds)), "wan": a.wan, "base": a.base, "tag": a.tag, "s1tag": a.s1tag,
                "attn": a.attn}
    if a.config:
        contract.update(config=a.config, config_sha256=sha256(Path(a.config)))
    if a.stage1_only:
        contract.update(stage1_only=True, exports=a.exports)
    elif not a.base:
        contract.update(ckpt=str(Path(a.ckpt).resolve()), exports=a.exports,
                        lora_sha256={k: sha256(Path(a.ckpt) / k / "lora_params.pt") for k in ("high", "low")},
                        adapter_sha256={k: sha256(Path(a.ckpt) / k / "motion_adapter.pt") for k in ("high", "low")})
    if a.plan_kind != "pred":   # only then: earlier folders keep their contract sha
        contract.update(plan_kind=a.plan_kind)
    contract["sha"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    cpath = work / "CONFIG.json"
    if cpath.is_file():
        old = json.loads(cpath.read_text())
        if old["sha"] != contract["sha"]:
            raise SystemExit(f"{cpath} belongs to another contract: {[k for k in contract if old.get(k) != contract[k]]}")
    else:
        tmp = cpath.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(contract, indent=1, ensure_ascii=False) + "\n")
        tmp.replace(cpath)

    def todo():
        partials = [p.stem for p in (work / "partial").glob("*.pt")]
        ids = [i for i in partials if i in by_id and i in order] + [i for i in order if i not in partials]
        return [by_id[i] for i in ids if not (run_dir / by_id[i]["out_name"]).is_file()]

    pending = todo()
    log(f"{len(pending)} of {len(order)} videos to do in {run_dir.name}; seed {a.seed}; deadline in {(deadline - time.time()) / 60:.0f} min")
    if not pending:
        log("GEN_VERIFIED_DONE nothing to do")
        return
    log(f"output durations measured with {duration_tool()}")          # raises before any model is loaded
    gen = Generator(a, contract) if a.profile == "official720" else NativeGenerator(a, contract)
    # provenance without host names or absolute paths: the code file relative to the repository, plus its sha256
    code = Path(__file__).resolve()
    env = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "torch": torch.__version__, "diffusers": __import__("diffusers").__version__,
           "code": code.relative_to(code.parents[2]).as_posix(), "code_sha256": sha256(code)}
    done = 0
    queue = todo()
    for k, item in enumerate(queue):
        gen.prefetch([item, queue[k + 1] if k + 1 < len(queue) else None])
        if time.time() + (300 if a.profile == "official720" else gen.time_needed()) > deadline:
            log("deadline: no time to start another video")
            break
        out = run_dir / item["out_name"]
        if out.is_file() or not claim(work / "claims", item["id"]):
            continue
        try:
            if out.is_file():
                continue
            partial = work / "partial" / f"{item['id']}.pt"
            result = gen.run(item, a.seed, deadline, partial, work / "claims")
            if result is None:
                break
            t0 = time.time()
            raw = work / "raw" / item["out_name"]
            tmp = work / f"tmp_{os.getpid()}_{item['out_name']}"
            write_official(result["frames"], raw, tmp, prof["fps"])
            info_raw, info = probe(raw), probe(tmp)
            meta = {"id": item["id"], "sid": item["sid"], "out_name": item["out_name"], "image": Path(item["image"]).name, "prompt": item["prompt"],
                    "setting": item["setting"], "seed": a.seed, "run_dir": run_dir.name, "contract_sha": contract["sha"],
                    "raw": info_raw, "trimmed": info, **result["stats"], "sec_write": round(time.time() - t0, 1), **env}
            if not a.base:
                meta["export"] = f"{Path(a.exports).name}/{item['sid']}.pt"
            ok = info["frames"] == 5 * prof["fps"] and abs(info["fps"] - prof["fps"]) < 1e-3 and info_raw["frames"] == prof["raw_frames"] and \
                (info["width"], info["height"]) == (prof["width"], prof["height"]) and abs(info["duration"] - 5.0) <= 1e-3
            if not ok:
                (work / "meta" / f"{item['id']}.BAD.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
                log(f"BAD output {item['id']}: raw {info_raw} trimmed {info}")
                tmp.unlink(missing_ok=True)
                continue
            os.replace(tmp, out)
            (work / "meta" / f"{item['id']}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
            partial.unlink(missing_ok=True)
            done += 1
            st = result["stats"]
            if a.profile == "native":
                log(f"done {item['id']} total {st['sec_this_segment']}s (pipeline {st['sec_pipeline']}s cond {st.get('sec_condition')}s) "
                    f"write {meta['sec_write']}s peak GiB {st['peak_gpu_gib_alloc_reserved']} raw {info_raw} trimmed {info}")
            else:
                log(f"done {item['id']} segments {st['segments']} high {st['steps_high']} steps {st['sec_high']:.0f}s low {st['steps_low']} steps "
                    f"{st['sec_low']:.0f}s encode {st.get('sec_vae_encode')}s cond {st.get('sec_condition')}s decode {st['sec_decode']}s "
                    f"write {meta['sec_write']}s peak GiB {st['peak_gpu_gib_alloc_reserved']} trimmed {info}")
            if a.max_videos and done >= a.max_videos:
                break
        finally:
            release(work / "claims", item["id"])
    log(f"GEN_VERIFIED_EXIT wrote {done}; {len(todo())} left in {run_dir.name}")


if __name__ == "__main__":
    main()
