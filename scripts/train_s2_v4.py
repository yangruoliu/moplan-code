#!/usr/bin/env python
"""Stage-2 training on real continuations with the v2 condition contract.

Every clip of a v2 cache (scripts/cache_stage2_cond_v2.py) carries four conditions rendered by one function:
  real  the plan measured on the clip      pred  the frozen Stage-1 plan      zero  the same objects at rest      null  no hint
--mix lists what the adapter is fed and from which data pool, e.g.
  S0  "p101:real:0.5,phyco:real:0.5"                      learn to follow a measured plan
  P   "p101:pred:0.5,p101:real:0.28,phyco:real:0.12,p101:null:0.1"   read the noisy Stage-1 plan, keep the clean skill, keep the prior
The target is always the real continuation.  Loss terms (each one logged on its own):
  flow matching                 plain, or weighted by the clip's moving-object tubes (--fm-weight-mode tube)
  keep (--keep-weight)          on null-fed samples the velocity is also pulled to a frozen reference policy (--reference-init):
                                "no hint" must stay the model we started from
  kinematic (--kin-weight)      direction / displacement of the one-step clean estimate read by a frozen latent motion
                                readout (--phi-lat) against the measured motion, inside a validated sigma window
The legacy self-warp agreement term is not available here.

Clips have different lengths, so the batch size is 1 and gradients are accumulated.  Sampling is a pure function of
(seed, sample cursor): pool, clip, condition kind and the sigma / noise draw are reproducible and identical after a
resume.  Checkpoints have the layout of the deployed model (lora_params.pt, motion_adapter.pt) and are never rotated.
<output-dir>/OVERRIDES.json may set any argument ({expert} is substituted).
The Wan2.2-I2V-A14B-Diffusers directory is $WAN_MODEL_DIR, else the config's paths.wan_model.

RAFT round (launch/raft.sh): --dpo-pairs <winners>, --dpo-prob 0.5, --dpo-sft-weight 1, --dpo-negative-weight 0, i.e.
plain flow matching on the selected candidates mixed with GT replay; see docs/TRAINING_RAFT.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_jepa import cond_v2  # noqa: E402
from motion_jepa.dpo import dpo_terms  # noqa: E402
from motion_jepa.stage2 import ExtendedOnlyFlowAdapter, OracleFlowWanBackend, Stage2Batch, load_latent_statistics  # noqa: E402
from motion_jepa.stage2_condition import interface_from_config, load_adapter_state, read_cached_condition  # noqa: E402
from scripts.infer_stage2_oracle import build_adapter  # noqa: E402


def checked_clip_grad(accelerator, parameters, max_norm):
    """Never let a nonfinite accumulated gradient enter the optimizer state."""
    norm = accelerator.clip_grad_norm_(parameters, max_norm)
    if not bool(torch.isfinite(torch.as_tensor(norm)).all()):
        raise FloatingPointError("nonfinite accumulated Stage2 gradient; optimizer step aborted")
    return norm

SAMPLER_VERSION = "cursor_mix_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--expert", required=True, choices=("low", "high"))
    p.add_argument("--output-dir", default="")
    p.add_argument("--pools", default="", help="name=cache_dir[@ids_file],... (ids_file: one sample id per line)")
    p.add_argument("--mix", default="", help="pool:kind:weight,... with kind in real|pred|zero|null")
    p.add_argument("--extra-channels", default=",".join(cond_v2.EXTRA_NAMES), help="extra condition channels the adapter may read; the others are zeroed")
    p.add_argument("--fm-weight-mode", default="none", choices=("none", "tube"))
    p.add_argument("--keep-weight", type=float, default=0.0)
    p.add_argument("--reference-init", default="", help="checkpoint (lora_params.pt + motion_adapter.pt) of the frozen reference policy")
    p.add_argument("--kin-weight", type=float, default=0.0)
    p.add_argument("--kin-alpha-disp", type=float, default=0.5)
    p.add_argument("--kin-sigma-min", type=float, default=0.3)
    p.add_argument("--kin-sigma-max", type=float, default=0.8)
    p.add_argument("--phi-lat", default="", help="frozen latent motion readout checkpoint")
    p.add_argument("--dpo-pairs", default="", help="preference pairs (scripts/dpo/build_pairs.py): <dir>/<sid>.pt with win / lose latents")
    p.add_argument("--dpo-prob", type=float, default=0.0, help="share of micro-steps that train on a preference pair")
    p.add_argument("--dpo-beta", type=float, default=2000.0)
    p.add_argument("--dpo-sft-weight", type=float, default=0.0, help="extra flow-matching weight on the winner of a pair (DPO + SFT; anchors the model)")
    p.add_argument("--dpo-negative-weight", type=float, default=1.0, help="scale of the DPO term; 0 with --dpo-sft-weight > 0 = fine-tuning on the winners only (RAFT)")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--stop-at-step", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="keep only the first N clips of every pool (smoke tests)")
    p.add_argument("--max-latent-frames", type=int, default=0,
                   help="train on at most the first N latent frames of a clip (the VAE is causal, so that is exactly the shorter clip); 0 = whole clips")
    p.add_argument("--resume-from", default="")
    p.add_argument("--init-from", default="", help="deployed or v4 checkpoint: lora_params.pt + motion_adapter.pt (fresh optimizer)")
    p.add_argument("--reset-input-stem", action="store_true", help="explicit legacy20/27 -> native5 initialization: reuse LoRA/body, freshly initialize the five-channel stem")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--lora-lr", type=float, default=-1.0)
    p.add_argument("--adapter-lr", type=float, default=-1.0)
    p.add_argument("--warmup-steps", type=int, default=-1)
    p.add_argument("--freeze-lora", action="store_true")
    p.add_argument("--gpu-memory-cap-gib", type=float, default=0.0)
    return p.parse_args()


def apply_overrides(args: argparse.Namespace) -> argparse.Namespace:
    path = Path(args.output_dir) / "OVERRIDES.json" if args.output_dir else None
    if path and path.is_file():
        overrides = json.loads(path.read_text())
        for key, value in overrides.items():
            if not hasattr(args, key):
                raise KeyError(f"unknown override {key}")
            if isinstance(value, str):
                value = value.replace("{expert}", args.expert)
            setattr(args, key, value)
        print(f"[overrides] {path}: {overrides}", flush=True)
    return args


def parse_pools(text: str) -> dict[str, tuple[Path, str]]:
    pools = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        name, _, rest = item.partition("=")
        folder, _, ids = rest.partition("@")
        if not name or not folder:
            raise ValueError(f"bad pool {item!r}: expected name=cache_dir[@ids_file]")
        pools[name] = (Path(folder), ids)
    if not pools:
        raise ValueError("--pools is empty")
    return pools


def parse_mix(text: str, pools) -> list[tuple[str, str, float]]:
    entries = []
    for item in filter(None, (part.strip() for part in text.split(","))):
        pool, kind, weight = item.split(":")
        if pool not in pools:
            raise ValueError(f"mix entry {item!r}: unknown pool {pool!r}")
        if kind not in cond_v2.KINDS:
            raise ValueError(f"mix entry {item!r}: kind must be one of {cond_v2.KINDS}")
        if float(weight) <= 0:
            raise ValueError(f"mix entry {item!r}: weight must be positive")
        entries.append((pool, kind, float(weight)))
    if not entries:
        raise ValueError("--mix is empty")
    total = sum(w for _, _, w in entries)
    return [(pool, kind, weight / total) for pool, kind, weight in entries]


class CursorStream(Sampler):
    """The sample cursors of a run segment; everything else is derived from (seed, cursor) by the dataset."""

    def __init__(self, start: int, stop: int) -> None:
        self.start, self.stop = int(start), int(stop)

    def __len__(self) -> int:
        return max(0, self.stop - self.start)

    def __iter__(self):
        return iter(range(self.start, self.stop))


class MixDataset(Dataset):
    """item(cursor): draw (pool, kind) from the mix and a clip of that pool, all from Random(seed:cursor)."""

    def __init__(self, pools: dict, mix: list, seed: int, extras: list[str], limit: int = 0, max_latent_frames: int = 0,
                 dpo_pairs: str = "", dpo_prob: float = 0.0, condition_interface: str = "legacy27", condition_source: str = "cached",
                 winner_only: bool = False) -> None:
        self.mix, self.seed, self.extras, self.max_latent_frames = mix, int(seed), list(extras), int(max_latent_frames)
        self.condition_interface, self.condition_source, self.winner_only = condition_interface, condition_source, winner_only
        self.dpo_prob, self.pairs = float(dpo_prob), []
        if dpo_prob > 0:
            # Pools named raft* hold the conditions the candidates were sampled under; search them first so a
            # replay pool that caches the same clip ID (with an older Stage-1 prediction) is never used instead.
            folders = list(dict.fromkeys(folder for name, (folder, _) in sorted(
                pools.items(), key=lambda item: (not item[0].startswith("raft"), item[0]))))
            for path in sorted(Path(dpo_pairs).glob("*.pt")):
                home = next((folder for folder in folders if (folder / path.name).is_file()), None)
                if home is not None:
                    self.pairs.append((path, home))
            if not self.pairs:
                raise RuntimeError(f"--dpo-prob {dpo_prob} but no pair under {dpo_pairs} belongs to a pool")
            print(f"[data] {len(self.pairs)} preference pairs from {dpo_pairs}", flush=True)
        self.pools = {}
        for name, (folder, ids_file) in pools.items():
            rows = sorted(p.stem for p in folder.glob("*.pt") if not p.name.startswith("summary-"))
            if ids_file:
                wanted = {line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()}
                rows = [r for r in rows if r in wanted]
            if limit:
                rows = rows[:limit]
            if not rows:
                raise RuntimeError(f"pool {name}: no cache files under {folder}")
            probe = torch.load(folder / f"{rows[0]}.pt", map_location="cpu", weights_only=True)
            if probe.get("cache_format_version") != cond_v2.CACHE_FORMAT:
                raise ValueError(f"pool {name}: {folder} is not a {cond_v2.CACHE_FORMAT} cache")
            if self.condition_interface == "legacy27" and list(probe.get("condition_channels_v2", [])) != list(cond_v2.EXTRA_NAMES):
                raise ValueError(f"pool {name}: extra channels {probe.get('condition_channels_v2')} differ from the contract")
            self.pools[name] = (folder, rows)
            print(f"[data] pool {name}: {len(rows)} clips from {folder}", flush=True)
        for pool, kind, _ in mix:
            folder, rows = self.pools[pool]
            probe = torch.load(folder / f"{rows[0]}.pt", map_location="cpu", weights_only=True)
            read_cached_condition(probe, kind, self.condition_interface, self.condition_source)
        self.full = set(extras) == set(cond_v2.EXTRA_NAMES)

    def __len__(self) -> int:
        return sum(len(rows) for _, rows in self.pools.values())

    def draw(self, cursor: int) -> tuple[str, str, str]:
        rng = random.Random(f"{self.seed}:{cursor}")
        value, total = rng.random(), 0.0
        pool, kind = self.mix[-1][0], self.mix[-1][1]
        for name, mode, weight in self.mix:
            total += weight
            if value < total:
                pool, kind = name, mode
                break
        rows = self.pools[pool][1]
        return pool, kind, rows[rng.randrange(len(rows))]

    def pair_item(self, cursor: int, rng: random.Random) -> dict:
        """A preference pair: two clips generated under the same condition of one cached sample (winner in video_latent)."""
        path, home = self.pairs[rng.randrange(len(self.pairs))]
        pair = torch.load(path, map_location="cpu", weights_only=True)
        payload = torch.load(home / path.name, map_location="cpu", weights_only=True)
        condition = read_cached_condition(payload, pair["kind"], self.condition_interface, self.condition_source)
        if self.condition_interface == "legacy27" and not self.full:
            condition = cond_v2.mask_extras(condition, self.extras)
        keep = min(pair["win"].shape[1], condition.shape[1], self.max_latent_frames or 10 ** 6)
        if not self.winner_only:
            keep = min(keep, pair["lose"].shape[1])
        return {"cursor": int(cursor), "pool": "pairs", "kind": "dpo", "sample_id": path.stem,
                "video_latent": pair["win"].float()[:, :keep], "lose_latent": None if self.winner_only else pair["lose"].float()[:, :keep],
                "image_cond_latent": payload["image_cond_latent"].float()[:, :keep], "motion_condition": condition[:, :keep],
                "null_condition": read_cached_condition(payload, "null", self.condition_interface, self.condition_source)[:, :keep], "prompt_embeds": payload["prompt_embeds"].float(), "fm_weight": None,
                "seconds": torch.tensor(payload["frame_seconds"], dtype=torch.float32)[:keep], "masks": None, "disp": None, "valid": None}

    def __getitem__(self, cursor: int) -> dict:
        if self.pairs:
            rng = random.Random(f"{self.seed}:pair:{int(cursor)}")
            if rng.random() < self.dpo_prob:
                return self.pair_item(int(cursor), rng)
        pool, kind, sample_id = self.draw(int(cursor))
        payload = torch.load(self.pools[pool][0] / f"{sample_id}.pt", map_location="cpu", weights_only=True)
        condition = read_cached_condition(payload, kind, self.condition_interface, self.condition_source)
        null = read_cached_condition(payload, "null", self.condition_interface, self.condition_source)
        if self.condition_interface == "legacy27" and not self.full:
            condition = cond_v2.mask_extras(condition, self.extras)
        labels = payload.get("labels_real")
        keep = payload["video_latent"].shape[1]
        if self.max_latent_frames:
            keep = min(keep, self.max_latent_frames)
        weight = payload["fm_weight"].float()[:keep] if "fm_weight" in payload else None
        if weight is not None:
            weight = weight * (weight.numel() / weight.sum())           # mean 1 over the frames that are trained on
        return {"cursor": int(cursor), "pool": pool, "kind": kind, "sample_id": sample_id,
                "video_latent": payload["video_latent"].float()[:, :keep], "image_cond_latent": payload["image_cond_latent"].float()[:, :keep],
                "motion_condition": condition[:, :keep], "null_condition": null[:, :keep], "prompt_embeds": payload["prompt_embeds"].float(),
                "fm_weight": weight,
                "seconds": torch.tensor(payload["frame_seconds"], dtype=torch.float32)[:keep],
                "masks": payload["object_masks_latent"].float() if "object_masks_latent" in payload else None,
                "disp": labels["disp"].float()[:, :keep] if labels is not None else None,
                "valid": labels["valid"].bool()[:, :keep] if labels is not None else None, "lose_latent": None}


def collate_one(batch: list[dict], prompt_pad_length: int) -> dict:
    """Batch of one clip (clips differ in length); tensors get their batch dimension, the prompt is zero-padded."""
    if len(batch) != 1:
        raise ValueError("the v4 trainer runs with per_gpu_batch = 1")
    item = batch[0]
    prompt = torch.zeros(1, prompt_pad_length, item["prompt_embeds"].shape[-1])
    embedding = item["prompt_embeds"][:prompt_pad_length]
    prompt[0, : embedding.shape[0]] = embedding
    out = {key: item[key] for key in ("cursor", "pool", "kind", "sample_id")}
    out["prompt_embeds"] = prompt
    for key in ("video_latent", "image_cond_latent", "motion_condition", "null_condition", "fm_weight", "seconds", "masks", "disp", "valid", "lose_latent"):
        out[key] = None if item[key] is None else item[key][None]
    return out


def load_adapter_weights(adapter, checkpoint: Path, *, reset_input_stem: bool = False) -> None:
    state = torch.load(checkpoint / "motion_adapter.pt", map_location="cpu", weights_only=True)
    report = load_adapter_state(adapter, state, reset_input_stem=reset_input_stem)
    if reset_input_stem:
        print(f"[adapter transfer] {json.dumps(report)}", flush=True)


def main() -> None:
    args = apply_overrides(parse_args())
    config = yaml.safe_load(Path(args.config).read_text())
    paths, data_cfg, model_cfg, train_cfg = config["paths"], config["data"], config["model"], config["train"]
    condition_interface = interface_from_config(model_cfg)
    condition_source = str(model_cfg.get("condition_source", "cached"))
    if args.reset_input_stem and (condition_interface != "native5" or not args.init_from or args.resume_from):
        raise SystemExit("--reset-input-stem requires native5, --init-from, and no --resume-from")
    if int(train_cfg["per_gpu_batch"]) != 1:
        raise SystemExit("the v4 trainer needs train.per_gpu_batch = 1 (clips differ in length)")
    max_steps = int(args.max_steps or train_cfg["max_steps"])
    grad_accum = int(train_cfg["grad_accum_steps"])
    save_every = int(args.save_every or train_cfg["save_every"])
    log_every = int(args.log_every or train_cfg["log_every"])
    target_step = min(max_steps, int(args.stop_at_step)) if args.stop_at_step else max_steps
    seed = int(config["seed"] if args.seed < 0 else args.seed)
    pools = parse_pools(args.pools)
    mix = parse_mix(args.mix, pools)
    extras = [name for name in args.extra_channels.split(",") if name]
    cond_v2.extra_channel_index(extras)
    if args.keep_weight > 0 and not (args.reference_init or args.init_from):
        raise SystemExit("--keep-weight needs --reference-init (or --init-from, which is then the reference)")
    if args.kin_weight > 0 and not args.phi_lat:
        raise SystemExit("--kin-weight needs --phi-lat")
    winner_only = args.dpo_prob > 0 and args.dpo_negative_weight == 0
    if winner_only and args.dpo_sft_weight <= 0:
        raise SystemExit("winner-only training requires --dpo-sft-weight > 0")
    if args.dpo_prob > 0 and (not args.dpo_pairs or (not winner_only and not (args.reference_init or args.init_from))):
        raise SystemExit("--dpo-prob needs --dpo-pairs; DPO with a negative term also needs a reference")
    needs_reference = args.keep_weight > 0 or (args.dpo_prob > 0 and not winner_only)
    reference_init = args.reference_init or args.init_from

    from accelerate import Accelerator
    from accelerate.utils import set_seed

    accelerator = Accelerator(gradient_accumulation_steps=grad_accum, mixed_precision="bf16")
    if accelerator.num_processes != 1:
        raise SystemExit("the v4 trainer is single-process (one expert per GPU)")
    set_seed(seed)
    device = accelerator.device
    if args.gpu_memory_cap_gib > 0 and device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(index).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_memory_cap_gib / total), index)
    output_dir = Path(args.output_dir or paths["output_dir"]) / args.expert
    output_dir.mkdir(parents=True, exist_ok=True)
    regime = {"training_regime": "s2_v4_real_video", "sampler": SAMPLER_VERSION, "seed": seed,
              "pools": {name: [folder.name, Path(ids).name if ids else ""] for name, (folder, ids) in pools.items()},      # site-independent
              "max_latent_frames": int(args.max_latent_frames),
              "condition_interface": condition_interface, "condition_source": condition_source,
              "reset_input_stem": bool(args.reset_input_stem),
              "mix": [[pool, kind, round(weight, 6)] for pool, kind, weight in mix], "extra_channels": extras,
              "fm_weight_mode": args.fm_weight_mode, "keep_weight": args.keep_weight,
              "reference_init": reference_init if needs_reference else None,
              "dpo_prob": args.dpo_prob, "dpo_beta": args.dpo_beta if args.dpo_prob > 0 else None,
              "dpo_sft_weight": args.dpo_sft_weight if args.dpo_prob > 0 else None, "dpo_negative_weight": args.dpo_negative_weight if args.dpo_prob > 0 else None, "dpo_pairs": Path(args.dpo_pairs).name if args.dpo_prob > 0 else None,
              "kin_weight": args.kin_weight, "kin_alpha_disp": args.kin_alpha_disp, "kin_sigma": [args.kin_sigma_min, args.kin_sigma_max],
              "phi_lat": args.phi_lat or None, "init_from": args.init_from or None, "grad_accum_steps": grad_accum,
              "freeze_lora": bool(args.freeze_lora), "lora_lr": None if args.lora_lr < 0 else args.lora_lr,
              "adapter_lr": None if args.adapter_lr < 0 else args.adapter_lr,
              "warmup_steps": None if args.warmup_steps < 0 else int(args.warmup_steps), "limit": int(args.limit)}
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump({**config, "selected_expert": args.expert, "resolved_max_steps": max_steps,
                                                                    "regime": regime}, sort_keys=False))
    dataset = MixDataset(pools, mix, seed, extras, limit=args.limit, max_latent_frames=args.max_latent_frames,
                         dpo_pairs=args.dpo_pairs, dpo_prob=args.dpo_prob, condition_interface=condition_interface,
                         condition_source=condition_source, winner_only=winner_only)

    wan_model = os.environ.get("WAN_MODEL_DIR", "") or str(paths.get("wan_model") or "")
    if not wan_model:
        raise SystemExit("no Wan2.2-I2V-A14B-Diffusers directory: set WAN_MODEL_DIR or paths.wan_model in the config")
    mean, std = load_latent_statistics(wan_model)
    backend = OracleFlowWanBackend(
        model_dir=wan_model, expert=args.expert, injection_blocks=list(model_cfg["injection_blocks"]),
        lora_rank=int(model_cfg["lora_rank"]), lora_alpha=int(model_cfg["lora_alpha"]), latents_mean=mean, latents_std=std,
        device=device, flow_shift=float(train_cfg["flow_shift"]), boundary_sigma=float(model_cfg["boundary_sigma"]),
        gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]))
    transformer_dim = backend.transformer.config.num_attention_heads * backend.transformer.config.attention_head_dim
    adapter = build_adapter(model_cfg, transformer_dim).to(device)
    lora_parameters = backend.trainable_parameters()
    adapter_parameters = [p for p in adapter.parameters() if p.requires_grad]
    if args.freeze_lora:
        for parameter in lora_parameters:
            parameter.requires_grad_(False)
    lora_lr = float(train_cfg["lora_learning_rate"]) if args.lora_lr < 0 else args.lora_lr
    adapter_lr = float(train_cfg["adapter_learning_rate"]) if args.adapter_lr < 0 else args.adapter_lr
    optimizer = torch.optim.AdamW([{"params": lora_parameters, "lr": 0.0 if args.freeze_lora else lora_lr},
                                   {"params": adapter_parameters, "lr": adapter_lr}],
                                  weight_decay=float(train_cfg["weight_decay"]), betas=(0.9, 0.999))
    warmup_steps = int(train_cfg["warmup_steps"]) if args.warmup_steps < 0 else int(args.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    if args.resume_from:
        checkpoint = Path(args.resume_from)
        backend.load_lora_parameters(checkpoint / "lora_params.pt")
        load_adapter_weights(adapter, checkpoint)
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        for key, default in (("condition_interface", "legacy27"), ("condition_source", "cached")):
            if state.get(key, default) != regime[key]:
                raise ValueError(f"resume would change {key}: {state.get(key, default)!r} != {regime[key]!r}")
        if int(state.get("max_steps", max_steps)) != max_steps:
            raise ValueError(f"resume would change the LR schedule: {state.get('max_steps')} != {max_steps}")
        tail = lambda value: "/".join(Path(value).parts[-2:]) if value else value        # noqa: E731  the same checkpoint under another root directory
        if tail(state.get("reference_init")) != tail(regime["reference_init"]):
            raise ValueError(f"resume would change reference_init: checkpoint {state.get('reference_init')!r} != {regime['reference_init']!r}")
        for key in ("seed", "mix", "pools", "extra_channels", "fm_weight_mode", "keep_weight", "kin_weight",
                    "kin_alpha_disp", "kin_sigma", "grad_accum_steps", "freeze_lora", "lora_lr", "adapter_lr", "warmup_steps", "limit",
                    "max_latent_frames", "dpo_prob", "dpo_beta", "dpo_pairs", "dpo_sft_weight", "dpo_negative_weight"):
            if state.get(key) != regime[key]:
                raise ValueError(f"resume would change {key}: checkpoint {state.get(key)!r} != {regime[key]!r}")
        start_step = int(state["step"])
        if not args.init_from and state.get("init_from"):
            regime["init_from"] = state["init_from"]
    elif args.init_from:
        backend.load_lora_parameters(Path(args.init_from) / "lora_params.pt")
        load_adapter_weights(adapter, Path(args.init_from), reset_input_stem=args.reset_input_stem)
        print(f"[init] weights from {args.init_from} (fresh optimizer, step 0)", flush=True)

    reference_adapter = None
    if needs_reference:
        # the frozen reference policy: a second LoRA on the same transformer + its own adapter copy
        backend.add_reference(Path(reference_init) / "lora_params.pt", lora_rank=int(model_cfg["lora_rank"]), lora_alpha=int(model_cfg["lora_alpha"]))
        if args.freeze_lora:
            for parameter in lora_parameters:
                parameter.requires_grad_(False)
        reference_adapter = build_adapter(model_cfg, transformer_dim)
        load_adapter_weights(reference_adapter, Path(reference_init))
        reference_adapter = accelerator.prepare_model(reference_adapter.to(device).eval().requires_grad_(False), evaluation_mode=True)
        print(f"[reference] {reference_init}: {backend.reference_parameters} LoRA tensors, active adapters {backend.active_adapters()}", flush=True)
    readout = None
    if args.kin_weight > 0:
        from motion_jepa.phi_lat import load_readout
        readout = load_readout(args.phi_lat, mean, std).to(device).eval().requires_grad_(False)

    sampler = CursorStream(start_step * grad_accum, target_step * grad_accum)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=int(train_cfg["num_workers"]), pin_memory=True,
                        collate_fn=lambda batch: collate_one(batch, int(data_cfg["prompt_pad_length"])))
    train_module, adapter, optimizer, loader = accelerator.prepare(backend.transformer, adapter, optimizer, loader)
    backend.transformer = train_module
    if args.resume_from:
        if (Path(args.resume_from) / "optimizer.pt").is_file():
            optimizer.load_state_dict(torch.load(Path(args.resume_from) / "optimizer.pt", map_location="cpu", weights_only=False))
            scheduler.load_state_dict(torch.load(Path(args.resume_from) / "scheduler.pt", map_location="cpu", weights_only=False))
        else:
            # a checkpoint copied without its optimizer state: the schedule continues at its step,
            # AdamW's moment estimates start again (they settle within tens of steps)
            scheduler.last_epoch = start_step
            for group, base in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base * lr_lambda(start_step)
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            print(f"[resume] {args.resume_from} has no optimizer state: weights + schedule at step {start_step}, fresh AdamW moments", flush=True)
    trainable = ([] if args.freeze_lora else lora_parameters) + adapter_parameters
    print(f"[model] expert={args.expert} trainable={sum(p.numel() for p in trainable) / 1e6:.2f}M clips={len(dataset)} "
          f"effective_batch={grad_accum} steps={start_step}->{target_step} of {max_steps} regime={json.dumps(regime)}", flush=True)

    def save_checkpoint(step: int) -> None:
        checkpoint = output_dir / f"checkpoint-{step:06d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        backend.save_lora(checkpoint, transformer=accelerator.unwrap_model(backend.transformer))
        torch.save(accelerator.unwrap_model(adapter).state_dict(), checkpoint / "motion_adapter.pt")
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
        (checkpoint / "trainer_state.json").write_text(json.dumps({"step": step, "expert": args.expert, "max_steps": max_steps,
                                                                   "condition_contract": cond_v2.CACHE_FORMAT, **regime}, indent=2, sort_keys=True) + "\n")
        print(f"[checkpoint] {checkpoint}", flush=True)

    metrics = (output_dir / "metrics.jsonl").open("a")
    trace = (output_dir / "steps.jsonl").open("a")
    step, started = start_step, time.time()
    kinds: Counter = Counter()
    sums: Counter = Counter()
    counts: Counter = Counter()
    frames_seen = 0
    core = getattr(adapter, "module", adapter)
    for raw in loader:
        micro = int(raw["cursor"])
        kinds[f"{raw['pool']}:{raw['kind']}"] += 1
        frames_seen += int(raw["video_latent"].shape[2])
        torch.manual_seed((seed * 1_000_003 + micro) * 31)                # sigma and noise are tied to the micro-step
        if raw["kind"] == "dpo" and not winner_only:
            # a preference pair: both clips at the same sigma and noise; the scalar weight first, then one backward per clip
            with accelerator.accumulate(train_module, adapter):
                sigmas, noise = backend.pair_noise(raw["video_latent"])
                shared = dict(sigmas=sigmas, noise=noise)
                clips = (raw["video_latent"], raw["lose_latent"])
                with torch.no_grad():
                    frozen = reference_adapter(raw["motion_condition"])
                    frozen_mix = getattr(reference_adapter, "module", reference_adapter).mix
                    r_win, r_lose = (backend.pair_error(clip, raw["image_cond_latent"], raw["prompt_embeds"], frozen, mixer=frozen_mix, reference=True, **shared)
                                     for clip in clips)
                    current = adapter(raw["motion_condition"])
                    e_win, e_lose = (backend.pair_error(clip, raw["image_cond_latent"], raw["prompt_embeds"], current, mixer=core.mix, **shared) for clip in clips)
                    terms = dpo_terms(e_win, r_win, e_lose, r_lose, args.dpo_beta)
                coefficients = ((clips[0], args.dpo_negative_weight * terms["weight"] + args.dpo_sft_weight), (clips[1], -args.dpo_negative_weight * terms["weight"]))
                for clip, coefficient in coefficients:
                    if not torch.is_tensor(coefficient) and coefficient == 0:
                        continue
                    if torch.is_tensor(coefficient) and not bool(coefficient.ne(0).any()):
                        continue
                    error = backend.pair_error(clip, raw["image_cond_latent"], raw["prompt_embeds"], adapter(raw["motion_condition"]), mixer=core.mix, **shared)
                    accelerator.backward((coefficient * error).mean())
                loss = terms["loss"].mean()
                backend.last_terms = {"dpo": float(loss), "dpo_d": float(terms["d"].mean()), "dpo_prefers_winner": float((terms["d"] < 0).float().mean()),
                                      "dpo_error_win": float(e_win.mean()), "dpo_error_lose": float(e_lose.mean())}
                if accelerator.sync_gradients:
                    gradient_norm = checked_clip_grad(accelerator, trainable, float(train_cfg["max_grad_norm"]))
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        else:
          with accelerator.accumulate(train_module, adapter):
              keep = None
              if reference_adapter is not None and args.keep_weight > 0 and raw["kind"] == "null":
                  with torch.no_grad():
                      reference_residuals = reference_adapter(raw["null_condition"])
                  keep = {"weight": args.keep_weight, "mask": torch.ones(1, dtype=torch.bool), "residuals": reference_residuals,
                          "mixer": getattr(reference_adapter, "module", reference_adapter).mix}
              kin = None
              if readout is not None and raw["disp"] is not None and raw["disp"].shape[1] > 0:
                  kin = {"weight": args.kin_weight, "alpha_disp": args.kin_alpha_disp, "sigma_min": args.kin_sigma_min, "sigma_max": args.kin_sigma_max,
                         "readout": readout, "disp": raw["disp"], "valid": raw["valid"], "masks": raw["masks"], "seconds": raw["seconds"]}
              residuals = adapter(raw["motion_condition"])
              batch = Stage2Batch(video_latent=raw["video_latent"], image_cond_latent=raw["image_cond_latent"], prompt_embeds=raw["prompt_embeds"],
                                  motion_condition=raw["motion_condition"],
                                  fm_weight=raw["fm_weight"] if args.fm_weight_mode == "tube" else None)
              out = backend.train_step_v2(batch, residuals, mixer=core.mix, keep=keep, kin=kin)
              loss, sigmas = out["loss"], out["sigmas"]
              if raw["kind"] == "dpo":
                  # Winner-RFT is ordinary FM on the accepted video. No loser
                  # or reference forward, DPO logit, or preference gradient.
                  backend.last_terms["winner_fm"] = float(loss.detach())
                  loss = loss * args.dpo_sft_weight
              accelerator.backward(loss)
              if accelerator.sync_gradients:
                  gradient_norm = checked_clip_grad(accelerator, trainable, float(train_cfg["max_grad_norm"]))
              optimizer.step()
              if accelerator.sync_gradients:
                  scheduler.step()
              optimizer.zero_grad(set_to_none=True)
        terms = dict(backend.last_terms)
        terms["total"] = float(loss.detach())
        for key, value in terms.items():
            sums[key] += value
            counts[key] += 1
        trace.write(json.dumps({"micro": micro, "pool": raw["pool"], "kind": raw["kind"], "sample_id": raw["sample_id"],
                                "frames": int(raw["video_latent"].shape[2]), "sigma": round(float(sigmas[0]), 6),
                                "loss": float(loss.detach()), **{k: round(v, 6) for k, v in terms.items()}}) + "\n")
        trace.flush()
        if not accelerator.sync_gradients:
            continue
        step += 1
        if step == start_step + 1 or step % log_every == 0:
            total = sum(kinds.values()) or 1
            record = {"step": step, "loss/last_microbatch": float(loss.detach()),
                      **{f"loss/{key}": sums[key] / counts[key] for key in sorted(sums)},
                      "grad_norm": float(gradient_norm), "lr/lora": scheduler.get_last_lr()[0], "lr/adapter": scheduler.get_last_lr()[1],
                      **{f"mix/{key}": value / total for key, value in sorted(kinds.items())},
                      "latent_frames/mean": frames_seen / total, "elapsed_seconds": time.time() - started}
            if device.type == "cuda":
                record["cuda/max_memory_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
            sums.clear()
            counts.clear()
        if step % save_every == 0 or step >= target_step:
            save_checkpoint(step)
        if step >= target_step:
            break
    marker = "done.json" if step >= max_steps else "paused.json"
    (output_dir / marker).write_text(json.dumps({"expert": args.expert, "steps": step, "max_steps": max_steps, **regime}, indent=2, sort_keys=True) + "\n")
    print(f"[{marker[:-5]}] step {step} of {max_steps}", flush=True)
    metrics.close()
    trace.close()


if __name__ == "__main__":
    main()
