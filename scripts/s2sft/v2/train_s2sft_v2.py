#!/usr/bin/env python
"""Stage-2 GT-condition SFT from the base model, multi-GPU (DDP), resumable -- one training run per expert.

Evaluation length: 41 frames @ 7 fps = 11 latent frames.
  * validation is also reported per latent frame: "late" = latents 8-10, split by what the latent shows -- "real"
    (all its frames are real footage), "held" (all its frames are the held last frame of a scene at rest:
    clip_window padding) or "mixed" (the junction latent); "early" = latents 1-7; "per_latent" = 0-10
  * training metrics loss/early_plain (latents 1-7), loss/late_plain (latents 8-10) and the share of late latents

A reduced form of scripts/train_s2_v4.py with what the single-run GT SFT needs:
  * condition contract v2, interface legacy27 (20 rendered + 7 extra channels), condition kind "real" (measured or
    simulator ground-truth plan) from cond_v2 caches; the target is the real continuation
  * flow matching, plain or weighted by the clip's moving-object tubes (--fm-weight-mode tube)
  * fresh LoRA (peft init: B = 0) and a fresh adapter when --init-from is empty: the run starts at the base model
  * accelerate DDP over N GPUs, per-GPU batch 1 (clips differ in length), global batch = N x grad_accum
  * sampling plan "epoch_mix_v1": the pool of global sample g is drawn from Random(seed:g) with the mix weights, the clip
    is the next one of that pool's own shuffled epoch (without replacement; reshuffled every epoch with the seed), so
    every clip of a pool is seen once before any is seen twice.  Global sample g of optimizer step s, micro-step m,
    rank r is g = (s * grad_accum + m) * N + r; the plan is a pure function of (seed, pools, mix), identical after a
    resume.  The plan's unique-clip coverage is logged and written to <out>/sampling_plan_summary.json, every rank logs
    the clips it actually trained on to steps_rank<r>.jsonl
  * SIGUSR1 / SIGTERM / --deadline / --stop-file: all ranks agree (all-reduce) to checkpoint at the next optimizer
    step and leave
  * --global-batch fixes N x grad_accum (grad_accum = global_batch / N), so a run may continue on another GPU count
    with the identical sample sequence
  * held-out validation FM loss (--val-pools, kind real, the first --max-latent-frames frames): every clip at
    --val-sigmas fixed (sigma, noise) draws from its own seeded generator (identical at every evaluation), plain and
    tube-weighted, per pool; at step 0 of a fresh run (the base model) and at every kept checkpoint -> val.jsonl
  * checkpoints: every --save-every steps (resume points; deleted once two newer exist) and every --keep-every steps
    (kept complete, never deleted)
Checkpoints: lora_params.pt + motion_adapter.pt (deployed layout) + optimizer.pt + scheduler.pt + trainer_state.json.

Overrides: a JSON file (--overrides, default <output-dir>/OVERRIDES.json when it exists) may set any argument; keys
starting with "_" are comments.  In its string values "{expert}" becomes the expert, "{data_root}" the --data-root
directory and "{repo_root}" the repository root.  A flag given on the command line (different from its default)
wins over the file.  The released recipe is configs/stage2/sft_overrides.json.
Base model: --wan-model-dir, else $WAN_MODEL_DIR, else the config's paths.wan_model.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_jepa import cond_v2  # noqa: E402
from motion_jepa.stage2 import OracleFlowWanBackend, Stage2Batch, load_latent_statistics  # noqa: E402
from motion_jepa.stage2_condition import interface_from_config, load_adapter_state, read_cached_condition  # noqa: E402
from scripts.infer_stage2_oracle import build_adapter  # noqa: E402

SAMPLER_VERSION = "epoch_mix_v1"
STOP = {"flag": False}


def parse_args() -> argparse.Namespace:
    p = build_parser()
    args = p.parse_args()
    args._defaults = {action.dest: action.default for action in p._actions}
    return args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--expert", required=True, choices=("low", "high"))
    p.add_argument("--output-dir", default="", help="run directory; checkpoints go to <output-dir>/<expert> (default: config paths.output_dir)")
    p.add_argument("--overrides", default="", help="JSON file of argument values (default: <output-dir>/OVERRIDES.json if it exists)")
    p.add_argument("--data-root", default=os.environ.get("MOPLAN_SFT_DATA", ""),
                   help="substituted for {data_root} in the overrides (default: $MOPLAN_SFT_DATA)")
    p.add_argument("--wan-model-dir", default="", help="Wan2.2-I2V-A14B-Diffusers directory (default: $WAN_MODEL_DIR, else config paths.wan_model)")
    p.add_argument("--pools", default="", help="name=cache_dir[@ids_file],... (ids_file: one sample id per line)")
    p.add_argument("--mix", default="", help="pool:kind:weight,... (GT SFT: kind real)")
    p.add_argument("--fm-weight-mode", default="none", choices=("none", "tube"))
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--stop-at-step", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="keep only the first N clips of every pool (smoke tests)")
    p.add_argument("--max-latent-frames", type=int, default=0)
    p.add_argument("--resume-from", default="", help="checkpoint dir, or 'latest' = newest checkpoint-* of the output dir")
    p.add_argument("--init-from", default="", help="empty = fresh LoRA + fresh adapter (the base model)")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--lora-lr", type=float, default=-1.0)
    p.add_argument("--adapter-lr", type=float, default=-1.0)
    p.add_argument("--warmup-steps", type=int, default=-1)
    p.add_argument("--grad-accum", type=int, default=0, help="0 = config train.grad_accum_steps (ignored with --global-batch)")
    p.add_argument("--global-batch", type=int, default=0, help="N x grad_accum; grad_accum = global_batch / N")
    p.add_argument("--val-pools", default="", help="name=cache_dir[@ids_file],... held-out clips for the validation FM loss")
    p.add_argument("--val-sigmas", type=int, default=2, help="fixed (sigma, noise) draws per validation clip")
    p.add_argument("--val-every", type=int, default=0, help="0 = at every kept checkpoint (--keep-every)")
    p.add_argument("--find-unused-parameters", action="store_true")
    p.add_argument("--keep-every", type=int, default=500, help="checkpoints kept complete for selection (never deleted); every "
                   "other checkpoint is a resume point, deleted once 2 newer ones exist")
    p.add_argument("--deadline", type=float, default=0.0, help="unix time: checkpoint at the first optimizer step after it and leave (time-limited jobs)")
    p.add_argument("--stop-file", default="", help="if this file exists: checkpoint at the next optimizer step and leave")
    p.add_argument("--save-init", action="store_true", help="write checkpoint-<start step> before any update and exit (the fresh "
                   "LoRA (B = 0) + adapter of this run's seed = the untrained Wan2.2; the start of RAFT without SFT)")
    return p


def substitute(value: str, args: argparse.Namespace) -> str:
    """Placeholders of an override string: {expert}, {data_root} (--data-root) and {repo_root}."""
    value = value.replace("{expert}", args.expert)
    if "{data_root}" in value:
        if not args.data_root:
            raise SystemExit("the overrides use {data_root}: pass --data-root (or set MOPLAN_SFT_DATA)")
        value = value.replace("{data_root}", str(Path(args.data_root).expanduser().resolve()))
    return value.replace("{repo_root}", str(REPO_ROOT))


def apply_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.overrides:
        path = Path(args.overrides)
        if not path.is_file():
            raise SystemExit(f"--overrides {path}: no such file")
    else:
        path = Path(args.output_dir) / "OVERRIDES.json" if args.output_dir else None
    if path and path.is_file():
        print(f"[overrides] {path}", flush=True)
        overrides = json.loads(path.read_text())
        for key, value in overrides.items():
            if key.startswith("_"):
                continue
            if not hasattr(args, key) or key in ("overrides", "data_root"):
                raise KeyError(f"unknown override {key}")
            if isinstance(value, str):
                value = substitute(value, args)
            if getattr(args, key) != args._defaults.get(key):        # a flag given on the command line wins (smoke tests)
                print(f"[overrides] {key}: command line {getattr(args, key)!r} kept over OVERRIDES {value!r}", flush=True)
                continue
            setattr(args, key, value)
    for key in ("pools", "val_pools", "init_from"):                    # placeholders given on the command line
        value = getattr(args, key)
        if isinstance(value, str) and "{" in value:
            setattr(args, key, substitute(value, args))
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
    total = sum(w for _, _, w in entries)
    return [(pool, kind, weight / total) for pool, kind, weight in entries]


class EpochMixPlan:
    """(pool, kind, clip) of every global sample index: pool from the mix, clip = next of the pool's shuffled epoch."""

    def __init__(self, pools_rows: dict[str, list[str]], mix, seed: int, total: int) -> None:
        self.mix, self.seed = mix, int(seed)
        used: Counter = Counter()
        orders: dict[tuple[str, int], list[str]] = {}
        self.items: list[tuple[str, str, str]] = []
        for g in range(total):
            rng = random.Random(f"{self.seed}:{g}")
            value, acc = rng.random(), 0.0
            pool, kind = mix[-1][0], mix[-1][1]
            for name, mode, weight in mix:
                acc += weight
                if value < acc:
                    pool, kind = name, mode
                    break
            rows = pools_rows[pool]
            k = used[pool]
            used[pool] += 1
            epoch, index = divmod(k, len(rows))
            if (pool, epoch) not in orders:
                order = list(rows)
                random.Random(f"{self.seed}:{pool}:epoch{epoch}").shuffle(order)
                orders[(pool, epoch)] = order
            self.items.append((pool, kind, orders[(pool, epoch)][index]))

    def summary(self, upto: int, pools_rows) -> dict:
        seen: dict[str, set] = {}
        draws: Counter = Counter()
        for pool, _, sid in self.items[:upto]:
            seen.setdefault(pool, set()).add(sid)
            draws[pool] += 1
        unique = set().union(*seen.values()) if seen else set()
        return {"samples": upto, "unique_clips": len(unique),
                "per_pool": {p: {"clips": len(pools_rows[p]), "draws": draws[p], "unique": len(seen.get(p, ())),
                                 "passes": round(draws[p] / max(1, len(pools_rows[p])), 3)} for p in pools_rows}}


class RankStream(Sampler):
    """Global sample indices of this rank for optimizer steps [start, stop)."""

    def __init__(self, start_step: int, stop_step: int, accum: int, world: int, rank: int) -> None:
        self.indices = [(s * accum + m) * world + rank for s in range(start_step, stop_step) for m in range(accum)]

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


class PlanDataset(Dataset):
    def __init__(self, pools: dict, plan: EpochMixPlan, max_latent_frames: int, interface: str) -> None:
        self.pools, self.plan, self.max_latent_frames, self.interface = pools, plan, int(max_latent_frames), interface

    def __len__(self) -> int:
        return len(self.plan.items)

    def __getitem__(self, g: int) -> dict:
        pool, kind, sample_id = self.plan.items[int(g)]
        payload = torch.load(self.pools[pool] / f"{sample_id}.pt", map_location="cpu", weights_only=True)
        condition = read_cached_condition(payload, kind, self.interface, "cached")
        keep = payload["video_latent"].shape[1]
        if self.max_latent_frames:
            keep = min(keep, self.max_latent_frames)
        weight = payload["fm_weight"].float()[:keep] if "fm_weight" in payload else None
        if weight is not None:
            weight = weight * (weight.numel() / weight.sum())
        return {"cursor": int(g), "pool": pool, "kind": kind, "sample_id": sample_id,
                "video_latent": payload["video_latent"].float()[:, :keep], "image_cond_latent": payload["image_cond_latent"].float()[:, :keep],
                "motion_condition": condition[:, :keep], "prompt_embeds": payload["prompt_embeds"].float(), "fm_weight": weight}


def collate_one(batch: list[dict], prompt_pad_length: int) -> dict:
    if len(batch) != 1:
        raise ValueError("per_gpu_batch must be 1")
    item = batch[0]
    prompt = torch.zeros(1, prompt_pad_length, item["prompt_embeds"].shape[-1])
    embedding = item["prompt_embeds"][:prompt_pad_length]
    prompt[0, : embedding.shape[0]] = embedding
    out = {key: item[key] for key in ("cursor", "pool", "kind", "sample_id")}
    out["prompt_embeds"] = prompt
    for key in ("video_latent", "image_cond_latent", "motion_condition", "fm_weight"):
        out[key] = None if item[key] is None else item[key][None]
    return out


def latest_checkpoint(folder: Path) -> Path | None:
    found = sorted(p for p in folder.glob("checkpoint-[0-9]*") if (p / "trainer_state.json").is_file())
    return found[-1] if found else None


def main() -> None:
    args = apply_overrides(parse_args())
    config = yaml.safe_load(Path(args.config).read_text())
    paths, data_cfg, model_cfg, train_cfg = config["paths"], config["data"], config["model"], config["train"]
    interface = interface_from_config(model_cfg)
    if interface != "legacy27":
        raise SystemExit(f"this trainer is for the legacy27 interface, config gives {interface}")
    if int(train_cfg["per_gpu_batch"]) != 1:
        raise SystemExit("train.per_gpu_batch must be 1")
    wan_model = args.wan_model_dir or os.environ.get("WAN_MODEL_DIR", "") or (paths.get("wan_model") or "")
    if not wan_model:
        raise SystemExit("no base model: pass --wan-model-dir or set WAN_MODEL_DIR (Wan2.2-I2V-A14B-Diffusers directory)")
    paths["wan_model"] = str(wan_model)
    max_steps = int(args.max_steps or train_cfg["max_steps"])
    grad_accum = int(args.grad_accum or train_cfg["grad_accum_steps"])
    save_every = int(args.save_every or train_cfg["save_every"])
    log_every = int(args.log_every or train_cfg["log_every"])
    target_step = min(max_steps, int(args.stop_at_step)) if args.stop_at_step else max_steps
    seed = int(config["seed"] if args.seed < 0 else args.seed)
    pools = parse_pools(args.pools)
    mix = parse_mix(args.mix, pools)

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed

    ddp = DistributedDataParallelKwargs(find_unused_parameters=bool(args.find_unused_parameters))
    world_env = int(os.environ.get("WORLD_SIZE", "1"))
    if args.global_batch:
        if args.global_batch % world_env:
            raise SystemExit(f"--global-batch {args.global_batch} is not a multiple of the {world_env} processes")
        grad_accum = args.global_batch // world_env
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum, mixed_precision="bf16", kwargs_handlers=[ddp])
    world, rank = accelerator.num_processes, accelerator.process_index
    if world != world_env:
        raise SystemExit(f"WORLD_SIZE {world_env} != accelerator processes {world}")
    set_seed(seed + rank)
    device = accelerator.device
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.__setitem__("flag", True))
    output_dir = Path(args.output_dir or paths["output_dir"]) / args.expert
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = args.resume_from
    if resume == "latest":
        found = latest_checkpoint(output_dir)
        resume = str(found) if found else ""

    pools_rows: dict[str, list[str]] = {}
    for name, (folder, ids_file) in pools.items():
        rows = sorted(p.stem for p in folder.glob("*.pt") if not p.name.startswith("summary-"))
        if ids_file:
            wanted = {line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()}
            rows = [r for r in rows if r in wanted]
        if args.limit:
            rows = rows[: args.limit]
        if not rows:
            raise RuntimeError(f"pool {name}: no cache files under {folder}")
        probe = torch.load(folder / f"{rows[0]}.pt", map_location="cpu", weights_only=True)
        if probe.get("cache_format_version") != cond_v2.CACHE_FORMAT:
            raise ValueError(f"pool {name}: {folder} is not a {cond_v2.CACHE_FORMAT} cache")
        if list(probe.get("condition_channels_v2", [])) != list(cond_v2.EXTRA_NAMES):
            raise ValueError(f"pool {name}: extra channels {probe.get('condition_channels_v2')} differ from the contract")
        pools_rows[name] = rows
    for pool, kind, _ in mix:
        probe = torch.load(pools[pool][0] / f"{pools_rows[pool][0]}.pt", map_location="cpu", weights_only=True)
        read_cached_condition(probe, kind, interface, "cached")
    total_samples = max_steps * grad_accum * world
    plan = EpochMixPlan(pools_rows, mix, seed, total_samples)
    ids_digest = {name: [len(rows), rows[0], rows[-1]] for name, rows in pools_rows.items()}
    regime = {"training_regime": "s2sft_gt_single_run_v2_len41", "sampler": SAMPLER_VERSION, "seed": seed,
              "pools": {name: [str(folder), Path(ids).name if ids else ""] for name, (folder, ids) in pools.items()},
              "pool_sizes": {name: len(rows) for name, rows in pools_rows.items()}, "ids_digest": ids_digest,
              "max_latent_frames": int(args.max_latent_frames), "condition_interface": interface, "condition_source": "cached",
              "mix": [[pool, kind, round(weight, 6)] for pool, kind, weight in mix], "fm_weight_mode": args.fm_weight_mode,
              "init_from": args.init_from or None, "grad_accum_steps": grad_accum, "world_size": world,
              "global_batch": grad_accum * world, "lora_lr": None if args.lora_lr < 0 else args.lora_lr,
              "adapter_lr": None if args.adapter_lr < 0 else args.adapter_lr,
              "warmup_steps": None if args.warmup_steps < 0 else int(args.warmup_steps), "limit": int(args.limit)}
    if accelerator.is_main_process:
        (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump({**config, "selected_expert": args.expert, "resolved_max_steps": max_steps,
                                                                        "regime": regime}, sort_keys=False))
        full = plan.summary(total_samples, pools_rows)
        (output_dir / "sampling_plan_summary.json").write_text(json.dumps({"total": full, "per_1000_steps": {
            str(s): plan.summary(min(total_samples, s * grad_accum * world), pools_rows)["unique_clips"] for s in range(500, max_steps + 1, 500)}}, indent=1) + "\n")
        print(f"[plan] {json.dumps(full)}", flush=True)
    dataset = PlanDataset({name: folder for name, (folder, _) in pools.items()}, plan, args.max_latent_frames, interface)
    val_items = []
    if args.val_pools:
        for name, (folder, ids_file) in parse_pools(args.val_pools).items():
            rows = sorted(p.stem for p in folder.glob("*.pt"))
            if ids_file:
                wanted = {line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()}
                rows = [r for r in rows if r in wanted]
            overlap = set(rows) & set().union(*map(set, pools_rows.values()))
            if overlap:
                raise SystemExit(f"validation pool {name} overlaps the training pools: {sorted(overlap)[:3]}")
            val_items += [(name, folder / f"{sid}.pt", sid, k) for sid in rows for k in range(args.val_sigmas)]
        if accelerator.is_main_process:
            print(f"[val] {len(val_items)} (clip, draw) items: " + json.dumps(dict(Counter(n for n, *_ in val_items))), flush=True)

    mean, std = load_latent_statistics(paths["wan_model"])
    backend = OracleFlowWanBackend(
        model_dir=paths["wan_model"], expert=args.expert, injection_blocks=list(model_cfg["injection_blocks"]),
        lora_rank=int(model_cfg["lora_rank"]), lora_alpha=int(model_cfg["lora_alpha"]), latents_mean=mean, latents_std=std,
        device=device, flow_shift=float(train_cfg["flow_shift"]), boundary_sigma=float(model_cfg["boundary_sigma"]),
        gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]))
    transformer_dim = backend.transformer.config.num_attention_heads * backend.transformer.config.attention_head_dim
    adapter = build_adapter(model_cfg, transformer_dim).to(device)
    lora_parameters = backend.trainable_parameters()
    adapter_parameters = [p for p in adapter.parameters() if p.requires_grad]
    lora_lr = float(train_cfg["lora_learning_rate"]) if args.lora_lr < 0 else args.lora_lr
    adapter_lr = float(train_cfg["adapter_learning_rate"]) if args.adapter_lr < 0 else args.adapter_lr
    optimizer = torch.optim.AdamW([{"params": lora_parameters, "lr": lora_lr}, {"params": adapter_parameters, "lr": adapter_lr}],
                                  weight_decay=float(train_cfg["weight_decay"]), betas=(0.9, 0.999))
    warmup_steps = int(train_cfg["warmup_steps"]) if args.warmup_steps < 0 else int(args.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    if resume:
        checkpoint = Path(resume)
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if int(state.get("max_steps", max_steps)) != max_steps:
            raise ValueError(f"resume would change the LR schedule: {state.get('max_steps')} != {max_steps}")
        for key in ("seed", "mix", "pools", "pool_sizes", "ids_digest", "fm_weight_mode", "global_batch", "lora_lr",
                    "adapter_lr", "warmup_steps", "limit", "max_latent_frames", "init_from", "condition_interface", "sampler"):
            mine = regime.get(key) if key != "sampler" else SAMPLER_VERSION
            if json.loads(json.dumps(state.get(key))) != json.loads(json.dumps(mine)):
                raise ValueError(f"resume would change {key}: checkpoint {state.get(key)!r} != {mine!r}")
        backend.load_lora_parameters(checkpoint / "lora_params.pt")
        load_adapter_state(adapter, torch.load(checkpoint / "motion_adapter.pt", map_location="cpu", weights_only=True))
        start_step = int(state["step"])
    elif args.init_from:
        backend.load_lora_parameters(Path(args.init_from) / "lora_params.pt")
        load_adapter_state(adapter, torch.load(Path(args.init_from) / "motion_adapter.pt", map_location="cpu", weights_only=True))
    else:
        print(f"[init] rank {rank}: fresh LoRA (B = 0) + fresh {type(adapter).__name__}: the base model", flush=True)

    sampler = RankStream(start_step, target_step, grad_accum, world, rank)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=int(train_cfg["num_workers"]), pin_memory=True,
                        collate_fn=functools.partial(collate_one, prompt_pad_length=int(data_cfg["prompt_pad_length"])), persistent_workers=False)
    train_module, adapter, optimizer = accelerator.prepare(backend.transformer, adapter, optimizer)
    backend.transformer = train_module
    if resume:
        optimizer.load_state_dict(torch.load(Path(resume) / "optimizer.pt", map_location="cpu", weights_only=False))
        scheduler.load_state_dict(torch.load(Path(resume) / "scheduler.pt", map_location="cpu", weights_only=False))
    trainable = lora_parameters + adapter_parameters
    if accelerator.is_main_process:
        print(f"[model] expert={args.expert} world={world} trainable={sum(p.numel() for p in trainable) / 1e6:.2f}M "
              f"global_batch={grad_accum * world} steps={start_step}->{target_step} of {max_steps} resume={resume or None} "
              f"regime={json.dumps(regime)}", flush=True)

    def save_checkpoint(step: int, val: dict | None = None) -> None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            checkpoint = output_dir / f"checkpoint-{step:06d}"
            partial = output_dir / f".partial-{step:06d}"
            partial.mkdir(parents=True, exist_ok=True)
            backend.save_lora(partial, transformer=accelerator.unwrap_model(backend.transformer))
            torch.save(accelerator.unwrap_model(adapter).state_dict(), partial / "motion_adapter.pt")
            torch.save(optimizer.state_dict(), partial / "optimizer.pt")
            torch.save(scheduler.state_dict(), partial / "scheduler.pt")
            coverage = plan.summary(step * grad_accum * world, pools_rows)
            (partial / "trainer_state.json").write_text(json.dumps({"step": step, "expert": args.expert, "max_steps": max_steps,
                                                                    "condition_contract": cond_v2.CACHE_FORMAT, "coverage": coverage, "val": val,
                                                                    **regime, "sampler": SAMPLER_VERSION}, indent=2, sort_keys=True) + "\n")
            if checkpoint.exists():
                import shutil
                shutil.rmtree(checkpoint)
            partial.rename(checkpoint)
            print(f"[checkpoint] {checkpoint} unique_clips={coverage['unique_clips']}", flush=True)
            import shutil
            found = sorted(output_dir.glob("checkpoint-[0-9]*"))
            for old in found[:-2]:                       # the two newest stay (resume); kept checkpoints are never touched
                number = int(old.name.split("-")[1])
                if not (args.keep_every and number % args.keep_every == 0) and number != max_steps:
                    shutil.rmtree(old)
        accelerator.wait_for_everyone()

    if args.save_init:
        save_checkpoint(start_step)
        return

    metrics = (output_dir / "metrics.jsonl").open("a") if accelerator.is_main_process else None
    trace = (output_dir / f"steps_rank{rank}.jsonl").open("a")
    step, started = start_step, time.time()
    kinds: Counter = Counter()
    sums: Counter = Counter()
    counts: Counter = Counter()
    frames_seen = 0
    core = getattr(adapter, "module", adapter)
    gradient_norm = torch.tensor(0.0)
    step_started, last_log_step = time.time(), start_step
    import zlib
    from motion_jepa.stage2 import sample_expert_sigmas
    val_every = args.val_every or args.keep_every

    LATE0, NLAT = 8, 11
    # columns of the per-pool validation accumulator
    C_TUBE, C_PLAIN, C_N = 0, 1, 2
    CATS = ("late_real", "late_held", "late_mixed", "late", "early")
    C_CAT = {c: 3 + 4 * i for i, c in enumerate(CATS)}          # plain sum, plain count, tube num, tube den
    C_LAT = 3 + 4 * len(CATS)                                      # per latent j: plain sum (j), count (NLAT + j)
    NCOL = C_LAT + 2 * NLAT

    def latent_kind(j: int, frames_real: int) -> str:
        """What latent j of a clip with `frames_real` real frames shows (Wan VAE: latent 0 = frame 0, latent j = frames 4j-3..4j)."""
        last_real = frames_real - 1
        if 4 * j <= last_real:
            return "real"
        if 4 * j - 3 > last_real:
            return "held"
        return "mixed"

    def validate(at_step: int) -> dict | None:
        if not val_items:
            return None
        sums_v: dict[str, list[float]] = {}
        names = sorted({n for n, *_ in val_items})
        local = torch.zeros(len(names), NCOL, device=device, dtype=torch.float64)
        wrapped = backend.transformer
        backend.transformer = accelerator.unwrap_model(wrapped)      # no DDP collectives: ranks hold different item counts
        with torch.no_grad():
            for name, path, sid, k in val_items[rank::world]:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                keep = payload["video_latent"].shape[1]
                if args.max_latent_frames:
                    keep = min(keep, args.max_latent_frames)
                condition = read_cached_condition(payload, "real", interface, "cached")[:, :keep]
                weight = payload["fm_weight"].float()[:keep]
                weight = weight * (weight.numel() / weight.sum())
                prompt = torch.zeros(1, int(data_cfg["prompt_pad_length"]), payload["prompt_embeds"].shape[-1])
                emb = payload["prompt_embeds"].float()[: prompt.shape[1]]
                prompt[0, : emb.shape[0]] = emb
                video = payload["video_latent"].float()[None, :, :keep]
                generator = torch.Generator().manual_seed(zlib.crc32(f"{sid}:{k}:{args.expert}".encode()))
                sig = sample_expert_sigmas(1, expert=args.expert, boundary_sigma=float(model_cfg["boundary_sigma"]),
                                           shift=float(train_cfg["flow_shift"]), device=torch.device("cpu"), generator=generator)
                noise = torch.randn(video.shape, generator=generator)
                batch = Stage2Batch(video_latent=video, image_cond_latent=payload["image_cond_latent"].float()[None, :, :keep],
                                    prompt_embeds=prompt, motion_condition=condition[None], fm_weight=weight[None])
                with accelerator.autocast():
                    out = backend.train_step_v2(batch, core(condition[None].to(device)), mixer=core.mix, sigmas=sig, noise=noise)
                i = names.index(name)
                local[i, C_TUBE] += float(out["per_sample"].mean())
                local[i, C_PLAIN] += float(backend.last_terms["flow_matching_plain"])
                local[i, C_N] += 1
                per_pixel = (out["model_output"].float() - out["target"].float()).square().mean(1)[0]     # [T, h, w]
                per_frame = per_pixel.mean((1, 2)).double().cpu()
                w = weight.to(per_pixel.device, per_pixel.dtype)
                tube_num = (per_pixel * w).sum((1, 2)).double().cpu()
                tube_den = w.sum((1, 2)).double().cpu()
                frames_real = int(payload.get("frames_real", payload.get("frames", 4 * keep - 3)))
                for j in range(min(keep, NLAT)):
                    local[i, C_LAT + j] += float(per_frame[j])
                    local[i, C_LAT + NLAT + j] += 1
                    cats = []
                    if j >= LATE0:
                        cats += ["late", f"late_{latent_kind(j, frames_real)}"]
                    elif j >= 1:
                        cats.append("early")
                    for c in cats:
                        col = C_CAT[c]
                        local[i, col] += float(per_frame[j])
                        local[i, col + 1] += 1
                        local[i, col + 2] += float(tube_num[j])
                        local[i, col + 3] += float(tube_den[j])
        backend.transformer = wrapped
        local = accelerator.reduce(local, reduction="sum")
        result = {"step": at_step}

        def summarize(row: torch.Tensor) -> dict:
            n = float(row[C_N].clamp_min(1))
            entry = {"tube": float(row[C_TUBE]) / n, "plain": float(row[C_PLAIN]) / n, "n": int(row[C_N])}
            for c in CATS:
                col = C_CAT[c]
                cnt = int(row[col + 1])
                entry[c] = {"plain": float(row[col]) / cnt if cnt else None,
                            "tube": float(row[col + 2]) / float(row[col + 3]) if cnt and float(row[col + 3]) > 0 else None, "latents": cnt}
            entry["per_latent"] = [round(float(row[C_LAT + j]) / float(row[C_LAT + NLAT + j]), 6) if float(row[C_LAT + NLAT + j]) else None for j in range(NLAT)]
            return entry

        for i, name in enumerate(names):
            result[name] = summarize(local[i])
        result["all"] = summarize(local.sum(0))
        if accelerator.is_main_process:
            with (output_dir / "val.jsonl").open("a") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            print(f"[val] {json.dumps(result, sort_keys=True)}", flush=True)
        return result

    if not resume and start_step == 0 and val_items:
        validate(0)
    for raw in loader:
        micro = int(raw["cursor"])
        kinds[f"{raw['pool']}:{raw['kind']}"] += 1
        frames_seen += int(raw["video_latent"].shape[2])
        torch.manual_seed((seed * 1_000_003 + micro) * 31)                # sigma and noise are tied to the global sample
        with accelerator.accumulate(train_module, adapter):
            # The loader is not passed to accelerator.prepare (RankStream shards it), so its batches are on the CPU.
            # With several processes DDP moves the adapter input to the GPU; a single process does not, hence the
            # explicit move.  It copies the same fp32 values, so multi-GPU runs are unchanged.
            residuals = adapter(raw["motion_condition"].to(device))
            batch = Stage2Batch(video_latent=raw["video_latent"], image_cond_latent=raw["image_cond_latent"], prompt_embeds=raw["prompt_embeds"],
                                motion_condition=raw["motion_condition"], fm_weight=raw["fm_weight"] if args.fm_weight_mode == "tube" else None)
            out = backend.train_step_v2(batch, residuals, mixer=core.mix)
            loss, sigmas = out["loss"], out["sigmas"]
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                gradient_norm = accelerator.clip_grad_norm_(trainable, float(train_cfg["max_grad_norm"]))
                if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).all()):
                    raise FloatingPointError("nonfinite accumulated gradient; optimizer step aborted")
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        terms = dict(backend.last_terms)
        terms["total"] = float(loss.detach())
        for key, value in terms.items():
            sums[key] += value
            counts[key] += 1
        with torch.no_grad():
            per_frame = (out["model_output"].detach().float() - out["target"].float()).square().mean((1, 3, 4))[0]   # [T]
            if per_frame.numel() > 1:
                early = per_frame[1:LATE0]
                sums["_early_sum"] += float(early.sum())
                sums["_early_n"] += early.numel()
            if per_frame.numel() > LATE0:
                late = per_frame[LATE0:]
                sums["_late_sum"] += float(late.sum())
                sums["_late_n"] += late.numel()
                sums["_late_clips"] += 1
        trace.write(json.dumps({"micro": micro, "pool": raw["pool"], "sample_id": raw["sample_id"], "frames": int(raw["video_latent"].shape[2]),
                                "sigma": round(float(sigmas[0]), 6), "loss": round(float(loss.detach()), 6)}) + "\n")
        if not accelerator.sync_gradients:
            continue
        trace.flush()
        step += 1
        if args.deadline and time.time() > args.deadline:
            STOP["flag"] = True
        if args.stop_file and Path(args.stop_file).exists():
            STOP["flag"] = True
        stop_now = torch.tensor([1.0 if STOP["flag"] else 0.0], device=device)
        stop_now = accelerator.reduce(stop_now, reduction="sum")
        if step == start_step + 1 or step % log_every == 0:
            local = torch.tensor([sums["flow_matching"], counts["flow_matching"], sums["flow_matching_plain"], frames_seen,
                                  sum(kinds.values()), sums["_early_sum"], sums["_early_n"], sums["_late_sum"], sums["_late_n"],
                                  sums["_late_clips"]], device=device, dtype=torch.float64)
            local = accelerator.reduce(local, reduction="sum")
            if accelerator.is_main_process:
                record = {"step": step, "loss/flow_matching": float(local[0] / local[1].clamp_min(1)),
                          "loss/flow_matching_plain": float(local[2] / local[1].clamp_min(1)),
                          "loss/early_plain": float(local[5] / local[6].clamp_min(1)),
                          "loss/late_plain": float(local[7] / local[8]) if float(local[8]) > 0 else None,
                          "data/share_clips_with_late": float(local[9] / local[4].clamp_min(1)),
                          "grad_norm": float(gradient_norm), "lr/lora": scheduler.get_last_lr()[0], "lr/adapter": scheduler.get_last_lr()[1],
                          "latent_frames/mean": float(local[3] / local[4].clamp_min(1)), "world": world,
                          "data/unique_clips_drawn": plan.summary(step * grad_accum * world, pools_rows)["unique_clips"],
                          "sec_per_step": (time.time() - step_started) / max(1, step - last_log_step),
                          "elapsed_seconds": time.time() - started,
                          **{f"mix_rank0/{key}": value / max(1, sum(kinds.values())) for key, value in sorted(kinds.items())}}
                if device.type == "cuda":
                    record["cuda/max_memory_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                metrics.write(json.dumps(record, sort_keys=True) + "\n")
                metrics.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
            sums.clear()
            counts.clear()
            kinds.clear()
            frames_seen = 0
            step_started, last_log_step = time.time(), step
        val = validate(step) if val_items and step % val_every == 0 else None
        if step % save_every == 0 or step >= target_step or float(stop_now) > 0 or val is not None:
            save_checkpoint(step, val)
        if step >= target_step or float(stop_now) > 0:
            break
    trace.close()
    if accelerator.is_main_process:
        marker = "done.json" if step >= max_steps else "paused.json"
        (output_dir / marker).write_text(json.dumps({"expert": args.expert, "steps": step, "max_steps": max_steps, "stopped_by_signal": STOP["flag"],
                                                     "coverage": plan.summary(step * grad_accum * world, pools_rows), **regime}, indent=2, sort_keys=True) + "\n")
        print(f"[{marker[:-5]}] step {step} of {max_steps}", flush=True)
        metrics.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
