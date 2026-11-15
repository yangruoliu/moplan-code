#!/usr/bin/env python3
"""Smoke run directory of the Stage-2 GT-SFT trainer (v2/train_s2sft_v2.py), and the check of a finished smoke run.

build:  make_smoke_run.py SMOKE_DIR RUN_DIR [--synthetic N]
  Payload folders per source: SMOKE_DIR/p101_final or SMOKE_DIR/p101, SMOKE_DIR/ssv2_final or SMOKE_DIR/ssv2,
  SMOKE_DIR/phyco (stage2_cond_v2 payloads, e.g. a few clips through the cache builders, or a few payloads of a full
  build).  Per source the first clip (sorted) is held out for validation and the rest are the training pool; a source
  with fewer than 2 clips is left out.  Mix shares of the released run (P101 .5, SSv2 .1, PhyCo .4), max_latent_frames
  11, global batch 16, max_steps 8000 (the LR schedule of a real run), val_sigmas 2.  Writes RUN_DIR/OVERRIDES.json
  and RUN_DIR/pools/{train,val}_<source>.txt.
  --synthetic N first writes N random payloads per source with the shapes and keys of the contract (no dataset is
  needed; the losses then mean nothing, the run only exercises loading, the model, the optimizer and checkpointing).
check:  make_smoke_run.py --check RUN_DIR [--expert high] [--step 2]
  Verifies RUN_DIR/<expert>: finite metrics.jsonl losses, checkpoint-<step>/{lora_params.pt, motion_adapter.pt,
  trainer_state.json} (step recorded), a nonzero LoRA B after training, finite val.jsonl entries, a done/paused marker.
"""
import argparse
import json
import math
import sys
from pathlib import Path

EXTRA_NAMES = ["known", "availability", "selector", "raw_flow_x", "raw_flow_y", "motion_prob", "source_mask"]


def synthetic_payloads(folder: Path, source: str, count: int, seed: int, latent_hw=(32, 56), full: int = 11, short: int = 8) -> None:
    """`count` random stage2_cond_v2 payloads (default: 448x256 work frame -> 32x56 latents, 16 latent channels, umT5
    width 4096); the second clip is shorter (`short` latents) so that variable clip lengths are exercised."""
    import torch

    folder.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    h, w = latent_hw
    for index in range(count):
        latents = short if index == 1 else full
        frames = 4 * latents - 3
        cond = torch.cat((0.1 * torch.randn(20, latents, h, w, generator=generator),
                          torch.rand(len(EXTRA_NAMES), latents, h, w, generator=generator)))
        weight = 1.0 + torch.rand(latents, h, w, generator=generator)
        payload = {"cache_format_version": "stage2_cond_v2", "condition_channels_v2": list(EXTRA_NAMES), "base_channels": 20,
                   "sample_id": f"synthetic_{source}_{index:03d}", "source": source, "prompt": "a synthetic smoke-test clip",
                   "frames": frames, "frames_real": frames, "latent_frames": latents, "fps": 7.0, "height": 256, "width": 448,
                   "video_latent": torch.randn(16, latents, h, w, generator=generator).half(),
                   "image_cond_latent": torch.randn(16, latents, h, w, generator=generator).half(),
                   "cond_real": cond.half(), "cond_null": torch.zeros(27, latents, h, w).half(),
                   "fm_weight": (weight / weight.mean()).half(),
                   "prompt_embeds": 0.1 * torch.randn(24, 4096, generator=generator)}
        torch.save(payload, folder / f"{payload['sample_id']}.pt")


def build(smoke: Path, run: Path, synthetic: int) -> None:
    if synthetic:
        for offset, name in enumerate(("p101", "ssv2", "phyco")):
            synthetic_payloads(smoke / name, name, synthetic, seed=1234 + offset)
    (run / "pools").mkdir(parents=True, exist_ok=True)
    pools, vals, mix = [], [], {"p101": 0.5, "ssv2": 0.1, "phyco": 0.4}
    for name in ("p101", "ssv2", "phyco"):
        folder = next((f for f in (smoke / f"{name}_final", smoke / name) if f.is_dir()), smoke / name)
        ids = sorted(p.stem for p in folder.glob("*.pt") if not p.name.startswith("summary-"))
        if len(ids) < 2:
            mix.pop(name)
            continue
        (run / "pools" / f"train_{name}.txt").write_text("".join(s + "\n" for s in ids[1:]))
        (run / "pools" / f"val_{name}.txt").write_text(ids[0] + "\n")
        pools.append(f"{name}={folder.resolve()}@{(run / 'pools' / f'train_{name}.txt').resolve()}")
        vals.append(f"{name}={folder.resolve()}@{(run / 'pools' / f'val_{name}.txt').resolve()}")
    if not mix:
        raise SystemExit(f"no source under {smoke} has 2 or more payloads")
    total = sum(mix.values())
    ov = {"pools": ",".join(pools), "mix": ",".join(f"{n}:real:{w / total:.5f}" for n, w in mix.items()), "fm_weight_mode": "tube",
          "max_latent_frames": 11, "init_from": "", "max_steps": 8000, "save_every": 100, "keep_every": 500, "global_batch": 16,
          "log_every": 10, "seed": 42, "val_pools": ",".join(vals), "val_sigmas": 2}
    (run / "OVERRIDES.json").write_text(json.dumps(ov, indent=1) + "\n")
    print(json.dumps(ov, indent=1))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def check(run: Path, expert: str, step: int) -> int:
    folder = run / expert
    problems = []
    metrics = [json.loads(l) for l in (folder / "metrics.jsonl").read_text().splitlines() if l.strip()] if (folder / "metrics.jsonl").is_file() else []
    if not metrics:
        problems.append("no metrics.jsonl records")
    for record in metrics:
        for key in ("loss/flow_matching", "loss/flow_matching_plain", "grad_norm"):
            if not finite(record.get(key)):
                problems.append(f"metrics step {record.get('step')}: {key} = {record.get(key)!r}")
    checkpoint = folder / f"checkpoint-{step:06d}"
    for name in ("lora_params.pt", "motion_adapter.pt", "trainer_state.json"):
        if not (checkpoint / name).is_file():
            problems.append(f"missing {checkpoint / name}")
    if (checkpoint / "trainer_state.json").is_file():
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if int(state.get("step", -1)) != step:
            problems.append(f"trainer_state step {state.get('step')} != {step}")
    if (checkpoint / "lora_params.pt").is_file():
        import torch
        lora = torch.load(checkpoint / "lora_params.pt", map_location="cpu", weights_only=True)
        b = [v for k, v in lora.items() if "lora_B" in k]
        if not b:
            problems.append("no lora_B tensors in lora_params.pt")
        elif step > 0 and not any(bool(v.float().abs().sum() > 0) for v in b):
            problems.append("every LoRA B is zero after training")
        if not all(bool(torch.isfinite(v.float()).all()) for v in lora.values()):
            problems.append("nonfinite LoRA parameters")
    val = [json.loads(l) for l in (folder / "val.jsonl").read_text().splitlines() if l.strip()] if (folder / "val.jsonl").is_file() else []
    for record in val:
        if not finite(record.get("all", {}).get("plain")):
            problems.append(f"val step {record.get('step')}: all.plain = {record.get('all', {}).get('plain')!r}")
    if not ((folder / "done.json").is_file() or (folder / "paused.json").is_file()):
        problems.append("no done.json / paused.json")
    summary = {"expert": expert, "step": step, "metrics_records": len(metrics), "val_records": len(val),
               "last_loss": metrics[-1].get("loss/flow_matching") if metrics else None, "problems": problems}
    print(json.dumps(summary, indent=1))
    return 1 if problems else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="SMOKE_DIR RUN_DIR (build) or RUN_DIR (--check)")
    p.add_argument("--synthetic", type=int, default=0, help="write N synthetic payloads per source first")
    p.add_argument("--check", action="store_true")
    p.add_argument("--expert", default="high", choices=("high", "low"))
    p.add_argument("--step", type=int, default=2)
    a = p.parse_args()
    if a.check:
        if len(a.paths) != 1:
            p.error("--check takes RUN_DIR")
        sys.exit(check(Path(a.paths[0]), a.expert, a.step))
    if len(a.paths) != 2:
        p.error("build takes SMOKE_DIR RUN_DIR")
    build(Path(a.paths[0]), Path(a.paths[1]), a.synthetic)


if __name__ == "__main__":
    main()
