#!/usr/bin/env python
"""GPU stage of a real Stage-1 source: flow cache (first frame + Wan VAE image latent + prior placeholder) and the
umT5 prompt table, with the same code paths as scripts/cache_stage1_event_inputs.py (direct objective, zero FM).

  flowdit/<sid>.pt      first_frame uint8 [3,160,160], image_cond_latent [16,1,32,32] f16 (VAE mode of the 256 letterboxed
                        frame), flow_latent = the audited zero-flow prior slot repeated over the latent frames
                        (latent_semantics constant_prior_placeholder_direct_only), label_source, horizon_seconds, provenance
  metadata/prompt_table.pt  {embeddings{sha256(caption) -> [L,4096] bf16}, negative_key, prompt_pad_length=64, captions}

The first frame is recomputed here and must hash to the first_frame_sha256 the CPU stage stamped into
sam2_conditions/<sid>.pt (otherwise the loader would reject the condition).
usage: cache_stage1_real_inputs.py --config configs/stage1/train.yaml --root ROOT --prior PRIOR.pt [--phase images|prompts|both]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.stage1_real.build_stage1_real_v3 import LABEL_SOURCE, first_frame_160  # noqa: E402  (installs the h5py stub)
from motion_jepa.flow_codec import build_codec  # noqa: E402
from motion_jepa.flow_dit_config import load_config, resolve_derived  # noqa: E402
from motion_jepa.flow_dit_data import CACHE_FORMAT_VERSION, caption_key, first_frame_tensor_sha256  # noqa: E402
from scripts.cache_stage1_flowdit import read_first_frame, to_model_range  # noqa: E402
from scripts.cache_stage1_event_inputs import load_prior  # noqa: E402
from scripts.cache_stage1_flowdit_prompts import encode_captions  # noqa: E402
from scripts.stage1_paths import placeholder_values, resolve_placeholders  # noqa: E402


def manifest_rows(root: Path) -> list[dict]:
    return [json.loads(line) for split in ("train", "val") for line in (root / "manifests" / f"{split}.jsonl").read_text().splitlines() if line.strip()]


@torch.inference_mode()
def images(cfg: dict, root: Path, prior_path: Path) -> dict:
    summary = json.loads((root / "summary.json").read_text())
    horizons = [float(v) for v in summary["horizon_seconds"]]
    derived = resolve_derived({**cfg, "data": {**cfg["data"], "horizon_seconds": horizons}})
    prior, prior_codec = load_prior(prior_path, cfg["codec"])
    prior_hash = hashlib.sha256(prior.view(torch.uint8).numpy().tobytes()).hexdigest()
    if tuple(prior.shape[-2:]) != (derived["latent_size"], derived["latent_size"]):
        raise ValueError("prior resolution differs")
    codec = build_codec(cfg["codec"], model_dir=cfg["paths"]["wan_model"], device=torch.device("cuda"),
                        latent_size=derived["latent_size"])
    out = root / "flowdit"
    out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for row in manifest_rows(root):
        sid, frame_path = row["sample_id"], row["video_path"]
        target = out / f"{sid}.pt"
        if target.is_file():
            skipped += 1
            continue
        frame = read_first_frame(frame_path, int(cfg["data"]["flow_resolution"]))
        encoded = codec.vae.encode(to_model_range(frame)[None, :, None].cuda()).latent_dist.mode()[0].cpu().half()
        first = first_frame_160(frame_path)
        condition = torch.load(root / "sam2_conditions" / f"{sid}.pt", map_location="cpu", weights_only=True)
        if condition["first_frame_sha256"] != first_frame_tensor_sha256(first):
            raise ValueError(f"{sid}: first frame differs from the one stamped into its SAM2 condition")
        sample = {"cache_format_version": CACHE_FORMAT_VERSION, "sample_id": sid, "scenario": row["scenario"],
                  "label_source": LABEL_SOURCE, "horizon_seconds": horizons, "first_frame": first,
                  "first_frame_source": frame_path, "image_cond_latent": encoded,
                  "flow_latent": prior.expand(-1, derived["latent_frames"], -1, -1).clone(),
                  "latent_semantics": "constant_prior_placeholder_direct_only",
                  "codec": {**prior_codec, "latent_frames": derived["latent_frames"]},
                  "provenance": {"prior_cache": str(prior_path), "prior_sha256": prior_hash,
                                 "first_frame_sha256": hashlib.sha256(Path(frame_path).read_bytes()).hexdigest(),
                                 "future_labels_used_for_inputs": False}}
        temp = target.with_name(target.name + f".tmp{os.getpid()}")
        torch.save(sample, temp)
        temp.replace(target)
        done += 1
        if done % 500 == 0:
            print(json.dumps({"root": str(root), "cached": done}), flush=True)
    return {"root": str(root), "cached": done, "already": skipped, "prior_sha256": prior_hash,
            "latent_frames": derived["latent_frames"]}


def prompts(cfg: dict, root: Path) -> dict:
    captions = json.loads((root / "metadata" / "captions.json").read_text())["captions"]
    captions[caption_key("")] = ""
    if any(caption_key(text) != key for key, text in captions.items()):
        raise ValueError("caption text does not match its key")
    pad = int(cfg["data"]["prompt_pad_length"])
    embeddings, provenance = encode_captions(captions, Path(cfg["paths"]["wan_model"]), pad)
    destination = root / "metadata" / "prompt_table.pt"
    temp = destination.with_name(destination.name + f".tmp{os.getpid()}")
    torch.save({"embeddings": embeddings, "negative_key": caption_key(""), "prompt_pad_length": pad,
                "captions": captions, "provenance": provenance}, temp)
    temp.replace(destination)
    lengths = [int(v.shape[0]) for v in embeddings.values()]
    return {"root": str(root), "prompts": len(embeddings), "max_tokens": max(lengths), "pad": pad}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--root", required=True, nargs="+", type=Path)
    p.add_argument("--prior", required=True, type=Path,
                   help="zero-motion prior: a flow-cache sample or the released prior_latent.pt")
    p.add_argument("--phase", choices=("images", "prompts", "both"), default="both")
    a = p.parse_args()
    torch.set_num_threads(4)
    cfg = resolve_placeholders(load_config(a.config), placeholder_values(), strict=False)
    if cfg["train"]["objective"] != "direct" or float(cfg["loss"]["flow_matching"]) != 0:
        raise ValueError("observation-only cache is only valid for direct, zero-FM training")
    for root in a.root:
        if a.phase in ("prompts", "both"):
            print("PROMPTS", json.dumps(prompts(cfg, root)), flush=True)
        if a.phase in ("images", "both"):
            print("IMAGES", json.dumps(images(cfg, root, a.prior)), flush=True)


if __name__ == "__main__":
    main()
