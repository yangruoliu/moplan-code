#!/usr/bin/env python
"""Cache observation-only inputs for deterministic, zero-FM event training.

Wan's direct path uses only the first (zero-flow) latent as prior. Encoding
future labels is unnecessary here. Share one RGB VAE encoding per scene and
explicitly mark all future latent entries as unsupervised placeholders.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import cv2
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_jepa.flow_dit_config import load_config, resolve_derived
from motion_jepa.flow_dit_data import CACHE_FORMAT_VERSION, caption_key
from motion_jepa.flow_codec import build_codec
from scripts.cache_stage1_flowdit import read_first_frame, to_model_range
from scripts.cache_stage1_flowdit_prompts import encode_captions
from scripts.stage1_paths import placeholder_values, resolve_placeholders


def load_prior(path, codec_config):
    """Return (prior [16,1,S,S], codec metadata) from a prior file.

    Accepts a Stage-1 flow-cache sample (``flow_latent`` [16,T,S,S] and
    ``codec``; its zero-motion slot 0 is used) or the released prior asset
    (``prior_latent`` [16,S,S], or a bare tensor). A flow-cache sample gives
    its own slot 0, which equals the released prior only if that sample was
    built with it; the reference FOT caches were not (see docs/DATA_STAGE1.md).
    The codec block is metadata only and falls back to the config's.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        value = payload["flow_latent"] if "flow_latent" in payload else payload["prior_latent"]
        codec = payload["codec"] if "codec" in payload else dict(codec_config)
    else:
        value, codec = payload, dict(codec_config)
    if value.ndim == 3:
        value = value[:, None]
    return value[:, :1].contiguous(), codec


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["images", "prompts"])
    p.add_argument("--config", required=True)
    p.add_argument("--prior-cache", type=Path,
                   help="zero-motion prior: a flow-cache sample or the released prior_latent.pt")
    p.add_argument("--source-name", default="")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args(); torch.set_num_threads(1)
    cfg = resolve_placeholders(load_config(a.config), placeholder_values(), strict=False)
    if a.source_name:
        source = next(s for s in cfg["data"]["additional_sources"] if s["name"] == a.source_name)
        cfg["paths"].update({k: source[k] for k in ("manifest_dir", "flow_cache_dir", "target_cache_dir", "prompt_table")})
        cfg["data"]["horizon_seconds"] = source["horizon_seconds"]
        cfg["provenance"]["label_source"] = source["label_source"]
    paths=cfg["paths"]; data=cfg["data"]; derived=resolve_derived(cfg)
    if cfg["train"]["objective"] != "direct" or cfg["loss"]["flow_matching"] != 0:
        raise ValueError("observation-only cache is only valid for direct, zero-FM training")
    root = Path(paths["manifest_dir"]).parent
    rows = [json.loads(line) for split in ("train", "val")
            for line in (root / "manifests" / (split + ".jsonl")).read_text().splitlines()]
    if a.phase == "prompts":
        captions = {caption_key(r["caption"]): r["caption"] for r in rows}
        for split in ("train", "val"):
            pairs = json.loads((root / f"event_pairs_{split}.json").read_text())["pairs"]
            for specs in pairs.values():
                for spec in specs:
                    captions[caption_key(spec["caption"])] = spec["caption"]
        captions[caption_key("")] = ""
        embeddings, receipt = encode_captions(captions, Path(paths["wan_model"]), int(data["prompt_pad_length"]))
        destination = Path(paths["prompt_table"])
        temporary = destination.with_suffix(".tmp")
        torch.save({"embeddings": embeddings, "negative_key": caption_key(""),
                    "prompt_pad_length": data["prompt_pad_length"], "provenance": receipt}, temporary)
        temporary.replace(destination)
        print(json.dumps({"prompts": len(embeddings)}), flush=True)
        return
    if a.prior_cache is None:
        raise ValueError("an audited legacy zero-flow prior cache is required")
    prior, prior_codec = load_prior(a.prior_cache, cfg["codec"])
    prior_hash = hashlib.sha256(prior.view(torch.uint8).numpy().tobytes()).hexdigest()
    if tuple(prior.shape[-2:]) != (derived["latent_size"], derived["latent_size"]):
        raise ValueError("prior resolution differs")
    groups = defaultdict(list)
    missing_targets = 0
    for row_index, row in enumerate(rows):
        if a.source_name:
            # Shard before reading simulator targets. Their own video_path is
            # authoritative; relocated manifests may contain an old root.
            if row_index % a.num_shards != a.shard_index:
                continue
            target_path = Path(paths["target_cache_dir"]) / (row["sample_id"] + ".pt")
            if not target_path.is_file():
                missing_targets += 1
                continue
            target = torch.load(target_path, map_location="cpu", weights_only=True)
            if target["horizon_seconds"] != data["horizon_seconds"]:
                raise ValueError("native simulator times differ from the configured source")
            frame_path = target["video_path"]
        else:
            frame_path = row["video_path"]
        groups[frame_path].append(row)
    codec = build_codec(cfg["codec"], model_dir=paths["wan_model"], device=torch.device("cuda"),
                        latent_size=derived["latent_size"])
    out = Path(paths["flow_cache_dir"]); out.mkdir(parents=True, exist_ok=True)
    count = 0
    frame_paths = sorted(groups) if a.source_name else sorted(groups)[a.shard_index::a.num_shards]
    for frame_path in frame_paths:
        members = groups[frame_path]
        if all((out / (r["sample_id"] + ".pt")).exists() for r in members):
            continue
        frame = read_first_frame(frame_path, int(data["flow_resolution"]))
        encoded = codec.vae.encode(to_model_range(frame)[None, :, None].cuda()).latent_dist.mode()[0].cpu().half()
        first = torch.from_numpy(cv2.resize(frame, (int(data["width"]), int(data["height"])),
                                            interpolation=cv2.INTER_AREA)).permute(2, 0, 1).contiguous()
        for row in members:
            sample = {"cache_format_version": CACHE_FORMAT_VERSION, "sample_id": row["sample_id"],
                      "scenario": row["scenario"], "label_source": cfg["provenance"]["label_source"],
                      "horizon_seconds": data["horizon_seconds"], "first_frame": first,
                      "first_frame_source": frame_path, "image_cond_latent": encoded,
                      "flow_latent": prior.expand(-1, derived["latent_frames"], -1, -1).clone(),
                      "latent_semantics": "constant_prior_placeholder_direct_only",
                      "codec": {**prior_codec, "latent_frames": derived["latent_frames"]},
                      "provenance": {"prior_cache": str(a.prior_cache), "prior_sha256": prior_hash,
                        "first_frame_sha256": hashlib.sha256(Path(frame_path).read_bytes()).hexdigest(),
                        "future_labels_used_for_inputs": False}}
            torch.save(sample, out / (row["sample_id"] + ".pt")); count += 1
        if count % 100 < len(members):
            print(json.dumps({"shard": a.shard_index, "cached": count}), flush=True)
    print(json.dumps({"shard": a.shard_index, "cached": count, "prior_sha256": prior_hash,
                      "missing_targets": missing_targets}), flush=True)

if __name__ == "__main__":
    main()
