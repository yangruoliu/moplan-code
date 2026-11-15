#!/usr/bin/env python
"""Stage-1 inference on arbitrary first frames: image + caption + SAM2 instance masks -> motion plan export.

The input is a manifest CSV (sample_id, first_frame_path, caption); no targets are needed.  The script builds the
model inputs directly:

* the first frame, letterboxed to the square grids the model saw in training (RGB at 160 for the decoder, VAE
  latent at flow_resolution);
* captions encoded and padded with umT5 before loading the DiT, so both large models are never resident together;
* the decoding config from --decode-from, or the recorded evaluation settings of --config.

Writes ``flows/<sample_id>.pt`` with the fields the Stage-2 condition renderer consumes (``flow_ungated``,
``activity_input_instance_masks``, ``task_object_slot_probability``, ``motion_mask``, ``horizon_seconds``,
``geometry``, ...).  Each payload records checkpoint/config/decode provenance and the letterbox geometry.  Input
files are recorded by file name (plus sha256 where the provenance has one), never by absolute path; the
``stage1_provenance`` record of motion_jepa/flow_dit_export.py names the checkpoint folder and the sha256 of its
state files, which scripts/clean_r2/refresh_predictions.py compares with the checkpoint in use.

Checkpoint directory (``--checkpoint``): backend_params.pt, decoder.pt, trainer_state.json, the inference config
resolved_config.yaml (default for --config), optional checkpoint_complete.json, and the direct-objective prior
latent prior_latent.pt (pass it with --prior-latent).  The Wan2.2 directory comes from ``--set paths.wan_model=DIR``
or, when the config leaves it empty, from $WAN_MODEL_DIR.

Text batching: captions are encoded in batches of --text-batch-size together with the empty negative prompt.  To
reproduce the exact embeddings of a batch run for a single sample, pass the batch run's manifest (and shard) with
--text-manifest / --text-shard-index / --text-num-shards.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_jepa.flow_codec import build_codec
from motion_jepa.flow_dit import build_decoder
from motion_jepa.flow_dit_config import apply_overrides, load_config, resolve_derived
from motion_jepa.flow_dit_data import pad_prompt
from motion_jepa.flow_dit_export import bridge_fields, checkpoint_provenance
from motion_jepa.flow_dit_sampler import (
    SamplerConfig, decode_samples, reduce_samples, sample_flow_latents,
)
from scripts.eval_stage1_flowdit import load_backend

NEGATIVE_PROMPT = ""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_weight_hashes(checkpoint: str | Path) -> dict[str, str]:
    """Bind inference to actual weights, including legacy checkpoints without markers."""
    return {name: file_sha256(Path(checkpoint) / name)
            for name in ("decoder.pt", "backend_params.pt")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 inference on first frames")
    parser.add_argument("--config", default="", help="Stage-1 config (default: <checkpoint>/resolved_config.yaml)")
    weights = os.environ.get("MOPLAN_WEIGHTS", "")
    parser.add_argument("--checkpoint", default=str(Path(weights) / "stage1") if weights else None, required=not weights,
                        help="Stage-1 checkpoint folder (default $MOPLAN_WEIGHTS/stage1)")
    parser.add_argument("--manifest", required=True, help="CSV with sample_id, first_frame_path, caption")
    parser.add_argument("--decode-from", default="",
                        help="decoding-tuning JSON (best_by_object_epe_px_micro); when omitted, "
                             "explicitly use and record --config evaluation settings")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-raw-flow", action="store_true",
        help="also save the same decoded samples reduced with gain=1 and "
             "shrinkage=0; does not run an additional model forward pass",
    )
    parser.add_argument(
        "--query-horizon-seconds", default="",
        help="optional comma-separated inference times; preserves the validated "
             "training label contract and requires the same anchor count, "
             "positive increasing finite times, all <= data.time_scale_seconds",
    )
    parser.add_argument("--prior-latent", default="",
                        help="optional raw [C,H,W] prior, or cache .pt containing flow_latent; "
                             "without a cache, encode zero flow/all-visible once with the codec")
    parser.add_argument("--text-batch-size", type=int, default=8,
                        help="caption batch size; all text is encoded before loading DiT")
    parser.add_argument(
        "--instance-masks", default="",
        help="directory of {sample_id}.pt with instance_masks [K,side,side] and "
             "instance_channels [2,side,side] (e.g. from SAM2); required when "
             "the checkpoint was trained with oracle masks",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--text-manifest", default="",
                        help="encode the captions of this manifest (in its order, as a batch run over it did) instead of "
                             "only the rows being exported; reproduces the text batch composition of that run")
    parser.add_argument("--text-shard-index", type=int, default=0)
    parser.add_argument("--text-num-shards", type=int, default=1)
    return parser.parse_args(argv)


def letterbox_square(image: np.ndarray, side: int) -> np.ndarray:
    height, width = image.shape[:2]
    geometry = letterbox_geometry(height, width, side)
    inner_h, inner_w = geometry["content_height"], geometry["content_width"]
    resized = cv2.resize(image, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((side, side, 3), np.uint8)
    top, left = geometry["content_top"], geometry["content_left"]
    out[top: top + inner_h, left: left + inner_w] = resized
    return out


def letterbox_geometry(height: int, width: int, side: int) -> dict:
    if min(height, width, side) < 1:
        raise ValueError("source dimensions and grid side must be positive")
    scale = side / max(height, width)
    inner_h, inner_w = max(1, round(height * scale)), max(1, round(width * scale))
    return {
        "flow_units": "normalized_by_max_side", "grid_height": side, "grid_width": side,
        "content_top": (side - inner_h) // 2, "content_left": (side - inner_w) // 2,
        "content_height": inner_h, "content_width": inner_w,
        "normalization_scale": side, "source_height": height, "source_width": width,
    }


def stable_sample_seed(seed: int, sample_id: str) -> int:
    """Independent of PYTHONHASHSEED, shard assignment, and row order."""
    offset = int.from_bytes(hashlib.sha256(sample_id.encode("utf-8")).digest()[:8], "big")
    return (int(seed) + offset) % (2**63)


def resolve_sampler(config: dict, decode_from: str, *, seed: int) -> tuple[SamplerConfig, dict]:
    sampler = SamplerConfig.from_config(config)
    provenance = {"kind": "config_evaluation", "evaluation": dict(config["evaluation"])}
    if decode_from:
        path = Path(decode_from)
        contents = path.read_bytes()
        chosen = json.loads(contents)["best_by_object_epe_px_micro"]
        for key in ("num_steps", "num_samples", "guidance_scale", "shrinkage_px",
                    "flow_shift", "gain", "reduce", "boundary_sigma"):
            if key in chosen:
                setattr(sampler, key, type(getattr(sampler, key))(chosen[key]))
        provenance = {
            "kind": "validation_decode_file", "path": path.name,
            "sha256": hashlib.sha256(contents).hexdigest(), "chosen": chosen,
        }
    sampler.seed = int(seed)
    sampler.eval_side = float(config["data"]["height"])
    if sampler.num_samples < 1 or sampler.num_steps < 1 or sampler.reduce not in ("mean", "first"):
        raise ValueError(f"invalid sampler: {sampler}")
    if any(not math.isfinite(getattr(sampler, key)) for key in (
        "guidance_scale", "shrinkage_px", "flow_shift", "gain", "boundary_sigma",
    )) or sampler.shrinkage_px < 0 or sampler.flow_shift <= 0 or sampler.gain < 0:
        raise ValueError(f"invalid sampler: {sampler}")
    return sampler, provenance


def raw_flow_payload(decoded: dict, sampler: SamplerConfig, *, eval_side: float) -> dict:
    """Expose pre-gain/pre-shrinkage flow from the already decoded samples."""
    raw = reduce_samples(
        decoded, reduce=sampler.reduce, shrinkage_px=0.0,
        eval_side=eval_side, gain=1.0,
    )
    return {
        "raw_flow": raw["flow"].cpu().half(),
        "raw_decode": {
            **asdict(sampler), "shrinkage_px": 0.0, "gain": 1.0,
            "kind": "pre_gain_pre_shrinkage_reduced_flow",
            "source": "same_decoded_samples",
        },
    }


def motion_activity_payload(reduced: dict, *, instance_masks: torch.Tensor | None,
                            eval_side: int) -> dict:
    """Export per-query activity and its predicted future location, without GT.

    The source-coordinate mask is the supervised activity prediction. Its
    future-coordinate splat is a derived visualization, not independently
    supervised future segmentation. Both alternative flows are unshrunk.
    """
    if "motion_mask" not in reduced:
        return {}
    dense = "dense_motion_mask_logits" in reduced
    if not dense and instance_masks is None:
        raise ValueError("motion activity export requires the actual input slots")
    from motion_jepa.video import forward_splat_mask

    probability = reduced["motion_mask"].float()
    flow = reduced["flow_ungated"].float()
    if probability.shape != (flow.shape[0], 1, eval_side, eval_side):
        raise ValueError("activity and flow query grids differ")
    # Treat each query as its own one-horizon batch: activity can change with
    # time, unlike the first-frame union used by the legacy splat interface.
    future = forward_splat_mask(probability, flow[:, None] * eval_side)[:, 0]
    result = {
        "future_motion_mask": future.cpu().half(),
        "motion_activity_contract": {
            "prediction_kind": "direct_full_grid_dense_mask" if dense else "input_slot_activity_plus_outside_pixels",
            "motion_mask_coordinates": "first_frame_letterbox_grid",
            "motion_mask_shape": "horizon,1,height,width",
            "meaning": "nonzero_displacement_relative_to_first_frame_at_query_time",
            "future_motion_mask": "derived_splat_of_motion_mask_using_flow_ungated_not_GT",
            "alternative_flow_units": "normalized_by_eval_side",
            "alternative_flow_readout": "flow_ungated_and_flow_activity_gated_have_gain1_shrinkage0",
            "sample_reduction": "same_samples_as_flow; average_gated_flows_not_product_of_averages",
            "hard_activity_threshold_applied": False,
        },
    }
    if instance_masks is not None:
        result["activity_input_instance_masks"] = instance_masks.cpu().bool()
    if dense:
        # Near logit zero, FP16/BF16 probabilities can round up to exactly .5
        # and invert the fixed classification decision. Keep this learned
        # segmentation output in FP32; flow export retains its existing dtype.
        result["motion_mask"] = probability.cpu()
        result["dense_motion_mask_logits"] = reduced["dense_motion_mask_logits"].float().cpu()
        result["motion_activity_contract"].update(
            mask_construction="sigmoid_of_direct_dense_logits_not_blended_with_instance_masks",
            supervision="all_trusted_moving_and_certified_static_pixels_inside_and_outside_input_instances",
            unknown_labels="ignored_not_static",
            mask_storage_dtype="float32",
        )
        slot_logits = reduced.get("task_object_slot_logits")
        if slot_logits is not None:
            if instance_masks is None:
                raise ValueError("task-object slot export requires the input instance masks")
            expected_logits = (instance_masks.shape[0], flow.shape[0])
            slot_mask = reduced.get("task_object_slot_mask")
            slot_flow = reduced.get("flow_task_object_gated")
            if (
                not isinstance(slot_logits, torch.Tensor)
                or tuple(slot_logits.shape) != expected_logits
                or not torch.isfinite(slot_logits).all()
                or not isinstance(slot_mask, torch.Tensor)
                or tuple(slot_mask.shape) != tuple(probability.shape)
                or not torch.isfinite(slot_mask).all()
                or ((slot_mask < 0) | (slot_mask > 1)).any()
                or not isinstance(slot_flow, torch.Tensor)
                or tuple(slot_flow.shape) != tuple(flow.shape)
                or not torch.isfinite(slot_flow).all()
            ):
                raise ValueError("invalid task-object slot prediction export")
            result.update(
                task_object_slot_logits=slot_logits.float().cpu(),
                task_object_slot_mask=slot_mask.float().cpu(),
                flow_task_object_gated=slot_flow.cpu().half(),
            )
            for key in (
                "task_object_selection_mask", "flow_task_object_soft_gated",
                "task_object_slot_probability",
                "task_object_identity_logits",
                "task_object_identity_probability",
                "task_object_motion_probability",
            ):
                value = reduced.get(key)
                if value is not None:
                    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                        raise ValueError(f"invalid {key} export")
                    result[key] = value.float().cpu()
            result["motion_activity_contract"].update(
                task_object_slot_prediction=(
                    "time_invariant_identity_probability_times_per_query_motion_probability"
                    if "task_object_identity_logits" in reduced
                    else "per_input_sam_proposal_per_query"
                ),
                task_object_slot_mask="probability_rendered_in_first_frame_coordinates",
                task_object_slot_gate="diagnostic_alternative_flow_not_ground_truth_support",
            )
    else:
        result["slot_activity_probabilities"] = reduced["slot_activity_logits"].sigmoid().cpu().half()
        result["motion_activity_contract"]["slot_activity_shape"] = "input_slot,horizon"
    return result


def resolve_query_horizons(config: dict, query: str) -> list[float]:
    """Inference-only query grid, separate from training-label timestamps."""
    training = [float(value) for value in config["data"]["horizon_seconds"]]
    if not query:
        return training
    try:
        times = [float(value.strip()) for value in query.split(",")]
        scale = float(config["data"]["time_scale_seconds"])
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError("query horizons require numeric times and an explicit finite time scale") from error
    if (len(times) != len(training) or not times
            or not math.isfinite(scale) or scale <= 0
            or any(not math.isfinite(t) or t <= 0 or t > scale for t in times)
            or any(a >= b for a, b in zip(times, times[1:]))):
        raise ValueError(
            "query horizons must preserve the training anchor count and be "
            "finite, positive, strictly increasing and within data.time_scale_seconds"
        )
    return times


@torch.inference_mode()
def encode_prompt_batch(prompts, tokenizer, text_encoder, *, pad_length, device):
    tokens = tokenizer(
        prompts, padding="max_length", max_length=pad_length, truncation=True,
        add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
    )
    hidden = text_encoder(
        input_ids=tokens.input_ids.to(device),
        attention_mask=tokens.attention_mask.to(device),
    ).last_hidden_state
    # Cached training prompts retain real tokens only; collate then zero-pads.
    # Tokenizer pad positions' T5 hidden states must not enter the DiT.
    lengths = tokens.attention_mask.sum(dim=1).tolist()
    return [pad_prompt(hidden[i, :max(int(length), 1)].cpu().float(), pad_length)
            for i, length in enumerate(lengths)]


def load_prompt_cache(rows: list[dict], config: dict, *, device, batch_size: int):
    from transformers import T5TokenizerFast, UMT5EncoderModel

    if batch_size < 1:
        raise ValueError("--text-batch-size must be positive")
    model_dir = Path(config["paths"]["wan_model"])
    tokenizer = T5TokenizerFast.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_dir / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True,
    ).to(device).eval()
    captions = list(dict.fromkeys([NEGATIVE_PROMPT] + [row["caption"] for row in rows]))
    cache = {}
    try:
        for start in range(0, len(captions), batch_size):
            batch = captions[start:start + batch_size]
            values = encode_prompt_batch(
                batch, tokenizer, text_encoder,
                pad_length=int(config["data"]["prompt_pad_length"]), device=device,
            )
            cache.update(zip(batch, values))
    finally:
        del text_encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    print(f"[text] {len(cache)} unique captions cached on CPU; T5 released", flush=True)
    return cache


def load_direct_prior(codec, config: dict, derived: dict, *, device, prior_file: str = ""):
    source = Path(prior_file) if prior_file else None
    if source is None:
        cache_dir = config["paths"].get("flow_cache_dir", "")
        if cache_dir:
            source = next(iter(sorted(Path(cache_dir).glob("*.pt"))), None)
    if source is not None:
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            value = payload.get("prior_latent", payload.get("flow_latent"))
        else:
            value = payload
        if not torch.is_tensor(value):
            raise ValueError(f"{source} must contain prior_latent or flow_latent")
        if value.ndim == 4:
            value = value[:, 0]
        provenance = {"kind": "prior_file" if prior_file else "flow_cache_zero_slot",
                      "path": source.name, "sha256": file_sha256(source)}
    else:
        # Exactly the prepend_zero_slot image: zero x/y flow and visibility 1.
        resolution = int(config["data"]["flow_resolution"])
        image = torch.zeros(1, 3, resolution, resolution, device=device)
        image[:, 2] = 1.0
        value = codec.encode(image)[:, 0]
        provenance = {"kind": "codec_zero_flow_all_visible", "codec": dict(config["codec"])}
    expected = (derived["latent_channels"], derived["latent_size"], derived["latent_size"])
    if tuple(value.shape) != expected or not torch.isfinite(value).all():
        raise ValueError(f"invalid direct prior shape/value: expected {expected}, got {tuple(value.shape)}")
    # Training writes raw VAE latents to fp16 cache and reads them as fp32.
    return value.to(device).half().float(), provenance


@torch.inference_mode()
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.limit < 0:
        raise ValueError("invalid shard index/count or negative limit")
    if not args.config:
        args.config = str(Path(args.checkpoint) / "resolved_config.yaml")
    config = apply_overrides(load_config(args.config), args.overrides)
    if not config["paths"].get("wan_model"):
        config["paths"]["wan_model"] = os.environ.get("WAN_MODEL_DIR", "")
        if not config["paths"]["wan_model"]:
            raise ValueError("no Wan2.2 model directory: pass --set paths.wan_model=DIR or set WAN_MODEL_DIR")
    device = torch.device("cuda")
    checkpoint = Path(args.checkpoint)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    export_provenance = checkpoint_provenance(checkpoint)
    weight_hashes = checkpoint_weight_hashes(checkpoint)
    if state.get("decoder_config"):
        config["model"]["decoder"] = dict(state["decoder_config"])
    expert = state.get("expert", config["model"]["expert"])
    derived = resolve_derived(config)
    horizon_seconds = resolve_query_horizons(config, args.query_horizon_seconds)
    derived["horizon_seconds"] = horizon_seconds
    sampler, decode_provenance = resolve_sampler(config, args.decode_from, seed=args.seed)
    decode_provenance["config_path"] = Path(args.config).name
    print(f"[decode] source={decode_provenance['kind']} settings={asdict(sampler)}", flush=True)

    with Path(args.manifest).open() as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.shard_index::args.num_shards]
    output_dir = Path(args.output_dir) / "flows"
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [row for row in rows if args.overwrite or not (output_dir / f"{row['sample_id']}.pt").is_file()]
    skipped = len(rows) - len(pending)
    requires_masks = bool(config["model"].get("use_instance_channels", False)) or bool(
        config["model"].get("decoder", {}).get("oracle_instance_masks", False)
    )
    if requires_masks and not args.instance_masks:
        raise ValueError("this checkpoint requires --instance-masks")
    for row in pending:
        if not Path(row["first_frame_path"]).is_file():
            raise FileNotFoundError(row["first_frame_path"])
        if args.instance_masks:
            mask_path = Path(args.instance_masks) / f"{row['sample_id']}.pt"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
    if not pending:
        print(json.dumps({"done": 0, "skipped": skipped, "shard": args.shard_index,
                          "num_shards": args.num_shards}))
        return

    # Cache text on CPU and release T5 before DiT/VAE allocation on 48 GiB GPUs.
    text_rows = pending
    if args.text_manifest:
        if args.text_num_shards < 1 or not 0 <= args.text_shard_index < args.text_num_shards:
            raise ValueError("invalid text shard index/count")
        with Path(args.text_manifest).open() as handle:
            text_rows = list(csv.DictReader(handle))[args.text_shard_index::args.text_num_shards]
        absent = sorted({row["caption"] for row in pending} - {row["caption"] for row in text_rows})
        if absent:
            raise ValueError(f"--text-manifest does not contain the caption(s) to export: {absent[:2]}")
    prompt_cache = load_prompt_cache(text_rows, config, device=device, batch_size=args.text_batch_size)
    negative = prompt_cache[NEGATIVE_PROMPT]
    backend = load_backend(config, derived, checkpoint, expert, device)
    codec = build_codec(config["codec"], model_dir=config["paths"]["wan_model"],
                        device=device, latent_size=derived["latent_size"])
    decoder = build_decoder(config, derived).to(device)
    decoder.load_state_dict(
        torch.load(checkpoint / "decoder.pt", map_location="cpu", weights_only=True)
    )
    if checkpoint_weight_hashes(checkpoint) != weight_hashes:
        raise RuntimeError("checkpoint weights changed while loading for inference")
    decoder.eval()
    resolution = int(config["data"]["flow_resolution"])
    side = int(config["data"]["height"])
    generator = torch.Generator(device="cuda")
    prior_latent, prior_provenance = None, {"kind": "gaussian_noise"}
    if backend.objective == "direct":
        prior_latent, prior_provenance = load_direct_prior(
            codec, config, derived, device=device, prior_file=args.prior_latent,
        )
        print(f"[direct] prior {prior_provenance}", flush=True)

    effective_config = json.dumps(config, sort_keys=True, default=str)
    provenance = {
        "config_path": Path(args.config).name, "config_overrides": args.overrides,
        "effective_config_sha256": hashlib.sha256(effective_config.encode()).hexdigest(),
        "training_provenance": config.get("provenance", {}),
        "checkpoint_path": checkpoint.resolve().name, "trainer_state": state,
        "checkpoint_sha256": weight_hashes,
        "trainer_state_sha256": hashlib.sha256((checkpoint / "trainer_state.json").read_bytes()).hexdigest(),
        "expert": expert, "objective": backend.objective,
        "time_scale_seconds": config["data"].get("time_scale_seconds"),
        "training_horizon_seconds": [float(v) for v in config["data"]["horizon_seconds"]],
        "query_horizon_seconds": horizon_seconds,
        "query_horizon_override": bool(args.query_horizon_seconds),
        "inference_contract": {
            "query_horizon_seconds": horizon_seconds,
            "time_scale_seconds": config["data"].get("time_scale_seconds"),
            "data": {key: config["data"].get(key) for key in (
                "height", "width", "flow_resolution", "prompt_pad_length",
                "max_instances", "emit_instance_masks",
            )},
            "model": {key: config["model"].get(key) for key in (
                "architecture", "lora_rank", "lora_alpha", "boundary_sigma",
                "use_time_channel", "use_instance_channels", "decoder",
            )},
            "codec": dict(config["codec"]), "expert": expert,
            "objective": backend.objective, "sampler": asdict(sampler),
        },
        "manifest_path": Path(args.manifest).name,
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "decode": decode_provenance, "prior": prior_provenance,
        "prompt_pad_length": int(config["data"]["prompt_pad_length"]),
        "seed_algorithm": "(base_seed + sha256(sample_id)[:8]_big_endian) mod 2**63",
        "mask_directory": Path(args.instance_masks).resolve().name if args.instance_masks else None,
    }
    run_metadata = {"provenance": provenance, "effective_config": config,
                    "decode": asdict(sampler), "sample_ids": [row["sample_id"] for row in pending]}
    metadata_path = output_dir.parent / f"inference-shard-{args.shard_index:04d}.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2, default=str) + "\n")

    done = 0
    for row in pending:
        output = output_dir / f"{row['sample_id']}.pt"
        image_bgr = cv2.imread(row["first_frame_path"])
        if image_bgr is None:
            raise ValueError(f"cannot decode image: {row['first_frame_path']}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        frame_vae = letterbox_square(image, resolution)
        pixels = torch.from_numpy(frame_vae).permute(2, 0, 1)[None].float()
        pixels = pixels.div_(127.5).sub_(1.0).to(device)
        if hasattr(codec, "vae"):
            image_cond_latent = codec.vae.encode(pixels.unsqueeze(2)).latent_dist.mode()[0]
        else:
            image_cond_latent = torch.nn.functional.interpolate(
                pixels, size=(derived["latent_size"], derived["latent_size"]), mode="area",
            )[0][:, None]
        # Match the raw fp16 cache -> fp32 dataset path used in training.
        image_cond_latent = image_cond_latent.half().float()
        first_frame = (
            torch.from_numpy(letterbox_square(image, side)).permute(2, 0, 1)
            .float().div_(127.5).sub_(1.0).to(device)
        )
        instance_masks = instance_channels = None
        mask_path = None
        if args.instance_masks:
            mask_path = Path(args.instance_masks) / f"{row['sample_id']}.pt"
            mask_payload = torch.load(mask_path, map_location="cpu", weights_only=True)
            instance_masks = mask_payload["instance_masks"].float().to(device)
            instance_channels = mask_payload["instance_channels"].float().to(device)
            expected_masks = (int(config["data"]["emit_instance_masks"]), side, side)
            if tuple(instance_masks.shape) != expected_masks or tuple(instance_channels.shape) != (2, side, side):
                raise ValueError(f"{mask_path}: masks/channels do not match model grid and slot count")
        prompts = prompt_cache[row["caption"]]
        sample_seed = stable_sample_seed(args.seed, row["sample_id"])
        generator.manual_seed(sample_seed)
        latents = sample_flow_latents(
            {expert: backend}, image_cond_latent=image_cond_latent,
            prompt_embeds=prompts.to(device), negative_prompt_embeds=negative.to(device),
            latent_shape=derived["latent_shape"], horizon_seconds=horizon_seconds,
            config=sampler, generator=generator, instance_channels=instance_channels,
            init_latent=prior_latent if backend.objective == "direct" else None,
        )
        decoded = decode_samples(
            decoder, latents, image_cond_latent=image_cond_latent,
            first_frame=first_frame, horizon_seconds=horizon_seconds, backend=backend,
            prompt_embeds=prompts.to(device),
            instance_masks=instance_masks,
        )
        reduced = reduce_samples(
            decoded, reduce=sampler.reduce, shrinkage_px=sampler.shrinkage_px,
            eval_side=side, gain=sampler.gain,
        )
        payload = {
            "sample_id": row["sample_id"],
            "stage1_provenance": export_provenance,
            **{key: value.cpu().half() for key, value in reduced.items()},
            "horizon_seconds": horizon_seconds, "decode": asdict(sampler),
            "checkpoint": checkpoint.resolve().name, "geometry": letterbox_geometry(*image.shape[:2], side),
            "provenance": provenance, "sample_seed": sample_seed,
            "first_frame_path": Path(row["first_frame_path"]).name, "caption": row["caption"],
            "instance_mask_path": mask_path.name if mask_path else None,
        }
        if args.save_raw_flow:
            payload.update(raw_flow_payload(decoded, sampler, eval_side=side))
        payload.update(motion_activity_payload(reduced, instance_masks=instance_masks, eval_side=side))
        payload.update(bridge_fields(decoded, reduced, reduction=sampler.reduce,
                                     eval_side=side, instance_masks=instance_masks))
        temporary = output.with_suffix(f".tmp-{os.getpid()}")
        torch.save(payload, temporary)
        temporary.replace(output)
        done += 1
        if done % 20 == 0:
            print(f"  {done} done", flush=True)
    print(json.dumps({"done": done, "skipped": skipped,
                      "shard": args.shard_index, "num_shards": args.num_shards}))


if __name__ == "__main__":
    main()
