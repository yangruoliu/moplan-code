#!/usr/bin/env python
"""Evaluate Stage-1 WanFlowDiT on a cached split and write the evaluation JSON; also provides load_backend().

``flows/*.pt`` stores **normalised** flow (divided by the evaluation grid side, 160).  Besides the per-sample metrics
the JSON carries:

* ``min_ade`` -- best-of-K over the sampled flows (only meaningful with several samples per input);
* ``decoding`` -- the sampler settings, because the result depends on them (tune them on val, never on test).

``load_backend`` builds the WanFlowDiT backend (Wan2.2 expert + LoRA + new input / output layers) and loads
``backend_params.pt``; scripts/infer_stage1_physicsiq.py uses it for inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_jepa.config import SCENARIOS
from motion_jepa.flow_dit import WanFlowDiTBackend, build_decoder
from motion_jepa.flow_dit_config import apply_overrides, load_config, resolve_derived
from motion_jepa.flow_dit_data import FlowDiTDataset, pad_prompt
from motion_jepa.flow_dit_export import bridge_fields
from motion_jepa.flow_dit_sampler import (
    SamplerConfig,
    decode_samples,
    min_ade_over_samples,
    reduce_samples,
    sample_flow_latents,
)
from motion_jepa.metrics import MetricAccumulator, sample_metrics
from motion_jepa.stage2 import load_latent_statistics

# eval_dawn_physion pulls in DINOv3 processors that only exist in newer
# transformers; import it lazily so this evaluator runs under the pinned
# training overlay (its only use is flow visualisation/stats below).


_DENSE_STATS_WARNED: list[bool] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 WanFlowDiT")
    parser.add_argument("--config", default="", help="Stage-1 config (default: <checkpoint>/resolved_config.yaml)")
    parser.add_argument("--checkpoint", required=True, help="low/single checkpoint directory")
    parser.add_argument("--checkpoint-high", default="", help="optional high-noise expert")
    parser.add_argument("--split", default="test_core")
    parser.add_argument("--manifest", default="", help="explicit manifest, overrides --split")
    parser.add_argument("--ids", default="", help="filter the admitted full manifest after caption contract validation")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--visualizations", type=int, default=-1)
    parser.add_argument("--save-flows", action="store_true", default=True)
    parser.add_argument("--no-save-flows", dest="save_flows", action="store_false")
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=-1.0)
    parser.add_argument("--shrinkage-px", type=float, default=-1.0)
    parser.add_argument(
        "--decode-from",
        default="",
        help="JSON of a decoding sweep; applies its best_by_object_epe_px_micro settings "
             "(num_steps, num_samples, guidance_scale, shrinkage_px, flow_shift, gain). "
             "Without it the eval uses the config defaults.",
    )
    parser.add_argument("--reduce", default="", choices=("", "mean", "first"))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument(
        "--source", default="",
        help="name of an entry in data.additional_sources: evaluate that secondary source "
        "(its caches, prompt table, label source, horizons, instance masks) instead of the primary one",
    )
    return parser.parse_args()


def load_backend(
    config: dict, derived: dict, checkpoint: Path, expert: str, device: torch.device
) -> WanFlowDiTBackend:
    model_cfg = config["model"]
    mean, std = load_latent_statistics(config["paths"]["wan_model"])
    backend = WanFlowDiTBackend(
        model_dir=config["paths"]["wan_model"],
        expert=expert,
        latent_channels=derived["latent_channels"],
        lora_rank=int(model_cfg["lora_rank"]),
        lora_alpha=int(model_cfg["lora_alpha"]),
        latents_mean=mean,
        latents_std=std,
        device=device,
        flow_shift=float(config["train"]["flow_shift"]),
        boundary_sigma=float(model_cfg["boundary_sigma"]),
        gradient_checkpointing=False,
        use_time_channel=bool(model_cfg["use_time_channel"]),
        use_instance_channels=bool(model_cfg.get("use_instance_channels", False)),
        objective=str(config["train"].get("objective", "flow_matching")),
        time_scale=config["data"].get("time_scale_seconds"),
    )
    backend.load_state_dict(
        torch.load(checkpoint / "backend_params.pt", map_location="cpu", weights_only=True)
    )
    backend.transformer.eval()
    return backend


def sampler_from_args(config: dict, args: argparse.Namespace) -> SamplerConfig:
    sampler = SamplerConfig.from_config(config)
    if args.decode_from:
        chosen = json.loads(Path(args.decode_from).read_text())["best_by_object_epe_px_micro"]
        for field in ("num_steps", "num_samples", "guidance_scale", "shrinkage_px",
                      "flow_shift", "gain"):
            if field in chosen:
                setattr(sampler, field, type(getattr(sampler, field))(chosen[field]))
        print(f"[decode] from {args.decode_from}: " + ", ".join(
            f"{f}={getattr(sampler, f)}" for f in
            ("num_steps", "num_samples", "guidance_scale", "shrinkage_px")), flush=True)
    if args.num_samples:
        sampler.num_samples = args.num_samples
    if args.num_steps:
        sampler.num_steps = args.num_steps
    if args.guidance_scale >= 0:
        sampler.guidance_scale = args.guidance_scale
    if args.shrinkage_px >= 0:
        sampler.shrinkage_px = args.shrinkage_px
    if args.reduce:
        sampler.reduce = args.reduce
    return sampler


def read_validation_history(run_dir: Path) -> tuple[dict, bool]:
    """Best/initial validation numbers written by the trainer."""
    path = run_dir / "history.jsonl"
    if not path.is_file():
        return (
            {
                "initial_object_epe_norm_micro": None,
                "best_object_epe_norm_micro": None,
                "initial_total_loss": None,
                "best_total_loss": None,
                "note": f"{path} not found; validation_epe_improved cannot be verified",
            },
            False,
        )
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    initial = next((item for item in records if item["type"] == "validation_initial"), None)
    trained = [item for item in records if item["type"] == "validation"]
    if initial is None or not trained:
        return (
            {
                "initial_object_epe_norm_micro": None,
                "best_object_epe_norm_micro": None,
                "initial_total_loss": None,
                "best_total_loss": None,
                "note": "history.jsonl lacks validation_initial or validation records",
            },
            False,
        )
    initial_epe = float(initial["metric/object_epe_norm_micro"])
    best_epe = min(float(item["metric/object_epe_norm_micro"]) for item in trained)
    losses = [
        float(item["loss/total"])
        for item in trained
        if math.isfinite(float(item.get("loss/total", float("nan"))))
    ]
    # The initial validation runs before any optimisation, so it has no loss and
    # stores NaN. Comparing against NaN is always False, which made
    # validation_total_loss_improved unfalsifiable; use the first trained
    # validation as the reference instead.
    initial_loss = float(initial.get("loss/total", float("nan")))
    if not math.isfinite(initial_loss) and losses:
        initial_loss = float(losses[0])
    # The initial validation runs before any optimisation, so it has no loss to
    # report and stores NaN. Comparing against NaN is always False, which made
    # validation_total_loss_improved unfalsifiable; fall back to the first
    # trained validation as the reference instead.
    return (
        {
            "initial_object_epe_norm_micro": initial_epe,
            "best_object_epe_norm_micro": best_epe,
            "initial_total_loss": initial_loss,
            "best_total_loss": min(losses) if losses else None,
        },
        best_epe < initial_epe,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not args.config:
        args.config = str(Path(args.checkpoint) / "resolved_config.yaml")
    config = apply_overrides(load_config(args.config), args.overrides)
    derived = resolve_derived(config)
    paths, data_cfg = dict(config["paths"]), config["data"]
    label_source = config["provenance"]["label_source"]
    horizon_seconds = [float(v) for v in data_cfg["horizon_seconds"]]
    if args.source:
        sources = [src for src in data_cfg.get("additional_sources", []) if src.get("name") == args.source]
        if len(sources) != 1:
            raise ValueError(f"--source {args.source!r} not found in data.additional_sources")
        source = sources[0]
        paths.update({k: source[k] for k in ("manifest_dir", "flow_cache_dir", "target_cache_dir", "prompt_table")})
        for key in ("affine_cache_dir", "exclusions_manifest", "supervision_cache_dir", "condition_instance_cache_dir",
                    "condition_mask_source", "supervision_instance_cache_dir", "supervision_instance_mask_source",
                    "supervision_mover_ids_manifest"):
            paths[key] = source.get(key, "")
        for key in ("supervision_instance_masks_from_condition", "supervision_from_target"):
            paths[key] = source.get(key, False)
        paths["temporal_subsample_indices"] = source.get("temporal_subsample_indices")
        paths["caption_overrides"] = source.get("caption_overrides")
        label_source = source["label_source"]
        horizon_seconds = [float(v) for v in source.get("horizon_seconds", horizon_seconds)]
        # The decoder and exported anchor times must use the same source-specific
        # times as the targets. Source clips can share the frame count but span a
        # different time grid from the primary FOT data.
        derived["horizon_seconds"] = horizon_seconds
        print(f"[source] {args.source}: {paths['flow_cache_dir']} label_source={label_source}", flush=True)
    side = int(data_cfg["height"])
    delta_time = float(data_cfg["delta_time"])

    checkpoint = Path(args.checkpoint)
    # history.jsonl is written next to the checkpoints, i.e. inside the expert
    # directory. Walking one level further up silently loses it and reports
    # validation_epe_improved=False, which drags the whole gate to False.
    run_dir = checkpoint.parent
    output_root = (
        run_dir.parent if run_dir.name in ("low", "high", "single") else run_dir
    )
    output = Path(args.output_dir or output_root / f"eval_{args.split}")
    output.mkdir(parents=True, exist_ok=True)
    flow_dir, visual_dir = output / "flows", output / "visuals"
    if args.save_flows:
        flow_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest) if args.manifest else Path(paths["manifest_dir"]) / f"{args.split}.jsonl"
    dataset = FlowDiTDataset(
        manifest,
        flow_cache_dir=paths["flow_cache_dir"],
        target_cache_dir=paths["target_cache_dir"],
        prompt_table_path=paths["prompt_table"],
        width=int(data_cfg["width"]),
        height=side,
        limit=args.limit,
        strict=False,
        label_source=label_source,
        affine_cache_dir=paths.get("affine_cache_dir", ""),
        emit_instance_masks=int(config["data"].get("emit_instance_masks", 0)),
        horizon_seconds=horizon_seconds,
        exclusions=paths.get("exclusions_manifest") or None,
        supervision_mode=str(config["data"].get("supervision_mode", config["loss"].get("supervision_mode", "legacy"))),
        supervision_cache_dir=paths.get("supervision_cache_dir", ""),
        supervision_affine=bool(config["data"].get("supervision_affine", False)),
        prompt_pad_length=config["data"].get("prompt_pad_length"),
        # the reviewed-caption overrides are a train-split contract: the prompt table was
        # built with them, so a train manifest row without them has no embedding
        caption_overrides=(paths.get("caption_overrides")
                           if paths.get("caption_overrides") and manifest.resolve() == (Path(paths["manifest_dir"]) / "train.jsonl").resolve()
                           else None),  # the override table is SHA-bound to the full train manifest; subsets apply it themselves
        condition_instance_cache_dir=paths.get("condition_instance_cache_dir", ""),
        condition_mask_source=paths.get("condition_mask_source", ""),
        supervision_instance_masks_from_condition=bool(paths.get("supervision_instance_masks_from_condition", False)),
        supervision_instance_cache_dir=paths.get("supervision_instance_cache_dir", ""),
        supervision_instance_mask_source=paths.get("supervision_instance_mask_source", ""),
        temporal_subsample_indices=paths.get("temporal_subsample_indices"),
        supervision_from_target=bool(paths.get("supervision_from_target", False)),
        supervision_mover_ids_manifest=paths.get("supervision_mover_ids_manifest", ""),
    )
    objective = str(config["train"].get("objective", "flow_matching"))
    dataset.rows = dataset.available_rows()
    if args.ids:
        selected = Path(args.ids).read_text().split()
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("export IDs must be nonempty and unique")
        by_id = {row.sample_id: row for row in dataset.rows}
        missing = set(selected) - by_id.keys()
        if missing:
            raise ValueError(f"selected IDs absent from admitted source: {sorted(missing)[:3]}")
        dataset.rows = [by_id[sid] for sid in selected]
    if args.num_shards > 1:
        dataset.rows = [
            row for index, row in enumerate(dataset.rows)
            if index % args.num_shards == args.shard_index
        ]
    if not dataset.rows:
        raise RuntimeError(f"no cached samples for {manifest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    completion_marker = checkpoint / "checkpoint_complete.json"
    export_provenance = {
        "checkpoint": checkpoint.resolve().name,
        "checkpoint_state_sha256": hashlib.sha256((checkpoint / "trainer_state.json").read_bytes()).hexdigest(),
        "checkpoint_complete_sha256": hashlib.sha256(completion_marker.read_bytes()).hexdigest() if completion_marker.is_file() else None,
    }
    expert = state.get("expert", config["model"]["expert"])
    backends = {expert: load_backend(config, derived, checkpoint, expert, device)}
    if args.checkpoint_high:
        high = Path(args.checkpoint_high)
        backends["high"] = load_backend(config, derived, high, "high", device)
    reference = backends[expert]

    # The checkpoint is authoritative about its own architecture; the config
    # default would happily build the wrong decoder and fail on key names.
    recorded_config = state.get("decoder_config")
    if recorded_config:
        # Flags like causal_slots rename the module's parameters, so rebuilding
        # from the config default raises on unexpected keys rather than quietly
        # producing a different architecture.
        differing = {k: v for k, v in recorded_config.items()
                     if config["model"]["decoder"].get(k) != v}
        if differing:
            print(f"[decoder] checkpoint overrides {differing}", flush=True)
        config["model"]["decoder"] = dict(recorded_config)
    recorded = state.get("decoder_type")
    if recorded and recorded != config["model"]["decoder"].get("type"):
        print(
            f"[decoder] checkpoint was trained with {recorded!r}; overriding config "
            f"{config['model']['decoder'].get('type')!r}",
            flush=True,
        )
        config["model"]["decoder"]["type"] = recorded
    decoder = build_decoder(config, derived).to(device)
    decoder.load_state_dict(
        torch.load(checkpoint / "decoder.pt", map_location="cpu", weights_only=True)
    )
    decoder.eval()

    sampler = sampler_from_args(config, args)
    negative = pad_prompt(
        dataset.negative_prompt(), int(data_cfg["prompt_pad_length"])
    ).to(device)

    overall, zero_overall = MetricAccumulator(), MetricAccumulator()
    per_scenario: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    zero_per_scenario: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    per_sample: list[dict] = []
    label_sources: set[str] = set()
    multimodality: list[dict[str, float]] = []
    visual_budget = (
        int(config["evaluation"].get("visualizations", 24))
        if args.visualizations < 0
        else args.visualizations
    )
    generator = torch.Generator(device=device).manual_seed(sampler.seed)

    for index in range(len(dataset.rows)):
        item = dataset[index]
        label_sources.add(item["label_source"])
        image_cond = item["image_cond_latent"].to(device)
        # Dataset entries retain their variable token length; training collate
        # pads them to the checkpoint's fixed length. Both inference consumers
        # must receive that same positive-prompt tensor, just like the negative.
        prompt_embeds = pad_prompt(
            item["prompt_embeds"], int(data_cfg["prompt_pad_length"])
        ).to(device)
        latents = sample_flow_latents(
            backends,
            image_cond_latent=image_cond,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative,
            latent_shape=derived["latent_shape"],
            horizon_seconds=derived["horizon_seconds"],
            config=sampler,
            generator=generator,
            init_latent=(
                item["flow_latent"][:, 0].to(device)
                if objective == "direct"
                else None
            ),
            instance_channels=(
                item["instance_channels"].to(device)
                if "instance_channels" in item
                else None
            ),
        )
        decoded = decode_samples(
            decoder,
            latents,
            image_cond_latent=image_cond,
            first_frame=item["first_frame"].to(device),
            horizon_seconds=derived["horizon_seconds"],
            backend=reference,
            prompt_embeds=prompt_embeds,
            instance_masks=(
                item["instance_masks"].to(device)
                if "instance_masks" in item
                else None
            ),
        )
        reduced = reduce_samples(
            decoded,
            reduce=sampler.reduce,
            shrinkage_px=sampler.shrinkage_px,
            gain=sampler.gain,
            eval_side=float(side),
        )

        target_flow = item["flow"].to(device)
        target_visibility = item["visibility"].to(device)
        object_mask = item["object_mask"].to(device)
        common = {
            "target_flow": target_flow,
            "target_visibility": target_visibility,
            "object_mask": object_mask,
            "height": side,
            "width": int(data_cfg["width"]),
            "instance_id": item["instance_id"].to(device) if "instance_id" in item else None,
            "centers": item["centers"].to(device) if "centers" in item else None,
            "center_valid": item["center_valid"].to(device) if "center_valid" in item else None,
            "delta_time": delta_time,
        }
        metrics = sample_metrics(reduced["flow"], reduced["visibility_logits"], **common)
        zero = sample_metrics(
            torch.zeros_like(reduced["flow"]),
            torch.full_like(reduced["visibility_logits"], 10.0),
            **common,
        )
        scenario = item["scenario_name"]
        overall.update(metrics)
        per_scenario[scenario].update(metrics)
        zero_overall.update(zero)
        zero_per_scenario[scenario].update(zero)
        if sampler.num_samples > 1:
            multimodality.append(
                min_ade_over_samples(
                    decoded["flow"], target_flow, object_mask, target_visibility,
                    width=int(data_cfg["width"]), height=side,
                )
            )

        predicted_px = (reduced["flow"] * side).cpu()
        target_px = (target_flow * side).cpu()
        # Dense stats and flow images come from eval_dawn_physion, which pulls in
        # DINOv3 processors that the pinned training overlay's transformers lacks.
        # They are reporting extras, not the scored metrics, so a missing module
        # degrades the visuals instead of failing the whole evaluation.
        try:
            from eval_dawn_physion import flow_rgb, flow_stats
        except ImportError as error:
            if not _DENSE_STATS_WARNED:
                print(f"[eval] dense stats/visuals unavailable: {error!r}", flush=True)
                _DENSE_STATS_WARNED.append(True)
            flow_rgb = flow_stats = None

        if flow_stats is not None:
            dense = flow_stats(
                predicted_px[2], target_px[2], target_visibility[2].cpu(), object_mask.cpu(), 0.6
            )
            zero_dense = flow_stats(
                torch.zeros_like(predicted_px[2]), target_px[2],
                target_visibility[2].cpu(), object_mask.cpu(), 0.6,
            )
        else:
            dense = zero_dense = {}
        if args.save_flows:
            export = {
                "sample_id": item["sample_id"],
                "stage1_provenance": export_provenance,
                "flow": reduced["flow"].cpu().float(),
                "visibility_logits": reduced["visibility_logits"].cpu().float(),
                "object_mask_logits": reduced["object_mask_logits"].cpu().float(),
                "horizon_seconds": list(horizon_seconds),
                "flow_units": "normalized_by_eval_side",
                "eval_side": side,
                "prompt_pad_length": int(prompt_embeds.shape[0]),
            }
            # Retain legacy diagnostic predictions. The bridge always consumes
            # ungated source flow and the actual first-frame SAM support.
            for key in ("flow_ungated", "flow_task_object_gated", "task_object_slot_mask", "motion_mask"):
                if isinstance(reduced.get(key), torch.Tensor):
                    export[key] = reduced[key].cpu().half()
            for key in ("task_object_slot_logits", "task_object_slot_probability"):
                if isinstance(reduced.get(key), torch.Tensor):
                    export[key] = reduced[key].cpu().float()
            export.update(bridge_fields(decoded, reduced, reduction=sampler.reduce,
                                        eval_side=side, instance_masks=item.get("instance_masks")))
            torch.save(export, flow_dir / f"{item['sample_id']}.pt")
        if len(per_sample) < visual_budget and flow_rgb is not None:
            from PIL import Image

            Image.fromarray(flow_rgb(predicted_px[2].numpy(), 16.0)).save(
                visual_dir / f"{item['sample_id']}_pred_flow_0p6s.png"
            )
            Image.fromarray(flow_rgb(target_px[2].numpy(), 16.0)).save(
                visual_dir / f"{item['sample_id']}_gt_flow_0p6s.png"
            )
        per_sample.append(
            {
                "sample_id": item["sample_id"],
                "scenario": scenario,
                "dense_0p6s": dense,
                "zero_0p6s": zero_dense,
                "full_trajectory": metrics,
                "full_trajectory_zero": zero,
            }
        )
        if len(per_sample) % 25 == 0 or len(per_sample) == len(dataset.rows):
            print(
                f"[{len(per_sample)}/{len(dataset.rows)}] {item['sample_id']} "
                + (f"EPE@0.6={dense['epe_px']:.3f}px" if dense else "(dense stats off)"),
                flush=True,
            )

    model_summary = overall.summary()
    zero_summary = zero_overall.summary()
    validation, validation_improved = read_validation_history(run_dir)

    required_metrics = (
        "object_epe_norm_micro",
        "object_epe_norm",
        "ade_norm",
        "fde_norm",
        "warped_mask_iou",
    )
    feasibility = {
        "beats_zero_object_epe": model_summary.get("object_epe_norm_micro", float("inf"))
        < zero_summary.get("object_epe_norm_micro", float("inf")),
        "beats_zero_ade": model_summary.get("ade_norm", float("inf"))
        < zero_summary.get("ade_norm", float("inf")),
        "validation_epe_improved": validation_improved,
        "validation_total_loss_improved": bool(
            validation.get("best_total_loss") is not None
            and validation.get("initial_total_loss") is not None
            and math.isfinite(validation["initial_total_loss"])
            and validation["best_total_loss"] < validation["initial_total_loss"]
        ),
        "all_scenarios_present": all(
            scenario in per_scenario
            and per_scenario[scenario].summary().get("num_samples", 0) > 0
            for scenario in SCENARIOS
        ),
        "metrics_finite": all(
            key in model_summary and math.isfinite(model_summary[key])
            for key in required_metrics
        ),
    }
    feasibility["passed"] = all(feasibility.values())

    result = {
        # File names plus the checkpoint's sha256, no machine paths (as the per-export stage1_provenance above).
        "checkpoint": checkpoint.resolve().name,
        "checkpoint_state_sha256": export_provenance["checkpoint_state_sha256"],
        "checkpoint_step": int(state["step"]),
        "split": args.split,
        "manifest": manifest.name,
        "text_condition": "correct",
        "label_sources": sorted(label_sources),
        "protocol": "native Physion HDF5 flow; grounded per-sample captions; horizons 0.2--1.4 s",
        "model": model_summary,
        "zero_flow_baseline": zero_summary,
        "relative_improvement": {
            key: (zero_summary[key] - model_summary[key]) / max(abs(zero_summary[key]), 1e-12)
            for key in ("object_epe_norm_micro", "object_epe_norm", "ade_norm", "fde_norm")
            if key in model_summary and key in zero_summary
        },
        "validation_improvement": validation,
        "per_scenario": {key: value.summary() for key, value in sorted(per_scenario.items())},
        "zero_per_scenario": {
            key: value.summary() for key, value in sorted(zero_per_scenario.items())
        },
        "feasibility_gate": feasibility,
        "decoding": {
            "num_steps": sampler.num_steps,
            "guidance_scale": sampler.guidance_scale,
            "num_samples": sampler.num_samples,
            "shrinkage_px": sampler.shrinkage_px,
            "reduce": sampler.reduce,
            "flow_shift": sampler.flow_shift,
            "experts": sorted(backends),
        },
        "samples": len(per_sample),
    }
    if multimodality:
        keys = multimodality[0].keys()
        result["multimodality"] = {
            key: sum(item[key] for item in multimodality) / len(multimodality) for key in keys
        }
    suffix = "" if args.num_shards <= 1 else f"-shard{args.shard_index:03d}of{args.num_shards:03d}"
    (output / f"metrics{suffix}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / f"per_sample{suffix}.jsonl").open("w") as handle:
        for row in per_sample:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_payload = {**result, "per_sample": per_sample}
    (output / f"summary{suffix}.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "per_scenario"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
