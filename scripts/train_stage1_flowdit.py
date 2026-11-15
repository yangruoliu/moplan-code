#!/usr/bin/env python
"""Train the Stage-1 motion planner (WanFlowDiT).

Trains LoRA adapters on one Wan2.2 expert (``--expert low|high|single``) plus
the new input/output projections and the slot decoder. ``train.objective``
selects the objective:

* ``direct`` (the released recipe): one deterministic forward pass from a
  constant zero-motion prior latent, supervised on the decoded 160x160 motion
  field and its object-level terms;
* ``flow_matching``: latent flow matching with the same decoded-field terms,
  ramped in over ``loss.aux_warmup_steps`` and down-weighted at high sigma.

Validation runs the real sampler rather than a teacher-forced shortcut, so
``history.jsonl`` is comparable with the final evaluation. A
``validation_initial`` record (step 0, untrained) is always written first.

Filesystem locations in the config may use the placeholders ``${MOPLAN_DATA}``,
``${WAN_MODEL_DIR}`` and ``${MOPLAN_WORK_DIR}``; they are filled from
``--data-root``, ``--wan-model-dir`` and ``--work-dir`` or from the environment
variables of the same names (see ``resolve_placeholders``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import time
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from motion_jepa.flow_dit import WanFlowDiTBackend, build_decoder
from motion_jepa.flow_dit_checkpoint import initialize_from_checkpoint
from motion_jepa.caption_overrides import (
    caption_override_data_contract,
    validate_caption_override_resume,
)
from motion_jepa.flow_dit_resume import (
    PROTOCOL as EXACT_RESUME_PROTOCOL,
    DataCursor,
    complete_accumulation_samples,
    capture_rng,
    commit_checkpoint,
    log_positions,
    log_snapshots,
    rank_rng_state,
    restore_rng,
    rewind_logs,
    runtime_contract,
    staging_directory,
    verify_checkpoint,
)
from motion_jepa.flow_dit_diagnostics import motion_sufficient_statistics, summary_statistics
from motion_jepa.flow_dit_config import (
    apply_overrides,
    load_config,
    recover_validation_state,
    resolve_derived,
    validation_selection,
    validate_selection_resume,
)
from motion_jepa.flow_dit_bridge import BRIDGE_LABEL_SOURCE
from motion_jepa.flow_dit_data import (
    FixedMixtureSampler,
    FlowDiTDataset,
    JointDataset,
    collate,
    pad_prompt,
)
from motion_jepa.flow_dit_losses import cosine_ramp, flow_space_losses, sigma_weight
from motion_jepa.flow_dit_coverage import ConsumedCoverage, SourceCycleSampler
from motion_jepa.flow_dit_sampler import (
    SamplerConfig,
    decode_samples,
    reduce_samples,
    sample_flow_latents,
)
from motion_jepa.metrics import MetricAccumulator, sample_metrics
from motion_jepa.stage2 import load_latent_statistics
from scripts.stage1_paths import (  # noqa: F401  (re-exported for data tools and tests)
    PLACEHOLDER_SOURCES,
    add_placeholder_arguments,
    placeholder_values,
    resolve_placeholders,
)


def load_training_config(path, overrides: list[str], values: dict[str, str]) -> dict:
    """Load a config, apply ``--set`` overrides, then fill path placeholders."""
    return resolve_placeholders(apply_overrides(load_config(path), overrides), values)


ACTIVITY_HEAD_PARAMETER_NAMES = frozenset({
    "slot_activity_head.weight", "slot_activity_head.bias",
    "pixel_activity_head.weight", "pixel_activity_head.bias",
})


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _state_tensor_hash(state: dict[str, torch.Tensor]) -> str:
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        result.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def validate_activity_heads_only_resume(state: dict, requested: bool) -> None:
    recorded = state.get("activity_heads_only", False)
    if not isinstance(recorded, bool) or recorded != requested:
        raise ValueError("resume activity_heads_only differs from the saved optimizer contract")


def validate_mask_heads_only_resume(state: dict, requested: bool) -> None:
    recorded = state.get("mask_heads_only", False)
    if not isinstance(recorded, bool) or recorded != requested:
        raise ValueError("resume mask_heads_only differs from the saved optimizer contract")


def validate_mask_head_parameter_names(
    names: frozenset[str], *, semantic_identity: bool
) -> None:
    """Require every independently supervised head without assuming its layout."""
    required_prefixes = [
        "dense_motion_mask_fuse.",
        "dense_motion_mask_up.",
        "dense_motion_mask_head.",
        "task_object_slot_head.",
        "task_object_identity_head.",
    ]
    if semantic_identity:
        required_prefixes.extend(
            ("task_identity_visual_project.", "task_identity_prompt_project.")
        )
    missing = [
        prefix for prefix in required_prefixes
        if not any(name.startswith(prefix) for name in names)
    ]
    if missing:
        raise ValueError(
            "mask_heads_only requires dense, identity and temporal task-object "
            f"heads; missing prefixes: {missing}"
        )


def validate_activity_heads_only_runtime(*, enabled: bool, num_processes: int,
                                       device: torch.device, checkpoint: str) -> None:
    if enabled and (num_processes != 1 or device.type != "cuda"):
        raise ValueError("activity_heads_only supports one GPU/process only")
    if enabled and not checkpoint:
        raise ValueError("activity_heads_only requires --init-from or a matching --resume-from")


def validate_mask_heads_only_runtime(*, enabled: bool, num_processes: int,
                                     device: torch.device, checkpoint: str) -> None:
    if enabled and (num_processes != 1 or device.type != "cuda"):
        raise ValueError("mask_heads_only supports one GPU/process only")
    if enabled and not checkpoint:
        raise ValueError("mask_heads_only requires --init-from or a matching --resume-from")


class ActivityHeadsOnlyControl:
    """Freeze the full old model and retain a byte-identical backend checkpoint.

    Strict initialization runs BEFORE freezing, so its expected backend keyset
    remains the complete historical trainable state. The optimizer then sees
    only the four newly supervised tensors. No frozen weight is dropped from
    inference checkpoints merely because requires_grad has changed.
    """

    def __init__(self, backend, decoder, checkpoint: Path, *, config: dict, expert: str,
                 output_dir: Path, resume: bool = False,
                 trainable_parameter_names: frozenset[str] = ACTIVITY_HEAD_PARAMETER_NAMES,
                 mode_key: str = "activity_heads_only") -> None:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if resume:
            if mode_key == "activity_heads_only":
                validate_activity_heads_only_resume(state, True)
            elif mode_key == "mask_heads_only":
                validate_mask_heads_only_resume(state, True)
            else:
                raise ValueError(f"unknown frozen-head mode: {mode_key}")
        self.trainable_parameter_names = frozenset(trainable_parameter_names)
        self.mode_key = mode_key
        self.initialization = initialize_from_checkpoint(
            backend, decoder, checkpoint, config=config, expert=expert,
        )
        self.backend_reference = torch.load(checkpoint / "backend_params.pt", map_location="cpu", weights_only=True)
        self.decoder_reference = {name: value.detach().cpu().clone() for name, value in decoder.state_dict().items()
                                  if name not in self.trainable_parameter_names}
        names = set(dict(decoder.named_parameters()))
        if not self.trainable_parameter_names or not self.trainable_parameter_names <= names:
            missing = sorted(self.trainable_parameter_names - names)
            raise ValueError(f"{mode_key} is missing trainable head parameters: {missing[:5]}")
        self.contract = {
            "schema_version": 2 if mode_key == "mask_heads_only" else 1,
            "mode": mode_key,
            "source_checkpoint": checkpoint.resolve().name,             # name only; source_backend_sha256 identifies it
            "source_backend_sha256": _sha256(checkpoint / "backend_params.pt"),
            "frozen_decoder_tensor_sha256": _state_tensor_hash(self.decoder_reference),
            "backend_parameter_names": sorted(self.backend_reference),
            "trainable_decoder_parameter_names": sorted(self.trainable_parameter_names),
            "validation_reference": None,
        }
        if resume:
            saved = state.get(f"{mode_key}_contract", {})
            for key in ("schema_version", "source_backend_sha256", "frozen_decoder_tensor_sha256",
                        "backend_parameter_names", "trainable_decoder_parameter_names"):
                if saved.get(key) != self.contract[key]:
                    raise ValueError(f"resume activity_heads_only frozen contract mismatch: {key}")
            if not saved.get("validation_reference"):
                raise ValueError("resume activity_heads_only checkpoint lacks frozen validation baseline")
            self.contract = dict(saved)
            self.initialization = state.get("initialization", self.initialization)
        self.backend_file = output_dir / "frozen_backend_params.pt"
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.backend_file.exists():
            if _sha256(self.backend_file) != self.contract["source_backend_sha256"]:
                raise ValueError("run's frozen backend bytes differ from the strict initialization source")
        else:
            shutil.copyfile(checkpoint / "backend_params.pt", self.backend_file)
        if _sha256(self.backend_file) != self.contract["source_backend_sha256"]:
            raise ValueError("frozen backend copy failed SHA256 verification")
        backend.transformer.requires_grad_(False)
        decoder.requires_grad_(False)
        for name, parameter in decoder.named_parameters():
            if name in self.trainable_parameter_names:
                parameter.requires_grad_(True)
        backend.transformer.eval()
        decoder.eval()
        self.assert_frozen(backend.transformer, decoder)

    def assert_frozen(self, transformer, decoder) -> None:
        named = dict(transformer.named_parameters())
        if any(value.requires_grad for value in named.values()):
            raise ValueError(f"{self.mode_key} unexpectedly has trainable backend parameters")
        active = {name for name, value in decoder.named_parameters() if value.requires_grad}
        if active != self.trainable_parameter_names:
            raise ValueError(f"{self.mode_key} optimizer contract changed")
        for name, reference in self.backend_reference.items():
            actual = named[name].detach().cpu()
            if not torch.equal(actual, reference.to(actual.dtype)):
                raise ValueError(f"frozen backend parameter changed: {name}")
        current = decoder.state_dict()
        if set(current) - self.trainable_parameter_names != set(self.decoder_reference):
            raise ValueError("frozen decoder state keys changed")
        for name, reference in self.decoder_reference.items():
            if not torch.equal(current[name].detach().cpu(), reference):
                raise ValueError(f"frozen decoder state changed: {name}")

    def validate_ungated(self, summary: dict) -> None:
        current = {key: value for key, value in summary.items() if "ungated_motion/" in key}
        if not current:
            raise ValueError("activity_heads_only requires full-support ungated validation metrics")
        baseline = self.contract["validation_reference"]
        if baseline is None:
            self.contract["validation_reference"] = current
        elif current != baseline:
            changed = sorted(key for key in set(current) | set(baseline) if current.get(key) != baseline.get(key))
            raise ValueError(f"activity_heads_only ungated validation changed from initialization: {changed[:5]}")

    def save_backend(self, path: Path, *, transformer, decoder) -> None:
        self.assert_frozen(transformer, decoder)
        if _sha256(self.backend_file) != self.contract["source_backend_sha256"]:
            raise ValueError("frozen source backend bytes changed during training")
        shutil.copyfile(self.backend_file, path)
        if _sha256(path) != self.contract["source_backend_sha256"]:
            raise ValueError("saved frozen backend is not byte-identical to initialization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage-1 motion planner (WanFlowDiT)")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "stage1" / "train.yaml"))
    add_placeholder_arguments(parser)
    parser.add_argument("--expert", default="", choices=("", "low", "high", "single"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--stop-at-step",
        type=int,
        default=0,
        help=(
            "End this invocation at an absolute optimizer step (bounded jobs) while "
            "keeping --max-steps as the fixed LR-schedule horizon"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument(
        "--val-split",
        default="val",
        help="usually 'val'; set to 'train' for an overfit check so validation "
        "scores the same clip that was trained on",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume-from", default="")
    checkpoint_group.add_argument(
        "--init-from", default="",
        help="Load complete model weights for a new run; reset optimizer, step and schedule",
    )
    parser.add_argument("--keep-last", type=int, default=0)
    parser.add_argument(
        "--allow-scenario-holes",
        action="store_true",
        help="proceed despite a scenario-level cache blackout; mechanics-only "
        "runs such as smoke tests, never a reported model",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        help="section.key=value override (YAML-parsed; the key must exist)")
    return parser.parse_args()


def build_dataset(
    config: dict,
    split: str,
    *,
    limit: int,
    strict: bool,
    source: dict | None = None,
    allow_scenario_holes: bool = False,
) -> FlowDiTDataset:
    """Build a split and keep only cached rows.

    Availability is filtered *before* ``limit`` is applied: with ``--limit 1``
    the other order can silently yield an empty dataset if the first manifest
    row happens not to be cached.

    ``source`` overrides the path/provenance block -- the joint-training
    secondary dataset (PhyCo) lives in different caches with a different label
    source and prompt table but is otherwise shape-identical.
    """
    paths = dict(config["paths"])
    label_source = config["provenance"]["label_source"]
    horizon_seconds = [float(v) for v in config["data"]["horizon_seconds"]]
    if source is not None:
        paths.update({k: source[k] for k in (
            "manifest_dir", "flow_cache_dir", "target_cache_dir", "prompt_table"
        )})
        paths["affine_cache_dir"] = source.get("affine_cache_dir", "")
        paths["exclusions_manifest"] = source.get("exclusions_manifest", "")
        paths["supervision_cache_dir"] = source.get("supervision_cache_dir", "")
        paths["condition_instance_cache_dir"] = source.get("condition_instance_cache_dir", "")
        paths["condition_mask_source"] = source.get("condition_mask_source", "")
        paths["supervision_instance_masks_from_condition"] = source.get(
            "supervision_instance_masks_from_condition", False
        )
        paths["supervision_instance_cache_dir"] = source.get(
            "supervision_instance_cache_dir", ""
        )
        paths["supervision_instance_mask_source"] = source.get(
            "supervision_instance_mask_source", ""
        )
        paths["supervision_from_target"] = source.get("supervision_from_target", False)
        paths["supervision_mover_ids_manifest"] = source.get(
            "supervision_mover_ids_manifest", ""
        )
        paths["temporal_subsample_indices"] = source.get("temporal_subsample_indices")
        # A secondary source never inherits the primary source's overrides.
        paths["caption_overrides"] = source.get("caption_overrides")
        label_source = source["label_source"]
        horizon_seconds = [float(v) for v in source["horizon_seconds"]]
    dataset = FlowDiTDataset(
        Path(paths["manifest_dir"]) / f"{split}.jsonl",
        flow_cache_dir=paths["flow_cache_dir"],
        target_cache_dir=paths["target_cache_dir"],
        prompt_table_path=paths["prompt_table"],
        prompt_table_mmap=bool(paths.get("prompt_table_mmap", False)),
        width=int(config["data"]["width"]),
        height=int(config["data"]["height"]),
        limit=0,
        strict=strict,
        label_source=label_source,
        affine_cache_dir=paths.get("affine_cache_dir", ""),
        emit_instance_masks=int(config["data"].get("emit_instance_masks", 0)),
        horizon_seconds=horizon_seconds,
        exclusions=paths.get("exclusions_manifest") or None,
        supervision_mode=str(config["data"].get(
            "supervision_mode", config["loss"].get("supervision_mode", "legacy")
        )),
        supervision_cache_dir=paths.get("supervision_cache_dir", ""),
        supervision_affine=bool(config["data"].get("supervision_affine", False)),
        supervision_static_threshold_px=float(config["loss"].get("flow_static_threshold_px", 0.5)),
        supervision_moving_threshold_px=float(config["loss"].get("flow_moving_threshold_px", 1.0)),
        caption_overrides=paths.get("caption_overrides") if split == "train" else None,
        prompt_pad_length=config["data"].get("prompt_pad_length"),
        condition_instance_cache_dir=paths.get("condition_instance_cache_dir", ""),
        condition_mask_source=paths.get("condition_mask_source", ""),
        supervision_instance_masks_from_condition=bool(
            paths.get("supervision_instance_masks_from_condition", False)
        ),
        supervision_instance_cache_dir=paths.get(
            "supervision_instance_cache_dir", ""
        ),
        supervision_instance_mask_source=paths.get(
            "supervision_instance_mask_source", ""
        ),
        temporal_subsample_indices=paths.get("temporal_subsample_indices"),
        supervision_from_target=bool(paths.get("supervision_from_target", False)),
        supervision_mover_ids_manifest=paths.get(
            "supervision_mover_ids_manifest", ""
        ),
    )
    if dataset.exclusion_counts:
        applied = ", ".join(
            f"{reason}={count}" for reason, count in sorted(dataset.exclusion_counts.items())
        )
        print(f"[{split}] excluded rows via manifest: {applied}")
    populations = Counter(row.scenario for row in dataset.rows)
    dataset.rows = dataset.available_rows()
    retained = Counter(row.scenario for row in dataset.rows)
    short = {
        scenario: f"{retained.get(scenario, 0)}/{population}"
        for scenario, population in sorted(populations.items())
        if retained.get(scenario, 0) < population
    }
    if short:
        print(f"[{split}] per-scenario availability below manifest: {short}")
    # A scenario-level cache blackout (e.g. every ball_drop latent missing
    # because its source videos were corrupt) slips straight through the
    # per-row availability filter and trains a model that has never seen the
    # scenario. Refuse that silently-skewed run unless explicitly overridden.
    holes = {
        scenario: f"{retained.get(scenario, 0)}/{population}"
        for scenario, population in sorted(populations.items())
        if population >= 20 and retained.get(scenario, 0) < 0.5 * population
    }
    if holes and not allow_scenario_holes:
        raise RuntimeError(
            f"[{split}] scenario-level cache holes (retention < 50%): {holes}; "
            "fix the caches or pass --allow-scenario-holes for a mechanics-only run"
        )
    if limit:
        dataset.rows = dataset.rows[:limit]
    return dataset


def rotate_checkpoints(output_dir: Path, keep_last: int) -> None:
    """Delete all but the newest ``keep_last`` checkpoints.

    153M LoRA parameters plus optimiser state is ~1.9GB per checkpoint, so
    unbounded retention can fill a disk.
    ``checkpoint-best`` is never rotated.
    """
    if keep_last <= 0:
        return
    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.name != "checkpoint-best"),
        key=lambda path: path.name,
    )
    for path in checkpoints[:-keep_last]:
        shutil.rmtree(path, ignore_errors=True)
        print(f"[checkpoint] rotated out {path.name}", flush=True)


def write_resolved_config(output_dir: Path, config: dict, *, expert: str,
                          max_steps: int, stop_at_step: int) -> None:
    """Publish run metadata only after resume/data validation has succeeded."""
    content = yaml.safe_dump({**config, "selected_expert": expert,
                              "resolved_max_steps": max_steps,
                              "resolved_stop_at_step": stop_at_step}, sort_keys=False)
    fd, name = tempfile.mkstemp(prefix=".resolved_config-", suffix=".yaml", dir=output_dir)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output_dir / "resolved_config.yaml")
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def run_validation(
    *,
    backend: WanFlowDiTBackend,
    decoder: torch.nn.Module,
    dataset: FlowDiTDataset,
    config: dict,
    derived: dict,
    sampler: SamplerConfig,
    device: torch.device,
    limit: int,
    seed: int,
    shard: tuple[int, int] = (0, 1),
    export_dir: Path | None = None,
    calibration_rows_by_id: dict | None = None,
    release_hold_recipes: dict | None = None,
    calibrated_threshold: float | None = None,
    calibration_only: bool = False,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Sample, decode and score -- the same path the final evaluation uses.

    Sharded across ranks.  Running it on rank 0 only leaves the other ranks
    blocked in a collective for the whole validation, which trips the 600s NCCL
    watchdog and aborts the job; it also wastes every other GPU.
    """
    was_training = decoder.training
    backend_was_training = backend.transformer.training
    decoder.eval()
    backend.transformer.eval()

    accumulator = MetricAccumulator()
    zero_accumulator = MetricAccumulator()
    calibration_records, event_observations = [], []
    if calibration_only and calibration_rows_by_id is None:
        raise ValueError("calibration-only validation needs frozen calibration membership")
    generator = torch.Generator(device=device).manual_seed(seed)
    side = int(config["data"]["height"])
    delta_time = float(config["data"]["delta_time"])
    objective = getattr(backend, "objective", "flow_matching")
    rows = dataset.rows[:limit] if limit else dataset.rows
    rank, world = shard
    export_count = min(len(rows), int(config.get("evaluation", {}).get("visualizations", 0)))
    # Fixed evenly spaced rows, independent of validation scores.
    export_indices = {int(i * len(rows) / export_count) for i in range(export_count)}

    for index in range(rank, len(rows), world):
        item = dataset[index]
        validation_backends = {backend.expert: backend}
        if getattr(backend, "refine_partner", None) is not None:
            # The low expert is only meaningful on the combined 2-step path;
            # validating it solo would score it at sigmas it never serves.
            validation_backends["high"] = backend.refine_partner
        latents = sample_flow_latents(
            validation_backends,
            image_cond_latent=item["image_cond_latent"].to(device),
            prompt_embeds=item["prompt_embeds"].to(device),
            negative_prompt_embeds=None,  # guidance is disabled during training validation
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
            image_cond_latent=item["image_cond_latent"].to(device),
            first_frame=item["first_frame"].to(device),
            horizon_seconds=derived["horizon_seconds"],
            backend=backend,
            prompt_embeds=item["prompt_embeds"].to(device),
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
        raw_reduced = reduce_samples(
            decoded, reduce=sampler.reduce, shrinkage_px=0.0, gain=1.0,
            eval_side=float(side),
        )
        sample_id = str(getattr(rows[index], "sample_id", index))
        if calibration_rows_by_id is not None:
            from motion_jepa.flow_dit_gate_calibration import identity_calibration_record
            if sample_id not in calibration_rows_by_id:
                raise ValueError("calibration inference contains a row outside its frozen manifest")
            membership = calibration_rows_by_id[sample_id]
            calibration_records.append(identity_calibration_record(
                sample_id, membership["domain"], membership["scene_id"],
                raw_reduced, item, config.get("loss", {}),
            ))
        if calibration_only:
            continue
        calibrated_flow = None
        if calibrated_threshold is not None:
            from motion_jepa.flow_dit_gate_calibration import calibrated_identity_readout
            calibrated_flow, calibrated_mask = calibrated_identity_readout(
                raw_reduced, item, calibrated_threshold,
            )
            raw_reduced["flow_calibrated_task_object_gated"] = calibrated_flow
            raw_reduced["calibrated_task_object_selection_mask"] = calibrated_mask
        if release_hold_recipes is not None and sample_id in release_hold_recipes:
            from motion_jepa.flow_dit_event_diagnostics import release_hold_observation
            event_observations.append(release_hold_observation(
                sample_id, raw_reduced, item, derived["horizon_seconds"],
                release_hold_recipes[sample_id], calibrated_flow=calibrated_flow,
            ))
        common = {
            "target_flow": item["flow"].to(device),
            "target_visibility": item["visibility"].to(device),
            "object_mask": item["object_mask"].to(device),
            "height": side,
            "width": side,
            "instance_id": item.get("instance_id", torch.empty(0)).to(device)
            if "instance_id" in item
            else None,
            "centers": item.get("centers").to(device) if "centers" in item else None,
            "center_valid": item.get("center_valid").to(device) if "center_valid" in item else None,
            "delta_time": delta_time,
        }
        measured = sample_metrics(reduced["flow"], reduced["visibility_logits"], **common)
        zero_measured = sample_metrics(
            torch.zeros_like(reduced["flow"]),
            torch.full_like(reduced["visibility_logits"], 10.0),
            **common,
        )
        if item.get("label_source") == BRIDGE_LABEL_SOURCE:
            # SAM is a condition, not complete object/occupancy truth, and
            # invalid teacher flow is unknown. Do not score its zero fill.
            measured.pop("warped_mask_iou", None)
            zero_measured.pop("warped_mask_iou", None)
            accumulator.update({"_bridge_occupancy_unknown": 1.0})
        accumulator.update(measured)
        zero_accumulator.update(zero_measured)
        if "flow_valid" in item:
            loss_config = config.get("loss", {})
            if any(float(loss_config.get(key, 0.0)) > 0 for key in (
                "task_object_affine_displacement",
                "task_object_affine_direction",
                "task_object_affine_parallel_magnitude",
                "task_object_affine_trajectory",
            )):
                from motion_jepa.flow_dit_losses import _task_object_affine_motion_losses

                slot_params = decoded.get("slot_params")
                if not isinstance(slot_params, torch.Tensor):
                    raise ValueError("task-object affine validation requires slot_params")
                reduced_slot_params = (
                    slot_params[0] if sampler.reduce == "first" else slot_params.mean(0)
                )
                affine_batch = {
                    key: value.to(device).unsqueeze(0)
                    for key, value in item.items()
                    if torch.is_tensor(value)
                }
                affine_batch["horizon_seconds"] = torch.tensor(
                    derived["horizon_seconds"], device=device, dtype=torch.float32
                ).unsqueeze(0)
                affine_terms = _task_object_affine_motion_losses(
                    reduced_slot_params.unsqueeze(0), affine_batch,
                    item["flow"].to(device).unsqueeze(0),
                    item["flow_valid"].to(device).unsqueeze(0),
                    loss_config, None, width=side, height=side,
                )
                support_map = {
                    "task_object_affine_displacement": "task_object_affine_support",
                    "task_object_affine_direction": "task_object_affine_moving_support",
                    "task_object_affine_parallel_magnitude": "task_object_affine_moving_support",
                    "task_object_affine_trajectory": "task_object_affine_trajectory_support",
                }
                for key, support_key in support_map.items():
                    if float(affine_terms[support_key]) > 0:
                        accumulator.update({
                            f"_task_affine_stat/{key}": float(affine_terms[key])
                        })
                accumulator.update({
                    f"_task_affine_stat/{key}": float(affine_terms[key])
                    for key in (
                        "task_object_affine_support",
                        "task_object_affine_moving_support",
                        "task_object_affine_trajectory_support",
                    )
                })
            if any(float(loss_config.get(key, 0.0)) > 0 for key in (
                "task_object_flow_displacement",
                "task_object_flow_direction",
                "task_object_flow_parallel_magnitude",
                "task_object_flow_trajectory",
            )):
                from motion_jepa.flow_dit_losses import _task_object_flow_motion_losses

                task_flow_batch = {
                    key: value.to(device).unsqueeze(0)
                    for key, value in item.items()
                    if torch.is_tensor(value)
                }
                task_flow_batch["horizon_seconds"] = torch.tensor(
                    derived["horizon_seconds"], device=device, dtype=torch.float32
                ).unsqueeze(0)
                task_flow_terms = _task_object_flow_motion_losses(
                    raw_reduced["flow"].unsqueeze(0), task_flow_batch,
                    item["flow"].to(device).unsqueeze(0),
                    item["flow_valid"].to(device).unsqueeze(0),
                    loss_config, None, width=side, height=side,
                )
                support_map = {
                    "task_object_flow_displacement": "task_object_flow_support",
                    "task_object_flow_direction": "task_object_flow_moving_support",
                    "task_object_flow_parallel_magnitude": "task_object_flow_moving_support",
                    "task_object_flow_trajectory": "task_object_flow_trajectory_support",
                }
                for key, support_key in support_map.items():
                    if float(task_flow_terms[support_key]) > 0:
                        accumulator.update({
                            f"_task_flow_stat/{key}": float(task_flow_terms[key])
                        })
                accumulator.update({
                    f"_task_flow_stat/{key}": float(task_flow_terms[key])
                    for key in (
                        "task_object_flow_support",
                        "task_object_flow_moving_support",
                        "task_object_flow_trajectory_support",
                    )
                })
            if export_dir is not None and index in export_indices:
                export_dir.mkdir(parents=True, exist_ok=True)
                sample_id = str(getattr(rows[index], "sample_id", index))
                payload = {
                    "sample_id": sample_id,
                    "caption": getattr(dataset, "conditioning_captions", {}).get(
                        sample_id, getattr(rows[index], "caption", "")),
                    "original_caption": getattr(rows[index], "caption", ""),
                    "scenario": getattr(rows[index], "scenario", ""),
                    "horizon_seconds": list(derived["horizon_seconds"]),
                    "flow_units": "normalized_by_width_height",
                    "coordinates": "first_frame_letterbox_grid",
                    "dense_motion_mask_target_mode": loss_config.get(
                        "dense_motion_mask_target_mode", "pixel_displacement_v1"
                    ),
                    "selection": "fixed_evenly_spaced_validation_rows_not_score_selected",
                    "calibrated_identity_threshold": calibrated_threshold,
                    "prediction": {key: value.detach().float().cpu() for key, value in raw_reduced.items()
                                   if torch.is_tensor(value)},
                    "reference": {key: item[key].detach().cpu() for key in (
                        "first_frame", "flow", "flow_valid", "static_certified", "content_valid",
                        "instance_masks", "supervision_instance_masks",
                        "supervision_instance_mask_ids",
                        "supervision_object_identity_authoritative")
                                  if torch.is_tensor(item.get(key))},
                }
                torch.save(payload, export_dir / f"{sample_id}.pt")
            candidates = [("raw", raw_reduced["flow"]), ("readout", reduced["flow"])]
            if calibrated_flow is not None:
                candidates.append(("calibrated_task_object_gated", calibrated_flow))
            if "motion_mask" in raw_reduced:
                batched_prediction = {key: value.unsqueeze(0) for key, value in raw_reduced.items()}
                batched_item = {key: value.to(device).unsqueeze(0) if torch.is_tensor(value) else value
                                for key, value in item.items()}
                if "dense_motion_mask_logits" in raw_reduced:
                    from motion_jepa.flow_dit_dense_mask import dense_mask_sufficient_statistics
                    statistic_fn, statistic_prefix = dense_mask_sufficient_statistics, "_dense_mask_stat/"
                else:
                    from motion_jepa.flow_dit_activity import activity_sufficient_statistics
                    statistic_fn, statistic_prefix = activity_sufficient_statistics, "_activity_stat/"
                activity_stats = statistic_fn(
                    batched_prediction, batched_item, width=side, height=side,
                    config=config["loss"],
                )
                accumulator.update({f"{statistic_prefix}{key}": value for key, value in activity_stats.items()})
                candidates.extend((tag, raw_reduced[key]) for tag, key in (
                    ("ungated", "flow_ungated"), ("activity_gated", "flow_activity_gated")))
                if "flow_task_object_gated" in raw_reduced:
                    candidates.append((
                        "task_object_gated", raw_reduced["flow_task_object_gated"]
                    ))
            for tag, candidate in candidates:
                statistics = motion_sufficient_statistics(
                    candidate, item["flow"].to(device), item["flow_valid"].to(device),
                    static_certified=item["static_certified"].to(device),
                    width=side, height=side,
                )
                accumulator.update({f"_{tag}_motion_stat/{key}": value for key, value in statistics.items()})
                if (loss_config.get("profile") == "minimal5_direction_v1"
                        and tag in {"raw", "task_object_gated", "calibrated_task_object_gated"}):
                    from motion_jepa.flow_dit_minimal import segment_direction_statistics
                    direction_batch = {key:value.to(device).unsqueeze(0) for key,value in item.items()
                                       if torch.is_tensor(value)}
                    direction_batch["horizon_seconds"] = derived["horizon_seconds"]
                    direction_stats = segment_direction_statistics(candidate.unsqueeze(0),direction_batch,loss_config)
                    accumulator.update({f"_minimal5_{tag}_direction_stat/{key}":value
                                        for key,value in direction_stats.items()})

    decoder.train(was_training)
    backend.transformer.train(backend_was_training)
    result = dict(accumulator.values)
    result["__calibration_records"] = calibration_records
    result["__release_hold_observations"] = event_observations
    return result, dict(zero_accumulator.values)


def summarise_validation(
    payloads: list[tuple[dict[str, list[float]], dict[str, list[float]]]]
) -> dict[str, float]:
    """Merge per-rank accumulator values, then summarise once."""
    accumulator, zero_accumulator = MetricAccumulator(), MetricAccumulator()
    for local, local_zero in payloads:
        for key, values in local.items():
            # These are explicit non-metric side channels for scene-audited
            # calibration and matched-event joins across validation domains.
            if key.startswith("__"):
                continue
            accumulator.values[key].extend(values)
        for key, values in local_zero.items():
            zero_accumulator.values[key].extend(values)
    summary = accumulator.summary()
    zero_summary = zero_accumulator.summary()
    result = {
        "object_epe_norm_micro": summary.get("object_epe_norm_micro", float("nan")),
        "object_epe_px": summary.get("object_epe_px", float("nan")),
        "angular_error_deg": summary.get("angular_error_deg", float("nan")),
        "ade_norm": summary.get("ade_norm", float("nan")),
        "fde_norm": summary.get("fde_norm", float("nan")),
        "instance_ade_px": summary.get("instance_ade_px", float("nan")),
        # The selection criterion. Absent only when no sample carried an
        # instance_id, which the caller treats as an error rather than
        # silently falling back to a metric that is 54% static suppression.
        "relative_epe_moving": summary.get("relative_epe_moving", float("nan")),
        "static_predicted_magnitude_px": summary.get(
            "static_predicted_magnitude_px_micro", float("nan")
        ),
        "object_epe_px_moving_micro": summary.get(
            "object_epe_px_moving_micro", float("nan")
        ),
        "zero_relative_epe_moving": zero_summary.get(
            "relative_epe_moving", float("nan")
        ),
        "instance_fde_px": summary.get("instance_fde_px", float("nan")),
        "warped_mask_iou": summary.get("warped_mask_iou", float("nan")),
        "visibility_f1": summary.get("visibility_f1", float("nan")),
        "num_samples": summary.get("num_samples", 0.0),
        "zero_object_epe_norm_micro": zero_summary.get(
            "object_epe_norm_micro", float("nan")
        ),
        "zero_ade_norm": zero_summary.get("ade_norm", float("nan")),
    }
    for tag in ("raw", "readout", "ungated", "activity_gated", "task_object_gated", "calibrated_task_object_gated"):
        prefix = f"_{tag}_motion_stat/"
        totals = {key[len(prefix):]: sum(values) for key, values in accumulator.values.items()
                  if key.startswith(prefix)}
        if totals:
            result.update({f"{tag}_motion_counts/{key}": value for key, value in totals.items()})
            result.update({f"{tag}_motion/{key}": value
                           for key, value in summary_statistics(totals).items()
                           if value is not None})
    activity_prefix = "_activity_stat/"
    activity_totals = {key[len(activity_prefix):]: sum(values)
                       for key, values in accumulator.values.items() if key.startswith(activity_prefix)}
    if activity_totals:
        from motion_jepa.flow_dit_activity import summary_activity_statistics
        result.update({f"activity_counts/{key}": value for key, value in activity_totals.items()})
        result.update({f"activity/{key}": value for key, value in
                       summary_activity_statistics(activity_totals).items() if value is not None})
    affine_prefix = "_task_affine_stat/"
    for key in (
        "task_object_affine_displacement",
        "task_object_affine_direction",
        "task_object_affine_parallel_magnitude",
        "task_object_affine_trajectory",
    ):
        values = accumulator.values.get(f"{affine_prefix}{key}", [])
        if values:
            result[f"task_affine/{key.removeprefix('task_object_affine_')}"] = (
                sum(values) / len(values)
            )
    for key in (
        "task_object_affine_support",
        "task_object_affine_moving_support",
        "task_object_affine_trajectory_support",
    ):
        values = accumulator.values.get(f"{affine_prefix}{key}", [])
        if values:
            result[f"task_affine/{key.removeprefix('task_object_affine_')}"] = sum(values)
    task_flow_prefix = "_task_flow_stat/"
    for key in (
        "task_object_flow_displacement",
        "task_object_flow_direction",
        "task_object_flow_parallel_magnitude",
        "task_object_flow_trajectory",
    ):
        values = accumulator.values.get(f"{task_flow_prefix}{key}", [])
        if values:
            result[f"task_flow/{key.removeprefix('task_object_flow_')}"] = (
                sum(values) / len(values)
            )
    for key in (
        "task_object_flow_support",
        "task_object_flow_moving_support",
        "task_object_flow_trajectory_support",
    ):
        values = accumulator.values.get(f"{task_flow_prefix}{key}", [])
        if values:
            result[f"task_flow/{key.removeprefix('task_object_flow_')}"] = sum(values)
    dense_prefix = "_dense_mask_stat/"
    dense_totals = {key[len(dense_prefix):]: sum(values)
                    for key, values in accumulator.values.items() if key.startswith(dense_prefix)}
    if dense_totals:
        from motion_jepa.flow_dit_dense_mask import summary_dense_mask_statistics
        result.update({f"dense_mask_counts/{key}": value for key, value in dense_totals.items()})
        result.update({f"dense_mask/{key}": value for key, value in
                       summary_dense_mask_statistics(dense_totals).items() if value is not None})
    for tag in ("raw", "task_object_gated", "calibrated_task_object_gated"):
        prefix = f"_minimal5_{tag}_direction_stat/"
        totals = {key[len(prefix):]:sum(values) for key,values in accumulator.values.items()
                  if key.startswith(prefix)}
        if totals:
            result.update({f"minimal5_{tag}_direction_counts/{key}":value for key,value in totals.items()})
            n = totals["eligible_cases"]
            for angle in (15,30):
                accuracy = totals[f"direction{angle}_correct_case_sum"]/n if n else float("nan")
                result[f"minimal5_{tag}_direction/direction{angle}_accuracy"] = accuracy
                result[f"minimal5_{tag}_direction/direction{angle}_failure_fraction"] = 1-accuracy
    if accumulator.values.get("_bridge_occupancy_unknown"):
        result.pop("warped_mask_iou", None)
        result["bridge_future_occupancy_unavailable"] = 1.0
    return result


def aggregate_domain_validation(
    summary: dict[str, float],
    domain_summaries: list[tuple[str, dict[str, float]]],
    *,
    selection_metric: str,
    static_guard_metric: str,
) -> None:
    """Aggregate every configured validation domain for selection and guards.

    The original joint trainer averaged Bridge and PhyCo before optional
    additional sources were evaluated.  That made a third source visible in
    logs but unable to influence checkpoint selection.  Mutate ``summary`` so
    every configured domain has equal weight, independent of corpus size.
    """
    if len(domain_summaries) < 2:
        return

    def finite_values(key: str) -> list[float]:
        values = []
        for _, domain in domain_summaries:
            value = float(domain.get(key, float("nan")))
            if math.isfinite(value):
                values.append(value)
            elif key.startswith("minimal5_") and key.endswith("_failure_fraction"):
                prefix = key.split("/",1)[0] + "_counts/"
                if not (domain.get(prefix+"eligible_cases") == 0 and domain.get(prefix+"known_cases",0)>0):
                    return []
            elif key.endswith(("_motion/moving_relative_epe", "_motion/moving_direction30_failure_fraction")):
                prefix = key.split("/", 1)[0] + "_counts/"
                # A proven all-static validation source has no moving EPE.
                # Its static guard still participates below. Unknown/missing
                # metrics on a source with motion must continue to fail closed.
                if not (domain.get(prefix + "moving_count") == 0
                        and domain.get(prefix + "valid_count", 0) > 0):
                    return []
            else:
                return []
        return values

    moving = finite_values("relative_epe_moving")
    if moving:
        summary["physion_relative_epe_moving"] = moving[0]
        summary["relative_epe_moving"] = sum(moving) / len(moving)

    static = finite_values("static_predicted_magnitude_px")
    if static:
        summary["static_predicted_magnitude_px"] = max(static)

    if selection_metric != "relative_epe_moving":
        source_key = selection_metric.removeprefix("balanced_")
        values = finite_values(source_key)
        summary[f"primary_{source_key}"] = float(
            domain_summaries[0][1].get(source_key, float("nan"))
        )
        summary[selection_metric] = (
            sum(values) / len(values) if values else float("nan")
        )
        summary["selection_moving_domain_count"] = float(len(values))

    if static_guard_metric != "static_predicted_magnitude_px":
        source_key = static_guard_metric.removeprefix("worst_domain_")
        values = finite_values(source_key)
        summary[f"primary_{source_key}"] = float(
            domain_summaries[0][1].get(source_key, float("nan"))
        )
        summary[static_guard_metric] = max(values) if values else float("nan")


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config, args.overrides, placeholder_values(args))
    from motion_jepa.flow_dit_numerics import configure_numerics
    configure_numerics(config)
    paths, data_cfg, model_cfg = config["paths"], config["data"], config["model"]
    loss_cfg, train_cfg = config["loss"], config["train"]
    activity_heads_only = bool(train_cfg.get("activity_heads_only", False))
    mask_heads_only = bool(train_cfg.get("mask_heads_only", False))
    # These profiles own a strict continuation contract. Other profiles keep
    # the legacy checkpoint path; legacy files cannot claim exact RNG or
    # data-stream continuation merely by loading their optimizer state.
    clean_trajectory = loss_cfg.get("profile") == "clean_trajectory_v1"
    unified_trajectory = loss_cfg.get("profile") == "unified_trajectory_v1"
    unified_campaign = unified_trajectory or bool(train_cfg.get("unified_campaign", False))
    fresh_trajectory = clean_trajectory or unified_campaign
    from motion_jepa.flow_dit_unified import validate_fresh_start, validate_unified_runtime
    validate_fresh_start(config, args.init_from)
    if clean_trajectory and args.init_from:
        raise ValueError("clean Stage1 starts from original Wan; use --resume-from only for its own checkpoints")
    exact_resume = bool(model_cfg["decoder"].get("dense_motion_mask", False) or train_cfg.get("exact_resume", False))
    stop_signal = {"number": None}
    if exact_resume:
        def request_checkpoint_stop(number, _frame):
            stop_signal["number"] = number
        for number in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(number, request_checkpoint_stop)
    exact_checkpoint = None
    if exact_resume and args.resume_from:
        exact_checkpoint = verify_checkpoint(Path(args.resume_from), config=config)
    derived = resolve_derived(config)

    expert = args.expert or model_cfg["expert"]
    max_steps = int(args.max_steps or train_cfg["max_steps"])
    stop_at_step = int(args.stop_at_step or max_steps)
    if not 0 < stop_at_step <= max_steps:
        raise ValueError("--stop-at-step must lie in (0, max_steps]")
    grad_accum = int(train_cfg["grad_accum_steps"])
    keep_last = int(args.keep_last or train_cfg.get("keep_last", 0))
    label_source = config["provenance"]["label_source"]

    from datetime import timedelta

    from accelerate import Accelerator
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        InitProcessGroupKwargs,
        broadcast_object_list,
        gather_object,
        set_seed,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=str(train_cfg.get("precision", "bf16")),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=False),
            # The default NCCL watchdog is 600s. Any rank-uneven stretch longer
            # than that (for example validation or checkpointing) aborts the job.
            InitProcessGroupKwargs(timeout=timedelta(hours=2)),
        ],
    )
    set_seed(int(config["seed"]))
    device, is_main = accelerator.device, accelerator.is_main_process
    if unified_campaign:
        validate_unified_runtime(config, accelerator.num_processes)
    if fresh_trajectory and "global_batch" in train_cfg:
        effective_batch = accelerator.num_processes * int(train_cfg["per_gpu_batch"]) * grad_accum
        if effective_batch != int(train_cfg["global_batch"]):
            raise ValueError(f"fresh Stage1 global batch mismatch: {effective_batch} != {train_cfg['global_batch']}; set grad_accum_steps for the actual GPU count")
    validate_activity_heads_only_runtime(
        enabled=activity_heads_only, num_processes=accelerator.num_processes,
        device=device, checkpoint=args.init_from or args.resume_from,
    )
    validate_mask_heads_only_runtime(
        enabled=mask_heads_only, num_processes=accelerator.num_processes,
        device=device, checkpoint=args.init_from or args.resume_from,
    )
    if args.resume_from:
        saved_state = json.loads((Path(args.resume_from) / "trainer_state.json").read_text())
        validate_fresh_start(config, resume_state=saved_state)
        validate_activity_heads_only_resume(saved_state, activity_heads_only)
        validate_mask_heads_only_resume(saved_state, mask_heads_only)
    output_dir = Path(args.output_dir or paths["output_dir"]) / expert
    if unified_campaign and not args.resume_from and any((output_dir / name).exists() for name in ("history.jsonl", "metrics.jsonl", "done.json")):
        raise FileExistsError("unified campaign requires a fresh output directory or its own exact resume")
    if args.init_from and any((output_dir / name).exists() for name in ("history.jsonl", "metrics.jsonl")):
        raise FileExistsError("--init-from requires a fresh output directory; use --resume-from for an existing run")
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_set = build_dataset(
        config, "train", limit=args.limit, strict=False,
        allow_scenario_holes=args.allow_scenario_holes,
    )
    secondary_cfg = config["data"].get("secondary")
    secondary_val = None
    additional_validation = []
    train_sampler = None
    extra_sources = []
    if secondary_cfg:
        extra_sources.append((str(secondary_cfg.get("name", "phyco")), secondary_cfg))
    for source in config["data"].get("additional_sources", []):
        extra_sources.append((str(source["name"]), source))
    if extra_sources:
        # One pool, shuffled together, mixed batches -- not alternating loaders.
        # Each sample carries its own horizon_seconds; JointDataset fills any
        # missing kinematic placeholders so the collate stays rectangular.
        children = [train_set]
        fractions = []
        for source_name, source in extra_sources:
            children.append(build_dataset(
                config, "train", limit=args.limit, strict=False, source=source,
                allow_scenario_holes=args.allow_scenario_holes,
            ))
            fractions.append(float(source["mix_fraction"]))
        train_set = JointDataset(
            children,
            max_instances=int(config["data"].get("max_instances", 64)),
        )
        sampler_num_samples = None
        if exact_resume:
            sampler_num_samples = complete_accumulation_samples(
                len(train_set),
                int(train_cfg["per_gpu_batch"]) * accelerator.num_processes,
                grad_accum,
            )
        sampler_kind = train_cfg.get("sampler", "fixed_mixture")
        if sampler_kind not in {"fixed_mixture", "source_cycle_v1"}:
            raise ValueError(f"unknown training sampler {sampler_kind!r}")
        sampler_class = SourceCycleSampler if sampler_kind == "source_cycle_v1" else FixedMixtureSampler
        train_sampler = sampler_class(
            train_set,
            [1.0 - sum(fractions), *fractions],
            num_samples=sampler_num_samples,
            seed=int(config["seed"]),
        )
        for index, (source_name, source) in enumerate(extra_sources):
            validation = build_dataset(
                config,
                "val",
                limit=args.val_limit or int(config["train"].get("eval_samples", 128)),
                strict=False,
                source=source,
                allow_scenario_holes=args.allow_scenario_holes,
            )
            if index == 0 and secondary_cfg:
                secondary_val = validation
            else:
                additional_validation.append((source_name, source, validation))
    if not train_set.rows:
        raise RuntimeError(
            f"no cached samples under {paths['flow_cache_dir']}; run cache_stage1_flowdit.py"
        )
    coverage = (ConsumedCoverage(
                    train_set, int(train_cfg.get("min_unique_scenes", 20000)),
                    minimum_by_source=train_cfg.get("min_unique_scenes_by_source", {}),
                    source_names=[str(data_cfg.get("name", "primary")), *[name for name, _ in extra_sources]])
                if isinstance(train_sampler, SourceCycleSampler) else None)
    if fresh_trajectory and coverage is None:
        raise ValueError("fresh trajectory requires a mixed source-cycle dataset")
    if coverage is not None and is_main:
        print("[coverage] " + json.dumps(coverage.summary(), sort_keys=True), flush=True)
    caption_override_contract = caption_override_data_contract(train_set)
    if args.resume_from:
        validate_caption_override_resume(
            json.loads((Path(args.resume_from) / "trainer_state.json").read_text()),
            caption_override_contract,
        )
    val_set = build_dataset(
        config,
        args.val_split,
        limit=args.val_limit or int(train_cfg.get("eval_samples", 128)),
        strict=False,
        allow_scenario_holes=args.allow_scenario_holes,
    )
    if paths.get("event_validation_pairs_manifest"):
        from motion_jepa.flow_dit_events import EventParaphraseValidation
        additional_validation.append(("event_paraphrase", data_cfg,
            EventParaphraseValidation(val_set, paths["event_validation_pairs_manifest"])))

    calibration_sets, calibration_manifest, event_diagnostic_manifest = [], None, None
    eval_cfg = config.get("evaluation", {})
    if eval_cfg.get("gate_calibration_manifest"):
        calibration_manifest_path = Path(eval_cfg["gate_calibration_manifest"])
        calibration_manifest = json.loads(calibration_manifest_path.read_text())
        calibration_root = calibration_manifest_path.parent
        before_calibration_build = capture_rng()
        try:
            calibration_config = {**config, "paths": {**paths,
                "manifest_dir": str(calibration_root / "fot")}}
            calibration_sets.append(("fot", data_cfg, build_dataset(
                calibration_config, "val", limit=0, strict=True)))
            phyco_source = next(source for name, source in extra_sources if name == "phyco_anchor")
            phyco_calibration = {**phyco_source, "manifest_dir": str(calibration_root / "phyco_anchor")}
            calibration_sets.append(("phyco_anchor", phyco_calibration, build_dataset(
                config, "val", limit=0, strict=True, source=phyco_calibration)))
        finally:
            restore_rng(before_calibration_build)
    if eval_cfg.get("release_hold_pairs_manifest"):
        event_diagnostic_manifest = json.loads(Path(eval_cfg["release_hold_pairs_manifest"]).read_text())

    event_contract = None
    loader_dataset = train_set
    collate_fn = lambda batch: collate(batch, int(data_cfg["prompt_pad_length"]))
    if paths.get("event_pairs_manifest"):
        from motion_jepa.flow_dit_events import EventPairedDataset, collate_event_pairs
        if float(train_cfg.get("caption_dropout", 0)) != 0:
            raise ValueError("matched-event training requires caption_dropout=0")
        if train_cfg.get("objective") != "direct":
            raise ValueError("matched-event training currently requires deterministic direct prediction")
        loader_dataset = EventPairedDataset(train_set, paths["event_pairs_manifest"],
                                           paths["event_prompt_table"],
                                           prompt_embeddings=(getattr(train_set, "children", [train_set])[0].prompt_table
                                               if Path(paths["event_prompt_table"]).resolve() == Path(paths["prompt_table"]).resolve()
                                               else None))
        event_contract = loader_dataset.contract
        if args.resume_from and saved_state.get("event_pair_contract") != event_contract:
            raise ValueError("event-pair resume contract changed")
        collate_fn = lambda batch: collate_event_pairs(batch, int(data_cfg["prompt_pad_length"]))
        if is_main:
            print("[event-pairs] " + json.dumps(event_contract), flush=True)
    elif any(float(loss_cfg.get("event_" + name, 0)) for name in
             ("invariance", "displacement", "relative_displacement", "direction", "velocity", "affine", "static_role",
              "direction_margin", "segment_direction", "field_direction")):
        raise ValueError("event losses require paths.event_pairs_manifest")
    loader = DataLoader(
        loader_dataset,
        batch_size=int(train_cfg["per_gpu_batch"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(train_cfg["num_workers"]) > 0,
        collate_fn=collate_fn,
    )
    if len(loader) == 0:
        raise ValueError("training dataset is smaller than one batch with drop_last=True")

    mean, std = load_latent_statistics(paths["wan_model"])
    backend = WanFlowDiTBackend(
        model_dir=paths["wan_model"],
        expert=expert,
        latent_channels=derived["latent_channels"],
        lora_rank=int(model_cfg["lora_rank"]),
        lora_alpha=int(model_cfg["lora_alpha"]),
        latents_mean=mean,
        latents_std=std,
        device=device,
        flow_shift=float(train_cfg["flow_shift"]),
        boundary_sigma=float(model_cfg["boundary_sigma"]),
        gradient_checkpointing=bool(model_cfg["gradient_checkpointing"]),
        use_time_channel=bool(model_cfg["use_time_channel"]),
        use_instance_channels=bool(config["model"].get("use_instance_channels", False)),
        objective=str(config["train"].get("objective", "flow_matching")),
        time_scale=config["data"].get("time_scale_seconds"),
    )
    refine_from = str(train_cfg.get("refine_from", ""))
    if refine_from and expert != "low":
        raise ValueError("train.refine_from is a low-expert setting; got expert="
                         f"{expert!r}")
    if refine_from:
        # Frozen high-expert partner: supplies the sigma=1 estimate whose
        # distribution the low expert must learn to refine, and joins the
        # validation sampler so selection runs the real 2-step path.
        partner = WanFlowDiTBackend(
            model_dir=paths["wan_model"],
            expert="high",
            lora_rank=int(model_cfg["lora_rank"]),
            lora_alpha=int(model_cfg["lora_alpha"]),
            latents_mean=mean,
            latents_std=std,
            device=device,
            flow_shift=float(train_cfg["flow_shift"]),
            boundary_sigma=float(model_cfg["boundary_sigma"]),
            gradient_checkpointing=False,
            use_time_channel=bool(model_cfg["use_time_channel"]),
            use_instance_channels=bool(config["model"].get("use_instance_channels", False)),
            objective="direct",
        )
        partner_state = torch.load(
            Path(refine_from) / "backend_params.pt", map_location="cpu", weights_only=True
        )
        partner_named = dict(partner.transformer.named_parameters())
        missing = [k for k in partner_state if k not in partner_named]
        if missing:
            raise KeyError(f"refine_from checkpoint has unknown params: {missing[:3]}")
        with torch.no_grad():
            for key, value in partner_state.items():
                partner_named[key].data.copy_(value.to(partner_named[key].dtype))
        partner.transformer.requires_grad_(False)
        partner.transformer.eval()
        print(f"[refine] frozen high partner loaded from {refine_from}", flush=True)

    decoder = build_decoder(config, derived).to(device)
    if clean_trajectory or unified_trajectory:
        from motion_jepa.flow_dit_clean import freeze_clean_unused_heads
        frozen_names = freeze_clean_unused_heads(decoder)
        if is_main:
            print(f"[{loss_cfg['profile']}] froze {len(frozen_names)} diagnostic parameter tensors", flush=True)
    if loss_cfg.get("profile") == "minimal5_direction_v1":
        from motion_jepa.flow_dit_minimal import freeze_minimal5_unused_heads
        frozen_names = freeze_minimal5_unused_heads(decoder)
        if is_main:
            print(f"[minimal5] froze {len(frozen_names)} diagnostic-only parameter tensors for DDP", flush=True)
    freeze_visibility = bool(train_cfg.get("freeze_visibility_heads", False))
    if freeze_visibility:
        if loss_cfg.get("profile") not in {"minimal5_direction_v1", "clean_trajectory_v1", "unified_trajectory_v1"} and float(loss_cfg.get("visibility_bce", 0.1)) != 0:
            raise ValueError("freeze_visibility_heads requires loss.visibility_bce=0")
        prefixes = ("visibility_fuse.", "visibility_up.", "visibility_head.")
        frozen_visibility_parameters = 0
        for name, parameter in decoder.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad_(False)
                frozen_visibility_parameters += parameter.numel()
        if not frozen_visibility_parameters:
            raise ValueError("no independent decoder visibility heads found to freeze")
        if is_main:
            print(f"[freeze] {frozen_visibility_parameters} visibility parameters; shared features remain trainable", flush=True)

    if refine_from:
        backend.refine_partner = partner

    activity_control = None
    if activity_heads_only:
        activity_control = ActivityHeadsOnlyControl(
            backend, decoder, Path(args.init_from or args.resume_from),
            config=config, expert=expert, output_dir=output_dir, resume=bool(args.resume_from),
        )
        if is_main:
            print(f"[freeze] activity heads only: {json.dumps(activity_control.contract, sort_keys=True)}", flush=True)
    elif mask_heads_only:
        from motion_jepa.flow_dit_optimization import MASK_HEAD_PREFIXES
        mask_head_names = frozenset(
            name for name, _ in decoder.named_parameters()
            if name.startswith(MASK_HEAD_PREFIXES)
        )
        validate_mask_head_parameter_names(
            mask_head_names,
            semantic_identity=bool(
                model_cfg["decoder"].get("task_object_identity_semantic", False)
            ),
        )
        activity_control = ActivityHeadsOnlyControl(
            backend, decoder, Path(args.init_from or args.resume_from),
            config=config, expert=expert, output_dir=output_dir, resume=bool(args.resume_from),
            trainable_parameter_names=mask_head_names, mode_key="mask_heads_only",
        )
        if is_main:
            print(f"[freeze] mask heads only: {json.dumps(activity_control.contract, sort_keys=True)}", flush=True)

    lora_parameters = [
        parameter
        for name, parameter in backend.transformer.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    head_parameters = [
        parameter
        for name, parameter in backend.transformer.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    decoder_parameters = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": float(train_cfg["lora_learning_rate"])},
            {"params": head_parameters + decoder_parameters, "lr": float(train_cfg["head_learning_rate"])},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.999),
    )
    warmup_steps = int(train_cfg["warmup_steps"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    # Deliberately kept out of accelerator.prepare: a prepared scheduler steps
    # once per process and silently compresses the schedule by world_size.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    resume_trainer_state: dict = {}
    initialization: dict = (activity_control.initialization if activity_control is not None
                            else {"kind": "pretrained_backbone_new_heads"})
    if args.init_from and activity_control is None:
        initialization = initialize_from_checkpoint(
            backend, decoder, args.init_from, config=config, expert=expert
        )
        if is_main:
            print(f"[initialize] {json.dumps(initialization, sort_keys=True)}", flush=True)
    if args.resume_from:
        checkpoint = Path(args.resume_from)
        if activity_control is None:
            backend.load_state_dict(
                torch.load(checkpoint / "backend_params.pt", map_location="cpu", weights_only=True)
            )
            decoder.load_state_dict(
                torch.load(checkpoint / "decoder.pt", map_location="cpu", weights_only=True)
            )
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        resume_trainer_state = state
        initialization = state.get("initialization", {"kind": "legacy_resume"})
        if state.get("label_source") != label_source:
            raise ValueError("resume checkpoint was trained on different supervision")
        if state.get("expert") != expert:
            raise ValueError(f"resume checkpoint is expert {state.get('expert')!r}, not {expert!r}")
        if state.get("supervision_mode", "legacy") != loss_cfg.get("supervision_mode", "legacy"):
            raise ValueError("resume supervision mode differs; use --init-from for a new training objective")
        if bool(state.get("freeze_visibility_heads", False)) != freeze_visibility:
            raise ValueError("resume freeze_visibility_heads differs from the saved optimizer contract")
        recorded_max_steps = int(state.get("max_steps", max_steps))
        if recorded_max_steps != max_steps:
            raise ValueError(
                "resume checkpoint used a different max_steps/LR schedule: "
                f"{recorded_max_steps} vs {max_steps}"
            )
        start_step = int(state["step"])

    train_module, decoder, optimizer, loader = accelerator.prepare(
        backend.transformer, decoder, optimizer, loader
    )
    backend.transformer = train_module
    if args.resume_from:
        checkpoint = Path(args.resume_from)
        optimizer.load_state_dict(
            torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
        )
        scheduler.load_state_dict(
            torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=False)
        )

    resume_contract = None
    data_cursor = None
    resume_rng = None
    if exact_resume:
        resume_contract = runtime_contract(
            config=config, expert=expert, max_steps=max_steps, sampler=train_sampler,
            loader=loader, train_limit=args.limit, val_limit=args.val_limit,
            val_split=args.val_split, world_size=accelerator.num_processes,
            gradient_accumulation_steps=grad_accum,
        )
        data_cursor = DataCursor(len(loader))
        if exact_checkpoint is not None:
            saved_rng = exact_checkpoint["rng_state"]
            if saved_rng["contract"] != resume_contract:
                raise ValueError("exact resume runtime, data population or sampler contract changed")
            data_cursor = DataCursor(**saved_rng["data_cursor"])
            if coverage is not None:
                if "consumed_coverage" not in saved_rng:
                    raise ValueError("source-cycle resume requires committed consumed coverage")
                coverage.load_state_dict(saved_rng["consumed_coverage"])
            resume_rng = rank_rng_state(
                saved_rng,
                process_index=accelerator.process_index,
                world_size=accelerator.num_processes,
            )
            if is_main:
                archived = rewind_logs(output_dir, saved_rng["log_positions"], step=start_step,
                                       snapshots=saved_rng["log_snapshots"])
                print(f"[resume] exact state step={start_step} cursor={data_cursor.state_dict()} "
                      f"archived_crash_tails={archived}", flush=True)
            accelerator.wait_for_everyone()

    trainable = lora_parameters + head_parameters + decoder_parameters
    separate_mask_clip = bool(train_cfg.get("separate_mask_gradient_clip", False))
    separate_task_object_clip = bool(
        train_cfg.get("separate_task_object_gradient_clip", False)
    )
    if separate_mask_clip:
        from motion_jepa.flow_dit_optimization import split_gradient_groups, clip_gradient_groups
        clip_groups = split_gradient_groups(
            trainable, accelerator.unwrap_model(decoder).named_parameters(),
            separate_task_object=separate_task_object_clip,
            allow_empty_motion=mask_heads_only,
        )
        motion_clip_parameters, mask_clip_parameters = clip_groups[:2]
        task_object_clip_parameters = (
            clip_groups[2] if separate_task_object_clip else None
        )
    if is_main:
        print(
            f"[model] expert={expert} lora={sum(p.numel() for p in lora_parameters)/1e6:.1f}M "
            f"heads={sum(p.numel() for p in head_parameters)/1e6:.2f}M "
            f"decoder={sum(p.numel() for p in decoder_parameters)/1e6:.2f}M "
            f"train={len(train_set)} val={len(val_set)} world={accelerator.num_processes} "
            f"effective_batch={int(train_cfg['per_gpu_batch']) * grad_accum * accelerator.num_processes}",
            flush=True,
        )

    sampler = SamplerConfig.from_config(config)
    validation_sampler = SamplerConfig(
        num_steps=sampler.num_steps,
        guidance_scale=1.0,
        flow_shift=sampler.flow_shift,
        boundary_sigma=sampler.boundary_sigma,
        num_samples=1,
        shrinkage_px=sampler.shrinkage_px,
        reduce="mean",
        seed=sampler.seed,
    )
    history_path = output_dir / "history.jsonl"
    metrics_file = (output_dir / "metrics.jsonl").open("a") if is_main else None

    def write_history(record: dict) -> None:
        if not is_main:
            return
        with history_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    def validate(step: int, kind: str, loss_total: float) -> dict[str, float]:
        if not len(val_set):
            return {}
        accelerator.wait_for_everyone()
        calibrated_threshold = None
        event_observations = []
        if calibration_manifest is not None:
            from motion_jepa.flow_dit_gate_calibration import calibrate_identity_threshold
            calibration_rows = {row["sample_id"]: row for row in calibration_manifest["rows"]}
            gathered_calibration = []
            # Extra calibration inference must not change dropout/sampler RNG
            # in this displacement-only training comparison.
            calibration_rng = capture_rng()
            wrapped_calibration = backend.transformer
            backend.transformer = accelerator.unwrap_model(wrapped_calibration)
            try:
                for _, source_config, calibration_set in calibration_sets:
                    local_calibration = run_validation(
                        backend=backend, decoder=accelerator.unwrap_model(decoder),
                        dataset=calibration_set, config=config,
                        derived={**derived, "horizon_seconds": source_config["horizon_seconds"]},
                        sampler=validation_sampler, device=device, limit=0, seed=int(config["seed"]),
                        shard=(accelerator.process_index, accelerator.num_processes),
                        calibration_rows_by_id=calibration_rows, calibration_only=True)
                    gathered = gather_object([local_calibration])
                    if is_main:
                        for payload, _ in gathered:
                            gathered_calibration.extend(payload.get("__calibration_records", []))
            finally:
                backend.transformer = wrapped_calibration
                restore_rng(calibration_rng)
            threshold_result = calibrate_identity_threshold(gathered_calibration, calibration_manifest) if is_main else None
            threshold_container = [threshold_result]
            broadcast_object_list(threshold_container, from_process=0)
            threshold_result = threshold_container[0]
            calibrated_threshold = threshold_result["threshold"]
            if is_main:
                destination = output_dir / "calibration" / f"step-{step:06d}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({**threshold_result, "records": gathered_calibration}, indent=2) + "\n")
        # Validate on the unwrapped transformer: DDP's forward has no work to do
        # here and swapping it out keeps the sampling loop free of collectives.
        wrapped = backend.transformer
        backend.transformer = accelerator.unwrap_model(wrapped)
        try:
            local = run_validation(
                backend=backend,
                decoder=accelerator.unwrap_model(decoder),
                dataset=val_set,
                config=config,
                derived=derived,
                sampler=validation_sampler,
                device=device,
                limit=0,
                seed=int(config["seed"]),
                shard=(accelerator.process_index, accelerator.num_processes),
                export_dir=output_dir / "validation_exports" / f"step-{step:06d}" / "bridge",
                calibrated_threshold=calibrated_threshold,
                release_hold_recipes=event_diagnostic_manifest["rows"] if event_diagnostic_manifest else None,
            )
        finally:
            backend.transformer = wrapped
        payloads = gather_object([local])
        summary: dict[str, float] = summarise_validation(payloads) if is_main else {}
        if is_main:
            for payload, _ in payloads:
                event_observations.extend(payload.get("__release_hold_observations", []))
        domain_summaries: list[tuple[str, dict[str, float]]] = (
            [("primary", dict(summary))] if is_main else []
        )
        if secondary_val is not None:
            secondary_derived = {
                **derived,
                "horizon_seconds": [
                    float(v) for v in secondary_cfg["horizon_seconds"]
                ],
            }
            wrapped = backend.transformer
            backend.transformer = accelerator.unwrap_model(wrapped)
            try:
                local_secondary = run_validation(
                    backend=backend,
                    decoder=accelerator.unwrap_model(decoder),
                    dataset=secondary_val,
                    config=config,
                    derived=secondary_derived,
                    sampler=validation_sampler,
                    device=device,
                    limit=0,
                    seed=int(config["seed"]),
                    shard=(accelerator.process_index, accelerator.num_processes),
                    export_dir=output_dir / "validation_exports" / f"step-{step:06d}" / "phyco",
                    calibrated_threshold=calibrated_threshold,
                )
            finally:
                backend.transformer = wrapped
            payloads_secondary = gather_object([local_secondary])
            if is_main:
                secondary_summary = summarise_validation(payloads_secondary)
                for key, value in secondary_summary.items():
                    # instance_ade/fde aggregate over center_valid, which is
                    # all-False placeholder on PhyCo rows (RAFT flow has no
                    # center tracks): those keys are structurally undefined
                    # for this domain, not missing data - drop them instead
                    # of writing NaN into every validation record.
                    if key.startswith("instance_") and not math.isfinite(float(value)):
                        continue
                    summary[f"phyco_{key}"] = value
                domain_summaries.append(("phyco", secondary_summary))
        for source_name, source_cfg, source_validation in additional_validation:
            source_derived = {
                **derived,
                "horizon_seconds": [float(v) for v in source_cfg["horizon_seconds"]],
            }
            wrapped = backend.transformer
            backend.transformer = accelerator.unwrap_model(wrapped)
            try:
                local_source = run_validation(
                    backend=backend,
                    decoder=accelerator.unwrap_model(decoder),
                    dataset=source_validation,
                    config=config,
                    derived=source_derived,
                    sampler=validation_sampler,
                    device=device,
                    limit=0,
                    seed=int(config["seed"]),
                    shard=(accelerator.process_index, accelerator.num_processes),
                    export_dir=(
                        output_dir / "validation_exports" / f"step-{step:06d}" / source_name
                    ),
                    calibrated_threshold=calibrated_threshold,
                    release_hold_recipes=(event_diagnostic_manifest["rows"]
                        if event_diagnostic_manifest and source_name == "event_static" else None),
                )
            finally:
                backend.transformer = wrapped
            payloads_source = gather_object([local_source])
            if is_main:
                source_summary = summarise_validation(payloads_source)
                for payload, _ in payloads_source:
                    event_observations.extend(payload.get("__release_hold_observations", []))
                for key, value in source_summary.items():
                    if key.startswith("instance_") and not math.isfinite(float(value)):
                        continue
                    summary[f"{source_name}_{key}"] = value
                domain_summaries.append((source_name, source_summary))
        if is_main:
            if calibrated_threshold is not None:
                summary["calibrated_selection_threshold"] = calibrated_threshold
            if event_diagnostic_manifest is not None:
                from motion_jepa.flow_dit_event_diagnostics import summarize_release_hold
                event_report = summarize_release_hold(event_observations, event_diagnostic_manifest)
                destination = output_dir / "release_hold" / f"step-{step:06d}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(event_report, indent=2) + "\n")
                for field, values in event_report["summary"].items():
                    summary.update({f"release_hold/{field}/{key}": value for key, value in values.items()})
                summary["release_hold/pairs"] = event_report["pairs"]
            if train_cfg.get("selection_guard_limits"):
                from motion_jepa.flow_dit_selection import add_selection_guard
                for _, domain_summary in domain_summaries:
                    add_selection_guard(domain_summary, train_cfg["selection_guard_limits"])
            aggregate_domain_validation(
                summary,
                domain_summaries,
                selection_metric=selection_metric,
                static_guard_metric=static_guard_metric,
            )
        # Broadcast so every rank reaches the same best-checkpoint decision.
        # Deciding on rank 0 alone and calling save_checkpoint() there deadlocks:
        # that function opens with wait_for_everyone(), a barrier the other ranks
        # never reach because they have already moved on to the next step's
        # gradient all-reduce.
        container = [summary]
        broadcast_object_list(container, from_process=0)
        summary = container[0]
        if activity_control is not None:
            activity_control.validate_ungated(summary)
        if is_main:
            write_history(
                {
                    "type": kind,
                    "step": step,
                    "loss/total": loss_total,
                    "selection_metric": selection_metric,
                    "static_guard_metric": static_guard_metric,
                    "static_magnitude_budget_px": static_budget,
                    "early_stop_min_delta": min_delta,
                    **{f"metric/{key}": value for key, value in summary.items()},
                }
            )
        accelerator.wait_for_everyone()
        return summary

    negative_prompt = pad_prompt(
        train_set.negative_prompt(), int(data_cfg["prompt_pad_length"])
    ).to(device)
    caption_dropout = float(train_cfg.get("caption_dropout", 0.0))
    aux_warmup = int(loss_cfg.get("aux_warmup_steps", 0))
    aux_mode = str(loss_cfg.get("aux_sigma_weight", "one_minus_sigma_sq"))
    selection_metric = str(
        train_cfg.get("selection_metric", "relative_epe_moving")
    )
    static_guard_metric = str(train_cfg.get("static_guard_metric", "static_predicted_magnitude_px"))
    static_budget = float(train_cfg.get("static_magnitude_budget_px", 2.0))
    patience = int(train_cfg.get("early_stop_patience", 0))
    min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))
    best_metric = float("inf")
    best_object_epe_norm_micro = float("inf")
    stale_validations = 0
    if args.resume_from:
        validate_selection_resume(
            resume_trainer_state, selection_metric=selection_metric,
            static_guard_metric=static_guard_metric, static_budget=static_budget,
            min_delta=min_delta,
        )
        saved_best = resume_trainer_state.get("best_selection_metric")
        saved_object = resume_trainer_state.get("best_object_epe_norm_micro")
        if saved_best is not None and math.isfinite(float(saved_best)):
            best_metric = float(saved_best)
            if saved_object is not None and math.isfinite(float(saved_object)):
                best_object_epe_norm_micro = float(saved_object)
            stale_validations = int(resume_trainer_state.get("stale_validations", 0))
        else:
            recovery_path = Path(args.resume_from).parent / "history.jsonl"
            if not recovery_path.is_file():
                recovery_path = Path(args.resume_from).parent / "metrics.jsonl"
            (
                best_metric,
                best_object_epe_norm_micro,
                stale_validations,
            ) = recover_validation_state(
                recovery_path,
                selection_metric=selection_metric,
                static_budget=static_budget,
                min_delta=min_delta,
                static_guard_metric=static_guard_metric,
            )
            if is_main:
                print(
                    f"[resume] recovered validation state from {recovery_path.name}: "
                    f"best={best_metric} stale={stale_validations}",
                    flush=True,
                )
    if getattr(backend, "objective", "flow_matching") == "direct":
        # At sigma=1 the (1-sigma)^2 weight is exactly zero: every decoded-flow
        # loss would silently vanish and the run would train on the latent MSE
        # alone while reporting all the aux terms as zeros.
        if aux_mode != "uniform":
            raise ValueError(
                "objective=direct requires loss.aux_sigma_weight=uniform; "
                f"got {aux_mode!r} whose weight at sigma=1 is zero"
            )

    if is_main:
        write_resolved_config(output_dir, config, expert=expert,
                              max_steps=max_steps, stop_at_step=stop_at_step)

    def save_checkpoint(step: int, *, best: bool = False) -> None:
        accelerator.wait_for_everyone()
        rank_rngs = (
            gather_object([capture_rng()]) if exact_resume and not best else None
        )
        if not is_main:
            accelerator.wait_for_everyone()
            return
        name = "checkpoint-best" if best else f"checkpoint-{step:06d}"
        destination = output_dir / name
        checkpoint = staging_directory(destination) if exact_resume and not best else destination
        checkpoint.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(backend.transformer)
        if activity_control is not None:
            activity_control.save_backend(
                checkpoint / "backend_params.pt", transformer=unwrapped,
                decoder=accelerator.unwrap_model(decoder),
            )
        else:
            torch.save(
                {
                    name_: parameter.detach().cpu()
                    for name_, parameter in unwrapped.named_parameters()
                    if parameter.requires_grad
                },
                checkpoint / "backend_params.pt",
            )
        torch.save(
            accelerator.unwrap_model(decoder).state_dict(), checkpoint / "decoder.pt"
        )
        if not best:
            torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
            torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "expert": expert,
                    "label_source": label_source,
                    "condition_scope": config["provenance"]["condition_scope"],
                    "max_steps": max_steps,
                    "stop_at_step": stop_at_step,
                    "selection_metric": selection_metric,
                    "static_guard_metric": static_guard_metric,
                    "static_magnitude_budget_px": static_budget,
                    "early_stop_min_delta": min_delta,
                    "best_selection_metric": (
                        best_metric if math.isfinite(best_metric) else None
                    ),
                    "best_object_epe_norm_micro": (
                        best_object_epe_norm_micro
                        if math.isfinite(best_object_epe_norm_micro)
                        else None
                    ),
                    "stale_validations": stale_validations,
                    "codec": config["codec"],
                    "decoder_type": str(model_cfg["decoder"].get("type", "dense")),
                    # The whole decoder block, not just its type: causal_slots and
                    # incremental_params change the module's parameter names, so a
                    # checkpoint that records only "slots" cannot be rebuilt.
                    "decoder_config": dict(config["model"]["decoder"]),
                    "flow_resolution": int(data_cfg["flow_resolution"]),
                    "initialization": initialization,
                    "loss_profile": loss_cfg.get("profile", "legacy"),
                    "unified_campaign": unified_campaign,
                    "supervision_mode": str(loss_cfg.get("supervision_mode", "legacy")),
                    "supervision_cache_dir": str(paths.get("supervision_cache_dir", "")),
                    "freeze_visibility_heads": freeze_visibility,
                    "activity_heads_only": activity_heads_only,
                    "mask_heads_only": mask_heads_only,
                    **({"consumed_coverage": coverage.summary()} if coverage is not None else {}),
                    **({"event_pair_contract": event_contract} if event_contract is not None else {}),
                    **({"caption_override_contract": caption_override_contract}
                       if caption_override_contract is not None else {}),
                    **({"resume_protocol": EXACT_RESUME_PROTOCOL,
                        "resume_contract": resume_contract,
                        "checkpoint_every": (int(train_cfg["save_every"]) if fresh_trajectory else min(int(train_cfg["save_every"]), 25))}
                       if exact_resume and not best else {}),
                    **({f"{activity_control.mode_key}_contract": activity_control.contract}
                       if activity_control is not None else {}),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if not best:
            if exact_resume:
                metrics_file.flush()
                torch.save({"protocol": EXACT_RESUME_PROTOCOL, "step": step,
                            "contract": resume_contract, "data_cursor": data_cursor.state_dict(),
                            **({"consumed_coverage": coverage.state_dict()} if coverage is not None else {}),
                            "rng_by_rank": rank_rngs,
                            "log_positions": log_positions(output_dir),
                            "log_snapshots": log_snapshots(output_dir)},
                           checkpoint / "rng_state.pt")
                shutil.copyfile(output_dir / "resolved_config.yaml", checkpoint / "resolved_config.yaml")
                commit_checkpoint(checkpoint, destination, step=step)
            rotate_checkpoints(output_dir, keep_last)
        print(f"[checkpoint] {destination}", flush=True)
        accelerator.wait_for_everyone()

    # The initial record must exist before any optimisation: it is both the
    # zero-flow sanity check (the flow head is zero-initialised) and the
    # untrained baseline of the run.
    raw_guard_initial = None
    if start_step == 0 and not (exact_resume and args.resume_from):
        initial = validate(0, "validation_initial", float("nan"))
        raw_guard_initial = initial
        if event_contract is not None and args.init_from and initial:
            reference = float(initial.get(selection_metric, float("nan")))
            if not math.isfinite(reference):
                raise ValueError("event run lacks a finite transferred-model validation reference")
            # The first trained checkpoint must beat the transferred model;
            # infinity would accept an arbitrary regression at the first eval.
            best_metric = reference
            best_object_epe_norm_micro = float(initial.get("object_epe_norm_micro", float("inf")))
            if is_main:
                print(f"[event-baseline] candidate selection starts at {best_metric:.6f}; static guard remains mandatory", flush=True)
        if is_main and initial:
            gap = abs(
                initial["object_epe_norm_micro"] - initial["zero_object_epe_norm_micro"]
            ) / max(initial["zero_object_epe_norm_micro"], 1e-9)
            if args.init_from:
                print(
                    f"[baseline] transferred weights validated at new-run step 0; "
                    f"absolute EPE gap from zero-flow {gap * 100:.3f}%",
                    flush=True,
                )
            else:
                print(
                    f"[sanity] untrained EPE differs from zero-flow by {gap * 100:.3f}% "
                    f"(expected ~0 with a zero-initialised flow head)",
                    flush=True,
                )
        if exact_resume:
            # A preemption before step 25 need not repeat the full step-zero
            # validation or reseed/reorder the first training epoch.
            save_checkpoint(0)

    # Early stopping (train.early_stop_patience > 0) watches the validation
    # metric, not the loss: the training loss can keep falling while
    # validation EPE rises. max_steps is then a ceiling and patience decides
    # the actual length. The released recipe uses patience 0 (no early stop).
    if start_step >= stop_at_step:
        raise ValueError(
            f"checkpoint step {start_step} already reached stop-at-step {stop_at_step}"
        )
    step, started = start_step, time.time()
    log_every = int(train_cfg["log_every"])
    eval_every = int(train_cfg["eval_every"])
    save_every = int(train_cfg["save_every"])
    if exact_resume and not fresh_trajectory:
        save_every = min(save_every, 25)
    pending_coverage_ids = []
    from motion_jepa.flow_dit_unified import UpdateLossAccumulator
    update_loss_accumulator = UpdateLossAccumulator() if unified_campaign else None
    done = bool(start_step >= stop_at_step or (exact_resume and stop_signal["number"] is not None))
    while not done:
        training_batches = (data_cursor.iterate(loader, train_sampler, resume_rng=resume_rng)
                            if exact_resume else loader)
        resume_rng = None
        for raw in training_batches:
            if activity_control is not None:
                # Features stay deterministic; explicit caption dropout below
                # remains identical to joint training.
                backend.transformer.eval()
                decoder.eval()
            # Mixed Physion+PhyCo batches: the temporal span rides on the
            # sample ("horizon_seconds" [B,T]); when absent this is a
            # single-domain run and the global list applies. The negative
            # prompt is shared -- verified bit-identical across both prompt
            # tables (same umT5 encoding of the empty string).
            batch_horizons = raw.get("horizon_seconds", derived["horizon_seconds"])
            if bool(raw.get("flow_latent_is_placeholder", torch.tensor(False)).any()):
                if train_cfg.get("objective") != "direct" or float(loss_cfg["flow_matching"]) != 0:
                    raise ValueError("constant prior caches require direct objective and zero latent/FM loss")
            with accelerator.accumulate(train_module, decoder):
                prompt = raw["prompt_embeds"]
                if caption_dropout > 0:
                    drop = torch.rand(prompt.shape[0], device=prompt.device) < caption_dropout
                    if drop.any():
                        prompt = prompt.clone()
                        prompt[drop] = negative_prompt.to(prompt.dtype)
                batch = {**raw, "prompt_embeds": prompt}
                outputs = backend.train_step(
                    batch, horizon_seconds=batch_horizons
                )
                prediction = decoder(
                    outputs["x0"],
                    backend.normalize_latents(raw["image_cond_latent"].to(device)),
                    raw["first_frame"].to(device),
                    horizon_seconds=(
                        batch_horizons.to(device)
                        if torch.is_tensor(batch_horizons)
                        else batch_horizons
                    ),
                    instance_masks=(
                        raw["instance_masks"].to(device)
                        if "instance_masks" in raw
                        else None
                    ),
                    prompt_embeds=prompt.to(device),
                )
                weight = sigma_weight(outputs["sigmas"], aux_mode)
                aux = flow_space_losses(
                    prediction,
                    {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in raw.items()},
                    loss_cfg,
                    sample_weight=weight,
                    width=int(data_cfg["width"]),
                    height=int(data_cfg["height"]),
                )
                minimal5_gradients = {}
                if (loss_cfg.get("profile") == "minimal5_direction_v1"
                        and (step == start_step or (step + 1) % log_every == 0)):
                    from motion_jepa.flow_dit_minimal import minimal5_gradient_diagnostics
                    minimal5_gradients = minimal5_gradient_diagnostics(aux, prediction)
                if event_contract is not None:
                    from motion_jepa.flow_dit_events import event_difference_loss
                    event_total, event_logs = event_difference_loss(
                        prediction,
                        {key: value.to(device) if torch.is_tensor(value) else value
                         for key, value in raw.items()}, loss_cfg,
                    )
                    aux["aux_total"] = aux["aux_total"] + event_total
                    aux.update(event_logs)
                    aux["event_total"] = event_total.detach()
                ramp = cosine_ramp(step, aux_warmup)
                loss = (
                    float(loss_cfg["flow_matching"]) * outputs["flow_matching"]
                    + ramp * aux["aux_total"]
                )
                if update_loss_accumulator is not None:
                    update_loss_accumulator.add(
                        {**aux, "total": loss, "flow_matching": outputs["flow_matching"]},
                        int(raw["flow"].shape[0]))
                if (event_contract is not None or fresh_trajectory) and not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite training loss; refusing optimizer update")
                accelerator.backward(loss)
                gradient_norm = torch.zeros((), device=device)
                activity_gradient_norms = {}
                if accelerator.sync_gradients:
                    if unified_campaign and step < 8:
                        from motion_jepa.flow_dit_unified import gradient_diagnostics, aggregate_gradient_diagnostics
                        gradient_scale = (float(accelerator.scaler.get_scale())
                                          if accelerator.scaler is not None else 1.)
                        local_diagnostics = gradient_diagnostics(
                            accelerator.unwrap_model(decoder), lora_parameters,
                            gradient_scale=gradient_scale)
                        rank_diagnostics = gather_object([local_diagnostics])
                        activity_gradient_norms.update(aggregate_gradient_diagnostics(rank_diagnostics))
                        activity_gradient_norms["gradient_diagnostic/update"] = step + 1
                        activity_gradient_norms["gradient_diagnostic/zero_upstream_expected"] = step == 0
                    if ((model_cfg["decoder"].get("motion_activity", False) or
                         model_cfg["decoder"].get("dense_motion_mask", False)) and
                            (step == start_step or (step + 1) % log_every == 0)):
                        # These are gradients of the complete weighted loss,
                        # before global clipping; log its effective allocation
                        # instead of assuming a larger scalar weight helps.
                        named = list(accelerator.unwrap_model(decoder).named_parameters())
                        for group, prefixes in (
                            ("object_activity", ("slot_activity_head.",)),
                            ("pixel_activity", ("pixel_activity_head.",)),
                            ("dense_motion_mask", ("dense_motion_mask_fuse.", "dense_motion_mask_up.", "dense_motion_mask_head.")),
                            ("task_object_slot", (
                                "task_object_slot_head.", "task_object_identity_head.",
                                "task_identity_visual_project.",
                                "task_identity_prompt_project.",
                            )),
                            ("affine", ("slot_params.",)),
                            ("residual", ("residual_head.",)),
                            ("shared_motion", ("latent_stem.", "latent_blocks.", "slot_project.")),
                        ):
                            squares = [p.grad.detach().float().square().sum() for name, p in named
                                       if name.startswith(prefixes) and p.grad is not None]
                            if squares:
                                activity_gradient_norms[f"grad_preclip/{group}"] = float(torch.stack(squares).sum().sqrt())
                        if event_contract is not None:
                            squares = [p.grad.detach().float().square().sum() for p in lora_parameters
                                       if p.grad is not None]
                            if squares:
                                activity_gradient_norms["grad_preclip/lora"] = float(torch.stack(squares).sum().sqrt())
                    if separate_mask_clip:
                        accelerator.unscale_gradients(optimizer)
                        gradient_norm, clip_diagnostics = clip_gradient_groups(
                            motion_clip_parameters, mask_clip_parameters,
                            float(train_cfg["max_grad_norm"]),
                            task_object=task_object_clip_parameters,
                        )
                        activity_gradient_norms.update(clip_diagnostics)
                    else:
                        gradient_norm = accelerator.clip_grad_norm_(
                            trainable, float(train_cfg["max_grad_norm"])
                        )
                if (event_contract is not None or fresh_trajectory) and accelerator.sync_gradients and not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError("nonfinite training gradient; refusing optimizer update and coverage credit")
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if coverage is not None:
                pending_coverage_ids.extend(raw["sample_id"])
            if not accelerator.sync_gradients:
                continue
            if coverage is not None:
                consumed_ids = gather_object(pending_coverage_ids)
                coverage.consume(consumed_ids, optimizer_step_applied=not accelerator.optimizer_step_was_skipped)
                pending_coverage_ids.clear()
            step += 1
            global_update_losses = {}
            global_update_samples = 0
            if update_loss_accumulator is not None:
                if step == start_step + 1 or step % log_every == 0 or step <= 8:
                    rank_keys = gather_object([sorted(update_loss_accumulator.sums)])
                    loss_keys = sorted({key for keys in rank_keys for key in keys})
                    reduced_losses = accelerator.reduce(update_loss_accumulator.vector(loss_keys), reduction="sum")
                    if is_main:
                        global_update_samples = int(reduced_losses[-1].item())
                        global_update_losses = dict(zip(loss_keys, (reduced_losses[:-1] / reduced_losses[-1]).tolist()))
                update_loss_accumulator.reset()
            if exact_resume:
                local_stop = torch.tensor(
                    int(stop_signal["number"] is not None),
                    device=device,
                    dtype=torch.int32,
                )
                any_stop = accelerator.reduce(local_stop, reduction="max")
                if int(any_stop.item()) and stop_signal["number"] is None:
                    stop_signal["number"] = -1
            if is_main and (step == start_step + 1 or step % log_every == 0 or (unified_campaign and step <= 8)):
                record = {
                    "step": step,
                    "loss/total": float(loss.detach()),
                    "loss/flow_matching": float(outputs["flow_matching"].detach()),
                    "loss/aux_total": float(aux["aux_total"].detach()),
                    "loss/flow": float(aux["flow"].detach()),
                    "loss/gate": float(aux["gate"].detach()),
                    "loss/visibility": float(aux["visibility"].detach()),
                    "loss/object_mask": float(aux["object_mask"].detach()),
                    "loss/slot_mask": float(aux["slot_mask"].detach()),
                    "loss/position": float(aux["position"].detach()),
                    "aux_ramp": ramp,
                    "sigma/mean": float(outputs["sigmas"].mean()),
                    "grad_norm": float(gradient_norm),
                    "lr/lora": scheduler.get_last_lr()[0],
                    "lr/head": scheduler.get_last_lr()[1],
                    "elapsed_seconds": time.time() - started,
                    **activity_gradient_norms,
                    **minimal5_gradients,
                    **({"coverage": coverage.summary()} if coverage is not None else {}),
                }
                if device.type == "cuda":
                    record.update({
                        "cuda/current_memory_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                        "cuda/max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                        "cuda/max_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                    })
                if activity_gradient_norms and not separate_mask_clip:
                    record["grad_clip_multiplier"] = min(1.0, float(train_cfg["max_grad_norm"]) / (float(gradient_norm) + 1e-6))
                for key, value in aux.items():
                    if key.startswith("unified_"):
                        record[f"loss/{key}"] = float(value.detach())
                    if key.startswith("minimal5_"):
                        record[f"loss/{key}"] = float(value.detach())
                    if key.startswith("event_"):
                        record[f"loss/{key}"] = float(value.detach())
                    if key.startswith("coarse_"):
                        record[f"loss/{key}"] = float(value.detach())
                        if key == "coarse_motion_total":
                            record[f"weighted_loss/{key}"] = float(value.detach())
                    if key.startswith("trajectory_segment"):
                        record[f"loss/{key}"] = float(value.detach())
                    if key.startswith(("partition_", "support_", "activity_", "dense_motion_mask", "task_object_slot_", "task_object_identity", "task_object_affine_", "task_object_flow_")) or key in ("object_activity", "pixel_activity", "slot_affine", "object_displacement", "moving_direction", "moving_parallel_magnitude", "object_motion_direction", "object_motion_parallel_magnitude", "gated_flow_charbonnier", "task_object_slot_activity", "task_object_identity"):
                        record[f"loss/{key}"] = float(value.detach())
                for key in ("object_activity", "pixel_activity", "dense_motion_mask", "task_object_slot_activity", "task_object_identity", "gated_flow_charbonnier", "slot_affine", "flow", "trajectory_segment", "moving_direction", "moving_parallel_magnitude", "object_motion_direction", "object_motion_parallel_magnitude", "task_object_affine_displacement", "task_object_affine_direction", "task_object_affine_parallel_magnitude", "task_object_affine_trajectory", "task_object_flow_displacement", "task_object_flow_direction", "task_object_flow_parallel_magnitude", "task_object_flow_trajectory"):
                    if key in aux:
                        if loss_cfg.get("profile") == "minimal5_direction_v1":
                            continue
                        coefficient = float(loss_cfg.get("flow_charbonnier" if key == "flow" else key, 0.0))
                        record[f"weighted_loss/{key}"] = coefficient * float(aux[key].detach())
                if unified_campaign:
                    record.update({f"loss/{key}": value for key, value in global_update_losses.items()})
                    for key in list(record):
                        if key.startswith("weighted_loss/"):
                            loss_key = key.removeprefix("weighted_loss/")
                            if loss_key in global_update_losses:
                                coefficient = float(loss_cfg.get("flow_charbonnier" if loss_key == "flow" else loss_key, 0.))
                                record[key] = coefficient * global_update_losses[loss_key]
                    record["loss_aggregation"] = "global_update_sample_mean"
                    record["loss_global_samples"] = global_update_samples
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
            if eval_every and step % eval_every == 0:
                summary = validate(step, "validation", float(loss.detach()))
                if train_cfg.get("raw_motion_guard", False):
                    from motion_jepa.flow_dit_collapse import raw_motion_guard
                    if raw_guard_initial is None:
                        # Exact resume retains the original step-zero baseline.
                        rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
                        raw_guard_initial = next((row for row in rows if row.get("step") == 0), None)
                        if raw_guard_initial is None:
                            raise ValueError("raw guard requires the original step-zero validation")
                    guard = raw_motion_guard(raw_guard_initial, summary)
                    if is_main:
                        with (output_dir / "raw_guard.jsonl").open("a") as handle:
                            handle.write(json.dumps(dict(step=step, **guard), sort_keys=True) + "\n")
                    if not guard["passed"]:
                        if is_main:
                            print("[raw-motion-guard] " + json.dumps(guard), flush=True)
                        save_checkpoint(step)
                        done = True
                        break
                # Every rank holds the same summary, so this branch is taken
                # identically everywhere and save_checkpoint's barrier is safe.
                if summary:
                    object_epe = summary.get("object_epe_norm_micro", float("nan"))
                    if math.isfinite(object_epe):
                        best_object_epe_norm_micro = min(
                            best_object_epe_norm_micro, object_epe
                        )
                    current, hallucination, within_budget, improved = validation_selection(
                        summary, best=best_metric, selection_metric=selection_metric,
                        static_guard_metric=static_guard_metric,
                        static_budget=static_budget, min_delta=min_delta,
                    )
                    if train_cfg.get("selection_guard_record_only", False):
                        within_budget = True
                        improved = math.isfinite(current) and current < best_metric * (1.0-min_delta)
                    if is_main and math.isfinite(hallucination) and not within_budget:
                        print(
                            f"[select] rejecting step {step}: invents "
                            f"guard value {hallucination:.3f} ({static_guard_metric}) "
                            f"(budget {static_budget:.3f})",
                            flush=True,
                        )
                    if improved:
                        best_metric = current
                        stale_validations = 0
                        save_checkpoint(step, best=True)
                    else:
                        stale_validations += 1
                        if is_main:
                            print(
                                f"[early-stop] no improvement for {stale_validations}"
                                f"/{patience or '-'} validations "
                                f"(best {best_metric:.6f}, now {current:.6f})",
                                flush=True,
                            )
                    # Every rank holds the same summary, so this decision is
                    # identical everywhere and the loop exits in lockstep.
                    if patience and stale_validations >= patience:
                        if is_main:
                            print(
                                f"[early-stop] stopping at step {step}: "
                                f"{patience} validations without a "
                                f"{min_delta:.1%} improvement",
                                flush=True,
                            )
                        save_checkpoint(step)
                        done = True
                        break
            if step % save_every == 0 or step >= stop_at_step or (exact_resume and stop_signal["number"] is not None):
                save_checkpoint(step)
            if step >= stop_at_step or (exact_resume and stop_signal["number"] is not None):
                done = True
                break

    accelerator.wait_for_everyone()
    if is_main:
        (output_dir / "done.json").write_text(
            json.dumps(
                {
                    "expert": expert,
                    "steps": step,
                    "max_steps": max_steps,
                    "stop_at_step": stop_at_step,
                    "target_complete": step >= max_steps and (coverage is None or coverage.complete),
                    **({"consumed_coverage": coverage.summary(),
                        "completion_blocked_by_coverage": step >= max_steps and not coverage.complete}
                       if coverage is not None else {}),
                    "label_source": label_source,
                    "selection_metric": selection_metric,
                    "static_guard_metric": static_guard_metric,
                    "static_magnitude_budget_px": static_budget,
                    "early_stop_min_delta": min_delta,
                    "best_selection_metric": (
                        best_metric if math.isfinite(best_metric) else None
                    ),
                    # This legacy field used to (incorrectly) duplicate the
                    # selection metric even when selection_metric was
                    # relative_epe_moving. Keep the key for consumers, but
                    # write the metric named by the key.
                    "best_object_epe_norm_micro": (
                        best_object_epe_norm_micro
                        if math.isfinite(best_object_epe_norm_micro)
                        else None
                    ),
                    "stopped_early": bool(patience and stale_validations >= patience),
                    "stale_validations": stale_validations,
                    **({"resume_protocol": EXACT_RESUME_PROTOCOL,
                        "stopped_for_signal": stop_signal["number"]} if exact_resume else {}),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        metrics_file.close()
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if step >= max_steps and coverage is not None and not coverage.complete:
        raise RuntimeError(f"training budget exhausted before required scene coverage: {coverage.summary()}")


if __name__ == "__main__":
    main()
