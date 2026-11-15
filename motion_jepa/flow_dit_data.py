"""Dataset for Stage-1 WanFlowDiT training.

Reads two caches and never touches HDF5 at train time:

* ``target_cache_dir`` -- the Stage-1 ground-truth target cache.  Reused
  verbatim so the supervision is byte-identical to what every baseline in the
  comparison was scored against.
* ``flow_cache_dir`` -- the new per-sample flow/image latents written by
  ``scripts/cache_stage1_flowdit.py``.

Prompt embeddings live in a *shared* table keyed by caption hash: the grounded
manifests contain far fewer unique captions than rows, so storing them per
sample would cost roughly ten times more disk.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import SCENARIO_TO_ID
from .data import ManifestRow, cache_path, read_jsonl
from .flow_dit_bridge import BRIDGE_LABEL_SOURCE, validate_bridge_target
from .bridge_mover_selection import CONDITION_SOURCE as BRIDGE_TASK_OBJECT_MASK_SOURCE
from .caption_overrides import load_caption_overrides, validate_override_prompt_embedding


CACHE_FORMAT_VERSION = 1
SUPERVISION_SCHEMA_VERSION = 1
SUPERVISION_MODE = "valid_partition_v1"
SUPERVISION_TENSOR_KEYS = (
    "flow_valid", "visibility_valid", "static_certified", "motion_target",
    "motion_valid", "supervision_object_mask", "content_valid",
)
LABEL_SOURCE = (
    "official_physion_hdf5_instance_id_plus_"
    "mesh_pose_or_flex_particle_camera_correspondence"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def instance_map_sha256(instance_id: torch.Tensor) -> str:
    """Hash categorical IDs, independently of their original integer dtype."""
    value = instance_id.detach().cpu().long().contiguous()
    digest = hashlib.sha256(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def first_frame_tensor_sha256(value: torch.Tensor) -> str:
    """Hash the CHW uint8 first frame exactly as the SAM cache producer does."""
    value = value.detach().cpu().to(torch.uint8).contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def validate_supervision_sidecar(
    sidecar: dict[str, Any],
    target: dict[str, Any],
    *,
    sample_id: str,
    target_sha256: str,
    horizon_seconds: list[float],
    height: int,
    width: int,
    static_threshold_px: float = 0.5,
    moving_threshold_px: float = 1.0,
) -> None:
    """Fail closed on stale labels, geometry, IDs or time in a supervision sidecar.

    These masks only refine decoded-flow supervision. They do not rewrite or
    mask the existing latent target, whose legacy contract remains unchanged.
    """
    prefix = f"{sample_id}: supervision sidecar"
    if (sidecar.get("schema_version") != SUPERVISION_SCHEMA_VERSION
            or sidecar.get("kind") != "stage1_supervision"
            or sidecar.get("supervision_mode") != SUPERVISION_MODE):
        raise ValueError(f"{prefix} has an unsupported schema/mode")
    if sidecar.get("sample_id") != sample_id or target.get("sample_id") != sample_id:
        raise ValueError(f"{prefix} sample_id mismatch")
    if sidecar.get("label_source") != target.get("label_source"):
        raise ValueError(f"{prefix} label_source mismatch")
    if target.get("label_source") not in {
        LABEL_SOURCE, "phyco_kubric_raft_flow_plus_rendered_instance_segmentation", BRIDGE_LABEL_SOURCE
    }:
        raise ValueError(f"{prefix} unsupported label source validity contract")
    if target.get("label_source") == BRIDGE_LABEL_SOURCE:
        validate_bridge_target(target)
    provenance = sidecar.get("provenance", {})
    if provenance.get("source_target_sha256") != target_sha256:
        raise ValueError(f"{prefix} source target SHA256 mismatch")
    if provenance.get("instance_map_sha256") != instance_map_sha256(target["instance_id"]):
        raise ValueError(f"{prefix} instance ID contract mismatch")
    if sidecar.get("spatial_shape") != [height, width]:
        raise ValueError(f"{prefix} spatial shape mismatch")
    times = sidecar.get("horizon_seconds", [])
    if (not times or not all(math.isfinite(float(t)) and float(t) > 0 for t in times)
            or any(float(a) >= float(b) for a, b in zip(times, times[1:]))):
        raise ValueError(f"{prefix} invalid horizon_seconds")
    if len(times) != len(horizon_seconds) or any(
        abs(float(a) - float(b)) > 1e-6 for a, b in zip(times, horizon_seconds)
    ):
        raise ValueError(f"{prefix} horizon_seconds mismatch")
    if target.get("horizon_seconds") is not None and (
        len(target["horizon_seconds"]) != len(times)
        or any(abs(float(a) - float(b)) > 1e-6
               for a, b in zip(target["horizon_seconds"], times))
    ):
        raise ValueError(f"{prefix} target horizon_seconds mismatch")
    for name, expected in (("static_threshold_px", static_threshold_px),
                           ("moving_threshold_px", moving_threshold_px)):
        observed = float(sidecar.get(name, float("inf")))
        if not math.isfinite(observed) or abs(observed - expected) > 1e-8:
            raise ValueError(f"{prefix} {name} mismatch")
    temporal_shape = (len(times), 1, height, width)
    spatial_shape = (1, height, width)
    if tuple(target["flow"].shape) != (len(times), 2, height, width):
        raise ValueError(f"{prefix} target flow shape mismatch")
    for key in SUPERVISION_TENSOR_KEYS:
        value = sidecar.get(key)
        expected = spatial_shape if key in {"supervision_object_mask", "content_valid"} else temporal_shape
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != expected
                or not torch.isfinite(value).all()
                or not ((value == 0) | (value == 1)).all()):
            raise ValueError(f"{prefix} invalid {key}")
    valid = sidecar["flow_valid"].bool()
    available = (target["visibility"] > 0.5) & torch.isfinite(target["flow"]).all(1, keepdim=True)
    available &= sidecar["content_valid"].bool()[None]
    if target.get("flow_valid") is not None:
        available &= target["flow_valid"] > 0.5
    if target.get("label_source") == LABEL_SOURCE:
        available &= (target["instance_id"] > 0)[None]
    if target.get("content_valid") is not None and not torch.equal(
        sidecar["content_valid"].bool(), target["content_valid"].bool()
    ):
        raise ValueError(f"{prefix} content_valid contract mismatch")
    if (valid & ~available).any():
        raise ValueError(f"{prefix} marks unknown/nonfinite correspondence valid")
    if not torch.equal(sidecar["supervision_object_mask"].bool(), target["instance_id"] > 0):
        raise ValueError(f"{prefix} supervision union is not the reference instance union")
    magnitude = target["flow"].float().square().sum(1, keepdim=True).sqrt()
    static = sidecar["static_certified"].bool()
    motion = sidecar["motion_target"].bool()
    if (static & (~valid | (magnitude > static_threshold_px))).any():
        raise ValueError(f"{prefix} certifies invalid or nonstatic flow")
    if not torch.equal(motion, valid & (magnitude > moving_threshold_px)):
        raise ValueError(f"{prefix} motion_target mismatch")
    if not torch.equal(sidecar["motion_valid"].bool(), static | motion):
        raise ValueError(f"{prefix} motion_valid mismatch")
    # Legacy visibility false conflates occlusion, missing correspondences and
    # failed optical flow. This version only knows reliable *visible* labels.
    if not torch.equal(sidecar["visibility_valid"].bool(), valid):
        raise ValueError(f"{prefix} visibility_valid must use known-visible support")
    table, affine_valid = sidecar.get("affine_by_instance"), sidecar.get("affine_valid")
    if (not isinstance(table, torch.Tensor) or table.ndim != 3 or table.shape[0] < 1
            or table.shape[1:] != (len(times), 6) or not torch.isfinite(table).all()
            or not isinstance(affine_valid, torch.Tensor)
            or tuple(affine_valid.shape) != tuple(table.shape[:2])
            or affine_valid.dtype != torch.bool or affine_valid[0].any()):
        raise ValueError(f"{prefix} invalid affine contract")
    ids = target["instance_id"].long()
    if int(ids.max()) >= table.shape[0]:
        raise ValueError(f"{prefix} affine table cannot address reference IDs")


def caption_key(caption: str) -> str:
    """Stable key for the shared prompt-embedding table."""
    return hashlib.sha256(caption.strip().encode("utf-8")).hexdigest()


def load_manifest_rows(path: str | Path) -> list[ManifestRow]:
    return read_jsonl(path)


def read_captions(path: str | Path) -> dict[str, str]:
    """Map ``sample_id -> caption`` straight from the manifest JSON.

    ``ManifestRow`` gained a ``caption`` field so ``read_jsonl`` keeps it, but
    reading the raw JSON here as well means a stale checkout cannot silently
    train on empty captions.
    """
    captions: dict[str, str] = {}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            captions[record["sample_id"]] = record.get("caption", "")
    return captions


def load_exclusions(path: str | Path) -> dict[str, str]:
    """Read a versioned exclusions manifest: ``sample_id -> reason``.

    Every id in the mapping is excluded, whatever the reason says -- resolved
    entries (e.g. rebuilt targets) belong under other keys, not in the mapping.
    Accepts a top-level mapping or a wrapper named ``exclusions``/``mapping``,
    so the file can carry reconciliation counters alongside. Values may be
    reason strings or objects with a ``reason``/``category`` field.
    """
    payload = json.loads(Path(path).read_text())
    # The restored PhyCo ledger uses {schema, mapping: {id: {category}}}.
    # Treating that wrapper as sample IDs silently ignored all exclusions.
    table = payload.get("exclusions", payload.get("mapping", payload))
    result: dict[str, str] = {}
    for key, value in table.items():
        if isinstance(value, str):
            result[key] = value
        elif isinstance(value, dict) and "reason" in value:
            result[key] = str(value["reason"])
        elif isinstance(value, dict) and "category" in value:
            result[key] = str(value["category"])
    return result


def load_supervision_mover_ids(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the audited causal-metadata mapping from sample IDs to mover IDs.

    PhyCo's historical ``object_mask`` was inferred from RAFT magnitude and can
    include static walls or supports.  The rendered ``instance_id`` map is
    exact, so the new contract selects its participating IDs from trusted
    simulator object types instead of treating every rendered instance as a
    moving object.
    """
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") not in {
        "phyco_supervision_mover_ids_v1", "stage1_supervision_mover_ids_v1"
    }:
        raise ValueError("unsupported supervision mover-ID manifest schema")
    mapping = payload.get("mapping")
    if not isinstance(mapping, dict) or int(payload.get("rows", -1)) != len(mapping):
        raise ValueError("supervision mover-ID manifest row count mismatch")
    result: dict[str, dict[str, Any]] = {}
    for sample_id, record in mapping.items():
        if not isinstance(record, dict) or record.get("sample_id") != sample_id:
            raise ValueError(f"invalid supervision mover-ID record: {sample_id}")
        identifiers = record.get("mover_instance_ids")
        if (
            not isinstance(identifiers, list)
            or not identifiers
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in identifiers)
            or identifiers != sorted(set(identifiers))
        ):
            raise ValueError(f"invalid mover instance IDs: {sample_id}")
        result[sample_id] = record
    return result


def resolve_target_path(target_cache_dir: str | Path, sample_id: str) -> Path:
    """Find a target in either cache layout.

    Physion nests targets under ``<dir>/targets/<id>.pt``; the PhyCo cache is
    flat ``<dir>/<id>.pt``. Trying the nested form first keeps Physion's
    behaviour byte-identical.
    """
    nested = cache_path(target_cache_dir, "targets", sample_id)
    if nested.is_file():
        return nested
    return Path(target_cache_dir) / f"{sample_id}.pt"


class FlowDiTDataset(Dataset):
    """Stage-1 flow-diffusion samples."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        flow_cache_dir: str | Path,
        target_cache_dir: str | Path,
        prompt_table_path: str | Path,
        prompt_table_mmap: bool = False,
        width: int = 160,
        height: int = 160,
        limit: int = 0,
        strict: bool = True,
        require_caption: bool = True,
        label_source: str = LABEL_SOURCE,
        affine_cache_dir: str | Path = "",
        emit_instance_masks: int = 0,
        horizon_seconds: list[float] | None = None,
        exclusions: str | Path | None = None,
        supervision_mode: str = "legacy",
        supervision_cache_dir: str | Path = "",
        supervision_affine: bool = False,
        supervision_static_threshold_px: float = 0.5,
        supervision_moving_threshold_px: float = 1.0,
        caption_overrides: str | Path | None = None,
        prompt_pad_length: int | None = None,
        condition_instance_cache_dir: str | Path = "",
        condition_mask_source: str = "",
        supervision_instance_masks_from_condition: bool = False,
        supervision_instance_cache_dir: str | Path = "",
        supervision_instance_mask_source: str = "",
        temporal_subsample_indices: list[int] | None = None,
        supervision_from_target: bool = False,
        supervision_mover_ids_manifest: str | Path = "",
    ) -> None:
        self.rows = load_manifest_rows(manifest)
        self.captions = read_captions(manifest)
        # Original captions remain the source/target contract. Only the prompt
        # lookup changes; a reviewed edit must never weaken Bridge provenance.
        self.original_captions = self.captions
        self.conditioning_captions = self.captions
        self.caption_override_contract: dict[str, Any] | None = None
        self._caption_override_ids: set[str] = set()
        overrides = None
        if caption_overrides:
            if isinstance(prompt_pad_length, bool) or not isinstance(prompt_pad_length, int) or prompt_pad_length < 1:
                raise ValueError("caption overrides require an explicit positive training prompt_pad_length")
            overrides = load_caption_overrides(
                caption_overrides, manifest=manifest, original_captions=self.original_captions,
            )
            self.conditioning_captions = {**self.captions, **overrides.captions}
            self._caption_override_ids = set(overrides.captions)
        # Known-bad rows leave through this versioned manifest, never through
        # runtime skipping: the training distribution stays a function of files
        # on disk, and a *new* corrupt sample still fails loudly in _validate.
        self.exclusion_counts: dict[str, int] = {}
        if exclusions:
            table = load_exclusions(exclusions)
            kept: list[ManifestRow] = []
            for row in self.rows:
                reason = table.get(row.sample_id)
                if reason is None:
                    kept.append(row)
                else:
                    self.exclusion_counts[reason] = (
                        self.exclusion_counts.get(reason, 0) + 1
                    )
            self.rows = kept
        if limit:
            self.rows = self.rows[:limit]
        self.flow_cache_dir = Path(flow_cache_dir)
        self.target_cache_dir = Path(target_cache_dir)
        # Optional: the least-squares affine fit of each instance's true flow,
        # keyed by instance_id. Absent means the direct slot_params supervision
        # is simply off, which is the pre-existing behaviour.
        self.affine_cache_dir = Path(affine_cache_dir) if affine_cache_dir else None
        # When positive: also emit per-instance binary masks (padded/area-capped
        # to this many slots) and the 2-channel union+boundary map, for the
        # oracle-mask arm. The instance_id map is a first-frame property -- no
        # future information -- but it is ground truth, so any arm consuming it
        # is reported separately from the first-frame+caption baseline.
        self.emit_instance_masks = int(emit_instance_masks)
        self.condition_instance_cache_dir = (
            Path(condition_instance_cache_dir) if condition_instance_cache_dir else None
        )
        self.condition_mask_source = str(condition_mask_source)
        self.supervision_instance_masks_from_condition = bool(
            supervision_instance_masks_from_condition
        )
        self.supervision_instance_cache_dir = (
            Path(supervision_instance_cache_dir)
            if supervision_instance_cache_dir else None
        )
        self.supervision_instance_mask_source = str(
            supervision_instance_mask_source
        )
        # Per-sample temporal span, so two datasets with different physical
        # times but the same horizon count can still share mixed batches.
        self.horizon_seconds = (
            [float(v) for v in horizon_seconds] if horizon_seconds else None
        )
        self.temporal_subsample_indices = (
            [int(value) for value in temporal_subsample_indices]
            if temporal_subsample_indices is not None else None
        )
        if self.temporal_subsample_indices is not None and (
            not self.temporal_subsample_indices
            or self.temporal_subsample_indices != sorted(set(self.temporal_subsample_indices))
            or self.temporal_subsample_indices[0] < 0
            or self.horizon_seconds is None
            or len(self.temporal_subsample_indices) != len(self.horizon_seconds)
        ):
            raise ValueError(
                "temporal_subsample_indices must be sorted unique nonnegative indices "
                "matching horizon_seconds"
            )
        self.supervision_from_target = bool(supervision_from_target)
        self.supervision_mover_ids = (
            load_supervision_mover_ids(supervision_mover_ids_manifest)
            if supervision_mover_ids_manifest else None
        )
        if self.supervision_mover_ids is not None:
            missing_movers = [row.sample_id for row in self.rows
                              if row.sample_id not in self.supervision_mover_ids]
            if missing_movers:
                raise ValueError(
                    "supervision mover-ID manifest misses selected sample: "
                    f"{missing_movers[0]}"
                )
            if not self.supervision_from_target:
                raise ValueError(
                    "supervision_mover_ids_manifest requires supervision_from_target"
                )
        if self.condition_instance_cache_dir is not None:
            if not self.emit_instance_masks:
                raise ValueError("condition_instance_cache_dir requires emit_instance_masks > 0")
            if not self.condition_mask_source:
                raise ValueError("condition_instance_cache_dir requires condition_mask_source")
        if self.supervision_instance_masks_from_condition:
            if self.condition_instance_cache_dir is None:
                raise ValueError(
                    "supervision_instance_masks_from_condition requires condition_instance_cache_dir"
                )
            if label_source != BRIDGE_LABEL_SOURCE:
                raise ValueError(
                    "supervision_instance_masks_from_condition is only valid for Bridge proxy labels"
                )
        if self.supervision_instance_cache_dir is not None:
            if not self.emit_instance_masks:
                raise ValueError(
                    "supervision_instance_cache_dir requires emit_instance_masks > 0"
                )
            if label_source != BRIDGE_LABEL_SOURCE:
                raise ValueError(
                    "external task-object supervision masks are only valid for Bridge"
                )
            if not self.supervision_instance_mask_source:
                raise ValueError(
                    "supervision_instance_cache_dir requires supervision_instance_mask_source"
                )
            if self.supervision_instance_mask_source != BRIDGE_TASK_OBJECT_MASK_SOURCE:
                raise ValueError(
                    "unsupported external Bridge task-object supervision source"
                )
            if self.supervision_instance_masks_from_condition:
                raise ValueError(
                    "choose either the condition masks or a separate cache for supervision"
                )
        elif self.supervision_instance_mask_source:
            raise ValueError(
                "supervision_instance_mask_source requires supervision_instance_cache_dir"
            )
        self.width = int(width)
        self.height = int(height)
        self.label_source = label_source
        if supervision_mode not in {"legacy", SUPERVISION_MODE}:
            raise ValueError(f"unknown supervision_mode: {supervision_mode!r}")
        self.supervision_mode = supervision_mode
        if label_source == BRIDGE_LABEL_SOURCE and supervision_mode != SUPERVISION_MODE:
            raise ValueError("Bridge requires valid_partition_v1; legacy empty-mask/unknown labels are unsafe")
        self.supervision_cache_dir = Path(supervision_cache_dir) if supervision_cache_dir else None
        self.supervision_affine = bool(supervision_affine)
        self.supervision_static_threshold_px = float(supervision_static_threshold_px)
        self.supervision_moving_threshold_px = float(supervision_moving_threshold_px)
        self._target_hashes: dict[str, tuple[tuple[int, int, int], str]] = {}
        if not 0 <= self.supervision_static_threshold_px < self.supervision_moving_threshold_px:
            raise ValueError("supervision thresholds must satisfy 0 <= static < moving")
        if supervision_mode == "legacy":
            if self.supervision_cache_dir is not None or self.supervision_affine or self.supervision_from_target:
                raise ValueError("supervision cache/runtime/affine requires explicit supervision_mode")
        elif (
            (self.supervision_cache_dir is None) == (not self.supervision_from_target)
            or self.horizon_seconds is None
        ):
            raise ValueError(
                "valid_partition_v1 requires exactly one of supervision_cache_dir or "
                "supervision_from_target, plus horizon_seconds"
            )
        if self.supervision_from_target and self.supervision_affine:
            raise ValueError("runtime target supervision does not refit affine labels")

        prompt_table_sha = None
        if overrides is not None:
            # Hash the exact bytes loaded, so an external table replacement
            # cannot bind one file's checksum to another file's embeddings.
            import io
            prompt_bytes = Path(prompt_table_path).read_bytes()
            prompt_table_sha = hashlib.sha256(prompt_bytes).hexdigest()
            payload = torch.load(io.BytesIO(prompt_bytes), map_location="cpu", weights_only=True)
            del prompt_bytes
        else:
            payload = torch.load(prompt_table_path, map_location="cpu", weights_only=True,
                                 **({"mmap": True} if prompt_table_mmap else {}))
        self.prompt_table: dict[str, torch.Tensor] = payload["embeddings"]
        self.negative_key: str = payload["negative_key"]
        if overrides is not None:
            # Validate every reviewed entry now, including rows later excluded
            # or limited out. Missing new text must not fall back to old text.
            missing = [sid for sid, text in overrides.captions.items()
                       if caption_key(text) not in self.prompt_table]
            if missing:
                raise ValueError(f"caption override prompt embedding missing: {missing[0]}")
            for sid, text in overrides.captions.items():
                embedding = self.prompt_table[caption_key(text)]
                validate_override_prompt_embedding(embedding, pad_length=prompt_pad_length, label=sid)
            self.caption_override_contract = {
                **overrides.contract, "prompt_table_sha256": prompt_table_sha,
                "training_prompt_pad_length": prompt_pad_length,
            }

        if require_caption:
            missing = [row.sample_id for row in self.rows if not self.conditioning_captions.get(row.sample_id)]
            if missing:
                raise ValueError(
                    f"{len(missing)} manifest rows have no caption, first: {missing[0]}"
                )
        if strict:
            absent = [row.sample_id for row in self.rows if not self._paths_exist(row.sample_id)]
            if absent:
                raise FileNotFoundError(
                    f"{len(absent)} samples lack cache entries, first: {absent[0]}"
                )

    def _paths_exist(self, sample_id: str) -> bool:
        return (
            (self.flow_cache_dir / f"{sample_id}.pt").is_file()
            and resolve_target_path(self.target_cache_dir, sample_id).is_file()
            and (self.supervision_cache_dir is None or self.supervision_from_target
                 or (self.supervision_cache_dir / f"{sample_id}.pt").is_file())
            and (self.condition_instance_cache_dir is None
                 or (self.condition_instance_cache_dir / f"{sample_id}.pt").is_file())
            and (self.supervision_instance_cache_dir is None
                 or (self.supervision_instance_cache_dir / f"{sample_id}.pt").is_file())
        )

    def available_rows(self) -> list[ManifestRow]:
        return [row for row in self.rows if self._paths_exist(row.sample_id)]

    def __len__(self) -> int:
        return len(self.rows)

    def prompt_for(self, sample_id: str) -> torch.Tensor:
        return self.prompt_table[caption_key(self.conditioning_captions[sample_id])].float()

    def negative_prompt(self) -> torch.Tensor:
        return self.prompt_table[self.negative_key].float()

    def _supervision_instance_id(
        self, target: dict[str, Any], sample_id: str
    ) -> torch.Tensor:
        """Return exact rendered IDs filtered by the audited mover-ID contract."""
        labels = target["instance_id"].detach().cpu().long()
        if self.supervision_mover_ids is None:
            return labels
        record = self.supervision_mover_ids[sample_id]
        target_scenario = str(target.get("scenario", ""))
        record_scenario = str(record.get("scenario", ""))
        if target_scenario != record_scenario:
            raise ValueError(f"{sample_id}: mover-ID scenario mismatch")
        selected = torch.zeros_like(labels, dtype=torch.bool)
        missing = []
        for identifier in record["mover_instance_ids"]:
            belongs = labels == int(identifier)
            if not belongs.any():
                missing.append(identifier)
            selected |= belongs
        # Causal metadata describes every participating mover, including an
        # object that can be fully occluded or outside the first-frame crop.
        # Such an ID cannot have a first-frame mask.  Keep every visible mover
        # and ignore only absent IDs; rejecting a partially visible multi-object
        # scene makes a valid sample fail nondeterministically when it is first
        # reached by the distributed sampler.  A scene with no visible mover is
        # still invalid for first-frame-only Stage1 supervision.
        if not selected.any():
            raise ValueError(
                f"{sample_id}: no mover ID is visible in the first-frame rendered "
                f"instance map; absent IDs: {missing}"
            )
        # The legacy object_mask is a RAFT-threshold heuristic.  It may contain
        # false positives or miss a true mover, so it must not veto the trusted
        # metadata-to-rendered-ID contract.
        return torch.where(selected, labels, torch.zeros_like(labels))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        payload = torch.load(
            self.flow_cache_dir / f"{row.sample_id}.pt", map_location="cpu", weights_only=True
        )
        if payload.get("cache_format_version") != CACHE_FORMAT_VERSION:
            raise ValueError(f"{row.sample_id}: stale flow cache version")
        if payload.get("label_source") != self.label_source:
            raise ValueError(
                f"{row.sample_id}: unexpected label_source {payload.get('label_source')!r}"
            )
        target = torch.load(
            resolve_target_path(self.target_cache_dir, row.sample_id),
            map_location="cpu",
            weights_only=True,
        )
        payload, target = self._temporal_view(payload, target, row.sample_id)
        if self.label_source == BRIDGE_LABEL_SOURCE:
            validate_bridge_target(target)
            provenance = payload.get("provenance", {})
            if (row.video_path != target["video_path"] or self.original_captions[row.sample_id] != target["caption"]
                    or payload.get("first_frame_source") != target["video_path"]
                    or provenance.get("first_frame_sha256") != target["provenance"]["first_frame_sha256"]
                    or provenance.get("source_target_sha256") != sha256_file(
                        resolve_target_path(self.target_cache_dir, row.sample_id))):
                raise ValueError(f"{row.sample_id}: Bridge image/prompt/target cache input provenance mismatch")

        flow = target["flow"].float().clone()
        flow[:, 0].div_(self.width)
        flow[:, 1].div_(self.height)

        sample: dict[str, Any] = {
            "sample_id": row.sample_id,
            "scenario_name": row.scenario,
            # Vestigial: the old MotionJEPA conditioned on a scenario embedding;
            # WanFlowDiT conditions on captions and never reads this tensor.
            # PhyCo scenarios are not in the Physion table, hence the fallback.
            "scenario": torch.tensor(SCENARIO_TO_ID.get(row.scenario, -1), dtype=torch.long),
            "label_source": target.get("label_source", "unknown"),
            "flow_latent": payload["flow_latent"].float(),
            "flow_latent_is_placeholder": torch.tensor(
                payload.get("latent_semantics") == "constant_prior_placeholder_direct_only"),
            "image_cond_latent": payload["image_cond_latent"].float(),
            "first_frame": payload["first_frame"].float().div_(127.5).sub_(1.0),
            "prompt_embeds": self.prompt_for(row.sample_id),
            "flow": flow,
            "visibility": target["visibility"].float(),
            "object_mask": target["object_mask"].float(),
            "future_mask": target["future_mask"].float(),
            "supervision_object_identity_authoritative": torch.tensor(
                self.supervision_mover_ids is not None, dtype=torch.bool
            ),
        }
        if row.sample_id in self._caption_override_ids:
            sample["caption_override_prompt_pad_length"] = self.caption_override_contract["training_prompt_pad_length"]
        for key in ("instance_id", "centers", "center_valid", "object_ids"):
            if key in target:
                sample[key] = target[key]
        # PhyCo ships these in the target itself; Physion gets them from the
        # sidecar cache. Same dense-by-instance_id layout either way.
        if (not self.supervision_affine
                and "affine_by_instance" in target and "affine_valid" in target):
            affine_table = target["affine_by_instance"]
            affine_valid = target["affine_valid"]
            self._validate_affine(affine_table, affine_valid, row.sample_id)
            sample["affine_by_instance"] = affine_table.float()
            sample["affine_valid"] = affine_valid.bool()
        elif not self.supervision_affine and self.affine_cache_dir is not None:
            path = self.affine_cache_dir / f"{row.sample_id}.pt"
            if path.is_file():
                affine = torch.load(path, map_location="cpu", weights_only=True)
                if int(affine.get("schema_version", 0)) < 2:
                    raise ValueError(
                        f"{row.sample_id}: affine sidecar schema v1 has only "
                        "instance-level validity; regenerate it with "
                        "scripts/cache_stage1_affine.py"
                    )
                affine_table = affine["affine_by_instance"]
                affine_valid = affine["affine_valid"]
                self._validate_affine(affine_table, affine_valid, row.sample_id)
                sample["affine_by_instance"] = affine_table.float()
                sample["affine_valid"] = affine_valid.bool()
        if self.horizon_seconds is not None:
            sample["horizon_seconds"] = torch.tensor(self.horizon_seconds)
        if self.emit_instance_masks and "instance_id" in target:
            supervision_instance_id = self._supervision_instance_id(
                target, row.sample_id
            )
            masks, mask_ids, channels = build_instance_inputs(
                supervision_instance_id, self.emit_instance_masks
            )
            if self.condition_instance_cache_dir is None:
                # Historical oracle-mask contract: the same exact labels enter
                # the model and supervise it. Retained only for old arms.
                sample["instance_masks"] = masks
                sample["instance_mask_ids"] = mask_ids
                sample["instance_channels"] = channels
            else:
                # Keep labels and deployable first-frame conditions distinct.
                # GT masks may supervise object-vector terms, but only the
                # cached SAM-style proposal is allowed into the decoder/backend.
                condition_path = self.condition_instance_cache_dir / f"{row.sample_id}.pt"
                condition = torch.load(condition_path, map_location="cpu", weights_only=True)
                condition_masks = condition.get("instance_masks")
                condition_channels = condition.get("instance_channels")
                condition_labels = condition.get("instance_labels")
                expected_masks = (self.emit_instance_masks, self.height, self.width)
                valid_provenance = (
                    condition.get("sample_id") != row.sample_id
                    or condition.get("condition_mask_source", condition.get("source"))
                    != self.condition_mask_source
                    or condition.get("first_frame_sha256")
                    != first_frame_tensor_sha256(payload["first_frame"])
                )
                if isinstance(condition_labels, torch.Tensor):
                    if (
                        valid_provenance
                        or tuple(condition_labels.shape) != (self.height, self.width)
                        or not torch.isfinite(condition_labels).all()
                        or (condition_labels < 0).any()
                        or (condition_labels > self.emit_instance_masks).any()
                        or not torch.equal(condition_labels, condition_labels.long())
                    ):
                        raise ValueError(f"{row.sample_id}: invalid compact external condition instance cache")
                    condition_masks, _, condition_channels = build_instance_inputs(
                        condition_labels.long()[None], self.emit_instance_masks
                    )
                elif (
                    valid_provenance
                    or not isinstance(condition_masks, torch.Tensor)
                    or tuple(condition_masks.shape) != expected_masks
                    or not torch.isfinite(condition_masks).all()
                    or not ((condition_masks == 0) | (condition_masks == 1)).all()
                    or (condition_masks.sum(dim=0) > 1).any()
                    or not isinstance(condition_channels, torch.Tensor)
                    or tuple(condition_channels.shape) != (2, self.height, self.width)
                    or not torch.isfinite(condition_channels).all()
                    or not ((condition_channels == 0) | (condition_channels == 1)).all()
                    or not torch.equal(condition_channels[0].bool(), condition_masks.bool().any(0))
                ):
                    raise ValueError(f"{row.sample_id}: invalid external condition instance cache")
                if self.supervision_instance_masks_from_condition:
                    # Bridge has no rendered instance GT.  For the opt-in
                    # canonical SAM arm, use the same audited first-frame
                    # partition as the object-level label geometry while flow
                    # still supplies every motion/static target.  PhyCo is
                    # forbidden from entering this branch and keeps rendered GT.
                    sample["supervision_instance_masks"] = condition_masks.float()
                    sample["supervision_instance_mask_ids"] = torch.full(
                        (self.emit_instance_masks,), -1, dtype=torch.long
                    )
                else:
                    sample["supervision_instance_masks"] = masks
                    sample["supervision_instance_mask_ids"] = mask_ids
                sample["instance_masks"] = condition_masks.float()
                sample["instance_channels"] = condition_channels.float()
                sample["condition_mask_source"] = self.condition_mask_source
        if self.supervision_instance_cache_dir is not None:
            label_path = self.supervision_instance_cache_dir / f"{row.sample_id}.pt"
            label = torch.load(label_path, map_location="cpu", weights_only=True)
            label_values = label.get("instance_labels")
            if (
                label.get("sample_id") != row.sample_id
                or label.get("source", label.get("supervision_mask_source"))
                != self.supervision_instance_mask_source
                or label.get("first_frame_sha256")
                != first_frame_tensor_sha256(payload["first_frame"])
                or not isinstance(label_values, torch.Tensor)
                or tuple(label_values.shape) != (self.height, self.width)
                or not torch.isfinite(label_values).all()
                or (label_values < 0).any()
                or (label_values > self.emit_instance_masks).any()
                or not torch.equal(label_values, label_values.long())
                or not (label_values > 0).any()
            ):
                raise ValueError(
                    f"{row.sample_id}: invalid external task-object supervision cache"
                )
            label_masks, label_ids, _ = build_instance_inputs(
                label_values.long()[None], self.emit_instance_masks
            )
            sample["supervision_instance_masks"] = label_masks.float()
            sample["supervision_instance_mask_ids"] = label_ids
            sample["supervision_object_identity_authoritative"] = torch.tensor(
                True, dtype=torch.bool
            )
            sample["supervision_instance_mask_source"] = (
                self.supervision_instance_mask_source
            )
        if self.supervision_mode == SUPERVISION_MODE:
            if self.supervision_from_target:
                self._add_runtime_supervision(sample, target, row.sample_id)
            else:
                self._add_supervision(sample, target, row.sample_id)
        self._validate(sample)
        return sample

    def _temporal_view(
        self, payload: dict[str, Any], target: dict[str, Any], sample_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Select a fixed audited subset without duplicating immutable caches."""
        indices = self.temporal_subsample_indices
        if indices is None:
            return payload, target
        source_times = [float(value) for value in target.get("horizon_seconds", [])]
        if (
            not source_times
            or payload.get("horizon_seconds") != target.get("horizon_seconds")
            or indices[-1] >= len(source_times)
            or any(
                abs(source_times[index] - expected) > 5e-5
                for index, expected in zip(indices, self.horizon_seconds or [])
            )
        ):
            raise ValueError(f"{sample_id}: temporal subsample does not match cached physical times")
        selector = torch.tensor(indices, dtype=torch.long)
        latent_selector = torch.tensor([0] + [index + 1 for index in indices], dtype=torch.long)
        target = dict(target)
        for key in ("flow", "visibility", "future_mask", "affine_3d"):
            value = target.get(key)
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(source_times):
                target[key] = value.index_select(0, selector).contiguous()
        for key in ("affine_2d", "affine_by_instance", "affine_valid"):
            value = target.get(key)
            if isinstance(value, torch.Tensor) and value.ndim > 1 and value.shape[1] == len(source_times):
                target[key] = value.index_select(1, selector).contiguous()
        for key in ("centers", "center_valid"):
            value = target.get(key)
            if isinstance(value, torch.Tensor) and value.ndim > 1 and value.shape[1] == len(source_times) + 1:
                target[key] = value.index_select(1, latent_selector).contiguous()
        target["horizon_seconds"] = [source_times[index] for index in indices]
        payload = dict(payload)
        latent = payload.get("flow_latent")
        if (
            not isinstance(latent, torch.Tensor)
            or latent.ndim != 4
            or latent.shape[1] != len(source_times) + 1
        ):
            raise ValueError(f"{sample_id}: temporal subsample requires source+future flow latents")
        payload["flow_latent"] = latent.index_select(1, latent_selector).contiguous()
        payload["horizon_seconds"] = list(target["horizon_seconds"])
        if isinstance(payload.get("codec"), dict):
            payload["codec"] = {**payload["codec"], "latent_frames": len(indices) + 1}
        return payload, target

    def _add_runtime_supervision(
        self, sample: dict[str, Any], target: dict[str, Any], sample_id: str
    ) -> None:
        """Derive the same partition masks from the already loaded target."""
        if target.get("label_source") != self.label_source:
            raise ValueError(f"{sample_id}: target label source mismatch")
        if self.label_source == BRIDGE_LABEL_SOURCE:
            validate_bridge_target(target)
        flow = target["flow"].detach().cpu().float()
        visible = target["visibility"].detach().cpu() > 0.5
        finite = torch.isfinite(flow).all(1, keepdim=True)
        content = target.get("content_valid")
        if content is None:
            content = torch.ones(1, self.height, self.width, dtype=torch.bool)
        else:
            content = content.detach().cpu().bool()
        valid = visible & finite & content[None]
        if target.get("flow_valid") is not None:
            valid &= target["flow_valid"].detach().cpu().bool()
        if self.label_source == LABEL_SOURCE:
            valid &= (target["instance_id"].detach().cpu() > 0)[None]
        magnitude = flow.square().sum(1, keepdim=True).sqrt()
        mover_mask = self._supervision_instance_id(target, sample_id) > 0
        # The audited mover-ID contract is authoritative.  Dynamic RAFT pixels
        # outside those rendered movers are correspondence trails, occluders or
        # another unlabeled cause.  They must be unknown to every flow/vector/
        # trajectory loss, while trustworthy static background remains useful.
        if self.supervision_mover_ids is not None:
            outside_dynamic = (
                valid & ~mover_mask[None]
                & (magnitude > self.supervision_moving_threshold_px)
            )
            valid &= ~outside_dynamic
        static = valid & (magnitude <= self.supervision_static_threshold_px)
        moving = (
            valid & mover_mask[None]
            & (magnitude > self.supervision_moving_threshold_px)
        )
        values = {
            "flow_valid": valid,
            "visibility_valid": valid,
            "static_certified": static,
            "motion_target": moving,
            "motion_valid": static | moving,
            "supervision_object_mask": mover_mask,
            "content_valid": content,
        }
        for key, value in values.items():
            sample[key] = value.float()
        sample["supervision_mode"] = self.supervision_mode
        sample["supervision_source_target_sha256"] = "runtime_from_loaded_target_v1"
        sample["flow"] = torch.nan_to_num(
            sample["flow"], nan=0.0, posinf=0.0, neginf=0.0
        )

    def _add_supervision(
        self, sample: dict[str, Any], target: dict[str, Any], sample_id: str
    ) -> None:
        if target.get("label_source") != self.label_source:
            raise ValueError(f"{sample_id}: supervision target/latent label_source mismatch")
        path = resolve_target_path(self.target_cache_dir, sample_id)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._target_hashes.get(sample_id)
        if cached is None or cached[0] != signature:
            cached = (signature, sha256_file(path))
            self._target_hashes[sample_id] = cached
        sidecar = torch.load(
            self.supervision_cache_dir / f"{sample_id}.pt",  # type: ignore[operator]
            map_location="cpu", weights_only=True,
        )
        validate_supervision_sidecar(
            sidecar, target, sample_id=sample_id, target_sha256=cached[1],
            horizon_seconds=self.horizon_seconds,  # type: ignore[arg-type]
            height=self.height, width=self.width,
            static_threshold_px=self.supervision_static_threshold_px,
            moving_threshold_px=self.supervision_moving_threshold_px,
        )
        for key in SUPERVISION_TENSOR_KEYS:
            sample[key] = sidecar[key].float()
        if bool(sample["supervision_object_identity_authoritative"]):
            mover_union = sample["supervision_instance_masks"].bool().any(
                dim=0, keepdim=True
            )
            magnitude = target["flow"].float().square().sum(
                1, keepdim=True
            ).sqrt()
            outside_dynamic = (
                sample["flow_valid"].bool()
                & ~mover_union[None]
                & (magnitude > self.supervision_moving_threshold_px)
            )
            sample["flow_valid"] = (
                sample["flow_valid"].bool() & ~outside_dynamic
            ).float()
            sample["visibility_valid"] = (
                sample["visibility_valid"].bool() & ~outside_dynamic
            ).float()
            sample["motion_target"] = (
                sample["motion_target"].bool()
                & mover_union[None]
                & sample["flow_valid"].bool()
            ).float()
            sample["static_certified"] = (
                sample["static_certified"].bool()
                & sample["flow_valid"].bool()
            ).float()
            sample["motion_valid"] = (
                sample["motion_target"].bool()
                | sample["static_certified"].bool()
            ).float()
            sample["supervision_object_mask"] = mover_union.float()
        sample["supervision_mode"] = self.supervision_mode
        sample["supervision_source_target_sha256"] = cached[1]
        # Nonfinite targets carry no decoded-flow gradient. Existing caches
        # normally sanitize them; do not let NaN * 0 pollute this explicit mode.
        sample["flow"] = torch.nan_to_num(sample["flow"], nan=0.0, posinf=0.0, neginf=0.0)
        if self.supervision_affine:
            sample["affine_by_instance"] = sidecar["affine_by_instance"].float()
            sample["affine_valid"] = sidecar["affine_valid"].bool()

    @staticmethod
    def _validate_affine(
        table: torch.Tensor, valid: torch.Tensor, sample_id: str
    ) -> None:
        if table.ndim != 3 or table.shape[-1] != 6:
            raise ValueError(
                f"{sample_id}: affine_by_instance must be [instance,horizon,6], "
                f"got {tuple(table.shape)}"
            )
        if valid.ndim != 2 or tuple(valid.shape) != tuple(table.shape[:2]):
            raise ValueError(
                f"{sample_id}: affine_valid must be [instance,horizon] and match "
                f"the table; got {tuple(valid.shape)} vs {tuple(table.shape)}"
            )

    def _validate(self, sample: dict[str, Any]) -> None:
        expected = {
            "flow_latent": 4,
            # (C, 1, h, w): the Wan VAE keeps a temporal axis even for a single
            # conditioning frame, and every consumer relies on that axis being
            # present to broadcast across the flow timeline.
            "image_cond_latent": 4,
            "first_frame": 3,
            "flow": 4,
            "visibility": 4,
            "object_mask": 3,
            "future_mask": 4,
        }
        for key, dimensions in expected.items():
            tensor = sample[key]
            if tensor.ndim != dimensions or not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{sample['sample_id']} invalid {key}: shape={tuple(tensor.shape)}, "
                    f"finite={bool(torch.isfinite(tensor).all())}"
                )
        if sample["object_mask"].sum() < 1 and self.supervision_mode == "legacy":
            raise ValueError(f"{sample['sample_id']} has an empty object mask")


def collate(batch: list[dict[str, Any]], prompt_pad_length: int) -> dict[str, Any]:
    """Stack a batch, padding prompt embeddings to a fixed length.

    Mirrors the Stage-2 collate but with a much shorter pad (the grounded
    captions are short, so 512 would be almost all padding).
    """
    result: dict[str, Any] = {
        "sample_id": [item["sample_id"] for item in batch],
        "scenario_name": [item["scenario_name"] for item in batch],
        "label_source": [item["label_source"] for item in batch],
        "condition_mask_source": [item.get("condition_mask_source", "legacy_target_instance")
                                  for item in batch],
    }
    prompts = []
    for item in batch:
        prompt = item["prompt_embeds"]
        length = prompt.shape[0]
        if "caption_override_prompt_pad_length" in item:
            if item["caption_override_prompt_pad_length"] != prompt_pad_length:
                raise ValueError("caption override collate prompt_pad_length differs from the validated training contract")
            validate_override_prompt_embedding(prompt, pad_length=prompt_pad_length, label=item["sample_id"])
        if length > prompt_pad_length:
            prompt = prompt[:prompt_pad_length]
        elif length < prompt_pad_length:
            padding = prompt.new_zeros(prompt_pad_length - length, prompt.shape[1])
            prompt = torch.cat((prompt, padding), dim=0)
        prompts.append(prompt)
    result["prompt_embeds"] = torch.stack(prompts)

    for key in (
        "flow_latent",
        "image_cond_latent",
        "first_frame",
        "flow",
        "visibility",
        "object_mask",
        "future_mask",
        "scenario",
    ):
        result[key] = torch.stack([item[key] for item in batch])
    for key in (
        "instance_id", "centers", "center_valid", "object_ids",
        "affine_by_instance", "affine_valid",
        "instance_masks", "instance_mask_ids", "instance_channels",
        "supervision_instance_masks", "supervision_instance_mask_ids",
        "supervision_object_identity_authoritative",
        "horizon_seconds",
        "flow_latent_is_placeholder",
    ):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])
    modes = {item.get("supervision_mode", "legacy") for item in batch}
    if len(modes) != 1:
        raise ValueError(f"mixed supervision modes in one batch: {sorted(modes)}")
    if modes == {SUPERVISION_MODE}:
        for key in SUPERVISION_TENSOR_KEYS:
            if not all(key in item for item in batch):
                raise ValueError(f"mixed/missing supervision field: {key}")
            result[key] = torch.stack([item[key] for item in batch])
        result["supervision_mode"] = [item["supervision_mode"] for item in batch]
        result["supervision_source_target_sha256"] = [
            item["supervision_source_target_sha256"] for item in batch
        ]
    return result


def pad_prompt(prompt: torch.Tensor, prompt_pad_length: int) -> torch.Tensor:
    """Pad or truncate a single prompt embedding, for inference paths."""
    length = prompt.shape[0]
    if length > prompt_pad_length:
        return prompt[:prompt_pad_length]
    if length < prompt_pad_length:
        padding = prompt.new_zeros(prompt_pad_length - length, prompt.shape[1])
        return torch.cat((prompt, padding), dim=0)
    return prompt


def build_instance_inputs(
    instance_id: torch.Tensor, num_slots: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn the exact instance map into the oracle-mask inputs.

    Returns ``masks [num_slots, H, W]`` (binary, largest instances first, spare
    slots empty), ``mask_ids [num_slots]`` (the instance_id each slot carries,
    -1 for empty -- the direct slot-to-instance assignment the affine loss uses
    instead of a Hungarian solve), and ``channels [2, H, W]`` = union mask plus
    instance boundaries for the backbone. The boundary channel is what tells the
    generator "these are five separate bodies, not one blob" -- the union alone
    cannot.
    """
    ids = instance_id[0].long()
    present = [int(v) for v in torch.unique(ids) if int(v) > 0]
    # Largest first, so a cap drops the smallest fragments into background.
    present.sort(key=lambda v: -int((ids == v).sum()))
    present = present[:num_slots]
    height, width = ids.shape
    masks = torch.zeros(num_slots, height, width)
    mask_ids = torch.full((num_slots,), -1, dtype=torch.long)
    for slot, value in enumerate(present):
        masks[slot] = (ids == value).float()
        mask_ids[slot] = value
    union = (ids > 0).float()
    boundary = torch.zeros_like(ids, dtype=torch.bool)
    boundary[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    boundary[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    boundary[1:, :] |= ids[1:, :] != ids[:-1, :]
    boundary[:-1, :] |= ids[:-1, :] != ids[1:, :]
    boundary &= ids > 0
    channels = torch.stack((union, boundary.float()))
    return masks, mask_ids, channels


class JointDataset(torch.utils.data.Dataset):
    """Concatenate shape-compatible datasets into one shuffled pool.

    Mixed batches are the point: a batch may hold Physion rows (0.2-1.4s
    horizons, exact centers) next to PhyCo rows (0.57-4.0s, RAFT flow, no
    centers), so every optional key must be present on every sample or the
    collate would stack ragged lists. Children must therefore be constructed
    with ``horizon_seconds`` set, and rows missing kinematic supervision get
    explicit invalid placeholders -- zeros with ``center_valid`` all False --
    which the losses and metrics already treat as "no signal" rather than
    "zero motion".
    """

    def __init__(self, children: list[FlowDiTDataset], max_instances: int = 64) -> None:
        if not children:
            raise ValueError("JointDataset needs at least one child")
        for child in children:
            if child.horizon_seconds is None:
                raise ValueError(
                    "every child must carry horizon_seconds so mixed batches "
                    "stay self-describing"
                )
        counts = {len(c.horizon_seconds) for c in children}
        if len(counts) != 1:
            raise ValueError(f"children disagree on horizon count: {counts}")
        self.children = children
        self.max_instances = int(max_instances)
        self._offsets = []
        total = 0
        for child in children:
            self._offsets.append(total)
            total += len(child)
        self._total = total

    def __len__(self) -> int:
        return self._total

    @property
    def rows(self):
        """Concatenated manifest rows, matching the FlowDiTDataset attribute."""
        merged = []
        for child in self.children:
            merged.extend(child.rows)
        return merged

    def negative_prompt(self) -> torch.Tensor:
        # Verified bit-identical across the tables (same umT5 on ""), so the
        # first child's is the joint one.
        return self.children[0].negative_prompt()

    def __getitem__(self, index: int) -> dict[str, Any]:
        for child, offset in zip(reversed(self.children), reversed(self._offsets)):
            if index >= offset:
                sample = child[index - offset]
                break
        horizons = sample["horizon_seconds"].shape[0]
        if "centers" not in sample:
            sample["centers"] = torch.zeros(self.max_instances, horizons + 1, 2)
            sample["center_valid"] = torch.zeros(
                self.max_instances, horizons + 1, dtype=torch.bool
            )
        if "object_ids" not in sample:
            sample["object_ids"] = torch.full((self.max_instances,), -1, dtype=torch.long)
        if "instance_masks" in sample and "instance_mask_ids" not in sample:
            # Deployable SAM proposals do not have a trustworthy one-to-one
            # correspondence with rendered GT IDs.  Keep the conditioning
            # slots batch-compatible, but mark that correspondence unknown so
            # an affine/object-ID loss cannot accidentally consume it.
            sample["instance_mask_ids"] = torch.full(
                (sample["instance_masks"].shape[0],), -1, dtype=torch.long
            )
        if "instance_masks" in sample and "supervision_instance_masks" not in sample:
            # Bridge targets already use first-frame SAM IDs as both the
            # deployable condition and their available segmentation proxy.
            # PhyCo carries separate rendered supervision masks.  Expose one
            # rectangular label ABI across the mixed batch while preserving
            # that distinction.
            sample["supervision_instance_masks"] = sample["instance_masks"]
            sample["supervision_instance_mask_ids"] = sample["instance_mask_ids"]
        return sample


class FixedMixtureSampler(torch.utils.data.Sampler[int]):
    """Sample a JointDataset at explicit source fractions, independent of size.

    Every iterator has exactly ``num_samples`` draws. Source counts are fixed
    (up to integer rounding), while rows within a source are sampled with
    replacement and the combined order is shuffled. This keeps a data-source
    experiment from silently changing its treatment when one dataset grows.
    """

    def __init__(
        self,
        dataset: JointDataset,
        fractions: list[float],
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        if len(fractions) != len(dataset.children):
            raise ValueError("one mixture fraction is required per child dataset")
        if any(value <= 0 for value in fractions):
            raise ValueError("mixture fractions must all be positive")
        total = float(sum(fractions))
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"mixture fractions must sum to one, got {total}")
        if any(len(child) == 0 for child in dataset.children):
            raise ValueError("cannot sample from an empty child dataset")
        self.dataset = dataset
        self.fractions = [float(value) for value in fractions]
        self.num_samples = int(num_samples or len(dataset))
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        raw_counts = [self.num_samples * value for value in self.fractions]
        counts = [int(value) for value in raw_counts]
        remainder = self.num_samples - sum(counts)
        order = sorted(
            range(len(counts)),
            key=lambda index: raw_counts[index] - counts[index],
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1

        indices = []
        for child_index, (child, count) in enumerate(zip(self.dataset.children, counts)):
            local = torch.randint(len(child), (count,), generator=generator)
            offset = self.dataset._offsets[child_index]
            indices.extend((local + offset).tolist())
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        return iter([indices[index] for index in permutation])
