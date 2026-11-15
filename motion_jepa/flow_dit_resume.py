"""Exact continuation for deterministic cached dense-mask training.

The model, AdamW and scheduler live in the normal checkpoint files. This module
adds RNG, a consumed-batch cursor, a frozen runtime contract and an atomic
completion manifest. Worker prefetch is replayed, never mistaken for consumed
training data. Random dataset transforms are deliberately outside this contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
import yaml

PROTOCOL = "stage1_exact_resume_v2"
REQUIRED_FILES = frozenset({"backend_params.pt", "decoder.pt", "optimizer.pt", "scheduler.pt",
                            "trainer_state.json", "rng_state.pt", "resolved_config.yaml"})


def complete_accumulation_samples(total_samples: int, batch_size: int,
                                  gradient_accumulation_steps: int) -> int:
    """Trim a sampled epoch to complete optimizer-step groups.

    ``Accelerate`` flushes an incomplete accumulation group at the end of a
    dataloader.  That makes an epoch boundary behave differently from an
    ordinary group and prevents an exact mid-run restart.  The mixture sampler
    samples with replacement, so dropping at most one group minus one draws is
    both deterministic and semantically neutral.
    """
    values = (total_samples, batch_size, gradient_accumulation_steps)
    if any(not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("sample count, batch size and accumulation steps must be positive integers")
    samples_per_step = batch_size * gradient_accumulation_steps
    complete = total_samples - total_samples % samples_per_step
    if complete < samples_per_step:
        raise ValueError("dataset is smaller than one complete gradient-accumulation group")
    return complete


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    if len(state["torch_cuda"]) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("exact resume CUDA RNG device count changed")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rank_rng_state(payload: dict, *, process_index: int, world_size: int) -> dict:
    """Select this rank's stochastic state from a committed checkpoint."""
    if not isinstance(process_index, int) or not isinstance(world_size, int):
        raise ValueError("process index and world size must be integers")
    if world_size < 1 or not 0 <= process_index < world_size:
        raise ValueError("invalid exact-resume process topology")
    states = payload.get("rng_by_rank")
    if states is not None:
        if not isinstance(states, list) or len(states) != world_size:
            raise ValueError("exact-resume rank RNG count differs from world size")
        return states[process_index]
    if world_size == 1 and "rng" in payload:
        return payload["rng"]
    raise ValueError("distributed exact-resume checkpoint lacks per-rank RNG states")


@dataclass
class DataCursor:
    batches_per_epoch: int
    epoch: int = 0
    batches_consumed: int = 0

    def __post_init__(self):
        if self.batches_per_epoch < 1 or self.epoch < 0 or not 0 <= self.batches_consumed < self.batches_per_epoch:
            raise ValueError("invalid exact-resume consumed-batch cursor")

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "batches_consumed": self.batches_consumed,
                "batches_per_epoch": self.batches_per_epoch}

    def consumed(self) -> None:
        self.batches_consumed += 1
        if self.batches_consumed == self.batches_per_epoch:
            self.epoch += 1
            self.batches_consumed = 0

    def iterate(self, loader, sampler, *, resume_rng: dict | None = None):
        """Replay the deterministic index stream, then restore training RNG.

        Keep model randomness independent of loader worker/base seeds, both in
        fresh and resumed execution. Persistent workers create a different
        number of base seeds after a restart; these seeds cannot affect this
        deterministic cached dataset and must not shift caption/model dropout.
        """
        sampler.epoch = self.epoch
        offset = self.batches_consumed
        training_rng = capture_rng() if resume_rng is None else resume_rng
        iterator = iter(loader)
        for _ in range(offset):
            next(iterator)
        first = next(iterator)
        restore_rng(training_rng)
        self.consumed()
        yield first
        for batch in iterator:
            self.consumed()
            yield batch


def runtime_contract(*, config: dict, expert: str, max_steps: int, sampler, loader,
                     train_limit: int, val_limit: int, val_split: str,
                     world_size: int, gradient_accumulation_steps: int) -> dict:
    from motion_jepa.flow_dit_data import FixedMixtureSampler

    if world_size < 1 or not isinstance(sampler, FixedMixtureSampler):
        raise ValueError("dense exact resume requires FixedMixtureSampler")
    if len(loader) % gradient_accumulation_steps:
        raise ValueError("dense exact resume requires complete gradient-accumulation groups per epoch")
    rows = [row.get("sample_id") if isinstance(row, dict) else getattr(row, "sample_id", None)
            for row in sampler.dataset.rows]
    if not rows or any(not isinstance(row, str) for row in rows):
        raise ValueError("dense exact resume requires stable cached sample identifiers")
    from .caption_overrides import caption_override_data_contract
    caption_contract = caption_override_data_contract(sampler.dataset)
    from .flow_dit_numerics import numerics_contract
    numerics = numerics_contract(config)
    return {"protocol": PROTOCOL, "world_size": world_size, "expert": expert, "max_steps": max_steps,
            "config_sha256": config_digest(config), "train_limit": train_limit,
            "val_limit": val_limit, "val_split": val_split,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            **({"numerics": numerics} if numerics is not None else {}),
            **({"caption_override_contract": caption_contract} if caption_contract is not None else {}),
            "sampler_contract": {"kind": type(sampler).__name__, "seed": sampler.seed,
                "fractions": sampler.fractions, "num_samples": sampler.num_samples,
                "child_lengths": [len(child) for child in sampler.dataset.children],
                "sample_ids_sha256": config_digest({"ids": rows}),
                "batches_per_epoch": len(loader),
                "deterministic_cached_dataset": True}}


def log_positions(output: Path) -> dict:
    return {name: (output / name).stat().st_size if (output / name).exists() else 0
            for name in ("metrics.jsonl", "history.jsonl")}


def log_snapshots(output: Path) -> dict:
    return {name: (output / name).read_bytes() if (output / name).exists() else b""
            for name in ("metrics.jsonl", "history.jsonl")}


def rewind_logs(output: Path, positions: dict, *, step: int, snapshots: dict | None = None) -> list[str]:
    """Archive crash tails before returning append-only logs to checkpoint state."""
    archived = []
    if set(positions) != {"metrics.jsonl", "history.jsonl"}:
        raise ValueError("exact resume log cursor is incomplete")
    if snapshots is not None and (set(snapshots) != set(positions)
                                 or any(not isinstance(snapshots[name], bytes)
                                        or len(snapshots[name]) != positions[name] for name in positions)):
        raise ValueError("checkpoint embedded log snapshots do not match byte cursors")
    for name, position in positions.items():
        path = output / name
        if snapshots is not None and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(snapshots[name])
        if not isinstance(position, int) or position < 0 or not path.exists() or path.stat().st_size < position:
            raise ValueError(f"resume log is missing or shorter than the saved cursor: {path}")
        if snapshots is not None:
            with path.open("rb") as stream:
                if stream.read(position) != snapshots[name]:
                    raise ValueError(f"existing log prefix differs from the committed checkpoint: {path}")
        if path.stat().st_size == position:
            continue
        folder = output / "resume_recovery"
        folder.mkdir(exist_ok=True)
        fd, backup = tempfile.mkstemp(prefix=f"after-step-{step:06d}-", suffix=f"-{name}", dir=folder)
        with os.fdopen(fd, "wb") as dest, path.open("rb") as source:
            source.seek(position)
            shutil.copyfileobj(source, dest)
        with path.open("r+b") as stream:
            stream.truncate(position)
        archived.append(backup)
    return archived


def staging_directory(destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite committed checkpoint: {destination}")
    return Path(tempfile.mkdtemp(prefix=f".{destination.name}.incomplete-", dir=destination.parent))


def commit_checkpoint(staging: Path, destination: Path, *, step: int) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    if not REQUIRED_FILES <= {path.name for path in staging.iterdir() if path.is_file()}:
        raise ValueError("cannot publish incomplete exact-resume checkpoint")
    marker = {"schema_version": 1, "protocol": PROTOCOL, "step": step,
              "files": {name: {"sha256": sha256(staging / name), "size_bytes": (staging / name).stat().st_size}
                        for name in sorted(REQUIRED_FILES)}}
    for name in REQUIRED_FILES:
        with (staging / name).open("rb") as stream:
            os.fsync(stream.fileno())
    with (staging / "checkpoint_complete.json").open("x") as stream:
        json.dump(marker, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    staging.rename(destination)
    return marker


def verify_checkpoint(path: Path, *, config: dict | None = None) -> dict:
    marker = json.loads((path / "checkpoint_complete.json").read_text())
    if (marker.get("schema_version") != 1 or marker.get("protocol") != PROTOCOL
            or set(marker.get("files", {})) != REQUIRED_FILES
            or path.name != f"checkpoint-{marker.get('step', -1):06d}"):
        raise ValueError("not a committed numeric exact-resume checkpoint")
    for name, record in marker["files"].items():
        file = path / name
        if (file.is_symlink() or file.stat().st_size != record["size_bytes"]
                or sha256(file) != record["sha256"]):
            raise ValueError(f"checkpoint file changed or is incomplete: {name}")
    state = json.loads((path / "trainer_state.json").read_text())
    rng = torch.load(path / "rng_state.pt", map_location="cpu", weights_only=False)
    if state.get("step") != marker["step"] or state.get("resume_protocol") != PROTOCOL:
        raise ValueError("checkpoint training state disagrees with completion marker")
    if rng.get("step") != marker["step"] or rng.get("protocol") != PROTOCOL:
        raise ValueError("checkpoint RNG state disagrees with completion marker")
    if state.get("resume_contract") != rng.get("contract"):
        raise ValueError("checkpoint RNG/runtime contracts disagree")
    saved_config = yaml.safe_load((path / "resolved_config.yaml").read_text())
    for key in ("selected_expert", "resolved_max_steps", "resolved_stop_at_step"):
        saved_config.pop(key, None)
    if config_digest(saved_config) != rng["contract"]["config_sha256"]:
        raise ValueError("checkpoint config content does not match resume contract")
    if config is not None and config_digest(config) != rng["contract"]["config_sha256"]:
        raise ValueError("exact resume requires the unchanged complete training config")
    DataCursor(**rng["data_cursor"])
    return {"marker": marker, "trainer_state": state, "rng_state": rng}
