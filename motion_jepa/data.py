from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .config import SCENARIO_TO_ID


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    scenario: str
    split: str
    video_path: str
    map_path: str = ""
    source: str = "dynamics"
    # The grounded manifests carry a per-sample caption. Without this field
    # ``from_dict`` silently drops it and Stage-1 falls back to one embedding
    # per scenario, which is what starved the previous text conditioning.
    caption: str = ""
    # Preserve identity across relocated asset roots and caption variants.
    # Coverage must not count the same scene twice in dynamic/static sources.
    scene_id: str = ""
    original_video_path: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ManifestRow":
        return cls(**{field: value.get(field, "") for field in cls.__dataclass_fields__})


def read_jsonl(path: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(ManifestRow.from_dict(json.loads(line)))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[ManifestRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, sort_keys=True) + "\n")
    temporary.replace(path)


def cache_path(cache_dir: str | Path, kind: str, sample_id: str) -> Path:
    return Path(cache_dir) / kind / f"{sample_id}.pt"


class Stage1Dataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        cache_dir: str | Path,
        *,
        width: int,
        height: int,
        limit: int = 0,
        strict: bool = True,
        text_embedding_path: str | Path | None = None,
    ) -> None:
        self.rows = read_jsonl(manifest)
        if limit:
            self.rows = self.rows[:limit]
        self.cache_dir = Path(cache_dir)
        self.width = width
        self.height = height
        self.strict = strict
        self.text_embedding_table: torch.Tensor | None = None
        if text_embedding_path:
            payload = torch.load(text_embedding_path, map_location="cpu", weights_only=True)
            table = payload["scenario_embeddings"].float()
            expected = len(SCENARIO_TO_ID)
            if table.ndim != 2 or table.shape[0] != expected or not torch.isfinite(table).all():
                raise ValueError(
                    f"invalid text embedding table {text_embedding_path}: {tuple(table.shape)}"
                )
            self.text_embedding_table = table
        if strict:
            missing = [row.sample_id for row in self.rows if not self._paths_exist(row.sample_id)]
            if missing:
                preview = ", ".join(missing[:5])
                raise FileNotFoundError(f"{len(missing)} samples lack cache entries, first: {preview}")

    def _paths_exist(self, sample_id: str) -> bool:
        return cache_path(self.cache_dir, "features", sample_id).is_file() and cache_path(
            self.cache_dir, "targets", sample_id
        ).is_file()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        feature = torch.load(
            cache_path(self.cache_dir, "features", row.sample_id), map_location="cpu", weights_only=True
        )
        target = torch.load(
            cache_path(self.cache_dir, "targets", row.sample_id), map_location="cpu", weights_only=True
        )
        flow = target["flow"].float()
        flow[:, 0].div_(self.width)
        flow[:, 1].div_(self.height)
        scenario_id = SCENARIO_TO_ID[row.scenario]
        sample = {
            "sample_id": row.sample_id,
            "scenario_name": row.scenario,
            "label_source": target.get("label_source", "unknown"),
            "scenario": torch.tensor(scenario_id, dtype=torch.long),
            "z0": feature["z0"].float(),
            "z_future": feature["z_future"].float(),
            "object_mask": target["object_mask"].float(),
            # Native instance labels supervise losses only. The deployable
            # predictor receives first-frame tokens plus text, never a GT mask.
            "condition_mask": torch.ones_like(target["object_mask"], dtype=torch.float32),
            "future_mask": target["future_mask"].float(),
            "flow": flow,
            "visibility": target["visibility"].float(),
        }
        for key in ("instance_id", "centers", "center_valid", "object_ids"):
            if key in target:
                sample[key] = target[key]
        if self.text_embedding_table is not None:
            sample["text_embedding"] = self.text_embedding_table[scenario_id].clone()
        self._validate(sample)
        return sample

    def _validate(self, sample: dict[str, Any]) -> None:
        expected = {
            "z0": 3,
            "z_future": 4,
            "object_mask": 3,
            "condition_mask": 3,
            "future_mask": 4,
            "flow": 4,
            "visibility": 4,
        }
        for key, dimensions in expected.items():
            tensor = sample[key]
            if tensor.ndim != dimensions or not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{sample['sample_id']} invalid {key}: shape={tuple(tensor.shape)}, "
                    f"finite={bool(torch.isfinite(tensor).all())}"
                )
        if sample["object_mask"].sum() < 1:
            raise ValueError(f"{sample['sample_id']} has an empty object mask")
        if "instance_id" in sample:
            if sample["instance_id"].shape != sample["object_mask"].shape:
                raise ValueError(f"{sample['sample_id']} instance_id and object_mask disagree")
            if not torch.equal(
                sample["instance_id"] > 0, sample["object_mask"] > 0
            ):
                raise ValueError(f"{sample['sample_id']} instance coverage disagrees with mask")
