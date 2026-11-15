"""Reviewed, train-only caption substitutions without rewriting source targets.

The override JSON binds the original manifest bytes. Every substituted caption
also binds an accepted row in an external review JSONL by file and text hashes.
This module does not create reviews or turn generated captions into approval.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "stage1_caption_overrides_v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def caption_sha256(caption: str) -> str:
    """Same canonical caption key used by the existing prompt table."""
    return _sha(caption.strip().encode("utf-8"))


def validate_override_prompt_embedding(embedding: Any, *, pad_length: int, label: str) -> None:
    """Validate the actual tensor against the effective training context size."""
    import torch
    if isinstance(pad_length, bool) or not isinstance(pad_length, int) or pad_length < 1:
        raise ValueError("caption overrides require an explicit positive training prompt_pad_length")
    if (not isinstance(embedding, torch.Tensor) or embedding.ndim != 2
            or not all(embedding.shape) or not torch.isfinite(embedding).all()):
        raise ValueError(f"{label}: invalid caption override prompt embedding")
    if embedding.shape[0] > pad_length:
        raise ValueError(f"{label}: caption override embedding length {embedding.shape[0]} "
                         f"exceeds training prompt_pad_length={pad_length}; refusing truncation")


@dataclass(frozen=True)
class CaptionOverrides:
    captions: dict[str, str]
    contract: dict[str, Any]


def load_caption_overrides(
    path: str | Path, *, manifest: str | Path, original_captions: dict[str, str],
) -> CaptionOverrides:
    """Validate the complete table before exclusions, cache filtering or limits."""
    path = Path(path)
    content = path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("split") != "train":
        raise ValueError("caption overrides require stage1_caption_overrides_v1 and split=train")
    manifest_bytes = Path(manifest).read_bytes()
    manifest_sha = _sha(manifest_bytes)
    if payload.get("source_manifest_sha256") != manifest_sha:
        raise ValueError("caption override source manifest SHA256 mismatch")
    rows = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines() if line.strip()]
    source: dict[str, dict] = {}
    for row in rows:
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in source:
            raise ValueError("caption overrides require unique nonempty manifest sample IDs")
        if row.get("split") != "train":
            raise ValueError("caption overrides may only be applied to a train manifest")
        source[sid] = row
    if {sid: row.get("caption", "") for sid, row in source.items()} != original_captions:
        raise ValueError("source manifest captions changed while loading caption overrides")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("caption overrides require a nonempty records list")
    captions: dict[str, str] = {}
    reviews: dict[Path, tuple[str, dict[str, dict]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("caption override record must be an object")
        sid = record.get("sample_id")
        if not isinstance(sid, str) or sid not in source or sid in captions:
            raise ValueError(f"caption override has an unknown or duplicate train sample ID: {sid!r}")
        if record.get("split") != "train":
            raise ValueError(f"{sid}: caption override record must declare split=train")
        if record.get("old_caption") != source[sid].get("caption", ""):
            raise ValueError(f"{sid}: caption override old_caption differs from original manifest")
        caption = record.get("new_caption")
        if not isinstance(caption, str) or not caption or caption != caption.strip():
            raise ValueError(f"{sid}: new_caption must be nonempty text without outer whitespace")
        caption_sha = caption_sha256(caption)
        if record.get("new_caption_sha256") != caption_sha:
            raise ValueError(f"{sid}: new caption SHA256 mismatch")
        review = record.get("review")
        if not isinstance(review, dict) or not isinstance(review.get("source_path"), str) or not review["source_path"]:
            raise ValueError(f"{sid}: caption override requires an external review source")
        review_path = Path(review["source_path"])
        if not review_path.is_absolute():
            review_path = path.parent / review_path
        review_path = review_path.resolve()
        if review_path not in reviews:
            review_bytes = review_path.read_bytes()
            review_rows: dict[str, dict] = {}
            for line in review_bytes.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                key = item.get("sample_id")
                if not isinstance(key, str) or not key or key in review_rows:
                    raise ValueError("caption review source requires unique nonempty sample IDs")
                review_rows[key] = item
            reviews[review_path] = (_sha(review_bytes), review_rows)
        review_sha, review_rows = reviews[review_path]
        if review.get("source_sha256") != review_sha:
            raise ValueError(f"{sid}: caption review source SHA256 mismatch")
        accepted = review_rows.get(sid, {})
        if (accepted.get("decision") != "accept" or accepted.get("split") != "train"
                or not isinstance(accepted.get("reviewer"), str) or not accepted["reviewer"].strip()
                or accepted.get("new_caption") != caption
                or accepted.get("new_caption_sha256") != caption_sha):
            raise ValueError(f"{sid}: review must explicitly accept this train sample and exact new caption")
        captions[sid] = caption
    return CaptionOverrides(captions=captions, contract={
        "schema": SCHEMA, "override_file_sha256": _sha(content),
        "source_manifest_sha256": manifest_sha, "record_count": len(captions),
        "review_source_sha256s": sorted({value[0] for value in reviews.values()}),
    })


def caption_override_data_contract(dataset: Any) -> dict | None:
    """Capture source order in a mixed dataset; no override preserves old state."""
    children = getattr(dataset, "children", [dataset])
    contracts = [getattr(child, "caption_override_contract", None) for child in children]
    if not any(contract is not None for contract in contracts):
        return None
    return {"schema": SCHEMA, "sources": contracts}


def validate_caption_override_resume(state: dict, current: dict | None) -> None:
    if state.get("caption_override_contract") != current:
        raise ValueError(
            "caption override data contract changed; use --init-from for a new caption training run, "
            "not --resume-from"
        )
