#!/usr/bin/env python
"""Build exact PhyCo mover-instance IDs from trusted causal metadata.

The historical target ``object_mask`` selected instances with a RAFT motion
threshold.  It can include static walls and supports.  This artifact instead
maps scenario-specific participating object types to the exact segmentation IDs
stored in the rendered first-frame ``instance_id`` target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_mover(scenario: str, kind: str) -> bool:
    if scenario in {"ball_drop_soft_v4", "ball_drop_v2", "ball_drop_v3"}:
        return kind == "ball" or kind.startswith("ball_")
    if scenario == "ball_wall_collision":
        return kind == "ball"
    if scenario == "cube_deform_soft_v2_noeff":
        return kind in {"jelly_cube", "weight"}
    if scenario == "friction_slide_flat_force_v3":
        return kind == "brick"
    if scenario == "friction_slide_flat_v2":
        return kind == "sliding_object"
    raise ValueError(f"unsupported PhyCo mover policy: {scenario}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = read_jsonl(args.candidate_manifest)
    metadata_rows = {row["sample_id"]: row for row in read_jsonl(args.metadata)}
    if len(metadata_rows) != len(candidates):
        raise ValueError("candidate/metadata row-count mismatch")
    mapping = {}
    type_counts: dict[str, int] = {}
    for candidate in candidates:
        sample_id = candidate["sample_id"]
        metadata = metadata_rows.get(sample_id)
        if metadata is None:
            raise ValueError(f"missing causal metadata: {sample_id}")
        scenario = candidate["category"].split("/")[-1]
        objects = metadata["object_data"]
        kinds = objects.get("type")
        identifiers = objects.get("segmentation_id")
        if not isinstance(kinds, list) or not isinstance(identifiers, list):
            raise ValueError(f"missing object types/segmentation IDs: {sample_id}")
        if len(kinds) != len(identifiers):
            raise ValueError(f"object type/segmentation ID length mismatch: {sample_id}")
        selected = []
        selected_types = []
        for kind, identifier in zip(kinds, identifiers):
            kind = str(kind)
            if not is_mover(scenario, kind):
                continue
            if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier <= 0:
                raise ValueError(f"invalid mover segmentation ID: {sample_id}/{identifier}")
            selected.append(identifier)
            selected_types.append(kind)
            type_counts[kind] = type_counts.get(kind, 0) + 1
        if not selected or len(selected) != len(set(selected)):
            raise ValueError(f"missing/duplicate mover IDs: {sample_id}/{selected}")
        order = sorted(range(len(selected)), key=lambda index: selected[index])
        selected = [selected[index] for index in order]
        selected_types = [selected_types[index] for index in order]
        mapping[sample_id] = {
            "sample_id": sample_id,
            "scenario": scenario,
            "mover_instance_ids": selected,
            "mover_types": selected_types,
            "source_metadata_sha256": metadata["metadata_sha256"],
        }
    if set(mapping) != {row["sample_id"] for row in candidates}:
        raise ValueError("mover mapping ID set differs from candidate manifest")
    payload = {
        "schema": "phyco_supervision_mover_ids_v1",
        "rows": len(mapping),
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "causal_metadata_sha256": sha256(args.metadata),
        "policy": {
            "source": "trusted_metadata_object_type_plus_rendered_segmentation_id",
            "legacy_raft_object_mask_used_as_gt": False,
            "future_rgb_used": False,
        },
        "mover_type_counts": dict(sorted(type_counts.items())),
        "mapping": mapping,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output), "rows": len(mapping),
        "sha256": sha256(args.output), "mover_type_counts": payload["mover_type_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
