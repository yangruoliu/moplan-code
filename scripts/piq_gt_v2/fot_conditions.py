#!/usr/bin/env python
"""FOT bookkeeping after caching.

`reps`   : one representative row per scene (variant v0, or the first cached one) -> JSONL for
           cache_sam2_masks.py (all variants of a scene share the identical first frame).
`expand` : copy each scene's SAM2 condition to every sibling variant with its own sample_id
           (the loader checks sample_id and the first-frame hash against the flow cache), then
           write inference-format masks + infer manifests for the val split.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


def scene_of(sample_id: str) -> str:
    return sample_id.rsplit("_v", 1)[0]


def all_rows(root: Path) -> list[dict]:
    rows = []
    for name in ("manifests", "manifests_static"):
        for split in ("train", "val"):
            path = root / name / f"{split}.jsonl"
            if path.exists():
                rows += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["reps", "expand"])
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = all_rows(root)
    groups = defaultdict(list)
    for row in rows:
        groups[scene_of(row["sample_id"])].append(row)
    if args.phase == "reps":
        with (root / "sam2_reps.jsonl").open("w") as handle:
            for scene, members in sorted(groups.items()):
                members.sort(key=lambda r: r["sample_id"])
                cached = [m for m in members if (root / "flowdit" / f"{m['sample_id']}.pt").exists()]
                if cached:
                    handle.write(json.dumps(cached[0]) + "\n")
        print("representatives", len(groups))
        return
    from motion_jepa.flow_dit_data import build_instance_inputs, first_frame_tensor_sha256
    conditions = root / "sam2_conditions"
    infer_dir = root / "eval" / "masks"
    infer_dir.mkdir(parents=True, exist_ok=True)
    written = missing = mismatched = 0
    manifest_rows = []
    for scene, members in sorted(groups.items()):
        source = next((conditions / f"{m['sample_id']}.pt" for m in sorted(members, key=lambda r: r["sample_id"])
                       if (conditions / f"{m['sample_id']}.pt").exists()), None)
        if source is None:
            missing += len(members)
            continue
        base = torch.load(source, map_location="cpu", weights_only=True)
        for member in members:
            sid = member["sample_id"]
            flow_path = root / "flowdit" / f"{sid}.pt"
            if not flow_path.exists():
                missing += 1
                continue
            frame = torch.load(flow_path, map_location="cpu", weights_only=True)["first_frame"]
            if first_frame_tensor_sha256(frame) != base["first_frame_sha256"]:
                mismatched += 1
                continue
            destination = conditions / f"{sid}.pt"
            if not destination.exists():
                torch.save({**base, "sample_id": sid}, destination)
                written += 1
            if member["split"] == "val":
                labels = base["instance_labels"].long()
                masks, _, channels = build_instance_inputs(labels[None], 16)
                torch.save({"sample_id": sid, "instance_masks": masks.to(torch.uint8),
                            "instance_channels": channels.to(torch.uint8),
                            "num_instances": int(masks.flatten(1).any(1).sum())}, infer_dir / f"{sid}.pt")
                manifest_rows.append({"sample_id": sid, "first_frame_path": member["video_path"],
                                      "caption": member["caption"]})
    with (root / "eval" / "val_infer_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "first_frame_path", "caption"])
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda r: r["sample_id"]))
    print(json.dumps({"copied": written, "missing": missing, "hash_mismatch": mismatched,
                      "val_infer_rows": len(manifest_rows)}))


if __name__ == "__main__":
    main()
