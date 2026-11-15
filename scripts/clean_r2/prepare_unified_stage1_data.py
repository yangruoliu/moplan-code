#!/usr/bin/env python3
"""Assemble one Stage-1 training source in the layout of configs/stage1/train.yaml.

The released ID lists (data/ids/stage1/<source>_{train,val}.tsv) define every
source exactly: which rows it contains and in which order. Row order matters,
because the source-cycle sampler indexes rows by position. This tool takes the
rows produced by the per-source builders (docs/DATA_STAGE1.md), keeps exactly
the listed ones in the listed order, and links their cache files:

    <out>/<layout>/manifests/{train,val}.jsonl
    <out>/<layout>/{targets,flowdit,sam2_conditions}/<sample_id>.pt
    <out>/<layout>/metadata/<name>          (files given with --metadata)

``<out>`` is ``$MOPLAN_DATA/stage1`` and ``<layout>`` is the source directory
used by the config (fot, event_static, phyco_anchor, pool_v2, real/p101,
real/p101r, real/ssv2).

Manifest lines are copied byte for byte from the input manifests. The PhyCo
caption overrides are bound to the SHA256 of the train manifest that
build_phyco_causal_captions.py wrote; copying its lines in the same order keeps
that binding valid.

Example (FOT: v4.4 training rows, the v2 release rows and the v4.4 relational
validation rows; all caches are reachable through the combined v4.4 root)::

    python scripts/clean_r2/prepare_unified_stage1_data.py --source fot \\
        --out $MOPLAN_DATA/stage1 \\
        --manifest $FOT/combined_v44/manifests/train.jsonl \\
        --manifest $FOT/v2/split/dynamic/train.jsonl \\
        --manifest $FOT/combined_v44/manifests_relational/val.jsonl \\
        --cache-root $FOT/combined_v44 \\
        --metadata supervision_mover_ids_manifest.json=$FOT/combined_v44/task_object_ids.json \\
        --metadata prompt_table.pt=$FOT/combined_v44/prompt_table.pt

The tool never overwrites a manifest with different bytes and never replaces an
existing link that points elsewhere. CPU only; no model code is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IDS = REPO_ROOT / "data" / "ids" / "stage1"
SPLITS = ("train", "val")
CACHE_FOLDERS = ("targets", "flowdit", "sam2_conditions")
# config source name -> directory under $MOPLAN_DATA/stage1
LAYOUT = {
    "fot": "fot",
    "event_static": "event_static",
    "phyco_anchor": "phyco_anchor",
    "pool_v2": "pool_v2",
    "real_p101": "real/p101",
    "real_p101r": "real/p101r",
    "real_ssv2": "real/ssv2",
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_id_list(path: Path) -> list[dict]:
    """Rows of a released ID list: sample_id, scenario, scene_id (may be empty)."""
    lines = path.read_text().splitlines()
    if not lines or lines[0].split("\t") != ["sample_id", "scenario", "scene_id"]:
        raise ValueError(f"{path}: expected the header sample_id<TAB>scenario<TAB>scene_id")
    rows = []
    for number, line in enumerate(lines[1:], 2):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0]:
            raise ValueError(f"{path}:{number}: malformed row")
        rows.append({"sample_id": fields[0], "scenario": fields[1], "scene_id": fields[2]})
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate sample ids")
    return rows


def read_candidates(paths: list[Path]) -> dict[str, tuple[dict, str]]:
    """sample_id -> (parsed row, original line) over every input manifest."""
    candidates: dict[str, tuple[dict, str]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("sample_id")
            if not isinstance(sid, str) or not sid:
                raise ValueError(f"{path}: row without sample_id")
            previous = candidates.get(sid)
            if previous is not None and previous[0] != row:
                raise ValueError(f"{sid}: listed with different content in two input manifests")
            candidates.setdefault(sid, (row, line))
    return candidates


def select_rows(wanted: list[dict], candidates: dict[str, tuple[dict, str]], split: str) -> tuple[list[str], list[str]]:
    """Return (manifest lines in ID-list order, missing sample ids); check split/scenario/scene_id."""
    lines, missing = [], []
    for entry in wanted:
        found = candidates.get(entry["sample_id"])
        if found is None:
            missing.append(entry["sample_id"])
            continue
        row, line = found
        if row.get("split") != split:
            raise ValueError(f"{entry['sample_id']}: row split {row.get('split')!r} differs from the {split} list")
        if row.get("scenario", "") != entry["scenario"]:
            raise ValueError(f"{entry['sample_id']}: scenario {row.get('scenario')!r} differs from the ID list")
        if entry["scene_id"] and str(row.get("scene_id", "")) != entry["scene_id"]:
            raise ValueError(f"{entry['sample_id']}: scene_id {row.get('scene_id')!r} differs from the ID list")
        lines.append(line)
    return lines, missing


def place(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"{destination} already exists and points elsewhere")
        return
    if copy:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(os.path.abspath(source))


def write_manifest(path: Path, lines: list[str]) -> str:
    raw = "".join(line + "\n" for line in lines).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError(f"{path} exists with different content")
    if not path.exists():
        path.write_bytes(raw)
    return sha256_bytes(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, choices=sorted(LAYOUT))
    parser.add_argument("--out", required=True, type=Path, help="$MOPLAN_DATA/stage1")
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS, help="directory with <source>_{train,val}.tsv")
    parser.add_argument("--manifest", type=Path, action="append", required=True,
                        help="input manifest (jsonl) with candidate rows; repeatable, any split")
    parser.add_argument("--cache-root", type=Path, action="append", default=[],
                        help="directory with targets/, flowdit/ and sam2_conditions/; repeatable, first match wins")
    parser.add_argument("--metadata", action="append", default=[],
                        help="NAME=FILE placed as <layout>/metadata/NAME (prompt_table.pt, "
                             "supervision_mover_ids_manifest.json, caption_overrides_full.json, ...)")
    parser.add_argument("--copy", action="store_true", help="copy files instead of creating symlinks")
    parser.add_argument("--allow-missing", action="store_true",
                        help="write what is available and report missing rows/files instead of failing")
    args = parser.parse_args()

    root = args.out / LAYOUT[args.source]
    candidates = read_candidates(args.manifest)
    report = {"source": args.source, "root": str(root), "splits": {}, "missing_cache_files": 0}
    missing_any = False
    for split in SPLITS:
        wanted = read_id_list(args.ids / f"{args.source}_{split}.tsv")
        lines, missing = select_rows(wanted, candidates, split)
        if missing and not args.allow_missing:
            raise SystemExit(f"{split}: {len(missing)} listed rows are not in the input manifests, "
                             f"e.g. {missing[:3]}")
        digest = write_manifest(root / "manifests" / f"{split}.jsonl", lines)
        absent = 0
        for line in lines:
            sid = json.loads(line)["sample_id"]
            for folder in CACHE_FOLDERS:
                source = next((base / folder / f"{sid}.pt" for base in args.cache_root
                               if (base / folder / f"{sid}.pt").is_file()), None)
                if source is None:
                    absent += 1
                    continue
                place(source, root / folder / f"{sid}.pt", args.copy)
        report["splits"][split] = {"listed": len(wanted), "written": len(lines), "missing_rows": len(missing),
                                   "manifest_sha256": digest}
        report["missing_cache_files"] += absent
        missing_any |= bool(missing or absent)
    for item in args.metadata:
        name, _, value = item.partition("=")
        if not name or not value or "/" in name:
            raise SystemExit(f"--metadata expects NAME=FILE, got {item!r}")
        place(Path(value), root / "metadata" / name, args.copy)
    print(json.dumps(report, indent=2), flush=True)
    if report["missing_cache_files"] and not args.allow_missing:
        print("some cache files are missing; the trainer would silently drop those rows", flush=True)
        return 1
    return 0 if (not missing_any or args.allow_missing) else 1


if __name__ == "__main__":
    raise SystemExit(main())
