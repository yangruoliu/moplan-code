#!/usr/bin/env python
"""Build the PhyCo full-population manifests (v1) from the target cache.

Population = every clip with a built Stage-1 target (94,758 at build time);
clips in the official clip list whose targets are absent go to the exclusions
appendix, never silently dropped. The manifest is NOT filtered by latent or
affine availability -- that is runtime business (available_rows / exclusions
contract), baking it in here would poison provenance.

Split rule (fixed): bucket =
int(sha256(group_key)[:8], 16) % 100; val = {0,1}, held-out test = {2,3}
(sealed once built), train = the rest. Hash membership never drifts as caches
backfill, so the manifest is written exactly once.

group_key = sample_id. Basis: clip ids are 6-hex directory names; across
95,869 clip dirs we observe 271 duplicated ids, which matches the birthday
expectation for independent draws from a 16.7M space
(95,869^2 / (2 * 16^6) ~= 275), and no id repeats within the same
scenario+date (asserted below). Hence one directory = one independent Kubric
simulation; there is no multi-camera/multi-window grouping to protect, and
sample_id (scenario + date + id) is already the finest simulation-unique key.

An earlier, smaller PhyCo subset is deliberately NOT reconstructed -- its
membership cannot be recovered, and imitating it would be fake provenance.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-dir", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--absent-ids", default="",
        help="newline list of sample_ids from the official clip list whose "
             "targets were never built; recorded in the exclusions appendix",
    )
    parser.add_argument(
        "--failed-ids", default="",
        help="subset of --absent-ids that failed with an error (rest are "
             "recorded as skipped-by-guard)",
    )
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def bucket(sample_id: str) -> int:
    return int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) % 100


def split_of(sample_id: str) -> str:
    b = bucket(sample_id)
    if b in (0, 1):
        return "val"
    if b in (2, 3):
        return "test"
    return "train"


def read_caption(path: Path) -> tuple[str, str]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return path.stem, str(payload.get("caption", ""))


def main() -> None:
    args = parse_args()
    raw = Path(args.raw_root)
    targets = Path(args.targets_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # sample_id -> (scenario, clip dir), derived from the raw tree so the
    # scenario/date/id decomposition never depends on parsing underscores.
    index: dict[str, tuple[str, Path]] = {}
    triple_ids = set()
    for clip in raw.glob("*/*/*"):
        if not clip.is_dir():
            continue
        scenario, date, cid = clip.parent.parent.name, clip.parent.name, clip.name
        sid = f"phyco_{scenario}_{date}_{cid}"
        assert sid not in triple_ids, f"duplicate simulation triple {sid}"
        triple_ids.add(sid)
        index[sid] = (scenario, clip)

    target_files = sorted(targets.glob("*.pt"))
    print(f"targets: {len(target_files)}, raw clips indexed: {len(index)}")

    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        captions = dict(pool.map(read_caption, target_files, chunksize=64))

    rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    unmatched = []
    for path in target_files:
        sid = path.stem
        if sid not in index:
            unmatched.append(sid)
            continue
        scenario, clip = index[sid]
        rows_by_split[split_of(sid)].append(
            {
                "sample_id": sid,
                "scenario": scenario,
                "split": split_of(sid),
                "video_path": str(clip / "rgba.mp4"),
                "caption": captions.get(sid, ""),
                "source": "phyco_kubric",
            }
        )
    if unmatched:
        raise SystemExit(f"{len(unmatched)} targets have no raw clip dir: {unmatched[:5]}")

    for split, rows in rows_by_split.items():
        rows.sort(key=lambda r: r["sample_id"])
        with (out / f"{split}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    absent = [l.strip() for l in open(args.absent_ids)] if args.absent_ids else []
    failed = {l.strip() for l in open(args.failed_ids)} if args.failed_ids else set()
    exclusions = {
        "schema": "phyco_exclusions_v1",
        "mapping": {
            sid: {"category": "backfill_failed" if sid in failed else "backfill_skipped_guard"}
            for sid in absent if sid
        },
    }
    (out / "phyco_exclusions_v1.json").write_text(json.dumps(exclusions, indent=1))

    empty_captions = sum(
        1 for rows in rows_by_split.values() for r in rows if not r["caption"]
    )
    summary = {
        "schema": "phyco_manifests_v1",
        "population": len(target_files),
        "split_rule": "sha256(sample_id)[:8] % 100 -> val{0,1} test{2,3} train{4..99}",
        "group_key": "sample_id (1 dir = 1 independent simulation; see docstring)",
        "counts": {k: len(v) for k, v in rows_by_split.items()},
        "scenario_counts": dict(Counter(
            r["scenario"] for rows in rows_by_split.values() for r in rows
        )),
        "empty_captions": empty_captions,
        "exclusions": len(exclusions["mapping"]),
        "generated_at": datetime.datetime.now().isoformat(),
        "targets_snapshot": str(targets),
    }
    (out / "summary_v1.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("population", "counts", "empty_captions", "exclusions")}, indent=1))


if __name__ == "__main__":
    main()
