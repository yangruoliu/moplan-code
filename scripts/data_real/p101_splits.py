#!/usr/bin/env python
"""Train / dev / test split of the Physics 101 clips BY OBJECT, plus fixed evaluation panels.

Every view and trial of an object goes to one fold, so a dev or test clip never shows an object seen in training.
Objects are ordered by a hash; the first --dev-objects go to dev, the next --test-objects to the sealed test set.
Panels (what is generated at every checkpoint) are drawn evenly over scenario x setting from the front camera.

usage: p101_splits.py --manifest manifest.jsonl --out DIR [--dev-objects 10] [--test-objects 10] [--panel 128]
writes  DIR/{train,dev,test}.jsonl  DIR/{dev,test}_panel.jsonl  DIR/splits.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def spread(rows: list[dict], count: int) -> list[dict]:
    """`count` rows spread evenly over (scenario, setting) groups, one trial per (object, setting) first."""
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        groups[(row["scenario"], row["setting"])].append(row)
    picked, depth = [], 0
    while len(picked) < count and any(len(g) > depth for g in groups.values()):
        for key in sorted(groups):
            seen = set()
            unique = [r for r in groups[key] if not ((r["object"], r["trial"]) in seen or seen.add((r["object"], r["trial"])))]
            order = sorted(unique, key=lambda r: (r["trial"], hashlib.sha256(r["object"].encode()).hexdigest()))
            if len(order) > depth and len(picked) < count:
                picked.append(order[depth])
        depth += 1
    return picked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dev-objects", type=int, default=10)
    p.add_argument("--test-objects", type=int, default=10)
    p.add_argument("--panel", type=int, default=128)
    p.add_argument("--panel-camera", default="cam1")
    a = p.parse_args()
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines() if line.strip()]
    objects = sorted({r["object"] for r in rows}, key=lambda o: hashlib.sha256(("p101-v4:" + o).encode()).hexdigest())
    dev, test = set(objects[: a.dev_objects]), set(objects[a.dev_objects: a.dev_objects + a.test_objects])
    fold = {o: "dev" if o in dev else "test" if o in test else "train" for o in objects}
    a.out.mkdir(parents=True, exist_ok=True)
    by_fold = defaultdict(list)
    for row in rows:
        row = dict(row, split=fold[row["object"]])
        by_fold[row["split"]].append(row)
    for name, part in by_fold.items():
        with open(a.out / f"{name}.jsonl", "w") as handle:
            for row in part:
                handle.write(json.dumps(row) + "\n")
    report = {"objects": {"train": len(objects) - len(dev) - len(test), "dev": sorted(dev), "test": sorted(test)}, "clips": {}}
    for name in ("train", "dev", "test"):
        report["clips"][name] = {"n": len(by_fold[name]), "by_scenario": dict(Counter(r["scenario"] for r in by_fold[name])),
                                 "by_camera": dict(Counter(r["camera"] for r in by_fold[name]))}
    for name in ("dev", "test"):
        panel = spread([r for r in by_fold[name] if r["camera"] == a.panel_camera], a.panel)
        with open(a.out / f"{name}_panel.jsonl", "w") as handle:
            for row in panel:
                handle.write(json.dumps(row) + "\n")
        report["clips"][name + "_panel"] = {"n": len(panel), "by_scenario": dict(Counter(r["scenario"] for r in panel))}
    (a.out / "splits.json").write_text(json.dumps(report, indent=1) + "\n")
    print("P101_SPLITS", json.dumps(report["clips"]))


if __name__ == "__main__":
    main()
