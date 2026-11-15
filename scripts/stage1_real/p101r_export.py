#!/usr/bin/env python
"""Physics-101 other-camera (p101r) Stage-1 real source, step 1: pick the clips and dump their clip metadata.

Runs where the Stage-2 measured-plan cache of the other-camera clips lives.  Train = <cache>/ids_train.txt (other
cameras of train trials only); val = the other-camera dev clips whose trial is one of the 96 Stage-2 dev trials
(cam1 ids_dev96), so the source has a held-out val.jsonl.  For every chosen clip it writes one JSON line with what
the Stage-1 builder needs from the Stage-2 cache (frames_real / frames / at_rest / fps / frame_seconds /
camera_max_px / label_source) and lists the per-clip input files (plan, first frame, SAM2 condition, real tracks)
in files.txt, relative to --files-root, for copying them next to the builder.

usage: p101r_export.py --cache $S2/cache/p101r_k16 --manifest $S2/data/p101/manifest_rest.jsonl
                       --data $S2/data/p101 --dev96-trials trials.txt --out DIR
       ($S2 = the Stage-2 data root, as in docs/DATA_STAGE1.md; --files-root defaults to two levels above --data)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path,
                   help="Physics-101 data root with first_frames/ and k16/{sam2,real_tracks}/")
    p.add_argument("--files-root", type=Path, default=None,
                   help="root that files.txt paths are relative to (default: two levels above --data)")
    p.add_argument("--dev96-trials", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    files_root = a.files_root if a.files_root is not None else a.data.parents[1]
    rows = {json.loads(l)["sample_id"]: json.loads(l) for l in a.manifest.read_text().splitlines() if l.strip()}
    dev_trials = {l.strip() for l in a.dev96_trials.read_text().splitlines() if l.strip()}
    train = [l.strip() for l in (a.cache / "ids_train.txt").read_text().splitlines() if l.strip()]
    dev = [l.strip() for l in (a.cache / "ids_dev.txt").read_text().splitlines() if l.strip()]
    val = [s for s in dev if rows[s]["trial_id"] in dev_trials]
    assert not {rows[s]["trial_id"] for s in train} & dev_trials, "p101r train shares a trial with the P101 val set"
    assert all(rows[s]["split"] == "train" for s in train)
    a.out.mkdir(parents=True, exist_ok=True)
    files, missing = [], []
    with open(a.out / "meta.jsonl", "w") as handle:
        for split, ids in (("train", train), ("val", val)):
            for sid in ids:
                need = {"plan": a.cache / "plans_real" / f"{sid}.npz", "first_frame": a.data / "first_frames" / f"{sid}.png",
                        "sam2": a.data / "k16" / "sam2" / f"{sid}.pt", "tracks": a.data / "k16" / "real_tracks" / f"{sid}.npz",
                        "stage2": a.cache / f"{sid}.pt"}
                if not all(path.is_file() for path in need.values()):
                    missing.append(sid)
                    continue
                x = torch.load(need["stage2"], map_location="cpu", weights_only=False, mmap=True)
                scene = json.loads(x.get("real_scene") or "{}")
                meta = {"sample_id": sid, "split": split, "frames": int(x["frames"]), "frames_real": int(x["frames_real"]),
                        "at_rest": bool(x.get("at_rest", False)), "fps": float(x["fps"]), "frame_seconds": list(x["frame_seconds"]),
                        "camera_max_px": float(scene.get("camera_max_px", float("nan"))), "label_source": x.get("label_source", ""),
                        "row": rows[sid]}
                handle.write(json.dumps(meta) + "\n")
                files += [str(need[k].relative_to(files_root)) for k in ("plan", "first_frame", "sam2", "tracks")]
    (a.out / "files.txt").write_text("\n".join(files) + "\n")
    print(json.dumps({"train": len(train), "val": len(val), "missing": len(missing), "files": len(files)}))


if __name__ == "__main__":
    main()
