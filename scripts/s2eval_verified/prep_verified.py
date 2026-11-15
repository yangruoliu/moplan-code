#!/usr/bin/env python3
"""One-time inputs of the Physics-IQ evaluation (CPU, seconds).

From a physics-IQ-benchmark checkout at commit c2bca8a and the Physics-IQ Verified switch frames it writes <out>/:
  manifest_op.jsonl / manifest_bpp.jsonl   198 take-1 rows: id, sid, image (switch frame), prompt (CSV `description`
                                           verbatim), out_name (= generated_video_name), setting, source csv
  s1_manifest_op.csv / s1_manifest_bpp.csv the Stage-1 export manifest (sample_id, first_frame_path, caption): the same
                                           switch frame and the same prompt as the generator gets in that setting
  INPUTS.json                              sha256 of the written files + checks:
    * repo commit = c2bca8a, CSV md5 = the pinned files
    * --masks DIR (optional): a k16 instance mask <sid>.pt exists for every switch frame
    * --reference-frames DIR (optional): the 198 switch frames are byte-identical to <DIR>/<sid>.jpg
    * --reference-manifest CSV (optional): sample ids whose caption differs from the op prompt are listed
usage: python scripts/s2eval_verified/prep_verified.py --repo PIQ_CODE --switch-frames DIR --out DIR [--masks DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

COMMIT = "c2bca8a5bf452d24e177e9975641af1888eb1492"
CSVS = {"op": ("descriptions/descriptions_original.csv", "beb32b7d239649f20bba30025f8c6c7e"),
        "bpp": ("descriptions/best_practice/descriptions_base.csv", "a88d8c7d9fa2088a5c1843b7d14a1390")}


def digest(path: Path, kind: str = "sha256") -> str:
    return hashlib.new(kind, path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, required=True, help="physics-IQ-benchmark checkout at commit c2bca8a")
    p.add_argument("--switch-frames", type=Path, required=True,
                   help="physics-IQ-benchmark-verified/switch-frames (take-1 frames are byte-identical to the original benchmark)")
    p.add_argument("--out", type=Path, required=True, help="output folder for the manifests")
    p.add_argument("--masks", type=Path, default=None, help="optional: k16 instance mask folder to check for completeness")
    p.add_argument("--reference-frames", type=Path, default=None, help="optional: folder of <sid>.jpg to compare byte by byte")
    p.add_argument("--reference-manifest", type=Path, default=None, help="optional: csv (sample_id, caption) to compare with op")
    a = p.parse_args()
    repo, switch, out = a.repo, a.switch_frames, a.out
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if commit != COMMIT:
        raise SystemExit(f"{repo} is at {commit}, not {COMMIT}")
    frames = {f.name[:4]: f for f in switch.glob("*_switch-frames_anyFPS_*.jpg") if int(f.name[:4]) <= 198}
    if sorted(frames) != [f"{i:04d}" for i in range(1, 199)]:
        raise SystemExit(f"expected 198 take-1 switch frames in {switch}, got {len(frames)}")
    report = {"repo": str(repo), "commit": commit, "switch_frames": str(switch), "settings": {}}
    old_caps = ({r["sample_id"]: r["caption"] for r in csv.DictReader(a.reference_manifest.open(newline=""))}
                if a.reference_manifest else None)
    for setting, (rel, md5) in CSVS.items():
        path = repo / rel
        if digest(path, "md5") != md5:
            raise SystemExit(f"{path}: md5 differs from the pinned file")
        rows = [r for r in csv.DictReader(path.open(newline="")) if "_take-1_" in r["scenario"]]
        rows.sort(key=lambda r: r["scenario"][:4])
        if [r["scenario"][:4] for r in rows] != [f"{i:04d}" for i in range(1, 199)]:
            raise SystemExit(f"{rel}: not the 198 take-1 rows 0001..0198")
        items, stage1, caption_diff = [], [], []
        for r in rows:
            i, name = r["scenario"][:4], r["generated_video_name"]
            sid = name[:-4]
            frame = frames[i]
            if frame.name.replace("_switch-frames_anyFPS", "").replace(".jpg", ".mp4") != name:
                raise SystemExit(f"{i}: switch frame {frame.name} does not match {name}")
            items.append({"id": i, "sid": sid, "image": str(frame), "prompt": r["description"], "out_name": name,
                          "setting": setting, "source": rel})
            stage1.append({"sample_id": sid, "first_frame_path": str(frame), "caption": r["description"]})
            if setting == "op" and old_caps is not None and old_caps.get(sid) != r["description"]:
                caption_diff.append(sid)
        man = out / f"manifest_{setting}.jsonl"
        man.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
        s1 = out / f"s1_manifest_{setting}.csv"
        with s1.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "first_frame_path", "caption"])
            writer.writeheader()
            writer.writerows(stage1)
        report["settings"][setting] = {"csv": str(path), "csv_md5": md5, "rows": len(items), "manifest": str(man),
                                       "manifest_sha256": digest(man), "s1_manifest": str(s1), "s1_manifest_sha256": digest(s1)}
        if setting == "op" and old_caps is not None:
            report["settings"][setting]["reference_manifest_caption_differs"] = caption_diff
    same, missing_masks = [], []
    for i, frame in sorted(frames.items()):
        sid = frame.name.replace("_switch-frames_anyFPS", "").replace(".jpg", "")
        if a.reference_frames is not None:
            old = a.reference_frames / f"{sid}.jpg"
            same.append(old.is_file() and digest(old, "md5") == digest(frame, "md5"))
        if a.masks is not None and not (a.masks / f"{sid}.pt").is_file():
            missing_masks.append(sid)
    if a.reference_frames is not None:
        report["switch_frames_identical_to_reference_frames"] = f"{sum(same)}/198"
    if a.masks is not None:
        report["k16_masks"] = str(a.masks)
        report["k16_masks_missing"] = missing_masks
    if (a.reference_frames is not None and sum(same) != 198) or missing_masks:
        raise SystemExit(json.dumps(report, indent=1))
    (out / "INPUTS.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
