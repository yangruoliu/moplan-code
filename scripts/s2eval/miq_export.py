#!/usr/bin/env python3
"""CPU: export one complete MoProbe generation folder of infer_stage2_cond_v2.py (<arm>__<sid>[__s<k>].npy) in the
layout the MoProbe measurement code reads for a model "arm":  <root>/<miq_set>/gen/<ARM>/s<k>/<miq_sid>.mp4 + .json.

  video  the native 448x256 work frame, all frames, h264 crf 10 yuv420p at the clip's own fps (7)
  json   fps (the measurement scripts default to 16 fps without it), frames, size, seed index, the caption we fed, the
         benchmark prompt, generator / Stage-1 tags, source folder
The manifest is the MoProbe deployment manifest of scripts/bench/miq_inputs.py / miqB_inputs.py (fields miq_set, miq_sid,
item_sid, cond, caption, benchmark_prompt).
usage: miq_export.py --gen G --manifest M --arm ARM --root R [--src-arm pred] [--num-seeds 1] [--generator TAG --stage1 S1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import export_videos as ev  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--arm", required=True, help="model name in the MoProbe tree (e.g. moplan)")
    p.add_argument("--src-arm", default="pred", help="arm of the npy clips (pred, base, zero)")
    p.add_argument("--root", required=True)
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--generator", default="")
    p.add_argument("--stage1", default="")
    a = p.parse_args()
    gen, root = Path(a.gen), Path(a.root)
    rows = [json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    n, missing = 0, []
    for r in rows:
        for k in range(a.num_seeds):
            try:
                frames, fps = ev.load_clip(gen, a.src_arm, r["sample_id"], k)
            except FileNotFoundError:
                missing.append(f"{r['sample_id']}__s{k}")
                continue
            out = root / r["miq_set"] / "gen" / a.arm / f"s{k}" / f"{r['miq_sid']}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            imgs = [f for f in frames]
            ev.write_mp4_h264(imgs, out, fps)
            info = ev.probe(out)
            if info["frames"] != len(imgs) or abs(info["fps"] - fps) > 0.01:
                raise SystemExit(f"{out}: wrote {len(imgs)} frames at {fps}, read back {info}")
            meta = {"model": a.arm, "fps": float(fps), "frames": len(imgs), "width": info["size"][0], "height": info["size"][1],
                    "seed_index": k, "caption": r["caption"], "instruction": r["benchmark_prompt"], "item_sid": r["item_sid"],
                    "cond": r["cond"], "miq_set": r["miq_set"], "generator": a.generator, "stage1": a.stage1,
                    "work_frame": "448x256 (the 1280x720 16:9 canvas stretched; no crop)", "source_npy": str(gen)}
            out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
            n += 1
    if missing:
        raise SystemExit(f"{len(missing)} clips missing in {gen} (e.g. {missing[:3]})")
    print(json.dumps({"arm": a.arm, "videos": n, "root": str(root)}))


if __name__ == "__main__":
    main()
