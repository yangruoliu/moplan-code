#!/usr/bin/env python3
"""Build manifests of the Stage-2 GT-SFT caches: the clips of the frozen id lists (data/ids/stage2_sft), resolved to
their sources.  JSON + stat calls only (no GPU, no tensors).

  phyco.jsonl / phyco_val.jsonl   sample_id, scenario, caption, video_path, stage1_target_path [, v1_payload]
                                  The video comes from the candidate table (--phyco-candidates) when its file exists,
                                  else from the Stage-1 manifests (--phyco-stage1-manifest: the path as written, or
                                  mapped onto --phyco-video-root).  The ground-truth target is
                                  <--phyco-targets>/<sid>.pt.  With --phyco-v1-payloads, an existing 29-frame payload
                                  (<dir>/pools/phyco/<sid>.pt or <dir>/val/phyco/<sid>.pt) is recorded as v1_payload:
                                  build_phyco_gt_cache_v2.py then takes the prompt from it (the caption of that build).
  phyco_missing.jsonl             ids without a raw video or a target: no 41-frame payload can be built; their 29-frame
                                  payload is used by make_pools_v2.py
  p101.jsonl                      rows of --p101-manifest (earlier files win per sample id) for the P101 train + val ids
  ssv2.jsonl                      rows of --ssv2-manifest for the SSv2 train + val ids
  MANIFESTS.json                  counts, and the ids whose builder inputs (plans/<sid>.npz, real_tracks/<sid>.npz under
                                  --p101-inputs / --ssv2-inputs) are missing

usage: make_manifests.py --ids-dir data/ids/stage2_sft --out DIR
         --phyco-targets DIR [--phyco-candidates F] [--phyco-stage1-manifest F ...] [--phyco-video-root DIR]
         [--phyco-v1-payloads DIR] --p101-manifest F [F ...] --p101-inputs DIR --ssv2-manifest F --ssv2-inputs DIR
"""
import argparse
import json
import os
from pathlib import Path

SCEN = ("fall", "ramp", "spring", "multi", "liquid")


def ids(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def jl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids-dir", required=True, help="data/ids/stage2_sft: train/<pool>.txt and val/ids_<name>.txt")
    p.add_argument("--out", required=True, help="output folder of the manifests")
    p.add_argument("--phyco-targets", default="", help="Stage-1 ground-truth targets <sid>.pt of the PhyCo clips")
    p.add_argument("--phyco-candidates", default="", help="jsonl: sample_id, video_path, caption, scenario (checked first)")
    p.add_argument("--phyco-stage1-manifest", action="append", default=[], help="jsonl: sample_id, video_path, caption, scenario")
    p.add_argument("--phyco-video-root", default="", help="extracted PhyCo Kubric root: a Stage-1 manifest video path that does not "
                   "exist is looked up as <root>/<path after '/phyco_raw/'>")
    p.add_argument("--phyco-v1-payloads", default="", help="folder with pools/phyco and val/phyco 29-frame payloads (optional)")
    p.add_argument("--p101-manifest", action="append", default=[], help="jsonl rows: sample_id, video_path, first_frame_path, caption, ...")
    p.add_argument("--p101-inputs", default="", help="folder with plans/<sid>.npz and real_tracks/<sid>.npz of the P101 clips")
    p.add_argument("--ssv2-manifest", default="", help="jsonl rows: sample_id, video_path, first_frame_path, caption, ...")
    p.add_argument("--ssv2-inputs", default="", help="folder with plans/<sid>.npz and real_tracks/<sid>.npz of the SSv2 clips")
    a = p.parse_args()
    lists = Path(a.ids_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    # ---- PhyCo
    if a.phyco_targets:
        cand = {r["sample_id"]: r for r in jl(a.phyco_candidates)} if a.phyco_candidates else {}
        s1 = {}
        for manifest in a.phyco_stage1_manifest:
            for r in jl(manifest):
                s1[r["sample_id"]] = r
        targets = Path(a.phyco_targets)
        v1root = Path(a.phyco_v1_payloads) if a.phyco_v1_payloads else None
        missing = []
        for name, id_list, v1sub in (("phyco", ids(lists / "train/phyco.txt"), "pools/phyco"), ("phyco_val", ids(lists / "val/ids_phyco.txt"), "val/phyco")):
            rows = []
            for sid in id_list:
                video = None
                if sid in cand and os.path.exists(cand[sid]["video_path"]):
                    video = cand[sid]["video_path"]
                elif sid in s1:
                    given = s1[sid]["video_path"]
                    if os.path.exists(given):
                        video = given
                    elif a.phyco_video_root:
                        mapped = Path(a.phyco_video_root) / given.split("/phyco_raw/")[-1].lstrip("/")
                        video = str(mapped) if mapped.exists() else None
                target = targets / f"{sid}.pt"
                v1 = (v1root / v1sub / f"{sid}.pt") if v1root else None
                v1_payload = str(v1.resolve()) if v1 is not None and v1.is_file() else None
                if video is None or not target.is_file():
                    missing.append({"sample_id": sid, "set": name, "video": video is not None, "target": target.is_file(), "v1_payload": v1_payload})
                    continue
                row = {"sample_id": sid, "scenario": (cand.get(sid) or s1.get(sid) or {}).get("scenario", ""),
                       "caption": (cand.get(sid) or s1.get(sid))["caption"], "video_path": video, "stage1_target_path": str(target)}
                if v1_payload:
                    row["v1_payload"] = v1_payload
                rows.append(row)
            (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            report[name] = {"ids": len(id_list), "resolved": len(rows), "with_v1_prompt": sum("v1_payload" in r for r in rows)}
        (out / "phyco_missing.jsonl").write_text("".join(json.dumps(r) + "\n" for r in missing))
        report["phyco_missing"] = len(missing)
    # ---- P101
    if a.p101_manifest:
        need = sorted(set().union(*(ids(lists / f"train/p_{s}.txt") for s in SCEN)) | set(ids(lists / "val/ids_p101.txt")))
        rows = {}
        for manifest in a.p101_manifest:
            for r in jl(manifest):
                rows.setdefault(r["sample_id"], r)
        inp = Path(a.p101_inputs) if a.p101_inputs else None
        p101 = [rows[s] for s in need if s in rows]
        bad = [s for s in need if s not in rows or not os.path.exists(rows[s]["video_path"]) or not os.path.exists(rows[s]["first_frame_path"])
               or inp is None or not (inp / "real_tracks" / f"{s}.npz").is_file() or not (inp / "plans" / f"{s}.npz").is_file()]
        (out / "p101.jsonl").write_text("".join(json.dumps(r) + "\n" for r in p101))
        report["p101"] = {"ids": len(need), "rows": len(p101), "inputs_missing": len(bad), "missing_examples": bad[:5]}
    # ---- SSv2
    if a.ssv2_manifest:
        need = sorted(set(ids(lists / "train/ssv2.txt")) | set(ids(lists / "val/ids_ssv2.txt")))
        rows = {r["sample_id"]: r for r in jl(a.ssv2_manifest)}
        ss = [rows[s] for s in need if s in rows]
        k = Path(a.ssv2_inputs) if a.ssv2_inputs else None
        bad = [s for s in need if s not in rows or k is None or not (k / "real_tracks" / f"{s}.npz").is_file() or not (k / "plans" / f"{s}.npz").is_file()]
        (out / "ssv2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ss))
        report["ssv2"] = {"ids": len(need), "rows": len(ss), "inputs_missing": len(bad), "missing_examples": bad[:5]}
    (out / "MANIFESTS.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
