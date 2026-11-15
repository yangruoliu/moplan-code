#!/usr/bin/env python
"""Contract audit of real Stage-1 sources against an admitted synthetic source (phyco_anchor by default).

  1. file contract   target / flow cache / SAM2 condition: keys, dtypes and shapes of the tensors the loader reads,
                     compared with a reference sample of the reference source
  2. loader          every manifest row through the trainer's own build_dataset(...) + FlowDiTDataset.__getitem__
                     (valid_partition_v1 runtime supervision, mover-ID manifest, SAM2 provenance / first_frame_sha256),
                     and the produced sample dict vs a reference-source sample (keys, shapes, dtypes)
  3. prompts         every manifest caption has an embedding (0 < L < pad) in the source prompt table + negative key
  4. statistics      per split: rows, unique scenes, movers, valid / moving / static supervision fractions per horizon
usage: audit_stage1_real.py --config configs/stage1/train.yaml --sources real_sources.json [--names p101 ssv2] [--workers 8] --out A.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.stage1_real.build_stage1_real_v3  # noqa: E402,F401  (installs the optional h5py stub)
from motion_jepa.flow_dit_config import load_config  # noqa: E402
from motion_jepa.flow_dit_data import caption_key  # noqa: E402
from scripts.stage1_paths import placeholder_values, resolve_placeholders  # noqa: E402

TARGET_KEYS = ("flow", "visibility", "object_mask", "future_mask", "instance_id")
FLOW_KEYS = ("first_frame", "image_cond_latent", "flow_latent")


def describe(value):
    if isinstance(value, torch.Tensor):
        return [str(value.dtype), list(value.shape)]
    return type(value).__name__


def first_row(source: dict, split: str = "train") -> str:
    return json.loads((Path(source["manifest_dir"]) / f"{split}.jsonl").read_text().splitlines()[0])["sample_id"]


def load(dir_: str, sid: str, nested: bool = False) -> dict:
    path = Path(dir_) / "targets" / f"{sid}.pt" if nested else Path(dir_) / f"{sid}.pt"
    if not path.is_file():
        path = Path(dir_) / f"{sid}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def file_contract(source: dict, reference: dict) -> dict:
    sid, ref = first_row(source), first_row(reference)
    out = {}
    for kind, keys, key_dir in (("target", TARGET_KEYS, "target_cache_dir"), ("flow", FLOW_KEYS, "flow_cache_dir")):
        mine, theirs = load(source[key_dir], sid), load(reference[key_dir], ref)
        out[kind] = {k: {"real": describe(mine.get(k)), "reference": describe(theirs.get(k)),
                         "same": describe(mine.get(k)) == describe(theirs.get(k))} for k in keys}
        out[kind + "_extra_keys_real"] = sorted(set(mine) - set(theirs))
        out[kind + "_missing_keys_real"] = sorted(set(theirs) - set(mine))
    cond = load(source["condition_instance_cache_dir"], sid)
    out["condition"] = {k: describe(v) for k, v in cond.items() if k != "provenance"}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--sources", required=True, type=Path)
    p.add_argument("--names", nargs="*")
    p.add_argument("--reference", default="phyco_anchor")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    sys.argv = [sys.argv[0]]
    from scripts.train_stage1_flowdit import build_dataset                    # the trainer's own dataset builder
    config = resolve_placeholders(load_config(a.config), placeholder_values(), strict=False)
    reference = next(s for s in config["data"]["additional_sources"] if s["name"] == a.reference)
    sources = json.loads(a.sources.read_text())
    names = a.names or list(sources)
    report = {"reference": a.reference}
    ref_set = build_dataset(config, "train", limit=1, strict=True, source=reference)
    ref_sample = ref_set[0]
    for name in names:
        source = sources[name]
        entry = {"file_contract": file_contract(source, reference)}
        table = torch.load(source["prompt_table"], map_location="cpu", weights_only=True, mmap=True)
        pad = int(table["prompt_pad_length"])
        for split in ("train", "val"):
            rows = [json.loads(l) for l in (Path(source["manifest_dir"]) / f"{split}.jsonl").read_text().splitlines() if l.strip()]
            keys = [caption_key(r["caption"]) for r in rows]
            lengths = [int(table["embeddings"][k].shape[0]) for k in keys if k in table["embeddings"]]
            dataset = build_dataset(config, split, limit=a.limit, strict=True, source=source)
            loader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=a.workers,
                                                 collate_fn=lambda sample: {
                                                     "sample_id": sample["sample_id"],
                                                     "keys": {k: describe(v) for k, v in sample.items()},
                                                     "per_h": [[float(sample[k][j].sum()) for k in ("flow_valid", "motion_target", "static_certified")]
                                                               for j in range(sample["flow"].shape[0])],
                                                     "mover_px": float(sample["supervision_object_mask"].sum()),
                                                     "slots": int((sample["instance_masks"].flatten(1).sum(1) > 0).sum()),
                                                     "label_source": sample["label_source"]})
            seen, errors, per_h, first_keys = 0, [], None, None
            mover_px, slots = [], []
            iterator = iter(loader)
            while True:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                except Exception as error:                                     # noqa: BLE001
                    errors.append(repr(error)[:300])
                    break
                seen += 1
                first_keys = first_keys or item["keys"]
                totals = torch.tensor(item["per_h"])
                per_h = totals if per_h is None else per_h + totals
                mover_px.append(item["mover_px"])
                slots.append(item["slots"])
            pixels = 160 * 160 * max(seen, 1)
            ref_keys = {k: describe(v) for k, v in ref_sample.items()}
            entry[split] = {
                "manifest_rows": len(rows), "loaded": seen, "loader_errors": errors,
                "unique_scenes": len({r.get("scene_id") for r in rows}),
                "events": dict(Counter(r.get("event") for r in rows)),
                "prompt_coverage": f"{sum(k in table['embeddings'] for k in keys)}/{len(keys)}",
                "prompt_max_tokens": max(lengths) if lengths else 0, "prompt_pad": pad,
                "negative_key_present": table["negative_key"] in table["embeddings"],
                "fraction_per_horizon": {"flow_valid": [round(float(v) / pixels, 4) for v in per_h[:, 0]],
                                         "moving_target": [round(float(v) / pixels, 4) for v in per_h[:, 1]],
                                         "static_certified": [round(float(v) / pixels, 4) for v in per_h[:, 2]]} if per_h is not None else {},
                "mean_mover_px": sum(mover_px) / max(len(mover_px), 1), "mean_condition_slots": sum(slots) / max(len(slots), 1),
                "sample_keys_vs_reference": {
                    "missing_in_real": sorted(set(ref_keys) - set(first_keys or {})),
                    "extra_in_real": sorted(set(first_keys or {}) - set(ref_keys)),
                    "shape_or_dtype_differs": {k: [first_keys[k], ref_keys[k]] for k in ref_keys
                                               if first_keys and k in first_keys and first_keys[k] != ref_keys[k]}},
            }
        report[name] = entry
        print(name, json.dumps({s: {k: entry[s][k] for k in ("manifest_rows", "loaded", "loader_errors", "prompt_coverage")} for s in ("train", "val")}), flush=True)
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print("AUDIT_DONE", a.out)


if __name__ == "__main__":
    main()
