#!/usr/bin/env python
"""Source entries (full_sources.json format) for the real Stage-1 sources, and optionally a config that adds them to
a formal config as additional_sources (for a smoke; the formal mixture is the caller's decision).

usage: make_real_config.py --real-root RD --out-sources RD/real_sources.json
           [--config configs/stage1/train.yaml --out-config SMOKE.yaml --output-dir RUN --fraction real_p101=0.08 ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

NAMES = {"p101": "real_p101", "p101r": "real_p101r", "ssv2": "real_ssv2"}
SCOPES = {"p101": "first_frame_rgb_plus_p101_template_caption_plus_first_frame_sam2_instance_masks",
          "p101r": "first_frame_rgb_plus_p101_template_caption_plus_first_frame_sam2_instance_masks",
          "ssv2": "first_frame_rgb_plus_ssv2_template_caption_plus_first_frame_sam2_instance_masks"}


def entry(root: Path, key: str, fraction: float) -> dict:
    summary = json.loads((root / "summary.json").read_text())
    return {
        "name": NAMES[key], "affine_cache_dir": "", "exclusions_manifest": "",
        "manifest_dir": str(root / "manifests"), "target_cache_dir": str(root / "targets"), "flow_cache_dir": str(root / "flowdit"),
        "condition_instance_cache_dir": str(root / "sam2_conditions"), "condition_mask_source": "first_frame_sam2_instance_masks",
        "condition_scope": SCOPES[key], "prompt_table": str(root / "metadata" / "prompt_table.pt"), "prompt_table_mmap": True,
        "supervision_from_target": True, "supervision_affine": False, "supervision_mode": "valid_partition_v1",
        "supervision_cache_dir": "",
        "supervision_mover_ids_manifest": str(root / "metadata" / "supervision_mover_ids_manifest.json"),
        "label_source": summary["label_source"], "horizon_seconds": [float(v) for v in summary["horizon_seconds"]],
        "mix_fraction": fraction,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-root", required=True, type=Path)
    p.add_argument("--keys", nargs="+", default=["p101", "p101r", "ssv2"])
    p.add_argument("--out-sources", required=True, type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--out-config", type=Path)
    p.add_argument("--output-dir", default="")
    p.add_argument("--fraction", action="append", default=[], help="name=value (default 0.0 in the sources file)")
    a = p.parse_args()
    fractions = {k: float(v) for k, v in (item.split("=", 1) for item in a.fraction)}
    entries = {NAMES[k]: entry(a.real_root / k, k, fractions.get(NAMES[k], 0.0)) for k in a.keys if (a.real_root / k / "summary.json").is_file()}
    a.out_sources.write_text(json.dumps(entries, indent=2) + "\n")
    print("sources", list(entries))
    if a.config and a.out_config:
        config = yaml.safe_load(a.config.read_text())
        config["data"]["additional_sources"] = list(config["data"]["additional_sources"]) + list(entries.values())
        extra = sum(float(s["mix_fraction"]) for s in config["data"]["additional_sources"])
        if not 0 < extra < 1:
            raise SystemExit(f"additional mix fractions sum to {extra}; the primary source needs 1 - sum > 0")
        if a.output_dir:
            config["paths"]["output_dir"] = a.output_dir
        experiment = config.get("unified_experiment", {})
        if isinstance(experiment.get("fractions"), dict):
            experiment["fractions"] = {**experiment["fractions"], **{n: e["mix_fraction"] for n, e in entries.items()},
                                       config["data"].get("name", "primary"): round(1 - extra, 6)}
            experiment["note"] = "smoke copy: real sources appended by scripts/stage1_real/make_real_config.py; not an admitted recipe"
        a.out_config.write_text(yaml.safe_dump(config, sort_keys=False))
        print("config", a.out_config, "primary fraction", round(1 - extra, 4))


if __name__ == "__main__":
    main()
