#!/usr/bin/env python3
"""Export a frozen Stage-1 checkpoint on Physics 101 first-frame inputs.

Inputs under --root: stage1_export/p101/manifest_<split>.csv (sample_id, first_frame_path, caption) and the 16 SAM2
instance masks stage1_export/p101/sam2/<sample_id>.pt.  scripts/infer_stage1_physicsiq.py is run with the query
horizons of the released Stage-1 model (QUERY_SECONDS) and writes <output>/flows/<sample_id>.pt; an
admission-<shard>.json receipt ties the output folder to one checkpoint.

Defaults: --checkpoint $MOPLAN_WEIGHTS/stage1, --prior $MOPLAN_WEIGHTS/stage1/prior_latent.pt, --wan-model
$WAN_MODEL_DIR.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from motion_jepa.flow_dit_config import load_config
from motion_jepa.flow_dit_export import checkpoint_provenance

QUERY_SECONDS = "0.20833333333333334,0.4166666666666667,0.625,0.8333333333333334,1.2,2,4"
CHECKPOINT_FILES = ("resolved_config.yaml", "trainer_state.json", "backend_params.pt", "decoder.pt")


def admit_checkpoint(checkpoint: Path) -> tuple[dict, dict]:
    """(trainer_state, resolved_config) of a frozen Stage-1 checkpoint folder, after checking its export contract.

    The folder must hold resolved_config.yaml, trainer_state.json, backend_params.pt and decoder.pt (optimizer,
    scheduler and RNG state are not needed for export).  The config must use the 64-token prompt contract on the
    256 VAE / 160 flow grid with 16 SAM slots."""
    checkpoint = Path(checkpoint)
    for name in CHECKPOINT_FILES:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    config = load_config(checkpoint / "resolved_config.yaml")
    if int(config["data"]["prompt_pad_length"]) != 64:
        raise ValueError("Stage-1 export expects the checkpoint's 64-token training contract")
    if int(config["data"]["flow_resolution"]) != 256 or int(config["data"]["height"]) != 160 or int(config["data"]["emit_instance_masks"]) != 16:
        raise ValueError("P101 inputs require 256 VAE / 160 flow / 16 SAM slots")
    return state, config


def _weights(*parts: str) -> Path | None:
    root = os.environ.get("MOPLAN_WEIGHTS", "")
    return Path(root, *parts) if root else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=_weights("stage1"), required=_weights("stage1") is None,
                   help="Stage-1 checkpoint folder (default: $MOPLAN_WEIGHTS/stage1)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("train", "dev96", "all"), default="all")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--wan-model", type=Path, default=Path(os.environ["WAN_MODEL_DIR"]) if os.environ.get("WAN_MODEL_DIR") else None,
                   required=not os.environ.get("WAN_MODEL_DIR"), help="Wan2.2-I2V-A14B-Diffusers directory (default: $WAN_MODEL_DIR)")
    p.add_argument("--prior", type=Path, default=_weights("stage1", "prior_latent.pt"), required=_weights("stage1") is None,
                   help="Stage-1 prior latent (default: $MOPLAN_WEIGHTS/stage1/prior_latent.pt)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not 0 <= a.shard_index < a.num_shards:
        p.error("invalid shard index/count")
    checkpoint = a.checkpoint.resolve()
    state, config = admit_checkpoint(checkpoint)
    base = a.root / "stage1_export" / "p101"
    manifest, masks = base / f"manifest_{a.split}.csv", base / "sam2"
    rows = list(csv.DictReader(manifest.open()))[a.shard_index::a.num_shards]
    for row in rows:
        for source in (Path(row["first_frame_path"]), masks / f"{row['sample_id']}.pt"):
            if not source.is_file():
                raise FileNotFoundError(source)
    if not a.prior.is_file():
        raise FileNotFoundError(a.prior)
    provenance = checkpoint_provenance(checkpoint)
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "infer_stage1_physicsiq.py"),
               "--config", str(checkpoint / "resolved_config.yaml"), "--checkpoint", str(checkpoint),
               "--manifest", str(manifest), "--output-dir", str(a.output), "--instance-masks", str(masks),
               "--prior-latent", str(a.prior), "--query-horizon-seconds", QUERY_SECONDS,
               "--text-batch-size", "4", "--seed", "42", "--shard-index", str(a.shard_index),
               "--num-shards", str(a.num_shards), "--set", f"paths.wan_model={a.wan_model}"]
    receipt = {"stage1_provenance": provenance, "consumed_coverage": state.get("consumed_coverage"),
               "manifest": str(manifest), "samples": len(rows), "shard_index": a.shard_index,
               "num_shards": a.num_shards, "prompt_pad_length": config["data"]["prompt_pad_length"],
               "command": command}
    print(json.dumps(receipt, indent=2), flush=True)
    if a.dry_run:
        return
    a.output.mkdir(parents=True, exist_ok=True)
    existing_contracts = list(a.output.glob("admission-*.json"))
    if not existing_contracts and any((a.output / "flows").glob("*.pt")):
        raise ValueError("existing exports lack an admitted checkpoint contract; use a fresh output directory")
    for existing in existing_contracts:
        if json.loads(existing.read_text())["stage1_provenance"] != provenance:
            raise ValueError("output directory contains another frozen checkpoint's exports")
    contract = a.output / f"admission-{a.shard_index:04d}.json"
    if contract.is_file() and json.loads(contract.read_text()) != receipt:
        raise ValueError("output shard already belongs to a different checkpoint/input contract")
    contract.write_text(json.dumps(receipt, indent=2) + "\n")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
