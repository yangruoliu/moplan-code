#!/usr/bin/env python3
"""Few-step, single-GPU smoke test of Stage-1 training on real data rows.

Runs the unchanged trainer (through ``scripts/s1chain/smoke_entry.py``, which
only lifts the 4/8-rank contract) on the first ``--limit`` cached training rows
of every source of the config, for ``--steps`` optimizer steps, then checks
that the run logged finite losses, wrote its checkpoint and ``done.json``.
With ``--resume-check`` a second invocation resumes from that checkpoint for
the same number of steps.

Model and loss settings are those of the config. Only the batch bookkeeping
(1 GPU, ``--grad-accum`` micro-batches per update), validation size, data
loader workers and the scene-coverage minimum (meaningless on a subset) are
overridden.

Example::

    python scripts/stage1_smoke.py --data-root $MOPLAN_DATA \\
        --wan-model-dir $WAN_MODEL_DIR --out work/stage1_smoke

Requirements: one GPU with >= 40 GB, the Wan2.2 ``transformer/`` weights and
``vae/config.json``, and a built Stage-1 data root (docs/DATA_STAGE1.md). A
subset of the data is enough when ``--allow-scenario-holes`` is passed.
``--dry-run`` only resolves the config, checks the inputs and prints the
commands; it needs no GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ENTRY = REPO_ROOT / "scripts" / "s1chain" / "smoke_entry.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage1" / "train.yaml"
SOURCE_KEYS = ("manifest_dir", "flow_cache_dir", "target_cache_dir", "prompt_table")


def disable_rank_contract() -> None:
    """Same patch as scripts/s1chain/smoke_entry.py, for checking the smoke config in this process."""
    import motion_jepa.flow_dit_unified as unified

    unified.validate_unified_runtime = lambda config, actual_world_size: None
    unified.unified_world_size = lambda config, required=False: None


def smoke_overrides(*, grad_accum: int, eval_every: int, val_limit: int) -> list[str]:
    """``--set`` overrides that turn the full recipe into a 1-GPU smoke run."""
    return [
        f"train.grad_accum_steps={grad_accum}",
        f"train.global_batch={grad_accum}",
        f"train.eval_every={eval_every}",
        f"train.eval_samples={val_limit}",
        "train.save_every=1000",
        "train.num_workers=0",
        "train.min_unique_scenes=0",
        "train.min_unique_scenes_by_source={}",
    ]


def source_blocks(config: dict) -> list[tuple[str, dict]]:
    """(name, path block) of the primary source and every additional source."""
    blocks = [(str(config["data"].get("name", "primary")), config["paths"])]
    secondary = config["data"].get("secondary")
    if secondary:
        blocks.append((str(secondary.get("name", "secondary")), secondary))
    blocks.extend((str(source["name"]), source) for source in config["data"].get("additional_sources", []))
    return blocks


def preflight(config: dict) -> dict:
    """Check that every input the trainer opens exists; return per-source row counts."""
    problems, counts = [], {}
    wan = Path(config["paths"]["wan_model"])
    for required in (wan / "transformer", wan / "vae" / "config.json"):
        if not required.exists():
            problems.append(f"missing Wan2.2 input: {required}")
    for name, block in source_blocks(config):
        for key in SOURCE_KEYS:
            if not Path(block[key]).exists():
                problems.append(f"{name}: missing {key} {block[key]}")
        for key in ("caption_overrides", "exclusions_manifest", "supervision_mover_ids_manifest",
                    "condition_instance_cache_dir"):
            value = block.get(key)
            if value and not Path(value).exists():
                problems.append(f"{name}: missing {key} {value}")
        rows = {}
        for split in ("train", "val"):
            manifest = Path(block["manifest_dir"]) / f"{split}.jsonl"
            if manifest.is_file():
                with manifest.open() as handle:
                    rows[split] = sum(1 for line in handle if line.strip())
            else:
                problems.append(f"{name}: missing manifest {manifest}")
        counts[name] = rows
    return {"problems": problems, "manifest_rows": counts}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_run(run_dir: Path, first_step: int, last_step: int) -> list[str]:
    """Return a list of failures for a run that should have reached ``last_step``."""
    failures = []
    records = {int(r["step"]): r for r in read_jsonl(run_dir / "metrics.jsonl")
               if "step" in r and "loss/total" in r}
    for step in range(first_step, last_step + 1):
        record = records.get(step)
        if record is None:
            failures.append(f"no training record for step {step}")
        elif not math.isfinite(float(record["loss/total"])):
            failures.append(f"non-finite loss/total at step {step}")
    checkpoint = run_dir / f"checkpoint-{last_step:06d}"
    if not (checkpoint / "checkpoint_complete.json").is_file():
        failures.append(f"checkpoint not committed: {checkpoint}")
    done = run_dir / "done.json"
    if not done.is_file():
        failures.append("done.json missing")
    elif int(json.loads(done.read_text()).get("steps", -1)) != last_step:
        failures.append(f"done.json does not report step {last_step}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", default="", help="fills ${MOPLAN_DATA} (default: $MOPLAN_DATA)")
    parser.add_argument("--wan-model-dir", default="", help="fills ${WAN_MODEL_DIR} (default: $WAN_MODEL_DIR)")
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh output directory; the trainer writes <out>/<expert>/")
    parser.add_argument("--steps", type=int, default=2, help="optimizer steps per invocation")
    parser.add_argument("--limit", type=int, default=16, help="cached training rows used per source")
    parser.add_argument("--val-limit", type=int, default=4, help="validation rows per source")
    parser.add_argument("--grad-accum", type=int, default=8, help="micro-batches per optimizer step")
    parser.add_argument("--resume-check", action="store_true",
                        help="resume from the first run's checkpoint for another --steps steps")
    parser.add_argument("--allow-scenario-holes", action="store_true",
                        help="accept a data root where some scenarios are only partly cached")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        help="extra section.key=value override passed to the trainer")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and print the commands only")
    args = parser.parse_args()
    if args.steps < 1 or args.limit < 1 or args.val_limit < 1 or args.grad_accum < 1:
        parser.error("--steps, --limit, --val-limit and --grad-accum must be positive")

    disable_rank_contract()
    from scripts.train_stage1_flowdit import load_training_config, placeholder_values

    work_dir = str(args.out.resolve())
    values = placeholder_values(argparse.Namespace(
        data_root=args.data_root, wan_model_dir=args.wan_model_dir, work_dir=work_dir))
    overrides = smoke_overrides(grad_accum=args.grad_accum, eval_every=args.steps,
                                val_limit=args.val_limit) + args.overrides
    config = load_training_config(args.config, overrides, values)
    report = preflight(config)
    print("[smoke] manifest rows " + json.dumps(report["manifest_rows"], sort_keys=True), flush=True)
    for problem in report["problems"]:
        print(f"[smoke] {problem}", flush=True)
    if report["problems"] and not args.dry_run:
        return 1

    expert = str(config["model"]["expert"])
    run_dir = args.out.resolve() / expert
    if (run_dir / "metrics.jsonl").exists() and not args.dry_run:
        print(f"[smoke] {run_dir} already holds a run; use a fresh --out", flush=True)
        return 1
    base = [sys.executable, str(ENTRY), "--config", str(Path(args.config).resolve()), "--work-dir", work_dir,
            "--output-dir", work_dir, "--limit", str(args.limit), "--val-limit", str(args.val_limit)]
    for name, flag in (("MOPLAN_DATA", "--data-root"), ("WAN_MODEL_DIR", "--wan-model-dir")):
        if name in values:
            base += [flag, values[name]]
    if args.allow_scenario_holes:
        base.append("--allow-scenario-holes")
    for override in overrides:
        base += ["--set", override]
    runs = [(base + ["--stop-at-step", str(args.steps)], 1, args.steps)]
    if args.resume_check:
        checkpoint = run_dir / f"checkpoint-{args.steps:06d}"
        runs.append((base + ["--resume-from", str(checkpoint), "--stop-at-step", str(2 * args.steps)],
                     args.steps + 1, 2 * args.steps))

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("OMP_NUM_THREADS", "4")
    for command, first_step, last_step in runs:
        print("[smoke] " + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            print(f"[smoke] trainer exited with {result.returncode}", flush=True)
            return result.returncode
        failures = check_run(run_dir, first_step, last_step)
        for failure in failures:
            print(f"[smoke] FAIL {failure}", flush=True)
        if failures:
            return 1
    if not args.dry_run:
        print(f"[smoke] SMOKE_OK {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
