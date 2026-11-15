"""CPU tests of the Stage-1 training config, launch helpers and data tools."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from motion_jepa.flow_dit_config import load_config  # noqa: E402
from scripts.stage1_paths import placeholder_values, resolve_placeholders  # noqa: E402
from scripts.train_stage1_flowdit import load_training_config  # noqa: E402

CONFIG = REPO / "configs" / "stage1" / "train.yaml"
IDS = REPO / "data" / "ids" / "stage1"
SOURCES = ("fot", "event_static", "phyco_anchor", "pool_v2", "real_p101", "real_p101r", "real_ssv2")
VALUES = {"MOPLAN_DATA": "/data", "WAN_MODEL_DIR": "/wan", "MOPLAN_WORK_DIR": "/work"}


def strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def test_config_loads_and_resolves():
    config = load_training_config(CONFIG, [], VALUES)
    assert not any("${" in text for text in strings(config))
    assert config["paths"]["wan_model"] == "/wan"
    assert config["paths"]["output_dir"] == "/work/runs/stage1"
    assert config["paths"]["manifest_dir"] == "/data/stage1/fot/manifests"
    names = [source["name"] for source in config["data"]["additional_sources"]]
    assert names == list(SOURCES[1:])
    for source in config["data"]["additional_sources"]:
        for key in ("manifest_dir", "flow_cache_dir", "target_cache_dir", "prompt_table"):
            assert source[key].startswith("/data/stage1/")


def test_recipe_values_of_the_reference_run():
    config = load_config(CONFIG)
    train, model, loss = config["train"], config["model"], config["loss"]
    assert (train["max_steps"], train["warmup_steps"], train["per_gpu_batch"], train["grad_accum_steps"],
            train["global_batch"]) == (6000, 200, 1, 4, 32)
    assert (train["lora_learning_rate"], train["head_learning_rate"], train["weight_decay"]) == (2e-5, 5e-5, 0.01)
    assert train["objective"] == "direct" and train["sampler"] == "source_cycle_v1" and config["seed"] == 42
    assert (model["expert"], model["lora_rank"], model["lora_alpha"]) == ("high", 32, 32)
    assert loss["profile"] == "legacy" and loss["flow_matching"] == 0.0
    assert config["unified_experiment"]["world_size"] == 8
    fractions = [source["mix_fraction"] for source in config["data"]["additional_sources"]]
    assert fractions == [0.12, 0.14, 0.07, 0.08, 0.08, 0.06]
    assert abs(1.0 - sum(fractions) - config["unified_experiment"]["fractions"]["fot"]) < 1e-9


def test_four_gpu_overrides_validate_and_mismatch_fails():
    config = load_training_config(
        CONFIG, ["unified_experiment.world_size=4", "train.grad_accum_steps=8"], VALUES)
    assert config["train"]["grad_accum_steps"] == 8
    with pytest.raises(ValueError):
        load_training_config(CONFIG, ["unified_experiment.world_size=4"], VALUES)


def test_placeholder_rules():
    config = {"a": "${MOPLAN_DATA}/x", "b": ["${WAN_MODEL_DIR}"], "c": 3, "d": "plain"}
    assert resolve_placeholders(config, VALUES) == {"a": "/data/x", "b": ["/wan"], "c": 3, "d": "plain"}
    with pytest.raises(ValueError, match="not set"):
        resolve_placeholders(config, {"MOPLAN_DATA": "/data"})
    partial = resolve_placeholders(config, {"MOPLAN_DATA": "/data"}, strict=False)
    assert partial["b"] == ["${WAN_MODEL_DIR}"] and partial["a"] == "/data/x"
    with pytest.raises(ValueError, match="unknown"):
        resolve_placeholders({"a": "${HOME_DIR}"}, VALUES)
    args = argparse.Namespace(data_root="rel/data", wan_model_dir="", work_dir="")
    values = placeholder_values(args, environ={"WAN_MODEL_DIR": "/env/wan", "MOPLAN_DATA": "/env/data"})
    assert values["MOPLAN_DATA"] == str(Path("rel/data").resolve())
    assert values["WAN_MODEL_DIR"] == "/env/wan" and "MOPLAN_WORK_DIR" not in values


def read_ids(source, split):
    lines = (IDS / f"{source}_{split}.tsv").read_text().splitlines()
    assert lines[0] == "sample_id\tscenario\tscene_id"
    return [line.split("\t") for line in lines[1:] if line]


def test_id_lists_define_the_training_population():
    train_rows = {"fot": 28143, "event_static": 6564, "phyco_anchor": 28491, "pool_v2": 3000,
                  "real_p101": 1423, "real_p101r": 2584, "real_ssv2": 850}
    total = 0
    all_train, all_val = set(), set()
    for source in SOURCES:
        train, val = read_ids(source, "train"), read_ids(source, "val")
        assert len(train) == train_rows[source]
        total += len(train)
        train_ids, val_ids = [row[0] for row in train], [row[0] for row in val]
        assert len(set(train_ids)) == len(train_ids) and len(set(val_ids)) == len(val_ids)
        assert not set(train_ids) & set(val_ids)
        scenes_train = {row[2] for row in train if row[2]}
        scenes_val = {row[2] for row in val if row[2]}
        assert not scenes_train & scenes_val
        all_train |= set(train_ids)
        all_val |= set(val_ids)
    assert total == 71055
    assert not all_train & all_val


def test_assembler_keeps_listed_rows_in_order(tmp_path):
    from scripts.clean_r2 import prepare_unified_stage1_data as assemble

    ids = tmp_path / "ids"
    ids.mkdir()
    (ids / "pool_v2_train.tsv").write_text("sample_id\tscenario\tscene_id\nb\tpool\t\na\tpool\t\n")
    (ids / "pool_v2_val.tsv").write_text("sample_id\tscenario\tscene_id\nc\tpool\t\n")
    rows = [{"sample_id": s, "split": sp, "scenario": "pool", "caption": "x"}
            for s, sp in (("a", "train"), ("c", "val"), ("z", "train"), ("b", "train"))]
    manifest = tmp_path / "in.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    cache = tmp_path / "cache"
    for folder in assemble.CACHE_FOLDERS:
        (cache / folder).mkdir(parents=True)
        for sid in "abcz":
            (cache / folder / f"{sid}.pt").write_bytes(sid.encode())
    out = tmp_path / "out"
    sys_argv = sys.argv
    try:
        sys.argv = ["x", "--source", "pool_v2", "--out", str(out), "--ids", str(ids),
                    "--manifest", str(manifest), "--cache-root", str(cache)]
        assert assemble.main() == 0
    finally:
        sys.argv = sys_argv
    written = (out / "pool_v2" / "manifests" / "train.jsonl").read_text().splitlines()
    assert [json.loads(line)["sample_id"] for line in written] == ["b", "a"]
    assert written[0] == json.dumps(rows[3], sort_keys=True)
    assert (out / "pool_v2" / "targets" / "a.pt").read_bytes() == b"a"
    assert not (out / "pool_v2" / "targets" / "z.pt").exists()


def test_load_prior_accepts_cache_sample_and_released_asset(tmp_path):
    import hashlib
    from scripts.cache_stage1_event_inputs import load_prior

    latent = torch.randn(16, 15, 32, 32).half()
    codec = {"kind": "radial_power"}
    torch.save({"flow_latent": latent, "codec": codec}, tmp_path / "cache.pt")
    torch.save({"prior_latent": latent[:, 0].clone()}, tmp_path / "prior_latent.pt")
    a, codec_a = load_prior(tmp_path / "cache.pt", {"kind": "config"})
    b, codec_b = load_prior(tmp_path / "prior_latent.pt", {"kind": "config"})
    assert a.shape == b.shape == (16, 1, 32, 32) and torch.equal(a, b)
    digest = lambda t: hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert digest(a) == digest(b)
    assert codec_a == codec and codec_b == {"kind": "config"}


def make_data_root(root: Path, config: dict) -> None:
    from scripts.stage1_smoke import source_blocks

    wan = Path(config["paths"]["wan_model"])
    (wan / "transformer").mkdir(parents=True)
    (wan / "vae").mkdir(parents=True)
    (wan / "vae" / "config.json").write_text("{}")
    for _, block in source_blocks(config):
        manifests = Path(block["manifest_dir"])
        manifests.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            (manifests / f"{split}.jsonl").write_text('{"sample_id": "s"}\n')
        for key in ("flow_cache_dir", "target_cache_dir", "condition_instance_cache_dir"):
            if block.get(key):
                Path(block[key]).mkdir(parents=True, exist_ok=True)
        for key in ("prompt_table", "caption_overrides", "exclusions_manifest", "supervision_mover_ids_manifest"):
            if block.get(key):
                Path(block[key]).parent.mkdir(parents=True, exist_ok=True)
                Path(block[key]).write_text("")


def test_smoke_dry_run(tmp_path):
    values = {"MOPLAN_DATA": str(tmp_path / "data"), "WAN_MODEL_DIR": str(tmp_path / "wan"),
              "MOPLAN_WORK_DIR": str(tmp_path / "work")}
    make_data_root(tmp_path, load_training_config(CONFIG, [], values))
    command = [sys.executable, str(REPO / "scripts" / "stage1_smoke.py"), "--data-root", values["MOPLAN_DATA"],
               "--wan-model-dir", values["WAN_MODEL_DIR"], "--out", str(tmp_path / "work"), "--dry-run",
               "--resume-check"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "missing" not in result.stdout
    assert result.stdout.count("smoke_entry.py") == 2 and "--resume-from" in result.stdout
    assert "train.global_batch=8" in result.stdout


def test_smoke_preflight_reports_missing_inputs(tmp_path):
    from scripts.stage1_smoke import preflight

    config = load_training_config(CONFIG, [], {"MOPLAN_DATA": str(tmp_path), "WAN_MODEL_DIR": str(tmp_path),
                                               "MOPLAN_WORK_DIR": str(tmp_path)})
    report = preflight(config)
    assert any("transformer" in problem for problem in report["problems"])
    assert any(problem.startswith("phyco_anchor:") for problem in report["problems"])


def test_event_scripts_require_their_inputs():
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "prepare_stage1_events_v44.py"), "merge",
                             "--output", "unused"], capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 2 and "--v2-task-ids" in result.stderr


def test_launcher_syntax():
    assert subprocess.run(["bash", "-n", str(REPO / "launch" / "train_stage1.sh")]).returncode == 0
