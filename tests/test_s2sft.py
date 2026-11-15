"""CPU tests of the Stage-2 GT-SFT component (scripts/s2sft, configs/stage2/sft*.{yaml,json}, data/ids/stage2_sft).

Run from the repository root:

    python -m pytest tests/test_s2sft.py -q

No GPU, model weights or datasets are needed.  The last test trains a tiny random Wan transformer for a few steps on
synthetic payloads (about half a minute); set MOPLAN_SKIP_SLOW=1 to skip it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

yaml = pytest.importorskip("yaml")

CONFIG = REPO_ROOT / "configs/stage2/sft.yaml"
OVERRIDES = REPO_ROOT / "configs/stage2/sft_overrides.json"
IDS = REPO_ROOT / "data/ids/stage2_sft"
TRAIN_POOLS = ("p_fall", "p_ramp", "p_spring", "p_multi", "p_liquid", "ssv2", "phyco")
VAL_POOLS = ("p101", "phyco", "ssv2")
SCEN = ("fall", "ramp", "spring", "multi", "liquid")


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def released() -> dict:
    return json.loads(OVERRIDES.read_text())


def parse_trainer_args(argv: list[str]):
    from scripts.s2sft.v2 import train_s2sft_v2 as trainer

    parser = trainer.build_parser()
    args = parser.parse_args(argv)
    args._defaults = {action.dest: action.default for action in parser._actions}
    return trainer.apply_overrides(args)


# --------------------------------------------------------------------------- config and recipe

def test_config_loads():
    from motion_jepa.stage2_condition import interface_from_config

    config = yaml.safe_load(CONFIG.read_text())
    data, model, train = config["data"], config["model"], config["train"]
    assert config["seed"] == 42
    assert config["paths"]["wan_model"] is None
    assert data["frames"] == 41 and data["max_latent_frames"] == 11
    assert data["latent_frame_indices"] == list(range(0, 41, 4))
    assert (data["height"], data["width"], data["fps"]) == (256, 448, 7.0)
    assert interface_from_config(model) == "legacy27"
    assert model["adapter_architecture"] == "onlyflow" and model["injection_blocks"] == list(range(40))
    assert (model["lora_rank"], model["lora_alpha"], model["boundary_sigma"]) == (32, 32, 0.9)
    assert train["per_gpu_batch"] == 1 and train["gradient_checkpointing"] is True
    assert (train["lora_learning_rate"], train["adapter_learning_rate"], train["warmup_steps"]) == (5e-5, 1e-4, 100)
    assert (train["weight_decay"], train["max_grad_norm"], train["flow_shift"]) == (0.01, 1.0, 5.0)


def test_released_overrides_values():
    o = released()
    assert o["max_steps"] == 8000 and o["global_batch"] == 16 and o["seed"] == 42
    assert o["fm_weight_mode"] == "tube" and o["max_latent_frames"] == 11 and o["init_from"] == ""
    assert (o["save_every"], o["keep_every"], o["log_every"], o["val_sigmas"]) == (100, 500, 10, 2)
    for key, value in o.items():
        if isinstance(value, str) and not key.startswith("_"):
            assert not value.startswith("/") and "=/" not in value and "@/" not in value, key
    mix = [item.split(":") for item in o["mix"].split(",")]
    assert [name for name, _, _ in mix] == list(TRAIN_POOLS)
    assert all(kind == "real" for _, kind, _ in mix)
    assert abs(sum(float(w) for _, _, w in mix) - 1.0) < 1e-4


def test_overrides_resolve_with_data_root(tmp_path):
    from scripts.s2sft.v2 import train_s2sft_v2 as trainer

    args = parse_trainer_args(["--config", str(CONFIG), "--expert", "high", "--output-dir", str(tmp_path / "run"),
                               "--overrides", str(OVERRIDES), "--data-root", str(tmp_path / "data"), "--global-batch", "4"])
    assert args.max_steps == 8000 and args.seed == 42 and args.save_every == 100
    assert args.global_batch == 4                      # a flag given on the command line wins over the file
    pools = trainer.parse_pools(args.pools)
    assert list(pools) == list(TRAIN_POOLS)
    data = (tmp_path / "data").resolve()
    for name, (folder, ids_file) in pools.items():
        expected = "p101" if name.startswith("p_") else name
        assert folder == data / "pools" / expected
        assert Path(ids_file) == IDS / "train" / f"{name}.txt" and Path(ids_file).is_file()
    val = trainer.parse_pools(args.val_pools)
    assert list(val) == list(VAL_POOLS)
    for name, (folder, ids_file) in val.items():
        assert folder == data / "val" / name and Path(ids_file) == IDS / "val" / f"ids_{name}.txt"
    mix = trainer.parse_mix(args.mix, pools)
    assert abs(sum(w for _, _, w in mix) - 1.0) < 1e-12


def test_data_root_required(tmp_path):
    with pytest.raises(SystemExit):
        parse_trainer_args(["--config", str(CONFIG), "--expert", "low", "--output-dir", str(tmp_path / "run"),
                            "--overrides", str(OVERRIDES), "--data-root", ""])


# --------------------------------------------------------------------------- id lists

def test_id_lists_disjoint_and_clean():
    train = {name: read_ids(IDS / "train" / f"{name}.txt") for name in TRAIN_POOLS}
    val = {name: read_ids(IDS / "val" / f"ids_{name}.txt") for name in VAL_POOLS}
    for name, ids in {**train, **{f"val_{k}": v for k, v in val.items()}}.items():
        assert ids and len(ids) == len(set(ids)), name
    for scen in SCEN:
        assert all(s.startswith(f"p101_{scen}_") for s in train[f"p_{scen}"])
    assert all(s.startswith("ssv2_") for s in train["ssv2"] + val["ssv2"])
    assert all(s.startswith("phyco_") for s in train["phyco"] + val["phyco"])
    train_all = set().union(*map(set, train.values()))
    val_all = set().union(*map(set, val.values()))
    assert len(train_all) == sum(len(v) for v in train.values())
    assert not train_all & val_all
    exclusions = json.loads((IDS / "exclusion_lists.json").read_text())
    excluded = set().union(*map(set, exclusions.values()))
    trials = {s.split("__")[0] for s in excluded if s.startswith("p101_")}
    assert not train_all & excluded
    assert not {s for s in train_all if s.startswith("p101_") and s.split("__")[0] in trials}
    assert set(read_ids(IDS / "fallback_29frame/phyco_train.txt")) <= set(train["phyco"])
    assert set(read_ids(IDS / "fallback_29frame/phyco_val.txt")) <= set(val["phyco"])


def test_id_lists_small():
    total = sum(p.stat().st_size for p in IDS.rglob("*") if p.is_file())
    assert total < 5 * 2**20


# --------------------------------------------------------------------------- pool building (make_pools_v2.py)

def fake_inventory(root: Path, missing: str = "") -> dict[str, Path]:
    """Empty payload files for every id: 41-frame caches, plus the 29-frame fallbacks of the clips without raw video."""
    cache, fallback = root / "cache", root / "fallback"
    fb_train = set(read_ids(IDS / "fallback_29frame/phyco_train.txt"))
    fb_val = set(read_ids(IDS / "fallback_29frame/phyco_val.txt"))
    for name in TRAIN_POOLS:
        pool = "p101" if name.startswith("p_") else name
        for sid in read_ids(IDS / "train" / f"{name}.txt"):
            folder = fallback / "pools" / pool if sid in fb_train else cache / pool
            folder.mkdir(parents=True, exist_ok=True)
            if sid != missing:
                (folder / f"{sid}.pt").touch()
    for name in VAL_POOLS:
        for sid in read_ids(IDS / "val" / f"ids_{name}.txt"):
            folder = fallback / "val" / name if sid in fb_val else cache / {"phyco": "phyco_val"}.get(name, name)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{sid}.pt").touch()
    return {"cache": cache, "fallback": fallback}


def run_make_pools(root: Path, ids_dir: Path = IDS / "train", extra: tuple = ()) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "scripts/s2sft/v2/make_pools_v2.py"), "--ids-dir", str(ids_dir),
           "--val-ids-dir", str(IDS / "val"), "--exclusions", str(IDS / "exclusion_lists.json"),
           "--cache-dir", str(root / "cache"), "--fallback-dir", str(root / "fallback"), "--out-dir", str(root / "data"),
           "--run", str(root / "run"), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


def test_make_pools_v2_reproduces_released_recipe(tmp_path):
    fake_inventory(tmp_path)
    result = run_make_pools(tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]
    made = json.loads((tmp_path / "run/OVERRIDES.json").read_text())
    ref = released()
    for key, value in ref.items():
        if not key.startswith("_") and key not in ("pools", "val_pools"):
            assert made[key] == value, key
    data = tmp_path / "data"
    from scripts.s2sft.v2 import train_s2sft_v2 as trainer
    pools = trainer.parse_pools(made["pools"])
    assert list(pools) == list(TRAIN_POOLS)
    for name, (folder, ids_file) in pools.items():
        assert folder == data / "pools" / ("p101" if name.startswith("p_") else name)
        assert read_ids(Path(ids_file)) == read_ids(IDS / "train" / f"{name}.txt")
    for name in VAL_POOLS:
        assert sorted(read_ids(data / "val" / f"ids_{name}.txt")) == sorted(read_ids(IDS / "val" / f"ids_{name}.txt"))
    receipt = json.loads((tmp_path / "run/POOLS_RECEIPT.json").read_text())
    assert receipt["v1_payload_fallback"]["phyco"]["n"] == len(read_ids(IDS / "fallback_29frame/phyco_train.txt"))
    assert receipt["v1_payload_fallback"]["p101"]["n"] == receipt["v1_payload_fallback"]["ssv2"]["n"] == 0
    assert receipt["validation"]["phyco"]["v1_payloads"] == len(read_ids(IDS / "fallback_29frame/phyco_val.txt"))
    assert receipt["validation"]["p101"]["v1_payloads"] == receipt["validation"]["ssv2"]["v1_payloads"] == 0
    linked = {p.stem for p in (data / "pools/phyco").glob("*.pt")}
    assert linked == set(read_ids(IDS / "train/phyco.txt"))


def test_make_pools_v2_refuses_leaks_and_missing(tmp_path):
    fake_inventory(tmp_path, missing=read_ids(IDS / "train/ssv2.txt")[0])
    result = run_make_pools(tmp_path)
    assert result.returncode != 0 and "payloads exist neither" in (result.stderr + result.stdout)
    leaky = tmp_path / "leaky_ids"
    leaky.mkdir()
    exclusions = json.loads((IDS / "exclusion_lists.json").read_text())
    for name in TRAIN_POOLS:
        ids = read_ids(IDS / "train" / f"{name}.txt")
        if name == "ssv2":
            ids = ids + [exclusions["ssv2_official_validation"][0]]
        (leaky / f"{name}.txt").write_text("".join(s + "\n" for s in ids))
    (tmp_path / "cache/ssv2" / f"{exclusions['ssv2_official_validation'][0]}.pt").touch()
    (tmp_path / "cache/ssv2" / f"{read_ids(IDS / 'train/ssv2.txt')[0]}.pt").touch()
    result = run_make_pools(tmp_path, ids_dir=leaky)
    assert result.returncode != 0 and "evaluation clips in the pools" in (result.stderr + result.stdout)


# --------------------------------------------------------------------------- manifests (make_manifests.py)

def test_make_manifests_fake_sources(tmp_path):
    phyco = read_ids(IDS / "train/phyco.txt")[:3] + read_ids(IDS / "val/ids_phyco.txt")[:1]
    p101 = read_ids(IDS / "train/p_fall.txt")[:2] + read_ids(IDS / "val/ids_p101.txt")[:1]
    ssv2 = read_ids(IDS / "train/ssv2.txt")[:2]
    (tmp_path / "targets").mkdir()
    rows = []
    for k, sid in enumerate(phyco):
        (tmp_path / "targets" / f"{sid}.pt").touch()
        video = tmp_path / "kubric" / sid / "rgba.mp4"
        video.parent.mkdir(parents=True)
        video.touch()
        # the first clip keeps a path that only exists under --phyco-video-root
        path = f"/elsewhere/phyco_raw/{sid}/rgba.mp4" if k == 0 else str(video)
        rows.append({"sample_id": sid, "video_path": path, "caption": f"caption {k}", "scenario": "s"})
    (tmp_path / "s1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    real_rows = {}
    for name, ids in (("p101", p101), ("ssv2", ssv2)):
        for sub in ("plans", "real_tracks"):
            (tmp_path / name / sub).mkdir(parents=True)
        for sid in ids:
            for sub in ("plans", "real_tracks"):
                (tmp_path / name / sub / f"{sid}.npz").touch()
            (tmp_path / name / f"{sid}.mp4").touch()
            (tmp_path / name / f"{sid}.png").touch()
        real_rows[name] = [{"sample_id": s, "video_path": str(tmp_path / name / f"{s}.mp4"),
                            "first_frame_path": str(tmp_path / name / f"{s}.png"), "caption": "c"} for s in ids]
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in real_rows[name]))
    cmd = [sys.executable, str(REPO_ROOT / "scripts/s2sft/v2/make_manifests.py"), "--ids-dir", str(IDS), "--out", str(tmp_path / "m"),
           "--phyco-targets", str(tmp_path / "targets"), "--phyco-stage1-manifest", str(tmp_path / "s1.jsonl"),
           "--phyco-video-root", str(tmp_path / "kubric"), "--p101-manifest", str(tmp_path / "p101.jsonl"),
           "--p101-inputs", str(tmp_path / "p101"), "--ssv2-manifest", str(tmp_path / "ssv2.jsonl"), "--ssv2-inputs", str(tmp_path / "ssv2")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads((tmp_path / "m/MANIFESTS.json").read_text())
    assert report["phyco"]["resolved"] == 3 and report["phyco_val"]["resolved"] == 1
    assert report["phyco_missing"] == report["phyco"]["ids"] + report["phyco_val"]["ids"] - 4
    made = [json.loads(l) for l in (tmp_path / "m/phyco.jsonl").read_text().splitlines()]
    assert made[0]["video_path"] == str(tmp_path / "kubric" / phyco[0] / "rgba.mp4") and "v1_payload" not in made[0]
    assert report["p101"]["rows"] == 3 and report["ssv2"]["rows"] == 2
    assert report["p101"]["inputs_missing"] == report["p101"]["ids"] - 3


# --------------------------------------------------------------------------- sampling plan

def test_sample_sequence_does_not_depend_on_gpu_count():
    from scripts.s2sft.v2.train_s2sft_v2 import RankStream

    for step in (0, 1, 3999):
        one = set(RankStream(step, step + 1, accum=16, world=1, rank=0).indices)
        eight = set().union(*(RankStream(step, step + 1, accum=2, world=8, rank=r).indices for r in range(8)))
        assert one == eight == set(range(16 * step, 16 * step + 16))


def test_epoch_mix_plan_is_deterministic_and_exhaustive():
    from scripts.s2sft.v2.train_s2sft_v2 import EpochMixPlan

    rows = {"a": [f"a{i}" for i in range(7)], "b": [f"b{i}" for i in range(3)]}
    mix = [("a", "real", 0.7), ("b", "real", 0.3)]
    plan = EpochMixPlan(rows, mix, seed=42, total=200)
    assert plan.items == EpochMixPlan(rows, mix, seed=42, total=200).items
    assert plan.items != EpochMixPlan(rows, mix, seed=43, total=200).items
    for pool, clips in rows.items():
        drawn = [sid for p, _, sid in plan.items if p == pool]
        for start in range(0, len(drawn) - len(clips) + 1, len(clips)):
            assert sorted(drawn[start:start + len(clips)]) == sorted(clips)       # one epoch = every clip once


def test_adapter_inputs_are_moved_to_the_device():
    """The trainer's loader is not passed to accelerator.prepare, so its batches stay on the CPU.  With several
    processes DDP moves the adapter input to the GPU; a single process (launch/smoke_sft.sh, NPROC=1) does not.  Every
    adapter call of the trainer must therefore move its condition to `device` itself.  The CPU tests cannot see a
    device mismatch, so this checks the source."""
    import ast

    tree = ast.parse((REPO_ROOT / "scripts/s2sft/v2/train_s2sft_v2.py").read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"adapter", "core"}]
    assert len(calls) >= 2                                                   # the training step and the validation
    for call in calls:
        (arg,) = call.args
        assert (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr == "to"
                and [ast.unparse(a) for a in arg.args] == ["device"]), ast.unparse(call)


# --------------------------------------------------------------------------- smoke helper

def test_make_smoke_run_synthetic(tmp_path):
    pytest.importorskip("torch")
    from scripts.s2sft.v2 import make_smoke_run as smoke
    from scripts.s2sft.v2.make_pools_v2 import check

    for offset, name in enumerate(("p101", "ssv2", "phyco")):
        smoke.synthetic_payloads(tmp_path / "payloads" / name, name, 3, seed=offset, latent_hw=(8, 8), full=3, short=2)
    smoke.build(tmp_path / "payloads", tmp_path / "run", 0)
    for path in sorted((tmp_path / "payloads").rglob("*.pt")):
        _, ok, info = check(str(path))
        assert ok, info
    made = json.loads((tmp_path / "run/OVERRIDES.json").read_text())
    assert made["max_steps"] == 8000 and made["global_batch"] == 16
    assert made["mix"] == "p101:real:0.50000,ssv2:real:0.10000,phyco:real:0.40000"


# --------------------------------------------------------------------------- tiny end-to-end training on the CPU

def tiny_wan(folder: Path) -> None:
    import torch
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    for sub in ("transformer", "transformer_2"):
        WanTransformer3DModel(num_attention_heads=2, attention_head_dim=12, in_channels=36, out_channels=16, text_dim=4096,
                              freq_dim=32, ffn_dim=24, num_layers=40, image_dim=None).save_pretrained(folder / sub)
    (folder / "vae").mkdir(parents=True, exist_ok=True)
    (folder / "vae/config.json").write_text(json.dumps({"_class_name": "AutoencoderKLWan", "latents_mean": [0.0] * 16,
                                                        "latents_std": [1.0] * 16, "z_dim": 16}))


@pytest.mark.skipif(os.environ.get("MOPLAN_SKIP_SLOW") == "1", reason="MOPLAN_SKIP_SLOW=1")
def test_tiny_training_with_released_overrides(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("peft")
    pytest.importorskip("accelerate")
    from scripts.s2sft.v2 import make_smoke_run as smoke

    tiny_wan(tmp_path / "wan")
    data = tmp_path / "data"
    # two payloads per training pool and one per validation pool, named after the first ids of the released lists
    for i, name in enumerate(TRAIN_POOLS):
        pool = "p101" if name.startswith("p_") else name
        scratch = tmp_path / "synthetic" / name
        smoke.synthetic_payloads(scratch, pool, 2, seed=i, latent_hw=(8, 8), full=3, short=2)
        for src, sid in zip(sorted(scratch.glob("*.pt")), read_ids(IDS / "train" / f"{name}.txt")[:2]):
            (data / "pools" / pool).mkdir(parents=True, exist_ok=True)
            src.rename(data / "pools" / pool / f"{sid}.pt")
    for i, name in enumerate(VAL_POOLS):
        scratch = tmp_path / "synthetic" / f"val_{name}"
        smoke.synthetic_payloads(scratch, name, 1, seed=100 + i, latent_hw=(8, 8), full=3)
        (data / "val" / name).mkdir(parents=True, exist_ok=True)
        next(scratch.glob("*.pt")).rename(data / "val" / name / f"{read_ids(IDS / 'val' / f'ids_{name}.txt')[0]}.pt")
    config = yaml.safe_load(CONFIG.read_text())
    config["train"]["num_workers"] = 0
    (tmp_path / "sft.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    env = {**os.environ, "ACCELERATE_USE_CPU": "true", "WAN_MODEL_DIR": str(tmp_path / "wan"),
           "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    base = [sys.executable, str(REPO_ROOT / "scripts/s2sft/v2/train_s2sft_v2.py"), "--config", str(tmp_path / "sft.yaml"),
            "--expert", "high", "--output-dir", str(tmp_path / "run"), "--overrides", str(OVERRIDES), "--data-root", str(data),
            "--global-batch", "1", "--save-every", "1", "--keep-every", "2", "--log-every", "1"]
    first = subprocess.run(base + ["--stop-at-step", "2"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    assert smoke.check(tmp_path / "run", "high", 2) == 0
    second = subprocess.run(base + ["--stop-at-step", "3", "--resume-from", "latest"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]
    assert "checkpoint-000002" in second.stdout and smoke.check(tmp_path / "run", "high", 3) == 0
    state = json.loads((tmp_path / "run/high/checkpoint-000003/trainer_state.json").read_text())
    assert state["max_steps"] == 8000 and state["seed"] == 42 and state["fm_weight_mode"] == "tube"
    assert state["pool_sizes"] == {name: 2 for name in TRAIN_POOLS}
    assert not (tmp_path / "run/high/checkpoint-000001").exists()          # resume points beyond the newest two are removed
