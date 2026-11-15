"""CPU tests of the light RAFT round (no GPU, no model weights).

  python -m pytest tests/test_raft.py

Covered: winner selection (build_pairs.choose and build_pairs.py end to end on synthetic reward files with a stand-in
VAE); reward v5 of synthetic candidates through the stored-measurement path of score_generated.py, and the winner it
implies; candidate file names; the training overrides template (valid trainer arguments, the replay mix, the pools);
the released RAFT config; the shipped id lists; the data order of the training run (every micro-step's pool, kind and
clip, reproduced from the id lists, the selected candidates and the seed); the launcher scripts' syntax.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motion_jepa import cond_v2  # noqa: E402
from motion_jepa.stage2_condition import interface_from_config  # noqa: E402
from scripts import train_s2_v4 as trainer  # noqa: E402
from scripts.control import real_scene as rs  # noqa: E402
from scripts.control import reward as rw  # noqa: E402
from scripts.control import score_generated as sg  # noqa: E402
from scripts.dpo import build_pairs as bp  # noqa: E402

IDS = ROOT / "data" / "ids" / "raft"
SFT_IDS = ROOT / "data" / "ids" / "stage2_sft" / "train"
POOLS = ("phyco", "phycoB", "p101", "p101B")
POOL_IDS = {"phyco": "ids_phyco.txt", "phycoB": "ids_phycoB.txt", "p101": "ids_p101_rl.txt", "p101B": "ids_p101B.txt"}
GROUP, MARGIN, MICRO_STEPS = 8, 0.2, 1200            # 300 optimizer steps x 4 accumulated micro-steps
# sha256 of the (pool, kind, sample_id) sequence of the 1,200 micro-steps of the released RAFT run, one tab-separated
# line per micro-step: what the trainer drew from the replay pools and the selected candidates.
RELEASED_TRACE_SHA256 = "f7b307663ab2b33f956f56e325b85a468e394d12cd9afcd72b62b04b18c82cb8"


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def released_winners() -> list[tuple[str, str, int]]:
    lines = (IDS / "winners.tsv").read_text().splitlines()
    assert lines[0].split("\t") == ["sample_id", "pool", "seed_index"]
    return [(sid, pool, int(seed)) for sid, pool, seed in (line.split("\t") for line in lines[1:] if line.strip())]


# ---------------------------------------------------------------------------------------------------------------
# winner selection


def test_choose_picks_best_and_worst_with_margin():
    rewards = {0: 0.1, 1: 0.7, 2: -0.3, 3: 0.5}
    assert bp.choose(rewards, 0.2) == (1, 2)
    assert bp.choose(rewards, 1.0) == (1, 2)
    assert bp.choose(rewards, 1.01) is None


def test_choose_ties_go_to_the_lower_seed_index():
    assert bp.choose({0: 0.5, 1: 0.9, 2: 0.9, 3: 0.1, 4: 0.1}, 0.2) == (1, 3)


def test_choose_rejects_small_or_nonfinite_groups():
    assert bp.choose({0: 0.4}, 0.0) is None
    assert bp.choose({0: 0.4, 1: float("nan")}, 0.0) is None
    assert bp.choose({0: 0.4, 1: 0.4}, 0.0) == (0, 0)          # margin 0 (smoke tests) keeps every finite group


class _Posterior:
    def __init__(self, video: torch.Tensor) -> None:
        self.video = video

    def mode(self) -> torch.Tensor:
        # [1, 3, F, H, W] -> a deterministic stand-in latent [1, 16, (F - 1) / 4 + 1, H / 8, W / 8]
        frames = self.video[:, :, ::4]
        pooled = torch.nn.functional.avg_pool3d(frames, (1, 8, 8))
        return pooled.repeat(1, 6, 1, 1, 1)[:, :16]


class _FakeVae:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def encode(self, video):
        return type("Encoded", (), {"latent_dist": _Posterior(video)})()


def _write_group(folder: Path, sid: str, rewards: list[float], seed_value: int) -> None:
    rng = np.random.default_rng(seed_value)
    for k, reward in enumerate(rewards):
        name = f"pred__{sid}" + (f"__s{k}" if k else "")
        np.save(folder / f"{name}.npy", rng.integers(0, 256, (9, 16, 32, 3), dtype=np.uint8))
        (folder / f"{name}.reward_v5.json").write_text(json.dumps({"reward": reward, "arm": "pred", "sample_id": sid, "seed_index": k}) + "\n")


def test_build_pairs_winner_only_end_to_end(tmp_path, monkeypatch):
    import diffusers

    generated, out = tmp_path / "panel", tmp_path / "winners"
    generated.mkdir()
    groups = {"clip_a": [0.1, 0.2, 0.9, 0.3, 0.2, 0.1, 0.0, 0.4],          # spread 0.9 -> winner seed 2
              "clip_b": [0.50, 0.55, 0.52, 0.51, 0.50, 0.53, 0.54, 0.56],   # spread 0.06 < 0.2 -> no winner
              "clip_c": [0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6],                # 7 of 8 candidates -> incomplete
              "clip_d": [0.3] * 8}                                          # not in --ids
    for index, (sid, rewards) in enumerate(groups.items()):
        _write_group(generated, sid, rewards, index)
    ids = tmp_path / "ids.txt"
    ids.write_text("clip_a\nclip_b\nclip_c\n")
    monkeypatch.setattr(diffusers, "AutoencoderKLWan", _FakeVae)
    monkeypatch.setattr(sys, "argv", ["build_pairs.py", "--generated", str(generated), "--model-dir", str(tmp_path), "--out", str(out),
                                      "--arm", "pred", "--margin", str(MARGIN), "--suffix", ".reward_v5.json", "--ids", str(ids),
                                      "--winner-only", "--expected-group-size", str(GROUP)])
    bp.main()
    assert sorted(p.name for p in out.glob("*.pt")) == ["clip_a.pt"]
    payload = torch.load(out / "clip_a.pt", weights_only=True)
    assert payload["seed_win"] == 2 and payload["seed_lose"] == 6 and payload["winner_only"] is True
    assert payload["candidate_count"] == GROUP and payload["kind"] == "pred" and "lose" not in payload
    frames = np.load(generated / "pred__clip_a__s2.npy")
    from scripts.cache_stage2_oracle import preprocess
    expected = _FakeVae().encode(preprocess(frames, 16, 32)).latent_dist.mode()[0].float().cpu().half()
    assert payload["win"].dtype == torch.float16 and torch.equal(payload["win"], expected)
    log = {row["sample_id"]: row for row in map(json.loads, (out / "pairs.jsonl").read_text().splitlines())}
    assert set(log) == {"clip_a", "clip_b", "clip_c"}
    assert log["clip_a"]["pair"] == [2, 6] and log["clip_b"]["pair"] is None and log["clip_c"]["pair"] is None
    receipt = json.loads((out / "selection_receipt.json").read_text())
    assert receipt["conditions_selected"] == 1 and receipt["candidate_videos_scored"] == 8 + 8 + 7
    assert receipt["expected_group_size"] == GROUP and receipt["margin"] == MARGIN and receipt["winner_only"] is True


def test_candidate_names():
    assert sg.parse_name("pred__p101_fall_x__cam1") == ("pred", "p101_fall_x__cam1", 0)
    assert sg.parse_name("pred__p101_fall_x__cam1__s7") == ("pred", "p101_fall_x__cam1", 7)
    assert sg.parse_name("pred__phyco_ball_2025-09-05_0a1b2c__s3") == ("pred", "phyco_ball_2025-09-05_0a1b2c", 3)
    assert sg.scene_id("clip@a2", "strip-at") == "clip" and sg.scene_id("clip@a2") == "clip@a2"
    assert sg.reward_suffix(type("A", (), {"suffix": "", "reward_version": "v5"})()) == ".reward_v5.json"


# ---------------------------------------------------------------------------------------------------------------
# reward v5 on synthetic candidates


HEIGHT, WIDTH, FPS, FRAMES = 64, 112, 7.0, 9


def _synthetic_clip():
    """A real clip: object 0 slides right by 2 px per frame, object 1 and the background stay put."""
    masks = np.zeros((2, HEIGHT, WIDTH), bool)
    masks[0, 20:36, 10:30] = True
    masks[1, 20:36, 80:100] = True
    object_points = [np.array([[x, y] for y in (22, 26, 30, 34) for x in (12, 18, 24)], np.float64),
                     np.array([[x, y] for y in (22, 26, 30, 34) for x in (82, 88, 94)], np.float64)]
    dense = rs.dense_grid((HEIGHT, WIDTH), 16)
    points = np.concatenate(object_points + [dense])
    owner = np.array([0] * 12 + [1] * 12 + [rs.DENSE_OWNER] * len(dense), np.int16)
    times = np.arange(FRAMES) / FPS
    return masks, points, owner, times


def _measurement(points, owner, times, velocity, still_shift=0.0):
    """tracks [F, N, 2] and mask-centroid displacements [F, K, 2] of a clip whose object 0 moves at `velocity` px per
    frame along x and whose object 1 drifts by `still_shift` px per frame."""
    steps = np.arange(len(times), dtype=np.float64)[:, None]
    tracks = np.repeat(points[None], len(times), 0).astype(np.float64)
    tracks[:, owner == 0, 0] += velocity * steps
    tracks[:, owner == 1, 0] += still_shift * steps
    object_disp = np.zeros((len(times), 2, 2), np.float32)
    object_disp[:, 0, 0] = velocity * steps[:, 0]
    object_disp[:, 1, 0] = still_shift * steps[:, 0]
    return tracks, np.ones(tracks.shape[:2], bool), object_disp, np.ones((len(times), 2), bool)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    masks, points, owner, times = _synthetic_clip()
    tracks, visible, disp, ok = _measurement(points, owner, times, 2.0)
    built = rw.build_scene(points, owner, [0, 1], masks, tracks, visible, times, float(times[-1]), fps=FPS, max_times=None,
                           object_disp=disp, object_ok=ok, object_times=times)
    path = tmp_path_factory.mktemp("scenes") / "clip.npz"
    rw.save_scene(path, built)
    return rw.load_scene(path), points, owner, times


def _stored_reward(tmp_path, scene_tuple, velocity, still_shift=0.0, name="pred__clip__s0"):
    """Reward v5 of a synthetic candidate through the arrays that `score_generated.py track` stores."""
    loaded, points, owner, times = scene_tuple
    tracks, visible, disp, ok = _measurement(points, owner, times, velocity, still_shift)
    direct = rw.score(loaded, tracks, visible, times, object_disp=disp, object_ok=ok, object_times=times, version="v5", max_times=12)
    path = tmp_path / f"{name}.measure.npz"
    np.savez_compressed(path, object_disp=disp, object_ok=ok, times=times.astype(np.float32), slots=loaded["slots"],
                        tracks=tracks.astype(np.float16), visible=visible)
    stored = sg.score_measure(loaded, np.load(path), version="v5", horizon=None, max_times=12)
    return direct, stored


def test_reward_v5_ranks_synthetic_candidates(tmp_path, scene):
    loaded = scene[0]
    assert loaded["mode"] == "mask" and loaded["version"] == rw.VERSION and len(loaded["times"]) == FRAMES - 1
    assert loaded["moving"][0].any() and not loaded["moving"][1].any() and loaded["static"][1].all()
    candidates = {"correct": (2.0, 0.0), "half_speed": (1.0, 0.0), "still_object_drifts": (2.0, 2.0), "frozen": (0.0, 0.0),
                  "reversed": (-2.0, 0.0)}
    rewards = {}
    for name, (velocity, drift) in candidates.items():
        direct, stored = _stored_reward(tmp_path, scene, velocity, drift, name=f"pred__clip__{name}")
        assert stored["version"] == rw.VERSION and stored["reward"] == pytest.approx(direct["reward"], abs=1e-6)
        assert -2.0 <= stored["reward"] <= 1.0 and np.isfinite(stored["reward_raw"])
        rewards[name] = stored
    assert rewards["correct"]["reward"] == pytest.approx(1.0)
    assert rewards["correct"]["static_leak"] == 0.0 and rewards["correct"]["p_identity"] == 0.0
    assert rewards["still_object_drifts"]["static_leak"] > 0.0
    order = ["correct", "half_speed", "still_object_drifts", "frozen", "reversed"]
    values = [rewards[name]["reward"] for name in order]
    assert values[0] > values[1] > values[3] > values[4] and values[0] > values[2] > values[3]
    # the group's winner under the RAFT rule is the correct candidate
    group = {k: rewards[name]["reward"] for k, name in enumerate(order)}
    assert bp.choose(group, MARGIN) == (0, 4)


# ---------------------------------------------------------------------------------------------------------------
# overrides template, config, id lists


def resolved_overrides(sft_data: Path, raft_data: Path, sft_checkpoint: Path, winners: Path) -> dict:
    """The template resolved like launch/raft.sh resolves it."""
    template = json.loads((ROOT / "configs/stage2/raft_overrides.json").read_text())
    values = {"{repo_root}": ROOT, "{sft_data}": sft_data, "{raft_data}": raft_data, "{sft_checkpoint}": sft_checkpoint, "{winners}": winners}
    out = {}
    for key, value in template.items():
        if key.startswith("_"):
            continue
        if isinstance(value, str):
            for token, path in values.items():
                value = value.replace(token, str(path))
        out[key] = value
    return out


def test_overrides_template_is_a_valid_trainer_recipe(tmp_path, monkeypatch):
    cfg = resolved_overrides(tmp_path / "sft", tmp_path / "raft", tmp_path / "ck", tmp_path / "winners")
    monkeypatch.setattr(sys, "argv", ["train_s2_v4.py", "--config", "x.yaml", "--expert", "high"])
    args = trainer.parse_args()
    assert all(hasattr(args, key) for key in cfg)
    assert not any("{" in value for key, value in cfg.items() if isinstance(value, str) and key != "init_from")
    assert cfg["init_from"] == f"{tmp_path / 'ck'}/{{expert}}"
    assert (cfg["max_steps"], cfg["save_every"], cfg["log_every"], cfg["seed"]) == (300, 100, 10, 20261020)
    assert (cfg["dpo_prob"], cfg["dpo_sft_weight"], cfg["dpo_negative_weight"], cfg["dpo_beta"]) == (0.5, 1.0, 0.0, 2000.0)
    assert cfg["fm_weight_mode"] == "tube" and cfg["max_latent_frames"] == 11
    pools = trainer.parse_pools(cfg["pools"])
    assert sorted(pools) == sorted(["p_fall", "p_ramp", "p_spring", "p_multi", "p_liquid", "ssv2", "phyco"] + [f"raft_{p}" for p in POOLS])
    for name, (folder, ids) in pools.items():
        assert Path(ids).is_file(), ids
        if name.startswith("raft_"):
            assert folder == tmp_path / "raft" / f"cache_{name[len('raft_'):]}" and Path(ids).name == POOL_IDS[name[len("raft_"):]]
        else:
            assert folder.parent == tmp_path / "sft" / "pools" and folder.name in ("p101", "ssv2", "phyco")
    mix = trainer.parse_mix(cfg["mix"], pools)
    assert [pool for pool, _, _ in mix] == ["p_fall", "p_ramp", "p_spring", "p_multi", "p_liquid", "ssv2", "phyco"]
    assert all(kind == "real" for _, kind, _ in mix) and sum(w for _, _, w in mix) == pytest.approx(1.0)
    assert dict((pool, w) for pool, _, w in mix)["phyco"] == pytest.approx(0.4 / 0.99999)


def test_raft_config():
    config = yaml.safe_load((ROOT / "configs/stage2/raft.yaml").read_text())
    assert config["paths"]["wan_model"] == ""
    assert interface_from_config(config["model"]) == "legacy27" and config["model"]["condition_interface"] == "legacy27"
    data, train, model = config["data"], config["train"], config["model"]
    assert (data["frames"], data["fps"], data["max_latent_frames"], data["height"], data["width"]) == (41, 7.0, 11, 256, 448)
    assert data["latent_frame_indices"] == list(range(0, 41, 4)) and data["prompt_pad_length"] == 512
    assert (train["per_gpu_batch"], train["grad_accum_steps"], train["warmup_steps"]) == (1, 4, 50)
    assert (train["lora_learning_rate"], train["adapter_learning_rate"], train["weight_decay"], train["max_grad_norm"]) == (2e-5, 5e-5, 0.01, 1.0)
    assert train["flow_shift"] == 5.0 and train["gradient_checkpointing"] is True
    assert (model["lora_rank"], model["lora_alpha"], model["boundary_sigma"], model["injection_blocks"]) == (32, 32, 0.9, list(range(40)))
    assert config["evaluation"] == {"inference_steps": 30, "guidance_scale": 5.0}


def test_id_lists():
    lists = {pool: read_ids(IDS / POOL_IDS[pool]) for pool in POOLS}
    assert all(len(ids) == 256 and len(set(ids)) == 256 for ids in lists.values())
    every = [sid for ids in lists.values() for sid in ids]
    assert len(set(every)) == 1024
    assert all(sid.startswith("phyco_") for pool in ("phyco", "phycoB") for sid in lists[pool])
    assert all(sid.startswith("p101_") and sid.endswith("__cam1") for pool in ("p101", "p101B") for sid in lists[pool])
    export_phyco = read_ids(IDS / "export_phyco_order.txt")
    assert len(export_phyco) == 512 and set(export_phyco) == set(lists["phyco"]) | set(lists["phycoB"])
    export_p101 = read_ids(IDS / "export_p101_order.txt")
    assert len(export_p101) == len(set(export_p101)) and set(lists["p101"]) | set(lists["p101B"]) <= set(export_p101)
    winners = released_winners()
    assert len({sid for sid, _, _ in winners}) == len(winners)
    assert all(sid in lists[pool] and 0 <= seed < GROUP for sid, pool, seed in winners)


# ---------------------------------------------------------------------------------------------------------------
# data order of the training run


def _fake_payload(path, map_location=None, weights_only=None):
    zeros = torch.zeros
    payload = {"cache_format_version": cond_v2.CACHE_FORMAT, "condition_channels_v2": list(cond_v2.EXTRA_NAMES),
               "video_latent": zeros(16, 2, 1, 1), "image_cond_latent": zeros(16, 2, 1, 1), "prompt_embeds": zeros(1, 4),
               "frame_seconds": [0.0, 4.0 / 7.0], "win": zeros(16, 2, 1, 1), "kind": "pred"}
    for kind in ("real", "pred", "null"):
        payload[f"cond_{kind}"] = zeros(27, 2, 1, 1)
    return payload


def test_training_data_order_matches_the_released_run(tmp_path, monkeypatch):
    """The trainer's sample stream is a pure function of (seed, cursor), the pool rows and the selected candidates.
    Rebuilt here from the shipped id lists and winners on empty stand-in files, it must equal the released run."""
    if not SFT_IDS.is_dir():
        pytest.skip("GT-SFT id lists not present")
    winners = released_winners()
    cfg = resolved_overrides(tmp_path / "sft", tmp_path / "raft", tmp_path / "ck", tmp_path / "winners_all")
    pools = trainer.parse_pools(cfg["pools"])
    for folder, ids in pools.values():
        folder.mkdir(parents=True, exist_ok=True)
        for sid in read_ids(Path(ids)):
            (folder / f"{sid}.pt").touch()
    (tmp_path / "winners_all").mkdir()
    for sid, _, _ in winners:
        (tmp_path / "winners_all" / f"{sid}.pt").touch()
    mix = trainer.parse_mix(cfg["mix"], pools)
    monkeypatch.setattr(torch, "load", _fake_payload)
    dataset = trainer.MixDataset(pools, mix, cfg["seed"], list(cond_v2.EXTRA_NAMES), max_latent_frames=cfg["max_latent_frames"],
                                 dpo_pairs=cfg["dpo_pairs"], dpo_prob=cfg["dpo_prob"], winner_only=True)
    assert len(dataset.pairs) == len(winners)
    homes = {path.stem: home.name for path, home in dataset.pairs}
    assert all(homes[sid] == f"cache_{pool}" for sid, pool, _ in winners)
    rows = [dataset[cursor] for cursor in range(MICRO_STEPS)]
    text = "".join(f"{row['pool']}\t{row['kind']}\t{row['sample_id']}\n" for row in rows)
    assert hashlib.sha256(text.encode()).hexdigest() == RELEASED_TRACE_SHA256
    share = sum(row["kind"] == "dpo" for row in rows) / MICRO_STEPS
    assert 0.45 < share < 0.55


def test_winner_draw_matches_the_trainer_rule():
    """Micro-step c trains on a selected candidate iff Random(f"{seed}:pair:{c}").random() < p."""
    seed, p = 20261020, 0.5
    picks = [random.Random(f"{seed}:pair:{c}").random() < p for c in range(8)]
    assert picks[:3] == [True, True, True]


def test_export_phyco_dry_run(tmp_path, monkeypatch, capsys):
    """The PhyCo Stage-1 export runs the evaluator with the as-run sampler flags on the merged ids, in file order."""
    from scripts.clean_r2 import export_phyco_stage1 as ex

    checkpoint = tmp_path / "stage1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}\n")
    (checkpoint / "resolved_config.yaml").write_text(yaml.safe_dump(
        {"data": {"additional_sources": [{"name": "phyco_anchor", "horizon_seconds": [0.2, 4.0]}]}}))
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("clip_2\nclip_1\n")
    second.write_text("clip_3\n")
    monkeypatch.setattr(sys, "argv", ["export_phyco_stage1.py", "--checkpoint", str(checkpoint), "--ids", str(first), str(second),
                                      "--output", str(tmp_path / "out"), "--dry-run"])
    ex.main()
    contract = json.loads(capsys.readouterr().out)
    command = contract["command"]
    assert command[1].endswith("scripts/eval_stage1_flowdit.py") and contract["ids"] == ["clip_2", "clip_1", "clip_3"]
    flags = dict(zip(command[2::2], command[3::2]))
    assert flags["--source"] == "phyco_anchor" and flags["--split"] == "train" and flags["--reduce"] == "first"
    assert (flags["--num-samples"], flags["--num-steps"], flags["--guidance-scale"], flags["--shrinkage-px"]) == ("1", "1", "1", "0")
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError):
        ex.merged_ids([first, first])


# ---------------------------------------------------------------------------------------------------------------
# launchers


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("script", ["launch/raft.sh", "launch/smoke_raft.sh"])
def test_launcher_syntax(script):
    path = ROOT / script
    assert path.is_file()
    text = path.read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text
    subprocess.run(["bash", "-n", str(path)], check=True)
