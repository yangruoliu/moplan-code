"""CPU tests of the inference entry points (no GPU, no model weights).

Run: PYTHONPATH=. python -m pytest tests/test_infer.py   (or: PYTHONPATH=. python tests/test_infer.py)
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate  # noqa: E402
from motion_jepa import cond_v2  # noqa: E402
from motion_jepa.stage2_condition import interface_from_config, read_cached_condition  # noqa: E402
from scripts import infer_stage2_cond_v2 as s2  # noqa: E402
from scripts.s2eval.hold_known_beyond_horizon import hold  # noqa: E402

NEGATIVE_PROMPT_SHA256 = "ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e"


# ------------------------------------------------------------------------------------------------ arguments
def test_parse_args_protocol_defaults():
    a = generate.parse_args(["--image", "x.png", "--prompt", "A ball falls.", "--output", "o.mp4", "--weights", "/w"])
    assert (a.protocol, a.frames, a.fps, a.caption_rule) == ("moprobe", 37, 7.0, "static-suffix")
    assert a.stage1_checkpoint == "/w/stage1"
    assert a.high_checkpoint == "/w/stage2/high" and a.low_checkpoint == "/w/stage2/low"
    assert (a.stage1_text_shard_index, a.stage1_text_num_shards) == (0, 1)
    b = generate.parse_args(["--image", "x.png", "--prompt", "p", "--output", "o.mp4", "--protocol", "physics_iq"])
    assert (b.frames, b.fps, b.caption_rule) == (41, 8.0, "exact")
    c = generate.parse_args(["--image", "x.png", "--prompt", "p", "--output", "o.mp4", "--frames", "17", "--fps", "7",
                             "--stage1-text-shard", "2/4", "--caption-rule", "exact"])
    assert (c.frames, c.fps, c.caption_rule, c.stage1_text_shard_index, c.stage1_text_num_shards) == (17, 7.0, "exact", 2, 4)


def test_parse_args_rejections():
    bad = [["--protocol", "physics_iq", "--frames", "37"],           # the Physics-IQ profile is fixed
           ["--protocol", "physics_iq", "--negative-prompt", "x"],
           ["--frames", "36"],                                       # not 4k+1
           ["--stage1-text-shard", "4/4"]]
    for extra in bad:
        try:
            generate.parse_args(["--image", "x.png", "--prompt", "p", "--output", "o.mp4", *extra])
        except SystemExit:
            continue
        raise AssertionError(f"accepted {extra}")
    try:
        generate.parse_args(["--image", "x.png", "--output", "o.mp4"])
    except SystemExit:
        pass
    else:
        raise AssertionError("accepted a missing prompt")


def test_caption_rule():
    assert generate.caption_of("A hand drops the ruler.", "static-suffix") == "A hand drops the ruler."
    assert generate.caption_of("Two balls roll.  The camera is fixed.", "static-suffix") == \
        "Two balls roll. The camera is fixed. Static shot with no camera movement."
    assert generate.caption_of("Static shot with no camera movement.", "static-suffix") == "Static shot with no camera movement."
    assert generate.caption_of("  keep   spacing ", "exact") == "  keep   spacing "


# ------------------------------------------------------------------------------------------------ seeds
def test_job_seed_values():
    piq = "0001_perspective-left_trimmed-ball-and-block-fall"
    assert [s2.job_seed(42, piq, k) for k in range(3)] == [16160, 1016163, 2016166]
    for sid, value in (("ssv2_9578-ex", 45066), ("ssv2_9578-cf", 19247), ("ps030-pd", 55675), ("rb00073-pd", 91096)):
        assert s2.job_seed(42, sid, 0) == value
        assert generate.job_seed(42, sid, 0) == value
    for k in range(4):
        assert generate.job_seed(7, "any-id", k) == s2.job_seed(7, "any-id", k)


def test_generation_seed_rule():
    a = generate.parse_args(["--image", "x.png", "--prompt", "p", "--output", "o.mp4", "--sample-id", "ssv2_9578-ex"])
    assert generate.generation_seed(a, "ssv2_9578-ex") == 45066
    b = generate.parse_args(["--image", "x.png", "--prompt", "p", "--output", "o.mp4"])
    assert generate.generation_seed(b, "x") == 42                      # no sample id: the flat seed
    c = generate.parse_args(["--image", "x.png", "--prompt", "p", "--output", "o.mp4", "--protocol", "physics_iq",
                             "--sample-id", "0014_x", "--seed", "43"])
    assert generate.generation_seed(c, "0014_x") == 43                 # Physics-IQ: flat seed per run


# ------------------------------------------------------------------------------------------------ hold rule
def test_hold_known_random():
    torch.manual_seed(0)
    for last in range(0, 9):
        x = torch.randn(27, 10, 4, 6)
        x[cond_v2.CHANNEL["known"], last + 1:] = 0
        x[cond_v2.CHANNEL["known"], last] = 1.0
        y, got_last, filled = hold(x)
        assert got_last == last and filled == 9 - last
        assert torch.equal(y[:, : last + 1], x[:, : last + 1])        # up to the last known latent: untouched
        for j in range(last + 1, 10):
            assert torch.equal(y[:, j], x[:, last])                    # all 27 channels copied, not only `known`
        from scripts.s2eval_verified.gen_verified import hold_known_beyond_horizon
        assert torch.equal(hold_known_beyond_horizon(x), y)            # the on-the-fly copy of the rule agrees
        assert torch.equal(hold_known_beyond_horizon(x[None])[0], y)
    x = torch.randn(27, 6, 3, 3)
    x[cond_v2.CHANNEL["known"]] = 0.5
    y, last, filled = hold(x)
    assert last == 5 and filled == 0 and y is x                        # last latent known: unchanged
    x[cond_v2.CHANNEL["known"]] = 0
    y, last, filled = hold(x)
    assert last == -1 and filled == 0 and y is x                       # no plan known at all: unchanged


def test_hold_known_fp16_roundtrip():
    x = torch.randn(27, 10, 32, 56).half()
    x[20, 8:] = 0
    y, _, _ = hold(x.float())
    z = y.to(x.dtype)
    assert z.dtype == torch.float16 and torch.equal(z[:, 8], x[:, 7]) and torch.equal(z[:, :8], x[:, :8])


# ------------------------------------------------------------------------------------------------ cache item contract
class _Dist:
    def __init__(self, value):
        self.value = value

    def mode(self):
        return self.value


class _Encoded:
    def __init__(self, value):
        self.latent_dist = _Dist(value)


class StubVAE:
    """Deterministic stand-in for AutoencoderKLWan.encode: [1, 3, F, H, W] -> [1, 16, (F-1)/4+1, H/8, W/8]."""

    def encode(self, video):
        b, _, f, h, w = video.shape
        pooled = torch.nn.functional.avg_pool3d(video[:, :, :1], (1, 8, 8))           # [1, 3, 1, h/8, w/8]
        base = pooled.repeat(1, 6, (f - 1) // 4 + 1, 1, 1)[:, :16]
        return _Encoded(base * 0.5)


def synthetic_export(caption: str = "A ball rolls to the right.") -> dict:
    """A Stage-1 export with one square object moving right in the letterbox of a 16:9 frame."""
    side, k, horizons = 160, 16, [0.20833333333333334, 0.4166666666666667, 0.625, 0.8333333333333334, 1.2, 2.0, 4.0]
    masks = torch.zeros(k, side, side, dtype=torch.bool)
    masks[0, 70:90, 40:60] = True
    masks[1, 50:60, 110:125] = True
    flow = torch.zeros(len(horizons), 2, side, side)
    for t, when in enumerate(horizons):
        flow[t, 0][masks[0]] = 8.0 * when / side                    # normalized by the grid side
    prob = torch.zeros(len(horizons), 1, side, side)
    prob[:, 0][:, masks[0]] = 0.9
    selector = torch.zeros(k, len(horizons))
    selector[0] = 0.8
    return {"sample_id": "toy", "caption": caption, "flow_ungated": flow.half(), "activity_input_instance_masks": masks,
            "horizon_seconds": horizons, "task_object_slot_probability": selector, "motion_mask": prob,
            "geometry": {"content_top": 35, "content_left": 0, "content_height": 90, "content_width": 160,
                         "source_height": 720, "source_width": 1280}}


def test_cache_item_contract():
    from scripts.control import plan_condition as pc
    frames, fps = 37, 7.0
    first = (np.arange(256 * 448 * 3).reshape(256, 448, 3) % 251).astype(np.uint8)
    latent = generate.encode_image_latent(StubVAE(), first, frames, 256, 448, torch.device("cpu"))
    assert latent.shape == (16, 10, 32, 56) and latent.dtype == torch.float32
    payload = generate.base_payload("toy", "A ball rolls to the right.", Path("/tmp/toy.png"), latent, frames, fps)
    item = generate.moprobe_cache_item(payload, synthetic_export(), (720, 1280))
    # what infer_stage2_cond_v2.py reads
    for key in ("cache_format_version", "frames", "frame_seconds", "height", "width", "fps", "prompt", "first_frame_path"):
        assert key in item, key
    assert item["cache_format_version"] == cond_v2.CACHE_FORMAT == "stage2_cond_v2"
    assert (item["frames"], item["height"], item["width"], item["fps"]) == (37, 256, 448, 7.0)
    assert item["frame_seconds"] == [round(t, 6) for t in pc.latent_times(37, 7.0)]
    assert item["image_cond_latent"].dtype == torch.float16 and item["image_cond_latent"].shape == (16, 10, 32, 56)
    for kind in ("pred", "zero", "null"):
        cond = item[f"cond_{kind}"]
        assert cond.dtype == torch.float16 and cond.shape == (27, 10, 32, 56) and torch.isfinite(cond.float()).all()
        assert read_cached_condition(item, kind, "legacy27", "cached").shape == (27, 10, 32, 56)
    known = item["cond_pred"][cond_v2.CHANNEL["known"]].float().flatten(1).abs().sum(-1)
    assert bool((known > 0).all())                                     # held: every latent carries `known`
    assert torch.equal(item["cond_pred"][:, 8], item["cond_pred"][:, 7]) and torch.equal(item["cond_pred"][:, 9], item["cond_pred"][:, 7])
    assert float(item["cond_null"][cond_v2.CHANNEL["availability"]].abs().sum()) == 0.0
    assert float(item["cond_zero"][cond_v2.CHANNEL["availability"]].float().min()) == 1.0
    assert item["hold_known_beyond_horizon"] is True
    # the conditions are rendered from the stored fp16 latent and the rounded times, as refresh_predictions.py does
    from scripts import motion_plan as mp
    plan = mp.from_stage1_export(synthetic_export(), (256, 448), native_hw=(720, 1280))
    direct = pc.render_v2("pred", plan, item["image_cond_latent"].float(), item["frame_seconds"], (256, 448),
                          export=synthetic_export(), native_hw=(720, 1280)).half()
    assert torch.equal(hold(direct.float())[0].half(), item["cond_pred"])
    # a cache this long clips consistently to a shorter 4k+1 prefix
    clipped = s2.clip_generation_payload(item, 29)
    assert clipped["frames"] == 29 and clipped["cond_pred"].shape[1] == 8 and len(clipped["frame_seconds"]) == 8


# ------------------------------------------------------------------------------------------------ masks
def test_filter_payload_matches_cli():
    from scripts import filter_sam2_masks_v2 as k16
    from motion_jepa.flow_dit_data import build_instance_inputs
    ids = torch.zeros(1, 160, 160, dtype=torch.long)
    ids[0, 40:70, 30:60] = 1          # small object
    ids[0, 100:125, :] = 2            # floor span touching the border ring: dropped by the border test
    ids[0, 60:80, 100:130] = 3
    masks, _, channels = build_instance_inputs(ids, 16)
    raw = {"sample_id": "toy", "instance_masks": masks.to(torch.uint8), "instance_channels": channels.to(torch.uint8),
           "source": "sam2.1_hiera_large_automatic+nearest_labels_v2", "num_instances": 3}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "raw").mkdir()
        torch.save(raw, tmp / "raw" / "toy.pt")
        threads = torch.get_num_threads()
        k16.main(["--in-dir", str(tmp / "raw"), "--out-dir", str(tmp / "k16")])
        torch.set_num_threads(threads)
        cli = torch.load(tmp / "k16" / "toy.pt", map_location="cpu")
    k16.configure(**generate.K16)
    mine, stats = k16.filter_payload(torch.load(_save_load(raw), map_location="cpu"))
    assert torch.equal(cli["instance_masks"], mine["instance_masks"]) and torch.equal(cli["instance_channels"], mine["instance_channels"])
    assert cli["filter"] == mine["filter"] and mine["instance_masks"].dtype == torch.float16
    assert mine["num_instances"] == 2 and stats["n_drop_border"] + stats["n_drop_area"] >= 1


def _save_load(payload: dict) -> str:
    path = Path(tempfile.mkdtemp()) / "raw.pt"
    torch.save(payload, path)
    return str(path)


def test_letterbox_canvas():
    from scripts.bench import bench_common as bc
    img = (np.random.default_rng(0).integers(0, 255, (480, 640, 3))).astype(np.uint8)
    canvas = bc.letterbox_16x9_array(img)
    assert canvas.shape == (720, 1280, 3)
    assert canvas[:, :160].max() == 0 and canvas[:, 1120:].max() == 0      # 4:3 -> pillarbox bars of 160 px
    assert canvas[:, 160:1120].max() > 0
    wide = bc.letterbox_16x9_array(np.full((720, 1280, 3), 7, np.uint8))
    assert (wide == 7).all()                                                # a 16:9 frame of the canvas size is unchanged


# ------------------------------------------------------------------------------------------------ constants / configs
def test_negative_prompt_bytes():
    from scripts.infer_stage2_oracle import WAN_BENCHMARK_NEGATIVE_PROMPT
    from scripts.s2eval_verified.gen_verified import OFFICIAL_NEG
    assert hashlib.sha256(WAN_BENCHMARK_NEGATIVE_PROMPT.encode()).hexdigest() == NEGATIVE_PROMPT_SHA256
    assert OFFICIAL_NEG == WAN_BENCHMARK_NEGATIVE_PROMPT


def test_inference_config():
    config = yaml.safe_load((ROOT / "configs" / "stage2" / "inference.yaml").read_text())
    model, evaluation = config["model"], config["evaluation"]
    assert evaluation == {"inference_steps": 30, "guidance_scale": 5.0}
    assert interface_from_config(model) == "legacy27"
    assert model["injection_blocks"] == list(range(40))
    expected = {"architecture": "wan2.2-i2v-a14b", "adapter_architecture": "onlyflow", "lora_rank": 32, "lora_alpha": 32,
                "boundary_sigma": 0.9, "motion_condition_channels": 20, "motion_condition_extra_channels": 7,
                "adapter_hidden_channels": 128, "adapter_depth": 2, "adapter_levels": 3, "adapter_heads": 8, "adapter_mixer_rank": 128}
    assert {k: model[k] for k in expected} == expected
    from scripts.s2eval_verified.gen_verified import N
    assert (N["steps"], N["guidance"], N["frames"], N["fps"], N["height"], N["width"]) == (30, 5.0, 41, 8, 256, 448)


def test_query_seconds():
    horizons = [float(v) for v in generate.QUERY_SECONDS.split(",")]
    assert horizons == [5 / 24, 10 / 24, 15 / 24, 20 / 24, 1.2, 2.0, 4.0]
    try:
        from scripts.clean_r2.export_p101_stage1 import QUERY_SECONDS
    except Exception:          # noqa: BLE001  (module optional in a trimmed checkout)
        return
    assert QUERY_SECONDS == generate.QUERY_SECONDS


def test_weights_manifest():
    manifest = json.loads((ROOT / "tools" / "weights_manifest.json").read_text())
    targets = {entry["target"]: entry for entry in manifest["files"]}
    for name in ("stage1/backend_params.pt", "stage1/decoder.pt", "stage1/trainer_state.json", "stage1/resolved_config.yaml",
                 "stage1/prior_latent.pt", "stage2/high/lora_params.pt", "stage2/high/motion_adapter.pt",
                 "stage2/low/lora_params.pt", "stage2/low/motion_adapter.pt"):
        assert name in targets and len(targets[name]["sha256"]) == 64, name
    assert manifest["repo_id"] == "yrlyrl/MoPlan"


def test_generate_moprobe_wiring():
    """generate.main for --protocol moprobe --base with the VAE and the three Stage-2 phases stubbed: checks the cache
    item it hands to infer_stage2_cond_v2.py, the phase arguments, the seed, and the written mp4 / npy / json."""
    import diffusers
    from PIL import Image
    calls, saved = [], {}

    def fake_main(argv):
        value = lambda flag: argv[argv.index(flag) + 1]                 # noqa: E731
        calls.append(list(argv))
        out_dir = Path(value("--output-dir"))
        out_dir.mkdir(parents=True, exist_ok=True)
        if value("--phase") == "low":
            cache = torch.load(Path(value("--cache-dir")) / "toy-id.pt", weights_only=False)
            saved["cache"] = cache
            frames = np.zeros((cache["frames"], 256, 448, 3), np.uint8)
            frames[:, :, :, 0] = np.arange(cache["frames"], dtype=np.uint8)[:, None, None]
            arm = "base" if "--base" in argv else "pred"
            np.save(out_dir / f"{arm}__toy-id.npy", frames)

    original_vae, original_main = diffusers.AutoencoderKLWan.from_pretrained, s2.main
    diffusers.AutoencoderKLWan.from_pretrained = staticmethod(lambda *a, **k: _StubVAEModule())
    s2.main = fake_main
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            Image.fromarray((np.arange(720 * 1280 * 3).reshape(720, 1280, 3) % 253).astype(np.uint8)).save(tmp / "frame.png")
            out = tmp / "out" / "clip.mp4"
            generate.main(["--image", str(tmp / "frame.png"), "--prompt", "A cup slides left.", "--output", str(out), "--base",
                           "--sample-id", "toy-id", "--wan-model", str(tmp / "wan"), "--save-intermediates"])
            meta = json.loads(out.with_suffix(".json").read_text())
            frames = np.load(out.with_suffix(".npy"))
            assert out.is_file() and frames.shape == (37, 256, 448, 3)
            assert meta["seed"] == s2.job_seed(42, "toy-id", 0) and meta["seed_rule"] == "job_seed(seed, sample_id, k)"
            assert meta["frames_sha256"] == hashlib.sha256(frames.tobytes()).hexdigest()
            assert meta["image"] == "frame.png" and meta["image_sha256"] == hashlib.sha256((tmp / "frame.png").read_bytes()).hexdigest()
            assert (tmp / "out" / "clip.work" / "cache" / "toy-id.pt").is_file()       # kept with --save-intermediates
            base_calls = list(calls)
            calls.clear()
            torch.save(synthetic_export("A cup slides left."), tmp / "export.pt")
            generate.main(["--image", str(tmp / "frame.png"), "--prompt", "A cup slides left.", "--output", str(tmp / "out" / "moplan.mp4"),
                           "--sample-id", "toy-id", "--wan-model", str(tmp / "wan"), "--weights", str(tmp / "weights"),
                           "--stage1-export", str(tmp / "export.pt")])
            assert not (tmp / "out" / "moplan.work").exists()                       # removed without --save-intermediates
            pred = saved["cache"]
            assert pred["hold_known_beyond_horizon"] is True and torch.equal(pred["cond_pred"][:, 9], pred["cond_pred"][:, 7])
            assert all(c[c.index("--high-checkpoint") + 1] == str(tmp / "weights" / "stage2" / "high") for c in calls)
            assert all(c[c.index("--arms") + 1] == "pred" and "--base" not in c for c in calls)
            calls[:] = base_calls
    finally:
        diffusers.AutoencoderKLWan.from_pretrained, s2.main = original_vae, original_main
    assert [c[c.index("--phase") + 1] for c in calls] == ["embed", "high", "low"]
    for c in calls:
        assert c[c.index("--seed-mode") + 1] == "flat" and int(c[c.index("--seed") + 1]) == s2.job_seed(42, "toy-id", 0)
        assert "--base" in c and c[c.index("--arms") + 1] == "base"
    cache = saved["cache"]
    assert cache["frames"] == 37 and cache["prompt"] == "A cup slides left." and cache["cond_null"].shape == (27, 10, 32, 56)


def test_generate_physics_iq_wiring():
    """generate.main for --protocol physics_iq with gen_verified's embedding, generator and writer stubbed."""
    from PIL import Image
    from scripts.s2eval_verified import gen_verified as gv
    seen = {}

    class FakeGenerator:
        def __init__(self, ns, contract):
            seen["ns"] = ns

        def condition(self, item, image):
            return torch.zeros(27, 11, 32, 56).half()

        def run(self, item, seed, deadline, partial, claims):
            assert (Path(claims) / item["id"]).is_dir()
            seen["item"], seen["seed"] = item, seed
            return {"frames": np.full((41, 256, 448, 3), 9, np.uint8), "stats": {"segments": 1}}

    def fake_embed(ns):
        seen["embed"] = ns
        Path(ns.embeds).write_bytes(b"x")

    def fake_write(frames, raw, tmp, fps):
        Path(raw).parent.mkdir(parents=True, exist_ok=True)
        np.save(raw, frames)                                            # stand-ins for the raw and trimmed mp4s
        np.save(tmp, frames[: 5 * fps])
        Path(str(raw) + ".npy").rename(raw)
        Path(str(tmp) + ".npy").rename(tmp)

    def fake_probe(path):
        n = int(np.load(path).shape[0])
        return {"frames": n, "fps": 8.0, "width": 448, "height": 256, "duration": n / 8.0}

    originals = (gv.NativeGenerator, gv.phase_embed, gv.write_official, gv.probe)
    gv.NativeGenerator, gv.phase_embed, gv.write_official, gv.probe = FakeGenerator, fake_embed, fake_write, fake_probe
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            Image.fromarray(np.zeros((2160, 3840, 3), np.uint8)).save(tmp / "switch.jpg")
            torch.save(synthetic_export("Static shot. A ball drops."), tmp / "export.pt")
            (tmp / "weights" / "stage2" / "high").mkdir(parents=True)
            (tmp / "weights" / "stage2" / "low").mkdir(parents=True)
            out = tmp / "piq.mp4"
            generate.main(["--protocol", "physics_iq", "--image", str(tmp / "switch.jpg"), "--prompt", "Static shot. A ball drops.",
                           "--output", str(out), "--seed", "43", "--wan-model", str(tmp / "wan"), "--weights", str(tmp / "weights"),
                           "--stage1-export", str(tmp / "export.pt"), "--save-intermediates"])
            assert out.is_file() and np.load(out.with_suffix(".npy")).shape == (41, 256, 448, 3)
            work = tmp / "piq.work"
            assert (work / "condition.pt").is_file() and (work / "native_exports" / "switch.pt").is_file()
            assert (work / "native_ckpt" / "high").resolve() == (tmp / "weights" / "stage2" / "high").resolve()
            meta = json.loads(out.with_suffix(".json").read_text())
            assert meta["seed"] == 43 and meta["seed_rule"] == "flat" and meta["fps"] == 8.0
    finally:
        gv.NativeGenerator, gv.phase_embed, gv.write_official, gv.probe = originals
    ns = seen["ns"]
    assert ns.native_hw == (2160, 3840) and ns.hold_known is True and ns.plan_kind == "pred" and ns.attn == "native" and not ns.base
    assert seen["item"]["prompt"] == "Static shot. A ball drops." and seen["seed"] == 43    # physics_iq: the prompt verbatim


class _StubVAEModule(StubVAE):
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


def test_infer_stage2_cli_parses():
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()) as text:
        try:
            s2.main(["--help"])
        except SystemExit as exit_:
            assert exit_.code == 0
    assert "--seed-mode" in text.getvalue() and "--wan-model" in text.getvalue()


# ------------------------------------------------------------------------------------------------ video tools
def test_duration_probe_never_skips():
    """gen_verified.probe: without a working ffprobe the duration comes from PyAV; with neither it raises (a NaN
    duration used to pass the 5.000 s check)."""
    import shutil
    import stat
    from scripts.s2eval_verified import gen_verified as gv
    original = shutil.which
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw, trimmed = tmp / "raw.mp4", tmp / "trimmed.mp4"
        gv.write_official(np.zeros((41, 64, 64, 3), np.uint8), raw, trimmed, 8)    # bundled ffmpeg: libx264 + -t 5
        broken = tmp / "ffprobe"
        broken.write_text("#!/bin/sh\nexit 127\n")                                # like an ffprobe missing a library
        broken.chmod(broken.stat().st_mode | stat.S_IEXEC)
        try:
            for which in (lambda name: None, lambda name: str(broken)):
                shutil.which = which
                assert gv.duration_tool() == "pyav"
                info = gv.probe(trimmed)
                assert info["frames"] == 40 and abs(info["duration"] - 5.0) <= 1e-3, info
                assert gv.probe(raw)["frames"] == 41
            (tmp / "bad.mp4").write_bytes(b"not a video")
            try:
                gv.container_duration(tmp / "bad.mp4")
            except RuntimeError:
                pass
            else:
                raise AssertionError("an unmeasurable duration must raise")
        finally:
            shutil.which = original


def test_clip_builders_use_one_ffmpeg():
    """pisa_clips.py and p101_clips_v2.py run the imageio-ffmpeg binary unless --ffmpeg / $FFMPEG say otherwise, and
    p101's frame count raises instead of reporting 0 frames when ffmpeg fails."""
    import os
    import subprocess
    import imageio_ffmpeg
    from scripts.data_real import p101_clips_v2 as p101
    from scripts.eval import pisa_clips
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    saved = os.environ.pop("FFMPEG", None)
    try:
        assert pisa_clips.ffmpeg_exe() == p101.ffmpeg_exe() == exe
        assert p101.ffmpeg_exe("/opt/ffmpeg") == "/opt/ffmpeg"
        os.environ["FFMPEG"] = "/opt/other/ffmpeg"
        assert pisa_clips.ffmpeg_exe() == p101.ffmpeg_exe() == "/opt/other/ffmpeg"
    finally:
        os.environ.pop("FFMPEG", None)
        if saved is not None:
            os.environ["FFMPEG"] = saved
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        subprocess.run([exe, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=7", "-frames:v", "9",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
        assert p101.probe_frames(clip, exe) == 9
        try:
            p101.probe_frames(Path(tmp) / "missing.mp4", exe)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a failed frame count must raise")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {error!r}")
    sys.exit(1 if failures else 0)
