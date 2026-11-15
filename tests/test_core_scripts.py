"""CPU tests of the shared plan, condition, reward and cache scripts (no GPU, no model weights).

  python -m pytest tests/test_core_scripts.py

Covered: the import closure of the shared scripts; motion_plan plans (synthetic, from a Stage-1 export, save / load);
plan_condition rendering of the 27-channel condition for every kind; the training labels and flow-matching weights;
the reward (v5 cell anchors, a synthetic scene scored with correct / frozen / reversed motion, mask mode, truncation,
the v4 path); the Stage-1 readout fields; the measured plan of a synthetic real clip; and the checkpoint / tracker
argument checks.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motion_jepa import cond_v2  # noqa: E402
from scripts import motion_plan as mp  # noqa: E402
from scripts.control import plan_condition as pc  # noqa: E402
from scripts.control import real_scene as rs  # noqa: E402
from scripts.control import reward as rw  # noqa: E402
from scripts.piq_gt_v2 import stage1_pred_fields as pf  # noqa: E402

MODULES = (
    "scripts.cache_stage1",
    "scripts.cache_stage2_cond_v2",
    "scripts.cache_stage2_oracle",
    "scripts.cache_stage2_phyco",
    "scripts.cache_stage2_phyco_v2",
    "scripts.clean_r2.export_p101_stage1",
    "scripts.control.cotrack",
    "scripts.control.masktrack",
    "scripts.control.measure",
    "scripts.control.plan_condition",
    "scripts.control.real_scene",
    "scripts.control.reconcile_plans",
    "scripts.control.relabel_v2",
    "scripts.control.reward",
    "scripts.infer_stage2_oracle",
    "scripts.motion_plan",
    "scripts.piq_gt_v2.stage1_pred_fields",
)

SIZE = (64, 112)                 # small work frame (H, W); the latent grid is H / 8 x W / 8
LATENT_HW = (8, 14)
FPS = 7.0


def _box(size, top, bottom, left, right) -> np.ndarray:
    mask = np.zeros(size, bool)
    mask[top:bottom, left:right] = True
    return mask


def _image_latent(frames: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(16, frames, *LATENT_HW, generator=generator)


def _falling_plan() -> mp.MotionPlan:
    mask = _box(SIZE, 8, 24, 40, 56)
    return mp.synthetic_translation("synthetic", mask, [0.5, 1.0, 2.0], direction_deg=90.0, magnitude_px=24.0)


def _export(side: int = 32, drop_grid_px: float = 3.0) -> dict:
    """A Stage-1 export of one object that moves down by `drop_grid_px` grid pixels per horizon (grid = work frame)."""
    horizons = [0.2083, 0.4167, 0.625, 0.8333, 1.2, 2.0, 4.0]
    masks = torch.zeros(16, side, side, dtype=torch.bool)
    masks[0, 4:12, 12:20] = True
    flow = torch.zeros(len(horizons), 2, side, side)
    for index in range(len(horizons)):
        flow[index, 1][masks[0]] = drop_grid_px * (index + 1) / side
    motion = torch.zeros(len(horizons), 1, side, side)
    motion[:, 0][:, masks[0]] = 0.9
    probability = torch.zeros(16, len(horizons))
    probability[0] = 0.8
    return {"sample_id": "synthetic", "flow_ungated": flow, "horizon_seconds": horizons, "activity_input_instance_masks": masks,
            "motion_mask": motion, "task_object_slot_probability": probability,
            "geometry": {"content_top": 0, "content_left": 0, "content_height": side, "content_width": side}}


# ------------------------------------------------------------------------------------------------------ imports
@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


# --------------------------------------------------------------------------------------------- plan and condition
def test_latent_times():
    assert pc.latent_times(9, FPS) == [0.0, 4.0 / FPS, 8.0 / FPS]
    assert len(pc.latent_times(41, 8.0)) == 11
    with pytest.raises(ValueError):
        pc.latent_times(10, FPS)


@pytest.mark.parametrize("kind", ("real", "zero", "null"))
def test_render_v2_shapes(kind):
    plan = _falling_plan()
    times = pc.latent_times(13, FPS)
    latent = _image_latent(len(times))
    condition = pc.render_v2(kind, plan if kind != "null" else None, latent, times, SIZE)
    assert condition.shape == (cond_v2.V2_CHANNELS, len(times), *LATENT_HW)
    assert condition.dtype == torch.float32
    assert torch.isfinite(condition).all()
    availability = condition[cond_v2.CHANNEL["availability"]]
    if kind == "null":
        assert torch.equal(condition, cond_v2.null_condition(latent))
        assert float(availability.abs().max()) == 0.0
    else:
        assert torch.all(availability == 1.0)
        source = condition[cond_v2.CHANNEL["source_mask"]]
        assert float(source.max()) > 0 and torch.equal(source[0], source[-1])


def test_render_zero_and_real_content():
    plan = _falling_plan()
    times = pc.latent_times(13, FPS)
    latent = _image_latent(len(times))
    zero = pc.render_v2("zero", plan, latent, times, SIZE)
    real = pc.render_v2("real", plan, latent, times, SIZE)
    assert float(zero[16:18].abs().max()) == 0.0                                   # flow at target pixels
    assert float(zero[cond_v2.CHANNEL["raw_flow_y"]].abs().max()) == 0.0
    assert float(zero[cond_v2.CHANNEL["known"]].max()) > 0.0                       # known and at rest
    assert float(real[17, 1:].max()) > 0.0                                         # moving down: +y at target pixels
    assert float(real[cond_v2.CHANNEL["raw_flow_y"], 1:].max()) > 0.0
    assert float(real[cond_v2.CHANNEL["raw_flow_x"]].abs().max()) < 1e-6
    rows = torch.arange(LATENT_HW[0], dtype=torch.float32)[:, None]
    target = real[18]                                                              # target mask Mt
    centre = [(float((target[t] * rows).sum() / target[t].sum())) for t in (0, len(times) - 1)]
    assert centre[1] > centre[0]
    assert torch.equal(pc.render_base(plan, latent, times, SIZE), real[: cond_v2.BASE_CHANNELS])


def test_render_pred_from_stage1_export():
    export = _export()
    plan = mp.from_stage1_export(export, SIZE)
    assert [o.slot for o in plan.objects] == [0]
    times = pc.latent_times(9, FPS)
    latent = _image_latent(len(times))
    condition = pc.render_v2("pred", plan, latent, times, SIZE, export=export)
    assert condition.shape == (cond_v2.V2_CHANNELS, len(times), *LATENT_HW)
    assert float(condition[cond_v2.CHANNEL["motion_prob"], 0].abs().max()) == 0.0   # zero at t = 0
    assert float(condition[cond_v2.CHANNEL["motion_prob"], 1:].max()) > 0.0
    assert float(condition[cond_v2.CHANNEL["selector"], 1:].max()) > 0.0
    # a pure translation of the grid is recovered in work pixels (grid 32 x 32 -> work 64 x 112: y scales by 2)
    last = plan.objects[0].displacement()[-1]
    assert last[0] == pytest.approx(0.0, abs=1e-6)
    assert last[1] == pytest.approx(3.0 * 7 * SIZE[0] / 32, rel=1e-6)
    raw_flow, prob = pc.stage1_dense(export, times, SIZE)
    assert raw_flow.shape == (len(times), 2, *SIZE) and prob.shape == (len(times), *SIZE)
    with pytest.raises(ValueError):
        pc.render_v2("pred", plan, latent, times, SIZE)


def test_plan_save_load_roundtrip(tmp_path):
    plan = mp.from_stage1_export(_export(), SIZE)
    path = tmp_path / "plan.npz"
    mp.save_plan(path, plan)
    loaded = mp.load_plan(path)
    assert loaded.size == plan.size and np.array_equal(loaded.anchors, plan.anchors)
    for a, b in zip(plan.objects, loaded.objects):
        assert a.slot == b.slot
        assert np.array_equal(a.mask, b.mask) and np.array_equal(a.affine, b.affine) and np.array_equal(a.known, b.known)
        assert np.array_equal(a.provenance(), b.provenance())
    times = pc.latent_times(9, FPS)
    latent = _image_latent(len(times))
    assert torch.equal(pc.render_v2("real", plan, latent, times, SIZE), pc.render_v2("real", loaded, latent, times, SIZE))


def test_affine_interpolation_and_hold():
    plan = _falling_plan()
    obj = plan.objects[0]
    identity, known = mp.affine_at(obj, plan.anchors, 0.0)
    assert known and np.array_equal(identity, mp.IDENTITY)
    half, _ = mp.affine_at(obj, plan.anchors, 0.25)
    assert half[1, 2] == pytest.approx(0.5 * obj.affine[0, 1, 2])
    held, known_after = mp.affine_at(obj, plan.anchors, 3.0)                       # after the last anchor: held, unknown
    assert np.array_equal(held, obj.affine[-1]) and not known_after
    assert mp.interp_hold(np.array([1.0, 2.0]), [1.0, 2.0], 5.0) == 2.0


def test_labels_and_tube_weights():
    plan = _falling_plan()
    times = pc.latent_times(13, FPS)
    labels = pc.labels_from_plan(plan, times, move_px=8.0, still_px=4.0)
    assert labels["disp"].shape == (1, len(times), 2)
    assert labels["moving"][0, -1] and not labels["moving"][0, 0]
    weight = pc.tube_weights(plan, labels, times, LATENT_HW)
    assert weight.shape == (len(times), *LATENT_HW)
    assert float(weight.mean()) == pytest.approx(1.0, rel=1e-5)
    assert float(weight[1:].max()) > float(weight[0].max())
    masks = pc.object_masks_latent(plan, LATENT_HW)
    assert masks.shape == (1, *LATENT_HW) and float(masks.max()) == 1.0


def test_clip_window():
    assert pc.clip_window(20, False, min_frames=5, max_frames=37, pad_to=13) == (17, 17)
    assert pc.clip_window(10, True, min_frames=5, max_frames=37, pad_to=13) == (13, 10)
    assert pc.clip_window(3, False, min_frames=5) == (0, 0)
    assert pc.largest_4k1(100) == 37


def test_plan_transforms():
    plan = _falling_plan()
    rotated = mp.rotated(plan, 90.0)
    down, turned = plan.objects[0].displacement()[-1], rotated.objects[0].displacement()[-1]
    assert np.allclose(turned, [-down[1], down[0]], atol=1e-9)
    zeroed = mp.zeroed(plan)
    assert np.allclose(zeroed.objects[0].displacement(), 0.0) and zeroed.objects[0].known.all()
    assert not mp.unknown(plan).objects[0].known.any()
    tracks, visible, owner = mp.to_tracks(plan, pc.latent_times(9, FPS), points_per_object=4)
    assert tracks.shape == (3, 4, 2) and visible.shape == (3, 4) and list(owner) == [0] * 4


# ---------------------------------------------------------------------------------------------- Stage-1 readouts
def test_stage1_fields():
    assert pf.canonical("flow_ungated") == "raw" and pf.canonical("flow_task_object_gated") == "soft"
    assert pf.canonical("binary_0.50") == "binary_0.5"
    with pytest.raises(ValueError):
        pf.canonical("binary_1.5")
    export = _export()
    raw, contract = pf.resolve_field(export, "raw")
    assert contract["field"] == "raw" and torch.equal(raw, export["flow_ungated"].float())
    gated, contract = pf.resolve_field(export, "binary_0.5")
    assert contract["selected_slots"] == [0] and torch.equal(gated, raw)
    empty, contract = pf.resolve_field(export, "binary_0.9")
    assert contract["selected_slots"] == [] and float(empty.abs().max()) == 0.0


def test_rotate_objects():
    flow = torch.zeros(3, 2, 16, 16)
    flow[:, 1] = 1.0
    masks = torch.zeros(2, 16, 16, dtype=torch.bool)
    masks[0, :8] = True
    first, angles = pf.rotate_objects(flow, masks, "clip", seed=7)
    second, again = pf.rotate_objects(flow, masks, "clip", seed=7)
    assert torch.equal(first, second) and angles == again and list(angles) == [0]
    assert 120.0 <= angles[0] <= 240.0
    assert torch.allclose(first.square().sum(1), flow.square().sum(1), atol=1e-6)
    assert torch.equal(first[:, :, 8:], flow[:, :, 8:])                           # outside every instance: unchanged
    assert pf.displacement_jump(flow) == 0.0


# ------------------------------------------------------------------------------------------------------- reward
REWARD_SIZE = (256, 448)
REWARD_TIMES = np.arange(13) / FPS


def _reward_layout():
    mover = _box(REWARD_SIZE, 40, 72, 200, 232)
    table = _box(REWARD_SIZE, 176, 240, 64, 384)
    masks, slots = [mover, table], [0, 1]
    points = [mp.sample_points(m, 12) for m in masks]
    owner = sum(([slot] * 12 for slot in slots), [])
    dense = rs.dense_grid(REWARD_SIZE, 16)
    points.append(dense)
    owner += [rs.DENSE_OWNER] * len(dense)
    return np.stack(masks), np.concatenate(points), np.array(owner, np.int16), slots


def _reward_tracks(points, owner, path):
    tracks = np.repeat(points[None], len(REWARD_TIMES), 0).astype(np.float64)
    tracks[:, owner == 0] += path[:, None]
    return tracks, np.ones(tracks.shape[:2], bool)


def _fall(scale: float = 1.0) -> np.ndarray:
    return np.stack([np.zeros(len(REWARD_TIMES)), 120.0 * scale * (REWARD_TIMES / REWARD_TIMES[-1]) ** 2], 1)


def test_cell_score_v5_anchors():
    truth = np.array([0.0, 40.0])
    assert float(rw.cell_score_v5(truth, truth)) == pytest.approx(1.0)
    assert float(rw.cell_score_v5(np.zeros(2), truth)) == pytest.approx(-0.5)
    assert float(rw.cell_score_v5(-truth, truth)) == pytest.approx(-1.0)
    assert float(rw.cell_score_v5(np.array([40.0, 0.0]), truth)) == pytest.approx(0.0, abs=1e-12)
    assert float(rw.cell_score_v5(2.0 * truth, truth)) == pytest.approx(0.75)
    assert float(rw.cell_score_v5(0.5 * truth, truth)) == pytest.approx(0.75)
    assert float(rw.rot_score_v5(np.array(20.0), np.array(20.0))) == pytest.approx(1.0)
    assert float(rw.rot_score_v5(np.array(-20.0), np.array(20.0))) == pytest.approx(-1.0)


def test_reward_v5_orders_correct_frozen_reversed():
    masks, points, owner, slots = _reward_layout()
    fall = _fall()
    scene = rw.build_scene(points, owner, slots, masks, *_reward_tracks(points, owner, fall), REWARD_TIMES, REWARD_TIMES[-1], max_times=None)
    assert scene["version"] == rw.VERSION and scene["mode"] == "points"
    assert scene["moving"][0].any() and not scene["moving"][1].any()
    results = {name: rw.score(scene, *_reward_tracks(points, owner, path), REWARD_TIMES)
               for name, path in (("correct", fall), ("frozen", np.zeros_like(fall)), ("reversed", -fall))}
    assert results["correct"]["reward"] == pytest.approx(1.0)
    assert results["correct"]["reward"] > results["frozen"]["reward"] > results["reversed"]["reward"]
    for result in results.values():
        assert result["version"] == rw.VERSION and -2.0 <= result["reward_raw"] and result["reward"] <= 1.0
    v4 = rw.score(scene, *_reward_tracks(points, owner, fall), REWARD_TIMES, version="v4")
    assert v4["version"] == rw.VERSION_V4
    with pytest.raises(ValueError):
        rw.score(scene, *_reward_tracks(points, owner, fall), REWARD_TIMES, version="v3")


def test_reward_mask_mode_and_io(tmp_path):
    masks, points, owner, slots = _reward_layout()
    fall = _fall()
    tracks, visible = _reward_tracks(points, owner, fall)
    object_disp = np.zeros((len(REWARD_TIMES), len(slots), 2), np.float32)
    object_disp[:, 0] = fall
    object_ok = np.ones((len(REWARD_TIMES), len(slots)), bool)
    scene = rw.build_scene(points, owner, slots, masks, tracks, visible, REWARD_TIMES, REWARD_TIMES[-1], max_times=None,
                           object_disp=object_disp, object_ok=object_ok, object_times=REWARD_TIMES)
    assert scene["mode"] == "mask"
    with pytest.raises(ValueError):
        rw.score(scene, tracks, visible, REWARD_TIMES)
    result = rw.score(scene, tracks, visible, REWARD_TIMES, object_disp=object_disp, object_ok=object_ok, object_times=REWARD_TIMES)
    assert result["reward"] == pytest.approx(1.0)
    assert np.array_equal(rw.scene_masks(scene), masks)
    path = tmp_path / "scene.npz"
    rw.save_scene(path, scene)
    loaded = rw.load_scene(path)
    assert loaded["mode"] == "mask" and loaded["version"] == rw.VERSION
    assert np.array_equal(loaded["real"], scene["real"])
    short = rw.truncate_scene(loaded, 6.0 / FPS)
    assert len(short["times"]) == 6 and short["real"].shape[1] == 6
    assert len(rw.select_times(scene["times"], FPS, None, 12)) == 12


# ------------------------------------------------------------------------------------------- measured real plan
def test_measured_plan_follows_tracks():
    masks, points, owner, slots = _reward_layout()
    fall = _fall()
    tracks, visible = _reward_tracks(points, owner, fall)
    plan_pred = mp.from_instance_affines("synthetic", masks, np.repeat(mp.IDENTITY[None, None], 2, 0).repeat(7, 1),
                                         [0.2083, 0.4167, 0.625, 0.8333, 1.2, 1.5, 1.7])
    anchors = REWARD_TIMES[1:]
    plan, owner2, info = rs.measured_plan("synthetic", plan_pred, tracks, visible, owner, points, REWARD_TIMES, anchors)
    assert [o.slot for o in plan.objects] == [0, 1] and info["stage1_objects"] == 2
    assert np.allclose(plan.object(0).displacement(), fall[1:], atol=1e-6)
    assert np.allclose(plan.object(1).displacement(), 0.0, atol=1e-6)
    assert not rs.at_rest(tracks, visible, owner, REWARD_TIMES)


# ---------------------------------------------------------------------------------------- arguments and loaders
def test_tracker_loaders_need_checkpoints(monkeypatch):
    from scripts.control import cotrack, masktrack
    monkeypatch.delenv("COTRACKER_CKPT", raising=False)
    monkeypatch.delenv("SAM2_CHECKPOINT", raising=False)
    with pytest.raises(RuntimeError, match="COTRACKER_CKPT"):
        cotrack.load_tracker("", "", torch.device("cpu"))
    with pytest.raises(RuntimeError, match="SAM2_CHECKPOINT"):
        masktrack.load_mask_tracker("", device="cpu")


def test_admit_checkpoint_requires_files(tmp_path):
    from scripts.clean_r2 import export_p101_stage1 as export
    with pytest.raises(FileNotFoundError):
        export.admit_checkpoint(tmp_path)
    assert len(export.QUERY_SECONDS.split(",")) == 7


def test_real_anchors():
    from scripts import cache_stage2_cond_v2 as cache
    times = pc.latent_times(13, FPS)
    anchors = cache.real_anchors([0.2083, 0.4167, 0.625, 2.0], times)
    assert np.all(np.diff(anchors) > 0) and anchors[-1] == pytest.approx(times[-1])
    assert 2.0 not in anchors.tolist()


def test_negative_prompt_literal():
    """The official Wan2.2 negative prompt must stay byte-identical: it changes the generated videos."""
    import hashlib
    from scripts import infer_stage2_oracle as oracle
    prompt = oracle.WAN_BENCHMARK_NEGATIVE_PROMPT
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == "ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e"
