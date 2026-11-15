"""CPU unit tests for the core library (``motion_jepa``).

Run from the repository root:

    python -m pytest tests/test_core_lib.py -q

No GPU, model weights or datasets are needed.
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

MODULES = sorted(path.stem for path in (REPO_ROOT / "motion_jepa").glob("*.py") if path.stem != "__init__")
# Modules whose only extra dependency is optional on CPU.
OPTIONAL_DEPENDENCIES = {"physion_hdf5": "h5py"}


# --------------------------------------------------------------------------- imports


def test_package_imports():
    package = importlib.import_module("motion_jepa")
    assert {"MotionJEPA", "SCENARIOS", "load_config"} <= set(package.__all__)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    if name in OPTIONAL_DEPENDENCIES:
        pytest.importorskip(OPTIONAL_DEPENDENCIES[name])
    importlib.import_module(f"motion_jepa.{name}")


# --------------------------------------------------------------------------- cond_v2 contract


def test_cond_v2_channel_contract():
    from motion_jepa import cond_v2

    assert cond_v2.BASE_CHANNELS == 20
    assert cond_v2.EXTRA_CHANNELS == 7
    assert cond_v2.V2_CHANNELS == 27
    assert cond_v2.EXTRA_NAMES == (
        "known", "availability", "selector", "raw_flow_x", "raw_flow_y", "motion_prob", "source_mask",
    )
    assert cond_v2.CHANNEL == {
        "known": 20,
        "availability": 21,
        "selector": 22,
        "raw_flow_x": 23,
        "raw_flow_y": 24,
        "motion_prob": 25,
        "source_mask": 26,
    }
    assert cond_v2.KINDS == ("real", "pred", "zero", "null")
    assert cond_v2.CACHE_FORMAT == "stage2_cond_v2"


def test_null_condition_shape_and_content():
    from motion_jepa.cond_v2 import V2_CHANNELS, null_condition

    generator = torch.Generator().manual_seed(0)
    image_latent = torch.randn(16, 5, 4, 6, generator=generator)
    condition = null_condition(image_latent)
    assert condition.shape == (V2_CHANNELS, 5, 4, 6)
    assert condition.dtype == image_latent.dtype
    # Channels 0-15 repeat the first latent frame at every time; everything else is zero.
    for frame in range(5):
        assert torch.equal(condition[:16, frame], image_latent[:, 0])
    assert torch.count_nonzero(condition[16:]) == 0


def test_extra_channel_index_and_unknown_names():
    from motion_jepa.cond_v2 import extra_channel_index

    assert extra_channel_index(["known", "source_mask"]) == [0, 6]
    assert extra_channel_index([]) == []
    with pytest.raises(ValueError):
        extra_channel_index(["not_a_channel"])


@pytest.mark.parametrize("batched", [False, True])
def test_mask_extras_zeroes_only_dropped_extras(batched):
    from motion_jepa.cond_v2 import BASE_CHANNELS, CHANNEL, V2_CHANNELS, mask_extras

    generator = torch.Generator().manual_seed(1)
    shape = (2, V2_CHANNELS, 3, 4, 4) if batched else (V2_CHANNELS, 3, 4, 4)
    condition = torch.rand(shape, generator=generator) + 0.5
    original = condition.clone()
    out = mask_extras(condition, keep=["known", "availability"])
    channel_axis = 1 if batched else 0
    assert torch.equal(condition, original), "input must not be modified"
    assert torch.equal(out.narrow(channel_axis, 0, BASE_CHANNELS), original.narrow(channel_axis, 0, BASE_CHANNELS))
    for name, index in CHANNEL.items():
        kept = out.select(channel_axis, index)
        if name in ("known", "availability"):
            assert torch.equal(kept, original.select(channel_axis, index))
        else:
            assert torch.count_nonzero(kept) == 0


# --------------------------------------------------------------------------- stage2_condition


def test_interface_from_config():
    from motion_jepa.stage2_condition import interface_from_config

    assert interface_from_config({}) == "legacy20"
    assert interface_from_config({"motion_condition_channels": 20, "motion_condition_extra_channels": 7}) == "legacy27"
    assert interface_from_config({"motion_condition_channels": 5}) == "native5"
    with pytest.raises(ValueError):
        interface_from_config({"motion_condition_channels": 20, "condition_interface": "legacy27"})


def test_read_cached_condition():
    from motion_jepa.stage2_condition import read_cached_condition

    payload = {"cond_pred": torch.rand(27, 3, 4, 4)}
    full = read_cached_condition(payload, "pred", "legacy27")
    assert full.shape == (27, 3, 4, 4)
    base = read_cached_condition(payload, "pred", "legacy20")
    assert torch.equal(base, payload["cond_pred"][:20])
    with pytest.raises(KeyError):
        read_cached_condition(payload, "zero", "legacy27")
    broken = {"cond_pred": torch.full((27, 3, 4, 4), float("nan"))}
    with pytest.raises(ValueError):
        read_cached_condition(broken, "pred", "legacy27")


# --------------------------------------------------------------------------- Stage-2 adapter

ADAPTER_KWARGS = dict(hidden_channels=8, transformer_dim=16, injection_blocks=[0, 1], depth=1, levels=3, heads=2,
                      mixer_rank=4)


def _tiny_extended_adapter(seed: int = 0):
    from motion_jepa.stage2 import ExtendedOnlyFlowAdapter

    torch.manual_seed(seed)
    return ExtendedOnlyFlowAdapter(base_channels=20, extra_channels=7, **ADAPTER_KWARGS).eval()


def test_extended_adapter_forward_shapes_and_zero_init():
    adapter = _tiny_extended_adapter()
    frames, height, width = 3, 8, 8
    generator = torch.Generator().manual_seed(2)
    condition = torch.randn(1, 27, frames, height, width, generator=generator)
    with torch.no_grad():
        residuals = adapter(condition)
        base_only = adapter(condition[:, :20])
    tokens = frames * (height // 2) * (width // 2)
    assert len(residuals) == len(ADAPTER_KWARGS["injection_blocks"])
    for value in residuals:
        assert value.shape == (1, tokens, ADAPTER_KWARGS["transformer_dim"])
    # The extra stem is zero-initialised: the extra channels cannot change the output at step 0.
    for full, base in zip(residuals, base_only):
        assert torch.equal(full, base)
    # The up projection is zero-initialised: the injected residual is exactly zero at step 0.
    hidden = torch.randn(1, tokens, ADAPTER_KWARGS["transformer_dim"], generator=generator)
    with torch.no_grad():
        mixed = adapter.mix(0, residuals[0], hidden)
    assert torch.count_nonzero(mixed) == 0


def test_extended_adapter_rejects_wrong_channel_count():
    adapter = _tiny_extended_adapter()
    with pytest.raises(ValueError):
        adapter(torch.zeros(1, 21, 3, 8, 8))


def test_extended_adapter_loads_base_checkpoint():
    from motion_jepa.stage2 import OnlyFlowMotionAdapter
    from motion_jepa.stage2_condition import load_adapter_state

    torch.manual_seed(3)
    deployed = OnlyFlowMotionAdapter(20, **ADAPTER_KWARGS)
    adapter = _tiny_extended_adapter(seed=4)
    report = load_adapter_state(adapter, deployed.state_dict())
    assert report["mode"] == "strict"
    for name, tensor in deployed.state_dict().items():
        assert torch.equal(adapter.state_dict()[name], tensor)
    assert torch.count_nonzero(adapter.stem_extra.weight) == 0
    with pytest.raises(RuntimeError):
        adapter.load_deployed({"unexpected.weight": torch.zeros(1)})


def test_motion_agreement_loss_zero_for_static_scene():
    from motion_jepa.stage2 import motion_agreement_loss

    latent = torch.rand(1, 16, 1, 4, 4).expand(-1, -1, 3, -1, -1).contiguous()
    condition = torch.zeros(1, 20, 3, 4, 4)
    condition[:, 18] = 1.0
    assert float(motion_agreement_loss(latent, condition)) == pytest.approx(0.0, abs=1e-6)


def test_sample_expert_sigmas_ranges():
    from motion_jepa.stage2 import sample_expert_sigmas

    kwargs = dict(boundary_sigma=0.9, shift=5.0, device=torch.device("cpu"))
    high = sample_expert_sigmas(256, expert="high", generator=torch.Generator().manual_seed(0), **kwargs)
    low = sample_expert_sigmas(256, expert="low", generator=torch.Generator().manual_seed(0), **kwargs)
    assert bool((high >= 0.9 - 1e-6).all()) and bool((high <= 0.999).all())
    assert bool((low <= 0.9 + 1e-6).all()) and bool((low >= 0.001).all())
    again = sample_expert_sigmas(256, expert="high", generator=torch.Generator().manual_seed(0), **kwargs)
    assert torch.equal(high, again)


# --------------------------------------------------------------------------- pure functions


def test_dpo_terms_at_reference():
    from motion_jepa.dpo import dpo_terms

    errors = torch.tensor([0.3, 1.2])
    terms = dpo_terms(errors, errors, errors, errors, beta=2.0)
    assert torch.allclose(terms["d"], torch.zeros(2))
    assert torch.allclose(terms["loss"], torch.full((2,), math.log(2.0)))
    assert torch.allclose(terms["weight"], torch.full((2,), 1.0))


def test_kinematic_losses():
    from motion_jepa.kinematics import kinematic_losses

    seconds = torch.tensor([[0.0, 0.5, 1.0, 1.5]])
    truth = torch.zeros(1, 2, 4, 2)
    truth[0, 0, :, 0] = torch.tensor([0.0, 10.0, 20.0, 30.0])       # object 0 moves right, object 1 is static
    valid = torch.ones(1, 2, 4, dtype=torch.bool)
    exact = kinematic_losses(truth, truth, seconds, valid)
    assert float(exact["direction"]) == pytest.approx(0.0, abs=1e-6)
    assert float(exact["displacement"]) == pytest.approx(0.0, abs=1e-6)
    assert float(exact["moving_objects"]) == 1.0 and float(exact["objects"]) == 2.0
    still = kinematic_losses(torch.zeros_like(truth), truth, seconds, valid)
    assert float(still["direction"]) == pytest.approx(1.0, abs=1e-6)
    assert float(still["displacement"]) > 0.0
    with pytest.raises(ValueError):
        kinematic_losses(truth, truth, torch.tensor([[0.0, 0.5, 0.5, 1.0]]), valid)


@pytest.mark.parametrize("kind", ["radial_power", "signed_power", "radial_log", "linear"])
def test_flow_codec_round_trip(kind):
    from motion_jepa.flow_codec import compress, decompress

    generator = torch.Generator().manual_seed(5)
    flow = (torch.rand(3, 2, 6, 6, generator=generator) - 0.5) * 0.2
    code = compress(flow, max_normalized=0.25, gamma=0.5, kind=kind)
    assert float(code.abs().max()) <= 1.0 + 1e-6
    back = decompress(code, max_normalized=0.25, gamma=0.5, kind=kind)
    assert torch.allclose(back, flow, atol=1e-5)
    with pytest.raises(ValueError):
        compress(flow, max_normalized=0.25, gamma=0.5, kind="hsv")


def test_flow_image_round_trip_and_zero_slot():
    from motion_jepa.flow_codec import flow_to_image, image_to_flow, prepend_zero_slot

    generator = torch.Generator().manual_seed(6)
    flow = (torch.rand(2, 2, 16, 16, generator=generator) - 0.5) * 8.0
    visibility = torch.ones(2, 1, 16, 16)
    image = flow_to_image(flow, visibility, size=16, max_pixels=40.0, gamma=0.5, eval_side=16.0)
    assert image.shape == (2, 3, 16, 16)
    back, vis = image_to_flow(image, out_size=16, max_pixels=40.0, gamma=0.5, eval_side=16.0)
    assert torch.allclose(back, flow, atol=1e-3)
    assert torch.allclose(vis, visibility)
    padded_flow, padded_vis = prepend_zero_slot(flow, visibility)
    assert padded_flow.shape[0] == 3 and torch.count_nonzero(padded_flow[0]) == 0
    assert torch.equal(padded_vis[0], torch.ones_like(padded_vis[0]))


def test_affine_helpers():
    from motion_jepa.flow_dit_slots import affine_flow, compose_affine_sequence, render_blended_affine

    zero = torch.zeros(1, 2, 3, 6)
    assert torch.count_nonzero(affine_flow(zero, 5, 7)) == 0
    assert torch.equal(compose_affine_sequence(zero), zero)
    shift = torch.zeros(1, 1, 3, 6)
    shift[..., 4] = 0.2                                             # tx = 0.2 per step
    cumulative = compose_affine_sequence(shift)
    assert torch.allclose(cumulative[0, 0, :, 4], torch.tensor([0.2, 0.4, 0.6]))
    flow = affine_flow(cumulative, 4, 4)
    assert torch.allclose(flow[0, 0, :, 0].mean((-1, -2)), torch.tensor([0.1, 0.2, 0.3]))
    masks = torch.ones(1, 1, 4, 4)
    assert torch.allclose(render_blended_affine(masks, cumulative, 4, 4), flow[:, 0])


def test_apply_shrinkage():
    from motion_jepa.flow_dit import apply_shrinkage, expert_subfolder

    flow = torch.zeros(1, 2, 1, 2)
    flow[0, 0, 0, 0] = 0.05
    flow[0, 0, 0, 1] = 0.5
    assert torch.equal(apply_shrinkage(flow, 0.0), flow)
    shrunk = apply_shrinkage(flow, 0.1)
    assert float(shrunk[0, 0, 0, 0]) == 0.0
    assert float(shrunk[0, 0, 0, 1]) == pytest.approx(0.4)
    assert expert_subfolder("high") == "transformer"
    assert expert_subfolder("low") == "transformer_2"
    with pytest.raises(ValueError):
        expert_subfolder("medium")


def test_normalized_horizon_times():
    from motion_jepa.flow_dit_time import normalized_horizon_times

    cpu = torch.device("cpu")
    scaled = normalized_horizon_times([0.5, 1.0, 2.0], batch_size=2, horizons=3, time_scale=4.0, device=cpu)
    assert scaled.shape == (2, 3)
    assert torch.allclose(scaled[0], torch.tensor([0.125, 0.25, 0.5]))
    relative = normalized_horizon_times([0.5, 1.0, 2.0], batch_size=1, horizons=3, time_scale=None, device=cpu)
    assert torch.allclose(relative[0], torch.tensor([0.25, 0.5, 1.0]))
    with pytest.raises(ValueError):
        normalized_horizon_times([1.0, 0.5, 2.0], batch_size=1, horizons=3, time_scale=4.0, device=cpu)


def test_apply_overrides_unknown_key():
    from motion_jepa.config import apply_overrides

    with pytest.raises(ValueError):
        apply_overrides({"model": {}}, ["model.not_a_key=1"])
    with pytest.raises(ValueError):
        apply_overrides({"model": {}}, ["no_equals_sign"])
