# motion_jepa: core library

`motion_jepa` holds the model code of both MoPlan stages. The package name is
historical. The scripts under `scripts/` import it from the repository root,
so the root has to be on `PYTHONPATH`. The package is not installed.

- **Stage 1 (motion planner).** Predicts first-to-future object motion from the
  first frame and the caption. It is called WanFlowDiT in the code.
- **Stage 2 (generator).** Wan2.2-I2V-A14B with LoRA and a motion adapter. It
  reads the Stage-1 plan as a 27-channel condition.

Importing the package runs `__init__.py`. That file imports the legacy
`MotionJEPA` model and `config.load_config`, so `torch` and `pyyaml` are
needed even for Stage-2-only use.

## Shared utilities

| module | contents |
|---|---|
| `config.py` | `SCENARIOS`, the YAML loader and `section.key=value` overrides for the legacy Stage-1 config |
| `data.py` | `ManifestRow`, `read_jsonl` and the legacy feature/target cache dataset |
| `video.py` | Video reading and sampling, Physion agent masks, forward splatting of masks along flow |

## Stage 1: motion planner (WanFlowDiT)

The released Stage-1 checkpoint uses the high-noise Wan expert, the slot
decoder and `oracle_instance_masks`. Its inputs are SAM 2 first-frame
proposals: the masks are inputs, not ground truth. The decoder configuration
is read from the checkpoint's `trainer_state.json`.

Model and inference:

| module | contents |
|---|---|
| `flow_dit.py` | `WanFlowDiTBackend` (one Wan2.2 expert with LoRA that denoises flow latents in Wan's i2v channel layout), the dense `LatentFlowDecoder`, `build_decoder`, `apply_shrinkage` |
| `flow_dit_slots.py` | `SlotTransformDecoder`: per-object affine slots, text-to-slot binding, task-object identity and activity heads, the dense motion mask; also affine fitting and rendering helpers |
| `flow_codec.py` | Invertible flow-to-image codecs and the Wan VAE / direct latent codecs |
| `flow_dit_config.py` | Config loading, validation and derived quantities (latent grid, horizons) |
| `flow_dit_sampler.py` | Sigma grid, latent sampling, decoding and K-sample reduction |
| `flow_dit_time.py` | Physical-time normalisation of the query horizons |
| `flow_dit_selection.py` | Object-selection readouts from proposal probabilities |
| `flow_dit_export.py` | The canonical export fields that Stage 2 reads (`bridge_fields`, checkpoint provenance) |
| `flow_dit_data.py` | `FlowDiTDataset`, joint and mixture datasets, collate, `pad_prompt`, `build_instance_inputs` (SAM masks to slot inputs) |

Training objectives:

| module | contents |
|---|---|
| `flow_dit_losses.py` | Decoded-flow losses: Charbonnier, gate, visibility, object-level and affine terms, `valid_partition_v1` supervision |
| `flow_dit_minimal.py` | The opt-in five-term objective (cone direction, displacement, selection BCE) |
| `flow_dit_unified.py` | The balanced material-point objective and its run contract, plus distributed gradient statistics |
| `flow_dit_clean.py`, `flow_dit_coarse.py` | The material-point trajectory objective and the coarse affine/direction objective |
| `flow_dit_dense_mask.py` | Dense task-object motion-mask labels, losses and statistics |
| `flow_dit_activity.py` | Per-object displacement-activity labels and balanced BCE |
| `flow_dit_events.py` | Paired-event datasets and the field-difference loss |

Training infrastructure:

| module | contents |
|---|---|
| `flow_dit_checkpoint.py` | Weight-only initialisation from an earlier checkpoint |
| `flow_dit_resume.py` | Exact resume: RNG, the consumed-batch cursor, completion manifests |
| `flow_dit_numerics.py` | Opt-in PyTorch determinism settings |
| `flow_dit_optimization.py` | Separate gradient-clip budgets for the auxiliary heads |
| `flow_dit_coverage.py` | Deterministic per-source sampling cycles and consumed-batch coverage |

Data contracts:

| module | contents |
|---|---|
| `flow_dit_bridge.py`, `bridge_mover_selection.py` | Bridge real-video labels and the task-object selection prompts and validation |
| `caption_overrides.py` | Reviewed train-only caption substitutions |
| `physion_hdf5.py` | Simulator-native Physion flow from TDW HDF5 files. Needs `h5py` and is only used for data building |

Diagnostics and evaluation:

| module | contents |
|---|---|
| `metrics.py` | Per-sample flow metrics and `MetricAccumulator` |
| `flow_dit_diagnostics.py`, `flow_dit_event_diagnostics.py`, `flow_dit_collapse.py`, `flow_dit_gate_calibration.py` | Motion statistics, release/hold diagnostics, collapse guards and identity-threshold calibration |

Legacy modules:

| module | contents |
|---|---|
| `model.py` | The legacy `MotionJEPA` model, imported by `__init__.py` |
| `losses.py` | Legacy losses; `trajectory_centers` is still used by the sampler |
| `vjepa.py` | Legacy V-JEPA 2 runtime RoPE grid |

## Stage-2 condition contract (`cond_v2.py`)

A condition is a tensor `[27, T, h, w]` on the Wan latent grid, with one slice
per latent frame:

| channels | name | meaning |
|---|---|---|
| 0-15 | warped latent | first-frame VAE latent, forward-splatted along the planned object flow |
| 16-17 | flow | object flow at target pixels, divided by W and H |
| 18 | target mask | object mask M_t at time t |
| 19 | visibility | visibility weight |
| 20 | `known` | 1 where the plan states the object pose at this time |
| 21 | `availability` | 1 when a plan is supplied, 0 for the null condition |
| 22 | `selector` | Stage-1 task-object score on the target footprint (metadata only) |
| 23-24 | `raw_flow_x`, `raw_flow_y` | Stage-1 dense displacement at source pixels, divided by W and H |
| 25 | `motion_prob` | Stage-1 dense motion probability |
| 26 | `source_mask` | M0, the union of the first-frame object masks |

- `BASE_CHANNELS = 20`, `EXTRA_NAMES`, `CHANNEL` (name to index) and
  `CACHE_FORMAT = "stage2_cond_v2"` define the layout.
- `KINDS` lists the four conditions a cache item can hold:
  - `real`: the plan measured on the real video. Training only.
  - `pred`: the Stage-1 plan. This is what inference uses.
  - `zero`: the same objects held at rest.
  - `null`: no plan (`null_condition`).
- `mask_extras` zeroes chosen extra channels for ablations.

`stage2_condition.py` picks the input interface from the model config:

- `legacy20`: the first 20 channels.
- `legacy27`: 20 + 7 channels, used by the released generator.
- `native5`: an explicit source-coordinate layout.

The file also has `read_cached_condition`, which reads and validates a cached
condition, and `load_adapter_state`.

## Stage 2: generator (`stage2.py`)

| name | role |
|---|---|
| `DEFAULT_LORA_TARGETS` | Attention and FFN projections that get LoRA in both Wan experts |
| `ExtendedOnlyFlowAdapter` | The released motion adapter, an `OnlyFlowMotionAdapter` with a second zero-initialised stem for the 7 extra channels. Its output is one token residual per injected transformer block, mixed in through a low-rank `mix` |
| `OnlyFlowMotionAdapter`, `MotionAdapter` | The 20-channel adapters that the extended adapter builds on |
| `OracleFlowWanBackend` | One Wan expert (`high` = `transformer`, `low` = `transformer_2`) with LoRA and block-level residual injection, plus flow-matching training steps and the frozen reference LoRA |
| `sample_expert_sigmas`, `load_latent_statistics` | Sigma sampling per expert, and the Wan VAE latent mean and std |

Preference and RAFT training:

| module | contents |
|---|---|
| `dpo.py` | Diffusion-DPO terms on flow-matching errors |
| `kinematics.py`, `phi_lat.py` | Kinematic losses and the latent motion readout. Only used when the kinematic weight is above zero |

## Entry points

| script | modules used |
|---|---|
| `scripts/infer_stage1_physicsiq.py` | `flow_codec`, `flow_dit`, `flow_dit_config`, `flow_dit_data`, `flow_dit_export`, `flow_dit_sampler`, `video` |
| `scripts/train_stage1_flowdit.py` | most `flow_dit_*` modules, `caption_overrides`, `metrics` |
| `scripts/cache_stage2_cond_v2.py` | `cond_v2`, `stage2`, `data`, `video` |
| `scripts/infer_stage2_cond_v2.py` | `cond_v2`, `stage2`, `stage2_condition` |
| `scripts/s2sft/v2/train_s2sft_v2.py`, `scripts/train_s2_v4.py` | `cond_v2`, `stage2`, `stage2_condition`, `dpo`, and optionally `phi_lat` |

## Tests

```
python -m pytest tests/test_core_lib.py -q
```

The tests run on CPU in a few seconds and need no weights. They cover:

- importing every module (`physion_hdf5` is skipped without `h5py`);
- the 27-channel contract and `null_condition`;
- adapter shapes and zero-initialisation;
- several pure helper functions.
