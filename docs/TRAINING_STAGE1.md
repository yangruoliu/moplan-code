# Stage-1 training (motion planner)

Stage 1 of MoPlan predicts a motion plan from the first frame, the caption and first-frame SAM 2 instance masks. The
plan is a cumulative 2D displacement field on a 160 x 160 grid at 7 horizons up to 4 s, with object slots, a dense
motion mask and per-object activity. This page explains how to train the released Stage-1 model and how to run a
smoke test. How the training data are built is in [DATA_STAGE1.md](DATA_STAGE1.md).

## Model

- Backbone: the high-noise expert of Wan2.2-I2V-A14B (`transformer/` of `Wan2.2-I2V-A14B-Diffusers`). It is frozen in
  bf16 with gradient checkpointing. LoRA rank 32 and alpha 32 is applied to `attn1` and `attn2`
  (`to_q`, `to_k`, `to_v`, `to_out.0`) and to `ffn.net.0.proj` and `ffn.net.2` (about 153M trainable parameters).
- New input patch embedding (Conv3d, 39 -> 5120 channels): 16 motion-latent channels, 4 mask channels, 16 image-latent
  channels, 1 time channel and 2 instance channels. The first 36 input channels are copied from the pretrained layer.
  The new output projection maps 5120 -> 64.
- Slot decoder (about 4.95M parameters): 16 temporal, causal slots pooled from the SAM 2 masks (`area` pooling),
  text-slot binding (4 heads), a residual branch (scale 0.05), a dense motion-mask head, a task-object activity head
  and a zero-initialised flow head. The visibility heads are frozen.
- Motion codec: the Wan VAE with a radial power encoding of the flow (gamma 0.5, max 64 px, visibility channel).
- Objective `direct`: one deterministic forward pass. The input motion latent is a constant zero-motion prior latent,
  not noise. There is no latent flow-matching loss (`loss.flow_matching: 0`). Validation uses the same single step.

## Losses

`loss.profile: legacy` with these weights (all others are 0):

| term | weight | notes |
|---|---|---|
| flow_charbonnier | 1.0 | eps 0.5 px. Partition weights: moving 4, static 2, weak 1, background 1. Moving > 1 px, static <= 0.5 px |
| gated_flow_charbonnier | 0.25 | |
| dense_motion_mask | 5.0 | dice weight 0.5, pixel BCE fraction 0.75, target `whole_object_motion_v3_task_object` |
| task_object_slot_activity | 5.0 | prior BCE fraction 0.75 |
| object_displacement, moving_direction, object_motion_direction | 1.0 each | |
| moving_parallel_magnitude, object_motion_parallel_magnitude | 1.0 each | beta 1 px, scale 16 px |
| trajectory_segment | 0.5 | velocity scale 32 px/s |
| slot_mask | 0.5 | |
| object_mask_bce | 0.1 | |

The supervision mode is `valid_partition_v1`: every target pixel is labelled moving, static or unknown, and unknown
pixels carry no loss.

The config also sets `loss.background_motion_weight: 0.02`. Under `valid_partition_v1` this key has no effect: it
only enters the flow weighting and the gate weighting of the other supervision modes. The background pixels are
weighted by `loss.flow_partition_weights.background`, which is 1 (the "background 1" above).

## Optimisation

- AdamW, betas (0.9, 0.999), weight decay 0.01. The learning rate is 2e-5 for LoRA and 5e-5 for the new layers and
  the decoder. Warmup is linear over 200 steps, then a cosine decay to 0.1x at step 6000.
- Gradient clipping at 1.0. The mask and task-object heads are clipped separately.
- Global batch 32: 1 sample per GPU x 4 accumulation steps x 8 GPUs. bf16 mixed precision. Seed 42.
- 6000 optimizer steps. The released model is the final checkpoint (`checkpoint-006000`), not `checkpoint-best`.
  `checkpoint-best` only tracks the in-training selection metric.
- Data: 7 sources mixed by `source_cycle_v1`. This sampler keeps fixed per-source ratios and visits each source
  without replacement. The ratios are FOT 0.45, event_static 0.12, phyco_anchor 0.14, pool_v2 0.07, real_p101 0.08,
  real_p101r 0.08 and real_ssv2 0.06.
- Validation every 200 steps on the first 96 rows of each source's validation split. A checkpoint is written every
  200 steps and the last 40 are kept (about 1.9 GB each with optimizer state).

All values are in [`configs/stage1/train.yaml`](../configs/stage1/train.yaml). They are identical to the reference
run. Only file locations are written as placeholders.

## First frame of the decoder: training vs inference

The image branch of the slot decoder reads the first frame at 160 x 160 (`first_frame`). Training and inference
prepare it slightly differently:

- The training caches letterbox the frame to 256 x 256 and then resize that image to 160 x 160 with `INTER_AREA`,
  i.e. two resamplings. Every Stage-1 cache builder does this (`read_first_frame` of `scripts/cache_stage1_flowdit.py`,
  reused by `scripts/cache_stage1_event_inputs.py` and `scripts/stage1_real/`).
- Inference from an image (`scripts/infer_stage1_physicsiq.py`, which `generate.py` and the evaluation launchers
  use) letterboxes the frame directly to 160 x 160 with `INTER_AREA`, i.e. one resampling.

The letterbox placement is the same. The pixel values differ little: around one grey level on average, up to a few
tens of grey levels at sharp edges. The VAE image latent (`image_cond_latent`) is built from the 256 x 256 letterboxed
frame in both paths. The released videos and the evaluation were produced with the inference path as it is, so it is
left unchanged. Exports that read the training caches (`scripts/eval_stage1_flowdit.py`, used for the PhyCo
conditions of the RAFT round) use the training preprocessing.

## Requirements

- One node with 8 H100 80 GB GPUs for the reference setting. Peak memory is about 32 GiB per GPU.
- Compute of the reference run (8 GPUs). It ran as 8 consecutive jobs chained by exact resume. The run times of those
  jobs add up to about 7 hours, i.e. about 55 GPU-hours, including model loading, validation and checkpointing. The
  optimizer steps alone take about 3.6 s each, about 48 GPU-hours for 6000 steps. From the start of the first job to
  the end of the last one took about 8.3 hours; the difference is the time between the jobs.
- The pinned Python environment of this repository.
- `WAN_MODEL_DIR`: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`. Training reads `transformer/` and `vae/config.json`.
- `MOPLAN_DATA`: the built Stage-1 data, laid out as in [DATA_STAGE1.md](DATA_STAGE1.md) (about 150 GB).

## Config placeholders

`configs/stage1/train.yaml` writes every location with three placeholders. The trainer fills them before anything
else reads the config:

| placeholder | command-line option | environment variable |
|---|---|---|
| `${MOPLAN_DATA}` | `--data-root` | `MOPLAN_DATA` |
| `${WAN_MODEL_DIR}` | `--wan-model-dir` | `WAN_MODEL_DIR` |
| `${MOPLAN_WORK_DIR}` | `--work-dir` | `MOPLAN_WORK_DIR` |

A command-line option wins over the environment. An unset placeholder is an error. Any existing key can still be
overridden with `--set section.key=value` (YAML-parsed). The data tools (`scripts/cache_stage1_*.py`,
`scripts/stage1_real/*`) fill the same placeholders from the environment.

## Train

```bash
export MOPLAN_DATA=/path/to/data          # contains stage1/<source>/...
export WAN_MODEL_DIR=/path/to/Wan2.2-I2V-A14B-Diffusers
export MOPLAN_WORK_DIR=/path/to/work      # default: <repo>/work
bash launch/train_stage1.sh
```

The run is written to `$MOPLAN_WORK_DIR/runs/stage1/high/`:

```
metrics.jsonl  history.jsonl  done.json  resolved_config.yaml
checkpoint-000000/ ... checkpoint-006000/   backend_params.pt decoder.pt optimizer.pt scheduler.pt rng_state.pt
                                            trainer_state.json resolved_config.yaml checkpoint_complete.json
checkpoint-best/
```

`backend_params.pt` (the LoRA and new layers) and `decoder.pt` of `checkpoint-006000` are the files that the released
`$MOPLAN_WEIGHTS/stage1/` contains. The inference documentation lists the small config files the loader also reads.

Notes:

- `NPROC=4 bash launch/train_stage1.sh` trains the same recipe on 4 GPUs with 8 accumulation steps (global batch
  still 32). The config's rank contract allows only 4 or 8 ranks, and the launcher sets
  `unified_experiment.world_size=4` and `train.grad_accum_steps=8` for you. Rows are assigned to ranks differently,
  so this run is not step-for-step identical to the 8-GPU run.
- Even with 8 GPUs, a rerun is not guaranteed to be bitwise identical to the reference run. The data must match the
  released ID lists row for row and in the same order. GPU kernels and collective reductions can also differ.
- The trainer refuses to start a fresh run in an output directory that already holds `history.jsonl`,
  `metrics.jsonl` or `done.json`.
- Rows whose cache files are missing are dropped silently. The trainer stops only when a scenario keeps less than
  50% of its rows. Check that the first log lines report `train=71055`.

### Interrupt and resume

Exact resume is on (`train.exact_resume: true`). Send `SIGUSR1` (or `SIGTERM`) to the worker processes, not to the
torchrun agent, for example `pkill -USR1 -f -- '-u scripts/train_stage1_flowdit.py'` (torchrun starts the workers as
`python -u scripts/...`; the plain pattern `scripts/train_stage1_flowdit.py` also matches the agent, which SIGUSR1
kills at once while the workers keep running). They then finish the current optimizer step, write an exact-resume
checkpoint (model, optimizer, scheduler, RNG states and data cursors) and exit. Resume with:

```bash
bash launch/train_stage1.sh --resume-from "$MOPLAN_WORK_DIR/runs/stage1/high/checkpoint-002400"
```

Use the same `NPROC` and the same `MOPLAN_DATA`, `WAN_MODEL_DIR` and `MOPLAN_WORK_DIR` values. The resolved config,
including its paths, is part of the resume contract. `--stop-at-step N` ends an invocation at step N while the LR
schedule keeps its 6000-step horizon. This is useful for jobs with a wall-time limit.

## Smoke test (1 GPU)

`scripts/stage1_smoke.py` runs the unchanged trainer for a few optimizer steps on one GPU, using the first rows of
every source. It needs one GPU with at least 40 GB, `WAN_MODEL_DIR` and a data root that holds at least those rows:

```bash
python scripts/stage1_smoke.py --data-root "$MOPLAN_DATA" --wan-model-dir "$WAN_MODEL_DIR" \
    --out work/stage1_smoke --steps 2 --limit 16 --val-limit 4 --resume-check
```

It goes through `scripts/s1chain/smoke_entry.py`, which only disables the 4/8-rank check. It overrides batch
bookkeeping (1 GPU x `--grad-accum` 8 micro-batches), validation size, loader workers and the scene-coverage minimum,
which has no meaning on a subset. Every model and loss setting is that of the config. The script checks:

- finite `loss/total` at every step;
- a committed `checkpoint-<steps>`;
- `done.json`;
- with `--resume-check`, an exact resume for another `--steps` steps.

It then prints `SMOKE_OK`. The log also shows `[sanity] untrained EPE differs from zero-flow by 0.000%` (the flow
head is zero-initialised). On an H100 this takes about 5 minutes, most of it loading the 14B transformer. Pass
`--allow-scenario-holes` when only part of the data is built, and `--dry-run` to check the inputs and print the
trainer commands without a GPU.

## CPU tests

```bash
PYTHONPATH=. python -m pytest -q tests/test_s1train.py
```

These tests check that the config loads and resolves, that it holds the recipe values, the 4-GPU override, the
placeholder rules and the ID lists (71,055 training rows, disjoint splits). They also cover the source assembler,
the prior-file loader and the smoke dry run.
