# Stage-2 GT-SFT training

The GT-SFT stage fine-tunes Wan2.2-I2V-A14B to follow a motion plan. Each expert (`high` = `transformer/`, the
high-noise expert; `low` = `transformer_2/`) is trained once, from the base model, with a fresh LoRA and a fresh
27-channel motion adapter, on real continuations paired with their ground-truth plans (`docs/DATA_STAGE2.md`). The
released generator is the result of one light RAFT round (`docs/TRAINING_RAFT.md`) started from step 4000 of this run.

**The GT-SFT checkpoint is not distributed.** The released weights hold only the Stage-1 planner and the final
generator (after RAFT). The RAFT round and the SFT-only arm of MoProbe (`docs/MOPROBE.md`) both start from
`checkpoint-004000` of this run, so you have to train it yourself with `launch/train_sft.sh`, once per expert. The
launcher keeps the 8000-step schedule and stops at step 4000 by default (see "Why step 4000 of an 8000-step schedule"
below). A retrained checkpoint is close to the one the released round started from, but not bitwise equal (see "GPU
count" below and the gaps listed in `docs/DATA_STAGE2.md`).

- Trainer: `scripts/s2sft/v2/train_s2sft_v2.py`
- Config: `configs/stage2/sft.yaml` (model, data geometry, optimizer)
- Recipe: `configs/stage2/sft_overrides.json` (pools, mix, schedule length, batch, cadence, validation)
- Launchers: `launch/train_sft.sh` (one expert, one node), `launch/smoke_sft.sh` (1 GPU smoke test)

## Recipe

| item | value |
|---|---|
| base model | Wan2.2-I2V-A14B-Diffusers, one expert per run, bf16 weights, gradient checkpointing |
| LoRA | rank 32, alpha 32, on `attn1.to_{q,k,v,out.0}`, `attn2.to_{q,k,v,out.0}`, `ffn.net.0.proj`, `ffn.net.2` of all 40 blocks; fresh (B = 0) |
| motion adapter | OnlyFlow-style multi-scale adapter (`onlyflow`): hidden 128, depth 2, 3 levels, 8 heads, mixer rank 128, injected into all 40 blocks; 20 base + 7 extra condition channels (`legacy27`); fresh (zero-initialised output) |
| condition | `cond_real`, the ground-truth plan of the clip, for every pool |
| loss | flow matching on the clip's VAE latent, weighted by the moving-object tubes (`fm_weight_mode tube`) |
| clip length | 41 frames at 7 fps, at most 11 latent frames |
| noise level | sigma = 5u / (1 + 4u), u uniform in the expert's band; boundary sigma 0.9 (high: 0.9 to 0.999, low: 0.001 to 0.9) |
| optimizer | AdamW, betas (0.9, 0.999), weight decay 0.01, gradient clipping 1.0 |
| learning rate | LoRA 5e-5, adapter 1e-4; linear warmup over 100 steps, then cosine decay to 0.1x at step 8000 |
| batch | 1 clip per GPU, global batch 16 (GPUs x gradient accumulation) |
| schedule | `max_steps 8000`; the RAFT round of the released generator started from `checkpoint-004000` (not distributed) |
| data | 7 pools, mix P101 0.5 (fall 0.0964, ramp 0.27855, spring 0.03828, multi 0.05524, liquid 0.03152), SSv2 0.1, PhyCo 0.4 |
| sampling | `epoch_mix_v1`: the pool of global sample g is drawn with Random(seed:g); within a pool every clip is seen once per epoch; sigma and noise of sample g come from `torch.manual_seed((seed * 1000003 + g) * 31)` |
| seed | 42 |
| checkpoints | every 100 steps (resume points, only the newest two are kept), every 500 steps kept |
| validation | flow-matching loss on 27 P101 + 32 PhyCo + 32 SSv2 held-out clips, 2 fixed (sigma, noise) draws each, at step 0 and every 500 steps |

### Why step 4000 of an 8000-step schedule

The run was configured for 8000 steps, and step 4000 is the middle of its cosine schedule (the learning rate there is
about 0.56x the peak, not the 0.1x floor). To reproduce `checkpoint-004000`, keep `max_steps 8000` and stop at step 4000
(`launch/train_sft.sh` does this with `STOP_AT_STEP=4000`). Setting `max_steps 4000` gives a different learning-rate
curve, and the trainer refuses to resume a run across a change of `max_steps`.

Step 4000 was chosen after training, by evaluating the kept checkpoints downstream. The run's own lowest validation
loss is not the selection criterion; do not select by `val.jsonl`.

## Requirements

- GPUs: H100 80 GB. Peak memory is about 41 GiB per GPU (11 latent frames, gradient checkpointing, bf16). The
  reference run used 8 GPUs per expert (global batch 16 = 8 GPUs x 2 accumulation steps) at about 4.2 s per optimizer
  step, i.e. about 5 hours to step 4000 per expert.
- Disk: about 3 GB per checkpoint (LoRA, adapter, optimizer state). Up to step 4000 that is 8 kept checkpoints plus 2
  resume points per expert.
- Data: `MOPLAN_SFT_DATA` with complete pools, see `docs/DATA_STAGE2.md`.
- Weights: `WAN_MODEL_DIR` = `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (the trainer loads `transformer/` or `transformer_2/`
  and `vae/config.json`; the text encoder is not needed, the prompts are pre-encoded in the payloads).

## Running

```bash
export WAN_MODEL_DIR=/path/to/Wan2.2-I2V-A14B-Diffusers
export MOPLAN_SFT_DATA=/path/to/sft_data          # pools/ and val/
export MOPLAN_WORK_DIR=/path/to/work              # default: <repo>/work

EXPERT=high NPROC=8 launch/train_sft.sh
EXPERT=low  NPROC=8 launch/train_sft.sh           # on another node, or MASTER_PORT=29501 on other GPUs
```

The launcher reads `$RUN_DIR/OVERRIDES.json` if it exists (written by `make_pools_v2.py`), else
`configs/stage2/sft_overrides.json` with `MOPLAN_SFT_DATA` substituted for `{data_root}`. Extra arguments are passed
to the trainer; a flag given on the command line wins over the overrides file.

Output (`RUN_DIR` defaults to `$MOPLAN_WORK_DIR/runs/stage2_sft`):

```
$RUN_DIR/<expert>/
  checkpoint-NNNNNN/{lora_params.pt, motion_adapter.pt, lora/, optimizer.pt, scheduler.pt, trainer_state.json}
  metrics.jsonl              training losses, gradient norm, learning rates, data mix, every 10 steps
  val.jsonl                  validation loss per pool and per latent frame
  steps_rank<r>.jsonl        every clip each rank trained on (pool, sample id, sigma, loss)
  sampling_plan_summary.json clip coverage of the sampling plan
  resolved_config.yaml       config and resolved run settings
  done.json / paused.json    written when the run ends at max_steps / stops earlier
$RUN_DIR/sft_step004000/{high,low} -> ../<expert>/checkpoint-004000
```

`$RUN_DIR/sft_step004000` has the layout the RAFT round expects for `MOPLAN_SFT_CHECKPOINT` (`high/` and `low/`
with `lora_params.pt` and `motion_adapter.pt`). Since this checkpoint is not distributed, this folder is the
`MOPLAN_SFT_CHECKPOINT` of `docs/TRAINING_RAFT.md`. The link of an expert is written only when its run has reached
step 4000. The same layout loads into the inference script (`--high-checkpoint .../high --low-checkpoint .../low`).

### Stopping and resuming

Relaunching the same command continues from the newest checkpoint (`--resume-from latest`). The sample sequence is a
pure function of the seed, the pools, the mix and the global batch, so a resumed run trains on exactly the samples an
uninterrupted run would. To stop cleanly, create `$RUN_DIR/STOP_<expert>` or send `SIGUSR1` to the worker processes:
the run writes a checkpoint after the current optimizer step and exits. A resume checks that the seed, pools, mix,
batch, learning rates and `max_steps` are unchanged.

### GPU count

`NPROC` may be any divisor of 16; the gradient accumulation is 16 / `NPROC`. Every choice trains on the same sequence
of samples (clip, sigma, noise). The order of floating-point reductions across GPUs differs, and GPU kernels are not
bitwise deterministic, so a retrained checkpoint is close to, but not bitwise equal to, the released one. Use the
released weights to reproduce the published videos.

### Starting point without SFT

`--save-init` writes `checkpoint-000000` (the fresh LoRA with B = 0 and the fresh adapter of the run's seed, i.e. the
untrained Wan2.2) and exits; it is the starting point of RAFT without SFT.

## Smoke test

```bash
export WAN_MODEL_DIR=/path/to/Wan2.2-I2V-A14B-Diffusers
launch/smoke_sft.sh                                  # synthetic payloads, 1 GPU
SMOKE_DATA=/path/to/payloads launch/smoke_sft.sh     # p101/, ssv2/, phyco/ folders of real payloads
```

It builds a run directory from 3 payloads per source (the first of each source is the validation set), trains 2
optimizer steps from the base model with validation at steps 0 and 2, resumes to step 3, and checks the losses, the
checkpoint files, the LoRA update and the markers. `GLOBAL_BATCH=16` uses the real global batch (16 micro-steps per
optimizer step on one GPU). Synthetic payloads have the right shapes and keys but random values, so their losses
carry no meaning.

CPU tests (no GPU, no weights; one test trains a tiny random transformer for 3 steps):

```bash
python -m pytest tests/test_s2sft.py -q
```
