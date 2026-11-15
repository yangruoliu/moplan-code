# Inference

MoPlan generates a video from a first frame and a text prompt in two stages:

1. **Stage-1 (motion planner).** SAM 2.1 proposes up to 16 object masks in the first frame. Then a Wan2.2 high-noise
   expert with a LoRA and a slot decoder predicts how each object moves at 7 query times up to 4 s. This prediction
   is called the *export*.
2. **Stage-2 (generator).** The export is rendered into a 27-channel condition on the latent grid. Both Wan2.2-I2V-A14B
   experts, each carrying a LoRA and a motion adapter, then sample the video at 448x256. Past the last Stage-1 horizon
   (4 s), the condition holds the predicted end state as *known*.

There are two ways to run inference:

- `generate.py` turns one image and one prompt into a video, end to end.
- The **batch path** runs the separate scripts of the benchmark runs over a whole manifest.

Both paths call the same code.

## Requirements

Install the environment as described in [INSTALL.md](INSTALL.md). It covers the pinned Python packages, SAM 2 and
the video tools; inference needs no system ffmpeg (it uses the ffmpeg bundled with imageio-ffmpeg and PyAV). You
need one 80 GB GPU (the reference runs used H100 80GB). The `physics_iq` protocol keeps both 14B experts
resident, which peaks at about 57 GiB.

Set these environment variables:

| variable | content |
|---|---|
| `WAN_MODEL_DIR` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` at revision `596658fd9ca6b7b71d5057529bbf319ecbc61d74` |
| `MOPLAN_WEIGHTS` | the MoPlan weights (layout below) |
| `SAM2_CHECKPOINT` | `sam2.1_hiera_large.pt` (SAM 2.1 Hiera-L); not needed with `--masks` |
| `OMP_NUM_THREADS` | `8`, as in the reference runs (the `launch/infer_*.sh` scripts set it) |

To download the weights, use the setup script. It pins every revision and checks the sha256 of every file
(see [INSTALL.md](INSTALL.md)):

```bash
python setup/download_weights.py --only wan,moplan,sam2     # add cotracker for training and reward scoring
python setup/check_env.py --inference-only                  # CoTracker and the ffmpeg on PATH only give WARN
```

Without `--inference-only`, `check_env.py` reports the missing CoTracker3 checkpoint, which only training data and
reward scoring need, as FAIL.

Alternatively, download them by hand. Pin the revisions recorded in `setup/checksums.json` (the `main` branch of a
Hugging Face repository can move), then check the files with `python setup/download_weights.py --verify-only
--only wan,moplan,sam2`:

```bash
hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers --revision 596658fd9ca6b7b71d5057529bbf319ecbc61d74 --local-dir "$WAN_MODEL_DIR"
hf download yrlyrl/MoPlan --revision b0db0e367f6508977ec9eca693ae3888c52be2dd --local-dir "$MOPLAN_WEIGHTS"
wget -O "$SAM2_CHECKPOINT" https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

The weights are laid out as follows (`tools/weights_manifest.json` lists sizes and sha256):

```
$MOPLAN_WEIGHTS/
  stage1/backend_params.pt       Stage-1 LoRA, input embedding and output layer
  stage1/decoder.pt              Stage-1 slot decoder
  stage1/trainer_state.json      decoder config and expert (read by the loader)
  stage1/resolved_config.yaml    Stage-1 config (data paths are placeholders; the Wan path comes from WAN_MODEL_DIR)
  stage1/prior_latent.pt         init latent of the direct Stage-1 objective
  stage1/checkpoint_complete.json   provenance only
  stage2/high/{lora_params.pt,motion_adapter.pt}   high-noise expert of the generator
  stage2/low/{lora_params.pt,motion_adapter.pt}    low-noise expert of the generator
```

To check the sha256 of every file in use, add `--verify-weights` to any `generate.py` call.

No optimizer, scheduler or RNG state is needed or shipped. The Stage-1 loader (`scripts/infer_stage1_physicsiq.py`)
reads only the files above.

## Quickstart

```bash
bash launch/infer_generate.sh --image frame.png --prompt "A hand pushes the mug from right to left." --output out/mug.mp4
```

`launch/infer_generate.sh` sets the environment and runs `generate.py` with the same arguments, on the first GPU of
`$GPUS`. A call writes these files:

| file | content |
|---|---|
| `out/mug.mp4` | the video |
| `out/mug.npy` | the exact uint8 frames `[F, 256, 448, 3]`; skip it with `--no-npy` |
| `out/mug.json` | protocol, seed, caption, inputs (the image by file name and sha256) and the sha256 of the frames |

Common options:

| option | effect |
|---|---|
| `--protocol moprobe` (default) / `physics_iq` | which paper pipeline to follow (see below) |
| `--sample-id SID` | sample id (default: the image file name); with `moprobe` it selects the benchmark seed `job_seed(seed, SID, k)` |
| `--seed N`, `--seed-index K` | base seed (default 42) and `k` |
| `--base` | the untrained Wan2.2-I2V-A14B, with no LoRA, adapter or condition; same sampler, image, prompt and seed |
| `--canvas 16x9` | first letterbox the image into a black 1280x720 canvas (the MoProbe input rule) |
| `--caption-rule auto` / `exact` / `static-suffix` | how the prompt becomes the caption (see below) |
| `--masks K16.pt` | use a k16 mask file instead of running SAM 2 |
| `--stage1-export EXPORT.pt` | use a Stage-1 export instead of running Stage-1 |
| `--save-intermediates` | keep the work folder `OUT.work/` (see "Intermediate files") |

Run `python generate.py --help` for the full list.

## Protocols

Both protocols share these settings:

- 448x256 resolution;
- the Wan2.2 pipeline's own UniPC schedule (flow shift 3.0) with 30 steps;
- guidance 5.0 for both experts, boundary ratio 0.9;
- Wan2.2's official negative prompt and 512 text tokens;
- the first frame resized to 448x256 with PIL LANCZOS (a stretch, with no crop);
- for Stage-1, the k16 masks, the 7 query horizons (5/24, 10/24, 15/24, 20/24, 1.2, 2, 4 s), text batch 4 and seed 42.

### `--protocol moprobe`

This is the path of `scripts/infer_stage2_cond_v2.py`, used for MoProbe, PisaBench, RigidBench, Morpheus and VBench.
`generate.py` feeds one caption to both Stage-1 and the generator and has no prompt table. Benchmark runs in which
the generator read other text than the Stage-1 caption (`--prompt-table PROMPT_TABLE.json`, written by
`scripts/bench/official_sp_inputs.py`, for example the Morpheus enhanced prompts) can only be reproduced with the
batch path below. A benchmark's own negative prompt can be passed to `generate.py` with `--negative-prompt`.

- **Clip.** 37 frames at 7 fps by default. `--frames` (any 4k+1) and `--fps` change it, for example 17 frames as in
  RigidBench.
- **Condition.** `generate.py` builds a `stage2_cond_v2` cache item the way `scripts/cache_stage2_cond_v2.py --no-video`
  does: the Wan VAE posterior mode of the stretched frame followed by zero frames, stored as fp16, with the frame times
  rounded to 1e-6 s. `refresh_predictions.refresh_payload` then renders `cond_pred` / `cond_zero` / `cond_null` from
  the export, the fp16 latent and those times. Finally `cond_pred` is held past the last known latent
  (`scripts/s2eval/hold_known_beyond_horizon.py`).
- **Generation.** `generate.py` then calls `infer_stage2_cond_v2.main` three times: `embed`, then `high` (the high-noise
  expert up to the boundary; it keeps the latent and the scheduler state), then `low`.
- **Seed.** With `--sample-id`, the seed is `job_seed(42, sample_id, k) = 42 + crc32(sample_id) % 100000 + 1000003 k`.
  Otherwise it is `--seed + k`.
- **Caption.** `bench_common.caption_for` normalises whitespace. It appends " Static shot with no camera movement."
  only when the prompt itself states a static camera.
- **Video.** The mp4 is h264, crf 10, yuv420p, at the clip's frame rate (the MoProbe export format).

### `--protocol physics_iq`

This is the native profile of `scripts/s2eval_verified/gen_verified.py` (`NativeGenerator`), used for Physics-IQ.

- **Clip.** 41 frames.
- **Condition.** The condition is rendered on the fly at the 8 fps latent times (0, 0.5, ..., 5 s), with the hold rule.
- **Generation.** Both experts stay resident.
- **Seed.** One flat seed per run: `--seed 42` is run 1, and 43, 44, 45 are runs 2 to 4.
- **Caption.** The prompt is used verbatim, and Stage-1 reads the same text.
- **Video.** The raw 41-frame video is written with imageio (libx264, quality 8) at 8 fps, then trimmed to 5.000 s
  (40 frames) with `ffmpeg -t 5` (the ffmpeg bundled with imageio-ffmpeg). The trimmed file must have 40 frames and a
  container duration of 5.000 s (measured with a working `ffprobe` on `PATH`, else with PyAV); otherwise the run
  stops with an error. `OUT.mp4` is the trimmed file. `OUT.npy` holds all 41 generated frames, and the raw
  mp4 is kept under `--save-intermediates`.
- **Fixed profile.** `--frames`, `--fps` and `--negative-prompt` are rejected for this protocol.

### The hold rule and its edge case

Both protocols hold the condition past the last *known* latent: every later latent copies the full 27-channel
condition of that latent (`scripts/s2eval/hold_known_beyond_horizon.py`, and `hold_known_beyond_horizon` in
`scripts/s2eval_verified/gen_verified.py`). The intended rule is to hold from the last Stage-1 horizon (4 s) on.

The code finds the last known latent from the condition itself: the last latent whose `known` channel (20) is
non-zero anywhere. Channel 20 is drawn only on the objects' transformed footprints
(`scripts/control/plan_condition.py`). If every planned object, static ones included, is outside the frame from some
latent time before 4 s on, those latents carry `known` = 0 everywhere. The hold then starts at an earlier latent and
overwrites the later latents, including those up to 4 s, with that earlier condition. Choosing the hold point by
time (the last latent at or before the last Stage-1 horizon) would avoid this. The code is kept as it is because the
released outputs were made with it; a change would alter outputs whenever the case occurs.

The case needs every object to leave the frame, so it is rare. `HOLD_KNOWN_RECEIPT.json`, written next to a held
cache, counts the last known latent of every item in `last_known_hist`. A value below the index of the last latent
at or before 4 s (7 for a 37-frame clip at 7 fps) marks an item where this happened.

### Reproducing the showcase panels

`assets/showcase/*/meta.json` records the command, prompt, seed and reference hash of every generated panel. These
commands print and check them:

```bash
python assets/showcase/showcase.py commands              # the generate.py commands
python assets/showcase/showcase.py verify out/showcase      # compare outputs with the reference hashes
```

The two RigidBench cases (8 and 9) are reported as `NOT-VERIFIABLE`: their references are hashes of the 1280x704
evaluation exports, not of the files `generate.py` writes, so they do not fail the exit code.

The MoProbe inputs in `assets/showcase` are already the 1280x720 canvases, so do not add `--canvas`.

## Intermediate files (`--save-intermediates`)

`OUT.work/` holds these files:

| path | content |
|---|---|
| `inputs/<sid>.png` | the 16:9 canvas (only with `--canvas 16x9`) |
| `masks/sam2_raw/<sid>.pt`, `masks/k16/<sid>.pt` | SAM 2.1 masks before and after the k16 filter (the Stage-1 input) |
| `stage1/flows/<sid>.pt`, `stage1/inference-shard-0000.json` | Stage-1 export and run record |
| `cache/<sid>.pt` | (moprobe) the held `stage2_cond_v2` cache item the generator read |
| `prompt_embeds.pt` | (moprobe) the umT5 embeddings in the `infer_stage2_cond_v2.py` format |
| `gen/pred__<sid>.npy`, `gen/pred__<sid>.json` | (moprobe) the raw generator output (`base__<sid>` with `--base`) |
| `condition.pt` | (physics_iq) the fp16 condition the adapters read |
| `prompt_embeds_native.pt` | (physics_iq) the trimmed embeddings in the `gen_verified.py` format |
| `raw/<name>.mp4` | (physics_iq) the 41-frame video before trimming |

Without `--save-intermediates`, the folder is removed at the end. A folder given with `--work-dir` is never removed.

## Batch path

Use the batch path to reproduce benchmark generations, or to run many samples.

**1. Inputs.** Build a manifest CSV with the columns `sample_id`, `first_frame_path` and `caption`. The benchmark
adapters write one, in the format of `scripts/bench/bench_common.py`; `launch/eval_bench_inputs.sh` runs the ones the
evaluation uses (for example `scripts/bench/miqB_inputs.py` for MoProbe-H, which needs the MoProbe bundle).

**2. Held condition caches.** This step runs SAM 2.1, the k16 filter, Stage-1, the no-video base cache, the
re-rendering of the Stage-1 conditions (`refresh_predictions.py`) and the hold rule:

```bash
MANIFEST=work/moprobe/inputs/manifest.csv WORK=work/moprobe FRAMES=37 bash launch/infer_batch_caches.sh
```

**3. Videos.** This step runs `infer_stage2_cond_v2.py`: `embed` on one GPU, then `high` and `low` sharded over all GPUs:

```bash
CACHE_DIR=work/moprobe/cache OUT=work/moprobe/gen MODEL=moplan bash launch/infer_batch_stage2.sh
CACHE_DIR=work/moprobe/cache OUT=work/moprobe/gen_base MODEL=base bash launch/infer_batch_stage2.sh
```

**MoProbe layout.** Set `MIQ_MANIFEST=work/moprobe/inputs/manifest.jsonl` to also export the clips as
`<set>/gen/<arm>/s<k>/<sid>.mp4` (`scripts/s2eval/miq_export.py`).

**Several seeds.** Set `NUM_SEEDS=5 SEED_MAJOR=1` for several seeds per sample (VBench used 5).

**Extra flags.** Flags given after the script name go to every phase, for example a benchmark's own
`--negative-prompt "..."` or `--prompt-table T.json`.

**Direct commands.** The same steps without the launcher:

```bash
python scripts/infer_stage2_cond_v2.py --config configs/stage2/inference.yaml --cache-dir C --ids IDS --arms pred \
  --high-checkpoint "$MOPLAN_WEIGHTS/stage2/high" --low-checkpoint "$MOPLAN_WEIGHTS/stage2/low" \
  --output-dir OUT --prompt-embeds OUT/prompt_embeds.pt --phase embed        # then --phase high, then --phase low
```

- `--phase high` and `--phase low` take `--shard-index i --num-shards n`.
- The low pass must get the same `--cache-dir` string and the same sampler flags as the high pass, because the
  stored stage files are checked.
- `--phase full` does everything in one call with CPU offload (about 70 GB of host RAM). It is not the path of the
  reference runs.
- The Wan2.2 directory: `infer_stage2_cond_v2.py` takes `--wan-model`, then `$WAN_MODEL_DIR`, then the config's
  `paths.wan_model`; `infer_stage1_physicsiq.py` takes the config's `paths.wan_model` and falls back to
  `$WAN_MODEL_DIR` only when it is empty. The shipped configs (`configs/stage2/inference.yaml` and the released
  `stage1/resolved_config.yaml`) leave it empty, so both stages read `$WAN_MODEL_DIR`. With a config that sets it,
  keep it equal to `$WAN_MODEL_DIR`.

**Physics-IQ batch runs.** Use `launch/eval_piq_prepare.sh` and `launch/eval_piq_generate.sh`
(`gen_verified.py --profile native`; see the evaluation documentation).

**Zero-flow ablation.** `scripts/s2eval/make_zero_caches.py` turns held caches into caches whose `cond_pred` is the
held `cond_zero`.

## Bit-exactness: known sources of difference

`generate.py` calls the same functions as the batch path. On CPU, for a fixed export, its moprobe cache item has been
checked to be bitwise identical to the chain `cache_stage2_cond_v2.py --no-video` -> `refresh_predictions.py` ->
`hold_known_beyond_horizon.py`.

Bit-for-bit equality of videos has only been observed on H100 GPUs with the pinned software stack. These factors can
still change pixels, mostly by small amounts:

1. **Stage-1 text batches.** Stage-1 encodes its captions with umT5 in batches of 4. The empty negative prompt comes
   first, followed by the captions of the shard in manifest order. A batch run therefore encodes each caption next to
   different neighbours than a single-sample run does (`["", caption]`), and bf16 GEMMs of a different batch shape
   can round differently.
   - To reproduce a batch run's export exactly, pass its manifest with `--stage1-text-manifest M.csv` and its shard
     with `--stage1-text-shard i/n`.
   - In `infer_batch_caches.sh`, keep the same manifest order and run the Stage-1 export in one shard (the default,
     `STAGE1_GPUS` = the first GPU), as the reference runs did.
2. **SAM 2 masks.** The k16 masks of the Physics-IQ reference runs were made by an older version of the mask builder
   that is not in this repository. Their records name the source `sam2.1_hiera_large_automatic+filter_v2`, while the
   released `scripts/cache_sam2_masks.py` + `scripts/filter_sam2_masks_v2.py` write
   `sam2.1_hiera_large_automatic+nearest_labels_v2+filter_v2`. These masks are not distributed, so Physics-IQ runs
   (and their Stage-1 plans) are not expected to be bit-exact, independent of the GPU (see [EVALUATION.md,
   Bit-exact reproduction](EVALUATION.md#bit-exact-reproduction)). Beyond that, automatic masks depend on the GPU,
   the sam2 commit and the CUDA kernels. Pass the original k16 mask file with `--masks` (or `MASKS_DIR=` in the batch
   path) to rule this out.
3. **Hardware and libraries.** GPU model, driver, CUDA, cuBLAS and cuDNN versions, and the versions of torch
   (2.5.1, CUDA 12.4), diffusers (0.38.0), transformers (5.11.0), tokenizers, ftfy, peft, opencv-python and pillow.
   Different versions can change the text embeddings, the VAE latents and the sampler. The flow shift 3.0 comes from
   the scheduler config of the pinned Wan2.2 snapshot.
4. **Attention backend.** Everything used torch's native SDPA. Another backend (`gen_verified.py --attn`) changes
   pixels.
5. **Offload.** The reference runs used `embed` / `high` / `low` without offload (moprobe) and both experts resident
   (physics_iq). `generate.py` does the same. `--phase full` with CPU offload is meant to be equal, but it was not
   the evaluated path.
6. **One process instead of one per stage.** `generate.py` runs every stage in one process. It records the
   process-global torch settings (threads, TF32, cuDNN flags, SDPA backends) at the start and restores them before
   each later stage. The CUDA allocator state differs, which is not expected to change results.
7. **CPU threads.** The condition is rendered on the CPU. The reference runs used `OMP_NUM_THREADS=8`, and some torch
   CPU reductions depend on the thread count.
8. **Image decoding and canvas.** Stage-1 and SAM 2 read the image with OpenCV, and Stage-2 reads it with PIL. JPEG
   decoding can differ across library builds. `--canvas 16x9` recomputes the MoProbe canvas with OpenCV resizing. To
   reproduce a MoProbe sample, pass the original 1280x720 canvas PNG instead.
9. **Caption.** Stage-1 truncates captions to 64 umT5 tokens. The generator reads the caption (moprobe:
   `caption_for(prompt)`; physics_iq: the prompt verbatim). Benchmarks that fed the generator other text
   (`--prompt-table`) or a different negative prompt need the same flags; `generate.py` takes `--negative-prompt` but
   has no prompt table, so runs with a prompt table need the batch path.
10. **Seed rule.** The moprobe benchmark seed needs the exact sample id (for example `ssv2_9578-ex`, or the full
    Physics-IQ name `0001_perspective-left_trimmed-ball-and-block-fall`, not `0001`). Physics-IQ uses a flat seed for
    every video of a run.
11. **Protocol-specific rounding.** The moprobe path renders from the fp16 cache latent and from frame times rounded
    to 1e-6 s. The physics_iq path renders from the fp32 latent and the exact times. Each protocol reproduces its own
    published path, and the two are not interchangeable.
12. **Video files.** mp4 bytes depend on the encoder build: PyAV/libx264 for moprobe, imageio-ffmpeg for physics_iq.
    Compare `OUT.npy` or the decoded frames rather than mp4 hashes, unless the encoder binaries are the same.
13. **Prompt embeddings.** The batch runs encoded every prompt one at a time (`encode_prompt`, batch 1), which is what
    `generate.py` does. A precomputed embedding file from another software stack can differ.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_infer.py      # CPU only, no weights needed
```

The tests cover these points:

- argument parsing and the protocol defaults;
- the seed rule against `infer_stage2_cond_v2.job_seed`;
- the hold rule and its on-the-fly copy;
- the cache-item key contract with a synthetic export;
- the k16 filter function against its command line;
- the 16:9 canvas;
- the negative-prompt bytes;
- the inference config;
- the weights manifest;
- the duration check of the `physics_iq` writer without a working `ffprobe` (PyAV fallback, error when neither works);
- the ffmpeg binary of the clip builders and their frame count.
