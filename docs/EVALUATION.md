# Evaluation

This page explains how to evaluate MoPlan and the untrained Wan2.2-I2V-A14B baseline under the same generation
protocol on Physics-IQ, PisaBench, RigidBench, Morpheus and VBench-I2V. MoProbe is covered in
[MOPROBE.md](MOPROBE.md).

For each benchmark you set up the official scorer, generate videos with MoPlan and with the base model, then score
both sets with the unmodified official code. The launchers under `launch/` are plain single-node bash scripts. You
configure them with environment variables, and each one documents its own variables in its header.

## 1. Common setup

Environment of the MoPlan pipeline (see the main README):

| variable | meaning |
|---|---|
| `WAN_MODEL_DIR` | `Wan2.2-I2V-A14B-Diffusers` directory |
| `MOPLAN_WEIGHTS` | downloaded MoPlan weights (`stage1/`, `stage2/high`, `stage2/low`) |
| `SAM2_CHECKPOINT` | SAM 2.1 hiera-large checkpoint (`sam2.1_hiera_large.pt`), used for the first-frame instance masks |
| `EVAL_ROOT` | working folder for all evaluation inputs and outputs (default `<repo>/eval_work`; a relative path is taken from the current folder) |
| `PYTHON` | python of the MoPlan environment (default `python`) |
| `GPUS` | space-separated GPU indices (default: every visible GPU) |

Each benchmark scorer runs in its own environment, never in the MoPlan environment. The environments used for the
reference evaluation runs are listed in `docs/eval_envs/` (installed distributions and python version).

For a smoke test, `LIMIT=n bash launch/eval_bench_generate.sh <bench> <model>` generates the first n ids only. The
benchmark's prompt-embedding file, which every model of the benchmark shares, is still encoded for the full id list,
so the full run can follow directly. `GEN_COMPLETE`, which the scoring launchers require, is written only by a run
without `LIMIT`, once every clip of the full id list exists.

### Generation protocol (all benchmarks)

Both models go through the same Stage-2 sampler:

- the diffusers `WanImageToVideoPipeline`
- UniPC with the pipeline's own schedule (flow shift 3), 30 steps
- CFG 5 on both experts, expert boundary 0.9
- a 448x256 work frame (the first frame is stretched to 448x256 with PIL LANCZOS)
- umT5 prompt embeddings with 512 tokens

The base model runs the identical call without the LoRA, the motion adapter or the motion-plan condition. It uses the
same images, prompts, negative prompts and noise.

The two paths differ as follows:

| | Physics-IQ (native profile) | PisaBench / RigidBench / Morpheus / VBench-I2V / MoProbe |
|---|---|---|
| entry | `scripts/s2eval_verified/gen_verified.py --profile native` | `scripts/infer_stage2_cond_v2.py` on condition caches |
| frames | 41 frames written at 8 fps, trimmed to 40 frames / 5.000 s (`ffmpeg -t 5`) | 37 frames @ 7 fps (RigidBench: 17) |
| noise | `torch.Generator("cuda").manual_seed(41 + run)` for every video of a run | `job_seed(42, sample_id, k) = 42 + crc32(sample_id) % 100000 + 1000003 k` |
| condition | rendered on the fly from the Stage-1 export; hold rule past 4 s (`--hold-known`) | held condition caches (`launch/eval_bench_prepare.sh`) |
| negative prompt | official Wan2.2 default | the benchmark's official one where it has one, else the Wan2.2 default |

The Stage-1 planner always reads the same prompt as the generator, with one exception: on Morpheus it reads the
plain English prompt (see below). Stage-1 takes the first frame and 16 SAM 2.1 instance masks of that frame. No
benchmark video, ground-truth mask or ground-truth motion is used anywhere.

### Bit-exact reproduction

The reference videos were generated on H100 80GB GPUs with the pinned MoPlan environment (torch 2.5.1, diffusers 0.38.0)
and the native torch SDPA attention. Other GPUs, library versions or attention kernels change pixels.

- **Stage-1 exports.** Stage-1 encodes its captions in batches of 4, and the batch composition can change bf16
  results. The reference exports of Physics-IQ, PisaBench, RigidBench, Morpheus and VBench-I2V were each made in one
  shard, in manifest order, and the launchers therefore run Stage-1 on a single GPU. MoProbe-H is the exception: its
  reference exports come from three separate Stage-1 runs over parts of the frame set, while
  `launch/eval_bench_inputs.sh moprobe_h` builds one manifest of all frames. The caption batches therefore differ,
  and MoProbe-H Stage-1 exports are not expected to be bit-exact.
- **SAM 2 masks of Physics-IQ.** The k16 masks of the Physics-IQ reference runs were made by an older version of the
  mask builder that is not in this repository, and these masks are not distributed (except the four of the showcase
  Physics-IQ cases, `assets/showcase/0[4-7]_*/masks_k16.pt`). Their records name the source
  `sam2.1_hiera_large_automatic+filter_v2`, while the released builder (`scripts/cache_sam2_masks.py`, then
  `scripts/filter_sam2_masks_v2.py`) writes `sam2.1_hiera_large_automatic+nearest_labels_v2+filter_v2`. On some
  Physics-IQ frames the two builders keep different instances, which changes the Stage-1 plan and the video. A
  Physics-IQ rerun with the released code is therefore not expected to be bit-exact, on any GPU. Given the reference
  masks, the released code reproduced the checked Physics-IQ reference videos exactly on the reference setup. For
  PisaBench, RigidBench and MoProbe-H the released builder was checked to rebuild the reference masks byte for byte;
  Morpheus and VBench-I2V were not checked. Automatic masks can also change with the GPU and the SAM 2 version.
  `MASKS_DIR` takes precomputed k16 masks (`<sid>.pt`) instead of running SAM 2.
- **Morpheus.** The dynamical score's PINN fits are not seeded upstream and can move between re-scorings. The
  physical-invariance score is deterministic.
- **VBench-I2V.** The camera-motion verdicts can flip between machines. Score both models on the same machine.
- **Derived first frames.** The PisaBench canvas (`scripts/eval/pisa_clips.py`, OpenCV) and the downscaled VBench crops
  (`vbench_inputs.py`, PIL) are computed from the benchmark data. Use the pinned MoPlan environment to get the same pixels.
- **ffmpeg builds.** The Physics-IQ trim (`ffmpeg -t 5`) uses the ffmpeg bundled with the pinned imageio-ffmpeg
  0.6.0 (`imageio_ffmpeg.get_ffmpeg_exe()`), so its bytes are reproducible with the pinned environment. The
  PisaBench 7-fps preview clips of `scripts/eval/pisa_clips.py` (not read by the evaluation, which uses the PNG first
  frames) are encoded with the same imageio-ffmpeg binary, or with the one given by `--ffmpeg` / `$FFMPEG`. The
  5.000 s duration check of the Physics-IQ generation uses an `ffprobe` on `PATH` when one runs, otherwise PyAV, and
  stops with an error when neither works. The `ffmpeg` / `ffprobe` found on `PATH` are still used by the Physics-IQ
  pre-check (`scripts/eval/piq_precheck_run.py`, duration and frame-rate checks) and by the official scorers
  themselves (for example the PisaBench frame-rate conversion). The build on `PATH` for the reference runs was not
  recorded. The ffmpeg 4.3 of the lock file does not start as installed (its `libavcodec` needs `libopenh264.so.5`,
  and the lock's openh264 provides only `.so.6`) and has no libx264; [INSTALL.md, Video
  tools](INSTALL.md#video-tools) gives the fix. Use one build for both models so that any scorer-side difference
  affects both equally.

## 2. Physics-IQ (original and Verified)

| item | value |
|---|---|
| code | [google-deepmind/physics-IQ-benchmark](https://github.com/google-deepmind/physics-IQ-benchmark): Verified scorer at commit `c2bca8a5bf452d24e177e9975641af1888eb1492` (`physiq/`), original-leaderboard scorer at commit `ec2c61105ace452609d984fb503e4412edfca21e` (`code/`) |
| data, original | public bucket `gs://physics-iq-benchmark`: `switch-frames/`, `split-videos/testing/{8,30}FPS`, `video-masks/real/8FPS` |
| data, Verified | Hugging Face dataset `Anates-Labs-Research/Physics-IQ-Verified` at revision `b65f567dd8446a4076c7f72446335b3c217a8464` (gated) |
| inputs | the 198 take-1 switch frames; prompts `op` = `descriptions/descriptions_original.csv`, `bpp` = `descriptions/best_practice/descriptions_base.csv` (both at c2bca8a) |
| runs | 4 runs per prompt setting, seeds 42, 43, 44, 45 |

### Scorer environment

1. Clone the repository twice and check out the two commits:
   - `PIQ_CODE_VERIFIED` at c2bca8a
   - `PIQ_CODE_ORIGINAL` at ec2c611
2. Create a Python 3.12 venv.
3. Install the c2bca8a lock file: `uv pip install -r pylock.toml`.
4. Install the c2bca8a checkout without dependencies: `uv pip install --no-deps -e $PIQ_CODE_VERIFIED`.
5. Set `PIQ_PYTHON` to this venv's python. The ec2c611 scorer runs in the same venv.
6. Put a working `ffmpeg` and `ffprobe` on `PATH` (see [INSTALL.md, Video tools](INSTALL.md#video-tools)).
7. Set `PIQ_DATA` to a folder that holds `physics-IQ-benchmark/` and `physics-IQ-benchmark-verified/`, each with
   `switch-frames/`, `split-videos/testing/<fps>FPS/` and `video-masks/real/<fps>FPS/`.

Check the data before you score:

- The native profile is scored at 8 fps, so the 8FPS testing videos and real masks are needed.
- The 30FPS testing folder is needed by both scorers, which validate file names against it.
- The precomputed GT is used as it is. The wrapper refuses to score when the GT for the run's fps is missing.

### Generate

```bash
bash launch/eval_piq_prepare.sh                      # manifests, k16 masks, prompt embeddings, Stage-1 exports (op + bpp)
for s in op bpp; do for r in 1 2 3 4; do
  bash launch/eval_piq_generate.sh moplan $s $r      # MoPlan        -> $EVAL_ROOT/physics_iq/gen/moplan-<s>-run_0<r>
  bash launch/eval_piq_generate.sh base   $s $r      # base Wan2.2   -> $EVAL_ROOT/physics_iq/gen/wan22-base-<s>-run_0<r>
done; done
# ablation: the MoPlan generator fed the zero-motion plan -> moplan-zero-<s>-run_0<r>
bash launch/eval_piq_generate.sh zero bpp 1
```

`eval_piq_generate.sh` behaves as follows:

- It starts one worker per GPU. The workers share the run folder through claims.
- Finished videos are skipped, so you can run the same command again, or on several nodes at once, until the folder
  holds 198 videos.
- Each run folder is exactly what the scorer expects: 198 mp4 files named `<id>_..._.mp4`, each 5.000 s at 8 fps and
  448x256.

### Score

```bash
export PIQ_PYTHON=... PIQ_CODE_VERIFIED=... PIQ_CODE_ORIGINAL=... PIQ_DATA=...
bash launch/eval_piq_score.sh $EVAL_ROOT/physics_iq/gen/moplan-bpp-run_01 moplan-bpp-run_01 verified   # one run, one scorer
bash launch/eval_piq_report.sh moplan wan22-base                                                      # everything below
```

`eval_piq_score.sh` runs these steps:

1. It checks the run folder (`scripts/eval/piq_precheck_run.py`).
2. It builds a private work tree with hard links of the videos, symlinks to the GT and a private generated-mask
   folder, so the scorer never renames your files or reuses a stale mask cache.
3. It runs the unpatched scorer:
   - `verified`: c2bca8a, descriptions chosen by `-op-` / `-bpp-` in the run name
   - `orig_new`: c2bca8a `--original_physics_iq`
   - `orig_ec2c611`: the original-leaderboard scorer
4. It writes `<NAME>__<MODE>.json` (`scripts/eval/piq_summarize_run.py`) and the scorer's CSV to `$PIQ_RESULTS`.

`eval_piq_report.sh LABEL [BASE]` runs these steps:

1. It scores every run of the label: op runs with all three scorers, bpp runs with `verified`.
2. It aggregates the runs with `scripts/s2eval_verified/agg_runs_v.py` (mean and std with ddof 1, plus the official
   `physiq/aggregate_runs_from_csvs.py --score-type verified`).
3. With a base label, it writes the per-category breakdown with a paired scenario bootstrap
   (`scripts/eval/piq_breakdown.py`: 10,000 replicates, seed 20261024). The breakdown re-implements the official
   aggregation as a function of scenario weights and checks it against the official code on every category.

The reported columns are:

- Verified: bpp runs scored with `verified`, the mean of the per-run scores.
- Original: op runs scored with `orig_ec2c611`, the mean of the unrounded per-run scores. `orig_new` gives the same
  value.

Both columns are rounded half up to 2 decimals from the stored values. `eval_piq_report.sh` prints them at the end
(`scripts/eval/piq_report_values.py`). The ec2c611 scorer prints each run's score rounded to 2 decimals, and
`agg_runs_v.py` averages these printed scores, so its Original mean (`agg__<label>-op__orig_ec2c611.json`, printed
rounded half up to 2 decimals) can differ from the reported value by 0.01. The unrounded per-run scores come from the breakdown, which
recomputes them from the scorer CSVs and checks them against the printed values. The Original column is therefore
taken from `breakdown_<label>_vs_<base>.json` when a base label is given; without one, the report prints the mean of
the printed scores and says so.

## 3. PisaBench (real-world split)

| item | value |
|---|---|
| code | [vision-x-nyu/pisa-experiments](https://github.com/vision-x-nyu/pisa-experiments) at commit `65866af0f698fba7b3f38bc2e15bb044584dd08a` |
| data | Hugging Face dataset `nyu-visionx/pisa-experiments`, `pisabench/real.zip` (sha256 `8006c91383cf8c6c1283b727042081731897b0ca9ceda9f65aa65659cc7bc3ef`), all 361 clips |
| masks | SAM 2.1 hiera-large (`facebook/sam2.1-hiera-large` from the Hugging Face hub) through upstream [facebookresearch/sam2](https://github.com/facebookresearch/sam2) at commit `c2ec8e1` |
| resolution | GT 1080x1080 converted to 256x256 by the official `convert_resolution.py`; predictions at 16 fps (the released `pd_fps`) |

### Scorer environment

Set up a folder `$PISA_HOME` with this layout:

- `code/` is the pisa-experiments checkout. Give it a venv `code/.venv` (we used Python 3.10 and the packages in
  `docs/eval_envs/pisabench.txt`).
- `data/real.zip` is the benchmark data.
- `run/sam2` is a symlink to an upstream sam2 checkout at c2ec8e1 with its extension built (`python setup.py
  build_ext --inplace`). The repository's own `sam2/` copy is modified and does not run `generate_mask.py`, so
  `generate_mask.py` is started from `run/`.

Download the SAM 2.1 weights into `HF_HOME` beforehand, because the scorer runs offline.

### Generate and score

```bash
export PISA_HOME=...
bash launch/eval_bench_inputs.sh pisabench          # first frames (scripts/eval/pisa_clips.py) + manifest with the official prompt / negative
bash launch/eval_bench_prepare.sh pisabench         # masks, Stage-1 exports, condition caches
bash launch/eval_bench_generate.sh pisabench moplan
bash launch/eval_bench_generate.sh pisabench base
bash launch/eval_score_pisabench.sh moplan
bash launch/eval_score_pisabench.sh base
```

- The first frame is the 1080x1080 clip frame placed in the middle of a 1280x720 canvas. The side bands are blurred
  border colours.
- The export (`scripts/bench/pisa_official_export.py`) crops the original square (work columns 98..350) back out of
  every generated frame, resizes it to 256x256 (PIL bicubic) and re-times it to 16 fps by nearest frame.
- `eval.py` scores the overlap of the generated and GT durations. Clips that raise an exception in `eval.py` are
  reported as failed, and the launcher requires all 361 clips to be scored.

Fall speed (effective gravity from the same masks): `scripts/s2eval/pisa_fall_speed.py`. It runs through
`launch/eval_bench_compare.sh`.

## 4. RigidBench

| item | value |
|---|---|
| code | [swarnim-j/RigidBench](https://github.com/swarnim-j/RigidBench) at commit `f0b8f29`, with `vendor/Video-Depth-Anything` at `4f5ae23` |
| data | Zenodo record 21649156 (v1.1), downloaded with `rigidbench download` into `<checkout>/data` |
| set | the 100 evaluation samples (5 tasks x 20) |

### Scorer environment

1. Set `RIGIDBENCH_HOME` to the checkout.
2. Install its own environment with `uv sync --frozen --extra eval`. This creates `.venv`.
3. Build SAM 2 for the evaluator: `sam2==1.1.0`, built with CUDA.
4. Pre-fetch the evaluator's models on a machine with network access. The scorer runs offline, and
   `launch/eval_score_rigidbench.sh` stops before any work when one of these entries is missing
   (`SKIP_CACHE_CHECK=1` skips the check).

The reference package set is in `docs/eval_envs/rigidbench.txt`.

The evaluator loads DINOv2 (identity drift) and CoTracker3 (point tracks) with `torch.hub.load`, without a pinned
version, and LPIPS loads the torchvision AlexNet weights. These go into the scorer's `TORCH_HOME`, which the launcher
sets to `RIGIDBENCH_TORCH_HOME` (default: `TORCH_HOME`, else torch's default `~/.cache/torch`). When the repository
folder is already in the cache, torch.hub uses it as it is; otherwise it downloads the current default branch, and the
scores are no longer tied to the reference code. The reference runs used this tree:

| path under `$RIGIDBENCH_TORCH_HOME/hub/` | content |
|---|---|
| `facebookresearch_co-tracker_main/` | a checkout of facebookresearch/co-tracker at commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` |
| `facebookresearch_dinov2_main/` | a checkout of facebookresearch/dinov2 at commit `7764ea0` |
| `checkpoints/scaled_offline.pth` | CoTracker3 offline weights (`facebook/cotracker3` on the Hugging Face hub) |
| `checkpoints/dinov2_vitl14_pretrain.pth` | DINOv2 ViT-L/14 weights (`https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth`) |
| `checkpoints/alexnet-owt-7be5be79.pth` | torchvision AlexNet weights (`https://download.pytorch.org/models/alexnet-owt-7be5be79.pth`) |

SAM 2.1 and Video Depth Anything come from the Hugging Face hub. The launcher sets the scorer's `HF_HOME` to
`RIGIDBENCH_HF_HOME` (default: `HF_HOME`, else the hub's default `~/.cache/huggingface`) and reads the hub cache in
its `hub/` subfolder:

| repository | file | reference revision |
|---|---|---|
| `facebook/sam2.1-hiera-large` | `sam2.1_hiera_large.pt` | `665f8e2ad61cf5f53d65644ff27c8ee525124610` |
| `depth-anything/Video-Depth-Anything-Large` | `video_depth_anything_vitl.pth` | `7aafbcb5c6af0bac741aad2b6471894fb4761afa` |

The evaluator asks for the `main` revision, and offline the hub resolves `main` through
`hub/models--<org>--<name>/refs/main`. To pin a revision, download it with
`HF_HOME=$RIGIDBENCH_HF_HOME hf download <repository> <file> --revision <sha>` and write the sha into that
`refs/main` file.

### Generate and score

```bash
export RIGIDBENCH_HOME=...
bash launch/eval_bench_inputs.sh rigidbench         # scene prompt + official suffix, official negative prompt, 17 frames @ 7 fps
bash launch/eval_bench_prepare.sh rigidbench
bash launch/eval_bench_generate.sh rigidbench moplan
bash launch/eval_bench_generate.sh rigidbench base
bash launch/eval_score_rigidbench.sh moplan
bash launch/eval_score_rigidbench.sh base
```

- The 448x256 work frames are stretched to 1280x704 and written at 7 fps. 17 frames cover the 2 s the evaluator scores.
- The launcher spreads the per-sample metrics over the GPUs. It then runs the official `rigidbench evaluate`, which
  aggregates and writes `results.json`.

## 5. Morpheus

| item | value |
|---|---|
| code | [physics-from-video/Morpheus](https://github.com/physics-from-video/Morpheus) at commit `ee3b6ae`, with its SAM 2 copy at `412c70c` |
| data | Hugging Face dataset `physics-from-video/morpheus-real-world` at revision `3100bb2` |
| set | single-frame conditioning: 16 experiments (non_holonomic_pendulum excluded) x {plain, enhanced} x 3 official seeds (937, 1888, 5318) = 96 videos |
| prompts | Stage-1 reads the plain English prompt. The generator reads the official prompt: plain is the repository text, enhanced is Wan2.2's own prompt extension of it. The enhanced prompts (Chinese, the output language of Wan2.2's extension) are not regenerable bit-exactly, so they ship verbatim in `scripts/eval/inputs/morpheus_enhanced.jsonl` |

The official seeds are part of the sample id (`morph_<exp>_<plain|enhanced>_<seed>`). The noise comes from
`job_seed(42, sample_id, 0)`, like every other benchmark.

### Scorer environment

1. Set `MORPHEUS_HOME` to the checkout.
2. Install it into `.venv` (Python 3.10) with its SAM 2 copy built, following its README. The SAM 2.1 hiera-large
   checkpoint goes to `checkpoints/sam2.1_hiera_large.pt` in the checkout (`checkpoints/download_ckpts.sh`).
3. Pre-fetch the depth model on a machine with network access. The scorer runs offline, and
   `launch/eval_score_morpheus.sh` stops before any work when the depth model or the SAM 2.1 checkpoint is missing
   (`SKIP_CACHE_CHECK=1` skips the check).

The reference package set is in `docs/eval_envs/morpheus.txt`.

The scorer loads its depth model with `transformers.pipeline("depth-estimation", "nielsr/depth-anything-large")`.
The launcher sets the scorer's `HF_HOME` to `MORPHEUS_HF_HOME` (default: `HF_HOME`, else the hub's default
`~/.cache/huggingface`).
Its hub cache (`$MORPHEUS_HF_HOME/hub/`) must hold `models--nielsr--depth-anything-large` with `refs/main`,
`config.json` and `model.safetensors`. The hub stores the model under the name the scorer asks for, so fetch it by
that name, for example with
`HF_HOME=$MORPHEUS_HF_HOME .venv/bin/python -c 'from transformers import pipeline; pipeline("depth-estimation", "nielsr/depth-anything-large")'`
from the checkout. On the hub this name redirects to `LiheYoung/depth-anything-large-hf`; the reference runs used
its revision `27ccb0920352c0c37b3a96441873c8d37bd52fb6`. To pin it, download that revision with `--revision` and write
the sha into `refs/main`. Morpheus loads nothing through torch.hub.

### Generate and score

```bash
export MORPHEUS_HOME=...
bash launch/eval_bench_inputs.sh morpheus
bash launch/eval_bench_prepare.sh morpheus
bash launch/eval_bench_generate.sh morpheus moplan
bash launch/eval_bench_generate.sh morpheus base
bash launch/eval_score_morpheus.sh moplan
bash launch/eval_score_morpheus.sh base
```

The launcher runs five steps:

1. It exports 1280x1024 videos into Morpheus' raw tree.
2. It resizes them with Morpheus' own function.
3. It tracks and scores them with `morpheus-track-and-score --calculate-scores` (`scripts/s2eval/morph_worker.py`,
   several workers per GPU).
4. It checks that every exported video has a score (`morph_worker.py check`: 96 videos with a
   `combined_scores.json`). The scorer can return without writing a score, for example when tracking fails, and the
   aggregation averages the scores it finds. Without this check, such a video would drop out of the means instead of
   counting 0. The launcher stops and lists the videos; running it again retries only the videos without a score.
5. It aggregates them with `scripts/bench/morph_aggregate_br.py`. Discarded videos count 0,
   Total = (Dyn + PhysInv) / 2, and the better prompt type by Total is reported.

The Morpheus model name is `MP-<label>`. It is registered in memory only and must not contain Morpheus path keywords
(`wan`, `plain`, `enhanced`, `falling`, ...).

## 6. VBench-I2V

| item | value |
|---|---|
| code | [Vchitect/VBench](https://github.com/Vchitect/VBench) at commit `fd18b3d`, `vbench2_beta_i2v` |
| set | the full suite: 1,118 entries (355 image-prompt pairs + 763 camera_motion prompts), 5 videos each, crops `7-4` |
| prompts | the suite prompt verbatim; first frames downscaled to at most 1344 px wide |

### Scorer environment

1. Set `VBENCH_HOME` to the checkout.
2. Install it with its own `.venv` (Python 3.10, torch 2.4.1; see `docs/eval_envs/vbench.txt`).
3. Download the VBench model weights into `VBENCH_CACHE_DIR`.
4. Prepare the torch.hub tree for camera_motion. VBench loads CoTracker2 with
   `torch.hub.load("facebookresearch/co-tracker", "cotracker2")`, which reads `TORCH_HOME`, not `VBENCH_CACHE_DIR`.
   The launcher sets `TORCH_HOME=${VBENCH_TORCH_HOME:-$VBENCH_CACHE_DIR/torch_home}`. Put there:
   - `hub/facebookresearch_co-tracker_main/`: a checkout of facebookresearch/co-tracker at commit
     `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`;
   - `hub/checkpoints/cotracker2.pth`: the CoTracker2 checkpoint (`facebook/cotracker`, `cotracker2.pth`).

   Without this tree torch.hub falls back to `~/.cache/torch/hub` or downloads the current co-tracker default branch,
   and the camera_motion verdicts are then no longer tied to the reference code.

The scorer runs offline from the checkout (it reads `vbench2_beta_i2v/data/crop/7-4`).

### Generate and score

```bash
export VBENCH_HOME=... VBENCH_CACHE_DIR=...        # optional: VBENCH_TORCH_HOME=... (default $VBENCH_CACHE_DIR/torch_home)
bash launch/eval_bench_inputs.sh vbench
bash launch/eval_bench_prepare.sh vbench
bash launch/eval_bench_generate.sh vbench moplan
bash launch/eval_bench_generate.sh vbench base
bash launch/eval_score_vbench.sh moplan
bash launch/eval_score_vbench.sh base
```

The scoring wrapper `scripts/bench/vbench_score_br.py` works as follows:

- It evaluates one dimension per call with `VBenchI2V.evaluate(..., mode="vbench_standard")`, on shards spread over
  the GPUs.
- It merges the per-video results.
- It aggregates Total / I2V / Quality exactly like `scripts/cal_i2v_final_score.py` (min-max normalisation,
  `DIM_WEIGHT_I2V`). A missing dimension counts 0.

## 7. Comparisons and ablations

```bash
bash launch/eval_bench_compare.sh moplan base zero
```

This writes the following files:

| file | content |
|---|---|
| `$EVAL_ROOT/RESULTS.json` | the table of every scored benchmark |
| `$EVAL_ROOT/paired/paired_pisabench_moplan_vs_base.json` | per-clip paired bootstrap |
| `paired_morpheus_*.json` | per video and per experiment cluster |
| `paired_vbench_all_*.json` | bootstrap over prompts, re-aggregated like the official script |
| `breakdown_rigid_*.json` | per task, 10 official metrics |
| `breakdown_morpheus_*.json` | per category |
| `pisa_fall_speed.json` | fall speed |

All bootstraps use 10,000 replicates with fixed seeds. In `pisa_fall_speed.json` the comparisons of all arms draw from
one generator in arm order (`launch/eval_bench_compare.sh` passes `ours`, `base`, then the extra labels in the order
given). The intervals of ours vs base therefore do not depend on the extra arms, but the intervals of an extra arm
depend on the arms listed before it. The point estimates do not depend on the order.

The zero-plan ablation runs as follows:

- Physics-IQ: `launch/eval_piq_generate.sh zero ...`
- other benchmarks: `launch/eval_bench_generate.sh <bench> zero`

`launch/eval_bench_prepare.sh` builds the zero-plan caches.

You can evaluate any other generator checkpoint under a new label. The label is a positional argument of the
launchers, and `STAGE2_CKPT` names the generator folder (with `high/` and `low/`):

```bash
STAGE2_CKPT=<dir> bash launch/eval_bench_generate.sh <bench> <label>     # PisaBench, RigidBench, Morpheus, VBench-I2V, MoProbe-H
STAGE2_CKPT=<dir> bash launch/eval_piq_generate.sh <label> op|bpp <run>  # Physics-IQ: run folder <label>-<op|bpp>-run_0<run>
bash launch/eval_score_<bench>.sh <label>                                 # pisabench, rigidbench, morpheus, vbench
bash launch/eval_piq_report.sh <label> wan22-base
```

## 8. Output layout

```
$EVAL_ROOT/
  physics_iq/{inputs,masks,stage1/<op|bpp>/flows,gen/<NAME>,work/<NAME>,results}
  <bench>/inputs/        manifest.{jsonl,csv}, ids_all.txt, NUM_SEEDS, NEGATIVE_PROMPT.txt, PROMPT_TABLE.json, SEED_MAJOR
  <bench>/masks/k16/     first-frame instance masks
  <bench>/stage1/flows/  Stage-1 exports
  <bench>/cache/         held condition caches (cache_zero: zero-plan ablation)
  <bench>/gen/<model>/   generated clips (<arm>__<sid>[__s<k>].npy + .json)
  pisabench/official/<model>/   rigidbench/{predictions,runs}/<model>/   morpheus/{raw,resized,results}/
  vbench/{videos,shards,units,results}/<model>/   paired/   RESULTS.json
```
