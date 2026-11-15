# Stage-2 GT-SFT data

The Stage-2 GT-condition SFT trains the Wan2.2-I2V-A14B generator on real continuations of three sources, each clip
paired with the ground-truth motion plan of that clip:

| pool | source | plan |
|---|---|---|
| `p_fall`, `p_ramp`, `p_spring`, `p_multi`, `p_liquid` | Physics 101, three camera views | measured on the clip (CoTracker3 point tracks + SAM 2.1 masks) |
| `ssv2` | Something-Something v2, official train split | measured on the clip |
| `phyco` | PhyCo Kubric renders | simulator ground truth (rendered flow and instance ids) |

The clip lists are frozen in `data/ids/stage2_sft/` (see its README). This page describes the payload format, the
directory layout the trainer reads, how to build the payloads, and what cannot be rebuilt exactly.

The payloads are not distributed with this repository: they are derived from third-party videos under their own
licences (see "Sources and licences").

## What the trainer reads

### Directory layout

`MOPLAN_SFT_DATA` (passed to the trainer as `--data-root`) holds one `.pt` payload per clip:

```
$MOPLAN_SFT_DATA/
  pools/p101/<sample_id>.pt     Physics 101 training clips (all five scenario pools read this folder)
  pools/ssv2/<sample_id>.pt     SSv2 training clips
  pools/phyco/<sample_id>.pt    PhyCo training clips
  val/p101/<sample_id>.pt       validation clips
  val/phyco/<sample_id>.pt
  val/ssv2/<sample_id>.pt
```

A pool is the sorted intersection of the folder's `*.pt` files with its id list, so a folder may hold more payloads
than the list (for example a full cache), and the entries may be symlinks. Every listed clip must be present:
a missing payload changes the pool and therefore the sampling plan. `make_pools_v2.py` (step 5 below) creates this
layout as symlinks to the built caches and checks it.

The same layout is the replay data of the RAFT round (`docs/TRAINING_RAFT.md`).

### Payload contract (`stage2_cond_v2`)

One `torch.save` dict per clip, about 3.5 MB (about 75 GB for all pools). Keys the trainer reads:

| key | shape / type | meaning |
|---|---|---|
| `cache_format_version` | `"stage2_cond_v2"` | contract version |
| `condition_channels_v2` | list | `known, availability, selector, raw_flow_x, raw_flow_y, motion_prob, source_mask` |
| `video_latent` | `[16, T, 32, 56]` fp16 | Wan2.2 VAE latent (mode) of the clip at 448x256, 7 fps, T <= 11 |
| `image_cond_latent` | `[16, T, 32, 56]` fp16 | VAE latent of the first frame followed by zero frames |
| `cond_real` | `[27, T, 32, 56]` fp16 | ground-truth motion condition: 20 rendered channels + the 7 extra channels |
| `fm_weight` | `[T, 32, 56]` fp16 | tube weights of the flow-matching loss (moving objects 3, static objects 1.5, else 1; mean 1) |
| `prompt_embeds` | `[L, 4096]` | umT5 embedding of the caption (padded to 512 tokens by the trainer) |
| `frames`, `frames_real` | int | frames in the window (4T-3) and how many of them are real footage |

A clip whose scene has come to rest before its end is extended with its last real frame to the full 41-frame window
(`frames_real < frames`), and its plan is held. Other keys (`cond_null`, `cond_zero`, `labels_real`,
`object_masks_latent`, plan summaries, `video_path`, `first_frame_path`, `prompt`) are written for analysis and for the
RAFT preparation. `video_path` and `first_frame_path` hold absolute paths of the machine that built the cache, because
the RAFT preparation opens the files by these paths (the data manifests hold the same paths for the same reason).
Other recorded inputs (checkpoints, manifests, priors, the PhyCo `target_path`, text-encoder folders and the sources
named in receipts and summaries) are given as file or folder names, with sha256 where the record has one.

## Sources and licences

| item | where | terms |
|---|---|---|
| Physics 101 | the Physics 101 dataset (Wu et al., 2016) | research use; see the dataset page |
| Something-Something v2 | the official release (train split videos and labels) | its licence does not allow redistribution of the videos; the payloads are derived from them |
| PhyCo Kubric renders | the PhyCo dataset (gated on Hugging Face; request access) | see its dataset card |
| Wan2.2-I2V-A14B-Diffusers | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (VAE and umT5 text encoder are used to build) | Apache-2.0 |
| CoTracker3 | `facebookresearch/co-tracker` checkout + `scaled_offline.pth` | CC BY-NC 4.0 |
| SAM 2.1 | `sam2.1_hiera_large.pt` and the `sam2` package | Apache-2.0 |

The reference build used the Wan2.2 Diffusers snapshot `596658fd9ca6b7b71d5057529bbf319ecbc61d74` and the co-tracker
commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`.

## Building the payloads

Environment for all steps (from the repository root):

```bash
export PYTHONPATH=$PWD
export WAN_MODEL_DIR=/path/to/Wan2.2-I2V-A14B-Diffusers
export COTRACKER_REPO=/path/to/co-tracker COTRACKER_CKPT=/path/to/scaled_offline.pth   # real clips only
export SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt                                  # real clips only
BUILD=/path/to/sft_build          # caches and work folders
```

### 1. Inputs per source

**PhyCo.** For every clip of `train/phyco.txt` and `val/ids_phyco.txt`:
- the render `rgba.mp4` (24 fps, 768x432; read letterboxed into 448x256 at 7 fps), and
- its Stage-1 ground-truth target `<sample_id>.pt` (`flow` per horizon, `instance_id`, `horizon_seconds`), built by
  `scripts/cache_phyco_targets.py` (see `docs/DATA_STAGE1.md`).

**Physics 101 and SSv2 (real clips).** For every clip:
- a video (`video_path`; any frame rate, frame k is the source frame nearest to k / 7 s), a lossless PNG of its first
  frame (`first_frame_path`) and a caption. The reference clips were:
  - Physics 101: per trial and view, 1280x720 at 7 fps from the native first frame, at most 37 frames, plus a PNG of
    the native first frame. Views: `cam1` = Camera_1, `kin` = Kinect_RGB, `top` = Camera_2 (ramp and multi only).
  - SSv2: 7 fps in real time from a selected start frame, at most 37 frames, the 240-pixel-high frame letterboxed into a
    448x256 canvas (bands filled from the first frame's blurred border), caption = the filled-in SSv2 label.

  The cutters are `scripts/data_real/p101_clips_v2.py` (then `scripts/data_real/p101_splits.py` for the by-object
  split) and `scripts/data_real/ssv2_clips.py` (from the selection of `scripts/stage1_real/ssv2_train_select.py`).
- `plans/<sample_id>.npz` (the object plan) and `real_tracks/<sample_id>.npz` (CoTracker3 tracks), in one folder per
  source. They come from the real-clip chain: SAM 2.1 instance proposals (`scripts/cache_sam2_masks.py`, then
  `scripts/filter_sam2_masks_v2.py --max-kept 16 --union-cap 1.0`), a Stage-1 export that fixes the object set
  (`scripts/infer_stage1_physicsiq.py`), and `scripts/control/track_real.py --manifest M.jsonl --exports <exports>/flows
  --out <inputs> --raw-masks <sam2 proposals>`, which writes `plans/` and `real_tracks/` under `--out`.

### 2. Manifests

```bash
python scripts/s2sft/v2/make_manifests.py --ids-dir data/ids/stage2_sft --out $BUILD/manifests \
  --phyco-targets /path/to/phyco_targets --phyco-stage1-manifest /path/to/phyco_manifest.jsonl \
  --phyco-video-root /path/to/phyco_kubric/extracted \
  --p101-manifest /path/to/p101/manifest.jsonl --p101-inputs /path/to/p101_inputs \
  --ssv2-manifest /path/to/ssv2/manifest.jsonl --ssv2-inputs /path/to/ssv2_inputs
```

This writes `phyco.jsonl`, `phyco_val.jsonl`, `phyco_missing.jsonl`, `p101.jsonl`, `ssv2.jsonl` and `MANIFESTS.json`
(counts and the clips whose inputs are missing). The input rows need `sample_id, video_path, caption` (PhyCo,
optionally `scenario`) or `sample_id, video_path, first_frame_path, caption` (real clips). Optional:
`--phyco-candidates` (a table checked before the Stage-1 manifests) and `--phyco-v1-payloads DIR` (29-frame payloads
under `DIR/pools/phyco` and `DIR/val/phyco`; their `prompt` then replaces the caption, as in the reference build).

### 3. GPU builds

Each builder skips finished clips, so it can be restarted. Parallelise with `--shard-index i --num-shards n`, or start
any number of processes with `--claim` (each takes the next unclaimed clip).

```bash
# PhyCo (loads the VAE and the umT5 text encoder)
python scripts/s2sft/v2/build_phyco_gt_cache_v2.py --manifest $BUILD/manifests/phyco.jsonl \
  --output $BUILD/cache/phyco --frames 41
python scripts/s2sft/v2/build_phyco_gt_cache_v2.py --manifest $BUILD/manifests/phyco_val.jsonl \
  --output $BUILD/cache/phyco_val --frames 41

# real clips (VAE + umT5 + CoTracker3 + SAM 2.1); the raw build goes to a work folder
python scripts/s2sft/v2/cache_stage2_cond_v2_long.py --manifest $BUILD/manifests/p101.jsonl \
  --data /path/to/p101_inputs --output $BUILD/work/p101 --max-frames 41 --pad-to 41 --no-pred
python scripts/s2sft/v2/cache_stage2_cond_v2_long.py --manifest $BUILD/manifests/ssv2.jsonl \
  --data /path/to/ssv2_inputs --output $BUILD/work/ssv2 --max-frames 41 --pad-to 41 --no-pred
```

`--no-pred` skips the Stage-1 prediction (`cond_pred`), which the GT SFT does not read; the objects still come from
`plans/<sample_id>.npz`.

### 4. Label chain of the real clips (CPU)

The measured plans are checked against the mask readings and re-rendered. Run per source (`p101`, `ssv2`); every step
works in place on the work folder and accepts `--shard-index i --num-shards n` for parallel processes. The reference
build ran these steps with one thread per process (`OMP_NUM_THREADS=1`, no GPU).

```bash
S=p101; WK=$BUILD/work/$S; FIELDS=$BUILD/work/${S}_fields; OUT=$BUILD/cache/$S
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
python scripts/control/relabel_v2.py --cache $WK
cp -r $WK/plans_real $WK/plans_real_pre_reconcile
python scripts/control/reconcile_plans.py --cache $WK
python scripts/control/relabel_v2.py --cache $WK
python scripts/control/rerender_real.py --cache $WK --out $FIELDS
python scripts/s2sft/v2/apply_fields.py --cache $WK --fields $FIELDS --out $OUT --workers 16
cp -r $WK/plans_real $WK/plans_real_pre_reconcile $WK/scenes $OUT/
```

`rerender_real.py` renders `cond_real`, `fm_weight`, `known_frames_real` and `labels_real` again from the reconciled
plans; `apply_fields.py` writes the final payloads (`plan_source = plans_real_reconciled_v1`) into `$OUT`.

### 5. Pools

```bash
python scripts/s2sft/v2/make_pools_v2.py --ids-dir data/ids/stage2_sft/train --val-ids-dir data/ids/stage2_sft/val \
  --exclusions data/ids/stage2_sft/exclusion_lists.json --cache-dir $BUILD/cache --fallback-dir $BUILD/fallback \
  --out-dir $MOPLAN_SFT_DATA --run $MOPLAN_WORK_DIR/runs/stage2_sft --validate --workers 32
```

It links every listed clip into `$MOPLAN_SFT_DATA/{pools,val}/`, takes a clip without a 41-frame payload from
`$BUILD/fallback/pools/<pool>/` or `$BUILD/fallback/val/<name>/` (the 29-frame payloads, see "Gaps"), checks every
training id against the exclusion lists, recomputes the mix from the list sizes and writes
`<run>/OVERRIDES.json` and `<run>/POOLS_RECEIPT.json`. `--validate` loads every payload, drops nonconforming ones and
reports the latent-frame statistics. Without `--validate` it stops if a payload is missing.

If the receipt lists dropped payloads or the mix differs from `configs/stage2/sft_overrides.json`, the pools differ
from the reference run. With complete pools, `launch/train_sft.sh` can use either the written `OVERRIDES.json` or the
released `configs/stage2/sft_overrides.json` with `MOPLAN_SFT_DATA`; both describe the same run.

## Gaps: what cannot be rebuilt exactly

- **Real-clip plans.** The object sets of the Physics 101 and SSv2 plans (`plans/`, and with them `cond_real` and the
  tube weights) came from an earlier Stage-1 checkpoint that is not released. A rebuild with the released Stage-1
  checkpoint gives other object sets and therefore other payloads.
- **Clip preparation.** The Physics 101 cutter reads a per-trial catalog of the raw dataset (one row per trial with
  the video path of each camera); the catalog builder is not part of this release. The 37-frame cap matters: the
  builder takes the largest window the clip provides. Re-encoded clips depend on the ffmpeg / x264 build.
- **PhyCo clips without renders.** 1075 training clips and 8 validation clips
  (`data/ids/stage2_sft/fallback_29frame/`) had no raw render available to the 41-frame build; the reference run used
  their 29-frame payloads (8 latent frames). `build_phyco_gt_cache_v2.py --frames 29 --no-pad` builds a clip at that
  length from its render and target, if you have them.
- **PhyCo prompts.** The reference payloads took the prompt of the earlier 29-frame payloads (`--phyco-v1-payloads`).
  Without them the manifest caption is used, which can differ.
- **Decoding.** Videos are decoded with OpenCV / ffmpeg; a different decoder build can change pixel values slightly
  and with them the latents.
