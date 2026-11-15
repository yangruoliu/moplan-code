# Stage-1 training data

Stage-1 trains on seven sources: three synthetic, one simulated with two variants, and three real. None of the
training data is redistributed. ImageNet, PhyCo, Physics 101 and Something-Something v2 each have their own terms.
This page lists what the trainer reads, which ID lists define the exact training set, and how each source is built
from public raw data with the scripts in this repository. It ends with the parts that cannot be reproduced exactly.

## What the trainer reads

`configs/stage1/train.yaml` expects this layout under `$MOPLAN_DATA/stage1/` (about 150 GB in total):

```
<source>/manifests/{train,val}.jsonl                 one row per sample: sample_id, scenario, split, video_path, caption, ...
<source>/targets/<sample_id>.pt                      motion target
<source>/flowdit/<sample_id>.pt                      model inputs
<source>/sam2_conditions/<sample_id>.pt              first-frame SAM 2 instance masks
<source>/metadata/prompt_table.pt                    umT5 caption embeddings
<source>/metadata/supervision_mover_ids_manifest.json   which instances are the task objects
phyco_anchor/metadata/caption_overrides_full.json    reviewed PhyCo captions (train split)
phyco_anchor/metadata/exclusions_manifest.json       PhyCo rows excluded at load time
```

`<source>` is one of `fot`, `event_static`, `phyco_anchor`, `pool_v2`, `real/p101`, `real/p101r` and `real/ssv2`.

Per sample (`motion_jepa/flow_dit_data.py`, `FlowDiTDataset`):

- `targets/<id>.pt`:
  - `flow` [7, 2, 160, 160] fp16, the cumulative displacement in pixels of the 160 grid from the first frame to each
    horizon (letterboxed frame, x first, y down);
  - `visibility`, `flow_valid`, `content_valid`;
  - `instance_id` [1, 160, 160] int16, `object_mask`, `future_mask`;
  - `horizon_seconds`, `label_source`, `caption`, `video_path`.
- `flowdit/<id>.pt`:
  - `first_frame` uint8 [3, 160, 160];
  - `image_cond_latent` [16, 1, 32, 32], the Wan VAE mode of the 256 x 256 letterboxed first frame;
  - `flow_latent`. Its first latent slot is the constant zero-motion prior latent. The released recipe
    (`train.objective: direct`, no flow-matching loss) reads only that slot.
- `sam2_conditions/<id>.pt`: SAM 2.1 first-frame instance masks (16 slots, or compact uint8 labels) and the
  `first_frame_sha256` of the flow-cache first frame. The loader checks that hash.
- `metadata/prompt_table.pt`: `{embeddings: {sha256(caption.strip()): [L, 4096] bf16}, negative_key,
  prompt_pad_length: 64}`.

## The sources

| config name | mix | train rows (scenes) | val rows | horizons (s) | label source | raw data |
|---|---|---|---|---|---|---|
| `fot` | 0.45 | 28,143 (3,000) | 1,209 | 0.1 0.2 0.4 0.8 1.2 2.4 4.0 | analytic composite motion | ImageNet ILSVRC-2012 |
| `event_static` | 0.12 | 6,564 (3,000) | 64 | as `fot` | analytic (zero motion) | same scenes as `fot` |
| `phyco_anchor` | 0.14 | 28,491 | 140 | 4k/7, k = 1..7 | RAFT flow + rendered segmentation | PhyCo Kubric, 7 scenarios |
| `pool_v2` | 0.07 | 3,000 | 148 | 4k/7 | RAFT flow + rendered segmentation | PhyCo `pool_table_force` |
| `real_p101` | 0.08 | 1,423 | 35 | 5/24 10/24 15/24 20/24 1.2 2.0 4.0 | measured motion plan | Physics 101, camera 1 |
| `real_p101r` | 0.08 | 2,584 (1,729 trials) | 64 | as `real_p101` | measured motion plan | Physics 101, other cameras |
| `real_ssv2` | 0.06 | 850 | 76 | 4k/7 | measured motion plan | SSv2 train split + Something-Else boxes |

That is 71,055 training rows in total. The trainer log line `[model] ... train=71055` confirms that every row was
found.

- FOT (text-conditioned flying objects) gives each first frame four or more captions that change which object moves
  and how. The image alone then does not determine the motion.
- `event_static` holds the "hold" captions of the same scenes, with zero motion.
- The real sources are labelled by measuring the motion of each first-frame object in the real clip (tracks plus mask
  centroids, one rigid fit per horizon). An object is a mover if it moves at least 3% of the work-frame diagonal,
  still if it stays within 1%, and unknown otherwise. Unknown pixels carry no loss.

## ID lists

`data/ids/stage1/<source>_{train,val}.tsv` lists every training and validation row of the reference run. The header
is `sample_id  scenario  scene_id`, and `scene_id` is empty where the manifest has none. The lists are in manifest
order. The order matters: the source-cycle sampler indexes rows by position, so a manifest with the same rows in a
different order produces a different data stream. Train and validation rows share no sample id and no scene.

`scripts/clean_r2/prepare_unified_stage1_data.py` assembles one source into the layout above. It keeps exactly the
listed rows in the listed order, copies their manifest lines byte for byte and links their cache files. Rows or files
that are missing are reported:

```bash
python scripts/clean_r2/prepare_unified_stage1_data.py --source pool_v2 --out "$MOPLAN_DATA/stage1" \
    --manifest POOL_TRAIN.jsonl --manifest POOL_VAL.jsonl \
    --cache-root DIR_A --cache-root DIR_B \
    --metadata prompt_table.pt=PROMPT_TABLE.pt \
    --metadata supervision_mover_ids_manifest.json=MOVER_IDS.json
```

Each `--cache-root` holds `targets/`, `flowdit/` and `sam2_conditions/` subfolders (any of them may be absent). For
every file the first root that has it wins. `--copy` copies files instead of linking them, and `--allow-missing`
writes what is available and reports the rest.

## Downloads

| what | used by | notes |
|---|---|---|
| ImageNet ILSVRC-2012: `ILSVRC/Data/CLS-LOC/{train,val}`, `LOC_val_solution.csv`, `LOC_synset_mapping.txt` | FOT | object cutouts come from validation images (localisation boxes), backgrounds from train images (train scenes) or validation images (validation scenes) |
| PhyCo Kubric renders (`rgba.mp4`, `segmentation.mp4`, `metadata.json`, scenario captions) | phyco_anchor, pool_v2 | gated dataset on Hugging Face; request access from its authors |
| Physics 101 (Wu et al. 2016) | real_p101, real_p101r | all cameras; the `spring` scenario is not used |
| Something-Something v2: videos and `labels/` (train split only) | real_ssv2 | no SSv2 validation clip is used in Stage-1 training or validation; the `real_ssv2` validation rows are held-out clips of the train split |
| Something-Else box annotations | real_ssv2 | boxes of the annotated objects |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (`vae/`, `text_encoder/`, `tokenizer/`) | all | `WAN_MODEL_DIR` |
| SAM 2.1 hiera-large (`sam2.1_hiera_large.pt`) and the `sam2` package | all | `SAM2_CHECKPOINT` |
| CoTracker3 (`scaled_offline.pth`) and a co-tracker checkout | real sources | `COTRACKER_CKPT`, `COTRACKER_REPO` |
| torchvision `raft_large` (`C_T_SKHT_V2` weights) | phyco_anchor, pool_v2 | downloaded by torchvision |
| `$MOPLAN_WEIGHTS/stage1/` prior latent | fot, event_static, real sources | the constant zero-motion prior latent (slot 0 of every flow cache; see below) |

The prior latent shipped with the weights (`stage1/prior_latent.pt`) is the first latent slot of a PhyCo flow-cache
sample. `scripts/cache_stage1_flowdit.py` (phyco_anchor, pool_v2) VAE-encodes the zero-flow first slot together with
the flow video. The other cache builders (`scripts/cache_stage1_event_inputs.py`,
`scripts/stage1_real/cache_stage1_real_inputs.py`) take slot 0 from a prior file instead (`--prior-cache` /
`--prior`): the released file or any flow-cache sample. They copy it into every cache they write and record the
sha256 of its raw fp16 bytes as `prior_sha256` in the cache provenance. For the released file that hash is
`2352391e9a1b7b257c265c45803fb074d17efc5cce76da0534725c5f45305915`.

The reference sources do not all hold these values in slot 0. `phyco_anchor`, `real_p101`, `real_p101r` and
`real_ssv2` hold exactly the released values. `fot` and `event_static` were built from another zero-motion prior
(raw-byte sha256 starting with `f6f9f57c`), and `pool_v2` holds a third one (starting with `efdf6075`). Neither of the
two is released. They differ from the released prior in 60 and 4,804 of the 16,384 values, by at most 0.002 and
0.004, i.e. by about one fp16 rounding step. The difference is negligible for training, but a rebuild of these three
sources does not reproduce their `flow_latent` bit for bit.

## Building each source

The commands below use `$BUILD` for a scratch directory, `$IMAGENET`, `$PHYCO_RAW` and so on for the downloads, and
`$PRIOR` for the prior latent file. Every Python command runs from the repository root with
`PYTHONPATH=$PWD`. Every step shards with `--shard-index i --num-shards n` where it offers them.

### FOT and event_static

The FOT data are built in four layers. Every later layer reuses the scenes, first frames, SAM 2 conditions and split
of the earlier ones.

1. **Cutouts** (GPU, SAM 2.1). These are objects of 33 movable or tabletop classes, taken from ImageNet validation
   images with one dominant localisation box and segmented with a box prompt. Merge the `cutouts_shard*.jsonl` files
   into `cutouts.jsonl`.

   ```bash
   python scripts/piq_gt_v2/flyobj_cutouts.py --imagenet "$IMAGENET" --output-dir "$BUILD/fot_cutouts" \
       --sam-checkpoint "$SAM2_CHECKPOINT"
   ```
2. **FOT v1 scenes** (CPU). These are 640 x 360 composites of 2-4 cutouts on an ImageNet background, written to
   `scenes/<scene_id>/{frame.jpg,labels.png,scene.json}`. The training sources use 3,000 train scenes
   (`fot_train_00000` ...) and 160 validation scenes.

   ```bash
   python scripts/piq_gt_v2/flyobj_compose.py scenes --output-dir "$BUILD/fot_v1" --imagenet "$IMAGENET" \
       --cutouts "$BUILD/fot_cutouts/cutouts.jsonl" --cutout-dir "$BUILD/fot_cutouts/cutouts" \
       --split train --num-scenes 3000          # then the same with --split val
   ```

   The motion recipes of each scene are then extended with the `extend`, `extend_towards`, `extend_negatives`,
   `extend_into`, `extend_absent` and `recalibrate` phases of the same script (default seed).
3. **Event curriculum v2** (CPU, then GPU caches). This layer adds analytic event variants per scene (pushes, rolls,
   falls, releases, holds, ...), paraphrases and counterfactual caption pairs.

   ```bash
   python scripts/prepare_stage1_events.py targets --source "$BUILD/fot_v1" --output "$BUILD/v2/data"
   python scripts/prepare_stage1_events.py finalize --source "$BUILD/fot_v1" --output "$BUILD/v2/data"
   python scripts/build_event_task_object_ids.py --data-root "$BUILD/v2/data" --output "$BUILD/v2/task_object_ids.json"
   ```

   Split the v2 manifests into dynamic rows (`$BUILD/v2/split/dynamic/{train,val}.jsonl`, `motion_kind` not
   `static`) and static rows (`$BUILD/v2/split/static/{train,val}.jsonl`). The static rows include the `hold`
   events and become `event_static`. Then build the v2 caches. For this step use a copy of
   `configs/stage1/train.yaml` whose `paths.manifest_dir`, `flow_cache_dir`, `target_cache_dir` and `prompt_table`
   point to `$BUILD/v2/data/{manifests,flowdit,targets,prompt_table.pt}`:

   ```bash
   python scripts/cache_stage1_event_inputs.py images  --config v2.yaml --prior-cache "$PRIOR"
   python scripts/cache_stage1_event_inputs.py prompts --config v2.yaml
   python scripts/piq_gt_v2/fot_conditions.py reps --root "$BUILD/v2/data"
   python scripts/cache_sam2_masks.py --manifest "$BUILD/v2/data/sam2_reps.jsonl" \
       --flow-cache-dir "$BUILD/v2/data/flowdit" --output-dir "$BUILD/v2/data/sam2_conditions" --compact-labels
   python scripts/piq_gt_v2/fot_conditions.py expand --root "$BUILD/v2/data"
   ```
4. **Relational and sustained events, v4.4** (CPU, then GPU prompts). v3 adds collide, knock-over, stop-short and
   topple events. Their captions name the objects instead of screen directions. v4 lengthens every decelerating
   motion so that objects keep moving through the 4 s horizon. v4.4 applies per-event contact constraints (pairs
   meet once, stop-short stops before contact, knock-over and collide reach their partner).
   `prepare_stage1_events_v44.py` imports v3 and v4 and runs their phases with these rules:

   ```bash
   V2="--source $BUILD/fot_v1 --v2-root $BUILD/v2/data --v2-data $BUILD/v2/data \
       --v2-manifests $BUILD/v2/split/dynamic --v2-static-manifests $BUILD/v2/split/static \
       --v2-task-ids $BUILD/v2/task_object_ids.json --wan-model $WAN_MODEL_DIR"
   python scripts/prepare_stage1_events_v44.py targets $V2 --output "$BUILD/v44"
   python scripts/prepare_stage1_events_v44.py merge   $V2 --output "$BUILD/v44" --combined "$BUILD/combined_v44" \
       --keep-fall 0.036 --keep-lower 0.036
   python scripts/prepare_stage1_events_v44.py prompts $V2 --output "$BUILD/v44" --combined "$BUILD/combined_v44"
   ```

   `merge` keeps the v4.4 rows and a 3.6% hash subsample of the plain v2 falls and lowers. It links the v2 caches
   into the combined root and copies same-scene flow and SAM 2 caches for the new rows. It also writes the combined
   task-object ids, and `prompts` writes the combined prompt table. If you run `targets` sharded, pass the same
   `--num-shards` to `merge`.

Then assemble the two sources:

- `fot` training rows: the 22,312 combined v4.4 training rows, followed by the 5,831 v2 `release` rows of
  `split/dynamic/train.jsonl`.
- `fot` validation rows: the 1,209 rows of `combined_v44/manifests_relational/val.jsonl`.
- `event_static`: the v2 static rows.

```bash
python scripts/clean_r2/prepare_unified_stage1_data.py --source fot --out "$MOPLAN_DATA/stage1" \
    --manifest "$BUILD/combined_v44/manifests/train.jsonl" --manifest "$BUILD/v2/split/dynamic/train.jsonl" \
    --manifest "$BUILD/combined_v44/manifests_relational/val.jsonl" --cache-root "$BUILD/combined_v44" \
    --metadata prompt_table.pt="$BUILD/combined_v44/prompt_table.pt" \
    --metadata supervision_mover_ids_manifest.json="$BUILD/combined_v44/task_object_ids.json"
python scripts/clean_r2/prepare_unified_stage1_data.py --source event_static --out "$MOPLAN_DATA/stage1" \
    --manifest "$BUILD/v2/split/static/train.jsonl" --manifest "$BUILD/v2/split/static/val.jsonl" \
    --cache-root "$BUILD/v2/data" --metadata prompt_table.pt="$BUILD/v2/data/prompt_table.pt" \
    --metadata supervision_mover_ids_manifest.json="$BUILD/v2/task_object_ids.json"
```

### phyco_anchor

1. Download and extract PhyCo to `$PHYCO/extracted/<scenario>/<date>/<hash>/`. Keep the raw download, which holds
   `<scenario>/common_caption_cosmos.txt`, at `$PHYCO_RAW`.
2. Targets (GPU, RAFT). There are 14 anchors over 4 s. The trainer uses every second one
   (`temporal_subsample_indices`), so the horizons are 4k/7 s:

   ```bash
   python scripts/cache_phyco_targets.py --root "$PHYCO/extracted" --caption-root "$PHYCO_RAW" \
       --output "$BUILD/phyco/targets" --horizons 14 --span-seconds 4.0
   ```
3. Manifests (hash split: buckets 0-1 val, 2-3 sealed test, rest train) and the exclusions appendix:

   ```bash
   python scripts/prepare_phyco_manifests.py --targets-dir "$BUILD/phyco/targets" --raw-root "$PHYCO_RAW" \
       --output-dir "$BUILD/phyco"
   ```
4. Causal captions and task objects. The reviewed captions are deterministic, Physics-IQ-style three-sentence
   captions built from the simulator's causal metadata and the first frame only. The builder writes the train
   manifest, the review file and `phyco_causal_v10_overrides.json`, which is bound to the SHA256 of that manifest:

   ```bash
   python scripts/build_phyco_causal_captions.py --candidate-manifest CANDIDATES.jsonl --metadata METADATA.jsonl \
       --output-dir "$BUILD/phyco/captions"
   python scripts/build_phyco_mover_ids.py --candidate-manifest CANDIDATES.jsonl --metadata METADATA.jsonl \
       --output "$BUILD/phyco/mover_ids.json"
   ```
5. Flow caches with `scripts/cache_stage1_flowdit.py --from-targets`, SAM 2 conditions with
   `scripts/cache_sam2_masks.py` (16 slots, 160 grid), and the prompt table with
   `scripts/cache_stage1_flowdit_prompts.py --source phyco_anchor`. The prompt table must hold the reviewed
   captions and the negative key. With reviewed overrides the script requires an explicit new `--output`, and
   `--base-prompt-table` adds only the missing caption keys to an existing table.

The validation rows are 20 per scenario, and the ID list defines them. Assemble the source with the assembler, using
the caption-builder manifest for the train rows so that the override binding stays valid. Pass the overrides as
`--metadata caption_overrides_full.json=...` and keep the review JSONL beside them: the override file refers to it by
a relative name. Pass the exclusions with `--metadata exclusions_manifest.json=...`.

### pool_v2

This is the PhyCo `pool_table_force` scenario with captions that name the struck ball by its rendered colour and give
its strike direction or target ball. The captions do not describe the outcome.

```bash
python scripts/piq_gt_v2/pool_ball_colors.py --manifest-dir PHYCO_MANIFESTS --output "$BUILD/pool_dir/ball_colors.json"
python scripts/piq_gt_v2/pool_direction_prepare.py --manifests PHYCO_MANIFESTS --targets PHYCO_TARGETS \
    --flowdit PHYCO_FLOWDIT --output-dir "$BUILD/pool_dir"
```

`PHYCO_MANIFESTS` is a PhyCo manifest directory that includes `pool_table_force` clips, with `train.jsonl`,
`val.jsonl` and `test_core.jsonl`. `PHYCO_TARGETS` and `PHYCO_FLOWDIT` are the matching target and flow caches. The
3,000 training rows are a fixed subset of the prepared pool rows, and `pool_v2_train.tsv` defines them. Its
targets come from `scripts/cache_phyco_targets.py` with 7 horizons at 4k/7 s, and its flow caches from
`scripts/cache_stage1_flowdit.py`. The SAM 2 conditions come from `scripts/cache_sam2_masks.py`, and the prompt
table from `scripts/cache_stage1_flowdit_prompts.py`.

### Real sources: real_p101, real_p101r, real_ssv2

The real sources start from the measured motion plans of the Stage-2 data pipeline. For each clip that pipeline
produces 7-fps clips with first frames, first-frame SAM 2 masks (16 slots after `scripts/filter_sam2_masks_v2.py`)
and dense CoTracker3 tracks (`real_tracks/<id>.npz`). It also produces the Stage-2 cache with `plans_real/<id>.npz`
and the mask-centroid readings (`scripts/control/track_real.py`, `scripts/s2sft/v2/cache_stage2_cond_v2_long.py`).
For SSv2 the clips come from the train candidates of `scripts/stage1_real/ssv2_train_candidates.py` and
`ssv2_train_select.py`, and `scripts/stage1_real/ssv2_supervise.py` marks the instances inside non-hand
Something-Else boxes as supervisable.

From there each source has a CPU stage and a GPU stage. The CPU stage writes targets, conditions, manifests, mover
ids and captions. The GPU stage writes the flow cache and the prompt table. Below, `$S2` stands for the root of that
Stage-2 data build (its `cache/` and `data/` folders, see [DATA_STAGE2.md](DATA_STAGE2.md)).

```bash
# clip timing from the Stage-2 cache
python scripts/stage1_real/stage2_cache_meta.py --cache "$S2/cache/p101_k16" --ids IDS_TRAIN IDS_VAL --out p101_meta.jsonl

# Physics 101, camera 1
python scripts/stage1_real/build_stage1_real_v3.py --source-name p101 --manifest P101_MANIFEST.jsonl \
    --plans "$S2/cache/p101_k16/plans_real" --sam2 P101_SAM2 --tracks P101_TRACKS --meta p101_meta.jsonl \
    --train-ids IDS_TRAIN --val-ids IDS_VAL --horizons query --out "$MOPLAN_DATA/stage1/real/p101" \
    --scene-prefix p101: --scene-key trial_id --source-tag physics101_cam1_measured_plan_v3 \
    --static-key scenario --static-values phys101/ramp,phys101/multi --exclude-scenarios phys101/spring \
    --caption-source p101_template_caption_v1 --border-actor-unknown

# Physics 101, other cameras: export the clip metadata and centroid readings, then build with the reconcile pass
python scripts/stage1_real/p101r_export.py --cache "$S2/cache/p101r_k16" --manifest P101_REST_MANIFEST.jsonl \
    --data P101_DATA --dev96-trials DEV96_TRIALS --out "$BUILD/p101r"
python scripts/stage1_real/p101r_labels.py --cache "$S2/cache/p101r_k16" --meta "$BUILD/p101r/meta.jsonl" \
    --out "$BUILD/p101r/labels"
python scripts/stage1_real/build_stage1_real_v3.py --source-name p101r ... --horizons query \
    --reconcile-labels "$BUILD/p101r/labels" --reconcile-scenes "$S2/cache/p101r_k16/scenes" \
    --source-tag physics101_othercam_measured_plan_v3 --border-actor-unknown ...   # other flags as for p101

# SSv2 train
python scripts/stage1_real/build_stage1_real_v3.py --source-name ssv2 --manifest SSV2_MANIFEST.jsonl \
    --plans "$S2/cache/ssv2_k16/plans_real" --sam2 SSV2_SAM2 --tracks SSV2_TRACKS --meta ssv2_meta.jsonl \
    --train-ids IDS_TRAIN --val-ids IDS_HOLD --horizons phyco --annotated-only --scenario-prefix ssv2/ \
    --scenario-key family --out "$MOPLAN_DATA/stage1/real/ssv2" --scene-prefix ssv2: --scene-key source_id \
    --source-tag ssv2_train_strict_measured_plan_v3 --caption-source ssv2_template_caption_v1 \
    --static-key family --static-values static

# GPU stage for each root: prompt table and flow cache
python scripts/stage1_real/cache_stage1_real_inputs.py --config configs/stage1/train.yaml \
    --root "$MOPLAN_DATA/stage1/real/p101" "$MOPLAN_DATA/stage1/real/p101r" "$MOPLAN_DATA/stage1/real/ssv2" \
    --prior "$PRIOR" --phase both
```

For p101r, the `meta.jsonl` rows written by `p101r_export.py` carry the manifest row under `row`. Write those rows to
a manifest, write `meta.jsonl` without them, and use the train and val ids of the export as `--train-ids` and
`--val-ids`.

Check a built real source against the reference PhyCo source with the trainer's own loader:

```bash
python scripts/stage1_real/make_real_config.py --real-root "$MOPLAN_DATA/stage1/real" --out-sources real_sources.json
python scripts/stage1_real/audit_stage1_real.py --config configs/stage1/train.yaml --sources real_sources.json --out audit.json
python scripts/stage1_real/overlay_stage1_real.py --root "$MOPLAN_DATA/stage1/real/p101" --ids ID1 ID2 --out overlays/
python scripts/stage1_real/check_landing.py --root "$MOPLAN_DATA/stage1/real/p101" --scenes "$S2/cache/p101_k16/scenes_v5" --out landing.json
```

## Checking a data root

```bash
python scripts/stage1_smoke.py --data-root "$MOPLAN_DATA" --wan-model-dir "$WAN_MODEL_DIR" --out /tmp/s1 --dry-run
```

This prints the manifest row counts of every source and every missing input. Compare the counts with the table above
and the ID lists. A full training run reports `train=71055` in its first log lines.

## Known gaps

- **No redistribution.** The training data are derived from ImageNet (non-commercial research terms, no image
  redistribution), PhyCo (gated), Physics 101 (research use) and SSv2 (licence with registration, no
  redistribution). Only the ID lists and the scripts are released. You need your own copies of the raw data.
- **FOT regeneration is not verified bit-exact.** The FOT v1 scene phase seeds its random generator with
  `seed + 1000 * shard_index`, and the scene and extension phases were run with shard counts that were not recorded.
  The v4.4 layer was built on top of those scenes. Regenerated FOT and event_static scenes will have the same ids but
  can differ in content from the reference data. The split into v2 dynamic and static manifests is described above
  but was not a scripted step.
- **PhyCo inputs.** The causal-metadata JSONL (`sample_id`, `object_data`, `causal_metadata`, `metadata_sha256` of
  each clip's `metadata.json`) and the candidate manifest with a `category` field are read by the caption and
  mover-id builders, but the helper that produced them is not part of this release. The exact SAM 2 and flow-cache
  invocations for PhyCo and the selection of the 140 validation rows and the 3,000 pool rows are not scripted here.
  The ID lists define the result.
- **Real-clip labels depend on an older Stage-1 model.** `scripts/control/track_real.py` builds the tracked object
  set of every real clip from a frozen Stage-1 export (its plan objects and SAM 2 instances), and the measured plans
  are derived from those tracks. That earlier Stage-1 checkpoint is not released. The scripts can rebuild real
  sources with any Stage-1 export, but the labels are then not identical to the reference ones. Some intermediate
  inputs of the reference build (Physics 101 first frames and SAM 2 exports) are also no longer available.
- **SSv2 selection uses the MoProbe SSv2 helpers.** `ssv2_train_select.py` imports the camera, box and template
  helpers of `scripts/motionbench/` (`ssv2_select.py`, `ssv2_instructions.py`, `ssv2_templates.json`). Their inputs
  come from `scripts/motionbench/ssv2_boxes.py` (Something-Else boxes; needs `pip install ijson`, which is not in
  `requirements.txt`) and `scripts/motionbench/ssv2_camera_prefilter.py` (per-clip camera statistics). The 7-fps
  clips are cut by `scripts/data_real/ssv2_clips.py` (SSv2) and `scripts/data_real/p101_clips_v2.py` +
  `scripts/data_real/p101_splits.py` (Physics 101). The Physics 101 cutter reads a per-trial catalog
  (`sample_id`, `scenario`, `caption`, `object`, `setting`, ... and the `video_path` / `kinect_rgb_path` /
  `side_video_path` of each camera) whose builder is not part of this release.
- **Licences of the tools.** CoTracker3 is CC BY-NC 4.0. SAM 2 and Wan2.2 are Apache-2.0.
