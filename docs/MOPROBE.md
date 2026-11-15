# MoProbe

MoProbe measures three motion failures of image-to-video models from one first frame and one or more prompts:

- **Inertia**: the model ignores the prompt.
- **Entanglement**: camera motion stands in for object motion.
- **Confabulation**: an invented cause appears.

Every measurement is object-level and runs on the generated clip only. The instance masks of the first frame are
propagated through the clip, and a background point grid gives the camera.

This page describes the probe sets, the measurement, the metrics and how to run MoProbe for MoPlan and the
same-protocol base model. The probe manifests (frame ids, prompts and set definitions) ship in `data/moprobe/`.

**What is not distributed.** The MoProbe measurement bundle is not distributed: the built items, the first-frame
instance masks, the measured references of the real continuations, the first frames and videos, and the other
models' generations. It cannot be rebuilt from this repository alone, because the repository has no builder for the
items, masks and references (the Something-Something v2 selection scripts stop at draft items). The scripts document
the method: what they read, how they measure and how they score. The manual labels of the blind Confabulation check
are per-clip verdicts on generated clips and are not distributed either (see section 5).

In the analysis scripts and their outputs, F1 = Inertia, F2 = Confabulation and F3 = camera motion (Entanglement).
These names are unrelated to the failure codes `F1`-`F9` of `motionbench/contract.py`.

## 1. Probe sets

### MoProbe-H (Inertia, Entanglement)

MoProbe-H has 414 first frames from the Something-Something v2 **validation** split. The frame ids are
`ssv2_<video id>`. Each frame belongs to one dimension and comes with a pair of prompts that ask for opposite
outcomes.

| condition | meaning |
|---|---|
| `ex` | the true prompt (what happens in the real video) |
| `cf` | the opposite prompt (same objects, opposite outcome) |
| `rex` / `rcf` | true / opposite prompt of the new-role diagnostic ("A hand hits P with I" with the roles swapped) |
| `nx` / `nt` | empty prompt (used for the open models only; MoPlan's Stage-1 always needs a caption) |

The dimensions are:

| dimension | the two prompts differ in |
|---|---|
| `relation` | two objects move apart vs closer together |
| `noun` | which of two objects is moved |
| `lr` | left vs right |
| `updown` | up vs down |
| `nothing` | an event happens vs nothing happens |
| `role`, `role2` | which object acts on which (diagnostic only, not scored) |
| `camera` | frames used only for the camera / direction analyses |

The five scored dimensions are `relation`, `noun`, `lr`, `updown` and `nothing`. Some frames were removed after an
item check showed that their opposite prompt cannot be carried out from the first frame
(`data/moprobe/rejected_frames.json`).

### MoProbe-P (Confabulation, Entanglement)

MoProbe-P has 110 first frames from five physics benchmarks, with that benchmark's own prompt (condition `pd`):

| source | frames |
|---|---|
| PisaBench (`ps`) | 40 |
| Physics-IQ (`piq`) | 20 |
| Morpheus (`mo`) | 20 |
| CausalCascade (`cc`) | 15 |
| RigidBench (`rb`) | 15 |

All of these are fixed-camera rigs. CausalCascade is the benchmark of Li et al., "A Plausible Future Is Not a Causal
Future: Can Video Models Generate How the World Unfolds?", arXiv:2610.15562 (2026). The prompts in
`moprobe_p_rows.jsonl` are the source benchmarks' own prompts, verbatim, and stay under the terms of those benchmarks
(see [LICENSE_NOTICE.md](../LICENSE_NOTICE.md)).

### Files in `data/moprobe/`

| file | content |
|---|---|
| `finding_set_B.json` | MoProbe-H: frame -> dimension (`H`), all dimensions (`dims`), available conditions (`conds`), item folder (`items_dir`), scored frames (`f1_frames`), diagnostic frames (`f1_diag_frames`), `role2` frames |
| `moprobe_h_rows.jsonl` | one row per MoProbe-H frame and condition: `frame`, `cond`, `instruction` (the prompt), `dims`, `items_dir` |
| `moprobe_p_rows.jsonl` | one row per MoProbe-P candidate frame: `source`, `frame`, `cond` (`pd`), `instruction` |
| `finding_set_A.json` | `P`: the 110 MoProbe-P frames per source; `H`: the frames of the earlier H version (read by `moplan_subsets_A.py`) |
| `rejected_frames.json` | MoProbe-H frames removed by the item check |

The letters A and B in the file and script names (`finding_set_A.json`, `finding_set_B.json`, `moplan_*_A.py`,
`miqB_inputs.py`) and in some manifest values are legacy names of two builds of the probe sets, kept so that the
outputs stay identical. `finding_set_B.json` is MoProbe-H as described on this page; `finding_set_A.json` holds the
110 MoProbe-P frames and the H frames of an earlier build.

## 2. Measurement

`scripts/motionbench/measure_clips.py` measures every clip the same way, real or generated.

- **Work frame.** The clip is mapped to the 448x256 work frame. Generated clips are 16:9 canvases. With
  `--pin-first-frame`, frame 0 is replaced by the item's stored first work frame, so every model starts from the same
  pixels.
- **Objects.** SAM 2.1 hiera-large propagates the first-frame instance masks through the clip. This gives the
  centroid, area and visibility of every object over time.
- **Camera.** CoTracker3 (`scaled_offline.pth`) tracks a background point grid. A per-frame similarity fit (scale,
  rotation, translation; `camera_deep.fit_camera`) gives the camera. Object motion is compensated by it.
- **Output.** The result is stored as a `Reading` (`<arm>__<sid>__s<seed>.measure.npz`). All scoring is a separate
  CPU step, so thresholds never trigger re-tracking.

Rules (`motionbench/metrics/`):

- **Moved.** An object moved if its camera-compensated peak displacement reaches max(16 px, 0.5 x object size).
- **Directions.** Left / right / up / down mean the peak displacement lies within 45 degrees of that side.
- **Relation.** The distance between the two objects changes by max(16 px, 0.5 x mean size).
- **Noun.** The named object moved and the other one moved less than half as far.
- **Nothing.** The prompt says what should stay still or not fall, and the rule checks that.
- **Camera.** The camera moved if, over the fitted frames, the translation reaches 8 px, the zoom 10 % or the rotation
  3 degrees. If the fit is unreliable on half of the frames or more, a witness decides instead: the median raw shift of
  the objects that are static in the real continuation. Clips that cannot be scored are reported, never counted as
  still.

  The fallback threshold is inclusive: `camera_rule.verdict` uses `lost >= FIT_LOST` with `FIT_LOST = 0.5`, so a clip
  whose fit is unreliable on exactly half of its frames goes to the witness. This can happen: the 4 s window of a
  7 fps clip has 28 frames after frame 0, and 14 unreliable frames out of 28 is exactly half. Two other gates use the
  strict form: the per-frame camera series (`camera_patterns.series`, used by Follow and the direction analyses) and
  the real-continuation fit (`moprobe_h_rules.camera_finding_analyse`) drop a clip only when fewer than half of its
  frames are fitted (`good[1:].mean() < 0.5`). At exactly half, such a clip therefore keeps its fitted series, while
  its camera verdict comes from the witness. A description that says "more than half" (strict) differs from the code
  only at this boundary; the code is what produced the reference results.

## 3. Metrics

All metrics compare MoPlan with the untrained Wan2.2-I2V-A14B under the same generation protocol (arm `wan_p448`), on
the first frames both have. 95 % intervals come from a first-frame bootstrap with 2,000 resamples and seed 0.

| metric | definition | better |
|---|---|---|
| Same (Inertia) | the outcome under the true and the opposite prompt is the same | lower |
| true / opposite carried out | the prompt's outcome happens | higher |
| dboth | both prompts carried out, minus the independence baseline p_true x p_opposite (per dimension, weighted over the five scored dimensions) | higher |
| Dir | right direction, given motion on the asked axis (`lr`, `updown`; motion >= 16 px) | higher |
| Cam (Entanglement) | camera motion on first frames whose real continuation has a still camera, true prompt | lower |
| Follow (Entanglement) | among pans >= 16 px with a moving main object, the share whose direction follows the object (cosine >= 0.5) | reported |
| Fab (Confabulation) | something absent from the first frame appears (a hand / person touching an object while it is still in its first motion, or a new / replaced object), on first frames without people, after the blind manual check | lower |
| Cam(P) | camera motion on the fixed-rig MoProbe-P frames | lower |

## 4. Running MoProbe-H for MoPlan

### The MoProbe bundle (`MOPROBE_ROOT`)

The measurement needs, per first frame, a built item. The bundle is not distributed, and this repository cannot
build it: the items come from Something-Something v2 frames (whose licence does not allow redistribution), and no
script here builds the items, masks or references. The layout the scripts read is:

```
$MOPROBE_ROOT/
  <items_dir>/items/<frame>.json       item: roles of the first-frame instances (actor / patient / hand / static), matched
                                       objects, template id, letterbox geometry of the work frame
  <items_dir>/masks/<frame>.npz        packed first-frame instance masks in the 448x256 work frame
  <items_dir>/ref/<frame>.npz          measurement of the real continuation (for the still-camera frames)
  <items_dir>/frames/<frame>_t0_work.png   first work frame (for --pin-first-frame)
  bench3/sel/gen_manifest.jsonl, gen_manifest_role.jsonl, gen_manifest_camB.jsonl, frames16x9/
                                       generation manifests + 1280x720 letterboxed first frames (read by
                                       scripts/bench/miqB_inputs.py)
```

`<items_dir>` is the folder named in `finding_set_B.json` (`items_bench`, `items_bench2`, `items_cfset`,
`items_dirbal`, `items_role2`).

`scripts/bench/miqB_inputs.py` reads all three generation manifests, in this order, and stops when one is missing.
Their rows are keyed by `<frame>-<condition>`, and the first manifest that has a key provides its prompt and first
frame. `gen_manifest_camB.jsonl` holds the camera-fixed conditions (`exc`, `cx`) of a camera diagnostic. The MoProbe-H
manifest takes only `ex`, `cf`, `rex` and `rcf`, so the rows of `gen_manifest_camB.jsonl` never reach it, but the
file must be present.

### Commands

```bash
export MOPROBE_ROOT=... SAM2_CHECKPOINT=... COTRACKER_REPO=... COTRACKER_CKPT=...
bash launch/eval_bench_inputs.sh moprobe_h            # manifest: every MoProbe-H frame x ex / cf / rex / rcf, 37 frames @ 7 fps
bash launch/eval_bench_prepare.sh moprobe_h           # masks, Stage-1 exports, condition caches (+ zero-plan caches)
bash launch/eval_bench_generate.sh moprobe_h moplan   # and: base, zero
bash launch/eval_moprobe.sh export moplan             # arm moplan; base -> wan_p448, zero -> moplan_zero
bash launch/eval_moprobe.sh export base
bash launch/eval_moprobe.sh measure                   # SAM 2.1-L + CoTracker3 -> $MOPROBE_ROOT/measure/
bash launch/eval_moprobe.sh analyze                   # -> $EVAL_ROOT/moprobe_h/analysis/
```

`analyze` runs the following scripts:

| script | what it does |
|---|---|
| `scripts/motionbench/moprobe_h_rules.py real-camera` | real-continuation camera verdict per frame; the still-camera frames for Cam |
| `scripts/motionbench/moprobe_h_paired.py` | Inertia records (`moprobe_h_rules.derive` / `record`) and per-clip camera verdicts (`cam_clip`) for MoPlan and the base model; Same / dboth / Dir overall and per dimension, Cam, Follow |
| `scripts/motionbench/moprobe_h_role2.py` | scores the `role2` items that `f1_diag_frames` lists only as `role`; prints why any (frame, dimension) has no record |
| `scripts/motionbench/moprobe_h_ablation.py` | the SFT-only generator (arm `moplan_norl`) vs MoPlan on dboth; the zero plan (arm `moplan_zero`) vs MoPlan on how far the main object moves (camera-compensated peak, share >= 8 / 16 / 20 px) |

Checks of `analyze`:

- **Missing items.** Every analysis script first checks that each MoProbe-H frame has its built item
  (`<items_dir>/items/<frame>.json` and `<items_dir>/masks/<frame>.npz`) under `MOPROBE_ROOT`, and stops with a list
  of the missing frames per item folder otherwise. A frame whose item cannot be loaded would otherwise drop out of
  every count while the run still succeeds. `ALLOW_MISSING=1 bash launch/eval_moprobe.sh analyze` (the scripts'
  `--allow-missing`) analyses without those frames and prints a warning. With a complete bundle, the outputs are the
  same as without the check.
- **Real-camera verdicts.** `$MOPROBE_ROOT/real_cam.jsonl` is kept from an earlier run only while it is current
  (`moprobe_h_rules.py real-camera --skip-if-current`). It is computed again when it covers other frames or item
  folders, when an item, mask or reference file it was computed from is newer than the file, or when it holds a
  `missing` verdict whose item and reference exist now.
- **Stage-1 exports.** The MoProbe-H reference exports came from three separate Stage-1 runs over parts of the frame
  set, while `eval_bench_inputs.sh moprobe_h` builds one manifest of all frames. Stage-1 encodes captions in batches
  of 4, so its exports are not expected to be bit-exact (see [EVALUATION.md](EVALUATION.md), "Bit-exact
  reproduction").

The SFT-only arm is the GT-SFT generator of [TRAINING_SFT.md](TRAINING_SFT.md) (`checkpoint-004000`, with `high/`
and `low/`); it is not released, so train it first. Generate it with `STAGE2_CKPT=<that folder> bash launch/eval_bench_generate.sh moprobe_h norl` and export
it with `bash launch/eval_moprobe.sh export norl` (arm `moplan_norl`).

Records of the five open models can be placed in `$MOPROBE_ROOT/context/` (`f1_records_B.jsonl`, `cam_records.jsonl`).
They only add context lines to the output.

## 5. MoProbe-P analyses

The MoProbe-P rows cannot be recomputed from this repository, even with the MoProbe bundle. The scripts below compute
the tables from per-clip intermediate records, but the steps that produce those records are not part of the release:
the generation of the MoProbe-P clips, the contact detection, the Grounding DINO agent detections and the camera
records. No released script writes `camera_bench_records.jsonl`, `contact_manifest.jsonl`, `contact/*.contact.npz` or
`agent_eval/records.jsonl`, and `launch/eval_bench_inputs.sh` builds inputs for MoProbe-H only. The MoProbe-P input
builder `scripts/bench/miq_inputs.py` expects a `clips` field in its finding set, which `data/moprobe/finding_set_A.json`
does not have. The scripts document how the MoProbe-P numbers are computed from those records.

The MoProbe-P tables come from these scripts:

- `scripts/motionbench/moplan_subsets_A.py` (`--dir` holding the per-clip records)
- the Confabulation chain:
  - `new_object_scan.py` (class-agnostic new-object scan with SAM 2 automatic masks and DINOv2 features;
    `DINOV2_MODEL`, default `facebook/dinov2-base`). The reference runs used a local copy of DINOv2-base whose Hub
    revision was not recorded, so the features of another revision can differ slightly.
  - `f2_touch_in_motion.py` (a hand touching an object while it is still in its first motion)
  - `f2_merged_summary.py` (merges both with the blind manual labels)
- `camera_deep.py` and `camera_patterns.py` for the camera records

These scripts read per-clip intermediate records (contact passes, agent detections and camera records of the
MoProbe-P clips) from `MOPROBE_ROOT`; like the bundle, these records are not distributed. Section 2 describes the
measurement behind them.

The manual labels of the blind Confabulation check are not distributed. `f2_merged_summary.py` and
`moplan_subsets_A.py` take them as explicit arguments: `--blind-key KEY.json` (label id -> clip) and
`--blind-labels DIR` (one `<label id>.json` with an `answer` per label). `moplan_subsets_A.py` skips its F2
(Confabulation) section when they are not given; F1 and F3 are computed either way.
