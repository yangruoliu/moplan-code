# Stage-2 RAFT round

The released generator is the GT-SFT generator after one light round of reward-ranked fine-tuning (RAFT). The
round has four steps, and `launch/raft.sh` runs them on one node:

1. **sample**: for every training condition, generate G = 8 candidate videos with the GT-SFT generator under the
   cached Stage-1 condition (arm `pred`).
2. **score**: measure each candidate with the same trackers that measured the real clip, and compute the physics
   reward v5 against the real continuation.
3. **pick**: per condition, keep the best candidate when the group is complete and the rewards spread by at least
   0.2.
4. **train**: fine-tune each expert for 300 optimizer steps from the GT-SFT checkpoint. Half of the micro-steps use
   flow matching on a picked candidate. The other half replay the GT-SFT data and mix.

The conditions are 1,024 training clips in four pools of 256: `phyco` and `phycoB` (PhyCo renders) and `p101` and
`p101B` (Physics 101, camera 1). Their id lists are in `data/ids/raft/`. None of them is an evaluation clip,
because the reward needs the real continuation of the clip.

**The starting point is not distributed.** The round starts from the GT-SFT generator, `checkpoint-004000` of each
expert of the GT-SFT run. The released weights hold only the Stage-1 planner and the final generator, not this
checkpoint. Train it first with `launch/train_sft.sh` ([TRAINING_SFT.md](TRAINING_SFT.md)), once per expert. The
launcher keeps the 8000-step schedule, stops at step 4000 and links the result as `$RUN_DIR/sft_step004000/{high,low}`,
which is the `MOPLAN_SFT_CHECKPOINT` below. The reference run took about 5 hours per expert on 8 H100 GPUs. A
retrained GT-SFT generator is close to the one the released round started from, but not bitwise equal, so a round
started from it does not reproduce the released candidates, rewards and picks exactly (see "Reproducibility").

## Requirements

- One node with H100 80 GB GPUs. Sampling loads one Wan2.2 expert in bf16 per process (about 48 GB). Training needs
  about 41 GiB per expert and runs one expert per GPU. With two or more GPUs the experts train in parallel. Sampling
  and scoring are sharded over all GPUs.
- The pinned Python environment of this repository, plus the `sam2` package (SAM 2.1) and a CoTracker3 checkout.
- Weights:
  - `WAN_MODEL_DIR`: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`.
  - `MOPLAN_SFT_CHECKPOINT`: the GT-SFT generator, i.e. `checkpoint-004000` of each expert of the GT-SFT run. It is
    not distributed; train it with `launch/train_sft.sh` (see above). Lay it out as `high/` and `low/`, each holding
    `lora_params.pt` and `motion_adapter.pt`; `$RUN_DIR/sft_step004000/` of the GT-SFT run has this layout.
  - `COTRACKER_REPO` (a facebookresearch/co-tracker checkout) and `COTRACKER_CKPT` (`scaled_offline.pth`).
  - `SAM2_CHECKPOINT` (`sam2.1_hiera_large.pt`).
- Data:
  - `MOPLAN_RAFT_DATA`: the condition caches (see "Preparing the condition caches" below).

    ```
    cache_phyco/<sample_id>.pt          stage2_cond_v2 payloads with the held Stage-1 condition cond_pred
    cache_phyco/scenes_v5/<sample_id>.npz   the real side of the reward (rescene_v5.py)
    cache_phycoB/  cache_p101/  cache_p101B/   the same
    manifest_<pool>.jsonl or .csv       optional: sample_id, first_frame_path (overrides the path stored in the cache)
    ```

    The first frame must be the same image file that the cache was built from: the pipeline re-encodes it.
  - `MOPLAN_SFT_DATA`: the GT-SFT payloads, `pools/{p101,ssv2,phyco}/<sample_id>.pt`. These are the same pools as
    in the GT-SFT stage, filtered by the id lists in `data/ids/stage2_sft/train/`.

## Running

```bash
export WAN_MODEL_DIR=/path/to/Wan2.2-I2V-A14B-Diffusers
export MOPLAN_WORK_DIR=/path/to/work                      # default: <repo>/work
export MOPLAN_SFT_CHECKPOINT=$MOPLAN_WORK_DIR/runs/stage2_sft/sft_step004000   # high/ low/, from launch/train_sft.sh
export MOPLAN_RAFT_DATA=/path/to/raft_conditions
export MOPLAN_SFT_DATA=/path/to/sft_data                   # pools/
export COTRACKER_REPO=/path/to/co-tracker COTRACKER_CKPT=/path/to/scaled_offline.pth
export SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt

launch/raft.sh              # all four steps
launch/raft.sh sample       # or one step at a time: sample, score, pick, train
```

Every step skips finished work, so you can rerun a step after an interruption. Sampling skips finished videos and
staged latents. Scoring skips scored videos. Picking skips pools that have a selection receipt. Training resumes
each expert from its latest checkpoint and skips an expert that has a `done.json`.

The header of `launch/raft.sh` lists every setting: `POOLS`, `GPUS`, `GROUP`, `MARGIN`, `SEED`, `STEPS`, `REPLAY_P`,
`EXPERTS`, and the smoke-test switches `LIMIT`, `SAMPLE_STEPS`, `REPLAY_POOLS`, `REPLAY_MIX` and `EXTRA_OVERRIDES`.
The defaults are the released round. Despite its name, `REPLAY_P` (default 0.5) is the probability that a micro-step
trains on a picked candidate (the trainer's `dpo_prob`); the other micro-steps are GT-SFT replay. `REPLAY_P=0.8`
therefore means 80 % picked-candidate steps and 20 % replay.

Outputs, under `$MOPLAN_WORK_DIR/raft/`:

```
raft.yaml                      configs/stage2/raft.yaml with paths.wan_model filled in
ids/<pool>.txt                 the conditions of each pool
panels/<pool>/pred__<sid>[__s<k>].npy   candidate k of clip sid, uint8 [F, 256, 448, 3] (k = 0 has no suffix)
panels/<pool>/*.json           sampler settings of each candidate; *.reward_v5.json its reward; *.measure.npz the readings
panels/<pool>/table.txt        per-pool reward summary (score_generated.py table)
winners/<pool>/<sid>.pt        the picked candidate as a VAE latent, plus pairs.jsonl and selection_receipt.json
winners_all/                   links to every picked candidate (the trainer's --dpo-pairs)
train/OVERRIDES.json           the resolved training recipe (configs/stage2/raft_overrides.json)
train/{high,low}/              checkpoint-000100/200/300, metrics.jsonl, steps.jsonl, resolved_config.yaml, done.json
final/{high,low}               links to checkpoint-000300: the generator
logs/                          one log per process
```

Use `final/high` and `final/low` as `--high-checkpoint` and `--low-checkpoint` of `scripts/infer_stage2_cond_v2.py`.

## Sampling

Sampling calls `scripts/infer_stage2_cond_v2.py --arms pred --num-seeds 8` with the `embed`, `high` and `low`
phases. It uses the Wan2.2 diffusers pipeline with its own UniPC scheduler, 30 steps, text guidance 5.0, the default
Wan2.2 negative prompt, 448x256 at 7 fps, and the cached clip length. Clips keep their cached length: most
conditions have 41 frames, short Physics 101 clips keep their real length, and some PhyCo clips have 29 frames.

The seed of candidate k of clip `sid` is `42 + crc32(sid) % 100000 + 1000003 * k` (`job_seed`). It depends neither
on the GPU count nor on the order of the jobs. Renaming a clip changes all of its seeds.

## Reward (v5)

`scripts/control/score_generated.py track` measures every candidate with the instrument that measured the real clip
(`scripts/control/measure.py`). Frames are upscaled 2x. SAM 2.1 propagates the first-frame object masks, and an
object's displacement is the motion of its mask centroid. CoTracker3 tracks the first-frame points of the scene: the
object points and a dense grid of free background cells, kept where the visibility score is above 0.5. All
displacements are taken relative to the camera motion, which is the median motion of the background. The scene
(`scenes_v5/<sid>.npz`) holds the same readings of the real clip at every frame. The reward uses up to 12 of those
frames as evaluation times, cut to the length of the candidate.

At an evaluation time, an object really moves when its real displacement is at least 3 % of the frame diagonal. It is
still when its displacement is at most 1 %. In words:

- **Moving objects.** At every evaluation time, the object's displacement in the candidate is compared with its real
  displacement:
  - the right direction at the right length scores 1;
  - the score falls with the angle and by half a point per octave of length error;
  - an object that does not move scores -0.5, and one that moves backwards scores -1;
  - an object that the trackers lose in the candidate, while it is visible in the real clip, scores -1;
  - an object that really rotates without moving its centroid is scored on its rotation angle instead (a rigid fit
    of its tracked points).

  The same comparison runs on each segment between consecutive evaluation times (weight 0.30), so arriving in one
  jump is not the same as travelling there. The onset term (weight 0.15) compares when the object starts to move.
  The cells themselves have weight 0.55. An object's value is the weighted mean of the terms it has, and the motion
  score is the mean over the moving objects.
- **Still objects.** A still object should stay where it really is. Its violation ramps softly from 0 (within 1 % of
  the diagonal) to 1 (at 5 %). The leak term is the mean violation of the worst still object.
- **Background and camera.** Quiet background points that move in the candidate are penalised on the same ramp. The camera
  term grows linearly with the largest drift of the candidate's camera from the real camera motion, and reaches 1 at
  twice the moving threshold.
- **Identity.** This term is the share of object cells that are lost only in the candidate, plus any drop in
  background visibility.

`R_raw = motion score - (0.6 leak + 0.2 background + 0.2 camera) - 0.5 identity`. When nothing really moves, the
motion score is replaced by `1 - 2 leak` and the leak is not subtracted again. The reward is `clip(R_raw, -2, 1)`.
The reward is a fixed function of the tracker readings and has no learned weights. `scripts/control/reward.py`
documents every constant.

## Picking the winners

`scripts/dpo/build_pairs.py --winner-only --margin 0.2 --expected-group-size 8 --suffix .reward_v5.json` picks one
candidate per condition:

- a condition with fewer than 8 scored candidates is skipped;
- a condition whose best and worst rewards differ by less than 0.2 is skipped, because the group carries no ranking
  signal;
- otherwise the candidate with the highest reward is picked, and equal rewards go to the lower seed index.

The picked video is encoded with the Wan VAE (fp32, posterior mode) and stored as fp16. There is no loser and no
preference term.

`data/ids/raft/winners.tsv` lists the condition and the seed index of every candidate that the released round picked.
To compare your selection with it:

```bash
python - <<'EOF'
import csv, json, glob
ours = {}
for path in glob.glob("work/raft/winners/*/pairs.jsonl"):
    for row in map(json.loads, open(path)):
        if row["pair"]:
            ours[row["sample_id"]] = row["pair"][0]
released = {r["sample_id"]: int(r["seed_index"]) for r in csv.DictReader(open("data/ids/raft/winners.tsv"), delimiter="\t")}
print("same picks:", sum(ours.get(k) == v for k, v in released.items()), "of", len(released), "| extra:", len(set(ours) - set(released)))
EOF
```

## Training

`scripts/train_s2_v4.py --config configs/stage2/raft.yaml --expert high|low --output-dir <work>/raft/train` runs
one process per expert on one GPU. `<work>/raft/train/OVERRIDES.json`, resolved from
`configs/stage2/raft_overrides.json`, sets the recipe. The released settings are:

| Setting | Value |
|---|---|
| Start | GT-SFT generator, step 4000 (`init_from`), fresh optimizer |
| Optimizer steps | 300 per expert, gradient accumulation 4 (1,200 micro-steps), batch 1 |
| Optimizer | AdamW, betas (0.9, 0.999), weight decay 0.01, gradient clip 1.0 |
| Learning rate | LoRA 2e-5, motion adapter 5e-5; 50 warmup steps, then cosine decay to 0.1x at step 300 |
| Precision | bf16 mixed precision, gradient checkpointing |
| Noise | training flow shift 5.0; each expert trains on its own sigma range (boundary sigma 0.9) |
| Clip length | at most 11 latent frames (41 frames) |
| Seed | 20261020 |
| Picked candidates | probability 0.5 per micro-step; plain flow matching on the candidate's latent under the condition it was sampled with (`dpo_sft_weight 1`, `dpo_negative_weight 0`: no preference term, no reference model) |
| Replay | the other micro-steps: GT-SFT pools and mix (P101 fall 0.0964, ramp 0.27855, spring 0.03828, multi 0.05524, liquid 0.03152; SSv2 0.1; PhyCo 0.4), `real` condition, tube-weighted flow matching |
| Checkpoints | every 100 steps: `lora_params.pt`, `motion_adapter.pt`, optimizer and scheduler state, `trainer_state.json` |

The sample stream is a pure function of the seed and the micro-step index: the pool, the clip, whether the
micro-step trains on a picked candidate, and the sigma and noise draw. With the shipped id lists and the released
picks it reproduces the released run exactly, and `tests/test_raft.py` checks this. The trainer refuses to resume a
run whose seed, pools, mix or replay settings changed.

The `raft_*` pools are not part of the replay mix. They hold the conditions of the picked candidates: each picked
candidate is trained under the cached condition of its own pool, which the trainer searches before the replay pools.

## Reproducibility

- **Starting point.** The released round started from a GT-SFT checkpoint that is not distributed. A retrained one
  differs slightly: the GPU reductions of training are not bitwise deterministic ([TRAINING_SFT.md](TRAINING_SFT.md),
  "GPU count"), and some GT-SFT payloads cannot be rebuilt exactly ([DATA_STAGE2.md](DATA_STAGE2.md), "Gaps"). The
  candidates sampled from it, and with them the rewards and the picks, therefore differ from those of the released
  round, and the trained generator is close to, not equal to, the released one. Use the released weights to
  reproduce the published videos.
- **Candidates** are a deterministic function of the seed, the cached condition, the first frame and the software
  stack. The released round ran on H100 GPUs with the pinned versions (torch 2.5.1, diffusers 0.38.0) and native
  PyTorch attention. Another GPU type, library version or attention kernel changes the pixels slightly.
- **Rewards** come from SAM 2.1 and CoTracker3 on the GPU. Small numeric differences in the trackers can flip the
  pick of a group whose best candidates are nearly tied. Compare your picks with `data/ids/raft/winners.tsv`.
- **Training** uses one process per expert. Given the picks, the sample stream is exact, and its numerics are those
  of bf16 training on the GPU. Each micro-step's sigma and noise come from the CUDA generator, seeded per micro-step.
  After a full run, the `micro`/`sigma` column of `steps.jsonl` should match the released run:

  ```bash
  python - <<'EOF'
  import hashlib, json
  released = {"high": "fb68944c0651c4fd9ac99cc06d306034647095546ff16b1c5c53a14e296c2d52",
              "low": "3c75c8d310a091076f8e526623c9697be8fe5daeef3678886eb7ce7319fad6cd"}
  for expert, digest in released.items():
      rows = [json.loads(line) for line in open(f"work/raft/train/{expert}/steps.jsonl")]
      text = "".join(f"{r['micro']}\t{r['sigma']}\n" for r in rows)
      print(expert, hashlib.sha256(text.encode()).hexdigest() == digest)
  EOF
  ```

## Smoke test

```bash
launch/smoke_raft.sh
```

The smoke test runs the whole pipeline on 2 conditions of the `phyco` pool with 2 candidates each and 4 sampler
steps. It picks winners without a margin and trains 2 optimizer steps per expert, with a checkpoint after every
step. It then checks the candidates, the rewards, the selection receipts, `checkpoint-000002/` of both experts, the
finite losses, and that at least one micro-step trained on a picked candidate. Its outputs go to
`<repo>/work/smoke/raft` by default; remove that folder to rerun from scratch.

The smoke test also needs `MOPLAN_SFT_CHECKPOINT`. To exercise the pipeline before a GT-SFT run exists, any
checkpoint in the `high/` + `low/` layout will do, for example the untrained starting point that `--save-init`
writes ([TRAINING_SFT.md](TRAINING_SFT.md), "Starting point without SFT"). The outputs then carry no meaning.

Without the full GT-SFT pools, point the replay at one pool:

```bash
REPLAY_POOLS="phyco=/path/to/sft_data/pools/phyco@$PWD/data/ids/stage2_sft/train/phyco.txt" REPLAY_MIX="phyco:real:1" \
  launch/smoke_raft.sh
```

## Preparing the condition caches

The condition caches are built from the real clips and the released Stage-1 checkpoint. The chain below is how the
released caches were built. Each command's `--help` documents its inputs.

A rebuild from the raw clips is close to the released caches but not bit-identical. The object sets that the real
clips were measured with (`plans/`, `real_tracks/` and the reward scenes) came from an earlier Stage-1 export that is
not released. The released Stage-1 checkpoint gives the predicted conditions (`cond_pred`).

1. **Stage-1 predictions.** Export them with the released Stage-1 checkpoint.
   - PhyCo: `scripts/clean_r2/export_phyco_stage1.py --checkpoint $MOPLAN_WEIGHTS/stage1 --ids
     data/ids/raft/export_phyco_order.txt --output <exports_phyco>` with one shard. The evaluator draws its noise
     from one generator in id order, so keep this order and the single shard to reproduce the released predictions.
   - Physics 101: `scripts/clean_r2/export_p101_stage1.py --split all` (3 shards), with the manifest rows in the order
     of `data/ids/raft/export_p101_order.txt`.
2. **41-frame source caches of the real clips.**
   - PhyCo: `scripts/control/track_real.py --raw-masks '' --max-seconds 4` (CoTracker3 on the real clip, with the
     objects of the Stage-1 export given by `--exports`), then
     `scripts/s2sft/v2/cache_stage2_cond_v2_long.py --max-frames 41 --pad-to 41 --no-pred`.
   - Physics 101: the GT-SFT Physics 101 caches, built by `cache_stage2_cond_v2_long.py` with the same flags and the
     GT-SFT post-processing.
3. **Reward scenes.** Run `scripts/control/rescene_v5.py --cache <source cache> --manifest <manifest>`, which writes
   `<source cache>/scenes_v5/`.
4. **Conditions.** Run `scripts/clean_r2/refresh_predictions.py --cache <source cache> --exports <exports>/flows
   --ids data/ids/raft/<pool ids> --output <pool cache> --stage1-checkpoint $MOPLAN_WEIGHTS/stage1 --purpose evaluation
   --manifest <manifest>` per pool. It renders `cond_pred`, `cond_zero` and `cond_null` from the Stage-1 export into
   a copy of the source cache. The manifest's first frames set the native image size that the plan is mapped from.
   The released PhyCo pools used the training purpose with the receipt of their pool selection. The purpose changes
   only the input checks and the receipt, not the rendered conditions.
5. **Hold rule.** Run `scripts/s2eval/hold_known_beyond_horizon.py <cache> <cache_held> --keys cond_pred`. Beyond the
   last Stage-1 horizon (4 s), this copies the last known latent's condition into every later latent. The generator
   was trained and is deployed with this rule.

The source videos (PhyCo renders and Physics 101 clips) are third-party datasets under their own licences. They are
not distributed with this repository.
