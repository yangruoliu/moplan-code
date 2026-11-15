# MoPlan

This repository contains the code of **MoPlan**: inference, training of both stages, evaluation, the MoProbe
measurement, and the showcase examples.

## Overview

MoPlan splits image-to-video generation into two decisions. Stage 1, the motion planner, looks at the first frame,
the text prompt and the object masks of the first frame, and predicts how each object moves: a per-object plan of
2D displacements over time. Stage 2, the generator, is Wan2.2-I2V-A14B with a LoRA and a motion adapter on each of
its two experts; it renders the plan into a condition on the latent grid and generates the video that follows it.
Stage 1 is trained on synthetic, simulated and real videos with measured or rendered motion. Stage 2 is first
fine-tuned on real continuations paired with their ground-truth plans, then refined with one round of reward-ranked
fine-tuning (RAFT) on its own samples under predicted plans.

## Repository layout

```
generate.py            first frame + prompt -> video, end to end (Stage 1, then Stage 2)
motion_jepa/           core library: Stage-1 model, flow codec, losses, sampler, Stage-2 condition contract
motionbench/           measurement library of MoProbe (instance tracks, camera rule, metrics)
scripts/               entry points: inference, cache builders, trainers, reward, evaluation, data building
configs/               stage1/train.yaml; stage2/{inference,sft,raft}.yaml and the SFT / RAFT recipe overrides
launch/                single-node bash launchers (inference, training, smoke tests, evaluation)
setup/                 environment check, third-party install, weight download and checksums
data/ids/              frozen id lists of the Stage-1, GT-SFT and RAFT training sets
data/moprobe/          MoProbe probe manifests (ids, prompts, set definitions; no frames)
assets/showcase/       9 qualitative comparison cases with inputs, commands and reference hashes
docs/                  documentation (index: docs/README.md)
tests/                 CPU tests (no GPU, weights or data needed)
tools/                 weights manifest (sizes and sha256 of the released weights)
```

The Python package keeps its historical name `motion_jepa`. Everything runs from the checkout; nothing is
pip-installed.

## Installation

See [docs/INSTALL.md](docs/INSTALL.md). In short (Linux x86_64, CUDA 12.4, one 80 GB GPU for inference):

```bash
conda create --name moplan --file setup/conda-linux-64.lock.txt && conda activate moplan   # the tested route
pip install -r requirements.txt
bash setup/install_third_party.sh                               # SAM 2 and co-tracker at pinned commits
python setup/download_weights.py                                # Wan2.2, MoPlan, SAM 2.1, CoTracker3 (sha256-checked)
cp setup/env.sh.example setup/env.sh && source setup/env.sh     # sets WAN_MODEL_DIR, MOPLAN_WEIGHTS, ...
python setup/check_env.py
```

The lock file is the only tested route; `environment.yml` (option B in INSTALL.md) pins the same Python packages but
has not been tested. An inference-only setup (`python setup/download_weights.py --only wan,moplan,sam2`) runs
`python setup/check_env.py --inference-only` instead. The lock file's ffmpeg needs one extra step before the
evaluation steps can use it; see [Video tools](docs/INSTALL.md#video-tools).

`source setup/env.sh` is needed in every shell before running a launcher. It sets the five shared variables:

| variable | content |
|---|---|
| `WAN_MODEL_DIR` | `Wan2.2-I2V-A14B-Diffusers` directory |
| `MOPLAN_WEIGHTS` | MoPlan weights from the Hugging Face repo `yrlyrl/MoPlan` (`stage1/`, `stage2/high/`, `stage2/low/`) |
| `SAM2_CHECKPOINT` | SAM 2.1 Hiera-L checkpoint `sam2.1_hiera_large.pt` |
| `COTRACKER_REPO` | co-tracker checkout (training data and RAFT reward only) |
| `COTRACKER_CKPT` | CoTracker3 `scaled_offline.pth` (training data and RAFT reward only) |

## Quickstart

Generate one video from a first frame and a prompt:

```bash
python generate.py --image assets/showcase/02_moprobe_two_boxes_fall/first_frame.png \
    --prompt "A white box and a blue and white box fall at the same time." \
    --sample-id ps030-pd --output out/two_boxes.mp4
```

This writes `out/two_boxes.mp4`, the exact frames `out/two_boxes.npy` and a JSON sidecar. Add `--base` for the
untrained Wan2.2-I2V-A14B with the same sampler, image, prompt and seed. `--protocol physics_iq` follows the
Physics-IQ generation protocol instead of the default MoProbe protocol. `launch/infer_generate.sh` runs the same
command with the reference runtime settings. See [docs/INFERENCE.md](docs/INFERENCE.md) for the protocols, the
intermediate files and the batch path of the benchmark runs.

## Training

The released model is trained in three steps. Each guide lists the data, the recipe, the launcher and a smoke test.

1. Stage-1 motion planner: [docs/TRAINING_STAGE1.md](docs/TRAINING_STAGE1.md), data in
   [docs/DATA_STAGE1.md](docs/DATA_STAGE1.md) (`launch/train_stage1.sh`).
2. Stage-2 ground-truth-plan fine-tuning (GT-SFT): [docs/TRAINING_SFT.md](docs/TRAINING_SFT.md), data in
   [docs/DATA_STAGE2.md](docs/DATA_STAGE2.md) (`launch/train_sft.sh`).
3. One RAFT round on the GT-SFT generator: [docs/TRAINING_RAFT.md](docs/TRAINING_RAFT.md) (`launch/raft.sh`).
   The GT-SFT checkpoint the round starts from is not distributed; train it first with `launch/train_sft.sh` (it
   stops at step 4000 of the 8000-step schedule).

## Evaluation

[docs/EVALUATION.md](docs/EVALUATION.md) covers Physics-IQ, PisaBench, RigidBench, Morpheus and VBench-I2V with their
official scorers; [docs/MOPROBE.md](docs/MOPROBE.md) covers MoProbe (the `motionbench/` package and
`scripts/motionbench/`). The MoProbe-P rows cannot be recomputed from this repository: the scripts that write their
per-clip intermediate records are not released (MOPROBE.md, section 5). The launchers are `launch/eval_*.sh`.
Besides the variables above, they read:

| variable | content |
|---|---|
| `EVAL_ROOT` | working folder of all evaluation inputs and outputs (default `eval_work/`) |
| `PIQ_PYTHON`, `PIQ_CODE_VERIFIED`, `PIQ_CODE_ORIGINAL`, `PIQ_DATA` | Physics-IQ scorer python, the two scorer checkouts and the benchmark data |
| `PISA_HOME` | PisaBench scorer folder (checkout, environment and data) |
| `RIGIDBENCH_HOME` | RigidBench checkout with its environment and data |
| `MORPHEUS_HOME` | Morpheus checkout with its environment and data |
| `VBENCH_HOME`, `VBENCH_CACHE_DIR` | VBench checkout with its environment, and its pre-fetched model weights (optional `VBENCH_TORCH_HOME`, default `$VBENCH_CACHE_DIR/torch_home`: the torch.hub tree with CoTracker2) |
| `MOPROBE_ROOT` | MoProbe bundle (first frames, masks, references; not distributed, see MOPROBE.md) |
| `DINOV2_MODEL` | DINOv2 model id or directory for the MoProbe new-object scan (default `facebook/dinov2-base`) |

## Showcase

[assets/showcase/](assets/showcase/README.md) holds 9 comparison cases of MoPlan and the untrained Wan2.2, grouped
by finding, with a video page at the repository root (`index.html`). Every case has its exact first frame, prompt,
seed, the `generate.py` command of each generated panel and the sha256 of the original output; the Physics-IQ cases
(4-7) also include the first-frame masks of the original run:

```bash
python assets/showcase/showcase.py commands --case 7     # print the commands of one case (1-9)
python assets/showcase/showcase.py verify out/showcase   # compare regenerated panels with the references
```

The commands need `WAN_MODEL_DIR` for both models, and `MOPLAN_WEIGHTS` and `SAM2_CHECKPOINT` for MoPlan. The two
RigidBench cases (8, 9) are reported as `NOT-VERIFIABLE`: their references are hashes of the evaluation exports, not
of the files `generate.py` writes (see [assets/showcase/README.md](assets/showcase/README.md)).

## Data

No training data is redistributed. The training sets are defined by the id lists in `data/ids/` and are rebuilt
from the raw datasets (ImageNet, PhyCo, Physics 101, Something-Something v2) with the scripts in this repository,
under each dataset's own terms. The benchmarks are downloaded from their official sources. What the repository does
contain from third parties: the benchmark prompts used as evaluation inputs (`scripts/eval/inputs/*.jsonl`,
`data/moprobe/moprobe_p_rows.jsonl`), and, in the showcase folder, first frames of Physics-IQ, PisaBench, RigidBench
and the Something-Something v2 validation split and short real clips of Physics-IQ and Something-Something v2, for
illustration. These stay under their sources' terms; [LICENSE_NOTICE.md](LICENSE_NOTICE.md) gives the authors,
licences and licence texts. The MoProbe measurement bundle is not distributed.

## Licence

The code is released under the Apache License 2.0 ([LICENSE](LICENSE)). The MoPlan weights and the showcase videos
and images in `assets/showcase/` are released under CC BY-NC 4.0 (non-commercial). Third-party models, packages and
datasets keep their own licences; see [LICENSE_NOTICE.md](LICENSE_NOTICE.md).
