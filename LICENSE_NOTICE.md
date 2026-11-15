# Licence notice

## This repository

- **Code and text** (everything in this repository except the media files `*.mp4`, `*.png` and `*.jpg` in
  `assets/showcase/`; this includes `assets/showcase/showcase.py`, `index.html`, the `meta.json` files and
  `README.md` there): Apache License 2.0, see [LICENSE](LICENSE), except for the third-party text listed below.
- **MoPlan weights** (Hugging Face `yrlyrl/MoPlan`) and the **showcase videos and images** in `assets/showcase/`:
  Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0),
  https://creativecommons.org/licenses/by-nc/4.0/ . The showcase material also contains frames and clips of the
  benchmark and dataset sources listed below, which stay under their own terms.

**Third-party text and code inside the repository.** The following is not the authors' own work. It is not covered
by the grants above and remains under its sources' terms:

- Evaluation inputs in `scripts/eval/inputs/`:
  - `morpheus_plain.jsonl`: the Morpheus prompts (`prompts/plain/<experiment>.txt`), verbatim. Morpheus,
    https://github.com/physics-from-video/Morpheus , Copyright (c) 2026 Antonios Tragoudaras, MIT License (full text
    under "Third-party licence texts" below).
  - `morpheus_enhanced.jsonl`: the outputs of Wan2.2's own prompt extension (in Chinese) for those Morpheus prompts,
    kept verbatim because they cannot be regenerated bit-exactly. Wan2.2 is Apache-2.0; the prompts they extend are
    Morpheus' (MIT).
  - `pisabench_real.jsonl`: the PisaBench real-split prompts and negative prompt, verbatim (PisaBench, Apache-2.0;
    see the datasets table).
  - VBench-I2V prompts are not shipped: `scripts/bench/vbench_inputs.py` reads them from your VBench checkout
    (Apache-2.0).
- MoProbe prompts: the MoProbe-P rows of `data/moprobe/moprobe_p_rows.jsonl` are the source benchmarks' own prompts,
  verbatim: PisaBench (`ps`, Apache-2.0), Physics-IQ (`piq`, CC BY 4.0), Morpheus (`mo`, MIT), CausalCascade (`cc`,
  dataset CC BY 4.0, code MIT) and RigidBench (`rb`, evaluation set CC BY 4.0, code MIT); see the datasets table and
  "Third-party licence texts" below. The MoProbe-H instructions (`moprobe_h_rows.jsonl`) and
  `scripts/motionbench/ssv2_templates.json` are the authors' own text.
- Showcase prompts and first frames: the `meta.json` files and the media in `assets/showcase/` contain PisaBench,
  Physics-IQ and RigidBench prompts and first frames, and a Something-Something v2 frame and clip (see
  "Showcase sources" below).
- Code adapted from Wan2.2 (https://github.com/Wan-Video/Wan2.2 at `1ea34ff`, Copyright 2024-2025 The Alibaba Wan
  Team Authors, Apache-2.0), modified: the sampling helpers of `scripts/s2eval_verified/gen_verified.py` (size rule,
  timesteps, mask and frame writing) and the default negative prompt.
- The Morpheus taxonomy constants copied into `scripts/bench/morpheus_inputs.py` (Morpheus, MIT License, full text
  below).

The third-party components below keep their own licences in every case.

The MoPlan weights are adapters (LoRA and motion modules) for Wan2.2-I2V-A14B. They were trained on the datasets
listed below and with training labels made with SAM 2.1 and CoTracker3. The terms of those datasets and models
also apply to how the weights can be used.

## Models

None of these weights is redistributed here; `setup/download_weights.py` fetches them from their official sources.

| model | used for | licence |
|---|---|---|
| Wan2.2-I2V-A14B, Diffusers format (`Wan-AI/Wan2.2-I2V-A14B-Diffusers`) | base model of Stage-1 and Stage-2, VAE, umT5 text encoder | Apache-2.0 |
| SAM 2 code and the SAM 2.1 Hiera-L checkpoint (`facebookresearch/sam2`, `facebook/sam2.1-hiera-large`) | first-frame object masks, mask propagation in videos | Apache-2.0 |
| DINOv2-base (`facebook/dinov2-base`) | features of the MoProbe new-object scan (`scripts/motionbench/new_object_scan.py`) | Apache-2.0 |
| CoTracker code and the CoTracker3 checkpoint (`facebookresearch/co-tracker`, `facebook/cotracker3`) | point tracks for the real-video training plans and the RAFT reward | CC BY-NC 4.0 (non-commercial) |
| torchvision RAFT-Large (`C_T_SKHT_V2`) | optical-flow targets when rebuilding the PhyCo Stage-1 data | distributed with torchvision (BSD-3-Clause) |

The default negative prompt in the generation code is the one published with Wan2.2 (Apache-2.0).

The benchmark scorers download their own models (see [docs/EVALUATION.md](docs/EVALUATION.md)). Some of them are
non-commercial, for example Video-Depth-Anything-Large (CC BY-NC 4.0, used by RigidBench).

## Python packages

Installed from PyPI or conda by `requirements.txt` / `environment.yml`; none is vendored in this repository.

| package | licence |
|---|---|
| torch, torchvision | BSD-3-Clause |
| numpy, scipy, psutil, h5py, protobuf, omegaconf, antlr4-python3-runtime, portalocker | BSD-3-Clause |
| diffusers, transformers, tokenizers, accelerate, peft, safetensors, huggingface_hub, hf-xet, sentencepiece, ftfy | Apache-2.0 |
| regex | Apache-2.0 and CNRI-Python |
| opencv-python | Apache-2.0; the wheels bundle FFmpeg (LGPL-2.1) and Qt (LGPL-3.0) |
| av (PyAV) | BSD-3-Clause; the wheels bundle FFmpeg and codec libraries under their own (LGPL or GPL) licences |
| imageio, imageio-ffmpeg | BSD-2-Clause; imageio-ffmpeg bundles an ffmpeg executable under its own licence |
| pillow | MIT-CMU |
| PyYAML, einops, hydra-core, iopath, pytest | MIT |
| tqdm | MPL-2.0 and MIT |

The conda environment also contains NVIDIA CUDA runtime libraries (NVIDIA EULA) and Intel MKL (Intel Simplified
Software License).

## Datasets

No training dataset is redistributed here. The repository contains benchmark prompts (see above) and, in
`assets/showcase/`, a few first frames and short clips of the evaluation sources for illustration. Obtain each
dataset from its official source and follow its terms.

| dataset | used for |
|---|---|
| PhyCo (Kubric renders; access by request) | Stage-1 and Stage-2 training data |
| Physics 101 | Stage-1 and Stage-2 training data; RAFT conditions (with PhyCo) |
| Something-Something v2 (registration required) | Stage-1 and Stage-2 training data (train split); MoProbe (validation split) |
| Something-Else box annotations for Something-Something v2 | Stage-1 training data |
| ImageNet ILSVRC-2012 train and validation images (non-commercial research use) | backgrounds (train images) and object cut-outs (validation images) of the synthetic Stage-1 scenes |
| Physics-IQ (https://github.com/google-deepmind/physics-IQ-benchmark , public bucket `gs://physics-iq-benchmark`; Copyright 2024 DeepMind Technologies Limited; materials CC BY 4.0, software Apache-2.0) | evaluation; first frames and prompts also in MoProbe-P; showcase cases 4–7 |
| Physics-IQ Verified (Hugging Face `Anates-Labs-Research/Physics-IQ-Verified`, derived from Physics-IQ, CC BY 4.0; a gated download whose access requests are approved automatically) | evaluation |
| PisaBench (Chenyu Li, Oscar Michel, Xichen Pan, Sainan Liu, Mike Roberts and Saining Xie, "PISA Experiments: Exploring Physics Post-Training for Video Diffusion Models by Watching Stuff Drop", arXiv:2503.09595; Hugging Face dataset `nyu-visionx/pisa-experiments`, Apache-2.0 as declared on its dataset card; code https://github.com/vision-x-nyu/pisa-experiments , Apache-2.0) | evaluation; first frames and prompts also in MoProbe-P; showcase cases 2 and 3 |
| RigidBench (Swarnim Jain and Shangzhe Wu, "RigidBench: simulator-grounded evaluation data for rigid-body physics in video generation", v1.1.0, https://doi.org/10.5281/zenodo.21649156 , CC BY 4.0; code https://github.com/swarnim-j/RigidBench , MIT, Copyright (c) 2026 Swarnim Jain) | evaluation; first frames and prompts also in MoProbe-P; showcase cases 8 and 9 |
| Morpheus (https://github.com/physics-from-video/Morpheus , MIT, Copyright (c) 2026 Antonios Tragoudaras; data: Hugging Face `physics-from-video/morpheus-real-world`, MIT as declared on its dataset card) | evaluation; first frames and prompts also in MoProbe-P |
| VBench-I2V | evaluation (prompts read from your VBench checkout, not shipped) |
| CausalCascade (Jialiang Li, Zhengning Xu, Zongyang Ma and Shaobo Cui, "A Plausible Future Is Not a Causal Future: Can Video Models Generate How the World Unfolds?", arXiv:2610.15562; dataset: Hugging Face `deepdeltalab/CausalCascade`, CC BY 4.0; code: https://github.com/SJTU-DeepDeltaLab/CausalCascade , MIT, Copyright (c) 2026 CausalCascade contributors) | MoProbe-P first frames and prompts |

## Showcase sources

The videos in `assets/showcase/` were generated from first frames of Physics-IQ, PisaBench, RigidBench and the
Something-Something v2 validation split, and some of them show the real Something-Something v2 or Physics-IQ clip
next to the generated ones. The source frames and clips remain subject to the terms of those datasets. Per case
(numbers as in `assets/showcase/README.md`; each `meta.json` has a `first_frame.attribution` field):

- **Case 1**, Something-Something v2 validation video 9578 (first frame, real panel and strip): (C) 2017 Twenty
  Billion Neurons GmbH, under the dataset's own terms. The first frame is letterboxed onto a 1280x720 canvas; the
  real clip is resampled to 7 fps and scaled to 448x256.
- **Cases 2 and 3**, PisaBench real clips 0090 and 0096 (first frames and prompts; authors and licence in the
  datasets table): the first frames are letterboxed onto a 1280x720 canvas; the prompts are verbatim.
- **Cases 4–7**, Physics-IQ (switch frames and the real takes `full-videos/take-1/30FPS`): files of the original
  Physics-IQ release (https://github.com/google-deepmind/physics-IQ-benchmark , public bucket
  `gs://physics-iq-benchmark`), Copyright 2024 DeepMind Technologies Limited, licensed under CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). The evaluation read the switch frames from the Physics-IQ Verified
  copy; for these four cases the first frames and the real takes were compared by sha256 with the files in the
  public bucket and are byte-identical. The switch frames are unmodified; the real clips are resampled to 8 fps,
  scaled to 448x256 and start at the switch frame (see `assets/showcase/README.md`).
- **Cases 8 and 9**, RigidBench samples 00077 and 00059 (first frames and prompts): RigidBench evaluation set v1.1.0
  by Swarnim Jain and Shangzhe Wu, https://doi.org/10.5281/zenodo.21649156 , licensed under CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). The first frames are pixel-identical to the RigidBench images, with
  the PNG text and EXIF chunks removed; the generator prompt is the benchmark prompt plus the sentence " Static shot
  with no camera movement.". RigidBench states that some of its rendered scenes and objects use BlenderKit assets
  under Royalty Free or CC0 licences.

## Third-party licence texts

**CC BY 4.0 material** (Physics-IQ, Physics-IQ Verified, RigidBench evaluation set, CausalCascade dataset): the
creators, copyright notices, modifications and links are given in the datasets table and under "Showcase sources"
above. Licence: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/legalcode .
The material is provided by its licensors as-is, without warranties.

**Apache-2.0 material** (PisaBench prompts, first frames and negative prompt; Wan2.2 code and negative prompt): the
Apache License 2.0 text is in [LICENSE](LICENSE). Changes are described where the material is used (letterboxed
PisaBench first frames; the Wan2.2 helpers listed above).

**MIT material**: the Morpheus prompts and taxonomy constants (`scripts/eval/inputs/morpheus_plain.jsonl`, the `mo`
rows of `data/moprobe/moprobe_p_rows.jsonl`, `scripts/bench/morpheus_inputs.py`), and the parts of the RigidBench
and CausalCascade prompts that come from their code repositories (the `rb` and `cc` rows of
`data/moprobe/moprobe_p_rows.jsonl`, the RigidBench prompts in `assets/showcase/08_*/meta.json` and
`assets/showcase/09_*/meta.json`) are covered by the following copyright notices and the MIT License text below:

```
Copyright (c) 2026 Antonios Tragoudaras          (Morpheus, https://github.com/physics-from-video/Morpheus)
Copyright (c) 2026 Swarnim Jain                  (RigidBench, https://github.com/swarnim-j/RigidBench)
Copyright (c) 2026 CausalCascade contributors    (CausalCascade, https://github.com/SJTU-DeepDeltaLab/CausalCascade)
```

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
