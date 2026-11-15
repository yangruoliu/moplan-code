# Installation

This page sets up the software environment, the two source-only dependencies (SAM 2 and CoTracker) and the model
weights. The reference runs used Linux x86_64, Python 3.11.9, PyTorch 2.5.1 with CUDA 12.4, and H100 80GB GPUs.

## Requirements

- Linux x86_64 with an NVIDIA driver that supports CUDA 12.4 (R550 or newer). The CUDA runtime comes with PyTorch;
  no system CUDA toolkit is needed.
- GPU memory: inference (448x256, sampling with both Wan2.2 experts) needs one 80 GB GPU. Training needs several
  80 GB GPUs; see [TRAINING_STAGE1.md](TRAINING_STAGE1.md), [TRAINING_SFT.md](TRAINING_SFT.md) and
  [TRAINING_RAFT.md](TRAINING_RAFT.md).
- Host memory: the single-process `--phase full` mode of `scripts/infer_stage2_cond_v2.py` offloads to the CPU and
  needs about 70 GB of RAM. The phased modes do not.
- `git`, and network access for the one-time downloads. After that, every run reads local files only.

Disk space:

| item | size |
|---|---|
| conda environment (or uv/pip virtual environment) | about 8 GB |
| Wan2.2-I2V-A14B-Diffusers, pinned revision | 126.2 GB (transformer 57.2, transformer_2 57.2, text_encoder 11.4, vae 0.5 GB) |
| MoPlan weights (Stage-1 and Stage-2) | about 2.4 GB |
| SAM 2.1 Hiera-L checkpoint | 0.9 GB |
| CoTracker3 offline checkpoint | 0.1 GB |

Data, caches and outputs need more; see the data and training pages.

## 1. Python environment

Clone this repository and run every command below from its root. The code runs from the checkout without
installation (`PYTHONPATH` points at the repository root, see step 4).

### Option A: the exact conda package set (recommended)

`setup/conda-linux-64.lock.txt` lists every conda package of the reference environment, the one the reference
outputs were made in. This is the only route that has been tested.

```bash
conda create --name moplan --file setup/conda-linux-64.lock.txt
conda activate moplan
pip install -r requirements.txt
```

The lock file takes four packages from the Anaconda main channel (`repo.anaconda.com`). Check that its terms of
service apply to you. The ffmpeg of this environment needs one extra step before it starts; see
[Video tools](#video-tools).

### Option B: conda from `environment.yml` (not tested)

```bash
conda env create -f environment.yml
conda activate moplan
```

`environment.yml` pins Python, the PyTorch 2.5.1 CUDA 12.4 conda build, torchvision, numpy and its BLAS build, and
every pip package at the version of the reference environment. It does not pin the other conda libraries, and no
environment built from it has been run. A dry-run solve of the file for linux-64 picks other low-level builds than
the lock file, for example:

| package | lock file (option A) | `environment.yml` solve |
|---|---|---|
| mkl | 2023.1.0 (Anaconda main channel) | 2023.2.0 (conda-forge) |
| intel-openmp | 2023.1.0 (Anaconda main channel) | not installed |
| `_openmp_mutex` | GNU OpenMP (`20_gnu`) | LLVM OpenMP (`kmp_llvm`) |
| libjpeg-turbo | 2.0.0 (pytorch channel) | 3.x (conda-forge) |
| ffmpeg | 4.3 (pytorch channel) | 4.4.2 (conda-forge, with libx264) |

The lock's mkl and intel-openmp builds exist only on the Anaconda main channel, so the channels of `environment.yml`
cannot pin them: conda-forge's mkl 2023.1.0 needs a newer llvm-openmp than the lock's 15.0.7, and conda-forge has
no intel-openmp 2023.1.0. CPU results that depend on BLAS or the OpenMP runtime can therefore differ from the
reference environment. PyTorch builds from the pytorch channel are also known to fail at `import torch` (undefined
symbol `iJIT_NotifyEvent`) with mkl 2024.1 or newer; if a solve picks such a version, add `mkl<2024.1` to the
conda dependencies. Use option A when exact reproduction matters.

### Option C: uv or pip

```bash
uv venv --python 3.11.9 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

With pip, create the virtual environment with Python 3.11 (`python3.11 -m venv .venv`) and run
`pip install -r requirements.txt`.

On Linux x86_64 the PyPI wheels of torch 2.5.1 and torchvision 0.20.1 are the CUDA 12.4 builds, the same cuBLAS
12.4.5.8 and cuDNN 9.1.0 as the conda build. To take them from the PyTorch index instead, install them before the
requirements:

```bash
uv pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

The reference outputs were produced in the environment that option A reproduces; option B has the same Python
packages. Whether the PyPI wheels give bit-identical results has not been checked, so use a conda option when exact
reproduction matters.

## 2. SAM 2 and CoTracker

```bash
bash setup/install_third_party.sh
```

The script

- installs SAM 2 from `facebookresearch/sam2` at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` into the active
  environment, with `--no-deps` and without its CUDA extension (`SAM2_BUILD_CUDA=0`);
- checks out `facebookresearch/co-tracker` at commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` into
  `third_party/co-tracker` (or `$COTRACKER_REPO`). The code adds this checkout to `sys.path`; it is not
  pip-installed.

The reference runs had no SAM 2 CUDA extension. Without it, the SAM 2 video predictor skips its small-hole filling
step, so a build with the extension gives slightly different propagated masks. Do not install SAM 2 any other way.
`bash setup/install_third_party.sh --check` verifies an existing installation. In a uv environment without pip the
script uses `uv pip` (set `INSTALLER=pip` or `INSTALLER=uv` to choose).

## 3. Weights

```bash
python setup/download_weights.py --dry-run     # what would be fetched, from remote metadata only
python setup/download_weights.py               # download everything
```

| component | source | variable |
|---|---|---|
| `wan` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` at revision `596658fd9ca6b7b71d5057529bbf319ecbc61d74` | `WAN_MODEL_DIR` |
| `moplan` | `yrlyrl/MoPlan` at revision `b0db0e367f6508977ec9eca693ae3888c52be2dd` | `MOPLAN_WEIGHTS` |
| `sam2` | `facebook/sam2.1-hiera-large`, file `sam2.1_hiera_large.pt` | `SAM2_CHECKPOINT` |
| `cotracker` | `facebook/cotracker3`, file `scaled_offline.pth` | `COTRACKER_CKPT` |

Each file goes to the path in its variable. When a variable is unset, the default is under `weights/` in the
repository (the defaults of `setup/env.sh.example`). The flags `--wan-dir`, `--moplan-dir`, `--sam2-checkpoint` and
`--cotracker-checkpoint` override both.

- `--only wan,sam2` downloads a subset.
- `--offline` lists the targets and their local state without any network access.
- `--verify-only` hashes the local files and downloads nothing.
- An interrupted download resumes when the same command runs again.
- Every file is checked against the size and sha256 in `setup/checksums.json`. Files that pass are recorded in
  `<target>/.moplan_verified.json`, so a rerun does not hash 126 GB again (`--rehash` forces it).
- To reuse an existing copy of the Wan2.2 revision above, point `WAN_MODEL_DIR` at it and run
  `--verify-only --only wan`.

The SAM 2.1 checkpoint is also published at
`https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` (the same file). A manually
downloaded copy can be checked with `--verify-only --only sam2`.

## 4. Environment variables

```bash
cp setup/env.sh.example setup/env.sh    # edit the paths if the weights are not under weights/
source setup/env.sh
```

| variable | content |
|---|---|
| `WAN_MODEL_DIR` | the Wan2.2-I2V-A14B-Diffusers directory |
| `MOPLAN_WEIGHTS` | the MoPlan weights directory (`stage1/`, `stage2/high/`, `stage2/low/`) |
| `SAM2_CHECKPOINT` | the SAM 2.1 Hiera-L `.pt` file |
| `COTRACKER_REPO` | the co-tracker checkout (the directory that contains `cotracker/`) |
| `COTRACKER_CKPT` | the CoTracker3 `scaled_offline.pth` file |

The file also sets `PYTHONPATH` to the repository root and the settings of the reference runs:
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`, `TQDM_DISABLE=1`, `OMP_NUM_THREADS=8`
and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `setup/download_weights.py` ignores `HF_HUB_OFFLINE`, but
`TQDM_DISABLE=1` hides its progress bars.

## 5. Check the installation

```bash
python setup/check_env.py               # versions, GPU, SAM 2, co-tracker checkout, video tools, weight paths
python setup/check_env.py --inference-only   # for inference only (weights from --only wan,moplan,sam2)
python -m pytest tests -q               # CPU tests; no GPU or weights needed
```

`check_env.py` compares every installed package with `requirements.txt` and reports OK, WARN or FAIL per check. It
also checks the video tools (see [Video tools](#video-tools)). With `--inference-only`, the co-tracker checkout, the
CoTracker3 checkpoint and the `ffmpeg` / `ffprobe` on `PATH` are reported as WARN instead of FAIL: only training data,
reward scoring and evaluation need them. See [tests/README.md](../tests/README.md) for the tests.

## Video tools

MoPlan's own code needs no system ffmpeg:

- the Physics-IQ writer (libx264) and its 5 s trim, and the clip builders (`scripts/eval/pisa_clips.py`,
  `scripts/data_real/p101_clips_v2.py`, `scripts/data_real/ssv2_clips.py`), run the ffmpeg binary bundled with
  imageio-ffmpeg 0.6.0 (`imageio_ffmpeg.get_ffmpeg_exe()`; the clip builders take `--ffmpeg` or `$FFMPEG` to use
  another binary);
- the MoProbe / VBench mp4 writer uses PyAV 17.1.0 (libx264);
- the 5.000 s duration check of the Physics-IQ generation reads the container duration with `ffprobe` when one on
  `PATH` runs, and otherwise with PyAV. It stops with an error when neither works.

Some evaluation steps call `ffmpeg` and `ffprobe` from `PATH`: the Physics-IQ pre-check
(`scripts/eval/piq_precheck_run.py`) and the official scorers (see [EVALUATION.md](EVALUATION.md)).

The ffmpeg 4.3 of the lock file (option A; a dependency of the torchvision conda package, from the pytorch channel)
does not start as installed: its `libavcodec` needs `libopenh264.so.5`, and the lock's openh264 2.1.1 provides only
`libopenh264.so.6`. It is also built without libx264. Use one of these fixes:

- Add the missing library name. `ffmpeg` and `ffprobe` 4.3 then start (still without libx264):

  ```bash
  ln -s libopenh264.so.6 "$CONDA_PREFIX/lib/libopenh264.so.5"
  ```

- Or put a separate ffmpeg build with libx264 first on `PATH`, without installing it into the MoPlan environment:

  ```bash
  conda create -y -p third_party/ffmpeg-env --override-channels -c conda-forge "ffmpeg=*=gpl*"
  mkdir -p third_party/ffmpeg/bin
  ln -sf "$PWD/third_party/ffmpeg-env/bin/ffmpeg" "$PWD/third_party/ffmpeg-env/bin/ffprobe" third_party/ffmpeg/bin/
  export PATH=$PWD/third_party/ffmpeg/bin:$PATH      # or uncomment the PATH line in setup/env.sh
  ```

  Only the two links go on `PATH`, so the other programs of that environment do not shadow the MoPlan ones.

With uv or pip (option C) there is no conda ffmpeg; install ffmpeg and ffprobe separately for the evaluation steps.
Use one ffmpeg build for both models of a comparison, so that any scorer-side difference affects both equally.
`python setup/check_env.py` reports whether the bundled binary and PyAV have libx264 and whether the `ffmpeg` and
`ffprobe` on `PATH` run.

## Reproducing the reference outputs exactly

Bit-identical videos are expected only with the same software and GPU type. The following matter:

- **GPU and libraries.** H100 (sm_90), PyTorch 2.5.1 with cuBLAS 12.4.5.8 and cuDNN 9.1.0. Other GPUs or library
  builds change the results in the last bits, and diffusion sampling amplifies such differences.
- **Text embeddings.** The prompt embeddings depend on transformers 5.11.0, tokenizers 0.22.2, ftfy 6.3.1 (the Wan
  prompt cleaning) and diffusers 0.38.0.
- **SAM 2 without its CUDA extension** (step 2).
- **Video files.** mp4 bytes depend on the encoder build: opencv-python 4.13.0.92 (its bundled FFmpeg, mp4v),
  PyAV 17.1.0 (libx264) and imageio-ffmpeg 0.6.0 (its bundled ffmpeg). Install `opencv-python`, not
  `opencv-python-headless`. Another encoder build can write different files from the same frames, so compare the
  `.npy` frame dumps written next to the videos rather than mp4 hashes.
- **Exact pins.** Do not upgrade packages in the environment.

## Benchmark scorers

The benchmark scorers run in their own environments, never in this one. [EVALUATION.md](EVALUATION.md) gives the
code commit and setup of each scorer, and `docs/eval_envs/` lists the package versions of the scorer environments.
Some evaluation steps call `ffmpeg` and `ffprobe` from `PATH`; see [Video tools](#video-tools).

## Troubleshooting

- **A download stalls at 0 bytes.** Some networks block the Xet storage backend that `huggingface_hub` uses for large
  files. Set `HF_HUB_DISABLE_XET=1` to fall back to plain HTTP downloads, then run the same command again.
- **Rate limits.** Set `HF_TOKEN` to a Hugging Face access token. The Wan2.2, SAM 2.1 and CoTracker3 repositories
  are not gated.
- **`import sam2` fails.** Install `requirements.txt` first. SAM 2 is installed without dependencies, and it needs
  hydra-core, omegaconf and iopath from the requirements.
