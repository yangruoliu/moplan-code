#!/usr/bin/env python3
"""Check that the active environment matches the MoPlan reference environment.

  python setup/check_env.py                  # packages, GPU, SAM 2, CoTracker, video tools, weight paths
  python setup/check_env.py --inference-only # what generate.py and the batch inference path need
  python setup/check_env.py --no-weights     # skip the weight-path checks

Each line is OK, WARN or FAIL; the exit status is 1 if any check fails.

--inference-only is for a setup made with `download_weights.py --only wan,moplan,sam2` (docs/INFERENCE.md): the
co-tracker checkout, the CoTracker3 checkpoint and the ffmpeg / ffprobe on PATH are needed only for training data,
reward scoring and evaluation, so their problems are reported as WARN instead of FAIL.

This script only looks at versions, paths and whether the video tools run; setup/download_weights.py --verify-only
checks the weight files byte by byte.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
COTRACKER_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
PYTHON_VERSION = (3, 11, 9)
TORCH_CUDA = "12.4"
REFERENCE_GPU_CAPABILITY = (9, 0)

# Files whose presence marks a complete download of each weight directory.
WEIGHT_MARKERS = {
    "WAN_MODEL_DIR": ["model_index.json", "transformer/config.json", "transformer_2/config.json",
                      "vae/diffusion_pytorch_model.safetensors", "text_encoder/config.json",
                      "tokenizer/tokenizer.json", "scheduler/scheduler_config.json"],
    "MOPLAN_WEIGHTS": ["stage1/backend_params.pt", "stage1/decoder.pt", "stage1/trainer_state.json",
                       "stage1/resolved_config.yaml", "stage1/prior_latent.pt",
                       "stage2/high/lora_params.pt", "stage2/high/motion_adapter.pt",
                       "stage2/low/lora_params.pt", "stage2/low/motion_adapter.pt"],
}
WEIGHT_FILES = ("SAM2_CHECKPOINT", "COTRACKER_CKPT")
# Needed only for training data, reward scoring and evaluation (WARN instead of FAIL with --inference-only).
NOT_FOR_INFERENCE = ("COTRACKER_CKPT",)
NOT_FOR_INFERENCE_NOTE = " (not needed for inference)"

failures = 0


def report(level: str, msg: str) -> None:
    global failures
    failures += level == "FAIL"
    print(f"{level:4s}  {msg}", flush=True)


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path) -> dict[str, tuple[str, bool]]:
    """{name: (version, strict)}; pins in the tests and transitive sections are not strict."""
    pins = {}
    strict = True
    for raw in path.read_text().splitlines():
        if raw.startswith("# ---"):
            section = raw.strip("# -").lower()
            strict = not (section.startswith("tests") or section.startswith("transitive"))
            continue
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name, sep, version = line.partition("==")
        if sep:
            pins[canonical(name)] = (version.strip(), strict)
    return pins


def public(version: str) -> str:
    return version.split("+", 1)[0]


def check_python() -> None:
    got = sys.version_info[:3]
    level = "OK" if got == PYTHON_VERSION else ("WARN" if got[:2] == PYTHON_VERSION[:2] else "FAIL")
    report(level, f"python {'.'.join(map(str, got))} (reference {'.'.join(map(str, PYTHON_VERSION))})")


def check_packages() -> None:
    pins = read_pins(REQUIREMENTS)
    mismatches = 0
    for name, (want, strict) in sorted(pins.items()):
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            have = None
        if have is not None and public(have) == public(want):
            continue
        mismatches += 1
        found = f"{have}" if have else "missing"
        report("FAIL" if strict else "WARN", f"package {name}: {found} (requirements.txt pins {want})")
    if not mismatches:
        report("OK", f"all {len(pins)} pinned packages match requirements.txt")
    try:
        headless = md.version("opencv-python-headless")
    except md.PackageNotFoundError:
        headless = None
    if headless:
        report("FAIL", f"opencv-python-headless {headless} is installed next to opencv-python; uninstall it")


def check_torch() -> None:
    try:
        import torch
    except Exception as exc:
        report("FAIL", f"import torch: {exc}")
        return
    cuda = torch.version.cuda
    report("OK" if cuda == TORCH_CUDA else "FAIL", f"torch {torch.__version__}, built for CUDA {cuda} (reference {TORCH_CUDA})")
    if not torch.cuda.is_available():
        report("WARN", "no CUDA device visible (fine for the CPU tests; inference and training need a GPU)")
        return
    report("OK", f"cuDNN {torch.backends.cudnn.version()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        cap = (props.major, props.minor)
        gib = props.total_memory / 2**30
        level = "OK" if cap == REFERENCE_GPU_CAPABILITY else "WARN"
        note = "" if level == "OK" else " (reference runs used sm_90 / H100; outputs may differ in the last bits)"
        report(level, f"cuda:{i} {props.name}, {gib:.0f} GiB, sm_{props.major}{props.minor}{note}")
        if gib < 79:
            report("WARN", f"cuda:{i} has less than 80 GB; inference was run on 80 GB GPUs")


def check_sam2() -> None:
    try:
        import sam2
    except Exception as exc:
        report("FAIL", f"import sam2: {exc} (run setup/install_third_party.sh)")
        return
    pkg = Path(sam2.__file__).parent
    ext = sorted(p.name for p in pkg.iterdir() if p.name.startswith("_C"))
    if ext:
        report("FAIL", f"sam2 has its CUDA extension ({', '.join(ext)}); reinstall with setup/install_third_party.sh")
    else:
        report("OK", "sam2 without the CUDA extension")
    if not (pkg / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml").is_file():
        report("FAIL", "sam2 packaged config configs/sam2.1/sam2.1_hiera_l.yaml is missing")
    try:
        info = json.loads(md.distribution("SAM-2").read_text("direct_url.json") or "{}")
        commit = info.get("vcs_info", {}).get("commit_id")
    except Exception:
        commit = None
    report("OK" if commit == SAM2_COMMIT else "WARN", f"sam2 commit {commit or 'unknown'} (reference {SAM2_COMMIT[:7]})")


def check_cotracker(inference_only: bool = False) -> None:
    bad, note = ("WARN", NOT_FOR_INFERENCE_NOTE) if inference_only else ("FAIL", "")
    repo = os.environ.get("COTRACKER_REPO")
    if not repo:
        report(bad, f"COTRACKER_REPO is not set (source setup/env.sh){note}")
        return
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        report(bad, f"COTRACKER_REPO={repo} is not a git checkout (run setup/install_third_party.sh){note}")
        return
    report("OK" if head == COTRACKER_COMMIT else bad, f"co-tracker checkout at {head[:7]} (reference {COTRACKER_COMMIT[:7]})")
    if not (Path(repo) / "cotracker" / "predictor.py").is_file():
        report(bad, f"{repo} has no cotracker/predictor.py{note}")


def run_tool(cmd: list[str]) -> str | None:
    """stdout of a command that exits with 0, else None (also when it cannot be started)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def has_encoder(exe: str, name: str) -> bool:
    out = run_tool([exe, "-hide_banner", "-encoders"])
    return bool(out) and re.search(rf"^\s*\S+\s+{re.escape(name)}\s", out, re.MULTILINE) is not None


def first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def check_video_tools(inference_only: bool = False) -> None:
    """The encoders MoPlan's own code uses, and the ffmpeg / ffprobe on PATH that evaluation steps call."""
    # imageio-ffmpeg's bundled binary: the Physics-IQ writer and 5 s trim, and the clip builders
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", f"imageio-ffmpeg binary: {exc}")
    else:
        version = run_tool([exe, "-hide_banner", "-version"])
        if version is None:
            report("FAIL", f"imageio-ffmpeg binary {exe} does not run")
        elif not has_encoder(exe, "libx264"):
            report("FAIL", f"imageio-ffmpeg binary has no libx264 encoder ({first_line(version)})")
        else:
            report("OK", f"imageio-ffmpeg binary with libx264 ({first_line(version)})")
    # PyAV: the MoProbe / VBench mp4 writer (libx264), and the duration check when there is no working ffprobe
    try:
        import av
        av.codec.Codec("libx264", "w")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", f"PyAV with the libx264 encoder: {exc}")
    else:
        report("OK", f"PyAV {av.__version__} with the libx264 encoder")
    # ffmpeg / ffprobe on PATH: Physics-IQ pre-check and the official scorers (docs/INSTALL.md, "Video tools")
    bad, note = ("WARN", NOT_FOR_INFERENCE_NOTE) if inference_only else ("FAIL", " (see docs/INSTALL.md, Video tools)")
    for tool in ("ffprobe", "ffmpeg"):
        path = shutil.which(tool)
        version = run_tool([path, "-hide_banner", "-version"]) if path else None
        if not path:
            report(bad, f"{tool} is not on PATH{note}")
        elif version is None:
            report(bad, f"{tool} on PATH ({path}) does not run{note}")
        else:
            report("OK", f"{tool} on PATH: {first_line(version)}")
            if tool == "ffmpeg" and not has_encoder(path, "libx264"):
                report("WARN", "ffmpeg on PATH has no libx264 encoder (MoPlan's own code does not need it; "
                               "see docs/INSTALL.md, Video tools)")


def check_weights(inference_only: bool = False) -> None:
    for var, markers in WEIGHT_MARKERS.items():
        root = os.environ.get(var)
        if not root:
            report("FAIL", f"{var} is not set (source setup/env.sh)")
            continue
        missing = [m for m in markers if not (Path(root) / m).is_file()]
        if missing:
            report("FAIL", f"{var}={root}: missing {', '.join(missing)}")
        else:
            report("OK", f"{var}={root}")
    for var in WEIGHT_FILES:
        bad, note = ("WARN", NOT_FOR_INFERENCE_NOTE) if inference_only and var in NOT_FOR_INFERENCE else ("FAIL", "")
        path = os.environ.get(var)
        if not path:
            report(bad, f"{var} is not set (source setup/env.sh){note}")
        elif not Path(path).is_file():
            report(bad, f"{var}={path}: file not found{note}")
        else:
            report("OK", f"{var}={path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--no-weights", action="store_true", help="skip the weight-path checks")
    parser.add_argument("--inference-only", action="store_true",
                        help="report what only training, reward scoring and evaluation need (co-tracker, CoTracker3 "
                             "checkpoint, ffmpeg / ffprobe on PATH) as WARN instead of FAIL")
    args = parser.parse_args(argv)
    check_python()
    check_packages()
    check_torch()
    check_sam2()
    check_cotracker(args.inference_only)
    check_video_tools(args.inference_only)
    if not args.no_weights:
        check_weights(args.inference_only)
    print(f"\n{failures} check(s) failed" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
