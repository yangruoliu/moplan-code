"""CPU tests for the installation files.

Covers the pinned requirements, the conda environment file, the checksum file,
the offline code paths of setup/download_weights.py and setup/check_env.py, the
shell scripts under setup/ and the .gitignore rules. No network, GPU or model
weights are needed.

    python -m pytest tests/test_setup.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "setup"
CONDA_PROVIDED = {"torch", "torchvision", "numpy"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dw():
    return load_module("moplan_download_weights", SETUP / "download_weights.py")


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(lines) -> dict[str, str]:
    pins = {}
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name, sep, version = line.partition("==")
        assert sep and version.strip(), f"not an exact pin: {raw!r}"
        key = canonical(name.strip())
        assert key not in pins, f"duplicate pin: {name}"
        pins[key] = version.strip()
    return pins


def requirement_pins() -> dict[str, str]:
    return parse_pins((REPO_ROOT / "requirements.txt").read_text().splitlines())


# --------------------------------------------------------------------------- requirements / conda


def test_requirements_are_exact_pins():
    pins = requirement_pins()
    for name, version in {"torch": "2.5.1", "torchvision": "0.20.1", "numpy": "2.4.6",
                          "diffusers": "0.38.0", "transformers": "5.11.0", "peft": "0.19.1",
                          "accelerate": "1.14.0", "opencv-python": "4.13.0.92", "av": "17.1.0",
                          "ftfy": "6.3.1", "safetensors": "0.8.0"}.items():
        assert pins.get(name) == version, name
    assert "opencv-python-headless" not in pins


def test_environment_yml_matches_requirements():
    yaml = pytest.importorskip("yaml")
    env = yaml.safe_load((REPO_ROOT / "environment.yml").read_text())
    conda = [d for d in env["dependencies"] if isinstance(d, str)]
    pip_lists = [d["pip"] for d in env["dependencies"] if isinstance(d, dict) and "pip" in d]
    assert len(pip_lists) == 1
    assert "python=3.11.9" in conda
    assert "pytorch=2.5.1=py3.11_cuda12.4_cudnn9.1.0_0" in conda
    assert "pytorch-cuda=12.4" in conda
    req = requirement_pins()
    conda_versions = {}
    for dep in conda:
        name, _, rest = dep.partition("=")
        conda_versions[{"pytorch": "torch"}.get(name, name)] = rest.split("=", 1)[0]
    for name in CONDA_PROVIDED:
        assert conda_versions.get(name) == req[name], name
    pip = parse_pins(pip_lists[0])
    expected = {k: v for k, v in req.items() if k not in CONDA_PROVIDED}
    assert pip == expected


def test_conda_lock_is_explicit():
    lines = (SETUP / "conda-linux-64.lock.txt").read_text().splitlines()
    assert "@EXPLICIT" in lines
    urls = [l for l in lines if l and not l.startswith("#") and l != "@EXPLICIT"]
    assert urls and all(u.startswith("https://") for u in urls)
    joined = "\n".join(urls)
    assert "pytorch-2.5.1-py3.11_cuda12.4_cudnn9.1.0_0" in joined
    assert "numpy-2.4.6-" in joined and "python-3.11.9-" in joined


# --------------------------------------------------------------------------- checksum file


def test_checksums_schema(dw):
    data = dw.load_checksums(SETUP / "checksums.json")
    comps = data["components"]
    assert set(comps) == set(dw.COMPONENTS)
    assert {c["env"] for c in comps.values()} == {"WAN_MODEL_DIR", "MOPLAN_WEIGHTS", "SAM2_CHECKPOINT", "COTRACKER_CKPT"}
    for name in ("wan", "sam2", "cotracker"):
        spec = comps[name]
        assert HEX40.match(spec["revision"]), name
        for rel, entry in spec["files"].items():
            assert isinstance(entry["size"], int) and entry["size"] > 0, rel
            assert HEX64.match(entry["sha256"]), rel
    assert comps["wan"]["revision"] == "596658fd9ca6b7b71d5057529bbf319ecbc61d74"
    assert comps["moplan"]["repo_id"] == "yrlyrl/MoPlan"
    for rel in comps["moplan"]["files"]:
        assert rel.startswith(("stage1/", "stage2/high/", "stage2/low/")), rel


def test_checksums_cover_every_wan_component(dw):
    files = dw.load_checksums(SETUP / "checksums.json")["components"]["wan"]["files"]
    assert "model_index.json" in files
    for sub in ("scheduler", "text_encoder", "tokenizer", "transformer", "transformer_2", "vae"):
        assert any(rel.startswith(sub + "/") for rel in files), sub
    for expert in ("transformer", "transformer_2"):
        shards = [r for r in files if r.startswith(expert + "/diffusion_pytorch_model-")]
        assert len(shards) == 12, expert


# --------------------------------------------------------------------------- download_weights.py


def write_checksums(path: Path, name: str, layout: str, files: dict) -> Path:
    spec = {
        "env": "MOPLAN_TEST_UNUSED", "default_path": "unused", "layout": layout, "download": "listed",
        "repo_id": "example/repo", "revision": "0" * 40, "files": files,
    }
    path.write_text(json.dumps({"format": 1, "components": {name: spec}}))
    return path


def test_offline_listing_does_not_import_the_hub(tmp_path):
    code = (
        "import runpy, sys\n"
        "sys.argv = ['download_weights.py', '--offline']\n"
        "try:\n"
        f"    runpy.run_path({str(SETUP / 'download_weights.py')!r}, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    code = exc.code\n"
        "assert 'huggingface_hub' not in sys.modules\n"
        "sys.exit(code)\n"
    )
    env = dict(os.environ)
    for var, sub in (("WAN_MODEL_DIR", "wan"), ("MOPLAN_WEIGHTS", "moplan"),
                     ("SAM2_CHECKPOINT", "sam2/s.pt"), ("COTRACKER_CKPT", "cot/c.pth")):
        env[var] = str(tmp_path / sub)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "596658fd9ca6b7b71d5057529bbf319ecbc61d74" in out.stdout
    assert "yrlyrl/MoPlan" in out.stdout
    assert not any(tmp_path.iterdir()), "listing must not create target directories"


def test_verify_and_stamp(dw, tmp_path):
    root = tmp_path / "w"
    (root / "sub").mkdir(parents=True)
    payload = b"moplan test payload"
    (root / "sub" / "a.bin").write_bytes(payload)
    entry = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    target = dw.Target("wan", {"layout": "dir", "files": {"sub/a.bin": entry}}, root, "test")
    stamps = {}
    assert dw.verify(target, "sub/a.bin", entry, stamps, rehash=False) == (True, "verified")
    assert dw.verify(target, "sub/a.bin", entry, stamps, rehash=False) == (True, "verified (cached)")
    assert dw.local_state(target, "sub/a.bin", entry, stamps) == "verified"
    # same size, different content, new mtime: the record no longer applies
    (root / "sub" / "a.bin").write_bytes(b"X" + payload[1:])
    st = (root / "sub" / "a.bin").stat()
    os.utime(root / "sub" / "a.bin", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    ok, msg = dw.verify(target, "sub/a.bin", entry, stamps, rehash=False)
    assert ok is False and "sha256" in msg
    assert dw.verify(target, "missing.bin", entry, stamps, rehash=False) == (False, "missing")
    assert dw.verify(target, "sub/a.bin", {"size": None, "sha256": None}, stamps, rehash=False)[0] is None


def test_verify_only_cli(dw, tmp_path):
    payload = b"checkpoint bytes"
    ckpt = tmp_path / "ckpt" / "renamed.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(payload)
    sums = write_checksums(tmp_path / "sums.json", "sam2", "file",
                           {"orig.pt": {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}})
    argv = ["--checksums", str(sums), "--only", "sam2", "--verify-only", "--sam2-checkpoint", str(ckpt)]
    assert dw.main(argv) == 0
    stamp = json.loads((ckpt.parent / dw.STAMP_NAME).read_text())
    assert "renamed.pt" in stamp["files"]
    ckpt.write_bytes(b"x" * len(payload))
    st = ckpt.stat()
    os.utime(ckpt, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert dw.main(argv) == 1


def test_bad_arguments(dw, tmp_path):
    assert dw.main(["--offline", "--only", "nope"]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert dw.main(["--offline", "--checksums", str(bad)]) == 2


# --------------------------------------------------------------------------- check_env.py


def test_check_env_runs():
    check_env = load_module("moplan_check_env", SETUP / "check_env.py")
    pins = check_env.read_pins(REPO_ROOT / "requirements.txt")
    assert pins["diffusers"] == ("0.38.0", True)
    assert pins["pytest"][1] is False and pins["certifi"][1] is False
    assert check_env.main(["--no-weights"]) in (0, 1)


def test_check_env_inference_only(tmp_path, monkeypatch):
    """--inference-only: an install made with `download_weights.py --only wan,moplan,sam2` (no CoTracker) passes the
    co-tracker and weight checks with WARN lines; the full check still fails on the missing CoTracker3 checkpoint."""
    check_env = load_module("moplan_check_env_inference", SETUP / "check_env.py")
    for var, markers in check_env.WEIGHT_MARKERS.items():
        for rel in markers:
            (tmp_path / var / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / var / rel).write_bytes(b"x")
        monkeypatch.setenv(var, str(tmp_path / var))
    (tmp_path / "sam2.pt").write_bytes(b"x")
    monkeypatch.setenv("SAM2_CHECKPOINT", str(tmp_path / "sam2.pt"))
    monkeypatch.delenv("COTRACKER_CKPT", raising=False)
    monkeypatch.delenv("COTRACKER_REPO", raising=False)
    check_env.failures = 0
    check_env.check_cotracker(inference_only=True)
    check_env.check_weights(inference_only=True)
    assert check_env.failures == 0
    check_env.check_weights()
    assert check_env.failures == 1
    check_env.check_cotracker()
    assert check_env.failures == 2


def test_check_env_video_tools(monkeypatch):
    """The bundled imageio-ffmpeg binary and PyAV are checked for libx264; a missing ffmpeg / ffprobe on PATH is a
    FAIL in the full check and a WARN with --inference-only."""
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("av")
    check_env = load_module("moplan_check_env_video", SETUP / "check_env.py")
    monkeypatch.setattr(check_env.shutil, "which", lambda name: None)
    check_env.failures = 0
    check_env.check_video_tools(inference_only=True)
    assert check_env.failures == 0
    check_env.check_video_tools()
    assert check_env.failures == 2                                  # ffprobe and ffmpeg


def test_docs_pin_the_weight_revisions():
    """Every manual `hf download` in the docs pins the revision recorded in setup/checksums.json."""
    comps = json.loads((SETUP / "checksums.json").read_text())["components"]
    revisions = {comps[name]["repo_id"]: comps[name]["revision"] for name in ("wan", "moplan")}
    seen = set()
    for doc in sorted((REPO_ROOT / "docs").glob("*.md")) + [REPO_ROOT / "README.md"]:
        for line in doc.read_text().splitlines():
            match = re.search(r"\bhf download (\S+)", line)
            if match and match.group(1) in revisions:
                assert f"--revision {revisions[match.group(1)]}" in line, f"{doc.name}: {line}"
                seen.add(match.group(1))
    assert seen == set(revisions)
    assert revisions["yrlyrl/MoPlan"] in (REPO_ROOT / "docs" / "INSTALL.md").read_text()


# --------------------------------------------------------------------------- shell scripts / gitignore

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


@needs_bash
def test_install_script_parses():
    out = subprocess.run(["bash", "-n", str(SETUP / "install_third_party.sh")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    text = (SETUP / "install_third_party.sh").read_text()
    assert "2b90b9f5ceec907a1c18123530e92e794ad901a4" in text
    assert "82e02e8029753ad4ef13cf06be7f4fc5facdda4d" in text
    assert "SAM2_BUILD_CUDA=0" in text


@needs_bash
def test_env_example_exports():
    names = ["WAN_MODEL_DIR", "MOPLAN_WEIGHTS", "SAM2_CHECKPOINT", "COTRACKER_REPO", "COTRACKER_CKPT",
             "PYTHONPATH", "HF_HUB_OFFLINE"]
    script = f"source {str(SETUP / 'env.sh.example')!r}; " + "; ".join(f'printf "%s\\n" "${n}"' for n in names)
    env = {k: v for k, v in os.environ.items() if k not in names}
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    values = dict(zip(names, out.stdout.splitlines()))
    root = Path(values["PYTHONPATH"].split(":")[0]).resolve()
    assert root == REPO_ROOT.resolve()
    for name in names[:5]:
        assert Path(values[name]).resolve().is_relative_to(root), name
    assert values["HF_HUB_OFFLINE"] == "1"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_gitignore_rules(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    shutil.copy(REPO_ROOT / ".gitignore", tmp_path / ".gitignore")

    def ignored(path: str) -> bool:
        out = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "--no-index", "-q", path])
        return out.returncode == 0

    for path in ["weights/Wan2.2-I2V-A14B-Diffusers/model_index.json", "third_party/co-tracker/setup.py",
                 "outputs/a/video.mp4", "workspace/x.json", "work/runs/x.json", "motion_jepa/__pycache__/a.pyc",
                 "ckpt/lora_params.pt", "cache/flow.npy", "clip.mp4", "setup/env.sh"]:
        assert ignored(path), path
    for path in ["assets/showcase/01_case/comparison.mp4", "assets/prior/prior_latent.pt",
                 "setup/env.sh.example", "motion_jepa/stage2.py", "configs/stage2/inference.yaml",
                 "setup/checksums.json"]:
        assert not ignored(path), path
