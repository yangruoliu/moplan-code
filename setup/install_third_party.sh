#!/usr/bin/env bash
# Install the two source-only dependencies at the commits used for the reference runs.
#
#   SAM 2     facebookresearch/sam2@2b90b9f, installed into the active Python environment
#             without its CUDA extension (SAM2_BUILD_CUDA=0). The reference runs had no
#             sam2._C, so the video predictor skipped its hole-filling post-processing;
#             building the extension changes the propagated masks.
#   CoTracker facebookresearch/co-tracker@82e02e8, a git checkout that the code imports
#             through $COTRACKER_REPO (it is not pip-installed).
#
# usage: bash setup/install_third_party.sh [--skip-sam2] [--skip-cotracker] [--check]
#   --check   only verify an existing installation
#
# environment:
#   PYTHON          interpreter of the target environment (default: python)
#   INSTALLER       pip | uv (default: pip if "$PYTHON -m pip" works, else uv)
#   COTRACKER_REPO  checkout location (default: <repo>/third_party/co-tracker)
#
# Install requirements.txt (or environment.yml) first: SAM 2 is installed with
# --no-deps and --no-build-isolation, so torch and its runtime dependencies must
# already be present (setuptools is added if it is missing).
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SAM2_URL=https://github.com/facebookresearch/sam2.git
SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
COTRACKER_URL=https://github.com/facebookresearch/co-tracker.git
COTRACKER_COMMIT=82e02e8029753ad4ef13cf06be7f4fc5facdda4d
COTRACKER_REPO=${COTRACKER_REPO:-$REPO_ROOT/third_party/co-tracker}
PYTHON=${PYTHON:-python}

DO_SAM2=1
DO_COTRACKER=1
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --skip-sam2) DO_SAM2=0 ;;
    --skip-cotracker) DO_COTRACKER=0 ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

command -v "$PYTHON" >/dev/null || { echo "error: $PYTHON not found" >&2; exit 1; }
command -v git >/dev/null || { echo "error: git not found" >&2; exit 1; }

if [[ -z ${INSTALLER:-} ]]; then
  if "$PYTHON" -m pip --version >/dev/null 2>&1; then INSTALLER=pip
  elif command -v uv >/dev/null; then INSTALLER=uv
  else echo "error: neither '$PYTHON -m pip' nor uv is available" >&2; exit 1
  fi
fi

py_install() {
  if [[ $INSTALLER == pip ]]; then "$PYTHON" -m pip install "$@"
  else uv pip install --python "$PYTHON" "$@"
  fi
}

py_uninstall() {
  if [[ $INSTALLER == pip ]]; then "$PYTHON" -m pip uninstall -y "$@"
  else uv pip uninstall --python "$PYTHON" "$@"
  fi
}

check_sam2() {
  "$PYTHON" - "$SAM2_COMMIT" <<'PY'
import importlib.metadata as md
import json
import os
import sys

expected = sys.argv[1]
try:
    import sam2
except Exception as exc:  # import also needs hydra-core, omegaconf and iopath
    sys.exit(f"sam2 check: import failed: {exc}")
pkg = os.path.dirname(sam2.__file__)
ext = sorted(f for f in os.listdir(pkg) if f.startswith("_C"))
if ext:
    sys.exit(f"sam2 check: the CUDA extension is present ({', '.join(ext)}); "
             "reinstall with this script so that it is not built")
cfg = os.path.join(pkg, "configs", "sam2.1", "sam2.1_hiera_l.yaml")
if not os.path.isfile(cfg):
    sys.exit(f"sam2 check: packaged config missing: {cfg}")
commit = None
try:
    info = json.loads(md.distribution("SAM-2").read_text("direct_url.json") or "{}")
    commit = info.get("vcs_info", {}).get("commit_id")
except Exception:
    pass
if commit != expected:
    print(f"sam2 check: warning: installed commit is {commit or 'unknown'}, expected {expected}")
print(f"sam2 check: OK ({pkg}, no _C extension)")
PY
}

install_sam2() {
  "$PYTHON" -c "import torch" 2>/dev/null || {
    echo "error: torch is not importable; install requirements.txt first" >&2
    exit 1
  }
  # build backend for --no-build-isolation (it only copies the Python sources)
  "$PYTHON" -c "import setuptools" 2>/dev/null || py_install setuptools==82.0.1
  if "$PYTHON" -c "import importlib.metadata as m; m.distribution('SAM-2')" 2>/dev/null; then
    echo "removing the existing SAM 2 installation"
    py_uninstall SAM-2
  fi
  local nocache=--no-cache-dir
  [[ $INSTALLER == uv ]] && nocache=--no-cache
  # The wheel cache is bypassed so that a wheel built earlier with the extension is not reused.
  echo "installing SAM 2 @ $SAM2_COMMIT (no CUDA extension)"
  export SAM2_BUILD_CUDA=0
  py_install --no-deps --no-build-isolation "$nocache" "git+$SAM2_URL@$SAM2_COMMIT"
}

check_cotracker() {
  [[ -d $COTRACKER_REPO/.git ]] || { echo "cotracker check: no checkout at $COTRACKER_REPO" >&2; return 1; }
  local head
  head=$(git -C "$COTRACKER_REPO" rev-parse HEAD)
  if [[ $head != "$COTRACKER_COMMIT" ]]; then
    echo "cotracker check: HEAD is $head, expected $COTRACKER_COMMIT" >&2
    return 1
  fi
  if [[ -n $(git -C "$COTRACKER_REPO" status --porcelain --untracked-files=no) ]]; then
    echo "cotracker check: warning: the checkout has local modifications"
  fi
  if PYTHONPATH="$COTRACKER_REPO" "$PYTHON" -c "from cotracker.predictor import CoTrackerPredictor" 2>/dev/null; then
    echo "cotracker check: OK ($COTRACKER_REPO @ $COTRACKER_COMMIT)"
  else
    echo "cotracker check: warning: 'from cotracker.predictor import CoTrackerPredictor' failed (is torch installed?)"
  fi
}

install_cotracker() {
  if [[ ! -d $COTRACKER_REPO/.git ]]; then
    if [[ -e $COTRACKER_REPO ]] && [[ -n $(ls -A "$COTRACKER_REPO") ]]; then
      echo "error: $COTRACKER_REPO exists and is not a git checkout" >&2
      exit 1
    fi
    mkdir -p "$COTRACKER_REPO"
    git -C "$COTRACKER_REPO" init -q
    git -C "$COTRACKER_REPO" remote add origin "$COTRACKER_URL"
  fi
  if ! git -C "$COTRACKER_REPO" cat-file -e "$COTRACKER_COMMIT^{commit}" 2>/dev/null; then
    echo "fetching co-tracker @ $COTRACKER_COMMIT"
    git -C "$COTRACKER_REPO" fetch -q --depth 1 origin "$COTRACKER_COMMIT" \
      || git -C "$COTRACKER_REPO" fetch -q origin
  fi
  git -C "$COTRACKER_REPO" -c advice.detachedHead=false checkout -q --detach "$COTRACKER_COMMIT"
}

status=0
if [[ $DO_SAM2 == 1 ]]; then
  [[ $CHECK_ONLY == 1 ]] || install_sam2
  check_sam2 || status=1
fi
if [[ $DO_COTRACKER == 1 ]]; then
  [[ $CHECK_ONLY == 1 ]] || install_cotracker
  check_cotracker || status=1
  echo
  echo "Add to setup/env.sh:"
  echo "  export COTRACKER_REPO=$(printf '%q' "$COTRACKER_REPO")"
fi
exit $status
