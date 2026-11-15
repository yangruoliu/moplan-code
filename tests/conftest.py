"""Shared pytest setup: make the repository root importable.

The code is used from a checkout without installation, so ``motion_jepa`` and
``scripts`` are imported from the repository root (as ``PYTHONPATH`` does in
setup/env.sh.example).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
