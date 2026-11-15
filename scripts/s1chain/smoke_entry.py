#!/usr/bin/env python3
"""Run the Stage-1 trainer with the rank-count contract disabled.

The released recipe pins the number of ranks (``unified_experiment.world_size``
must be 4 or 8 and match the runtime) so that exact resume stays valid. This
entry point disables only that check and forwards every argument unchanged to
``scripts/train_stage1_flowdit.py``. It is meant for smoke tests on fewer GPUs;
set ``train.grad_accum_steps`` and ``train.global_batch`` so that
ranks * per_gpu_batch * grad_accum_steps == global_batch.
"""
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import motion_jepa.flow_dit_unified as unified  # noqa: E402

unified.validate_unified_runtime = lambda config, actual_world_size: None
unified.unified_world_size = lambda config, required=False: None
TRAINER = REPO_ROOT / "scripts" / "train_stage1_flowdit.py"
sys.argv = [str(TRAINER), *sys.argv[1:]]
runpy.run_path(str(TRAINER), run_name="__main__")
