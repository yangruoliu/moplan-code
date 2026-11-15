#!/usr/bin/env bash
# One video with generate.py (first image + prompt -> MoPlan video), in the reference environment settings.
#
# usage: bash launch/infer_generate.sh --image FRAME --prompt TEXT --output OUT.mp4 [generate.py options]
#   e.g. bash launch/infer_generate.sh --protocol moprobe --image assets/showcase/02_moprobe_two_boxes_fall/first_frame.png \
#          --prompt 'A white box and a blue and white box fall at the same time.' --sample-id ps030-pd --output out/ps030.mp4
# env: WAN_MODEL_DIR, MOPLAN_WEIGHTS, SAM2_CHECKPOINT (see launch/infer_common.sh); GPUS selects the GPU (first one is used)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/infer_common.sh"
cd "$REPO"
CUDA_VISIBLE_DEVICES=$(first_gpu) exec "$PYTHON" -u generate.py "$@"
