#!/usr/bin/env bash
set -euo pipefail
# single-GPU: keeps device_map from splitting (failures surface at load, not as a 2nd-GPU OOM)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -m train.sft.trainer --config configs/sft_config.yaml "$@"
