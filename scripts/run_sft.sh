#!/usr/bin/env bash
set -euo pipefail
# single-GPU: keeps device_map from splitting (failures surface at load, not as a 2nd-GPU OOM)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# loading+bnb conversion fragments the cache; segmented growth reclaims holes
# (~0.5-1.5GiB on a 16GiB T4 where resident E4B + sliced logits sit at the edge)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -m train.sft.trainer --config configs/sft_config.yaml "$@"
