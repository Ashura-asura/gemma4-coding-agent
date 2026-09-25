#!/usr/bin/env bash
set -euo pipefail
python -m train.sft.trainer --config configs/sft_config.yaml "$@"
