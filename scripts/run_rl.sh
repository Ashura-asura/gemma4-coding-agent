#!/usr/bin/env bash
set -euo pipefail
python -m train.rl.grpo --config configs/rl_config.yaml "$@"
