#!/usr/bin/env bash
set -euo pipefail
python -m eval.run_eval --config configs/eval_config.yaml "$@"
