#!/usr/bin/env bash
set -euo pipefail
# model-written code runs here: pid namespace + read-only rootfs + dropped caps, fail closed (AUDIT C2).
# Set SANDBOX_FS_ISOLATION=best_effort only if the host cannot provide them (results are then NOT trustworthy).
export SANDBOX_FS_ISOLATION="${SANDBOX_FS_ISOLATION:-required}"
python -m train.rl.grpo --config configs/rl_config.yaml "$@"
