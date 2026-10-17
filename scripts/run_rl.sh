#!/usr/bin/env bash
set -euo pipefail
# model-written code runs here: pid namespace + read-only rootfs + dropped caps, fail closed (AUDIT C2).
# "required" is the default whenever this host can create a network namespace.
# Hosts that cannot (Kaggle: unshare is seccomp-blocked even as root) would abort
# every rollout under "required" — default to best_effort there, loudly.
if [ -z "${SANDBOX_FS_ISOLATION:-}" ]; then
  if command -v unshare >/dev/null 2>&1 && unshare -n true 2>/dev/null; then
    export SANDBOX_FS_ISOLATION=required
  else
    export SANDBOX_FS_ISOLATION=best_effort
    echo "WARNING: unshare(netns) unavailable on this host — SANDBOX_FS_ISOLATION=best_effort" >&2
    echo "WARNING: sandbox isolation is degraded; reward/eval numbers are not fully isolated" >&2
  fi
fi
python -m train.rl.grpo --config configs/rl_config.yaml "$@"
