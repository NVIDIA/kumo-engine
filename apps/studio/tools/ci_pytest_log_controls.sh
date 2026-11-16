# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared log-noise controls for GitHub Actions pytest shards.

configure_pytest_ci_output() {
  # Keep local pytest output unchanged. In CI, capture successful test output,
  # suppress warning summaries, and still emit captured output on failure.
  PYTEST_CI_OUTPUT_ARGS=(--capture=fd --verbosity=0 --disable-warnings)
}

start_ci_memory_monitor() {
  ./tools/print_memory_usage.py \
    --interval "${KUMO_CI_MEMORY_LOG_INTERVAL_SECONDS:-60}" \
    --min-used-pct "${KUMO_CI_MEMORY_LOG_MIN_USED_PCT:-0.85}" \
    --always-first &
  MEMORY_MONITOR_PID=$!
  trap 'if [ -n "${MEMORY_MONITOR_PID:-}" ]; then kill "$MEMORY_MONITOR_PID" 2>/dev/null || true; fi' EXIT
}

move_coverage_artifact() {
  if [ -f .coverage ]; then
    mv .coverage "$ARTIFACT_NAME_SAFE"
  fi
}
