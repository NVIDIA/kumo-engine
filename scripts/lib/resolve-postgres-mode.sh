#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Resolve KUMO_POSTGRES_MODE consistently for compose-local entrypoints.
set -euo pipefail

mode="${KUMO_POSTGRES_MODE:-}"
if [[ -z "$mode" && $# -gt 0 ]]; then
  mode="$(awk -F= '/^KUMO_POSTGRES_MODE=/{v=$2} END{print v}' "$@" 2>/dev/null || true)"
fi

mode="$(printf '%s' "${mode:-bundled}" | tr '[:upper:]' '[:lower:]')"
case "$mode" in
  bundled|external)
    printf '%s\n' "$mode"
    ;;
  *)
    echo "ERROR: KUMO_POSTGRES_MODE must be bundled or external, got '$mode'" >&2
    exit 1
    ;;
esac
