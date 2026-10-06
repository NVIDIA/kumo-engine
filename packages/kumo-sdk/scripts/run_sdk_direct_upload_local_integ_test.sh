#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SDK Direct Upload — Local Integration Tests (MVP)
#   chmod +x run_sdk_direct_upload_local_integ_test.sh
#   ./run_sdk_direct_upload_local_integ_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# Required
export RUN_KUMO_LOCAL_INTEG="${RUN_KUMO_LOCAL_INTEG:-1}"
export KUMO_API_URL="${KUMO_API_URL:-http://localhost:10002/api}"
export KUMO_API_TOKEN="${KUMO_API_TOKEN:?Set KUMO_API_TOKEN before running}"

# Optional
export KUMO_KEEP="${KUMO_KEEP:-0}"   # 1=keep local temp data
export KUMO_HEAVY="${KUMO_HEAVY:-0}" # 1=run big many-shard parquet test
# export KUMO_BASE_DIR="/absolute/path"  # default: system temp

echo "=== SDK Local Upload Integration Tests ==="
echo "RUN_KUMO_LOCAL_INTEG=$RUN_KUMO_LOCAL_INTEG"
echo "KUMO_API_URL=$KUMO_API_URL"
echo "KUMO_KEEP=$KUMO_KEEP  KUMO_HEAVY=$KUMO_HEAVY"
echo "KUMO_BASE_DIR=${KUMO_BASE_DIR:-<system-temp>}"
echo "========================================="

"$KUMO_SDK_PYTHON" -m pytest -s -vv \
  test/connector/test_kumo_local_integration_e2e.py
