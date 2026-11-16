#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HELM_BIN=${HELM:-helm}
FIXTURE="${ROOT}/test/manifests/values-aws-minimal.yaml"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

config_value() {
  local manifest=$1 key=$2
  awk -v key="$key" '
    $1 == key ":" {
      found++
      sub(/^[^:]+:[[:space:]]*/, "")
      gsub(/^"|"$/, "")
      value = $0
    }
    END {
      if (found != 1) exit 2
      print value
    }
  ' "$manifest"
}

command -v "$HELM_BIN" >/dev/null 2>&1 || fail "Helm is required"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kumo-pool-nofile-contract.XXXXXX")
trap 'rm -r "$TMP_DIR"' EXIT

default_manifest="${TMP_DIR}/default.yaml"
"$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --show-only templates/configmap-kumo-config.yaml >"$default_manifest"
[[ $(config_value "$default_manifest" POOL_NOFILE_SOFT_LIMIT) == 65535 ]] ||
  fail "stock pool workers must render POOL_NOFILE_SOFT_LIMIT=65535"

override_manifest="${TMP_DIR}/override.yaml"
"$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --set kumoConfig.diskgraph.poolWorkerNofileSoftLimit=131072 \
  --show-only templates/configmap-kumo-config.yaml >"$override_manifest"
[[ $(config_value "$override_manifest" POOL_NOFILE_SOFT_LIMIT) == 131072 ]] ||
  fail "explicit pool-worker nofile override was not rendered"

protected_manifest="${TMP_DIR}/protected.yaml"
"$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --set kumoConfig.diskgraph.poolWorkerNofileSoftLimit=8192 \
  --set-string workloads.rest.extraEnv.POOL_NOFILE_SOFT_LIMIT=7 \
  --show-only templates/configmap-kumo-config.yaml >"$protected_manifest"
[[ $(config_value "$protected_manifest" POOL_NOFILE_SOFT_LIMIT) == 8192 ]] ||
  fail "generic extraEnv must not duplicate the dedicated nofile setting"

if "$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --set kumoConfig.diskgraph.poolWorkerNofileSoftLimit=0 >/dev/null 2>&1; then
  fail "values schema accepted zero pool-worker nofile limit"
fi

echo "PASS: pool-worker nofile contract"
