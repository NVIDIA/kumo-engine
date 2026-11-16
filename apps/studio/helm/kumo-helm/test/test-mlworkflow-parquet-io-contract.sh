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

env_value() {
  local manifest=$1 wanted=$2
  awk -v wanted="$wanted" '
    $1 == "-" && $2 == "name:" && $3 == wanted {
      found++
      if (getline <= 0 || $1 != "value:") exit 3
      sub(/^[[:space:]]*value:[[:space:]]*/, "")
      gsub(/^"|"$/, "")
      result = $0
    }
    END {
      if (found != 1) exit 2
      print result
    }
  ' "$manifest"
}

command -v "$HELM_BIN" >/dev/null 2>&1 || fail "Helm is required"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kumo-parquet-io-contract.XXXXXX")
trap 'rm -r "$TMP_DIR"' EXIT

default_manifest="${TMP_DIR}/default.yaml"
"$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --show-only templates/deployment-mlworkflow.yaml >"$default_manifest"

[[ $(env_value "$default_manifest" MAX_CONCURRENT_PARQUET_IO) == 1 ]] ||
  fail "stock mlworkflow must render MAX_CONCURRENT_PARQUET_IO=1"
grep -Fq 'memory: 5Gi' "$default_manifest" ||
  fail "stock mlworkflow memory limit must remain 5Gi"

large_manifest="${TMP_DIR}/large.yaml"
"$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --set workloads.mlworkflow.maxConcurrentParquetIO=2 \
  --set-string workloads.mlworkflow.resources.limits.memory=12Gi \
  --show-only templates/deployment-mlworkflow.yaml >"$large_manifest"

[[ $(env_value "$large_manifest" MAX_CONCURRENT_PARQUET_IO) == 2 ]] ||
  fail "explicit larger-pod override must render concurrency 2"
grep -Fq 'memory: 12Gi' "$large_manifest" ||
  fail "larger-pod memory override was not rendered"

config_manifest="${TMP_DIR}/configmap.yaml"
"$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --show-only templates/configmap-kumo-config.yaml >"$config_manifest"
if grep -Fq MAX_CONCURRENT_PARQUET_IO "$config_manifest"; then
  fail "Parquet IO concurrency must remain scoped to mlworkflow"
fi

if "$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --set workloads.mlworkflow.maxConcurrentParquetIO=0 >/dev/null 2>&1; then
  fail "values schema accepted zero Parquet IO concurrency"
fi

echo "PASS: mlworkflow Parquet IO resource contract"
