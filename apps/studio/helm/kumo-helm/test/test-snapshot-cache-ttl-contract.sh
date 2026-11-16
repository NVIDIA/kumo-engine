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

assert_cache_ttl() {
  local manifest=$1 expected=$2
  grep -Eq -- '\\"pool_max_ssd_idle_secs\\":'"${expected}"'([,}])' "$manifest" ||
    fail "$manifest does not configure pool_max_ssd_idle_secs=${expected}"
}

command -v "$HELM_BIN" >/dev/null 2>&1 || fail "Helm is required"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kumo-snapshot-cache-ttl-contract.XXXXXX")
trap 'rm -r "$TMP_DIR"' EXIT

default_manifest="${TMP_DIR}/default.yaml"
"$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --show-only templates/configmap-kumo-config.yaml >"$default_manifest"
assert_cache_ttl "$default_manifest" 3600

override_manifest="${TMP_DIR}/override.yaml"
"$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --set kumoConfig.diskgraph.snapshotCacheTtlSeconds=7200 \
  --show-only templates/configmap-kumo-config.yaml >"$override_manifest"
assert_cache_ttl "$override_manifest" 7200

minimum_manifest="${TMP_DIR}/minimum.yaml"
"$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --set kumoConfig.diskgraph.snapshotCacheTtlSeconds=300 \
  --show-only templates/configmap-kumo-config.yaml >"$minimum_manifest"
assert_cache_ttl "$minimum_manifest" 300

for invalid_ttl in 299 0 -1; do
  if "$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
    --set kumoConfig.diskgraph.snapshotCacheTtlSeconds="$invalid_ttl" \
    >/dev/null 2>&1; then
    fail "values schema accepted snapshot cache TTL ${invalid_ttl}; minimum is 300"
  fi
done

extra_minimum_manifest="${TMP_DIR}/extra-minimum.yaml"
"$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --set kumoConfig.extra.data_compute.pool_max_ssd_idle_secs=300 \
  --show-only templates/configmap-kumo-config.yaml >"$extra_minimum_manifest"
assert_cache_ttl "$extra_minimum_manifest" 300

if "$HELM_BIN" template kumo "$ROOT" -f "$FIXTURE" \
  --set kumoConfig.extra.data_compute.pool_max_ssd_idle_secs=299 \
  >/dev/null 2>&1; then
  fail "kumoConfig.extra bypassed the 300-second snapshot cache TTL minimum"
fi

echo "PASS: snapshot cache TTL contract"
