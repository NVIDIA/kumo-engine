#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
UI_DIR="$ROOT/ui"

status() {
  local label="$1"
  local value="$2"
  printf '%s\t%s\n' "$label" "$value"
}

probe_dev_server() {
  if [[ "${BUILD_UI_DEV_SERVER_NOT_NEEDED:-}" == "true" ]]; then
    status "dev-server" "not-needed"
    return
  fi
  if [[ -n "${BUILD_UI_DEV_SERVER_URL:-}" ]]; then
    if curl -fsS --max-time 2 "$BUILD_UI_DEV_SERVER_URL" >/dev/null 2>&1; then
      status "dev-server" "available"
    else
      status "dev-server" "unavailable"
    fi
  else
    status "dev-server" "unknown"
  fi
}

probe_backend() {
  if [[ "${BUILD_UI_BACKEND_NOT_NEEDED:-}" == "true" ]]; then
    status "backend" "not-needed"
    return
  fi
  if [[ -n "${BUILD_UI_BACKEND_URL:-}" ]]; then
    if curl -fsS --max-time 2 "$BUILD_UI_BACKEND_URL" >/dev/null 2>&1; then
      status "backend" "available"
    else
      status "backend" "unavailable"
    fi
  else
    status "backend" "unknown"
  fi
}

probe_browser() {
  if [[ "${BUILD_UI_BROWSER_NOT_NEEDED:-}" == "true" ]]; then
    status "browser-tooling" "not-needed"
    return
  fi
  if [[ -n "${CI_SCREENSHOT_SCRIPT:-}" ]]; then
    status "browser-tooling" "available"
  else
    status "browser-tooling" "unknown"
  fi
}

probe_runtime_config() {
  if [[ "${BUILD_UI_RUNTIME_CONFIG_NOT_NEEDED:-}" == "true" ]]; then
    status "runtime-config" "not-needed"
    return
  fi
  local env_file="$UI_DIR/env.js"
  if [[ -f "$env_file" ]]; then
    if grep -Eq "AUTH_DISABLED[[:space:]]*:[[:space:]]*true" "$env_file"; then
      status "runtime-config" "available"
    elif grep -Eq "AUTH_DISABLED[[:space:]]*:[[:space:]]*(false|__AUTH_DISABLED__)" "$env_file"; then
      status "runtime-config" "unavailable"
    else
      status "runtime-config" "unknown"
    fi
  else
    status "runtime-config" "unknown"
  fi
}

probe_catalog() {
  if [[ "${BUILD_UI_CATALOG_NOT_NEEDED:-}" == "true" ]]; then
    status "catalog-package" "not-needed"
    return
  fi
  if [[ -d "$UI_DIR/node_modules" ]]; then
    status "catalog-package" "available"
  else
    status "catalog-package" "unknown"
  fi
}

main() {
  probe_dev_server
  probe_backend
  probe_browser
  probe_runtime_config
  probe_catalog
}

main "$@"
