#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTALLER="${ROOT}/bin/kumo"
CONFIG="${ROOT}/config.yaml"

usage() {
  cat <<'EOF'
Usage:
  bash install.sh             Install or reconcile the complete stack
  bash install.sh platform    Update only the selected platform chart
  bash install.sh temporal    Update only the Temporal chart
  bash install.sh core        Update only the Kumo core chart

Edit config.yaml before running this script. Component updates use that same
configuration and the editable chart sources in this directory.
EOF
}

[[ -r "${INSTALLER}" ]] || {
  printf 'error: installer is missing or unreadable: %s\n' "${INSTALLER}" >&2
  exit 1
}
[[ -r "${CONFIG}" ]] || {
  printf 'error: configuration is missing or unreadable: %s\n' "${CONFIG}" >&2
  exit 1
}

case "${1:-}" in
  "")
    exec bash "${INSTALLER}" install --values "${CONFIG}"
    ;;
  platform|temporal|core)
    [[ $# -eq 1 ]] || {
      usage >&2
      exit 2
    }
    exec bash "${INSTALLER}" install "$1" --values "${CONFIG}"
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
