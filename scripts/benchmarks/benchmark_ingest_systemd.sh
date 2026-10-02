#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
UNIT_TEMPLATE="${SCRIPT_DIR}/benchmark_ingest.service"
UNIT_NAME="benchmark_ingest.service"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
UNIT_DEST="${SYSTEMD_USER_DIR}/${UNIT_NAME}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

install_unit() {
  require_cmd systemctl
  mkdir -p "${SYSTEMD_USER_DIR}"
  sed "s|__REPO_ROOT__|${REPO_ROOT}|g" "${UNIT_TEMPLATE}" > "${UNIT_DEST}"
  systemctl --user daemon-reload
  systemctl --user enable --now "${UNIT_NAME}"
  systemctl --user status "${UNIT_NAME}" --no-pager
}

case "${1:-}" in
  install)
    install_unit
    ;;
  status|start|stop|restart|enable|disable)
    require_cmd systemctl
    systemctl --user "$1" "${UNIT_NAME}"
    ;;
  *)
    cat <<USAGE
Usage:
  ${0##*/} install
  ${0##*/} status|start|stop|restart|enable|disable
USAGE
    exit 2
    ;;
 esac
