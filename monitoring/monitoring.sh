#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Set the current working directory to the script's directory
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO="kumo-grafana"
KUMO_GRAFANA_REF="${KUMO_GRAFANA_REF:-main}"
KUMO_GRAFANA_HOME="${KUMO_GRAFANA_HOME:-$(pwd)/$REPO}"
KUMO_GRAFANA_REPO_URL="${KUMO_GRAFANA_REPO_URL:-}"

function abs_path() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

# stop all containers when this script exits
function teardown() {
  if [ -x "$KUMO_GRAFANA_HOME/stop.sh" ]; then
    "$KUMO_GRAFANA_HOME/stop.sh"
  fi
}
trap teardown EXIT

set -a
: ${GRAFANA_PORT:=3000}
: ${REST_API_PORT:=10001}
: ${PROMETHEUS_PORT:=9090}
#: ${NODE_EXPORTER_PORT:=9100}
: ${JAEGER_OTLP_PORT:=4317}
: ${JAEGER_UI_PORT:=16686}
set +a

if ! [ -d "$KUMO_GRAFANA_HOME" ]; then
  if [ -z "$KUMO_GRAFANA_REPO_URL" ]; then
    echo "Set KUMO_GRAFANA_HOME to a compatible local checkout or set KUMO_GRAFANA_REPO_URL." >&2
    exit 2
  fi
  mkdir -p "$(dirname "$KUMO_GRAFANA_HOME")"
  git clone "$KUMO_GRAFANA_REPO_URL" "$KUMO_GRAFANA_HOME"
fi

git -C "$KUMO_GRAFANA_HOME" fetch origin "$KUMO_GRAFANA_REF"
git -C "$KUMO_GRAFANA_HOME" checkout --detach FETCH_HEAD
git -C "$KUMO_GRAFANA_HOME" reset --hard FETCH_HEAD

KUMO_GRAFANA_SKIP_FORWARDING_INSTRUCTIONS=1 "$KUMO_GRAFANA_HOME/start.sh"
tail -f /dev/null
