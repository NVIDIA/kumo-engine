#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the UI development server in docker.
#
# Usage:
#   bash ./scripts/run_ui_devserver.sh <ui_port_number> <api_port_number> <label> [ui_catalog_path] [node_modules_volume]
#
# Examples:
#   bash ./scripts/run_ui_devserver.sh 10000 10001 devserver
#   bash ./scripts/run_ui_devserver.sh 10000 10001 devserver packages/kumo-ui-catalog

# This script needs to be run from the root of the Kumo repository.

set -xe

if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "Usage: ./scripts/run_ui_devserver.sh <ui_port_number> <api_port_number> <label> [ui_catalog_path] [node_modules_volume]"
    false
fi

UI_PORT_NUMBER=$1
API_PORT_NUMBER=$2
LABEL=$3
UI_CATALOG_PATH="${4:-}"
UI_NODE_MODULES_VOLUME="${5:-}"
UI_CATALOG_HOST_PATH="$(cd -- "${UI_CATALOG_PATH:-packages/kumo-ui-catalog}" && pwd)"
UI_NODE_MODULES_MOUNT=()

if [ -n "$UI_NODE_MODULES_VOLUME" ]; then
  if [[ ! "$UI_NODE_MODULES_VOLUME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo "Invalid UI node_modules volume name: $UI_NODE_MODULES_VOLUME"
    exit 2
  fi
  docker volume create "$UI_NODE_MODULES_VOLUME" >/dev/null
  UI_NODE_MODULES_MOUNT=(-v "$UI_NODE_MODULES_VOLUME:/src/node_modules")
fi

UI_LOCK_HASH="$(cat ui/package-lock.json "$UI_CATALOG_HOST_PATH/package-lock.json" | sha256sum | awk '{print $1}')"

if [ "$(uname -s)" = "Darwin" ] && [ "${KUMO_UI_DEVSERVER_DOCKER:-0}" != "1" ]; then
  pushd ui
  npm --prefix "$UI_CATALOG_HOST_PATH" ci --legacy-peer-deps
  npm --prefix "$UI_CATALOG_HOST_PATH" run build
  npm ci

  NPM_SCRIPT="start"
  if [ -n "$UI_CATALOG_PATH" ]; then
    export UI_CATALOG_PATH="$UI_CATALOG_HOST_PATH"
    NPM_SCRIPT="start:catalog"
  fi

  unset API_TARGET UI_REMOTE
  export UI_LOCAL=true
  export API_PROXY_TARGET="http://127.0.0.1:${API_PORT_NUMBER}"
  export REACT_APP_BASE_URL_DEV="/"
  export BROWSER=none
  export REACT_APP_PORT="${UI_PORT_NUMBER}"
  export USE_POLLING=true
  npm run "$NPM_SCRIPT"
  popd
  exit 0
fi

# https://hub.docker.com/_/node
IMAGE=node:22

# Make sure we don't point docker-cli to minikube's internal docker engine
eval $(minikube docker-env -u)

docker run \
  -v "$(pwd)/ui:/src" -w /src \
  -v "$UI_CATALOG_HOST_PATH:/packages/kumo-ui-catalog" \
  "${UI_NODE_MODULES_MOUNT[@]}" \
  -ti --rm \
  -l $LABEL \
  -e "KUMO_UI_LOCK_HASH=$UI_LOCK_HASH" \
  $IMAGE bash -e -c '
  npm run install:catalog
  if [ -f node_modules/.kumo-package-lock-sha256 ] \
      && [ "$(cat node_modules/.kumo-package-lock-sha256)" = "$KUMO_UI_LOCK_HASH" ]; then
    echo "Reusing node_modules matching ui/package-lock.json"
    exit 0
  fi

  npm ci && printf "%s\\n" "$KUMO_UI_LOCK_HASH" > node_modules/.kumo-package-lock-sha256'

UI_CATALOG_ENV=""
NPM_SCRIPT="start"

if [ -n "$UI_CATALOG_PATH" ]; then
    UI_CATALOG_ENV="-e UI_CATALOG_PATH=/packages/kumo-ui-catalog"
    NPM_SCRIPT="start:catalog"
fi

docker run \
  -v "$(pwd)/ui:/src" -w /src \
  -v "$UI_CATALOG_HOST_PATH:/packages/kumo-ui-catalog" \
  "${UI_NODE_MODULES_MOUNT[@]}" \
  -ti --rm \
  -l $LABEL \
  --add-host=host.docker.internal:host-gateway \
  -p $UI_PORT_NUMBER:$UI_PORT_NUMBER \
  -e "UI_LOCAL=true" \
  -e "API_PROXY_TARGET=http://host.docker.internal:${API_PORT_NUMBER}" \
  -e "REACT_APP_BASE_URL_DEV=/" \
  -e "BROWSER=none" \
  -e "REACT_APP_PORT=${UI_PORT_NUMBER}" \
  -e "USE_POLLING=true" \
  $UI_CATALOG_ENV \
  $IMAGE npm run $NPM_SCRIPT
