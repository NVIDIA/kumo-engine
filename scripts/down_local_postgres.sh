#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

CONTAINER="${KUMO_LOCAL_POSTGRES_CONTAINER:-kumo-local-postgres}"
DATA_DIR="${KUMO_LOCAL_POSTGRES_DATA_DIR:-}"
INIT_DIR="${KUMO_LOCAL_POSTGRES_INIT_DIR:-$REPO_ROOT/.cache/kumo/local-postgres/initdb}"
DEFAULT_DEVSERVER_HOME="${KUMO_HOME:-$REPO_ROOT/.cache/kumo/devserver-postgres}"
SECRETS_DIR="${KUMO_LOCAL_POSTGRES_SECRETS_DIR:-$DEFAULT_DEVSERVER_HOME/secrets}"
DEVSERVER_HOME="${KUMO_LOCAL_POSTGRES_DEVSERVER_HOME:-$DEFAULT_DEVSERVER_HOME}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

normalize_path() {
    local path="$1"
    local dir
    local base
    case "$path" in
        /*) ;;
        *) path="$REPO_ROOT/$path" ;;
    esac
    dir="$(dirname "$path")"
    base="$(basename "$path")"
    mkdir -p "$dir"
    dir="$(cd "$dir" && pwd -P)"
    printf '%s/%s\n' "$dir" "$base"
}

safe_rm_dir() {
    local path="$1"
    local abs
    [ -n "$path" ] || return 0
    abs="$(normalize_path "$path")"
    case "$abs" in
        "$REPO_ROOT/.cache/kumo"|"$REPO_ROOT/.cache/kumo"/*)
            rm -rf -- "$abs"
            ;;
        /tmp/kumo-local-postgres|/tmp/kumo-local-postgres/*)
            rm -rf -- "$abs"
            ;;
        *)
            if [ "${KUMO_LOCAL_POSTGRES_ALLOW_EXTERNAL_DATA_DIR:-0}" = "1" ]; then
                rm -rf -- "$abs"
            else
                fail "refusing to remove non-cache path $abs; set KUMO_LOCAL_POSTGRES_ALLOW_EXTERNAL_DATA_DIR=1 to allow it"
            fi
            ;;
    esac
}

clean_devserver_state() {
    local child
    for child in artifacts data-persist-dir diskgraph-data jobs kumo-home pool uploaded-datasets; do
        safe_rm_dir "$DEVSERVER_HOME/$child"
    done
}

case "$CONTAINER" in
    ''|*[!A-Za-z0-9_.-]*)
        fail "KUMO_LOCAL_POSTGRES_CONTAINER contains unsupported characters: $CONTAINER"
        ;;
esac

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

if [ "${KUMO_LOCAL_POSTGRES_KEEP_DATA:-0}" != "1" ]; then
    safe_rm_dir "$DATA_DIR"
    clean_devserver_state
fi
safe_rm_dir "$INIT_DIR"
safe_rm_dir "$SECRETS_DIR"

echo "Local Postgres container $CONTAINER is stopped"
