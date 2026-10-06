#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

CONTAINER="${KUMO_LOCAL_POSTGRES_CONTAINER:-kumo-local-postgres}"
IMAGE="${KUMO_LOCAL_POSTGRES_IMAGE:-postgres:16-alpine}"
HOST="${KUMO_LOCAL_POSTGRES_HOST:-127.0.0.1}"
PORT="${KUMO_LOCAL_POSTGRES_PORT:-15432}"
USER_NAME="${POSTGRES_USER:-kumo}"
PASSWORD="${POSTGRES_PASSWORD:-kumo}"
STATE_DB_NAME="${KUMO_STATE_DB_NAME:-kumo}"
RESET="${KUMO_LOCAL_POSTGRES_RESET:-1}"
TMPFS_SIZE="${KUMO_LOCAL_POSTGRES_TMPFS_SIZE:-512m}"
DATA_DIR="${KUMO_LOCAL_POSTGRES_DATA_DIR:-}"
DEFAULT_DEVSERVER_HOME="${KUMO_HOME:-$REPO_ROOT/.cache/kumo/devserver-postgres}"
SECRETS_DIR="${KUMO_LOCAL_POSTGRES_SECRETS_DIR:-$DEFAULT_DEVSERVER_HOME/secrets}"
DEVSERVER_HOME="${KUMO_LOCAL_POSTGRES_DEVSERVER_HOME:-$DEFAULT_DEVSERVER_HOME}"
CHECK_NETWORK="${KUMO_POSTGRES_DOCKER_NETWORK:-host}"

if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
fi

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

validate_identifier() {
    local label="$1"
    local value="$2"
    case "$value" in
        ''|*[!A-Za-z0-9_]*)
            fail "$label must contain only letters, numbers, and underscore: $value"
            ;;
    esac
}

validate_container_name() {
    case "$CONTAINER" in
        ''|*[!A-Za-z0-9_.-]*)
            fail "KUMO_LOCAL_POSTGRES_CONTAINER contains unsupported characters: $CONTAINER"
            ;;
    esac
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

wait_for_postgres() {
    local attempt
    for attempt in $(seq 1 60); do
        if docker exec "$CONTAINER" pg_isready -h 127.0.0.1 -p 5432 -U "$USER_NAME" -d "$STATE_DB_NAME" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    docker logs "$CONTAINER" >&2 || true
    fail "Postgres did not become ready in $CONTAINER"
}

verify_database_exists() {
    local db_name="$1"
    local exists
    exists="$(
        docker exec -e "PGPASSWORD=$PASSWORD" "$CONTAINER" \
            psql -h 127.0.0.1 -p 5432 -U "$USER_NAME" -d "$STATE_DB_NAME" -tAc \
            "SELECT 1 FROM pg_database WHERE datname = '$db_name'"
    )"
    [ "$exists" = "1" ] || fail "database was not created: $db_name"
}

write_secrets() {
    safe_rm_dir "$SECRETS_DIR"
    mkdir -p "$SECRETS_DIR"
    STATE_DB_DSN_FILE="$SECRETS_DIR/STATE_DB_DSN" \
    POSTGRES_HOST="$HOST" \
    POSTGRES_PORT="$PORT" \
    POSTGRES_USER="$USER_NAME" \
    POSTGRES_PASSWORD="$PASSWORD" \
    KUMO_STATE_DB_NAME="$STATE_DB_NAME" \
        "$PYTHON_BIN" "$REPO_ROOT/deploy/shared/scripts/write-postgres-secrets.py"
    chmod 600 "$SECRETS_DIR/STATE_DB_DSN"
}

validate_grants() {
    KUMO_POSTGRES_DOCKER_NETWORK="$CHECK_NETWORK" \
    KUMO_POSTGRES_PYTHON_BIN="$PYTHON_BIN" \
        "$REPO_ROOT/deploy/shared/scripts/check-postgres-dsns.sh" \
        "$SECRETS_DIR/STATE_DB_DSN"
}

validate_container_name
validate_identifier "POSTGRES_USER" "$USER_NAME"
validate_identifier "KUMO_STATE_DB_NAME" "$STATE_DB_NAME"
case "$PORT" in
    ''|*[!0-9]*) fail "KUMO_LOCAL_POSTGRES_PORT must be numeric: $PORT" ;;
esac
[ "$PORT" -ge 1024 ] || fail "KUMO_LOCAL_POSTGRES_PORT must be >= 1024"
[ -n "$PYTHON_BIN" ] || fail "No python interpreter found. Set PYTHON_BIN."

if [ "$RESET" = "1" ]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    safe_rm_dir "$SECRETS_DIR"
    clean_devserver_state
    if [ -n "$DATA_DIR" ]; then
        safe_rm_dir "$DATA_DIR"
    fi
elif [ "$RESET" != "0" ]; then
    fail "KUMO_LOCAL_POSTGRES_RESET must be 0 or 1"
fi

if docker ps -q -f "name=^/${CONTAINER}$" | grep -q .; then
    echo "Local Postgres container $CONTAINER is already running"
else
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    MOUNT_ARGS=()
    if [ -n "$DATA_DIR" ]; then
        DATA_DIR="$(normalize_path "$DATA_DIR")"
        mkdir -p "$DATA_DIR"
        MOUNT_ARGS+=(-v "$DATA_DIR:/var/lib/postgresql/data")
    else
        MOUNT_ARGS+=(--tmpfs "/var/lib/postgresql/data:rw,size=$TMPFS_SIZE")
    fi

    docker run -d \
        --name "$CONTAINER" \
        -p "$HOST:$PORT:5432" \
        -e "POSTGRES_USER=$USER_NAME" \
        -e "POSTGRES_PASSWORD=$PASSWORD" \
        -e "POSTGRES_DB=$STATE_DB_NAME" \
        "${MOUNT_ARGS[@]}" \
        "$IMAGE" >/dev/null
fi

wait_for_postgres
verify_database_exists "$STATE_DB_NAME"
write_secrets
validate_grants

echo "Local Postgres is ready on $HOST:$PORT using container $CONTAINER"
echo "Generated DSN secrets under $SECRETS_DIR"
