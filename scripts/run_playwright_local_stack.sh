#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="$REPO_ROOT/ui"
ENV_FILE="$REPO_ROOT/docker/compose/.env.compose-playwright-ci"
LOG_DIR="${PLAYWRIGHT_LOCAL_LOG_DIR:-/tmp/kumo-playwright/logs}"

REUSE_STACK=0
KEEP_STACK=0
SKIP_BUILD=0
SKIP_SEED=0
SKIP_TESTS=0
PLAYWRIGHT_ARGS=()
BASE_URL="${BASE_URL:-}"
STACK_STARTED=0
SCRIPT_START_SECONDS=$SECONDS
TOTAL_TIMEOUT_SECONDS="${PLAYWRIGHT_LOCAL_STACK_TOTAL_TIMEOUT_SECONDS:-3600}"

usage() {
    cat <<'USAGE'
Usage: scripts/run_playwright_local_stack.sh [options]

Options:
  --reuse-stack        Reuse the currently running compose stack.
  --keep-stack         Leave the stack running after tests.
  --base-url URL       Override Playwright base URL.
  --skip-build         Reuse existing local Docker images.
  --skip-seed          Do not seed fixture data.
  --skip-tests         Start/seed only.
  -- ARGS              Pass remaining arguments to Playwright.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reuse-stack) REUSE_STACK=1 ;;
        --keep-stack) KEEP_STACK=1 ;;
        --base-url)
            BASE_URL="$2"
            shift
            ;;
        --skip-build) SKIP_BUILD=1 ;;
        --skip-seed) SKIP_SEED=1 ;;
        --skip-tests) SKIP_TESTS=1 ;;
        --)
            shift
            PLAYWRIGHT_ARGS=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

mkdir -p "$LOG_DIR"

CALLER_STACK_SET="${STACK+x}"
CALLER_STACK="${STACK-}"
CALLER_PORT_OFFSET_SET="${PORT_OFFSET+x}"
CALLER_PORT_OFFSET="${PORT_OFFSET-}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    . "$ENV_FILE"
    set +a
fi

if [[ -n "$CALLER_STACK_SET" ]]; then
    STACK="$CALLER_STACK"
fi
if [[ -n "$CALLER_PORT_OFFSET_SET" ]]; then
    PORT_OFFSET="$CALLER_PORT_OFFSET"
fi

STACK="${STACK:-playwright}"
PORT_OFFSET="${PORT_OFFSET:-10000}"
REST_API_PORT="${REST_API_PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$((8888 + PORT_OFFSET))/}"
BASE_URL="${BASE_URL%/}"

collect_diagnostics() {
    local status=$?
    mkdir -p "$LOG_DIR"
    if command -v docker >/dev/null 2>&1; then
        timeout --kill-after=10s 30s \
            docker ps > "$LOG_DIR/docker-ps.txt" 2>&1 || true
        diagnostics_timeout="${PLAYWRIGHT_DIAGNOSTICS_TIMEOUT_SECONDS:-180}"
        timeout --kill-after=30s "${diagnostics_timeout}s" \
            make -C "$REPO_ROOT" compose-local-logs-once \
            > "$LOG_DIR/compose-logs.txt" 2>&1 || true
    fi
    return "$status"
}

finish() {
    local status=$?
    if [[ "$status" -ne 0 || "${PLAYWRIGHT_COLLECT_DIAGNOSTICS_ALWAYS:-0}" == "1" ]]; then
        collect_diagnostics || true
    fi
    if [[ "$STACK_STARTED" == "1" && "$KEEP_STACK" != "1" && "$SKIP_TESTS" != "1" ]]; then
        teardown_timeout="${PLAYWRIGHT_COMPOSE_TEARDOWN_TIMEOUT_SECONDS:-300}"
        timeout --kill-after=30s "${teardown_timeout}s" \
            make -C "$REPO_ROOT" compose-local-down || true
    fi
    exit "$status"
}

trap finish EXIT

bounded_phase_timeout() {
    local requested="$1"
    local phase="$2"
    local elapsed=$((SECONDS - SCRIPT_START_SECONDS))
    local remaining=$((TOTAL_TIMEOUT_SECONDS - elapsed))

    if (( remaining <= 0 )); then
        echo "ERROR: Playwright local stack exceeded total timeout ${TOTAL_TIMEOUT_SECONDS}s before ${phase}" >&2
        exit 124
    fi
    if (( requested > remaining )); then
        echo "$remaining"
    else
        echo "$requested"
    fi
}

export AUTH_DISABLED=true
export BASE_URL="$BASE_URL/"
export PLAYWRIGHT_HTML_OPEN="${PLAYWRIGHT_HTML_OPEN:-never}"
export COMPOSE_LOCAL_BASE_ENV_FILE="$ENV_FILE"
export STACK
export PORT_OFFSET
export KUMO_IMAGE_TAG="${KUMO_IMAGE_TAG:-local}"
export KUMO_BACKEND_VARIANT="${KUMO_BACKEND_VARIANT:-cpu}"
export BUILD_GPU="${BUILD_GPU:-0}"
export COMPOSE_LOCAL_CLOUD_PROVIDER="${COMPOSE_LOCAL_CLOUD_PROVIDER:-unknown}"
export COMPOSE_LOCAL_GLOVE_LEVEL="${COMPOSE_LOCAL_GLOVE_LEVEL:-none}"

echo "==> Playwright local stack total timeout ${TOTAL_TIMEOUT_SECONDS}s"

# compose-local's local dataplane fallback resolves to this stack-scoped mount.
# Pre-create the Playwright CSV fixture and writable leaves before the non-root
# backend containers touch the bind mount.
PLAYWRIGHT_DATAPLANE_ROOT="$REPO_ROOT/.cache/kumo/compose-local/dataplane/$STACK"
PLAYWRIGHT_CSV_HOST_ROOT="$PLAYWRIGHT_DATAPLANE_ROOT/csv-data"
mkdir -p \
    "$PLAYWRIGHT_DATAPLANE_ROOT/$STACK/fixtures" \
    "$PLAYWRIGHT_DATAPLANE_ROOT/artifacts/$STACK" \
    "$PLAYWRIGHT_DATAPLANE_ROOT/$STACK/data-persist-dir/data_backend_cache" \
    "$PLAYWRIGHT_CSV_HOST_ROOT"
cp -R "$REPO_ROOT/test/csv_data/." "$PLAYWRIGHT_CSV_HOST_ROOT/"
chmod -R 777 "$PLAYWRIGHT_DATAPLANE_ROOT" 2>/dev/null || true

if [[ "$REUSE_STACK" != "1" ]]; then
    compose_down_timeout="${PLAYWRIGHT_COMPOSE_DOWN_TIMEOUT_SECONDS:-300}"
    compose_down_timeout="$(bounded_phase_timeout "$compose_down_timeout" "compose-local-down-volumes")"
    echo "==> Removing existing local Playwright stack (timeout ${compose_down_timeout}s)"
    timeout --kill-after=30s "${compose_down_timeout}s" \
        make -C "$REPO_ROOT" compose-local-down-volumes || true
    if [[ "$SKIP_BUILD" == "1" ]]; then
        export SKIP_COMPOSE_LOCAL_BUILD=1
    fi
    compose_start_timeout="${PLAYWRIGHT_COMPOSE_START_TIMEOUT_SECONDS:-1800}"
    compose_start_timeout="$(bounded_phase_timeout "$compose_start_timeout" "compose-local")"
    echo "==> Starting local Playwright stack (timeout ${compose_start_timeout}s)"
    set +e
    timeout --kill-after=60s "${compose_start_timeout}s" \
        make -C "$REPO_ROOT" compose-local
    compose_status=$?
    set -e
    if [[ "$compose_status" -eq 124 || "$compose_status" -eq 137 ]]; then
        echo "ERROR: compose-local timed out after ${compose_start_timeout}s" >&2
    fi
    if [[ "$compose_status" -ne 0 ]]; then
        exit "$compose_status"
    fi
    STACK_STARTED=1
fi

echo "==> Waiting for local Kumo at $BASE_URL"
bootstrap_timeout="${PLAYWRIGHT_LOCAL_BOOTSTRAP_TIMEOUT:-1200}"
bootstrap_timeout="$(bounded_phase_timeout "$bootstrap_timeout" "local Kumo readiness")"
deadline=$((SECONDS + bootstrap_timeout))
while (( SECONDS < deadline )); do
    tmp="$(mktemp)"
    code="$(curl -s -o "$tmp" -m 15 -w "%{http_code}" "$BASE_URL/v0/authentication/client-config" || echo 000)"
    auth_disabled="$(python3 -c 'import json, sys; print(str(json.load(open(sys.argv[1])).get("auth_disabled")).lower())' "$tmp" 2>/dev/null || true)"
    rm -f "$tmp"
    echo "  client-config endpoint -> HTTP $code, auth_disabled=${auth_disabled:-unknown}"
    if [[ "$code" == "200" && "$auth_disabled" == "true" ]]; then
        break
    fi
    sleep 10
done
if (( SECONDS >= deadline )); then
    echo "ERROR: local Kumo did not become ready before timeout" >&2
    exit 1
fi

if [[ "$SKIP_SEED" != "1" ]]; then
    rest_container="${STACK}-rest"
    seed_training_timeout="${PLAYWRIGHT_SEED_TRAINING_TIMEOUT_SECONDS:-1200}"
    seed_exec_timeout="${PLAYWRIGHT_SEED_EXEC_TIMEOUT_SECONDS:-$((seed_training_timeout + 300))}"
    seed_exec_timeout="$(bounded_phase_timeout "$seed_exec_timeout" "Playwright local stack seed")"
    seed_copy_timeout="${PLAYWRIGHT_SEED_COPY_TIMEOUT_SECONDS:-60}"
    seed_copy_timeout="$(bounded_phase_timeout "$seed_copy_timeout" "Playwright local stack seed fixture copy")"
    timeout --kill-after=10s "${seed_copy_timeout}s" \
        docker cp "$REPO_ROOT/test/rest/integ/test_cases/test_e2e_fixtures.json" "$rest_container:/tmp/playwright-e2e-fixtures.json"
    seed_copy_timeout="${PLAYWRIGHT_SEED_COPY_TIMEOUT_SECONDS:-60}"
    seed_copy_timeout="$(bounded_phase_timeout "$seed_copy_timeout" "Playwright local stack seed connector copy")"
    timeout --kill-after=10s "${seed_copy_timeout}s" \
        docker cp "$REPO_ROOT/test/rest/integ/connector_args/test_e2e_connector_args.json" "$rest_container:/tmp/playwright-e2e-connectors.json"
    seed_copy_timeout="${PLAYWRIGHT_SEED_COPY_TIMEOUT_SECONDS:-60}"
    seed_copy_timeout="$(bounded_phase_timeout "$seed_copy_timeout" "Playwright local stack seed script copy")"
    timeout --kill-after=10s "${seed_copy_timeout}s" \
        docker cp "$REPO_ROOT/scripts/seed_playwright_local_stack.py" "$rest_container:/tmp/seed_playwright_local_stack.py"

    echo "==> Seeding local Playwright stack (timeout ${seed_exec_timeout}s)"
    set +e
    timeout --kill-after=30s "${seed_exec_timeout}s" \
    docker exec \
        -e "PLAYWRIGHT_SEED_BASE_URL=http://localhost:${REST_API_PORT}" \
        -e "PLAYWRIGHT_SEED_CONNECTORS=/tmp/playwright-e2e-connectors.json" \
        -e "PLAYWRIGHT_SEED_CSV_ROOT=$PLAYWRIGHT_CSV_HOST_ROOT" \
        -e "PLAYWRIGHT_SEED_TRAINING_TIMEOUT_SECONDS=$seed_training_timeout" \
        "$rest_container" \
        python3 /tmp/seed_playwright_local_stack.py
    seed_status=$?
    set -e
    if [[ "$seed_status" -eq 124 || "$seed_status" -eq 137 ]]; then
        echo "ERROR: Playwright local stack seed command timed out after ${seed_exec_timeout}s" >&2
        exit "$seed_status"
    elif [[ "$seed_status" -ne 0 ]]; then
        echo "ERROR: Playwright local stack seed command failed with status $seed_status" >&2
        exit "$seed_status"
    fi
fi

if [[ "$SKIP_TESTS" == "1" ]]; then
    echo "==> Local Playwright stack is ready at $BASE_URL"
    exit 0
fi

cd "$UI_DIR"
playwright_test_timeout="${PLAYWRIGHT_TEST_COMMAND_TIMEOUT_SECONDS:-1800}"
playwright_test_timeout="$(bounded_phase_timeout "$playwright_test_timeout" "Playwright test command")"
playwright_reporter="${PLAYWRIGHT_REPORTER:-html,json}"
echo "==> Running Playwright tests (timeout ${playwright_test_timeout}s)"
set +e
if (( ${#PLAYWRIGHT_ARGS[@]} > 0 )); then
    timeout --kill-after=30s "${playwright_test_timeout}s" \
        node node_modules/@playwright/test/cli.js test --reporter="$playwright_reporter" "${PLAYWRIGHT_ARGS[@]}"
else
    timeout --kill-after=30s "${playwright_test_timeout}s" \
        node node_modules/@playwright/test/cli.js test --reporter="$playwright_reporter"
fi
playwright_status=$?
set -e
if [[ "$playwright_status" -eq 124 || "$playwright_status" -eq 137 ]]; then
    echo "ERROR: Playwright test command timed out after ${playwright_test_timeout}s" >&2
fi
exit "$playwright_status"
