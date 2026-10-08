#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON_BIN=""
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    DEFAULT_PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    DEFAULT_PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
if [ -z "$PYTHON_BIN" ]; then
    echo "No python interpreter found on PATH. Install python3 or set PYTHON_BIN." >&2
    exit 1
fi

# The official installer uses this stable user-local path. Reuse it on later
# runs without requiring the user to modify a shell startup file.
if ! command -v temporal >/dev/null 2>&1 && [ -x "$HOME/.temporalio/bin/temporal" ]; then
    export PATH="$PATH:$HOME/.temporalio/bin"
fi

TEMPORAL_PORT="${TEMPORAL_PORT:-7233}"
TEMPORAL_UI_PORT="${TEMPORAL_UI_PORT:-$((TEMPORAL_PORT + 1000))}"
TEMPORAL_NAMESPACE="${TEMPORAL_NAMESPACE:-devtest}"
TEMPORAL_SESSION_NAME="${TEMPORAL_SESSION_NAME:-local_temporal}"
TEMPORAL_ADDRESS="${TEMPORAL_ADDRESS:-127.0.0.1:${TEMPORAL_PORT}}"
TEMPORAL_DB_FILENAME="${TEMPORAL_DB_FILENAME:-}"
TEMPORAL_READY_TIMEOUT="${TEMPORAL_READY_TIMEOUT:-120}"

if DEPLOY_STAGE=dev TEMPORAL_SERVER=localhost TEMPORAL_ADDRESS="$TEMPORAL_ADDRESS" TEMPORAL_NAMESPACE="$TEMPORAL_NAMESPACE" "$PYTHON_BIN" temporalio_commons/wait_server_ready.py -t 1; then
    echo "temporal server is already running at ${TEMPORAL_ADDRESS}"
else
    if ! command -v temporal &> /dev/null; then
        echo "Install Temporal CLI, see https://docs.temporal.io/cli"
        temporal_installer_commit="daa3f466fcb5cc37d5a1081a73106523299e0a56"
        temporal_cli_version="1.9.1"
        for i in 1 2 3 4 5; do
            curl -sSf \
                "https://raw.githubusercontent.com/temporalio/cli/${temporal_installer_commit}/install.sh" && break
            sleep 5
        done | sh -s -- --version "${temporal_cli_version}"
        export PATH=$PATH:$HOME/.temporalio/bin
        SHELL_NAME="$(basename "${SHELL:-bash}")"
        case "$SHELL_NAME" in
            zsh) SHELL_RC="$HOME/.zshrc" ;;
            bash) SHELL_RC="$HOME/.bashrc" ;;
            *) SHELL_RC="your shell startup file" ;;
        esac
        echo "Temporal CLI was installed in $HOME/.temporalio/bin."
        echo "To make it available in future shells, add this to ${SHELL_RC}:"
        echo '  export PATH="$PATH:$HOME/.temporalio/bin"'
    else
        echo "Temporal CLI is already installed"
    fi

    temporal_command=(
        temporal server start-dev
        --namespace "$TEMPORAL_NAMESPACE"
        --port "$TEMPORAL_PORT"
        --ui-port "$TEMPORAL_UI_PORT"
    )
    if [ -n "$TEMPORAL_DB_FILENAME" ]; then
        mkdir -p "$(dirname "$TEMPORAL_DB_FILENAME")"
        temporal_command+=(--db-filename "$TEMPORAL_DB_FILENAME")
    fi

    tmux kill-session -t "$TEMPORAL_SESSION_NAME" || true
    tmux new-session -d -s "$TEMPORAL_SESSION_NAME"
    printf -v temporal_command_string "%q " "${temporal_command[@]}"
    tmux send-keys -t "$TEMPORAL_SESSION_NAME" \
        "$temporal_command_string; status=\$?; echo TEMPORAL_SERVER_EXIT_STATUS=\$status; sleep 300; exit \$status" \
        ENTER
    echo "Local Temporal Cluster is up and running in '${TEMPORAL_SESSION_NAME}' tmux session at service port=${TEMPORAL_PORT}; web UI port=${TEMPORAL_UI_PORT}"
    if ! DEPLOY_STAGE=dev TEMPORAL_SERVER=localhost TEMPORAL_ADDRESS="$TEMPORAL_ADDRESS" TEMPORAL_NAMESPACE="$TEMPORAL_NAMESPACE" "$PYTHON_BIN" temporalio_commons/wait_server_ready.py -t "$TEMPORAL_READY_TIMEOUT"; then
        echo "Temporal did not become ready. Last '${TEMPORAL_SESSION_NAME}' pane output:" >&2
        tmux capture-pane -p -S -200 -t "$TEMPORAL_SESSION_NAME" >&2 || true
        exit 1
    fi
fi
