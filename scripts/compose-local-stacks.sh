#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Manage N compose-local stacks (kumo1..kumoN).
#
# Usage:
#   compose-local-stacks.sh [--no-auth] <command> <n>
#
# Options:
#   --no-auth  Disable auth (sets DISABLE_AUTH=true)
#
# Commands:
#   up     - build (if needed) and start all stacks
#   down   - stop all stacks
#   logs   - tail logs from all stacks (one tmux pane per stack, or serial)
#   status - show running containers for all stacks
#
# Examples:
#   ./scripts/compose-local-stacks.sh up 4
#   ./scripts/compose-local-stacks.sh --no-auth up 4
#   ./scripts/compose-local-stacks.sh down 4
#
# Pass extra env vars before the script as usual:
#   LOCAL_DISKGRAPH_DATA=/fast/ssd ./scripts/compose-local-stacks.sh up 4

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    sed -n '/^# Usage/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 1
}

NO_AUTH=0
if [[ "${1:-}" == "--no-auth" ]]; then
    NO_AUTH=1
    shift
fi

[[ $# -lt 2 ]] && usage

CMD="$1"
N="$2"

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: <n> must be a positive integer, got: $N" >&2
    exit 1
fi

stacks=()
for i in $(seq 1 "$N"); do
    stacks+=("kumo$i")
done

run_stack() {
    local stack="$1"; shift
    echo "==> [STACK=$stack] $*"
    if [[ "$NO_AUTH" == "1" ]]; then
        STACK="$stack" KUMO_IMAGE_TAG=local DISABLE_AUTH=true DEPLOY_STAGE=preprod "$@"
    else
        STACK="$stack" KUMO_IMAGE_TAG=local "$@"
    fi
}

case "$CMD" in
up)
    for stack in "${stacks[@]}"; do
        run_stack "$stack" make -C "$REPO_ROOT" compose-local
    done
    ;;

down)
    _manifest_dir="${KUMO_STACK_MANIFEST_DIR:-$REPO_ROOT/../kumo-tests/stacks}"
    for stack in "${stacks[@]}"; do
        run_stack "$stack" make -C "$REPO_ROOT" compose-local-down
        rm -f "$_manifest_dir/$stack.json"
    done
    ;;

logs)
    if command -v tmux >/dev/null 2>&1 && [[ -z "${TMUX:-}" ]]; then
        echo "==> Opening tmux session with one pane per stack"
        SESSION="kumo-stacks"
        tmux new-session -d -s "$SESSION" -x 220 -y 50 \
            "STACK=${stacks[0]} make -C '$REPO_ROOT' compose-local-logs"
        for stack in "${stacks[@]:1}"; do
            tmux split-window -t "$SESSION" -h \
                "STACK=$stack make -C '$REPO_ROOT' compose-local-logs"
            tmux select-layout -t "$SESSION" tiled
        done
        tmux attach -t "$SESSION"
    elif command -v tmux >/dev/null 2>&1 && [[ -n "${TMUX:-}" ]]; then
        echo "==> Already inside tmux — opening one window per stack"
        for stack in "${stacks[@]}"; do
            tmux new-window -n "$stack" \
                "STACK=$stack make -C '$REPO_ROOT' compose-local-logs"
        done
    else
        echo "==> tmux not available — tailing stacks serially (Ctrl-C to stop each)"
        for stack in "${stacks[@]}"; do
            run_stack "$stack" make -C "$REPO_ROOT" compose-local-logs &
        done
        wait
    fi
    ;;

status)
    for stack in "${stacks[@]}"; do
        echo ""
        echo "==> STACK=$stack"
        docker ps --filter "name=${stack}-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null \
            || echo "  (no containers found)"
    done
    ;;

*)
    echo "ERROR: unknown command '$CMD'" >&2
    usage
    ;;
esac
