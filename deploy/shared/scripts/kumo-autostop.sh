#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Auto-stop the instance after a configurable idle period with no Temporal
# workflow activity.
#
# "Idle" means: no user-initiated workflows are currently running, AND no
# user-initiated workflows have completed within the last IDLE_MINUTES.
# Background workflows (health checks, RPC polling) are ignored.
# The instance also never stops within IDLE_MINUTES of boot.
#
# Runs automatically via cron (/etc/cron.d/kumo-autostop) every 5 minutes.
# No-op unless KUMO_AUTOSTOP_IDLE_MINUTES is set in /etc/kumo/autostop.conf:
#   echo "KUMO_AUTOSTOP_IDLE_MINUTES=60" > /etc/kumo/autostop.conf
#
# Run history: /var/log/kumo-autostop.log
# Example log lines:
#   [2026-04-03T14:30:00+00:00] instance booted 12m ago, need 60m — skipping
#   [2026-04-03T14:35:00+00:00] running workflows found — skipping
#   [2026-04-03T14:40:00+00:00] workflows completed in last 60m — skipping
#   [2026-04-03T15:45:00+00:00] no activity in last 60m — stopping instance
set -uo pipefail

log() { echo "[$(date -Iseconds)] $*"; }

# Source config if it exists.
# shellcheck source=/dev/null
. /etc/kumo/autostop.conf 2>/dev/null || true

IDLE_MINUTES="${KUMO_AUTOSTOP_IDLE_MINUTES:-}"
if [ -z "$IDLE_MINUTES" ]; then
    exit 0
fi
if ! [[ "$IDLE_MINUTES" =~ ^[0-9]+$ ]] || [ "$IDLE_MINUTES" -lt 10 ]; then
    log "invalid KUMO_AUTOSTOP_IDLE_MINUTES=$IDLE_MINUTES (must be integer >= 10) — skipping"
    exit 1
fi

compose_env_value() {
    local key="$1"
    local env_file="/opt/kumo/compose/.env"
    grep -E "^${key}=" "$env_file" 2>/dev/null | tail -1 | cut -d= -f2- || true
}

NAMESPACE="${TEMPORAL_NAMESPACE:-$(compose_env_value TEMPORAL_NAMESPACE)}"
NAMESPACE="${NAMESPACE:-prod}"
TEMPORAL_CONTAINER="kumo-temporal"
# Exclude background workflows: health checks (WorkerInfoWorkflow) and RPC
# polling workflows (RpcWorkflow_*) that run continuously.
BACKGROUND_FILTER="NOT (WorkflowId STARTS_WITH 'health-check-') AND NOT (WorkflowType STARTS_WITH 'RpcWorkflow_')"

# -- helpers ------------------------------------------------------------------

temporal_query() {
    docker exec "$TEMPORAL_CONTAINER" \
        temporal workflow list -n "$NAMESPACE" \
        --query "$1" --limit 1 --output json 2>/dev/null
}

# -- guards -------------------------------------------------------------------

# Don't stop if we booted less than IDLE_MINUTES ago.
boot_epoch=$(date -d "$(uptime -s)" +%s 2>/dev/null) \
    || { log "failed to determine boot time — skipping"; exit 0; }
now=$(date +%s)
boot_age_min=$(( (now - boot_epoch) / 60 ))
if [ "$boot_age_min" -lt "$IDLE_MINUTES" ]; then
    log "instance booted ${boot_age_min}m ago, need ${IDLE_MINUTES}m — skipping"
    exit 0
fi

# -- check running workflows (excluding background workflows) ----------------

if ! running_json=$(temporal_query \
    "ExecutionStatus='Running' AND $BACKGROUND_FILTER"); then
    log "temporal query failed — assuming active, skipping"
    exit 0
fi

running_count=$(echo "$running_json" | jq 'length')
if [ "${running_count:-1}" -gt 0 ]; then
    log "running workflows found — skipping"
    exit 0
fi

# -- check recently completed workflows --------------------------------------

# Any non-background workflow that closed within the idle window means the
# instance is still "active".
cutoff=$(date -u -d "$IDLE_MINUTES minutes ago" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null) \
    || { log "failed to compute cutoff time — skipping"; exit 0; }

if ! recent_json=$(temporal_query \
    "CloseTime > '$cutoff' AND $BACKGROUND_FILTER"); then
    log "temporal query for recent completions failed — assuming active, skipping"
    exit 0
fi

recent_count=$(echo "$recent_json" | jq 'length')
if [ "${recent_count:-1}" -gt 0 ]; then
    log "workflows completed in last ${IDLE_MINUTES}m — skipping"
    exit 0
fi

log "no activity in last ${IDLE_MINUTES}m — stopping instance"
sudo shutdown -h now
