#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Kumo health watchdog. Runs periodically via cron to monitor and auto-recover services.
#
# Runs automatically via cron (/etc/cron.d/kumo-watchdog) every 5 minutes.
# Run history: /var/log/kumo-watchdog.log (rotated via logrotate)
#
# Checks:
#   1. All expected containers are running
#   2. Docker healthcheck status for containers that define one
#   3. REST APIs responding
#   4. Disk space thresholds
#
# Recovery actions:
#   - Restart containers that are exited or unhealthy
set -uo pipefail

COMPOSE_DIR="/opt/kumo/compose"
COMPOSE_FILE="docker-compose.singlehost.yml"
COMPOSE_POSTGRES_FILE="docker-compose.singlehost.postgres.yml"
COMPOSE_GPU_FILE="docker-compose.singlehost.gpu.yml"
BOOT_COMPLETE_FLAG="/var/lib/kumo/.boot-complete"
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"

log() { echo "${LOG_PREFIX} $1"; }

compose_env_value() {
  local key="$1"
  local env_file="$COMPOSE_DIR/.env"
  grep -E "^${key}=" "$env_file" 2>/dev/null | tail -1 | cut -d= -f2- || true
}

KUMO_POSTGRES_MODE="${KUMO_POSTGRES_MODE:-$(compose_env_value KUMO_POSTGRES_MODE)}"
KUMO_POSTGRES_MODE="${KUMO_POSTGRES_MODE:-bundled}"

COMPOSE_ARGS=(-f "$COMPOSE_FILE")
if [ "$KUMO_POSTGRES_MODE" != "external" ]; then
  COMPOSE_ARGS+=(-f "$COMPOSE_POSTGRES_FILE")
fi
COMPOSE_ARGS+=(-f "$COMPOSE_GPU_FILE")

# Skip if boot hasn't completed yet (avoid racing with kumo-boot.sh)
if [ ! -f "$BOOT_COMPLETE_FLAG" ]; then
  log "SKIP: boot not complete yet ($BOOT_COMPLETE_FLAG missing)"
  exit 0
fi

ISSUES=0

# Map container names to compose service names for recovery
# (docker inspect uses container names, docker compose uses service names)
container_to_service() {
  case "$1" in
    kumo-temporal)       echo "temporal" ;;
    kumo-postgres)       echo "postgres" ;;
    kumo-prometheus)     echo "prometheus" ;;
    kumo-jaeger)         echo "jaeger" ;;
    kumo-nginx)          echo "nginx" ;;
    kumo-rest)           echo "kumo-rest" ;;
    kumo-rest-public)    echo "kumo-rest-public" ;;
    kumo-worker-data)    echo "kumo-worker-data" ;;
    kumo-worker-mlworkflow) echo "kumo-worker-mlworkflow" ;;
    kumo-pool-controller)   echo "kumo-pool-controller" ;;
    kumo-pool-worker)       echo "kumo-pool-worker" ;;
    kumo-ui)             echo "kumo-ui" ;;
    *) echo "$1" ;;
  esac
}

container_health() {
  local container="$1"
  local port="$2"
  docker exec "$container" python3 - "$port" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

urllib.request.urlopen(
    f"http://127.0.0.1:{sys.argv[1]}/health", timeout=10).read()
PY
}

# ============================================================
# 1. Check expected containers are running
# ============================================================
EXPECTED_CONTAINERS="kumo-temporal kumo-rest kumo-rest-public kumo-worker-data kumo-worker-mlworkflow kumo-pool-controller kumo-pool-worker kumo-ui kumo-nginx kumo-prometheus kumo-jaeger"
if [ "$KUMO_POSTGRES_MODE" != "external" ]; then
  EXPECTED_CONTAINERS="kumo-postgres $EXPECTED_CONTAINERS"
fi

for c in $EXPECTED_CONTAINERS; do
  status=$(docker inspect "$c" --format '{{.State.Status}}' 2>/dev/null || echo "missing")
  if [ "$status" != "running" ]; then
    svc=$(container_to_service "$c")
    log "WARN: $c is $status, restarting service $svc..."
    cd "$COMPOSE_DIR" || { log "ERROR: cannot enter $COMPOSE_DIR"; exit 1; }
    docker compose "${COMPOSE_ARGS[@]}" up -d "$svc"
    ISSUES=$((ISSUES + 1))
  fi
done

# ============================================================
# 2. Check docker healthcheck status
# ============================================================
HEALTHCHECK_CONTAINERS="kumo-temporal kumo-worker-data kumo-worker-mlworkflow"
if [ "$KUMO_POSTGRES_MODE" != "external" ]; then
  HEALTHCHECK_CONTAINERS="kumo-postgres $HEALTHCHECK_CONTAINERS"
fi
for c in $HEALTHCHECK_CONTAINERS; do
  health=$(docker inspect "$c" --format '{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
  if [ "$health" = "unhealthy" ]; then
    log "WARN: $c is unhealthy, restarting..."
    docker restart "$c" 2>/dev/null
    ISSUES=$((ISSUES + 1))
  fi
done

# ============================================================
# 3. Check REST APIs (only if containers are running)
# ============================================================
rest_status=$(docker inspect kumo-rest --format '{{.State.Status}}' 2>/dev/null)
if [ "$rest_status" = "running" ]; then
  if ! container_health kumo-rest 8000; then
    # REST container running but not responding — check if it's still starting up
    uptime=$(docker inspect kumo-rest --format '{{.State.StartedAt}}' 2>/dev/null)
    age_sec=$(python3 -c "
from datetime import datetime, timezone
import sys
try:
    started = datetime.fromisoformat('${uptime}'.replace('Z','+00:00'))
    age = (datetime.now(timezone.utc) - started).total_seconds()
    print(int(age))
except:
    print(0)
" 2>/dev/null)
    if [ "$age_sec" -gt 300 ]; then
      log "WARN: kumo-rest running but not responding (age=${age_sec}s), restarting..."
      docker restart kumo-rest 2>/dev/null
      ISSUES=$((ISSUES + 1))
    fi
  fi
fi

# ============================================================
# 4. Disk space checks
# ============================================================
root_pct=$(df / | awk 'NR==2{print $5}' | tr -d '%')
if [ "$root_pct" -gt 90 ]; then
  log "WARN: Root disk usage ${root_pct}% (>90%)"
  ISSUES=$((ISSUES + 1))
fi

if [ -d /mnt/localssd ]; then
  nvme_pct=$(df /mnt/localssd 2>/dev/null | awk 'NR==2{print $5}' | tr -d '%')
  if [ -n "$nvme_pct" ] && [ "$nvme_pct" -gt 90 ]; then
    log "WARN: NVMe usage ${nvme_pct}% (>90%)"
    ISSUES=$((ISSUES + 1))
  fi
fi

# ============================================================
# Summary
# ============================================================
if [ $ISSUES -gt 0 ]; then
  log "Watchdog found $ISSUES issue(s) and took recovery actions."
else
  log "OK"
fi
