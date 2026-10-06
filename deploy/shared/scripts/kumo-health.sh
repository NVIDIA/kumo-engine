#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Kumo single-host health check.
# Usage: kumo-health
# Exit code 0 = all healthy, non-zero = issues found.
set -uo pipefail

ERRORS=0
COMPOSE_DIR="/opt/kumo/compose"

compose_env_value() {
    local key="$1"
    local env_file="$COMPOSE_DIR/.env"
    grep -E "^${key}=" "$env_file" 2>/dev/null | tail -1 | cut -d= -f2- || true
}

TEMPORAL_NAMESPACE="${TEMPORAL_NAMESPACE:-$(compose_env_value TEMPORAL_NAMESPACE)}"
TEMPORAL_NAMESPACE="${TEMPORAL_NAMESPACE:-prod}"
KUMO_POSTGRES_MODE="${KUMO_POSTGRES_MODE:-$(compose_env_value KUMO_POSTGRES_MODE)}"
KUMO_POSTGRES_MODE="${KUMO_POSTGRES_MODE:-bundled}"

check() {
    local name="$1" cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "  OK   $name"
    else
        echo "  FAIL $name"
        ERRORS=$((ERRORS + 1))
    fi
}

echo "=== Kumo Health Check ==="

# 1. Container status
echo ""
echo "Containers:"
CONTAINERS="kumo-temporal kumo-rest kumo-rest-public kumo-worker-data kumo-worker-mlworkflow kumo-pool-controller kumo-pool-worker kumo-ui kumo-nginx kumo-prometheus kumo-jaeger"
if [ "$KUMO_POSTGRES_MODE" != "external" ]; then
    CONTAINERS="kumo-postgres $CONTAINERS"
fi
for c in $CONTAINERS; do
    check "$c" "docker ps --format '{{.Names}}' | grep -q '^${c}$'"
done

# 2. State DB
if [ "$KUMO_POSTGRES_MODE" != "external" ]; then
    echo ""
    echo "Postgres:"
    check "state DB ready" \
        "docker exec kumo-postgres sh -c 'pg_isready -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\"'"
fi

# 3. Temporal
echo ""
echo "Temporal:"
check "namespace ${TEMPORAL_NAMESPACE} reachable" \
    "docker exec kumo-temporal temporal workflow list --limit 1 -n ${TEMPORAL_NAMESPACE}"

# 4. REST APIs
echo ""
echo "REST APIs:"
check "internal REST (8000)" \
    "docker exec kumo-rest python3 -c 'import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8000/health\", timeout=5).read()'"
check "public REST (8001)" \
    "docker exec kumo-rest-public python3 -c 'import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8001/health\", timeout=5).read()'"
check "nginx HTTP (80)" "curl -sf --connect-timeout 5 http://localhost/"

# 5. GPU
echo ""
echo "GPU:"
check "nvidia-smi" "nvidia-smi"

# 6. Disk
echo ""
echo "Storage:"
check "NVMe/localssd exists" "test -d /mnt/localssd"
nvme_free=$(df /mnt/localssd 2>/dev/null | awk 'NR==2{print $5}' | tr -d '%')
if [ -n "$nvme_free" ] && [ "$nvme_free" -gt 90 ]; then
    echo "  WARN NVMe usage ${nvme_free}% (>90%)"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK   NVMe usage ${nvme_free:-?}%"
fi
root_free=$(df / | awk 'NR==2{print $5}' | tr -d '%')
if [ "$root_free" -gt 90 ]; then
    echo "  WARN Root disk usage ${root_free}% (>90%)"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK   Root disk usage ${root_free}%"
fi

# 7. Monitoring
echo ""
echo "Monitoring:"
check "Prometheus" "docker exec kumo-prometheus wget -qO- http://127.0.0.1:9090/-/ready"
check "Jaeger" "docker exec kumo-jaeger wget -qO- http://127.0.0.1:16686/"

# Summary
echo ""
if [ $ERRORS -eq 0 ]; then
    echo "=== ALL HEALTHY ==="
else
    echo "=== $ERRORS ISSUE(S) FOUND ==="
fi
exit $ERRORS
