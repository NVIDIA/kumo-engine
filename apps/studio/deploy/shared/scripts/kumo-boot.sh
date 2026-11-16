#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Kumo single-host boot script. Runs on every boot via cloud-init per-boot.
# Cloud-agnostic: works on AWS and GCP.
set -euo pipefail

echo "=== Kumo boot script starting ==="

compose_env_value() {
  local key="$1"
  local env_file="/opt/kumo/compose/.env"
  grep -E "^${key}=" "$env_file" 2>/dev/null | tail -1 | cut -d= -f2- || true
}

container_health() {
  local container="$1"
  local port="$2"
  docker exec "$container" python3 - "$port" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

urllib.request.urlopen(
    f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2).read()
PY
}

# 1. Run firstboot if config doesn't exist yet.
#    Defensive fallback: cloud-init per-once should run firstboot before
#    per-boot, but stale semaphores (e.g., from Packer build) can cause
#    per-once to be skipped. This ensures firstboot always runs.
if [ ! -f /etc/kumo/deploy_env_config.json ]; then
  echo "First boot: running firstboot script..."
  /var/lib/cloud/scripts/per-once/kumo-firstboot.sh
fi

# One GPU worker slot per GPU, counted each time unless .env sets it.
if [ -z "${ML_MAX_GPU_WORKER_SLOTS:-}" ] \
    && [ -z "$(compose_env_value ML_MAX_GPU_WORKER_SLOTS)" ]; then
  ML_MAX_GPU_WORKER_SLOTS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  [ "$ML_MAX_GPU_WORKER_SLOTS" -ge 1 ] || ML_MAX_GPU_WORKER_SLOTS=1
  export ML_MAX_GPU_WORKER_SLOTS
fi

TEMPORAL_NAMESPACE="${TEMPORAL_NAMESPACE:-$(compose_env_value TEMPORAL_NAMESPACE)}"
TEMPORAL_NAMESPACE="${TEMPORAL_NAMESPACE:-prod}"
KUMO_POSTGRES_MODE="${KUMO_POSTGRES_MODE:-$(compose_env_value KUMO_POSTGRES_MODE)}"
KUMO_POSTGRES_MODE="${KUMO_POSTGRES_MODE:-bundled}"

COMPOSE_ARGS=(-f docker-compose.singlehost.yml)
if [ "$KUMO_POSTGRES_MODE" != "external" ]; then
  COMPOSE_ARGS+=(-f docker-compose.singlehost.postgres.yml)
fi
COMPOSE_ARGS+=(-f docker-compose.singlehost.gpu.yml)

# 2. Start Temporal first (other services depend on it)
cd /opt/kumo/compose
docker compose "${COMPOSE_ARGS[@]}" up -d temporal

# 3. Wait for Temporal to be healthy before proceeding
echo "Waiting for Temporal to be healthy..."
timeout=120
elapsed=0
while [ $elapsed -lt $timeout ]; do
  if docker exec kumo-temporal temporal workflow list --limit 1 -n "$TEMPORAL_NAMESPACE" >/dev/null 2>&1; then
    echo "Temporal is healthy."
    break
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done
if [ $elapsed -ge $timeout ]; then
  echo "WARNING: Temporal health check timed out after ${timeout}s, proceeding anyway."
fi

# 4. Terminate stale workflows from previous boot (NVMe data may be lost)
echo "Terminating stale running workflows..."
docker exec kumo-temporal temporal workflow list -n "$TEMPORAL_NAMESPACE" \
  --query 'ExecutionStatus="Running"' -o json 2>/dev/null \
  | TEMPORAL_NAMESPACE_FOR_CLEANUP="$TEMPORAL_NAMESPACE" python3 -c '
import json, os, sys, subprocess
namespace = os.environ["TEMPORAL_NAMESPACE_FOR_CLEANUP"]
wfs = json.load(sys.stdin)
for wf in wfs:
    wid = wf["execution"]["workflowId"]
    print(f"  Terminating: {wid}")
    subprocess.run([
        "docker", "exec", "kumo-temporal", "temporal", "workflow", "terminate",
        "-w", wid, "-n", namespace, "--reason", "Instance reboot: stale workflow cleanup"
    ], capture_output=True)
print(f"Terminated {len(wfs)} stale workflow(s).")
' 2>/dev/null || echo "No stale workflows to clean up."

# 5. Start all remaining services
docker compose "${COMPOSE_ARGS[@]}" up -d

# 6. Wait for REST APIs to be ready (cold start takes ~4min due to torch/CUDA in base image).
# Use unauthenticated health endpoints so auth mode does not affect startup checks.
echo "Waiting for REST APIs to start (may take several minutes on cold boot)..."
timeout=300
elapsed=0
while [ $elapsed -lt $timeout ]; do
  if container_health kumo-rest 8000 && container_health kumo-rest-public 8001; then
    echo "REST APIs ready (${elapsed}s)."
    break
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done
if [ $elapsed -ge $timeout ]; then
  echo "WARNING: REST APIs not ready after ${timeout}s."
fi

# Signal boot complete (watchdog checks this before running)
mkdir -p /var/lib/kumo
touch /var/lib/kumo/.boot-complete

echo "=== Kumo boot script complete ==="
