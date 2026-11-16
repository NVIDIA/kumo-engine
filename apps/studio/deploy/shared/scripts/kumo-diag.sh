#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Kumo diagnostic bundle tool.
# Collects time-windowed logs, metrics, traces, workflow history, and system info.
#
# Usage:
#   kumo-diag                              # last 24 hours (default)
#   kumo-diag --since 2h                   # last 2 hours
#   kumo-diag --since 3h --until 2h        # 3 hours ago to 2 hours ago
#   kumo-diag --since 30m                  # last 30 minutes
#   kumo-diag --since 7d --until 6d        # day 7 ago to day 6 ago
#
# Time format: <number><unit> where unit is m(inutes), h(ours), d(ays)
# --since: start of window (how far back). Default: 24h
# --until: end of window (how far back). Default: 0 (now)
# --debug: include DEBUG-level logs (default: INFO+ only, excludes noisy heartbeat/protobuf spam)
#
# Output: /tmp/kumo-diag-<timestamp>.tar.gz
set -uo pipefail

# Parse args
SINCE="24h"
UNTIL="0m"
INCLUDE_DEBUG=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --since) SINCE="$2"; shift 2 ;;
    --until) UNTIL="$2"; shift 2 ;;
    --debug) INCLUDE_DEBUG=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

compose_env_value() {
  local key="$1"
  local env_file="/opt/kumo/compose/.env"
  grep -E "^${key}=" "$env_file" 2>/dev/null | tail -1 | cut -d= -f2- || true
}

redact_env_file() {
  local env_file="$1"
  python3 - "$env_file" <<'PY'
import re
import sys

secret_key_pattern = re.compile(
    r'(SECRET|KEY|PASSWORD|TOKEN|CREDENTIAL|DSN|URL_OVERRIDE)', re.IGNORECASE)

for raw_line in open(sys.argv[1], encoding='utf-8'):
    line = raw_line.rstrip('\n')
    if not line or line.lstrip().startswith('#') or '=' not in line:
        print(line)
        continue
    key, _ = line.split('=', 1)
    if secret_key_pattern.search(key):
        print(f'{key}=<redacted>')
    else:
        print(line)
PY
}

TEMPORAL_NAMESPACE="${TEMPORAL_NAMESPACE:-$(compose_env_value TEMPORAL_NAMESPACE)}"
TEMPORAL_NAMESPACE="${TEMPORAL_NAMESPACE:-prod}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DIAG_DIR="/tmp/kumo-diag-${TIMESTAMP}"
OUTPUT="/tmp/kumo-diag-${TIMESTAMP}.tar.gz"

mkdir -p "$DIAG_DIR/logs" "$DIAG_DIR/metrics" "$DIAG_DIR/traces" "$DIAG_DIR/temporal" "$DIAG_DIR/system" "$DIAG_DIR/config"

# Convert duration string to seconds-ago
duration_to_secs() {
  python3 -c "
import re, sys
m = re.match(r'^(\d+)([mhd])$', '$1')
if not m:
    print('ERROR: invalid duration \"$1\". Use format: <number><m|h|d> (e.g., 2h, 30m, 7d)', file=sys.stderr)
    sys.exit(1)
n, unit = int(m.group(1)), m.group(2)
print(n * {'m': 60, 'h': 3600, 'd': 86400}[unit])
"
}

SINCE_SECS=$(duration_to_secs "$SINCE")
UNTIL_SECS=$(duration_to_secs "$UNTIL")

# Epoch timestamps
SINCE_EPOCH=$(python3 -c "import time; print(time.time() - ${SINCE_SECS})")
UNTIL_EPOCH=$(python3 -c "import time; print(time.time() - ${UNTIL_SECS})")
SINCE_EPOCH_US=$(python3 -c "print(int(${SINCE_EPOCH} * 1e6))")
UNTIL_EPOCH_US=$(python3 -c "print(int(${UNTIL_EPOCH} * 1e6))")

# Human-readable window for display
SINCE_HUMAN=$(date -d "@$(printf '%.0f' "$SINCE_EPOCH")" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r "$(printf '%.0f' "$SINCE_EPOCH")" '+%Y-%m-%d %H:%M:%S')
UNTIL_HUMAN=$(date -d "@$(printf '%.0f' "$UNTIL_EPOCH")" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r "$(printf '%.0f' "$UNTIL_EPOCH")" '+%Y-%m-%d %H:%M:%S')

echo "Collecting Kumo diagnostics [${SINCE_HUMAN} → ${UNTIL_HUMAN}]..."

# ============================================================
# 1. Container logs (time-windowed)
# ============================================================
echo "  [1/7] Collecting container logs..."
# Use docker's native --since/--until for time filtering (fast, no streaming through Python)
# Then optionally filter DEBUG level
DOCKER_SINCE=$(date -d "@$(printf '%.0f' "$SINCE_EPOCH")" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date -r "$(printf '%.0f' "$SINCE_EPOCH")" '+%Y-%m-%dT%H:%M:%S')
DOCKER_UNTIL=$(date -d "@$(printf '%.0f' "$UNTIL_EPOCH")" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date -r "$(printf '%.0f' "$UNTIL_EPOCH")" '+%Y-%m-%dT%H:%M:%S')

for container in $(docker ps --format '{{.Names}}'); do
  if [ "$INCLUDE_DEBUG" = "true" ]; then
    docker logs --since "$DOCKER_SINCE" --until "$DOCKER_UNTIL" "$container" \
      > "$DIAG_DIR/logs/${container}.log" 2>&1 &
  else
    docker logs --since "$DOCKER_SINCE" --until "$DOCKER_UNTIL" "$container" 2>&1 \
      | grep -v " - DEBUG \|^DEBUG:" \
      > "$DIAG_DIR/logs/${container}.log" &
  fi
done
wait

# ============================================================
# 2. Prometheus metrics
# ============================================================
echo "  [2/7] Collecting Prometheus metrics..."

# Snapshot (if admin API enabled)
SNAP=$(curl -sf -X POST http://localhost:9090/api/v1/admin/tsdb/snapshot 2>/dev/null)
if [ -n "$SNAP" ]; then
  echo "$SNAP" > "$DIAG_DIR/metrics/snapshot_info.json"
fi

# Key metrics via query_range
PROM_START="$SINCE_EPOCH"
PROM_END="$UNTIL_EPOCH"
PROM_STEP="60"

for metric in "up" "process_resident_memory_bytes" "process_cpu_seconds_total" "scrape_duration_seconds"; do
  curl -sf "http://localhost:9090/api/v1/query_range?query=${metric}&start=${PROM_START}&end=${PROM_END}&step=${PROM_STEP}" \
    > "$DIAG_DIR/metrics/${metric}.json" 2>/dev/null || true
done

# Target health
curl -sf http://localhost:9090/api/v1/targets > "$DIAG_DIR/metrics/targets.json" 2>/dev/null || true

# ============================================================
# 3. Jaeger traces
# ============================================================
echo "  [3/7] Collecting Jaeger traces..."

# Get all services
SERVICES=$(curl -sf http://localhost:16686/api/services 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
for s in d.get('data', []):
    if s != 'jaeger-all-in-one':
        print(s)
" 2>/dev/null)

for svc in $SERVICES; do
  curl -sf "http://localhost:16686/api/traces?service=${svc}&start=${SINCE_EPOCH_US}&end=${UNTIL_EPOCH_US}&limit=50" \
    > "$DIAG_DIR/traces/${svc}.json" 2>/dev/null || true
done

# ============================================================
# 4. Temporal workflow history
# ============================================================
echo "  [4/7] Collecting Temporal workflow history..."

# Recent failed workflows
docker exec kumo-temporal temporal workflow list -n "$TEMPORAL_NAMESPACE" \
  --query 'ExecutionStatus="Failed"' --limit 20 -o json \
  > "$DIAG_DIR/temporal/failed_workflows.json" 2>/dev/null || true

# Recent terminated workflows
docker exec kumo-temporal temporal workflow list -n "$TEMPORAL_NAMESPACE" \
  --query 'ExecutionStatus="Terminated"' --limit 20 -o json \
  > "$DIAG_DIR/temporal/terminated_workflows.json" 2>/dev/null || true

# Detailed history for each failed workflow
if [ -f "$DIAG_DIR/temporal/failed_workflows.json" ]; then
  python3 -c "
import json, subprocess, sys
try:
    wfs = json.load(open('$DIAG_DIR/temporal/failed_workflows.json'))
except:
    sys.exit(0)
for wf in wfs[:10]:
    wid = wf.get('execution', {}).get('workflowId', '')
    if not wid:
        continue
    safe_name = wid.replace('/', '_')[:100]
    result = subprocess.run(
        ['docker', 'exec', 'kumo-temporal', 'temporal', 'workflow', 'show',
         '-w', wid, '-n', '${TEMPORAL_NAMESPACE}', '-o', 'json'],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        with open(f'$DIAG_DIR/temporal/history_{safe_name}.json', 'w') as f:
            f.write(result.stdout)
" 2>/dev/null || true
fi

# ============================================================
# 5. System info
# ============================================================
echo "  [5/7] Collecting system info..."

# Service status
cd /opt/kumo/compose || exit 1
KUMO_POSTGRES_MODE=$(grep -E '^KUMO_POSTGRES_MODE=' .env 2>/dev/null | tail -1 | cut -d= -f2- || true)
KUMO_POSTGRES_MODE=${KUMO_POSTGRES_MODE:-bundled}
if [ "$KUMO_POSTGRES_MODE" = "external" ]; then
  docker compose -f docker-compose.singlehost.yml ps > "$DIAG_DIR/system/containers.txt" 2>&1 || true
else
  docker compose -f docker-compose.singlehost.yml -f docker-compose.singlehost.postgres.yml ps > "$DIAG_DIR/system/containers.txt" 2>&1 || true
fi

# Container restart counts and health
docker ps -a --format '{{.Names}}\t{{.Status}}' > "$DIAG_DIR/system/container_status.txt" 2>&1 || true
for container in $(docker ps --format '{{.Names}}'); do
  restarts=$(docker inspect "$container" --format '{{.RestartCount}}' 2>/dev/null)
  health=$(docker inspect "$container" --format '{{.State.Health.Status}}' 2>/dev/null || echo "n/a")
  echo "${container}: restarts=${restarts} health=${health}" >> "$DIAG_DIR/system/container_health.txt"
done

# GPU
nvidia-smi > "$DIAG_DIR/system/nvidia-smi.txt" 2>&1 || echo "nvidia-smi not available" > "$DIAG_DIR/system/nvidia-smi.txt"

# Disk
df -h > "$DIAG_DIR/system/disk.txt"
du -sh /mnt/localssd/* 2>/dev/null > "$DIAG_DIR/system/nvme_usage.txt" || true
iostat -x 1 1 > "$DIAG_DIR/system/iostat.txt" 2>/dev/null || true

# Memory
free -h > "$DIAG_DIR/system/memory.txt"
ps aux --sort=-rss | head -20 > "$DIAG_DIR/system/top_memory_processes.txt" 2>/dev/null || true

# Per-container resource usage
docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.PIDs}}' \
  > "$DIAG_DIR/system/container_resources.txt" 2>/dev/null || true

# Network: listening ports
ss -tlnp > "$DIAG_DIR/system/listening_ports.txt" 2>/dev/null || true

# Thread dumps for Python processes in containers (helps debug stuck activities)
for container in kumo-rest kumo-worker-data kumo-worker-mlworkflow kumo-pool-controller kumo-pool-worker; do
  docker exec "$container" bash -c 'kill -USR1 1 2>/dev/null; python3 -c "import faulthandler,sys; faulthandler.dump_traceback(sys.stdout)" 2>/dev/null' \
    > "$DIAG_DIR/system/threads_${container}.txt" 2>/dev/null || true
done

# Kernel errors (OOM, GPU hangs)
dmesg -T 2>/dev/null | grep -iE "oom|kill|gpu|error|nvme" | tail -50 > "$DIAG_DIR/system/dmesg_errors.txt" || true

# cloud-init log
tail -100 /var/log/cloud-init-output.log > "$DIAG_DIR/system/cloud-init.txt" 2>/dev/null || true

# REST /health endpoints (if available)
curl -sf --connect-timeout 5 http://localhost:8000/health > "$DIAG_DIR/system/rest_health.json" 2>/dev/null || true
curl -sf --connect-timeout 5 http://localhost:8001/health > "$DIAG_DIR/system/rest_public_health.json" 2>/dev/null || true

# ============================================================
# 6. Config (secrets redacted)
# ============================================================
echo "  [6/7] Collecting config..."

cat /etc/kumo/image_tag > "$DIAG_DIR/config/image_tag.txt" 2>/dev/null || true
cat /etc/kumo/deploy_env_config.json > "$DIAG_DIR/config/deploy_env_config.json" 2>/dev/null || true
redact_env_file /opt/kumo/compose/.env > "$DIAG_DIR/config/compose_env.txt" 2>/dev/null || true

# kumo-health output
kumo-health > "$DIAG_DIR/system/kumo-health.txt" 2>&1 || true

# ============================================================
# 7. Summary and error extraction (for quick triage)
# ============================================================
echo "  [7/7] Generating summary..."

# Extract errors/exceptions from all logs
# Match lines with ERROR/FATAL log level or Python exception patterns
# Exclude false positives: endpoint paths containing "error", Jaeger gRPC transport noise
for f in "$DIAG_DIR"/logs/*.log; do
  name=$(basename "$f" .log)
  grep -nE " - ERROR |^ERROR:|Traceback |FATAL|panic:" "$f" 2>/dev/null | while IFS= read -r line; do
    echo "[$name] $line"
  done
done > "$DIAG_DIR/ERRORS.txt" 2>/dev/null || true

# Workflow failure summary
python3 -c "
import json, sys, os
out = []
for fname in ['failed_workflows.json', 'terminated_workflows.json']:
    path = '$DIAG_DIR/temporal/' + fname
    if not os.path.exists(path):
        continue
    try:
        wfs = json.load(open(path))
    except:
        continue
    for wf in wfs:
        status = fname.replace('_workflows.json', '').upper()
        wtype = wf.get('type', {}).get('name', '?')
        wid = wf.get('execution', {}).get('workflowId', '?')
        start = wf.get('startTime', '?')[:19]
        close = wf.get('closeTime', '?')[:19]
        out.append(f'{status}: {wtype}')
        out.append(f'  id: {wid}')
        out.append(f'  time: {start} -> {close}')
        out.append('')
print('\n'.join(out))
" > "$DIAG_DIR/temporal/FAILURES_SUMMARY.txt" 2>/dev/null || true

# Top-level summary
cat > "$DIAG_DIR/SUMMARY.txt" << EOSUMMARY
Kumo Diagnostic Bundle
======================
Time window: ${SINCE_HUMAN} → ${UNTIL_HUMAN}
Generated:   $(date '+%Y-%m-%d %H:%M:%S')
Image tag:   $(cat /etc/kumo/image_tag 2>/dev/null || echo '?')

Containers:
$(docker ps --format '  {{.Names}}: {{.Status}}' 2>/dev/null)

Error count by container:
$(for f in "$DIAG_DIR"/logs/*.log; do
  name=$(basename "$f" .log)
  count=$(grep -cE " - ERROR |^ERROR:|Traceback |FATAL" "$f" 2>/dev/null || echo 0)
  [ "$count" -gt 0 ] && echo "  $name: $count errors"
done)

Failed/terminated workflows:
$(wc -l < "$DIAG_DIR/temporal/FAILURES_SUMMARY.txt" 2>/dev/null | awk '{print "  " int($1/4) " workflow(s)"}' || echo "  0")

Health check:
$(kumo-health 2>&1 | grep -E "FAIL|ALL HEALTHY|ISSUE" || echo "  (unavailable)")
EOSUMMARY

# ============================================================
# Bundle
# ============================================================
tar -czf "$OUTPUT" -C /tmp "kumo-diag-${TIMESTAMP}"
rm -rf "$DIAG_DIR"

echo ""
echo "Diagnostic bundle: $OUTPUT"
ls -lh "$OUTPUT"
echo ""
echo "Contents:"
tar -tzf "$OUTPUT" | head -30
echo "..."
echo ""
echo "Please review contents before sharing with Kumo support."
