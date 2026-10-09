#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Restart all Kumo services.
# Customer-facing tool — no SSH expertise required.
set -euo pipefail

compose_env_value() {
  local key="$1"
  local env_file="/opt/kumo/compose/.env"
  grep -E "^${key}=" "$env_file" 2>/dev/null | tail -1 | cut -d= -f2- || true
}

KUMO_POSTGRES_MODE="${KUMO_POSTGRES_MODE:-$(compose_env_value KUMO_POSTGRES_MODE)}"
KUMO_POSTGRES_MODE="${KUMO_POSTGRES_MODE:-bundled}"

# One GPU worker slot per GPU, counted each time unless .env sets it.
if [ -z "${ML_MAX_GPU_WORKER_SLOTS:-}" ] \
    && [ -z "$(compose_env_value ML_MAX_GPU_WORKER_SLOTS)" ]; then
  ML_MAX_GPU_WORKER_SLOTS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  [ "$ML_MAX_GPU_WORKER_SLOTS" -ge 1 ] || ML_MAX_GPU_WORKER_SLOTS=1
  export ML_MAX_GPU_WORKER_SLOTS
fi

COMPOSE_ARGS=(-f docker-compose.singlehost.yml)
if [ "$KUMO_POSTGRES_MODE" != "external" ]; then
  COMPOSE_ARGS+=(-f docker-compose.singlehost.postgres.yml)
fi
COMPOSE_ARGS+=(-f docker-compose.singlehost.gpu.yml)

echo "Restarting Kumo services..."
cd /opt/kumo/compose
docker compose "${COMPOSE_ARGS[@]}" down
docker compose "${COMPOSE_ARGS[@]}" up -d
echo "Done. Services restarting — may take a minute to become fully ready."
