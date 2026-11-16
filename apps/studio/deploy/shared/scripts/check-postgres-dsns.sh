#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate BYO Postgres secrets from the same Docker networking surface Kumo uses.
set -euo pipefail

usage() {
  echo "Usage: $0 <postgres-secret-file> [postgres-secret-file ...]" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
PSQL_IMAGE="${KUMO_POSTGRES_PSQL_IMAGE:-postgres:16-alpine}"
CONNECT_TIMEOUT="${KUMO_POSTGRES_CONNECT_TIMEOUT:-10}"
PYTHON_BIN="${KUMO_POSTGRES_PYTHON_BIN:-python3}"
DOCKER_ARGS=(--rm -i)
if [[ -n "${KUMO_POSTGRES_DOCKER_NETWORK:-}" ]]; then
  DOCKER_ARGS+=(--network "$KUMO_POSTGRES_DOCKER_NETWORK")
fi
if [[ -n "${KUMO_POSTGRES_SECRETS_DIR:-}" ]]; then
  DOCKER_ARGS+=(-v "${KUMO_POSTGRES_SECRETS_DIR}:/opt/kumo/secrets:ro")
fi

normalize_dsn() {
  # psql/libpq does not understand SQLAlchemy driver names like
  # postgresql+asyncpg://, but the host/user/db parts are otherwise compatible.
  printf '%s' "$1" | sed -E 's#^postgresql\+[^:]+://#postgresql://#'
}

dsn_from_secret_file() {
  local secret_file="$1"
  if [[ ! -f "$secret_file" ]]; then
    echo "ERROR: Postgres secret file does not exist: $secret_file" >&2
    return 1
  fi

  "$PYTHON_BIN" - "$secret_file" <<'PY'
import json
import sys
from urllib.parse import quote, urlencode

secret_file = sys.argv[1]
with open(secret_file, encoding="utf-8") as f:
    secret = json.load(f)

missing = [
    key for key in ("username", "host", "database")
    if not secret.get(key)
]
if missing:
    raise SystemExit(
        f"ERROR: {secret_file} is missing structured fields {missing}")

userinfo = quote(str(secret["username"]), safe="")
if secret.get("password"):
    userinfo += f":{quote(str(secret['password']), safe='')}"
hostport = str(secret["host"])
if secret.get("port"):
    hostport += f":{secret['port']}"
tls = secret.get("tls")
params = tls if isinstance(tls, dict) else {}
suffix = f"?{urlencode(params)}" if params else ""
print(f"postgresql://{userinfo}@{hostport}/{quote(str(secret['database']), safe='')}{suffix}")
PY
}

check_secret_file() {
  local label="$1"
  local secret_file="$2"
  local dsn
  local table="kumo_byo_postgres_preflight_$$"

  dsn="$(dsn_from_secret_file "$secret_file")"
  if [[ -z "$dsn" ]]; then
    echo "ERROR: $label Postgres DSN is empty" >&2
    return 1
  fi

  dsn="$(normalize_dsn "$dsn")"
  echo "Checking $label Postgres connectivity and grants..."

  docker run "${DOCKER_ARGS[@]}" \
    -e "PGCONNECT_TIMEOUT=$CONNECT_TIMEOUT" \
    -e "KUMO_POSTGRES_DSN=$dsn" \
    "$PSQL_IMAGE" \
    sh -eu -c 'psql "$KUMO_POSTGRES_DSN" -v ON_ERROR_STOP=1 -q -X' >/dev/null <<SQL
BEGIN;
SELECT 1;
CREATE TABLE ${table} (
  id integer PRIMARY KEY,
  checked_at timestamptz DEFAULT now()
);
INSERT INTO ${table} (id) VALUES (1);
UPDATE ${table} SET checked_at = now() WHERE id = 1;
DELETE FROM ${table} WHERE id = 1;
DROP TABLE ${table};
COMMIT;
SQL
}

for secret_file in "$@"; do
  label="$(basename "$secret_file")"
  check_secret_file "$label" "$secret_file"
done
echo "BYO Postgres preflight passed."
