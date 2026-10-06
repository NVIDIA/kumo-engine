#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

validate_db_name() {
  case "$1" in
    ""|*[!A-Za-z0-9_]*)
      echo "Invalid Postgres database name: $1" >&2
      exit 1
      ;;
  esac
}

create_database_if_missing() {
  db_name="$1"
  validate_db_name "$db_name"
  if [ "$db_name" = "$POSTGRES_DB" ]; then
    return
  fi
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
SELECT 'CREATE DATABASE "$db_name"'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db_name')\gexec
SQL
}

create_database_if_missing "${TEMPORAL_DB_NAME:-temporal}"
create_database_if_missing "${TEMPORAL_VISIBILITY_DB_NAME:-temporal_visibility}"
