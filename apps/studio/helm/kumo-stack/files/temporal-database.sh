#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Credentials arrive through the shared tmpfs and Temporal Secret; neither value
# appears in arguments or the stored Job manifest.
set -eu

bootstrap_dir=/opt/kumo/bootstrap
credential_dir=/run/kumo

admin_psql() {
  PGPASSWORD=$(cat "${credential_dir}/admin-password")
  export PGPASSWORD
  psql --no-password "$@"
}

temporal_psql() {
  PGPASSWORD=${KUMO_TEMPORAL_PASSWORD}
  export PGPASSWORD
  psql --no-password --username "${KUMO_TEMPORAL_ROLE}" "$@"
}

case "${1:-}" in
  bootstrap)
    admin_psql --file "${bootstrap_dir}/bootstrap-temporal.sql"
    ;;
  verify)
    for database in temporal temporal_visibility; do
      owned=$(admin_psql --dbname "${database}" --tuples-only --no-align \
        --file "${bootstrap_dir}/verify-temporal.sql")
      if [ "${owned}" != t ]; then
        printf '%s owner/public schema permissions do not match %s\n' \
          "${database}" "${KUMO_TEMPORAL_ROLE}" >&2
        exit 1
      fi
      temporal_psql --dbname "${database}" --command "SELECT 1" >/dev/null
    done
    ;;
  *)
    printf 'usage: temporal-database.sh bootstrap|verify\n' >&2
    exit 2
    ;;
esac
