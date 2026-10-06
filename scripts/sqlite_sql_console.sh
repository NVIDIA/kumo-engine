#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SQLite console for the devserver StateDB.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    DEFAULT_PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    DEFAULT_PYTHON_BIN="python3"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"

if [ -n "${STATE_DB_PATH:-}" ]; then
    :
elif [ -n "${DB_STORAGE_ROOT:-}" ]; then
    STATE_DB_PATH="$DB_STORAGE_ROOT/state.sqlite"
else
    STATE_DB_PATH="$("$PYTHON_BIN" -c '
from rest.settings import get_db_storage_root
from statedb.db.config import default_sqlite_path

print(default_sqlite_path(get_db_storage_root()))
')"
fi

echo "Opening devserver StateDB: $STATE_DB_PATH"
exec sqlite3 "$STATE_DB_PATH" -cmd ".mode line"
