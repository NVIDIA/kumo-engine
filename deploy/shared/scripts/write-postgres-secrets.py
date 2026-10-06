#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write Kumo Postgres secret files from structured env vars."""

import json
import os
import sys
from pathlib import Path

SUPPORTED_SSLMODES = {"verify-ca", "verify-full"}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _password() -> str:
    password_file = _env("POSTGRES_PASSWORD_FILE")
    if password_file:
        return Path(password_file).read_text(encoding="utf-8").strip()
    return _env("POSTGRES_PASSWORD")


def _structured_secret(database: str) -> dict[str, object]:
    host = _env("POSTGRES_HOST")
    user = _env("POSTGRES_USER")
    if not host or not user:
        missing = [
            name for name, value in {
                "POSTGRES_HOST": host,
                "POSTGRES_USER": user,
            }.items() if not value
        ]
        raise SystemExit("ERROR: structured Postgres config is missing " +
                         ", ".join(missing))

    secret: dict[str, object] = {
        "username": user,
        "host": host,
        "port": _env("POSTGRES_PORT", "5432"),
        "database": database,
    }
    password = _password()
    if password:
        secret["password"] = password
    sslmode = _env("POSTGRES_SSLMODE")
    sslrootcert = _env("POSTGRES_SSLROOTCERT")
    if sslmode or sslrootcert:
        if sslmode not in SUPPORTED_SSLMODES:
            raise SystemExit(
                "ERROR: POSTGRES_SSLMODE must be verify-ca or verify-full")
        tls: dict[str, str] = {"sslmode": sslmode}
        if sslrootcert:
            tls["sslrootcert"] = sslrootcert
        secret["tls"] = tls
    return secret


def _write(path: str, payload: dict[str, object]) -> None:
    # DSNs embed the Postgres password; create with restrictive perms before
    # writing content so the secret is never briefly world-readable. The
    # O_CREAT mode only applies when the file is created, so fchmod the fd
    # explicitly to also tighten a pre-existing permissive file before its
    # (truncated) contents are rewritten.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
    os.fchmod(fd, 0o640)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")


def main() -> int:
    state_file = _env("STATE_DB_DSN_FILE")
    temporal_file = _env("TEMPORAL_DB_DSN_FILE")
    visibility_file = _env("TEMPORAL_VISIBILITY_DB_DSN_FILE")
    if not state_file and not temporal_file and not visibility_file:
        raise SystemExit(
            "ERROR: at least one Postgres DSN output file is required")

    if state_file:
        state_secret = _structured_secret(_env("KUMO_STATE_DB_NAME", "kumo"))
        _write(state_file, state_secret)

    if temporal_file:
        temporal_secret = _structured_secret(
            _env("TEMPORAL_DB_NAME", "temporal"))
        _write(temporal_file, temporal_secret)

    if visibility_file:
        visibility_secret = _structured_secret(
            _env("TEMPORAL_VISIBILITY_DB_NAME", "temporal_visibility"))
        _write(visibility_file, visibility_secret)
    return 0


if __name__ == "__main__":
    sys.exit(main())
