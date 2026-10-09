# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncpg

from statedb.db.bootstrap import PostgresSchemaMigration
from statedb.db.schema import STATE_DB_TABLES, IndexSpec

_LEGACY_API_KEY_INDEXES = (
    "api_keys_customer_type_owner_idx",
    "api_keys_verification_lookup_idx",
    "api_keys_owner_name_unique_idx",
)


def _current_api_key_indexes() -> tuple[IndexSpec, ...]:
    return next(table.indexes for table in STATE_DB_TABLES
                if table.name == "api_keys")


async def _migrate_postgres_12_to_13(connection: asyncpg.Connection, ) -> None:
    """Remove the retired customer scope from persisted API-key indexes."""
    for index_name in _LEGACY_API_KEY_INDEXES:
        await connection.execute(f"DROP INDEX IF EXISTS {index_name}")
    for index in _current_api_key_indexes():
        await connection.execute(index.ddl("api_keys", "postgres"))


async def _migrate_postgres_13_to_14(connection: asyncpg.Connection, ) -> None:
    """Drop resource analytics for the retired external ML job launchers."""
    await connection.execute("DROP TABLE IF EXISTS job_resources")


async def _migrate_postgres_14_to_15(connection: asyncpg.Connection, ) -> None:
    """Drop metadata for the retired baseline-job subsystem."""
    await connection.execute("DROP TABLE IF EXISTS baseline_jobs")


async def _migrate_postgres_15_to_16(connection: asyncpg.Connection, ) -> None:
    """Drop metadata for the retired online-serving subsystem."""
    await connection.execute("DROP TABLE IF EXISTS online_serving_jobs")


async def _migrate_postgres_16_to_17(connection: asyncpg.Connection, ) -> None:
    """Release tombstoned identity principals, names and slugs for reuse."""
    identities = next(table for table in STATE_DB_TABLES
                      if table.name == "identities")
    for index in identities.indexes:
        if index.unique:
            await connection.execute(f"DROP INDEX IF EXISTS {index.name}")
            await connection.execute(index.ddl("identities", "postgres"))


POSTGRES_SCHEMA_MIGRATIONS: dict[int, PostgresSchemaMigration] = {
    12: _migrate_postgres_12_to_13,
    13: _migrate_postgres_13_to_14,
    14: _migrate_postgres_14_to_15,
    15: _migrate_postgres_15_to_16,
    16: _migrate_postgres_16_to_17,
}
