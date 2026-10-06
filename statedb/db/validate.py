# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and, when empty, initialize cloud PostgreSQL StateDB."""

from __future__ import annotations

import asyncio
import os

from service_framework._raw_deploy_config import PostgresMetadataStorageConfig
from statedb.db.config import open_state_db_from_postgres_config


async def validate() -> None:
    config = PostgresMetadataStorageConfig(
        auth_mode=os.environ["STATE_AUTH_MODE"],
        host=os.environ["PGHOST"],
        port=int(os.environ["PGPORT"]),
        database=os.environ["PGDATABASE"],
        username=os.environ["PGUSER"],
        region=os.environ.get("AWS_REGION") or None,
        sslrootcert=os.environ["PGSSLROOTCERT"],
    )
    state_db = await open_state_db_from_postgres_config(config)
    try:
        status = await state_db.get_schema_status()
        if not status.ok:
            raise RuntimeError(  # noqa: TRY003
                f"StateDB schema validation failed: {status.error_message or status}")
    finally:
        await state_db.close()


if __name__ == "__main__":
    asyncio.run(validate())
