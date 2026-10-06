# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from io_utils import fsutil
from service_framework._raw_deploy_config import (
    PostgresMetadataStorageConfig,
    SqliteMetadataStorageConfig,
)
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage
from service_framework.postgres_dsn import (
    PostgresConn,
    postgres_conn_from_secret,
    postgresql_dsn,
)
from statedb.db.runtime import StateDB

PostgresPassword = Callable[[], Awaitable[str]]


def resolve_state_db_postgres_dsn(config: PostgresMetadataStorageConfig) -> str:
    if config.dsn:
        return config.dsn
    if config.dsn_secret_name:
        from io_utils.secrets import get_secrets
        secret = get_secrets(config.dsn_secret_name)
        conn = postgres_conn_from_secret(secret, config.dsn_secret_name)
        return postgresql_dsn(conn)
    conn = PostgresConn(
        username=config.username or '',
        password=None,
        host=config.host or '',
        port=str(config.port),
        database=config.database or '',
        sslmode='verify-full',
        sslrootcert=config.sslrootcert,
    )
    return postgresql_dsn(conn)


def _postgres_password(
    config: PostgresMetadataStorageConfig,
) -> PostgresPassword | None:
    if config.auth_mode == 'aws_iam':

        async def aws_password() -> str:
            import boto3

            def generate() -> str:
                return boto3.client(
                    'rds', region_name=config.region).generate_db_auth_token(
                        DBHostname=config.host,
                        Port=config.port,
                        DBUsername=config.username,
                    )

            return await asyncio.to_thread(generate)

        return aws_password
    if config.auth_mode == 'azure_entra':

        async def azure_password() -> str:
            from azure.identity.aio import WorkloadIdentityCredential
            credential = WorkloadIdentityCredential()
            try:
                token = await credential.get_token(
                    'https://ossrdbms-aad.database.windows.net/.default')
                return token.token
            finally:
                await credential.close()

        return azure_password
    return None


def default_sqlite_path(state_root: str) -> str:
    if fsutil.is_s3(state_root) or "://" in state_root:
        digest = hashlib.sha256(state_root.encode("utf-8")).hexdigest()[:16]
        root = Path("/tmp") / "kumo-state-db" / digest
    else:
        root = Path(state_root).expanduser().resolve()
    return str(root / "state.sqlite")


async def open_state_db_from_deploy_config(
    deploy_config: DeployEnvConfig,
    *,
    state_root: str,
) -> StateDB:
    storage = deploy_config.metadata_storage
    if isinstance(storage, PostgresMetadataStorageConfig):
        return await open_state_db_from_postgres_config(storage)
    if isinstance(storage, SqliteMetadataStorageConfig):
        if deploy_config.deploy_mode == DeployStage.PROD:
            raise RuntimeError(
                'SQLite metadata fallback is only allowed for non-prod '
                f'deploy_mode "{deploy_config.deploy_mode.value}".')
        sqlite_path = default_sqlite_path(state_root)
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        return await StateDB.open_sqlite(sqlite_path)
    raise TypeError(f"Unsupported State DB storage config: {storage!r}")


async def open_state_db_from_postgres_config(
    config: PostgresMetadataStorageConfig,
) -> StateDB:
    """Open StateDB through the production passwordless/bootstrap path."""
    return await StateDB.open_postgres(
        resolve_state_db_postgres_dsn(config),
        password=_postgres_password(config),
    )
