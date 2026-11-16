# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast

import pytest

from service_framework._raw_deploy_config import (
    PostgresMetadataStorageConfig,
    SqliteMetadataStorageConfig,
)
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage
from statedb.db import (
    default_sqlite_path,
    open_state_db_from_deploy_config,
    open_state_db_from_postgres_config,
    resolve_state_db_postgres_dsn,
)
from statedb.db.config import _postgres_password


def test_resolve_state_db_postgres_dsn_prefers_literal_dsn() -> None:
    config = PostgresMetadataStorageConfig(
        dsn="postgresql://user:password@localhost/kumo")

    assert resolve_state_db_postgres_dsn(config) == (
        "postgresql://user:password@localhost/kumo")


def test_resolve_passwordless_postgres_dsn() -> None:
    config = PostgresMetadataStorageConfig(
        auth_mode="aws_iam",
        username="kumo",
        host="state.example.rds.amazonaws.com",
        database="state",
        region="us-west-2",
        sslrootcert="/var/run/secrets/kumo/ca.crt",
    )

    assert resolve_state_db_postgres_dsn(config) == (
        "postgresql://kumo@state.example.rds.amazonaws.com:5432/state"
        "?sslmode=verify-full&sslrootcert="
        "%2Fvar%2Frun%2Fsecrets%2Fkumo%2Fca.crt")


@pytest.mark.asyncio
async def test_aws_password_uses_rds_iam_token(monkeypatch) -> None:
    config = PostgresMetadataStorageConfig(
        auth_mode="aws_iam",
        username="kumo",
        host="state.example.rds.amazonaws.com",
        database="state",
        region="us-west-2",
    )
    calls = {}

    class Client:
        def generate_db_auth_token(self, **kwargs):
            calls.update(kwargs)
            return "token"

    monkeypatch.setattr("boto3.client", lambda service, region_name: Client())
    password = _postgres_password(config)

    assert password is not None
    assert await password() == "token"
    assert calls == {
        "DBHostname": config.host,
        "Port": 5432,
        "DBUsername": "kumo",
    }


@pytest.mark.asyncio
async def test_azure_password_uses_workload_identity(monkeypatch) -> None:
    config = PostgresMetadataStorageConfig(
        auth_mode="azure_entra",
        username="kumo-wi",
        host="state.postgres.database.azure.com",
        database="postgres",
    )

    class Credential:
        closed = False

        async def get_token(self, scope):
            assert scope == (
                "https://ossrdbms-aad.database.windows.net/.default")
            return SimpleNamespace(token="token")

        async def close(self):
            self.closed = True

    credential = Credential()
    monkeypatch.setattr(
        "azure.identity.aio.WorkloadIdentityCredential",
        lambda: credential,
    )
    password = _postgres_password(config)

    assert password is not None
    assert await password() == "token"
    assert credential.closed is True


@pytest.mark.asyncio
async def test_open_postgres_config_reuses_runtime_bootstrap(
        monkeypatch) -> None:
    config = PostgresMetadataStorageConfig(
        auth_mode="azure_entra",
        username="kumo-wi",
        host="state.postgres.database.azure.com",
        database="kumo",
        sslrootcert="/etc/ssl/certs/ca-certificates.crt",
    )
    captured = {}
    expected = object()

    async def open_postgres(dsn, *, password) -> object:
        captured.update(dsn=dsn, password=password)
        return expected

    monkeypatch.setattr(
        "statedb.db.config.StateDB.open_postgres",
        open_postgres,
    )

    actual = await open_state_db_from_postgres_config(config)

    assert actual is expected
    assert captured["dsn"] == resolve_state_db_postgres_dsn(config)
    assert captured["password"] is not None


def test_default_sqlite_path_uses_local_root() -> None:
    assert default_sqlite_path("/tmp/kumo-state") == (
        "/tmp/kumo-state/state.sqlite")


def test_default_sqlite_path_maps_uri_roots_to_local_cache() -> None:
    path = default_sqlite_path("s3://bucket/state")

    assert path.startswith("/tmp/kumo-state-db/")
    assert path.endswith("/state.sqlite")


@pytest.mark.asyncio
async def test_open_state_db_from_deploy_config_uses_sqlite(tmp_path) -> None:
    deploy_config = cast(
        DeployEnvConfig,
        SimpleNamespace(
            metadata_storage=SqliteMetadataStorageConfig(),
            deploy_mode=DeployStage.DEV,
        ),
    )

    state_db = await open_state_db_from_deploy_config(
        deploy_config,
        state_root=str(tmp_path),
    )

    assert state_db.backend == "sqlite"
    assert state_db.bootstrap_result.created_schema_row is True


@pytest.mark.asyncio
async def test_open_state_db_from_deploy_config_blocks_sqlite_in_prod(
        tmp_path) -> None:
    deploy_config = cast(
        DeployEnvConfig,
        SimpleNamespace(
            metadata_storage=SqliteMetadataStorageConfig(),
            deploy_mode=DeployStage.PROD,
        ),
    )

    with pytest.raises(RuntimeError, match="SQLite metadata fallback"):
        await open_state_db_from_deploy_config(
            deploy_config,
            state_root=str(tmp_path),
        )
