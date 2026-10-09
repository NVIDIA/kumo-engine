# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import subprocess
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Optional

import asyncpg
import pytest
import pytest_asyncio

_POSTGRES_IMAGE = "postgres:16-alpine"
_POSTGRES_USER = "kumo_test"
_POSTGRES_PASSWORD = "kumo_test"
_POSTGRES_DB = "kumo_test"
_POSTGRES_ENV_VAR = "STATEDB_POSTGRES_DSN"


def _get_mapped_port(container_name: str) -> int:
    output = subprocess.check_output(
        ["docker", "port", container_name, "5432/tcp"],
        text=True,
    ).strip()
    host_port = output.rsplit(":", maxsplit=1)[-1]
    return int(host_port)


async def _wait_for_postgres(dsn: str, retries: int = 40) -> None:
    for _ in range(retries):
        try:
            connection = await asyncpg.connect(dsn)
        except Exception:
            await asyncio.sleep(0.25)
            continue
        await connection.close()
        return
    raise RuntimeError("Timed out waiting for postgres test server to become ready")


@pytest.fixture(scope="module")
def postgres_dsn() -> Generator[str, None, None]:
    configured_dsn = os.environ.get(_POSTGRES_ENV_VAR)
    if configured_dsn:
        yield configured_dsn
        return

    if subprocess.run(["docker", "info"],
                      stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode != 0:
        pytest.skip(
            "Docker is required for statedb Postgres tests unless "
            f"{_POSTGRES_ENV_VAR} is set."
        )

    container_name = f"kumo-statedb-postgres-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            container_name,
            "--env",
            f"POSTGRES_USER={_POSTGRES_USER}",
            "--env",
            f"POSTGRES_PASSWORD={_POSTGRES_PASSWORD}",
            "--env",
            f"POSTGRES_DB={_POSTGRES_DB}",
            "--publish",
            "127.0.0.1::5432",
            _POSTGRES_IMAGE,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    dsn: Optional[str] = None
    try:
        port = _get_mapped_port(container_name)
        dsn = (
            f"postgresql://{_POSTGRES_USER}:{_POSTGRES_PASSWORD}"
            f"@127.0.0.1:{port}/{_POSTGRES_DB}"
        )
        asyncio.run(_wait_for_postgres(dsn))
        yield dsn
    finally:
        subprocess.run(["docker", "rm", "--force", container_name],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)


@pytest_asyncio.fixture
async def postgres_connection(
    postgres_dsn: str) -> AsyncGenerator[asyncpg.Connection]:
    connection: asyncpg.Connection = await asyncpg.connect(postgres_dsn)
    try:
        yield connection
    finally:
        await connection.close()
