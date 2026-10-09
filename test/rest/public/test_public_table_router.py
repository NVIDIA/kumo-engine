# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import create_autospec

import kumoapi.source_table as source_table_api
import kumoapi.table as table_api
import pytest
from kumo_core.types.schema import FPID
from kumoapi.data_source import DEFAULT_PROJECT_ID
from kumoapi.typing import Dtype, Stype

from rest.config.table_v2 import TableDefinition
from rest.deps.authentication import Authentication
from rest.driver.table_driver import TableDriver
from rest.public.routers.table import TableRouter
from statedb.config.common import ProjectID, TableIDv2, TableName


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return False


@pytest.fixture
def _rest_install_dg_col_stats_fallback(
):  # pyright: ignore[reportUnusedFunction]
    yield


@pytest.fixture(scope='session')
def setup_rest_test_session():
    yield


@pytest.fixture
def auto_cleanup_per_func():
    yield


@pytest.fixture(scope='session')
def force_local_cache_dir_for_public_tests():
    yield


@pytest.fixture(scope='session')
def temporal_workers_ready():
    yield True


def _table_definition(
        path: str = 'events.parquet') -> table_api.TableDefinition:
    return table_api.TableDefinition(
        cols=[
            table_api.Column(
                name='event_id',
                stype=Stype.ID,
                dtype=Dtype.int64,
            )
        ],
        source_table=source_table_api.S3SourceTable(
            connector_id='connector_' + '1' * 32,
            source=source_table_api.S3RelativeSource(path=path),
            file_type=source_table_api.FileType.PARQUET,
        ),
        pkey='event_id',
    )


def _router() -> tuple[TableRouter, Any]:
    driver = cast(Any, create_autospec(TableDriver, instance=True))
    driver.create_full_table.return_value = _table_id()
    auth = cast(Authentication, SimpleNamespace())
    return TableRouter(lambda: None, driver, auth), driver


def _table_id() -> TableIDv2:
    return TableIDv2.public(
        table_name=TableName('NA'),
        fpid=FPID('a' * 32),
        pquery_id=None,
    )


@pytest.mark.asyncio
async def test_create_named_table_forwards_keyword_only_overwrite_args(
) -> None:
    router, driver = _router()
    request = _table_definition()

    result = await router.create_table(
        request,
        project_id=DEFAULT_PROJECT_ID,
        name_alias='sdk_smoke_table',
        force_rename=True,
    )

    expected = TableDefinition(**dataclasses.asdict(request))
    driver.create_full_table.assert_awaited_once_with(
        expected,
        alias='sdk_smoke_table',
        force_rename=True,
        project_id=ProjectID(DEFAULT_PROJECT_ID),
    )
    assert result.id == _table_id()
