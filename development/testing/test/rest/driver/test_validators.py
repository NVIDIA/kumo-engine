# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kumo_core.types.schema import FPID, SourceTableID, VersionedSourceTableID
from pytest_mock import MockerFixture

from kumo.data.backend_cache import (
    AsyncDataTaskStatus,
    TableDataTaskResult,
    TableRefreshStage,
)
from kumo.validation.validation import ValidationError, ValidationResponse
from rest.backends.data_backend.context import DataBackendContext
from rest.driver.validators import TableValidator
from statedb.config.common import ProjectID, TableIDv1, TableIDv2, TableName

_PROJECT_ID = ProjectID("project_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


def _patch_backend_cache(mocker: MockerFixture, cache: AsyncMock) -> None:
    """Patch ``rest.driver.validators.get_active_cache`` to return the
    supplied AsyncMock. Patches at the call boundary because callers
    invoke ``get_active_cache()`` directly rather than going through a
    backend object.
    """
    mocker.patch('rest.driver.validators.get_active_cache', return_value=cache)


def _table_validator_with_state_table(data_backend_ctx: DataBackendContext):
    fpid = FPID("a" * 32)
    table_id = TableIDv2.public(
        table_name=TableName("table"),
        fpid=fpid,
        pquery_id=None,
    )
    table_definition = MagicMock()
    table_definition.source_table_id = SourceTableID("connector", "table")
    table_definition.cache_key.return_value = "table"
    table_definition.cols = []
    table_service = SimpleNamespace(get_table_entries=AsyncMock(
        return_value={fpid: SimpleNamespace(config=table_definition)}))
    return (
        TableValidator(
            data_backend_ctx=data_backend_ctx,
            job_base_dir="/tmp/jobs",
            state_services=SimpleNamespace(tables=table_service),
            project_id=_PROJECT_ID,
        ),
        table_id,
    )


@pytest.mark.asyncio
async def test_validate_ingested_data_basic(
        mocker: MockerFixture, data_backend_ctx: DataBackendContext):
    table_validator, table_id = _table_validator_with_state_table(
        data_backend_ctx)
    table_validator._table_logic = AsyncMock()
    cache = AsyncMock()
    # Call corresponding to failed ingestion will yield a more recent version
    cache.table_version_cache.get_many.coroutine.return_value = ([
        VersionedSourceTableID(
            'connector',
            'table',
            '3873873983',  # smaller
        )
    ])
    cache.table_refresh_result.get_many.coroutine.return_value = [
        TableDataTaskResult(
            validation=ValidationResponse(
                errors=[ValidationError(
                    'Error',
                    'table ingestion failed',
                )], ),
            status=AsyncDataTaskStatus.FAILED,
            stage=TableRefreshStage.INGEST_TABLE,
        )
    ]
    _patch_backend_cache(mocker, cache)
    result = (await table_validator.validate_tables([table_id]))
    assert len(result) == 1
    err = result[0][0]
    assert err.title == 'Error'
    assert err.description == 'table ingestion failed'
    assert err.is_error


@pytest.mark.asyncio
async def test_validate_ingested_data_with_None_version(
        mocker: MockerFixture, data_backend_ctx: DataBackendContext):
    table_validator, table_id = _table_validator_with_state_table(
        data_backend_ctx)
    table_validator._table_logic = AsyncMock()  # noqa
    cache = AsyncMock()
    cache.table_version_cache.get_many.coroutine.return_value = [
        None,
        VersionedSourceTableID(
            'connector',
            'table',
            '3873873983',
        ),
    ]
    cache.table_refresh_result.get_many.coroutine.return_value = [
        TableDataTaskResult(
            validation=ValidationResponse(
                errors=[
                    ValidationError(
                        'Error - only failure version present',
                        'table ingestion failed',
                    )
                ], ),
            status=AsyncDataTaskStatus.FAILED,
            stage=TableRefreshStage.INGEST_TABLE,
        )
    ]
    _patch_backend_cache(mocker, cache)
    result = (await table_validator.validate_tables([table_id]))

    assert len(result) == 2
    err = result[1][0]
    assert err.title == 'Error - only failure version present'
    assert err.description == 'table ingestion failed'
    assert err.is_error


@pytest.mark.asyncio
async def test_validate_ingested_data_both_None(
        mocker: MockerFixture, data_backend_ctx: DataBackendContext):
    table_validator = TableValidator(AsyncMock(), data_backend_ctx, AsyncMock(),
                                     project_id=_PROJECT_ID)
    table_validator._table_logic = AsyncMock()  # noqa
    cache = AsyncMock()
    cache.table_version_cache.get_many.coroutine.return_value = ([
        None,
        None,
    ])
    _patch_backend_cache(mocker, cache)
    connector_id = f"connector_{'1' * 32}"
    (await
     table_validator.validate_tables([TableIDv1(f'{connector_id}.table')]))
