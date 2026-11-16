# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import kumoapi.table as table_api
import pytest
from kumoapi.data_source import DataSourceType as ApiDataSourceType
from kumoapi.source_table import S3SourceTable as ApiS3SourceTable
from kumoapi.typing import Dtype, Stype

from kumo.data.backend_cache import TableDataTaskResult, TableIngestionInfo
from kumo.data.cache.table_version_cache import TableIngestionStatus
from kumo.validation.validation import ValidationResponse
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.file_type import FileType
from rest.config.table_v2 import (
    Column,
    DataSourceType,
    S3SourceTable,
    TableDBEntryV2,
    TableDefinition,
)
from rest.deps.authentication import UserAuthorization
from rest.deps.context import _current_user_ctx
from rest.driver.table_driver import TableDriver
from rest.driver.validators import TableValidator
from statedb.config.common import ConnectorID, ProjectID, TableIDv2, TableName
from statedb.config.table import TableDataVersion
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.connector_authorization import CONNECTOR_TABLE_SOURCE_ACTIONS
from statedb.utils.errors import TableDoesNotExistError

_TEST_PROJECT_ID = ProjectID("project_" + "1" * 32)
_TEST_CONNECTOR_ID = ConnectorID("connector_" + "1" * 32)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func(monkeypatch):
    monkeypatch.setattr(TableDriver, "_require_project_access", AsyncMock())
    yield


@contextmanager
def _authenticated_principal():
    token = _current_user_ctx.set(UserAuthorization.disabled())
    try:
        yield _current_user_ctx.get()
    finally:
        _current_user_ctx.reset(token)


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


def _timestamps() -> ResourceTimestamps:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ResourceTimestamps(created_at=now, updated_at=now)


def _table_definition(
    source_table_name: str = "customers",
    *,
    connector_id: ConnectorID = _TEST_CONNECTOR_ID,
) -> TableDefinition:
    return TableDefinition(
        source_table=S3SourceTable(
            source={
                'mode': 'relative',
                'path': source_table_name
            },
            connector_id=connector_id,
            file_type=FileType.CSV,  # type: ignore[arg-type]
            data_source_type=DataSourceType.S3,
        ),
        cols=[Column(name="customer_id", stype=Stype.ID, dtype=Dtype.int64)],
        pkey="customer_id",
    )


def _table_entry(
    table_definition: TableDefinition,
    alias: str = "customers",
    *,
    project_id: ProjectID = _TEST_PROJECT_ID,
) -> TableDBEntryV2:
    return TableDBEntryV2(
        id=table_definition.fpid,
        project_id=project_id,
        alias=alias,
        config=table_definition,
        col_dtypes={
            col.name: col.dtype
            for col in table_definition.cols if col.dtype
        },
        timestamps=_timestamps(),
    )


def _metadata_request(
    connector_id: ConnectorID = _TEST_CONNECTOR_ID,
) -> table_api.TableMetadataRequest:
    return table_api.TableMetadataRequest(
        source_table=ApiS3SourceTable(
            source={
                'mode': 'relative',
                'path': "customers"
            },
            connector_id=connector_id,
            file_type=FileType.CSV,  # type: ignore[arg-type]
            data_source_type=ApiDataSourceType.S3,
        ),
        cols=[
            table_api.ColumnMetadataRequest(
                name="customer_id",
                stype=Stype.ID,
                dtype=Dtype.int64,
            )
        ],
        pkey="customer_id",
    )


def _driver(state_services):
    driver = TableDriver(
        data_backend_ctx=MagicMock(),
        data_scale_limits=MagicMock(),
    )
    driver._configure_state_services(state_services)
    return driver


@pytest.mark.asyncio
async def test_infer_table_metadata_uses_state_services_connector():
    reference = ConnectorExecutionRef(project_id=_TEST_PROJECT_ID,
                                      connector_id=_TEST_CONNECTOR_ID)
    connector_service = SimpleNamespace(execution_ref_for_project=AsyncMock(
        return_value=reference))
    driver = _driver(
        SimpleNamespace(
            tables=SimpleNamespace(),
            connectors=connector_service,
            projects=SimpleNamespace(),
        ))
    expected_response = MagicMock()

    with patch(
            "rest.driver.table_driver.DataBackend._execute_single_activity",
            new=AsyncMock(return_value=expected_response),
    ) as execute_activity:
        result = await driver.infer_table_metadata(_metadata_request(),
                                                   str(_TEST_PROJECT_ID))

    assert result == expected_response
    connector_service.execution_ref_for_project.assert_awaited_once_with(
        None, _TEST_PROJECT_ID, _TEST_CONNECTOR_ID,
        CONNECTOR_TABLE_SOURCE_ACTIONS)
    # A physical (non-SQL) table keeps the prior 2-arg [reference, request]
    # dispatch arity (no Custom SQL stype-inheritance 3rd arg), so a rolling
    # deploy to an old 2-param data-backend worker still infers physical
    # schemas. Only a Custom SQL table appends the reference-stype hints.
    assert execute_activity.await_args.args[2] == [
        reference, _metadata_request()
    ]


@pytest.mark.asyncio
async def test_create_full_table_project_scope_uses_state_services():
    project_id = ProjectID.generate()
    table_definition = _table_definition()
    table_entry = _table_entry(table_definition, project_id=project_id)
    table_id = TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=table_definition.fpid,
    )

    table_service = SimpleNamespace(
        get_table_entry_by_alias=AsyncMock(return_value=None),
        get_table_entry=AsyncMock(return_value=table_entry),
        save_table_definition=AsyncMock(return_value=table_entry),
    )
    project_service = SimpleNamespace(
        validate_project_for_creation=AsyncMock(), )
    driver = _driver(
        SimpleNamespace(tables=table_service, projects=project_service))

    with _authenticated_principal() as principal:
        result = await driver.create_full_table(
            table=table_definition,
            alias="customers",
            force_rename=False,
            project_id=str(project_id),
        )

    assert result == table_id
    project_service.validate_project_for_creation.assert_awaited_once_with(
        str(project_id),
        principal,
    )
    table_service.get_table_entry_by_alias.assert_awaited_once_with(
        project_id, "customers")
    table_service.get_table_entry.assert_awaited_once_with(
        project_id, table_definition.fpid)
    table_service.save_table_definition.assert_awaited_once()
    assert table_service.save_table_definition.await_args.args == (
        project_id,
        table_definition,
    )
    assert table_service.save_table_definition.await_args.kwargs == {
        "alias": "customers",
        "force_rename": False,
        "increment_ref_count": False,
        "created_by": "_AUTHORIZATION_DISABLED_",
        "created_by_name": "",
    }


@pytest.mark.asyncio
async def test_create_full_table_new_definition_validation_uses_state_services(
):
    project_id = ProjectID.generate()
    table_definition = _table_definition()
    table_entry = _table_entry(table_definition, project_id=project_id)
    reference = ConnectorExecutionRef(project_id=project_id,
                                      connector_id=_TEST_CONNECTOR_ID)

    table_service = SimpleNamespace(
        get_table_entry_by_alias=AsyncMock(return_value=None),
        get_table_entry=AsyncMock(return_value=None),
        save_table_definition=AsyncMock(return_value=table_entry),
    )
    connector_service = SimpleNamespace(execution_ref_for_project=AsyncMock(
        return_value=reference))
    project_service = SimpleNamespace(
        validate_project_for_creation=AsyncMock(), )
    driver = _driver(
        SimpleNamespace(
            tables=table_service,
            connectors=connector_service,
            projects=project_service,
        ))

    with _authenticated_principal() as principal, patch(
            "rest.driver.table_driver.DataBackend._execute_single_activity",
            new=AsyncMock(return_value=None),
    ) as execute_activity:
        result = await driver.create_full_table(
            table=table_definition,
            alias="customers",
            force_rename=False,
            project_id=str(project_id),
        )

    assert TableIDv2.parse(result)[2] == table_definition.fpid
    table_service.get_table_entry.assert_awaited_once_with(
        project_id, table_definition.fpid)
    connector_service.execution_ref_for_project.assert_awaited_once_with(
        principal, project_id, table_definition.get_connector_id(),
        CONNECTOR_TABLE_SOURCE_ACTIONS)
    assert execute_activity.await_args.args[2] == [
        reference, table_definition, {}
    ]
    table_service.save_table_definition.assert_awaited_once()


@pytest.mark.asyncio
async def test_validate_table_uses_state_services():
    project_id = ProjectID.generate()
    table_definition = _table_definition()
    reference = ConnectorExecutionRef(project_id=project_id,
                                      connector_id=_TEST_CONNECTOR_ID)
    connector_service = SimpleNamespace(execution_ref_for_project=AsyncMock(
        return_value=reference))
    driver = _driver(
        SimpleNamespace(
            tables=SimpleNamespace(),
            connectors=connector_service,
            projects=SimpleNamespace(),
        ))
    limits = driver._data_scale_limits.parse_from_env.return_value
    limits.max_table_size_bytes_by_connector = {"s3": 123}

    with patch(
            "rest.driver.table_driver.DataBackend._execute_single_activity",
            new=AsyncMock(return_value=None),
    ) as execute_activity:
        result = await driver.validate_table(table_definition,
                                             project_id=str(project_id))

    assert result == ValidationResponse()
    connector_service.execution_ref_for_project.assert_awaited_once_with(
        None, project_id, table_definition.get_connector_id(),
        CONNECTOR_TABLE_SOURCE_ACTIONS)
    assert execute_activity.await_args.args[2] == [
        reference, table_definition, {
            "s3": 123
        }
    ]


@pytest.mark.asyncio
async def test_table_validator_uses_state_table_service_for_v2_definitions():
    table_definition = _table_definition()
    table_id = TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=table_definition.fpid,
    )
    table_service = SimpleNamespace(get_table_entries=AsyncMock(
        return_value={table_definition.fpid: _table_entry(table_definition)}))
    validator = TableValidator(
        data_backend_ctx=MagicMock(),
        job_base_dir="/tmp",
        state_services=SimpleNamespace(tables=table_service),
        project_id=_TEST_PROJECT_ID,
    )

    version_cache_get_many = AsyncMock(return_value=[None])
    refresh_result_get_many = AsyncMock(return_value=[])
    cache = SimpleNamespace(
        table_version_cache=SimpleNamespace(get_many=SimpleNamespace(
            coroutine=version_cache_get_many)),
        table_refresh_result=SimpleNamespace(get_many=SimpleNamespace(
            coroutine=refresh_result_get_many)),
    )

    with patch("rest.driver.validators.get_active_cache", return_value=cache):
        result = await validator.validate_tables([table_id])

    assert result == [[]]
    table_service.get_table_entries.assert_awaited_once_with(
        _TEST_PROJECT_ID, [table_definition.fpid])
    version_cache_get_many.assert_awaited_once_with(
        [table_definition.source_table_id],
        TableIngestionStatus.COMPLETED,
    )
    refresh_result_get_many.assert_awaited_once_with([], TableDataTaskResult)


@pytest.mark.asyncio
async def test_create_full_table_force_rename_uses_state_project_service():
    project_id = ProjectID.generate()
    table_definition = _table_definition()
    table_entry = _table_entry(table_definition, project_id=project_id)
    table_id = TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=table_definition.fpid,
    )

    table_service = SimpleNamespace(
        get_table_entry=AsyncMock(return_value=table_entry),
        save_table_definition=AsyncMock(return_value=table_entry),
    )
    project_service = SimpleNamespace(
        validate_project_for_creation=AsyncMock(), )
    driver = _driver(
        SimpleNamespace(tables=table_service, projects=project_service))

    with _authenticated_principal() as principal:
        result = await driver.create_full_table(
            table=table_definition,
            alias="customers",
            force_rename=True,
            project_id=str(project_id),
        )

    assert result == table_id
    project_service.validate_project_for_creation.assert_awaited_once_with(
        str(project_id),
        principal,
    )
    table_service.save_table_definition.assert_awaited_once()


@pytest.mark.asyncio
async def test_project_table_read_delete_and_validate_use_state_services():
    project_id = ProjectID.generate()
    table_definition = _table_definition()
    table_id = TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=table_definition.fpid,
    )
    table_entry = _table_entry(table_definition, project_id=project_id)
    table_service = SimpleNamespace(
        get_table_entry_by_alias=AsyncMock(return_value=table_entry),
        list_named_tables=AsyncMock(return_value=[table_entry]),
        clear_table_alias=AsyncMock(return_value=table_entry),
    )
    driver = _driver(
        SimpleNamespace(tables=table_service, projects=SimpleNamespace()))

    resolved = await driver._resolve_project_table(str(project_id), "customers")
    listed = await driver._list_project_table_names(str(project_id))
    validation = await driver._validate_project_table_name(
        str(project_id),
        "customers",
        exclude_table_id=table_id,
    )
    deleted = await driver.delete_table_alias(
        "customers",
        project_id=str(project_id),
    )

    assert resolved == table_entry
    assert len(listed) == 1
    assert listed[0].name == "customers"
    assert listed[0].id == table_id
    assert validation.is_valid
    assert deleted == table_id
    assert table_service.get_table_entry_by_alias.await_count == 2
    table_service.list_named_tables.assert_awaited_once_with(project_id)
    table_service.clear_table_alias.assert_awaited_once_with(
        project_id, "customers")


@pytest.mark.asyncio
async def test_get_table_validation_uses_state_table_service():
    table_definition = _table_definition()
    table_id = TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=table_definition.fpid,
    )
    table_service = SimpleNamespace(get_table_entry=AsyncMock(
        return_value=_table_entry(table_definition)))
    driver = _driver(
        SimpleNamespace(tables=table_service, projects=SimpleNamespace()))

    version_cache_get = AsyncMock(return_value="version-1")
    cache = SimpleNamespace(table_version_cache=SimpleNamespace(
        get=SimpleNamespace(coroutine=version_cache_get)))
    expected_result = MagicMock(spec=TableDataTaskResult)
    get_refresh_results = AsyncMock(return_value=[expected_result])

    with patch(
            "rest.driver.table_driver.get_active_cache", return_value=cache
    ), patch(
            "rest.driver.table_driver.DataBackend.get_many_table_refresh_results",
            new=get_refresh_results,
    ):
        result = await driver.get_table_validation(str(table_id),
                                                   str(_TEST_PROJECT_ID))

    assert result == expected_result
    table_service.get_table_entry.assert_awaited_once_with(
        _TEST_PROJECT_ID, table_definition.fpid)
    version_cache_get.assert_awaited_once_with(
        table_definition.get_connector_id(),
        table_definition.get_source_table_name(),
    )
    get_refresh_results.assert_awaited_once_with([
        (table_definition, TableDataVersion("version-1"))
    ])


@pytest.mark.asyncio
async def test_get_table_last_ingested_timestamp_uses_state_table_service():
    table_definition = _table_definition()
    table_id = TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=table_definition.fpid,
    )
    table_service = SimpleNamespace(get_table_definition=AsyncMock(
        return_value=table_definition))
    driver = _driver(
        SimpleNamespace(tables=table_service, projects=SimpleNamespace()))

    ingest_start_time = datetime(2026, 1, 2, tzinfo=timezone.utc)
    ingest_end_time = datetime(2026, 1, 3, tzinfo=timezone.utc)
    version_cache_get_many = AsyncMock(
        return_value=[SimpleNamespace(version="version-1")])
    ingestion_info_get_many = AsyncMock(return_value=[
        TableIngestionInfo(
            ingest_start_time=ingest_start_time,
            ingest_end_time=ingest_end_time,
            temporal_workflow_id="workflow-id",
            code_version="test-version",
        )
    ])
    cache = SimpleNamespace(
        table_version_cache=SimpleNamespace(get_many=SimpleNamespace(
            coroutine=version_cache_get_many)),
        ingestion_info=SimpleNamespace(get_many=SimpleNamespace(
            coroutine=ingestion_info_get_many)),
    )

    with patch("rest.driver.table_driver.get_active_cache", return_value=cache):
        result = await driver.get_table_last_ingested_timestamp(
            table_id, str(_TEST_PROJECT_ID))

    assert result == ingest_start_time
    table_service.get_table_definition.assert_awaited_once_with(
        _TEST_PROJECT_ID, table_definition.fpid)
    version_cache_get_many.assert_awaited_once_with(
        tables=[table_definition.source_table_id])
    ingestion_info_get_many.assert_awaited_once_with(
        [(
            table_definition.get_connector_id(),
            table_definition.get_source_table_name(),
            "version-1",
        )],
        TableIngestionInfo,
    )


@pytest.mark.asyncio
async def test_refresh_table_uses_state_table_service():
    table_definition = _table_definition()
    table_id = TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=table_definition.fpid,
    )
    table_entry = _table_entry(table_definition)
    table_service = SimpleNamespace(get_table_entry=AsyncMock(
        return_value=table_entry))
    execution_spec = ConnectorExecutionRef(
        project_id=_TEST_PROJECT_ID,
        connector_id=table_definition.get_connector_id(),
        authorization_token='version-token',
    )
    refresh_execution_spec = dataclasses.replace(
        execution_spec, authorization_token='refresh-token')
    connector_service = SimpleNamespace(
        execution_ref_for_project=AsyncMock(return_value=execution_spec),
        fresh_execution_refs_for_workflow=AsyncMock(
            return_value=(refresh_execution_spec, )),
    )
    driver = _driver(
        SimpleNamespace(
            tables=table_service,
            connectors=connector_service,
            projects=SimpleNamespace(),
        ))

    version_cache_get = AsyncMock(return_value="old-version")
    cache = SimpleNamespace(table_version_cache=SimpleNamespace(
        get=SimpleNamespace(coroutine=version_cache_get)))
    get_table_data_version = AsyncMock(return_value="version-1")
    get_refresh_results = AsyncMock(return_value=[None])
    start_refresh_table = AsyncMock()

    with patch(
            "rest.driver.table_driver.get_active_cache", return_value=cache
    ), patch(
            "rest.driver.table_driver.DataBackend."
            "get_table_data_version_in_activity",
            new=get_table_data_version,
    ), patch(
            "rest.driver.table_driver.DataBackend.get_many_table_refresh_results",
            new=get_refresh_results,
    ), patch(
            "rest.driver.table_driver.DataBackend.start_refresh_table",
            new=start_refresh_table,
    ):
        result = await driver.refresh_table(table_id, str(_TEST_PROJECT_ID))

    assert result is None
    table_service.get_table_entry.assert_awaited_once_with(
        _TEST_PROJECT_ID, table_definition.fpid)
    connector_service.execution_ref_for_project.assert_awaited_once_with(
        None, _TEST_PROJECT_ID, table_definition.get_connector_id(),
        CONNECTOR_TABLE_SOURCE_ACTIONS)
    version_cache_get.assert_awaited_once_with(
        table_definition.get_connector_id(),
        str(table_definition.get_source_table_name()),
    )
    get_table_data_version.assert_awaited_once_with(
        driver._data_backend_ctx,
        execution_spec,
        str(table_definition.get_source_table_name()),
        refresh_table_version=True,
        table_def=table_definition,
    )
    connector_service.fresh_execution_refs_for_workflow.assert_awaited_once_with(
        None,
        (execution_spec, ),
        CONNECTOR_TABLE_SOURCE_ACTIONS,
    )
    get_refresh_results.assert_awaited_once_with([
        (table_definition, TableDataVersion("version-1"))
    ])
    start_refresh_table.assert_awaited_once_with(
        driver._data_backend_ctx,
        refresh_execution_spec,
        table_definition,
        TableDataVersion("version-1"),
        alias=TableName("customers"),
    )


@pytest.mark.asyncio
async def test_refresh_table_state_service_missing_table_avoids_storage():
    table_definition = _table_definition()
    table_id = TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=table_definition.fpid,
    )
    table_service = SimpleNamespace(get_table_entry=AsyncMock(
        return_value=None))
    driver = _driver(
        SimpleNamespace(tables=table_service, projects=SimpleNamespace()))

    with pytest.raises(TableDoesNotExistError):
        await driver.refresh_table(table_id, str(_TEST_PROJECT_ID))

    table_service.get_table_entry.assert_awaited_once_with(
        _TEST_PROJECT_ID, table_definition.fpid)


@pytest.mark.asyncio
async def test_get_col_stats_and_sample_rows_uses_state_services():
    table_definition = _table_definition()
    table_id = TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=table_definition.fpid,
    )
    table_service = SimpleNamespace(get_table_definition=AsyncMock(
        return_value=table_definition))
    reference = ConnectorExecutionRef(
        project_id=_TEST_PROJECT_ID,
        connector_id=table_definition.get_connector_id(),
    )
    connector_service = SimpleNamespace(execution_ref_for_project=AsyncMock(
        return_value=reference))
    driver = _driver(
        SimpleNamespace(
            tables=table_service,
            connectors=connector_service,
            projects=SimpleNamespace(),
        ))

    expected_response = MagicMock()
    with patch(
            "rest.driver.table_driver.DataBackend._execute_single_activity",
            new=AsyncMock(return_value=expected_response),
    ) as execute_activity:
        result = await driver.get_col_stats_and_sample_rows(
            table_id, str(_TEST_PROJECT_ID))

    assert result == expected_response
    table_service.get_table_definition.assert_awaited_once_with(
        _TEST_PROJECT_ID, table_definition.fpid)
    connector_service.execution_ref_for_project.assert_awaited_once_with(
        None, _TEST_PROJECT_ID, table_definition.get_connector_id(),
        CONNECTOR_TABLE_SOURCE_ACTIONS)
    assert execute_activity.await_args.args[2] == [reference, table_definition]


@pytest.mark.asyncio
async def test_get_col_stats_and_sample_rows_missing_state_table_avoids_storage(
):
    table_definition = _table_definition()
    table_id = TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=table_definition.fpid,
    )
    table_service = SimpleNamespace(get_table_definition=AsyncMock(
        return_value=None))
    connector_service = SimpleNamespace(get_required=AsyncMock())
    driver = _driver(
        SimpleNamespace(
            tables=table_service,
            connectors=connector_service,
            projects=SimpleNamespace(),
        ))

    with pytest.raises(TableDoesNotExistError):
        await driver.get_col_stats_and_sample_rows(table_id,
                                                   str(_TEST_PROJECT_ID))

    table_service.get_table_definition.assert_awaited_once_with(
        _TEST_PROJECT_ID, table_definition.fpid)
    connector_service.get_required.assert_not_awaited()


@pytest.mark.asyncio
async def test_table_to_sdk_uses_state_table_service():
    table_definition = _table_definition()
    table_id = TableIDv2.public(
        pquery_id=None,
        table_name=TableName("NA"),
        fpid=table_definition.fpid,
    )
    table_entry = _table_entry(table_definition)
    table_service = SimpleNamespace(get_table_entries=AsyncMock(
        return_value={table_definition.fpid: table_entry}))
    connector_service = SimpleNamespace(
        source_table_connector_for_project=AsyncMock(
            return_value=SimpleNamespace(name="Shared source")))
    driver = _driver(
        SimpleNamespace(
            tables=table_service,
            connectors=connector_service,
            projects=SimpleNamespace(),
        ))

    result = await driver.table_to_sdk(table_id, str(_TEST_PROJECT_ID))

    assert 'kumo.S3Connector.get_by_name("Shared source")' in result
    assert f'project="{_TEST_PROJECT_ID}"' in result
    assert "customers = kumo.Table(" in result
    table_service.get_table_entries.assert_awaited_once_with(
        _TEST_PROJECT_ID, [table_definition.fpid])
    connector_service.source_table_connector_for_project.assert_awaited_once_with(
        _TEST_PROJECT_ID,
        _TEST_CONNECTOR_ID,
        None,
    )
