# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import os
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from kumoapi import table as table_api
from kumoapi.data_source import DataSourceType as ApiDataSourceType
from kumoapi.source_table import ConnectorType as ApiSourceConnectorType
from kumoapi.source_table import \
    CustomSQLSourceTable as ApiCustomSQLSourceTable
from kumoapi.source_table import S3RelativeSource, S3SourceTable
from kumoapi.typing import Dtype
from kumoapi.typing import Dtype as InternalDtype
from kumoapi.typing import Stype

from common_utils.error import PublicException
from kumo.connector.connector import ConnectorError
from kumo.data import reset_active_cache_for_test
from kumo.validation import ValidationException
from rest.backends.data_backend.activities import custom_table_validation
from rest.backends.data_backend.context import DataBackendContext
from rest.config.connector import FileConnectorResourceConfig
from rest.config.connector_db import ConnectorDBEntry, InternalConnectorDBEntry
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.file_type import FileType
from rest.config.table_v2 import Column as RestColumn
from rest.config.table_v2 import \
    CustomSQLSourceTable as RestCustomSQLSourceTable
from rest.config.table_v2 import TableDBEntryV2
from rest.config.table_v2 import TableDefinition as RestTableDefinition
from rest.config.table_v2 import TableKind
from rest.driver import table_logic as table_logic_module
from rest.driver.error_constants import TABLE_COLUMN_NOT_FOUND_TITLE
from rest.driver.table_driver import TableDriver
from rest.driver.table_logic import (
    _get_sample_rows,
    _resolve_table_metadata_with_connector,
    _validate_col_names_v2,
    _validate_cols_are_unique_v2,
    _validate_pkey_and_time_col_v2,
    infer_table_metadata_activity,
    validate_table_definition_activity,
)
from rest.settings import DataScaleLimits
from statedb.config.common import ConnectorID, ProjectID
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.connector_authorization import CONNECTOR_TABLE_SOURCE_ACTIONS

_TEST_PROJECT_ID = ProjectID("project_" + "1" * 32)
_TEST_CONNECTOR_ID = ConnectorID("connector_" + "1" * 32)
_SECOND_CONNECTOR_ID = ConnectorID("connector_" + "2" * 32)


@pytest.fixture(scope="function")
def temp_csv_dir() -> Generator[str, None, None]:
    """Create a temporary directory with CSV test data.

    Generates a customers.csv file with the following structure:

    +-----------+--------------+-----+------------+----------------+----------+
    |customer_id| customer_name| age | signup_date| account_balance| is_active|
    +===========+==============+=====+============+================+==========+
    | 1         | Alice        | 25  | 2023-01-15 | 1000.50        | True     |
    | 2         | Bob          | 30  | 2023-02-20 | 2500.75        | True     |
    | 3         | Charlie      | 35  | 2023-03-10 | 3200.00        | False    |
    | 4         | David        | 40  | 2023-04-05 | 1500.25        | True     |
    | 5         | Eve          | 45  | 2023-05-12 | 4100.80        | True     |
    +-----------+--------------+-----+------------+----------------+----------+
    """
    temp_dir = None
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        # Create sample CSV data with various column types
        df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4, 5],
            "customer_name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "signup_date": [
                "2023-01-15",
                "2023-02-20",
                "2023-03-10",
                "2023-04-05",
                "2023-05-12",
            ],
            "account_balance": [
                1000.50,
                2500.75,
                3200.00,
                1500.25,
                4100.80,
            ],
            "is_active": [True, True, False, True, True],
        })
        csv_path = os.path.join(temp_dir, "customers.csv")
        df.to_csv(csv_path, index=False)

        yield temp_dir
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception as e:
                import logging

                logging.warning(
                    f"Failed to delete test directory {temp_dir}: {e}")


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    async def execute_test_activity(_ctx, activity_fn, activity_inputs,
                                    _workflow_id_prefix):
        if activity_fn is infer_table_metadata_activity:
            return _metadata_response(activity_inputs[1])
        if activity_fn is validate_table_definition_activity:
            definition = activity_inputs[1]
            _validate_col_names_v2(definition)
            _validate_cols_are_unique_v2(definition)
            _validate_pkey_and_time_col_v2(definition)
            return None
        return None

    # The data-backend activities are fully mocked here, so the driver never
    # touches the (deploy_env-derived) data persist dir; just drop the lazy
    # BackendCacheManager around the test for isolation.
    with patch.object(
            TableDriver,
            "_require_project_access",
            new=AsyncMock(),
    ), patch(
            "rest.driver.table_driver.DataBackend._execute_single_activity",
            new=AsyncMock(side_effect=execute_test_activity),
    ):
        reset_active_cache_for_test()
        yield
        reset_active_cache_for_test()


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


def _create_table_driver(mock_connectors_dao: AsyncMock) -> TableDriver:
    """Helper function to create a TableDriver with mocked dependencies.

    Args:
        mock_connectors_dao: Mocked connectors DAO to use

    Returns:
        TableDriver instance with all dependencies mocked
    """
    data_backend_ctx = AsyncMock(spec=DataBackendContext)
    data_scale_limits = DataScaleLimits.parse_from_env()

    table_driver = TableDriver(
        data_backend_ctx=data_backend_ctx,
        data_scale_limits=data_scale_limits,
    )
    table_driver._configure_state_services(
        SimpleNamespace(
            tables=SimpleNamespace(),
            projects=SimpleNamespace(),
        ))
    return table_driver


def _configure_local_connector(
    table_driver: TableDriver,
    root_dir: str,
    connector_id: ConnectorID = _TEST_CONNECTOR_ID,
) -> "_ConnectorService":
    connector_service = _ConnectorService()
    config = FileConnectorResourceConfig(
        name=str(connector_id),
        root_dir=root_dir,
    )
    connector_service.entries[connector_id] = _connector_entry(
        config,
        "Project-owned test file connector",
        connector_id=connector_id,
    )
    table_driver._configure_state_services(
        SimpleNamespace(
            tables=SimpleNamespace(),
            projects=SimpleNamespace(),
            connectors=connector_service,
        ))
    return connector_service


class _ConnectorService:
    def __init__(self):
        self.entries = {}
        self.execution_ref_calls = []

    async def execution_ref_for_project(self, principal, project_id,
                                        connector_id, actions):
        self.execution_ref_calls.append(
            (principal, project_id, connector_id, actions))
        entry = self.entries[ConnectorID(connector_id)]
        if entry.project_id != project_id:
            raise KeyError(connector_id)
        return ConnectorExecutionRef(project_id=project_id,
                                     connector_id=entry.id)


def _connector_entry(
    config: FileConnectorResourceConfig,
    description: str,
    *,
    connector_id: ConnectorID = _TEST_CONNECTOR_ID,
    project_id: ProjectID = _TEST_PROJECT_ID,
) -> ConnectorDBEntry:
    return InternalConnectorDBEntry(
        id=connector_id,
        project_id=project_id,
        name=config.name,
        normalized_name=config.name,
        config=config,
        timestamps=ResourceTimestamps(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        description=description,
    )


def _create_table_metadata_request(
    source_table_name: str,
    column_names: list[str],
    pkey: str,
    time_col: str | None = None,
    end_time_col: str | None = None,
    timestamp_format: table_api.TimestampUnit | None = None,
    connector_id: str | ConnectorID = _TEST_CONNECTOR_ID,
) -> table_api.TableMetadataRequest:
    """Helper function to create a TableMetadataRequest with column definitions.

    Args:
        source_table_name: Name of the source table file
        column_names: List of column names to include
        pkey: Primary key column name
        time_col: Optional time column name
        end_time_col: Optional end time column name
        timestamp_format: Optional timestamp format for timestamp columns

    Returns:
        TableMetadataRequest object with specified columns
    """
    # Mapping of known column names to their types
    column_specs = {
        # Customer columns
        "customer_id": (Stype.categorical, Dtype.int64),
        "customer_name": (Stype.categorical, Dtype.string),
        "age": (Stype.numerical, Dtype.int64),
        "signup_date": (Stype.timestamp, Dtype.time),
        "account_balance": (Stype.numerical, Dtype.float64),
        "is_active": (Stype.categorical, Dtype.bool),
        # Order columns
        "order_id": (Stype.categorical, Dtype.int64),
        "order_timestamp": (Stype.timestamp, Dtype.int64),
        "amount": (Stype.numerical, Dtype.float64),
        # Test columns
        "non_existent_column": (Stype.categorical, Dtype.string),
    }

    cols: List[table_api.ColumnMetadataRequest] = []
    for col_name in column_names:
        if col_name in column_specs:
            stype, dtype = column_specs[col_name]
        else:
            # Default to categorical string for unknown columns
            stype, dtype = Stype.categorical, Dtype.string

        # Determine if this column should have timestamp_format
        col_timestamp_format = None
        if stype == Stype.timestamp and timestamp_format is not None:
            col_timestamp_format = timestamp_format
        cols.append(
            table_api.ColumnMetadataRequest(
                name=col_name,
                stype=stype,
                dtype=dtype,
                timestamp_format=col_timestamp_format,
            ))

    return table_api.TableMetadataRequest(
        source_table=S3SourceTable(
            source={
                'mode': 'relative',
                'path': source_table_name
            },
            connector_id=connector_id,
            file_type=FileType.CSV,  # pyright: ignore
            data_source_type=ApiDataSourceType.S3,
        ),
        cols=cols,  # type: ignore
        pkey=pkey,
        time_col=time_col,
        end_time_col=end_time_col,
    )


def _metadata_response(
    request: table_api.TableMetadataRequest,
) -> table_api.TableMetadataResponse:
    return table_api.TableMetadataResponse(
        cols=[
            table_api.Column(
                name=col.name,
                stype=col.stype,
                dtype=col.dtype,
                timestamp_format=col.timestamp_format,
            ) for col in request.cols
        ],
        source_table=request.source_table,
        pkey=request.pkey,
        time_col=request.time_col,
        end_time_col=request.end_time_col,
    )


@pytest.mark.asyncio
async def test_infer_table_metadata_activity_closes_connector():
    connector = MagicMock()
    request = _create_table_metadata_request('customers', ['customer_id'],
                                             pkey='customer_id')
    expected = _metadata_response(request)

    with patch.object(
            table_logic_module,
            'get_connector_from_spec_async',
            new=AsyncMock(return_value=connector),
    ), patch.object(
            table_logic_module,
            '_resolve_table_metadata_with_connector',
            new=AsyncMock(return_value=(expected, None, MagicMock())),
    ):
        result = await infer_table_metadata_activity(
            MagicMock(),
            request,
        )

    assert result == expected
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'request_stype,request_dtype',
    [
        (Stype.multicategorical, Dtype.string),
        (None, None),
    ],
)
async def test_infer_table_metadata_preserves_multicategorical_delimiter(
        request_stype, request_dtype):
    request = table_api.TableMetadataRequest(
        source_table=S3SourceTable(
            source=S3RelativeSource(path='customers'),
            connector_id=_TEST_CONNECTOR_ID,
            file_type=FileType.CSV,  # pyright: ignore
            data_source_type=ApiDataSourceType.S3,
        ),
        cols=[
            table_api.ColumnMetadataRequest(
                name='customer_id',
                stype=Stype.ID,
                dtype=Dtype.int64,
            ),
            table_api.ColumnMetadataRequest(
                name='tags',
                stype=request_stype,
                dtype=request_dtype,
                multicategorical_delimiter='|',
            ),
        ],
        pkey='customer_id',
    )
    source_metadata = MagicMock()
    source_metadata.all_cols.return_value = [
        SimpleNamespace(name='customer_id'),
        SimpleNamespace(name='tags'),
    ]
    connector = MagicMock()

    with patch.object(
            table_logic_module,
            '_get_source_table_metadata',
            new=SimpleNamespace(coroutine=AsyncMock(
                return_value=source_metadata)),
    ), patch.object(
            table_logic_module.Table,
            '_infer_metadata_blocking',
            new=SimpleNamespace(coroutine=AsyncMock()),
    ), patch.object(
            table_logic_module.Table,
            '_infer_dtypes_from_stypes_blocking',
            new=SimpleNamespace(coroutine=AsyncMock()),
    ):
        response, _, _ = await _resolve_table_metadata_with_connector(
            request, connector)

    tags = next(col for col in response.cols if col.name == 'tags')
    assert tags.stype == Stype.multicategorical
    assert tags.dtype == Dtype.string
    assert tags.multicategorical_delimiter == '|'
    connector.read_table_sample.assert_not_called()


@pytest.mark.asyncio
async def test_infer_metadata_does_not_resample_unresolved_multicategorical():
    request = _create_table_metadata_request('customers', ['customer_name'],
                                             pkey=None)
    request.cols[0] = dataclasses.replace(
        request.cols[0],
        stype=Stype.multicategorical,
        dtype=Dtype.string,
    )
    connector = MagicMock()
    source_metadata = MagicMock()
    source_metadata.all_cols.return_value = [
        SimpleNamespace(name='customer_name'),
    ]

    with patch.object(
            table_logic_module,
            '_get_source_table_metadata',
            new=SimpleNamespace(coroutine=AsyncMock(
                return_value=source_metadata)),
    ), patch.object(
            table_logic_module.Table,
            '_infer_metadata_blocking',
            new=SimpleNamespace(coroutine=AsyncMock()),
    ), patch.object(
            table_logic_module.Table,
            '_infer_dtypes_from_stypes_blocking',
            new=SimpleNamespace(coroutine=AsyncMock()),
    ):
        with pytest.raises(ValidationException,
                           match='existing approximate-stats sample'):
            await _resolve_table_metadata_with_connector(request, connector)

    connector.read_table_sample.assert_not_called()


@pytest.mark.asyncio
async def test_validate_table_definition_activity_closes_after_failure():
    connector = MagicMock()
    definition = RestTableDefinition(**dataclasses.asdict(
        _create_table_definition(
            'customers',
            [
                table_api.Column(
                    name='customer_id',
                    stype=Stype.ID,
                    dtype=Dtype.int64,
                )
            ],
            pkey='customer_id',
        )))
    failure = RuntimeError('metadata failure')

    with patch.object(
            table_logic_module,
            'get_connector_from_spec_async',
            new=AsyncMock(return_value=connector),
    ), patch.object(
            table_logic_module,
            '_get_source_table_metadata',
            new=SimpleNamespace(coroutine=AsyncMock(side_effect=failure)),
    ):
        with pytest.raises(RuntimeError) as caught:
            await validate_table_definition_activity(
                MagicMock(),
                definition,
                {},
            )

    assert caught.value is failure
    connector.open.assert_called_once_with()
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_validate_table_definition_activity_runs_checks_off_event_loop():
    connector = MagicMock()

    def sync_lifecycle_only():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return connector
        raise AssertionError('connector lifecycle ran on the event loop')

    connector.open.side_effect = sync_lifecycle_only
    connector.close.side_effect = sync_lifecycle_only
    definition = RestTableDefinition(**dataclasses.asdict(
        _create_table_definition(
            'customers',
            [
                table_api.Column(
                    name='customer_id',
                    stype=Stype.ID,
                    dtype=Dtype.int64,
                )
            ],
            pkey='customer_id',
        )))
    source_metadata = MagicMock()
    metadata_check = AsyncMock()
    size_check = AsyncMock()
    permissions_check = AsyncMock()

    with patch.object(
            table_logic_module,
            'get_connector_from_spec_async',
            new=AsyncMock(return_value=connector),
    ), patch.object(
            table_logic_module,
            '_get_source_table_metadata',
            new=SimpleNamespace(coroutine=AsyncMock(
                return_value=source_metadata)),
    ), patch.object(
            table_logic_module,
            '_validate_table_metadata_v2',
            new=SimpleNamespace(coroutine=metadata_check),
    ), patch.object(
            table_logic_module,
            '_validate_table_size_v2',
            new=SimpleNamespace(coroutine=size_check),
    ), patch.object(
            table_logic_module,
            '_validate_table_permissions_v2',
            new=SimpleNamespace(coroutine=permissions_check),
    ):
        await validate_table_definition_activity(
            MagicMock(),
            definition,
            {'default': 10},
        )

    connector.open.assert_called_once_with()
    connector.close.assert_called_once_with()
    metadata_check.assert_awaited_once_with(definition, connector,
                                            source_metadata)
    size_check.assert_awaited_once_with(definition, connector, {'default': 10})
    permissions_check.assert_awaited_once_with(definition, connector)


@pytest.mark.asyncio
async def test_validate_table_retry_uses_fresh_connector_after_transient_failure(
):
    first_connector = MagicMock()
    retry_connector = MagicMock()
    connector_factory = AsyncMock(
        side_effect=[first_connector, retry_connector])
    definition = RestTableDefinition(**dataclasses.asdict(
        _create_table_definition(
            'customers',
            [
                table_api.Column(
                    name='customer_id',
                    stype=Stype.ID,
                    dtype=Dtype.int64,
                )
            ],
            pkey='customer_id',
        )))
    transient_error = RuntimeError('transient warehouse outage')
    source_metadata = MagicMock()
    metadata_lookup = AsyncMock(side_effect=[transient_error, source_metadata])
    metadata_check = AsyncMock()
    size_check = AsyncMock()
    permissions_check = AsyncMock()

    with patch.object(
            table_logic_module,
            'get_connector_from_spec_async',
            new=connector_factory,
    ), patch.object(
            table_logic_module,
            '_get_source_table_metadata',
            new=SimpleNamespace(coroutine=metadata_lookup),
    ), patch.object(
            table_logic_module,
            '_validate_table_metadata_v2',
            new=SimpleNamespace(coroutine=metadata_check),
    ), patch.object(
            table_logic_module,
            '_validate_table_size_v2',
            new=SimpleNamespace(coroutine=size_check),
    ), patch.object(
            table_logic_module,
            '_validate_table_permissions_v2',
            new=SimpleNamespace(coroutine=permissions_check),
    ):
        with pytest.raises(RuntimeError) as caught:
            await validate_table_definition_activity(MagicMock(), definition,
                                                     {})
        await validate_table_definition_activity(MagicMock(), definition, {})

    assert caught.value is transient_error
    assert connector_factory.await_count == 2
    first_connector.open.assert_called_once_with()
    first_connector.close.assert_called_once_with()
    retry_connector.open.assert_called_once_with()
    retry_connector.close.assert_called_once_with()
    metadata_check.assert_awaited_once_with(definition, retry_connector,
                                            source_metadata)
    size_check.assert_awaited_once_with(definition, retry_connector, {})
    permissions_check.assert_awaited_once_with(definition, retry_connector)


@pytest.mark.asyncio
async def test_get_sample_rows_closes_connector():
    connector = MagicMock()
    definition = RestTableDefinition(**dataclasses.asdict(
        _create_table_definition(
            'customers',
            [
                table_api.Column(
                    name='customer_id',
                    stype=Stype.ID,
                    dtype=Dtype.int64,
                )
            ],
            pkey='customer_id',
        )))
    sample = pd.DataFrame({'customer_id': [1, 2]})

    with patch.object(
            table_logic_module,
            'get_connector_from_spec_async',
            new=AsyncMock(return_value=connector),
    ), patch.object(
            table_logic_module,
            'get_sample_rows_raw',
            new=SimpleNamespace(coroutine=AsyncMock(return_value=sample)),
    ):
        result = await _get_sample_rows(definition, MagicMock())

    assert result == {'customer_id': [1, 2]}
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'metadata_error',
    [
        ValueError('source missing'),
        PublicException(message='source missing'),
    ],
)
async def test_custom_table_validation_activity_closes_on_early_return(
        metadata_error):
    connector = MagicMock()
    request = _create_table_metadata_request('customers', ['customer_id'],
                                             pkey='customer_id')

    with patch(
            'rest.backends.data_backend.activities.custom_table_validation.'
            'get_connector_from_spec_async',
            new=AsyncMock(return_value=connector),
    ), patch(
            'rest.backends.data_backend.activities.custom_table_validation.'
            '_get_source_table_metadata',
            new=SimpleNamespace(coroutine=AsyncMock(
                side_effect=metadata_error)),
    ):
        response = await custom_table_validation.validate_custom_train_table_activity(
            MagicMock(),
            request.source_table,
            'weight',
            'train-table',
            False,
        )

    assert response.errors
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_custom_table_validation_uses_physical_train_table_path():
    connector = MagicMock()
    request = _create_table_metadata_request('customers', ['customer_id'],
                                             pkey='customer_id')
    with patch(
            'rest.backends.data_backend.activities.custom_table_validation.'
            'get_connector_from_spec_async',
            new=AsyncMock(return_value=connector),
    ), patch(
            'rest.backends.data_backend.activities.custom_table_validation.'
            '_get_source_table_metadata',
            new=SimpleNamespace(coroutine=AsyncMock(
                side_effect=ValueError('stop after resolution'))),
    ):
        response = await custom_table_validation.validate_custom_train_table_activity(
            MagicMock(),
            request.source_table,
            'weight',
            '/immutable/bundle/tt.parquet',
            False,
        )

    assert response.errors
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_custom_sql_validation_returns_actionable_query_error():
    connector = MagicMock()
    connector._get_connector_type.return_value = 'databricks'
    connector.get_query_metadata.side_effect = ConnectorError(
        'PARSE_SYNTAX_ERROR near FROM')
    source = ApiCustomSQLSourceTable(
        connector_id='connector_' + 'a' * 32,
        sql='SELECT * FROM missing_table',
        table='custom_sql_invalid',
        connector_type=ApiSourceConnectorType.SNOWFLAKE,
    )

    with patch(
            'rest.backends.data_backend.activities.custom_table_validation.'
            'get_connector_from_spec_async',
            new=AsyncMock(return_value=connector),
    ), patch('kumo.data.custom_sql.advisory_read_only_check',
             ) as advisory, patch(
                 'kumo.data.custom_sql.validate_connector_sql_scope', ), patch(
                     'kumo.data.custom_sql.subquery_safe_sql',
                     side_effect=lambda sql, _connector_type: sql,
                 ):
        response = await custom_table_validation.validate_custom_train_table_activity(
            MagicMock(),
            source,
            'WEIGHT',
            'train-table',
            False,
        )

    assert response.errors[0].title == 'Warehouse query failed'
    assert response.errors[0].message == (
        'The warehouse could not run this query. '
        'PARSE_SYNTAX_ERROR near FROM')
    assert advisory.call_args.args[1].value == 'databricks'
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_custom_table_validation_reports_missing_required_column():
    connector = MagicMock()
    request = _create_table_metadata_request('customers', ['ENTITY1', 'WEIGHT'],
                                             pkey='ENTITY1')
    from kumo_core.training import custom_table

    metadata = SimpleNamespace(cols=[
        SimpleNamespace(name='WEIGHT', dtype=InternalDtype.float32),
    ])

    with patch(
            'rest.backends.data_backend.activities.custom_table_validation.'
            'get_connector_from_spec_async',
            new=AsyncMock(return_value=connector),
    ), patch(
            'rest.backends.data_backend.activities.custom_table_validation.'
            '_get_source_table_metadata',
            new=SimpleNamespace(coroutine=AsyncMock(return_value=metadata)),
    ), patch(
            'rest.backends.data_backend.activities.custom_train_table.'
            'missing_source_column_errors',
            side_effect=lambda names, path, weight, split: (
                custom_table.missing_column_errors(
                    names, required=['ENTITY1'], weight_col=weight,
                    split_col=split)),
    ):
        response = await custom_table_validation.validate_custom_train_table_activity(
            MagicMock(),
            request.source_table,
            'WEIGHT',
            'train-table',
            False,
        )

    assert response.errors[0].title == 'Required column not found'
    assert 'ENTITY1' in response.errors[0].message
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_custom_table_validation_reports_missing_split_column():
    """The name check the activity runs against the warehouse's own column
    metadata has to see ``split_col`` the same way it sees ``weight_col``,
    or a misnamed split column would only surface after the export."""
    connector = MagicMock()
    request = _create_table_metadata_request('customers', ['ENTITY1', 'W'],
                                             pkey='ENTITY1')
    metadata = SimpleNamespace(cols=[
        SimpleNamespace(name='ENTITY1', dtype=InternalDtype.int),
        SimpleNamespace(name='W', dtype=InternalDtype.float32),
    ])

    with patch(
            'rest.backends.data_backend.activities.custom_table_validation.'
            'get_connector_from_spec_async',
            new=AsyncMock(return_value=connector),
    ), patch(
            'rest.backends.data_backend.activities.custom_table_validation.'
            '_get_source_table_metadata',
            new=SimpleNamespace(coroutine=AsyncMock(return_value=metadata)),
    ):
        response = await custom_table_validation.validate_custom_train_table_activity(
            MagicMock(),
            request.source_table,
            'W',
            'train-table',
            False,
            'SPLIT',
        )

    assert response.errors[0].title == 'Split column not found'
    assert 'SPLIT' in response.errors[0].message
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_infer_table_metadata_with_db_enabled_existing_connector(
    temp_csv_dir: str, ):
    connector_config = FileConnectorResourceConfig(
        name=str(_TEST_CONNECTOR_ID),
        root_dir=temp_csv_dir,
    )
    mock_connectors_dao = AsyncMock()

    table_driver = _create_table_driver(mock_connectors_dao)
    connector_service = _ConnectorService()
    connector_service.entries[ConnectorID(
        connector_config.name)] = (_connector_entry(
            connector_config,
            "Test CSV connector for metadata inference",
        ))
    table_driver._configure_state_services(
        SimpleNamespace(connectors=connector_service))

    request = _create_table_metadata_request(
        source_table_name="customers",
        column_names=[
            "customer_id",
            "customer_name",
            "age",
            "signup_date",
            "account_balance",
            "is_active",
        ],
        pkey="customer_id",
        time_col="signup_date",
        connector_id=_TEST_CONNECTOR_ID,
    )

    response = await table_driver.infer_table_metadata(request,
                                                       str(_TEST_PROJECT_ID))

    assert response is not None
    assert isinstance(response, table_api.TableMetadataResponse)
    assert len(response.cols) == 6

    col_names = {col.name for col in response.cols}
    assert col_names == {
        "customer_id",
        "customer_name",
        "age",
        "signup_date",
        "account_balance",
        "is_active",
    }

    for col in response.cols:
        assert col.stype is not None, f"stype not inferred for {col.name}"
        assert col.dtype is not None, f"dtype not inferred for {col.name}"

    col_dict = {col.name: col for col in response.cols}
    assert col_dict["customer_id"].stype in [Stype.categorical, Stype.ID]
    assert col_dict["customer_name"].stype == Stype.categorical
    assert col_dict["age"].stype == Stype.numerical
    assert col_dict["account_balance"].stype == Stype.numerical
    assert col_dict["is_active"].stype == Stype.categorical

    assert response.pkey == "customer_id"
    assert response.time_col == "signup_date"
    assert isinstance(response.source_table, S3SourceTable)
    assert isinstance(response.source_table.source, S3RelativeSource)
    assert response.source_table.source.path == "customers"
    assert not hasattr(response.source_table, "s3_path")
    assert connector_service.execution_ref_calls == [
        (None, _TEST_PROJECT_ID, _TEST_CONNECTOR_ID,
         CONNECTOR_TABLE_SOURCE_ACTIONS)
    ]


@pytest.mark.asyncio
async def test_list_project_table_names_stamps_custom_sql_kind():
    # The Tables-list route returns ``ListResponse`` rows; a Custom SQL table
    # must be stamped ``SQL`` here (not only on the ``TableResourceResponse``
    # detail/connector-list builders) or the list "Kind" filter cannot see it.
    valid_project_id = "project_" + "a" * 32
    table_driver = _create_table_driver(AsyncMock())

    table_definition = RestTableDefinition(
        source_table=RestCustomSQLSourceTable(
            connector_id=_TEST_CONNECTOR_ID,
            connector_type="snowflake",
            sql="SELECT customer_id FROM customers",
            table="custom_sql_abc",
            referenced_tables=["customers"],
        ),
        cols=[
            RestColumn(name="customer_id", stype=Stype.ID, dtype=Dtype.int64)
        ],
        pkey="customer_id",
    )
    table_entry = MagicMock(
        alias="orders_sql",
        id=table_definition.fpid,
        config=table_definition,
        timestamps=ResourceTimestamps(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )
    table_service = SimpleNamespace(list_named_tables=AsyncMock(
        return_value=[table_entry]))
    table_driver._configure_state_services(
        SimpleNamespace(tables=table_service, projects=SimpleNamespace()))

    result = await table_driver._list_project_table_names(valid_project_id)

    assert len(result) == 1
    assert result[0].table_kind == TableKind.SQL


@pytest.mark.asyncio
async def test_infer_table_metadata_invalid_column_name(temp_csv_dir: str, ):
    # Setup connector
    connector_config = FileConnectorResourceConfig(
        name=str(_TEST_CONNECTOR_ID),
        root_dir=temp_csv_dir,
    )

    mock_connectors_dao = AsyncMock()

    # Create TableDriver
    table_driver = _create_table_driver(mock_connectors_dao)
    connector_service = _ConnectorService()
    connector_service.entries[ConnectorID(
        connector_config.name)] = (_connector_entry(connector_config,
                                                    "Test CSV connector"))

    table_driver._configure_state_services(
        SimpleNamespace(connectors=connector_service))

    # Create a request with a non-existent column
    request = _create_table_metadata_request(
        source_table_name="customers",
        column_names=["customer_id", "non_existent_column"],
        pkey="customer_id",
        connector_id=_TEST_CONNECTOR_ID,
    )

    # Execute the method under test and expect an error. The real infer
    # activity raises the allowlisted ``ValidationException`` for this case so
    # the actionable message survives the data-backend single-activity wrapper
    # (kumo-ai/kumo#29794); mirror that type here instead of a bare
    # ``ValueError``.
    expected_error = ValidationException(
        title=TABLE_COLUMN_NOT_FOUND_TITLE,
        message="Column non_existent_column does not exist in table customers")
    with patch(
            "rest.driver.table_driver.DataBackend._execute_single_activity",
            new=AsyncMock(side_effect=expected_error),
    ), pytest.raises(ValidationException) as exc_info:
        await table_driver.infer_table_metadata(request, str(_TEST_PROJECT_ID))

    # Verify the error message is descriptive
    error_msg = str(exc_info.value)
    assert "non_existent_column" in error_msg
    assert "does not exist" in error_msg
    assert "customers" in error_msg


@pytest.mark.asyncio
async def test_infer_table_metadata_with_timestamp_format(temp_csv_dir: str, ):
    """
    Test infer_table_metadata correctly handles timestamp format specifications.

    This tests that:
    1. Timestamp formats are properly propagated through the inference
    2. The returned columns include the timestamp format information
    """
    # Create a CSV with timestamp data
    df = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105],
        "customer_id": [1, 2, 1, 3, 2],
        "order_timestamp": [
            1672531200000,  # Unix timestamp in milliseconds
            1672617600000,
            1672704000000,
            1672790400000,
            1672876800000,
        ],
        "amount": [99.99, 149.50, 75.25, 200.00, 125.75],
    })
    csv_path = os.path.join(temp_csv_dir, "orders.csv")
    df.to_csv(csv_path, index=False)
    connector_config = FileConnectorResourceConfig(
        name=str(_SECOND_CONNECTOR_ID),
        root_dir=temp_csv_dir,
    )

    mock_connectors_dao = AsyncMock()

    table_driver = _create_table_driver(mock_connectors_dao)
    connector_service = _ConnectorService()
    connector_service.entries[ConnectorID(
        connector_config.name)] = (_connector_entry(
            connector_config,
            "Test orders connector",
            connector_id=_SECOND_CONNECTOR_ID,
        ))
    table_driver._configure_state_services(
        SimpleNamespace(connectors=connector_service))

    request = _create_table_metadata_request(
        source_table_name="orders",
        column_names=["order_id", "customer_id", "order_timestamp", "amount"],
        pkey="order_id",
        time_col="order_timestamp",
        timestamp_format=table_api.TimestampUnit.SECOND,
        connector_id=_SECOND_CONNECTOR_ID,
    )

    response = await table_driver.infer_table_metadata(request,
                                                       str(_TEST_PROJECT_ID))

    assert response is not None
    assert len(response.cols) == 4

    col_dict = {col.name: col for col in response.cols}
    timestamp_col = col_dict["order_timestamp"]
    assert timestamp_col.stype == Stype.timestamp
    assert timestamp_col.timestamp_format is not None
    assert response.time_col == "order_timestamp"


def _create_table_definition(
    source_table_name: str,
    cols: List[table_api.Column],
    connector_id: str | ConnectorID = _TEST_CONNECTOR_ID,
    pkey: str | None = None,
    time_col: str | None = None,
    end_time_col: str | None = None,
) -> table_api.TableDefinition:
    """Helper function to create a TableDefinition for validation testing.

    Args:
        source_table_name: Name of the source table file
        cols: List of columns with name, stype, dtype
        connector_id: Optional connector ID for the table
        pkey: Optional primary key column name
        time_col: Optional time column name
        end_time_col: Optional end time column name

    Returns:
        TableDefinition object for validation testing
    """
    return table_api.TableDefinition(
        source_table=S3SourceTable(
            source={
                'mode': 'relative',
                'path': source_table_name
            },
            connector_id=connector_id,
            file_type=FileType.CSV,  # type: ignore
            data_source_type=ApiDataSourceType.S3,
        ),
        cols=cols,
        pkey=pkey,
        time_col=time_col,
        end_time_col=end_time_col,
    )


def _create_rest_table_definition(temp_csv_dir: str) -> RestTableDefinition:
    table_definition = _create_table_definition(
        source_table_name="customers",
        cols=[
            table_api.Column(
                name="customer_id",
                stype=Stype.ID,
                dtype=Dtype.int64,
            )
        ],
        connector_id=_TEST_CONNECTOR_ID,
        pkey="customer_id",
    )
    return RestTableDefinition(**dataclasses.asdict(table_definition))


def _create_table_db_entry(
    table_definition: RestTableDefinition,
    alias: str | None = "customers",
) -> TableDBEntryV2:
    return TableDBEntryV2(
        id=table_definition.fpid,
        project_id=_TEST_PROJECT_ID,
        alias=alias,
        config=table_definition,
        col_dtypes={
            col.name: col.dtype
            for col in table_definition.cols if col.dtype
        },
        timestamps=ResourceTimestamps(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )


def _patch_empty_table_stats():
    cache = SimpleNamespace(get_many_min_latest_exact_stats=SimpleNamespace(
        coroutine=AsyncMock(return_value=[None])))
    return patch(
        "rest.driver.table_driver.get_active_cache",
        return_value=cache,
    )


@pytest.mark.asyncio
async def test_get_table_by_name_uses_state_service(temp_csv_dir: str):
    table_driver = _create_table_driver(AsyncMock())
    table_definition = _create_rest_table_definition(temp_csv_dir)
    table_entry = _create_table_db_entry(table_definition, alias="customers")
    table_service = SimpleNamespace(get_table_entry_by_alias=AsyncMock(
        return_value=table_entry))
    table_driver._configure_state_services(
        SimpleNamespace(tables=table_service))

    with _patch_empty_table_stats():
        result = await table_driver.get_table_by_name("customers",
                                                      str(_TEST_PROJECT_ID))

    assert result.id == table_definition.fpid
    assert result.name_alias == "customers"
    assert result.table == table_definition
    table_service.get_table_entry_by_alias.assert_awaited_once_with(
        _TEST_PROJECT_ID, "customers")


@pytest.mark.asyncio
async def test_list_tables_uses_state_service(temp_csv_dir: str):
    table_driver = _create_table_driver(AsyncMock())
    table_definition = _create_rest_table_definition(temp_csv_dir)
    table_entry = _create_table_db_entry(table_definition, alias="customers")
    table_service = SimpleNamespace(list_tables=AsyncMock(
        return_value=[table_entry]))
    table_driver._configure_state_services(
        SimpleNamespace(tables=table_service))

    with _patch_empty_table_stats():
        result = await table_driver.list_tables(
            project_id=str(_TEST_PROJECT_ID),
            connector_id=_TEST_CONNECTOR_ID,
            fetch_column_stats=False,
        )

    assert len(result) == 1
    assert result[0].id == table_definition.fpid
    assert result[0].name_alias == "customers"
    table_service.list_tables.assert_awaited_once_with(
        _TEST_PROJECT_ID, connector_id=_TEST_CONNECTOR_ID)


@pytest.mark.asyncio
async def test_validate_table_success(temp_csv_dir: str):
    mock_connectors_dao = AsyncMock()

    table_driver = _create_table_driver(mock_connectors_dao)
    _configure_local_connector(table_driver, temp_csv_dir)

    # Create a valid table definition
    cols = [
        table_api.Column(
            name="customer_id",
            stype=Stype.ID,
            dtype=Dtype.int64,
        ),
        table_api.Column(
            name="customer_name",
            stype=Stype.categorical,
            dtype=Dtype.string,
        ),
        table_api.Column(
            name="age",
            stype=Stype.numerical,
            dtype=Dtype.int64,
        ),
        table_api.Column(
            name="signup_date",
            stype=Stype.timestamp,
            dtype=Dtype.time,
        ),
    ]

    table_definition = _create_table_definition(
        source_table_name="customers",
        cols=cols,
        pkey="customer_id",
        time_col="signup_date",
    )

    response = await table_driver.validate_table(
        table_definition, project_id=str(_TEST_PROJECT_ID))

    assert response is not None
    assert len(response.errors) == 0


@pytest.mark.asyncio
async def test_validate_table_duplicate_column_names(temp_csv_dir: str):
    mock_connectors_dao = AsyncMock()

    table_driver = _create_table_driver(mock_connectors_dao)
    _configure_local_connector(table_driver, temp_csv_dir)

    # Create table definition with duplicate column names
    cols = [
        table_api.Column(
            name="customer_id",
            stype=Stype.ID,
            dtype=Dtype.int64,
        ),
        table_api.Column(
            name="customer_id",  # Duplicate name
            stype=Stype.categorical,
            dtype=Dtype.string,
        ),
    ]

    table_definition = _create_table_definition(
        source_table_name="customers",
        cols=cols,
        pkey="customer_id",
    )

    result = await table_driver.validate_table(table_definition,
                                               project_id=str(_TEST_PROJECT_ID))

    assert len(result.errors) > 0
    assert "duplicate" in result.errors[0].message.lower()


@pytest.mark.asyncio
async def test_validate_table_pkey_not_in_columns(temp_csv_dir: str):
    mock_connectors_dao = AsyncMock()

    table_driver = _create_table_driver(mock_connectors_dao)
    _configure_local_connector(table_driver, temp_csv_dir)

    cols = [
        table_api.Column(
            name="customer_name",
            stype=Stype.categorical,
            dtype=Dtype.string,
        ),
        table_api.Column(
            name="age",
            stype=Stype.numerical,
            dtype=Dtype.int64,
        ),
    ]

    table_definition = _create_table_definition(
        source_table_name="customers",
        cols=cols,
        pkey="customer_id",  # Not in cols
    )

    result = await table_driver.validate_table(table_definition,
                                               project_id=str(_TEST_PROJECT_ID))

    assert len(result.errors) > 0
    assert "primary key" in result.errors[0].message.lower()
    assert "must be present" in result.errors[0].message.lower()


@pytest.mark.asyncio
async def test_validate_table_pkey_not_id_type(temp_csv_dir: str):
    mock_connectors_dao = AsyncMock()

    table_driver = _create_table_driver(mock_connectors_dao)
    _configure_local_connector(table_driver, temp_csv_dir)

    cols = [
        table_api.Column(
            name="customer_id",
            stype=Stype.categorical,  # Should be ID
            dtype=Dtype.int64,
        ),
        table_api.Column(
            name="customer_name",
            stype=Stype.categorical,
            dtype=Dtype.string,
        ),
    ]

    table_definition = _create_table_definition(
        source_table_name="customers",
        cols=cols,
        pkey="customer_id",
    )

    result = await table_driver.validate_table(table_definition,
                                               project_id=str(_TEST_PROJECT_ID))

    assert len(result.errors) > 0
    assert "primary key" in result.errors[0].message.lower()
    assert "id" in result.errors[0].message.lower()


@pytest.mark.asyncio
async def test_validate_table_time_col_not_in_columns(temp_csv_dir: str):
    mock_connectors_dao = AsyncMock()

    table_driver = _create_table_driver(mock_connectors_dao)
    _configure_local_connector(table_driver, temp_csv_dir)

    cols = [
        table_api.Column(
            name="customer_id",
            stype=Stype.ID,
            dtype=Dtype.int64,
        ),
        table_api.Column(
            name="customer_name",
            stype=Stype.categorical,
            dtype=Dtype.string,
        ),
    ]

    table_definition = _create_table_definition(
        source_table_name="customers",
        cols=cols,
        pkey="customer_id",
        time_col="signup_date",  # Not in cols
    )

    result = await table_driver.validate_table(table_definition,
                                               project_id=str(_TEST_PROJECT_ID))

    assert len(result.errors) > 0
    assert "time" in result.errors[0].message.lower()
    assert "must be present" in result.errors[0].message.lower()


@pytest.mark.asyncio
async def test_validate_table_time_col_same_as_pkey(temp_csv_dir: str):
    mock_connectors_dao = AsyncMock()

    table_driver = _create_table_driver(mock_connectors_dao)
    _configure_local_connector(table_driver, temp_csv_dir)

    cols = [
        table_api.Column(
            name="customer_id",
            stype=Stype.ID,
            dtype=Dtype.int64,
        ),
        table_api.Column(
            name="customer_name",
            stype=Stype.categorical,
            dtype=Dtype.string,
        ),
    ]

    table_definition = _create_table_definition(
        source_table_name="customers",
        cols=cols,
        pkey="customer_id",
        time_col="customer_id",  # Same as pkey
    )

    result = await table_driver.validate_table(table_definition,
                                               project_id=str(_TEST_PROJECT_ID))

    assert len(result.errors) > 0
    assert "time column" in result.errors[0].message.lower()
    assert "cannot be the same" in result.errors[0].message.lower()


@pytest.mark.asyncio
async def test_validate_table_time_col_not_timestamp_type(temp_csv_dir: str):
    mock_connectors_dao = AsyncMock()

    table_driver = _create_table_driver(mock_connectors_dao)
    _configure_local_connector(table_driver, temp_csv_dir)

    cols = [
        table_api.Column(
            name="customer_id",
            stype=Stype.ID,
            dtype=Dtype.int64,
        ),
        table_api.Column(
            name="signup_date",
            stype=Stype.categorical,  # Should be timestamp
            dtype=Dtype.string,
        ),
    ]

    table_definition = _create_table_definition(
        source_table_name="customers",
        cols=cols,
        pkey="customer_id",
        time_col="signup_date",
    )

    result = await table_driver.validate_table(table_definition,
                                               project_id=str(_TEST_PROJECT_ID))

    assert len(result.errors) > 0
    assert "time column" in result.errors[0].message.lower()
    assert "timestamp" in result.errors[0].message.lower()


@pytest.mark.asyncio
async def test_validate_table_column_not_in_source(temp_csv_dir: str):
    mock_connectors_dao = AsyncMock()

    table_driver = _create_table_driver(mock_connectors_dao)
    _configure_local_connector(table_driver, temp_csv_dir)

    cols = [
        table_api.Column(
            name="customer_id",
            stype=Stype.ID,
            dtype=Dtype.int64,
        ),
        table_api.Column(
            name="nonexistent_column",  # Doesn't exist in source
            stype=Stype.categorical,
            dtype=Dtype.string,
        ),
    ]

    table_definition = _create_table_definition(
        source_table_name="customers",
        cols=cols,
        pkey="customer_id",
    )

    with patch(
            "rest.driver.table_driver.DataBackend._execute_single_activity",
            new=AsyncMock(side_effect=ValueError(
                "Column nonexistent_column does not exist in source table")),
    ):
        result = await table_driver.validate_table(
            table_definition, project_id=str(_TEST_PROJECT_ID))

    assert len(result.errors) > 0
    assert "does not exist" in result.errors[0].message.lower()
