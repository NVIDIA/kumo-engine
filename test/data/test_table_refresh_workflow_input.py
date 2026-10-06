# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
from kumoapi.source_table import FileType, S3SourceTable
from temporalio.workflow import ParentClosePolicy

from rest.backends.data_backend.data_backend import DataBackend
from rest.backends.data_backend.workflows import (
    GraphRefreshWorkflow,
    TableRefreshWorkflowInput,
)
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.table_v2 import TableDefinition
from statedb.config.common import ConnectorID, ProjectID, TableID, TableName
from statedb.config.table import TableDataVersion
from temporalio_commons.dataclass_converter import KumoPayloadConverter


@pytest.mark.asyncio
async def test_start_refresh_table_emits_typed_input_without_positional_shift():
    connector_id = ConnectorID('connector_' + '1' * 32)
    connector_ref = ConnectorExecutionRef(
        connector_id=connector_id,
        project_id=ProjectID('project_' + '2' * 32),
        authorization_token='capability',
    )
    table = TableDefinition(
        cols=[],
        source_table=S3SourceTable(
            connector_id=connector_id,
            source={'mode': 'relative', 'path': 'orders'},
            file_type=FileType.PARQUET,
        ),
        pkey=None,
        time_col=None,
    )
    data_version = TableDataVersion('test-version')
    alias = TableName('orders')
    client = mock.AsyncMock()
    provider = mock.Mock()
    provider.get = mock.AsyncMock(return_value=client)
    context = mock.Mock()
    context._settings.disable_async_ingestion = False
    context.temporal_client.return_value = provider

    await DataBackend.start_refresh_table(context, connector_ref, table,
                                          data_version, alias)

    args = client.start_workflow.await_args.kwargs['args']
    assert len(args) == 2
    assert args[0] is context
    assert isinstance(args[1], TableRefreshWorkflowInput)
    assert args[1].connector_spec == connector_ref
    assert args[1].table == table
    assert args[1].data_version == data_version
    assert args[1].job_id is None
    assert args[1].base_tables_args_list is None
    assert args[1].base_tables_connector_spec is None
    assert args[1].alias == alias
    assert args[1].defer_col_stats_validation is False
    assert args[1].timestamp_units_inferred_by_parent is False

    converter = KumoPayloadConverter()
    payloads = converter.to_payloads([args[1]])
    assert converter.from_payloads(
        payloads, [TableRefreshWorkflowInput])[0] == args[1]


@pytest.mark.asyncio
async def test_graph_refresh_marks_child_timestamp_units_as_already_inferred():
    connector_id = ConnectorID('connector_' + '1' * 32)
    connector_ref = ConnectorExecutionRef(
        connector_id=connector_id,
        project_id=ProjectID('project_' + '2' * 32),
        authorization_token='capability',
    )
    table_id = cast(TableID, TableName('orders'))
    table = TableDefinition(
        cols=[],
        source_table=S3SourceTable(
            connector_id=connector_id,
            source={'mode': 'relative', 'path': 'orders'},
            file_type=FileType.PARQUET,
        ),
        pkey=None,
        time_col=None,
    )
    data_version = TableDataVersion('test-version')
    workflow = GraphRefreshWorkflow()
    workflow._ctx = mock.Mock()
    workflow._job_id = 'test-job'
    workflow._input = cast(Any, SimpleNamespace(
        table_aliases={table_id: TableName('orders')},
        connectors={connector_id: connector_ref},
        tables={table_id: mock.Mock()},
    ))
    workflow._graph_version_info = cast(Any, SimpleNamespace(
        table_data_versions={TableName('orders'): data_version}))
    start_child_refresh = mock.AsyncMock()
    with mock.patch.object(
            workflow, 'start_child_refresh', new=start_child_refresh):
        await workflow._start_refresh_table(
            'refresh-orders', table_id, table,
            ParentClosePolicy.REQUEST_CANCEL)

    child_call = start_child_refresh.await_args
    assert child_call is not None
    child_input = cast(TableRefreshWorkflowInput,
                       child_call.kwargs['args'][1])
    assert child_input.timestamp_units_inferred_by_parent is True
