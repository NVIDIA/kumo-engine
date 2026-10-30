# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from test.rest.conftest import TestClient
from test.rest.integ import fixture_connector_args
from typing import Any, Dict

import kumoapi.source_table as source_table_api
import kumoapi.table as table_api
import pytest
from kumoapi.source_table import (
    SourceTableDataResponse,
    SourceTableListResponse,
)
from kumoapi.typing import Stype

from io_utils import fsutil
from kumo.data import MaterializationStage
from kumo.data.backend_cache import TableDataTaskResult
from kumo.data.common import TableRefreshStage
from kumo.data.utils import get_materialize_path
from kumo.testing.rest import (
    fixture_admin_user_authorization_context,
    get_and_expect_success,
    post_and_expect_success,
)
from kumo.validation import ValidationResponse
from rest.backends.data_backend.workflows import TableRefreshWorkflow
from rest.config.status import EmptyResponse, IDResponse
from rest.driver.rest_api_driver import RestApiDriver
from statedb.config.common import TableIDv2
from statedb.config.table import TableDataVersion
from temporalio_commons import TemporalClient

from .config_for_test import (
    ConnectorType,
    FailStage,
    parquet_args,
    parquet_time_col_args,
    unsupported_parquet_args,
)

# Pool fixtures (storage_backend, _diskgraph_deploy_config, _pool_subprocesses)
# live in test/rest/integ/conftest.py; the diskgraph per-table refresh
# (TableRefreshWorkflow -> ingest_table_data_activity_v2) needs the pool
# subprocesses to run. We pin storage_backend to s3 (mirroring the green
# sibling test/rest/integ/test_snowflake_ingestion_diskgraph_e2e.py): the
# session-scoped worker reuses a single deploy_env, and the local->s3 flip
# within one session trips a known harness deploy_env leak.
pytestmark = [
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", [
        pytest.param("s3", marks=pytest.mark.private_resource('aws')),
    ], indirect=True),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [
    parquet_args,
    unsupported_parquet_args,
    parquet_time_col_args,
])
async def test_table_lifecycle(client: TestClient,
                               temporal_client: TemporalClient,
                               rest_api_driver: RestApiDriver, storage_backend,
                               args: Dict[str, Any]):
    """
    Test the table lifecycle from discovery through snapshot across supported
    parquet data types.
    """
    assert args["connector_type"] == ConnectorType.file
    root_dir = ("s3://kumocloud-test-bucket/ux_e2e_stability_tests/"
                "generated_data/data_variety_test/")
    project_id, connector_entry = await fixture_connector_args.provision_project_upload_connector(
        client,
        source_root=root_dir,
        api_prefix='/v0',
    )
    # 1. Call all the APIs that are called before
    # a user clicks `Add Table` ######################
    connector_id = str(connector_entry.id)
    # list all source tables
    source_tables = get_and_expect_success(
        client,
        f'/v0/projects/{project_id}/connectors/{connector_id}/source-tables',
        {}, SourceTableListResponse)
    source_table_names = source_tables.table_names
    assert args['source_name'] in source_table_names

    # Get table samples with default type suggestion
    table_samples = get_and_expect_success(
        client, f'/v0/projects/{project_id}/connectors/{connector_id}/sample',
        {'table_name': args['source_name']}, SourceTableDataResponse)
    assert len(args['default_stypes']) == len(table_samples.cols)
    for col in table_samples.cols:
        assert col.stype == args['default_stypes'][col.name], col.name

    # Only include columns that are supported
    # when creating the table.
    columns = []
    for col in table_samples.cols:
        if col.stype != Stype.unsupported:
            columns.append(
                table_api.Column(name=col.name, stype=col.stype,
                                 dtype=col.dtype))

    table_args = table_api.TableDefinition(
        cols=columns,
        source_table=source_table_api.S3SourceTable(
            connector_id=connector_id,
            source={
                'mode': 'relative',
                'path': args['source_name']
            },
            file_type=source_table_api.FileType.PARQUET,
            data_source_type=source_table_api.DataSourceType.S3,
        ),
        pkey=args['pkey'][0] if args['pkey'] else None,
        time_col=args['time_cols'][0] if args['time_cols'] else None,
    )

    val_response = post_and_expect_success(
        client,
        "/v1/tables/validate",
        table_args,
        ValidationResponse,
        project_id=str(project_id),
    )
    if args['fail_stage'] == FailStage.VALIDATE:
        assert not val_response.ok
        assert args['fail_message'] in val_response.error_message()
        return
    assert val_response.ok

    # 2. Add the table ##############################
    table_response = post_and_expect_success(
        client,
        "/v1/tables/",
        table_args,
        IDResponse,
        name_alias=args['table_name'],
        project_id=str(project_id),
    )
    table_id = table_response.id
    assert table_id is not None

    # 3. Refresh(ingest) the table ##############################
    # The persisted upload name is a reserved metadata sentinel. Runtime
    # connectors are keyed by their immutable connector ID, matching
    # ConnectorIdentityService._compose_internal.
    connector = replace(connector_entry.config,
                        name=connector_id).as_connector()
    table_version = await connector.get_data_version.coroutine(
        args['source_name'])
    post_and_expect_success(
        client,
        f"/v1/tables/{table_id}/refresh",
        {},
        EmptyResponse,
        project_id=str(project_id),
    )

    # wait for table to be ingested
    normalized_table_id = (table_id if '.' in table_id else f'NA.{table_id}')
    table_fpid = TableIDv2.parse(TableIDv2.from_public(normalized_table_id))[2]
    assert table_fpid is not None
    with fixture_admin_user_authorization_context():
        table_resource = await rest_api_driver.table_driver.get_table(
            table_fpid, str(project_id))
    table_def = table_resource.table
    assert table_def is not None
    wf_id = TableRefreshWorkflow.compose_wf_id(table_def,
                                               TableDataVersion(table_version))
    result: TableDataTaskResult = await temporal_client.result(wf_id)
    if args['fail_stage'] == FailStage.REFRESH:
        assert not result.validation.ok
        assert args['fail_message'] in result.validation.error_message()
        return
    assert result.validation.ok
    assert result.stage == TableRefreshStage.COMPLETED

    # 4. Assert the expected source handoff.
    #
    # Eligible ambient object-store Parquet is handed to DiskGraph directly,
    # so table refresh must not create a raw ingest materialization. Other
    # sources retain the staged path and its non-empty Parquet contract.
    ingest_path = get_materialize_path(table_def.fingerprint(), table_version,
                                       MaterializationStage.INGEST)
    if connector.uses_direct_diskgraph_source_for(args['source_name']):
        assert not fsutil.exists(ingest_path), (
            f'direct source unexpectedly staged raw parquet at {ingest_path}')
        return

    assert fsutil.exists(ingest_path), (
        f"expected staged ingest output at {ingest_path}, but the path "
        f"does not exist after a COMPLETED refresh")
    staged_parquet = [
        entry for entry in fsutil.listdir(ingest_path)
        if entry.endswith('.parquet')
    ]
    assert staged_parquet, (
        f"expected staged parquet under {ingest_path}, but found none. "
        f"Listing: {fsutil.listdir(ingest_path)}")
    for entry in staged_parquet:
        size = fsutil.filesize(entry)
        assert size and size > 0, f"empty staged parquet at {entry}"
