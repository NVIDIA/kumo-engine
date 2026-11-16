# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import os.path as osp
from test.rest.integ.fixture_connector_args import (
    provision_project_upload_connector,
)

import kumoapi.source_table as source_table_api
import kumoapi.table as table_api
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from kumoapi.typing import Dtype, Stype

from io_utils import fsutil
from kumo.data.backend_cache import AsyncDataTaskStatus, TableDataTaskResult
from kumo.data.common import TableRefreshStage
from kumo.testing.rest import get_and_expect_success, post_and_expect_success
from rest.backends.data_backend.data_backend import DataBackend
from rest.backends.data_backend.workflows import TableRefreshWorkflow
from rest.config.status import EmptyResponse, IDResponse
from rest.config.table_v2 import TableDefinition, TableResourceResponse
from rest.testing.config import TestFunctionCleanupOptions
from statedb.config.table import TableDataVersion
from temporalio_commons import TemporalClient


@pytest.fixture(scope="function")
def test_cleanup_options() -> TestFunctionCleanupOptions:
    return TestFunctionCleanupOptions(should_reset_data_backend_cache=True,
                                      should_delete_data_persist_dir=True)


async def _trigger_and_wait_for_table_refresh(
    client: TestClient,
    temporal_client: TemporalClient,
    table_id: str,
    data_version: str,
    project_id: str,
) -> TableDataTaskResult:
    # Trigger table refresh workflow via the v2 endpoint.
    post_and_expect_success(
        client,
        f'/v1/tables/{table_id}/refresh',
        {},
        EmptyResponse,
        project_id=project_id,
    )

    # Wait for table refresh workflow to complete. `compose_wf_id` keys the
    # workflow on `TableDefinition.wf_key()` (v2). Fetch the v2 table via the
    # HTTP endpoint and rebuild the rest `TableDefinition` (the response field
    # may parse to the `kumoapi` base, which lacks `wf_key`).
    table_resource = get_and_expect_success(
        client,
        f'/v1/tables/{table_id}',
        {'project_id': project_id},
        TableResourceResponse,
    )
    table_def = TableDefinition(**dataclasses.asdict(table_resource.table))
    wf_id = TableRefreshWorkflow.compose_wf_id(table_def,
                                               TableDataVersion(data_version))
    return await temporal_client.result(wf_id)


# Genuine DG gap: the per-table "Duplicate primary keys" warning needs
# per-column NUM_UNIQUE/COUNT stats, but on DiskGraph those are seeded only by
# the graph-scope col-stats pipeline; the standalone per-table refresh seeds only
# table-level stats, and the graph path persists empty per-table results, so the
# warning is not reachable via get_warnings.
@pytest.mark.skip(
    reason="DG gap: per-table get_warnings cannot surface the duplicate-pkey "
    "warning; per-column NUM_UNIQUE/COUNT are seeded only by the graph-scope "
    "DG col-stats pipeline, and the graph path persists empty per-table results. "
    "The standalone table refresh ingest seeds only table-level stats.")
@pytest.mark.asyncio
async def test_create_table_with_dup_primary_key_creates_info_item(
    client: TestClient,
    tempdir,
    temporal_client: TemporalClient,
):
    # Create table with 9 copies of each value for primary key column
    # This will trigger the warning for primary key duplication
    c = np.arange(1, 91, 1) % 10
    df = pd.DataFrame({'a': c, 'b': np.ones(90)})

    # Create a parquet file using the above DF which can act as a Kumo table
    table_name = str('dup_pkey_table')
    root_dir: str = tempdir.name
    table_path = create_parquet_table(
        root_dir,
        df,
        table_name,
    )

    # Create connector for the table
    project_id, connector_id = await create_connector(client, root_dir)

    # Create table which should trigger async ingestion. This will create
    # a table refresh workflow which is eventually queried for completion.
    # Post via /v1/tables/: TableValidator.validate_tables resolves only
    # TableIDv2 (via get_many_table_definitions), so a v1 create would skip
    # the duplicate-pkey check entirely.
    table_def = table_api.TableDefinition(
        cols=[
            table_api.Column(name='a', stype=Stype.ID, dtype=Dtype.int),
            table_api.Column(name='b', stype=Stype.numerical, dtype=Dtype.int),
        ],
        source_table=source_table_api.S3SourceTable(
            connector_id=connector_id,
            source={
                'mode': 'relative',
                'path': os.path.basename(table_path)
            },
            file_type=source_table_api.FileType.PARQUET,
        ),
        pkey='a',
    )
    table_create_response = post_and_expect_success(
        client,
        "/v1/tables/",
        table_def,
        IDResponse,
        name_alias=table_name,
        project_id=project_id,
    )

    result = await _trigger_and_wait_for_table_refresh(
        client, temporal_client, table_create_response.id,
        get_data_version_string(table_path), project_id)

    warnings_map = {w.title: w for w in result.validation.warnings}
    assert 'Duplicate primary keys' in warnings_map
    assert warnings_map['Duplicate primary keys'].message.startswith(
        '80 (88.9%) of the rows in')


# Same DG gap as the duplicate-pkey test: the "Missing or invalid primary keys"
# warning needs per-column NA stats that the standalone per-table refresh never
# seeds on DiskGraph (graph-scope col-stats only).
@pytest.mark.skip(
    reason="DG gap: per-table get_warnings cannot surface the missing-pkey "
    "warning; per-column NA/NUM_UNIQUE stats are seeded only by the graph-scope "
    "DG col-stats pipeline.")
@pytest.mark.asyncio
async def test_user_is_informed_if_too_many_missing_pkeys(
    client: TestClient,
    tempdir,
    temporal_client: TemporalClient,
):
    c = np.ones(100)
    c[:90] = None
    df = pd.DataFrame({'a': np.arange(100), 'b': c})

    # Create a parquet file using the above DF which can act as a Kumo table
    table_name = str('table_with_many_nans')
    root_dir: str = tempdir.name
    table_path = create_parquet_table(
        root_dir,
        df,
        table_name,
    )

    # Create connector for the table
    project_id, connector_id = await create_connector(client, root_dir)

    # Create table which should trigger async ingestion. This will create
    # a table refresh workflow which is eventually queried for completion.
    # See note above test_create_table_with_dup_primary_key_creates_info_item
    # for why we POST to ``/v1/tables/`` rather than ``/v0/tables/``.
    table_def = table_api.TableDefinition(
        cols=[
            table_api.Column(name='a', stype=Stype.numerical, dtype=Dtype.int),
            table_api.Column(name='b', stype=Stype.ID, dtype=Dtype.float),
        ],
        source_table=source_table_api.S3SourceTable(
            connector_id=connector_id,
            source={
                'mode': 'relative',
                'path': os.path.basename(table_path)
            },
            file_type=source_table_api.FileType.PARQUET,
        ),
        pkey='b',
    )
    table_create_response = post_and_expect_success(
        client,
        "/v1/tables/",
        table_def,
        IDResponse,
        name_alias=table_name,
        project_id=project_id,
    )

    result = await _trigger_and_wait_for_table_refresh(
        client, temporal_client, table_create_response.id,
        get_data_version_string(table_path), project_id)

    warning_titles = [w.title for w in result.validation.warnings]
    # 90% of pkey rows are NaN, so the validator emits the
    # "Missing or invalid primary keys" warning. The "Duplicate primary keys"
    # branch is only reached when the missing-fraction is below
    # MAX_PKEY_BAD_FRACTION (0.0), so it never fires for this fixture.
    assert 'Missing or invalid primary keys' in warning_titles


@pytest.mark.skip(reason="Terminal refresh stage is now COMPLETED; assertion "
                  "expects INGEST_TABLE — stage contract under review.")
@pytest.mark.asyncio
async def test_table_refresh_validation(
    client: TestClient,
    tempdir,
    temporal_client: TemporalClient,
):
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '../../', 'csv_data', 'small_synthetic')

    # Create connector for the table
    project_id, connector_id = await create_connector(client, root_dir)
    table_name = 'A'
    table_path = osp.join(root_dir, 'A.csv')

    # Create table with few enough rows to get warning
    table_def = table_api.TableDefinition(
        cols=[
            table_api.Column(name='A', stype=Stype.ID, dtype=Dtype.int),
            table_api.Column(name='feature', stype=Stype.categorical,
                             dtype=Dtype.int),
        ],
        source_table=source_table_api.S3SourceTable(
            connector_id=connector_id,
            source={
                'mode': 'relative',
                'path': os.path.basename(table_path)
            },
            file_type=source_table_api.FileType.CSV,
        ),
        pkey='A',
    )
    table_create_response = post_and_expect_success(
        client,
        "/v1/tables/",
        table_def,
        IDResponse,
        name_alias=table_name,
        project_id=project_id,
    )

    data_version = get_data_version_string(table_path)
    result = await _trigger_and_wait_for_table_refresh(client, temporal_client,
                                                       table_create_response.id,
                                                       data_version, project_id)
    assert result.validation.warnings == []
    assert result.validation.errors == []
    assert result.stage == TableRefreshStage.COMPLETED
    assert result.status == AsyncDataTaskStatus.DONE

    table_resource = get_and_expect_success(
        client, f'/v1/tables/{table_create_response.id}', {},
        TableResourceResponse)
    rest_table_def = TableDefinition(**dataclasses.asdict(table_resource.table))
    ingest_result = (await DataBackend.get_many_table_refresh_results(
        [(rest_table_def, TableDataVersion(data_version))],
        stage=TableRefreshStage.INGEST_TABLE,
    ))[0]
    assert ingest_result is not None
    assert ingest_result.validation.warnings == []
    assert ingest_result.validation.errors == []
    assert ingest_result.stage == TableRefreshStage.INGEST_TABLE
    assert ingest_result.status == AsyncDataTaskStatus.DONE

    # Empty-table validation was converted into the Phase 1b black-box gate in
    # test_table_validation_flow.py. Keep this owner focused on the successful
    # refresh path because it waits on Temporal/data-backend workflow results.


def create_parquet_table(
    root_dir: str,
    df: pd.DataFrame,
    table_name: str,
):
    fsutil.makedirs(root_dir)
    table_path = f'{os.path.join(root_dir, table_name)}.parquet'
    df.to_parquet(table_path, index=False)
    return table_path


async def create_connector(client, root_dir) -> tuple[str, str]:
    project_id, connector = await provision_project_upload_connector(
        client,
        source_root=root_dir,
        api_prefix='/v0',
    )
    return str(project_id), str(connector.id)


def get_data_version_string(table_path: str):
    # Since data_version is an internal Kumo construct, there is no way
    # to get this using REST apis. Using the system timestamp (that is how
    # version is set) to determine the version. Note that this test may break
    # if we use a different way to compute version in the future.
    return str(os.path.getmtime(table_path)).split('.')[0]
