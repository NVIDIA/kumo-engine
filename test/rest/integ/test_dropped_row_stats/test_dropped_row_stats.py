# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that dropped row statistics are computed correctly as a colstat"""
from pathlib import Path
from test.rest.conftest import TestClient
from test.rest.integ import fixture_connector_args

import kumoapi.source_table as source_table_api
import kumoapi.table as table_api
import pytest
from kumo_core.types import ColStatType, TableStatType
from kumo_core.types.schema import BaseColumnConfig
from kumoapi.source_table import (
    SourceTableDataResponse,
    SourceTableListResponse,
)
from kumoapi.typing import Stype
from kumoapi.typing import Stype as ApiStype

from kumo.data import get_active_cache
from kumo.data.backend_cache import TableDataTaskResult
from kumo.data.common import TableRefreshStage
from kumo.testing.rest import get_and_expect_success, post_and_expect_success
from kumo.validation import ValidationResponse
from rest.backends.data_backend.workflows import TableRefreshWorkflow
from rest.config.status import EmptyResponse, IDResponse
from rest.driver.rest_api_driver import RestApiDriver
from statedb.config.common import TableIDv2
from statedb.config.table import TableDataVersion
from temporalio_commons import TemporalClient

from .config_for_test import dropped_rows_200_na_args, dropped_rows_no_na_args


@pytest.mark.asyncio
async def test_dropped_row_stats_both_tables(client: TestClient,
                                             temporal_client: TemporalClient,
                                             tmp_path: Path,
                                             rest_api_driver: RestApiDriver):
    """
    Test that dropped row statistics are correctly recorded during ingestion
    for both test cases using the same source file with different column
    combinations.

    Test cases:
    1. Using article_id1 + time_col1: Should have 0 dropped rows in
       DROPPED_ROWS col stat (both columns have no NAs)
    2. Using article_id2 + time_col2: Should have 200 dropped rows in
       DROPPED_ROWS col stat (100 NAs in each column, non-overlapping rows)

    Both should have the same table stats (NUM_ROWS, SOURCE_SIZE) but different
    column-level DROPPED_ROWS statistics for the time_col.
    """
    # Setup file connector pointing to the dropped rows test data
    root_dir = ("s3://kumocloud-test-bucket/ux_e2e_stability_tests/"
                "generated_data/dropped_rows_stats_data/")
    project_id, connector_entry = await fixture_connector_args.provision_project_upload_connector(
        client,
        source_root=root_dir,
        api_prefix='/v0',
    )

    # 1. Create connector and list source tables
    connector_id = str(connector_entry.id)

    source_tables = get_and_expect_success(
        client,
        f'/v0/projects/{project_id}/connectors/{connector_id}/source-tables',
        {}, SourceTableListResponse)
    source_table_names = source_tables.table_names

    # Verify the test table exists (both test cases use the same source file)
    assert dropped_rows_no_na_args['source_name'] in source_table_names
    assert dropped_rows_200_na_args['source_name'] in source_table_names

    test_cases = [dropped_rows_no_na_args, dropped_rows_200_na_args]

    # Step 1: Ingest both tables first
    ingested_tables = {}

    for test_case in test_cases:
        # 2. Get table samples with automatic type inference
        table_samples = get_and_expect_success(
            client,
            f'/v0/projects/{project_id}/connectors/{connector_id}/sample',
            {'table_name': test_case['source_name']}, SourceTableDataResponse)

        # Only include columns that are supported when creating the table
        columns = []
        for col in table_samples.cols:
            if col.stype != Stype.unsupported:
                # Force primary key column to be ID type
                if col.name == test_case['pkey']:
                    columns.append(
                        table_api.Column(name=col.name, stype=ApiStype.ID,
                                         dtype=col.dtype))
                elif col.stype is not None:
                    columns.append(
                        table_api.Column(name=col.name, stype=col.stype,
                                         dtype=col.dtype))

        pkey_name = test_case['pkey']
        time_col_name = test_case['time_cols']

        # Add pkey if not already included
        if pkey_name and not any(col.name == pkey_name for col in columns):
            # Find the column in table_samples to get its dtype
            pkey_col = next(
                (col for col in table_samples.cols if col.name == pkey_name),
                None)
            if pkey_col:
                columns.append(
                    table_api.Column(name=pkey_name, stype=ApiStype.ID,
                                     dtype=pkey_col.dtype))

        # Add time_col if not already included
        if time_col_name and not any(col.name == time_col_name
                                     for col in columns):
            # Find the column in table_samples to get its dtype
            time_col_col = next(
                (col
                 for col in table_samples.cols if col.name == time_col_name),
                None)
            if time_col_col:
                columns.append(
                    table_api.Column(name=time_col_name,
                                     stype=ApiStype.timestamp,
                                     dtype=time_col_col.dtype))

        # Create table definition with pkey and time_col
        table_args = table_api.TableDefinition(
            cols=columns,
            source_table=source_table_api.S3SourceTable(
                connector_id=connector_id,
                source={
                    'mode': 'relative',
                    'path': test_case['source_name']
                },
                file_type=source_table_api.FileType.PARQUET,
            ),
            pkey=test_case['pkey'] if test_case['pkey'] else None,
            time_col=test_case['time_cols'] if test_case['time_cols'] else None,
        )

        # 3. Validate table definition
        val_response = post_and_expect_success(
            client,
            "/v1/tables/validate",
            table_args,
            ValidationResponse,
            project_id=str(project_id),
        )
        assert val_response.ok

        # 4. Add the table
        table_response = post_and_expect_success(
            client,
            "/v1/tables/",
            table_args,
            IDResponse,
            name_alias=test_case['table_name'],
            project_id=str(project_id),
        )
        table_id = table_response.id
        assert table_id is not None

        # 5. Refresh(ingest) the table
        connector = connector_entry.config.as_connector()
        table_version = await connector.get_data_version.coroutine(
            test_case['source_name'])

        post_and_expect_success(
            client,
            f"/v1/tables/{table_id}/refresh",
            {},
            EmptyResponse,
            project_id=str(project_id),
        )

        # Wait for table to be ingested
        normalized_table_id = (table_id
                               if '.' in table_id else f'NA.{table_id}')
        table_fpid = TableIDv2.parse(
            TableIDv2.from_public(normalized_table_id))[2]
        assert table_fpid is not None
        table_resource = await rest_api_driver.table_driver.get_table(
            table_fpid, str(project_id))
        table_def = table_resource.table
        assert table_def is not None
        wf_id = TableRefreshWorkflow.compose_wf_id(
            table_def, TableDataVersion(table_version))
        result: TableDataTaskResult = await temporal_client.result(wf_id)

        # Verify ingestion completed successfully
        assert result.validation.ok
        assert result.stage == TableRefreshStage.COMPLETED

        # Store refreshed table results for later comparison
        ingested_tables[test_case['table_name']] = {
            'table_id': table_id,
            'table_version': table_version,
            'table_def': table_def,
            'test_case': test_case,
            'expected_dropped_rows': test_case['expected_dropped_rows']
        }

    # Step 2: Now get stats for both tables and compare them
    results = {}

    for table_name, table_info in ingested_tables.items():
        table_id = table_info['table_id']
        table_version = table_info['table_version']
        table_def = table_info['table_def']
        test_case = table_info['test_case']
        expected_dropped_rows = table_info['expected_dropped_rows']

        # 6. Verify dropped row statistics
        cache = get_active_cache()

        workflow_fingerprint = table_def.fingerprint()

        # Get table stats - should be the same for both tables
        table_stats = cache.get_table_stats(
            workflow_fingerprint,
            [TableStatType.NUM_ROWS, TableStatType.SOURCE_SIZE], table_version)

        # Verify table stats exist
        assert 'NUM_ROWS' in table_stats
        assert 'SOURCE_SIZE' in table_stats

        # Needed since the cache in this test process is different than
        # the cache used in the API call.
        cache.load_col_stats_cache(workflow_fingerprint, table_version)

        # Create the stats_dict that maps column configs to desired stat types
        stats_dict = {}
        for col in table_def.cols:
            if col.name == test_case['time_cols']:
                col_config = table_def.col_to_base_config(col.name)

                # Ensure timestamp_unit is set to 's' to match workflow
                if col_config.timestamp_unit != 's':
                    col_config = BaseColumnConfig(name=col_config.name,
                                                  stype=col_config.stype,
                                                  dtype=col_config.dtype,
                                                  metadata=col_config.metadata,
                                                  timestamp_unit='s')

                stats_dict[col_config] = [ColStatType.DROPPED_ROWS]

        col_stats = cache.get_col_stats(workflow_fingerprint, stats_dict,
                                        table_version, approx=False)

        time_col_config = None
        for col_config in col_stats.keys():
            if col_config.name == test_case['time_cols']:
                time_col_config = col_config
                break

        assert time_col_config is not None, (
            f"time_col {test_case['time_cols']} not found in column stats")

        # Verify DROPPED_ROWS statistic exists and has the expected value
        time_col_stats = col_stats[time_col_config]
        assert ColStatType.DROPPED_ROWS in time_col_stats, (
            "DROPPED_ROWS statistic not found for time_col")

        dropped_rows_value = time_col_stats[ColStatType.DROPPED_ROWS].value
        expected_dropped_rows = test_case['expected_dropped_rows']

        # Store results for comparison (don't assert yet)
        results[table_name] = {
            'dropped_rows': dropped_rows_value,
            'table_rows': table_stats['NUM_ROWS'].value,
            'table_size': table_stats['SOURCE_SIZE'].value,
            'source_name': test_case['source_name'],
            'expected_dropped_rows': expected_dropped_rows
        }

    # First, verify individual table stats match expectations
    for table_name, table_result in results.items():
        expected = table_result['expected_dropped_rows']
        actual = table_result['dropped_rows']
        assert actual == expected, (
            f"Expected {expected} dropped rows for {table_name}, "
            f"but got {actual}")

    # Verify that both tables have the same table-level stats
    # (since they're the same data, just with different NA patterns)
    first_table = list(results.keys())[0]
    second_table = list(results.keys())[1]

    first_table_rows = results[first_table]['table_rows']
    second_table_rows = results[second_table]['table_rows']
    assert first_table_rows == second_table_rows, (
        f"Table row counts should be the same: "
        f"{first_table_rows} vs {second_table_rows}")

    no_na_size = results['dropped_rows_no_na']['table_size']
    na_200_size = results['dropped_rows_200_na']['table_size']

    # Both tables should have the same size since they use the same source file
    assert na_200_size == no_na_size
    # Verify specific dropped row values for each table
    no_na_dropped = results['dropped_rows_no_na']['dropped_rows']
    na_200_dropped = results['dropped_rows_200_na']['dropped_rows']
    assert no_na_dropped == 0
    assert na_200_dropped == 200
