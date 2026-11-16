# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import re

import kumoapi.table as table_api
import pandas as pd
from fastapi.testclient import TestClient
from kumoapi.graph import ColumnKey, ColumnKeyGroup
from kumoapi.graph import GraphDefinition as ApiGraphDefinition
from kumoapi.source_table import FileType, S3SourceTable
from kumoapi.typing import Dtype, Stype

from io_utils import fsutil
from io_utils.json import load_json
from kumo.testing.rest import (
    get_and_expect_success,
    post_and_expect_success,
    randomize_fixture_ids_v2,
    remap_fixture_connector_ids_v2,
)
from rest.config.api_command import FixtureDefinition, FixturePQueryRecord
from rest.config.graph_v2 import GraphDefinition
from rest.config.graph_v2 import GraphDefinition as RestGraphDefinition
from rest.config.predictive_query_v2 import PQueryResource, PQueryResponse
from rest.config.status import EmptyResponse, IDResponse
from rest.config.table_v2 import TableDefinition
from rest.driver.rest_api_driver import RestApiDriver
from rest.routers.fixture_utils import fixture_to_pquery_resources_v2
from rest.routers.fixtures import LoadJsonArgs, SavePQueryArgs
from statedb.config.common import PredictiveQueryID, ProjectID


def _create_local_roundtrip_pquery(
    client: TestClient,
    rest_api_driver: RestApiDriver,
    tmp_path,
) -> tuple[PredictiveQueryID, ProjectID]:
    root = tmp_path / 'fixture_roundtrip'
    root.mkdir()
    pd.DataFrame({
        'customer_id': ['customer-1'],
    }).to_parquet(root / 'customers.parquet')
    pd.DataFrame({
        't_dat': pd.to_datetime(['2024-01-01']),
        'customer_id': ['customer-1'],
    }).to_parquet(root / 'transactions.parquet')

    async def create_project_connector() -> tuple[ProjectID, str]:
        services = rest_api_driver._require_state_services()
        _, project = await services.groups.ensure_default_resources()
        connector_id = await services.connectors.register_project_upload_connector(
            project.id, str(root))
        return project.id, str(connector_id)

    project_id, connector_id = asyncio.run(create_project_connector())
    graph = ApiGraphDefinition(
        tables={
            'customers':
            table_api.TableDefinition(
                cols=[
                    table_api.Column(
                        name='customer_id',
                        stype=Stype.ID,
                        dtype=Dtype.string,
                    ),
                ],
                source_table=S3SourceTable(
                    connector_id=connector_id,
                    source={'mode': 'relative', 'path': 'customers.parquet'},
                    file_type=FileType.PARQUET,
                ),
                pkey='customer_id',
            ),
            'transactions':
            table_api.TableDefinition(
                cols=[
                    table_api.Column(
                        name='t_dat',
                        stype=Stype.timestamp,
                        dtype=Dtype.time,
                    ),
                    table_api.Column(
                        name='customer_id',
                        stype=Stype.ID,
                        dtype=Dtype.string,
                    ),
                ],
                source_table=S3SourceTable(
                    connector_id=connector_id,
                    source={'mode': 'relative', 'path': 'transactions.parquet'},
                    file_type=FileType.PARQUET,
                ),
                time_col='t_dat',
            ),
        },
        col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey(
                    table_name='transactions',
                    col_name='customer_id',
                ),
                ColumnKey(
                    table_name='customers',
                    col_name='customer_id',
                ),
            )),
        ],
    )
    post_and_expect_success(client, f'/v1/graphs/?project_id={project_id}',
                            graph, IDResponse)
    pquery_id = PredictiveQueryID('fixture_roundtrip_query')
    post_and_expect_success(
        client,
        f'/v1/predictive_queries/?project_id={project_id}',
        PQueryResource(
            name=str(pquery_id),
            query_string='PREDICT COUNT(transactions.*, 0, 20) FOR EACH '
            'customers.customer_id',
            graph=RestGraphDefinition(**dataclasses.asdict(graph)),
            desc='Fixture export round-trip test query',
        ),
        IDResponse,
    )
    return pquery_id, project_id


def test_save_and_load_json_uses_existing_project_connectors(
    fixture_export_path: str,
    rest_api_driver: RestApiDriver,
    client: TestClient,
    tmp_path,
):
    pquery_id, project_id = _create_local_roundtrip_pquery(
        client, rest_api_driver, tmp_path)
    json_file_path = post_and_expect_success(
        client,
        '/v0/fixtures/save_json',
        SavePQueryArgs(
            project_id=project_id,
            pquery_ids=[pquery_id],
            output_dir=fixture_export_path,
            fixture_name='my_fixture',
        ),
        str,
    ).strip('"')

    assert fsutil.exists(json_file_path)
    saved = load_json(json_file_path, FixtureDefinition)
    assert saved.graphs
    connector_ids = {
        table.get_connector_id()
        for graph in saved.graphs
        for table in graph.tables.values()
    }
    assert len(connector_ids) == 1
    assert all(not hasattr(table.source_table, 's3_path')
               for graph in saved.graphs for table in graph.tables.values())

    post_and_expect_success(
        client,
        '/v0/fixtures/load_json',
        LoadJsonArgs(file_path=json_file_path, project_id=project_id),
        EmptyResponse,
    )
    assert get_and_expect_success(
        client,
        f'/v1/predictive_queries/{pquery_id}',
        {'project_id': project_id},
        PQueryResponse,
    )


def test_fixture_to_pquery_resources_v2_preserves_model_planner_yaml(
    graph_definition_v2, ):
    graph_def = GraphDefinition(**dataclasses.asdict(graph_definition_v2))
    model_plan_yaml = 'training_table_generation:\n  lag_timesteps: 3'
    table_defs = [
        TableDefinition(**dataclasses.asdict(table))
        for table in graph_definition_v2.tables.values()
    ]
    fixture = FixtureDefinition(
        tables=table_defs,
        graphs=[graph_def],
        queries=[
            FixturePQueryRecord(
                query_str='PREDICT COUNT(transactions.*, 0, 20) FOR EACH '
                'customers.customer_id',
                graph_fpid=graph_def.fpid,
                pquery_name='pquery_with_model_plan',
                model_planner_yaml=model_plan_yaml,
            )
        ],
        predictions=None,
    )
    resources = fixture_to_pquery_resources_v2(fixture)
    assert len(resources) == 1
    assert resources[0].advanced_options_yaml == model_plan_yaml
    assert resources[0].list_distinct_target_is_array is None


def test_randomize_fixture_ids_v2_generates_canonical_connector_ids(
    graph_definition_v2, ):
    # Checked-in integration fixtures predate durable connector IDs and carry
    # descriptive import aliases.  The importer must canonicalize those raw
    # references before any runtime code validates ConnectorID.
    legacy_tables = {
        table_name:
        dataclasses.replace(
            table,
            source_table=dataclasses.replace(
                table.source_table,
                connector_id='financial_csv_local_connector',
            ),
        )
        for table_name, table in graph_definition_v2.tables.items()
    }
    graph_def = GraphDefinition(**dataclasses.asdict(
        dataclasses.replace(graph_definition_v2, tables=legacy_tables)))
    fixture = FixtureDefinition(
        tables=[
            TableDefinition(**dataclasses.asdict(table))
            for table in graph_definition_v2.tables.values()
        ],
        graphs=[graph_def],
        queries=[
            FixturePQueryRecord(
                query_str='PREDICT COUNT(*) FOR EACH customer_id',
                graph_fpid=graph_def.fpid,
                pquery_name='fixture_connector_remap',
            )
        ],
        predictions=None,
    )
    randomized = randomize_fixture_ids_v2(fixture)
    randomized_ids = {
        str(table.get_connector_id())
        for table in randomized.graphs[0].tables.values()
    }

    assert 'financial_csv_local_connector' not in randomized_ids
    assert all(
        re.fullmatch(r'connector_[a-f0-9]{32}', connector_id)
        for connector_id in randomized_ids)

    created_connector_id = f'connector_{"a" * 32}'
    remapped = remap_fixture_connector_ids_v2(
        randomized,
        {next(iter(randomized_ids)): created_connector_id},
    )
    assert {
        str(table.get_connector_id())
        for table in remapped.graphs[0].tables.values()
    } == {created_connector_id}
    assert remapped.graphs[0].fpid != randomized.graphs[0].fpid
    assert remapped.queries[0].graph_fpid == remapped.graphs[0].fpid
