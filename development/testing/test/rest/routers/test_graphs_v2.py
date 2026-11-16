# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import kumoapi.graph as api_graph
import kumoapi.table as api_table
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from kumoapi.json_serde import to_json_dict
from kumoapi.typing import Dtype, Stype

from rest.config.graph_v2 import ListResponse
from rest.config.table_v2 import DataSourceType, FileType, S3SourceTable
from rest.routers.graphs_v2 import GraphRouter

_PROJECT_ID = "project_" + "1" * 32
_CONNECTOR_ID = "connector_" + "1" * 32


@pytest.fixture(autouse=True, scope="session")
def setup_rest_test_session():
    yield


@pytest.fixture(autouse=True, scope="session")
def temporal_workers_ready():
    yield True


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def graph_router():
    auth = SimpleNamespace(authenticate_request_header=lambda: None)
    graph_driver = AsyncMock()
    return GraphRouter(auth, graph_driver)


@pytest.fixture
def graph_definition_v2():
    """Minimal two-table graph definition for unit tests (no server required)."""
    return api_graph.GraphDefinition(
        tables={
            "transactions":
            api_table.TableDefinition(
                cols=[
                    api_table.Column(name="customer_id", stype=Stype.ID,
                                     dtype=Dtype.string),
                    api_table.Column(name="article_id", stype=Stype.ID,
                                     dtype=Dtype.string),
                ],
                source_table=S3SourceTable(
                    connector_id=_CONNECTOR_ID,
                    source={
                        'mode': 'relative',
                        'path': "transactions"
                    },
                    file_type=FileType.PARQUET,
                    data_source_type=DataSourceType.S3,
                ),
                pkey=None,
                time_col=None,
                end_time_col=None,
            ),
            "customers":
            api_table.TableDefinition(
                cols=[
                    api_table.Column(name="customer_id", stype=Stype.ID,
                                     dtype=Dtype.string),
                ],
                source_table=S3SourceTable(
                    connector_id=_CONNECTOR_ID,
                    source={
                        'mode': 'relative',
                        'path': "customers"
                    },
                    file_type=FileType.PARQUET,
                    data_source_type=DataSourceType.S3,
                ),
                pkey="customer_id",
                time_col=None,
                end_time_col=None,
            ),
        },
        col_groups=[
            api_graph.ColumnKeyGroup(columns=(
                api_graph.ColumnKey(table_name="transactions",
                                    col_name="customer_id"),
                api_graph.ColumnKey(table_name="customers",
                                    col_name="customer_id"),
            )),
        ],
    )


@pytest.mark.asyncio
async def test_create_graph_v2_project_scoped(
    graph_router: GraphRouter,
    graph_definition_v2,
):
    """Happy path passes the one project-local alias to the driver."""
    graph_router._graph_driver.create_full_graph_v2.return_value = (
        "graph-1234abcd")

    response = await graph_router.create_graph_v2(
        graph=graph_definition_v2,
        project_id=_PROJECT_ID,
        name_alias="orders_graph",
    )

    graph_router._graph_driver.create_full_graph_v2.assert_called_once_with(
        ANY, alias="orders_graph", force_rename=False, skip_validation=False,
        project_id=_PROJECT_ID)
    assert response.id == "graph-1234abcd"


@pytest.mark.asyncio
async def test_list_graph_names_is_project_scoped(graph_router: GraphRouter):
    expected = [
        ListResponse(name="g1", id="graph-abc", connector_id=None,
                     source_table_name=None, timestamps=None)
    ]
    graph_router._graph_driver.list_project_graph_names.return_value = expected

    actual = await graph_router.list_graph_names_v2(project_id=_PROJECT_ID)

    graph_router._graph_driver.list_project_graph_names.assert_called_once_with(
        _PROJECT_ID)
    assert actual == expected


@pytest.mark.asyncio
async def test_validate_name_is_project_scoped(graph_router: GraphRouter):
    expected = SimpleNamespace(is_valid=True, error_msg=None)
    graph_router._graph_driver.validate_project_graph_name.return_value = expected

    result = await graph_router.validate_name_v2("new_graph",
                                                 project_id=_PROJECT_ID)

    graph_router._graph_driver.validate_project_graph_name.assert_called_once_with(
        project_id=_PROJECT_ID, name="new_graph")
    assert result is expected


@pytest.mark.asyncio
async def test_create_graph_v2_empty_name_alias_treated_as_no_alias(
    graph_router: GraphRouter,
    graph_definition_v2,
):
    """Empty name_alias must not reach the driver; it would cause get_by_alias
    to match all anonymous graphs (normalized_alias='') and crash."""
    graph_router._graph_driver.create_full_graph_v2.return_value = (
        "graph-1234abcd")

    await graph_router.create_graph_v2(
        graph=graph_definition_v2,
        project_id=_PROJECT_ID,
        name_alias="",
    )

    graph_router._graph_driver.create_full_graph_v2.assert_called_once_with(
        ANY, alias=None, force_rename=False, skip_validation=False,
        project_id=_PROJECT_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'removed_field',
    ('name_alias_in_project', 'display_name_in_project', 'created_by_name'),
)
async def test_create_graph_rejects_removed_query_fields_over_asgi(
    graph_router: GraphRouter,
    graph_definition_v2,
    removed_field: str,
) -> None:
    app = FastAPI()
    app.include_router(graph_router.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport,
                           base_url='http://test') as client:
        response = await client.post(
            '/v1/graphs/',
            params={
                'project_id': _PROJECT_ID,
                removed_field: 'ignored-before-strict-validation',
            },
            json=to_json_dict(graph_definition_v2),
        )

    assert response.status_code == 422
    assert removed_field in response.text
    graph_router._graph_driver.create_full_graph_v2.assert_not_awaited()
