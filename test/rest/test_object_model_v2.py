# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from io_utils.json import load_json
from kumo.testing.rest import get_and_expect_success
from rest.config.graph_v2 import GraphDefinition, GraphResourceResponse
from rest.config.predictive_query_v2 import PQueryResource
from rest.config.table_v2 import TableResourceResponse
from rest.deps.authentication import RequestContext
from rest.deps.context import _current_user_ctx
from rest.driver.rest_api_driver import RestApiDriver
from statedb.config.common import (
    ID,
    GraphIDv2,
    PredictiveQueryID,
    ProjectID,
    TableIDv2,
    TableName,
)


def _load_pquery_res() -> PQueryResource:
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
    pquery_res_path = os.path.join(dir, 'pquery_resource.json')
    return load_json(pquery_res_path, PQueryResource)


async def _register_project_file_connector(
    rest_api_driver: RestApiDriver,
    pquery_res: PQueryResource,
    project_id: ProjectID,
) -> None:
    state_services = rest_api_driver._state_services
    assert state_services is not None
    root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                            'csv_data', 'FINANCIAL')
    connector_id = await state_services.connectors.register_project_upload_connector(
        project_id, root_dir)
    for table_name, table in list(pquery_res.graph.tables.items()):
        pquery_res.graph.tables[table_name] = dataclasses.replace(
            table,
            source_table=dataclasses.replace(
                table.source_table,
                connector_id=str(connector_id),
            ),
        )


@pytest_asyncio.fixture(scope='function', autouse=True)
async def clear_db_before_test(storage_dao):
    # Start with a clean StateDB.
    await storage_dao.reset_storage()


@pytest.fixture
def driver_principal(ctx: RequestContext):
    token = _current_user_ctx.set(ctx.user_authorization)
    try:
        yield
    finally:
        _current_user_ctx.reset(token)


@pytest.mark.asyncio
async def test_pquery_v2(client: TestClient, rest_api_driver: RestApiDriver,
                         ctx: RequestContext, default_project_id: str,
                         driver_principal):
    project_id = ProjectID(default_project_id)
    pquery_res = _load_pquery_res()
    await _register_project_file_connector(rest_api_driver, pquery_res,
                                           project_id)
    pquery_driver = rest_api_driver.pquery_driver
    pquery_id = await pquery_driver.create_predictive_query_v2(
        pquery_res, project_id)
    _, fpid = PredictiveQueryID.parse(pquery_id)
    assert fpid is not None

    pquery_resource = await pquery_driver.get_predictive_query_v2(
        pquery_id, project_id)
    # desc is not persisted in PQueryDefinition, so exclude it from comparison.
    pquery_res.desc = pquery_resource.desc
    assert pquery_resource == pquery_res

    # V2 LIST returns the created pquery (driver-level; no HTTP equivalent on
    # the internal app).
    pqueries = await pquery_driver.list_predictive_queries_v2(project_id)
    assert len(pqueries) == 1
    assert pqueries[0] == pquery_res

    # V2 GET graph by public id.
    graph_v2_id = GraphIDv2.public(pquery_res.graph.fpid)
    graph_resource = get_and_expect_success(client, f'/v1/graphs/{graph_v2_id}',
                                            {'project_id': project_id},
                                            GraphResourceResponse)
    assert graph_resource.id == graph_v2_id
    # Compare via asdict because the response may decode `graph` as the public
    # `kumoapi.graph.GraphDefinition` while `pquery_res.graph` is the internal
    # `rest.config.graph_v2.GraphDefinition`; dataclass __eq__ requires same
    # type.
    assert (dataclasses.asdict(graph_resource.graph) == dataclasses.asdict(
        pquery_res.graph))

    # V2 GET each table by composite id; fpids come from the graph response.
    for table_name, table_fpid in graph_resource.table_name_to_fpid_mapping.items(
    ):
        table_v2_id = TableIDv2.public(
            pquery_id=None,
            table_name=TableName("NA"),
            fpid=table_fpid,
        )
        table_resource = get_and_expect_success(client,
                                                f'/v1/tables/{table_v2_id}', {
                                                    'project_id': project_id,
                                                }, TableResourceResponse)
        assert table_resource.id == table_fpid
        assert (dataclasses.asdict(table_resource.table) == dataclasses.asdict(
            pquery_res.graph.tables[table_name]))


@pytest.mark.asyncio
async def test_graph_deletion(rest_api_driver: RestApiDriver,
                              ctx: RequestContext, default_project_id: str,
                              driver_principal):
    project_id = ProjectID(default_project_id)
    graph_driver = rest_api_driver.graph_driver
    pquery_driver = rest_api_driver.pquery_driver
    pquery_res = _load_pquery_res()
    await _register_project_file_connector(rest_api_driver, pquery_res,
                                           project_id)

    # Create a pquery with the graph.
    pquery_id = await pquery_driver.create_predictive_query_v2(
        pquery_res, project_id)

    # Attempt to create the same graph again.
    graph_alias = ID('test_graph')
    graph_def = GraphDefinition(**dataclasses.asdict(pquery_res.graph))
    graph_id = await graph_driver.create_full_graph_v2(graph_def,
                                                       project_id=project_id,
                                                       alias=graph_alias)

    graph_resource, _, _ = await graph_driver.get_graph_v2_by_alias(
        graph_alias, project_id)
    assert graph_resource.graph == pquery_res.graph
    assert graph_resource.id == graph_id

    # Delete the pquery, but graph should remain
    await pquery_driver.delete_predictive_query(ctx, pquery_id, project_id)
    graph_resource, _, _ = await graph_driver.get_graph_v2_by_alias(
        graph_alias, project_id)
    assert graph_resource.graph == pquery_res.graph
    assert graph_resource.id == graph_id

    # Attempt to create the same graph again
    assert graph_id == await graph_driver.create_full_graph_v2(
        graph_def, project_id=project_id, alias='test_graph')
    # Force rename the graph
    graph_alias2 = ID('test_graph2')
    assert graph_id == await graph_driver.create_full_graph_v2(
        graph_def, project_id=project_id, alias=graph_alias2, force_rename=True)
    graph_resource, _, _ = await graph_driver.get_graph_v2_by_alias(
        graph_alias2, project_id)
    assert graph_resource.graph == pquery_res.graph
    assert graph_resource.id == graph_id
    assert graph_resource.name_alias == graph_alias2

    with pytest.raises(KeyError):
        # This graph alias doesn't exist anymore (graph was renamed)
        await graph_driver.get_graph_v2_by_alias(graph_alias, project_id)
