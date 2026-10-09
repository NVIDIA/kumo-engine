# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from http import HTTPStatus
from types import SimpleNamespace
from typing import List

import pytest
from fastapi.testclient import TestClient

from kumo.testing.rest import (
    get_and_expect_success,
    post_and_expect_error,
    post_and_expect_success,
)
from rest.config.graph_v2 import (
    ColumnKeyGroup,
    GraphDefinition,
    GraphResourceResponse,
)
from rest.config.status import EmptyResponse, IDResponse
from statedb.db.identities import ConnectorService


def test_refresh_graph(client: TestClient, fake_graph_v2: GraphResourceResponse,
                       default_project_id: str):
    # Should refresh if graph exists
    post_and_expect_success(
        client,
        f"/v1/graphs/{fake_graph_v2.id}/refresh?project_id={default_project_id}",
        {}, EmptyResponse)

    # Should return error if graph does not exist. Use a well-formed (but
    # nonexistent) v2 public graph id so it resolves past id parsing and hits
    # the GraphDoesNotExistError -> 404 path.
    post_and_expect_error(
        client,
        f"/v1/graphs/graph-{'a' * 32}/refresh?project_id={default_project_id}",
        {}, HTTPStatus.NOT_FOUND, 'does not exist')


def test_graph_to_sdk_s3_connector(
    client: TestClient,
    graph_definition_v2: GraphDefinition,
    default_project_id: str,
    monkeypatch: pytest.MonkeyPatch,
):
    graph_def = GraphDefinition(**dataclasses.asdict(graph_definition_v2))

    graph_id = post_and_expect_success(
        client=client,
        path=f'/v1/graphs/?project_id={default_project_id}',
        payload=graph_def,
        response_type=IDResponse,
    )

    assert len(graph_id.id) > 0
    assert graph_id.id.startswith("graph-") is True

    # The shared HNM fixture is deliberately an internal local-file connector
    # so ingestion can run without cloud credentials. SDK replay only supports
    # public project connectors, so expose the fixture's safe name at that
    # boundary without exposing its internal root.
    async def visible_test_connector(self, project_id, connector_id,
                                     principal):
        connector = await self.get_internal(project_id, connector_id)
        return SimpleNamespace(id=connector.id, name=connector.name)

    monkeypatch.setattr(ConnectorService,
                        'source_table_connector_for_project',
                        visible_test_connector)

    graph_sdk_script = get_and_expect_success(
        client=client,
        path=f"/v1/graphs/{graph_id.id}/sdk",
        params={'project_id': default_project_id},
        response_type=str,
    )
    graph_sdk_script = json.loads(graph_sdk_script)
    assert graph_sdk_script != ""
    assert f'project="{default_project_id}"' in graph_sdk_script
    assert 'Connector.get_by_name(' in graph_sdk_script
    assert 'root_dir=' not in graph_sdk_script
    assert "graph = kumo.Graph(" in graph_sdk_script
    assert "graph.validate()" in graph_sdk_script


def test_suggest_links_v2(
    client: TestClient,
    graph_definition_v2_no_edges: GraphDefinition,
    default_project_id: str,
):
    graph_def = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2_no_edges))

    col_groups = post_and_expect_success(
        client,
        f"/v1/graphs/suggest_links?project_id={default_project_id}",
        graph_def,
        List[ColumnKeyGroup],
    )
    assert len(col_groups) == 2  # 2 column groups expected.
    assert len(col_groups[0].columns) == 2  # 2 entries in first group
    assert len(col_groups[1].columns) == 2  # 2 entries in second group
