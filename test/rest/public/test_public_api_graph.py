# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from http import HTTPStatus
from pathlib import Path
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from kumoapi.data_source import DEFAULT_PROJECT_ID

from kumo.validation import ValidationError, ValidationResponse
from rest.driver.table_driver import TableDriver

# Converted owner for Phase 1b graph lifecycle coverage. Keep this file
# API-visible only: public_api_client plus raw HTTP JSON assertions.
_PQUERY_RESOURCE_PATH = (Path(__file__).resolve().parents[1] / "test_data" /
                         "pquery_resource.json")


def _load_graph_definition(client: TestClient) -> dict:
    pquery = json.loads(_PQUERY_RESOURCE_PATH.read_text(encoding="utf-8"))
    graph = pquery["graph"]
    project_id = _project_id(client)
    response = client.get(
        f"/v1/projects/{project_id}/connectors/uploads/connector")
    assert response.status_code == HTTPStatus.OK, response.text
    connector_id = response.json()["connector_id"]
    for table in graph["tables"].values():
        table["source_table"]["connector_id"] = connector_id
    return graph


def _project_id(client: TestClient) -> str:
    del client
    return DEFAULT_PROJECT_ID


def _post_graph(
    client: TestClient,
    graph: dict,
    *,
    params: dict | None = None,
) -> dict:
    request_params = dict(params or {})
    request_params["project_id"] = _project_id(client)
    request_params.setdefault("skip_validation", "true")
    response = client.post("/v1/graphs", params=request_params, json=graph)
    assert response.status_code == HTTPStatus.OK, response.text
    body = response.json()
    assert body["id"].startswith("graph-")
    return body


def _get_graph(client: TestClient, graph_id_or_alias: str) -> dict:
    response = client.get(
        f"/v1/graphs/{graph_id_or_alias}",
        params={"project_id": _project_id(client)},
    )
    assert response.status_code == HTTPStatus.OK, response.text
    body = response.json()
    assert body["id"].startswith("graph-")
    assert "graph" in body
    return body


def _assert_error(
    response,
    status_code: HTTPStatus,
    expected_text: str,
) -> None:
    assert response.status_code == status_code, response.text
    assert expected_text in response.text


def _assert_graph_matches_requested_definition(
    actual: dict,
    expected: dict,
) -> None:
    assert actual["col_groups"] == expected["col_groups"]
    assert set(actual["tables"]) == set(expected["tables"])
    for table_name, expected_table in expected["tables"].items():
        actual_table = actual["tables"][table_name]
        assert len(actual_table["cols"]) == len(expected_table["cols"])
        for actual_col, expected_col in zip(actual_table["cols"],
                                            expected_table["cols"]):
            for key, value in expected_col.items():
                assert actual_col[key] == value
        assert actual_table["pkey"] == expected_table["pkey"]
        assert actual_table["time_col"] == expected_table["time_col"]
        for key, value in expected_table["source_table"].items():
            assert actual_table["source_table"][key] == value


def test_graph_resource(public_api_client: TestClient):
    _assert_error(
        public_api_client.get(
            "/v1/graphs/graph-xyz",
            params={"project_id": _project_id(public_api_client)},
        ),
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "Graph resource id is malformed",
    )
    _assert_error(
        public_api_client.get(
            "/v1/graphs/graph_not_found",
            params={"project_id": _project_id(public_api_client)},
        ),
        HTTPStatus.NOT_FOUND,
        "not found",
    )

    graph_definition = _load_graph_definition(public_api_client)
    graph_id = _post_graph(public_api_client, graph_definition)["id"]
    assert graph_id.startswith("graph-")

    graph_resource = _get_graph(public_api_client, graph_id)
    assert graph_resource["id"] == graph_id
    assert graph_resource["name_alias"] is None
    _assert_graph_matches_requested_definition(
        graph_resource["graph"],
        graph_definition,
    )

    name_alias = "test_graph_xyz"
    # Assign an alias to an existing unnamed graph and get the primary id back.
    assert _post_graph(
        public_api_client,
        graph_definition,
        params={"name_alias": name_alias},
    )["id"] == graph_id
    # Recreating with the same alias is idempotent.
    assert _post_graph(
        public_api_client,
        graph_definition,
        params={"name_alias": name_alias},
    )["id"] == graph_id
    # Recreating with no alias is also idempotent.
    assert _post_graph(public_api_client, graph_definition)["id"] == graph_id

    _assert_error(
        public_api_client.post(
            "/v1/graphs",
            params={
                "name_alias": "new_name",
                "project_id": _project_id(public_api_client),
            },
            json=graph_definition,
        ),
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "cannot create another graph with the same definition "
        "under a different name",
    )

    graph_by_id = _get_graph(public_api_client, graph_id)
    graph_by_alias = _get_graph(public_api_client, name_alias)
    assert graph_by_id == graph_by_alias
    assert graph_by_id["name_alias"] == name_alias


def test_graph_rename(public_api_client: TestClient):
    graph_def1 = _load_graph_definition(public_api_client)
    graph_def2 = copy.deepcopy(graph_def1)
    assert graph_def1["tables"]["TRANS"]["cols"][3]["name"] == "TYPE"
    assert graph_def1["tables"]["TRANS"]["cols"][3]["stype"] == "categorical"
    graph_def2["tables"]["TRANS"]["cols"][3]["stype"] = "text"
    graph_def2["tables"]["TRANS"]["cols"][3]["dtype"] = "string"

    graph_id1 = _post_graph(
        public_api_client,
        graph_def1,
        params={"name_alias": "graph1"},
    )

    assert _post_graph(
        public_api_client,
        graph_def1,
        params={"name_alias": "graph1"},
    ) == graph_id1
    _assert_error(
        public_api_client.post(
            "/v1/graphs",
            params={
                "name_alias": "graph123",
                "project_id": _project_id(public_api_client),
            },
            json=graph_def1,
        ),
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "cannot create another graph with the same definition under "
        "a different name",
    )

    _post_graph(
        public_api_client,
        graph_def1,
        params={
            "name_alias": "graph123",
            "force_rename": "true"
        },
    )
    _assert_error(
        public_api_client.get(
            "/v1/graphs/graph1",
            params={"project_id": _project_id(public_api_client)},
        ),
        HTTPStatus.NOT_FOUND,
        "not found",
    )
    graph1 = _get_graph(public_api_client, "graph123")
    assert graph1["id"] == graph_id1["id"]
    assert graph1["name_alias"] == "graph123"

    graph_id2 = _post_graph(public_api_client, graph_def2)

    _assert_error(
        public_api_client.post(
            "/v1/graphs",
            params={
                "name_alias": "graph123",
                "project_id": _project_id(public_api_client),
            },
            json=graph_def2,
        ),
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "Graph name graph123 is already used in this project.",
    )
    _assert_error(
        public_api_client.post(
            "/v1/graphs",
            params={
                "name_alias": "graph123",
                "force_rename": "false",
                "project_id": _project_id(public_api_client),
            },
            json=graph_def2,
        ),
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "Graph name graph123 is already used in this project.",
    )

    assert _post_graph(
        public_api_client,
        graph_def2,
        params={
            "name_alias": "graph123",
            "force_rename": "true"
        },
    ) == graph_id2

    graph = _get_graph(public_api_client, "graph123")
    assert graph["id"] == graph_id2["id"]
    assert graph["name_alias"] == "graph123"
    assert _get_graph(public_api_client, graph1["id"])["id"] == graph1["id"]


def test_graph_skip_validation(
    public_api_client: TestClient,
    monkeypatch,
):
    graph_definition = _load_graph_definition(public_api_client)

    validator = AsyncMock(return_value=ValidationResponse(errors=[
        ValidationError(
            title="Table Validation Failure",
            message="source table does not exist",
        )
    ]))
    monkeypatch.setattr(TableDriver, "validate_table", validator)

    _assert_error(
        public_api_client.post(
            "/v1/graphs",
            params={
                "project_id": _project_id(public_api_client),
                "skip_validation": "false",
            },
            json=graph_definition,
        ),
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "Table Validation Failure",
    )
    graph_id1 = _post_graph(public_api_client, graph_definition)
    assert graph_id1["id"].startswith("graph-")

    graph_id2 = _post_graph(
        public_api_client,
        graph_definition,
        params={"skip_validation": "true"},
    )
    assert graph_id2["id"] == graph_id1["id"]

    name_alias = "test_skip_validation_graph"
    graph_id3 = _post_graph(
        public_api_client,
        graph_definition,
        params={
            "name_alias": name_alias,
            "skip_validation": "true"
        },
    )
    assert graph_id3["id"] == graph_id1["id"]

    graph_resource = _get_graph(public_api_client, name_alias)
    assert graph_resource["id"] == graph_id1["id"]
    assert graph_resource["name_alias"] == name_alias
    # Only the first (skip_validation=false) request validates, and it
    # validates every table so all failures are reported together.
    assert validator.await_count == len(graph_definition["tables"])
