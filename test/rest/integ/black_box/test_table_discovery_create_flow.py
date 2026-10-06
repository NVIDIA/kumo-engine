# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import uuid
from http import HTTPStatus
from pathlib import Path
from test.rest.integ.black_box.project_upload_helpers import upload_project_csv

from fastapi.testclient import TestClient

_SMALL_SYNTHETIC_DIR = (Path(__file__).resolve().parents[3] / "csv_data" /
                        "small_synthetic")
_DEFAULT_PROJECT_ID = "default"


def _response_json(response):
    assert response.status_code == HTTPStatus.OK, response.text
    return response.json()


def _table_definition(connector_id: str, source_table_name: str) -> dict:
    return {
        "cols": [
            {
                "name": "A",
                "stype": "ID",
                "dtype": "int",
            },
            {
                "name": "feature",
                "stype": "categorical",
                "dtype": "int",
            },
        ],
        "source_table": {
            "data_source_type": "S3",
            "connector_id": connector_id,
            "source_table_name": source_table_name,
            "file_type": "CSV",
        },
        "pkey":
        "A",
    }


def _column_by_name(columns: list[dict]) -> dict[str, dict]:
    return {column["name"]: column for column in columns}


def _listed_table_names(tables: list[dict]) -> set[str]:
    names: set[str] = set()
    for table in tables:
        if table.get("name_alias"):
            names.add(table["name_alias"])
        if table.get("name"):
            names.add(table["name"])
    return names


def test_local_table_discovery_create_pre_refresh_is_api_visible(
    client: TestClient, ):
    suffix = uuid.uuid4().hex[:8]
    table_alias = f"phase1b_table_{suffix}"

    connector_id = upload_project_csv(client, _DEFAULT_PROJECT_ID,
                                      _SMALL_SYNTHETIC_DIR / "A.csv")
    assert upload_project_csv(
        client,
        _DEFAULT_PROJECT_ID,
        _SMALL_SYNTHETIC_DIR / "empty.csv",
    ) == connector_id

    source_tables = _response_json(
        client.get(
            f"/v0/projects/{_DEFAULT_PROJECT_ID}/connectors/{connector_id}/source-tables",
        ))
    source_table_names = set(source_tables["table_names"])
    assert {"A", "empty"}.issubset(source_table_names)

    samples = _response_json(
        client.get(
            f"/v0/projects/{_DEFAULT_PROJECT_ID}/connectors/{connector_id}/sample",
            params={
                "table_name": "A",
                "sample_rows": 2,
            },
        ))
    sample_columns = _column_by_name(samples["cols"])
    assert sample_columns["A"]["dtype"] == "int64"
    assert sample_columns["feature"]["dtype"] == "int64"
    sample_rows = json.loads(samples["sample_rows"])
    assert len(sample_rows["data"]) >= 2
    assert sample_rows["data"][0]["A"] == "31"
    assert sample_rows["data"][0]["feature"] == "1"

    table_definition = _table_definition(connector_id, "A")
    validation = _response_json(
        client.post(
            "/v1/tables/validate",
            params={"project_id": _DEFAULT_PROJECT_ID},
            json=table_definition,
        ))
    assert validation["errors"] == []

    created_table = _response_json(
        client.post(
            "/v1/tables/",
            params={
                "project_id": _DEFAULT_PROJECT_ID,
                "name_alias": table_alias,
            },
            json=table_definition,
        ))
    table_id = created_table["id"]
    assert table_id
    table_fpid = table_id.rsplit("-", maxsplit=1)[-1]

    listed_tables = _response_json(
        client.get("/v1/tables/", params={"project_id": _DEFAULT_PROJECT_ID}))
    assert table_alias in _listed_table_names(listed_tables)

    fetched_by_alias = _response_json(
        client.get(
            f"/v1/tables/{table_alias}",
            params={"project_id": _DEFAULT_PROJECT_ID},
        ))
    assert fetched_by_alias["id"] == table_fpid
    assert fetched_by_alias["name_alias"] == table_alias
    assert fetched_by_alias["table"]["pkey"] == "A"
    assert fetched_by_alias["table"]["source_table"]["connector_id"] == (
        connector_id)

    fetched_by_id = _response_json(
        client.get(
            f"/v1/tables/{table_id}",
            params={"project_id": _DEFAULT_PROJECT_ID},
        ))
    assert fetched_by_id["id"] == table_fpid
    assert fetched_by_id["name_alias"] == table_alias
    assert fetched_by_id["table"]["source_table"]["connector_id"] == (
        connector_id)

    connector_tables = _response_json(
        client.get(
            f"/v1/tables/{connector_id}/list_tables",
            params={
                "project_id": _DEFAULT_PROJECT_ID,
                "fetch_column_stats": "false",
            },
        ))
    assert table_fpid in {table["id"] for table in connector_tables}

    delete_alias = _response_json(
        client.delete(
            f"/v1/tables/{table_alias}",
            params={"project_id": _DEFAULT_PROJECT_ID},
        ))
    assert delete_alias["id"]

    get_after_delete = client.get(
        f"/v1/tables/{table_alias}",
        params={"project_id": _DEFAULT_PROJECT_ID},
    )
    assert get_after_delete.status_code == HTTPStatus.NOT_FOUND

    listed_after_delete = _response_json(
        client.get("/v1/tables/", params={"project_id": _DEFAULT_PROJECT_ID}))
    assert table_alias not in _listed_table_names(listed_after_delete)
