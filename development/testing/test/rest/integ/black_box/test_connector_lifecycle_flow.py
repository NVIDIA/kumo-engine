# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from test.rest.integ.black_box.project_upload_helpers import (
    project_upload_connector_id,
)

from fastapi.testclient import TestClient

_DEFAULT_PROJECT_ID = "default"


def _response_json(response):
    assert response.status_code == HTTPStatus.OK, response.text
    return response.json()


def test_project_upload_connector_is_idempotent_hidden_and_read_only(
    client: TestClient, ):
    connector_id = project_upload_connector_id(client, _DEFAULT_PROJECT_ID)
    assert project_upload_connector_id(client,
                                       _DEFAULT_PROJECT_ID) == (connector_id)

    listed_connectors = _response_json(
        client.get(f"/v0/projects/{_DEFAULT_PROJECT_ID}/connectors/"))
    assert connector_id not in {
        connector["id"]
        for connector in listed_connectors
    }

    fetched = client.get(
        f"/v0/projects/{_DEFAULT_PROJECT_ID}/connectors/{connector_id}/")
    assert fetched.status_code == HTTPStatus.NOT_FOUND

    deleted = client.delete(
        f"/v0/projects/{_DEFAULT_PROJECT_ID}/connectors/{connector_id}")
    assert deleted.status_code == HTTPStatus.NOT_FOUND

    assert project_upload_connector_id(client,
                                       _DEFAULT_PROJECT_ID) == (connector_id)
