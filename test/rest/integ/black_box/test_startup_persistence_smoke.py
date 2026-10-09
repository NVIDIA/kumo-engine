# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi.testclient import TestClient


def test_startup_and_api_key_persistence_smoke(client: TestClient):
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    key_name = "phase1a-smoke-key"
    create_response = client.post("/v0/api-keys", json={"name": key_name})
    assert create_response.status_code == 201
    created = create_response.json()
    assert created["name"] == key_name
    assert created["value"]

    list_response = client.get("/v0/api-keys")
    assert list_response.status_code == 200
    assert key_name in {row["name"] for row in list_response.json()}

    delete_response = client.delete(f"/v0/api-keys/{created['id']}")
    assert delete_response.status_code == 204

    final_list_response = client.get("/v0/api-keys")
    assert final_list_response.status_code == 200
    assert key_name not in {row["name"] for row in final_list_response.json()}
