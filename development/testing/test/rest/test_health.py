# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from rest.app import _state_db_health_response
from statedb.db import SCHEMA_VERSION, StateDBSchemaStatus, schema_identity


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


def test_internal_rest_health(client: TestClient):
    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


@pytest.mark.asyncio
async def test_internal_rest_health_reports_state_db_schema_failure():
    status = StateDBSchemaStatus(
        backend="sqlite",
        expected_schema_version=SCHEMA_VERSION,
        stored_schema_version=0,
        expected_schema_hash=schema_identity("sqlite"),
        stored_schema_hash="old-schema",
        expected_table_count=3,
        existing_state_table_count=2,
        missing_table_names=("projects",),
        ok=False,
        error_type=None,
        error_message=None,
    )
    get_schema_status = AsyncMock(return_value=status)
    state_services = SimpleNamespace(
        status=SimpleNamespace(get_schema_status=get_schema_status),
    )

    response = await _state_db_health_response(state_services)

    assert response.status_code == 503
    assert json.loads(response.body) == {
        "status": "unhealthy",
        "state_db": {
            "backend": "sqlite",
            "expected_schema_version": SCHEMA_VERSION,
            "stored_schema_version": 0,
            "expected_schema_hash": schema_identity("sqlite"),
            "stored_schema_hash": "old-schema",
            "expected_table_count": 3,
            "existing_state_table_count": 2,
            "missing_table_names": ["projects"],
            "ok": False,
            "error_type": None,
            "error_message": None,
        },
    }
    get_schema_status.assert_awaited_once_with()


def test_public_rest_health_without_api_key(public_api_client: TestClient):
    public_api_client.headers.pop('X-API-KEY', None)

    response = public_api_client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
