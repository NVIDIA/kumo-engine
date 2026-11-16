# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rest.routers import api_keys_v2
from statedb.config.common import APIKeyID
from statedb.db.api_keys import APIKeyRepository
from statedb.db.services import APIKeyService


def _set_request_user(monkeypatch: pytest.MonkeyPatch, user_id: str):
    monkeypatch.setattr(
        api_keys_v2,
        "get_user_authorization_from_request",
        lambda _request: SimpleNamespace(user_id=user_id, ),
    )


async def _get_secret_record(client: TestClient, api_key_id: str):
    repo = APIKeyRepository()
    state_services = client.app_builder._state_services
    assert state_services is not None
    state_db = state_services.state_db
    async with state_db.read_tx() as tx:
        return await repo.get_secret(tx, APIKeyID(api_key_id))


async def _list_metadata(client: TestClient):
    state_services = client.app_builder._state_services
    assert state_services is not None
    return await state_services.api_keys.list_metadata()


@pytest.mark.asyncio
async def test_admin_create_persists_secure_record_and_returns_value(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _set_request_user(monkeypatch, "direct_connect|admin1")

    res = client.post(
        "/v0/api-keys/admin",
        json={
            "name": "new-admin",
            "type": "admin"
        },
    )
    assert res.status_code == 201

    payload = res.json()
    assert payload["type"] == "admin"
    assert payload["value"]
    assert len(payload["value"].split(":")) == 3

    secret_record = await _get_secret_record(client, payload["id"])
    assert secret_record is not None
    assert secret_record.hash_version == "v1"
    assert secret_record.hash_algorithm == "scrypt"
    assert secret_record.hash_params["n"] == 2**14
    assert secret_record.hash_params["r"] == 8
    assert secret_record.hash_params["p"] == 1
    assert secret_record.hash_params["dklen"] == 32
    assert secret_record.hash_b64 != payload["value"]
    assert secret_record.salt_b64 != payload["value"]
    assert not hasattr(secret_record, "name")
    assert not hasattr(secret_record, "type")
    assert not hasattr(secret_record, "owner_user_id")

    admin_list = client.get(
        "/v0/api-keys/admin",
        params={"type": "admin"},
    )
    assert admin_list.status_code == 200
    assert "new-admin" in {row["name"] for row in admin_list.json()}


@pytest.mark.asyncio
async def test_admin_create_legacy_rejected_and_not_persisted(
    client: TestClient, ):
    res = client.post(
        "/v0/api-keys/admin",
        json={
            "name": "legacy-disabled",
            "type": "legacy"
        },
    )

    assert res.status_code == 422

    assert await _list_metadata(client) == []
    assert await _get_secret_record(client, "missing") is None


@pytest.mark.asyncio
async def test_admin_create_db_failure_has_no_partial_side_effect(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _set_request_user(monkeypatch, "direct_connect|admin1")

    def _raise_write_error(*args, **kwargs):
        raise RuntimeError("db write failed")

    monkeypatch.setattr(APIKeyService, "create_unique", _raise_write_error)

    pre_list = client.get(
        "/v0/api-keys/admin",
        params={"type": "admin"},
    )
    assert pre_list.json() == []

    res = client.post(
        "/v0/api-keys/admin",
        json={
            "name": "should-not-exist",
            "type": "admin"
        },
    )
    assert res.status_code == 500
    assert res.json()["detail"] == "Failed to securely persist API key"

    post_list = client.get(
        "/v0/api-keys/admin",
        params={"type": "admin"},
    )
    assert post_list.json() == []

    assert await _list_metadata(client) == []
