# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rest.routers import api_keys_v2
from statedb.db.api_keys import APIKeyRepository


def _set_request_user(monkeypatch: pytest.MonkeyPatch, user_id: str):
    monkeypatch.setattr(
        api_keys_v2,
        "get_user_authorization_from_request",
        lambda _request: SimpleNamespace(user_id=user_id, ),
    )


def _create_admin_key(client: TestClient, name: str, type: str = "admin"):
    return client.post(
        "/v0/api-keys/admin",
        json={
            "name": name,
            "type": type
        },
    )


def _create_user_key(client: TestClient, name: str):
    return client.post("/v0/api-keys", json={"name": name})


def test_admin_list_requires_type(client: TestClient):
    res = client.get("/v0/api-keys/admin")
    assert res.status_code == 422


def test_malformed_key_matches_headerless_authority_when_auth_disabled(
    unauthenticated_client: TestClient, ):
    # Auth-disabled mode has no identity boundary: every request is the one
    # anonymous administrator principal. A malformed key must not create a
    # header-controlled privilege differential -- it resolves to the *same*
    # `disabled_request` principal as a headerless request (S-16), so both
    # reach the admin route identically.
    headerless = unauthenticated_client.get(
        "/v0/api-keys/admin",
        params={"type": "admin"},
    )
    malformed = unauthenticated_client.get(
        "/v0/api-keys/admin",
        params={"type": "admin"},
        headers={"X-API-Key": "not-an-api-key"},
    )

    assert headerless.status_code == 200
    assert malformed.status_code == headerless.status_code


def test_absent_external_admin_owner_cannot_use_admin_route(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    driver = client.app_builder._api_key_driver
    created = asyncio.run(
        driver.create_admin_api_key(
            name="external-former-admin-route",
            type="admin",
            user_id="external|former-admin-route",
            user_name="Former Admin",
        ))

    class _MissingOwnerStore:
        async def get_user(self, _user_id: str):
            return None

    monkeypatch.setattr(driver, "_direct_connect_auth_store",
                        _MissingOwnerStore())

    response = client.get(
        "/v0/api-keys/admin",
        params={"type": "admin"},
        headers={"X-API-Key": created.value},
    )

    assert response.status_code == 403


def test_legacy_admin_api_key_info_route_is_removed(client: TestClient):
    res = client.get("/v0/admin/api_key_info")
    assert res.status_code == 404


def test_legacy_admin_api_key_create_delete_routes_are_removed(
        client: TestClient):
    create_res = client.post("/v0/admin/api_key/default")
    delete_res = client.delete("/v0/admin/api_key")

    assert create_res.status_code == 404
    assert delete_res.status_code == 404


def test_admin_list_by_tab(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _set_request_user(monkeypatch, "direct_connect|admin1")
    create_admin = _create_admin_key(client, "admin-ci", "admin")
    create_user_1 = _create_admin_key(client, "user-notebook", "user")
    create_user_2 = _create_admin_key(client, "user-pipeline", "user")

    assert create_admin.status_code == 201
    assert create_admin.json()["type"] == "admin"
    assert create_user_1.status_code == 201
    assert create_user_1.json()["type"] == "user"
    assert create_user_2.status_code == 201
    assert create_user_2.json()["type"] == "user"

    admin_res = client.get("/v0/api-keys/admin", params={"type": "admin"})
    user_res = client.get("/v0/api-keys/admin", params={"type": "user"})
    legacy_res = client.get("/v0/api-keys/admin", params={"type": "legacy"})

    assert admin_res.status_code == 200
    assert user_res.status_code == 200
    assert legacy_res.status_code == 422
    assert {row["name"] for row in admin_res.json()} == {"admin-ci"}
    assert {row["name"]
            for row in user_res.json()} == {
                "user-notebook",
                "user-pipeline",
            }


def test_phase1_contract_admin_legacy_api_key_tab_is_rejected(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _set_request_user(monkeypatch, "direct_connect|admin1")

    res = client.get("/v0/api-keys/admin", params={"type": "legacy"})

    assert res.status_code == 422


def test_admin_search_filter(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _set_request_user(monkeypatch, "direct_connect|admin1")
    _create_admin_key(client, "user-notebook", "user")
    _create_admin_key(client, "user-pipeline", "user")

    res = client.get(
        "/v0/api-keys/admin",
        params={
            "type": "user",
            "search": "pipeline"
        },
    )

    assert res.status_code == 200
    assert [row["name"] for row in res.json()] == ["user-pipeline"]


def test_admin_delete_db_key(client: TestClient,
                             monkeypatch: pytest.MonkeyPatch):
    _set_request_user(monkeypatch, "direct_connect|admin1")
    created = _create_admin_key(client, "delete-me", "user").json()

    delete_res = client.delete(f"/v0/api-keys/admin/{created['id']}")
    list_res = client.get("/v0/api-keys/admin", params={"type": "user"})

    assert delete_res.status_code == 204
    assert "delete-me" not in {row["name"] for row in list_res.json()}


@pytest.mark.parametrize(
    "case,key_id,expected_status,expected_detail",
    [
        ("missing", "missing-key", 404, None),
    ],
)
def test_admin_delete_disallowed_cases(
    client: TestClient,
    case: str,
    key_id: str,
    expected_status: int,
    expected_detail: str | None,
):
    if case != "missing":
        raise AssertionError(f"Unknown case: {case}")

    res = client.delete(f"/v0/api-keys/admin/{key_id}")
    assert res.status_code == expected_status
    if expected_detail is not None:
        assert res.json()["detail"] == expected_detail


@pytest.mark.parametrize(
    "path,name",
    [
        ("/v0/api-keys/admin", "bad:name"),
        ("/v0/api-keys/admin", "bad name"),
        ("/v0/api-keys/admin", "bad@name"),
        ("/v0/api-keys/admin", "bad\tname"),
        ("/v0/api-keys/admin", ""),
        ("/v0/api-keys", "bad:name"),
        ("/v0/api-keys", "bad name"),
        ("/v0/api-keys", "bad@name"),
        ("/v0/api-keys", "bad\tname"),
        ("/v0/api-keys", ""),
    ],
    ids=[
        "admin-colon",
        "admin-space",
        "admin-at",
        "admin-tab",
        "admin-empty",
        "user-colon",
        "user-space",
        "user-at",
        "user-tab",
        "user-empty",
    ],
)
def test_create_rejects_invalid_name_characters(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    name: str,
):
    _set_request_user(monkeypatch, "direct_connect|admin1")
    payload = {"name": name}
    if path == "/v0/api-keys/admin":
        payload["type"] = "admin"

    res = client.post(path, json=payload)

    assert res.status_code == 400
    assert res.json()["detail"] == "API key name must match [A-Za-z0-9_-]+"


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/v0/api-keys/admin", {
            "name": "my_admin_key-01",
            "type": "admin"
        }),
        ("/v0/api-keys", {
            "name": "my_user_key-01"
        }),
    ],
    ids=["admin", "user"],
)
def test_create_accepts_valid_name(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    payload: dict,
    event_loop,
):
    _set_request_user(monkeypatch, "direct_connect|admin1")

    res = client.post(path, json=payload)

    assert res.status_code == 201, res.text
    assert res.json()["name"] == payload["name"]

    async def _read_state_metadata():
        repo = APIKeyRepository()
        state_services = client.app_builder._state_services
        assert state_services is not None
        state_db = state_services.state_db
        async with state_db.read_tx() as tx:
            return await repo.get_metadata(tx, res.json()["id"])

    metadata = event_loop.run_until_complete(_read_state_metadata())
    assert metadata is not None
    assert metadata.owner_user_hash == "c14f9333"


@pytest.mark.parametrize(
    "case,expected_names",
    [
        ("owner-scope", {"user1-key"}),
        ("type-filtering", {"user-visible-key"}),
    ],
    ids=["owner-scope", "type-filtering"],
)
def test_user_list_visibility(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_names: set[str],
):
    if case == "owner-scope":
        _set_request_user(monkeypatch, "direct_connect|user1")
        _create_user_key(client, "user1-key")
        _set_request_user(monkeypatch, "direct_connect|user2")
        _create_user_key(client, "user2-key")
        _set_request_user(monkeypatch, "direct_connect|user1")
    elif case == "type-filtering":
        _set_request_user(monkeypatch, "direct_connect|user1")
        _create_user_key(client, "user-visible-key")
        _create_admin_key(client, "user-hidden-admin-key", "admin")
    else:
        raise AssertionError(f"Unknown case: {case}")

    res = client.get("/v0/api-keys")

    assert res.status_code == 200
    assert {row["name"] for row in res.json()} == expected_names


def test_user_delete_owned_key_succeeds(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _set_request_user(monkeypatch, "direct_connect|user1")
    key_id = _create_user_key(client, "owned-delete").json()["id"]

    res = client.delete(f"/v0/api-keys/{key_id}")
    list_res = client.get("/v0/api-keys")

    assert res.status_code == 204
    assert list_res.json() == []


@pytest.mark.parametrize(
    "case",
    [
        "owned-admin",
        "non-owned-user",
    ],
)
def test_user_delete_disallowed_cases_return_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
):
    if case == "owned-admin":
        _set_request_user(monkeypatch, "direct_connect|user1")
        key_id = _create_admin_key(client, "owned-admin-key",
                                   "admin").json()["id"]
    elif case == "non-owned-user":
        _set_request_user(monkeypatch, "direct_connect|user2")
        key_id = _create_user_key(client, "user2-owned").json()["id"]
    else:
        raise AssertionError(f"Unknown case: {case}")

    _set_request_user(monkeypatch, "direct_connect|user1")
    res = client.delete(f"/v0/api-keys/{key_id}")

    assert res.status_code == 404
