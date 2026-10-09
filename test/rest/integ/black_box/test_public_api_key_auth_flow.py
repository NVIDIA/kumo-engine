# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from http import HTTPStatus

from fastapi.testclient import TestClient


def _assert_auth_error(response, status_code: HTTPStatus, detail: str) -> None:
    assert response.status_code == status_code, response.text
    assert detail in response.text


def _assert_authentication_success(response) -> None:
    assert response.status_code == HTTPStatus.NO_CONTENT, response.text
    assert response.content == b""


def _assert_documentation_success(response, expected_text: str) -> None:
    assert response.status_code == HTTPStatus.OK, response.text
    assert expected_text in response.text


def _assert_created_api_key(
    api_key: dict,
    *,
    expected_name: str,
    expected_type: str,
) -> None:
    assert api_key["id"].startswith("k-")
    assert api_key["name"] == expected_name
    assert api_key["type"] == expected_type
    assert api_key["value"]
    assert api_key["value_preview"]
    assert api_key["value_preview"] != api_key["value"]
    assert api_key["created_time"]


def _api_key_names(rows: list[dict]) -> set[str]:
    return {row["name"] for row in rows}


def _clear_api_key_headers(client: TestClient) -> None:
    for header_name in list(client.headers):
        if header_name.lower() == "x-api-key":
            client.headers.pop(header_name, None)


def test_public_documentation_links_are_proxy_prefix_safe(
    public_api_client: TestClient,
):
    response = public_api_client.get("/", follow_redirects=False)
    assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT
    assert response.headers["location"] == "redoc"

    for path in ("/docs", "/redoc"):
        response = public_api_client.get(path)
        assert response.status_code == HTTPStatus.OK
        assert "./openapi.json" in response.text


def test_public_api_key_request_auth_is_api_visible(
    client: TestClient,
    public_api_client: TestClient,
):
    _clear_api_key_headers(public_api_client)

    protected_route = "/v1/authentication"
    authentication_response = public_api_client.get(protected_route)
    assert authentication_response.status_code == HTTPStatus.NO_CONTENT, (
        authentication_response.text)
    assert authentication_response.content == b""
    # Public API keys are header-only. The legacy query parameter is ignored,
    # so this remains an ordinary disabled-auth request.
    _assert_authentication_success(
        public_api_client.get(protected_route,
                              params={"api_key": "invalid_key"}))

    _assert_documentation_success(
        public_api_client.get("/openapi.json"),
        "openapi",
    )
    documentation_routes = {
        "/docs": "swagger-ui",
        "/redoc": "ReDoc",
    }
    for path, expected_text in documentation_routes.items():
        _assert_documentation_success(
            public_api_client.get(path),
            expected_text,
        )
    key_name = f"phase1b_api_key_auth_{uuid.uuid4().hex[:8]}"
    create_response = client.post("/v0/api-keys", json={"name": key_name})
    assert create_response.status_code == HTTPStatus.CREATED, create_response.text
    created_key = create_response.json()
    assert created_key["name"] == key_name
    assert created_key["id"]
    assert created_key["value"]

    list_response = client.get("/v0/api-keys")
    assert list_response.status_code == HTTPStatus.OK, list_response.text
    assert key_name in {api_key["name"] for api_key in list_response.json()}

    created_key_value = created_key["value"]
    _assert_authentication_success(
        public_api_client.get(
            "/v1/authentication",
            headers={"X-API-Key": created_key_value},
        ))
    _assert_documentation_success(
        public_api_client.get("/openapi.json",
                              headers={"X-API-Key": created_key_value}),
        "openapi",
    )
    for path, expected_text in documentation_routes.items():
        _assert_documentation_success(
            public_api_client.get(path,
                                  params={"api_key": created_key_value}),
            expected_text,
        )
        _assert_documentation_success(
            public_api_client.get(
                path, headers={"X-API-Key": created_key_value}),
            expected_text,
        )
    delete_response = client.delete(f"/v0/api-keys/{created_key['id']}")
    assert delete_response.status_code == HTTPStatus.NO_CONTENT, (
        delete_response.text)

    final_list_response = client.get("/v0/api-keys")
    assert final_list_response.status_code == HTTPStatus.OK, (
        final_list_response.text)
    assert key_name not in {
        api_key["name"]
        for api_key in final_list_response.json()
    }

    # A well-formed header key must still be verified in disabled-auth mode.
    # Once deleted, it is an explicit failed credential rather than an
    # ordinary disabled-auth request.
    _assert_auth_error(
        public_api_client.get(
            "/v1/authentication",
            headers={"X-API-Key": created_key_value},
        ),
        HTTPStatus.UNAUTHORIZED,
        "Failed. API Key is not valid",
    )


def test_api_key_admin_management_search_is_api_visible(client: TestClient):
    suffix = uuid.uuid4().hex[:8]
    admin_key_name = f"phase1b_admin_key_{suffix}"
    user_key_name = f"phase1b_user_key_{suffix}"
    other_user_key_name = f"phase1b_other_user_key_{suffix}"

    missing_type_response = client.get("/v0/api-keys/admin")
    assert missing_type_response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    admin_create_response = client.post(
        "/v0/api-keys/admin",
        json={
            "name": admin_key_name,
            "type": "admin"
        },
    )
    assert admin_create_response.status_code == HTTPStatus.CREATED, (
        admin_create_response.text)
    admin_key = admin_create_response.json()
    _assert_created_api_key(
        admin_key,
        expected_name=admin_key_name,
        expected_type="admin",
    )

    user_create_response = client.post(
        "/v0/api-keys/admin",
        json={
            "name": user_key_name,
            "type": "user"
        },
    )
    assert user_create_response.status_code == HTTPStatus.CREATED, (
        user_create_response.text)
    user_key = user_create_response.json()
    _assert_created_api_key(
        user_key,
        expected_name=user_key_name,
        expected_type="user",
    )

    other_user_create_response = client.post(
        "/v0/api-keys/admin",
        json={
            "name": other_user_key_name,
            "type": "user"
        },
    )
    assert other_user_create_response.status_code == HTTPStatus.CREATED, (
        other_user_create_response.text)

    admin_list_response = client.get(
        "/v0/api-keys/admin",
        params={"type": "admin"},
    )
    assert admin_list_response.status_code == HTTPStatus.OK, (
        admin_list_response.text)
    admin_rows = admin_list_response.json()
    assert admin_key_name in _api_key_names(admin_rows)
    assert user_key_name not in _api_key_names(admin_rows)
    assert {row["type"]
            for row in admin_rows
            if row["name"] == admin_key_name} == {"admin"}

    user_list_response = client.get(
        "/v0/api-keys/admin",
        params={"type": "user"},
    )
    assert user_list_response.status_code == HTTPStatus.OK, (
        user_list_response.text)
    user_rows = user_list_response.json()
    assert user_key_name in _api_key_names(user_rows)
    assert other_user_key_name in _api_key_names(user_rows)
    assert admin_key_name not in _api_key_names(user_rows)
    assert {
        row["type"]
        for row in user_rows if row["name"] in {
            user_key_name,
            other_user_key_name,
        }
    } == {"user"}
    assert all("value" not in row for row in user_rows)
    assert all(row["value_preview"] for row in user_rows)

    search_response = client.get(
        "/v0/api-keys/admin",
        params={
            "type": "user",
            "search": user_key_name,
        },
    )
    assert search_response.status_code == HTTPStatus.OK, search_response.text
    assert _api_key_names(search_response.json()) == {user_key_name}

    delete_response = client.delete(f"/v0/api-keys/admin/{user_key['id']}")
    assert delete_response.status_code == HTTPStatus.NO_CONTENT, (
        delete_response.text)

    final_user_list_response = client.get(
        "/v0/api-keys/admin",
        params={"type": "user"},
    )
    assert final_user_list_response.status_code == HTTPStatus.OK, (
        final_user_list_response.text)
    assert user_key_name not in _api_key_names(final_user_list_response.json())
    assert other_user_key_name in _api_key_names(
        final_user_list_response.json())

    for path, payload in {
            "/v0/api-keys/admin": {
                "name": "bad name",
                "type": "admin"
            },
            "/v0/api-keys": {
                "name": "bad name"
            },
    }.items():
        invalid_name_response = client.post(path, json=payload)
        assert invalid_name_response.status_code == HTTPStatus.BAD_REQUEST
        assert invalid_name_response.json()["detail"] == (
            "API key name must match [A-Za-z0-9_-]+")
