# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from typing import Optional

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from rest.config.authentication import AuthenticationSettings
from rest.deps.authentication import (
    Authentication,
    Permissions,
    UserAuthorization,
)
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from rest.routers.admin import AdminResource
from rest.utils.driver_common import get_current_timestamps
from statedb.config.common import UserID
from statedb.config.direct_connect_auth import DirectConnectAuthConfigRecord
from statedb.config.timestamp import now_utc
from statedb.db import StateDB
from statedb.db.direct_connect_auth import DirectConnectAuthService


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


async def _build_store(sqlite_path: str) -> DirectConnectAuthStore:
    state_db = await StateDB.open_sqlite(sqlite_path)
    store = DirectConnectAuthStore(DirectConnectAuthService(state_db))
    await store.set_config(
        DirectConnectAuthConfigRecord(
            timestamps=get_current_timestamps(),
            issuer="https://idp.example.com",
            client_id="kumo",
            encrypted_client_secret="secret",
            groups_claim="groups",
            admin_emails=["alice@example.com"],
            enabled=True,
        ))
    await store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="alice-subject",
        email="alice@example.com",
        display_name="Alice Example",
        refresh_token="alice-refresh-token",
        email_verified=True,
    )
    await store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="bob-subject",
        email="bob@example.com",
        display_name="Bob Example",
        refresh_token="bob-refresh-token",
    )
    return store


@pytest.mark.parametrize(
    ("email_verified", "expected_role"),
    [
        (True, "admin"),
        (False, "user"),
        (None, "user"),
        ("true", "user"),
        ("definitely", "user"),
        (1, "user"),
    ],
    ids=[
        "true", "false", "missing", "string-true", "hostile-string",
        "truthy-number"
    ],
)
@pytest.mark.asyncio
async def test_oidc_bootstrap_admin_requires_exact_verified_boolean(
        tmp_path, email_verified, expected_role):
    async def create_first_user():
        state_db = await StateDB.open_sqlite(
            str(tmp_path / "bootstrap-verification.db"))
        store = DirectConnectAuthStore(DirectConnectAuthService(state_db))
        await store.set_config(
            DirectConnectAuthConfigRecord(
                timestamps=get_current_timestamps(),
                issuer="https://idp.example.com",
                client_id="kumo",
                encrypted_client_secret="secret",
                groups_claim="groups",
                admin_emails=["alice@example.com"],
                enabled=True,
            ))
        _, user = await store.upsert_user_from_idp(
            issuer="https://idp.example.com",
            subject="first-subject",
            email="alice@example.com",
            display_name="Alice Example",
            refresh_token="refresh-token",
            email_verified=email_verified,
        )
        await state_db.close()
        return user

    user = await create_first_user()
    assert user.role == expected_role


def _build_store_for_test(event_loop, tmp_path) -> DirectConnectAuthStore:
    return event_loop.run_until_complete(
        _build_store(str(tmp_path / "direct-connect-auth.db")))


def _user_auth(is_admin: bool = True) -> UserAuthorization:
    permissions = [
        Permissions.READ.compose(),
        Permissions.WRITE.compose(),
    ]
    if is_admin:
        permissions.append(Permissions.ADMIN.compose())
    return UserAuthorization(
        user_id=UserID("direct_connect|admin"),
        access_token="direct-connect-session",
        permissions=permissions,
    )


def _app(
    store: Optional[DirectConnectAuthStore],
    *,
    is_admin=True,
    disable=False,
    direct_connect_login_method="oidc",
) -> FastAPI:
    auth = Authentication(
        AuthenticationSettings(
            direct_connect_login_method=direct_connect_login_method,
            disable=disable,
        ),
        is_unit_test=True,
    )
    app = FastAPI()

    @app.middleware("http")
    async def set_user_authorization(request: Request, call_next):
        request.state.user_authorization = _user_auth(is_admin=is_admin)
        return await call_next(request)

    app.include_router(
        AdminResource(
            auth,
            direct_connect_auth_store=store,
            direct_connect_login_method=direct_connect_login_method,
        ).router)
    return app


def test_list_direct_connect_users_returns_users(event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)

    with TestClient(_app(store)) as client:
        response = client.get("/v0/admin/users")

    assert response.status_code == 200
    payload = response.json()
    assert [user["email"] for user in payload] == [
        "alice@example.com",
        "bob@example.com",
    ]
    assert payload[0]["role"] == "admin"
    assert payload[1]["role"] == "user"


def test_update_direct_connect_user_role_and_status(event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)
    bob_user_id = event_loop.run_until_complete(
        store.get_user_by_email("bob@example.com"))[0]

    with TestClient(_app(store)) as client:
        response = client.patch(
            f"/v0/admin/users/{bob_user_id}",
            json={
                "role": "admin",
                "disabled": True
            },
        )

    assert response.status_code == 200
    assert response.json()["role"] == "admin"
    assert response.json()["disabled"] is True

    updated = event_loop.run_until_complete(store.get_user(bob_user_id))
    assert updated is not None
    assert updated.role == "admin"
    assert updated.disabled is True


def test_update_direct_connect_user_role_change_invalidates_sessions(
        event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)
    bob_user_id = event_loop.run_until_complete(
        store.get_user_by_email("bob@example.com"))[0]
    expires_time = now_utc() + timedelta(hours=8)

    def create_bob_session() -> str:
        return event_loop.run_until_complete(
            store.create_session(
                user_id=bob_user_id,
                expires_time=expires_time,
            ))

    # A role change ends the user's existing sessions so they must
    # re-authenticate under the new privileges.
    session_id = create_bob_session()
    with TestClient(_app(store)) as client:
        response = client.patch(
            f"/v0/admin/users/{bob_user_id}",
            json={"role": "admin"},
        )
    assert response.status_code == 200
    assert event_loop.run_until_complete(store.get_session(session_id)) is None

    # A no-op update (role unchanged) keeps the session alive.
    session_id = create_bob_session()
    with TestClient(_app(store)) as client:
        response = client.patch(
            f"/v0/admin/users/{bob_user_id}",
            json={"role": "admin"},
        )
    assert response.status_code == 200
    assert event_loop.run_until_complete(
        store.get_session(session_id)) is not None

    # Disabling the user also ends their sessions.
    with TestClient(_app(store)) as client:
        response = client.patch(
            f"/v0/admin/users/{bob_user_id}",
            json={"disabled": True},
        )
    assert response.status_code == 200
    assert event_loop.run_until_complete(store.get_session(session_id)) is None


def test_direct_connect_users_are_admin_only(event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)

    with TestClient(_app(store, is_admin=False)) as client:
        response = client.get("/v0/admin/users")

    assert response.status_code == 403


def test_get_organization_members_maps_roles(event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)

    with TestClient(_app(store)) as client:
        response = client.get("/v0/admin/members")

    assert response.status_code == 200
    payload = response.json()
    assert [(member["email"], member["roles"]) for member in payload] == [
        ("alice@example.com", ["Admin"]),
        ("bob@example.com", ["User"]),
    ]
    assert payload[0]["name"] == "Alice Example"
    users = event_loop.run_until_complete(store.list_users())
    assert [member["uid"]
            for member in payload] == [user.user_id for user in users]


def test_get_organization_members_is_admin_only(event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)

    with TestClient(_app(store, is_admin=False)) as client:
        response = client.get("/v0/admin/members")

    assert response.status_code == 403


def test_get_organization_members_empty_without_store():
    with TestClient(_app(None)) as client:
        response = client.get("/v0/admin/members")

    assert response.status_code == 200
    assert response.json() == []


def test_user_routes_disabled_without_direct_connect_store():
    with TestClient(_app(None)) as client:
        list_response = client.get("/v0/admin/users")
        update_response = client.patch(
            "/v0/admin/users/direct_connect%7Cmissing",
            json={"disabled": True},
        )

    assert list_response.status_code == 200
    assert list_response.json() == []
    assert update_response.status_code == 404


def test_create_direct_connect_user(event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)

    with TestClient(_app(store)) as client:
        response = client.post(
            "/v0/admin/users",
            json={
                "email": "LOCAL@example.com",
                "display_name": "Local User",
                "role": "user",
                "disabled": False,
            },
        )

    assert response.status_code == 200
    assert response.json()["email"] == "local@example.com"
    assert response.json()["display_name"] == "Local User"
    assert response.json()["role"] == "user"


def test_create_direct_connect_user_with_password_in_password_mode(
        event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)

    with TestClient(_app(store,
                         direct_connect_login_method="password")) as client:
        response = client.post(
            "/v0/admin/users",
            json={
                "email": "LOCAL@example.com",
                "display_name": "Local User",
                "role": "user",
                "disabled": False,
                "password": "SeedPass12345",
            },
        )

    assert response.status_code == 200
    assert response.json()["email"] == "local@example.com"
    status, reset_token, login_user_id, _record = event_loop.run_until_complete(
        store.password_login(
            email="local@example.com",
            password="SeedPass12345",
            reset_expires_time=now_utc() + timedelta(minutes=10),
        ))
    assert status == "password_reset_required"
    assert reset_token
    assert login_user_id == response.json()["user_id"]


def test_create_direct_connect_user_requires_password_in_password_mode(
        event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)

    with TestClient(_app(store,
                         direct_connect_login_method="password")) as client:
        response = client.post(
            "/v0/admin/users",
            json={
                "email": "local@example.com",
                "display_name": "Local User",
                "role": "user",
                "disabled": False,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Temporary password is required"
    assert event_loop.run_until_complete(
        store.get_user_by_email("local@example.com")) is None


def test_create_direct_connect_user_rejects_password_in_oidc_mode(
        event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)

    with TestClient(_app(store)) as client:
        response = client.post(
            "/v0/admin/users",
            json={
                "email": "local@example.com",
                "display_name": "Local User",
                "role": "user",
                "disabled": False,
                "password": "SeedPass12345",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Password is only supported when Direct Connect password login is enabled"
    )
    assert event_loop.run_until_complete(
        store.get_user_by_email("local@example.com")) is None


def test_create_direct_connect_user_rejects_weak_password_without_creating_user(
        event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)

    with TestClient(_app(store,
                         direct_connect_login_method="password")) as client:
        response = client.post(
            "/v0/admin/users",
            json={
                "email": "local@example.com",
                "display_name": "Local User",
                "role": "user",
                "disabled": False,
                "password": "short1",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Password must be at least 12 characters")
    assert event_loop.run_until_complete(
        store.get_user_by_email("local@example.com")) is None


def test_set_direct_connect_user_temporary_password_rejects_weak_password(
        event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)
    user = event_loop.run_until_complete(
        store.create_local_user(email="local@example.com"))

    with TestClient(_app(store)) as client:
        response = client.post(
            f"/v0/admin/users/{user.user_id}/password",
            json={"password": "short1"},
        )

    assert response.status_code == 400
    assert (
        response.json()["detail"] == "Password must be at least 12 characters")


def test_set_direct_connect_user_temporary_password(event_loop, tmp_path):
    store = _build_store_for_test(event_loop, tmp_path)
    user = event_loop.run_until_complete(
        store.create_local_user(email="local@example.com"))

    with TestClient(_app(store)) as client:
        response = client.post(
            f"/v0/admin/users/{user.user_id}/password",
            json={"password": "SeedPass12345"},
        )

    assert response.status_code == 200
    status, reset_token, login_user_id, _record = event_loop.run_until_complete(
        store.password_login(
            email="local@example.com",
            password="SeedPass12345",
            reset_expires_time=now_utc() + timedelta(minutes=10),
        ))
    assert status == "password_reset_required"
    assert reset_token
    assert login_user_id == user.user_id


def test_admin_endpoints_degrade_when_store_unconfigured():
    # Auth-disabled constructs the store but never configures its auth service.
    # Direct Connect user management must degrade (empty reads / "not enabled"
    # writes) rather than raising "auth service is not configured".
    store = DirectConnectAuthStore()  # unconfigured -> no auth service
    with TestClient(_app(store, disable=True)) as client:
        users = client.get("/v0/admin/users")
        members = client.get("/v0/admin/members")
        create = client.post("/v0/admin/users",
                             json={"email": "new@example.com"})
    assert users.status_code == 200 and users.json() == []
    assert members.status_code == 200 and members.json() == []
    assert create.status_code == 404
