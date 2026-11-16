# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import timedelta
from http.cookies import Morsel, SimpleCookie
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from rest.config.encryption import encrypt_secret_string, hash_idp_group_id
from rest.deps.authentication import (
    AUTHORIZATION_DIRECT_CONNECT,
    Permissions,
    UserAuthorization,
    authorization_middleware,
    get_user_authorization_from_request,
)
from rest.deps.direct_connect_oidc_client import DirectConnectOIDCTokens
from rest.deps.group_membership import ResolvedGroupMembership
from rest.driver import direct_connect_auth_storage as storage
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from rest.routers import direct_connect_auth as routes
from rest.routers.direct_connect_auth import (
    DIRECT_CONNECT_LOGIN_COOKIE,
    DIRECT_CONNECT_SESSION_COOKIE,
    DIRECT_CONNECT_SESSION_SECONDS,
    DirectConnectAuthResource,
    _safe_return_to,
)
from rest.utils.driver_common import get_current_timestamps
from statedb.config.common import UserID
from statedb.config.direct_connect_auth import DirectConnectAuthConfigRecord
from statedb.config.timestamp import now_utc
from statedb.db import StateDB
from statedb.db.direct_connect_auth import DirectConnectAuthService
from statedb.rest.group import ExternalIdentityProviderID


def _direct_connect_group_hash(group_id: str) -> str:
    return hash_idp_group_id(ExternalIdentityProviderID.DIRECT_CONNECT,
                             group_id)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


class _FakeOIDCClient:
    def __init__(self) -> None:
        self.exchange_code = SimpleNamespace(coroutine=self._exchange_code)
        self.verify_id_token = SimpleNamespace(coroutine=self._verify_id_token)
        self.authorization_request = None
        self.exchange_request = None
        self.verify_request = None
        self.email_verified_claim = True

    async def authorization_url(self, **kwargs):
        self.authorization_request = kwargs
        return f"https://idp.example.com/authorize?state={kwargs['state']}"

    async def _exchange_code(self, **kwargs):
        self.exchange_request = kwargs
        return DirectConnectOIDCTokens(
            id_token="id-token",
            refresh_token="refresh-token",
        )

    async def _verify_id_token(self, **kwargs):
        self.verify_request = kwargs
        claims = {
            "sub": "subject-1",
            "email": "Alice@Example.com",
            "name": "Alice Example",
            "_kumo_idp_groups": ["idp-admin"],
        }
        if self.email_verified_claim is not _MISSING_EMAIL_VERIFIED:
            claims["email_verified"] = self.email_verified_claim
        return claims


_MISSING_EMAIL_VERIFIED = object()


class _RecordingGroupResolver:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self._group_hashes_by_user: dict[str, tuple[str, ...]] = {}
        self.calls = []

    def remember_external_groups(self, user_id: str,
                                 external_group_ids: list[str]) -> None:
        self._group_hashes_by_user[user_id] = tuple(
            _direct_connect_group_hash(group_id)
            for group_id in external_group_ids)

    async def resolve_groups(self, user_id: str, auth_context: dict):
        self.calls.append((user_id, auth_context))
        return ResolvedGroupMembership(
            tuple(self._mapping[group_hash]
                  for group_hash in self._group_hashes_by_user.get(user_id, ())
                  if group_hash in self._mapping))


class _RecordingProjectResolver:
    def __init__(self, project_ids_by_group: dict[str, list[str]]) -> None:
        self._project_ids_by_group = project_ids_by_group
        self.calls = []

    async def resolve_projects(self, user_authorization: UserAuthorization):
        self.calls.append(user_authorization)
        project_ids = []
        for group_id in user_authorization.kumo_group_ids:
            project_ids.extend(self._project_ids_by_group.get(group_id, []))
        return project_ids


@pytest.fixture
def auth_store(event_loop, tmp_path):
    state_db = event_loop.run_until_complete(
        StateDB.open_sqlite(str(tmp_path / "direct-connect-auth.db")))
    store = DirectConnectAuthStore(DirectConnectAuthService(state_db))
    event_loop.run_until_complete(
        store.set_config(
            DirectConnectAuthConfigRecord(
                timestamps=get_current_timestamps(),
                issuer="https://idp.example.com",
                client_id="kumo",
                encrypted_client_secret="secret",
                groups_claim="groups",
                admin_emails=["alice@example.com"],
                enabled=True,
            )))
    return store


def _build_app_and_fake_client(
    auth_store,
    monkeypatch,
    *,
    login_method=None,
    allowed_return_origins=None,
    group_resolver=None,
    password_login_ip_rate_limit_enabled=True,
):
    fake_client = _FakeOIDCClient()
    resource = DirectConnectAuthResource(
        auth_store,
        group_resolver=group_resolver,
        login_method=login_method,
        allowed_return_origins=allowed_return_origins,
        password_login_ip_rate_limit_enabled=(
            password_login_ip_rate_limit_enabled),
    )

    async def _build_client():
        return fake_client

    monkeypatch.setattr(resource, "_build_client", _build_client)
    app = FastAPI()
    app.include_router(resource.router)
    return app, fake_client


def _add_direct_connect_auth_probe(
    app: FastAPI,
    auth_store: DirectConnectAuthStore,
    *,
    group_resolver=None,
    project_resolver=None,
) -> None:
    @app.middleware("http")
    async def _direct_connect_auth_middleware(request: Request, call_next):
        return await authorization_middleware(
            request=request,
            call_next=call_next,
            auth_disabled=False,
            auth_proxied=False,
            is_unit_test=True,
            group_resolver=group_resolver,
            project_resolver=project_resolver,
            direct_connect_auth_store=auth_store,
        )

    @app.get("/v0/direct-connect-auth-probe")
    async def _direct_connect_auth_probe(request: Request):
        user_auth = get_user_authorization_from_request(request)
        return {
            "access_token": user_auth.access_token,
            "error": user_auth.error,
            "is_admin": user_auth.is_admin,
            "kumo_group_ids": list(user_auth.kumo_group_ids),
            "permissions": list(user_auth.permissions),
            "project_ids": list(user_auth.project_ids),
            "user_id": user_auth.user_id,
        }


@pytest.fixture
def app_and_fake_client(auth_store, monkeypatch):
    return _build_app_and_fake_client(auth_store, monkeypatch)


def _login(client: TestClient):
    response = client.get("/v0/auth/login?return_to=/projects",
                          follow_redirects=False)
    assert response.status_code in (302, 307)
    state = parse_qs(urlparse(response.headers["location"]).query)["state"][0]
    _assert_cookie_set(response, DIRECT_CONNECT_LOGIN_COOKIE, path="/v0/auth")
    return state


def _set_cookie_header(response, name: str) -> Optional[str]:
    # The httpx-backed TestClient exposes individual Set-Cookie headers via
    # response.headers.get_list("Set-Cookie"). Do NOT use response.cookies
    # (its CookieJar silently drops the max-age=0 cleared cookies these
    # assertions check for).
    prefix = f"{name}="
    for header in response.headers.get_list("Set-Cookie"):
        if header.startswith(prefix):
            return header
    return None


def _set_cookie_morsel(response, name: str) -> Optional[Morsel]:
    header = _set_cookie_header(response, name)
    if header is None:
        return None
    cookie = SimpleCookie()
    cookie.load(header)
    return cookie.get(name)


def _assert_cookie_set(response, name: str, *, path: str = "/") -> str:
    morsel = _set_cookie_morsel(response, name)
    assert morsel is not None
    assert morsel.value
    assert morsel["path"] == path
    return morsel.value


def _assert_cookie_cleared(response, name: str, *, path: str = "/") -> None:
    morsel = _set_cookie_morsel(response, name)
    assert morsel is not None
    assert morsel.value == ""
    assert morsel["max-age"] == "0"
    assert morsel["path"] == path


def _assert_cookie_not_set(response, name: str) -> None:
    assert _set_cookie_header(response, name) is None


@pytest.mark.parametrize(
    "return_to",
    [
        r"/\\evil.com",
        "/%5cevil.com",
        "/%5Cevil.com",
        "/projects%0d%0aLocation:%20https://evil.com",
        "//evil.com/path",
    ],
)
def test_safe_return_to_rejects_redirect_bypass_vectors(return_to):
    assert _safe_return_to(return_to, None) == "/"


def test_safe_return_to_reconstructs_allowed_absolute_url():
    assert _safe_return_to(
        "https://kumo.example.com/projects/?tab=jobs#ignored",
        None,
        allowed_return_origins=["https://kumo.example.com"],
    ) == "/projects/?tab=jobs"


def test_login_sets_pkce_cookie_and_redirects_to_idp(app_and_fake_client):
    app, fake_client = app_and_fake_client
    with TestClient(app) as client:
        state = _login(client)

    request = fake_client.authorization_request
    assert request["state"] == state
    assert request["redirect_uri"].endswith("/v0/auth/callback")
    assert request["nonce"]
    assert request["code_challenge"]


def test_login_uses_exact_browser_origin_for_callback(app_and_fake_client):
    app, fake_client = app_and_fake_client
    with TestClient(app, base_url="http://localhost:12000") as client:
        _login(client)

    assert fake_client.authorization_request["redirect_uri"] == (
        "http://localhost:12000/v0/auth/callback")


def test_login_cookie_preserves_security_attributes(app_and_fake_client):
    app, _fake_client = app_and_fake_client
    with TestClient(app, base_url="http://localhost:12000") as client:
        response = client.get("/v0/auth/login?return_to=/home",
                              follow_redirects=False)

    cookie = _set_cookie_morsel(response, DIRECT_CONNECT_LOGIN_COOKIE)
    assert cookie is not None
    assert cookie["httponly"]
    assert cookie["samesite"].lower() == "lax"
    assert cookie["path"] == "/v0/auth"
    assert not cookie["secure"]


def test_https_login_cookie_remains_secure(app_and_fake_client):
    app, fake_client = app_and_fake_client
    with TestClient(app, base_url="https://localhost:12000") as client:
        response = client.get("/v0/auth/login?return_to=/home",
                              follow_redirects=False)

    cookie = _set_cookie_morsel(response, DIRECT_CONNECT_LOGIN_COOKIE)
    assert cookie is not None
    assert cookie["secure"]
    assert fake_client.authorization_request["redirect_uri"] == (
        "https://localhost:12000/v0/auth/callback")


def test_oidc_login_and_callback_are_disabled_in_password_mode(
        auth_store, monkeypatch):
    app, fake_client = _build_app_and_fake_client(auth_store, monkeypatch,
                                                  login_method="password")

    with TestClient(app) as client:
        login_response = client.get("/v0/auth/login", follow_redirects=False)
        callback_response = client.get(
            "/v0/auth/callback?code=code&state=state",
            follow_redirects=False,
        )

    assert login_response.status_code == 404
    assert callback_response.status_code == 404
    assert fake_client.authorization_request is None


def test_callback_creates_user_session_and_clears_login_cookie(
        app_and_fake_client):
    app, fake_client = app_and_fake_client
    with TestClient(app) as client:
        state = _login(client)
        response = client.get(
            f"/v0/auth/callback?code=auth-code&state={state}",
            follow_redirects=False,
        )

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/projects"
    _assert_cookie_set(response, DIRECT_CONNECT_SESSION_COOKIE)
    _assert_cookie_cleared(response, DIRECT_CONNECT_LOGIN_COOKIE,
                           path="/v0/auth")
    assert fake_client.exchange_request["code"] == "auth-code"
    assert fake_client.verify_request["id_token"] == "id-token"


def test_callback_persists_user_and_session(app_and_fake_client, auth_store,
                                            event_loop):
    app, _fake_client = app_and_fake_client
    with TestClient(app) as client:
        state = _login(client)
        before_callback = now_utc()
        response = client.get(
            f"/v0/auth/callback?code=auth-code&state={state}",
            follow_redirects=False,
        )
        session_id = _assert_cookie_set(response, DIRECT_CONNECT_SESSION_COOKIE)

    user_lookup = event_loop.run_until_complete(
        auth_store.get_user_by_email("alice@example.com"))
    assert user_lookup is not None
    user_id, user = user_lookup
    assert user.role == "admin"

    session = event_loop.run_until_complete(auth_store.get_session(session_id))
    assert session is not None
    assert session.user_id == user_id
    assert not hasattr(session, "idp_group_hashes")
    assert before_callback + timedelta(
        seconds=DIRECT_CONNECT_SESSION_SECONDS) <= session.expires_time
    assert session.expires_time <= now_utc() + timedelta(
        seconds=DIRECT_CONNECT_SESSION_SECONDS)


@pytest.mark.parametrize(
    ("email_verified_claim", "expected_role"),
    [
        (True, "admin"),
        (_MISSING_EMAIL_VERIFIED, "user"),
        (False, "user"),
        ("true", "user"),
        ("definitely", "user"),
        (1, "user"),
    ],
    ids=[
        "true", "missing", "false", "string-true", "hostile-string",
        "truthy-number"
    ],
)
@pytest.mark.asyncio
async def test_callback_bootstrap_admin_requires_verified_boolean_claim(
        tmp_path, monkeypatch, email_verified_claim, expected_role):
    state_db = await StateDB.open_sqlite(str(tmp_path / "callback-auth.db"))
    auth_store = DirectConnectAuthStore(DirectConnectAuthService(state_db))
    await auth_store.set_config(
        DirectConnectAuthConfigRecord(
            timestamps=get_current_timestamps(),
            issuer="https://idp.example.com",
            client_id="kumo",
            encrypted_client_secret="secret",
            groups_claim="groups",
            admin_emails=["alice@example.com"],
            enabled=True,
        ))
    fake_client = _FakeOIDCClient()
    fake_client.email_verified_claim = email_verified_claim
    resource = DirectConnectAuthResource(auth_store)

    async def build_client():
        return fake_client

    monkeypatch.setattr(resource, "_build_client", build_client)
    encrypted_login = encrypt_secret_string(
        json.dumps({
            "state": "state",
            "nonce": "nonce",
            "code_verifier": "verifier",
            "redirect_uri": "https://kumo.example.com/v0/auth/callback",
            "return_to": "/",
        }))
    request = Request({
        "type":
        "http",
        "asgi": {
            "version": "3.0"
        },
        "http_version":
        "1.1",
        "method":
        "GET",
        "scheme":
        "https",
        "path":
        "/v0/auth/callback",
        "raw_path":
        b"/v0/auth/callback",
        "query_string":
        b"",
        "headers": [(
            b"cookie",
            (f"{DIRECT_CONNECT_LOGIN_COOKIE}="
             f"{encrypted_login}").encode(),
        )],
        "client": ("127.0.0.1", 1234),
        "server": ("kumo.example.com", 443),
        "root_path":
        "",
    })

    response = await resource.callback(request, code="auth-code", state="state")

    assert response.status_code in (302, 307)
    user_lookup = await auth_store.get_user_by_email("alice@example.com")
    assert user_lookup is not None
    assert user_lookup[1].role == expected_role
    await state_db.close()


def test_oidc_session_cookie_authenticates_through_middleware_with_rbac(
    auth_store,
    monkeypatch,
    event_loop,
):
    idp_group_hash = _direct_connect_group_hash("idp-admin")
    group_resolver = _RecordingGroupResolver({idp_group_hash: "kumo-admins"})
    app, _fake_client = _build_app_and_fake_client(
        auth_store,
        monkeypatch,
        group_resolver=group_resolver,
    )
    project_resolver = _RecordingProjectResolver(
        {"kumo-admins": ["project-admin"]})
    _add_direct_connect_auth_probe(
        app,
        auth_store,
        group_resolver=group_resolver,
        project_resolver=project_resolver,
    )

    with TestClient(app) as client:
        state = _login(client)
        callback_response = client.get(
            f"/v0/auth/callback?code=auth-code&state={state}",
            follow_redirects=False,
        )
        session_id = _assert_cookie_set(callback_response,
                                        DIRECT_CONNECT_SESSION_COOKIE)
        probe_response = client.get("/v0/direct-connect-auth-probe")

    session = event_loop.run_until_complete(auth_store.get_session(session_id))
    assert session is not None
    assert not hasattr(session, "idp_group_hashes")

    assert probe_response.status_code == 200
    probe_payload = probe_response.json()
    assert probe_payload["error"] is None
    assert probe_payload["access_token"] == AUTHORIZATION_DIRECT_CONNECT
    assert probe_payload["is_admin"]
    assert probe_payload["user_id"].startswith("direct_connect|")
    assert probe_payload["kumo_group_ids"] == ["kumo-admins"]
    assert probe_payload["project_ids"] == ["project-admin"]
    assert group_resolver.calls == [(
        probe_payload["user_id"],
        {
            "auth_source": "direct_connect_session",
        },
    )]
    assert len(project_resolver.calls) == 1
    assert project_resolver.calls[0].kumo_group_ids == ("kumo-admins", )


def test_callback_rejects_state_mismatch(app_and_fake_client):
    app, _fake_client = app_and_fake_client
    with TestClient(app) as client:
        _login(client)
        response = client.get(
            "/v0/auth/callback?code=auth-code&state=wrong-state",
            follow_redirects=False,
        )

    assert response.status_code == 401
    _assert_cookie_not_set(response, DIRECT_CONNECT_SESSION_COOKIE)


def test_logout_deletes_session(app_and_fake_client, auth_store, event_loop):
    app, _fake_client = app_and_fake_client
    with TestClient(app) as client:
        state = _login(client)
        callback_response = client.get(
            f"/v0/auth/callback?code=auth-code&state={state}",
            follow_redirects=False,
        )
        session_id = _assert_cookie_set(callback_response,
                                        DIRECT_CONNECT_SESSION_COOKIE)
        logout_response = client.post("/v0/auth/logout", follow_redirects=False)

    assert logout_response.status_code == 303
    _assert_cookie_cleared(logout_response, DIRECT_CONNECT_SESSION_COOKIE)
    assert event_loop.run_until_complete(
        auth_store.get_session(session_id)) is None


def test_me_returns_current_direct_connect_user(auth_store, monkeypatch,
                                                event_loop):
    user_id, _user = event_loop.run_until_complete(
        auth_store.upsert_user_from_idp(
            issuer="https://idp.example.com",
            subject="subject-1",
            email="alice@example.com",
            display_name="Alice Example",
            refresh_token="refresh-token",
        ))
    permissions = [Permissions.READ.compose()]
    monkeypatch.setattr(
        routes,
        "get_user_authorization_from_request",
        lambda _request: UserAuthorization(
            user_id=UserID(user_id),
            access_token="direct-connect-session",
            permissions=permissions,
            kumo_group_ids=("group-1", ),
            project_ids=("project-1", ),
        ),
    )
    app = FastAPI()
    app.include_router(DirectConnectAuthResource(auth_store).router)

    with TestClient(app) as client:
        response = client.get("/v0/auth/me")

    assert response.status_code == 200
    assert response.json() == {
        "user_id": user_id,
        "email": "alice@example.com",
        "display_name": "Alice Example",
        "role": "user",
        "permissions": permissions,
        "kumo_group_ids": ["group-1"],
        "project_ids": ["project-1"],
    }


def test_password_login_requires_password_mode(app_and_fake_client, auth_store,
                                               event_loop):
    app, _fake_client = app_and_fake_client
    user = event_loop.run_until_complete(
        auth_store.create_local_user(email="local@example.com"))
    event_loop.run_until_complete(
        auth_store.set_temporary_password(user.user_id, "SeedPass12345"))

    with TestClient(app) as client:
        response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "local@example.com",
                "password": "SeedPass12345"
            },
        )

    assert response.status_code == 404


def test_password_login_returns_reset_token_for_temporary_password(
        auth_store, event_loop, monkeypatch):
    app, _fake_client = _build_app_and_fake_client(auth_store, monkeypatch,
                                                   login_method="password")
    user = event_loop.run_until_complete(
        auth_store.create_local_user(email="local@example.com"))
    event_loop.run_until_complete(
        auth_store.set_temporary_password(user.user_id, "SeedPass12345"))

    with TestClient(app) as client:
        response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "LOCAL@example.com",
                "password": "SeedPass12345"
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "password_reset_required"
    assert response.json()["reset_token"]
    _assert_cookie_not_set(response, DIRECT_CONNECT_SESSION_COOKIE)


def test_password_reset_sets_session_cookie_and_enables_normal_login(
        auth_store, event_loop, monkeypatch):
    app, _fake_client = _build_app_and_fake_client(auth_store, monkeypatch,
                                                   login_method="password")
    user = event_loop.run_until_complete(
        auth_store.create_local_user(email="local@example.com"))
    event_loop.run_until_complete(
        auth_store.set_temporary_password(user.user_id, "SeedPass12345"))

    with TestClient(app) as client:
        login_response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "local@example.com",
                "password": "SeedPass12345"
            },
        )
        reset_token = login_response.json()["reset_token"]
        reset_response = client.post(
            "/v0/auth/password/complete-reset",
            json={
                "reset_token": reset_token,
                "new_password": "FinalPass67890"
            },
        )
        second_login_response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "local@example.com",
                "password": "FinalPass67890"
            },
        )

    assert reset_response.status_code == 200
    assert reset_response.json() == {"status": "ok", "reset_token": None}
    _assert_cookie_set(reset_response, DIRECT_CONNECT_SESSION_COOKIE)
    assert second_login_response.status_code == 200
    assert second_login_response.json() == {"status": "ok", "reset_token": None}
    _assert_cookie_set(second_login_response, DIRECT_CONNECT_SESSION_COOKIE)


def test_password_session_cookie_authenticates_through_middleware_without_groups(
    auth_store,
    event_loop,
    monkeypatch,
):
    app, _fake_client = _build_app_and_fake_client(auth_store, monkeypatch,
                                                   login_method="password")
    group_resolver = _RecordingGroupResolver(
        {_direct_connect_group_hash("idp-admin"): "kumo-admins"})
    project_resolver = _RecordingProjectResolver(
        {"kumo-admins": ["project-admin"]})
    _add_direct_connect_auth_probe(
        app,
        auth_store,
        group_resolver=group_resolver,
        project_resolver=project_resolver,
    )
    user = event_loop.run_until_complete(
        auth_store.create_local_user(email="local@example.com"))
    event_loop.run_until_complete(
        auth_store.set_temporary_password(user.user_id, "SeedPass12345"))

    with TestClient(app) as client:
        login_response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "local@example.com",
                "password": "SeedPass12345"
            },
        )
        reset_response = client.post(
            "/v0/auth/password/complete-reset",
            json={
                "reset_token": login_response.json()["reset_token"],
                "new_password": "FinalPass67890"
            },
        )
        session_id = _assert_cookie_set(reset_response,
                                        DIRECT_CONNECT_SESSION_COOKIE)
        probe_response = client.get("/v0/direct-connect-auth-probe")

    session = event_loop.run_until_complete(auth_store.get_session(session_id))
    assert session is not None
    assert not hasattr(session, "idp_group_hashes")

    assert probe_response.status_code == 200
    probe_payload = probe_response.json()
    assert probe_payload["error"] is None
    assert probe_payload["access_token"] == AUTHORIZATION_DIRECT_CONNECT
    assert not probe_payload["is_admin"]
    assert probe_payload["user_id"] == user.user_id
    assert probe_payload["kumo_group_ids"] == []
    assert probe_payload["project_ids"] == []
    assert group_resolver.calls == [(
        user.user_id,
        {
            "auth_source": "direct_connect_session",
        },
    )]
    assert len(project_resolver.calls) == 1
    assert project_resolver.calls[0].kumo_group_ids == ()


def test_normal_password_login_session_cookie_authenticates_through_middleware_without_groups(
    auth_store,
    event_loop,
    monkeypatch,
):
    app, _fake_client = _build_app_and_fake_client(auth_store, monkeypatch,
                                                   login_method="password")
    group_resolver = _RecordingGroupResolver(
        {_direct_connect_group_hash("idp-admin"): "kumo-admins"})
    project_resolver = _RecordingProjectResolver(
        {"kumo-admins": ["project-admin"]})
    _add_direct_connect_auth_probe(
        app,
        auth_store,
        group_resolver=group_resolver,
        project_resolver=project_resolver,
    )
    user = event_loop.run_until_complete(
        auth_store.create_local_user(email="local@example.com"))
    event_loop.run_until_complete(
        auth_store.set_temporary_password(user.user_id, "SeedPass12345"))

    with TestClient(app) as client:
        login_response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "local@example.com",
                "password": "SeedPass12345"
            },
        )
        reset_response = client.post(
            "/v0/auth/password/complete-reset",
            json={
                "reset_token": login_response.json()["reset_token"],
                "new_password": "FinalPass67890"
            },
        )
        _assert_cookie_set(reset_response, DIRECT_CONNECT_SESSION_COOKIE)
        second_login_response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "local@example.com",
                "password": "FinalPass67890"
            },
        )
        session_id = _assert_cookie_set(second_login_response,
                                        DIRECT_CONNECT_SESSION_COOKIE)
        group_resolver.calls.clear()
        project_resolver.calls.clear()
        probe_response = client.get("/v0/direct-connect-auth-probe")

    session = event_loop.run_until_complete(auth_store.get_session(session_id))
    assert session is not None
    assert not hasattr(session, "idp_group_hashes")

    assert probe_response.status_code == 200
    probe_payload = probe_response.json()
    assert probe_payload["error"] is None
    assert probe_payload["access_token"] == AUTHORIZATION_DIRECT_CONNECT
    assert not probe_payload["is_admin"]
    assert probe_payload["user_id"] == user.user_id
    assert probe_payload["kumo_group_ids"] == []
    assert probe_payload["project_ids"] == []
    assert group_resolver.calls == [(
        user.user_id,
        {
            "auth_source": "direct_connect_session",
        },
    )]
    assert len(project_resolver.calls) == 1
    assert project_resolver.calls[0].kumo_group_ids == ()


def test_password_reset_rejects_weak_password(auth_store, event_loop,
                                              monkeypatch):
    app, _fake_client = _build_app_and_fake_client(auth_store, monkeypatch,
                                                   login_method="password")
    user = event_loop.run_until_complete(
        auth_store.create_local_user(email="local@example.com"))
    event_loop.run_until_complete(
        auth_store.set_temporary_password(user.user_id, "SeedPass12345"))

    with TestClient(app) as client:
        login_response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "local@example.com",
                "password": "SeedPass12345"
            },
        )
        reset_response = client.post(
            "/v0/auth/password/complete-reset",
            json={
                "reset_token": login_response.json()["reset_token"],
                "new_password": "short1",
            },
        )

    assert reset_response.status_code == 400
    assert reset_response.json()["detail"] == (
        "Password must be at least 12 characters")


def test_password_login_rejects_wrong_password(app_and_fake_client, auth_store,
                                               event_loop, monkeypatch):
    app, _fake_client = _build_app_and_fake_client(auth_store, monkeypatch,
                                                   login_method="password")
    user = event_loop.run_until_complete(
        auth_store.create_local_user(email="local@example.com"))
    event_loop.run_until_complete(
        auth_store.set_temporary_password(user.user_id, "SeedPass12345"))

    with TestClient(app) as client:
        response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "local@example.com",
                "password": "wrong-password"
            },
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password."


def test_password_login_can_omit_untrusted_shared_client_ip(
        auth_store, event_loop, monkeypatch):
    check_allowed = AsyncMock(wraps=auth_store.check_password_login_allowed)
    record_failure = AsyncMock(wraps=auth_store.record_password_login_failure)
    monkeypatch.setattr(auth_store, "check_password_login_allowed",
                        check_allowed)
    monkeypatch.setattr(auth_store, "record_password_login_failure",
                        record_failure)
    app, _fake_client = _build_app_and_fake_client(
        auth_store,
        monkeypatch,
        login_method="password",
        password_login_ip_rate_limit_enabled=False,
    )
    user = event_loop.run_until_complete(
        auth_store.create_local_user(email="local@example.com"))
    event_loop.run_until_complete(
        auth_store.set_temporary_password(user.user_id, "SeedPass12345"))

    with TestClient(app) as client:
        response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "local@example.com",
                "password": "wrong-password",
            },
        )

    assert response.status_code == 401
    assert check_allowed.await_args.kwargs["source_ip"] is None
    assert record_failure.await_args.kwargs["source_ip"] is None


def test_password_login_disabled_user_uses_uniform_failure(
        auth_store, event_loop, monkeypatch):
    app, _fake_client = _build_app_and_fake_client(auth_store, monkeypatch,
                                                   login_method="password")
    user = event_loop.run_until_complete(
        auth_store.create_local_user(
            email="local@example.com",
            disabled=True,
        ))
    event_loop.run_until_complete(
        auth_store.set_temporary_password(user.user_id, "SeedPass12345"))

    with TestClient(app) as client:
        response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "local@example.com",
                "password": "SeedPass12345",
            },
        )

    assert response.status_code == 401


def test_password_login_throttles_repeated_failures(auth_store, event_loop,
                                                    monkeypatch):
    app, _fake_client = _build_app_and_fake_client(auth_store, monkeypatch,
                                                   login_method="password")
    user = event_loop.run_until_complete(
        auth_store.create_local_user(email="local@example.com"))
    event_loop.run_until_complete(
        auth_store.set_temporary_password(user.user_id, "SeedPass12345"))

    with TestClient(app) as client:
        for _ in range(storage._PASSWORD_LOGIN_EMAIL_LIMIT):
            response = client.post(
                "/v0/auth/password/login",
                json={
                    "email": "local@example.com",
                    "password": "wrong-password",
                },
            )
            assert response.status_code == 401

        locked_response = client.post(
            "/v0/auth/password/login",
            json={
                "email": "local@example.com",
                "password": "SeedPass12345",
            },
        )

    assert locked_response.status_code == 429
    assert locked_response.json()["detail"] == (
        "Too many sign-in attempts. Please wait and try again.")
