# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import cachetools
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from rest.config.encryption import encrypt_secret_string
from rest.config.status import RestStatusCode
from rest.deps import direct_connect_oidc_client as oidc
from rest.deps.authentication import Permissions, UserAuthorization
from rest.deps.direct_connect_group_resolver import (
    _GROUP_CACHE_TTL_SECONDS,
    DirectConnectGroupResolver,
)
from rest.deps.direct_connect_oidc_client import DirectConnectOIDCClient
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from rest.utils.driver_common import get_current_timestamps
from statedb.config.common import UserID
from statedb.config.direct_connect_auth import DirectConnectAuthConfigRecord
from statedb.db import StateDB, StateServices
from statedb.rest.group import KumoGroupCreateSpec


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clear_oidc_caches():
    oidc._DISCOVERY_CACHE.clear()
    oidc._JWKS_CLIENT_CACHE.clear()


def _client() -> DirectConnectOIDCClient:
    return DirectConnectOIDCClient(
        issuer="https://idp.example.com",
        client_id="kumo",
        client_secret="secret",
        groups_claim="groups",
    )


@pytest.mark.asyncio
async def test_authorization_url_requests_standard_oidc_scopes(monkeypatch):
    monkeypatch.setattr(
        oidc.requests,
        "get",
        lambda *_args, **_kwargs: _Response(
            {
                "issuer": "https://idp.example.com",
                "authorization_endpoint": "https://idp.example.com/authorize",
            }),
    )

    url = await _client().authorization_url(
        redirect_uri="https://kumo.example.com/auth/callback",
        state="state",
        nonce="nonce",
        code_challenge="challenge",
    )

    query = parse_qs(urlparse(url).query)
    assert query["scope"] == ["openid email profile offline_access"]


@pytest.mark.asyncio
async def test_exchange_code_requires_refresh_token(monkeypatch):
    oidc._DISCOVERY_CACHE["https://idp.example.com"] = {
        "issuer": "https://idp.example.com",
        "token_endpoint": "https://idp.example.com/token",
    }
    monkeypatch.setattr(
        oidc.requests,
        "post",
        lambda *_args, **_kwargs: _Response({"id_token": "id-token"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _client().exchange_code.coroutine(
            code="code",
            redirect_uri="https://kumo.example.com/auth/callback",
            code_verifier="verifier",
        )

    assert exc_info.value.status_code == 503
    assert "refresh token" in exc_info.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(("groups", "expected"), [
    (["idp-a", "idp-a", "idp-b"], ["idp-a", "idp-b"]),
    ([], []),
    (None, []),
])
async def test_refresh_groups_uses_userinfo_and_rotated_token(
        monkeypatch, groups, expected):
    _mock_group_refresh(monkeypatch, token="must-not-use-this-token",
                        userinfo={"groups": groups})

    result = await _client().refresh_groups.coroutine(
        refresh_token="refresh-token")

    assert result.external_group_ids == expected
    assert result.refresh_token == "rotated-refresh-token"


@pytest.mark.asyncio
@pytest.mark.parametrize("userinfo", [None, {}, {"email": "alice@example.com"}])
@pytest.mark.parametrize(("groups", "expected"), [
    (["idp-admin", {
        "name": "idp-user"
    }], ["idp-admin", "idp-user"]),
    (["idp-user"], ["idp-user"]),
    ([], []),
])
async def test_refresh_groups_falls_back_to_verified_id_token(
        monkeypatch, userinfo, groups, expected):
    token = _signed_id_token(monkeypatch, payload_overrides={"groups": groups})
    _mock_group_refresh(monkeypatch, token=token, userinfo=userinfo)

    result = await _client().refresh_groups.coroutine(
        refresh_token="refresh-token")

    assert result.external_group_ids == expected
    assert result.refresh_token == "rotated-refresh-token"


def _mock_group_refresh(monkeypatch, *, token, userinfo):
    oidc._DISCOVERY_CACHE["https://idp.example.com"] = {
        "issuer": "https://idp.example.com",
        "token_endpoint": "https://idp.example.com/token",
        "userinfo_endpoint": "https://idp.example.com/userinfo",
        "jwks_uri": "https://idp.example.com/jwks",
    }
    received_tokens = []

    def post(_url, *, data, **_kwargs):
        received_tokens.append(data["refresh_token"])
        access_token = "access-token" if userinfo is not None else None
        return _Response({
            "access_token": access_token,
            "id_token": token,
            "refresh_token": "rotated-refresh-token",
        })

    monkeypatch.setattr(oidc.requests, "post", post)
    monkeypatch.setattr(oidc.requests, "get",
                        lambda *_args, **_kwargs: _Response(userinfo))
    return received_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    "expired", "audience", "issuer", "signature", "missing-exp", "missing-token"
])
async def test_refresh_groups_rejects_invalid_fallback_token(
        monkeypatch, invalid):
    overrides = {
        "expired": {
            "exp": datetime.now(timezone.utc) - timedelta(minutes=5)
        },
        "audience": {
            "aud": "another-client"
        },
        "issuer": {
            "iss": "https://another-idp.example.com"
        },
    }
    token = _signed_id_token(
        monkeypatch, payload_overrides=overrides.get(invalid),
        drop_claims=["exp"] if invalid == "missing-exp" else None)
    if invalid == "signature":
        token = token.rsplit(".", 1)[0] + ".AAAA"
    elif invalid == "missing-token":
        token = None
    _mock_group_refresh(monkeypatch, token=token, userinfo={})

    with pytest.raises(HTTPException) as exc_info:
        await _client().refresh_groups.coroutine(refresh_token="refresh-token")

    assert exc_info.value.status_code == (503 if invalid == "missing-token" else
                                          401)


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh", ["retained", "removed", "invalid"])
async def test_group_only_project_access_across_cache_expiry(
        monkeypatch, tmp_path, refresh):
    token = _signed_id_token(
        monkeypatch, payload_overrides={
            "groups": [] if refresh == "removed" else ["idp-user"],
        })
    received_tokens = _mock_group_refresh(
        monkeypatch, token="invalid" if refresh == "invalid" else token,
        userinfo={"email": "alice@example.com"})
    db = await StateDB.open_sqlite(str(tmp_path / "group-access.db"))
    try:
        services = StateServices(db)
        store = DirectConnectAuthStore(services.direct_connect_auth)
        await store.set_config(
            DirectConnectAuthConfigRecord(
                timestamps=get_current_timestamps(),
                issuer="https://idp.example.com",
                client_id="kumo",
                encrypted_client_secret=encrypt_secret_string("secret"),
                groups_claim="groups",
                admin_emails=[],
                enabled=True,
            ))
        user_id, user = await store.upsert_user_from_idp(
            issuer="https://idp.example.com",
            subject="subject-1",
            email="alice@example.com",
            display_name="Alice Example",
            refresh_token="refresh-token",
        )
        assert user.role == "user"
        admin = UserAuthorization(
            user_id=UserID("admin"),
            access_token="test",
            permissions=[Permissions.ADMIN.compose()],
        )
        group = (await services.groups.create_kumo_groups(
            [KumoGroupCreateSpec(display_name="Test group")], admin))[0]
        await services.groups.create_external_mapping(group.id,
                                                      "direct_connect",
                                                      "idp-user",
                                                      principal=admin)
        project = await services.projects.create_project(
            name="Group-only project", group_id=group.id,
            user_authorization=admin)
        now = [0.0]
        resolver = DirectConnectGroupResolver(
            state_services=services,
            auth_store=store,
            cache=cachetools.TTLCache(maxsize=10, ttl=_GROUP_CACHE_TTL_SECONDS,
                                      timer=lambda: now[0]),
        )
        resolver.remember_external_groups(user_id, ["idp-user"])

        async def visible_projects():
            membership = await resolver.resolve_groups(user_id, {})
            principal = UserAuthorization(
                user_id=UserID(user_id),
                access_token="test",
                permissions=[Permissions.READ.compose()],
                kumo_group_ids=membership.group_ids,
                external_membership_unavailable=membership.
                external_membership_unavailable,
            )
            assert not principal.is_admin
            return await services.projects.list_visible_project_ids(principal)

        assert await visible_projects() == [str(project.id)]
        now[0] = _GROUP_CACHE_TTL_SECONDS - 1
        assert await visible_projects() == [str(project.id)]
        assert received_tokens == []

        now[0] = _GROUP_CACHE_TTL_SECONDS
        expected = [str(project.id)] if refresh == "retained" else []
        assert await visible_projects() == expected
        assert received_tokens == ["refresh-token"]
        now[0] = 2 * _GROUP_CACHE_TTL_SECONDS
        assert await visible_projects() == expected
        assert received_tokens[-1] == ("refresh-token" if refresh == "invalid"
                                       else "rotated-refresh-token")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_refresh_groups_token_failure_has_safe_context(
        monkeypatch, caplog):
    oidc._DISCOVERY_CACHE["https://idp.example.com"] = {
        "issuer": "https://idp.example.com",
        "token_endpoint": "https://idp.example.com/token",
    }

    class _TokenErrorResponse:
        status_code = 401

        def raise_for_status(self):
            raise oidc.requests.HTTPError(
                "401 Client Error at https://idp.example.com/token?"
                "code=sensitive-auth-code",
                response=self,
            )

        def json(self):
            return {
                "error": "sensitive-provider-error",
                "error_description": "sensitive-provider-body-token",
            }

    monkeypatch.setattr(
        oidc.requests,
        "post",
        lambda *_args, **_kwargs: _TokenErrorResponse(),
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException) as exc_info:
            await _client().refresh_groups.coroutine(
                refresh_token="sensitive-refresh-token")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == RestStatusCode.AUTH_INVALID_TOKEN.value
    assert "OIDC refresh token exchange failed" in caplog.text
    assert "error_type=HTTPError" in caplog.text
    assert "status=401" in caplog.text
    assert "sensitive-refresh-token" not in caplog.text
    assert "sensitive-auth-code" not in caplog.text
    assert "sensitive-provider-error" not in caplog.text
    assert "sensitive-provider-body-token" not in caplog.text


@pytest.mark.asyncio
async def test_exchange_code_exception_text_and_url_are_not_logged(
        monkeypatch, caplog):
    oidc._DISCOVERY_CACHE["https://idp.example.com"] = {
        "issuer": "https://idp.example.com",
        "token_endpoint": "https://idp.example.com/token",
    }

    def _raise_connection_error(*_args, **_kwargs):
        raise oidc.requests.ConnectionError(
            "connection failed for https://idp.example.com/token?"
            "client_secret=sensitive-client-secret&code=sensitive-auth-code")

    monkeypatch.setattr(oidc.requests, "post", _raise_connection_error)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException) as exc_info:
            await _client().exchange_code.coroutine(
                code="sensitive-auth-code",
                redirect_uri="https://kumo.example.com/auth/callback",
                code_verifier="sensitive-code-verifier",
            )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == RestStatusCode.AUTH_INVALID_TOKEN.value
    assert "OIDC code exchange failed" in caplog.text
    assert "error_type=ConnectionError" in caplog.text
    assert "status=unknown" in caplog.text
    assert "idp.example.com/token" not in caplog.text
    assert "sensitive-client-secret" not in caplog.text
    assert "sensitive-auth-code" not in caplog.text
    assert "sensitive-code-verifier" not in caplog.text


@pytest.mark.asyncio
async def test_refresh_groups_userinfo_failure_has_safe_context(
        monkeypatch, caplog):
    oidc._DISCOVERY_CACHE["https://idp.example.com"] = {
        "issuer": "https://idp.example.com",
        "token_endpoint": "https://idp.example.com/token",
        "userinfo_endpoint": "https://idp.example.com/userinfo",
    }
    monkeypatch.setattr(
        oidc.requests,
        "post",
        lambda *_args, **_kwargs: _Response({
            "access_token": "sensitive-access-token",
        }),
    )

    class _UserinfoErrorResponse:
        status_code = 503

        def raise_for_status(self):
            raise oidc.requests.HTTPError("503 Server Error", response=self)

        def json(self):
            return {"error": "temporarily_unavailable"}

    monkeypatch.setattr(
        oidc.requests,
        "get",
        lambda *_args, **_kwargs: _UserinfoErrorResponse(),
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException) as exc_info:
            await _client().refresh_groups.coroutine(
                refresh_token="sensitive-refresh-token")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == RestStatusCode.AUTH_INVALID_TOKEN.value
    assert "OIDC userinfo group refresh failed" in caplog.text
    assert "sensitive-access-token" not in caplog.text
    assert "sensitive-refresh-token" not in caplog.text


def test_discovery_rejects_issuer_mismatch(monkeypatch):
    monkeypatch.setattr(
        oidc.requests,
        "get",
        lambda *_args, **_kwargs: _Response({
            "issuer":
            "https://other-idp.example.com",
        }),
    )

    with pytest.raises(HTTPException) as exc_info:
        _client().discovery()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "OIDC discovery issuer mismatch"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        ("admin, user ,,", ["admin", "user"]),
        (
            ["admin", {
                "name": "group-name"
            }, 7, None],
            ["admin", "group-name", "7"],
        ),
        ({
            "unexpected": "shape"
        }, []),
    ],
)
def test_normalize_idp_groups(raw, expected):
    assert oidc.normalize_idp_groups(raw) == expected


@pytest.mark.parametrize(
    ("claims", "expected"),
    [
        ({
            "name": "Alice Example",
            "given_name": "Ignored"
        }, "Alice Example"),
        ({
            "given_name": "Alice",
            "family_name": "Example"
        }, "Alice Example"),
        ({
            "given_name": "Alice"
        }, "Alice"),
        ({}, ""),
    ],
)
def test_display_name_from_claims(claims, expected):
    assert oidc.display_name_from_claims(claims) == expected


def _signed_id_token(monkeypatch, *, payload_overrides=None, drop_claims=None):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    class _FakeJwksClient:
        def __init__(self, jwks_uri):
            self.jwks_uri = jwks_uri

        def get_signing_key_from_jwt(self, _token):
            return SimpleNamespace(key=public_key)

    monkeypatch.setattr(oidc.jwt, "PyJWKClient", _FakeJwksClient)
    monkeypatch.setattr(
        oidc.requests,
        "get",
        lambda *_args, **_kwargs: _Response(
            {
                "issuer": "https://idp.example.com",
                "jwks_uri": "https://idp.example.com/jwks",
            }),
    )
    now = datetime.now(timezone.utc)
    payload = {
        "iss": "https://idp.example.com",
        "aud": "kumo",
        "sub": "subject-1",
        "email": " Alice@Example.COM ",
        "nonce": "nonce-1",
        "groups": ["idp-admin", {
            "name": "idp-user"
        }],
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    for claim in drop_claims or []:
        payload.pop(claim, None)
    payload.update(payload_overrides or {})
    return jwt.encode(
        payload,
        private_key,
        algorithm="RS256",
        headers={"kid": "kid-1"},
    )


def test_verify_id_token_validates_signature_and_normalizes_claims(monkeypatch):
    token = _signed_id_token(monkeypatch)

    payload = _client().verify_id_token(id_token=token, nonce="nonce-1")

    assert payload["sub"] == "subject-1"
    assert payload["email"] == "alice@example.com"
    assert payload["_kumo_idp_groups"] == ["idp-admin", "idp-user"]


def test_verify_id_token_requires_exp(monkeypatch):
    token = _signed_id_token(monkeypatch, drop_claims=["exp"])

    with pytest.raises(HTTPException) as exc_info:
        _client().verify_id_token(id_token=token, nonce="nonce-1")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == RestStatusCode.AUTH_INVALID_TOKEN.value


def test_verify_id_token_applies_exp_leeway(monkeypatch):
    token = _signed_id_token(
        monkeypatch,
        payload_overrides={
            "exp": datetime.now(timezone.utc) - timedelta(seconds=30),
        },
    )

    payload = _client().verify_id_token(id_token=token, nonce="nonce-1")

    assert payload["sub"] == "subject-1"


@pytest.mark.parametrize(
    ("overrides", "nonce", "detail"),
    [
        ({}, "wrong-nonce", "Invalid OIDC nonce"),
        ({
            "sub": ""
        }, "nonce-1", "OIDC token missing subject"),
        ({
            "email": ""
        }, "nonce-1", "OIDC token missing email"),
    ],
)
def test_verify_id_token_rejects_required_claim_failures(
        monkeypatch, overrides, nonce, detail):
    token = _signed_id_token(monkeypatch, payload_overrides=overrides)

    with pytest.raises(HTTPException) as exc_info:
        _client().verify_id_token(id_token=token, nonce=nonce)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == detail


class _GoogleCreds:
    valid = True
    token = "sa-token"


def _google_client() -> DirectConnectOIDCClient:
    return DirectConnectOIDCClient(
        issuer="https://idp.example.com",
        client_id="kumo",
        client_secret="secret",
        groups_claim="groups",
        google_sa_key='{"type": "service_account"}',
        google_admin_email="workspace-admin@example.com",
    )


@pytest.mark.parametrize(
    ("google_sa_key", "google_admin_email"),
    [
        ('{"type": "service_account"}', ""),
        ("", "workspace-admin@example.com"),
    ],
)
def test_client_rejects_partial_google_groups_config(google_sa_key,
                                                     google_admin_email):
    with pytest.raises(ValueError, match="both"):
        DirectConnectOIDCClient(
            issuer="https://idp.example.com",
            client_id="kumo",
            client_secret="secret",
            groups_claim="groups",
            google_sa_key=google_sa_key,
            google_admin_email=google_admin_email,
        )


def test_client_rejects_malformed_google_sa_key():
    with pytest.raises(ValueError, match="not valid JSON"):
        DirectConnectOIDCClient(
            issuer="https://idp.example.com",
            client_id="kumo",
            client_secret="secret",
            groups_claim="groups",
            google_sa_key="not-json",
            google_admin_email="workspace-admin@example.com",
        )


def test_fetch_google_groups_paginates_and_returns_group_emails(monkeypatch):
    client = _google_client()
    monkeypatch.setattr(client, "_google_credentials", _GoogleCreds)
    calls = []

    def _get(url, params=None, headers=None, timeout=None):
        calls.append((url, dict(params), headers))
        if params.get("pageToken"):
            return _Response({"groups": [{"email": "eng@example.com"}]})
        return _Response({
            "groups": [
                {
                    "email": "ml-team@example.com"
                },
                {
                    "id": "entry-without-email-ignored"
                },
            ],
            "nextPageToken":
            "page-2",
        })

    monkeypatch.setattr(oidc.requests, "get", _get)

    groups = client._fetch_google_groups("alice@example.com")

    assert groups == ["ml-team@example.com", "eng@example.com"]
    assert [url for url, _, _ in calls] == [
        oidc._GOOGLE_DIRECTORY_GROUPS_URL,
        oidc._GOOGLE_DIRECTORY_GROUPS_URL,
    ]
    assert calls[0][1]["userKey"] == "alice@example.com"
    assert calls[0][2] == {"Authorization": "Bearer sa-token"}
    assert "pageToken" not in calls[0][1]
    assert calls[1][1]["pageToken"] == "page-2"


def test_fetch_google_groups_failure_degrades_without_logging_provider_details(
        monkeypatch, caplog):
    client = _google_client()
    monkeypatch.setattr(client, "_google_credentials", _GoogleCreds)

    def _get(*_args, **_kwargs):
        raise oidc.requests.RequestException(
            "directory unavailable: sensitive-google-bearer-token")

    monkeypatch.setattr(oidc.requests, "get", _get)

    with caplog.at_level(logging.WARNING):
        assert client._fetch_google_groups("alice@example.com") == []

    assert "Google Workspace group lookup failed" in caplog.text
    assert "error_type=RequestException" in caplog.text
    assert "sensitive-google-bearer-token" not in caplog.text
    assert "alice@example.com" not in caplog.text


def test_verify_id_token_uses_google_directory_groups_over_claim(monkeypatch):
    token = _signed_id_token(monkeypatch)
    client = _google_client()
    monkeypatch.setattr(
        client,
        "_fetch_google_groups",
        lambda email: [f"ml-team@example.com:{email}"],
    )

    payload = client.verify_id_token(id_token=token, nonce="nonce-1")

    assert payload["_kumo_idp_groups"] == [
        "ml-team@example.com:alice@example.com"
    ]


@pytest.mark.asyncio
async def test_refresh_groups_uses_google_directory_with_userinfo_email(
        monkeypatch):
    oidc._DISCOVERY_CACHE["https://idp.example.com"] = {
        "issuer": "https://idp.example.com",
        "token_endpoint": "https://idp.example.com/token",
        "userinfo_endpoint": "https://idp.example.com/userinfo",
    }
    monkeypatch.setattr(
        oidc.requests,
        "post",
        lambda *_args, **_kwargs: _Response({
            "access_token": "access-token",
            "refresh_token": "rotated-refresh-token",
        }),
    )
    monkeypatch.setattr(
        oidc.requests,
        "get",
        lambda *_args, **_kwargs: _Response({
            "email": " Alice@Example.COM ",
            "groups": ["ignored-oidc-group"],
        }),
    )
    client = _google_client()
    emails = []
    monkeypatch.setattr(
        client,
        "_fetch_google_groups",
        lambda email: emails.append(email) or ["workspace@example.com"],
    )

    result = await client.refresh_groups.coroutine(
        refresh_token="refresh-token")

    assert result.external_group_ids == ["workspace@example.com"]
    assert result.refresh_token == "rotated-refresh-token"
    assert emails == ["alice@example.com"]


@pytest.mark.asyncio
async def test_refresh_groups_uses_google_directory_with_id_token_email(
        monkeypatch):
    token = _signed_id_token(monkeypatch)
    oidc._DISCOVERY_CACHE["https://idp.example.com"] = {
        "issuer": "https://idp.example.com",
        "token_endpoint": "https://idp.example.com/token",
        "jwks_uri": "https://idp.example.com/jwks",
    }
    monkeypatch.setattr(
        oidc.requests,
        "post",
        lambda *_args, **_kwargs: _Response({
            "id_token": token,
        }),
    )
    client = _google_client()
    emails = []
    monkeypatch.setattr(
        client,
        "_fetch_google_groups",
        lambda email: emails.append(email) or ["workspace@example.com"],
    )

    result = await client.refresh_groups.coroutine(
        refresh_token="refresh-token")

    assert result.external_group_ids == ["workspace@example.com"]
    assert result.refresh_token == "refresh-token"
    assert emails == ["alice@example.com"]


@pytest.mark.asyncio
async def test_exchange_code_redacts_idp_error_response(monkeypatch, caplog):
    monkeypatch.setattr(
        oidc.requests,
        "get",
        lambda *_args, **_kwargs: _Response(
            {
                "issuer": "https://idp.example.com",
                "token_endpoint": "https://idp.example.com/token",
            }),
    )

    class _IdpErrorResponse:
        status_code = 401

        def raise_for_status(self):
            raise oidc.requests.HTTPError(
                "401 Client Error: sensitive-exception-token", response=self)

        def json(self):
            return {
                "error": "sensitive-invalid-client",
                "error_description":
                "sensitive-provider-description: Invalid client secret.",
            }

    monkeypatch.setattr(
        oidc.requests,
        "post",
        lambda *_args, **_kwargs: _IdpErrorResponse(),
    )
    client = DirectConnectOIDCClient(
        issuer="https://idp.example.com",
        client_id="kumo",
        client_secret="rotated-client-secret-value",
        groups_claim="groups",
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException) as exc_info:
            await client.exchange_code.coroutine(
                code="code",
                redirect_uri="https://kumo.example.com/auth/callback",
                code_verifier="verifier",
            )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == RestStatusCode.AUTH_INVALID_TOKEN.value
    assert "OIDC code exchange failed" in caplog.text
    assert "error_type=HTTPError" in caplog.text
    assert "status=401" in caplog.text
    assert "sensitive-invalid-client" not in caplog.text
    assert "sensitive-provider-description" not in caplog.text
    assert "sensitive-exception-token" not in caplog.text
    assert "rotated-client-secret-value" not in caplog.text


def test_verify_id_token_redacts_failure_reason(monkeypatch, caplog):
    token = _signed_id_token(monkeypatch, drop_claims=["exp"])

    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException):
            _client().verify_id_token(id_token=token, nonce="nonce-1")

    assert "OIDC id_token verification failed" in caplog.text
    assert "error_type=MissingRequiredClaimError" in caplog.text
    assert "status=unknown" in caplog.text
    assert "Token is missing the \"exp\" claim" not in caplog.text
