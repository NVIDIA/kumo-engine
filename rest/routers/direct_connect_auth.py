# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import secrets
from datetime import timedelta
from http import HTTPStatus
from typing import Optional, Sequence
from urllib.parse import quote, unquote, urlparse

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse

from rest.config.encryption import decrypt_secret_string, encrypt_secret_string
from rest.config.status import RestStatusCode
from rest.deps.authentication import get_user_authorization_from_request
from rest.deps.direct_connect_group_resolver import DirectConnectGroupResolver
from rest.deps.direct_connect_oidc_client import (
    DirectConnectOIDCClient,
    build_pkce_challenge,
    display_name_from_claims,
)
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from rest.utils.frontend_proxy_metadata import clear_cookie as _clear_cookie
from rest.utils.frontend_proxy_metadata import client_ip as _client_ip
from rest.utils.frontend_proxy_metadata import redirect_uri as _redirect_uri
from rest.utils.frontend_proxy_metadata import set_cookie as _set_cookie
from statedb.config.direct_connect_auth import (
    DirectConnectMeResponse,
    DirectConnectPasswordLoginRequest,
    DirectConnectPasswordLoginResponse,
    DirectConnectPasswordResetRequest,
)
from statedb.config.timestamp import now_utc

DIRECT_CONNECT_SESSION_COOKIE = "kumo_direct_connect_session"
DIRECT_CONNECT_LOGIN_COOKIE = "kumo_direct_connect_login"
DIRECT_CONNECT_SESSION_SECONDS = 8 * 60 * 60
DIRECT_CONNECT_LOGIN_SECONDS = 10 * 60
DIRECT_CONNECT_PASSWORD_RESET_SECONDS = 10 * 60
_RETURN_TO_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _password_login_enabled(login_method: Optional[str] = None) -> bool:
    return (login_method or "oidc").lower() == "password"


def _oidc_login_enabled(login_method: Optional[str] = None) -> bool:
    return (login_method or "oidc").lower() == "oidc"


def _origin(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _safe_relative_return_to(value: str) -> str:
    if _RETURN_TO_CONTROL_CHARS.search(value):
        return "/"
    decoded = unquote(value)
    if ("\\" in value or "\\" in decoded
            or _RETURN_TO_CONTROL_CHARS.search(decoded)):
        return "/"

    parsed = urlparse(decoded)
    if (parsed.scheme or parsed.netloc or not parsed.path.startswith("/")
            or parsed.path.startswith("//")):
        return "/"

    path = quote(parsed.path, safe="/!$&'()*+,;=:@~-._")
    return path + (("?" + parsed.query) if parsed.query else "")


def _safe_return_to(
    return_to: Optional[str],
    request: Optional[Request],
    allowed_return_origins: Optional[Sequence[str]] = None,
) -> str:
    if not return_to:
        return "/"

    parsed = urlparse(return_to)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "/"

        allowed_origins = {
            origin.strip()
            for origin in (allowed_return_origins or ()) if origin.strip()
        }
        if request is not None:
            allowed_origins.add(_origin(str(request.base_url)))
        if _origin(return_to) not in allowed_origins:
            return "/"
        return _safe_relative_return_to(parsed.path + (
            ("?" + parsed.query) if parsed.query else ""))

    return _safe_relative_return_to(return_to)


def _claim_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    # OIDC claims are provider-controlled input. Unknown strings, numbers,
    # collections, and other truthy values must not become authorization
    # facts, especially for first-admin bootstrap.
    return None


class DirectConnectAuthResource:
    def __init__(
        self,
        auth_store: DirectConnectAuthStore,
        *,
        group_resolver: Optional[DirectConnectGroupResolver] = None,
        login_method: Optional[str] = None,
        allowed_return_origins: Optional[Sequence[str]] = None,
        password_login_ip_rate_limit_enabled: bool = True,
    ) -> None:
        self._auth_store = auth_store
        self._group_resolver = group_resolver
        self._login_method = login_method
        self._allowed_return_origins = tuple(allowed_return_origins or ())
        self._password_login_ip_rate_limit_enabled = (
            password_login_ip_rate_limit_enabled)
        self.router = APIRouter(prefix="/v0/auth", tags=["Auth"])
        self.router.add_api_route(
            "/login",
            self.login,
            methods=["GET"],
            response_class=RedirectResponse,
            include_in_schema=False,
        )
        self.router.add_api_route(
            "/callback",
            self.callback,
            methods=["GET"],
            response_class=RedirectResponse,
            include_in_schema=False,
        )

        self.router.add_api_route(
            "/password/login",
            self.password_login,
            methods=["POST"],
            response_model=DirectConnectPasswordLoginResponse,
        )
        self.router.add_api_route(
            "/password/complete-reset",
            self.complete_password_reset,
            methods=["POST"],
            response_model=DirectConnectPasswordLoginResponse,
        )
        self.router.add_api_route(
            "/me",
            self.me,
            methods=["GET"],
            response_model=DirectConnectMeResponse,
        )
        self.router.add_api_route(
            "/logout",
            self.logout,
            methods=["POST"],
            response_class=RedirectResponse,
            include_in_schema=False,
        )

    async def _create_session_response(
        self,
        request: Request,
        response,
        *,
        user_id: str,
    ) -> None:
        session_id = await self._auth_store.create_session(
            user_id=user_id,
            expires_time=now_utc() +
            timedelta(seconds=DIRECT_CONNECT_SESSION_SECONDS),
        )
        _set_cookie(
            response,
            request,
            name=DIRECT_CONNECT_SESSION_COOKIE,
            value=session_id,
            max_age=DIRECT_CONNECT_SESSION_SECONDS,
        )
        await self._auth_store.cleanup_expired_sessions()
        await self._auth_store.cleanup_expired_password_reset_tokens()

    async def _build_client(self) -> DirectConnectOIDCClient:
        config = await self._auth_store.get_config()
        if config is None or not config.enabled:
            raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE,
                                "Direct Connect auth is not configured")
        client_secret = await self._auth_store.get_client_secret()
        google_sa_key, google_admin_email = (
            await self._auth_store.get_google_groups_settings())
        return DirectConnectOIDCClient(
            issuer=config.issuer,
            client_id=config.client_id,
            client_secret=client_secret,
            groups_claim=config.groups_claim,
            google_sa_key=google_sa_key,
            google_admin_email=google_admin_email,
        )

    async def login(
            self,
            request: Request,
            return_to: Optional[str] = Query(default="/"),
    ) -> RedirectResponse:
        if not _oidc_login_enabled(self._login_method):
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                "Direct Connect OIDC login is not enabled")
        client = await self._build_client()
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64)
        redirect_uri = _redirect_uri(request)
        payload = {
            "state":
            state,
            "nonce":
            nonce,
            "code_verifier":
            code_verifier,
            "redirect_uri":
            redirect_uri,
            "return_to":
            _safe_return_to(
                return_to,
                request,
                allowed_return_origins=self._allowed_return_origins,
            ),
        }
        auth_url = await client.authorization_url(
            redirect_uri=redirect_uri,
            state=state,
            nonce=nonce,
            code_challenge=build_pkce_challenge(code_verifier),
        )
        response = RedirectResponse(auth_url)
        _set_cookie(
            response,
            request,
            name=DIRECT_CONNECT_LOGIN_COOKIE,
            value=encrypt_secret_string(json.dumps(payload)),
            max_age=DIRECT_CONNECT_LOGIN_SECONDS,
            path="/v0/auth",
        )
        return response

    async def callback(
            self,
            request: Request,
            code: str = Query(...),
            state: str = Query(...),
    ) -> RedirectResponse:
        if not _oidc_login_enabled(self._login_method):
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                "Direct Connect OIDC login is not enabled")
        encrypted_login = request.cookies.get(DIRECT_CONNECT_LOGIN_COOKIE)
        if not encrypted_login:
            raise HTTPException(HTTPStatus.UNAUTHORIZED,
                                RestStatusCode.AUTH_INVALID_TOKEN.value)
        try:
            login_payload = json.loads(decrypt_secret_string(encrypted_login))
        except Exception as exc:
            raise HTTPException(
                HTTPStatus.UNAUTHORIZED,
                RestStatusCode.AUTH_INVALID_TOKEN.value) from exc
        if login_payload.get("state") != state:
            raise HTTPException(HTTPStatus.UNAUTHORIZED,
                                RestStatusCode.AUTH_INVALID_TOKEN.value)

        client = await self._build_client()
        tokens = await client.exchange_code.coroutine(
            code=code,
            redirect_uri=login_payload["redirect_uri"],
            code_verifier=login_payload["code_verifier"],
        )
        claims = await client.verify_id_token.coroutine(
            id_token=tokens.id_token,
            nonce=login_payload["nonce"],
        )

        config = await self._auth_store.get_config()
        if config is None:
            raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE,
                                "Direct Connect auth is not configured")
        idp_groups = list(claims.get("_kumo_idp_groups", []))
        user_id, user = await self._auth_store.upsert_user_from_idp(
            issuer=config.issuer,
            subject=str(claims["sub"]),
            email=str(claims["email"]),
            display_name=display_name_from_claims(claims),
            refresh_token=tokens.refresh_token,
            email_verified=_claim_bool(claims.get("email_verified")),
        )
        if user.disabled:
            raise HTTPException(HTTPStatus.FORBIDDEN,
                                RestStatusCode.INVALID_PERMISSIONS.value)

        response = RedirectResponse(login_payload.get("return_to") or "/")
        _clear_cookie(response, request, name=DIRECT_CONNECT_LOGIN_COOKIE,
                      path="/v0/auth")
        if self._group_resolver is not None:
            self._group_resolver.remember_external_groups(user_id, idp_groups)
        await self._create_session_response(
            request,
            response,
            user_id=user_id,
        )
        return response

    async def password_login(
        self,
        request: Request,
        payload: DirectConnectPasswordLoginRequest,
        response: Response,
    ) -> DirectConnectPasswordLoginResponse:
        if not _password_login_enabled(self._login_method):
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                "Direct Connect password login is not enabled")
        # An arbitrary customer ingress does not provide a portable trusted
        # original-client-IP contract. In that topology its peer address would
        # collapse all users into one source-IP bucket, allowing one attacker
        # to lock password login for the tenant. The deployment can therefore
        # disable only the IP-wide bucket while retaining the email bucket.
        source_ip = (_client_ip(request)
                     if self._password_login_ip_rate_limit_enabled else None)
        await self._auth_store.check_password_login_allowed(
            email=payload.email, source_ip=source_ip)
        try:
            (status, reset_token, user_id,
             _user) = await self._auth_store.password_login(
                 email=payload.email,
                 password=payload.password,
                 reset_expires_time=now_utc() +
                 timedelta(seconds=DIRECT_CONNECT_PASSWORD_RESET_SECONDS),
             )
        except HTTPException as exc:
            if exc.status_code in {
                    HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN
            }:
                await self._auth_store.record_password_login_failure(
                    email=payload.email, source_ip=source_ip)
                raise HTTPException(
                    HTTPStatus.UNAUTHORIZED,
                    RestStatusCode.AUTH_WRONG_CREDENTIALS.value) from exc
            raise
        await self._auth_store.record_password_login_success(email=payload.email
                                                             )
        if status == "password_reset_required":
            await self._auth_store.cleanup_expired_password_reset_tokens()
            return DirectConnectPasswordLoginResponse(
                status="password_reset_required",
                reset_token=reset_token,
            )
        await self._create_session_response(
            request,
            response,
            user_id=user_id,
        )
        return DirectConnectPasswordLoginResponse(status="ok")

    async def complete_password_reset(
        self,
        request: Request,
        payload: DirectConnectPasswordResetRequest,
        response: Response,
    ) -> DirectConnectPasswordLoginResponse:
        if not _password_login_enabled(self._login_method):
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                "Direct Connect password login is not enabled")
        user_id, _user = await self._auth_store.complete_password_reset(
            reset_token=payload.reset_token,
            new_password=payload.new_password,
        )
        await self._create_session_response(
            request,
            response,
            user_id=user_id,
        )
        return DirectConnectPasswordLoginResponse(status="ok")

    async def me(self, request: Request) -> DirectConnectMeResponse:
        user_auth = get_user_authorization_from_request(request)
        if user_auth.error:
            raise HTTPException(HTTPStatus.UNAUTHORIZED, user_auth.error)
        user = await self._auth_store.get_user(str(user_auth.user_id))
        if user is None:
            raise HTTPException(HTTPStatus.UNAUTHORIZED,
                                RestStatusCode.AUTH_INVALID_TOKEN.value)
        return DirectConnectMeResponse(
            user_id=str(user_auth.user_id),
            email=user.email,
            display_name=user.display_name,
            role=user.role,
            permissions=list(user_auth.permissions),
            kumo_group_ids=list(user_auth.kumo_group_ids),
            project_ids=list(user_auth.project_ids),
        )

    async def logout(self, request: Request) -> RedirectResponse:
        session_id = request.cookies.get(DIRECT_CONNECT_SESSION_COOKIE)
        if session_id:
            await self._auth_store.delete_session(session_id)
        await self._auth_store.cleanup_expired_sessions()
        response = RedirectResponse("/", status_code=HTTPStatus.SEE_OTHER)
        _clear_cookie(response, request, name=DIRECT_CONNECT_SESSION_COOKIE)
        return response
