# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, List
from urllib.parse import urlencode

import jwt
import requests
from fastapi import HTTPException
from jwt.exceptions import PyJWKClientError

from async_utils.sync import blocking_method
from io_utils import logging
from rest.config.status import RestStatusCode
from rest.driver import direct_connect_auth_storage

log = logging.getLogger(__name__)

_OIDC_SCOPES = "openid email profile offline_access"
_DISCOVERY_CACHE: Dict[str, Dict[str, Any]] = {}
_JWKS_CLIENT_CACHE: Dict[str, jwt.PyJWKClient] = {}

# Google Workspace Directory API lookup. Google's OIDC tokens never carry
# group membership, so when a delegated service account is configured we
# fetch the user's groups per login instead of reading the groups claim.
_GOOGLE_DIRECTORY_SCOPE = (
    "https://www.googleapis.com/auth/admin.directory.group.readonly")
_GOOGLE_DIRECTORY_GROUPS_URL = (
    "https://admin.googleapis.com/admin/directory/v1/groups")
_GOOGLE_DIRECTORY_PAGE_SIZE = 200
_GOOGLE_SA_CREDS_CACHE: Dict[str, Any] = {}


def build_pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _normalize_issuer(value: Any) -> str:
    return str(value or "").rstrip("/")


def normalize_idp_groups(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [value.strip() for value in raw.split(",") if value.strip()]
    if isinstance(raw, list):
        result: List[str] = []
        for item in raw:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict) and "name" in item:
                result.append(str(item["name"]))
            elif item is not None:
                result.append(str(item))
        return result
    return []


def display_name_from_claims(payload: Dict[str, Any]) -> str:
    name = str(payload.get("name") or "").strip()
    if name:
        return name
    given = str(payload.get("given_name") or "").strip()
    family = str(payload.get("family_name") or "").strip()
    return f"{given} {family}".strip()


def _invalid_oidc_token(exc: BaseException, context: str) -> HTTPException:
    # Provider failures routinely contain request URLs, authorization codes,
    # refresh/access tokens, client credentials, or provider-controlled error
    # bodies.  Only emit fixed-shape diagnostics derived from local types and
    # the numeric HTTP status; never stringify the exception or response.
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    safe_status = status if isinstance(status, int) else "unknown"
    log.warning(
        "OIDC %s failed (error_type=%s status=%s)",
        context,
        type(exc).__name__,
        safe_status,
    )
    return HTTPException(
        HTTPStatus.UNAUTHORIZED,
        RestStatusCode.AUTH_INVALID_TOKEN.value,
    )


@dataclass(frozen=True)
class DirectConnectOIDCTokens:
    id_token: str
    refresh_token: str


@dataclass(frozen=True)
class DirectConnectOIDCGroupRefresh:
    external_group_ids: List[str]
    refresh_token: str


class DirectConnectOIDCClient:
    def __init__(
        self,
        *,
        issuer: str,
        client_id: str,
        client_secret: str,
        groups_claim: str,
        google_sa_key: str = "",
        google_admin_email: str = "",
    ) -> None:
        self._issuer = issuer.rstrip("/")
        self._client_id = client_id
        self._client_secret = client_secret
        self._groups_claim = groups_claim or "groups"
        self._google_sa_key = google_sa_key
        self._google_admin_email = google_admin_email
        if bool(google_sa_key) != bool(google_admin_email):
            raise ValueError(
                "Direct Connect google_groups requires both a service "
                "account key and an admin_email, or neither")
        if google_sa_key:
            try:
                json.loads(google_sa_key)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Direct Connect google_groups service account key is "
                    "not valid JSON") from exc

    @blocking_method()
    def discovery(self) -> Dict[str, Any]:
        return self._discovery()

    def _discovery(self) -> Dict[str, Any]:
        cached = _DISCOVERY_CACHE.get(self._issuer)
        if cached is not None:
            return cached
        response = requests.get(
            f"{self._issuer}/.well-known/openid-configuration",
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if _normalize_issuer(payload.get("issuer")) != self._issuer:
            raise HTTPException(500, "OIDC discovery issuer mismatch")
        _DISCOVERY_CACHE[self._issuer] = payload
        return payload

    async def authorization_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        nonce: str,
        code_challenge: str,
    ) -> str:
        discovery = await self.discovery.coroutine()
        auth_endpoint = discovery.get("authorization_endpoint")
        if not auth_endpoint:
            raise HTTPException(500, "OIDC authorization_endpoint missing")
        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "scope": _OIDC_SCOPES,
            "state": state,
            "nonce": nonce,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{auth_endpoint}?{urlencode(params)}"

    @blocking_method()
    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> DirectConnectOIDCTokens:
        discovery = self._discovery()
        token_endpoint = discovery.get("token_endpoint")
        if not token_endpoint:
            raise HTTPException(500, "OIDC token_endpoint missing")
        try:
            response = requests.post(
                token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code_verifier": code_verifier,
                },
                headers={"content-type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("OIDC token response must be a JSON object")
            id_token = payload.get("id_token")
            refresh_token = payload.get("refresh_token")
        except requests.RequestException as exc:
            raise _invalid_oidc_token(exc, "code exchange") from exc
        except (AttributeError, TypeError, ValueError) as exc:
            raise _invalid_oidc_token(exc, "code exchange") from exc
        if not id_token:
            raise HTTPException(401, "OIDC token response missing id_token")
        if not refresh_token:
            raise HTTPException(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "OIDC provider must issue a refresh token so current group "
                "membership can be resolved",
            )
        return DirectConnectOIDCTokens(
            id_token=str(id_token),
            refresh_token=str(refresh_token),
        )

    @blocking_method()
    def refresh_groups(
        self,
        *,
        refresh_token: str,
    ) -> DirectConnectOIDCGroupRefresh:
        discovery = self._discovery()
        token_endpoint = discovery.get("token_endpoint")
        if not token_endpoint:
            raise HTTPException(500, "OIDC token_endpoint missing")
        try:
            response = requests.post(
                token_endpoint,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"content-type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("OIDC token response must be a JSON object")
        except requests.RequestException as exc:
            raise _invalid_oidc_token(exc, "refresh token exchange") from exc
        except (AttributeError, TypeError, ValueError) as exc:
            raise _invalid_oidc_token(exc, "refresh token exchange") from exc

        groups: List[str] | None = None
        user_email: str | None = None
        access_token = payload.get("access_token")
        userinfo_endpoint = discovery.get("userinfo_endpoint")
        if access_token and userinfo_endpoint:
            try:
                userinfo_response = requests.get(
                    str(userinfo_endpoint),
                    headers={"authorization": f"Bearer {access_token}"},
                    timeout=10,
                )
                userinfo_response.raise_for_status()
                userinfo = userinfo_response.json()
                if not isinstance(userinfo, dict):
                    raise ValueError("OIDC userinfo response must be an object")
                if self._google_sa_key and self._google_admin_email:
                    if userinfo.get("email"):
                        user_email = direct_connect_auth_storage.\
                            normalize_direct_connect_email(
                                str(userinfo["email"]))
                else:
                    groups = normalize_idp_groups(
                        userinfo.get(self._groups_claim))
            except requests.RequestException as exc:
                raise _invalid_oidc_token(exc,
                                          "userinfo group refresh") from exc
            except (AttributeError, TypeError, ValueError) as exc:
                raise _invalid_oidc_token(exc,
                                          "userinfo group refresh") from exc

        if user_email is not None:
            groups = self._fetch_google_groups(user_email)
        elif groups is None:
            id_token = payload.get("id_token")
            if not id_token:
                raise HTTPException(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "OIDC refresh response has neither usable userinfo nor "
                    "an ID token for group resolution",
                )
            claims = self._decode_id_token(str(id_token))
            if self._google_sa_key and self._google_admin_email:
                if not claims.get("email"):
                    raise HTTPException(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "OIDC refresh ID token is missing email for Google "
                        "Workspace group resolution",
                    )
                groups = self._fetch_google_groups(
                    direct_connect_auth_storage.normalize_direct_connect_email(
                        str(claims["email"])))
            else:
                groups = normalize_idp_groups(claims.get(self._groups_claim))

        return DirectConnectOIDCGroupRefresh(
            external_group_ids=list(dict.fromkeys(groups)),
            refresh_token=str(payload.get("refresh_token") or refresh_token),
        )

    @blocking_method()
    def verify_id_token(
        self,
        *,
        id_token: str,
        nonce: str,
    ) -> Dict[str, Any]:
        payload = self._decode_id_token(id_token)
        if payload.get("nonce") != nonce:
            raise HTTPException(401, "Invalid OIDC nonce")
        if not payload.get("sub"):
            raise HTTPException(401, "OIDC token missing subject")
        if not payload.get("email"):
            raise HTTPException(401, "OIDC token missing email")
        payload["email"] = direct_connect_auth_storage.\
            normalize_direct_connect_email(str(payload["email"]))
        payload["_kumo_idp_groups"] = normalize_idp_groups(
            self._resolve_raw_idp_groups(payload))
        return payload

    def _decode_id_token(self, id_token: str) -> Dict[str, Any]:
        discovery = self._discovery()
        jwks_uri = discovery.get("jwks_uri")
        if not jwks_uri:
            raise HTTPException(500, "OIDC jwks_uri missing")
        jwks_client = _JWKS_CLIENT_CACHE.setdefault(str(jwks_uri),
                                                    jwt.PyJWKClient(jwks_uri))
        try:
            signing_key = jwks_client.get_signing_key_from_jwt(id_token).key
            payload = jwt.decode(
                id_token,
                signing_key,
                algorithms=["RS256"],
                audience=self._client_id,
                issuer=discovery.get("issuer", self._issuer),
                options={"require": ["exp"]},
                leeway=60,
            )
        except (jwt.exceptions.InvalidTokenError, PyJWKClientError,
                ValueError) as exc:
            raise _invalid_oidc_token(exc, "id_token verification") from exc
        return payload

    def _resolve_raw_idp_groups(self, payload: Dict[str, Any]) -> Any:
        if self._google_sa_key and self._google_admin_email:
            return self._fetch_google_groups(str(payload["email"]))
        return payload.get(self._groups_claim)

    def _google_credentials(self) -> Any:
        cache_key = hashlib.sha256(
            f"{self._google_admin_email}\x00{self._google_sa_key}".encode(
            )).hexdigest()
        creds = _GOOGLE_SA_CREDS_CACHE.get(cache_key)
        if creds is None:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_info(
                json.loads(self._google_sa_key),
                scopes=[_GOOGLE_DIRECTORY_SCOPE],
                subject=self._google_admin_email,
            )
            _GOOGLE_SA_CREDS_CACHE[cache_key] = creds
        return creds

    def _fetch_google_groups(self, email: str) -> List[str]:
        """Fetch the user's Google Workspace groups (as group emails).

        Failures degrade to "no groups" rather than blocking the login:
        IdP groups only feed group-based role mapping, and a hard failure
        here would lock every user out on a Directory API hiccup.
        """
        try:
            import google.auth.transport.requests as google_auth_requests
            creds = self._google_credentials()
            if not creds.valid:
                creds.refresh(google_auth_requests.Request())
            groups: List[str] = []
            page_token = ""
            while True:
                params: Dict[str, Any] = {
                    "userKey": email,
                    "maxResults": _GOOGLE_DIRECTORY_PAGE_SIZE,
                }
                if page_token:
                    params["pageToken"] = page_token
                response = requests.get(
                    _GOOGLE_DIRECTORY_GROUPS_URL,
                    params=params,
                    headers={"Authorization": f"Bearer {creds.token}"},
                    timeout=10,
                )
                response.raise_for_status()
                body = response.json()
                groups.extend(
                    str(group["email"]) for group in body.get("groups") or []
                    if isinstance(group, dict) and group.get("email"))
                page_token = str(body.get("nextPageToken") or "")
                if not page_token:
                    return groups
        except Exception as exc:
            log.warning(
                "Google Workspace group lookup failed "
                "(error_type=%s); continuing login with no IdP groups",
                type(exc).__name__,
            )
            return []
