# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Callable, Optional

from fastapi import APIRouter, Depends, Request, Security
from fastapi.exceptions import HTTPException
from fastapi.security.api_key import APIKeyHeader

from io_utils import logging
from rest.config.api_key import ParsedAPIKey
from rest.config.authentication import ClientAuthConfigResponse
from rest.config.status import RestStatusCode
from rest.deps.authentication import (
    Authentication,
    GroupResolverProtocol,
    ProjectResolver,
    UserAuthorization,
    resolve_api_key_user_authorization_context,
    set_request_user_authorization,
)
from rest.driver.api_key_driver import APIKeyDriver

log = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    if not api_key_header:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN,
                            detail="Not authenticated (API Key missing)")
    return api_key_header


def get_optional_api_key(
        api_key_header: Optional[str] = Security(API_KEY_HEADER)):
    return api_key_header


def _reject_invalid_api_key_format(api_key: str) -> None:
    try:
        ParsedAPIKey.parse(api_key)
    except ValueError as ex:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail=RestStatusCode.AUTH_INVALID_API_KEY.value,
        ) from ex


def _reuse_public_middleware_authorization(
    request: Request,
    api_key: str,
) -> bool:
    """Return True when request.state auth can be reused for this API key."""
    # Public middleware resolves header-based API key auth and stores the
    # result in request.state. Reuse that result to avoid double resolution.
    header_api_key = request.headers.get("X-API-Key")
    if not header_api_key or api_key != header_api_key:
        return False

    user_authorization = getattr(request.state, "user_authorization", None)
    if user_authorization is None:
        return False

    error_detail = getattr(user_authorization, "error", None)
    if error_detail:
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED,
                            detail=error_detail)
    return True


def api_key_auth(
    api_key_driver: Optional[APIKeyDriver] = None,
    group_resolver: Optional[GroupResolverProtocol] = None,
    project_resolver: Optional[ProjectResolver] = None,
    auth_disabled: bool = False,
    auth_proxied: bool = False,
) -> Callable:
    log.info(f"Using auth_disabled: {auth_disabled}")

    async def _api_key_auth(
            request: Request,
            api_key: Optional[str] = Depends(get_optional_api_key),
    ) -> None:
        if not api_key:
            if not auth_disabled:
                raise HTTPException(
                    status_code=HTTPStatus.UNAUTHORIZED,
                    detail="Not authenticated (API Key missing)",
                )
            # Public middleware has already resolved current native groups and
            # projects for this ordinary disabled-auth request user. Preserve
            # that request-local authority instead of replacing it with the
            # legacy admin-shaped disabled sentinel.
            user_authorization = getattr(request.state, 'user_authorization',
                                         None)
            if (user_authorization is None
                    or not user_authorization.is_disabled):
                user_authorization = UserAuthorization.disabled_request()
            set_request_user_authorization(request, user_authorization)
            return

        if not auth_disabled:
            _reject_invalid_api_key_format(api_key)

        if _reuse_public_middleware_authorization(request, api_key):
            return

        user_authorization = await resolve_api_key_user_authorization_context(
            api_key=api_key,
            auth_disabled=auth_disabled,
            auth_proxied=auth_proxied,
            api_key_driver=api_key_driver,
            group_resolver=group_resolver,
            project_resolver=project_resolver,
        )
        set_request_user_authorization(request, user_authorization)

    return _api_key_auth


class AuthenticationResource:
    def __init__(self, auth: Authentication) -> None:
        self._auth = auth
        self.router = APIRouter(
            prefix="/v0/authentication",
            tags=["Authentication"],
            responses={
                HTTPStatus.NOT_FOUND: {
                    "description": "Not found (AUTH)"
                }
            },
        )
        self._auth_req_header = auth.authenticate_request_header

        self.router.add_api_route(
            "/client-config",
            self.get_client_config,
            response_model=ClientAuthConfigResponse,
            methods=['GET'],
        )

    async def get_client_config(self) -> ClientAuthConfigResponse:
        return ClientAuthConfigResponse(
            auth_disabled=self._auth.settings.disable,
            direct_connect_login_method=(
                self._auth.settings.direct_connect_login_method),
        )
