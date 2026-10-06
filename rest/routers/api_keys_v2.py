# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from rest.config.api_key import (
    APIKeySummary,
    CreateAdminAPIKeyRequest,
    CreateAPIKeyResponse,
    CreateUserAPIKeyRequest,
)
from rest.deps.authentication import (
    Authentication,
    get_user_authorization_from_request,
)
from rest.driver.api_key_driver import APIKeyDriver
from service_framework.deploy_stage import deploy_stage
from statedb.config.common import APIKeyID
from statedb.rest.api_key import APIKeyType

_API_KEY_STAGING_ENV_VAR = "KUMO_ENABLE_UNAUTHENTICATED_API_KEY_STAGING"
_STAGED_API_KEY_OWNER = "phase1b-api-key-staging"


def _api_key_staging_enabled() -> bool:
    enabled = os.getenv(_API_KEY_STAGING_ENV_VAR, "").lower() in {
        "1",
        "true",
        "yes",
    }
    if enabled:
        stage = deploy_stage()
        if not (stage.is_dev or stage.is_test):
            raise RuntimeError(
                'Unauthenticated API key staging is restricted to local '
                'development and test deployments')
    return enabled


def _get_request_user_id(request: Request) -> str:
    """Return the authenticated user ID attached to this request."""
    user_authorization = get_user_authorization_from_request(request)
    return str(getattr(user_authorization, "user_id", ""))


class UserAPIKeyResource:
    """Expose owner-bound StateDB-backed API key endpoints."""
    def __init__(
        self,
        auth: Authentication,
        api_key_driver: APIKeyDriver,
    ) -> None:
        _api_key_staging_enabled()
        self._api_key_driver = api_key_driver
        self.router = APIRouter(
            prefix="/v0/api-keys",
            tags=["API Keys"],
            responses={HTTPStatus.NOT_FOUND: {
                "description": "Not found"
            }},
        )

        self.router.add_api_route(
            "/admin",
            self.list_admin_api_keys,
            response_model=List[APIKeySummary],
            dependencies=[Depends(auth.require_customer_admin)],
            methods=["GET"],
        )

        self.router.add_api_route(
            "/admin",
            self.create_admin_api_key,
            response_model=CreateAPIKeyResponse,
            dependencies=[Depends(auth.require_customer_admin)],
            status_code=HTTPStatus.CREATED,
            methods=["POST"],
        )

        self.router.add_api_route(
            "/admin/{id}",
            self.delete_admin_api_key,
            dependencies=[Depends(auth.require_customer_admin)],
            response_model=None,
            status_code=HTTPStatus.NO_CONTENT,
            methods=["DELETE"],
        )

        self.router.add_api_route(
            "",
            self.list_user_api_keys,
            response_model=List[APIKeySummary],
            dependencies=[Depends(auth.authenticate_request_header)],
            methods=["GET"],
        )

        self.router.add_api_route(
            "",
            self.create_user_api_key,
            response_model=CreateAPIKeyResponse,
            dependencies=[Depends(auth.authenticate_request_header)],
            status_code=HTTPStatus.CREATED,
            methods=["POST"],
        )

        self.router.add_api_route(
            "/stage",
            self.create_staged_user_api_key,
            response_model=CreateAPIKeyResponse,
            status_code=HTTPStatus.CREATED,
            methods=["POST"],
            include_in_schema=False,
        )

        self.router.add_api_route(
            "/stage/{id}",
            self.delete_staged_user_api_key,
            response_model=None,
            status_code=HTTPStatus.NO_CONTENT,
            methods=["DELETE"],
            include_in_schema=False,
        )

        self.router.add_api_route(
            "/{id}",
            self.delete_user_api_key,
            response_model=None,
            dependencies=[Depends(auth.authenticate_request_header)],
            status_code=HTTPStatus.NO_CONTENT,
            methods=["DELETE"],
        )

    async def _create_api_key(
        self,
        *,
        request: Request,
        name: str,
        type: Optional[APIKeyType] = None,
    ) -> CreateAPIKeyResponse:
        """Create either a user-scoped or admin-scoped API key."""
        user_authorization = get_user_authorization_from_request(request)
        user_id = str(getattr(user_authorization, "user_id", ""))
        user_name = str(getattr(user_authorization, "user_name", ""))
        if type is None:
            return await self._api_key_driver.create_user_api_key(
                name=name,
                user_id=user_id,
                user_name=user_name,
            )
        return await self._api_key_driver.create_admin_api_key(
            name=name,
            type=type,
            user_id=user_id,
            user_name=user_name,
        )

    async def list_admin_api_keys(
            self,
            request: Request,
            type: APIKeyType = Query(...),
            search: Optional[str] = Query(default=None),
    ) -> List[APIKeySummary]:
        """List API keys visible to an admin caller."""
        return await self._api_key_driver.list_admin_api_keys(
            type=type,
            search=search,
        )

    async def create_admin_api_key(
        self,
        payload: CreateAdminAPIKeyRequest,
        request: Request,
    ) -> CreateAPIKeyResponse:
        """Create an admin-managed API key for the current user."""
        return await self._create_api_key(
            request=request,
            name=payload.name,
            type=payload.type,
        )

    async def delete_admin_api_key(self, id: APIKeyID,
                                   request: Request) -> Response:
        """Delete an admin-managed API key by identifier."""
        await self._api_key_driver.delete_admin_api_key(api_key_id=id, )
        return Response(status_code=HTTPStatus.NO_CONTENT)

    async def list_user_api_keys(self, request: Request) -> List[APIKeySummary]:
        """List API keys owned by the authenticated user."""
        return await self._api_key_driver.list_user_api_keys(
            user_id=_get_request_user_id(request), )

    async def create_user_api_key(
        self,
        payload: CreateUserAPIKeyRequest,
        request: Request,
    ) -> CreateAPIKeyResponse:
        """Create a user-scoped API key for the authenticated user."""
        return await self._create_api_key(
            request=request,
            name=payload.name,
        )

    async def create_staged_user_api_key(
        self,
        payload: CreateUserAPIKeyRequest,
        request: Request,
    ) -> CreateAPIKeyResponse:
        """Create a test bootstrap key when explicitly enabled for staging."""
        if not _api_key_staging_enabled():
            raise HTTPException(HTTPStatus.NOT_FOUND, "Not found")
        return await self._api_key_driver.create_user_api_key(
            name=payload.name,
            user_id=_STAGED_API_KEY_OWNER,
            user_name=_STAGED_API_KEY_OWNER,
        )

    async def delete_staged_user_api_key(self, id: APIKeyID,
                                         request: Request) -> Response:
        """Delete a staged test bootstrap key when explicitly enabled."""
        if not _api_key_staging_enabled():
            raise HTTPException(HTTPStatus.NOT_FOUND, "Not found")
        await self._api_key_driver.delete_user_api_key(
            api_key_id=id,
            user_id=_STAGED_API_KEY_OWNER,
        )
        return Response(status_code=HTTPStatus.NO_CONTENT)

    async def delete_user_api_key(self, id: APIKeyID,
                                  request: Request) -> Response:
        """Delete a user-scoped API key by identifier."""
        await self._api_key_driver.delete_user_api_key(
            api_key_id=id,
            user_id=_get_request_user_id(request),
        )
        return Response(status_code=HTTPStatus.NO_CONTENT)
