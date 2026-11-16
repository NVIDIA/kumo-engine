# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from io_utils import logging
from rest.config.authentication import (
    AdminInformation,
    AdminResponse,
    OrganizationMember,
)
from rest.deps.authentication import (
    Authentication,
    get_user_authorization_from_request,
)
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from statedb.config.direct_connect_auth import (
    DirectConnectSetPasswordRequest,
    DirectConnectUserCreateRequest,
    DirectConnectUserResponse,
    DirectConnectUserUpdateRequest,
)

log = logging.getLogger(__name__)


class AdminResource:
    def __init__(
        self,
        auth: Authentication,
        direct_connect_auth_store: Optional[DirectConnectAuthStore] = None,
        direct_connect_login_method: Optional[str] = None,
    ) -> None:
        self._auth = auth
        self._direct_connect_auth_store = direct_connect_auth_store
        self._direct_connect_login_method = (
            direct_connect_login_method
            or auth.settings.direct_connect_login_method)
        self.router = APIRouter(
            prefix="/v0/admin",
            tags=["Admin"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={
                HTTPStatus.NOT_FOUND: {
                    "description": "Not found (ADMIN)"
                }
            },
        )
        # Doesn't require admin, but an admin check is done in the code to
        # conditionally exclude admin-only parts of the response
        self.router.add_api_route(
            "/info",
            self.get_admin_info,
            response_model=AdminResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/members",
            self.get_organization_members,
            dependencies=[Depends(auth.require_customer_admin)],
            response_model=List[OrganizationMember],
            methods=['GET'],
        )

        self.router.add_api_route(
            "/users",
            self.list_direct_connect_users,
            dependencies=[Depends(auth.require_customer_admin)],
            response_model=List[DirectConnectUserResponse],
            methods=['GET'],
        )

        self.router.add_api_route(
            "/users",
            self.create_direct_connect_user,
            dependencies=[Depends(auth.require_customer_admin)],
            response_model=DirectConnectUserResponse,
            methods=['POST'],
        )

        self.router.add_api_route(
            "/users/{user_id}",
            self.update_direct_connect_user,
            dependencies=[Depends(auth.require_customer_admin)],
            response_model=DirectConnectUserResponse,
            methods=['PATCH'],
        )

        self.router.add_api_route(
            "/users/{user_id}/password",
            self.set_direct_connect_user_password,
            dependencies=[Depends(auth.require_customer_admin)],
            response_model=DirectConnectUserResponse,
            methods=['POST'],
        )

    @property
    def _dc_store(self) -> Optional[DirectConnectAuthStore]:
        # Auth-disabled never configures the store, so treat it as absent and let
        # Direct Connect user management degrade. Under auth-enabled an
        # unconfigured store is a real startup error -- let it surface.
        store = self._direct_connect_auth_store
        if store is None or (self._auth.settings.disable
                             and not store.is_configured):
            return None
        return store

    async def get_organization_members(
            self, request: Request) -> List[OrganizationMember]:
        """
        Get information about the members of the user's organization.
        """
        store = self._dc_store
        if store is None:
            return []
        users = await store.list_users()
        return [
            OrganizationMember(
                name=user.display_name or user.email,
                email=user.email,
                uid=user.user_id,
                roles=['Admin'] if user.role == 'admin' else ['User'],
            ) for user in users
        ]

    async def list_direct_connect_users(
            self, request: Request) -> List[DirectConnectUserResponse]:
        store = self._dc_store
        if store is None:
            return []
        return await store.list_users()

    async def create_direct_connect_user(
        self,
        payload: DirectConnectUserCreateRequest,
        request: Request,
    ) -> DirectConnectUserResponse:
        store = self._dc_store
        if store is None:
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                'Direct Connect users are not enabled')
        if self._direct_connect_password_login_enabled():
            if not payload.password:
                raise HTTPException(HTTPStatus.BAD_REQUEST,
                                    'Temporary password is required')
        elif payload.password is not None:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                'Password is only supported when Direct Connect password login is enabled',
            )
        return await store.create_local_user(
            email=payload.email,
            display_name=payload.display_name,
            role=payload.role,
            disabled=payload.disabled,
            password=payload.password,
        )

    def _direct_connect_password_login_enabled(self) -> bool:
        return self._direct_connect_login_method.lower() == "password"

    async def update_direct_connect_user(
        self,
        user_id: str,
        payload: DirectConnectUserUpdateRequest,
        request: Request,
    ) -> DirectConnectUserResponse:
        store = self._dc_store
        if store is None:
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                'Direct Connect users are not enabled')
        return await store.update_user(
            user_id,
            role=payload.role,
            disabled=payload.disabled,
        )

    async def set_direct_connect_user_password(
        self,
        user_id: str,
        payload: DirectConnectSetPasswordRequest,
        request: Request,
    ) -> DirectConnectUserResponse:
        store = self._dc_store
        if store is None:
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                'Direct Connect users are not enabled')
        return await store.set_temporary_password(
            user_id,
            payload.password,
        )

    async def get_admin_info(self, request: Request) -> AdminResponse:
        """ Get information about the Organization administrators"""
        user_authorization = get_user_authorization_from_request(request)
        return AdminResponse(
            organization_name='Kumo',
            user_is_customer_admin=user_authorization.is_admin,
            customer_admin_emails=[],
            customer_admin_info=AdminInformation(
                api_enabled=True) if user_authorization.is_admin else None,
        )
