# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from http import HTTPStatus

import kumoapi.data_source as api
from fastapi import APIRouter, Depends, Request

from rest.deps.authentication import (
    Authentication,
    get_user_authorization_from_request,
)
from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.routers.identity_rotation import preserves_s3_ca_certificate
from rest.utils.public import wrap_driver_errors
from statedb.config.common import IdentityID, ProjectID
from statedb.utils.errors import IdentityConflictError, IdentityInUseError


class IdentityRouter:
    """SDK identity discovery plus administrator lifecycle management.

    Project-scoped discovery (``/projects/{project_id}/identities``) is
    available to any project member. The administrator lifecycle routes
    (``/identities``) mirror the UI's admin surface and require a
    customer-admin API key, enforced by ``require_customer_admin``.
    """
    def __init__(self, api_key_auth: Callable, driver: ConnectorApiDriver,
                 auth: Authentication) -> None:
        self._driver = driver
        self.router = APIRouter(
            prefix='/projects/{project_id}/identities',
            tags=['API v1: Identities'],
            dependencies=[Depends(api_key_auth)],
        )
        self.admin_router = APIRouter(
            prefix='/identities',
            tags=['API v1: Identity Administration'],
            dependencies=[
                Depends(api_key_auth),
                Depends(auth.require_customer_admin),
            ],
            responses={HTTPStatus.NOT_FOUND: {
                'description': 'Not found'
            }},
        )

        # Project-scoped discovery (any project member).
        self.router.add_api_route(
            '/by-slug/{slug}', self.get_available_by_slug, methods=['GET'],
            response_model=api.IdentityResponse,
            summary='Get an identity available to a project by slug.')
        self.router.add_api_route(
            '/{identity_id}', self.get_available_by_id, methods=['GET'],
            response_model=api.IdentityResponse,
            summary='Get an identity available to a project by ID.')
        self.router.add_api_route(
            '', self.list_available, methods=['GET'],
            response_model=list[api.IdentityResponse],
            summary='List identities available to a project.')

        # Administrator lifecycle (customer-admin API key only).
        self.admin_router.add_api_route(
            '', self.create, methods=['POST'],
            response_model=api.IdentityResponse, summary='Create an identity.')
        self.admin_router.add_api_route(
            '', self.list_admin, methods=['GET'],
            response_model=list[api.IdentityResponse],
            summary='List all identities.')
        self.admin_router.add_api_route(
            '/{identity_id}', self.get_admin, methods=['GET'],
            response_model=api.IdentityResponse, summary='Get an identity.')
        self.admin_router.add_api_route(
            '/{identity_id}', self.update, methods=['PATCH'],
            response_model=api.IdentityResponse,
            summary='Update identity metadata.')
        self.admin_router.add_api_route(
            '/{identity_id}', self.delete, methods=['DELETE'],
            response_model=api.IdentityResponse, summary='Delete an identity.')
        self.admin_router.add_api_route(
            '/{identity_id}/credentials', self.rotate_credentials,
            methods=['PUT'], response_model=api.IdentityResponse,
            summary='Rotate identity credentials.')
        self.admin_router.add_api_route(
            '/{identity_id}/test', self.test_credentials, methods=['POST'],
            response_model=api.IdentityResponse,
            summary='Test identity credentials against the provider.')
        self.admin_router.add_api_route(
            '/{identity_id}/groups', self.replace_groups, methods=['PUT'],
            response_model=api.IdentityGroupsResponse,
            summary='Replace identity group bindings.')
        self.admin_router.add_api_route(
            '/{identity_id}/groups', self.list_groups, methods=['GET'],
            response_model=api.IdentityGroupsResponse,
            summary='List identity group bindings.')

    @property
    def _service(self):
        return self._driver._require_state_services().identities

    @staticmethod
    def _principal(request: Request):
        return get_user_authorization_from_request(request)

    # -- Project-scoped discovery ------------------------------------------

    async def list_available(
        self,
        project_id: ProjectID,
        request: Request,
    ) -> list[api.IdentityResponse]:
        with wrap_driver_errors():
            return await self._service.available_for_project(
                project_id, self._principal(request))

    async def get_available_by_slug(
        self,
        project_id: ProjectID,
        slug: str,
        request: Request,
    ) -> api.IdentityResponse:
        with wrap_driver_errors():
            return await self._service.available_for_project_by_slug(
                project_id, slug, self._principal(request))

    async def get_available_by_id(
        self,
        project_id: ProjectID,
        identity_id: IdentityID,
        request: Request,
    ) -> api.IdentityResponse:
        with wrap_driver_errors():
            return await self._service.available_for_project_by_id(
                project_id, identity_id, self._principal(request))

    # -- Administrator lifecycle -------------------------------------------

    async def create(self, args: api.CreateIdentityArgs,
                     request: Request) -> api.IdentityResponse:
        with wrap_driver_errors({IdentityConflictError: HTTPStatus.CONFLICT}):
            return await self._service.create(args, self._principal(request))

    async def list_admin(self, request: Request) -> list[api.IdentityResponse]:
        with wrap_driver_errors():
            return await self._service.list_admin(self._principal(request))

    async def get_admin(self, identity_id: IdentityID,
                        request: Request) -> api.IdentityResponse:
        with wrap_driver_errors():
            return await self._service.get_admin(identity_id,
                                                 self._principal(request))

    async def update(self, identity_id: IdentityID,
                     args: api.UpdateIdentityArgs,
                     request: Request) -> api.IdentityResponse:
        with wrap_driver_errors({IdentityConflictError: HTTPStatus.CONFLICT}):
            return await self._service.update_metadata(identity_id, args,
                                                       self._principal(request))

    async def delete(self, identity_id: IdentityID,
                     request: Request) -> api.IdentityResponse:
        with wrap_driver_errors({IdentityInUseError: HTTPStatus.CONFLICT}):
            return await self._service.delete(identity_id,
                                              self._principal(request))

    async def rotate_credentials(
        self,
        identity_id: IdentityID,
        args: api.RotateIdentityCredentialsArgs,
        request: Request,
    ) -> api.IdentityResponse:
        with wrap_driver_errors():
            return await self._service.rotate_credentials(
                identity_id,
                args,
                self._principal(request),
                preserve_s3_ca_certificate=(
                    await preserves_s3_ca_certificate(request)),
            )

    async def test_credentials(self, identity_id: IdentityID,
                               request: Request) -> api.IdentityResponse:
        with wrap_driver_errors():
            return await self._driver.test_identity_credentials(
                identity_id, self._principal(request))

    async def replace_groups(
        self,
        identity_id: IdentityID,
        args: api.ReplaceIdentityGroupsArgs,
        request: Request,
    ) -> api.IdentityGroupsResponse:
        with wrap_driver_errors():
            return await self._service.replace_groups(identity_id, args,
                                                      self._principal(request))

    async def list_groups(self, identity_id: IdentityID,
                          request: Request) -> api.IdentityGroupsResponse:
        with wrap_driver_errors():
            return await self._service.list_groups(identity_id,
                                                   self._principal(request))
