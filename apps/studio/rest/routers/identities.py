# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


class IdentitiesResource:
    """Administrator-only identity lifecycle operations."""
    def __init__(self, auth: Authentication,
                 driver: ConnectorApiDriver) -> None:
        self._driver = driver
        authenticated = [Depends(auth.authenticate_request_header)]
        admin = [Depends(auth.require_customer_admin)]
        self.router = APIRouter(
            prefix='/v0/identities',
            tags=['Identities'],
            dependencies=[*authenticated, *admin],
            responses={HTTPStatus.NOT_FOUND: {
                'description': 'Not found'
            }},
        )
        self.project_router = APIRouter(
            prefix='/v0/projects/{project_id}/identities',
            tags=['Identities'],
            dependencies=authenticated,
        )
        self.router.add_api_route('/', self.create, methods=['POST'],
                                  response_model=api.IdentityResponse)
        self.router.add_api_route('/', self.list_admin, methods=['GET'],
                                  response_model=list[api.IdentityResponse])
        self.router.add_api_route('/{identity_id}', self.get_admin,
                                  methods=['GET'],
                                  response_model=api.IdentityResponse)
        self.router.add_api_route('/{identity_id}', self.update,
                                  methods=['PATCH'],
                                  response_model=api.IdentityResponse)
        self.router.add_api_route('/{identity_id}', self.delete,
                                  methods=['DELETE'],
                                  response_model=api.IdentityResponse)
        self.router.add_api_route('/{identity_id}/credentials',
                                  self.rotate_credentials, methods=['PUT'],
                                  response_model=api.IdentityResponse)
        self.router.add_api_route('/{identity_id}/test', self.test_credentials,
                                  methods=['POST'],
                                  response_model=api.IdentityResponse)
        self.router.add_api_route('/{identity_id}/groups', self.replace_groups,
                                  methods=['PUT'],
                                  response_model=api.IdentityGroupsResponse)
        self.router.add_api_route('/{identity_id}/groups', self.list_groups,
                                  methods=['GET'],
                                  response_model=api.IdentityGroupsResponse)
        self.router.add_api_route(
            '/{identity_id}/connectors', self.list_connectors, methods=['GET'],
            response_model=api.IdentityConnectorsResponse)
        self.project_router.add_api_route(
            '/by-slug/{slug}', self.get_available_for_project_by_slug,
            methods=['GET'], response_model=api.IdentityResponse)
        self.project_router.add_api_route(
            '/{identity_id}', self.get_available_for_project_by_id,
            methods=['GET'], response_model=api.IdentityResponse)
        self.project_router.add_api_route(
            '/', self.list_available_for_project, methods=['GET'],
            response_model=list[api.IdentityResponse])

    @property
    def _service(self):
        return self._driver._require_state_services().identities

    @staticmethod
    def _principal(request: Request):
        return get_user_authorization_from_request(request)

    async def create(self, args: api.CreateIdentityArgs,
                     request: Request) -> api.IdentityResponse:
        with wrap_driver_errors({IdentityConflictError: HTTPStatus.CONFLICT}):
            return await self._service.create(args, self._principal(request))

    async def list_admin(self, request: Request) -> list[api.IdentityResponse]:
        with wrap_driver_errors():
            return await self._service.list_admin(self._principal(request))

    async def list_available_for_project(
        self,
        project_id: ProjectID,
        request: Request,
    ) -> list[api.IdentityResponse]:
        with wrap_driver_errors():
            return await self._service.available_for_project(
                project_id, self._principal(request))

    async def get_available_for_project_by_slug(
        self,
        project_id: ProjectID,
        slug: str,
        request: Request,
    ) -> api.IdentityResponse:
        with wrap_driver_errors():
            return await self._service.available_for_project_by_slug(
                project_id, slug, self._principal(request))

    async def get_available_for_project_by_id(
        self,
        project_id: ProjectID,
        identity_id: IdentityID,
        request: Request,
    ) -> api.IdentityResponse:
        with wrap_driver_errors():
            return await self._service.available_for_project_by_id(
                project_id, identity_id, self._principal(request))

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

    async def replace_groups(
        self,
        identity_id: IdentityID,
        args: api.ReplaceIdentityGroupsArgs,
        request: Request,
    ) -> api.IdentityGroupsResponse:
        with wrap_driver_errors():
            return await self._service.replace_groups(identity_id, args,
                                                      self._principal(request))

    async def test_credentials(self, identity_id: IdentityID,
                               request: Request) -> api.IdentityResponse:
        with wrap_driver_errors():
            return await self._driver.test_identity_credentials(
                identity_id, self._principal(request))

    async def list_groups(self, identity_id: IdentityID,
                          request: Request) -> api.IdentityGroupsResponse:
        with wrap_driver_errors():
            return await self._service.list_groups(identity_id,
                                                   self._principal(request))

    async def list_connectors(
        self, identity_id: IdentityID,
        request: Request) -> api.IdentityConnectorsResponse:
        with wrap_driver_errors():
            return await self._service.list_connectors_using_identity(
                identity_id, self._principal(request))

    async def delete(self, identity_id: IdentityID,
                     request: Request) -> api.IdentityResponse:
        with wrap_driver_errors({IdentityInUseError: HTTPStatus.CONFLICT}):
            return await self._service.delete(identity_id,
                                              self._principal(request))
