# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request, Response

from rest.config.group import (
    AddGroupMemberRequest,
    AdminGroupDetailResponse,
    CreateExternalGroupMappingRequest,
    CreateKumoGroupRequest,
    CreateProjectInGroupRequest,
    ExternalGroupMappingResponse,
    ReplaceExternalGroupMappingsRequest,
    ReplaceExternalGroupMappingsResponse,
    ReplaceGroupMembersRequest,
    ReplaceGroupMembersResponse,
    UpdateKumoGroupRequest,
)
from rest.deps.authentication import (
    Authentication,
    get_user_authorization_from_request,
)
from rest.driver.groups_driver import GroupsDriver
from rest.utils.public import wrap_driver_errors
from statedb.config.common import ExternalGroupMappingID, GroupID
from statedb.rest.group import GroupMemberEntry, KumoGroupDBEntry
from statedb.rest.project import ProjectDBEntry
from statedb.utils.errors import (
    GroupAlreadyExists,
    GroupInputError,
    GroupNotFoundError,
    GroupRelationConflictError,
)

_GROUP_ERRORS = {
    GroupNotFoundError: HTTPStatus.NOT_FOUND,
    GroupAlreadyExists: HTTPStatus.CONFLICT,
}


class GroupsRouter:
    """Administrator API for native teams and their membership sources."""
    def __init__(self, auth: Authentication,
                 groups_driver: GroupsDriver) -> None:
        self._driver = groups_driver
        admin_only = [Depends(auth.require_customer_admin)]
        self.router = APIRouter(
            prefix='/v0/groups',
            tags=['Groups'],
            dependencies=[Depends(auth.authenticate_request_header)],
        )
        self.v1_router = APIRouter(
            prefix='/v1/groups',
            tags=['Groups'],
            dependencies=[
                Depends(auth.authenticate_request_header),
                Depends(auth.require_customer_admin),
            ],
        )

        self.router.add_api_route('', self.create_group, methods=['POST'],
                                  status_code=HTTPStatus.CREATED,
                                  response_model=KumoGroupDBEntry,
                                  dependencies=admin_only)
        self.router.add_api_route('', self.list_groups, methods=['GET'],
                                  response_model=list[KumoGroupDBEntry])
        self.router.add_api_route('/by-slug/{slug}', self.get_group_by_slug,
                                  methods=['GET'],
                                  response_model=KumoGroupDBEntry,
                                  dependencies=admin_only)
        self.router.add_api_route('/{group_id}', self.get_group,
                                  methods=['GET'],
                                  response_model=AdminGroupDetailResponse,
                                  dependencies=admin_only)
        self.router.add_api_route('/{group_id}', self.update_group,
                                  methods=['PATCH'],
                                  response_model=KumoGroupDBEntry,
                                  dependencies=admin_only)
        self.router.add_api_route('/{group_id}', self.delete_group,
                                  methods=['DELETE'],
                                  status_code=HTTPStatus.NO_CONTENT,
                                  dependencies=admin_only)

        self.router.add_api_route('/{group_id}/members', self.list_members,
                                  methods=['GET'],
                                  response_model=list[GroupMemberEntry],
                                  dependencies=admin_only)
        self.router.add_api_route('/{group_id}/members', self.add_member,
                                  methods=['POST'],
                                  status_code=HTTPStatus.CREATED,
                                  response_model=GroupMemberEntry,
                                  dependencies=admin_only)
        self.router.add_api_route('/{group_id}/members/{principal_id}',
                                  self.remove_member, methods=['DELETE'],
                                  status_code=HTTPStatus.NO_CONTENT,
                                  dependencies=admin_only)

        self.router.add_api_route(
            '/{group_id}/external-mappings', self.list_external_mappings,
            methods=['GET'], response_model=list[ExternalGroupMappingResponse],
            dependencies=admin_only)
        self.router.add_api_route('/{group_id}/external-mappings',
                                  self.create_external_mapping,
                                  methods=['POST'
                                           ], status_code=HTTPStatus.CREATED,
                                  response_model=ExternalGroupMappingResponse,
                                  dependencies=admin_only)
        self.router.add_api_route('/{group_id}/external-mappings/{mapping_id}',
                                  self.delete_external_mapping,
                                  methods=['DELETE'],
                                  status_code=HTTPStatus.NO_CONTENT,
                                  dependencies=admin_only)
        self.v1_router.add_api_route(
            '/{group_id}/members',
            self.replace_members,
            methods=['PUT'],
            response_model=ReplaceGroupMembersResponse,
        )
        self.v1_router.add_api_route(
            '/{group_id}/external-mappings',
            self.replace_external_mappings,
            methods=['PUT'],
            response_model=ReplaceExternalGroupMappingsResponse,
        )

        self.router.add_api_route('/{group_id}/all-authenticated-users',
                                  self.get_all_authenticated_users,
                                  methods=['GET'], response_model=bool,
                                  dependencies=admin_only)
        self.router.add_api_route('/{group_id}/all-authenticated-users',
                                  self.enable_all_authenticated_users,
                                  methods=['PUT'],
                                  status_code=HTTPStatus.NO_CONTENT,
                                  dependencies=admin_only)
        self.router.add_api_route('/{group_id}/all-authenticated-users',
                                  self.disable_all_authenticated_users,
                                  methods=['DELETE'],
                                  status_code=HTTPStatus.NO_CONTENT,
                                  dependencies=admin_only)

        self.router.add_api_route('/{group_id}/projects', self.create_project,
                                  methods=['POST'],
                                  status_code=HTTPStatus.CREATED,
                                  response_model=ProjectDBEntry)
        self.router.add_api_route('/{group_id}/projects', self.list_projects,
                                  methods=['GET'],
                                  response_model=list[ProjectDBEntry])

    async def create_group(
        self,
        request: CreateKumoGroupRequest,
    ) -> KumoGroupDBEntry:
        with wrap_driver_errors({
                **_GROUP_ERRORS, ValueError:
                HTTPStatus.UNPROCESSABLE_ENTITY
        }):
            result = await self._driver._create_kumo_groups([request])
            return await self._driver._get_kumo_group(result[0].id)

    async def list_groups(self, request: Request) -> list[KumoGroupDBEntry]:
        return await self._driver.list_visible_groups(
            get_user_authorization_from_request(request))

    async def get_group(self,
                        group_id: GroupID) -> AdminGroupDetailResponse:
        with wrap_driver_errors(_GROUP_ERRORS):
            return await self._driver.get_admin_group_detail(group_id)

    async def get_group_by_slug(self, slug: str) -> KumoGroupDBEntry:
        with wrap_driver_errors(_GROUP_ERRORS):
            return await self._driver.get_kumo_group_by_slug(slug)

    async def update_group(
        self,
        group_id: GroupID,
        request: UpdateKumoGroupRequest,
    ) -> KumoGroupDBEntry:
        with wrap_driver_errors({
                **_GROUP_ERRORS, ValueError:
                HTTPStatus.BAD_REQUEST
        }):
            return await self._driver._update_kumo_group(
                group_id, display_name=request.display_name,
                description=request.description)

    async def delete_group(self, group_id: GroupID) -> Response:
        with wrap_driver_errors({
                **_GROUP_ERRORS, ValueError: HTTPStatus.UNPROCESSABLE_ENTITY
        }):
            await self._driver._delete_kumo_group(str(group_id))
        return Response(status_code=HTTPStatus.NO_CONTENT)

    async def list_members(self, group_id: GroupID) -> list[GroupMemberEntry]:
        with wrap_driver_errors(_GROUP_ERRORS):
            return await self._driver.list_members(group_id)

    async def add_member(
        self,
        group_id: GroupID,
        request: AddGroupMemberRequest,
    ) -> GroupMemberEntry:
        with wrap_driver_errors({
                **_GROUP_ERRORS,
                GroupInputError: HTTPStatus.BAD_REQUEST,
                GroupRelationConflictError: HTTPStatus.CONFLICT,
        }):
            return await self._driver.add_member(group_id, request)

    async def remove_member(self, group_id: GroupID,
                            principal_id: str) -> Response:
        with wrap_driver_errors(_GROUP_ERRORS):
            await self._driver.remove_member(group_id, principal_id)
        return Response(status_code=HTTPStatus.NO_CONTENT)

    async def replace_members(
        self,
        group_id: GroupID,
        request: ReplaceGroupMembersRequest,
    ) -> ReplaceGroupMembersResponse:
        with wrap_driver_errors({
                **_GROUP_ERRORS,
                GroupInputError: HTTPStatus.BAD_REQUEST,
                GroupRelationConflictError: HTTPStatus.CONFLICT,
        }):
            return await self._driver.replace_members(group_id, request)

    async def list_external_mappings(
        self,
        group_id: GroupID,
    ) -> list[ExternalGroupMappingResponse]:
        with wrap_driver_errors(_GROUP_ERRORS):
            return await self._driver.list_external_mappings(group_id)

    async def create_external_mapping(
        self,
        group_id: GroupID,
        request: CreateExternalGroupMappingRequest,
    ) -> ExternalGroupMappingResponse:
        with wrap_driver_errors({
                **_GROUP_ERRORS,
                GroupInputError: HTTPStatus.BAD_REQUEST,
                GroupRelationConflictError: HTTPStatus.CONFLICT,
        }):
            return await self._driver.create_external_mapping(group_id, request)

    async def delete_external_mapping(
        self,
        group_id: GroupID,
        mapping_id: ExternalGroupMappingID,
    ) -> Response:
        with wrap_driver_errors(_GROUP_ERRORS):
            await self._driver.delete_external_mapping(group_id, mapping_id)
        return Response(status_code=HTTPStatus.NO_CONTENT)

    async def replace_external_mappings(
        self,
        group_id: GroupID,
        request: ReplaceExternalGroupMappingsRequest,
    ) -> ReplaceExternalGroupMappingsResponse:
        with wrap_driver_errors({
                **_GROUP_ERRORS,
                GroupInputError: HTTPStatus.BAD_REQUEST,
                GroupRelationConflictError: HTTPStatus.CONFLICT,
        }):
            return await self._driver.replace_external_mappings(
                group_id, request)

    async def enable_all_authenticated_users(
        self,
        group_id: GroupID,
    ) -> Response:
        with wrap_driver_errors(_GROUP_ERRORS):
            await self._driver.set_all_authenticated_users(group_id, True)
        return Response(status_code=HTTPStatus.NO_CONTENT)

    async def get_all_authenticated_users(
        self,
        group_id: GroupID,
    ) -> bool:
        with wrap_driver_errors(_GROUP_ERRORS):
            return await self._driver.has_all_authenticated_users(group_id)

    async def disable_all_authenticated_users(
        self,
        group_id: GroupID,
    ) -> Response:
        with wrap_driver_errors(_GROUP_ERRORS):
            await self._driver.set_all_authenticated_users(group_id, False)
        return Response(status_code=HTTPStatus.NO_CONTENT)

    async def create_project(
        self,
        group_id: GroupID,
        request: CreateProjectInGroupRequest,
    ) -> ProjectDBEntry:
        with wrap_driver_errors({
                **_GROUP_ERRORS, ValueError: HTTPStatus.CONFLICT
        }):
            return await self._driver.create_project_in_group(group_id, request)

    async def list_projects(
        self,
        group_id: GroupID,
    ) -> list[ProjectDBEntry]:
        with wrap_driver_errors(_GROUP_ERRORS):
            return await self._driver.list_projects_in_group(group_id)
