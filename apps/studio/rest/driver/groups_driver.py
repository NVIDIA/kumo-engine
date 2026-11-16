# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from rest.config.encryption import decrypt_idp_id
from rest.config.group import (
    AddGroupMemberRequest,
    AdminExternalGroupMappingResponse,
    AdminGroupDetailResponse,
    AdminGroupMemberResponse,
    CreatedGroupEntryResponse,
    CreateExternalGroupMappingRequest,
    CreateKumoGroupRequest,
    CreateProjectInGroupRequest,
    ExternalGroupMappingResponse,
    GroupMemberResponse,
    ListUserKumoGroupsResponse,
    ReplaceExternalGroupMappingsRequest,
    ReplaceExternalGroupMappingsResponse,
    ReplaceGroupMembersRequest,
    ReplaceGroupMembersResponse,
)
from rest.deps.context import _current_user_ctx
from statedb.config.common import ExternalGroupMappingID, GroupID
from statedb.db.auth import AuthPrincipal
from statedb.db.services import GroupService, StateServices
from statedb.rest.group import (
    ExternalGroupMappingSpec,
    GroupMemberEntry,
    GroupMemberSpec,
    KumoGroupCreateSpec,
    KumoGroupDBEntry,
)
from statedb.rest.project import ProjectDBEntry


class GroupsDriver:
    def __init__(self):
        self._state_services: Optional[StateServices] = None

    def _configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services

    @property
    def _group_service(self) -> GroupService:
        if self._state_services is None:
            raise RuntimeError("GroupsDriver StateServices is not configured")
        return self._state_services.groups

    # ========================= KumoGroup CRUD =========================

    async def _list_kumo_groups(self) -> List[KumoGroupDBEntry]:
        return await self._group_service.list_kumo_groups(
            _current_user_ctx.get())

    async def list_visible_groups(
        self,
        principal: AuthPrincipal,
    ) -> List[KumoGroupDBEntry]:
        if principal.is_admin:
            return await self._group_service.list_kumo_groups(principal)
        return await self._group_service.list_user_groups(principal)

    async def _get_kumo_group(self, group_id: GroupID) -> KumoGroupDBEntry:
        return await self._group_service.get_kumo_group(str(group_id),
                                                        _current_user_ctx.get())

    async def get_admin_group_detail(
        self,
        group_id: GroupID,
    ) -> AdminGroupDetailResponse:
        group = await self._group_service.get_kumo_group(
            str(group_id), _current_user_ctx.get())
        members = await self.list_members(group_id)
        mappings = await self.list_external_mappings(group_id)
        return AdminGroupDetailResponse(
            id=group.id,
            slug=group.slug,
            display_name=group.display_name,
            description=group.description,
            members=[
                AdminGroupMemberResponse(
                    id=member.id,
                    principal_id=member.principal_id,
                    principal_type=member.principal_type,
                    source=member.source,
                ) for member in members
            ],
            external_group_mappings=[
                AdminExternalGroupMappingResponse(
                    id=mapping.id,
                    identity_provider_id=mapping.identity_provider_id,
                    external_group_id=mapping.external_group_id,
                ) for mapping in mappings
            ],
        )

    async def _get_kumo_group_by_alias(self,
                                       display_name: str) -> KumoGroupDBEntry:
        return await self._group_service.get_kumo_group_by_alias(
            display_name,
            _current_user_ctx.get(),
        )

    async def get_kumo_group_by_slug(self, slug: str) -> KumoGroupDBEntry:
        return await self._group_service.get_kumo_group_by_slug(
            slug,
            _current_user_ctx.get(),
        )

    async def _create_kumo_groups(
        self,
        groups: List[CreateKumoGroupRequest],
    ) -> List[CreatedGroupEntryResponse]:
        specs = [
            KumoGroupCreateSpec(
                display_name=group.display_name,
                slug=group.slug,
                description=group.description,
            ) for group in groups
        ]
        entries = await self._group_service.create_kumo_groups(
            specs, _current_user_ctx.get())
        return [
            CreatedGroupEntryResponse(
                id=entry.id,
                slug=entry.slug,
                display_name=entry.display_name,
                description=entry.description,
            ) for entry in entries
        ]

    async def _update_kumo_group(
        self,
        group_id: GroupID,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> KumoGroupDBEntry:
        return await self._group_service.update_kumo_group(
            group_id,
            display_name=display_name,
            description=description,
            principal=_current_user_ctx.get(),
        )

    async def _delete_kumo_group(self, group_id: str) -> None:
        await self._group_service.delete_kumo_group(group_id,
                                                    _current_user_ctx.get())

    # ========================= User group membership =========================

    async def _list_user_groups(self) -> List[ListUserKumoGroupsResponse]:
        """Return groups the current user belongs to.

        Group membership is read from _current_user_ctx, which the
        middleware populates via the group resolver at request time.
        """
        groups = await self._group_service.list_user_groups(
            _current_user_ctx.get())
        return [
            ListUserKumoGroupsResponse(
                group_id=group.id,
                slug=group.slug,
                display_name=group.display_name,
            ) for group in groups
        ]

    async def list_user_groups(self) -> List[ListUserKumoGroupsResponse]:
        return await self._list_user_groups()

    async def add_member(
        self,
        group_id: GroupID,
        request: AddGroupMemberRequest,
    ) -> GroupMemberEntry:
        entry = await self._group_service.add_member(
            group_id, request.principal_id, request.principal_type,
            principal=_current_user_ctx.get())
        return entry

    async def list_members(self, group_id: GroupID) -> List[GroupMemberEntry]:
        return await self._group_service.list_members(group_id,
                                                      _current_user_ctx.get())

    async def remove_member(self, group_id: GroupID, principal_id: str) -> None:
        await self._group_service.remove_member(
            group_id, principal_id, principal=_current_user_ctx.get())

    async def replace_members(
        self,
        group_id: GroupID,
        request: ReplaceGroupMembersRequest,
    ) -> ReplaceGroupMembersResponse:
        entries = await self._group_service.replace_members(
            group_id,
            [
                GroupMemberSpec(
                    principal_id=member.principal_id,
                    principal_type=member.principal_type,
                ) for member in request.members
            ],
            principal=_current_user_ctx.get(),
        )
        return ReplaceGroupMembersResponse(members=[
            GroupMemberResponse(
                id=entry.id,
                group_id=entry.group_id,
                principal_id=entry.principal_id,
                principal_type=entry.principal_type,
                source=entry.source,
            ) for entry in entries
        ])

    async def create_external_mapping(
        self,
        group_id: GroupID,
        request: CreateExternalGroupMappingRequest,
    ) -> ExternalGroupMappingResponse:
        entry = await self._group_service.create_external_mapping(
            group_id,
            request.identity_provider_id,
            request.external_group_id,
            principal=_current_user_ctx.get(),
        )
        return self._external_mapping_response(entry)

    async def list_external_mappings(
        self,
        group_id: GroupID,
    ) -> List[ExternalGroupMappingResponse]:
        entries = await self._group_service.list_external_mappings(
            group_id, _current_user_ctx.get())
        return [self._external_mapping_response(entry) for entry in entries]

    async def delete_external_mapping(
        self,
        group_id: GroupID,
        mapping_id: ExternalGroupMappingID,
    ) -> None:
        await self._group_service.delete_external_mapping(
            group_id, mapping_id, principal=_current_user_ctx.get())

    async def replace_external_mappings(
        self,
        group_id: GroupID,
        request: ReplaceExternalGroupMappingsRequest,
    ) -> ReplaceExternalGroupMappingsResponse:
        entries = await self._group_service.replace_external_mappings(
            group_id,
            [
                ExternalGroupMappingSpec(
                    identity_provider_id=mapping.identity_provider_id,
                    external_group_id=mapping.external_group_id,
                ) for mapping in request.mappings
            ],
            principal=_current_user_ctx.get(),
        )
        return ReplaceExternalGroupMappingsResponse(mappings=[
            self._external_mapping_response(entry) for entry in entries
        ])

    @staticmethod
    def _external_mapping_response(entry) -> ExternalGroupMappingResponse:
        return ExternalGroupMappingResponse(
            id=entry.id,
            group_id=entry.group_id,
            identity_provider_id=entry.identity_provider_id,
            external_group_id=decrypt_idp_id(entry.encrypted_external_group_id),
        )

    async def set_all_authenticated_users(
        self,
        group_id: GroupID,
        enabled: bool,
    ) -> bool:
        await self._group_service.set_all_authenticated_users(
            group_id, enabled, principal=_current_user_ctx.get())
        return enabled

    async def has_all_authenticated_users(self, group_id: GroupID) -> bool:
        return await self._group_service.has_all_authenticated_users(
            group_id, _current_user_ctx.get())

    async def create_project_in_group(
        self,
        group_id: GroupID,
        request: CreateProjectInGroupRequest,
    ) -> ProjectDBEntry:
        if self._state_services is None:
            raise RuntimeError('GroupsDriver StateServices is not configured')
        return await self._state_services.projects.create_project(
            name=request.name,
            group_id=str(group_id),
            slug=request.slug,
            description=request.description,
            is_private=request.is_private,
            user_authorization=_current_user_ctx.get(),
        )

    async def list_projects_in_group(
        self,
        group_id: GroupID,
    ) -> List[ProjectDBEntry]:
        if self._state_services is None:
            raise RuntimeError('GroupsDriver StateServices is not configured')
        return await self._state_services.projects.list_projects_in_group(
            group_id, _current_user_ctx.get())
