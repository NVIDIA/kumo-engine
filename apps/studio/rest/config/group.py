# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import List, Optional

from kumoapi.data_source import ResourceSlug

from io_utils.dataclasses import kumo_dataclass
from statedb.config.common import (
    ExternalGroupMappingID,
    GroupID,
    GroupMemberID,
)
from statedb.rest.group import PrincipalType


@kumo_dataclass
class CreateKumoGroupRequest:
    display_name: str
    slug: Optional[ResourceSlug] = None
    description: str = ''


@kumo_dataclass
class CreatedGroupEntryResponse:
    id: GroupID
    slug: ResourceSlug
    display_name: str
    description: str = ''


@kumo_dataclass
class UpdateKumoGroupRequest:
    display_name: Optional[str] = None
    description: Optional[str] = None


@kumo_dataclass
class ListUserKumoGroupsResponse:
    group_id: GroupID
    slug: ResourceSlug
    display_name: str


@kumo_dataclass
class AddGroupMemberRequest:
    principal_id: str
    principal_type: PrincipalType = PrincipalType.USER


@kumo_dataclass
class GroupMemberResponse:
    id: GroupMemberID
    group_id: GroupID
    principal_id: str
    principal_type: PrincipalType
    source: str = 'manual'


@kumo_dataclass
class ReplaceGroupMembersRequest:
    members: List[AddGroupMemberRequest] = field(default_factory=list)


@kumo_dataclass
class ReplaceGroupMembersResponse:
    members: List[GroupMemberResponse] = field(default_factory=list)


@kumo_dataclass
class CreateExternalGroupMappingRequest:
    identity_provider_id: str
    external_group_id: str


@kumo_dataclass
class ExternalGroupMappingResponse:
    id: ExternalGroupMappingID
    group_id: GroupID
    identity_provider_id: str
    external_group_id: str


@kumo_dataclass
class ReplaceExternalGroupMappingsRequest:
    mappings: List[CreateExternalGroupMappingRequest] = field(
        default_factory=list)


@kumo_dataclass
class ReplaceExternalGroupMappingsResponse:
    mappings: List[ExternalGroupMappingResponse] = field(default_factory=list)


@kumo_dataclass
class AdminExternalGroupMappingResponse:
    id: ExternalGroupMappingID
    identity_provider_id: str
    external_group_id: str


@kumo_dataclass
class AdminGroupMemberResponse:
    id: GroupMemberID
    principal_id: str
    principal_type: PrincipalType
    source: str = 'manual'


@kumo_dataclass
class AdminGroupDetailResponse:
    id: GroupID
    slug: ResourceSlug
    display_name: str
    description: str = ''
    members: List[AdminGroupMemberResponse] = field(default_factory=list)
    external_group_mappings: List[
        AdminExternalGroupMappingResponse] = field(default_factory=list)


@kumo_dataclass
class CreateProjectInGroupRequest:
    name: str
    slug: Optional[ResourceSlug] = None
    description: str = ''
    is_private: bool = False
