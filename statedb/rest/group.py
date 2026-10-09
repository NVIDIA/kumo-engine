# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Optional

from kumoapi.data_source import ResourceSlug

from io_utils.dataclasses import kumo_dataclass
from statedb.config.common import (
    DBEntry,
    ExternalGroupMappingID,
    GroupID,
    GroupMemberID,
    GroupMembershipRuleID,
)


class GroupStatus(str, Enum):
    ACTIVE = 'active'
    DELETED = 'deleted'


class PrincipalType(str, Enum):
    USER = 'user'
    SERVICE_ACCOUNT = 'service_account'


class GroupMembershipRuleType(str, Enum):
    ALL_AUTHENTICATED_USERS = 'all_authenticated_users'


class ExternalIdentityProviderID(str, Enum):
    """Identity providers that can supply external group membership."""

    DIRECT_CONNECT = 'direct_connect'

    @classmethod
    def normalize(
        cls,
        value: str | 'ExternalIdentityProviderID',
    ) -> 'ExternalIdentityProviderID':
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().casefold())


@kumo_dataclass
class KumoGroupCreateSpec:
    display_name: str
    slug: Optional[ResourceSlug] = None
    description: str = ''


@kumo_dataclass
class KumoGroupDBEntry(DBEntry):
    id: GroupID
    slug: ResourceSlug
    display_name: str
    normalized_display_name: str
    description: str = ''
    status: GroupStatus = GroupStatus.ACTIVE


@kumo_dataclass
class GroupMemberEntry(DBEntry):
    id: GroupMemberID
    group_id: GroupID
    principal_id: str
    principal_type: PrincipalType
    source: str = 'manual'


@kumo_dataclass
class GroupMemberSpec:
    principal_id: str
    principal_type: PrincipalType = PrincipalType.USER


@kumo_dataclass
class ExternalGroupMappingEntry(DBEntry):
    id: ExternalGroupMappingID
    group_id: GroupID
    identity_provider_id: ExternalIdentityProviderID
    encrypted_external_group_id: str
    external_group_id_hash: str


@kumo_dataclass
class ExternalGroupMappingSpec:
    """Plaintext mapping input used only at the service boundary."""

    identity_provider_id: str
    external_group_id: str


@kumo_dataclass
class GroupMembershipRuleEntry(DBEntry):
    id: GroupMembershipRuleID
    group_id: GroupID
    type: GroupMembershipRuleType
