# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Optional

from kumoapi.data_source import IdentityConfig, ResourceSlug
from pydantic import Field

from io_utils import kumo_dataclass
from statedb.config.common import DBEntry, GroupID, IdentityGroupID, IdentityID
from statedb.rest.credentials import EncryptedIdentityCredentials


@kumo_dataclass
class IdentityDBEntry(DBEntry):
    id: IdentityID
    slug: ResourceSlug
    display_name: str
    normalized_display_name: str
    description: str
    identity_key: str
    credentials_rotated_at: Optional[datetime]
    config: IdentityConfig = Field(discriminator='type')
    encrypted_credentials: Optional[EncryptedIdentityCredentials] = Field(
        default=None, discriminator='kind')
    tombstoned_at: Optional[datetime] = None


@kumo_dataclass
class IdentityGroupEntry(DBEntry):
    id: IdentityGroupID
    identity_id: IdentityID
    group_id: GroupID
