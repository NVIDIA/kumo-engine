# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from statedb.config.common import GroupID
from statedb.db.identities import IdentityService


async def ensure_startup_ambient_identities(
    identities: IdentityService,
    default_group_id: GroupID,
    *,
    auth_disabled: bool,
) -> None:
    if auth_disabled:
        await identities.ensure_ambient_s3_identity_bound_to_group(
            default_group_id)
    else:
        await identities.ensure_ambient_s3_identity()
    await identities.ensure_ambient_adls_identity_bound_on_create(
        default_group_id)
