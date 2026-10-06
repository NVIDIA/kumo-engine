# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from rest.deps.authentication import UserAuthorization
from statedb.db import StateDB
from statedb.db.groups import GroupService
from statedb.rest.group import (
    ExternalGroupMappingSpec,
    GroupMemberSpec,
    KumoGroupCreateSpec,
    PrincipalType,
)
from statedb.utils.errors import GroupRelationConflictError


async def _reset_postgres_schema(connection) -> None:
    await connection.execute('DROP SCHEMA public CASCADE')
    await connection.execute('CREATE SCHEMA public')


@pytest.mark.asyncio
async def test_postgres_concurrent_replacement_uses_authoritative_uniqueness(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    service = GroupService(state_db)
    admin = UserAuthorization.disabled()
    try:
        first, second = await service.create_kumo_groups([
            KumoGroupCreateSpec(display_name='First'),
            KumoGroupCreateSpec(display_name='Second'),
        ], admin)
        results = await asyncio.gather(
            service.replace_external_mappings(
                first.id,
                [ExternalGroupMappingSpec(
                    identity_provider_id='direct_connect',
                    external_group_id='contended',
                )],
                principal=admin,
            ),
            service.replace_external_mappings(
                second.id,
                [ExternalGroupMappingSpec(
                    identity_provider_id='direct_connect',
                    external_group_id='contended',
                )],
                principal=admin,
            ),
            return_exceptions=True,
        )

        assert sum(isinstance(result, list) for result in results) == 1
        assert sum(isinstance(result, GroupRelationConflictError)
                   for result in results) == 1
        persisted = (await service.list_external_mappings(first.id, admin)
                     + await service.list_external_mappings(second.id, admin))
        assert len(persisted) == 1
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)


@pytest.mark.asyncio
async def test_postgres_concurrent_member_replacements_are_complete_sets(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    service = GroupService(state_db)
    admin = UserAuthorization.disabled()
    try:
        group = (await service.create_kumo_groups(
            [KumoGroupCreateSpec(display_name='Member race')], admin))[0]
        first = [
            GroupMemberSpec(principal_id='first-a'),
            GroupMemberSpec(principal_id='first-b'),
        ]
        second = [
            GroupMemberSpec(
                principal_id='second-service',
                principal_type=PrincipalType.SERVICE_ACCOUNT,
            ),
            GroupMemberSpec(principal_id='second-user'),
        ]

        results = await asyncio.gather(
            service.replace_members(group.id, first, principal=admin),
            service.replace_members(group.id, second, principal=admin),
        )

        assert all(len(result) == 2 for result in results)
        final = {(entry.principal_type, entry.principal_id)
                 for entry in await service.list_members(group.id, admin)}
        assert final in (
            {
                (PrincipalType.USER, 'first-a'),
                (PrincipalType.USER, 'first-b'),
            },
            {
                (PrincipalType.SERVICE_ACCOUNT, 'second-service'),
                (PrincipalType.USER, 'second-user'),
            },
        )
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)
