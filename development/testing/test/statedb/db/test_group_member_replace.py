# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import pytest_asyncio

from rest.deps.authentication import UserAuthorization
from statedb.config.common import GroupMemberID
from statedb.config.timestamp import get_current_timestamps
from statedb.db import StateDB
from statedb.db.groups import GroupService
from statedb.rest.group import (
    GroupMemberEntry,
    GroupMemberSpec,
    KumoGroupCreateSpec,
    PrincipalType,
)
from statedb.utils.errors import GroupInputError, GroupRelationConflictError

ADMIN = UserAuthorization.disabled()


def _member(
    principal_id: str,
    principal_type: PrincipalType = PrincipalType.USER,
) -> GroupMemberSpec:
    return GroupMemberSpec(
        principal_id=principal_id,
        principal_type=principal_type,
    )


@pytest_asyncio.fixture
async def service(tmp_path):
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    try:
        yield GroupService(db)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_replace_members_adds_multiple_trims_dedupes_and_sorts(
    service,
) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Engineering')], ADMIN))[0]

    members = await service.replace_members(
        group.id,
        [
            _member(' user-b '),
            _member('service-a', PrincipalType.SERVICE_ACCOUNT),
            _member('user-a'),
            _member('user-b'),
        ],
        principal=ADMIN,
    )

    assert [(member.principal_type, member.principal_id)
            for member in members] == [
                (PrincipalType.SERVICE_ACCOUNT, 'service-a'),
                (PrincipalType.USER, 'user-a'),
                (PrincipalType.USER, 'user-b'),
            ]
    assert all(member.source == 'manual' for member in members)
    assert await service.resolve_native_memberships(
        'user-a',
        principal_type=PrincipalType.USER,
        authenticated=True,
    ) == [str(group.id)]


@pytest.mark.asyncio
async def test_one_principal_can_be_direct_member_of_several_groups(service) -> None:
    first, second = await service.create_kumo_groups([
        KumoGroupCreateSpec(display_name='First'),
        KumoGroupCreateSpec(display_name='Second'),
    ], ADMIN)

    await service.replace_members(
        first.id, [_member('stable-user-id')], principal=ADMIN)
    await service.replace_members(
        second.id, [_member('stable-user-id')], principal=ADMIN)

    assert await service.resolve_native_memberships(
        'stable-user-id',
        principal_type=PrincipalType.USER,
        authenticated=True,
    ) == sorted([str(first.id), str(second.id)])


@pytest.mark.asyncio
async def test_replace_members_preserves_ids_and_removes_omissions(
    service,
) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Data')], ADMIN))[0]
    before = await service.replace_members(
        group.id,
        [_member('keep'), _member('remove')],
        principal=ADMIN,
    )
    keep_id = next(member.id for member in before
                   if member.principal_id == 'keep')

    after = await service.replace_members(
        group.id,
        [_member(' keep '), _member('new')],
        principal=ADMIN,
    )

    assert next(member.id for member in after
                if member.principal_id == 'keep') == keep_id
    assert {member.principal_id for member in after} == {'keep', 'new'}


@pytest.mark.asyncio
async def test_empty_replace_only_clears_manual_members(service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Independent')], ADMIN))[0]
    await service.replace_members(
        group.id, [_member('manual-user')], principal=ADMIN)
    mapping = await service.create_external_mapping(
        group.id, 'direct_connect', 'external-team', principal=ADMIN)
    await service.set_all_authenticated_users(group.id, True, principal=ADMIN)
    system_member = GroupMemberEntry(
        id=GroupMemberID.generate(),
        group_id=group.id,
        principal_id='system-user',
        principal_type=PrincipalType.USER,
        source='system',
        timestamps=get_current_timestamps(),
    )
    async with service._state_db.write_tx() as tx:
        await service._members.upsert(
            tx, str(system_member.id), system_member)

    assert await service.replace_members(
        group.id, [], principal=ADMIN) == []
    assert await service.list_members(group.id, ADMIN) == []
    async with service._state_db.read_tx() as tx:
        rows = await service._members.list_by_field(
            tx, 'group_id', str(group.id))
    assert [row.payload for row in rows] == [system_member]
    assert [entry.id for entry in await service.list_external_mappings(
        group.id, ADMIN)] == [mapping.id]
    assert await service.has_all_authenticated_users(group.id, ADMIN)
    assert await service.resolve_native_memberships(
        'system-user',
        principal_type=PrincipalType.USER,
        authenticated=True,
    ) == [str(group.id)]


@pytest.mark.asyncio
async def test_non_manual_collision_is_safe_and_does_not_mutate(service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='System managed')], ADMIN))[0]
    system_member = GroupMemberEntry(
        id=GroupMemberID.generate(),
        group_id=group.id,
        principal_id='managed-user',
        principal_type=PrincipalType.USER,
        source='system',
        timestamps=get_current_timestamps(),
    )
    async with service._state_db.write_tx() as tx:
        await service._members.upsert(
            tx, str(system_member.id), system_member)

    with pytest.raises(GroupRelationConflictError, match='another source'):
        await service.replace_members(
            group.id,
            [_member('managed-user'), _member('new-user')],
            principal=ADMIN,
        )
    async with service._state_db.read_tx() as tx:
        rows = await service._members.list_by_field(
            tx, 'group_id', str(group.id))
    assert [row.payload for row in rows] == [system_member]


@pytest.mark.asyncio
async def test_member_persistence_failure_rolls_back_replacement(
    service,
    monkeypatch,
) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Rollback')], ADMIN))[0]
    before = await service.replace_members(
        group.id, [_member('old-user')], principal=ADMIN)
    original_delete = service._members.delete

    async def fail_delete(tx, member_id):
        del tx, member_id
        raise RuntimeError('injected member delete failure')

    monkeypatch.setattr(service._members, 'delete', fail_delete)
    with pytest.raises(RuntimeError, match='injected member delete failure'):
        await service.replace_members(
            group.id, [_member('new-user')], principal=ADMIN)
    monkeypatch.setattr(service._members, 'delete', original_delete)

    after = await service.list_members(group.id, ADMIN)
    assert [member.id for member in after] == [before[0].id]
    assert after[0].principal_id == 'old-user'


@pytest.mark.asyncio
async def test_member_validation_does_not_mutate_existing_rows(service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Validation')], ADMIN))[0]
    before = await service.replace_members(
        group.id, [_member('existing')], principal=ADMIN)

    with pytest.raises(GroupInputError, match='principal_id'):
        await service.replace_members(
            group.id, [_member(' ')], principal=ADMIN)
    assert [member.id for member in await service.list_members(
        group.id, ADMIN)] == [before[0].id]


@pytest.mark.asyncio
async def test_concurrent_member_replacements_never_mix_requested_sets(
    service,
) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Concurrent members')], ADMIN))[0]
    first_set = [_member('first-a'), _member('first-b')]
    second_set = [
        _member('second-service', PrincipalType.SERVICE_ACCOUNT),
        _member('second-user'),
    ]

    results = await asyncio.gather(
        service.replace_members(group.id, first_set, principal=ADMIN),
        service.replace_members(group.id, second_set, principal=ADMIN),
    )

    assert [{(entry.principal_type, entry.principal_id) for entry in result}
            for result in results] == [
                {
                    (PrincipalType.USER, 'first-a'),
                    (PrincipalType.USER, 'first-b'),
                },
                {
                    (PrincipalType.SERVICE_ACCOUNT, 'second-service'),
                    (PrincipalType.USER, 'second-user'),
                },
            ]
    final = {(entry.principal_type, entry.principal_id)
             for entry in await service.list_members(group.id, ADMIN)}
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
