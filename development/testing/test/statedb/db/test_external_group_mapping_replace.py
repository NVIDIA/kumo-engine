# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import pytest_asyncio

from rest.config.encryption import decrypt_idp_id
from rest.deps.authentication import UserAuthorization
from statedb.db import StateDB
from statedb.db.groups import GroupService
from statedb.rest.group import (
    ExternalGroupMappingSpec,
    GroupStatus,
    KumoGroupCreateSpec,
)
from statedb.utils.errors import GroupInputError, GroupRelationConflictError

ADMIN = UserAuthorization.disabled()


def _mapping(provider: str, external_id: str) -> ExternalGroupMappingSpec:
    return ExternalGroupMappingSpec(
        identity_provider_id=provider,
        external_group_id=external_id,
    )


@pytest_asyncio.fixture
async def service(tmp_path):
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    try:
        yield GroupService(db)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_replace_external_mappings_is_normalized_encrypted_and_or_resolved(
    service,
) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Engineering')], ADMIN))[0]

    entries = await service.replace_external_mappings(
        group.id,
        [
            _mapping(' DIRECT_CONNECT ', ' engineering '),
            _mapping('direct_connect', 'data-science'),
            _mapping('direct_connect', 'engineering'),
        ],
        principal=ADMIN,
    )

    assert [entry.identity_provider_id for entry in entries] == [
        'direct_connect',
        'direct_connect',
    ]
    assert [decrypt_idp_id(entry.encrypted_external_group_id)
            for entry in entries] == ['data-science', 'engineering']
    assert all(entry.encrypted_external_group_id not in {
        'data-science', 'engineering'
    } for entry in entries)
    assert await service.map_idp_group_ids(
        ['engineering'], identity_provider_id='direct_connect') == [
            str(group.id)
        ]
    assert await service.map_idp_group_ids(
        ['data-science'], identity_provider_id='direct_connect') == [
            str(group.id)
        ]
    assert await service.map_idp_group_ids(
        ['engineering', 'data-science', 'engineering'],
        identity_provider_id='direct_connect',
    ) == [str(group.id)]


@pytest.mark.asyncio
async def test_replace_preserves_retained_ids_deletes_omissions_and_can_clear(
    service,
) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Data')], ADMIN))[0]
    before = await service.replace_external_mappings(
        group.id,
        [_mapping('direct_connect', 'keep'),
         _mapping('direct_connect', 'remove')],
        principal=ADMIN,
    )
    keep_id = next(entry.id for entry in before
                   if decrypt_idp_id(entry.encrypted_external_group_id) == 'keep')

    after = await service.replace_external_mappings(
        group.id,
        [_mapping('DIRECT_CONNECT', ' keep '),
         _mapping('direct_connect', 'new')],
        principal=ADMIN,
    )
    assert next(entry.id for entry in after
                if decrypt_idp_id(entry.encrypted_external_group_id) == 'keep'
                ) == keep_id
    assert {decrypt_idp_id(entry.encrypted_external_group_id)
            for entry in after} == {'keep', 'new'}

    assert await service.replace_external_mappings(
        group.id, [], principal=ADMIN) == []
    assert await service.list_external_mappings(group.id, ADMIN) == []


@pytest.mark.asyncio
async def test_cross_group_conflict_rolls_back_entire_replacement(service) -> None:
    first, second = await service.create_kumo_groups([
        KumoGroupCreateSpec(display_name='First'),
        KumoGroupCreateSpec(display_name='Second'),
    ], ADMIN)
    before = await service.replace_external_mappings(
        first.id,
        [_mapping('direct_connect', 'keep'),
         _mapping('direct_connect', 'remove')],
        principal=ADMIN,
    )
    await service.replace_external_mappings(
        second.id,
        [_mapping('direct_connect', 'owned-by-second')],
        principal=ADMIN,
    )

    with pytest.raises(GroupRelationConflictError, match='already mapped'):
        await service.replace_external_mappings(
            first.id,
            [
                _mapping('direct_connect', 'keep'),
                _mapping('direct_connect', 'new'),
                _mapping('direct_connect', 'owned-by-second'),
            ],
            principal=ADMIN,
        )

    assert [entry.id for entry in await service.list_external_mappings(
        first.id, ADMIN)] == [entry.id for entry in before]


@pytest.mark.asyncio
async def test_persistence_failure_rolls_back_insert_and_delete(
    service,
    monkeypatch,
) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Rollback')], ADMIN))[0]
    before = await service.replace_external_mappings(
        group.id,
        [_mapping('direct_connect', 'old')],
        principal=ADMIN,
    )
    original_delete = service._external_mappings.delete

    async def fail_delete(tx, mapping_id):
        del tx, mapping_id
        raise RuntimeError('injected delete failure')

    monkeypatch.setattr(service._external_mappings, 'delete', fail_delete)
    with pytest.raises(RuntimeError, match='injected delete failure'):
        await service.replace_external_mappings(
            group.id,
            [_mapping('direct_connect', 'new')],
            principal=ADMIN,
        )
    monkeypatch.setattr(service._external_mappings, 'delete', original_delete)

    after = await service.list_external_mappings(group.id, ADMIN)
    assert [entry.id for entry in after] == [before[0].id]
    assert decrypt_idp_id(after[0].encrypted_external_group_id) == 'old'


@pytest.mark.asyncio
async def test_replace_rejects_empty_values_without_mutating(service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Validation')], ADMIN))[0]
    before = await service.replace_external_mappings(
        group.id, [_mapping('direct_connect', 'existing')], principal=ADMIN)

    for mapping, field_name in (
        (_mapping(' ', 'valid'), 'identity_provider_id'),
        (_mapping('direct_connect', ' '), 'external_group_id'),
    ):
        with pytest.raises(GroupInputError, match=field_name):
            await service.replace_external_mappings(
                group.id, [mapping], principal=ADMIN)
    assert [entry.id for entry in await service.list_external_mappings(
        group.id, ADMIN)] == [before[0].id]


@pytest.mark.asyncio
async def test_deleted_groups_do_not_resolve_external_membership(service) -> None:
    group = (await service.create_kumo_groups(
        [KumoGroupCreateSpec(display_name='Disabled')], ADMIN))[0]
    await service.replace_external_mappings(
        group.id, [_mapping('direct_connect', 'external')], principal=ADMIN)
    async with service._state_db.write_tx() as tx:
        row = await service._groups.get(tx, str(group.id))
        assert row is not None
        row.payload.status = GroupStatus.DELETED
        await service._groups.upsert(tx, row.id, row.payload)

    assert await service.map_idp_group_ids(['external']) == []


@pytest.mark.asyncio
async def test_concurrent_cross_group_replacements_are_unique_and_atomic(
    service,
) -> None:
    first, second = await service.create_kumo_groups([
        KumoGroupCreateSpec(display_name='Race First'),
        KumoGroupCreateSpec(display_name='Race Second'),
    ], ADMIN)
    results = await asyncio.gather(
        service.replace_external_mappings(
            first.id,
            [_mapping('direct_connect', 'first-only'),
             _mapping('direct_connect', 'contended')],
            principal=ADMIN,
        ),
        service.replace_external_mappings(
            second.id,
            [_mapping('direct_connect', 'second-only'),
             _mapping('direct_connect', 'contended')],
            principal=ADMIN,
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, list) for result in results) == 1
    assert sum(isinstance(result, GroupRelationConflictError)
               for result in results) == 1
    first_ids = {decrypt_idp_id(entry.encrypted_external_group_id)
                 for entry in await service.list_external_mappings(
                     first.id, ADMIN)}
    second_ids = {decrypt_idp_id(entry.encrypted_external_group_id)
                  for entry in await service.list_external_mappings(
                      second.id, ADMIN)}
    assert (first_ids, second_ids) in (
        ({'first-only', 'contended'}, set()),
        (set(), {'second-only', 'contended'}),
    )
