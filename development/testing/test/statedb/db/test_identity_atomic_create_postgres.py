# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import kumoapi.data_source as api
import pytest

import statedb.db.groups as groups_module
import statedb.db.identities as identities_module
from rest.deps.authentication import UserAuthorization
from statedb.db import StateDB
from statedb.db.connector_authorizations import ConnectorAuthorizationService
from statedb.db.groups import GroupService
from statedb.db.identities import IdentityService
from statedb.rest.group import KumoGroupCreateSpec


async def _reset_postgres_schema(connection) -> None:
    await connection.execute('DROP SCHEMA public CASCADE')
    await connection.execute('CREATE SCHEMA public')


@pytest.mark.asyncio
async def test_postgres_group_delete_serializes_with_atomic_identity_create(
    postgres_dsn: str,
    postgres_connection,
    monkeypatch,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    admin = UserAuthorization.disabled()
    groups = GroupService(state_db)
    identities = IdentityService(
        state_db,
        ConnectorAuthorizationService(state_db),
        identity_validator=lambda *_: None,
    )
    try:
        group = (await groups.create_kumo_groups(
            [KumoGroupCreateSpec(display_name='Atomic identity group')],
            admin,
        ))[0]
        original_lock = identities_module.lock_group_relations
        delete_has_lock = asyncio.Event()
        create_is_waiting = asyncio.Event()
        release_delete = asyncio.Event()

        async def controlled_lock(tx, group_ids) -> None:
            task = asyncio.current_task()
            if task is not None and task.get_name() == 'delete-group':
                await original_lock(tx, group_ids)
                delete_has_lock.set()
                await asyncio.wait_for(release_delete.wait(), timeout=5)
                return
            create_is_waiting.set()
            await original_lock(tx, group_ids)

        monkeypatch.setattr(groups_module, 'lock_group_relations',
                            controlled_lock)
        monkeypatch.setattr(identities_module, 'lock_group_relations',
                            controlled_lock)

        delete_task = asyncio.create_task(
            groups.delete_kumo_group(str(group.id), admin),
            name='delete-group',
        )
        await asyncio.wait_for(delete_has_lock.wait(), timeout=5)
        create_task = asyncio.create_task(
            identities.create(
                api.CreateIdentityArgs(
                    display_name='Must not bind deleted group',
                    group_ids=[str(group.id)],
                    config=api.S3IdentityConfig(
                        authentication=api.S3AssumeRoleAuthentication(
                            partition='aws',
                            role_arn=('arn:aws:iam::123456789012:role/atomic'),
                        ), ),
                ),
                admin,
            ),
            name='create-identity',
        )
        await asyncio.wait_for(create_is_waiting.wait(), timeout=5)
        release_delete.set()
        await delete_task

        result = await asyncio.gather(create_task, return_exceptions=True)
        assert len(result) == 1
        assert isinstance(result[0], ValueError)
        assert 'Unknown or inactive groups' in str(result[0])
        async with state_db.read_tx() as tx:
            assert await identities._identities.list_all(tx) == []
            assert await identities._bindings.list_all(tx) == []
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)
