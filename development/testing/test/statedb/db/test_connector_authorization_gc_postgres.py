# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from datetime import timedelta

import pytest

from statedb.config.common import ConnectorAuthorizationUseID, UserID
from statedb.config.timestamp import get_current_timestamps, now_utc
from statedb.db import StateDB
from statedb.db.connector_authorizations import (
    ConnectorAuthorizationRetentionPolicy,
    ConnectorAuthorizationService,
)
from statedb.rest.connector_authorization import (
    ConnectorAuthorizationResourceKind,
    ConnectorAuthorizationUseDBEntry,
    ConnectorCapabilityAction,
)


@dataclass(frozen=True)
class _Principal:
    user_id: UserID = UserID('postgres-user')
    user_name: str = 'Postgres User'
    kumo_group_ids: tuple[str, ...] = ('group-1', )
    external_membership_unavailable: bool = False
    is_admin: bool = False


async def _reset_postgres_schema(connection) -> None:
    await connection.execute('DROP SCHEMA public CASCADE')
    await connection.execute('CREATE SCHEMA public')


async def _issue(service: ConnectorAuthorizationService):
    return await service.issue(
        principal=_Principal(),
        auth_backend='direct_connect',
        resource_kind=ConnectorAuthorizationResourceKind.CONNECTOR,
        resource_id='connector-1',
        project_id='project-1',
        request_digest='',
        allowed_actions=(
            ConnectorCapabilityAction.GET_CONNECTOR_SAMPLE_ROWS, ),
        deployment_id='customer:instance:client',
        temporal_namespace='namespace',
    )


@pytest.mark.asyncio
async def test_postgres_concurrent_replicas_cleanup_once_and_preserve_live(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    state_db = await StateDB.open_postgres(postgres_dsn)
    policy = ConnectorAuthorizationRetentionPolicy(
        expired_retention=timedelta(0),
        revoked_retention=timedelta(0),
        authorization_batch_size=10,
        use_batch_size=10,
        opportunistic_min_interval=timedelta(days=1),
        maintenance_interval=timedelta(days=1),
        maintenance_batches_per_tick=1,
    )
    first = ConnectorAuthorizationService(
        state_db, retention_policy=policy)
    second = ConnectorAuthorizationService(
        state_db, retention_policy=policy)
    try:
        stale, _ = await _issue(first)
        active, active_token = await _issue(first)
        current = now_utc()
        stale.expires_at = current - timedelta(seconds=1)
        active.expires_at = current + timedelta(hours=1)
        use = ConnectorAuthorizationUseDBEntry(
            id=ConnectorAuthorizationUseID.generate(),
            authorization_id=stale.id,
            workflow_id='stale-workflow',
            workflow_run_id='stale-run',
            activity_id='stale-activity',
            action=ConnectorCapabilityAction.GET_CONNECTOR_SAMPLE_ROWS,
            timestamps=get_current_timestamps(),
        )
        async with state_db.write_tx() as tx:
            await first._authorizations.upsert(tx, str(stale.id), stale)
            await first._authorizations.upsert(tx, str(active.id), active)
            await first._uses.upsert(tx, str(use.id), use)

        results = await asyncio.gather(
            first.purge_stale(now=current),
            second.purge_stale(now=current),
        )

        assert sum(result.authorizations_deleted for result in results) == 1
        assert sum(result.uses_deleted for result in results) == 1
        async with state_db.read_tx() as tx:
            assert await first._authorizations.get(tx, str(stale.id)) is None
            assert await first._authorizations.get(tx, str(active.id)) is not None
        bound, newly_bound = await first.bind_workflow(
            active_token,
            deployment_id='customer:instance:client',
            temporal_namespace='namespace',
            workflow_id='active-workflow',
            expires_at=current + timedelta(hours=1),
        )
        assert newly_bound
        assert bound.id == active.id
    finally:
        await state_db.close()
        await _reset_postgres_schema(postgres_connection)
