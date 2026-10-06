# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass, fields, replace
from datetime import timedelta

import pytest

from rest.backends.data_backend.activities.connector_authorization import (
    _actions_for_child_workflow,
)
from statedb.config.common import ConnectorAuthorizationUseID, UserID
from statedb.config.timestamp import get_current_timestamps, now_utc
from statedb.db import StateDB
from statedb.db.connector_authorizations import (
    ConnectorAuthorizationCleanupResult,
    ConnectorAuthorizationError,
    ConnectorAuthorizationRetentionPolicy,
    ConnectorAuthorizationService,
)
from statedb.rest.connector_authorization import (
    CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS,
    CONNECTOR_GRAPH_ACTIONS,
    ConnectorAuthorizationDBEntry,
    ConnectorAuthorizationResourceKind,
    ConnectorAuthorizationUseDBEntry,
    ConnectorCapabilityAction,
)


@dataclass(frozen=True)
class _Principal:
    user_id: UserID = UserID('user-1')
    user_name: str = 'User One'
    kumo_group_ids: tuple[str, ...] = ('group-1', )
    external_membership_unavailable: bool = False
    is_admin: bool = False
    is_disabled: bool = False
    is_proxied: bool = False


_DEPLOYMENT_ID = 'customer:instance:client'
_NAMESPACE = 'namespace'
_PROJECT_ID = 'project-1'
_RESOURCE_ID = 'connector-1'
_READ_ACTION = ConnectorCapabilityAction.GET_CONNECTOR_SAMPLE_ROWS


@pytest.fixture
def db(tmp_path):
    state_db = asyncio.run(
        StateDB.open_sqlite(str(tmp_path / 'state.db'), pool_size=1))
    try:
        yield state_db
    finally:
        asyncio.run(state_db.close())


async def _issue(
    service: ConnectorAuthorizationService,
    *,
    principal: _Principal = _Principal(),
    auth_backend: str = 'direct_connect',
    resource_id: str = _RESOURCE_ID,
    allowed_actions=(
        _READ_ACTION,
        ConnectorCapabilityAction.DELEGATE_CHILD,
    ),
):
    return await service.issue(
        principal=principal,
        auth_backend=auth_backend,
        resource_kind=ConnectorAuthorizationResourceKind.CONNECTOR,
        resource_id=resource_id,
        project_id=_PROJECT_ID,
        request_digest='',
        allowed_actions=allowed_actions,
        deployment_id=_DEPLOYMENT_ID,
        temporal_namespace=_NAMESPACE,
    )


async def _bind(
    service: ConnectorAuthorizationService,
    token: str,
    workflow_id: str = 'workflow-1',
    *,
    expires_at=None,
):
    return await service.bind_workflow(
        token,
        deployment_id=_DEPLOYMENT_ID,
        temporal_namespace=_NAMESPACE,
        workflow_id=workflow_id,
        expires_at=expires_at or now_utc() + timedelta(hours=1),
    )


def _activity_scope(**overrides):
    scope = dict(
        deployment_id=_DEPLOYMENT_ID,
        temporal_namespace=_NAMESPACE,
        workflow_id='workflow-1',
        workflow_run_id='run-1',
        activity_id='activity-1',
        action=_READ_ACTION,
        resource_kind=ConnectorAuthorizationResourceKind.CONNECTOR,
        resource_id=_RESOURCE_ID,
        project_id=_PROJECT_ID,
    )
    scope.update(overrides)
    return scope


async def _authorize(
    service: ConnectorAuthorizationService,
    token: str,
    **overrides,
):
    return await service.authorize_activity(
        token,
        **_activity_scope(**overrides),
    )


async def _delegating_parent(
    service: ConnectorAuthorizationService,
    *,
    resource_id: str,
    allowed_actions,
    workflow_id: str,
    activity_id: str,
):
    _, token = await _issue(
        service,
        resource_id=resource_id,
        allowed_actions=allowed_actions,
    )
    await _bind(service, token, workflow_id)
    return await _authorize(
        service,
        token,
        workflow_id=workflow_id,
        workflow_run_id='parent-run',
        activity_id=activity_id,
        action=ConnectorCapabilityAction.DELEGATE_CHILD,
        resource_id=resource_id,
    ), token


def test_durable_capability_does_not_snapshot_membership_or_admin() -> None:
    durable_fields = {
        field.name
        for field in fields(ConnectorAuthorizationDBEntry)
    }
    assert 'principal_group_ids' not in durable_fields
    assert 'principal_was_admin' not in durable_fields


def _retention_policy(
    *,
    retention: timedelta = timedelta(days=7),
    authorization_batch_size: int = 100,
    use_batch_size: int = 1000,
    opportunistic_min_interval: timedelta = timedelta(days=1),
    maintenance_interval: timedelta = timedelta(days=1),
    maintenance_batches_per_tick: int = 4,
) -> ConnectorAuthorizationRetentionPolicy:
    return ConnectorAuthorizationRetentionPolicy(
        expired_retention=retention,
        revoked_retention=retention,
        authorization_batch_size=authorization_batch_size,
        use_batch_size=use_batch_size,
        opportunistic_min_interval=opportunistic_min_interval,
        maintenance_interval=maintenance_interval,
        maintenance_batches_per_tick=maintenance_batches_per_tick,
    )


async def _set_inactive_time(
    service: ConnectorAuthorizationService,
    entry: ConnectorAuthorizationDBEntry,
    *,
    expires_at,
    revoked_at=None,
) -> None:
    entry.expires_at = expires_at
    entry.revoked_at = revoked_at
    async with service._state_db.write_tx() as tx:
        await service._authorizations.upsert(tx, str(entry.id), entry)


async def _add_use(
    service: ConnectorAuthorizationService,
    entry: ConnectorAuthorizationDBEntry,
    suffix: str,
) -> ConnectorAuthorizationUseDBEntry:
    use = ConnectorAuthorizationUseDBEntry(
        id=ConnectorAuthorizationUseID.generate(),
        authorization_id=entry.id,
        workflow_id=f'workflow-{suffix}',
        workflow_run_id=f'run-{suffix}',
        activity_id=f'activity-{suffix}',
        action=ConnectorCapabilityAction.GET_CONNECTOR_SAMPLE_ROWS,
        timestamps=get_current_timestamps(),
    )
    async with service._state_db.write_tx() as tx:
        await service._uses.upsert(tx, str(use.id), use)
    return use


async def _stored_authorization_ids(
    service: ConnectorAuthorizationService, ) -> set[str]:
    async with service._state_db.read_tx() as tx:
        return {row.id for row in await service._authorizations.list_all(tx)}


async def _stored_use_count(service: ConnectorAuthorizationService, ) -> int:
    async with service._state_db.read_tx() as tx:
        return len(await service._uses.list_all(tx))


def test_retention_policy_defaults_and_environment_overrides(
        monkeypatch) -> None:
    defaults = ConnectorAuthorizationRetentionPolicy()
    assert defaults.expired_retention == timedelta(days=7)
    assert defaults.revoked_retention == timedelta(days=7)

    monkeypatch.setenv('CONNECTOR_AUTHORIZATION_EXPIRED_RETENTION_SECONDS',
                       '11')
    monkeypatch.setenv('CONNECTOR_AUTHORIZATION_REVOKED_RETENTION_SECONDS',
                       '12')
    monkeypatch.setenv('CONNECTOR_AUTHORIZATION_GC_BATCH_SIZE', '13')
    monkeypatch.setenv('CONNECTOR_AUTHORIZATION_USE_GC_BATCH_SIZE', '14')
    monkeypatch.setenv('CONNECTOR_AUTHORIZATION_GC_MIN_INTERVAL_SECONDS', '15')
    monkeypatch.setenv(
        'CONNECTOR_AUTHORIZATION_GC_MAINTENANCE_INTERVAL_SECONDS', '16')
    monkeypatch.setenv('CONNECTOR_AUTHORIZATION_GC_BATCHES_PER_TICK', '17')

    configured = ConnectorAuthorizationRetentionPolicy.from_env()
    assert configured == ConnectorAuthorizationRetentionPolicy(
        expired_retention=timedelta(seconds=11),
        revoked_retention=timedelta(seconds=12),
        authorization_batch_size=13,
        use_batch_size=14,
        opportunistic_min_interval=timedelta(seconds=15),
        maintenance_interval=timedelta(seconds=16),
        maintenance_batches_per_tick=17,
    )


@pytest.mark.parametrize(
    'override',
    [
        {
            'expired_retention': timedelta(seconds=-1)
        },
        {
            'revoked_retention': timedelta(seconds=-1)
        },
        {
            'authorization_batch_size': 0
        },
        {
            'use_batch_size': 0
        },
        {
            'opportunistic_min_interval': timedelta(seconds=-1)
        },
        {
            'maintenance_interval': timedelta(0)
        },
        {
            'maintenance_batches_per_tick': 0
        },
    ],
    ids=[
        'expired-retention',
        'revoked-retention',
        'authorization-batch',
        'use-batch',
        'opportunistic-interval',
        'maintenance-interval',
        'maintenance-batches',
    ],
)
def test_retention_policy_rejects_unbounded_or_invalid_configuration(
    override, ) -> None:
    with pytest.raises(ValueError):
        ConnectorAuthorizationRetentionPolicy(**override)


@pytest.mark.asyncio
async def test_capability_binds_run_and_activity_retry_is_idempotent(
        db: StateDB) -> None:
    service = ConnectorAuthorizationService(db)
    entry, token = await _issue(service)
    assert token not in repr(entry)
    async with db.read_tx() as tx:
        row = await service._authorizations.get(tx, str(entry.id))
    assert row is not None
    assert row.payload.token_hash != token

    bound, newly_bound = await _bind(service, token)
    assert bound.workflow_id == 'workflow-1'
    assert newly_bound

    first = await _authorize(service, token)
    retried = await _authorize(service, token)
    assert first.workflow_run_id == retried.workflow_run_id == 'run-1'
    async with db.read_tx() as tx:
        uses = await service._uses.list_all(tx)
    assert len(uses) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'override',
    [
        {
            'workflow_id': 'workflow-2'
        },
        {
            'workflow_run_id': 'run-2'
        },
        {
            'resource_id': 'connector-2'
        },
        {
            'project_id': 'project-2'
        },
        {
            'action': ConnectorCapabilityAction.WRITE_ARTIFACTS_TO_CUSTOMER
        },
    ],
    ids=['workflow', 'run', 'resource', 'project', 'action'],
)
async def test_capability_rejects_cross_scope_replay(
    db: StateDB,
    override,
) -> None:
    service = ConnectorAuthorizationService(db)
    _, token = await _issue(service)
    await _bind(service, token)
    if override.get('workflow_run_id') == 'run-2':
        await _authorize(service, token)
    with pytest.raises(ConnectorAuthorizationError):
        await _authorize(service, token, **override)


@pytest.mark.asyncio
async def test_capability_rejects_guess_expiry_and_revocation(
        db: StateDB) -> None:
    service = ConnectorAuthorizationService(db)
    _, token = await _issue(service)
    with pytest.raises(ConnectorAuthorizationError):
        await _bind(service, token + '-guess')


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['disabled', 'proxied'])
async def test_request_principal_flags_can_never_mint_internal_service(
    db: StateDB,
    mode: str,
) -> None:
    service = ConnectorAuthorizationService(db)
    principal = _Principal(
        user_id=UserID('internal_service:scheduled_workflow'),
        is_disabled=mode == 'disabled',
        is_proxied=mode == 'proxied',
    )
    entry, _ = await _issue(
        service,
        principal=principal,
        auth_backend=f'native_{mode}',
        allowed_actions=(_READ_ACTION, ),
    )
    assert entry.principal_id == 'internal_service:scheduled_workflow'
    assert entry.auth_backend == f'native_{mode}'


@pytest.mark.asyncio
async def test_expired_capability_is_rejected(db: StateDB, monkeypatch) -> None:
    service = ConnectorAuthorizationService(db)
    _, token = await _issue(service)
    current = now_utc()
    monkeypatch.setattr(
        'statedb.db.connector_authorizations.now_utc',
        lambda: current + timedelta(days=1),
    )
    with pytest.raises(ConnectorAuthorizationError):
        await _bind(
            service,
            token,
            expires_at=current + timedelta(days=2),
        )


@pytest.mark.asyncio
async def test_child_delegation_is_exact_and_cannot_expand_scope(
        db: StateDB) -> None:
    service = ConnectorAuthorizationService(db)
    _, parent_token = await _issue(service)
    await _bind(service, parent_token, 'parent-workflow')
    parent = await _authorize(
        service,
        parent_token,
        workflow_id='parent-workflow',
        workflow_run_id='parent-run',
        activity_id='delegate-activity',
        action=ConnectorCapabilityAction.DELEGATE_CHILD,
    )
    child, child_token = await service.delegate_to_child(
        parent,
        child_workflow_id='child-workflow',
        child_actions=(ConnectorCapabilityAction.GET_CONNECTOR_SAMPLE_ROWS, ),
    )
    assert child.parent_authorization_id == parent.id
    assert child.workflow_id == 'child-workflow'
    assert child_token != parent_token
    await _authorize(
        service,
        child_token,
        workflow_id='child-workflow',
        workflow_run_id='child-run',
        activity_id='child-activity',
    )
    with pytest.raises(ConnectorAuthorizationError):
        await service.delegate_to_child(
            parent,
            child_workflow_id='overbroad-child',
            child_actions=(
                ConnectorCapabilityAction.WRITE_ARTIFACTS_TO_CUSTOMER, ),
        )
    forged_parent = replace(
        parent,
        allowed_actions=(
            *parent.allowed_actions,
            ConnectorCapabilityAction.WRITE_ARTIFACTS_TO_CUSTOMER,
        ),
        project_id='forged-project',
    )
    with pytest.raises(ConnectorAuthorizationError,
                       match='stored parent scope'):
        await service.delegate_to_child(
            forged_parent,
            child_workflow_id='forged-action-child',
            child_actions=(
                ConnectorCapabilityAction.WRITE_ARTIFACTS_TO_CUSTOMER, ),
        )
    stored_scope_child, _ = await service.delegate_to_child(
        forged_parent,
        child_workflow_id='forged-project-child',
        child_actions=(ConnectorCapabilityAction.GET_CONNECTOR_SAMPLE_ROWS, ),
    )
    assert stored_scope_child.project_id == parent.project_id
    assert stored_scope_child.resource_id == parent.resource_id
    assert child.principal_id == parent.principal_id
    assert child.principal_name == parent.principal_name


@pytest.mark.asyncio
async def test_graph_parent_delegates_exact_diskgraph_snapshot_authority(
    db: StateDB, ) -> None:
    """Reproduce GraphRefresh -> dg-ingest -> ensure_snapshot authority."""
    service = ConnectorAuthorizationService(db)
    parent, _ = await _delegating_parent(
        service,
        resource_id='warehouse-connector',
        allowed_actions=CONNECTOR_GRAPH_ACTIONS,
        workflow_id='graph-refresh',
        activity_id='delegate-dg-ingest',
    )

    child_actions = _actions_for_child_workflow('DiskgraphIngestWorkflow')
    child, child_token = await service.delegate_to_child(
        parent,
        child_workflow_id='dg-ingest:graph@ver=1',
        child_actions=child_actions,
    )
    assert child_actions == {ConnectorCapabilityAction.ENSURE_SNAPSHOT}
    assert set(
        child.allowed_actions) == {ConnectorCapabilityAction.ENSURE_SNAPSHOT}
    await _authorize(
        service,
        child_token,
        workflow_id='dg-ingest:graph@ver=1',
        workflow_run_id='dg-run',
        activity_id='ensure-snapshot',
        action=ConnectorCapabilityAction.ENSURE_SNAPSHOT,
        resource_id='warehouse-connector',
    )

    with pytest.raises(ConnectorAuthorizationError,
                       match='action is not allowed'):
        await _authorize(
            service,
            child_token,
            workflow_id='dg-ingest:graph@ver=1',
            workflow_run_id='dg-run',
            activity_id='unrelated-table-action',
            resource_id='warehouse-connector',
        )


@pytest.mark.asyncio
async def test_training_postprocess_uses_only_child_workflow_authority(
    db: StateDB, ) -> None:
    service = ConnectorAuthorizationService(db)
    parent, parent_token = await _delegating_parent(
        service,
        resource_id='warehouse-connector',
        allowed_actions=CONNECTOR_GRAPH_ACTIONS,
        workflow_id='training-parent',
        activity_id='delegate-training-job',
    )
    child_actions = _actions_for_child_workflow('TrainingJobWorkflow')
    assert child_actions == {
        ConnectorCapabilityAction.POSTPROCESS_TRAINING_JOB_OUTPUT,
    }
    _, child_token = await service.delegate_to_child(
        parent,
        child_workflow_id='training-job-child',
        child_actions=child_actions,
    )

    await _authorize(
        service,
        child_token,
        workflow_id='training-job-child',
        workflow_run_id='child-run',
        activity_id='postprocess',
        action=ConnectorCapabilityAction.POSTPROCESS_TRAINING_JOB_OUTPUT,
        resource_id='warehouse-connector',
    )

    common = dict(
        workflow_run_id='confused-run',
        activity_id='confused-postprocess',
        action=ConnectorCapabilityAction.POSTPROCESS_TRAINING_JOB_OUTPUT,
        resource_id='warehouse-connector',
    )
    with pytest.raises(ConnectorAuthorizationError,
                       match='workflow provenance mismatch'):
        await _authorize(
            service,
            parent_token,
            workflow_id='training-job-child',
            **common,
        )
    with pytest.raises(ConnectorAuthorizationError,
                       match='workflow provenance mismatch'):
        await _authorize(
            service,
            child_token,
            workflow_id='training-parent',
            **common,
        )


@pytest.mark.asyncio
async def test_batch_prediction_delegates_exact_artifact_output_authority(
    db: StateDB, ) -> None:
    service = ConnectorAuthorizationService(db)
    parent, _ = await _delegating_parent(
        service,
        resource_id='output-connector',
        allowed_actions=CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS,
        workflow_id='batch-prediction-parent',
        activity_id='delegate-batch-prediction-job',
    )
    batch_actions = _actions_for_child_workflow('BatchPredictionJobWorkflow')
    assert batch_actions == set(
        CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS)
    batch_child, batch_token = await service.delegate_to_child(
        parent,
        child_workflow_id='batch-prediction-job-child',
        child_actions=batch_actions,
    )
    batch_child = await _authorize(
        service,
        batch_token,
        workflow_id='batch-prediction-job-child',
        workflow_run_id='batch-run',
        activity_id='delegate-artifact-export',
        action=ConnectorCapabilityAction.DELEGATE_CHILD,
        resource_id='output-connector',
    )
    artifact_actions = _actions_for_child_workflow('ArtifactExportWorkflow')
    artifact_child, artifact_token = await service.delegate_to_child(
        batch_child,
        child_workflow_id='artifact-export-child',
        child_actions=artifact_actions,
    )
    assert set(artifact_child.allowed_actions) == artifact_actions
    assert ConnectorCapabilityAction.DELEGATE_CHILD not in artifact_actions
    assert ConnectorCapabilityAction.START_BATCH_PREDICTION_STATS not in (
        artifact_actions)
    assert ConnectorCapabilityAction.CLEANUP_DATABRICKS_VOLUME_OUTPUTS not in (
        artifact_actions)
    await _authorize(
        service,
        artifact_token,
        workflow_id='artifact-export-child',
        workflow_run_id='artifact-run',
        activity_id='write-artifacts',
        action=ConnectorCapabilityAction.WRITE_ARTIFACTS_TO_CUSTOMER,
        resource_id='output-connector',
    )
    with pytest.raises(ConnectorAuthorizationError,
                       match='action is not allowed'):
        await _authorize(
            service,
            artifact_token,
            workflow_id='artifact-export-child',
            workflow_run_id='artifact-run',
            activity_id='unrelated-read',
            resource_id='output-connector',
        )


@pytest.mark.asyncio
async def test_cancel_revoke_is_scoped_to_deployment_and_namespace(
        db: StateDB) -> None:
    service = ConnectorAuthorizationService(db)
    _, token = await _issue(service)
    await _bind(service, token)
    assert await service.revoke_workflow(
        workflow_id='workflow-1',
        deployment_id='other:instance:client',
        temporal_namespace=_NAMESPACE,
    ) == 0
    assert await service.revoke_workflow(
        workflow_id='workflow-1',
        deployment_id=_DEPLOYMENT_ID,
        temporal_namespace=_NAMESPACE,
    ) == 1
    with pytest.raises(ConnectorAuthorizationError):
        await _authorize(service, token)
    assert await service.revoke(token)
    with pytest.raises(ConnectorAuthorizationError):
        await _bind(service, token)


@pytest.mark.asyncio
async def test_cleanup_retains_live_and_recent_rows_and_cascades_old_uses(
    db: StateDB,
    caplog,
) -> None:
    service = ConnectorAuthorizationService(
        db, retention_policy=_retention_policy())
    issued = [await _issue(service) for _ in range(5)]
    entries = [item[0] for item in issued]
    tokens = [item[1] for item in issued]
    current = now_utc()
    active, recent_expired, old_expired, recent_revoked, old_revoked = entries
    await _set_inactive_time(service, active,
                             expires_at=current + timedelta(days=1))
    await _set_inactive_time(service, recent_expired,
                             expires_at=current - timedelta(days=1))
    await _set_inactive_time(service, old_expired,
                             expires_at=current - timedelta(days=8))
    await _set_inactive_time(
        service,
        recent_revoked,
        expires_at=current + timedelta(days=1),
        revoked_at=current - timedelta(days=1),
    )
    await _set_inactive_time(
        service,
        old_revoked,
        expires_at=current + timedelta(days=1),
        revoked_at=current - timedelta(days=8),
    )
    await _add_use(service, old_expired, 'expired')
    await _add_use(service, old_revoked, 'revoked')

    result = await service.purge_stale(now=current)

    assert result == ConnectorAuthorizationCleanupResult(
        authorizations_deleted=2, uses_deleted=2)
    assert await _stored_authorization_ids(service) == {
        str(active.id),
        str(recent_expired.id),
        str(recent_revoked.id),
    }
    assert await _stored_use_count(service) == 0
    assert vars(result) == {
        'authorizations_deleted': 2,
        'uses_deleted': 2,
    }
    for secret in (*tokens, _Principal().user_name, str(_Principal().user_id)):
        assert secret not in repr(result)
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_cleanup_batches_are_bounded_and_repeated_calls_drain_backlog(
    db: StateDB, ) -> None:
    service = ConnectorAuthorizationService(
        db,
        retention_policy=_retention_policy(
            retention=timedelta(0),
            authorization_batch_size=2,
            use_batch_size=2,
            maintenance_batches_per_tick=2,
        ),
    )
    entries = [(await _issue(service))[0] for _ in range(3)]
    current = now_utc()
    for index, entry in enumerate(entries):
        await _set_inactive_time(service, entry,
                                 expires_at=current - timedelta(seconds=1))
        await _add_use(service, entry, f'{index}-first')
        await _add_use(service, entry, f'{index}-second')

    results = []
    for _ in range(10):
        result = await service.purge_stale(now=current)
        results.append(result)
        assert result.authorizations_deleted <= 2
        assert result.uses_deleted <= 2
        if not await _stored_authorization_ids(service):
            break

    assert await _stored_authorization_ids(service) == set()
    assert await _stored_use_count(service) == 0
    assert sum(item.authorizations_deleted for item in results) == 3
    assert sum(item.uses_deleted for item in results) == 6


@pytest.mark.asyncio
async def test_maintenance_tick_caps_work_then_next_tick_drains(
    db: StateDB, ) -> None:
    service = ConnectorAuthorizationService(
        db,
        retention_policy=_retention_policy(
            retention=timedelta(0),
            authorization_batch_size=2,
            maintenance_batches_per_tick=2,
        ),
    )
    entries = [(await _issue(service))[0] for _ in range(5)]
    current = now_utc()
    for entry in entries:
        await _set_inactive_time(service, entry,
                                 expires_at=current - timedelta(seconds=1))

    first = await service.run_maintenance_tick(now=current)
    assert first.authorizations_deleted == 4
    assert len(await _stored_authorization_ids(service)) == 1

    second = await service.run_maintenance_tick(now=current)
    assert second.authorizations_deleted == 1
    assert await _stored_authorization_ids(service) == set()


@pytest.mark.asyncio
async def test_periodic_maintenance_purges_idle_backlog_and_stops_cleanly(
    db: StateDB, ) -> None:
    service = ConnectorAuthorizationService(
        db,
        retention_policy=_retention_policy(
            retention=timedelta(0),
            maintenance_interval=timedelta(milliseconds=10),
            maintenance_batches_per_tick=1,
        ),
    )
    entry, _ = await _issue(service)
    await service.start_maintenance()
    try:
        await _set_inactive_time(
            service,
            entry,
            expires_at=now_utc() - timedelta(seconds=1),
        )
        for _ in range(100):
            if not await _stored_authorization_ids(service):
                break
            await asyncio.sleep(0.01)
        assert await _stored_authorization_ids(service) == set()
    finally:
        await service.stop_maintenance()
    assert service._maintenance_task is None


@pytest.mark.asyncio
async def test_write_volume_forces_cleanup_at_or_above_ingest_capacity(
    db: StateDB, ) -> None:
    service = ConnectorAuthorizationService(
        db,
        retention_policy=_retention_policy(
            retention=timedelta(0),
            authorization_batch_size=3,
            opportunistic_min_interval=timedelta(days=1),
        ),
    )
    stale = [(await _issue(service))[0] for _ in range(9)]
    current = now_utc()
    for entry in stale:
        await _set_inactive_time(service, entry,
                                 expires_at=current - timedelta(seconds=1))

    new_entries = [(await _issue(service))[0] for _ in range(9)]

    stored = await _stored_authorization_ids(service)
    assert not stored.intersection(str(entry.id) for entry in stale)
    assert stored == {str(entry.id) for entry in new_entries}


@pytest.mark.asyncio
async def test_cleanup_is_safe_with_concurrent_issue_validate_delegate_and_retry(
    db: StateDB, ) -> None:
    service = ConnectorAuthorizationService(
        db,
        retention_policy=_retention_policy(
            retention=timedelta(0),
            authorization_batch_size=2,
            use_batch_size=2,
        ),
    )
    stale, _ = await _issue(service)
    parent, parent_token = await _issue(service)
    current = now_utc()
    await _set_inactive_time(service, stale,
                             expires_at=current - timedelta(seconds=1))
    await _bind(
        service,
        parent_token,
        'parent-workflow',
        expires_at=current + timedelta(hours=1),
    )
    activity_scope = dict(
        workflow_id='parent-workflow',
        workflow_run_id='parent-run',
        activity_id='parent-activity',
    )
    parent = await _authorize(service, parent_token, **activity_scope)

    cleanup, retried, delegated, issued = await asyncio.gather(
        service.run_maintenance_tick(now=current),
        _authorize(service, parent_token, **activity_scope),
        service.delegate_to_child(
            parent,
            child_workflow_id='child-workflow',
            child_actions=(_READ_ACTION, ),
        ),
        _issue(service),
    )

    assert cleanup.authorizations_deleted in (0, 1)
    assert retried.id == parent.id
    child, child_token = delegated
    assert issued[0].id != parent.id
    await _authorize(
        service,
        child_token,
        workflow_id='child-workflow',
        workflow_run_id='child-run',
        activity_id='child-activity',
    )
    assert str(stale.id) not in await _stored_authorization_ids(service)
