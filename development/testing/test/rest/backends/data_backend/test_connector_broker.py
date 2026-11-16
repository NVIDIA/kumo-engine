# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from rest.backends.data_backend.connector_broker import (
    ConnectorCredentialBroker,
    ConnectorCredentialBrokerTimeout,
)
from rest.config.data_backend import (
    ActivityAuthorizationRef,
    ConnectorExecutionRef,
)
from statedb.config.common import ConnectorID, ProjectID
from statedb.rest.connector_authorization import (
    ConnectorAuthorizationResourceKind,
    ConnectorCapabilityAction,
)
from statedb.utils.errors import (
    ConnectorDoesNotExistError,
    IdentityForbiddenError,
)

_CONNECTOR_ID = ConnectorID('connector_' + 'b' * 32)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


@pytest.fixture
def activity_context(monkeypatch):
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_broker.activity.in_activity',
        lambda: True,
    )
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_broker.activity.info',
        lambda: SimpleNamespace(
            activity_id='activity-1',
            activity_type='get_connector_sample_rows',
            namespace='namespace',
            workflow_id='workflow-1',
            workflow_run_id='run-1',
        ),
    )


def _reference() -> ConnectorExecutionRef:
    return ConnectorExecutionRef(
        project_id=ProjectID.generate(),
        connector_id=ConnectorID.generate(),
    )


def _delegation_broker(recheck_effects, *, delegated=None):
    references = [_reference(), _reference()]
    action = ConnectorCapabilityAction.WRITE_ARTIFACTS_TO_CUSTOMER
    parent = SimpleNamespace(allowed_actions=(
        ConnectorCapabilityAction.DELEGATE_CHILD,
        action,
    ))
    recheck = AsyncMock(side_effect=recheck_effects)
    delegate = AsyncMock(return_value=delegated)
    revoke = AsyncMock(return_value=True)
    services = SimpleNamespace(
        connectors=SimpleNamespace(require_delegatable_for_project=recheck),
        connector_authorizations=SimpleNamespace(
            delegate_to_child=delegate,
            revoke=revoke,
        ),
    )
    broker = ConnectorCredentialBroker()
    broker._authorize_reference = AsyncMock(return_value=parent)
    broker.run = Mock(
        side_effect=lambda operation: asyncio.run(operation(services)))
    return broker, references, action, parent, recheck, delegate, revoke


def test_broker_resolves_only_the_durable_project_connector_reference() -> None:
    broker = ConnectorCredentialBroker()
    project_id = ProjectID.generate()
    reference = ConnectorExecutionRef(
        connector_id=_CONNECTOR_ID,
        project_id=project_id,
    )

    resolved = object()
    resolve_for_activity = AsyncMock(return_value=resolved)
    services = SimpleNamespace(connectors=SimpleNamespace(
        resolve_for_activity=resolve_for_activity))

    def run(operation):
        return asyncio.run(operation(services))

    broker.run = Mock(side_effect=run)
    broker._authorize_reference = AsyncMock(return_value=object())

    with patch('rest.backends.data_backend.connector_broker.activity.info',
               return_value=SimpleNamespace()):
        assert broker.resolve(reference) is resolved

    resolve_for_activity.assert_awaited_once_with(
        project_id,
        _CONNECTOR_ID,
    )
    broker.run.assert_called_once()


@pytest.mark.parametrize(
    ('project_owned', 'rechecks_principal'),
    [(True, False), (False, True)],
    ids=['accepted-project-work', 'management-work'],
)
def test_broker_rechecks_principal_only_for_management_work(
    activity_context,
    monkeypatch,
    project_owned,
    rechecks_principal,
) -> None:
    authorization = object()
    authorize = AsyncMock(return_value=authorization)
    resolver = SimpleNamespace(resolve=AsyncMock(
        side_effect=PermissionError('submitter was demoted')))
    services = SimpleNamespace(connector_authorizations=SimpleNamespace(
        authorize_activity=authorize))
    broker = ConnectorCredentialBroker()
    broker._services = services
    broker._principal_authority = resolver
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_broker.deploy_env',
        lambda: SimpleNamespace(
            instance_id='instance',
            client_id='client',
        ),
    )

    method = (broker.authorize_project_activity_scope
              if project_owned else broker.authorize_activity_scope)
    kwargs = dict(
        resource_kind=(
            ConnectorAuthorizationResourceKind.PROJECT_CONNECTOR_CREATE
            if project_owned else
            ConnectorAuthorizationResourceKind.IDENTITY_VALIDATE),
        resource_id='identity-1',
        project_id='project-1' if project_owned else '',
        request_digest='digest',
        action=(ConnectorCapabilityAction.CREATE_PROJECT_CONNECTOR if
                project_owned else ConnectorCapabilityAction.VALIDATE_IDENTITY),
    )
    if rechecks_principal:
        with pytest.raises(PermissionError, match='demoted'):
            method(ActivityAuthorizationRef('opaque-token'), **kwargs)
        resolver.resolve.assert_awaited_once_with(authorization)
    else:
        assert method(ActivityAuthorizationRef('opaque-token'),
                      **kwargs) is authorization
        resolver.resolve.assert_not_awaited()


def test_broker_rejects_resolution_outside_an_activity() -> None:
    broker = ConnectorCredentialBroker()
    reference = ConnectorExecutionRef(
        connector_id=_CONNECTOR_ID,
        project_id=ProjectID.generate(),
    )

    with pytest.raises(RuntimeError, match='Not in activity context'):
        broker.resolve(reference)


def test_multi_delegation_prevalidates_all_before_minting() -> None:
    broker, references, action, _parent, recheck, delegate, revoke = (
        _delegation_broker([
            None,
            ConnectorDoesNotExistError(_CONNECTOR_ID),
        ]))

    with patch('rest.backends.data_backend.connector_broker.activity.info',
               return_value=SimpleNamespace()):
        with pytest.raises(ConnectorDoesNotExistError):
            broker.delegate_connector_references(
                references,
                child_workflow_id='child',
                child_actions={action},
            )

    assert recheck.await_count == 2
    delegate.assert_not_awaited()
    revoke.assert_not_awaited()


def test_multi_delegation_revokes_earlier_mint_on_late_live_denial() -> None:
    # Two phase-one validations, then one validation immediately before each
    # mint. The second reference loses its binding during phase two.
    broker, references, action, parent, _recheck, delegate, revoke = (
        _delegation_broker(
            [None, None, None,
             IdentityForbiddenError('binding removed')],
            delegated=(object(), 'first-child-token'),
        ))

    with patch('rest.backends.data_backend.connector_broker.activity.info',
               return_value=SimpleNamespace()):
        with pytest.raises(IdentityForbiddenError, match='binding removed'):
            broker.delegate_connector_references(
                references,
                child_workflow_id='child',
                child_actions={action},
            )

    delegate.assert_awaited_once_with(
        parent,
        child_workflow_id='child',
        child_actions={action},
    )
    revoke.assert_awaited_once_with('first-child-token')


@pytest.mark.asyncio
async def test_broker_opens_the_shared_state_db_root() -> None:
    broker = ConnectorCredentialBroker()
    project_id = ProjectID.generate()
    deploy_config = object()
    state_db = object()
    resolved = object()
    resolve_for_activity = AsyncMock(return_value=resolved)
    open_state_db = AsyncMock(return_value=state_db)

    with (patch(
            'rest.backends.data_backend.connector_broker.deploy_env',
            return_value=deploy_config,
    ),
          patch(
              'rest.backends.data_backend.connector_broker.'
              'get_db_storage_root',
              return_value='/shared/db',
          ),
          patch(
              'rest.backends.data_backend.connector_broker.'
              'open_state_db_from_deploy_config',
              open_state_db,
          ),
          patch(
              'rest.backends.data_backend.connector_broker.StateServices',
              return_value=SimpleNamespace(connectors=SimpleNamespace(
                  resolve_for_activity=resolve_for_activity)),
          )):

        async def resolve(services):
            return await services.connectors.resolve_for_activity(
                project_id,
                _CONNECTOR_ID,
            )

        result = await broker._run(resolve)

    assert result is resolved
    open_state_db.assert_awaited_once_with(
        deploy_config,
        state_root='/shared/db',
    )
    assert resolve_for_activity.await_args.args == (project_id, _CONNECTOR_ID)


@pytest.mark.asyncio
async def test_concurrent_cold_broker_opens_shared_state_db_once(
        monkeypatch) -> None:
    state_db = object()
    services = object()
    opens = 0

    async def open_state_db(*_args, **_kwargs):
        nonlocal opens
        opens += 1
        await asyncio.sleep(0.01)
        return state_db

    monkeypatch.setattr(
        'rest.backends.data_backend.connector_broker.'
        'open_state_db_from_deploy_config',
        open_state_db,
    )
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_broker.StateServices',
        lambda opened: services if opened is state_db else None,
    )
    broker = ConnectorCredentialBroker()

    resolved = await asyncio.gather(*(broker._get_services()
                                      for _ in range(16)))

    assert opens == 1
    assert resolved == [services] * 16


@pytest.mark.asyncio
async def test_async_resolve_rechecks_each_access_without_blocking_event_loop(
        activity_context) -> None:
    reference = _reference()

    class _Connectors:
        calls = 0

        async def resolve_for_activity(self, project_id, connector_id):
            assert (project_id, connector_id) == (
                reference.project_id,
                reference.connector_id,
            )
            self.calls += 1
            generation = self.calls
            await asyncio.sleep(0)
            return f'generation-{generation}'

    connectors = _Connectors()
    broker = ConnectorCredentialBroker(timeout_seconds=1)
    broker._services = SimpleNamespace(connectors=connectors)
    broker._authorize_reference = AsyncMock(return_value=object())

    results = await asyncio.gather(*(broker.resolve_async(reference)
                                     for _ in range(8)))

    assert set(results) == {f'generation-{index}' for index in range(1, 9)}
    assert connectors.calls == 8
    assert broker._authorize_reference.await_count == 8


@pytest.mark.asyncio
async def test_async_hung_state_db_open_times_out_and_is_cancelled(
    activity_context,
    monkeypatch,
) -> None:
    started = threading.Event()
    cancelled = threading.Event()
    provider_secret = 'provider-secret-must-not-escape'

    async def hung_open(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(
        'rest.backends.data_backend.connector_broker.'
        'open_state_db_from_deploy_config',
        hung_open,
    )
    broker = ConnectorCredentialBroker(timeout_seconds=0.05)
    resolve_task = asyncio.create_task(broker.resolve_async(_reference()))

    assert await asyncio.to_thread(started.wait, 1)
    # A broker wait must yield to unrelated activity-loop work.
    await asyncio.sleep(0)
    event_loop_progress = True
    with pytest.raises(ConnectorCredentialBrokerTimeout) as caught:
        await resolve_task

    assert event_loop_progress
    assert await asyncio.to_thread(cancelled.wait, 1)
    assert provider_secret not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_async_waiter_cancellation_cancels_broker_operation(
        activity_context) -> None:
    started = threading.Event()
    cancelled = threading.Event()
    broker = ConnectorCredentialBroker(timeout_seconds=1)
    broker._services = SimpleNamespace()

    async def hung(_services):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    task = asyncio.create_task(broker.run_async(hung))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(cancelled.wait, 1)


def test_sync_hung_resolution_has_bounded_wait_and_cancels(
        activity_context) -> None:
    started = threading.Event()
    cancelled = threading.Event()
    broker = ConnectorCredentialBroker(timeout_seconds=0.05)
    broker._services = SimpleNamespace()

    async def hung(_services):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    with pytest.raises(ConnectorCredentialBrokerTimeout) as caught:
        broker.run(hung)

    assert started.is_set()
    assert cancelled.wait(1)
    assert str(caught.value) == 'Connector credential resolution timed out'
    assert caught.value.__cause__ is None


def test_broker_operations_remain_activity_only(monkeypatch) -> None:
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_broker.activity.in_activity',
        lambda: False,
    )
    broker = ConnectorCredentialBroker(timeout_seconds=1)

    async def operation(_services):
        return None

    with pytest.raises(RuntimeError, match='only inside a Temporal activity'):
        broker.run(operation)

    async def invoke_async():
        with pytest.raises(RuntimeError,
                           match='only inside a Temporal activity'):
            await broker.run_async(operation)

    asyncio.run(invoke_async())
