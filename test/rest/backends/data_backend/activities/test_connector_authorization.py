# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from temporalio import activity
from temporalio.api.failure.v1 import Failure
from temporalio.common import Priority
from temporalio.converter import (
    DefaultFailureConverter,
    DefaultPayloadConverter,
)
from temporalio.exceptions import ApplicationError
from temporalio.worker._interceptor import ExecuteActivityInput

from rest.backends.data_backend.activities.connector_authorization import (
    ConnectorDelegationDeniedError,
    delegate_connector_authorizations,
)
from rest.backends.data_backend.principal_authority import (
    PrincipalAuthorityDeniedError,
    PrincipalAuthorityUnavailableError,
)
from rest.config.data_backend import ConnectorExecutionRef
from statedb.config.common import ConnectorID, ProjectID
from statedb.db.connector_authorizations import ConnectorAuthorizationError
from statedb.utils.errors import (
    ConnectorDoesNotExistError,
    IdentityForbiddenError,
    ProjectNotFoundError,
)
from temporalio_commons.auto_rpc_exception import _AutoRpcNonRetriableError
from temporalio_commons.error_utils import default_non_retryable_error_types
from temporalio_commons.worker_interceptors import (
    KumoActivityInboundInterceptor,
)


def _reference() -> ConnectorExecutionRef:
    return ConnectorExecutionRef(
        project_id=ProjectID.generate(),
        connector_id=ConnectorID.generate(),
        authorization_token='parent-token',
    )


def _activity_info() -> activity.Info:
    now = datetime.now(timezone.utc)
    return activity.Info(
        activity_id='delegation-activity',
        activity_type='delegate_connector_authorizations',
        attempt=1,
        current_attempt_scheduled_time=now,
        heartbeat_details=(),
        heartbeat_timeout=None,
        is_local=False,
        namespace='namespace',
        schedule_to_close_timeout=None,
        scheduled_time=now,
        start_to_close_timeout=None,
        started_time=now,
        task_queue='task-queue',
        task_token=b'token',
        workflow_id='parent-workflow',
        workflow_namespace='namespace',
        workflow_run_id='parent-run',
        workflow_type='ParentWorkflow',
        priority=Priority(),
        retry_policy=None,
    )


def _interceptor_for(monkeypatch, failure):
    next_interceptor = SimpleNamespace(
        execute_activity=AsyncMock(side_effect=failure))
    interceptor = KumoActivityInboundInterceptor(next_interceptor)
    interceptor._outbound = SimpleNamespace(track_last_heartbeat=False)
    info = _activity_info()
    monkeypatch.setattr(
        'temporalio_commons.worker_interceptors.activity.info', lambda: info)
    monkeypatch.setattr(
        'temporalio_commons.worker_interceptors._kumo_record_event',
        lambda *_args, **_kwargs: None,
    )
    activity_input = ExecuteActivityInput(
        fn=lambda: None,
        args=(),
        executor=None,
        headers={},
    )
    return interceptor, activity_input


def _rendered_history(error: ApplicationError) -> str:
    failure = Failure()
    DefaultFailureConverter().to_failure(
        error, DefaultPayloadConverter(), failure)
    return f'{error!s} {error!r} {failure!s}'


@pytest.mark.parametrize(
    ('failure', 'permanent'),
    [
        (ConnectorDoesNotExistError(ConnectorID('connector_' + 'a' * 32)),
         True),
        (IdentityForbiddenError('sensitive identity binding'), True),
        (ProjectNotFoundError('sensitive project id'), True),
        (ConnectorAuthorizationError('sensitive capability reason'), True),
        (PrincipalAuthorityDeniedError('sensitive principal reason'), True),
        (RuntimeError('database temporarily unavailable'), False),
        (PrincipalAuthorityUnavailableError(
            'membership temporarily unavailable'), False),
    ],
    ids=(
        'missing-connector',
        'forbidden-identity',
        'missing-project',
        'invalid-capability',
        'denied-principal',
        'database-unavailable',
        'principal-unavailable',
    ),
)
def test_delegation_failure_classification(failure, permanent) -> None:
    with patch(
            'rest.backends.data_backend.activities.connector_authorization.'
            'connector_credential_broker.delegate_connector_references',
            side_effect=failure,
    ):
        expected_type = (ConnectorDelegationDeniedError
                         if permanent else type(failure))
        with pytest.raises(expected_type) as caught:
            delegate_connector_authorizations(
                [_reference()],
                'child-workflow',
                'TableRefreshWorkflow',
            )

    if permanent:
        assert caught.value.non_retryable is True
        assert caught.value.http_code == 403
        rendered = f'{caught.value!s} {caught.value!r}'
        assert 'sensitive' not in rendered
        assert str(failure) not in rendered
        assert 'parent-token' not in rendered
    else:
        assert caught.value is failure


@pytest.mark.asyncio
async def test_activity_interceptor_emits_one_attempt_nonretryable_envelope(
        monkeypatch) -> None:
    denial = ConnectorDelegationDeniedError(
        message='sanitized delegation denial',
        non_retryable=True,
    )
    interceptor, activity_input = _interceptor_for(monkeypatch, denial)

    with pytest.raises(ApplicationError) as caught:
        await interceptor.execute_activity(activity_input)

    assert caught.value.non_retryable is True
    assert caught.value.type == 'ConnectorDelegationDeniedError'
    assert caught.value.message == 'sanitized delegation denial'
    assert caught.value.__cause__ is None
    assert interceptor.next.execute_activity.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('failure_type', 'expected_message'),
    [
        (RuntimeError, 'The activity failed due to an internal error.'),
        (ValueError, 'The activity failed due to an internal error.'),
        (BrokenProcessPool, None),
    ],
    ids=('runtime', 'value', 'broken-process'),
)
async def test_activity_interceptor_sanitizes_unknown_failure_history_and_logs(
        monkeypatch, caplog, failure_type, expected_message) -> None:
    canary = 'provider-body password=activity-secret-canary'
    interceptor, activity_input = _interceptor_for(
        monkeypatch, failure_type(canary))

    with pytest.raises(ApplicationError) as caught:
        await interceptor.execute_activity(activity_input)

    error = caught.value
    if expected_message is not None:
        assert error.message == expected_message
    assert error.type == failure_type.__name__
    assert error.non_retryable is False
    assert error.__cause__ is None
    rendered = _rendered_history(error)
    assert canary not in rendered
    assert 'activity-secret-canary' not in rendered
    assert canary in caplog.text
    if failure_type is ValueError:
        assert error.type in default_non_retryable_error_types


@pytest.mark.asyncio
async def test_activity_interceptor_preserves_safe_autorpc_type_chain(
        monkeypatch, caplog) -> None:
    canary = 'autorpc password=nonretryable-secret-canary'
    interceptor, activity_input = _interceptor_for(
        monkeypatch,
        _AutoRpcNonRetriableError(ValueError, canary),
    )

    with pytest.raises(ApplicationError) as caught:
        await interceptor.execute_activity(activity_input)

    error = caught.value
    assert error.type == '_AutoRpcNonRetriableError'
    assert error.non_retryable is True
    assert isinstance(error.__cause__, ApplicationError)
    safe_cause = error.__cause__
    assert safe_cause.type == 'ValueError'
    assert safe_cause.non_retryable is True
    rendered = _rendered_history(error)
    assert canary not in rendered
    assert 'nonretryable-secret-canary' not in rendered
    assert canary in caplog.text


@pytest.mark.asyncio
async def test_activity_interceptor_preserves_non_exception_base_failure(
        monkeypatch) -> None:
    failure = SystemExit('system-exit-canary')
    interceptor, activity_input = _interceptor_for(monkeypatch, failure)

    with pytest.raises(SystemExit) as caught:
        await interceptor.execute_activity(activity_input)

    assert caught.value is failure
