# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kumo_core.public_error import PublicTrainingError
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.common import Priority
from temporalio.exceptions import ApplicationError
from temporalio.worker._interceptor import ExecuteActivityInput

from async_utils.sync import blocking_func, blocking_method
from common_utils.error import PublicException
from io_utils.errors import mark_exception_restricted
from temporalio_commons import worker_interceptors
from temporalio_commons.auto_rpc import autorpc_handler_wrapper, rpc_handler
from temporalio_commons.auto_rpc_exception import (
    RpcError,
    _AutoRpcNonRetriableError,
    wf_failure_to_rpc_error,
)


class _SafePublicError(PublicException):
    pass


def _trusted_error(kind: str) -> Exception:
    if kind == 'public':
        return _SafePublicError('curated placement rejection')
    return ApplicationError(
        'curated application failure',
        {'placement': 'single-host-limit'},
        type='PlacementRejected',
        non_retryable=True,
    )


def _activity_info() -> activity.Info:
    now = datetime.now(timezone.utc)
    return activity.Info(
        activity_id='autorpc-contract-activity',
        activity_type='autorpc_contract_handler',
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
        workflow_id='autorpc-contract-workflow',
        workflow_namespace='namespace',
        workflow_run_id='run-id',
        workflow_type='AutoRpcContractWorkflow',
        priority=Priority(),
        retry_policy=None,
    )


def _interceptor_for(monkeypatch, failure):
    next_interceptor = SimpleNamespace(execute_activity=AsyncMock(
        side_effect=failure))
    interceptor = worker_interceptors.KumoActivityInboundInterceptor(
        next_interceptor)
    interceptor._outbound = SimpleNamespace(track_last_heartbeat=False)
    monkeypatch.setattr('temporalio_commons.worker_interceptors.activity.info',
                        _activity_info)
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


class MyService:
    def __init__(self) -> None:
        self._info = 'myservice'

    @rpc_handler
    def foo(self, param: str) -> str:  # pylint: disable=disallowed-name
        return f'{self._info}-{param}'

    @rpc_handler
    def foo_multiparams(self, p1: int, p2: str, *args) -> str:
        return ','.join([str(p1), p2] + list(args))

    @rpc_handler
    async def foo_async(self, param: str) -> str:
        return f'{self._info}-{param}'

    @rpc_handler
    async def foo_async_multiparams(self, p1: int, p2: str, *args) -> str:
        return ','.join([str(p1), p2] + list(args))


@pytest.mark.parametrize('kind', ('public', 'application'))
def test_sync_autorpc_handler_preserves_trusted_error(kind: str) -> None:
    failure = _trusted_error(kind)

    @activity.defn(name=f'test_sync_autorpc_{kind}_error')
    def handler() -> None:
        raise failure

    wrapped = autorpc_handler_wrapper(handler)

    with pytest.raises(type(failure)) as caught:
        wrapped()

    assert caught.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ('public', 'application'))
async def test_async_autorpc_handler_preserves_trusted_error(kind: str) -> None:
    failure = _trusted_error(kind)

    @activity.defn(name=f'test_async_autorpc_{kind}_error')
    async def handler() -> None:
        raise failure

    wrapped = autorpc_handler_wrapper(handler)

    with pytest.raises(type(failure)) as caught:
        await wrapped()

    assert caught.value is failure


def test_autorpc_retry_output_withholds_only_marked_detail(
        caplog, capsys) -> None:
    canary = 'provider response password=autorpc-retry-secret'
    failure = RuntimeError(canary)
    mark_exception_restricted(
        failure,
        safe_message='Kubernetes API operation failed (status=503)',
    )

    @activity.defn(name='test_sync_autorpc_restricted_retry')
    def handler() -> None:
        raise failure

    wrapped = autorpc_handler_wrapper(handler)

    with caplog.at_level('DEBUG', logger='temporalio_commons.auto_rpc.server'):
        with pytest.raises(RuntimeError) as caught:
            wrapped()

    assert caught.value is failure
    stdout = capsys.readouterr().out
    assert canary not in stdout
    assert 'Kubernetes API operation failed (status=503)' in stdout
    assert canary in caplog.text


@pytest.mark.asyncio
async def test_autorpc_public_error_reaches_public_activity_contract(
        monkeypatch) -> None:
    failure = _SafePublicError('curated placement rejection')

    @activity.defn(name='test_autorpc_public_activity_contract')
    async def handler() -> None:
        raise failure

    wrapped = autorpc_handler_wrapper(handler)

    async def execute_wrapped(_input):
        return await wrapped()

    interceptor, activity_input = _interceptor_for(monkeypatch, execute_wrapped)

    with pytest.raises(ApplicationError) as caught:
        await interceptor.execute_activity(activity_input)

    error = caught.value
    assert error.message == failure.message
    assert error.type == '_SafePublicError'
    assert tuple(error.details) == (failure.to_dict(), )
    assert error.non_retryable is True


@pytest.mark.asyncio
async def test_core_public_error_reaches_legacy_activity_contract(
        monkeypatch) -> None:
    failure = PublicTrainingError('curated training rejection')
    interceptor, activity_input = _interceptor_for(monkeypatch, failure)

    with pytest.raises(ApplicationError) as caught:
        await interceptor.execute_activity(activity_input)

    error = caught.value
    assert error.message == failure.message
    assert error.type == PublicException.__name__
    assert PublicException.from_dict(
        error.details[0]).get_user_message() == (failure.get_user_message())
    assert error.non_retryable is True


@pytest.mark.asyncio
async def test_activity_interceptor_preserves_prebuilt_application_error(
        monkeypatch) -> None:
    cause = RuntimeError('controller failure')
    failure = ApplicationError(
        'curated application failure',
        {'placement': 'single-host-limit'},
        type='PlacementRejected',
        non_retryable=True,
    )
    failure.__cause__ = cause
    interceptor, activity_input = _interceptor_for(monkeypatch, failure)

    with pytest.raises(ApplicationError) as caught:
        await interceptor.execute_activity(activity_input)

    assert caught.value is failure
    assert caught.value.message == 'curated application failure'
    assert caught.value.type == 'PlacementRejected'
    assert tuple(caught.value.details) == ({'placement': 'single-host-limit'}, )
    assert caught.value.non_retryable is True
    assert caught.value.__cause__ is cause


@pytest.mark.asyncio
async def test_autorpc_unknown_error_history_is_sanitized_and_logs_message(
        monkeypatch, caplog) -> None:
    canary = 'provider-body password=autorpc-secret-canary'

    @activity.defn(name='test_autorpc_unknown_error_sanitized')
    async def handler() -> None:
        raise ValueError(canary)

    wrapped = autorpc_handler_wrapper(handler)

    async def execute_wrapped(_input):
        return await wrapped()

    interceptor, activity_input = _interceptor_for(monkeypatch, execute_wrapped)

    with pytest.raises(ApplicationError) as caught:
        await interceptor.execute_activity(activity_input)

    error = caught.value
    assert error.message == 'The activity failed due to an internal error.'
    assert error.type == _AutoRpcNonRetriableError.__name__
    assert error.non_retryable is True
    assert isinstance(error.__cause__, ApplicationError)
    assert error.__cause__.type == 'ValueError'
    rendered = f'{error!s} {error!r} {error.__cause__!s}'
    assert canary not in rendered
    assert 'autorpc-secret-canary' not in rendered
    assert canary in caplog.text


@pytest.mark.parametrize('non_retryable', (False, True))
def test_wf_failure_preserves_direct_application_retryability(
        non_retryable: bool) -> None:
    failure = ApplicationError(
        'curated application failure',
        type='PlacementRejected',
        non_retryable=non_retryable,
    )

    with pytest.raises(RpcError) as caught:
        wf_failure_to_rpc_error(
            'PoolService',
            'place',
            WorkflowFailureError(cause=failure),
        )

    assert caught.value.error_type == 'PlacementRejected'
    assert caught.value.message == 'curated application failure'
    assert caught.value.non_retryable is non_retryable


def test_wf_failure_keeps_legacy_autorpc_envelope_contract() -> None:
    original = ApplicationError(
        'legacy failure',
        type='ValueError',
        non_retryable=True,
    )
    envelope = ApplicationError(
        'sanitized envelope',
        type=_AutoRpcNonRetriableError.__name__,
        non_retryable=True,
    )
    envelope.__cause__ = original

    with pytest.raises(RpcError) as caught:
        wf_failure_to_rpc_error(
            'PoolService',
            'place',
            WorkflowFailureError(cause=envelope),
        )

    assert caught.value.error_type == 'ValueError'
    assert caught.value.message == 'legacy failure'
    assert caught.value.non_retryable is True


def test_instance_method_sync():
    myservice = MyService()
    assert myservice.foo('test') == 'myservice-test'
    assert myservice.foo.sync('test') == 'myservice-test'
    assert myservice.foo_async.sync('test') == 'myservice-test'

    assert myservice.foo_multiparams(1, '2', '3', '4') == '1,2,3,4'
    assert myservice.foo_multiparams.sync(1, '2', '3,4') == '1,2,3,4'
    assert myservice.foo_async_multiparams.sync(1, '2', '3,4') == '1,2,3,4'


@pytest.mark.asyncio
async def test_instance_method_async():
    myservice = MyService()
    assert await myservice.foo.coroutine('test') == 'myservice-test'
    assert await myservice.foo_async('test') == 'myservice-test'
    assert await myservice.foo_async.coroutine('test') == 'myservice-test'

    ret = await myservice.foo_multiparams.coroutine(1, '2', '3', '4')
    assert ret == '1,2,3,4'
    ret = await myservice.foo_async_multiparams(1, '2', '3,4')
    assert ret == '1,2,3,4'
    ret = await myservice.foo_async_multiparams.coroutine(1, '2', '3,4')
    assert ret == '1,2,3,4'


@blocking_func()
def sync_func(p1: int, p2: int) -> int:
    return p1 + p2


class SyncClass:
    def __init__(self, value: int) -> None:
        self._value = value

    @blocking_method()
    def sync_method(self, p1: int, p2: int) -> int:
        return p1 + p2 + self._value


@pytest.mark.asyncio
async def test_blocking_call_async():
    assert await sync_func.coroutine(1, 2) == 3

    with pytest.raises(Exception,
                       match='Not allowed to call a blocking function'):
        sync_func(1, 2)

    with pytest.raises(Exception,
                       match='Not allowed to call a blocking function'):
        sync_func.sync(1, 2)

    instance = SyncClass(3)
    assert await instance.sync_method.coroutine(1, 2) == 6

    with pytest.raises(Exception,
                       match='Not allowed to call a blocking function'):
        instance.sync_method(1, 2)

    with pytest.raises(Exception,
                       match='Not allowed to call a blocking function'):
        instance.sync_method.sync(1, 2)


def test_blocking_call():
    assert sync_func(1, 2) == 3
    assert sync_func.sync(1, 2) == 3
    instance = SyncClass(3)
    assert instance.sync_method(1, 2) == 6
    assert instance.sync_method.sync(1, 2) == 6
