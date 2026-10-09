# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanitization contract for ``_wrap_pickle_safe`` (process-pool activities).

Root cause these guard against (#30109): the generic-exception path of
``process_pool_activity_defn`` built its ``ApplicationError`` from raw
``str(ex)`` plus a full ``traceback.format_exc()`` details payload, and the
dead-cluster path attached the same raw traceback. Pre-built
``ApplicationError``s cross the Temporal boundary verbatim (the activity
interceptor trusts them, see #30048/#30104), so arbitrary subprocess exception
text (cloud-SDK/driver messages carrying request bodies, SQL, or credentials)
flowed into Temporal workflow history, job-status messages, and centralized
logs. These tests pin the fixed contract: exception TYPE names survive for
diagnosis and retryability classification is preserved, but no raw exception
text or traceback crosses the boundary; user-actionable text must come from
``PublicException``, which passes through untouched.

Logs are a separate support channel. Ordinary failures are diagnostic at
``ERROR`` by default; explicitly marked integration-origin chains retain safe
context at ``ERROR`` and expose raw rendering only with enabled ``DEBUG``.
"""
import asyncio
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from kumo_core.public_error import PublicTrainingError
from temporalio import activity
from temporalio.common import Priority
from temporalio.exceptions import ApplicationError
from temporalio.worker._interceptor import ExecuteActivityInput

from common_utils.error import PublicException
from temporalio_commons import worker_interceptors
from temporalio_commons.process_pool_utils import process_pool_activity_defn

_CANARY = 'password=proc-pool-secret-canary-30109'
_ACTIVITY_TYPE = 'proc_pool_sanitization_canary'


class _ClusterPublicError(PublicException):
    non_retryable = False


class _RequestsProviderError(Exception):
    __module__ = 'requests.exceptions'
    _kumo_restricted_log_message = True


class _MarkedRestrictedError(RuntimeError):
    _kumo_restricted_log_message = True
    _kumo_safe_log_message = 'converter failed (exit code 17)'


def _activity_info() -> activity.Info:
    now = datetime.now(timezone.utc)
    return activity.Info(
        activity_id='proc-pool-sanitization-activity',
        activity_type=_ACTIVITY_TYPE,
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
        workflow_id='proc-pool-sanitization-workflow',
        workflow_namespace='namespace',
        workflow_run_id='run-id',
        workflow_type='ProcPoolSanitizationWorkflow',
        priority=Priority(),
        retry_policy=None,
    )


@pytest.fixture
def patched_activity_info(monkeypatch):
    monkeypatch.setattr('temporalio_commons.process_pool_utils.activity.info',
                        _activity_info)


def _rendered(error: ApplicationError) -> str:
    parts = [str(error), repr(error), str(error.message)]
    parts.extend(str(detail) for detail in error.details)
    current = error.__cause__
    while current is not None:
        parts.append(str(current))
        parts.append(repr(current))
        current = current.__cause__
    return ' '.join(parts)


def _interceptor_for(monkeypatch, raiser):
    """Drive ``KumoActivityInboundInterceptor`` around ``raiser``."""
    async def execute_wrapped(_input):
        return raiser()

    next_interceptor = SimpleNamespace(execute_activity=AsyncMock(
        side_effect=execute_wrapped))
    interceptor = worker_interceptors.KumoActivityInboundInterceptor(
        next_interceptor)
    interceptor._outbound = cast(
        worker_interceptors.KumoActivityOutboundInterceptor,
        SimpleNamespace(track_last_heartbeat=False),
    )
    monkeypatch.setattr('temporalio_commons.worker_interceptors.activity.info',
                        _activity_info)
    monkeypatch.setattr(
        'temporalio_commons.worker_interceptors._kumo_record_event',
        lambda *_args, **_kwargs: None,
    )
    return interceptor, ExecuteActivityInput(fn=lambda: None, args=(),
                                             executor=None, headers={})


def test_generic_exception_is_sanitized_but_logged(
        patched_activity_info, caplog) -> None:
    @process_pool_activity_defn
    def handler() -> None:
        raise ValueError(f'provider body {_CANARY}')

    with caplog.at_level('ERROR'):
        with pytest.raises(ApplicationError) as caught:
            handler()

    error = caught.value
    rendered = _rendered(error)
    assert _CANARY not in rendered
    assert 'Traceback (most recent call last)' not in rendered
    # Type name and activity name survive for diagnosis.
    assert error.type == 'ValueError'
    assert 'ValueError' in error.message
    assert _ACTIVITY_TYPE in error.message
    # ValueError is in DEFAULT_NON_RETRYABLE_EX.
    assert error.non_retryable is True
    assert _CANARY in caplog.text


def test_cause_type_survives_but_cause_text_does_not_cross_temporal(
        patched_activity_info, caplog) -> None:
    @process_pool_activity_defn
    def handler() -> None:
        try:
            raise KeyError(f'inner {_CANARY}')
        except KeyError as inner:
            raise RuntimeError(f'outer {_CANARY}') from inner

    with caplog.at_level('ERROR'):
        with pytest.raises(ApplicationError) as caught:
            handler()

    error = caught.value
    rendered = _rendered(error)
    assert _CANARY not in rendered
    assert error.type == 'RuntimeError'
    assert 'KeyError' in error.message
    assert f'outer {_CANARY}' in caplog.text
    assert f'inner {_CANARY}' in caplog.text


def test_retryable_generic_exception_stays_retryable(
        patched_activity_info, caplog) -> None:
    @process_pool_activity_defn
    def handler() -> None:
        raise ConnectionError(f'transient {_CANARY}')

    with caplog.at_level('ERROR'):
        with pytest.raises(ApplicationError) as caught:
            handler()

    error = caught.value
    assert _CANARY not in _rendered(error)
    assert error.type == 'ConnectionError'
    assert error.non_retryable is False
    # Broad builtin families are visible unless an operation marks them.
    assert _CANARY in caplog.text


def test_restricted_process_pool_failure_is_raw_only_at_debug(
        patched_activity_info, caplog) -> None:
    @process_pool_activity_defn
    def handler() -> None:
        raise _RequestsProviderError(f'provider response {_CANARY}')

    with caplog.at_level('ERROR'):
        with pytest.raises(ApplicationError):
            handler()
    assert _CANARY not in caplog.text
    assert '_RequestsProviderError' in caplog.text

    caplog.clear()
    with caplog.at_level('DEBUG'):
        with pytest.raises(ApplicationError):
            handler()
    assert _CANARY in caplog.text


def test_explicit_marker_keeps_safe_context_and_retryability(
        patched_activity_info, caplog) -> None:
    @process_pool_activity_defn
    def handler() -> None:
        raise _MarkedRestrictedError(f'converter stderr {_CANARY}')

    with caplog.at_level('ERROR'):
        with pytest.raises(ApplicationError) as caught:
            handler()

    error = caught.value
    assert error.type == '_MarkedRestrictedError'
    assert error.non_retryable is False
    assert _CANARY not in _rendered(error)
    assert _CANARY not in caplog.text
    assert 'converter failed (exit code 17)' in caplog.text

    caplog.clear()
    with caplog.at_level('DEBUG'):
        with pytest.raises(ApplicationError):
            handler()
    assert _CANARY in caplog.text


def test_fixed_outer_message_does_not_expose_restricted_cause_at_error(
        patched_activity_info, caplog) -> None:
    @process_pool_activity_defn
    def handler() -> None:
        try:
            raise _RequestsProviderError(f'provider response {_CANARY}')
        except _RequestsProviderError as inner:
            raise RuntimeError('fixed Kumo operation failure') from inner

    with caplog.at_level('ERROR'):
        with pytest.raises(ApplicationError):
            handler()
    assert 'fixed Kumo operation failure' in caplog.text
    assert _CANARY not in caplog.text

    caplog.clear()
    with caplog.at_level('DEBUG'):
        with pytest.raises(ApplicationError):
            handler()
    assert _CANARY in caplog.text


def test_dead_cluster_error_keeps_public_message_drops_traceback(
        patched_activity_info, monkeypatch) -> None:
    monkeypatch.setattr(
        'temporalio_commons.error_utils.is_retryable_cluster_error',
        lambda _ex: True)

    @process_pool_activity_defn
    def handler() -> None:
        raise _ClusterPublicError('cluster went away, retrying is safe')

    with pytest.raises(ApplicationError) as caught:
        handler()

    error = caught.value
    # The PublicException message is public-safe by contract and must
    # survive; the raw traceback must not.
    assert 'cluster went away, retrying is safe' in error.message
    assert 'Traceback (most recent call last)' not in _rendered(error)
    assert error.type == '_ClusterPublicError'
    assert error.non_retryable is False


def test_public_exception_passes_through_untouched(
        patched_activity_info) -> None:
    failure = _ClusterPublicError('curated user-facing message')

    @process_pool_activity_defn
    def handler() -> None:
        raise failure

    with pytest.raises(_ClusterPublicError) as caught:
        handler()

    assert caught.value is failure


def test_core_public_error_uses_legacy_public_contract(
        patched_activity_info) -> None:
    @process_pool_activity_defn
    def handler() -> None:
        raise PublicTrainingError('curated training failure')

    with pytest.raises(PublicException) as caught:
        handler()

    assert type(caught.value) is PublicException
    assert caught.value.get_user_message() == 'curated training failure'


@pytest.mark.asyncio
async def test_no_canary_through_activity_interceptor(patched_activity_info,
                                                      monkeypatch) -> None:
    """End-to-end pin through the worker boundary: the sanitized
    ApplicationError built by the process-pool wrapper rides the
    interceptor's trusted pre-built-ApplicationError pass-through (#30104)
    with its sanitized message, type, and retryability intact, and no raw
    exception text leaks past the boundary."""
    @process_pool_activity_defn
    def handler() -> None:
        raise ValueError(f'provider body {_CANARY}')

    interceptor, activity_input = _interceptor_for(monkeypatch, handler)

    with pytest.raises(ApplicationError) as caught:
        await interceptor.execute_activity(activity_input)

    error = caught.value
    assert _CANARY not in _rendered(error)
    # The pass-through must preserve the wrapper's sanitized contract, not
    # rewrap it into a generic internal-failure ApplicationError.
    assert error.type == 'ValueError'
    assert 'ValueError' in error.message
    assert _ACTIVITY_TYPE in error.message
    assert error.non_retryable is True


class _NativeDiskgraphError(Exception):
    """Stand-in for a ``diskgraph._c`` pybind class, which needs the wheel."""


class _UnrenderableError(Exception):
    __module__ = 'diskgraph._c'

    def __str__(self) -> str:
        raise RuntimeError('str() exploded')


class _FalseyError(Exception):
    def __bool__(self) -> bool:
        return False


_NativeDiskgraphError.__module__ = 'diskgraph._c'

_NATIVE_MESSAGE = 'cannot open /mnt/localssd/idx/train_tables/x for read'
_PUBLIC_MESSAGE = 'curated user-facing message'
_CORE_PLATFORM_MESSAGE = 'curated portable training failure'
_SOURCE_CANARY = 'source-line-canary'
_INTERNAL_MESSAGE = 'The activity failed due to an internal error.'
_OOM_MESSAGE = ('Activity failed as the worker process was killed abruptly, '
                'this could be caused by segment fault or insufficient '
                'resource such as OOM')


def _raise_native() -> None:
    def _execute_tt_ingest() -> None:
        raise _NativeDiskgraphError(_NATIVE_MESSAGE)

    _execute_tt_ingest()


def _raise_native_from_a_line_holding_a_secret() -> None:
    def _connect(_dsn: str) -> None:
        raise _NativeDiskgraphError(_NATIVE_MESSAGE)

    _connect('postgres://user:source-line-canary@host/db')


def _raise_native_chaining_a_provider_error() -> None:
    try:
        raise _RequestsProviderError(f'provider body {_CANARY}')
    except _RequestsProviderError as inner:
        raise _NativeDiskgraphError(_NATIVE_MESSAGE) from inner


def _raise_native_suppressing_a_provider_context() -> None:
    try:
        raise ValueError(f'provider body {_CANARY}')
    except ValueError:
        raise _NativeDiskgraphError(_NATIVE_MESSAGE) from None


@pytest.mark.asyncio
@pytest.mark.parametrize('failure, in_log, not_in_log, payload', [
    (_raise_native, ('_execute_tt_ingest', _NATIVE_MESSAGE), (),
     (_INTERNAL_MESSAGE, '_NativeDiskgraphError', False)),
    (_raise_native_from_a_line_holding_a_secret, ('_connect', _NATIVE_MESSAGE),
     (_SOURCE_CANARY, ), (_INTERNAL_MESSAGE, '_NativeDiskgraphError', False)),
    (_raise_native_chaining_a_provider_error,
     ('_RequestsProviderError', _NATIVE_MESSAGE), (_CANARY, ),
     (_INTERNAL_MESSAGE, '_NativeDiskgraphError', False)),
    (_raise_native_suppressing_a_provider_context, (_NATIVE_MESSAGE, ),
     (_CANARY, 'ValueError'),
     (_INTERNAL_MESSAGE, '_NativeDiskgraphError', False)),
    (ValueError(f'ordinary failure {_CANARY}'),
     ('ValueError', 'raise_it', _CANARY), (),
     (_INTERNAL_MESSAGE, 'ValueError', False)),
    (_RequestsProviderError(f'provider body {_CANARY}'),
     ('_RequestsProviderError', 'raise_it'), (_CANARY, ),
     (_INTERNAL_MESSAGE, '_RequestsProviderError', False)),
    (_UnrenderableError(), ('_UnrenderableError', 'diagnostic unavailable'), (),
     (_INTERNAL_MESSAGE, '_UnrenderableError', False)),
    (PublicTrainingError(_CORE_PLATFORM_MESSAGE),
     ('PublicTrainingError', 'raise_it', _CORE_PLATFORM_MESSAGE), (),
     (_CORE_PLATFORM_MESSAGE, 'PublicException', True)),
    (_ClusterPublicError(_PUBLIC_MESSAGE), (_PUBLIC_MESSAGE, 'raise_it'), (),
     (_PUBLIC_MESSAGE, '_ClusterPublicError', False)),
    (ApplicationError(f'provider body {_CANARY}', type='ProviderFailure',
                      non_retryable=False),
     ('ApplicationError', 'raise_it', _CANARY), (),
     (f'provider body {_CANARY}', 'ProviderFailure', False)),
    (BrokenProcessPool(f'provider body {_CANARY}'), ('raise_it', _CANARY), (),
     (_OOM_MESSAGE, 'BrokenProcessPool', False)),
])
async def test_diagnostic_records_call_sites_and_gates_messages(
        monkeypatch, caplog, failure, in_log, not_in_log, payload) -> None:
    def raise_it() -> None:
        if isinstance(failure, BaseException):
            raise failure
        failure()

    interceptor, activity_input = _interceptor_for(monkeypatch, raise_it)

    with caplog.at_level('ERROR'):
        with pytest.raises(ApplicationError) as caught:
            await interceptor.execute_activity(activity_input)

    for text in in_log:
        assert text in caplog.text
    for text in not_in_log:
        assert text not in caplog.text
    assert (caught.value.message, caught.value.type,
            caught.value.non_retryable) == payload
    assert caught.value.__cause__ is None
    assert _NATIVE_MESSAGE not in _rendered(caught.value)


@pytest.mark.asyncio
async def test_restricted_activity_failure_is_raw_only_at_debug(
        monkeypatch, caplog) -> None:
    def raise_provider() -> None:
        raise _RequestsProviderError(f'provider body {_CANARY}')

    interceptor, activity_input = _interceptor_for(monkeypatch, raise_provider)
    with caplog.at_level('ERROR'):
        with pytest.raises(ApplicationError):
            await interceptor.execute_activity(activity_input)
    assert _CANARY not in caplog.text

    caplog.clear()
    interceptor, activity_input = _interceptor_for(monkeypatch, raise_provider)
    with caplog.at_level('DEBUG'):
        with pytest.raises(ApplicationError):
            await interceptor.execute_activity(activity_input)
    assert _CANARY in caplog.text


@pytest.mark.asyncio
async def test_visible_activity_with_restricted_cause_requires_debug_for_cause(
        monkeypatch, caplog) -> None:
    interceptor, activity_input = _interceptor_for(
        monkeypatch, _raise_native_chaining_a_provider_error)
    with caplog.at_level('ERROR'):
        with pytest.raises(ApplicationError):
            await interceptor.execute_activity(activity_input)
    assert _NATIVE_MESSAGE in caplog.text
    assert _CANARY not in caplog.text

    caplog.clear()
    interceptor, activity_input = _interceptor_for(
        monkeypatch, _raise_native_chaining_a_provider_error)
    with caplog.at_level('DEBUG'):
        with pytest.raises(ApplicationError):
            await interceptor.execute_activity(activity_input)
    assert _CANARY in caplog.text


def test_diagnostic_terminates_on_a_cyclic_chain() -> None:
    first = ValueError('first')
    second = ValueError('second')
    first.__cause__ = second
    second.__cause__ = first

    rendered = worker_interceptors._activity_failure_diagnostic(first)

    assert rendered.count('Caused by:') == 1


def test_diagnostic_caps_exception_chain_depth() -> None:
    first = ValueError('first')
    current = first
    for index in range(worker_interceptors._MAX_DIAGNOSTIC_CHAIN_DEPTH + 3):
        following = ValueError(f'level {index}')
        current.__cause__ = following
        current = following

    rendered = worker_interceptors._activity_failure_diagnostic(first)

    assert rendered.count('Caused by:') == (
        worker_interceptors._MAX_DIAGNOSTIC_CHAIN_DEPTH - 1)


def test_diagnostic_follows_a_falsey_explicit_cause() -> None:
    first = _FalseyError('first')
    second = _FalseyError('second')
    first.__cause__ = second

    rendered = worker_interceptors._activity_failure_diagnostic(first)

    assert rendered.count('Caused by:') == 1


def test_diagnostic_failure_is_contained(monkeypatch, caplog) -> None:
    def fail_diagnostic(_error):
        raise RuntimeError('diagnostic exploded')

    monkeypatch.setattr(worker_interceptors, '_activity_failure_diagnostic',
                        fail_diagnostic)
    with caplog.at_level('ERROR'):
        worker_interceptors._log_activity_failure_diagnostic(
            _activity_info(), ValueError('provider detail'))

    assert 'Activity failure diagnostic unavailable' in caplog.text
    assert 'ValueError' in caplog.text


@pytest.mark.parametrize('signal_type', [
    asyncio.CancelledError,
    KeyboardInterrupt,
    SystemExit,
])
def test_log_diagnostic_reraises_cancellation_and_shutdown_signals(
        monkeypatch, signal_type) -> None:
    signal = signal_type()

    def raise_signal(_error):
        raise signal

    monkeypatch.setattr(worker_interceptors, '_activity_failure_diagnostic',
                        raise_signal)

    with pytest.raises(signal_type) as caught:
        worker_interceptors._log_activity_failure_diagnostic(
            _activity_info(), ValueError())

    assert caught.value is signal


def test_diagnostic_keeps_the_raising_frame_past_the_frame_cap() -> None:
    def _raise_here() -> None:
        raise _NativeDiskgraphError(_NATIVE_MESSAGE)

    def _recurse(depth: int) -> None:
        if depth:
            _recurse(depth - 1)
        else:
            _raise_here()

    cap = worker_interceptors._MAX_DIAGNOSTIC_FRAMES
    with pytest.raises(_NativeDiskgraphError) as caught:
        _recurse(cap + 5)

    rendered = worker_interceptors._activity_failure_diagnostic(caught.value)

    assert rendered.count('  File "') == cap
    assert '_raise_here' in rendered and '_recurse' in rendered


def test_diagnostic_bounds_message_and_total_length(monkeypatch) -> None:
    error = _NativeDiskgraphError('m' * 4096)
    monkeypatch.setattr(worker_interceptors, '_MAX_DIAGNOSTIC_MESSAGE_CHARS',
                        16)
    assert 'm' * 17 not in worker_interceptors._activity_failure_diagnostic(
        error)

    monkeypatch.setattr(worker_interceptors, '_MAX_DIAGNOSTIC_CHARS', 30)
    rendered = worker_interceptors._activity_failure_diagnostic(error)

    assert len(rendered) == 30 and rendered.endswith('...')


@pytest.mark.parametrize('limit, expected', [
    (2, 'xx'),
    (3, '...'),
    (5, 'xx...'),
    (12, 'xxxxxxxxx...'),
])
def test_truncate_never_exceeds_its_limit(limit: int, expected: str) -> None:
    assert worker_interceptors._truncate('x' * 20, limit) == expected
