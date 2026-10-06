# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import json
import threading
import time
import traceback
from concurrent.futures.process import BrokenProcessPool
from contextvars import copy_context
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from logging import DEBUG, WARNING
from pprint import pformat
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union

from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError
from temporalio.common import RetryPolicy, TypedSearchAttributes
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    CancelledError,
    ChildWorkflowError,
    TimeoutError,
    WorkflowAlreadyStartedError,
)
from temporalio.worker import (
    ActivityInboundInterceptor,
    ActivityOutboundInterceptor,
    ExecuteActivityInput,
    ExecuteWorkflowInput,
    Interceptor,
    StartActivityInput,
    StartChildWorkflowInput,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
    WorkflowOutboundInterceptor,
)
from temporalio.workflow import ActivityHandle

from common_utils.error import PublicException
from common_utils.error_diagnostics import (
    exception_chain_contains_restricted_detail,
    exception_message_should_be_logged_at_error,
    safe_exception_message_for_log,
)
from common_utils.kumo_timing import record_event as _kumo_record_event
from io_utils import logging
from kumo.platform_errors import (
    CORE_PLATFORM_ERROR_TYPES,
    as_platform_public_exception,
)
from service_framework.deploy_stage import _DEPLOY_STAGE, DeployStage
from temporalio_commons.auto_rpc_exception import _AutoRpcNonRetriableError
from temporalio_commons.dataclass_converter import KumoPayloadConverter
from temporalio_commons.heartbeat_utils import (
    DEFAULT_HEARTBEAT_INTERVAL,
    auto_heartbeater,
)
from temporalio_commons.temporal_client import ID_SUFFIX_MEMO_FIELD

from .error_utils import (
    default_non_retryable_error_types,
    is_cancelled_exception,
)

log = logging.getLogger(__name__)

_MAX_SIZE_ACTIVITY_RESULT_TO_LOG = 4 * 1024  # 4 KB
_DEFAULT_HEARTBEAT_TIMEOUT = timedelta(minutes=5)
# This number is decided by the data expiration time. And we don't timeout
# customer long running workflow if customer don't cancel it.
_DEFAULT_ACTIVITY_SCHEDULE_TO_CLOSE_TIMEOUT = timedelta(days=21)
_INTERNAL_ACTIVITY_FAILURE_MESSAGE = (
    'The activity failed due to an internal error.')

_WITHHELD_DETAIL = '<message withheld: may carry provider text or credentials>'

_MAX_DIAGNOSTIC_CHAIN_DEPTH = 10
_MAX_DIAGNOSTIC_FRAMES = 20
_MAX_DIAGNOSTIC_MESSAGE_CHARS = 2048
_MAX_DIAGNOSTIC_CHARS = 16384


def _truncate(text: str, limit: int) -> str:
    """Clip ``text`` to at most ``limit`` characters, marking any clip."""
    if len(text) <= limit:
        return text
    return text[:limit] if limit < 3 else text[:limit - 3] + '...'


def _message_is_safe_to_log(ex: BaseException) -> bool:
    """Whether ``ex``'s message may be written to this process's log.

    Preserve #30332's public/DiskGraph visibility, expose ordinary failures by
    default, and withhold explicitly marked integration details.
    """
    return exception_message_should_be_logged_at_error(ex)


def _activity_failure_diagnostic(ex: BaseException) -> str:
    """Render a failing activity's call sites, plus each message that is safe.

    Each chain level contributes its innermost frames as filename, line
    number and function name, never source text. Explicit causes and
    unsuppressed contexts are followed. Output is bounded, and ordinary
    rendering failures degrade to a type-only record without swallowing
    cancellation or shutdown signals.
    """
    sections: list[str] = []
    try:
        seen: set[int] = set()
        current: Optional[BaseException] = ex
        while (current is not None and id(current) not in seen
               and len(sections) < _MAX_DIAGNOSTIC_CHAIN_DEPTH):
            seen.add(id(current))
            walked = list(traceback.walk_tb(current.__traceback__))
            lines = [
                f'  File "{frame.f_code.co_filename}", line {lineno}, in '
                f'{frame.f_code.co_name}'
                for frame, lineno in walked[-_MAX_DIAGNOSTIC_FRAMES:]
            ]
            if _message_is_safe_to_log(current):
                detail = _truncate(str(current), _MAX_DIAGNOSTIC_MESSAGE_CHARS)
            else:
                detail = _truncate(
                    safe_exception_message_for_log(current)
                    or _WITHHELD_DETAIL,
                    _MAX_DIAGNOSTIC_MESSAGE_CHARS,
                )
            lines.append(f'{type(current).__name__}: {detail}')
            sections.append('\n'.join(lines))
            cause = current.__cause__
            if cause is not None:
                current = cause
            else:
                current = (None if current.__suppress_context__ else
                           current.__context__)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # pylint: disable=broad-except
        sections = [f'{type(ex).__name__}: <diagnostic unavailable>']
    return _truncate('\n\nCaused by:\n\n'.join(sections), _MAX_DIAGNOSTIC_CHARS)


def _log_activity_failure_diagnostic(info: activity.Info,
                                     ex: BaseException) -> None:
    """Contain ordinary logging failures without replacing the activity
    failure.
    """
    try:
        diagnostic = _activity_failure_diagnostic(ex)
        log.error(
            'Activity %s(id=%s, attempt=%d) in %s failed! '
            'Exception (type=%s)\n%s', info.activity_type, info.activity_id,
            info.attempt, info.workflow_id,
            type(ex).__name__, diagnostic)
        if exception_chain_contains_restricted_detail(ex):
            log.debug(
                'Restricted raw activity failure for explicit diagnostic '
                'logging',
                exc_info=(type(ex), ex, ex.__traceback__),
            )
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # pylint: disable=broad-except
        try:
            log.error('Activity failure diagnostic unavailable (type=%s)',
                      type(ex).__name__)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # pylint: disable=broad-except
            pass


# These children receive large job envelopes that may contain connector
# references, but never perform credential-bearing work. Strip parent tokens
# instead of granting authority or allowing an unknown workflow to inherit a
# parent-bound token it cannot legitimately consume.
_CONNECTOR_AUTHORITY_FREE_CHILD_WORKFLOWS = frozenset({
    'WorkerInfoWorkflow',
})

DEFAULT_RETRY_POLICY = RetryPolicy()

# Whitelist (exceptions) of activities that have to be executed with retry
# disabled. Please document the reason for each and every exception.
RETRY_DISABLED_ACTIVITY_WHITELIST = (
    # Writing artifacts are not idempotent for certain scenarios of exporting
    # to external data warehouse such as APPEND.
    'write_artifacts_to_customer',

    # Test-only activities:
    'hc_sync_activity',
    'hc_async_activity',
    'test_get_session')

# Global dictionary to help monitor liveness of executor for synchronous
# activities.
#
# Keyed by task_queue, last timestamp of any sync activity heartbeat
last_sync_activity_heartbeat: Dict[str, datetime] = {}
#
# Keyed by task_queue, last timestamp of any sync activity completion
last_sync_activity_completion: Dict[str, datetime] = {}


class KumoActivityOutboundInterceptor(ActivityOutboundInterceptor):
    def __init__(self, next: ActivityOutboundInterceptor) -> None:
        """Create the outbound interceptor.

        Args:
            next: The next interceptor in the chain. The default implementation
                of all calls is to delegate to the next interceptor.
        """
        self.next = next
        self.track_last_heartbeat: bool = False

    def heartbeat(self, *details: Any) -> None:
        """Called for every :py:func:`temporalio.activity.heartbeat` call."""
        if self.track_last_heartbeat:
            last_sync_activity_heartbeat[
                activity.info().task_queue] = datetime.now(timezone.utc)

        self.next.heartbeat(*details)


# pylint: disable=too-few-public-methods
class KumoActivityInboundInterceptor(ActivityInboundInterceptor):
    def __init__(self, next: ActivityInboundInterceptor) -> None:
        super().__init__(next)
        self._converter = KumoPayloadConverter()

    def init(self, outbound: ActivityOutboundInterceptor) -> None:
        """Initialize with an outbound interceptor.

        To add a custom outbound interceptor, wrap the given interceptor before
        sending to the next ``init`` call.
        """
        self._outbound = KumoActivityOutboundInterceptor(outbound)
        self.next.init(self._outbound)

    async def execute_activity(self, input: ExecuteActivityInput) -> Any:
        """Called to invoke the activity."""
        is_sync_activity = bool(input.executor is not None
                                and not inspect.iscoroutinefunction(input.fn))
        self._outbound.track_last_heartbeat = is_sync_activity

        info = activity.info()
        startActivitylogLevel = WARNING if info.attempt > 1 else DEBUG
        if log.isEnabledFor(startActivitylogLevel):
            log.log(startActivitylogLevel, 'Executing activity:\n%s',
                    pformat(asdict(info), indent=2))

        heartbeat_required = (info.heartbeat_timeout is not None
                              and not info.is_local)
        if heartbeat_required:
            # Automatically send heartbeat when heartbeat is required
            log.info('Enable auto-heartbeating for %s.%s', info.activity_type,
                     info.activity_id)
            input.fn = auto_heartbeater(input.fn)

        # For process-pool (sync) activities, also heartbeat from the main
        # process throughout execution. The subprocess auto_heartbeater runs
        # in a background thread but GIL contention from blocking C-extension
        # I/O (e.g. Databricks gRPC) can starve
        # that thread for the full duration of the call — confirmed in
        # production as the root cause of materialize_prediction_task timeouts
        # on large datasets. The main process event loop is unaffected by
        # subprocess GIL pressure, so a Thread here keeps Temporal alive while
        # the subprocess is blocked.
        # Thread-based (not async) to leave the SDK's SharedHeartbeatSender
        # IPC bridge intact; execute_activity is never wrapped or cancelled.
        hb_done: Optional[threading.Event] = None
        if is_sync_activity and heartbeat_required:
            _hb_done = threading.Event()
            hb_done = _hb_done
            _hb_ctx = copy_context()
            _hb_interval = DEFAULT_HEARTBEAT_INTERVAL.total_seconds()
            _hb_activity_type = info.activity_type
            _hb_activity_id = info.activity_id
            # Stop backup heartbeating before start_to_close_timeout so that a
            # truly deadlocked subprocess still times out on Temporal's side
            # rather than being kept alive indefinitely.
            _sto = info.start_to_close_timeout or timedelta(hours=1)
            _max_backup_secs = _sto.total_seconds() * 0.8

            def _main_process_heartbeat():
                started_at = time.monotonic()
                while not _hb_done.wait(timeout=_hb_interval):
                    elapsed = time.monotonic() - started_at
                    if elapsed > _max_backup_secs:
                        log.error(
                            'Main-process heartbeat for %s has run for %.0fs '
                            '(> %.0fs threshold). Subprocess may be deadlocked;'
                            ' stopping backup heartbeat so Temporal can time '
                            'out the activity.',
                            _hb_activity_type,
                            elapsed,
                            _max_backup_secs,
                        )
                        break
                    try:
                        _hb_ctx.run(
                            activity.heartbeat,
                            f'backup-heartbeat;subprocess_silent_for={elapsed:.0f}s',
                        )
                        log.debug(
                            'Main-process heartbeat sent for %s '
                            '(subprocess_silent_for=%.0fs)',
                            _hb_activity_type,
                            elapsed,
                        )
                    except BaseException as e:  # pylint: disable=broad-except
                        log.warning('Main-process heartbeat failed for %s: %s',
                                    _hb_activity_type, e)
                        break

            threading.Thread(
                target=_main_process_heartbeat,
                daemon=True,
                name=f'main-hb-{_hb_activity_id}',
            ).start()

        start_time = datetime.now(timezone.utc)
        phase1b_activity_error_type: Optional[str] = None
        try:
            result = await self.next.execute_activity(input)
        except PublicException as exception:
            phase1b_activity_error_type = exception.__class__.__name__
            _log_activity_failure_diagnostic(info, exception)
            raise ApplicationError(
                exception.message,
                exception.to_dict(),
                type=exception.__class__.__name__,
                non_retryable=exception.non_retryable,
            ) from None
        except CORE_PLATFORM_ERROR_TYPES as core_exception:
            phase1b_activity_error_type = core_exception.__class__.__name__
            platform_exception = as_platform_public_exception(core_exception)
            assert platform_exception is not None
            _log_activity_failure_diagnostic(info, core_exception)
            raise ApplicationError(
                platform_exception.message,
                platform_exception.to_dict(),
                type=platform_exception.__class__.__name__,
                non_retryable=platform_exception.non_retryable,
            ) from None
        except ApplicationError as exception:
            phase1b_activity_error_type = exception.__class__.__name__
            _log_activity_failure_diagnostic(info, exception)
            raise
        except BaseException as ex:
            phase1b_activity_error_type = ex.__class__.__name__
            if isinstance(ex, (asyncio.CancelledError, CancelledError)):
                log.warning(
                    'Activity %s(id=%s, attempt=%d) in %s is cancelled.',
                    info.activity_type, info.activity_id, info.attempt,
                    info.workflow_id)
            else:
                # Cloud SDK and database driver messages can carry
                # request bodies, SQL, or credentials, so the record keeps
                # bounded call sites but gates every message.
                _log_activity_failure_diagnostic(info, ex)

            if isinstance(ex, BrokenProcessPool):
                # For broken process pool exception, we will wrap the
                # exception as ApplicationError so that the worker would remain
                # alive and HealingProcessPool would re-create a new process
                # pool upon the next activity task.
                raise ApplicationError(
                    ('Activity failed as the worker process was killed '
                     'abruptly, this could be caused by segment fault '
                     'or insufficient resource such as OOM'),
                    type=BrokenProcessPool.__name__,
                    non_retryable=False,
                ) from None
            if isinstance(ex, _AutoRpcNonRetriableError):
                # AutoRPC uses a two-level failure chain to retain the
                # original exception type while marking the call terminal.
                # Rebuild that chain from fixed text so its client contract is
                # preserved without serializing the original message/cause.
                original_error_type = (ex.args[0] if ex.args else Exception)
                if (not isinstance(original_error_type, type)
                        or not issubclass(original_error_type, BaseException)):
                    original_error_type = Exception
                safe_cause = ApplicationError(
                    _INTERNAL_ACTIVITY_FAILURE_MESSAGE,
                    type=original_error_type.__name__,
                    non_retryable=True,
                )
                raise ApplicationError(
                    _INTERNAL_ACTIVITY_FAILURE_MESSAGE,
                    type=_AutoRpcNonRetriableError.__name__,
                    non_retryable=True,
                ) from safe_cause
            if isinstance(ex, (asyncio.CancelledError, CancelledError)):
                raise
            if not isinstance(ex, Exception):
                raise
            raise ApplicationError(
                _INTERNAL_ACTIVITY_FAILURE_MESSAGE,
                type=type(ex).__name__,
                non_retryable=False,
            ) from None
        finally:
            # Stop the main-process heartbeat thread if one was started.
            if hb_done is not None:
                hb_done.set()
            # Keep track of most recent completion time of any activity on
            # this task queue.
            if is_sync_activity:
                last_sync_activity_completion[info.task_queue] = datetime.now(
                    timezone.utc)
            phase1b_finished_at = datetime.now(timezone.utc)
            _kumo_record_event(
                'temporal.activity.done',
                workflow_type=info.workflow_type,
                activity_type=info.activity_type,
                task_queue=info.task_queue,
                attempt=info.attempt,
                is_local=info.is_local,
                is_sync_activity=is_sync_activity,
                schedule_delay_seconds=(start_time -
                                        info.scheduled_time).total_seconds(),
                execution_seconds=(phase1b_finished_at -
                                   start_time).total_seconds(),
                total_since_scheduled_seconds=(
                    phase1b_finished_at - info.scheduled_time).total_seconds(),
                error_type=phase1b_activity_error_type,
            )

        if not log.isEnabledFor(DEBUG):
            # Showing detailed debug info is expensive, skip if logging is not
            # enabled for DEBUG.
            return result

        activity_summary = _activity_summary(info, start_time)
        payload = self._converter.to_payloads([result])[0]
        if (payload.data
                and len(payload.data) > _MAX_SIZE_ACTIVITY_RESULT_TO_LOG):
            activity_summary['result'] = f'...({len(payload.data)} bytes)'
        elif (payload.data
              and payload.metadata['encoding'].decode().startswith('json')
              and not payload.metadata['class'].decode().startswith('str_')
              and 'binary' not in payload.metadata):
            activity_summary['result'] = json.loads(payload.data.decode())
        else:
            activity_summary['result'] = str(result)
        log.debug('Activity completed, summary:\n%s', pformat(activity_summary))
        return result


# pylint: disable=too-few-public-methods
class KumoActivityInterceptor(Interceptor):
    def intercept_activity(
            self,
            next: ActivityInboundInterceptor) -> ActivityInboundInterceptor:
        return KumoActivityInboundInterceptor(next)


def _activity_summary(info: activity.Info,
                      start_time: datetime) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    exec_sec = (now - start_time).total_seconds()
    total_sec = (now - info.scheduled_time).total_seconds()
    return {
        'type': f'{info.workflow_type}.{info.activity_type}',
        'workflow_id': {info.workflow_id},
        'task-queue': {info.task_queue},
        'timing (seconds)': {
            'function execution': f'{exec_sec:.4f}',
            'total since scheduled': f'{total_sec:.4f}',
        }
    }


WFError = Union[WorkflowFailureError, ChildWorkflowError, ActivityError]

ReturnType = TypeVar('ReturnType')


class _ActivityHandleWithFailureTracking(Generic[ReturnType]):
    """Wraps ActivityHandle to capture failures for timeout error enrichment.

    When an activity is killed mid-execution by start_to_close timeout (before
    it can raise an error), Temporal reports only TimeoutError without the last
    failure from previous retry attempts. This wrapper tracks activity failures
    and enriches TimeoutError with prior failure context via __cause__.

    Example scenario:
    1. Activity attempt 1: Fails with ValueError("Database connection failed")
    2. Activity attempt 2: Is killed mid-execution by start_to_close timeout
    3. Without this wrapper: Error shows only TimeoutError
    4. With this wrapper: TimeoutError.__cause__ contains the ValueError
    """
    def __init__(self, handle: ActivityHandle[ReturnType], activity_name: str,
                 interceptor: 'KumoWorkflowOutboundInterceptor') -> None:
        self._handle = handle
        self._activity_name = activity_name
        self._interceptor = interceptor

    @staticmethod
    def _find_innermost_timeout_in_chain(
            exc: BaseException) -> Optional[TimeoutError]:
        """Find the innermost TimeoutError in the exception chain.

        Walks the chain following .cause and __cause__. Returns the last/deepest
        TimeoutError found (the one without a TimeoutError cause), so we can
        attach the prior failure to it without overwriting existing chain links.
        """
        innermost_timeout: Optional[TimeoutError] = None
        current = exc
        seen: set[int] = set()
        while id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, TimeoutError):
                innermost_timeout = current
            next_cause = getattr(current, 'cause', None) or getattr(
                current, '__cause__', None)
            if next_cause is None:
                break
            current = next_cause
        return innermost_timeout

    def __await__(self):
        # Make the wrapper awaitable - wrap the inner __await__ to catch errors
        return self._awaitable_wrapper().__await__()

    async def _awaitable_wrapper(self) -> ReturnType:
        """Wrapper coroutine that catches exceptions and tracks failures."""
        try:
            result = await self._handle
            # Clear stored failure on success - activity recovered
            self._interceptor._last_activity_failures.pop(
                self._activity_name, None)
            return result
        except ActivityError as e:
            innermost_timeout = self._find_innermost_timeout_in_chain(e)
            if innermost_timeout is not None:
                last_failure = self._interceptor._last_activity_failures.get(
                    self._activity_name)
                if last_failure:
                    # Set __cause__ on the innermost TimeoutError to chain the prior failure
                    if innermost_timeout.__cause__ is None:
                        innermost_timeout.__cause__ = last_failure
                    log.info(
                        'Enriched timeout error with prior failure for activity %s',
                        self._activity_name)
            else:
                self._interceptor._last_activity_failures[
                    self._activity_name] = e
            raise
        except BaseException as e:
            self._interceptor._last_activity_failures[self._activity_name] = e
            raise

    def __getattr__(self, name: str) -> Any:
        # Delegate all other attributes to the wrapped handle
        return getattr(self._handle, name)


class KumoWorkflowOutboundInterceptor(WorkflowOutboundInterceptor):
    """Kumo's in-house workflow outbound interceptor that intercepts:
    1) workflow.start_child_workflow() call: automatically propagate custom
    search attributes, workflow id and task queue suffix, and memo (custom
    metadata) field, from parent workflow to child workflow.

    2) workflow.start_activity() call: enforce ALL long-running activities to
    be configured with heartbeat timeout.

    3) Activity failure tracking: tracks the last failure per activity type
    to enrich TimeoutError with prior failure context when an activity is
    killed mid-execution by start_to_close timeout.

    """
    def __init__(self, next: WorkflowOutboundInterceptor) -> None:
        super().__init__(next)
        # Track last activity failure per activity type for timeout enrichment
        self._last_activity_failures: Dict[str, BaseException] = {}

    def _get_id_suffix(self) -> Optional[str]:
        parent_wf_info = self.info()
        id_suffix_payload = parent_wf_info.raw_memo.get(ID_SUFFIX_MEMO_FIELD)
        if id_suffix_payload:
            id_suffix = id_suffix_payload.data.decode()
            assert parent_wf_info.workflow_id.endswith(id_suffix)
            assert parent_wf_info.task_queue.endswith(id_suffix)
            return id_suffix
        return None

    def _update_activity_retry_policy(self, input: StartActivityInput) -> None:
        is_autorpc = self.info().workflow_type.startswith('RpcWorkflow_')
        if is_autorpc:
            return  # AutoRPC disables activity-level retry by design.

        retry_policy = input.retry_policy
        if retry_policy is None:
            # use Default values.
            retry_policy = RetryPolicy()

        if (retry_policy.maximum_attempts == 1
                and input.activity not in RETRY_DISABLED_ACTIVITY_WHITELIST
                and self.info().workflow_type
                not in ('RunSingleActivityWorkflow', 'DummyTestWorkflow',
                        'TestHeartbeatWorkflow')):
            log.error(f'Activity {input.activity} must enable retry!')
            if _DEPLOY_STAGE.parse_value() != DeployStage.PROD:
                raise ApplicationError(
                    f'Activity {input.activity} must enable retry!',
                    type=AssertionError.__name__, non_retryable=True)

        non_retryable_error_types = list(default_non_retryable_error_types)
        if retry_policy.non_retryable_error_types:
            non_retryable_error_types.extend(
                retry_policy.non_retryable_error_types)
        retry_policy.non_retryable_error_types = list(
            set(non_retryable_error_types))

        # set updated retry policy
        input.retry_policy = retry_policy

    async def start_child_workflow(
            self,
            input: StartChildWorkflowInput) -> workflow.ChildWorkflowHandle:
        # Child workflows inherit custom search attributes from parent
        parent_wf_info = self.info()
        parent_search_attrs = parent_wf_info.search_attributes
        if parent_search_attrs:
            child_search_attrs = dict(parent_search_attrs)
            if input.search_attributes:
                assert not isinstance(input.search_attributes,
                                      TypedSearchAttributes)
                child_search_attrs.update(input.search_attributes)
            input.search_attributes = child_search_attrs

        # Child workflows inherit id suffix from parent
        id_suffix = self._get_id_suffix()
        if id_suffix:
            # Append suffix to task_queue and id of the child workflow
            if (input.task_queue and not input.task_queue.endswith(id_suffix)):
                input.task_queue += id_suffix
            if not input.id.endswith(id_suffix):
                input.id += id_suffix

            # Make sure the child also carries id_suffix in the memo
            memo = {ID_SUFFIX_MEMO_FIELD: id_suffix.encode()}
            if input.memo:
                memo.update(input.memo)
            input.memo = memo

        # Try to propagate workflow task timeout setting from parent.
        if not input.task_timeout:
            input.task_timeout = parent_wf_info.task_timeout
        from rest.backends.data_backend.connector_authorization_client import \
            connector_execution_references  # pants: no-infer-dep
        from rest.backends.data_backend.connector_authorization_client import \
            replace_connector_authorization_tokens  # pants: no-infer-dep
        references = connector_execution_references(input.args)
        delegated_tokens: list[str] = []
        if references:
            if input.workflow in _CONNECTOR_AUTHORITY_FREE_CHILD_WORKFLOWS:
                replacements = {
                    reference.authorization_token: ''
                    for reference in references
                }
            else:
                replacements = await workflow.execute_activity(
                    'delegate_connector_authorizations',
                    args=[references, input.id, input.workflow],
                    schedule_to_close_timeout=timedelta(minutes=1),
                )
                delegated_tokens = list(replacements.values())
            input.args = [
                replace_connector_authorization_tokens(arg, replacements)
                for arg in input.args
            ]
        try:
            return await self.next.start_child_workflow(input)
        except WorkflowAlreadyStartedError:
            if delegated_tokens:
                try:
                    await workflow.execute_activity(
                        'revoke_connector_authorizations',
                        args=[delegated_tokens],
                        schedule_to_close_timeout=timedelta(minutes=1),
                    )
                except BaseException:
                    log.exception('Failed best-effort cleanup of duplicate '
                                  'child-start capability')
            raise

    def start_activity(self, input: StartActivityInput) -> ActivityHandle:
        """Called for every :py:func:`temporalio.workflow.start_activity` and
        :py:func:`temporalio.workflow.execute_activity` call.
        """
        if not input.heartbeat_timeout:
            activity_timeout = (input.start_to_close_timeout
                                or input.schedule_to_close_timeout)
            if (not activity_timeout
                    or activity_timeout > _DEFAULT_HEARTBEAT_TIMEOUT):
                log.warning('Setting heartbeat timeout %s for %s.%s',
                            _DEFAULT_HEARTBEAT_TIMEOUT, input.activity,
                            input.activity_id)
                input.heartbeat_timeout = _DEFAULT_HEARTBEAT_TIMEOUT

        if not input.schedule_to_close_timeout:
            if _DEPLOY_STAGE.parse_value() != DeployStage.PROD:
                raise ApplicationError(f'Activity {input.activity} must set '
                                       'schedule_to_close_timeout')

            input.schedule_to_close_timeout = _DEFAULT_ACTIVITY_SCHEDULE_TO_CLOSE_TIMEOUT  # noqa: E501

        # Update retry policy
        self._update_activity_retry_policy(input)

        # Append suffix to task_queue of outbound activity.
        if input.task_queue:
            id_suffix = self._get_id_suffix()
            if id_suffix and not input.task_queue.endswith(id_suffix):
                input.task_queue += id_suffix

        handle = self.next.start_activity(input)
        # Wrap handle to track failures for timeout error enrichment
        return _ActivityHandleWithFailureTracking(
            handle, input.activity, self)  # type: ignore[return-value]


class KumoWorkflowInboundInterceptor(WorkflowInboundInterceptor):
    def init(self, outbound: WorkflowOutboundInterceptor) -> None:
        """Initialize with an outbound interceptor.

        To add a custom outbound interceptor, wrap the given interceptor before
        sending to the next ``init`` call.
        """
        self.next.init(KumoWorkflowOutboundInterceptor(outbound))

    async def execute_workflow(self, input: ExecuteWorkflowInput):
        """Execute workflow with error handling for PublicException and cancellation propagation.

        This interceptor is automatically invoked by Temporal for all workflow executions when
        KumoWorkflowInterceptor is registered.

        This method:
        1. Converts direct PublicException raises to ApplicationError
        2. Detects cancellation buried in ActivityError → ApplicationError → CancelledError chain
           and re-raises as CancelledError so the workflow is properly marked as cancelled
        3. Preserves full error chain for other errors

        The cancellation detection is important for activities like `wait_on_workflow` that poll
        other workflows - if the polled workflow is cancelled, the error chain contains
        CancelledError but wrapped in ApplicationError. Without this interceptor, the parent
        workflow would be marked as FAILED instead of CANCELLED.
        """
        try:
            return await self.next.execute_workflow(input)

        except ActivityError as ex:
            # Check for cancellation buried in ActivityError → ApplicationError chain.
            # This happens when an activity (e.g., wait_on_workflow) fails because
            # the workflow it was waiting on was cancelled.
            if isinstance(ex.cause, ApplicationError):
                cause = ex.cause.__cause__
                while cause:
                    if is_cancelled_exception(cause):
                        raise CancelledError(
                            'Workflow cancelled due to cancelled child/activity'
                        ) from cause
                    cause = getattr(cause, '__cause__', None)
            raise

        except PublicException as e:
            raise ApplicationError(
                e.message,
                e.to_dict(),
                type=e.__class__.__name__,
                non_retryable=e.non_retryable,
            ) from None


class KumoWorkflowInterceptor(Interceptor):
    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> Optional[Type[WorkflowInboundInterceptor]]:
        return KumoWorkflowInboundInterceptor
