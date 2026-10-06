# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import List, Optional

from pydantic import ValidationError
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    CancelledError,
    ChildWorkflowError,
    TemporalError,
    TimeoutError,
    TimeoutType,
)
from temporalio.service import RPCError as TemporalRPCError

from common_utils.error import PublicException
from io_utils import logging

from .auto_rpc_exception import RpcError as AutoRpcError
from .auto_rpc_exception import _AutoRpcNonRetriableError

# Common utilities for handling error/failure/exception related to Temporal
# workflow/activity execution.

log = logging.getLogger(__name__)


def is_cancelled_exception(exception: BaseException) -> bool:
    """Check whether the given exception is considered a cancellation exception
    according to Temporal.

    This is often used in a conditional of a catch clause to check whether a
    cancel occurred inside of a workflow. This can occur from
    :py:class:`asyncio.CancelledError` or :py:class:`CancelledError` or either
    :py:class:`ActivityError` or :py:class:`ChildWorkflowError` if either of
    those latter two have a :py:class:`CancelledError` cause.

    Args:
        exception: Exception to check.

    Returns:
        True if a cancelled exception, false if not.
    """
    return (isinstance(exception, asyncio.CancelledError)
            or isinstance(exception, CancelledError)
            or ((isinstance(exception, (ActivityError, ChildWorkflowError))
                 and isinstance(exception.cause, CancelledError))))


def wf_failure_to_temporal_trace(err: WorkflowFailureError) -> str:
    messages = ['Workflow execution failed!\n']
    _find_root_cause(err, messages)
    return '\n'.join(messages)


# Find root cause of TemporalError while appending error messages to the list.
def _find_root_cause(err: TemporalError, msgs: List[str]):
    cause = err.cause
    while isinstance(cause, BaseException):
        if isinstance(cause, ApplicationError):
            app_error_type = cause.type or 'application error'
            msgs.append(f'caused by {app_error_type}, message: {cause.message}')
            break

        if isinstance(cause, TimeoutError):
            if not cause.type:
                msgs.append(f'timeout error has occured: {cause.message}')
                break
            assert isinstance(cause.type, TimeoutType)
            if cause.type == TimeoutType.HEARTBEAT:
                msgs.append(
                    'caused by worker heartbeat timeout, which means the '
                    'worker process died or was restarted mid-task, most '
                    'commonly because it ran out of memory while processing '
                    'large data; a plain retry on the same data is likely '
                    'to fail the same way, so reduce the data volume or '
                    'contact Kumo support.')
            else:
                msgs.append(f'Task {cause.type.name} deadline has exceeded.')
            break

        if isinstance(cause, ChildWorkflowError):
            msgs.append('caused by failure in child workflow '
                        f'{cause.workflow_type}, message: {cause.message}')
            cause = cause.cause
        elif isinstance(cause, ActivityError):
            msgs.append(f'caused by failure in activity '
                        f'{cause.activity_type}, message: {cause.message}')
            cause = cause.cause
        elif isinstance(cause, ApplicationError):
            app_error_type = cause.type or 'application error'
            msgs.append(f'caused by {app_error_type}, message: {cause.message}')
            break
        elif isinstance(cause, TemporalRPCError):
            msgs.append(
                f'caused by RPC error status['
                f'{cause.status.name}={cause.status.value}]: {cause.message}')
            cause = cause.cause
        elif isinstance(cause, TemporalError):
            msgs.append(f'caused by {type(cause)}: {cause}')
            cause = cause.cause
        elif isinstance(cause, AutoRpcError):
            msgs.append(
                f'caused by RPC error={cause.error_type} to Kumo internal '
                f'service {cause.rpc_service}.{cause.rpc_method}, '
                f'message: {cause.message}')
            cause = cause.__cause__
        else:  # other non-Temporal exception
            msgs.append(f'caused by {type(cause)}: {cause}')
            cause = cause.__cause__

    return


DEFAULT_NON_RETRYABLE_EX = (
    ValueError,
    KeyError,
    AssertionError,
    AttributeError,
    ValidationError,
    PermissionError,
    TypeError,
    NotImplementedError,
    ArithmeticError,
    ImportError,
    # Kumo specific non-retriable error types:
    _AutoRpcNonRetriableError,
)

# Common error types to included by default in all Temporal
# RetryPolicy.non_retryable_error_types
# This will be used in interceptor and applied to all retry policy everywhere.
default_non_retryable_error_types = tuple(e.__name__
                                          for e in DEFAULT_NON_RETRYABLE_EX)

_TRANSIENT_CLUSTER_PATTERNS = (
    'StatusCode.UNAVAILABLE',
    'Connection refused',
)

_DEAD_CLUSTER_PATTERNS = (
    'created by JobLauncher',
    'Cannot start cluster',
    'Cluster terminated',
    'cluster has been terminated',
    'ClusterNotFoundException',
    'INVALID_STATE: Expected RUNNING',
)


def is_transient_cluster_error(e: BaseException) -> bool:
    r"""Check if an exception might be a transient cluster connectivity issue.

    These errors (e.g., momentary network blip, gRPC UNAVAILABLE) may resolve
    on their own and should be retried against the same cluster before assuming
    it is dead.
    """
    err_str = str(e)
    return any(p in err_str for p in _TRANSIENT_CLUSTER_PATTERNS)


def is_dead_cluster_error(e: BaseException) -> bool:
    r"""Check if an exception indicates a dead/terminated Databricks job
    cluster.

    This detects errors from Databricks Connect when the underlying
    job cluster has been terminated (e.g., idle timeout, max lifetime, or
    cancelled run).

    Use :func:`is_retryable_cluster_error` to check for *either* dead or
    transient cluster errors (e.g. in process_pool_utils where both
    categories should trigger a Temporal activity retry).
    """
    err_str = str(e)
    return any(p in err_str for p in _DEAD_CLUSTER_PATTERNS)


def is_retryable_cluster_error(e: BaseException) -> bool:
    r"""Check if an exception is a retryable cluster error (dead or transient).

    Both dead cluster errors and transient connectivity issues are retryable
    at the Temporal activity level — the activity will be retried with a
    potentially new cluster.
    """
    return is_dead_cluster_error(e) or is_transient_cluster_error(e)


def is_error_non_retryable(ex: BaseException):
    """Determine if exception is one of common non-retryable error types."""
    # DiskGraph's native runner lives in kumo-core and deliberately does not
    # depend on PublicException or Temporal. Its stable retry classification is
    # translated here at the platform boundary.
    from kumo_core.diskgraph.runner import DiskgraphIngestError

    if isinstance(ex, DiskgraphIngestError):
        return ex.non_retryable

    if isinstance(ex, ApplicationError):
        return ex.non_retryable

    if isinstance(ex, AutoRpcError):
        return ex.non_retryable

    if isinstance(ex, PublicException):
        return ex.non_retryable

    return isinstance(ex, DEFAULT_NON_RETRYABLE_EX)


_TIMEOUT_MESSAGES = {
    TimeoutType.SCHEDULE_TO_CLOSE:
    'Your job timed out before it could finish. This usually happens '
    'when the data is too large to process within the allowed time. '
    'Try reducing the data volume or contact Kumo support for help.',
    TimeoutType.START_TO_CLOSE:
    'Your job timed out while it was running. This usually happens '
    'when the data is too large to process within the allowed time. '
    'Try reducing the data volume or contact Kumo support for help.',
    TimeoutType.HEARTBEAT:
    'Your job stopped reporting progress and was aborted. This usually '
    'means the worker process running it crashed or was terminated, most '
    'commonly because it ran out of memory while processing large data. '
    'A plain retry on the same data is likely to fail the same way; '
    'reduce the data volume or contact Kumo support.',
    TimeoutType.SCHEDULE_TO_START:
    'Your job timed out waiting for an available worker. The system '
    'may be under heavy load. Please retry later.',
}


def _timeout_to_public_exception(
        error: TimeoutError,
        activity_name: Optional[str] = None) -> PublicException:
    """Convert a Temporal TimeoutError to a user-friendly PublicException.

    If the TimeoutError has a __cause__ (e.g. enriched by
    _ActivityHandleWithFailureTracking with the last retry failure),
    we prefer extracting a PublicException from that cause chain since
    it carries more specific context than a generic timeout message.

    ``activity_name`` (the wrapping ActivityError's activity_type, when the
    timeout came from an activity) names the timed-out step in the message,
    so a support ticket carries the failing activity instead of only a
    generic timeout sentence.
    """
    # Check __cause__ first — may contain a more specific error from a
    # prior retry attempt (see PR #26276).
    if error.__cause__ is not None:
        from_cause = extract_public_exception(error.__cause__)
        if from_cause is not None:
            return from_cause

    suffix = f' (timed-out step: {activity_name})' if activity_name else ''
    timeout_type = error.type
    if timeout_type is not None:
        message = _TIMEOUT_MESSAGES.get(timeout_type)
        if message is not None:
            return PublicException(message=message + suffix)

    return PublicException(
        message='Your job timed out. Please retry or contact Kumo support.'
        f'{suffix}')


def extract_public_exception(error: BaseException):
    """Extract PublicException from wrapped Temporal errors.

    This utility extracts and reconstructs PublicException from various Temporal
    error wrappers, allowing you to retrieve the original error information.

    Args:
        error: The exception to extract from (can be ApplicationError,
               ActivityError, ChildWorkflowError, or direct PublicException)

    Returns:
        Reconstructed PublicException if found, None otherwise

    Example:
        try:
            await workflow.execute_activity(...)
        except ActivityError as e:
            public_ex = extract_public_exception(e)
            if public_ex:
                return {"error": public_ex.message, "http_code": public_ex.http_code}
    """
    # Direct PublicException
    if isinstance(error, PublicException):
        return error

    # TimeoutError — produce a user-friendly message instead of leaking
    # raw Temporal strings like "activity ScheduleToClose timeout".
    if isinstance(error, TimeoutError):
        return _timeout_to_public_exception(error)

    # ApplicationError with PublicException details
    if isinstance(error, ApplicationError):
        # ApplicationError stores details as a tuple in the 'details' property
        details = getattr(error, 'details', None)
        if details and len(details) > 0:
            error_data = details[0]
            if isinstance(error_data, dict) and 'type' in error_data:
                try:
                    return PublicException.from_dict(error_data)
                except Exception as ex:
                    log.warning('Failed to reconstruct PublicException: %s', ex)

    # Wrapped in ActivityError or ChildWorkflowError
    if isinstance(error, (ActivityError, ChildWorkflowError)):
        cause = getattr(error, 'cause', None)
        if cause:
            # A timeout of an activity keeps the activity's name so the
            # public message can say which step timed out (an ActivityError
            # carries activity_type; a ChildWorkflowError does not).
            if isinstance(cause, TimeoutError):
                return _timeout_to_public_exception(
                    cause, activity_name=getattr(error, 'activity_type', None))
            return extract_public_exception(cause)

    # Generic error with cause attribute (fallback)
    if hasattr(error, 'cause'):
        cause = getattr(error, 'cause', None)
        if cause:
            return extract_public_exception(cause)

    return None
