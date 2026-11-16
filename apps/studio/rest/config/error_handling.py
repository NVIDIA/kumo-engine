# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared error handling utilities for workflows."""
from typing import Optional

from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ActivityError, FailureError
from temporalio.exceptions import TimeoutError as TemporalTimeoutError
from temporalio.exceptions import TimeoutType

from io_utils import logging
from kumo.utils.formatting import format_exception
from kumo.validation.validation import ValidationResponse
from statedb.config.common import JobFailureInfo
from statedb.utils.errors import maybe_get_public_message
from temporalio_commons.error_utils import wf_failure_to_temporal_trace

log = logging.getLogger(__name__)

# Temporal's own convention for timeout names in failure messages, e.g.
# "activity StartToClose timeout".
_TIMEOUT_TYPE_NAMES = {
    TimeoutType.START_TO_CLOSE: 'StartToClose',
    TimeoutType.SCHEDULE_TO_START: 'ScheduleToStart',
    TimeoutType.SCHEDULE_TO_CLOSE: 'ScheduleToClose',
    TimeoutType.HEARTBEAT: 'Heartbeat',
}


def root_cause_message(ex: BaseException) -> str:
    """Descend an exception chain to the deepest failure message, naming the
    failing activity when the chain identifies one.

    A failed child workflow arrives wrapped in nested ``FailureError`` layers
    (ChildWorkflowError -> ActivityError -> ApplicationError) whose outer
    ``str()`` is only generic wrapper text; the real cause is on ``.__cause__``.
    The wrappers also hold metadata the deepest message lacks: a bare Temporal
    timeout reads "activity StartToClose timeout" without ever naming the
    activity, so the enclosing ``ActivityError.activity_type`` is folded into
    the returned message.  Deterministic (no IO/clock/randomness), so safe in
    workflow code.  Falls back to ``str(ex)`` when the chain has no temporalio
    failure.
    """
    message = str(ex)
    deepest: Optional[FailureError] = None
    activity_type: Optional[str] = None
    cur: Optional[BaseException] = ex
    seen: set = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ActivityError) and cur.activity_type:
            # The innermost ActivityError encloses the deepest failure.
            activity_type = cur.activity_type
        # Deepest FailureError.message wins; plain __cause__ links don't
        # override.
        if isinstance(cur, FailureError) and cur.message:
            message = cur.message
            deepest = cur
        cur = cur.__cause__
    if activity_type is None:
        return message
    if isinstance(deepest, TemporalTimeoutError):
        timeout_name = (_TIMEOUT_TYPE_NAMES.get(deepest.type)
                        if deepest.type is not None else None)
        suffix = f' ({timeout_name})' if timeout_name else ''
        return f'activity {activity_type} timed out{suffix}'
    if activity_type in message:
        return message
    return f'activity {activity_type} failed: {message}'


def extract_failure_info(
        err_msg: Optional[str] = None, stacktrace: Optional[str] = None,
        exception: Optional[BaseException] = None,
        validation: Optional[ValidationResponse] = None) -> JobFailureInfo:
    """
    Extract failure information.

    User message and debug stacktrace are extracted independently:

    User message priority (what to show to users):
    1. ValidationResponse converted to PublicException message (via to_exception())
    2. err_msg (e.g. from Train/Pred status file)
    3. PublicException extracted from exception object (``Exception``
       instances only; a ``BaseException``-only exception such as a
       cancellation never carries a public message)
    4. "" (empty - no user-facing message available)

    Debug stacktrace priority (what to show to developers):
    1. Explicit stacktrace parameter
    2. Exception formatted with full trace
    3. err_msg (e.g. from Train/Pred status file) as fallback
    4. "" (empty)

    Args:
        err_msg: User-facing error message from status file
        stacktrace: Stacktrace, e.g. from training / prediction status
        exception: Exception object (may be wrapped in Temporal exceptions);
            accepts ``BaseException`` since workflow crash handlers catch
            ``BaseException``, and such exceptions still yield a stacktrace
        validation: ValidationResponse (if validation failed)

    Returns:
        JobFailureInfo with user-friendly message and debug stacktrace
    """
    user_message = ""
    debug_stacktrace = ""

    # Extract user message
    if validation and not validation.ok:
        try:
            exc = validation.to_exception()
            user_message = exc.get_user_message()
        except Exception:
            log.warning("Failed to convert ValidationResponse to PublicException")
            user_message = validation.error_message()
    elif err_msg:
        user_message = err_msg
    elif isinstance(exception, Exception):
        user_message = maybe_get_public_message(exception)

    # Extract debug stacktrace
    if stacktrace:
        debug_stacktrace = stacktrace
    elif exception:
        if isinstance(exception, WorkflowFailureError):
            debug_stacktrace = wf_failure_to_temporal_trace(exception)
        else:
            debug_stacktrace = format_exception(exception)
    elif err_msg:
        debug_stacktrace = err_msg

    return JobFailureInfo(
        message=user_message,
        stacktrace=debug_stacktrace
    )
