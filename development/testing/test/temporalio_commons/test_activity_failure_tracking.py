# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _ActivityHandleWithFailureTracking.

Tests the failure tracking and timeout enrichment logic in isolation using mocks.
"""
from typing import Any
from unittest.mock import MagicMock

import pytest
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    RetryState,
    TimeoutError,
    TimeoutType,
)

from temporalio_commons.worker_interceptors import (
    KumoWorkflowOutboundInterceptor,
    _ActivityHandleWithFailureTracking,
)


@pytest.fixture
def mock_interceptor():
    """Create a mock interceptor with an empty _last_activity_failures dict."""
    interceptor = MagicMock(spec=KumoWorkflowOutboundInterceptor)
    interceptor._last_activity_failures = {}
    return interceptor


def create_activity_error(message="Activity failed", cause=None):
    """Create an ActivityError with optional cause."""
    error = ActivityError(
        message,
        scheduled_event_id=1,
        started_event_id=2,
        identity="test-worker",
        activity_type="test_activity",
        activity_id="activity-1",
        retry_state=RetryState.TIMEOUT,
    )
    if cause:
        error.__cause__ = cause
    return error


def create_timeout_error(message="Activity timed out", cause=None):
    """Create a TimeoutError with optional cause."""
    error = TimeoutError(
        message,
        type=TimeoutType.START_TO_CLOSE,
        last_heartbeat_details=[],
    )
    if cause:
        error.__cause__ = cause
    return error


def mock_activity_handle(result: Any = None, exception: BaseException | None = None) -> Any:
    """Create a mock activity handle that can be awaited."""
    async def coro():
        if exception:
            raise exception
        return result
    return coro()


@pytest.mark.asyncio
async def test_failure_tracking_stores_activity_error(mock_interceptor):
    """When activity raises ActivityError, it's stored in _last_activity_failures."""
    original_error = create_activity_error("Database connection failed")

    wrapper = _ActivityHandleWithFailureTracking(
        mock_activity_handle(exception=original_error), "test_activity", mock_interceptor
    )

    with pytest.raises(ActivityError) as exc_info:
        await wrapper

    assert exc_info.value is original_error
    assert mock_interceptor._last_activity_failures["test_activity"] is original_error


@pytest.mark.asyncio
async def test_timeout_enrichment_sets_cause(mock_interceptor):
    """TimeoutError.__cause__ is set to prior failure when available."""
    # Store a prior failure
    prior_failure = create_activity_error("Prior database error")
    mock_interceptor._last_activity_failures["test_activity"] = prior_failure

    # Create timeout error chain: ActivityError -> TimeoutError
    timeout_error = create_timeout_error("Start-to-close timeout")
    activity_error = create_activity_error(message="Activity timed out", cause=timeout_error)

    wrapper = _ActivityHandleWithFailureTracking(
        mock_activity_handle(exception=activity_error), "test_activity", mock_interceptor
    )

    with pytest.raises(ActivityError):
        await wrapper

    # Verify the timeout error's __cause__ is set to the prior failure
    assert timeout_error.__cause__ is prior_failure


@pytest.mark.asyncio
async def test_successful_activity_clears_stored_failure(mock_interceptor):
    """Successful activity clears any stored failure (prevents memory growth)."""
    # Store a prior failure
    mock_interceptor._last_activity_failures["test_activity"] = create_activity_error()

    wrapper = _ActivityHandleWithFailureTracking(
        mock_activity_handle(result="success"), "test_activity", mock_interceptor
    )

    result = await wrapper

    assert result == "success"
    assert "test_activity" not in mock_interceptor._last_activity_failures


@pytest.mark.parametrize("scenario", ["nested", "no_timeout"])
def test_find_innermost_timeout_in_chain(scenario):
    """Test _find_innermost_timeout_in_chain finds deepest timeout or None."""
    if scenario == "nested":
        # Chain: ActivityError -> TimeoutError (outer) -> TimeoutError (inner)
        inner_timeout = create_timeout_error("Inner timeout")
        outer_timeout = create_timeout_error("Outer timeout", cause=inner_timeout)
        exc = create_activity_error(message="Activity failed", cause=outer_timeout)
        expected = inner_timeout
    else:  # no_timeout
        # Chain: ActivityError -> ApplicationError
        app_error = ApplicationError("Application error")
        exc = create_activity_error(message="Activity failed", cause=app_error)
        expected = None

    result = _ActivityHandleWithFailureTracking._find_innermost_timeout_in_chain(exc)
    assert result is expected
