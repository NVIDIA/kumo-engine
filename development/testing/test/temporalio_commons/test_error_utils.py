# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from temporalio.client import WorkflowFailureError
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    ChildWorkflowError,
    TimeoutError,
    TimeoutType,
)

from common_utils.error import PublicException
from temporalio_commons.error_utils import (
    extract_public_exception,
    wf_failure_to_temporal_trace,
)


def _make_timeout_error(timeout_type):
    """Create a TimeoutError with the given type."""
    return TimeoutError("timeout", type=timeout_type, last_heartbeat_details=[])


def _wrap_in_activity_error(cause):
    """Wrap an exception in an ActivityError."""
    err = ActivityError(
        message="activity failed",
        scheduled_event_id=1,
        started_event_id=2,
        identity="",
        activity_type="generate_training_table",
        activity_id="1",
        retry_state=None,
    )
    err.__cause__ = cause
    return err


def _wrap_in_child_workflow_error(cause):
    """Wrap an exception in a ChildWorkflowError."""
    err = ChildWorkflowError(
        message="child workflow failed",
        namespace="default",
        workflow_id="wf-123",
        run_id="run-456",
        workflow_type="GenerateTrainingTableWorkflow",
        initiated_event_id=1,
        started_event_id=2,
        retry_state=None,
    )
    err.__cause__ = cause
    return err


class TestExtractPublicExceptionTimeout:
    """Tests that extract_public_exception handles TimeoutError correctly,
    producing user-friendly messages instead of raw Temporal strings."""
    def test_schedule_to_close_timeout_via_activity_error(self):
        timeout = _make_timeout_error(TimeoutType.SCHEDULE_TO_CLOSE)
        activity_err = _wrap_in_activity_error(timeout)

        result = extract_public_exception(activity_err)

        assert result is not None
        msg = result.get_user_message()
        assert 'timed out' in msg
        assert 'SCHEDULE_TO_CLOSE' not in msg
        # An activity timeout names the timed-out step for supportability.
        assert 'generate_training_table' in msg

    def test_heartbeat_timeout_via_activity_error(self):
        timeout = _make_timeout_error(TimeoutType.HEARTBEAT)
        activity_err = _wrap_in_activity_error(timeout)

        result = extract_public_exception(activity_err)

        assert result is not None
        msg = result.get_user_message()
        # kumo#30084: a heartbeat timeout means the worker process died or
        # was restarted mid-step (commonly the kernel OOM killer), not that
        # the job is merely slow; heartbeats are auto-emitted while the
        # worker is alive. The message must be honest about that, name the
        # timed-out step, and not present a plain retry as the fix.
        assert 'unresponsive' not in msg
        assert 'out of memory' in msg
        assert 'generate_training_table' in msg

    def test_heartbeat_timeout_direct_has_no_step_suffix(self):
        timeout = _make_timeout_error(TimeoutType.HEARTBEAT)

        result = extract_public_exception(timeout)

        assert result is not None
        msg = result.get_user_message()
        assert 'timed-out step' not in msg
        assert 'out of memory' in msg

    def test_start_to_close_timeout(self):
        timeout = _make_timeout_error(TimeoutType.START_TO_CLOSE)
        activity_err = _wrap_in_activity_error(timeout)

        result = extract_public_exception(activity_err)

        assert result is not None
        assert 'timed out' in result.get_user_message()

    def test_schedule_to_start_timeout(self):
        timeout = _make_timeout_error(TimeoutType.SCHEDULE_TO_START)
        activity_err = _wrap_in_activity_error(timeout)

        result = extract_public_exception(activity_err)

        assert result is not None
        assert 'timed out' in result.get_user_message()

    def test_timeout_nested_in_child_workflow_and_activity(self):
        """ActivityError(TimeoutError) wrapped in ChildWorkflowError."""
        timeout = _make_timeout_error(TimeoutType.SCHEDULE_TO_CLOSE)
        activity_err = _wrap_in_activity_error(timeout)
        child_wf_err = _wrap_in_child_workflow_error(activity_err)

        result = extract_public_exception(child_wf_err)

        assert result is not None
        assert 'timed out' in result.get_user_message()

    def test_bare_timeout_error(self):
        """TimeoutError passed directly (not wrapped)."""
        timeout = _make_timeout_error(TimeoutType.SCHEDULE_TO_CLOSE)

        result = extract_public_exception(timeout)

        assert result is not None
        assert 'timed out' in result.get_user_message()

    def test_timeout_with_no_type(self):
        timeout = TimeoutError("unknown timeout", type=None,
                               last_heartbeat_details=[])

        result = extract_public_exception(timeout)

        assert result is not None
        assert 'timed out' in result.get_user_message()

    def test_application_error_still_works(self):
        """Existing ApplicationError handling is not broken."""
        app_err = ApplicationError(message="Something went wrong")
        activity_err = _wrap_in_activity_error(app_err)

        result = extract_public_exception(activity_err)

        # ApplicationError without PublicException details returns None
        # (existing behavior)
        assert result is None

    def test_timeout_with_prior_failure_cause(self):
        """TimeoutError enriched with prior failure via __cause__ (PR #26276).

        When _ActivityHandleWithFailureTracking attaches a prior retry
        failure to TimeoutError.__cause__, we should prefer the more
        specific PublicException from that cause over the generic timeout
        message.
        """
        # Simulate the prior failure: ApplicationError wrapping a PublicException
        # Note: ApplicationError takes details as *args, not keyword
        public_ex = PublicException(message="DiskGraph worker OOM")
        prior_app_err = ApplicationError(
            "activity failed",
            public_ex.to_dict(),
            type="DiskGraphError",
        )
        prior_activity_err = _wrap_in_activity_error(prior_app_err)

        # Simulate the timeout with __cause__ set to the prior failure
        timeout = _make_timeout_error(TimeoutType.START_TO_CLOSE)
        timeout.__cause__ = prior_activity_err

        activity_err = _wrap_in_activity_error(timeout)

        result = extract_public_exception(activity_err)

        assert result is not None
        assert 'DiskGraph worker OOM' in result.get_user_message()

    def test_timeout_with_non_public_cause_falls_back_to_timeout_msg(self):
        """TimeoutError.__cause__ without PublicException falls back to
        generic timeout message."""
        timeout = _make_timeout_error(TimeoutType.SCHEDULE_TO_CLOSE)
        timeout.__cause__ = ValueError("some internal error")

        result = extract_public_exception(timeout)

        assert result is not None
        assert 'timed out' in result.get_user_message()

    def test_none_for_unknown_error(self):
        """Non-Temporal errors without PublicException still return None."""
        cause = ValueError("bad input")
        activity_err = _wrap_in_activity_error(cause)

        result = extract_public_exception(activity_err)

        assert result is None


class TestWfFailureToTemporalTrace:
    """Tests for the internal trace builder used in JobFailureInfo."""
    def test_heartbeat_timeout_trace_matches_public_semantics(self):
        """The heartbeat-timeout trace states that the worker process died
        or was restarted (commonly OOM) and does not advise a plain retry,
        consistent with _TIMEOUT_MESSAGES[TimeoutType.HEARTBEAT]."""
        timeout = _make_timeout_error(TimeoutType.HEARTBEAT)
        activity_err = _wrap_in_activity_error(timeout)
        wf_err = WorkflowFailureError(cause=activity_err)

        trace = wf_failure_to_temporal_trace(wf_err)

        assert 'caused by worker heartbeat timeout' in trace
        assert 'died or was restarted' in trace
        assert 'ran out of memory' in trace
        assert 'unresponsive' not in trace
        assert 'please retry' not in trace

    def test_non_heartbeat_timeout_trace_unchanged(self):
        timeout = _make_timeout_error(TimeoutType.SCHEDULE_TO_CLOSE)
        activity_err = _wrap_in_activity_error(timeout)
        wf_err = WorkflowFailureError(cause=activity_err)

        trace = wf_failure_to_temporal_trace(wf_err)

        assert 'Task SCHEDULE_TO_CLOSE deadline has exceeded.' in trace
