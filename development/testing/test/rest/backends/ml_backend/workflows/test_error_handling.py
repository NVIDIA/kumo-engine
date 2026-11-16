# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for error handling utilities and ValidationResponse.to_exception()"""
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from temporalio.client import WorkflowExecutionStatus, WorkflowFailureError
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    CancelledError,
    ChildWorkflowError,
)
from temporalio.exceptions import TimeoutError as TemporalTimeoutError
from temporalio.exceptions import TimeoutType

from common_utils.error import PublicException
from kumo.validation.validation import (
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)
from rest.backends.ml_backend.ml_job_backend import (
    _job_cancelled_message,
    _update_execution_upon_wf_error,
)
from rest.config.batch_prediction_job import BatchPredictionJobOutput
from rest.config.error_handling import extract_failure_info, root_cause_message
from rest.config.training_job import TrainingJobOutput
from statedb.config.common import (
    JobEventLogEntry,
    JobFailureInfo,
    JobStatus,
    JobType,
)
from statedb.utils.errors import (
    ColumnStatsError,
    TableIngestionError,
    maybe_get_public_message,
)


# Exception classes for testing - defined at module level to avoid class registry pollution
# Note: Don't start with "Test" to avoid pytest collection warnings
class SamplePublicException(PublicException):
    """Sample exception for testing PublicException serialization/deserialization"""
    template = "Public exception message"


class SamplePublicExceptionForPriority(PublicException):
    """Sample exception for testing priority over ValidationResponse"""
    template = "Public error message"


class TestValidationResponseToException:
    """Tests for ValidationResponse.to_exception() method"""
    def test_to_exception_with_single_error(self):
        """Test converting ValidationResponse with single error to PublicException"""
        validation = ValidationResponse(
            errors=[
                ValidationError("Missing Column",
                                "Required column 'id' is missing")
            ], warnings=[], info_items=[])

        exc = validation.to_exception()

        # Check primary message includes error (no context prefix)
        assert "Missing Column" in exc.message

        # Check get_user_message() returns the primary error
        user_msg = exc.get_user_message()
        assert "Missing Column" in user_msg

    def test_to_exception_with_multiple_errors(self):
        """Test converting ValidationResponse with multiple errors"""
        validation = ValidationResponse(
            errors=[
                ValidationError("Missing Column",
                                "Required column 'id' is missing"),
                ValidationError("Invalid Type", "Column 'age' has wrong type")
            ], warnings=[], info_items=[])

        exc = validation.to_exception()

        # error_message() should serialize all errors
        user_msg = exc.get_user_message()
        # Both errors should be in the message via error_message()
        assert "Missing Column" in user_msg or "Invalid Type" in user_msg

    def test_to_exception_with_warnings(self):
        """Test that warnings are added as related_errors but not in user message"""
        validation = ValidationResponse(
            errors=[
                ValidationError("Missing Column",
                                "Required column 'id' is missing")
            ], warnings=[
                ValidationWarning("Low Row Count", "Only 10 rows found"),
                ValidationWarning("Old Data", "Data is from 2020")
            ], info_items=[])

        exc = validation.to_exception()

        # Warnings are added as plain Warning instances to related_errors
        assert len(exc.related_errors) == 2

        # But warnings DON'T appear in get_user_message() because they're not PublicExceptions
        user_message = exc.get_user_message()
        assert "Missing Column" in user_message  # Error is present
        assert "Low Row Count" not in user_message  # Warning is NOT present
        assert "Old Data" not in user_message  # Warning is NOT present

    def test_to_exception_raises_on_ok_validation(self):
        """Test that to_exception() raises ValueError when validation.ok is True"""
        validation = ValidationResponse(errors=[], warnings=[], info_items=[])

        with pytest.raises(ValueError,
                           match="Cannot convert successful validation"):
            validation.to_exception()

    def test_to_exception_serialization_roundtrip(self):
        """Test that PublicException from ValidationResponse can be serialized/deserialized"""
        validation = ValidationResponse(
            errors=[ValidationError("Test Error", "This is a test error")],
            warnings=[
                ValidationWarning("Test Warning", "This is a test warning")
            ], info_items=[])

        exc = validation.to_exception()

        # Serialize and deserialize
        serialized = exc.to_dict()
        deserialized = PublicException.from_dict(serialized)

        # Check messages are preserved
        assert deserialized.get_user_message() == exc.get_user_message()


class TestExtractFailureInfo:
    """Tests for extract_failure_info() utility function"""
    def test_extract_from_validation_response(self):
        """Test extracting failure_info from ValidationResponse"""
        validation = ValidationResponse(
            errors=[ValidationError("Data Error", "Invalid data format")],
            warnings=[], info_items=[])

        failure_info = extract_failure_info(validation=validation)

        # No context prefix - just the validation error
        assert "Data Error" in failure_info.message
        assert failure_info.stacktrace == ""

    def test_extract_from_err_msg(self):
        """Test extracting failure_info when only stacktrace is available (no user message)"""
        # Empty user message (internal error, not PublicException)
        failure_info = extract_failure_info(
            err_msg="", stacktrace="Job failed due to timeout")

        assert failure_info.message == ""
        assert failure_info.stacktrace == "Job failed due to timeout"

    def test_extract_from_exception(self):
        """Test extracting failure_info from Exception object"""
        try:
            raise RuntimeError("Something went wrong")
        except RuntimeError as e:
            failure_info = extract_failure_info(exception=e)

            assert failure_info.message == ""
            assert "RuntimeError" in failure_info.stacktrace
            assert "Something went wrong" in failure_info.stacktrace

    def test_priority_validation_over_public_error(self):
        """Test that ValidationResponse has priority over err_msg/stacktrace"""
        validation = ValidationResponse(
            errors=[ValidationError("Validation Error", "Priority error")],
            warnings=[], info_items=[])

        # Use module-level exception to avoid class registry pollution
        exc = SamplePublicExceptionForPriority()
        err_msg = exc.get_user_message()
        failure_info = extract_failure_info(err_msg=err_msg,
                                            stacktrace="test stack",
                                            validation=validation)

        # Validation should take priority
        assert "Validation Error" in failure_info.message
        assert "Public error message" not in failure_info.message

    def test_priority_public_error_over_err_msg(self):
        """Test that err_msg/stacktrace has priority over plain err_msg"""
        # Use module-level TestPublicException to avoid class registry pollution
        exc = SamplePublicException()
        err_msg = exc.get_user_message()

        failure_info = extract_failure_info(err_msg=err_msg,
                                            stacktrace="test stack trace")

        # err_msg/stacktrace should be used
        assert failure_info.message == "Public exception message"
        assert failure_info.stacktrace == "test stack trace"

    def test_fallback_to_generic_message(self):
        """Test fallback when no error information is provided"""
        failure_info = extract_failure_info()

        assert failure_info.message == ""
        assert failure_info.stacktrace == ""

    def test_invalid_public_error_dict_fallback(self):
        """Test that when only err_msg is provided (no stacktrace), both are populated with err_msg"""
        # Simulate case where only err_msg is available (backward compat)
        failure_info = extract_failure_info(err_msg="Fallback error message")

        assert failure_info.message == "Fallback error message"
        assert failure_info.stacktrace == "Fallback error message"

    def test_stacktrace_from_exception_with_validation(self):
        """The validation supplies the user message and the exception (threaded
        through by the workflow crash path) supplies the debug stacktrace.

        Before the fix, the workflow catch passed only the ValidationResponse,
        so failure_info.stacktrace, and hence the public API's error
        description field, was empty by construction for every workflow-crash
        failure."""
        validation = ValidationResponse(
            errors=[
                ValidationError("Training failed",
                                "activity place_snapshot_activity timed out")
            ], warnings=[], info_items=[])

        try:
            raise RuntimeError("placement blew up")
        except RuntimeError as e:
            failure_info = extract_failure_info(exception=e,
                                                validation=validation)

        assert "Training failed" in failure_info.message
        assert "RuntimeError" in failure_info.stacktrace
        assert "placement blew up" in failure_info.stacktrace


# Custom exception for testing chain walking
class CustomError(Exception):
    """Custom exception for testing recursive chain processing."""


class TestPublicExceptionRecursiveChain:
    """Tests for PublicException recursive exception chain processing.

    These tests verify that PublicException correctly walks exception chains
    (__cause__) and extracts all relevant error messages.
    """
    def test_simple_two_level_chain(self):
        """Test A(error=B) where both are PublicException."""
        b = PublicException(message="Inner error")
        a = PublicException(message="Outer error", error=b)

        result = str(a)
        expected = "Outer error\nInner error"
        assert result == expected

    def test_three_level_chain(self):
        """Test A(error=B(error=C)) where all are PublicException."""
        c = PublicException(message="Innermost error")
        b = PublicException(message="Middle error", error=c)
        a = PublicException(message="Outermost error", error=b)

        result = str(a)
        expected = "Outermost error\nMiddle error\nInnermost error"
        assert result == expected

    def test_value_error_hidden(self):
        """Test that ValueError is hidden when passed as error."""
        value_error = ValueError("Internal validation error")
        pub_exc = PublicException(message="User-facing error",
                                  error=value_error)

        result = str(pub_exc)
        assert result == "User-facing error"

    def test_temporal_error_hidden(self):
        """Test that Temporal framework exceptions are hidden."""
        # ApplicationError is a subclass of TemporalError
        temporal_error = ApplicationError("Workflow failed",
                                          "Internal workflow state")
        pub_exc = PublicException(message="User-facing error",
                                  error=temporal_error)

        result = str(pub_exc)
        assert result == "User-facing error"

    def test_temporal_error_with_custom_cause_exposed(self):
        """Test custom exceptions exposed when wrapped by Temporal errors."""
        # Create chain: Multiple Temporal errors -> Custom error
        custom = CustomError("Database connection failed")
        inner_temporal = ApplicationError("Workflow execution failed")
        inner_temporal.__cause__ = custom
        outer_temporal = ApplicationError("Activity failed")
        outer_temporal.__cause__ = inner_temporal

        pub_exc = PublicException(message="Operation failed",
                                  error=outer_temporal)

        result = str(pub_exc)
        expected = ("Operation failed\n"
                    "The following error occurred:\n"
                    "CustomError: Database connection failed")
        assert result == expected

    def test_custom_exception_exposed(self):
        """Test that custom exceptions are exposed."""
        custom = CustomError("Custom error message")
        pub_exc = PublicException(message="Wrapper error", error=custom)

        result = str(pub_exc)
        expected = ("Wrapper error\n"
                    "The following error occurred:\n"
                    "CustomError: Custom error message")
        assert result == expected

    def test_stops_at_public_exception(self):
        """Test that walking stops at first PublicException in chain."""
        # Create chain: CustomError -> PublicException
        custom = CustomError("Custom error")
        b = PublicException(message="B error", error=custom)
        a = PublicException(message="A error", error=b)

        result = str(a)
        expected = ("A error\n"
                    "B error\n"
                    "The following error occurred:\n"
                    "CustomError: Custom error")
        assert result == expected

    def test_walks_cause_chain(self):
        """Test that __cause__ chain is walked correctly."""
        # Create chain using __cause__
        innermost = CustomError("Innermost error")
        middle = CustomError("Middle error")
        middle.__cause__ = innermost

        outer = PublicException(message="Outer error", error=middle)

        result = str(outer)
        expected = ("Outer error\n"
                    "The following error occurred:\n"
                    "CustomError: Middle error\n"
                    "The following error occurred:\n"
                    "CustomError: Innermost error")
        assert result == expected

    def test_real_world_validation_pattern(self):
        """Test real-world pattern: validation error from backend."""
        # Simulate backend.py validation error
        validation = PublicException(
            message="Statistic 'MEDIAN' not supported for type 'text'")
        # Activity wraps with context
        stats_error = ColumnStatsError(
            message="Failed to compute stats for table 'users'",
            error=validation)

        result = str(stats_error)
        expected = ("Failed to compute stats for table 'users'\n"
                    "Statistic 'MEDIAN' not supported for type 'text'")
        assert result == expected

    def test_nested_subclasses(self):
        """Test nested PublicException subclasses."""
        validation = PublicException(message="Invalid column type")
        stats_error = ColumnStatsError(message="Column stats failed",
                                       error=validation)
        ingest_error = TableIngestionError(message="Table ingestion failed",
                                           error=stats_error)

        result = str(ingest_error)
        expected = ("Table ingestion failed\n"
                    "Column stats failed\n"
                    "Invalid column type")
        assert result == expected

    def test_no_error_parameter(self):
        """Test PublicException without error parameter."""
        exc = PublicException(message="Simple error")
        assert str(exc) == "Simple error"

    def test_empty_message_with_error(self):
        """Test PublicException with empty message but with error."""
        inner = PublicException(message="Inner error")
        outer = PublicException(message="", error=inner)

        result = str(outer)
        assert result == "Inner error"

    def test_multiple_builtin_exceptions_hidden(self):
        """Test that multiple builtin exceptions in chain are all hidden."""
        # Create chain: ValueError -> TypeError -> RuntimeError
        runtime_error = RuntimeError("Runtime issue")
        type_error = TypeError("Type issue")
        type_error.__cause__ = runtime_error
        value_error = ValueError("Value issue")
        value_error.__cause__ = type_error

        exc = PublicException(message="User-facing message", error=value_error)

        result = str(exc)
        assert result == "User-facing message"

    def test_mixed_chain_custom_and_builtin(self):
        """Test chain with both custom and builtin exceptions."""
        # Create chain: CustomError -> ValueError -> CustomError
        inner_custom = CustomError("Inner custom error")
        value_error = ValueError("Internal error")
        value_error.__cause__ = inner_custom
        outer_custom = CustomError("Outer custom error")
        outer_custom.__cause__ = value_error

        exc = PublicException(message="Wrapper", error=outer_custom)

        result = str(exc)
        expected = ("Wrapper\n"
                    "The following error occurred:\n"
                    "CustomError: Outer custom error\n"
                    "The following error occurred:\n"
                    "CustomError: Inner custom error")
        assert result == expected


class TestJobCancelledMessage:
    """Tests for _job_cancelled_message which maps workflow errors to job status.

    When a workflow fails with CancelledError as cause, the job should be
    marked CANCELLED (not FAILED). This is critical for the parent workflow
    cancellation fix.
    """
    def test_failed_workflow_with_cancelled_error_returns_cancel_message(self):
        """Test that FAILED workflow with CancelledError cause is treated as cancelled.

        This is the key case: when the workflow interceptor detects cancellation
        and re-raises CancelledError, the workflow FAILS (not CANCELS in Temporal),
        but _job_cancelled_message should still return a cancel message.
        """
        wf_error = Mock(spec=WorkflowFailureError)
        wf_error.cause = CancelledError("Child workflow was canceled")

        wf_description = Mock()
        wf_description.status = WorkflowExecutionStatus.FAILED

        result = _job_cancelled_message(wf_error, wf_description)
        assert result is not None
        assert "cancelled" in result.lower()

    def test_canceled_workflow_returns_cancel_message(self):
        """Test that CANCELED workflow status returns cancel message."""
        wf_error = Mock(spec=WorkflowFailureError)
        wf_error.cause = CancelledError("Workflow cancelled")

        wf_description = Mock()
        wf_description.status = WorkflowExecutionStatus.CANCELED

        result = _job_cancelled_message(wf_error, wf_description)
        assert result is not None

    def test_failed_workflow_without_cancelled_error_returns_none(self):
        """Test that regular FAILED workflow returns None (not a cancellation)."""
        wf_error = Mock(spec=WorkflowFailureError)
        wf_error.cause = RuntimeError("Some other error")

        wf_description = Mock()
        wf_description.status = WorkflowExecutionStatus.FAILED

        result = _job_cancelled_message(wf_error, wf_description)
        assert result is None

    def test_child_workflow_cancelled_returns_cancel_message(self):
        """Test that ChildWorkflowError with CancelledError cause is treated as cancelled.

        This is the case when a child workflow (e.g., GenerateTrainingTableWorkflow)
        is canceled - the parent workflow fails with ChildWorkflowError wrapping
        CancelledError.
        """
        child_wf_error = ChildWorkflowError(
            "Child Workflow execution cancelled",
            namespace="default",
            workflow_id="gen-train-table-123",
            run_id="run-456",
            workflow_type="GenerateTrainingTableWorkflow",
            initiated_event_id=1,
            started_event_id=2,
            retry_state=None,
        )
        child_wf_error.__cause__ = CancelledError("Cancelled")

        wf_error = Mock(spec=WorkflowFailureError)
        wf_error.cause = child_wf_error

        wf_description = Mock()
        wf_description.status = WorkflowExecutionStatus.FAILED

        result = _job_cancelled_message(wf_error, wf_description)
        assert result is not None
        assert "cancelled" in result.lower()


class TestWorkflowInterceptorCancellationDetection:
    """Tests for is_cancelled_exception helper used by workflow interceptor.

    The interceptor detects cancellation buried in ActivityError → ApplicationError
    → CancelledError chain and re-raises as CancelledError.
    """
    def test_detects_cancellation_in_activity_error_chain(self):
        """Test detection of ActivityError with CancelledError cause."""
        from temporalio_commons.error_utils import is_cancelled_exception

        # Direct CancelledError
        assert is_cancelled_exception(CancelledError("cancelled"))

        # ActivityError with CancelledError cause (direct)
        activity_error = ActivityError(
            "Activity failed",
            scheduled_event_id=1,
            started_event_id=2,
            identity="worker",
            activity_type="test",
            activity_id="act-1",
            retry_state=None,
        )
        activity_error.__cause__ = CancelledError("cancelled")
        assert is_cancelled_exception(activity_error)

    def test_does_not_match_non_cancellation(self):
        """Test that non-cancellation errors are not matched."""
        from temporalio_commons.error_utils import is_cancelled_exception

        assert not is_cancelled_exception(RuntimeError("some error"))
        assert not is_cancelled_exception(ApplicationError("app error"))


class TestGenerateTableErrorDetailFallback:
    """Generate-table failures expose curated or generic messages only."""
    _GENERIC = (
        "An internal error occurred while generating the table. Please retry "
        "the job. If the problem continues, contact Kumo support."
    )

    def _build_detail(self, ex: Exception) -> str:
        """Mirror the detail expression used by the workflow handlers."""
        return maybe_get_public_message(ex) or self._GENERIC

    def test_non_public_exception_detail_is_generic(self):
        ex = RuntimeError("connector timed out")

        detail = self._build_detail(ex)

        assert detail == self._GENERIC
        assert "connector timed out" not in detail

    def test_value_error_detail_does_not_expose_raw_message(self):
        ex = ValueError("Failed to compute target time range for InvoiceDate")

        detail = self._build_detail(ex)

        assert detail == self._GENERIC
        assert "InvoiceDate" not in detail

    def test_public_exception_message_is_preferred(self):
        """When a public message exists, it takes priority over the fallback."""
        ex = SamplePublicException()

        detail = self._build_detail(ex)

        assert detail == "Public exception message"

    def test_validation_response_contains_no_raw_internal_detail(self):
        ex = RuntimeError("connector timed out")

        validation_response = ValidationResponse(errors=[
            ValidationError(
                title="Unexpected error in generating training table",
                message=self._build_detail(ex),
            )
        ])

        assert not validation_response.ok
        assert self._GENERIC in validation_response.error_message()
        assert "connector timed out" not in validation_response.error_message()


def _activity_error(
        activity_type: str = 'place_snapshot_activity') -> ActivityError:
    return ActivityError(
        'Activity task failed',
        scheduled_event_id=1,
        started_event_id=2,
        identity='worker@host',
        activity_type=activity_type,
        activity_id='act-1',
        retry_state=None,
    )


def _child_workflow_error() -> ChildWorkflowError:
    return ChildWorkflowError(
        'Child Workflow execution failed',
        namespace='default',
        workflow_id='training-job-workflow-abc',
        run_id='run-1',
        workflow_type='TrainingJobWorkflow',
        initiated_event_id=1,
        started_event_id=2,
        retry_state=None,
    )


class TestRootCauseMessage:
    """root_cause_message keeps the deepest failure message AND the failing
    activity's name.

    The wrappers Temporal puts around a failure (ChildWorkflowError ->
    ActivityError -> TimeoutError/ApplicationError) hold the activity name,
    which the deepest message alone lacks: a bare timeout reads
    "activity StartToClose timeout" without naming any activity.
    """
    def test_timeout_names_activity_and_timeout_type(self):
        """An activity timeout is rendered with the activity name and timeout
        type instead of Temporal's bare 'activity StartToClose timeout'."""
        timeout = TemporalTimeoutError('activity StartToClose timeout',
                                       type=TimeoutType.START_TO_CLOSE,
                                       last_heartbeat_details=[])
        activity = _activity_error()
        activity.__cause__ = timeout
        child = _child_workflow_error()
        child.__cause__ = activity

        assert root_cause_message(child) == (
            'activity place_snapshot_activity timed out (StartToClose)')

    def test_wrapped_message_names_activity(self):
        """A message wrapped in an ActivityError carries the activity name
        (e.g. the interceptor's masked constant, see kumo#30048: the constant
        itself is that issue's business, but the activity name is available
        here regardless)."""
        masked = ApplicationError(
            'The activity failed due to an internal error.',
            type='ApplicationError')
        activity = _activity_error()
        activity.__cause__ = masked

        assert root_cause_message(activity) == (
            'activity place_snapshot_activity failed: '
            'The activity failed due to an internal error.')

    def test_real_cause_message_survives_with_activity_context(self):
        """A descriptive deepest message is kept, prefixed with the activity."""
        real = ApplicationError('KeyError: train_seeds', type='KeyError')
        activity = _activity_error('materialize_train_task')
        activity.__cause__ = real
        child = _child_workflow_error()
        child.__cause__ = activity

        assert root_cause_message(child) == (
            'activity materialize_train_task failed: KeyError: train_seeds')

    def test_message_already_naming_activity_not_double_prefixed(self):
        real = ApplicationError(
            'activity place_snapshot_activity ran out of memory',
            type='RuntimeError')
        activity = _activity_error()
        activity.__cause__ = real

        assert root_cause_message(activity) == (
            'activity place_snapshot_activity ran out of memory')

    def test_timeout_without_activity_keeps_temporal_message(self):
        """A workflow-level timeout (no ActivityError wrapper) keeps
        Temporal's own message."""
        timeout = TemporalTimeoutError('workflow execution timeout',
                                       type=TimeoutType.START_TO_CLOSE,
                                       last_heartbeat_details=[])

        assert root_cause_message(timeout) == 'workflow execution timeout'

    def test_chain_without_activity_returns_deepest_message(self):
        """No ActivityError in the chain: behavior is unchanged, the deepest
        FailureError message wins over the wrapper text."""
        real = ApplicationError('KeyError: pred_abc/train_seeds',
                                type='KeyError')
        child = _child_workflow_error()
        child.__cause__ = real

        assert root_cause_message(child) == 'KeyError: pred_abc/train_seeds'

    def test_plain_exception_falls_back_to_str(self):
        assert root_cause_message(
            RuntimeError('materialize blew up')) == 'materialize blew up'


class TestUpdateExecutionUponWfErrorFallback:
    """_update_execution_upon_wf_error (the reconciliation safety net for
    workflows that died without persisting failure_info) must never store an
    empty failure_info.message.

    It builds JobFailureInfo directly, bypassing the JobExecution.failed()
    constructors and their non-empty fallback, so before the fix it stored
    message='' whenever the Temporal chain carried no PublicException.
    """
    _TS = datetime(2026, 7, 16, tzinfo=timezone.utc)

    def _wf_description(self):
        wf_description = Mock()
        wf_description.status = WorkflowExecutionStatus.FAILED
        wf_description.close_time = self._TS
        return wf_description

    def test_message_never_empty_without_public_exception(self):
        """A chain with no PublicException used to produce message=''."""
        cause = ApplicationError(
            'The activity failed due to an internal error.',
            type='ApplicationError')
        wf_error = WorkflowFailureError(cause=cause)
        execution = TrainingJobOutput(status=JobStatus.RUNNING,
                                      log=JobEventLogEntry.starting(self._TS))

        _update_execution_upon_wf_error(JobType.TRAINING_JOB,
                                        execution, wf_error,
                                        self._wf_description(), self._TS)

        assert execution.status == JobStatus.FAILED
        assert execution.failure_info is not None
        # Mirrors TrainingJobOutput.failed()'s human-readable fallback.
        assert execution.failure_info.message == 'Training job failed'
        # The Temporal error detail is still preserved for debugging.
        assert 'internal error' in execution.failure_info.stacktrace

    def test_fallback_message_human_readable_for_batch_prediction(self):
        """The fallback matches BatchPredictionJobOutput.failed()'s
        human-readable text."""
        cause = ApplicationError(
            'The activity failed due to an internal error.',
            type='ApplicationError')
        wf_error = WorkflowFailureError(cause=cause)
        execution = BatchPredictionJobOutput(
            status=JobStatus.RUNNING, log=JobEventLogEntry.starting(self._TS))

        _update_execution_upon_wf_error(JobType.BATCH_PREDICTION_JOB,
                                        execution, wf_error,
                                        self._wf_description(), self._TS)

        assert execution.status == JobStatus.FAILED
        assert execution.failure_info is not None
        assert execution.failure_info.message == 'Batch prediction job failed'

    def test_public_exception_message_not_overridden(self):
        """A real public message from the chain is kept, not replaced by the
        fallback."""
        wf_error = WorkflowFailureError(cause=SamplePublicException())
        execution = TrainingJobOutput(status=JobStatus.RUNNING,
                                      log=JobEventLogEntry.starting(self._TS))

        _update_execution_upon_wf_error(JobType.TRAINING_JOB,
                                        execution, wf_error,
                                        self._wf_description(), self._TS)

        assert execution.status == JobStatus.FAILED
        assert execution.failure_info is not None
        assert execution.failure_info.message == 'Public exception message'

    def test_workflow_set_failure_info_preserved(self):
        """failure_info persisted by the workflow itself is never overwritten
        by the reconciliation path (existing behavior, pinned)."""
        cause = ApplicationError('irrelevant', type='ApplicationError')
        wf_error = WorkflowFailureError(cause=cause)
        execution = TrainingJobOutput(
            status=JobStatus.RUNNING, log=JobEventLogEntry.starting(self._TS),
            failure_info=JobFailureInfo(
                message='Training failed. activity x timed out',
                stacktrace='real trace'))

        _update_execution_upon_wf_error(JobType.TRAINING_JOB,
                                        execution, wf_error,
                                        self._wf_description(), self._TS)

        assert execution.status == JobStatus.FAILED
        assert execution.failure_info is not None
        assert execution.failure_info.message == (
            'Training failed. activity x timed out')
        assert execution.failure_info.stacktrace == 'real trace'
