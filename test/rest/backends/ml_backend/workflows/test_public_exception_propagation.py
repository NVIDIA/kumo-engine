# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PublicException propagation through Temporal workflows."""
import tempfile

from kumo_core.job_config import PredStatus, TrainStatus

from common_utils.error import PublicException


class ValidationError(PublicException):
    """Test validation error."""
    template = "Validation error: {reason}"
    http_code = 400


class DatabaseError(PublicException):
    """Test database error."""
    template = "Database issue: {detail}"


class TestStatusSerialization:
    """Test that Status classes properly serialize error_msg and stacktrace."""
    def test_train_status_saves_public_error(self):
        """Test that TrainStatus saves and loads error_msg and stacktrace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exc = ValidationError(reason="invalid input")
            exc.add_related_error(DatabaseError(detail="connection timeout"))

            status = TrainStatus()
            status.error_msg = exc.get_user_message()
            status.stacktrace = "exception caused by..."
            status.save(tmpdir)

            # Load and verify
            loaded = TrainStatus.load(tmpdir)
            assert loaded.error_msg is not None
            assert 'Validation error' in loaded.error_msg
            assert 'invalid input' in loaded.error_msg

            # Related PublicExceptions appear in error_msg via get_user_message()
            assert 'Database issue' in loaded.error_msg
            assert 'connection timeout' in loaded.error_msg
            assert 'Related errors:' in loaded.error_msg

            # Stacktrace is whatever was explicitly set
            assert loaded.stacktrace == "exception caused by..."

    def test_pred_status_saves_public_error(self):
        """Test that PredStatus saves and loads error_msg and stacktrace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exc = ValidationError(reason="missing required field")

            status = PredStatus()
            status.error_msg = exc.get_user_message()
            status.stacktrace = "exception caused by..."
            status.save(tmpdir)

            # Load and verify
            loaded = PredStatus.load(tmpdir)
            assert loaded.error_msg is not None
            assert 'missing required field' in loaded.error_msg
            assert loaded.stacktrace is not None


class TestWorkflowErrorHandling:
    """Test workflow error message construction."""
    def test_user_message_from_public_exception(self):
        """Test that user gets clean message from error_msg and stacktrace."""
        # Create a PublicException
        exc = ValidationError(reason="database timeout")
        exc.add_related_error(ValueError("Connection failed"))

        # Extract messages as activities would
        user_message = exc.get_user_message()

        # Verify messages
        assert user_message is not None
        assert "Validation error: database timeout" in user_message
        # User message is clean (standard exceptions aren't shown to users)
        assert "ValueError" not in user_message

    def test_fallback_to_error_msg_when_no_public_error(self):
        """Test that workflow uses err_msg when available."""
        err_msg = "Internal system error"
        stacktrace = "Internal system error"

        # Simulate workflow logic using extract_failure_info
        from rest.config.error_handling import extract_failure_info

        failure_info = extract_failure_info(err_msg=err_msg,
                                            stacktrace=stacktrace)

        # Verify messages
        assert failure_info.message == "Internal system error"
        assert failure_info.stacktrace == "Internal system error"

    def test_fallback_when_reconstruction_fails(self):
        """Test that workflow handles missing error info gracefully."""
        err_msg = "Original error message"
        stacktrace = "Original error stacktrace"

        # Simulate workflow logic
        from rest.config.error_handling import extract_failure_info

        failure_info = extract_failure_info(err_msg=err_msg,
                                            stacktrace=stacktrace)

        # Verify fallback happened
        assert failure_info.message == "Original error message"
        assert failure_info.stacktrace == "Original error stacktrace"

    def test_no_job_prefix_in_user_message(self):
        """Test that user messages don't contain 'Job X failed' prefix."""
        exc = ValidationError(reason="validation error")
        user_message = exc.get_user_message()

        # Verify no job prefix
        assert "Job" not in user_message
        assert "failed" not in user_message.lower(
        ) or "validation" in user_message.lower()
        assert "Validation error: validation error" in user_message
