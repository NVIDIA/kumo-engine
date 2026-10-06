# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from common_utils.error import PublicException, PublicMessage


class SimpleMessage(PublicMessage):
    template = "Hello {name}"


class SimpleError(PublicException):
    template = "Error: {reason}"


class RelatedError(PublicException):
    template = "Related: {detail}"


class TestPublicMessage:
    def test_template_formatting(self):
        msg = SimpleMessage(name="Alice")
        assert str(msg) == "Hello Alice"
        assert msg.kwargs == {"name": "Alice"}

    def test_missing_template_key(self):
        msg = SimpleMessage()
        assert "[Missing key: name]" in str(msg)


class TestPublicException:
    def test_basic_exception(self):
        exc = SimpleError(reason="not found")
        assert str(exc) == "Error: not found"
        assert exc.non_retryable is True
        assert exc.http_code == 422
        assert exc.related_errors == []

    def test_can_be_raised(self):
        with pytest.raises(SimpleError):
            raise SimpleError(reason="server error")

    def test_add_related_errors(self):
        main = SimpleError(reason="server error")
        main.add_related_error(RelatedError(detail="timeout"))
        main.add_related_error(ValueError("validation failed"))

        assert len(main.related_errors) == 2
        assert isinstance(main.related_errors[0], PublicException)
        assert isinstance(main.related_errors[1], ValueError)


class TestGetUserMessage:
    def test_simple_message(self):
        exc = SimpleError(reason="not found")
        assert exc.get_user_message() == "Error: not found"

    def test_with_public_related_errors(self):
        main = SimpleError(reason="server error")
        main.add_related_error(RelatedError(detail="timeout"))
        main.add_related_error(RelatedError(detail="retry failed"))

        expected = """Error: server error

Related errors:
  - Related: timeout
  - Related: retry failed"""
        assert main.get_user_message() == expected

    def test_excludes_standard_exceptions(self):
        main = SimpleError(reason="server error")
        main.add_related_error(ValueError("internal"))
        main.add_related_error(RelatedError(detail="user-facing"))

        expected = """Error: server error

Related errors:
  - Related: user-facing"""
        assert main.get_user_message() == expected


class TestSerialization:
    def test_serialize_simple(self):
        exc = SimpleError(reason="not found")
        data = exc.to_dict()

        assert data['type'] == 'SimpleError'
        assert data['message'] == "Error: not found"
        assert data['kwargs'] == {'reason': 'not found'}
        assert data['http_code'] == 422
        assert data['non_retryable'] is True

    def test_deserialize_simple(self):
        original = SimpleError(reason="not found")
        data = original.to_dict()
        restored = PublicException.from_dict(data)

        assert restored.message == original.message
        assert restored.kwargs == original.kwargs

    def test_roundtrip_with_related_errors(self):
        main = SimpleError(reason="server error")
        main.add_related_error(RelatedError(detail="timeout"))
        main.add_related_error(ValueError("validation"))

        data = main.to_dict()
        restored = PublicException.from_dict(data)

        assert restored.message == main.message
        assert len(restored.related_errors) == 1
        first_error = restored.related_errors[0]
        assert isinstance(first_error, PublicException)
        assert first_error.message == "Related: timeout"
        assert 'validation' not in str(data)

    def test_standard_related_error_never_enters_temporal_payload(self):
        canary = 'provider-body password=related-error-secret-canary'
        main = SimpleError(reason='server error')
        main.add_related_error(ValueError(canary))

        data = main.to_dict()

        assert data['related_errors'] == []
        assert canary not in str(data)

    def test_nested_serialization(self):
        level2 = RelatedError(detail="nested")
        level1 = SimpleError(reason="server error")
        level1.add_related_error(level2)

        data = level1.to_dict()
        assert len(data['related_errors']) == 1
        assert data['related_errors'][0]['type'] == 'RelatedError'
        assert data['related_errors'][0]['message'] == 'Related: nested'


class TestRealWorldScenarios:
    def test_parallel_failures(self):
        """Multiple items failing in parallel operation"""
        class BatchError(PublicException):
            template = "Failed to process batch"

        batch = BatchError()
        batch.add_related_error(SimpleError(reason="item 1 failed"))
        batch.add_related_error(SimpleError(reason="item 2 failed"))
        batch.add_related_error(ValueError("internal database error"))

        expected_user = """Failed to process batch

Related errors:
  - Error: item 1 failed
  - Error: item 2 failed"""
        assert batch.get_user_message() == expected_user

    def test_temporal_workflow_serialization(self):
        """Verify serialization works for Temporal ApplicationError details"""
        class WorkflowError(PublicException):
            template = "Workflow failed: {reason}"
            non_retryable = False
            http_code = 500

        exc = WorkflowError(reason="timeout")
        exc.add_related_error(RelatedError(detail="activity failed"))

        details = exc.to_dict()
        restored = PublicException.from_dict(details)

        assert restored.message == exc.message
        assert restored.non_retryable is False
        assert restored.http_code == 500
        assert len(restored.related_errors) == 1
