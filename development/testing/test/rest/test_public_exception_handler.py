# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for PublicException exception handler in the FastAPI app.
"""
from fastapi.testclient import TestClient
from kumo_core.errors import PublicError, TrainingValidationError
from kumo_core.public_error import PublicTrainingError

from common_utils.error import PublicException


class SampleClientError(PublicException):
    template = "Test error: {reason}"
    http_code = 400
    non_retryable = True


class SampleServerError(PublicException):
    template = "Server error occurred: {message}"
    http_code = 500
    non_retryable = False


class TestPublicExceptionHandler:
    """Test the PublicException handler in the FastAPI app"""
    def test_public_exception_handler_basic(self, client: TestClient):
        """Test that PublicException is handled with correct status code and message"""
        # Create a test endpoint that raises a PublicException
        @client.app.get("/test/public-exception")
        async def test_endpoint():
            raise SampleClientError(reason="something went wrong")

        response = client.get("/test/public-exception")

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "Test error: something went wrong"
        assert data["error_type"] == "SampleClientError"

    def test_public_exception_handler_with_custom_http_code(
            self, client: TestClient):
        """Test that custom HTTP codes are respected"""
        @client.app.get("/test/server-error")
        async def test_endpoint():
            raise SampleServerError(message="database connection failed")

        response = client.get("/test/server-error")

        assert response.status_code == 500
        data = response.json()
        assert data[
            "error"] == "Server error occurred: database connection failed"
        assert data["error_type"] == "SampleServerError"

    def test_public_exception_handler_with_related_errors(
            self, client: TestClient):
        """Test that related errors are included in user message"""
        class RelatedError(PublicException):
            template = "Related issue: {detail}"

        @client.app.get("/test/with-related")
        async def test_endpoint():
            main_error = SampleClientError(reason="batch processing failed")
            main_error.add_related_error(RelatedError(detail="item 1 failed"))
            main_error.add_related_error(RelatedError(detail="item 2 failed"))
            # Internal errors should not be in user message
            main_error.add_related_error(ValueError("internal error"))
            raise main_error

        response = client.get("/test/with-related")

        assert response.status_code == 400
        data = response.json()

        # Check that user message includes related PublicExceptions
        expected_message = """Test error: batch processing failed

Related errors:
  - Related issue: item 1 failed
  - Related issue: item 2 failed"""
        assert data["error"] == expected_message
        assert data["error_type"] == "SampleClientError"

    def test_public_exception_handler_with_missing_template_key(
            self, client: TestClient):
        """Test that missing template keys are handled gracefully"""
        @client.app.get("/test/missing-key")
        async def test_endpoint():
            # Missing required 'reason' parameter
            raise SampleClientError()

        response = client.get("/test/missing-key")

        assert response.status_code == 400
        data = response.json()
        assert "[Missing key: reason]" in data["error"]

    def test_public_exception_does_not_affect_other_endpoints(
            self, client: TestClient):
        """Test that the handler doesn't break existing endpoints"""
        # Test that normal endpoints still work
        response = client.get("/")
        assert response.status_code == 200

    def test_public_exception_vs_validation_error(self, client: TestClient):
        """Test that PublicException handler doesn't interfere with validation errors"""
        # Validation errors should still work as before
        # This is tested implicitly by the existing test suite, but we can
        # verify the handler is registered correctly
        handlers = client.app.exception_handlers
        assert PublicException in handlers

    def test_core_validation_error_keeps_legacy_response(
            self, client: TestClient):
        @client.app.get('/test/core-validation-error')
        async def test_endpoint():
            raise TrainingValidationError(
                title='GPU out-of-memory error',
                message='Reduce output_embedding_dim.',
            )

        response = client.get('/test/core-validation-error')

        assert response.status_code == 422
        assert response.json() == {
            'error': 'GPU out-of-memory error: Reduce output_embedding_dim.',
            'error_type': 'ValidationException',
        }

    def test_core_public_error_keeps_legacy_response(self, client: TestClient):
        @client.app.get('/test/core-public-error')
        async def test_endpoint():
            raise PublicTrainingError('Training configuration is invalid.')

        response = client.get('/test/core-public-error')

        assert response.status_code == 422
        assert response.json() == {
            'error': 'Training configuration is invalid.',
            'error_type': 'PublicException',
        }


class TestPublicAppOptimisticLockHandler:
    """OptimisticLockError is NOT a PublicException, so the public (SDK) app
    must register a dedicated handler or a concurrent-write conflict from an
    in-process metadata driver surfaces as a generic 500 instead of 409."""
    def test_optimistic_lock_handler_registered(self,
                                                public_api_client: TestClient):
        from rest.deps.db_common import OptimisticLockError
        assert OptimisticLockError in public_api_client.app.exception_handlers

    def test_core_error_handlers_registered(self,
                                            public_api_client: TestClient):
        handlers = public_api_client.app.exception_handlers
        assert TrainingValidationError in handlers
        assert PublicError in handlers

    def test_optimistic_lock_returns_409(self, public_api_client: TestClient):
        from rest.deps.db_common import OptimisticLockError

        @public_api_client.app.get('/test/optimistic-lock')
        async def _raise_optimistic_lock():
            raise OptimisticLockError('row changed concurrently')

        response = public_api_client.get('/test/optimistic-lock')

        assert response.status_code == 409
        data = response.json()
        assert data['error_type'] == 'OptimisticLockError'
        assert data['retry'] is True
