# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PublicException integration with TrainStatus and PredStatus."""
import subprocess
import sys
import tempfile

import pytest
from kumo_core.job_config import PredStatus, TrainStatus

from common_utils.error import PublicException


class TestError(PublicException):
    template = "Test error: {reason}"
    http_code = 400


class RelatedError(PublicException):
    template = "Related issue: {detail}"


class TestTrainStatusPublicError:
    def test_save_and_load_with_error(self):
        """Test that error_msg is preserved through save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a PublicException
            exc = TestError(reason="validation failed")
            exc.add_related_error(RelatedError(detail="missing field"))

            # Create status with the exception
            status = TrainStatus()
            status.error_msg = exc.get_user_message()
            status.stacktrace = "Mock stacktrace"

            # Save and reload
            status.save(tmpdir)
            loaded_status = TrainStatus.load(tmpdir)

            # Verify error_msg is preserved
            assert loaded_status.error_msg is not None
            assert "Test error: validation failed" in loaded_status.error_msg
            assert "missing field" in loaded_status.error_msg
            assert loaded_status.stacktrace == "Mock stacktrace"


class TestPredStatusPublicError:
    def test_save_and_load_with_error(self):
        """Test that error_msg is preserved through save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a PublicException
            exc = TestError(reason="invalid input")

            # Create status with the exception
            status = PredStatus()
            status.error_msg = exc.get_user_message()
            status.stacktrace = "Mock stacktrace"

            # Save and reload
            status.save(tmpdir)
            loaded_status = PredStatus.load(tmpdir)

            # Verify error_msg is preserved
            assert loaded_status.error_msg is not None
            assert "Test error: invalid input" in loaded_status.error_msg
            assert loaded_status.stacktrace == "Mock stacktrace"


class TestWorkerExceptionPattern:
    """Test the pattern used in worker exception handlers."""
    def test_training_worker_exception_pattern(self):
        """Test the exception handling pattern from kumo-auto."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate what kumo-auto does when catching PublicException
            try:
                raise TestError(reason="GPU memory error")
            except Exception as e:
                status = TrainStatus()

                # This is the pattern from kumo-auto
                if isinstance(e, PublicException):
                    status.error_msg = e.get_user_message()
                    status.allow_k8s_retry = not e.non_retryable
                else:
                    status.error_msg = str(e)

                status.save(tmpdir)

            # Verify it worked correctly
            loaded_status = TrainStatus.load(tmpdir)
            assert loaded_status.error_msg is not None
            assert loaded_status.allow_k8s_retry is False  # TestError.non_retryable=True
            assert "GPU memory error" in loaded_status.error_msg

    def test_prediction_worker_exception_pattern(self):
        """Test the exception handling pattern from kumo-pred."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate what kumo-pred does when catching PublicException
            try:
                raise TestError(reason="input validation failed")
            except Exception as e:
                pred_status = PredStatus()

                # This is the pattern from kumo-pred
                if isinstance(e, PublicException):
                    pred_status.error_msg = e.get_user_message()
                else:
                    pred_status.error_msg = str(e)

                pred_status.save(tmpdir)

            # Verify it worked correctly
            loaded_status = PredStatus.load(tmpdir)
            assert loaded_status.error_msg is not None
            assert "input validation failed" in loaded_status.error_msg

    def test_non_public_exception_pattern(self):
        """Test that non-PublicException errors are handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                raise ValueError("Internal error")
            except Exception as e:
                status = TrainStatus()

                if isinstance(e, PublicException):
                    status.error_msg = e.get_user_message()
                else:
                    status.error_msg = str(e)

                status.save(tmpdir)

            # Verify non-PublicException sets error_msg correctly
            loaded_status = TrainStatus.load(tmpdir)
            assert loaded_status.error_msg == "Internal error"


class TestOOMErrorMessagePreservation:
    """Test that the outer kumo-auto handler doesn't wipe error_msg set by
    auto_trainer.run(). Regression test for #26704."""
    @staticmethod
    def _run_kumo_auto_handler(tmpdir, exception):
        """Replicates the except-Exception handler from bin/kumo-auto
        (lines 229-275) so we can test it in isolation."""
        import traceback

        stacktrace = ''.join(
            traceback.format_exception(exception, exception,
                                       exception.__traceback__))

        status = TrainStatus.load(tmpdir)

        if status.stacktrace is None:
            status.stacktrace = stacktrace

        if isinstance(exception, PublicException):
            status.error_msg = exception.get_user_message()
            status.allow_k8s_retry = not exception.non_retryable
        elif "Unable to locate credentials" in str(stacktrace):
            status.allow_k8s_retry = True
        elif "out of memory" in stacktrace.lower():
            status.allow_k8s_retry = False
            status.error_msg = (
                "Out-of-memory (OOM) error: The training job ran out of "
                "memory. To resolve this issue, consider:\n"
                "- Reducing the batch size in your training "
                "configuration\n"
                "- Reducing the number of sampled neighbors "
                "(num_neighbors)\n"
                "- Excluding high-cardinality columns from the graph\n\n"
                f"Error: {exception}")
        elif "Failed to initialize NVML: Unknown Error" in str(stacktrace):
            status.allow_k8s_retry = True
        else:
            status.allow_k8s_retry = False
            if 'message larger than max' in str(exception):
                status.error_msg = str(exception)

        status.save(tmpdir)
        return status

    def test_generic_reraise_does_not_wipe_oom_message(self):
        """Reproduces #26704: auto_trainer.run() saves detailed OOM message,
        then raises RuntimeError("Failed to run any AutoML trial").
        The outer handler must NOT overwrite it with ""."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: auto_trainer.run() saves detailed OOM message
            status = TrainStatus()
            status.error_msg = (
                "Out of 2 experiment(s), 2 experiment(s) failed due to "
                "GPU out-of-memory. Encoder: 13056 parameters (50MB)")
            status.save(tmpdir)

            # Step 2: auto_trainer.run() raises generic RuntimeError
            try:
                raise RuntimeError("Failed to run any AutoML trial")
            except Exception as e:
                self._run_kumo_auto_handler(tmpdir, e)

            # Verify: detailed message must survive
            loaded = TrainStatus.load(tmpdir)
            assert "GPU out-of-memory" in loaded.error_msg
            assert "13056 parameters" in loaded.error_msg
            assert loaded.error_msg != ""
            assert loaded.allow_k8s_retry is False

    def test_direct_oom_without_prior_message_gets_generic(self):
        """When OOM text IS in the stacktrace and no prior message exists,
        the generic OOM message should be set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            status = TrainStatus()
            status.save(tmpdir)

            try:
                raise RuntimeError("CUDA out of memory. Tried to allocate "
                                   "51.42 GiB")
            except Exception as e:
                self._run_kumo_auto_handler(tmpdir, e)

            loaded = TrainStatus.load(tmpdir)
            assert "Out-of-memory (OOM) error" in loaded.error_msg
            assert loaded.allow_k8s_retry is False


class TestRetryLogic:
    """Test retry logic based on PublicException.non_retryable."""
    def test_retryable_exception(self):
        """Test that retryable exceptions set allow_k8s_retry correctly."""
        class RetryableError(PublicException):
            template = "Retryable: {reason}"
            non_retryable = False  # This should be retried

        with tempfile.TemporaryDirectory() as tmpdir:
            exc = RetryableError(reason="temporary network issue")

            status = TrainStatus()
            status.error_msg = exc.get_user_message()
            status.allow_k8s_retry = not exc.non_retryable

            status.save(tmpdir)
            loaded_status = TrainStatus.load(tmpdir)

            # Should allow retry
            assert loaded_status.allow_k8s_retry is True
            assert loaded_status.error_msg is not None
            assert "temporary network issue" in loaded_status.error_msg

    def test_non_retryable_exception(self):
        """Test that non-retryable exceptions set allow_k8s_retry correctly."""
        class NonRetryableError(PublicException):
            template = "Non-retryable: {reason}"
            non_retryable = True

        with tempfile.TemporaryDirectory() as tmpdir:
            exc = NonRetryableError(reason="invalid configuration")

            status = TrainStatus()
            status.error_msg = exc.get_user_message()
            status.allow_k8s_retry = not exc.non_retryable

            status.save(tmpdir)
            loaded_status = TrainStatus.load(tmpdir)

            # Should NOT allow retry
            assert loaded_status.allow_k8s_retry is False
            assert loaded_status.error_msg is not None
            assert "invalid configuration" in loaded_status.error_msg


# Mirrors the entry-point wrapper in bin/kumo-auto and bin/kumo-pred. The
# non-daemon thread models the diskgraph sampler's native thread, which joins
# forever at interpreter shutdown on py3.14; os._exit forces a clean exit on
# both the success and error paths.
_ENTRYPOINT_WRAPPER = """
import os
import sys
import threading
import time
import traceback


def main():
    threading.Thread(target=lambda: time.sleep(3600), daemon=False).start()
    if {raise_in_main}:
        raise RuntimeError("boom")


if __name__ == '__main__':
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (
            0 if exc.code is None else 1)
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
"""


class TestEntrypointForcesCleanExit:
    """The entry-point wrapper must force the process to exit even when a
    lingering non-daemon thread would otherwise hang interpreter shutdown,
    leaving the parent ``train`` activity's ``proc.wait()`` blocked forever.
    """
    def _run(self, raise_in_main: bool) -> subprocess.CompletedProcess:
        script = _ENTRYPOINT_WRAPPER.format(raise_in_main=str(raise_in_main))
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(script)
            path = f.name
        # Without os._exit the thread hangs shutdown; the timeout turns that
        # regression into a failure.
        return subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=30)

    def test_exits_promptly_on_success(self):
        """main() returns -> process exits 0 promptly, not hung."""
        try:
            result = self._run(raise_in_main=False)
        except subprocess.TimeoutExpired:
            pytest.fail("entry point hung at interpreter shutdown instead of "
                        "exiting via os._exit")
        assert result.returncode == 0

    def test_exits_nonzero_on_failure(self):
        """A failing main() still force-exits non-zero so the activity reports
        FAILED."""
        try:
            result = self._run(raise_in_main=True)
        except subprocess.TimeoutExpired:
            pytest.fail("entry point hung at interpreter shutdown on the "
                        "failure path")
        assert result.returncode == 1
