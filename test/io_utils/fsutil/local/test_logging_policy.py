# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

from common_utils.error_diagnostics import (
    exception_message_should_be_logged_at_error,
    safe_exception_message_for_log,
)
from io_utils import fsutil
from io_utils.fsutil.logging_policy import logging_policy


@contextmanager
def _isolated_capture(logger: logging.Logger):
    # Attach a private handler and turn off propagation so the test does not
    # depend on pytest's caplog/root-logger level — caplog.at_level(level,
    # logger=...) overrides the very logger.setLevel() under test.
    captured: list[logging.LogRecord] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _CaptureHandler(level=logging.NOTSET)
    prior_propagate = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        yield captured
    finally:
        logger.removeHandler(handler)
        logger.propagate = prior_propagate


def test_logging_policy_honors_logger_level():
    logger = logging.getLogger(
        'test_logging_policy_honors_logger_level.fsutil_module')
    policy = logging_policy(logger)

    @policy()
    def op():
        return 'ok'

    with _isolated_capture(logger) as captured:
        logger.setLevel(logging.INFO)
        op()
        assert not [r for r in captured if r.levelno == logging.DEBUG], (
            'DEBUG logging_policy records leaked through despite '
            'logger.setLevel(INFO)')

        captured.clear()
        logger.setLevel(logging.DEBUG)
        op()
        assert any(r.levelno == logging.DEBUG
                   and 'success: op' in r.getMessage() for r in captured)


def test_logging_policy_failure_records_emit_at_error_level():
    logger = logging.getLogger(
        'test_logging_policy_failure_records_emit_at_error_level.fsutil_module'
    )
    policy = logging_policy(logger)

    # The failure-emit path only fires when allowed_exceptions is set.
    # ValueError here doesn't match RuntimeError, so the unhandled-exception
    # branch logs at ERROR before re-raising.
    @policy(allowed_exceptions=ValueError)
    def op_fail():
        raise RuntimeError('boom provider-secret-sentinel')

    with _isolated_capture(logger) as captured:
        logger.setLevel(logging.INFO)
        with pytest.raises(RuntimeError, match='boom'):
            op_fail()
        assert any(r.levelno == logging.ERROR
                   and 'failure[RuntimeError]: op_fail' in r.getMessage()
                   for r in captured)
        assert all('provider-secret-sentinel' not in r.getMessage()
                   for r in captured)
        assert all(r.exc_info is None for r in captured)


def test_remote_logging_policy_marks_unexpected_failure_without_changing_type():
    logger = logging.getLogger('test_remote_logging_policy.fsutil_module')
    policy = logging_policy(logger, restricted_failures=True)
    error = RuntimeError('signed-url=provider-secret-sentinel')

    @policy()
    def download():
        raise error

    with pytest.raises(RuntimeError) as raised:
        download()

    assert raised.value is error
    assert not exception_message_should_be_logged_at_error(error)
    assert safe_exception_message_for_log(error) == (
        'Remote filesystem operation download failed')


def test_local_logging_policy_leaves_unexpected_failure_visible():
    logger = logging.getLogger('test_local_logging_policy.fsutil_module')
    policy = logging_policy(logger)
    error = RuntimeError('ordinary Kumo filesystem failure')

    @policy()
    def read_local():
        raise error

    with pytest.raises(RuntimeError) as raised:
        read_local()

    assert raised.value is error
    assert exception_message_should_be_logged_at_error(error)


def test_remote_logging_policy_marks_allowed_path_failure() -> None:
    logger = logging.getLogger('test_remote_allowed_logging_policy.fsutil_module')
    policy = logging_policy(logger, restricted_failures=True)
    error = FileNotFoundError('/customer/private/source/table.parquet')

    @policy(allowed_exceptions=FileNotFoundError)
    def read_remote():
        raise error

    with pytest.raises(FileNotFoundError) as raised:
        read_remote()

    assert raised.value is error
    assert not exception_message_should_be_logged_at_error(error)
    assert safe_exception_message_for_log(error) == (
        'Remote filesystem operation read_remote failed')


@pytest.mark.parametrize('phase', ('enter', 'read', 'exit'))
def test_remote_stream_marks_provider_failure_across_handle_lifecycle(
        phase: str) -> None:
    error = ClientError(
        {
            'Error': {
                'Code': 'AccessDenied',
                'Message': 'signed-url=stream-secret-canary',
            }
        },
        'GetObject',
    )

    class FailingHandle:
        def __enter__(self):
            if phase == 'enter':
                raise error
            return self

        def __exit__(self, *_args):
            if phase == 'exit':
                raise error
            return False

        def read(self):
            if phase == 'read':
                raise error

    with (
            patch('io_utils.fsutil.impl_s3.threadsafe_boto.s3_client'),
            patch('io_utils.fsutil.impl_s3.smart_open',
                  return_value=FailingHandle()),
    ):
        with pytest.raises(ClientError) as raised:
            with fsutil.get_binary_handle('s3://bucket/object', 'rb') as handle:
                handle.read()

    assert raised.value is error
    assert not exception_message_should_be_logged_at_error(error)
    assert safe_exception_message_for_log(error) == (
        'S3 stream operation failed')


def test_remote_stream_does_not_mark_caller_failure() -> None:
    caller_error = ValueError('ordinary Kumo caller failure')

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    with (
            patch('io_utils.fsutil.impl_s3.threadsafe_boto.s3_client'),
            patch('io_utils.fsutil.impl_s3.smart_open', return_value=Handle()),
    ):
        with pytest.raises(ValueError) as raised:
            with fsutil.get_binary_handle('s3://bucket/object', 'rb'):
                raise caller_error

    assert raised.value is caller_error
    assert exception_message_should_be_logged_at_error(caller_error)
