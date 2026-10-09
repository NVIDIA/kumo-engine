# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging as py_logging
import sys

import pytest

from io_utils import logging as kumo_logging
from io_utils.logging_json import JsonLogFormatter
from service_framework._raw_deploy_config import LoggingOutputFormat


def _build_config(output_format: LoggingOutputFormat):
    original = kumo_logging._get_output_format
    kumo_logging._get_output_format = lambda: output_format
    try:
        return kumo_logging._get_config(
            mode=kumo_logging.LoggingMode.DEVELOPMENT,
            log_to_console=True,
            log_to_file=False,
            log_to_cloud_watch=False,
            log_process_id=False,
            logging_level_overrides={},
        )
    finally:
        kumo_logging._get_output_format = original


def test_get_config_uses_json_formatter():
    config = _build_config(LoggingOutputFormat.JSON)
    assert 'json' in config.formatters
    assert (config.formatters['json'].cls ==
            'io_utils.logging_json.JsonLogFormatter')
    assert config.handlers['console'].formatter == 'json'
    assert config.handlers['error_console'].formatter == 'json'


def test_get_config_uses_text_formatter():
    config = _build_config(LoggingOutputFormat.TEXT)
    assert 'standard' in config.formatters
    assert 'error' in config.formatters
    assert config.handlers['console'].formatter == 'standard'
    assert config.handlers['error_console'].formatter == 'error'


def test_loki_export_requires_a_configured_url(monkeypatch):
    monkeypatch.setattr(kumo_logging, '_LOKI_EXPORT', True)
    monkeypatch.setattr(kumo_logging, '_LOKI_URL', '')

    with pytest.raises(ValueError, match='LOKI_URL must be set'):
        _build_config(LoggingOutputFormat.TEXT)


def test_loki_export_uses_the_configured_url(monkeypatch):
    url = 'https://logs.example.test/loki/api/v1/push'
    monkeypatch.setattr(kumo_logging, '_LOKI_EXPORT', True)
    monkeypatch.setattr(kumo_logging, '_LOKI_URL', url)
    monkeypatch.setattr(kumo_logging, '_LOKI_TAGS', '{}')

    config = _build_config(LoggingOutputFormat.TEXT)

    assert config.handlers['loki'].url == url


def test_pytest_fsutil_debug_logs_disabled_overrides_fsutil_logger(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(kumo_logging._PYTEST_ENABLE_FSUTIL_DEBUG_LOGS_ENV, '0')

    config = kumo_logging._get_config(
        mode=kumo_logging.LoggingMode.DEBUG,
        log_to_console=True,
        log_to_file=False,
        log_to_cloud_watch=False,
        log_process_id=False,
        logging_level_overrides=(
            kumo_logging._with_pytest_logging_level_overrides({})),
    )

    assert config.loggers['io_utils'].level == 'DEBUG'
    assert config.loggers['io_utils.fsutil'].level == 'INFO'


def test_pytest_fsutil_debug_logs_enabled_keeps_fsutil_debuggable(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(kumo_logging._PYTEST_ENABLE_FSUTIL_DEBUG_LOGS_ENV, '1')

    config = kumo_logging._get_config(
        mode=kumo_logging.LoggingMode.DEBUG,
        log_to_console=True,
        log_to_file=False,
        log_to_cloud_watch=False,
        log_process_id=False,
        logging_level_overrides=(
            kumo_logging._with_pytest_logging_level_overrides({})),
    )

    assert config.loggers['io_utils'].level == 'DEBUG'
    assert 'io_utils.fsutil' not in config.loggers


def test_json_formatter_emits_json():
    formatter = JsonLogFormatter()
    logger = py_logging.getLogger('test.json')
    record = logger.makeRecord(
        name='test.json',
        level=py_logging.INFO,
        fn='/tmp/f.py',
        lno=12,
        msg='hello %s',
        args=('world', ),
        exc_info=None,
    )
    payload = json.loads(formatter.format(record))
    assert payload['message'] == 'hello world'
    assert payload['level'] == 'INFO'
    assert payload['logger'] == 'test.json'


def test_json_formatter_includes_exception():
    formatter = JsonLogFormatter()
    logger = py_logging.getLogger('test.json.exception')
    try:
        raise ValueError('boom')
    except ValueError:
        record = logger.makeRecord(
            name='test.json.exception',
            level=py_logging.ERROR,
            fn='/tmp/f.py',
            lno=99,
            msg='failed',
            args=(),
            exc_info=sys.exc_info(),
        )

    payload = json.loads(formatter.format(record))
    assert 'exception' in payload
    assert 'ValueError: boom' in payload['exception']


def test_json_formatter_single_line_with_caplog(
        caplog: pytest.LogCaptureFixture):
    formatter = JsonLogFormatter()
    logger = py_logging.getLogger('test.json.caplog')

    with caplog.at_level(py_logging.INFO, logger='test.json.caplog'):
        logger.info('line1\nline2')
        try:
            raise RuntimeError('boom')
        except RuntimeError:
            logger.exception('failure with traceback')

    assert len(caplog.records) >= 2
    for record in caplog.records:
        formatted = formatter.format(record)
        # One JSON object must stay on one physical line.
        assert '\n' not in formatted
        assert '\r' not in formatted
        json.loads(formatted)


def test_json_line_round_trip_matches_text_line():
    logger = py_logging.getLogger('test.json.roundtrip')
    record = logger.makeRecord(
        name='test.json.roundtrip',
        level=py_logging.WARNING,
        fn='/tmp/roundtrip_file.py',
        lno=321,
        msg='item=%s count=%d',
        args=('abc', 42),
        exc_info=None,
        func='roundtrip_func',
    )

    text_formatter = py_logging.Formatter(
        '%(levelname)s|%(name)s|%(module)s|%(filename)s|%(lineno)d|'
        '%(funcName)s|%(process)d|%(processName)s|%(thread)d|'
        '%(threadName)s|%(message)s')
    expected_text_line = text_formatter.format(record)

    json_formatter = JsonLogFormatter()
    json_line = json_formatter.format(record)
    payload = json.loads(json_line)

    round_trip_text_line = (
        f"{payload['level']}|{payload['logger']}|{payload['module']}|"
        f"{payload['filename']}|{payload['lineno']}|{payload['function']}|"
        f"{payload['process_id']}|{payload['process_name']}|"
        f"{payload['thread_id']}|{payload['thread_name']}|"
        f"{payload['message']}")
    assert round_trip_text_line == expected_text_line
