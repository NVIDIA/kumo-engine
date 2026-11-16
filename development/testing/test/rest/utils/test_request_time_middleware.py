# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from rest.utils.request_time_middleware import SensitiveQueryLogFilter


def test_uvicorn_access_filter_redacts_query_credentials() -> None:
    query_secret = 'known-query-api-key-material'
    record = logging.LogRecord(
        name='uvicorn.access',
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=(
            '127.0.0.1:1234',
            'GET',
            f'/v1/projects?api_key={query_secret}&limit=10',
            '1.1',
            401,
        ),
        exc_info=None,
    )

    assert SensitiveQueryLogFilter().filter(record)

    message = record.getMessage()
    assert query_secret not in message
    assert 'redacted_query=%3Credacted%3E' in message
    assert 'limit=10' in message


def test_uvicorn_access_filter_does_not_inspect_or_log_headers() -> None:
    header_secret = 'known-header-api-key-material'
    record = logging.LogRecord(
        name='uvicorn.access',
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=('127.0.0.1:1234', 'GET', '/v1/projects', '1.1', 200),
        exc_info=None,
    )

    assert SensitiveQueryLogFilter().filter(record)
    assert header_secret not in record.getMessage()
