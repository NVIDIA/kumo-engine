# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import http
import logging as std_logging
import time
from datetime import timedelta
from logging import ERROR, INFO
from typing import List
from urllib.parse import parse_qsl, urlencode

from asgiref.compatibility import guarantee_single_callable
from fastapi import Request, Response
from opentelemetry.trace import format_trace_id
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)

from io_utils import logging
from rest.utils.tracing import get_span_context

log = logging.getLogger(__name__)

_SENSITIVE_QUERY_KEY_PARTS = (
    "api_key",
    "authorization",
    "client_secret",
    "code",
    "cookie",
    "password",
    "refresh_token",
    "reset",
    "secret",
    "session",
    "state",
    "token",
)


def _is_sensitive_query_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_QUERY_KEY_PARTS)


def _redacted_query_string(request: Request) -> str:
    return _redact_query_string(request.url.query)


def _redact_query_string(query: str) -> str:
    redacted_params = []
    for key, value in parse_qsl(query, keep_blank_values=True):
        redacted_params.append(
            ("redacted_query", "<redacted>")
            if _is_sensitive_query_key(key) else (key, value))
    return urlencode(redacted_params)


class SensitiveQueryLogFilter(std_logging.Filter):
    """Redact credential-like query values from Uvicorn access records."""
    def filter(self, record: std_logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(
                args[2], str):
            path, separator, query = args[2].partition('?')
            if separator:
                redacted_args = list(args)
                redacted_args[2] = f'{path}?{_redact_query_string(query)}'
                record.args = tuple(redacted_args)
        return True


class RequestTimeMiddleware(BaseHTTPMiddleware):
    """The ASGI application middleware.

    This class is an ASGI middleware that logs information about all request
    that take longer than X seconds.

    Args:
        app: The ASGI application callable to forward requests to.
        threshold: Number of seconds for a request to be considered "slow"
        exclude_prefixes: HTTP routes to exclude from slow-request logging
    """
    def __init__(
        self,
        app,
        exclude_prefixes: List[str] = [],
        threshold_sec: float = 10,
    ):
        super().__init__(guarantee_single_callable(app))
        self._exclude_prefixes = exclude_prefixes
        self._threshold = timedelta(seconds=threshold_sec)

    async def dispatch(self, request: Request,
                       call_next: RequestResponseEndpoint) -> Response:
        if any(request.url.path.startswith(p) for p in self._exclude_prefixes):
            return await call_next(request)

        host = getattr(getattr(request, "client", None), "host", None)
        port = getattr(getattr(request, "client", None), "port", None)
        url = (f"{request.url.path}?{_redacted_query_string(request)}"
               if request.query_params else request.url.path)
        status_phrase = http.HTTPStatus.INTERNAL_SERVER_ERROR.phrase
        status_code = 500

        start = time.perf_counter()
        try:
            resp = await call_next(request)
            status_code = resp.status_code
            status_phrase = http.HTTPStatus(resp.status_code).phrase
            return resp
        finally:
            full_path = (
                f'{host}:{port} - "{request.method} {url}" {status_code} '
                f'{status_phrase}')
            self._print_request_time_log(full_path, start, status_code)

    def _print_request_time_log(self, full_path: str, start: float,
                                status_code: int) -> None:
        # Calculate duration and compare to threshold
        end = time.perf_counter()
        duration = end - start
        formatted_duration = "{0:.2f}".format(duration)
        if duration < self._threshold.total_seconds():
            log.log(INFO if status_code < 400 else ERROR,
                    f"{full_path}: {formatted_duration}s")
            return

        trace_id = format_trace_id(get_span_context().trace_id)
        log.error(f"Slow Request! {full_path}: {formatted_duration}s, "
                  f"Trace ID {trace_id}")
