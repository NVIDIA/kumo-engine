# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON log formatter utilities."""

import json
import logging
from datetime import datetime, timezone

_LOG_RECORD_BASE_FIELDS = frozenset(logging.makeLogRecord({}).__dict__.keys())


class JsonLogFormatter(logging.Formatter):
    """Serialize log records as one-line JSON objects."""
    def formatTime(self, record: logging.LogRecord,
                   datefmt: str | None = None) -> str:
        if datefmt is not None:
            return super().formatTime(record, datefmt)
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return timestamp.isoformat(timespec='milliseconds')

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'filename': record.filename,
            'lineno': record.lineno,
            'function': record.funcName,
            'process_id': record.process,
            'process_name': record.processName,
            'thread_id': record.thread,
            'thread_name': record.threadName,
        }

        if record.exc_info:
            payload['exception'] = self.formatException(record.exc_info)
        if record.stack_info:
            payload['stack'] = self.formatStack(record.stack_info)

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _LOG_RECORD_BASE_FIELDS and key != 'message'
        }
        if extra:
            payload['extra'] = extra

        return json.dumps(payload, default=str)
