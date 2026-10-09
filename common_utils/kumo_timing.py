# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import logging
import multiprocessing
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

_ENV_ENABLED = "KUMO_TIMING"
_ENV_TIMING_FILE = "KUMO_TIMING_FILE"
_MARKER = "KUMO_TIMING"

_REDACTED = "<redacted>"
_SENSITIVE_KEY_PARTS = (
    "access_key",
    "api_key",
    "authorization",
    "cookie",
    "password",
    "private_key",
    "secret",
    "token",
)
_LOCATION_KEY_PARTS = (
    "path",
    "uri",
    "url",
)


def kumo_timing_enabled() -> bool:
    return os.environ.get(_ENV_ENABLED, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _timestamp_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _safe_value(key: str, value: Any) -> Any:
    key_lower = key.lower()
    if any(part in key_lower for part in _SENSITIVE_KEY_PARTS):
        return _REDACTED
    if any(part in key_lower for part in _LOCATION_KEY_PARTS):
        return _REDACTED
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(child_key): _safe_value(str(child_key), child_value)
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(key, item) for item in value]
    return repr(value)


def _safe_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _safe_value(key, value)
        for key, value in fields.items()
        if value is not None
    }


def _base_record(name: str, **fields: Any) -> dict[str, Any]:
    record_fields = dict(fields)
    status = record_fields.pop("status", "event")
    elapsed_seconds = record_fields.pop("elapsed_seconds", None)
    error_type = record_fields.pop("error_type", None)
    record = {
        "event": "kumo_timing",
        "fields": _safe_fields(record_fields),
        "name": name,
        "pid": os.getpid(),
        "process_name": multiprocessing.current_process().name,
        "status": status,
        "thread_id": threading.get_ident(),
        "timestamp_utc": _timestamp_utc(),
    }
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = elapsed_seconds
    if error_type is not None:
        record["error_type"] = error_type
    return record


def _write_timing_record(record: dict[str, Any]) -> None:
    timing_file = os.environ.get(_ENV_TIMING_FILE)
    if not timing_file:
        return
    try:
        line = json.dumps(record, sort_keys=True) + "\n"
        with Path(timing_file).open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        log.exception("%s failed to write timing file", _MARKER)


def kumo_timing_event(name: str, **fields: Any) -> None:
    if not kumo_timing_enabled():
        return
    record = _base_record(name, **fields)
    log.info("%s %s", _MARKER, json.dumps(record, sort_keys=True))
    _write_timing_record(record)


def record_event(name: str, **fields: Any) -> None:
    kumo_timing_event(name, **fields)


@contextmanager
def kumo_timing(name: str, **fields: Any) -> Iterator[None]:
    if not kumo_timing_enabled():
        yield
        return
    started = time.perf_counter()
    kumo_timing_event(f"{name}.start", status="start", **fields)
    try:
        yield
    except BaseException as exc:
        kumo_timing_event(
            f"{name}.failed",
            elapsed_seconds=round(time.perf_counter() - started, 3),
            error_type=exc.__class__.__name__,
            status="failed",
            **fields,
        )
        raise
    else:
        kumo_timing_event(
            f"{name}.done",
            elapsed_seconds=round(time.perf_counter() - started, 3),
            status="done",
            **fields,
        )
