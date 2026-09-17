# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small Snowflake stage store used by submission and remote observation."""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import PurePosixPath
from typing import Any, Mapping

_IDENTIFIER = r'(?:[A-Za-z_][A-Za-z0-9_$]*|"(?:[^"]|"")+")'
_STAGE = re.compile(rf'^@{_IDENTIFIER}\.{_IDENTIFIER}\.{_IDENTIFIER}'
                    r'(?:/[A-Za-z0-9_.+$=-]+)*/?$')
_MAX_STAGE_NAME_LENGTH = 4096


def validate_stage_name(stage_name: str) -> str:
    """Return a normalized, fully-qualified internal-stage path."""
    if (not isinstance(stage_name, str)
            or len(stage_name) > _MAX_STAGE_NAME_LENGTH
            or not _STAGE.fullmatch(stage_name)):
        raise ValueError(
            'stage_name must be a fully-qualified Snowflake stage path such '
            "as '@DB.SCHEMA.KUMO_JOBS'")
    return stage_name.rstrip('/')


def safe_relative_path(path: str) -> str:
    """Reject a relative key that could leave the integration prefix."""
    parsed = PurePosixPath(path)
    if (not path or path.startswith('/') or '\\' in path
            or any(part in {'', '.', '..'} for part in parsed.parts)
            or not re.fullmatch(r'[A-Za-z0-9_.+$/=-]+', path)):
        raise ValueError(f'unsafe Snowflake stage-relative path {path!r}')
    return parsed.as_posix()


def row_dict(row: Any) -> dict[str, Any]:
    """Normalize Snowpark ``Row`` and mapping results."""
    if isinstance(row, Mapping):
        values = row
    else:
        as_dict = getattr(row, 'as_dict', None)
        if not callable(as_dict):
            raise TypeError(
                f'unsupported Snowflake row shape {type(row).__name__!r}')
        values = as_dict()
    return {str(key).lower(): value for key, value in values.items()}


def _file_result(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return {str(key).lower(): value for key, value in result.items()}
    as_dict = getattr(result, 'as_dict', None)
    if callable(as_dict):
        return {str(key).lower(): value for key, value in as_dict().items()}
    return {
        name: getattr(result, name)
        for name in ('status', 'target') if hasattr(result, name)
    }


def _stage_object_missing(error: BaseException) -> bool:
    if getattr(error, 'errno', None) == 250016:
        return True
    return any(fragment in str(error).lower()
               for fragment in ('does not exist', 'not found',
                                'no files matched'))


class StageStore:
    """JSON and file operations below one caller-supplied internal stage."""
    def __init__(self, session: Any, stage_name: str) -> None:
        if session is None or not callable(getattr(session, 'sql', None)):
            raise TypeError('session must be an active Snowpark session')
        self.session = session
        self.stage_name = validate_stage_name(stage_name)

    def uri(self, relative: str) -> str:
        return f'{self.stage_name}/{safe_relative_path(relative)}'

    def put_bytes(self, relative: str, data: bytes) -> None:
        result = self.session.file.put_stream(
            io.BytesIO(data),
            self.uri(relative),
            auto_compress=False,
            overwrite=True,
        )
        if result is None:
            raise RuntimeError(
                f'Snowflake reported no result while writing {relative!r}')
        values = _file_result(result)
        status = str(values.get('status', '')).upper()
        target = str(values.get('target', '')).rsplit('/', 1)[-1]
        expected = PurePosixPath(relative).name
        if status and status not in {'UPLOADED', 'SKIPPED'}:
            raise RuntimeError(
                f'Snowflake did not confirm writing {relative!r}: {values!r}')
        if target and target != expected:
            raise RuntimeError(
                f'Snowflake wrote {target!r}; expected {expected!r}')

    def put_json(self, relative: str, payload: Mapping[str, Any]) -> None:
        data = json.dumps(
            payload,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        ).encode('utf-8')
        self.put_bytes(relative, data)

    def get_bytes(self, relative: str, *,
                  missing_ok: bool = False) -> bytes | None:
        uri = self.uri(relative)
        for attempt in range(3):
            try:
                stream = self.session.file.get_stream(uri, decompress=False)
                try:
                    return stream.read()
                finally:
                    close = getattr(stream, 'close', None)
                    if callable(close):
                        close()
            except Exception as error:
                if missing_ok and _stage_object_missing(error):
                    return None
                if attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))
        return None

    def get_json(self, relative: str, *,
                 missing_ok: bool = False) -> Any | None:
        for attempt in range(3):
            data = self.get_bytes(relative, missing_ok=missing_ok)
            if data is None:
                return None
            try:
                return json.loads(data.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))
        return None

    def list(self, relative: str) -> list[dict[str, Any]]:
        rows = self.session.sql(f'LIST {self.uri(relative)}').collect()
        return [row_dict(row) for row in rows]

    def remove(self, relative: str) -> None:
        self.session.sql(f'REMOVE {self.uri(relative)}').collect()

    def copy_file(self, source_relative: str, target_directory: str) -> None:
        source = PurePosixPath(safe_relative_path(source_relative))
        target = self.uri(target_directory).rstrip('/') + '/'
        filename = source.name.replace("'", "''")
        parent = source.parent.as_posix()
        source_dir = (self.stage_name
                      if parent == '.' else self.uri(parent)).rstrip('/') + '/'
        rows = self.session.sql(
            f"COPY FILES INTO {target} FROM {source_dir} FILES = "
            f"('{filename}') DETAILED_OUTPUT = TRUE").collect()
        copied = [
            str(row_dict(row).get('file', '')).rsplit('/', 1)[-1]
            for row in rows
        ]
        if copied != [source.name]:
            raise RuntimeError(
                f'Snowflake copied {copied!r}; expected {[source.name]!r}')
