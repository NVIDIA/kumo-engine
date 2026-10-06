# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cursor stand-in every SQL warehouse's unit tests can record against."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class RecordingCursor:
    """Records every statement and replays rows matched by SQL substring."""
    def __init__(self, responses: Mapping[str, Sequence[Any]] | None = None,
                 raises: Mapping[str, BaseException] | None = None) -> None:
        self.statements: list[tuple[str, Any]] = []
        self.responses = dict(responses or {})
        self.raises = dict(raises or {})
        self._result: Sequence[Any] = ()

    def execute(self, sql: str, params: Any = None) -> "RecordingCursor":
        self.statements.append((sql, params))
        for marker, error in self.raises.items():
            if marker in sql:
                raise error
        self._result = ()
        for marker, rows in self.responses.items():
            if marker in sql:
                self._result = rows
                break
        return self

    def fetchall(self) -> list[Any]:
        return list(self._result)

    def close(self) -> None:
        return None

    def sql(self) -> list[str]:
        return [statement for statement, _ in self.statements]

    def matching(self, marker: str) -> list[str]:
        return [statement for statement in self.sql() if marker in statement]


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self.cursor_object = cursor

    def cursor(self) -> RecordingCursor:
        return self.cursor_object
