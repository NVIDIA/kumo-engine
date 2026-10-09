# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for a Snowflake cursor and an HTTP session, so units need no network."""

from __future__ import annotations

from threading import Barrier
from typing import Any, Mapping

from sources.recording import RecordingConnection, RecordingCursor

FakeConnection = RecordingConnection


class FakeColumn:
    """One column of a compiled statement, as ``describe`` reports it."""
    def __init__(self, name: str, type_name: str) -> None:
        from snowflake.connector.constants import FIELD_NAME_TO_ID
        self.name = name
        self.type_code = FIELD_NAME_TO_ID[type_name]


class FakeCursor(RecordingCursor):
    def __init__(self, responses=None, raises=None,
                 declared: Mapping[str, str] | None = None) -> None:
        super().__init__(responses, raises)
        self.description = [
            FakeColumn(name, type_name)
            for name, type_name in (declared or {}).items()
        ]

    def describe(self, sql: str) -> None:
        self.execute(sql)


class FakeResponse:
    def __init__(self, body: bytes,
                 raises: BaseException | None = None) -> None:
        self.body = body
        self.raises = raises

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exception: Any) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self.raises is not None:
            raise self.raises

    def iter_content(self, chunk_size: int = 1) -> Any:
        for start in range(0, len(self.body), max(chunk_size, 1)):
            yield self.body[start:start + chunk_size]


class FakeSession:
    """Serves a canned body per url, failing or refusing on demand."""
    def __init__(self, bodies: Mapping[str, bytes],
                 raises: BaseException | None = None,
                 refuses: BaseException | None = None,
                 arrivals: Barrier | None = None) -> None:
        self.bodies = dict(bodies)
        self.raises = raises
        self.refuses = refuses
        self.arrivals = arrivals
        self.closed = False
        self.adapters = {}

    def mount(self, prefix: str, adapter: Any) -> None:
        self.adapters[prefix] = adapter

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if self.arrivals is not None:
            self.arrivals.wait()
        if self.raises is not None:
            raise self.raises
        return FakeResponse(self.bodies.get(url, b""), self.refuses)

    def close(self) -> None:
        self.closed = True


class FakeSessions:
    """Stands in for ``requests.Session`` itself, so a test sees every session
    a read opened and whether that read gave it back."""
    def __init__(self, bodies: Mapping[str, bytes],
                 arrivals: Barrier | None = None) -> None:
        self.bodies = dict(bodies)
        self.arrivals = arrivals
        self.raises: BaseException | None = None
        self.refuses: BaseException | None = None
        self.opened: list[FakeSession] = []

    def __call__(self) -> FakeSession:
        session = FakeSession(self.bodies, self.raises, self.refuses,
                              self.arrivals)
        self.opened.append(session)
        return session

    @property
    def released(self) -> bool:
        return all(session.closed for session in self.opened)
