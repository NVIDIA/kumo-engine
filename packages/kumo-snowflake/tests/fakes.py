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


class FakeStage:
    """An in-memory named internal stage behind a Snowpark-shaped session.

    LIST and REMOVE match by string prefix, as Snowflake does, and report
    names under the lower-cased stage name. ``ciphertext`` makes LIST report
    padded sizes and presigned fetches return encrypted bytes, as a
    ``SNOWFLAKE_FULL`` stage does.
    """
    def __init__(self, stage: str = '@DB.SCHEMA.KUMO_ARTIFACTS') -> None:
        self.stage = stage
        self.objects: dict[str, bytes] = {}
        self.statements: list[str] = []
        self.puts: list[dict[str, Any]] = []
        self.ranges: list[str] = []
        self.ciphertext = False
        self.ignores_range = False
        self.remove_pattern_ignored = False
        self.gets: list[str] = []

    def _relative(self, location: str) -> str:
        location = location.strip("'")
        assert location.startswith(self.stage), location
        return location[len(self.stage):].lstrip('/')

    def _listed(self, relative: str) -> str:
        return f"{self.stage.rsplit('.', 1)[1].lower()}/{relative}"

    def _matched(self, relative: str) -> str:
        # What Snowflake matches REMOVE/GET PATTERN against: the listed name
        # behind further internal path segments.
        return f'internal/{self._listed(relative)}'

    def session(self) -> Any:
        return _stage_session(self)

    def sql(self, sql: str) -> list[tuple]:
        import re
        self.statements.append(sql)
        verb, _, rest = sql.partition(' ')
        if verb == 'LIST':
            prefix = self._relative(rest)
            # A client-side encrypted stage lists padded ciphertext sizes.
            return [(self._listed(name), (len(data) // 16 + 1) *
                     16 if self.ciphertext else len(data), 'md5', 'date')
                    for name, data in sorted(self.objects.items())
                    if name.startswith(prefix)]
        if verb == 'REMOVE':
            location, _, pattern = rest.partition(' PATTERN = ')
            prefix = self._relative(location)
            for name in [n for n in self.objects if n.startswith(prefix)]:
                if not pattern or (not self.remove_pattern_ignored
                                   and re.fullmatch(pattern.strip("'"),
                                                    self._matched(name))):
                    del self.objects[name]
            return []
        raise AssertionError(f'unexpected statement {sql!r}')

    def put(self, local: str, target: str, **options: Any) -> list[str]:
        import os
        self.puts.append(options)
        name = self._relative(target).rstrip('/')
        key = f'{name}/{os.path.basename(local)}' if name else \
            os.path.basename(local)
        with open(local, 'rb') as reader:
            self.objects[key] = reader.read()
        return ['UPLOADED']

    def get(self, source: str, directory: str, *, pattern: str,
            **options: Any) -> list[str]:
        import os
        import re
        prefix = self._relative(source)
        got = []
        for name, data in self.objects.items():
            if name.startswith(prefix) and re.fullmatch(pattern,
                                                        self._matched(name)):
                with open(os.path.join(directory, os.path.basename(name)),
                          'wb') as writer:
                    writer.write(data)
                got.append(name)
        self.gets += got
        return got

    def presign(self, sql: str) -> list[tuple]:
        import re
        assert sql.startswith('SELECT'), sql
        paths = re.findall(r"\('((?:[^']|'')*)'\)", sql) or re.findall(
            r", '((?:[^']|'')*)', \d+\)", sql)
        return [(path, f'https://stage.invalid/{path}?signature=secret')
                for path in paths]

    def http_get(self, url: str, headers: Mapping[str, str],
                 **kwargs: Any) -> Any:
        import re

        import requests
        name = url.split('https://stage.invalid/', 1)[1].split('?')[0]
        response = requests.Response()
        response.url = url
        if name not in self.objects:
            response.status_code = 404
            return response
        data = self.objects[name]
        if self.ciphertext:
            data = bytes(b ^ 0xFF for b in data)
        self.ranges.append(headers['Range'])
        start, end = map(
            int,
            re.fullmatch(r'bytes=(\d+)-(\d+)', headers['Range']).groups())
        if self.ignores_range:
            response.status_code, response._content = 200, data
        else:
            response.status_code = 206
            response._content = data[start:end + 1]
            response.headers['Content-Range'] = (
                f'bytes {start}-{start + len(response._content) - 1}/'
                f'{len(data)}')
        return response


def _stage_session(stage: FakeStage) -> Any:
    from types import SimpleNamespace

    from snowflake.snowpark import Session

    class _Cursor:
        def execute(self, sql: str) -> None:
            self.rows = stage.presign(sql)

        def fetchall(self) -> list[tuple]:
            return self.rows

        def close(self) -> None:
            pass

    class _Session(Session):
        def __init__(self) -> None:
            pass

        file = SimpleNamespace(put=lambda *a, **k: stage.put(*a, **k),
                               get=lambda *a, **k: stage.get(*a, **k))
        connection = SimpleNamespace(cursor=_Cursor)

        def sql(self, sql: str, params: Any = None) -> Any:
            rows = stage.sql(sql)
            return SimpleNamespace(collect=lambda **_: rows)

    return _Session()
