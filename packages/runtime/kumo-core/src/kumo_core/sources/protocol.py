# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from kumoapi.typing import Dtype

from ..types.enums import ConnectorType
from .errors import UnsupportedReadModeError
from .types import ReadMode, StagedTable


@runtime_checkable
class SourceReader(Protocol):
    """An implementation receives an already-authenticated client, configures
    warehouse-side staging on itself rather than per call, and leaves no staged
    remote object outliving :meth:`prepare`.
    """
    @property
    def source_type(self) -> ConnectorType:
        ...

    def read_modes(self, table: str) -> Sequence[ReadMode]:
        """Modes this reader can run under its configuration, cheapest
        first; the caller may take the first or override it."""

    def data_version(self, table: str, *, sql: str | None = None,
                     columns: Mapping[str, Dtype] | None = None) -> str:
        """Stamp ``table`` without reading it, and cheaply.

        A query is stamped by what it reads and by the ``columns`` contract it
        produces; a physical table ignores ``columns``.
        """

    def prepare(
        self,
        table: str,
        dest: str,
        *,
        columns: Mapping[str, Dtype],
        mode: ReadMode,
        aliases: Mapping[str, str] | None = None,
        expected_rows: int | None = None,
        worker_ceiling: int | None = None,
        sql: str | None = None,
        version: str | None = None,
    ) -> StagedTable:
        """Read ``columns`` of ``table`` into local Parquet under ``dest``.

        ``sql`` replaces the physical table with a read-only query, embedded
        in parenthesized subquery position and never rewritten; ``table`` is
        then the synthetic name staging prefixes and errors key on, and is
        never looked up as a table. Readers without SQL support accept
        ``None`` and reject a query with :class:`UnsupportedReadModeError`
        before reading the source or modifying ``dest``.

        ``worker_ceiling`` bounds every pool this read opens; it is per call
        because it describes this host at this moment, not the reader.

        ``version`` is the stamp the caller already derived and keyed its
        output on; given one, an implementation must not derive a second.

        An implementation must: clear ``dest`` and any staging prefix before
        writing and drop the prefix afterwards, so a retry cannot mix a
        half-finished attempt in; project exactly ``columns``, casting by
        dtype unless ingest converts to it, and naming the result back through
        ``aliases``; stamp the source *before* the read, since a version taken
        afterwards describes rows the read never saw; reconcile
        ``expected_rows`` where a cheap count exists; and refuse a mode
        outside :meth:`read_modes` rather than substituting another.
        """


def staged_first(staged: bool) -> tuple[ReadMode, ...]:
    """Staging leads where available: compressed Parquet, no table held in
    memory."""
    return ((ReadMode.STAGED, ReadMode.STREAMED) if staged else
            (ReadMode.STREAMED, ))


def require_mode(table: str, mode: ReadMode,
                 offered: Sequence[ReadMode]) -> None:
    if mode not in offered:
        raise UnsupportedReadModeError(
            f"cannot read {table} {mode.value}; this reader offers "
            f"{', '.join(m.value for m in offered)}", table=table)
