# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Mapping

from kumoapi.typing import Dtype

from .errors import (
    RowCountMismatchError,
    SourceAccessDeniedError,
    UnsupportedReadModeError,
)
from .protocol import SourceReader
from .transfer import reset_destination
from .types import Attestation, ReadMode, StagedTable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadOutput:
    paths: tuple[str, ...]
    num_rows: int | None = None
    rows_attested_by: Attestation = Attestation.LOCAL
    scanned_bytes: int | None = None


def run_prepare(
    table: str,
    dest: str,
    mode: ReadMode,
    *,
    stamp: Callable[[], str],
    read: Callable[[], ReadOutput],
    expected_rows: int | None = None,
    version: str | None = None,
) -> StagedTable:
    """Stamped once, before the read. Stamping either side and refusing a
    difference cannot catch a torn read -- every read here is a single
    statement the warehouse gives one snapshot -- and would fail a view read
    that crossed the hour boundary its fallback version is cut on.

    A caller that already holds the version keyed its output on passes it,
    and ``stamp`` is not called: deriving it a second time only produces a
    value one side has to discard.
    """
    reset_destination(dest)
    if version is None:
        version = stamp()
    produced = read()
    if expected_rows is not None and produced.num_rows != expected_rows:
        raise RowCountMismatchError(expected=expected_rows,
                                    actual=produced.num_rows or 0, table=table)
    return StagedTable(
        directory=dest,
        paths=produced.paths,
        version=version,
        source_size=sum(os.path.getsize(path) for path in produced.paths),
        mode_used=mode,
        num_rows=produced.num_rows,
        rows_attested_by=produced.rows_attested_by,
        scanned_bytes=produced.scanned_bytes,
    )


def read_table(
    reader: SourceReader,
    table: str,
    dest: str,
    *,
    columns: Mapping[str, Dtype],
    mode: ReadMode | None = None,
    aliases: Mapping[str, str] | None = None,
    expected_rows: int | None = None,
    worker_ceiling: int | None = None,
    sql: str | None = None,
    version: str | None = None,
) -> StagedTable:
    """Read ``table`` through ``reader``, taking the reader's cheapest mode.

    A denial, and only a denial, retries the table streamed: the staging grant
    cannot always be established without trying, whereas any other permanent
    failure would fail the same way twice.

    An explicit ``mode`` runs neither the fallback nor the offer -- a caller
    who asked for staging wants to know staging worked, and asking what is
    offered is not free (BigQuery answers by running a real export and
    clearing what it wrote).
    """
    def read(chosen: ReadMode) -> StagedTable:
        return reader.prepare(table, dest, columns=columns, mode=chosen,
                              aliases=aliases, expected_rows=expected_rows,
                              worker_ceiling=worker_ceiling, sql=sql,
                              version=version)

    if mode is not None:
        # `ReadMode` is a `str` enum, so a bare "staged" passes every
        # membership check and then fails every `is` branch inside the
        # reader, silently running the mode the caller asked to avoid.
        return read(ReadMode(mode))
    offered = reader.read_modes(table)
    if not offered:
        raise UnsupportedReadModeError(
            f"{reader.source_type.value} offers no way to read {table}",
            table=table)
    first = offered[0]
    try:
        return read(first)
    except SourceAccessDeniedError:
        if first is ReadMode.STREAMED or ReadMode.STREAMED not in offered:
            raise
        logger.warning(
            "Staging %s was denied at read time; streaming it instead", table)
        return read(ReadMode.STREAMED)
