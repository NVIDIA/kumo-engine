# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class SourceError(Exception):
    def __init__(self, message: str, *, table: str | None = None) -> None:
        super().__init__(message)
        self.table = table


class TransientSourceError(SourceError):
    """A retry of the same read may succeed."""


class PermanentSourceError(SourceError):
    """Retrying will not help; the request or the grants must change."""


class ReconnectRequiredError(PermanentSourceError):
    """The credential lapsed; the connector definition is still correct, so
    prompt for reconnection rather than discard it."""


class SourceNotFoundError(PermanentSourceError):
    """The table named is not in the warehouse.

    Warehouses that cannot tell "absent" from "not granted" apart report the
    denial instead, rather than guess and send the caller to fix the wrong
    thing.
    """


class SourceAccessDeniedError(PermanentSourceError):
    """A grant the chosen read mode needs is missing. Narrow on purpose:
    this alone selects the streamed fallback."""


class UnsupportedReadModeError(PermanentSourceError):
    """The requested mode cannot satisfy this table."""


class RowCountMismatchError(PermanentSourceError):
    """Fewer rows arrived than the source reported; warehouses have shipped
    silent truncation with a success flag."""
    def __init__(self, *, expected: int, actual: int,
                 table: str | None = None) -> None:
        super().__init__(f"read {actual} rows against an expected {expected}",
                         table=table)
        self.expected = expected
        self.actual = actual


class SchemaChangedError(PermanentSourceError):
    """The source changed schema partway through a read; shards are written
    independently, so an unnoticed change only fails at ingest."""
