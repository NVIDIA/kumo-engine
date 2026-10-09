# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Attestation(str, Enum):
    """Only a warehouse-reported count can reveal a source that returned
    fewer rows than it held."""

    WAREHOUSE = "warehouse"
    LOCAL = "local"


class Capability(str, Enum):
    """``UNKNOWN`` is what no answer looks like: never cached, never
    raised."""

    ALLOWED = "allowed"
    DENIED = "denied"
    UNKNOWN = "unknown"


class ReadMode(str, Enum):
    STAGED = "staged"
    STREAMED = "streamed"


@dataclass(frozen=True)
class StagedTable:
    """``source_size`` is the Parquet this read produced, not the size of
    the table in the warehouse, so it means the same whichever mode ran."""

    directory: str
    paths: tuple[str, ...]
    version: str
    source_size: int
    mode_used: ReadMode
    num_rows: int | None = None
    rows_attested_by: Attestation = Attestation.LOCAL
    scanned_bytes: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.paths, str):
            raise ValueError("paths must be a collection, not a single string")
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(self, "rows_attested_by",
                           Attestation(self.rows_attested_by))
        if not self.directory:
            raise ValueError("a staged table requires a directory")
        if not self.version:
            raise ValueError("a staged table requires the version it read")
        if not self.paths and self.num_rows != 0:
            raise ValueError(
                "a staged table requires at least one path unless it read "
                "no rows")
        if any(not path for path in self.paths):
            raise ValueError("staged paths cannot be empty")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("staged paths must be unique")
        if self.source_size < 0:
            raise ValueError("source_size cannot be negative")
        if self.num_rows is not None and self.num_rows < 0:
            raise ValueError("num_rows cannot be negative")
        if self.scanned_bytes is not None and self.scanned_bytes < 0:
            raise ValueError("scanned_bytes cannot be negative")
