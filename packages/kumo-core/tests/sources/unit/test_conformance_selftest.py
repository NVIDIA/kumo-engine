# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The suite must fail a reader that lies. Verified against deliberate breakage."""

import inspect
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.errors import (
    RowCountMismatchError,
    UnsupportedReadModeError,
)
from kumo_core.sources.types import ReadMode, StagedTable
from tests.sources.conformance import SourceReaderConformance

ROWS = 100
COLUMNS = {"id": "int64", "amount": "float32"}


def fixture_table() -> pa.Table:
    return pa.table({
        "id":
        pa.array(range(ROWS), pa.int64()),
        "amount":
        pa.array([float(i) for i in range(ROWS)], pa.float32()),
        "unused":
        pa.array(["x"] * ROWS, pa.string()),
    })


class MemoryReader:
    """Reference reader: both modes, correct behaviour, until told to lie."""
    def __init__(self, **lies):
        self.lies = lies
        self._modes = lies.get("modes", frozenset(ReadMode))

    def read_modes(self, table):
        return sorted(self._modes, key=lambda mode: mode.value)

    def data_version(self, table, *, sql=None, columns=None):
        return "1"

    def prepare(self, table, dest, *, columns, mode, aliases=None,
                expected_rows=None, worker_ceiling=None, sql=None,
                version=None):
        lie = self.lies.get
        if mode not in self._modes and not lie("substitute_mode"):
            raise UnsupportedReadModeError(f"{mode.value} unsupported")
        if not lie("skip_mkdir"):
            os.makedirs(dest, exist_ok=True)
        source = fixture_table()
        if not lie("ignore_projection"):
            source = source.select(list(columns))
        if aliases and not lie("ignore_aliases"):
            source = source.rename_columns(
                [aliases.get(name, name) for name in source.column_names])
        if lie("modes_disagree") and mode is ReadMode.STAGED:
            source = source.select(list(reversed(source.column_names)))
        source = source.slice(0, source.num_rows - lie("drop_rows", 0))
        path = f"{dest}/part-00000.parquet"
        pq.write_table(source, path)
        if not lie("swallow_short_read") and expected_rows is not None:
            if source.num_rows != expected_rows:
                raise RowCountMismatchError(expected=expected_rows,
                                            actual=source.num_rows, table=table)
        return StagedTable(
            directory=dest,
            paths=(path, ),
            version=lie("wrong_version") or version or self.data_version(table),
            source_size=os.path.getsize(path) *
            (10 if lie("wrong_size") else 1),
            mode_used=lie("wrong_mode") or mode,
            num_rows=source.num_rows,
        )


class ConformanceHarness(SourceReaderConformance):
    reader_kwargs: dict = {}

    def reader(self):
        return MemoryReader(**self.reader_kwargs)

    def table(self):
        return "orders"

    def columns(self):
        return COLUMNS

    def truth_rows(self):
        return ROWS


def run_suite(**reader_kwargs) -> list[str]:
    """The conformance checks a reader built this way fails, run without pytest."""
    harness = ConformanceHarness()
    harness.reader_kwargs = reader_kwargs
    root = Path(tempfile.mkdtemp())
    cache: dict = {}

    def staged():
        if not cache:
            cache["tables"] = {
                mode: harness.read(root, mode)
                for mode in harness.modes()
            }
        return cache["tables"]

    failed = []
    for name in sorted(dir(harness)):
        if not name.startswith("test_"):
            continue
        check = getattr(harness, name)
        parameters = inspect.signature(check).parameters
        kwargs = {}
        if "tmp_path" in parameters:
            kwargs["tmp_path"] = Path(tempfile.mkdtemp())
        try:
            if "staged" in parameters:
                kwargs["staged"] = staged()
            check(**kwargs)
        except pytest.skip.Exception:
            continue
        except BaseException:
            failed.append(name)
    return failed


LIES = {
    "test_every_mode_reads_every_row":
    dict(drop_rows=1),
    "test_the_projection_is_pushed_down":
    dict(ignore_projection=True),
    "test_a_short_read_is_reported_not_swallowed":
    dict(swallow_short_read=True),
    "test_a_mode_is_available":
    dict(modes=frozenset()),
    "test_an_alias_renames_the_staged_column":
    dict(ignore_aliases=True),
    "test_the_reported_version_describes_the_source":
    dict(wrong_version="9"),
    "test_modes_agree_with_each_other":
    dict(modes_disagree=True),
    "test_the_mode_used_is_reported":
    dict(wrong_mode=ReadMode.STAGED, modes=frozenset({ReadMode.STREAMED})),
    "test_an_unsupported_mode_is_refused_not_substituted":
    dict(substitute_mode=True, modes=frozenset({ReadMode.STREAMED})),
    "test_the_destination_is_created_by_the_read":
    dict(skip_mkdir=True),
    "test_the_size_reported_is_the_parquet_that_landed":
    dict(wrong_size=True),
}


@pytest.mark.parametrize("caught", LIES)
def test_the_suite_catches_a_reader_that_lies(caught):
    assert caught in run_suite(**LIES[caught])


def test_a_correct_reader_passes_everything():
    assert run_suite() == []
