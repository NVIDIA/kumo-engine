# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A suite every source reader must pass, run once per vendor.

Subclass :class:`SourceReaderConformance`, supply a reader and a fixture
table, and the vendor inherits the contract. Nothing here is mocked: a reader
that cannot reach its warehouse fails rather than passes vacuously.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import pyarrow.parquet as pq
import pytest
from kumo_core.sources.errors import (
    RowCountMismatchError,
    UnsupportedReadModeError,
)
from kumo_core.sources.prepare import read_table
from kumo_core.sources.protocol import SourceReader
from kumo_core.sources.types import ReadMode
from kumoapi.typing import Dtype

SIZE_TOLERANCE = 2


def _parquet(directory: str) -> list[str]:
    names = [n for n in sorted(os.listdir(directory)) if n.endswith(".parquet")]
    if not names:
        raise AssertionError(f"no parquet written to {directory}")
    return [os.path.join(directory, name) for name in names]


def parquet_rows(directory: str) -> int:
    return sum(pq.ParquetFile(p).metadata.num_rows for p in _parquet(directory))


def parquet_bytes(directory: str) -> int:
    return sum(os.path.getsize(path) for path in _parquet(directory))


def parquet_columns(directory: str) -> list[str]:
    return list(pq.ParquetFile(_parquet(directory)[0]).schema.names)


class SourceReaderConformance:
    def reader(self) -> SourceReader:
        raise NotImplementedError

    def table(self) -> str:
        raise NotImplementedError

    def columns(self) -> Mapping[str, Dtype]:
        raise NotImplementedError

    def truth_rows(self) -> int:
        """Row count established independently of the reader."""
        raise NotImplementedError

    def sql(self) -> str | None:
        """A query source overrides this; ``table()`` then names a synthetic
        alias rather than anything the warehouse has."""
        return None

    def query(self) -> Mapping[str, object]:
        return {"sql": self.sql(), "columns": self.columns()}

    def modes(self) -> Sequence[ReadMode]:
        return self.reader().read_modes(self.table())

    def read(self, root, mode: ReadMode, name: str = "", **kwargs):
        dest = os.path.join(str(root), f"{mode.value}{name}")
        return read_table(self.reader(), self.table(), dest,
                          columns=self.columns(), mode=mode, sql=self.sql(),
                          **kwargs)

    @pytest.fixture(scope="class")
    def staged(self, tmp_path_factory) -> Mapping[ReadMode, object]:
        """One plain read per mode, shared by every check that only reads it.

        Seven checks against a two-mode reader is fourteen full reads of a
        fixture measured in millions of rows, and not one of them asks for a
        table the others have not already got.
        """
        root = tmp_path_factory.mktemp("conformance")
        return {mode: self.read(root, mode) for mode in self.modes()}

    def test_a_mode_is_available(self):
        assert self.modes(), "reader supports no mode for this table"

    def test_every_mode_reads_every_row(self, staged):
        truth = self.truth_rows()
        assert truth > 0, "fixture table is empty; the check would be vacuous"
        for mode, table in staged.items():
            assert parquet_rows(table.directory) == truth, mode

    def test_modes_agree_with_each_other(self, staged):
        """Column order is part of the agreement, not only membership."""
        if len(staged) < 2:
            pytest.skip("reader supports one mode for this table")
        counts = {parquet_rows(table.directory) for table in staged.values()}
        schemas = {
            tuple(parquet_columns(table.directory))
            for table in staged.values()
        }
        assert len(counts) == 1, f"modes disagree on row count: {counts}"
        assert len(schemas) == 1, f"modes disagree on schema: {schemas}"

    def test_the_projection_is_pushed_down(self, staged):
        for mode, table in staged.items():
            assert set(parquet_columns(table.directory)) == set(
                self.columns()), mode

    def test_the_mode_used_is_reported(self, staged):
        for mode, table in staged.items():
            assert table.mode_used is mode

    def test_an_unsupported_mode_is_refused_not_substituted(self, tmp_path):
        unsupported = set(ReadMode) - set(self.modes())
        if not unsupported:
            pytest.skip("reader supports every mode for this table")
        with pytest.raises(UnsupportedReadModeError):
            self.read(tmp_path, unsupported.pop())

    def test_the_version_is_stable_while_the_source_is(self):
        """Retried once, because a source with no version of its own falls
        back to a time window that a run may straddle the boundary of."""
        reader, table = self.reader(), self.table()
        for _ in range(2):
            if reader.data_version(table, **self.query()) == reader.data_version(
                    table, **self.query()):
                return
        raise AssertionError(f"{table} reports a different version each read")

    def test_a_short_read_is_reported_not_swallowed(self, tmp_path):
        for mode in self.modes():
            with pytest.raises(RowCountMismatchError):
                self.read(tmp_path, mode, expected_rows=self.truth_rows() + 1)

    def test_the_reported_version_describes_the_source(self, staged):
        """A fabricated version would key a snapshot to nothing."""
        for mode, table in staged.items():
            assert table.version == self.reader().data_version(
                self.table(), **self.query()), mode

    def test_an_alias_renames_the_staged_column(self, tmp_path):
        source_name = next(iter(self.columns()))
        aliases = {source_name: "renamed_by_conformance"}
        for mode in self.modes():
            table = self.read(tmp_path, mode, aliases=aliases,
                              version="caller-owned-version")
            assert table.version == "caller-owned-version", mode
            written = set(parquet_columns(table.directory))
            assert "renamed_by_conformance" in written, mode
            assert source_name not in written, mode

    def test_the_destination_is_created_by_the_read(self, staged):
        """A caller should not have to guess who owns the directory."""
        for mode, table in staged.items():
            assert os.path.isdir(table.directory), mode

    def test_the_size_reported_is_the_parquet_that_landed(self, staged):
        for mode, table in staged.items():
            on_disk = parquet_bytes(table.directory)
            assert on_disk > 0, mode
            assert on_disk / SIZE_TOLERANCE <= table.source_size, mode
            assert table.source_size <= on_disk * SIZE_TOLERANCE, mode
