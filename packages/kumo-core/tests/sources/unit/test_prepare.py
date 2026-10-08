# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One sequence for every read, so no reader can drift out of it alone."""

import os

import pytest
from kumo_core.sources.errors import RowCountMismatchError
from kumo_core.sources.prepare import ReadOutput, run_prepare
from kumo_core.sources.types import ReadMode

PARQUET = b"PAR1" + b"body" * 32


class Warehouse:
    """A source that writes one file and can move under the read."""
    def __init__(self, versions=("7", "7", "7"), rows=100, **output):
        self.versions = list(versions)
        self.rows = rows
        self.output = output
        self.stamps = 0
        self.order = []

    def stamp(self):
        self.order.append("stamp")
        version = self.versions[min(self.stamps, len(self.versions) - 1)]
        self.stamps += 1
        return version

    def read(self, dest):
        def run():
            self.order.append("read")
            path = os.path.join(dest, "part-00000.parquet")
            with open(path, "wb") as handle:
                handle.write(PARQUET)
            return ReadOutput(paths=(path, ), num_rows=self.rows, **self.output)

        return run


def prepared(tmp_path, warehouse=None, mode=ReadMode.STAGED, **kwargs):
    source = warehouse or Warehouse()
    dest = str(tmp_path / "orders")
    return source, run_prepare("orders", dest, mode, stamp=source.stamp,
                               read=source.read(dest), **kwargs)


def test_the_mode_used_is_reported(tmp_path):
    _, staged = prepared(tmp_path, mode=ReadMode.STREAMED)
    assert staged.mode_used is ReadMode.STREAMED


class TestStamping:
    def test_the_source_is_stamped_once_before_the_read(self, tmp_path):
        source, staged = prepared(tmp_path,
                                  Warehouse(versions=("7", "9", "9")))
        assert source.order == ["stamp", "read"]
        assert staged.version == "7"


class TestReconciliation:
    def test_a_matching_row_count_is_accepted(self, tmp_path):
        _, staged = prepared(tmp_path, expected_rows=100)
        assert staged.num_rows == 100

    @pytest.mark.parametrize("source,expected", [(None, 101),
                                                 (Warehouse(rows=None), 1)])
    def test_a_read_that_cannot_meet_the_count_is_reported(
            self, tmp_path, source, expected):
        with pytest.raises(RowCountMismatchError) as failure:
            prepared(tmp_path, source, expected_rows=expected)
        assert failure.value.expected == expected


class TestWhatIsReported:
    def test_an_earlier_attempts_parquet_is_not_mixed_in(self, tmp_path):
        stale = tmp_path / "orders"
        stale.mkdir()
        (stale / "part-99999.parquet").write_bytes(b"PAR1stale")
        _, staged = prepared(tmp_path)
        assert os.path.isdir(staged.directory)
        assert sorted(os.listdir(staged.directory)) == ["part-00000.parquet"]

    def test_the_size_is_measured_and_the_warehouses_scan_carried(
            self, tmp_path):
        _, staged = prepared(tmp_path, Warehouse(scanned_bytes=66629424))
        assert staged.source_size == len(PARQUET)
        assert staged.scanned_bytes == 66629424
