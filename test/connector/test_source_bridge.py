# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp

import pytest
from kumo_core.sources.errors import SourceAccessDeniedError
from kumo_core.sources.types import ReadMode, StagedTable
from kumo_core.types.enums import ConnectorType
from kumo_core.types.schema import IngestedTableFingerprint
from kumoapi.typing import Dtype

from kumo.connector import source_bridge
from kumo.connector.source_bridge import (
    read_source,
    stage_source,
    worker_ceiling,
)
from kumo.data import MaterializationStage
from kumo.data.utils import get_materialize_path
from rest.constants import DATA_BACKEND_WORKER_NAME
from temporalio_commons.worker_main import GLOBAL_WORKER_OPTIONS

POOL_WIDTH = GLOBAL_WORKER_OPTIONS[DATA_BACKEND_WORKER_NAME].max_worker_threads

FILES = ('part-0.parquet', 'part-1.parquet')


class FakeReader:
    source_type = ConnectorType.SNOWFLAKE

    def __init__(self, version='v-reader', num_rows=3, source_size=11,
                 scanned_bytes=None, modes=(ReadMode.STAGED, ReadMode.STREAMED),
                 error=None):
        self.version = version
        self.num_rows = num_rows
        self.source_size = source_size
        self.scanned_bytes = scanned_bytes
        self.modes = modes
        self.error = error
        self.ceilings = []
        self.dest = None
        self.mode = None
        self.sql = None
        self.tables_read = []
        self.modes_tried = []
        self.versions = []

    def read_modes(self, table):
        return self.modes

    def data_version(self, table):
        return self.version

    def prepare(self, table, dest, *, columns, mode, aliases=None,
                expected_rows=None, worker_ceiling=None, sql=None,
                version=None):
        self.dest, self.mode, self.sql = dest, mode, sql
        self.tables_read.append(table)
        self.versions.append(version)
        self.modes_tried.append(mode)
        self.ceilings.append(worker_ceiling)
        if self.error is not None and mode is ReadMode.STAGED:
            raise self.error
        os.makedirs(dest, exist_ok=True)
        paths = tuple(osp.join(dest, name) for name in FILES)
        for path in paths:
            with open(path, 'wb') as f:
                f.write(b'parquet')
        return StagedTable(directory=dest, paths=paths, version=version
                           or self.version, source_size=self.source_size,
                           mode_used=mode, num_rows=self.num_rows,
                           scanned_bytes=self.scanned_bytes)


@pytest.fixture
def table_fp():
    return IngestedTableFingerprint(
        connector_name='snowflake',
        source_name='CUSTOMERS',
        cols={
            'id': Dtype.int64,
            'name': Dtype.string
        },
        pkey='id',
        time_col=None,
        end_time_col=None,
    )


def ingest_dir(table_fp, data_version):
    return get_materialize_path(table_fp, data_version,
                                MaterializationStage.INGEST)


def test_the_parquet_lands_flat_under_the_callers_own_data_version(table_fp):
    reader = FakeReader(version='v-reader')
    read_source(reader, 'CUSTOMERS', table_fp, 'v-caller')
    assert reader.versions == ['v-caller']
    assert sorted(os.listdir(ingest_dir(table_fp, 'v-caller'))) == sorted(FILES)
    assert not osp.exists(ingest_dir(table_fp, 'v-reader'))


def test_source_size_is_the_parquet_written_not_what_the_warehouse_scanned(
        table_fp):
    """SOURCE_SIZE reserves the disk an ingest needs, so only bytes that
    landed on it can mean it, whatever else a vendor reported."""
    reader = FakeReader(source_size=11, scanned_bytes=9000)
    assert read_source(reader, 'CUSTOMERS', table_fp, 'v1').source_size == 11


def test_a_read_divides_the_budget_between_the_reads_that_can_run(monkeypatch):
    """Every table of a graph refresh reads in this one process, but only as
    many as the worker has threads can be reading at once."""
    monkeypatch.setattr(source_bridge, 'cpu_budget', lambda: 64)
    assert worker_ceiling(None) is None
    assert worker_ceiling(4) == 16
    assert worker_ceiling(POOL_WIDTH * 3) == 64 // POOL_WIDTH

    monkeypatch.setattr(source_bridge, 'cpu_budget', lambda: 2)
    assert worker_ceiling(50) == 1


def test_a_table_count_narrows_the_read_and_its_absence_leaves_it_alone(
        table_fp):
    untouched = FakeReader()
    read_source(untouched, 'CUSTOMERS', table_fp, 'v1')
    assert untouched.ceilings == [None]

    narrowed = FakeReader()
    read_source(narrowed, 'CUSTOMERS', table_fp, 'v1', graph_table_count=4)
    assert narrowed.ceilings == [worker_ceiling(4)]


def test_the_scratch_directory_is_removed_however_the_read_ends(table_fp):
    """It sits on the ingest root, which is sized for one graph at a time."""
    reader = FakeReader()
    read_source(reader, 'CUSTOMERS', table_fp, 'v1')
    assert not osp.exists(reader.dest)

    failed = FakeReader(error=RuntimeError('warehouse said no'),
                        modes=(ReadMode.STAGED, ))
    with pytest.raises(RuntimeError):
        read_source(failed, 'CUSTOMERS', table_fp, 'v1')
    assert not osp.exists(failed.dest)


def test_the_bridge_delegates_the_choice_of_mode_to_kumo_core(table_fp):
    """The one check here that mode selection happens at all: which mode runs
    and when a denial retries streamed are ``kumo_core``'s, and are pinned by
    ``packages/kumo-core/tests/sources/test_read_table.py``. What this adds is
    that the bridge routes through it -- a fallback reads twice into the same
    scratch directory, and the destination still gets what the second read
    wrote.
    """
    reader = FakeReader(error=SourceAccessDeniedError('no stage grant'))
    metrics = read_source(reader, 'CUSTOMERS', table_fp, 'v1')
    assert reader.modes_tried == [ReadMode.STAGED, ReadMode.STREAMED]
    assert reader.versions == ['v1', 'v1']
    assert metrics.num_rows == 3
    assert sorted(os.listdir(ingest_dir(table_fp, 'v1'))) == sorted(FILES)


@pytest.mark.parametrize('num_rows', [None, 0, 7])
def test_num_rows_passes_through_to_the_export_metrics(table_fp, num_rows):
    metrics = read_source(FakeReader(num_rows=num_rows), 'CUSTOMERS', table_fp,
                          'v1')
    assert metrics.num_rows == num_rows


def test_a_destination_that_cannot_be_cleared_stops_the_publish(
        table_fp, monkeypatch):
    """The copy is entry-by-entry, so a delete that only partly succeeded
    would leave the previous attempt's shards for this one to overlay, and
    ingest would read the mixture as one table."""
    copied = []

    def refuse(*args, **kwargs):
        raise OSError('one shard could not be deleted')

    monkeypatch.setattr(source_bridge.fsutil, 'rm', refuse)
    monkeypatch.setattr(source_bridge.fsutil, 'copy',
                        lambda *args, **kwargs: copied.append(args))

    with pytest.raises(RuntimeError, match='could not clear') as excinfo:
        read_source(FakeReader(), 'CUSTOMERS', table_fp, 'v1')

    assert ingest_dir(table_fp, 'v1') in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)
    assert not copied


def test_a_query_reaches_the_reader_as_sql_not_as_a_table_name(table_fp):
    """A Custom SQL table has no source table to read; the name is only what
    staging prefixes and errors key on."""
    reader = FakeReader()
    metrics = read_source(reader, 'fpv3.sql.dbx.orders', table_fp, 'v1',
                          sql='SELECT id FROM orders')

    assert reader.sql == 'SELECT id FROM orders'
    assert reader.tables_read == ['fpv3.sql.dbx.orders']
    assert metrics.num_rows == 3
    assert not metrics.approx


def test_a_table_read_has_no_sql_and_preserves_the_callers_version(table_fp):
    reader = FakeReader()
    assert read_source(reader, 'CUSTOMERS', table_fp, 'v1').num_rows == 3
    assert reader.sql is None
    assert reader.versions == ['v1']


def test_a_pinned_mode_replaces_the_readers_offer(tmp_path):
    """A read that must not write to the source account pins ``STREAMED``,
    and pinning has to actually reach ``read_table``: an unpinned read takes
    the staged offer first, which for every warehouse means unloading through
    the customer's own storage."""
    reader = FakeReader(modes=(ReadMode.STAGED, ReadMode.STREAMED))
    stage_source(reader, 'CUSTOMERS', str(tmp_path / 'dest'),
                 columns={'id': Dtype.int64}, mode=ReadMode.STREAMED)
    assert reader.modes_tried == [ReadMode.STREAMED]


def test_an_unpinned_stage_source_still_takes_the_readers_first_offer(tmp_path):
    reader = FakeReader(modes=(ReadMode.STAGED, ReadMode.STREAMED))
    stage_source(reader, 'CUSTOMERS', str(tmp_path / 'dest'),
                 columns={'id': Dtype.int64})
    assert reader.modes_tried == [ReadMode.STAGED]


@pytest.mark.parametrize('failure_type', [RuntimeError, KeyboardInterrupt])
def test_failed_publisher_does_not_delete_peer_output(tmp_path, monkeypatch,
                                                      caplog, failure_type):
    destination = tmp_path / 'destination'
    error = failure_type('private publication detail')

    def failed_copy(source, target, **kwargs):
        destination.mkdir()
        (destination / 'peer.parquet').write_bytes(b'other publisher output')
        raise error

    monkeypatch.setattr(source_bridge.fsutil, 'copy', failed_copy)
    reader = FakeReader()
    with pytest.raises(failure_type) as caught:
        stage_source(reader, 'CUSTOMERS', str(destination),
                     columns={'id': Dtype.int64})
    assert caught.value is error
    assert (destination /
            'peer.parquet').read_bytes() == b'other publisher output'
    assert not os.path.exists(reader.dest)
    assert 'private publication detail' not in caplog.text
