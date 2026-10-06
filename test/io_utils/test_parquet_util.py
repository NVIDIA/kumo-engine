# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import multiprocessing
import os.path as osp
import pathlib
import random
import string
import threading
import time
import types
from copy import deepcopy

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from io_utils import fsutil, parquet_util


def random_letters(length=20):
    return ''.join(random.choices(string.ascii_letters, k=length))


def _sleep_forever():
    # Picklable worker that never returns, to trigger the subprocess timeout.
    while True:
        time.sleep(0.2)


def _parquet_bytes(values):
    payload = io.BytesIO()
    pd.DataFrame({'value': values}).to_parquet(payload)
    return payload.getvalue()


def _mock_cloud_parquet(monkeypatch, payloads, children=None):
    children = children or {}
    monkeypatch.setattr(parquet_util.fsutil, 'isdir',
                        lambda path: path in children)
    monkeypatch.setattr(parquet_util.fsutil, 'is_s3', lambda _: False)
    monkeypatch.setattr(parquet_util.fsutil, 'is_gcs', lambda _: True)
    monkeypatch.setattr(parquet_util, 'is_ucv', lambda _: False)
    monkeypatch.setattr(parquet_util.fsutil, 'listdir',
                        lambda path: children[path])
    monkeypatch.setattr(
        parquet_util, '_get_dataset',
        lambda _: pytest.fail('cloud parquet must use fsutil handles'))

    @contextlib.contextmanager
    def binary_handle(path, mode):
        assert mode == 'rb'
        yield io.BytesIO(payloads[path])

    monkeypatch.setattr(parquet_util.fsutil, 'get_binary_handle', binary_handle)


@pytest.mark.parametrize(
    ('uri', 'expected_path'),
    [
        ('gs://bucket/path', 'bucket/path'),
        ('s3://bucket/path', 'bucket/path'),
        (
            'abfss://container@account.dfs.core.windows.net/path',
            'path',
        ),
    ],
)
def test_get_dataset_uses_scoped_cloud_filesystem(
    monkeypatch,
    uri,
    expected_path,
):
    scoped_filesystem = object()
    dataset = object()
    monkeypatch.setattr(parquet_util, 'has_filesystem_credential_scope',
                        lambda: True)
    monkeypatch.setattr(parquet_util.fsutil, 'pyarrow_filesystem',
                        lambda _: scoped_filesystem)
    monkeypatch.setattr(
        parquet_util, 'arrow_cloud_filesystem',
        lambda _: pytest.fail('ambient filesystem must not be used'))
    monkeypatch.setattr(
        parquet_util.fsutil,
        'copy',
        lambda *_args, **_kwargs: pytest.fail(
            'cloud parquet must not be staged locally'),
    )

    def fake_dataset(path, *, filesystem, **kwargs):
        assert path == expected_path
        assert filesystem is scoped_filesystem
        return dataset

    monkeypatch.setattr(parquet_util.pyarrow.dataset, 'dataset', fake_dataset)

    with parquet_util._get_dataset(uri) as actual:
        assert actual is dataset


def test_get_dataset_preserves_ambient_cloud_filesystem(monkeypatch):
    ambient_filesystem = object()
    dataset = object()
    monkeypatch.setattr(parquet_util, 'has_filesystem_credential_scope',
                        lambda: False)
    monkeypatch.setattr(parquet_util, 'arrow_cloud_filesystem',
                        lambda _: ambient_filesystem)
    monkeypatch.setattr(
        parquet_util.fsutil, 'pyarrow_filesystem',
        lambda _: pytest.fail('scoped filesystem must not be used'))

    def fake_dataset(path, *, filesystem, **kwargs):
        assert path == 'bucket/path'
        assert filesystem is ambient_filesystem
        return dataset

    monkeypatch.setattr(parquet_util.pyarrow.dataset, 'dataset', fake_dataset)

    with parquet_util._get_dataset('s3://bucket/path') as actual:
        assert actual is dataset


@pytest.mark.parametrize(('values', 'expected'), [([], True), ([1], False)])
def test_exact_cloud_is_empty_reads_through_fsutil(monkeypatch, values,
                                                   expected):
    path = 'gs://bucket/table.parquet'
    _mock_cloud_parquet(monkeypatch, {path: _parquet_bytes(values)})

    assert parquet_util._is_empty_helper(path) is expected


@pytest.mark.parametrize(('values', 'expected'), [([], True), ([1], False)])
def test_cloud_directory_is_empty_reads_exact_parquet_files(
        monkeypatch, values, expected):
    root = 'gs://bucket/table'
    part = f'{root}/part.parquet'
    _mock_cloud_parquet(
        monkeypatch,
        {part: _parquet_bytes(values)},
        {root: [f'{root}/_SUCCESS', part]},
    )

    assert parquet_util._is_empty_helper(root) is expected


def test_local_directory_is_empty_preserves_dataset_scan(monkeypatch):
    monkeypatch.setattr(parquet_util.fsutil, 'isdir', lambda _: True)
    monkeypatch.setattr(parquet_util.fsutil, 'is_s3', lambda _: False)
    monkeypatch.setattr(parquet_util.fsutil, 'is_gcs', lambda _: False)
    monkeypatch.setattr(parquet_util, 'is_ucv', lambda _: False)
    monkeypatch.setattr(parquet_util.fsutil, 'listdir',
                        lambda _: ['/tmp/table/part.parquet'])

    class Dataset:
        def count_rows(self):
            return 1

    @contextlib.contextmanager
    def get_dataset(path):
        assert path == '/tmp/table/part.parquet'
        yield Dataset()

    monkeypatch.setattr(parquet_util, '_get_dataset', get_dataset)

    assert parquet_util._is_empty_helper('/tmp/table') is False


def test_exact_cloud_num_rows_reads_through_fsutil(monkeypatch):
    path = 'gs://bucket/table.parquet'
    _mock_cloud_parquet(monkeypatch, {path: _parquet_bytes([1, 2, 3])})
    assert parquet_util._count_rows_per_file(path) == 3


def test_cloud_directory_num_rows_recurses_and_filters_sidecars(monkeypatch):
    payloads = {
        path: _parquet_bytes(values)
        for path, values in {
            'gs://bucket/table/part-0.parquet': [1, 2],
            'gs://bucket/table/date=2026/part-1.parquet': [3],
        }.items()
    }

    directories = {'gs://bucket/table', 'gs://bucket/table/date=2026'}
    children = {
        'gs://bucket/table': [
            'gs://bucket/table/_SUCCESS',
            'gs://bucket/table/part-0.parquet',
            'gs://bucket/table/date=2026',
        ],
        'gs://bucket/table/date=2026': [
            'gs://bucket/table/date=2026/part-1.parquet',
        ],
    }
    assert directories == set(children)
    _mock_cloud_parquet(monkeypatch, payloads, children)

    assert parquet_util._count_rows_per_file('gs://bucket/table') == 3


@pytest.mark.parametrize(
    'remote_path',
    [
        'gs://bucket/artifact.parquet',
        ('abfss://container@account.dfs.core.windows.net/'
         'artifact.parquet'),
    ],
)
def test_cloud_parquet_metadata_reads_remote_dataset_directly(
        monkeypatch, tmp_path, remote_path):
    local_source = tmp_path / 'source.parquet'
    expected = pd.DataFrame({'value': [1, 2, 3]})
    expected.to_parquet(local_source)
    local_dataset = parquet_util._get_dataset

    @contextlib.contextmanager
    def remote_dataset(path):
        assert path == remote_path
        with local_dataset(str(local_source)) as dataset:
            yield dataset

    monkeypatch.setattr(parquet_util, '_get_dataset', remote_dataset)
    monkeypatch.setattr(
        parquet_util.fsutil,
        'copy',
        lambda *_args, **_kwargs: pytest.fail(
            'metadata must not copy the cloud payload'),
    )

    metadata = parquet_util.get_parquet_metadata(remote_path)

    assert metadata.cols == ['value']
    assert metadata.num_rows == 3
    assert metadata.num_cols == 1
    assert metadata.avg_row_size > 0


@pytest.mark.parametrize(
    'remote_path',
    [
        'gs://bucket/artifact',
        'abfss://container@account.dfs.core.windows.net/artifact',
    ],
)
def test_cloud_directory_metadata_reads_fragments_directly(
        monkeypatch, tmp_path, remote_path):
    local_source = tmp_path / 'source'
    (local_source / 'date=2026').mkdir(parents=True)
    pd.DataFrame({'value': [1, 2]}).to_parquet(local_source / 'part-0.parquet')
    pd.DataFrame({
        'value': [3, 4, 5]
    }).to_parquet(local_source / 'date=2026' / 'part-1.parquet')
    (local_source / '_SUCCESS').touch()
    local_dataset = parquet_util._get_dataset

    @contextlib.contextmanager
    def remote_dataset(path):
        assert path == remote_path
        with local_dataset(str(local_source)) as dataset:
            yield dataset

    monkeypatch.setattr(parquet_util, '_get_dataset', remote_dataset)
    monkeypatch.setattr(
        parquet_util.fsutil,
        'copy',
        lambda *_args, **_kwargs: pytest.fail(
            'metadata must not copy the cloud payload'),
    )

    metadata = parquet_util.get_parquet_metadata(remote_path)

    assert metadata.cols == ['value']
    assert metadata.num_rows == 5
    assert metadata.num_cols == 1
    assert metadata.avg_row_size > 0


def test_count_rows(tmp_path: pathlib.Path):
    a = pd.DataFrame({'a': [1, 2, 4, 5, 6], 'v': ['a', 'b', 'c', 'd', 'e']})

    fsutil.makedirs(osp.join(tmp_path, 'test_parquet_util'))
    a.to_parquet(osp.join(tmp_path, 'test_parquet_util', 'a.parquet'))
    assert not parquet_util.is_empty(osp.join(tmp_path, 'test_parquet_util'))
    info = parquet_util.get_parquet_metadata(
        osp.join(tmp_path, 'test_parquet_util', 'a.parquet'))
    assert set(info.cols) == {'a', 'v'}
    assert info.num_cols == 2
    assert info.num_rows == 5


def test_count_rows_paritions(tmp_path: pathlib.Path):
    a = pd.DataFrame({'a': [1, 2, 4, 5, 6], 'v': ['a', 'b', 'c', 'd', 'e']})

    fsutil.makedirs(osp.join(tmp_path, 'test_parquet_util'))
    a.to_parquet(osp.join(tmp_path, 'test_parquet_util', 'a_0.parquet'))
    a.to_parquet(osp.join(tmp_path, 'test_parquet_util', 'a_1.parquet'))
    assert not parquet_util.is_empty(osp.join(tmp_path, 'test_parquet_util'))
    info = parquet_util.get_parquet_metadata(
        osp.join(tmp_path, 'test_parquet_util'))
    assert set(info.cols) == {'a', 'v'}
    assert info.num_cols == 2
    assert info.num_rows == 10


def test_get_parquet_metadata_wildcard(tmp_path: pathlib.Path,
                                       deploy_env_config):
    # parquet_util.get_parquet_metadata on subsets of .parquet files in a
    # dir that match a prefix+wildcard path
    a = pd.DataFrame({'a': [1, 2, 4, 5, 6], 'v': ['a', 'b', 'c', 'd', 'e']})

    # Unique prefix per session avoids race conditions between concurrent CI runs
    s3_base = f"s3://kumo-test-dataplane-bucket/fsutil-test/{deploy_env_config.instance_id}/"
    for base_path in [tmp_path, s3_base]:
        test_path = osp.join(base_path, 'test_parquet_util')
        fsutil.makedirs(test_path)
        a.to_parquet(osp.join(test_path, 'a_0.parquet'))
        a.to_parquet(osp.join(test_path, 'a_1.parquet'))
        a.to_parquet(osp.join(test_path, 'b_1.parquet'))
        a.to_parquet(osp.join(test_path, 'a1.parquet'))

        # match a_0.parquet, a_1.parquet
        info = parquet_util.get_parquet_metadata(osp.join(test_path, 'a_*'))
        assert set(info.cols) == {'a', 'v'}
        assert info.num_cols == 2
        assert info.num_rows == 10

        # match a_0.parquet, a_1.parquet, a1.parquet
        info = parquet_util.get_parquet_metadata(osp.join(test_path, 'a*'))
        assert set(info.cols) == {'a', 'v'}
        assert info.num_cols == 2
        assert info.num_rows == 15

        # match all files in the directory
        info = parquet_util.get_parquet_metadata(osp.join(test_path, '*'))
        assert set(info.cols) == {'a', 'v'}
        assert info.num_cols == 2
        assert info.num_rows == 20

        fsutil.rm(test_path)


def test_count_rows_mixed_file_types(tmp_path: pathlib.Path):
    temp_path = tmp_path.resolve().as_posix()
    a = pd.DataFrame({'a': [1, 2, 4, 5, 6], 'v': ['a', 'b', 'c', 'd', 'e']})

    path = osp.join(temp_path, 'test_parquet_util')
    fsutil.makedirs(path)
    a.to_parquet(osp.join(path, 'a_0.parquet'))
    a.to_parquet(osp.join(path, 'a_1.parquet'))
    fsutil.touch(osp.join(path, 'a_2.sst'))
    assert parquet_util.parquet_num_rows(path) == 10


def test_to_shards(tmp_path):
    df = pd.DataFrame()
    # Create dataframe of about 8MB (each row is 8 bytes)
    df['items'] = np.arange(0, 1 << 10)

    # write to 10 shards, totaling 80MB
    for i in range(10):
        df.to_parquet(str(tmp_path / f"part-{i}.parquet"))

    # iter_parquet with 1MB target should return about 80 shards
    shards = list(parquet_util.to_batches(str(tmp_path), 1 << 10))
    assert len(shards) > 60
    assert len(shards) < 100


def test_to_batches_multi_file_unchanged(tmp_path):
    # Across a multi-file directory the scan preserves every row, yields a
    # monotonic contiguous batch_number, and is deterministically ordered by
    # sorted file name.
    num_files = 5
    rows_per_file = 100
    for i in range(num_files):
        df = pd.DataFrame(
            {'x': np.arange(i * rows_per_file, (i + 1) * rows_per_file)})
        df.to_parquet(str(tmp_path / f"part-{i}.parquet"))
    total_rows = num_files * rows_per_file

    shards = list(parquet_util.to_batches(str(tmp_path), 1 << 10))

    # Monotonic, contiguous batch numbers across files.
    assert [s.batch_number for s in shards] == list(range(len(shards)))
    # No rows dropped or duplicated.
    assert sum(len(s.df) for s in shards) == total_rows
    # Deterministic, sorted-by-file-name order: files part-0..part-4 hold
    # arange(0..99), arange(100..199), ... so the concatenation is range().
    values = pd.concat([s.df for s in shards], ignore_index=True)['x'].tolist()
    assert values == list(range(total_rows))
    # A repeated scan yields the identical order.
    shards_again = list(parquet_util.to_batches(str(tmp_path), 1 << 10))
    values_again = pd.concat([s.df for s in shards_again],
                             ignore_index=True)['x'].tolist()
    assert values_again == values


def test_to_batches_nested_dir_not_truncated(tmp_path):
    # Regression guard: a nested / partitioned directory (parquet files in
    # subdirs, discovered recursively by the metadata dataset) must return
    # EVERY row. The prior non-recursive listdir(suffix='.parquet') saw no
    # top-level files and silently truncated the scan; deriving files from the
    # dataset's own fragments keeps discovery recursive and name-agnostic.
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    rows_per_file = 100
    pd.DataFrame({
        'x': np.arange(0, rows_per_file)
    }).to_parquet(str(tmp_path / 'a' / 'part-0.parquet'))
    pd.DataFrame({
        'x': np.arange(rows_per_file, 2 * rows_per_file)
    }).to_parquet(str(tmp_path / 'b' / 'part-1.parquet'))
    total_rows = 2 * rows_per_file

    shards = list(parquet_util.to_batches(str(tmp_path), 1 << 10))

    # Every row across both nested files is returned -- not truncated.
    assert sum(len(s.df) for s in shards) == total_rows
    # Monotonic, contiguous batch numbers across the recursive scan.
    assert [s.batch_number for s in shards] == list(range(len(shards)))


def test_to_batches_reacquires_dataset_per_file(tmp_path, monkeypatch):
    # The filesystem must not be held for the whole scan: each file re-enters
    # _get_dataset (and thus arrow_cloud_filesystem) so a scan outliving the
    # frozen-credential TTL picks up refreshed credentials between files.
    num_files = 4
    for i in range(num_files):
        df = pd.DataFrame({'x': np.arange(0, 100)})
        df.to_parquet(str(tmp_path / f"part-{i}.parquet"))

    real_get_dataset = parquet_util._get_dataset
    acquired: list[str] = []

    def counting_get_dataset(path, **kwargs):
        acquired.append(path)
        return real_get_dataset(path, **kwargs)

    monkeypatch.setattr(parquet_util, '_get_dataset', counting_get_dataset)

    shards = list(parquet_util.to_batches(str(tmp_path), 1 << 10))

    # Re-acquired more than once (not one FS for the whole scan): one metadata
    # acquire up front plus one per file.
    assert len(acquired) > 1
    per_file = [p for p in acquired if p.endswith('.parquet')]
    assert len(per_file) == num_files
    # Every file was scanned; rows preserved.
    assert sum(len(s.df) for s in shards) == num_files * 100


def test_to_batches_wildcard_local(tmp_path):
    # The wildcard branch resolves the prefix-matched files, preserves every
    # row, and yields them in deterministic sorted-by-name order with a
    # monotonic contiguous batch_number.
    num_files = 4
    rows_per_file = 100
    for i in range(num_files):
        df = pd.DataFrame(
            {'x': np.arange(i * rows_per_file, (i + 1) * rows_per_file)})
        df.to_parquet(str(tmp_path / f"part-{i}.parquet"))
    total_rows = num_files * rows_per_file

    wildcard = osp.join(str(tmp_path), 'part-*')
    shards = list(parquet_util.to_batches(wildcard, 1 << 10))

    # Monotonic, contiguous batch numbers.
    assert [s.batch_number for s in shards] == list(range(len(shards)))
    # No rows dropped or duplicated.
    assert sum(len(s.df) for s in shards) == total_rows
    # Deterministic, sorted-by-file-name order: part-0..part-3 hold
    # arange(0..99), arange(100..199), ... so the concatenation is range().
    values = pd.concat([s.df for s in shards], ignore_index=True)['x'].tolist()
    assert values == list(range(total_rows))


def test_to_batches_normalizes_s3a_wildcard_scheme(monkeypatch):
    # Latent regression guard: an s3a:// wildcard must be normalized to s3://
    # at the top of to_batches so the wildcard branch reattaches a real scheme
    # and every per-file _get_dataset sees an s3:// path -- not a bare,
    # scheme-less path that would be misread as a local file (FileNotFound).
    # Fully offline via a fake dataset; the full cloud wildcard path is not
    # locally runnable (no S3), so we assert the scheme routing directly.
    file_tables = {
        'bucket/data/part-0.parquet': pa.table({'x': [0, 1, 2]}),
        'bucket/data/part-1.parquet': pa.table({'x': [3, 4, 5]}),
    }
    meta_table = pa.table({'x': [0, 1, 2, 3, 4, 5]})

    class _FakeDataset:
        def __init__(self, table, fragment_paths=()):
            self._table = table
            self._fragment_paths = list(fragment_paths)

        @property
        def schema(self):
            return self._table.schema

        def count_rows(self):
            return self._table.num_rows

        def head(self, n):
            return self._table.slice(0, n)

        def get_fragments(self):
            # Cloud fragment paths carry no scheme (bucket/key), mirroring what
            # pyarrow returns for a cloud wildcard scan.
            return [types.SimpleNamespace(path=p) for p in self._fragment_paths]

        def to_batches(self, **kwargs):
            return self._table.to_batches()

    seen_paths = []

    @contextlib.contextmanager
    def fake_get_dataset(path, **kwargs):
        seen_paths.append(path)
        if path.endswith('*'):
            yield _FakeDataset(meta_table, fragment_paths=list(file_tables))
        else:
            yield _FakeDataset(file_tables[path.removeprefix('s3://')])

    monkeypatch.setattr(parquet_util, '_get_dataset', fake_get_dataset)
    # Keep the isdir probe offline and off the wildcard branch.
    monkeypatch.setattr(parquet_util.fsutil, 'isdir', lambda path: False)

    shards = list(parquet_util.to_batches('s3a://bucket/data/part-*', 1 << 20))

    # The per-file scans must all receive s3://-schemed paths (scheme
    # reattached after normalization), never bare/scheme-less paths.
    per_file = [p for p in seen_paths if p.endswith('.parquet')]
    assert per_file == [
        's3://bucket/data/part-0.parquet',
        's3://bucket/data/part-1.parquet',
    ]
    # Rows preserved, deterministic sorted-by-name order, monotonic batches.
    assert [s.batch_number for s in shards] == list(range(len(shards)))
    values = pd.concat([s.df for s in shards], ignore_index=True)['x'].tolist()
    assert values == [0, 1, 2, 3, 4, 5]


def test_save_df_timestamp(tmp_path):
    table = pa.table({
        'timestamp':
        pa.array(
            [1717386536614035456, 1717386536614035126, 1717386536614035226],
            type=pa.timestamp('ns'),
        ),
        'string_col':
        pa.array(['A', 'B', 'C'], type=pa.string())
    })

    # Convert the table to a Pandas DataFrame
    df = table.to_pandas()

    # PyArrow 14 now supports nanosecond timestamps natively
    # Test that both df.to_parquet and parquet_util.write_df work
    df.to_parquet(osp.join(tmp_path, "test_standard.parquet"))
    assert fsutil.exists(osp.join(tmp_path, "test_standard.parquet"))

    parquet_util.write_df(df, osp.join(tmp_path, "test_util.parquet"))
    assert fsutil.exists(osp.join(tmp_path, "test_util.parquet"))

    # Verify both files can be read back
    df_standard = pd.read_parquet(osp.join(tmp_path, "test_standard.parquet"))
    df_util = pd.read_parquet(osp.join(tmp_path, "test_util.parquet"))

    # Check that timestamps are preserved correctly
    assert len(df_standard) == 3
    assert len(df_util) == 3
    assert df_standard['timestamp'].dtype == df['timestamp'].dtype
    assert df_util['timestamp'].dtype == df['timestamp'].dtype


def test_is_parquet_table_empty(tmp_path):
    path = tmp_path.resolve().as_posix()
    assert parquet_util.parquet_num_rows(path) == 0

    nonexist_path = osp.join(path, 'nonexistant_file')
    assert parquet_util.parquet_num_rows(nonexist_path) == 0

    fsutil.makedirs(nonexist_path)
    assert parquet_util.parquet_num_rows(nonexist_path) == 0

    nonexist_path = osp.join(path, 'nonexistant_file.parquet')
    assert parquet_util.parquet_num_rows(nonexist_path) == 0

    pd_df = pd.DataFrame()
    pd_df.to_parquet(osp.join(path, 'empty_df.parquet'))
    assert parquet_util.parquet_num_rows(osp.join(path,
                                                  'empty_df.parquet')) == 0

    pd_df['test'] = [1, 2, 3]
    pd_df.to_parquet(osp.join(path, 'test_df.parquet'))

    assert parquet_util.parquet_num_rows(osp.join(path, 'test_df.parquet')) == 3


def test_write_df(tmp_path):
    path = tmp_path.resolve().as_posix()

    df_a = pd.DataFrame({'a': [1, 2, 4, 5, 6], 'v': ['a', 'b', 'c', 'd', 'e']})
    path_a = osp.join(path, "a")
    df_a.to_parquet(path_a)

    df_b: pd.DataFrame = df_a[0:0]  # empty with schema
    path_b = osp.join(path, "b")
    fsutil.makedirs(path_b)
    # to_parquet cannot handle an existing dir
    with pytest.raises(IsADirectoryError):
        df_b.to_parquet(path_b)

    # write_df also expects a filename
    with pytest.raises(ValueError):
        parquet_util.write_df(df_b, path_b)

    file_b = osp.join(path_b, 'snappy.parquet')
    parquet_util.write_df(df_b, file_b)


@pytest.mark.parametrize('uri', [
    'gs://bucket/output.parquet',
    's3://bucket/output.parquet',
])
def test_write_df_cloud_uses_fsutil_binary_handle(monkeypatch, uri):
    payload = io.BytesIO()
    real_write_table = parquet_util.pq.write_table

    monkeypatch.setattr(parquet_util.fsutil, 'isdir', lambda _: False)
    monkeypatch.setattr(parquet_util.fsutil, 'is_local', lambda _: False)
    monkeypatch.setattr(parquet_util.fsutil, 'is_s3',
                        lambda path: path.startswith('s3://'))
    monkeypatch.setattr(parquet_util.fsutil, 'is_gcs',
                        lambda path: path.startswith('gs://'))

    @contextlib.contextmanager
    def binary_handle(path, mode):
        assert path == uri
        assert mode == 'wb'
        yield payload

    def guarded_write_table(destination_table, destination, **kwargs):
        assert not isinstance(destination, str), (
            'cloud URI must not reach PyArrow implicit filesystem resolution')
        assert kwargs == {
            'use_deprecated_int96_timestamps': True,
            'flavor': 'spark',
        }
        return real_write_table(destination_table, destination, **kwargs)

    monkeypatch.setattr(parquet_util.fsutil, 'get_binary_handle', binary_handle)
    monkeypatch.setattr(parquet_util.pq, 'write_table', guarded_write_table)

    expected = pd.DataFrame({'value': [1, 2, 3]})
    parquet_util.write_df(expected, uri)

    actual = pd.read_parquet(io.BytesIO(payload.getvalue()))
    pd.testing.assert_frame_equal(actual, expected)


def test_write_df_local_preserves_direct_path(monkeypatch, tmp_path):
    output_path = str(tmp_path / 'output.parquet')
    real_write_table = parquet_util.pq.write_table

    monkeypatch.setattr(
        parquet_util.fsutil, 'get_binary_handle',
        lambda *_: pytest.fail('local path must remain a direct PyArrow write'))

    def guarded_write_table(destination_table, destination, **kwargs):
        assert destination == output_path
        return real_write_table(destination_table, destination, **kwargs)

    monkeypatch.setattr(parquet_util.pq, 'write_table', guarded_write_table)

    expected = pd.DataFrame({'value': [1, 2, 3]})
    parquet_util.write_df(expected, output_path)

    pd.testing.assert_frame_equal(pd.read_parquet(output_path), expected)


@pytest.mark.parametrize('prefix', [
    'gs://bucket/output',
    's3://bucket/output',
])
def test_write_in_chunks_cloud_uses_fsutil_binary_handle(monkeypatch, prefix):
    payloads = {}

    monkeypatch.setattr(parquet_util.fsutil, 'is_s3',
                        lambda path: path.startswith('s3://'))
    monkeypatch.setattr(parquet_util.fsutil, 'is_gcs',
                        lambda path: path.startswith('gs://'))
    monkeypatch.setattr(parquet_util.fsutil, 'is_local', lambda _: False)

    @contextlib.contextmanager
    def binary_handle(path, mode):
        assert path == f'{prefix}-00000.parquet'
        assert mode == 'wb'
        payload = io.BytesIO()
        payloads[path] = payload
        yield payload

    monkeypatch.setattr(parquet_util.fsutil, 'get_binary_handle', binary_handle)

    expected = pd.DataFrame({'value': [1, 2, 3]})
    parquet_util.write_in_chunks(expected, prefix, 1 << 20, save_index=False)

    payload = payloads[f'{prefix}-00000.parquet'].getvalue()
    pd.testing.assert_frame_equal(pd.read_parquet(io.BytesIO(payload)),
                                  expected)


def test_head_parquet_ucv(temp_ucv, tmp_path: pathlib.Path):
    """Test head_parquet with Databricks UCV for four cases:
    1. Single parquet file with 5 rows
    2. Directory with all parquet files (2 files, 5 rows each)
    3. Directory with mixed file types (2 parquets and 1 dummy file)
    4. Same as case 3 but we set max_rows to 3 in head_parquet
    5. Non-existent directory
    6. Empty directory
    """
    # Create test data
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v']})

    # Case 1: Single parquet file
    single_file = deepcopy(temp_ucv)
    single_file.path = f"single-{random_letters()}.parquet"

    # Write to local temp file first
    local_file = osp.join(tmp_path, "single.parquet")
    df.to_parquet(local_file)

    # Copy to UCV
    fsutil.copy_file(local_file, single_file.as_url())

    # Test head_parquet on single file
    result = parquet_util.head_parquet(single_file.as_url())
    assert isinstance(result, pa.Table)
    assert len(result) == 5
    assert result.column_names == ['a', 'b']

    # Case 2: Directory with all parquet files
    parquet_dir = deepcopy(temp_ucv)
    parquet_dir.path = f"parquet_dir-{random_letters()}"
    fsutil.makedirs(parquet_dir.as_url())

    # Create two parquet files
    for i in range(2):
        local_file = osp.join(tmp_path, f"parquet_{i}.parquet")
        df.to_parquet(local_file)
        fsutil.copy_file(local_file,
                         osp.join(parquet_dir.as_url(), f"file_{i}.parquet"))

    # Test head_parquet on directory with all parquet files
    result = parquet_util.head_parquet(parquet_dir.as_url())
    assert isinstance(result, pa.Table)
    assert len(result) == 10  # 5 rows from each parquet file
    assert result.column_names == ['a', 'b']

    # Case 3: Directory with multiple files (2 parquets and 1 non-parquet file)
    multi_dir = deepcopy(temp_ucv)
    multi_dir.path = f"multi-{random_letters()}"
    fsutil.makedirs(multi_dir.as_url())

    # Create two parquet files
    for i in range(2):
        local_file = osp.join(tmp_path, f"multi_{i}.parquet")
        df.to_parquet(local_file)
        fsutil.copy_file(local_file,
                         osp.join(multi_dir.as_url(), f"file_{i}.parquet"))

    # Create a dummy non-parquet file
    dummy_file = osp.join(multi_dir.as_url(), "dummy.txt")
    fsutil.write_binary(dummy_file, b"dummy content")

    # Test head_parquet on directory
    result = parquet_util.head_parquet(multi_dir.as_url())
    assert isinstance(result, pa.Table)
    assert len(result) == 10  # 5 rows from each parquet file
    assert result.column_names == ['a', 'b']

    # Case 4: Directory with head limit
    result = parquet_util.head_parquet(multi_dir.as_url(), max_rows=3)
    assert isinstance(result, pa.Table)
    assert len(result) == 3  # Limited to 3 rows
    assert result.column_names == ['a', 'b']

    # Case 5: Non-existent directory
    nonexistent_path = deepcopy(temp_ucv)
    nonexistent_path.path = f"nonexistent-{random_letters()}"

    # Test head_parquet on non-existent path
    with pytest.raises(
            Exception):  # Should raise an error for non-existent path
        parquet_util.head_parquet(nonexistent_path.as_url())

    # Case 6: Empty directory
    empty_dir = deepcopy(temp_ucv)
    empty_dir.path = f"empty_dir-{random_letters()}"
    fsutil.makedirs(empty_dir.as_url())

    # Test head_parquet on empty directory
    with pytest.raises(Exception):  # Should raise an error for empty directory
        parquet_util.head_parquet(empty_dir.as_url())


def test_parquet_num_rows_ucv(temp_ucv, tmp_path: pathlib.Path):
    """Test parquet_num_rows with Databricks UCV for various cases:
    1. Single parquet file
    2. Directory with all parquet files
    3. Directory with mixed file types (parquet and non-parquet)
    4. Non-existent path
    5. Empty directory
    """
    # Create test data
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v']})

    # Case 1: Single parquet file
    single_file = deepcopy(temp_ucv)
    single_file.path = f"single-{random_letters()}.parquet"

    # Write to local temp file first
    local_file = osp.join(tmp_path, "single.parquet")
    df.to_parquet(local_file)

    # Copy to UCV
    fsutil.copy_file(local_file, single_file.as_url())

    # Test parquet_num_rows on single file
    assert parquet_util.parquet_num_rows(single_file.as_url()) == 5
    assert parquet_util.parquet_num_rows(single_file.as_url(),
                                         skip_check=True) == 5

    # Case 2: Directory with all parquet files
    parquet_dir = deepcopy(temp_ucv)
    parquet_dir.path = f"parquet_dir-{random_letters()}"
    fsutil.makedirs(parquet_dir.as_url())

    # Create two parquet files
    for i in range(2):
        local_file = osp.join(tmp_path, f"parquet_{i}.parquet")
        df.to_parquet(local_file)
        fsutil.copy_file(local_file,
                         osp.join(parquet_dir.as_url(), f"file_{i}.parquet"))

    # Test parquet_num_rows on directory with all parquet files
    assert parquet_util.parquet_num_rows(parquet_dir.as_url()) == 10
    assert parquet_util.parquet_num_rows(parquet_dir.as_url(),
                                         skip_check=True) == 10

    # Case 3: Directory with mixed file types
    multi_dir = deepcopy(temp_ucv)
    multi_dir.path = f"multi_dir-{random_letters()}"
    fsutil.makedirs(multi_dir.as_url())

    # Create two parquet files
    for i in range(2):
        local_file = osp.join(tmp_path, f"multi_{i}.parquet")
        df.to_parquet(local_file)
        fsutil.copy_file(local_file,
                         osp.join(multi_dir.as_url(), f"file_{i}.parquet"))

    # Create a dummy non-parquet file
    dummy_file = osp.join(multi_dir.as_url(), "dummy.txt")
    fsutil.write_binary(dummy_file, b"dummy content")

    # Test parquet_num_rows on directory with mixed files
    assert parquet_util.parquet_num_rows(multi_dir.as_url()) == 10
    assert parquet_util.parquet_num_rows(multi_dir.as_url(),
                                         skip_check=True) == 10

    # Case 4: Non-existent path
    nonexistent_path = deepcopy(temp_ucv)
    nonexistent_path.path = f"nonexistent-{random_letters()}"

    # Test parquet_num_rows on non-existent path
    assert parquet_util.parquet_num_rows(nonexistent_path.as_url()) == 0
    assert parquet_util.parquet_num_rows(nonexistent_path.as_url(),
                                         skip_check=True) == 0

    # Case 5: Empty directory
    empty_dir = deepcopy(temp_ucv)
    empty_dir.path = f"empty_dir-{random_letters()}"
    fsutil.makedirs(empty_dir.as_url())

    # Test parquet_num_rows on empty directory
    assert parquet_util.parquet_num_rows(empty_dir.as_url()) == 0
    assert parquet_util.parquet_num_rows(empty_dir.as_url(),
                                         skip_check=True) == 0


def test_get_parquet_metadata_ucv(temp_ucv, tmp_path: pathlib.Path):
    """Test get_parquet_metadata with Databricks UCV for various cases:
    1. Single parquet file
    2. Directory with all parquet files
    3. Directory with mixed file types (parquet and non-parquet)
    3b. Wildcard path matching multiple but not all files in directory
    4. Non-existent path
    5. Empty directory
    """
    # Create test data
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': ['x', 'y', 'z', 'w', 'v'],
        'c': [1.1, 2.2, 3.3, 4.4, 5.5]
    })

    # Case 1: Single parquet file
    single_file = deepcopy(temp_ucv)
    single_file.path = f"single-{random_letters()}.parquet"

    local_file = osp.join(tmp_path, "single.parquet")
    df.to_parquet(local_file)

    # Copy to UCV
    fsutil.copy_file(local_file, single_file.as_url())

    metadata = parquet_util.get_parquet_metadata(single_file.as_url())
    assert set(metadata.cols) == {'a', 'b', 'c'}
    assert metadata.num_rows == 5
    assert metadata.num_cols == 3
    assert metadata.avg_row_size > 0  # Should be positive

    # Case 2: Directory with all parquet files
    parquet_dir = deepcopy(temp_ucv)
    parquet_dir.path = f"parquet_dir-{random_letters()}"
    fsutil.makedirs(parquet_dir.as_url())

    # Create two parquet files
    for i in range(2):
        local_file = osp.join(tmp_path, f"parquet_{i}.parquet")
        df.to_parquet(local_file)
        fsutil.copy_file(local_file,
                         osp.join(parquet_dir.as_url(), f"file_{i}.parquet"))

    # Test get_parquet_metadata on directory with all parquet files
    metadata = parquet_util.get_parquet_metadata(parquet_dir.as_url())
    assert set(metadata.cols) == {'a', 'b', 'c'}
    assert metadata.num_rows == 10  # 5 rows from each file
    assert metadata.num_cols == 3
    assert metadata.avg_row_size > 0

    # Case 3: Directory with mixed file types
    multi_dir = deepcopy(temp_ucv)
    multi_dir.path = f"multi_dir-{random_letters()}"
    fsutil.makedirs(multi_dir.as_url())

    # Create two parquet files
    for i in range(2):
        local_file = osp.join(tmp_path, f"multi_{i}.parquet")
        df.to_parquet(local_file)
        fsutil.copy_file(local_file,
                         osp.join(multi_dir.as_url(), f"file_{i}.parquet"))

    # Create a dummy non-parquet file
    dummy_file = osp.join(multi_dir.as_url(), "dummy.txt")
    fsutil.write_binary(dummy_file, b"dummy content")

    # Test get_parquet_metadata on directory with mixed files
    metadata = parquet_util.get_parquet_metadata(multi_dir.as_url())
    assert set(metadata.cols) == {'a', 'b', 'c'}
    assert metadata.num_rows == 10  # 5 rows from each parquet file
    assert metadata.num_cols == 3
    assert metadata.avg_row_size > 0

    # Case 3b: Wildcard path matching multiple but not all files in directory,
    # reusing dir from case 2:
    fsutil.copy_file(local_file,
                     osp.join(multi_dir.as_url(), f"diff_{i}.parquet"))
    prefix_dir = osp.join(multi_dir.as_url(), "file*")
    metadata = parquet_util.get_parquet_metadata(prefix_dir)
    assert set(metadata.cols) == {'a', 'b', 'c'}
    assert metadata.num_rows == 10  # 5 rows from each parquet file
    assert metadata.num_cols == 3
    assert metadata.avg_row_size > 0

    # for now verify that matching a non-parquet file raises an error
    extenionless_file = osp.join(multi_dir.as_url(), "file_0")
    fsutil.write_binary(extenionless_file, b"dummy content")

    with pytest.raises(ValueError, match="Found non-parquet file"):
        parquet_util.get_parquet_metadata(prefix_dir)

    # Create two parquet files
    # Case 4: Non-existent path
    nonexistent_path = deepcopy(temp_ucv)
    nonexistent_path.path = f"nonexistent-{random_letters()}"

    # Test get_parquet_metadata on non-existent path
    with pytest.raises(
            Exception):  # Should raise an error for non-existent path
        parquet_util.get_parquet_metadata(nonexistent_path.as_url())

    # Case 5: Empty directory
    empty_dir = deepcopy(temp_ucv)
    empty_dir.path = f"empty_dir-{random_letters()}"
    fsutil.makedirs(empty_dir.as_url())

    # Test get_parquet_metadata on empty directory
    with pytest.raises(Exception):  # Should raise an error for empty directory
        parquet_util.get_parquet_metadata(empty_dir.as_url())


def test_is_empty_helper_ucv(temp_ucv, tmp_path: pathlib.Path):
    """Test _is_empty_helper with Databricks UCV for various cases:
    1. Single parquet file with data
    2. Single empty parquet file
    3. Directory with all parquet files
    4. Directory with mixed file types (parquet and non-parquet)
    5. Non-existent path
    6. Empty directory
    """
    # Create test data
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v']})
    empty_df = pd.DataFrame()

    # Case 1: Single parquet file with data
    single_file = deepcopy(temp_ucv)
    single_file.path = f"single-{random_letters()}.parquet"

    # Write to local temp file first
    local_file = osp.join(tmp_path, "single.parquet")
    df.to_parquet(local_file)

    # Copy to UCV
    fsutil.copy_file(local_file, single_file.as_url())

    # Test _is_empty_helper on single file with data
    assert not parquet_util._is_empty_helper(single_file.as_url())

    # Case 2: Single empty parquet file
    empty_file = deepcopy(temp_ucv)
    empty_file.path = f"empty-{random_letters()}.parquet"

    # Write to local temp file first
    local_file = osp.join(tmp_path, "empty.parquet")
    empty_df.to_parquet(local_file)

    # Copy to UCV
    fsutil.copy_file(local_file, empty_file.as_url())

    # Test _is_empty_helper on empty file
    assert parquet_util._is_empty_helper(empty_file.as_url())

    # Case 3: Directory with all parquet files
    parquet_dir = deepcopy(temp_ucv)
    parquet_dir.path = f"parquet_dir-{random_letters()}"
    fsutil.makedirs(parquet_dir.as_url())

    # Create two parquet files
    for i in range(2):
        local_file = osp.join(tmp_path, f"parquet_{i}.parquet")
        df.to_parquet(local_file)
        fsutil.copy_file(local_file,
                         osp.join(parquet_dir.as_url(), f"file_{i}.parquet"))

    # Test _is_empty_helper on directory with all parquet files
    assert not parquet_util._is_empty_helper(parquet_dir.as_url())

    # Case 4: Directory with mixed file types
    multi_dir = deepcopy(temp_ucv)
    multi_dir.path = f"multi_dir-{random_letters()}"
    fsutil.makedirs(multi_dir.as_url())

    # Create two parquet files
    for i in range(2):
        local_file = osp.join(tmp_path, f"multi_{i}.parquet")
        df.to_parquet(local_file)
        fsutil.copy_file(local_file,
                         osp.join(multi_dir.as_url(), f"file_{i}.parquet"))

    # Create a dummy non-parquet file
    dummy_file = osp.join(multi_dir.as_url(), "dummy.txt")
    fsutil.write_binary(dummy_file, b"dummy content")

    # Test _is_empty_helper on directory with mixed files
    assert not parquet_util._is_empty_helper(multi_dir.as_url())

    # Case 5: Non-existent path
    nonexistent_path = deepcopy(temp_ucv)
    nonexistent_path.path = f"nonexistent-{random_letters()}"

    # Test _is_empty_helper on non-existent path
    assert parquet_util._is_empty_helper(nonexistent_path.as_url())

    # Case 6: Empty directory
    empty_dir = deepcopy(temp_ucv)
    empty_dir.path = f"empty_dir-{random_letters()}"
    fsutil.makedirs(empty_dir.as_url())

    # Test _is_empty_helper on empty directory
    assert parquet_util._is_empty_helper(empty_dir.as_url())


def test_read_parquet_offset_size():
    file_path = 's3://kumo-public-datasets/large_input_data/test1G128/'

    df = parquet_util.read_parquet(file_path, offset=1000, size=200)
    assert df.shape == (200, 128)

    df2 = parquet_util.read_parquet(file_path, offset=900, size=200)
    assert df2.shape == (200, 128)

    pd.testing.assert_series_equal(df.iloc[0], df2.iloc[100], check_names=False)


# --- py3.14 subprocess-isolation regression ---------------------------------
# is_empty()/parquet_num_rows() run the cloud-fs read in a throwaway process to
# isolate poisoned S3FS state. On py3.14 the old bare pool.apply() (default
# forkserver/spawn, no timeout) re-imported kumo and deadlocked forever. The
# fix forces a 'fork' context and a hard timeout that raises rather than
# blocking or falling back in-process.


def test_run_in_isolated_subprocess_uses_fork_context(monkeypatch):
    # The helper must force a 'fork' context (not the py3.14 default).
    requested = {}
    real_get_context = multiprocessing.get_context

    def spy_get_context(method=None):
        requested['method'] = method
        return real_get_context(method)

    monkeypatch.setattr(parquet_util.multiprocessing, 'get_context',
                        spy_get_context)

    result = parquet_util._run_in_isolated_subprocess(int, ('7', ))
    assert result == 7
    assert requested['method'] == 'fork'


def test_run_in_isolated_subprocess_times_out_and_raises(monkeypatch):
    # On timeout the helper must raise TimeoutError (never block or fall back
    # in-process) and tear the pool down promptly rather than hang.
    if 'fork' not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method unavailable on this platform")

    monkeypatch.setenv('PARQUET_ISOLATED_SUBPROCESS_TIMEOUT_SECS', '1')

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        parquet_util._run_in_isolated_subprocess(_sleep_forever, ())
    elapsed = time.monotonic() - start

    # Bounded: the raising timeout is the backstop against a wedged child.
    assert elapsed < 15, elapsed


def test_s3_branch_routes_through_isolated_subprocess(monkeypatch):
    # The S3 branch must go through the py3.14-safe helper.
    monkeypatch.setattr(parquet_util.fsutil, 'is_s3', lambda path: True)
    monkeypatch.setattr(parquet_util, '_is_empty_helper', lambda path: False)
    monkeypatch.setattr(parquet_util, '_parquet_num_rows_helper',
                        lambda path, skip_check: 42)

    calls = []

    def fake_run(func, args):
        calls.append((func, args))
        return func(*args)

    monkeypatch.setattr(parquet_util, '_run_in_isolated_subprocess', fake_run)

    assert parquet_util.is_empty('s3://bucket/key') is False
    assert parquet_util.parquet_num_rows('s3://bucket/key') == 42
    assert len(calls) == 2
    assert calls[0][0] is parquet_util._is_empty_helper
    assert calls[1][0] is parquet_util._parquet_num_rows_helper


def test_gcs_branch_stays_in_process(monkeypatch):
    # Forking a threaded process after creating a GCS filesystem can wedge
    # PyArrow. The poisoned-client isolation is specifically for S3.
    monkeypatch.setattr(parquet_util.fsutil, 'is_s3', lambda path: False)
    monkeypatch.setattr(parquet_util.fsutil, 'is_gcs', lambda path: True)
    monkeypatch.setattr(parquet_util, '_is_empty_helper', lambda path: False)
    monkeypatch.setattr(parquet_util, '_parquet_num_rows_helper',
                        lambda path, skip_check: 42)

    def fail_isolation(*args):
        raise AssertionError('GCS must not use fork isolation')

    monkeypatch.setattr(parquet_util, '_run_in_isolated_subprocess',
                        fail_isolation)

    assert parquet_util.is_empty('gs://bucket/key') is False
    assert parquet_util.parquet_num_rows('gs://bucket/key') == 42


# --- credentialed skip_check count + no-more-masking-403-as-0 regression -----
# The skip_check fast path used to open a dataset on the bare directory prefix,
# whose pyarrow-native HeadObject 403s on cross-account buckets, and the broad
# except masked that as "0 rows". The fix counts explicit *.parquet keys and
# surfaces access/IO errors instead of swallowing them.


def test_num_rows_skip_check_counts_multi_file_dir(tmp_path):
    # skip_check counts each *.parquet key and ignores non-parquet sidecars.
    path = tmp_path.resolve().as_posix()
    table_dir = osp.join(path, 'multi')
    fsutil.makedirs(table_dir)
    pd.DataFrame({
        'a': [1, 2, 3]
    }).to_parquet(osp.join(table_dir, 'part-0.parquet'))
    pd.DataFrame({
        'a': [4, 5]
    }).to_parquet(osp.join(table_dir, 'part-1.parquet'))
    fsutil.touch(osp.join(table_dir, '_SUCCESS'))

    assert parquet_util.parquet_num_rows(table_dir, skip_check=True) == 5
    assert parquet_util.parquet_num_rows(table_dir, skip_check=False) == 5


def test_num_rows_counts_partitioned_dir(tmp_path):
    # A partitioned table stores its .parquet files under partition subdirs.
    # Enumerating only direct-child .parquet keys (non-recursive listdir) would
    # report 0; the recursive dataset count must find the nested files.
    path = tmp_path.resolve().as_posix()
    table_dir = osp.join(path, 'partitioned')
    fsutil.makedirs(osp.join(table_dir, 'part=1'))
    fsutil.makedirs(osp.join(table_dir, 'part=2'))
    pd.DataFrame({
        'a': [1, 2, 3]
    }).to_parquet(osp.join(table_dir, 'part=1', 'data.parquet'))
    pd.DataFrame({
        'a': [4, 5]
    }).to_parquet(osp.join(table_dir, 'part=2', 'data.parquet'))

    assert parquet_util.parquet_num_rows(table_dir, skip_check=True) == 5
    assert parquet_util.parquet_num_rows(table_dir, skip_check=False) == 5


def test_num_rows_raises_on_corrupt_fragment(tmp_path):
    # A corrupt .parquet alongside a good one must RAISE, never be silently
    # pruned and undercounted (the anti-masking guarantee). This would fail
    # against an exclude_invalid_files=True directory count, which returns the
    # good rows and drops the corrupt fragment.
    path = tmp_path.resolve().as_posix()
    table_dir = osp.join(path, 'corrupt')
    fsutil.makedirs(table_dir)
    pd.DataFrame({
        'a': [1, 2, 3]
    }).to_parquet(osp.join(table_dir, 'good.parquet'))
    fsutil.write_binary(osp.join(table_dir, 'bad.parquet'),
                        b'not a parquet file at all, garbage bytes')

    with pytest.raises(pa.lib.ArrowInvalid):
        parquet_util.parquet_num_rows(table_dir, skip_check=True)
    with pytest.raises(pa.lib.ArrowInvalid):
        parquet_util.parquet_num_rows(table_dir, skip_check=False)


def test_cloud_dir_num_rows_reads_footers_concurrently(monkeypatch):
    # The S3/GCS directory branch overlaps the per-fragment footer reads: a
    # serial sum over thousands of parts blows the subprocess timeout. Sizing a
    # barrier to the part count proves every read is in flight at once (a serial
    # count would deadlock the barrier and time out).
    table = 'gs://bucket/big'
    parts = [f'{table}/part-{i}.parquet' for i in range(16)]
    monkeypatch.setattr(parquet_util.fsutil, 'is_s3', lambda _: False)
    monkeypatch.setattr(parquet_util.fsutil, 'is_gcs', lambda _: True)
    monkeypatch.setattr(parquet_util.fsutil, 'isdir', lambda p: p == table)
    monkeypatch.setattr(parquet_util.fsutil, 'listdir', lambda _: parts)

    barrier = threading.Barrier(len(parts), timeout=10)

    def fake_read(path):
        barrier.wait()  # only clears if all reads run at the same time
        return int(path.rsplit('-', 1)[1].removesuffix('.parquet'))

    monkeypatch.setattr(parquet_util, '_cloud_parquet_file_num_rows', fake_read)

    assert parquet_util._count_rows_per_file(table) == sum(range(16))


def test_cloud_dir_num_rows_propagates_fragment_error(monkeypatch):
    # A footer read that raises must surface, never be dropped and undercount.
    table = 'gs://bucket/big'
    parts = [f'{table}/part-{i}.parquet' for i in range(8)]
    monkeypatch.setattr(parquet_util.fsutil, 'is_s3', lambda _: False)
    monkeypatch.setattr(parquet_util.fsutil, 'is_gcs', lambda _: True)
    monkeypatch.setattr(parquet_util.fsutil, 'isdir', lambda p: p == table)
    monkeypatch.setattr(parquet_util.fsutil, 'listdir', lambda _: parts)

    def fake_read(path):
        if path.endswith('part-3.parquet'):
            raise OSError('read failed')
        return 1

    monkeypatch.setattr(parquet_util, '_cloud_parquet_file_num_rows', fake_read)

    with pytest.raises(OSError, match='read failed'):
        parquet_util._count_rows_per_file(table)


def test_cloud_dir_num_rows_recurses_parquet_suffixed_directory(monkeypatch):
    # An object-store "directory" is a key prefix, so a partition dir can itself
    # end in .parquet. It must be recursed into (directory-first), not read as a
    # single footer.
    tree = {
        'gs://bucket/big': ['gs://bucket/big/col=a.parquet'],
        'gs://bucket/big/col=a.parquet':
        ['gs://bucket/big/col=a.parquet/p.parquet'],
    }
    monkeypatch.setattr(parquet_util.fsutil, 'is_s3', lambda _: False)
    monkeypatch.setattr(parquet_util.fsutil, 'is_gcs', lambda _: True)
    monkeypatch.setattr(parquet_util.fsutil, 'isdir', lambda p: p in tree)
    monkeypatch.setattr(parquet_util.fsutil, 'listdir', lambda p: tree[p])

    def fake_read(path):
        assert path == 'gs://bucket/big/col=a.parquet/p.parquet'
        return 5

    monkeypatch.setattr(parquet_util, '_cloud_parquet_file_num_rows', fake_read)

    assert parquet_util._count_rows_per_file('gs://bucket/big') == 5


def test_num_rows_skip_check_propagates_access_error(tmp_path, monkeypatch):
    # An access/read error propagates instead of being masked as 0 rows.
    path = tmp_path.resolve().as_posix()
    table_dir = osp.join(path, 'multi')
    fsutil.makedirs(table_dir)
    pd.DataFrame({
        'a': [1, 2, 3]
    }).to_parquet(osp.join(table_dir, 'part-0.parquet'))

    def boom(p, **kwargs):
        raise OSError("AWS Error ACCESS_DENIED: Forbidden (403)")

    monkeypatch.setattr(parquet_util, '_get_dataset', boom)
    with pytest.raises(OSError, match="ACCESS_DENIED"):
        parquet_util.parquet_num_rows(table_dir, skip_check=True)


def test_is_empty_propagates_access_error(tmp_path, monkeypatch):
    # is_empty must surface an access/IO error, not report the table as empty.
    path = tmp_path.resolve().as_posix()
    table_dir = osp.join(path, 'multi')
    fsutil.makedirs(table_dir)
    pd.DataFrame({
        'a': [1, 2, 3]
    }).to_parquet(osp.join(table_dir, 'part-0.parquet'))

    def boom(p, **kwargs):
        raise OSError("AWS Error ACCESS_DENIED: Forbidden (403)")

    monkeypatch.setattr(parquet_util, '_get_dataset', boom)
    with pytest.raises(OSError, match="ACCESS_DENIED"):
        parquet_util.is_empty(table_dir)


def test_is_empty_propagates_access_error_from_isdir(monkeypatch):
    # The isdir probe is the earliest read; an access/IO failure there must
    # surface too, not be swallowed as "empty".
    def boom(path):
        raise OSError("connection timed out")

    monkeypatch.setattr(parquet_util.fsutil, "isdir", boom)
    with pytest.raises(OSError, match="timed out"):
        parquet_util.is_empty("/tmp/some-table")


def test_is_access_or_io_error_classifier():
    # Access/IO failures are classified by exception type; a missing or
    # parse-error path is not an access error.
    assert parquet_util._is_access_or_io_error(OSError("ACCESS_DENIED 403"))
    assert parquet_util._is_access_or_io_error(ConnectionError("reset"))
    assert parquet_util._is_access_or_io_error(TimeoutError())
    assert not parquet_util._is_access_or_io_error(ValueError("bad footer"))
    assert not parquet_util._is_access_or_io_error(FileNotFoundError("missing"))


def test_streaming_dataframe_writer_appends_and_rotates(tmp_path):
    prefix = str(tmp_path / 'part-00003')
    writer = parquet_util.StreamingDataFrameParquetWriter(
        prefix,
        max_file_size=20,
        max_buffer_size=20,
        save_index=False,
    )
    writer.write(pd.DataFrame({'value': [1, 2]}))
    writer.write(pd.DataFrame({'value': [3, 4]}))
    writer.close()

    files = sorted(tmp_path.glob('part-00003-*.parquet'))
    assert len(files) == 2
    actual = pd.concat([pd.read_parquet(path) for path in files],
                       ignore_index=True)
    pd.testing.assert_frame_equal(actual, pd.DataFrame({'value': [1, 2, 3, 4]}))


def test_streaming_dataframe_writer_slices_before_arrow_conversion(tmp_path):
    writer = parquet_util.StreamingDataFrameParquetWriter(
        str(tmp_path / 'part'),
        max_file_size=1024,
        max_buffer_size=64,
        save_index=False,
    )
    converted_sizes = []
    to_table = writer._to_table

    def record_conversion(df):
        converted_sizes.append(len(df))
        return to_table(df)

    writer._to_table = record_conversion
    source = pd.DataFrame({'value': ['x' * 32] * 20})
    writer.write(source)
    writer.close()

    assert sum(converted_sizes) == len(source)
    assert max(converted_sizes) < len(source)


def test_streaming_dataframe_writer_preserves_contiguous_index(tmp_path):
    prefix = str(tmp_path / 'embedding-part')
    with parquet_util.StreamingDataFrameParquetWriter(
            prefix,
            max_file_size=1024,
            max_buffer_size=1024,
            save_index=True,
    ) as writer:
        writer.write(pd.DataFrame({'value': ['a', 'b']}))
        writer.write(pd.DataFrame({'value': ['c']}))

    actual = pd.read_parquet(f'{prefix}-00000.parquet')
    assert list(actual.index) == [0, 1, 2]
    assert list(actual['value']) == ['a', 'b', 'c']


def test_streaming_dataframe_writer_rejects_column_changes(tmp_path):
    writer = parquet_util.StreamingDataFrameParquetWriter(
        str(tmp_path / 'part'),
        max_file_size=1024,
        max_buffer_size=1024,
        save_index=False,
    )
    writer.write(pd.DataFrame({'value': [1]}))
    with pytest.raises(ValueError, match='schema columns changed'):
        writer.write(pd.DataFrame({'other': [2]}))
    writer.close()


def test_streaming_dataframe_writer_cloud_uses_fsutil_handle(monkeypatch):
    payload = io.BytesIO()
    expected_path = 's3://bucket/part-00000.parquet'
    monkeypatch.setattr(parquet_util.fsutil, 'is_local', lambda _: False)
    monkeypatch.setattr(parquet_util.fsutil, 'is_s3',
                        lambda path: path.startswith('s3://'))
    monkeypatch.setattr(parquet_util.fsutil, 'is_gcs', lambda _: False)

    @contextlib.contextmanager
    def binary_handle(path, mode):
        assert path == expected_path
        assert mode == 'wb'
        yield payload

    monkeypatch.setattr(parquet_util.fsutil, 'get_binary_handle', binary_handle)
    with parquet_util.StreamingDataFrameParquetWriter(
            's3://bucket/part',
            max_file_size=1024,
            max_buffer_size=1024,
            save_index=False,
    ) as writer:
        writer.write(pd.DataFrame({'value': [1, 2]}))
        writer.write(pd.DataFrame({'value': [3]}))

    actual = pd.read_parquet(io.BytesIO(payload.getvalue()))
    pd.testing.assert_frame_equal(actual, pd.DataFrame({'value': [1, 2, 3]}))
