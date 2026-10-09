# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from io_utils import fsutil
from io_utils.fsutil.api import PathDetail
from kumo.connector.file_connector import _scan_dataset_files


def _file(path):
    return PathDetail(path=path, is_directory=False)


def _directory(path):
    return PathDetail(path=path, is_directory=True)


def test_large_scan_uses_listing_types_without_object_stats(monkeypatch):
    root = 's3://customer-data/events'
    parts = [
        _file(f'{root}/part-{index:05d}.parquet')
        for index in reversed(range(10_000))
    ]
    listdir = Mock(return_value=parts)
    isdir = Mock(side_effect=AssertionError('must not stat listed entries'))
    monkeypatch.setattr(fsutil, 'listdir', listdir)
    monkeypatch.setattr(fsutil, 'isdir', isdir)

    files = _scan_dataset_files(root, ('parquet', ))

    assert len(files) == 10_000
    assert files == sorted(files)
    listdir.assert_called_once_with(root, max_items=None, detail=True)
    isdir.assert_not_called()


def test_scan_uses_directory_type_for_nested_and_colliding_paths(monkeypatch):
    root = 'gs://customer-data/events'
    partition = f'{root}/year=2026.parquet'
    children = {
        root: [_file(partition), _directory(partition)],
        partition: [_file(f'{partition}/part-0.parquet')],
    }

    def listdir(path, *, max_items=None, detail=False):
        assert detail is True
        return children[path]

    monkeypatch.setattr(fsutil, 'listdir', listdir)
    monkeypatch.setattr(
        fsutil, 'isdir',
        Mock(side_effect=AssertionError('listing metadata is authoritative')))

    assert _scan_dataset_files(root, ('parquet', )) == [(
        'year=2026.parquet/part-0.parquet',
        f'{partition}/part-0.parquet',
    )]


def test_scan_uses_the_stat_fallback_for_adls(monkeypatch):
    root = 'abfss://data@account.dfs.core.windows.net/events'
    partition = f'{root}/year=2026'
    children = {
        root: [f'{root}/part-0.parquet', partition],
        partition: [f'{partition}/part-1.parquet'],
    }

    def listdir(path, *, max_items=None, detail=False):
        assert detail is False
        assert max_items is None
        return children[path]

    monkeypatch.setattr(fsutil, 'listdir', listdir)
    monkeypatch.setattr(fsutil, 'isdir', lambda path: path in children)

    assert _scan_dataset_files(root, ('parquet', ), max_entries=3) == [
        ('part-0.parquet', f'{root}/part-0.parquet'),
        ('year=2026/part-1.parquet', f'{partition}/part-1.parquet'),
    ]


def test_scan_checks_fallback_limit_before_entry_stats(monkeypatch):
    root = 'abfss://data@account.dfs.core.windows.net/events'
    entries = [f'{root}/part-{index:02d}.parquet' for index in range(10)]
    isdir = Mock(return_value=False)
    monkeypatch.setattr(fsutil, 'listdir', lambda *_args, **_kwargs: entries)
    monkeypatch.setattr(fsutil, 'isdir', isdir)

    with pytest.raises(OverflowError, match='supported entry limit'):
        _scan_dataset_files(root, ('parquet', ), max_entries=3)

    assert isdir.call_count == 3
