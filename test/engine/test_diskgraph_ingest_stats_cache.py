# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import types

import pytest
from kumo_core.types import TableStatType
from kumoapi.data_source import ConnectorType

import kumo.diskgraph.ingest_stats_cache as stats_cache
from rest.config.table_v2 import CustomSQLSourceTable


class _TableCacheDir:
    def __init__(self, root):
        self.col_stats = str(root / 'col_stats')
        self.min_col_stats = str(root / 'min_col_stats')


class _Cache:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.edge_stats_path_called = False
        self.graph_cache = self
        self.table_stats = None
        self.added_table_stats = None

    def create_table_cache_dir(self, *_args):
        return _TableCacheDir(self.tmp_path)

    def edge_stats_path(self, *_args):
        self.edge_stats_path_called = True
        return str(self.tmp_path / 'edge_stats')

    def save_table_stats(self, _fingerprint, _version, value):
        self.table_stats = value

    def add_table_stats(self, fingerprint, value):
        self.added_table_stats = (fingerprint, value)


class _Table:
    source_table = CustomSQLSourceTable(
        connector_id='conn',
        connector_type=ConnectorType.DATABRICKS,
        sql='SELECT id FROM users',
        table='users',
    )

    def __init__(self, source_name='users'):
        self.source_name = source_name

    def get_connector_id(self):
        return 'conn'

    def get_source_table_name(self):
        return self.source_name

    def fingerprint(self):
        return 'fp'


class _FileTable(_Table):
    source_table = object()


def _install_diskgraph(monkeypatch, edge_stats):
    diskgraph = types.SimpleNamespace(
        read_col_stats=lambda *_args: '{}',
        read_edge_stats=lambda *_args: edge_stats,
    )
    monkeypatch.setitem(sys.modules, 'diskgraph', diskgraph)


def _write_ingest_stats(tmp_path):
    path = tmp_path / 'ingest_stats.json'
    path.write_text(
        json.dumps({
            'stats': {
                'source_rows_by_table': {
                    'users': 10,
                    'loans': 5,
                },
                'source_bytes_by_table': {
                    'users': 1024,
                    'loans': 512,
                },
            },
        }))
    return str(path)


def test_no_edge_graph_does_not_require_edge_stats(monkeypatch, tmp_path):
    _install_diskgraph(monkeypatch, edge_stats=None)
    cache = _Cache(tmp_path)
    monkeypatch.setattr(stats_cache, 'get_active_cache', lambda: cache)

    stats_cache.copy_ingest_stats_to_backend_cache(
        index_path='/index',
        ingest_stats_path=_write_ingest_stats(tmp_path),
        graph_key='graph',
        graph_data_version='v1',
        tables={'users': _Table()},
        table_data_versions={'users': 'tv1'},
        has_edges=False,
    )

    assert (tmp_path / 'col_stats').read_text() == '{}'
    # The ingest report no longer republishes source row/byte counts: every
    # source is staged before ingest and counted there.
    assert cache.table_stats is None
    assert not cache.edge_stats_path_called


def test_edge_graph_still_requires_edge_stats(monkeypatch, tmp_path):
    _install_diskgraph(monkeypatch, edge_stats=None)
    monkeypatch.setattr(stats_cache, 'get_active_cache',
                        lambda: _Cache(tmp_path))

    with pytest.raises(RuntimeError, match='edge_stats.json'):
        stats_cache.copy_ingest_stats_to_backend_cache(
            index_path='/index',
            ingest_stats_path=_write_ingest_stats(tmp_path),
            graph_key='graph',
            graph_data_version='v1',
            tables={'users': _Table()},
            table_data_versions={'users': 'tv1'},
            has_edges=True,
        )


@pytest.mark.parametrize(
    'edge_key',
    [
        'loans__conn__users__conn__user_id__user_id__EDGE_COUNT',
        'warehouse/loans__conn__warehouse/users__conn__user_id__user_id'
        '__EDGE_COUNT',
    ],
)
def test_edge_stats_use_source_table_names(monkeypatch, tmp_path, edge_key):
    edge_stats = json.dumps({
        edge_key: {
            'value': 3,
        },
    })
    _install_diskgraph(monkeypatch, edge_stats=edge_stats)
    monkeypatch.setattr(stats_cache, 'get_active_cache',
                        lambda: _Cache(tmp_path))

    stats_cache.copy_ingest_stats_to_backend_cache(
        index_path='/index',
        ingest_stats_path=_write_ingest_stats(tmp_path),
        graph_key='graph',
        graph_data_version='v1',
        tables={
            'loans': _Table('warehouse/loans'),
            'users': _Table('warehouse/users'),
        },
        table_data_versions={
            'loans': 'tv1',
            'users': 'tv1',
        },
    )

    written = json.loads((tmp_path / 'edge_stats').read_text())
    assert list(written) == [
        'warehouse/loans__conn__warehouse/users__conn__user_id__user_id'
        '__EDGE_COUNT',
    ]


def test_num_rows_backfilled_from_diskgraph_node_counts(
        monkeypatch, tmp_path):
    cache = _Cache(tmp_path)
    monkeypatch.setattr(stats_cache, 'get_active_cache', lambda: cache)
    ingest_stats_path = tmp_path / 'ingest_stats.json'
    ingest_stats_path.write_text(
        json.dumps({'stats': {
            'node_counts': {
                'users': 17,
            },
        }}))

    stats_cache._save_num_rows_from_node_counts(
        str(ingest_stats_path),
        {'users': _FileTable()},
        {'users': 'tv1'},
    )

    fingerprint, written = cache.added_table_stats
    assert fingerprint == 'fp'
    assert written[TableStatType.NUM_ROWS].value == 17
    assert written[TableStatType.NUM_ROWS].approx is False
    assert written[TableStatType.NUM_ROWS].data_version == 'tv1'


def test_duplicate_counts_returned_without_rewriting_pkey_stats(
        monkeypatch, tmp_path):
    def col_stats(num_unique):
        return json.dumps({
            'id__ID__COUNT__False': {
                'value': 100,
                'approx': False,
                'data_version': 'tv1',
            },
            'id__ID__NUM_UNIQUE__False': {
                'value': num_unique,
                'approx': False,
                'data_version': 'tv1',
            },
            'id__ID__UNIQUE_FRACTION__False': {
                'value': num_unique / 100,
                'approx': False,
                'data_version': 'tv1',
            },
        })

    diskgraph = types.SimpleNamespace(
        read_col_stats=lambda *_args: col_stats(99),
        read_edge_stats=lambda *_args: None,
    )
    monkeypatch.setitem(sys.modules, 'diskgraph', diskgraph)

    class Cache(_Cache):
        def create_table_cache_dir(self, *args):
            source = args[1]
            root = self.tmp_path / source
            root.mkdir(exist_ok=True)
            return _TableCacheDir(root)

    class Table:
        source_table = object()

        def __init__(self, source):
            self.source = source
            self.table_name = source
            self.name = source
            self.data_version = 'tv1'

        def get_connector_id(self):
            return 'conn'

        def get_source_table_name(self):
            return self.source

        def fingerprint(self):
            return types.SimpleNamespace(
                connector_name='conn',
                source_name=self.source,
            )

    ingest_stats_path = tmp_path / 'ingest_stats.json'
    ingest_stats_path.write_text(
        json.dumps({
            'stats': {
                'pk_duplicate_counts': {
                    'users': 1,
                    'users:shard-00001': 2,
                    'users:shard-00002': 3,
                },
            },
        }))
    cache = Cache(tmp_path)
    monkeypatch.setattr(stats_cache, 'get_active_cache', lambda: cache)

    duplicate_counts = stats_cache.copy_ingest_stats_to_backend_cache(
        index_path=str(tmp_path),
        ingest_stats_path=str(ingest_stats_path),
        graph_key='graph',
        graph_data_version='v1',
        tables={
            'users': Table('users'),
            'events': Table('events'),
        },
        table_data_versions={
            'users': 'tv1',
            'events': 'tv1',
        },
        has_edges=False,
    )
    assert duplicate_counts == {'users': 6, 'events': 0}

    for filename in ('col_stats', 'min_col_stats'):
        users = json.loads((tmp_path / 'users' / filename).read_text())
        assert users['id__ID__NUM_UNIQUE__False']['value'] == 99
        assert users['id__ID__UNIQUE_FRACTION__False']['value'] == 0.99

        events = json.loads((tmp_path / 'events' / filename).read_text())
        assert events['id__ID__NUM_UNIQUE__False']['value'] == 99
        assert events['id__ID__UNIQUE_FRACTION__False']['value'] == 0.99
