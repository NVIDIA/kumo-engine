# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

from kumo.engine.graph import Graph, _warn_ingest_data_loss


def test_snapshot_warns_once_per_table_with_duplicate_primary_keys(tmp_path):
    stats = tmp_path / 'ingest_stats.json'
    stats.write_text(
        '{"stats":{"pk_duplicate_counts":'
        '{"accounts:shard-00000":1,"accounts:shard-00001":2}}}')
    result = SimpleNamespace(ingest_stats_path=str(stats))
    graph_config = {
        'tables': [{
            'table_name': 'accounts',
            'pkey': {'name': 'account_id'},
        }]
    }

    with pytest.warns(UserWarning, match=(
            r"3 duplicate primary-key row\(s\) for "
            r"accounts\.account_id; canonical-first")) as caught:
        _warn_ingest_data_loss(result, graph_config)

    assert len(caught) == 1


def test_snapshot_warns_when_invalid_primary_keys_drop_rows(tmp_path):
    stats = tmp_path / 'ingest_stats.json'
    stats.write_text(
        '{"stats":{"key_failures_by_context":'
        '{"pkey:accounts.account_id":{"parse_failures":1}},'
        '"source_rows_by_table":{"accounts":4}}}')
    result = SimpleNamespace(ingest_stats_path=str(stats))
    graph_config = {
        'tables': [{
            'table_name': 'accounts',
            'pkey': {'name': 'account_id'},
        }]
    }

    with pytest.warns(UserWarning, match=(
            r"dropped 1 row\(s\).*null or invalid primary key in "
            r"accounts\.account_id \(25\.0% of source rows\)")):
        _warn_ingest_data_loss(result, graph_config)


@pytest.mark.parametrize('failure', [None, 'stage', 'ingest', 'cached'])
def test_snapshot_discards_only_owned_staging(tmp_path, monkeypatch, failure):
    import kumo_core.diskgraph.ingest as ingest
    import kumo_core.diskgraph.snapshot as snapshots
    import kumo_core.sources.transfer as transfer
    from filelock import FileLock, Timeout

    import kumo.engine._scratch as scratch

    monkeypatch.setattr(scratch, '_root', tmp_path)
    discard = transfer.discard_destination

    def discard_under_lock(path):
        with pytest.raises(Timeout):
            with FileLock(f'{path}.lock', timeout=0):
                pytest.fail('cleanup must hold the snapshot lock')
        discard(path)

    monkeypatch.setattr(transfer, 'discard_destination', discard_under_lock)

    original = tmp_path / 'customer-source.parquet'
    original.write_bytes(b'original')
    index = tmp_path / 'retained-index'
    index.mkdir()
    (index / 'index').write_bytes(b'index')
    result = SimpleNamespace(index_path=str(index))
    calls = []

    class Source:
        def prepare_ingest(self, connector, table_name, destination, **kwargs):
            dest = Path(destination)
            (dest / 'part.parquet').write_bytes(b'staged')
            calls.append(table_name)
            if failure == 'stage' and table_name == 'second':
                raise ValueError('source failure')
            return SimpleNamespace(credentials=None, version='v1',
                                   source={'path': str(dest)})

    source = Source()
    tables = [
        SimpleNamespace(table_name=name,
                        definition=SimpleNamespace(cols=[], source_table=None),
                        connector=SimpleNamespace(
                            source_type='LOCAL_FILE',
                            source_backend=source,
                            validate_scratch_dir=lambda path: None,
                        ))
        for name in ('first', 'second')
    ]
    graph = Graph(tables=tables)
    graph.__dict__['id'] = 'graph-key'
    graph.__dict__['_graph_config'] = {
        'tables': [{
            'table_name': table.table_name,
            'source_table_name': f'fixture.public.{table.table_name}',
            'pkey': {'name': 'id', 'dtype': 'int64', 'stype': 'ID'},
            'cols': [],
            'fkeys': [],
        } for table in tables]
    }
    monkeypatch.setattr(snapshots, 'check_diskgraph_output', lambda path: result
                        if failure == 'cached' else None)
    monkeypatch.setattr(snapshots.DiskgraphIngestResult, 'for_index',
                        lambda path: SimpleNamespace(to_json=lambda: '{}'))

    def execute(graph_config, spec):
        assert all(Path(path).is_dir() for path in spec.source_paths.values())
        if failure == 'ingest':
            raise ValueError('ingest failure')
        return result

    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest', execute)
    if failure in ('stage', 'ingest'):
        with pytest.raises(ValueError, match='failure'):
            graph.snapshot()
    else:
        assert graph.snapshot().index_path == index
    assert calls == ['first', 'second']
    assert not (tmp_path / 'snapshots' / 'graph-key' / 'source_tables').exists()
    assert not list(tmp_path.rglob('*.discarded-*'))
    assert original.read_bytes() == b'original'
    assert (index / 'index').read_bytes() == b'index'


def test_snapshot_warns_on_foreign_keys_that_mostly_miss(tmp_path, recwarn):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumoapi.graph import Edge
    from kumoapi.table import Column
    from kumoapi.typing import Dtype, Stype

    import kumo.engine as kumo
    from kumo.engine import FileConnector, Table
    from kumo.engine.table import TableDefinition

    kumo.configure(scratch_dir=tmp_path / 'scratch')
    frames = {
        'users': pa.table({'id': range(10)}),
        'items': pa.table({'id': range(5)}),
        'orders': pa.table({
            'id': range(10),
            'user_id': [0, 1, 2] + list(range(100, 107)),
            'item_id': [0, 1, 2, 3, 4] * 2,
        }),
    }
    source = tmp_path / 'inputs'
    source.mkdir()
    connector = FileConnector(root_dir=str(source))
    tables = []
    for name, frame in frames.items():
        pq.write_table(frame, source / f'{name}.parquet')
        tables.append(
            Table(
                connector, name,
                TableDefinition(
                    source_table=connector.source_table_for(name), pkey='id',
                    cols=[
                        Column(name=col, dtype=Dtype.int64, stype=Stype.ID)
                        for col in frame.column_names
                    ])))
    graph = Graph(tables, [
        Edge('orders', 'user_id', 'users'),
        Edge('orders', 'item_id', 'items'),
    ])

    graph.snapshot()

    messages = [
        str(w.message) for w in recwarn.list if 'matched' in str(w.message)
    ]
    assert messages == [
        'Foreign key orders.user_id matched 3 of 10 non-null rows (30.00%) '
        'to users.id; unmatched rows get no edge. Check that both columns '
        'hold the same IDs.'
    ]
