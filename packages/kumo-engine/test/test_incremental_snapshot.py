# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native incremental ingestion: reuse, invalidation, and parent immutability."""
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge

from kumo.engine import FileConnector, Graph, Table, configure


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file()
    }


@pytest.fixture
def graph(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    pq.write_table(pa.table({
        'id': [1, 2, 3],
        'name': ['a', 'b', 'c']
    }), data / 'users.parquet')
    pq.write_table(pa.table({
        'id': [1, 2, 3],
        'price': [10., 20., 30.]
    }), data / 'products.parquet')
    for name in ['events', 'audits']:
        values = {'id': [1, 2, 3], 'user_id': [1, 2, 3]}
        if name == 'events':
            values['product_id'] = [1, 2, 3]
        pq.write_table(pa.table(values), data / f'{name}.parquet')
    configure(scratch_dir=tmp_path / 'scratch')
    connector = FileConnector(root_dir=str(data))
    tables = {
        name: Table.from_source_table(connector[name], primary_key='id')
        for name in ['users', 'products', 'events', 'audits']
    }
    edges = [
        Edge(src_table=name, fkey='user_id', dst_table='users')
        for name in ['events', 'audits']
    ]
    edges.append(
        Edge(src_table='events', fkey='product_id', dst_table='products'))
    return Graph(tables=tables, edges=edges), data


def update_info(snapshot):
    return json.loads(
        (snapshot.index_path / '_incremental/update.json').read_text())


def test_unchanged_native_reuse_is_bit_identical(graph):
    graph, _ = graph
    first = graph.snapshot(enable_updates=True)
    original = hashes(first.index_path)
    second = graph.snapshot(previous_snapshot=first)
    assert second.index_path != first.index_path
    assert update_info(second)['rebuilt_tables'] == []
    assert len(update_info(second)['reused_edges']) == 6
    assert hashes(first.index_path / 'nodes') == hashes(second.index_path /
                                                        'nodes')
    assert hashes(first.index_path) == original


def test_only_changed_table_and_incident_edges_rebuilt(graph):
    graph, data = graph
    first = graph.snapshot(enable_updates=True)
    original = hashes(first.index_path)
    pq.write_table(
        pa.table({
            'id': [1, 2, 3, 4],
            'price': [10., 20., 30., 400.]
        }), data / 'products.parquet')
    second = graph.snapshot(previous_snapshot=first)
    info = update_info(second)
    assert info['rebuilt_tables'] == ['products']
    assert info['reused_tables'] == ['audits', 'events', 'users']
    assert len(info['reused_edges']) == 4
    for name in info['reused_tables']:
        relative = Path('_incremental/encode/_tmp/nodes') / name
        assert hashes(first.index_path / relative) == hashes(second.index_path /
                                                             relative)
    for name in graph._named_tables:
        relative = Path('nodes') / name / 'node_data.ctx'
        assert (first.index_path /
                relative).read_bytes() == (second.index_path /
                                           relative).read_bytes()
    assert hashes(first.index_path) == original


@pytest.mark.parametrize('source_format', [',', ';', 'parquet'])
@pytest.mark.parametrize('shards', [1, 3])
@pytest.mark.parametrize('api', ['update', 'snapshot'])
def test_unchanged_normalized_sources_are_not_restaged(
        tmp_path, monkeypatch, source_format, shards, api):
    import diskgraph
    import pyarrow.csv as csv

    data = tmp_path / 'data'
    data.mkdir()
    events = pa.table({
        'id': ['001', '002', '003'], 'product_id': [1, 1, 2],
        'event_time': ['2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z',
                       '2026-01-03T00:00:00Z'],
        'amount': ['4', '5', '6'], 'tags': ['a|b', 'b|c', 'a|c'],
    })
    if source_format == 'parquet':
        pq.write_table(events, data / 'events.parquet')
    else:
        csv.write_csv(events, data / 'events.csv',
                      write_options=csv.WriteOptions(delimiter=source_format))
    pq.write_table(pa.table({'id': [1, 2], 'price': [10, 20]}),
                   data / 'products.parquet')
    configure(scratch_dir=tmp_path / 'scratch')
    connector = FileConnector(root_dir=str(data))
    table = Table.from_source_table(connector['events'], primary_key='id',
                                    time_column='event_time')
    table = table.with_column('id', dtype='string').with_column(
        'amount', dtype='int64').with_column(
            'tags', stype='multicategorical', multicategorical_delimiter='|')
    graph = Graph(tables={
        'events': table,
        'products': Table.from_source_table(connector['products'],
                                            primary_key='id'),
    }, edges=[Edge(src_table='events', fkey='product_id', dst_table='products')])
    converter = diskgraph.DiskgraphConfigConverter

    class ShardedConverter(converter):
        def convert(self):
            config = super().convert()
            config['num_shards'] = shards
            return config

    monkeypatch.setattr(diskgraph, 'DiskgraphConfigConverter', ShardedConverter)
    first = graph.snapshot(enable_updates=True)
    # Legacy snapshots still exercise the normalized-metadata rebuild path.
    shutil.rmtree(first.index_path / '_incremental/sources')
    original = hashes(first.index_path)
    monkeypatch.setattr(diskgraph, 'DiskgraphConfigConverter', converter)
    staged = []
    prepare = connector.source_backend.prepare_ingest

    def record_stage(connector, name, *args, **kwargs):
        staged.append(name)
        return prepare(connector, name, *args, **kwargs)

    monkeypatch.setattr(connector.source_backend, 'prepare_ingest', record_stage)
    pq.write_table(pa.table({'id': [1, 2], 'price': [11, 20]}),
                   data / 'products.parquet')
    updated = (first.update() if api == 'update' else
               graph.snapshot(previous_snapshot=first))
    assert staged == ['products']
    assert updated.tables_rebuilt == ['products']
    config = json.loads((updated.index_path / 'diskgraph_config.json').read_text())
    assert config['num_shards'] == shards
    full = graph.snapshot(enable_updates=True, force_full_rebuild=True)
    assert updated.validate() == full.validate()
    assert sample_edges(updated, 'events', '001')[1] == 1
    assert sample_edges(updated, 'products', 1, cutoff=1767312000)[1] == (
        sample_edges(full, 'products', 1, cutoff=1767312000)[1])
    assert hashes(first.index_path) == original


def test_forced_encoding_and_legacy_snapshot_error(graph):
    graph, _ = graph
    first = graph.snapshot(enable_updates=True)
    second = graph.snapshot(previous_snapshot=first, force_tables=['events'])
    assert update_info(second)['rebuilt_tables'] == ['events']
    with pytest.raises(ValueError, match='Unknown force_tables'):
        graph.snapshot(previous_snapshot=first, force_tables=['missing'])
    from kumo.engine.graph import GraphSnapshot
    with pytest.raises(ValueError, match='no incremental cache'):
        graph.snapshot(
            previous_snapshot=GraphSnapshot('legacy', first.index_path /
                                            'missing', first.graph_config))


def test_schema_change_refits_only_affected_table(graph):
    graph, data = graph
    first = graph.snapshot(enable_updates=True)
    frame = pq.read_table(data / 'products.parquet')
    pq.write_table(frame.append_column('new_feature', pa.array([1, 2, 3])),
                   data / 'products.parquet')
    tables = dict(graph._named_tables)
    tables['products'] = Table.from_source_table(
        tables['products'].connector['products'], primary_key='id')
    changed = Graph(tables=tables, edges=graph.edges)
    second = changed.snapshot(previous_snapshot=first)
    assert update_info(second)['rebuilt_tables'] == ['products']
    assert len(update_info(second)['reused_edges']) == 4
    full = changed.snapshot(force_full_rebuild=True, enable_updates=True)
    assert (second.index_path /
            'nodes/products/node_data.ctx').read_bytes() == (
                full.index_path / 'nodes/products/node_data.ctx').read_bytes()


@pytest.mark.parametrize('reuse_encoders', [True, False])
def test_full_update_honors_encoder_reuse(graph, reuse_encoders):
    graph, data = graph
    first = graph.snapshot(enable_updates=True)
    pq.write_table(
        pa.table({
            'id': [1, 2, 3, 4],
            'price': [10., 20., 30., 400.]
        }), data / 'products.parquet')
    second = first.update(mode='full', reuse_encoders=reuse_encoders)
    assert set(second.tables_rebuilt) == set(graph._named_tables)
    if reuse_encoders:
        assert (second.index_path /
                'nodes/products/node_data.ctx').read_bytes() == (
                    first.index_path /
                    'nodes/products/node_data.ctx').read_bytes()
    else:
        full = graph.snapshot(force_full_rebuild=True, enable_updates=True)
        assert hashes(second.index_path / 'nodes') == hashes(full.index_path /
                                                             'nodes')


def test_duplicate_keys_and_missing_foreign_keys_keep_counts(graph):
    graph, data = graph
    pq.write_table(pa.table({
        'id': [1, 1, 3],
        'user_id': [1, 2, 99]
    }), data / 'audits.parquet')
    first = graph.snapshot(enable_updates=True)
    second = graph.snapshot(previous_snapshot=first)

    def stats(snapshot):
        return json.loads(
            (snapshot.index_path / 'ingest_stats.json').read_text())['stats']

    for key in [
            'pk_duplicate_counts', 'edge_counts', 'key_parse_failures',
            'key_lookup_failures', 'key_membership_failures',
            'key_failures_by_context'
    ]:
        assert stats(first).get(key) == stats(second).get(key), key


def test_temporal_edges_match_full_build(graph):
    graph, data = graph
    events = pq.read_table(data / 'events.parquet').append_column(
        'event_time',
        pa.array([1000000000, 1000000060, 1000000120], type=pa.timestamp('s')))
    pq.write_table(events, data / 'events.parquet')
    tables = dict(graph._named_tables)
    tables['events'] = Table.from_source_table(
        tables['events'].connector['events'], primary_key='id',
        time_column='event_time')
    graph = Graph(tables=tables, edges=graph.edges)
    first = graph.snapshot(enable_updates=True)
    pq.write_table(pa.table({
        'id': [1, 2, 3],
        'price': [100., 20., 30.]
    }), data / 'products.parquet')
    second = graph.snapshot(previous_snapshot=first, reuse_encoders=False)
    full = graph.snapshot(force_full_rebuild=True, enable_updates=True)
    assert hashes(second.index_path / 'nodes') == hashes(full.index_path /
                                                         'nodes')
    assert len(update_info(second)['reused_edges']) == 4


def test_native_rejects_parent_overlap_before_writing(graph, tmp_path):
    import os
    import subprocess

    graph, _ = graph
    snapshot = graph.snapshot(enable_updates=True)
    original = hashes(snapshot.index_path)
    config = tmp_path / 'incremental.json'
    config.write_text(json.dumps({'parent': str(snapshot.index_path)}))
    for output in [snapshot.index_path, snapshot.index_path / 'missing/deeper']:
        result = subprocess.run([
            os.environ['DISKGRAPH_INGEST_BIN'], '--source',
            str(snapshot.index_path / 'diskgraph_config.json'), '--output',
            str(output), '--incremental-config',
            str(config), '--worker'
        ], capture_output=True, text=True)
        assert result.returncode != 0
        assert 'incremental output' in result.stderr
        assert hashes(snapshot.index_path) == original


def test_fingerprints_and_lineage_are_persisted(graph):
    graph, data = graph
    first = graph.snapshot(enable_updates=True)
    second = graph.snapshot(previous_snapshot=first)
    info = update_info(second)
    assert info['snapshot_id'] == second.id
    assert info['parent_snapshot_id'] == first.id
    assert info['creation_mode'] == 'incremental'
    assert info['update_chain_length'] == 1
    assert all(table['encoder_state'] == 'inherited'
               for table in info['tables'].values())
    fps = json.loads((second.index_path /
                      '_incremental/source_fingerprints.json').read_text())
    assert fps['events']['row_count'] == 3
    assert fps['events']['partition_sizes']['events.parquet'] == (
        data / 'events.parquet').stat().st_size


def sample_edges(snapshot, table, key, cutoff=None, fused=True):
    from diskgraph import SamplerEngine, TimeUnit
    engine = SamplerEngine(str(snapshot.index_path), random_seed=42,
                           num_threads=1)
    metadata = json.loads(
        (snapshot.index_path / 'index_metadata.json').read_text())
    node = next(n for n in metadata['node_types'] if n['name'] == table)
    ids = [
        i for shard in node['shards']
        for i in range(shard['min_id'], shard['max_id'] + 1)
    ]
    mask = engine.canonical_node_mask(table, ids, pa.array([key] * len(ids)))
    seed = [i for i, match in zip(ids, mask) if match]
    assert len(seed) == 1
    plan = {
        'layers': [{
            'fanouts': {},
            'default_fanout': {
                'fanout': 100,
                'mode': 'temporal_most_recent'
                if cutoff is not None else 'deterministic_uniform',
                'absent_key_policy': 'uniform_fallback'
            }
        }]
    }
    options = {} if cutoff is None else {
        'seed_timestamps': {
            table: [cutoff]
        },
        'seed_timestamp_unit': TimeUnit.Seconds
    }
    events = list(
        engine.sample({table: seed}, json.dumps(plan), fused=fused, **options))
    return seed[0], sum(event['batch'].num_rows for event in events
                        if event['type'] == 'edge')


def test_append_stable_ids_and_both_edge_directions(graph):
    graph, data = graph
    first = graph.snapshot(enable_updates=True)
    original = hashes(first.index_path)
    old_id, old_edges = sample_edges(first, 'users', 1)
    frame = pq.read_table(data / 'events.parquet')
    appended = pa.table({'id': [4], 'user_id': [1], 'product_id': [2]})
    pq.write_table(pa.concat_tables([frame, appended]), data / 'events.parquet',
                   row_group_size=3)
    second = first.update(mode='append')
    assert sample_edges(second, 'users', 1) == (old_id, old_edges + 1)
    assert sample_edges(second, 'events', 4)[1] == 2
    assert update_info(second)['tables']['events']['rows_added'] == 1
    assert second.rows_appended == 1
    assert second.duration_seconds > 0
    assert hashes(first.index_path) == original


def test_append_rejects_historical_edits(graph):
    graph, data = graph
    first = graph.snapshot(enable_updates=True)
    frame = pa.table({'id': [1, 2, 3, 4], 'price': [999., 20., 30., 40.]})
    pq.write_table(frame, data / 'products.parquet', row_group_size=3)
    with pytest.raises(ValueError, match='not a proven append'):
        first.update(mode='append')


def test_new_entity_then_event_and_compaction(graph):
    graph, data = graph
    first = graph.snapshot(enable_updates=True)
    users = pq.read_table(data / 'users.parquet')
    pq.write_table(
        pa.concat_tables([users, pa.table({
            'id': [4],
            'name': ['d']
        })]), data / 'users.parquet', row_group_size=3)
    entity = first.update(mode='append')
    new_id, edges = sample_edges(entity, 'users', 4)
    assert edges == 0
    original = hashes(entity.index_path)
    events = pq.read_table(data / 'events.parquet')
    pq.write_table(
        pa.concat_tables(
            [events,
             pa.table({
                 'id': [4],
                 'user_id': [4],
                 'product_id': [2]
             })]), data / 'events.parquet', row_group_size=3)
    linked = entity.update(mode='append')
    assert sample_edges(linked, 'users', 4) == (new_id, 1)
    assert sample_edges(linked, 'users', 4, fused=False) == (new_id, 1)
    assert all(ratio == 0 for ratio in linked.fragmentation.values())
    assert linked.update_history[-1]['creation_mode'] == 'compaction'
    assert hashes(entity.index_path) == original
    assert linked.compact() is linked
    # A later append must resolve keys introduced before compaction.
    events = pq.read_table(data / 'events.parquet')
    with pq.ParquetWriter(data / 'events.parquet', events.schema) as writer:
        writer.write_table(events.slice(0, 3))
        writer.write_table(events.slice(3))
        writer.write_table(
            pa.table({
                'id': [5],
                'user_id': [4],
                'product_id': [3]
            }))
    again = linked.update(mode='append')
    assert sample_edges(again, 'users', 4) == (new_id, 2)
    assert sample_edges(again, 'events', 5)[1] == 2
    assert again.validate()['nodes']['events'] == 5
    # A subsequent edit must retain the rows introduced before compaction.
    pq.write_table(pa.table({
        'id': [1, 2, 3],
        'price': [11., 20., 30.]
    }), data / 'products.parquet')
    rebuilt = again.update()
    assert rebuilt.validate()['nodes']['events'] == 5
    assert sample_edges(rebuilt, 'users', 4)[1] == 2


def test_future_append_is_filtered_before_and_after_compaction(graph):
    graph, data = graph
    events = pq.read_table(data / 'events.parquet').append_column(
        'event_time',
        pa.array([1000000000, 1000000060, 1000000120], type=pa.timestamp('s')))
    pq.write_table(events, data / 'events.parquet')
    tables = dict(graph._named_tables)
    tables['events'] = Table.from_source_table(
        tables['events'].connector['events'], primary_key='id',
        time_column='event_time')
    graph = Graph(tables=tables, edges=graph.edges)
    first = graph.snapshot(enable_updates=True)
    baseline = sample_edges(first, 'users', 1, cutoff=1000000500)
    future = pa.table({
        'id': [4],
        'user_id': [1],
        'product_id': [2],
        'event_time':
        pa.array([1000001000], type=pa.timestamp('s'))
    })
    pq.write_table(pa.concat_tables([events, future]), data / 'events.parquet',
                   row_group_size=3)
    updated = first.update(mode='append')
    assert sample_edges(updated, 'users', 1, cutoff=1000000500) == baseline
    assert sample_edges(updated, 'users', 1,
                        cutoff=1000002000)[1] == baseline[1] + 1
    assert sample_edges(updated, 'users', 1, cutoff=1000000500,
                        fused=False) == baseline


def test_mixed_append_and_edit(graph):
    graph, data = graph
    first = graph.snapshot(enable_updates=True)
    events = pq.read_table(data / 'events.parquet')
    pq.write_table(
        pa.concat_tables(
            [events,
             pa.table({
                 'id': [4],
                 'user_id': [1],
                 'product_id': [2]
             })]), data / 'events.parquet', row_group_size=3)
    pq.write_table(pa.table({
        'id': [1, 2, 3],
        'price': [999., 20., 30.]
    }), data / 'products.parquet')
    updated = first.update()
    intermediate = updated.update_history[1]
    assert intermediate['rebuilt_tables'] == ['products']
    assert intermediate['tables']['events']['status'] == 'reused'
    assert sample_edges(updated, 'events', 4)[1] == 2
    assert updated.validate()['nodes']['events'] == 4
    assert updated.tables_rebuilt == ['products']
    assert updated.rows_appended == 1


@pytest.fixture
def deterministic_torch(monkeypatch, request):
    import torch
    requested_precision = getattr(request, 'param', 'highest')
    previous = torch.are_deterministic_algorithms_enabled()
    precision = torch.get_float32_matmul_precision()
    set_precision = torch.set_float32_matmul_precision
    monkeypatch.setattr(torch, 'set_float32_matmul_precision',
                        lambda _: set_precision(requested_precision))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    try:
        yield requested_precision
    finally:
        torch.use_deterministic_algorithms(previous)
        set_precision(precision)


@pytest.mark.parametrize('update_kind', ['append', 'features', 'expansion'])
@pytest.mark.parametrize('deterministic_torch', ['highest', 'medium'],
                         indirect=True)
def test_updated_graph_serves_new_queries_without_changing_old_results(
        tmp_path, deterministic_torch, update_kind):
    from urllib.parse import urlsplit

    import pandas as pd
    from kumoapi.model_plan import ModelPlan

    from kumo.engine import PredictiveQuery, Trainer

    data = tmp_path / 'data'
    data.mkdir()
    configure(scratch_dir=tmp_path / 'scratch')
    users = pa.table({
        'id':
        range(200),
        'label': [float(i % 7) if i < 180 else None for i in range(200)]
    })
    events = pa.table({
        'id': range(1000),
        'user_id': [i % 200 for i in range(1000)],
        'amount': [float(i % 13) for i in range(1000)]
    })
    pq.write_table(users, data / 'users.parquet')
    pq.write_table(events, data / 'events.parquet')
    connector = FileConnector(root_dir=str(data))
    graph = Graph(
        tables={
            name: Table.from_source_table(connector[name], primary_key='id')
            for name in ['users', 'events']
        }, edges=[Edge(src_table='events', fkey='user_id', dst_table='users')])
    first = graph.snapshot(enable_updates=True)
    query = PredictiveQuery(graph=graph,
                            query='PREDICT users.label FOR EACH users.id')
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.training_job.num_experiments = 1
    plan.training_job.refit_trainval = False
    plan.training_job.manual_seed = 42
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = Trainer(model_plan=plan)
    trainer.fit(query.generate_training_table(first), snapshot=first)
    previous = trainer.predict(query.generate_prediction_table(first),
                               snapshot=first)
    old_files = [Path(urlsplit(path).path) for path in previous.files]
    old_hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in old_files
    ]
    old_graph = hashes(first.index_path)
    if update_kind == 'append':
        extra_users = pa.table({'id': [200], 'label': [None]}, schema=users.schema)
        pq.write_table(pa.concat_tables([users, extra_users]),
                       data / 'users.parquet', row_group_size=200)
        extra_events = pa.table({
            'id': [1000, 1001],
            'user_id': [190, 200],
            'amount': [12., 12.]
        })
        pq.write_table(pa.concat_tables([events, extra_events]),
                       data / 'events.parquet', row_group_size=1000)
        updated = first.update(mode='append')
        full = graph.snapshot(force_full_rebuild=True, enable_updates=True)
    elif update_kind == 'features':
        values = events['amount'].to_pylist()
        values[:5] = [12.] * 5
        pq.write_table(events.set_column(2, 'amount', pa.array(values)), data / 'events.parquet')
        updated = first.update()
        assert update_info(updated)['creation_mode'] == 'feature_update'
        full = updated.update(mode='full')
    else:
        pq.write_table(events.slice(0, 200), data / 'devices.parquet')
        graph = Graph(tables={**graph._named_tables, 'devices': Table.from_source_table(
            connector['devices'], primary_key='id')},
            edges=[*graph.edges, Edge(src_table='devices', fkey='user_id', dst_table='users')])
        updated = graph.snapshot(previous_snapshot=first)
        assert update_info(updated)['creation_mode'] == 'expansion'
        full = updated.update(mode='full')
        query = PredictiveQuery(graph=graph, query=query.query)
        # A trained heterogeneous GNN has no encoder for a new node type,
        # regardless of whether the expanded graph was rebuilt or reused.
        for candidate in (updated, full):
            with pytest.raises(KeyError, match='devices'):
                trainer.predict(query.generate_prediction_table(candidate), snapshot=candidate)
        trainer = Trainer(model_plan=plan)
        trainer.fit(query.generate_training_table(full), snapshot=full)
    appended = trainer.predict(query.generate_prediction_table(updated),
                               snapshot=updated).frame
    rebuilt = trainer.predict(query.generate_prediction_table(full),
                              snapshot=full).frame
    assert previous.num_rows == 20
    assert len(appended) == len(rebuilt) == (21 if update_kind == 'append' else 20)
    if update_kind == 'append':
        assert 200 in appended['ENTITY'].to_list()
    pd.testing.assert_frame_equal(
        appended.sort_values(list(appended.columns)).reset_index(drop=True),
        rebuilt.sort_values(list(rebuilt.columns)).reset_index(drop=True),
        check_exact=True)
    assert updated.validate() == full.validate()
    assert hashes(first.index_path) == old_graph
    assert [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in old_files
    ] == old_hashes


@pytest.mark.parametrize('separator', [',', ';', '\t', '|'])
@pytest.mark.parametrize('mode', ['append', 'auto'])
def test_csv_append_keeps_string_keys_and_rejects_duplicates(
        tmp_path, separator, mode):
    data = tmp_path / 'data'
    data.mkdir()
    path = data / 'users.csv'
    path.write_text('id,name\n001,alice\n002,bob\n'.replace(',', separator))
    configure(scratch_dir=tmp_path / 'scratch')
    connector = FileConnector(root_dir=str(data))
    table = Table.from_source_table(connector['users'], primary_key='id')
    # Declare string IDs explicitly so zero padding is part of the key.
    table = table.with_column('id', dtype='string')
    graph = Graph(tables={'users': table}, edges=[])
    first = graph.snapshot(enable_updates=True)
    original = hashes(first.index_path)
    with path.open('a') as stream:
        stream.write('003,carol\n'.replace(',', separator))
    second = first.update(mode=mode)
    assert second.validate()['nodes']['users'] == 3
    assert sample_edges(second, 'users', '003')[0] == 2
    with path.open('a') as stream:
        stream.write('001,duplicate\n'.replace(',', separator))
    from diskgraph import DiskgraphSystemError
    with pytest.raises(DiskgraphSystemError,
                       match='existing or duplicate primary key'):
        second.update(mode='append')
    assert hashes(first.index_path) == original


def test_new_entity_resolves_orphan_from_previous_append(graph):
    graph, data = graph
    first = graph.snapshot(enable_updates=True)
    events = pq.read_table(data / 'events.parquet')
    pq.write_table(
        pa.concat_tables(
            [events,
             pa.table({
                 'id': [4],
                 'user_id': [99],
                 'product_id': [2]
             })]), data / 'events.parquet', row_group_size=3)
    orphan = first.update(mode='append')
    users = pq.read_table(data / 'users.parquet')
    pq.write_table(
        pa.concat_tables([users, pa.table({
            'id': [99],
            'name': ['new']
        })]), data / 'users.parquet', row_group_size=3)
    with pytest.warns(UserWarning, match='historical orphan'):
        restored = orphan.update()
    assert sample_edges(restored, 'events', 4)[1] == 2
    assert restored.validate()['nodes']['users'] == 4


def test_append_filters_existing_destination_timestamps(graph):
    graph, data = graph
    users = pq.read_table(data / 'users.parquet').append_column(
        'created',
        pa.array([1000001000, 1000000000, 1000000000], type=pa.timestamp('s')))
    pq.write_table(users, data / 'users.parquet')
    tables = dict(graph._named_tables)
    tables['users'] = Table.from_source_table(
        tables['users'].connector['users'], primary_key='id',
        time_column='created')
    graph = Graph(tables=tables, edges=graph.edges)
    events = pa.table({
        'id': range(1, 13),
        'user_id': [1, 2, 3] * 4,
        'product_id': [1, 2, 3] * 4
    })
    pq.write_table(events, data / 'events.parquet')
    first = graph.snapshot(enable_updates=True)
    pq.write_table(
        pa.concat_tables(
            [events,
             pa.table({
                 'id': [13],
                 'user_id': [1],
                 'product_id': [2]
             })]), data / 'events.parquet', row_group_size=12)
    updated = first.update(mode='append')
    assert updated.update_history[-1]['creation_mode'] == 'append'
    full = graph.snapshot(force_full_rebuild=True, enable_updates=True)
    for cutoff, expected in [(1000000500, 1), (1000002000, 2)]:
        assert sample_edges(updated, 'events', 13, cutoff=cutoff)[1] == expected
        assert sample_edges(full, 'events', 13, cutoff=cutoff)[1] == expected
    assert updated.validate() == full.validate()


def test_hive_append_preserves_partition_columns(tmp_path):
    from kumo_core.diskgraph.incremental import (
        detect_append,
        fingerprint_files,
    )

    from kumo.engine.incremental import _stage_additions
    data = tmp_path / 'data/events'
    (data / 'day=1').mkdir(parents=True)
    pq.write_table(pa.table({'id': [1, 2]}), data / 'day=1/part.parquet')
    connector = FileConnector(root_dir=str(data.parent))
    table = Table.from_source_table(connector['events'], primary_key='id')
    graph = Graph(tables={'events': table}, edges=[])
    configure(scratch_dir=tmp_path / 'scratch')
    first = graph.snapshot(enable_updates=True)
    before = fingerprint_files('events', data)
    (data / 'day=2').mkdir()
    pq.write_table(pa.table({'id': [3]}), data / 'day=2/part.parquet')
    parts = detect_append(data, before, fingerprint_files('events', data))
    staged = _stage_additions(parts, table, tmp_path / 'staged')
    assert pq.read_table(staged[0]).to_pydict() == {'id': [3], 'day': [2]}
    updated = first.update(mode='append')
    assert updated.validate()['nodes']['events'] == 3


def test_requested_compaction_without_source_changes(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    pq.write_table(pa.table({'id': [1]}), data / 'users.parquet')
    events = pa.table({'id': range(20), 'user_id': [1] * 20})
    pq.write_table(events, data / 'events.parquet')
    connector = FileConnector(root_dir=str(data))
    graph = Graph(
        tables={
            name: Table.from_source_table(connector[name], primary_key='id')
            for name in ('users', 'events')
        }, edges=[Edge(src_table='events', fkey='user_id', dst_table='users')])
    configure(scratch_dir=tmp_path / 'scratch')
    first = graph.snapshot(enable_updates=True)
    pq.write_table(
        pa.concat_tables([events,
                          pa.table({
                              'id': [20],
                              'user_id': [1]
                          })]), data / 'events.parquet', row_group_size=20)
    appended = first.update(mode='append')
    compacted = appended.update(compact=True)
    assert compacted.id != appended.id
    assert not any(compacted.fragmentation.values())
    assert compacted.validate() == appended.validate()


def test_snapshot_cache_is_explicit(graph, monkeypatch):
    graph, _ = graph
    from kumo.engine.connector.file import _FileReader
    flags = []
    fingerprint = _FileReader.fingerprint

    def record(self, table, **kwargs):
        flags.append(kwargs.get('include_row_groups', True))
        return fingerprint(self, table, **kwargs)

    monkeypatch.setattr(_FileReader, 'fingerprint', record)
    ordinary = graph.snapshot()
    assert flags and not any(flags)
    assert not (ordinary.index_path / '_incremental').exists()
    with pytest.raises(ValueError, match='enable_updates=True'):
        ordinary.update()
    enabled = graph.snapshot(enable_updates=True)
    assert enabled.id != ordinary.id
    assert (enabled.index_path / '_incremental/manifest.json').is_file()
    assert hashes(enabled.index_path / 'nodes') == hashes(ordinary.index_path /
                                                          'nodes')


def test_entity_only_compaction_consolidates_keys_and_shards(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    path = data / 'users.csv'
    path.write_text('id,name\n1,first\n')
    connector = FileConnector(root_dir=str(data))
    graph = Graph(
        tables={
            'users': Table.from_source_table(connector['users'],
                                             primary_key='id')
        }, edges=[])
    configure(scratch_dir=tmp_path / 'scratch')
    first = graph.snapshot(enable_updates=True)
    original = hashes(first.index_path)
    current = first
    for identifier in range(2, 6):
        with path.open('a') as stream:
            stream.write(f'{identifier},first\n')
        current = current.update(mode='append')
    prior_ids = {i: sample_edges(current, 'users', i)[0] for i in range(1, 6)}
    compacted = current.update(compact=True)
    manifest = json.loads(
        (compacted.index_path /
         '_incremental/deltas.json').read_text())['tables']['users']
    assert len(manifest['primary_key_logs']) == 1
    assert len(manifest['shard_layouts']) == manifest['base_shards'] + 1
    assert {
        i: sample_edges(compacted, 'users', i)[0]
        for i in range(1, 6)
    } == prior_ids
    with path.open('a') as stream:
        stream.write('6,first\n')
    later = compacted.update(mode='append')
    assert later.validate()['nodes']['users'] == 6
    assert {
        i: sample_edges(later, 'users', i)[0]
        for i in range(1, 6)
    } == prior_ids
    assert hashes(first.index_path) == original


@pytest.mark.parametrize('extension', ['parquet', 'csv'])
def test_append_advances_temporal_queries_and_preserves_parent(
        tmp_path, extension):
    from datetime import datetime, timedelta

    import pyarrow.csv as csv

    from kumo.engine import PredictiveQuery

    configure(scratch_dir=tmp_path / 'scratch')
    data = tmp_path / 'data'
    data.mkdir()
    pq.write_table(pa.table({'id': [1, 2]}), data / 'users.parquet')
    start = datetime(2026, 1, 1)
    events = pa.table({
        'id': range(20),
        'user_id': [1, 2] * 10,
        'ts': [start + timedelta(days=i) for i in range(20)]
    })

    def write(table):
        if extension == 'csv':
            csv.write_csv(table, data / 'events.csv')
        else:
            pq.write_table(table, data / 'events.parquet', row_group_size=20)

    write(events)
    connector = FileConnector(root_dir=str(data))
    graph = Graph(
        tables={
            'users':
            Table.from_source_table(connector['users'], primary_key='id'),
            'events':
            Table.from_source_table(connector['events'], primary_key='id',
                                    time_column='ts')
        }, edges=[Edge(src_table='events', fkey='user_id', dst_table='users')])
    query = PredictiveQuery(
        graph=graph,
        query='PREDICT COUNT(events.*, 0, 1, days) > 0 FOR EACH users.id')
    first = graph.snapshot(enable_updates=True)
    previous_times = query.generate_prediction_table(first).timestamps
    original = hashes(first.index_path)
    # One new maximum and one late arrival exercise both timestamp bounds.
    write(
        pa.concat_tables([
            events,
            pa.table({
                'id': [20, 21],
                'user_id': [1, 2],
                'ts': [start + timedelta(days=20), start - timedelta(days=1)]
            })
        ]))
    updated = first.update(mode='append')
    full = graph.snapshot(force_full_rebuild=True, enable_updates=True)
    assert query.generate_prediction_table(
        updated).timestamps == query.generate_prediction_table(full).timestamps
    assert query.generate_prediction_table(updated).timestamps != previous_times
    for kind in ['full', 'min']:
        relative = f'kumo_col_stats/col_stats_{kind}/events.json'
        actual = json.loads((updated.index_path / relative).read_text())
        expected = json.loads((full.index_path / relative).read_text())
        for metric in ['MIN', 'MAX']:
            key = f'ts__kumo.timestamp__{metric}__False'
            assert actual[key]['value'] == expected[key]['value']
    compacted = updated.compact()
    assert query.generate_prediction_table(
        compacted).timestamps == query.generate_prediction_table(
            full).timestamps
    assert hashes(first.index_path) == original
    for name in ['users', 'events']:
        relative = f'nodes/{name}/node_data.ctx'
        assert (first.index_path /
                relative).read_bytes() == (updated.index_path /
                                           relative).read_bytes()


@pytest.mark.parametrize('addition', [[None, 3], [], [3]])
def test_append_invalid_keys_and_empty_partitions(tmp_path, addition):
    configure(scratch_dir=tmp_path / 'scratch')
    root = tmp_path / 'data' / 'users'
    root.mkdir(parents=True)
    schema = pa.schema([('id', pa.int64())])
    pq.write_table(pa.table({'id': [1, 2]}, schema=schema), root / '0.parquet')
    connector = FileConnector(root_dir=str(root.parent))
    graph = Graph(
        tables={
            'users': Table.from_source_table(connector['users'],
                                             primary_key='id')
        }, edges=[])
    first = graph.snapshot(enable_updates=True)
    original = hashes(first.index_path)
    pq.write_table(pa.table({'id': []}, schema=schema), root / '1.parquet')
    pq.write_table(pa.table({'id': addition}, schema=schema),
                   root / '2.parquet')
    if None in addition:
        with pytest.raises(Exception, match='requires full table encoding'):
            first.update(mode='append')
    updated = first.update()
    full = graph.snapshot(force_full_rebuild=True, enable_updates=True)
    assert updated.validate() == full.validate()
    assert updated.update().id == updated.id
    assert hashes(first.index_path) == original


@pytest.mark.parametrize('incompatible_cache', [False, True])
def test_rebuild_stages_only_needed_tables_and_checks_native_reuse(
        graph, monkeypatch, incompatible_cache):
    from kumo.engine.connector.file import _FileReader

    graph, data = graph
    first = graph.snapshot(enable_updates=True)
    # Legacy snapshots still exercise the normalized-metadata rebuild path.
    shutil.rmtree(first.index_path / '_incremental/sources')
    if incompatible_cache:
        path = first.index_path / '_incremental/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['num_shards'] += 1
        path.write_text(json.dumps(manifest))
    pq.write_table(pa.table({
        'id': [1, 2, 3],
        'price': [11., 20., 30.]
    }), data / 'products.parquet')
    prepared = []
    prepare = _FileReader.prepare

    def record(self, table, *args, **kwargs):
        prepared.append(table)
        return prepare(self, table, *args, **kwargs)

    monkeypatch.setattr(_FileReader, 'prepare', record)
    updated = first.update()
    assert set(prepared) == (set(graph._named_tables)
                             if incompatible_cache else {'products'})
    full = graph.snapshot(force_full_rebuild=True, enable_updates=True)
    assert updated.validate() == full.validate()


def test_append_category_drift_stops_tracking_after_warning_threshold(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kumo.engine import incremental

    key = 'category__kumo.categorical__CATEGORY_COUNTS__False'
    path = tmp_path / 'kumo_col_stats/col_stats_full'
    path.mkdir(parents=True)
    (path / 'users.json').write_text(
        json.dumps({key: {
            'value': [(str(i), 1) for i in range(100)]
        }}))
    incoming = tmp_path / 'incoming.parquet'
    pq.write_table(pa.table({'category': [str(i) for i in range(10000)]}),
                   incoming)
    observed = []
    monkeypatch.setattr(incremental, 'warn_encoder_drift',
                        lambda name, old, new: observed.append(new))
    incremental._warn_append_drift(SimpleNamespace(index_path=tmp_path),
                                   'users', [incoming])
    assert len(
        observed[0][key]['value']
    ) == 6  # Enough to prove >5%, bounded independently of incoming cardinality.


@pytest.mark.parametrize('separator', [',', ';'])
@pytest.mark.parametrize('new_partition', [False, True])
def test_csv_append_normalization_matches_full_decoded_features(
        tmp_path, separator, new_partition):
    import csv

    from diskgraph import SamplerEngine

    configure(scratch_dir=tmp_path / 'scratch')
    data = tmp_path / 'data'
    users = data / 'users'
    users.mkdir(parents=True)
    source = users / '0.csv'

    def write(path, rows, mode='w'):
        with path.open(mode, newline='') as stream:
            writer = csv.writer(stream, delimiter=separator)
            if mode == 'w':
                writer.writerow(['id', 'amount'])
            writer.writerows(rows)

    write(source, [('001', 1), ('002', 2)])
    connector = FileConnector(root_dir=str(data))
    table = Table.from_source_table(connector['users'], primary_key='id')
    table = table.with_column('id', dtype='string').with_column(
        'amount', dtype='int64')
    graph = Graph(tables={'users': table}, edges=[])
    first = graph.snapshot(enable_updates=True)
    original = hashes(first.index_path)
    additions = [('003', '3.6'), ('004', '1_234.6'), ('005', '+5.6'),
                 ('006', '-3.6'), ('007', 'NA'), ('008', '9007199254740993')]
    write(users / '1.csv' if new_partition else source, additions,
          'w' if new_partition else 'a')
    updated = first.update(mode='append')
    full = graph.snapshot(force_full_rebuild=True, enable_updates=True)

    def decoded(snapshot):
        engine = SamplerEngine(str(snapshot.index_path), random_seed=42,
                               num_threads=1)
        rows = []
        for event in engine.fetch_features([{
                'node_type': 'users',
                'node_ids': list(range(8)),
                'feature_names': []
        }]):
            if event.get('batch') is not None:
                rows.extend(event['batch'].to_pylist())
        return sorted(rows, key=lambda row: row['id'])

    assert decoded(updated) == decoded(full) == [{
        'id': key,
        'amount': value
    } for key, value in zip(
        ['001', '002', '003', '004', '005', '006', '007', '008'],
        [1, 2, 4, 1235, 6, -4, None, 9007199254740993])]
    assert hashes(first.index_path) == original
