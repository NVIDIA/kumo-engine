# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan
from kumoapi.train import TrainingTableSpec
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.trainer.output_config import TrainingTableOutputConfig

SPLITS = {'train': 0, 'val': 1, 'test': 2}
FREQ = '__kumo_internal_frequency_column__'


@pytest.fixture(scope='module')
def graph(tmp_path_factory):
    root = tmp_path_factory.mktemp('graph')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    rng = np.random.default_rng(0)
    data = root / 'data'
    data.mkdir()
    pd.DataFrame({
        'ID': np.arange(50),
        'LABEL': rng.integers(0, 2, 50).astype(bool),
    }).to_parquet(data / 'ents.parquet')
    pd.DataFrame({'IID': np.arange(20)}).to_parquet(data / 'items.parquet')
    pd.DataFrame({
        'EID':
        np.arange(3000),
        'ENT_ID':
        rng.integers(0, 50, 3000),
        'ITEM_ID':
        rng.integers(0, 20, 3000),
        'TS':
        pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 3000), unit='D'),
    }).to_parquet(data / 'events.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    ents = kumo.Table.from_source_table(connector['ents.parquet'],
                                        primary_key='ID').infer_metadata()
    items = kumo.Table.from_source_table(connector['items.parquet'],
                                         primary_key='IID').infer_metadata()
    events = kumo.Table.from_source_table(
        connector['events.parquet'],
        primary_key='EID', time_column='TS').infer_metadata().with_column(
            'ENT_ID', stype=Stype.ID).with_column('ITEM_ID', stype=Stype.ID)
    graph = kumo.Graph(
        tables={
            'ents': ents,
            'items': items,
            'events': events
        }, edges=[
            Edge(src_table='events', fkey='ENT_ID', dst_table='ents'),
            Edge(src_table='events', fkey='ITEM_ID', dst_table='items'),
        ])
    out = root / 'out'
    out.mkdir()
    yield graph, graph.snapshot(), kumo.FileConnector(root_dir=str(out)), out
    kumo.configure(scratch_dir=previous)


def _spec(table):
    return json.loads((table.staging_dir / 'tt_ingest_spec.json').read_text())


def _seeds(table):
    stages = {s['key']: s for s in _spec(table)['split_stages']}
    return {
        key: pa.concat_tables(
            [pq.read_table(table.staging_dir / p) for p in s['parts']])
        for key, s in stages.items()
    }, {key: s['row_count']
        for key, s in stages.items()}


def _rows(table, columns):
    values = [table[c].to_pylist() for c in columns]
    return Counter(
        tuple(tuple(v) if isinstance(v, list) else v for v in row)
        for row in zip(*values))


def _export(graph, query, name):
    graph, snapshot, connector, out = graph
    pquery = kumo.PredictiveQuery(graph=graph, query=query)
    table = pquery.generate_training_table(snapshot)
    table.export(TrainingTableOutputConfig(connector, name))
    parts = sorted((out / name).glob('*.parquet'))
    return pquery, table, pa.concat_tables([pq.read_table(p) for p in parts])


@pytest.mark.parametrize('name, query', [
    ('temporal', 'PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH ents.ID'),
    ('static', 'PREDICT ents.LABEL FOR EACH ents.ID'),
])
def test_node_export_roundtrip(graph, name, query):
    _, snapshot, connector, out = graph
    pquery, table, exported = _export(graph, query, name)
    seeds, counts = _seeds(table)

    columns = ['ENTITY1', 'TIMESTAMP', 'TARGET1']
    assert exported.schema.names == columns + ['SPLIT']
    assert exported.schema.types == [
        pa.int64(), pa.timestamp('ns'), pa.bool_(), pa.int64()
    ]
    expected = Counter()
    for key, seed in seeds.items():
        expected += _rows(seed.append_column('SPLIT', pa.array(
            [SPLITS[key]] * len(seed), pa.int64())), exported.schema.names)
    assert _rows(exported, exported.schema.names) == expected
    for key, split in SPLITS.items():
        assert exported['SPLIT'].to_pylist().count(split) == counts[key]

    frame = exported.to_pandas()
    frame['WEIGHT'] = 1.0 + frame['ENTITY1'] % 3
    frame.to_parquet(out / f'{name}_w.parquet')
    reimported = kumo.TrainingTable.from_source_table(
        connector[f'{name}_w.parquet'],
        TrainingTableSpec(weight_col='WEIGHT', split_col='SPLIT'),
        pq=pquery, snapshot=snapshot)
    assert _seeds(reimported)[1] == counts

    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    kumo.Trainer(model_plan=plan).fit(reimported, snapshot=snapshot)


def test_link_export_roundtrip(graph):
    _, snapshot, connector, out = graph
    query = ('PREDICT LIST_DISTINCT(events.ITEM_ID, 0, 7, days) '
             'RANK TOP 3 FOR EACH ents.ID')
    pquery, table, exported = _export(graph, query, 'link')
    seeds, counts = _seeds(table)
    assert exported.schema.names == [
        'ENTITY1', 'TIMESTAMP', 'TARGET1', FREQ, 'SPLIT'
    ]
    assert pa.types.is_list(exported.schema.field('TARGET1').type)
    columns = ['ENTITY1', 'TIMESTAMP', 'TARGET1', FREQ]
    assert _rows(exported, columns) == sum(
        (_rows(s, columns) for s in seeds.values()), Counter())

    frame = exported.to_pandas()
    frame.to_parquet(out / 'link_w.parquet')
    reimported = kumo.TrainingTable.from_source_table(
        connector['link_w.parquet'], TrainingTableSpec(split_col='SPLIT'),
        pq=pquery, snapshot=snapshot)
    again, again_counts = _seeds(reimported)
    assert again_counts == counts
    for key, seed in seeds.items():
        assert _rows(again[key], columns) == _rows(seed, columns)


def test_forecasting_export_roundtrip(graph):
    _, snapshot, connector, out = graph
    query = ('PREDICT COUNT(events.*, 0, 7, days) FORECAST 3 TIMEFRAMES '
             'FOR EACH ents.ID')
    pquery, table, exported = _export(graph, query, 'forecast')
    seeds, counts = _seeds(table)
    columns = [
        'ENTITY1', 'TIMESTAMP', 'TARGET1', 'autoregression_labels',
        'future_labels'
    ]
    assert exported.schema.names == columns + ['SPLIT']
    assert _rows(exported, columns) == sum(
        (_rows(seed, columns) for seed in seeds.values()), Counter())
    for key, split in SPLITS.items():
        assert exported['SPLIT'].to_pylist().count(split) == counts[key]

    exported.to_pandas().to_parquet(out / 'forecast_w.parquet')
    reimported = kumo.TrainingTable.from_source_table(
        connector['forecast_w.parquet'], TrainingTableSpec(),
        pq=pquery, snapshot=snapshot)
    assert _seeds(reimported)[1] == counts


def test_reexport_and_trailing_slash(graph):
    _, _, connector, out = graph
    query = 'PREDICT ents.LABEL FOR EACH ents.ID'
    _, table, first = _export(graph, query, 'retry')

    def listing():
        parts = sorted((out / 'retry').glob('*.parquet'))
        return [p.name for p in parts], sum(
            pq.ParquetFile(p).metadata.num_rows for p in parts)

    before = listing()
    config = TrainingTableOutputConfig(connector, 'retry')
    table.export(config)
    table.export(config)
    assert config.table_name == 'retry/'
    assert listing() == before
    assert before[1] == first.num_rows == sum(_seeds(table)[1].values())


@pytest.mark.parametrize('prediction', [False, True])
def test_custom_link_table_ingests_multiple_weighted_parts(
        graph, monkeypatch, prediction):
    from kumo_core.sources import object_store

    from kumo.engine.pquery import _custom_source

    _, snapshot, connector, out = graph
    query = ('PREDICT LIST_DISTINCT(events.ITEM_ID, 0, 7, days) '
             'RANK TOP 3 FOR EACH ents.ID')
    pquery, table, exported = _export(graph, query,
                                      f'multipart_base_{prediction}')
    monkeypatch.setattr(
        object_store, 'materialize_arrow_streams',
        partial(object_store.materialize_arrow_streams, max_rows_per_shard=20))
    stage_source_table = _custom_source.stage_source_table

    def stage_multipart_source(*args, **kwargs):
        path, version = stage_source_table(*args, **kwargs)
        assert len(list(path.glob('*.parquet'))) > 1
        return path, version

    monkeypatch.setattr(_custom_source, 'stage_source_table',
                        stage_multipart_source)
    if prediction:
        source = exported.select(['ENTITY1', 'TIMESTAMP'])
        # Two files force multipart staging for prediction too.
        root = out / 'multipart_prediction'
        root.mkdir()
        middle = source.num_rows // 2
        pq.write_table(source.slice(0, middle), root / 'part-0.parquet')
        pq.write_table(source.slice(middle), root / 'part-1.parquet')
        predicted = kumo.PredictionTable.from_source_table(
            connector['multipart_prediction'], pq=pquery, snapshot=snapshot)
        assert predicted.row_count == source.num_rows
    else:
        weights = pa.array([[float(index + 1) for index in range(len(values))]
                            for values in exported['TARGET1'].to_pylist()],
                           pa.list_(pa.float64()))
        weighted = exported.append_column('WEIGHT', weights)
        pq.write_table(weighted, out / 'multipart_weighted.parquet')
        reimported = kumo.TrainingTable.from_source_table(
            connector['multipart_weighted.parquet'],
            TrainingTableSpec(weight_col='WEIGHT', split_col='SPLIT'),
            pq=pquery, snapshot=snapshot)
        seeds, counts = _seeds(reimported)
        original, original_counts = _seeds(table)
        assert counts == original_counts
        for key in original:
            columns = ['ENTITY1', 'TIMESTAMP', 'TARGET1']
            assert _rows(seeds[key], columns) == _rows(original[key], columns)
            expected = weighted.filter(
                pa.array([
                    split == SPLITS[key]
                    for split in weighted['SPLIT'].to_pylist()
                ]))
            assert _rows(seeds[key],
                         columns + ['WEIGHT']) == _rows(expected,
                                                        columns + ['WEIGHT'])


@pytest.mark.parametrize('unit,divisor', [
    ('seconds', 10**9), ('milliseconds', 10**6), ('microseconds', 10**3),
    ('nanoseconds', 1),
])
def test_custom_training_table_declares_integer_timestamp_unit(graph, unit,
                                                               divisor):
    _, snapshot, connector, out = graph
    query = 'PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH ents.ID'
    pquery, original, exported = _export(graph, query, f'unit_base_{unit}')
    timestamps = exported['TIMESTAMP'].cast(pa.int64()).to_pylist()
    source = exported.set_column(
        exported.schema.get_field_index('TIMESTAMP'), 'TIMESTAMP',
        pa.array([value // divisor for value in timestamps], pa.int64()))
    name = f'unit_{unit}.parquet'
    pq.write_table(source, out / name)
    imported = kumo.TrainingTable.from_source_table(
        connector[name], TrainingTableSpec(split_col='SPLIT', timestamp_unit=unit),
        pq=pquery, snapshot=snapshot)
    seeds, counts = _seeds(imported)
    original_seeds, original_counts = _seeds(original)
    assert counts == original_counts
    for stage in seeds:
        columns = ['ENTITY1', 'TIMESTAMP', 'TARGET1']
        assert _rows(seeds[stage], columns) == _rows(original_seeds[stage], columns)


def test_manifest_and_export_sidecar(graph, tmp_path):
    from kumo_core.training.export import export_train_table

    _, snapshot, connector, out = graph
    query = 'PREDICT ents.LABEL FOR EACH ents.ID'
    pquery, table, _ = _export(graph, query, 'lineage')
    manifest = table.manifest
    assert manifest['id'] == table.tt_id == _spec(table)['tt_id']
    assert manifest['parents'] == {'snapshot': snapshot.id}
    assert manifest['query'] == query
    assert 'rows' not in manifest and 'source' not in manifest
    assert table.source_mode == 'pquery'
    assert manifest['split_seed'] == 0

    directory = out / 'lineage'
    baseline = tmp_path / 'baseline'
    export_train_table(str(table.staging_dir), str(baseline))
    for part in baseline.glob('*.parquet'):
        assert (directory / part.name).read_bytes() == part.read_bytes()
    assert json.loads(
        (directory / '_kumo_manifest.json').read_bytes()) == manifest

    reloaded = kumo.TrainingTable.from_source_table(
        connector['lineage'], TrainingTableSpec(split_col='SPLIT'), pq=pquery,
        snapshot=snapshot)
    manifest = reloaded.manifest
    assert reloaded.source_mode == 'custom_tt'
    assert manifest['parents']['source_table'] == {
        'connector': 'LOCAL_FILE',
        'location': str(directory.resolve()),
        'version': connector.source_backend.get_data_version(
            connector, 'lineage'),
        'version_kind': 'listing',
    }
    assert _seeds(reloaded)[1] == _seeds(table)[1]


def test_custom_table_follows_the_snapshots_record_sources(graph):
    graph_, _, connector, _ = graph
    pquery, _, _ = _export(graph, 'PREDICT ents.LABEL FOR EACH ents.ID',
                           'unlocated')
    try:
        redacted = graph_.snapshot(record_sources=False)
        table = kumo.TrainingTable.from_source_table(
            connector['unlocated'], TrainingTableSpec(split_col='SPLIT'),
            pq=pquery, snapshot=redacted)
    finally:
        graph_.snapshot()
    assert table.manifest['parents']['source_table']['location'] == 'unlocated'


def test_manifest_write_failure_keeps_generation_and_export(
        graph, monkeypatch):
    from kumo.engine.pquery import training_table

    _, _, connector, out = graph

    def fail(*args, **kwargs):
        raise OSError('read-only scratch')

    monkeypatch.setattr(training_table, 'write_manifest', fail)
    with pytest.warns(UserWarning, match='read-only scratch'):
        _, table, exported = _export(
            graph, 'PREDICT ents.LABEL FOR EACH ents.ID', 'no_manifest')
    assert table.manifest is None
    assert not (out / 'no_manifest' / '_kumo_manifest.json').exists()
    assert exported.num_rows == sum(_seeds(table)[1].values())


def test_unreadable_manifest_still_exports(graph):
    graph_, snapshot, connector, out = graph
    table = kumo.PredictiveQuery(
        graph=graph_, query='PREDICT ents.LABEL FOR EACH ents.ID'
    ).generate_training_table(snapshot)
    manifest = table.staging_dir / 'training-table-manifest.json'
    manifest.chmod(0)
    try:
        with pytest.warns(UserWarning, match='_kumo_manifest.json'):
            table.export(TrainingTableOutputConfig(connector, 'unreadable'))
    finally:
        manifest.chmod(0o644)
    assert list((out / 'unreadable').glob('*.parquet'))
    assert not (out / 'unreadable' / '_kumo_manifest.json').exists()


def test_no_sidecar_for_warehouse_or_csv_exports(graph):
    graph_, snapshot, connector, out = graph
    table = kumo.PredictiveQuery(
        graph=graph_, query='PREDICT ents.LABEL FOR EACH ents.ID'
    ).generate_training_table(snapshot)
    calls = []

    def write_files(connector, table_name, files, write_mode, **kwargs):
        calls.append(kwargs)

    warehouse = SimpleNamespace(
        source_type='SNOWFLAKE',
        sink_backend=SimpleNamespace(write_files=write_files))
    table.export(TrainingTableOutputConfig(warehouse, 'DB.SCHEMA.TT'))
    assert calls == [{'export_id': f'{table.tt_id}:tt-export-v1'}]
    table.export(TrainingTableOutputConfig(connector, 'rows.csv/'))
    assert not (out / 'rows.csv' / '_kumo_manifest.json').exists()
