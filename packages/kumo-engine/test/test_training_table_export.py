# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections import Counter
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumoapi.train import TrainingTableSpec
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._manifest import SourceTableEntry, TrainingTableParents
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.trainer.output_config import TrainingTableOutputConfig

SPLITS = {'train': 0, 'val': 1, 'test': 2}
FREQ = '__kumo_internal_frequency_column__'
LABEL = 'PREDICT ents.LABEL FOR EACH ents.ID'


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


def _generate(graph, **kwargs):
    graph, snapshot, _, _ = graph
    pquery = kumo.PredictiveQuery(graph=graph, query=LABEL)
    return pquery, pquery.generate_training_table(snapshot, **kwargs)


def _export(graph, query, name, plan=None):
    graph, snapshot, connector, out = graph
    pquery = kumo.PredictiveQuery(graph=graph, query=query)
    table = pquery.generate_training_table(snapshot, plan)
    table.export(TrainingTableOutputConfig(connector, name))
    parts = sorted((out / name).glob('*.parquet'))
    return pquery, table, pa.concat_tables([pq.read_table(p) for p in parts])


@pytest.mark.parametrize('name, query', [
    ('temporal', 'PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH ents.ID'),
    ('static', LABEL),
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


@pytest.mark.parametrize('reload', [False, True])
@pytest.mark.parametrize('window', ['7, days', '1, months'])
@pytest.mark.parametrize('invalid_targets', [False, True])
def test_link_export_roundtrip(graph, reload, window, invalid_targets,
                               tmp_path):
    base, _, connector, out = graph
    catalog = pd.read_parquet(out.parent / 'data/events.parquet')[['EID', 'TS']]
    catalog['END'] = catalog['TS'] + pd.Timedelta(days=10)
    catalog.loc[catalog['EID'] % 3 != 0, 'END'] = pd.NaT
    catalog.to_parquet(tmp_path / 'items.parquet')
    items = kumo.Table.from_source_table(
        kumo.FileConnector(root_dir=str(tmp_path))['items.parquet'],
        primary_key='EID', time_column='TS',
        end_time_column='END').infer_metadata()
    base = replace(base, tables=dict(base.tables, items=items))
    snapshot = base.snapshot()
    query = (f'PREDICT LIST_DISTINCT(events.ITEM_ID, 0, {window}) '
             'RANK TOP 3 FOR EACH ents.ID')
    name = f'link_{reload}_{invalid_targets}_{window.split(",")[1].strip()}'
    plan = TrainingTableGenerationPlan(
        timeframe_step=1, split='ApproxDateOffsetSplit([0.6, 0.2, 0.2])')
    pquery, table, exported = _export((base, snapshot, connector, out), query,
                                      name, plan)
    seeds, counts = _seeds(table)
    assert exported.schema.names == [
        'ENTITY1', 'TIMESTAMP', 'TARGET1', FREQ, 'SPLIT'
    ]
    assert pa.types.is_list(exported.schema.field('TARGET1').type)
    columns = ['ENTITY1', 'TIMESTAMP', 'TARGET1', FREQ]
    assert _rows(exported, columns) == sum(
        (_rows(s, columns) for s in seeds.values()), Counter())

    frame = exported.to_pandas()
    if invalid_targets:
        saw_future = saw_expired = False
        end, unit = window.split(', ')
        offset = pd.DateOffset(**{unit: int(end)})
        for i, row in frame.iterrows():
            future = catalog.loc[catalog['TS'] > row['TIMESTAMP'] + offset,
                                 'EID'].tolist()
            expired = catalog.loc[catalog['END'] <= row['TIMESTAMP'],
                                  'EID'].tolist()
            saw_future |= bool(future)
            saw_expired |= bool(expired)
            invalid = [999999] + future[:1] + expired[:1]
            frame.at[i, 'TARGET1'] = list(row['TARGET1']) + invalid
            frame.at[i, FREQ] = list(row[FREQ]) + [999] * len(invalid)
        assert saw_future and saw_expired
    frame.to_parquet(out / 'link_w.parquet')
    if reload:
        load = partial(kumo.TrainingTable.load_from_connector, connector,
                       path='link_w.parquet', pquery=pquery)
    else:
        load = partial(kumo.TrainingTable.from_source_table,
                       connector['link_w.parquet'],
                       TrainingTableSpec(split_col='SPLIT'), pq=pquery)
    cfg = _volume_root(tmp_path)
    with kumo.mlops(cfg):
        reimported = load(snapshot=snapshot)
        assert reimported.manifest.parents.source_table.version
        assert load(snapshot=snapshot).id == reimported.id
    stored = kumo.TrainingTable.load_from_connector(
        cfg, path=f'training_tables/{reimported.id}', pquery=pquery,
        snapshot=snapshot)
    assert _seeds(stored) == _seeds(reimported)
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


def test_reexport_and_trailing_slash(graph, monkeypatch, caplog):
    import kumo_core.artifact_export

    _, _, connector, out = graph
    query = 'PREDICT ents.LABEL FOR EACH ents.ID'
    _, table, first = _export(graph, query, 'retry')

    def listing():
        parts = sorted((out / 'retry').glob('*.parquet'))
        return [p.name for p in parts], sum(
            pq.ParquetFile(p).metadata.num_rows for p in parts)

    before = listing()
    config = TrainingTableOutputConfig(connector, 'retry')
    monkeypatch.setattr(kumo_core.artifact_export, 'export_selected_files',
                        lambda *a, **k: pytest.fail('uploaded again'))
    with caplog.at_level('INFO', logger='kumo_core.object_store_sink'):
        table.export(config)
        table.export(config)
    assert 'reused it' in caplog.text
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
    pquery, table, _ = _export(graph, LABEL, 'lineage')
    manifest = table.manifest
    assert manifest.id == table.id and table.id.startswith('tt-')
    assert table.id != _spec(table)['tt_id']
    assert manifest.parents == TrainingTableParents(
        snapshot=snapshot.manifest)
    assert manifest.query == LABEL
    assert manifest.extra == {}
    assert manifest.split_seed == 0
    again = pquery.generate_training_table(snapshot)
    assert again.id == table.id
    assert again.tt_id != table.tt_id

    directory = out / 'lineage'
    baseline = tmp_path / 'baseline'
    export_train_table(str(table.staging_dir), str(baseline))
    for part in baseline.glob('*.parquet'):
        assert (directory / part.name).read_bytes() == part.read_bytes()
    assert json.loads(
        (directory / '_provenance.json').read_bytes()) == manifest.to_dict()

    reloaded = kumo.TrainingTable.load_from_connector(
        connector, path='lineage', pquery=pquery, snapshot=snapshot)
    manifest = reloaded.manifest
    assert reloaded.id != table.id
    assert manifest.parents.source_table == SourceTableEntry(
        connector='LOCAL_FILE',
        location=str(directory.resolve()),
        version=connector.source_backend.get_data_version(
            connector, 'lineage'),
        version_kind='listing',
    )
    assert _seeds(reloaded)[1] == _seeds(table)[1]


def test_custom_table_follows_the_snapshots_record_sources(
        graph, monkeypatch):
    graph_, snapshot, connector, _ = graph
    pquery, _, _ = _export(graph, LABEL, 'unlocated')

    def location(snapshot):
        table = kumo.TrainingTable.from_source_table(
            connector['unlocated'], TrainingTableSpec(split_col='SPLIT'),
            pq=pquery, snapshot=snapshot)
        return table.manifest.parents.source_table.location

    try:
        assert location(graph_.snapshot(record_sources=False)) == 'unlocated'
    finally:
        graph_.snapshot()
    assert location(snapshot) != 'unlocated'
    monkeypatch.setattr(kumo.GraphSnapshot, 'manifest', property(lambda _: None))
    assert location(replace(snapshot, record_sources=None)) == 'unlocated'


def test_manifest_write_failure_keeps_generation_and_export(
        graph, monkeypatch):
    from kumo.engine.pquery import training_table

    out = graph[3]
    monkeypatch.setattr(training_table, 'write_manifest',
                        Mock(side_effect=OSError('read-only scratch')))
    with pytest.warns(UserWarning, match='read-only scratch'):
        _, table, exported = _export(graph, LABEL, 'no_manifest')
    assert table.manifest is None
    assert not (out / 'no_manifest' / '_provenance.json').exists()
    assert exported.num_rows == sum(_seeds(table)[1].values())


def test_unreadable_manifest_still_exports(graph):
    _, _, connector, out = graph
    _, table = _generate(graph)
    manifest = table.staging_dir / '_provenance.json'
    manifest.chmod(0)
    try:
        with pytest.warns(UserWarning, match='_provenance.json'):
            table.export(TrainingTableOutputConfig(connector, 'unreadable'))
    finally:
        manifest.chmod(0o644)
    assert list((out / 'unreadable').glob('*.parquet'))
    assert not (out / 'unreadable' / '_provenance.json').exists()


def test_no_sidecar_for_warehouse_or_csv_exports(graph):
    _, _, connector, out = graph
    _, table = _generate(graph)
    write_files = Mock()
    warehouse = SimpleNamespace(
        source_type='SNOWFLAKE',
        sink_backend=SimpleNamespace(write_files=write_files))
    table.export(TrainingTableOutputConfig(warehouse, 'DB.SCHEMA.TT'))
    assert [call.kwargs for call in write_files.call_args_list] == [{
        'export_id': f'{table.tt_id}:tt-export-v1'}]
    table.export(TrainingTableOutputConfig(connector, 'rows.csv/'))
    assert not (out / 'rows.csv' / '_provenance.json').exists()


def test_the_store_spills_at_most_two_parts(graph, tmp_path, monkeypatch):
    from kumo_core import object_store_sink
    from kumo_core.training import export

    stage_parts = export.stage_parts
    monkeypatch.setattr(
        export, 'stage_parts', lambda spec: {
            split: parts * 4
            for split, parts in stage_parts(spec).items()
        })
    upload_file = object_store_sink.upload_file
    spilled = []

    def upload(filesystem, source, destination):
        spilled.append(len(os.listdir(os.path.dirname(source))))
        upload_file(filesystem, source, destination)

    monkeypatch.setattr(object_store_sink, 'upload_file', upload)
    _, table = _generate(graph, mlops=kumo.MLOps(artifact_root=str(tmp_path)))
    stored = tmp_path / 'training_tables' / table.id
    assert len(spilled) == len(list(stored.glob('part-*.parquet'))) >= 8
    assert max(spilled) <= 2


def test_a_mlops_block_stores_each_table_once_under_the_root(graph, tmp_path):
    snapshot = graph[1]
    root = tmp_path / 'artifacts'
    cfg = kumo.MLOps(artifact_root=str(root))
    with kumo.mlops(cfg):
        pquery, table = _generate(graph)
        stored = root / 'training_tables' / table.id
        parts = {p.name: p.stat().st_mtime_ns for p in stored.iterdir()}
        pquery.generate_training_table(snapshot)
    assert '_provenance.json' in parts and '_SUCCESS' in parts
    assert table.manifest.build.disk_bytes > 0
    assert pquery.generate_training_table(snapshot).manifest.build is None
    assert {p.name: p.stat().st_mtime_ns for p in stored.iterdir()} == parts
    assert json.loads((stored / '_provenance.json').read_text())['id'] == (
        table.id)
    reloaded = kumo.TrainingTable.load_from_connector(
        cfg, path=f'training_tables/{table.id}', pquery=pquery, snapshot=snapshot)
    assert _seeds(reloaded)[1] == _seeds(table)[1]
    off = pquery.generate_training_table(
        snapshot, mlops=kumo.MLOps(artifact_root=str(tmp_path / 'other'),
                                   store={'model'}))
    assert off.id == table.id and not (tmp_path / 'other').exists()


def _volume_root(tmp_path, *, other_catalog=False, require_staged=False):
    from types import SimpleNamespace

    from pyarrow.fs import LocalFileSystem, SubTreeFileSystem

    (tmp_path / 'volume').mkdir()
    files = SubTreeFileSystem(str(tmp_path / 'volume'), LocalFileSystem())
    volume = kumo.DatabricksConnector(
        catalog='other_catalog' if other_catalog else 'c',
        schema_name='other_schema' if other_catalog else 's',
        warehouse_id='warehouse', require_staged=require_staged,
        staging_volume_path='/Volumes/c/s/v/staging' if require_staged else None)
    volume.__dict__['sink_backend'] = SimpleNamespace(
        holds_files=lambda connector, path: path.startswith('/Volumes/'),
        filesystem=lambda connector, path: (files, path.lstrip('/')))
    return kumo.MLOps((volume, '/Volumes/c/s/v/kumo'))


@pytest.mark.parametrize('other_catalog,require_staged', [(False, False), (True, False), (False, True)])
def test_a_table_stored_on_a_volume_loads_back(
        graph, tmp_path, other_catalog, require_staged):
    cfg = _volume_root(tmp_path, other_catalog=other_catalog, require_staged=require_staged)
    assert cfg.root.connector.source_backend is not None
    with kumo.mlops(cfg):
        pquery, table = _generate(graph)
    path = f'training_tables/{table.id}'

    def load():
        return kumo.TrainingTable.load_from_connector(
            cfg, path=path, pquery=pquery, snapshot=graph[1])

    reloaded = load()
    assert _seeds(reloaded)[1] == _seeds(table)[1]
    assert reloaded.manifest.parents.source_table.location == (
        f'/Volumes/c/s/v/kumo/{path}')
    assert load().id == reloaded.id


def test_a_reloaded_table_keeps_unrecorded_sources_unrecorded(graph, tmp_path):
    cfg = _volume_root(tmp_path)
    snapshot = graph[0].snapshot(record_sources=False)
    pquery = kumo.PredictiveQuery(graph=graph[0], query=LABEL)
    with kumo.mlops(cfg):
        table = pquery.generate_training_table(snapshot)
    reloaded = kumo.TrainingTable.load_from_connector(
        cfg, path=f'training_tables/{table.id}', pquery=pquery,
        snapshot=snapshot)
    assert reloaded.manifest.parents.source_table.location == table.id
    assert '/Volumes' not in json.dumps(reloaded.manifest.to_dict())


def test_a_string_root_records_the_full_uri(graph, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pyarrow.fs import LocalFileSystem, SubTreeFileSystem

    files = SubTreeFileSystem(str(tmp_path), LocalFileSystem())
    store = SimpleNamespace(source_type='s3', sink_backend=SimpleNamespace(
        holds_files=lambda connector, path: True,
        filesystem=lambda connector, path: (files, path)))
    monkeypatch.setattr('kumo.engine.connector.connector_for_root',
                        lambda uri: store)
    cfg = kumo.MLOps('s3://bucket/kumo')
    with kumo.mlops(cfg):
        pquery, table = _generate(graph)
    path = f'training_tables/{table.id}'
    reloaded = kumo.TrainingTable.load_from_connector(
        cfg, path=path, pquery=pquery, snapshot=graph[1])
    assert reloaded.manifest.parents.source_table.location == (
        f's3://bucket/kumo/{path}')


@pytest.mark.parametrize('remote', [True, False])
def test_an_incomplete_stored_table_says_so(graph, tmp_path, remote):
    cfg = (_volume_root(tmp_path) if remote else
           kumo.MLOps(str(tmp_path / 'kumo')))
    with kumo.mlops(cfg):
        pquery, table = _generate(graph)
    stored = (tmp_path / 'volume/Volumes/c/s/v/kumo' if remote else
              tmp_path / 'kumo')
    (stored / 'training_tables' / table.id / '_SUCCESS').unlink()
    with pytest.raises(FileNotFoundError, match='has no _SUCCESS marker'):
        kumo.TrainingTable.load_from_connector(
            cfg, path=f'training_tables/{table.id}', pquery=pquery,
            snapshot=graph[1])


@pytest.mark.parametrize('rewrite', [
    lambda stored: (stored / 'part-new.parquet').write_bytes(
        next(stored.glob('*.parquet')).read_bytes()),
    lambda stored: (stored / '_SUCCESS').unlink()])
def test_a_table_rewritten_while_loading_says_so(graph, tmp_path,
                                                 monkeypatch, rewrite):
    from kumo.engine.trainer import trainer

    cfg = _volume_root(tmp_path)
    with kumo.mlops(cfg):
        pquery, table = _generate(graph)
    stored = (tmp_path / 'volume/Volumes/c/s/v/kumo/training_tables' /
              table.id)
    fetch = trainer.fetch_parquet

    def rewrite_after(*args):
        fetch(*args)
        rewrite(stored)

    monkeypatch.setattr(trainer, 'fetch_parquet', rewrite_after)
    with pytest.raises(RuntimeError, match='changed while it was being '
                       'loaded; retry'):
        kumo.TrainingTable.load_from_connector(
            cfg, path=f'training_tables/{table.id}', pquery=pquery,
            snapshot=graph[1])


def test_volume_folder_load_uses_the_source_backend_before_artifact_routing(
        monkeypatch):
    connector = kumo.DatabricksConnector(catalog='catalog',
                                         schema_name='schema',
                                         warehouse_id='warehouse')
    connector.__dict__['sink_backend'] = SimpleNamespace(
        holds_files=lambda *args: pytest.fail(
            'Volume sources must use scoped source reads'))
    observed = []

    def read(cls, source, spec, *, pq, snapshot):
        observed.append((source.table_name, source.connector))
        return 'source-reader-result'

    monkeypatch.setattr(kumo.TrainingTable, 'from_source_table',
                        classmethod(read))
    path = '/Volumes/catalog/schema/volume/plain-folder'
    assert kumo.TrainingTable.load_from_connector(
        connector, path=path, pquery=None,
        snapshot=None) == 'source-reader-result'
    assert observed == [(path, connector)]


@pytest.mark.parametrize('other_catalog,require_staged', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('complete', [False, True])
def test_stored_volume_routing_checks_commit_before_source_loading(
        tmp_path, monkeypatch, other_catalog, require_staged, complete):
    cfg = _volume_root(tmp_path, other_catalog=other_catalog, require_staged=require_staged)
    assert cfg.root.connector.source_backend is not None
    stored = tmp_path / 'volume/Volumes/c/s/v/kumo/training_tables/table-id'
    stored.mkdir(parents=True)
    pq.write_table(pa.table({'ENTITY1': [1, 2], 'SPLIT': [0, 1]}), stored / 'part.parquet')
    if complete:
        (stored / '_SUCCESS').touch()
    observed = []

    def from_rows(cls, source_table, spec, **kwargs):
        rows = Path(source_table.connector.config.root_dir) / source_table.table_name
        observed.append((pq.read_table(rows).to_pydict(), kwargs['source'].location))
        return 'loaded export'

    monkeypatch.setattr(kumo.TrainingTable, '_from_rows', classmethod(from_rows))
    path = 'training_tables/table-id'
    if complete:
        assert kumo.TrainingTable.load_from_connector(
            cfg, path=path, pquery=None, snapshot=SimpleNamespace(record_sources=True)) == 'loaded export'
        assert observed == [({'ENTITY1': [1, 2], 'SPLIT': [0, 1]},
                             '/Volumes/c/s/v/kumo/' + path)]
    else:
        with pytest.raises(FileNotFoundError, match='has no _SUCCESS marker'):
            kumo.TrainingTable.load_from_connector(
                cfg, path=path, pquery=None, snapshot=SimpleNamespace(record_sources=True))
        assert observed == []
