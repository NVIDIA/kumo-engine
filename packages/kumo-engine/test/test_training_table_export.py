# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter

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
