# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from kumoapi.graph import Edge
from kumoapi.jobs import MetadataField
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.trainer.output_config import PredictionOutputConfig

ANCHOR = MetadataField.ANCHOR_TIMESTAMP
JOB = MetadataField.JOB_TIMESTAMP
ARMS = [None, [], [JOB], [ANCHOR], [ANCHOR, JOB]]


@pytest.fixture(scope='module')
def graph(tmp_path_factory):
    root = tmp_path_factory.mktemp('graph')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    rng = np.random.default_rng(0)
    data = root / 'data'
    data.mkdir()
    pd.DataFrame({'ID': np.arange(50)}).to_parquet(data / 'ents.parquet')
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
        'V':
        rng.normal(size=3000),
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
    yield graph, graph.snapshot(), root
    kumo.configure(scratch_dir=previous)


def _fit(graph, query):
    graph, snapshot, _ = graph
    pquery = kumo.PredictiveQuery(graph=graph, query=query)
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(model_plan=plan)
    trainer.fit(pquery.generate_training_table(snapshot), snapshot=snapshot)
    return trainer, pquery, pquery.generate_prediction_table(snapshot)


@pytest.fixture(scope='module')
def forecast(graph):
    return _fit(
        graph, 'PREDICT COUNT(events.*, 0, 7, days) '
        'FORECAST 3 TIMEFRAMES FOR EACH ents.ID')


@pytest.fixture(scope='module')
def classifier(graph):
    return _fit(graph, 'PREDICT COUNT(events.*, 0, 7, days) > 0 '
                'FOR EACH ents.ID')


@pytest.fixture(scope='module')
def link(graph):
    return _fit(
        graph, 'PREDICT LIST_DISTINCT(events.ITEM_ID, 0, 7, days) '
        'RANK TOP 3 FOR EACH ents.ID')


def _predict(graph, fitted, fields, kinds=('predictions', ), **options):
    """Return each exported artifact; ``fields=None`` exports nothing."""
    _, snapshot, root = graph
    trainer, _, table = fitted
    if fields is None:
        return {
            'predictions':
            trainer.predict(table, snapshot=snapshot, **options).frame
        }
    name = '-'.join(field.value for field in fields) or 'none'
    paths = {
        kind: f'{trainer.scratch_dir.name}/{name}/{kind}/'
        for kind in kinds
    }
    output = PredictionOutputConfig(
        connector=kumo.FileConnector(root_dir=str(root / 'out')),
        output_paths=paths, extra_fields=fields)
    trainer.predict(table, snapshot=snapshot, output_config=output, **options)
    return {
        kind:
        pd.concat(
            pd.read_parquet(path)
            for path in sorted((root / 'out' / paths[kind]).glob('*.parquet')))
        for kind in kinds
    }


def _metadata(fields):
    return [field.value for field in (ANCHOR, JOB) if field in (fields or [])]


def test_time_column_is_a_feature(graph, forecast):
    [path] = forecast[0].scratch_dir.glob(
        'runs/*-fit-*/task_dataset/feature_schema.json')
    schema = json.loads(path.read_text())['node_types']

    assert schema['events']['feature_columns'] == ['V', 'TS']


@pytest.mark.parametrize('fields', ARMS)
def test_forecast_output_columns(graph, forecast, fields):
    # #31405
    frame = _predict(graph, forecast, fields)['predictions']
    anchor = pd.Timestamp(forecast[2].timestamps[0], unit='ns')

    assert set(frame) == {
        'ENTITY', 'TIMESTAMP', 'TARGET_PRED', *_metadata(fields)
    }
    horizons = frame.groupby('ENTITY')['TIMESTAMP'].agg(sorted)
    assert horizons.map(len).eq(3).all()
    assert (horizons.map(lambda times: times[0]) == anchor).all()
    if ANCHOR in (fields or []):
        assert set(frame['ANCHOR_TIMESTAMP']) == {anchor}


@pytest.mark.parametrize('fields', ARMS)
def test_classification_output_columns(graph, classifier, fields):
    kinds = ('predictions', ) if fields is None else ('predictions',
                                                      'embeddings')
    frames = _predict(graph, classifier, fields, kinds,
                      binary_classification_threshold=0.5)
    anchor = pd.Timestamp(classifier[2].timestamps[0], unit='ns')
    metadata = _metadata(fields)

    assert list(frames['predictions']) == [
        'ENTITY', *metadata, 'TARGET_PRED', 'False_PROB', 'True_PROB'
    ]
    if fields is not None:
        assert list(frames['embeddings']) == ['ENTITY', *metadata, 'EMBEDDING']
    for frame in frames.values():
        if ANCHOR in metadata:
            assert set(frame['ANCHOR_TIMESTAMP']) == {anchor}
        if JOB in metadata:
            assert frame['JOB_TIMESTAMP'].nunique() == 1


@pytest.mark.parametrize('fields', [[], [ANCHOR, JOB]])
def test_link_prediction_output_columns(graph, link, fields):
    frames = _predict(graph, link, fields, ('predictions', 'embeddings'),
                      num_classes_to_return=3)
    metadata = _metadata(fields)

    assert list(
        frames['predictions']) == ['ENTITY', 'CLASS', 'SCORE', *metadata]
    assert list(frames['embeddings']) == ['ID', 'TYPE', 'EMBEDDING', *metadata]


def test_later_export_rejects_extra_fields(graph, classifier, tmp_path):
    _, snapshot, _ = graph
    trainer, _, table = classifier
    result = trainer.predict(table, snapshot=snapshot,
                             binary_classification_threshold=0.5)
    connector = kumo.FileConnector(root_dir=str(tmp_path))

    with pytest.raises(ValueError, match='Trainer.predict'):
        result.export(
            PredictionOutputConfig(connector=connector,
                                   output_paths={'predictions': 'bad/'},
                                   extra_fields=[ANCHOR]))
    assert not (tmp_path / 'bad').exists()

    result.export(
        PredictionOutputConfig(connector=connector,
                               output_paths={'predictions': 'good/'}))
    copied = sorted((tmp_path / 'good').glob('*.parquet'))
    written = [p.read_bytes() for p in copied]
    expected = [Path(f).read_bytes() for f in sorted(result.files)]
    assert written == expected
