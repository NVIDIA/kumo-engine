# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.errors import SourceNotFoundError
from kumo_core.training.export import STAGE_SPLITS, export_train_table
from kumoapi.graph import Edge
from kumoapi.model_plan import TrainingTableGenerationPlan
from kumoapi.train import TrainingTableSpec
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import scoped_scratch_dir
from kumo.engine.connector.file import FileConnector
from kumo.engine.integrations.pytorch import (
    stage_prediction_table,
    stage_tables,
    stage_training_table,
)

api = pytest.importorskip('kumo.pytorch.api')

_UTC = pa.timestamp('ms', tz='UTC')


def test_staged_tables_feed_a_kumo_pytorch_batch(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    pq.write_table(pa.table({'user_id': [1, 2]}), data / 'users.parquet')
    pq.write_table(
        pa.table({
            'purchase_id': [101, 102, 103, 201],
            'user_id': [1, 1, 1, 2],
            'at':
            pa.array([datetime(2024, 1, day) for day in (2, 8, 12, 6)], _UTC),
            'amount':
            pa.array([Decimal(v) for v in ('12.00', '30.50', '99.00', '18.25')],
                     pa.decimal128(12, 2)),
        }), data / 'purchases.parquet')
    pq.write_table(
        pa.table({
            'user_id': [1, 2],
            'at': pa.array([datetime(2024, 1, 10)] * 2, _UTC),
        }), data / 'requests.parquet')
    connector = FileConnector(root_dir=str(data))

    graph = stage_tables(
        {name: connector[name]
         for name in ('users', 'purchases')}, root=tmp_path / 'staged')
    requests = stage_tables({'requests': connector['requests']},
                            root=tmp_path / 'staged')

    purchases = graph.tables['purchases']
    amount = pq.read_table(purchases.path).schema.field('amount')
    assert amount.type == pa.decimal128(12, 2)
    assert purchases.version is not None

    schema = api.DataSchema(
        tables=(
            api.TableSpec(name='users', primary_key='user_id',
                          time_column=api.NON_TEMPORAL),
            api.TableSpec(name='purchases', primary_key='purchase_id',
                          time_column='at'),
        ),
        relations=(api.RelationSpec(name='user_purchases', left_table='users',
                                    left_key='user_id', right_table='purchases',
                                    right_key='user_id'), ),
    )
    past = api.Rows(path=('user_purchases', ),
                    time=api.TimeWindow(column='at', stop=0))
    sample = api.SampleSpec(
        entity=api.EntitySpec(table='users', entity_type='user'),
        events={
            'purchases':
            api.EventSpec(rows=past,
                          fields={'amount': api.Field(column='amount')})
        },
    )
    batch = api.BatchSpec(
        inputs={
            'history':
            api.Input(
                source=api.EventSource(events=('purchases', )),
                representation=api.Fields(fields=(
                    api.Encode(field='amount', encoding=api.ZScore()), )),
                layout=api.Padded(),
            ),
        },
        batching=api.Batching(max_samples=2),
    )
    contract = api.DataLoaderSpec(schema=schema, sample=sample, batch=batch)
    anchors = api.AnchorTable(source=requests.source('requests'),
                              entity_id='user_id', as_of_time='at')
    engine = api.DiskGraphEngine(index=tmp_path / 'index')
    with api.prepare(
            contract=contract, dataset=graph.dataset(),
            engine=engine) as plan, plan.materialize(anchors=anchors) as task:
        result = next(iter(task.batches(output_format='model')))

    # Purchases before January 10: two by user 1, one by user 2.
    assert result.inputs['history']['valid'].sum(dim=1).tolist() == [2, 1]


def test_stage_tables_leaves_nothing_when_a_table_fails(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    pq.write_table(pa.table({'user_id': [1]}), data / 'users.parquet')
    connector = FileConnector(root_dir=str(data))

    with pytest.raises(SourceNotFoundError):
        stage_tables({name: connector[name]
                      for name in ('users', 'missing')},
                     root=tmp_path / 'staged')

    assert list((tmp_path / 'staged').iterdir()) == []


@pytest.fixture
def anchor_scratch(tmp_path):
    with scoped_scratch_dir(tmp_path / 'scratch'):
        yield


def _anchor_graph(tmp_path, key_type, user_time):
    rng = np.random.default_rng(0)
    data = tmp_path / 'data'
    data.mkdir()

    def ids(values):
        if key_type == 'string':
            return pa.array([f'u{value}' for value in values])
        return pa.array(values, getattr(pa, key_type)())

    def times(count, days):
        return pa.array([
            datetime(2024, 1, 1) + timedelta(days=int(day))
            for day in rng.integers(0, days, count)
        ], pa.timestamp('us'))

    pq.write_table(
        pa.table({
            'ID': ids(range(40)),
            'AGE': rng.integers(18, 80, 40),
            'SCORE': rng.random(40),
            'SEGMENT': rng.choice(['a', 'b', 'c'], 40),
            'JOINED': times(40, 60),
        }), data / 'users.parquet')
    pq.write_table(
        pa.table({
            'EID': np.arange(600),
            'USER_ID': ids(rng.integers(0, 40, 600)),
            'AMOUNT': rng.random(600),
            'TS': times(600, 120),
        }), data / 'events.parquet')
    connector = FileConnector(root_dir=str(data))
    users = kumo.Table.from_source_table(
        connector['users.parquet'], primary_key='ID',
        time_column=user_time).infer_metadata()
    events = kumo.Table.from_source_table(
        connector['events.parquet'], primary_key='EID',
        time_column='TS').infer_metadata().with_column('USER_ID',
                                                       stype=Stype.ID)
    graph = kumo.Graph(
        tables={
            'users': users,
            'events': events
        }, edges=[Edge(src_table='events', fkey='USER_ID', dst_table='users')])
    return connector, graph


def _read(parts):
    return pa.concat_tables([pq.read_table(part) for part in parts])


def _seconds(time):
    return (time - datetime(1970, 1, 1)) // timedelta(seconds=1)


def _served(batches):
    r"""Each served anchor's entity, time, label and weight, by sample id."""
    def values(column):
        return column.tolist() if hasattr(column, 'tolist') else list(column)

    served = {}
    for batch in batches:
        metadata = batch.metadata
        entities = values(metadata['entity_ids'])
        times = values(metadata['as_of_time'])
        label = batch.targets.get('label')
        labels = values(label['values']) if label else [None] * len(times)
        for i, sample_id in enumerate(metadata['sample_ids']):
            if not metadata['is_padding'][i]:
                assert sample_id not in served
                served[sample_id] = (
                    entities[i], times[i], labels[i],
                    metadata['request_metadata'][i].get('WEIGHT'))
    return served


# The timed-entity case also uses a random split and an entity value, which
# takes kumo-pytorch's scalar materialization path.
@pytest.mark.usefixtures('anchor_scratch')
@pytest.mark.parametrize(
    'key_type, query, label_dtype, user_time, has_time, weighted', [
        ('int64', 'PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH users.ID',
         'bool', None, True, False),
        ('string', 'PREDICT users.SEGMENT FOR EACH users.ID', 'string', None,
         False, True),
        ('int32', 'PREDICT users.SCORE FOR EACH users.ID', 'float64', 'JOINED',
         True, False),
    ], ids=[
        'temporal-binary', 'static-multiclass-weighted',
        'static-timed-regression-random'
    ])
def test_staged_anchors_match_engine_rows(tmp_path, key_type, query,
                                          label_dtype, user_time, has_time,
                                          weighted):
    connector, graph = _anchor_graph(tmp_path, key_type, user_time)
    snapshot = graph.snapshot()
    pquery = kumo.PredictiveQuery(graph=graph, query=query)
    generation = TrainingTableGenerationPlan(
        split='RandomSplit([0.6, 0.2, 0.2])') if user_time else None
    table = pquery.generate_training_table(snapshot, generation)
    if weighted:
        rows = _read(
            export_train_table(str(table.staging_dir),
                               str(tmp_path / 'unweighted')))
        rows = rows.append_column('WEIGHT',
                                  pa.array(np.arange(rows.num_rows) + 1.0))
        pq.write_table(rows, tmp_path / 'data' / 'weighted.parquet')
        table = kumo.TrainingTable.from_source_table(
            connector['weighted.parquet'],
            TrainingTableSpec(weight_col='WEIGHT', split_col='SPLIT'),
            pq=pquery, snapshot=snapshot)

    anchors = stage_training_table(table, pquery, target='label',
                                   root=tmp_path / 'anchors')
    assert anchors.as_of_time == ('TIMESTAMP' if has_time else None)
    staged = stage_tables(
        {name: connector[f'{name}.parquet']
         for name in ('users', 'events')}, root=tmp_path / 'staged')
    schema = api.DataSchema(
        tables=(
            api.TableSpec(name='users', primary_key='ID', time_column=user_time
                          or api.NON_TEMPORAL),
            api.TableSpec(name='events', primary_key='EID',
                          time_column='TS' if has_time else api.NON_TEMPORAL),
        ),
        relations=(api.RelationSpec(name='user_events', left_table='users',
                                    left_key='ID', right_table='events',
                                    right_key='USER_ID'), ),
    )
    sample = api.SampleSpec(
        entity=api.EntitySpec(table='users', entity_type='user'),
        entity_values={'age': api.Field(column='AGE')} if user_time else {},
        neighborhoods={
            'graph':
            api.NeighborhoodSpec((api.GraphLayer(
                (api.GraphSample('user_events', 4), )), ))
        },
        targets={'label': api.ExternalTarget(dtype=api.DType(label_dtype))},
    )
    inputs = {
        'graph':
        api.Input(
            source=api.NeighborhoodSource('graph'), representation=api.Graph(
                node_fields={
                    'events': api.Fields((api.Encode('AMOUNT', api.Raw()), ))
                }), layout=None),
    }
    if user_time:
        inputs['user'] = api.Input(
            source=api.EntitySource(), representation=api.Fields(
                (api.Encode('age', api.Raw()), )), layout=api.Separate())
    batch = api.BatchSpec(
        inputs=inputs,
        targets={'label': api.LabelTarget(source='label', encoding=api.Raw())},
        batching=api.Batching(max_samples=64),
    )
    contract = api.DataLoaderSpec(schema=schema, sample=sample, batch=batch)
    engine = api.DiskGraphEngine(index=tmp_path / 'index')
    assert api.check(contract=contract, dataset=staged.dataset(), engine=engine,
                     anchors=anchors) == ()

    # Sample ids are positions in the rows `TrainingTable.export` writes.
    expected = {split: {} for split in STAGE_SPLITS}
    split_names = {code: split for split, code in STAGE_SPLITS.items()}
    reference = export_train_table(str(table.staging_dir),
                                   str(tmp_path / 'reference'))
    for i, row in enumerate(_read(reference).to_pylist()):
        time = _seconds(row['TIMESTAMP']) if has_time else 0
        expected[split_names[row['SPLIT']]][i] = (row['ENTITY1'], time,
                                                  row['TARGET1'],
                                                  row.get('WEIGHT'))
    with api.prepare(contract=contract, dataset=staged.dataset(),
                     engine=engine) as plan, plan.materialize(
                         anchors, fit_split='train') as task:
        assert {
            split: _served(task.batches(split=split, shuffle=False))
            for split in STAGE_SPLITS
        } == expected

        if has_time and not user_time:  # The temporal query.
            # A plan's forecast length makes this query a forecasting task.
            count = kumo.PredictiveQuery(
                graph=graph,
                query='PREDICT COUNT(events.*, 0, 7, days) FOR EACH users.ID')
            forecast = TrainingTableGenerationPlan(forecast_length=2,
                                                   lag_timesteps=1)
            for stage, generated in [
                (stage_training_table,
                 count.generate_training_table(snapshot, forecast)),
                (stage_prediction_table,
                 count.generate_prediction_table(snapshot, plan=forecast)),
            ]:
                with pytest.raises(ValueError, match='Staging forecasting'):
                    stage(generated, count)

            prediction = pquery.generate_prediction_table(snapshot)
            prediction_anchors = stage_prediction_table(
                prediction, pquery, root=tmp_path / 'predictions')
            rows = _read(sorted(prediction.table_path.glob('*.parquet')))
            with plan.materialize(prediction_anchors, fitted=task.fitted_state,
                                  inference=True) as inference:
                assert _served(inference.batches(shuffle=False)) == {
                    i: (row['ENTITY1'], _seconds(row['TIMESTAMP']), None, None)
                    for i, row in enumerate(rows.to_pylist())
                }
