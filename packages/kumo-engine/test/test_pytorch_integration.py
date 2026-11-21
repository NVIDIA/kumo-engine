# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.errors import SourceNotFoundError

from kumo.engine.connector.file import FileConnector
from kumo.engine.integrations.pytorch import stage_tables

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
