# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge
from kumoapi.model_plan import TrainingTableGenerationPlan
from kumoapi.typing import Stype

from io_utils import fsutil
from io_utils.parquet_util import read_parquet
from kumo.engine import (
    FileConnector,
    Graph,
    PredictiveQuery,
    Table,
    TrainingTableOutputConfig,
    configure,
)
from kumo.engine._scratch import get_scratch_dir

START = pd.Timestamp('2024-01-10')
END = pd.Timestamp('2024-02-20')
TASKS = [
    ('binary_classification', 'COUNT(events.*, 0, 2, days) > 0'),
    ('forecasting', 'COUNT(events.*, 0, 2, days) FORECAST 2 TIMEFRAMES'),
    ('multilabel_classification',
     'LIST_DISTINCT(events.category, 0, 2, days) CLASSIFY TOP 3'),
]


@pytest.fixture(scope='module')
def temporal_graph(tmp_path_factory, request):
    root = tmp_path_factory.mktemp('absolute-bounds')
    previous = get_scratch_dir()
    configure(scratch_dir=root / 'scratch')
    request.addfinalizer(lambda: configure(scratch_dir=previous))
    source = root / 'data'
    fsutil.makedirs(str(source))
    events = [(user, day) for user in range(24) for day in range(60)
              if (user + day) % 5 < 2]
    frames = {
        'users': pa.table({'id': range(24), 'value': range(24)}),
        'events': pa.table({
            'id': range(len(events)),
            'user_id': [user for user, _ in events],
            'ts': [pd.Timestamp('2024-01-01') + pd.Timedelta(days=day)
                   for _, day in events],
            'category': [f'category-{day % 3}' for _, day in events],
        }),
    }
    for name, frame in frames.items():
        with fsutil.get_binary_handle(str(source / f'{name}.parquet'),
                                      'wb') as handle:
            pq.write_table(frame, handle)
    connector = FileConnector(root_dir=str(source))
    graph = Graph({
        'users': Table.from_source_table(connector['users'], primary_key='id'),
        'events': Table.from_source_table(
            connector['events'], primary_key='id', time_column='ts'
        ).with_column('user_id', stype=Stype.ID).with_column(
            'category', stype=Stype.categorical),
    }, [Edge('events', 'user_id', 'users')])
    return graph, graph.snapshot(), frames['events'].to_pandas()


def _generate(temporal_graph, tmp_path, target, **options):
    graph, snapshot, _ = temporal_graph
    query = PredictiveQuery(graph, f'PREDICT {target} FOR EACH users.id')
    table = query.generate_training_table(
        snapshot, TrainingTableGenerationPlan(timeframe_step=2, **options))
    destination = tmp_path / 'export'
    fsutil.makedirs(str(destination))
    table.export(TrainingTableOutputConfig(
        FileConnector(root_dir=str(destination)), 'rows'))
    return query, table, read_parquet(str(destination / 'rows'))


@pytest.mark.parametrize('task,target', TASKS)
def test_temporal_node_bounds(temporal_graph, tmp_path, monkeypatch, task,
                              target):
    monkeypatch.delenv('USE_DECOUPLED_SPLITS', raising=False)
    options = dict(start_time=START.isoformat(), end_time=END.isoformat())
    if task == 'forecasting':
        options.update(forecast_length=3, lag_timesteps=3,
                       year_over_year=False)
    query, table, rows = _generate(temporal_graph, tmp_path, target, **options)
    assert query.get_task_type().value == task
    assert set(rows['SPLIT']) == {0, 1, 2}
    assert rows['TIMESTAMP'].min() >= START
    horizon = 6 if task == 'forecasting' else 2
    assert rows['TIMESTAMP'].max() + pd.Timedelta(days=horizon) <= END
    assert table.generation_plan.start_time == START.isoformat()
    assert table.generation_plan.end_time == END.isoformat()
    if task == 'forecasting':
        assert table.generation_plan.forecast_length == 3
        assert rows['TIMESTAMP'].max() == END - pd.Timedelta(days=6)
    if task == 'binary_classification':
        events = temporal_graph[2]
        for row in rows.itertuples():
            count = len(events[(events['user_id'] == row.ENTITY1)
                               & (events['ts'] > row.TIMESTAMP)
                               & (events['ts'] <= row.TIMESTAMP +
                                  pd.Timedelta(days=2))])
            assert row.TARGET1 == (count > 0)
