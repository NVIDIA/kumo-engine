# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from kumo_core.io.json import to_json_dict
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumoapi.typing import Stype

from io_utils import fsutil
from io_utils.parquet_util import read_parquet
from kumo.engine import (
    FileConnector,
    Graph,
    PredictiveQuery,
    Table,
    Trainer,
    TrainingTableOutputConfig,
    configure,
)
from kumo.engine._scratch import get_scratch_dir

START = pd.Timestamp('2024-01-10')
END = pd.Timestamp('2024-02-20')
SOURCE_END = pd.Timestamp('2024-02-29')
TASKS = [
    ('binary_classification', 'COUNT(events.*, 0, 2, days) > 0'),
    ('regression', 'COUNT(events.*, 0, 2, days)'),
    ('multiclass_classification', 'FIRST(events.category, 0, 2, days)'),
    ('multilabel_classification',
     'LIST_DISTINCT(events.category, 0, 2, days) CLASSIFY TOP 3'),
    ('multilabel_ranking',
     'LIST_DISTINCT(events.category, 0, 2, days) RANK TOP 3'),
    ('forecasting', 'COUNT(events.*, 0, 2, days) FORECAST 3 TIMEFRAMES'),
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
        options.update(lag_timesteps=3, year_over_year=False)
    query, table, rows = _generate(temporal_graph, tmp_path, target, **options)
    assert query.get_task_type().value == task
    assert set(rows['SPLIT']) == {0, 1, 2}
    assert rows['TIMESTAMP'].min() >= START
    horizon = 6 if task == 'forecasting' else 2
    assert rows['TIMESTAMP'].max() + pd.Timedelta(days=horizon) <= END
    assert table.generation_plan.start_time == START.isoformat()
    assert table.generation_plan.end_time == END.isoformat()
    if task in {'binary_classification', 'regression'}:
        events = temporal_graph[2]
        for row in rows.itertuples():
            count = len(events[(events['user_id'] == row.ENTITY1)
                               & (events['ts'] > row.TIMESTAMP)
                               & (events['ts'] <= row.TIMESTAMP +
                                  pd.Timedelta(days=2))])
            assert row.TARGET1 == (count > 0 if task.startswith('binary')
                                   else count)


@pytest.mark.parametrize('bounds', [
    {'start_time': '2023-12-01', 'end_time': '2024-04-01'},
    {'start_time': '2024-01-10'},
    {'end_time': '2024-02-20'},
    {'start_time': '-INF', 'end_time': '2024-02-20'},
])
def test_partial_and_clipped_bounds(temporal_graph, tmp_path, monkeypatch,
                                    bounds):
    monkeypatch.setenv('USE_DECOUPLED_SPLITS', 'false')
    _, _, rows = _generate(temporal_graph, tmp_path,
                           'COUNT(events.*, 0, 2, days)', **bounds)
    lower = max(pd.Timestamp('2024-01-01'),
                pd.Timestamp(bounds['start_time'])
                if bounds.get('start_time') not in {None, '-INF'} else
                pd.Timestamp('2024-01-01'))
    upper = min(SOURCE_END, pd.Timestamp(bounds.get('end_time', SOURCE_END)))
    assert not rows.empty
    assert rows['TIMESTAMP'].min() >= lower
    assert rows['TIMESTAMP'].max() + pd.Timedelta(days=2) <= upper


@pytest.mark.parametrize('override', [False, True])
def test_forecast_complete_horizon(temporal_graph, tmp_path, monkeypatch,
                                   override):
    monkeypatch.setenv('USE_DECOUPLED_SPLITS', 'false')
    target = 'COUNT(events.*, 0, 2, days)'
    options = dict(start_time=START.isoformat(), lag_timesteps=3,
                   year_over_year=False)
    if override:
        options.update(forecast_length=3, end_time=END.isoformat())
    else:
        target += ' FORECAST 3 TIMEFRAMES'
    _, table, rows = _generate(temporal_graph, tmp_path, target, **options)
    cutoff = END if override else SOURCE_END
    assert not rows.empty
    assert rows['TIMESTAMP'].max() + pd.Timedelta(days=6) <= cutoff
    assert table.generation_plan.forecast_length == 3


@pytest.mark.parametrize('split', [None, 'RandomSplit([0.6, 0.2, 0.2])',
                                   'TemporalSplit([0.6, 0.2, 0.2])',
                                   'ApproxDateOffsetSplit([0.6, 0.2, 0.2])'])
def test_bound_timezone_equivalence(temporal_graph, tmp_path, monkeypatch,
                                    split):
    monkeypatch.delenv('USE_DECOUPLED_SPLITS', raising=False)
    options = {} if split is None else {'split': split}
    _, _, utc = _generate(
        temporal_graph, tmp_path / 'utc', 'COUNT(events.*, 0, 2, days)',
        start_time='2024-01-10T00:00:00Z',
        end_time='2024-02-20T00:00:00Z', **options)
    _, _, local = _generate(
        temporal_graph, tmp_path / 'local', 'COUNT(events.*, 0, 2, days)',
        start_time='2024-01-09T18:00:00-06:00',
        end_time='2024-02-20T01:00:00+01:00', **options)
    columns = ['TIMESTAMP', 'ENTITY1', 'SPLIT']
    pd.testing.assert_frame_equal(
        utc.sort_values(columns).reset_index(drop=True),
        local.sort_values(columns).reset_index(drop=True))
    if split == 'ApproxDateOffsetSplit([0.6, 0.2, 0.2])':
        expected = pd.date_range('2024-01-11', '2024-02-18', freq='2D')
        assert set(utc['TIMESTAMP']) == set(expected)


@pytest.mark.parametrize('options', [
    {},
    {'train_start_offset': 40, 'train_end_offset': 10},
    {'split': "TimeRangeSplit([('2024-01-10', '2024-01-24'), "
              "('2024-01-24', '2024-02-07'), "
              "('2024-02-07', '2024-02-20')])"},
])
def test_existing_window_controls(temporal_graph, tmp_path, monkeypatch,
                                  options):
    monkeypatch.delenv('USE_DECOUPLED_SPLITS', raising=False)
    _, _, rows = _generate(temporal_graph, tmp_path,
                           'COUNT(events.*, 0, 2, days)', **options)
    assert set(rows['SPLIT']) == {0, 1, 2}
    assert rows['TIMESTAMP'].min() >= pd.Timestamp('2024-01-01')
    assert rows['TIMESTAMP'].max() + pd.Timedelta(days=2) <= SOURCE_END


@pytest.mark.parametrize('start,end', [
    ('2024-03-10', '2024-03-20'),
    ('2024-01-10', '2024-01-11'),
])
def test_no_complete_window(temporal_graph, tmp_path, monkeypatch, start, end):
    monkeypatch.delenv('USE_DECOUPLED_SPLITS', raising=False)
    with pytest.raises(ValueError, match='No usable training examples'):
        _generate(temporal_graph, tmp_path, 'COUNT(events.*, 0, 2, days)',
                  start_time=start, end_time=end)


def test_bounds_fit_reload_predict(temporal_graph, tmp_path, monkeypatch,
                                   request):
    monkeypatch.delenv('USE_DECOUPLED_SPLITS', raising=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    query, table, _ = _generate(
        temporal_graph, tmp_path, 'COUNT(events.*, 0, 2, days) > 0',
        start_time='2024-01-10T00:00:00Z',
        end_time='2024-02-19T18:00:00-06:00')
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.training_job.refit_trainval = False
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    plan.optimization.batch_size = [32]
    plan.model_architecture.channels = [16]
    snapshot = temporal_graph[1]
    trainer = Trainer(plan, accelerator='cpu')
    result = trainer.fit(table, snapshot=snapshot)
    restored = Trainer.load(result.bundle_dir, accelerator='cpu')
    expected = to_json_dict(table.generation_plan)
    assert to_json_dict(trainer.training_table_generation_plan) == expected
    assert to_json_dict(restored.training_table_generation_plan) == expected
    prediction_table = query.generate_prediction_table(snapshot)
    before = trainer.predict(prediction_table, snapshot=snapshot,
                             binary_classification_threshold=0.5).frame
    after = restored.predict(prediction_table, snapshot=snapshot,
                             binary_classification_threshold=0.5).frame
    assert len(after) == 24
    pd.testing.assert_frame_equal(before, after, atol=0.005, rtol=0)
