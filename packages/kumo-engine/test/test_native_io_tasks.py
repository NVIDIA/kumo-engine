# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real CPU training/prediction across the native sampler backends.

From the repository root, with an installed candidate DiskGraph wheel:
DG_TEST_NATIVE_TASKS=1 pytest -q packages/kumo-engine/test/test_native_io_tasks.py
Each task/backend gets a fresh interpreter; a denied io_uring setup fails.
"""

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from wheel_gpu_smoke import (
    TASKS,
    compare_predictions,
    make_plan,
    prediction_decisions,
    prediction_values,
    target_classes,
)

from io_utils import fsutil

QUERIES = {
    'binary': 'PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH users.id',
    'multiclass': 'PREDICT FIRST(events.daypart, 0, 14, days) FOR EACH users.id',
    'multilabel': (
        'PREDICT LIST_DISTINCT(events.daypart, 0, 14, days) '
        'CLASSIFY TOP 3 FOR EACH users.id'),
    'multilabel_ranking': (
        'PREDICT LIST_DISTINCT(events.daypart, 0, 14, days) '
        'RANK TOP 3 FOR EACH users.id'),
    'regression': 'PREDICT COUNT(events.*, 0, 30, days) FOR EACH users.id',
    'forecasting': (
        'PREDICT COUNT(events.*, 0, 7, days) FORECAST 12 TIMEFRAMES '
        'FOR EACH users.id'),
    'static_link': (
        'PREDICT LIST_DISTINCT(events.item_id) RANK TOP 12 '
        'FOR EACH users.id'),
    'temporal_link': (
        'PREDICT LIST_DISTINCT(events.item_id, 0, 7, days) '
        'RANK TOP 12 FOR EACH users.id'),
}
BACKENDS = ('mmap', 'buffered', 'direct')


def write_parquet(table: pa.Table, path: Path) -> None:
    with fsutil.get_binary_handle(str(path), 'wb') as stream:
        pq.write_table(table, stream)


def read_parquet(path: str) -> pa.Table:
    with fsutil.get_binary_handle(path, 'rb') as stream:
        return pq.read_table(stream)


def graph_for_source(source: Path, static: bool):
    from kumoapi.graph import Edge
    from kumoapi.typing import Stype

    import kumo.engine as kumo

    connector = kumo.FileConnector(root_dir=str(source))
    tables = {
        name: kumo.Table.from_source_table(
            connector[f'{name}.parquet'], primary_key='id',
            time_column='time' if name == 'events' and not static else None)
        for name in ('users', 'items', 'events')
    }
    tables['events'] = tables['events'].with_column(
        'user_id', stype=Stype.ID).with_column('item_id', stype=Stype.ID)
    graph = kumo.Graph(tables, [
        Edge('events', 'user_id', 'users'),
        Edge('events', 'item_id', 'items'),
    ])
    graph.validate()
    return graph


def fixture_graph(root: Path, static: bool):
    source = root / 'source'
    fsutil.makedirs(str(source), exist_ok=True)
    start = datetime(2020, 1, 1)
    events = [(user, week, item)
              for user in range(40) for week in range(72)
              if (user + week) % 3 != 0 for item in range(2)]
    frames = {
        'users': pa.table({'id': range(40), 'value': np.arange(40) / 40}),
        'items': pa.table({'id': range(12), 'value': np.arange(12) / 12}),
        'events': pa.table({
            'id': range(len(events)),
            'user_id': [user for user, _, _ in events],
            'item_id': [(user * 5 + week + item * 3) % 12
                        for user, week, item in events],
            'time': [start + timedelta(days=week * 7 + user % 3)
                     for user, week, _ in events],
            'daypart': [['morning', 'afternoon', 'evening'][(user * 2 + week) % 3]
                        for user, week, _ in events],
            'value': [(user + week + item) / 100
                      for user, week, item in events],
        }),
    }
    for name, table in frames.items():
        write_parquet(table, source / f'{name}.parquet')
    graph = graph_for_source(source, static)
    return graph, graph.snapshot(), start, start + timedelta(days=72 * 7)


def task_inputs(task: str, root: Path):
    from kumoapi.model_plan import TrainingTableGenerationPlan

    import kumo.engine as kumo

    graph, snapshot, start, end = fixture_graph(root, task == 'static_link')
    query = kumo.PredictiveQuery(graph, QUERIES[task])
    query.validate()
    assert query.get_task_type().value == TASKS[task]['task_type']
    if task == 'static_link':
        generation = TrainingTableGenerationPlan(
            split='RandomLinkSplit([0.6, 0.2, 0.2])')
    else:
        horizon = timedelta(days=TASKS[task]['days'])
        ranges = [
            ((start if offset == 3 else end - offset * horizon).isoformat(),
             (end - (offset - 1) * horizon).isoformat())
            for offset in (3, 2, 1)
        ]
        generation = TrainingTableGenerationPlan(
            split=f'TimeRangeSplit({ranges!r}, timeframe={horizon.days}, '
                  'flexible_boundaries=False)',
            timeframe_step=7 if task == 'forecasting' else horizon.days,
            **({'lag_timesteps': 12, 'year_over_year': False}
               if task == 'forecasting' else {}))
    training = query.generate_training_table(snapshot, plan=generation)
    export = root / 'export'
    fsutil.makedirs(str(export), exist_ok=True)
    training.export(kumo.TrainingTableOutputConfig(
        kumo.FileConnector(root_dir=str(export)), 'rows'))
    rows = pa.concat_tables([
        read_parquet(path)
        for path in fsutil.listdir(str(export / 'rows'), suffix='.parquet')
    ])
    assert set(rows['SPLIT'].to_pylist()) == {0, 1, 2}
    test_rows = rows.filter(pc.equal(rows['SPLIT'], 2))
    ordering = [('ENTITY1', 'ascending')]
    if task != 'static_link':
        ordering.append(('TIMESTAMP', 'ascending'))
    test_rows = test_rows.sort_by(ordering)
    indices, seen = [], set()
    for index, entity in enumerate(test_rows['ENTITY1'].to_pylist()):
        if entity not in seen:
            indices.append(index)
            seen.add(entity)
        if len(indices) == 8:
            break
    selected = test_rows.take(indices)
    ids = selected['ENTITY1'].to_pylist()
    assert len(ids) == 8 and len(set(ids)) == 8
    columns = ['ENTITY1'] if task == 'static_link' else ['ENTITY1', 'TIMESTAMP']
    if task == 'forecasting':
        columns.append('autoregression_labels')
        assert all(len(values) == 12
                   for values in selected['autoregression_labels'].to_pylist())
    fsutil.makedirs(str(root / 'prediction-inputs'), exist_ok=True)
    inputs = root / 'prediction-inputs' / 'inputs.parquet'
    write_parquet(selected.select(columns), inputs)
    classes = (list(range(12)) if task.endswith('link') else
               target_classes(task, rows['TARGET1'].to_pylist())
               if task in ('multiclass', 'multilabel', 'multilabel_ranking')
               else [])
    anchor = (None if task == 'static_link'
              else selected['TIMESTAMP'][0].as_py())
    return query, snapshot, training, inputs, ids, classes, anchor


def run_case(task: str, root: Path, reference: Path | None = None) -> None:
    import diskgraph._c as native
    import kumo_core
    import torch
    from kumo_core.io.filesystem import local_path
    from kumo_core.train.checkpoint import load_kumo_checkpoint
    from kumoapi.jobs import MetadataField
    from kumoapi.model_plan import GNNModelPlan

    import kumo.engine as kumo
    from kumo.engine.graph import GraphSnapshot

    repo = Path(__file__).resolve().parents[3]
    assert Path(kumo.__file__).resolve().is_relative_to(
        repo / 'packages/kumo-engine')
    assert Path(kumo_core.__file__).resolve().is_relative_to(
        repo / 'packages/kumo-core')
    assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix))
    torch.set_num_threads(1)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    kumo.configure(scratch_dir=root / 'scratch')
    query, snapshot, training, inputs, ids, classes, anchor = task_inputs(
        task, root)
    plan = make_plan(task)
    plan.optimization.max_epochs = 1
    plan.optimization.min_steps_per_epoch = 2
    plan.optimization.max_steps_per_epoch = 2
    plan.optimization.max_val_steps = 2
    plan.optimization.max_test_steps = 2
    plan.optimization.batch_size = [16]
    plan.model_architecture.model = [GNNModelPlan(
        channels=[8], aggregation=[['mean']], dropout=[0.0])]
    plan.model_architecture.num_post_message_passing_layers = [1]
    trainer = kumo.Trainer(plan, accelerator='cpu')
    fitted = trainer.fit(training, snapshot=snapshot)
    checkpoint = load_kumo_checkpoint(local_path(
        str(fitted.artifact_paths['selected_checkpoint'])), map_location='cpu')
    steps = checkpoint['global_step']
    assert isinstance(steps, int) and steps >= 2

    def predict(model, stage: str):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        table = kumo.PredictionTable.from_source_table(
            kumo.FileConnector(root_dir=str(inputs.parent))[inputs.name], pq=query,
            snapshot=snapshot, plan=model.training_table_generation_plan)
        options = {'num_classes_to_return': len(classes)} if classes else {}
        if task == 'binary':
            options['binary_classification_threshold'] = .5
        if task != 'static_link':
            options['output_config'] = kumo.PredictionOutputConfig(
                connector=kumo.FileConnector(root_dir=str(root)),
                output_paths={'predictions': f'predictions/{stage}'},
                extra_fields=[MetadataField.ANCHOR_TIMESTAMP])
        frame = model.predict(table, snapshot=snapshot, num_workers=1,
                              **options).frame
        values = prediction_values(task, frame, ids, classes=classes,
                                   anchor=anchor)
        decisions = prediction_decisions(task, frame, ids, classes=classes)
        return values, decisions

    before, before_decisions = predict(trainer, 'initial')
    repeated, repeat_decisions = predict(trainer, 'repeat')
    restored = kumo.Trainer.load(fitted.bundle_dir, accelerator='cpu')
    after, after_decisions = predict(restored, 'reload')
    for values, decisions in ((repeated, repeat_decisions),
                              (after, after_decisions)):
        compare_predictions(before, values, task=task, atol=1e-6,
                            before_decisions=before_decisions,
                            after_decisions=decisions)
    if reference is None:
        fsutil.write_text(str(root / 'reference.json'), json.dumps({
            'bundle': str(fitted.bundle_dir), 'snapshot_id': snapshot.id,
            'index_path': str(snapshot.index_path),
            'graph_config': snapshot.graph_config, 'inputs': str(inputs),
            'ids': ids, 'classes': classes,
        }))
    else:
        saved = json.loads(fsutil.read_text(str(reference / 'reference.json')))
        assert ids == saved['ids'] and classes == saved['classes']
        query = kumo.PredictiveQuery(
            graph_for_source(reference / 'source', task == 'static_link'),
            QUERIES[task])
        snapshot = GraphSnapshot(saved['snapshot_id'], Path(saved['index_path']),
                                 saved['graph_config'])
        inputs = Path(saved['inputs'])
        common_model = kumo.Trainer.load(saved['bundle'], accelerator='cpu')
        before, before_decisions = predict(common_model, 'shared-reference')
    print('NATIVE_TASK_RESULT=' + json.dumps({
        'task': task, 'backend': os.environ['DG_EDGE_BACKEND'],
        'optimizer_steps': steps, 'ids': ids, 'values': before.tolist(),
        'decisions': (None if before_decisions is None
                      else before_decisions.tolist()),
    }))


@pytest.mark.skipif(os.environ.get('DG_TEST_NATIVE_TASKS') != '1',
                    reason='requires candidate wheel and native Linux I/O')
@pytest.mark.parametrize('task', tuple(TASKS))
def test_native_task(task: str, tmp_path: Path) -> None:
    reference = None
    for backend in BACKENDS:
        directory = tmp_path / backend
        environment = {
            **os.environ, 'DG_EDGE_BACKEND': backend, 'DG_DIRECT_CACHE_GB': '1',
            'DG_IO_RINGS': '2', 'DG_IO_QD': '8', 'RAYON_NUM_THREADS': '2',
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'PYTHONHASHSEED': '0',
        }
        completed = subprocess.run(
            [sys.executable, __file__, task, str(directory),
             str(tmp_path / 'mmap')],
            env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=300, check=False)
        assert completed.returncode == 0, completed.stdout[-20000:]
        result = next(json.loads(line.split('=', 1)[1])
                      for line in completed.stdout.splitlines()
                      if line.startswith('NATIVE_TASK_RESULT='))
        assert result['optimizer_steps'] >= 2
        print(f"{task}/{backend}: {result['optimizer_steps']} optimizer steps, "
              f"{len(result['values'])} predicted entities", flush=True)
        if reference is None:
            reference = result
        compare_predictions(
            reference['values'], result['values'], task=task, atol=1e-6,
            before_decisions=reference['decisions'],
            after_decisions=result['decisions'])


if __name__ == '__main__':
    reference = (Path(sys.argv[3]) if len(sys.argv) > 3
                 and os.environ['DG_EDGE_BACKEND'] != 'mmap' else None)
    run_case(sys.argv[1], Path(sys.argv[2]), reference)
