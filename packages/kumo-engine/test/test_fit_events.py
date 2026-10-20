# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pandas as pd
import pytest
from kumoapi.model_plan import ModelPlan

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir


@pytest.fixture
def table_and_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr('kumo.engine._events._progress', None)
    previous_scratch = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        rng = np.random.default_rng(0)
        (tmp_path / 'data').mkdir()
        pd.DataFrame({
            'ID': np.arange(400),
            'Y': rng.random(400),
            'F1': rng.normal(size=400),
        }).to_parquet(tmp_path / 'data' / 'ents.parquet')
        connector = kumo.FileConnector(root_dir=str(tmp_path / 'data'))
        ents = kumo.Table.from_source_table(connector['ents.parquet'],
                                            primary_key='ID').infer_metadata()
        graph = kumo.Graph(tables={'ents': ents}, edges=[])
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph=graph,
                                     query='PREDICT ents.Y FOR EACH ents.ID')
        yield query.generate_training_table(snapshot), snapshot
    finally:
        kumo.configure(scratch_dir=previous_scratch)


def _bundle_files(result):
    return sorted(
        str(path.relative_to(result.bundle_dir))
        for path in result.bundle_dir.rglob('*'))


def test_fit_reports_progress_events(table_and_snapshot, capsys, caplog,
                                     monkeypatch):
    import kumo.engine.trainer._runtime as runtime_module

    table, snapshot = table_and_snapshot
    plan = ModelPlan()
    plan.training_job.num_experiments = 2
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 2
    plan.optimization.max_steps_per_epoch = 2
    bind = runtime_module.local_training_bindings
    sinks = []

    def spy(*args, on_event=None, **kwargs):
        sinks.append(on_event)
        return bind(*args, on_event=on_event, **kwargs)

    monkeypatch.setattr(runtime_module, 'local_training_bindings', spy)
    monkeypatch.delenv('KUMO_ENGINE_PROGRESS', raising=False)
    kumo.configure(progress='off')
    baseline = kumo.Trainer(plan).fit(table, snapshot=snapshot)

    assert sinks == [None]
    assert '"event": "kumo.engine"' not in capsys.readouterr().out

    events = []
    broken_calls = []

    def broken(event):
        broken_calls.append(event)
        raise RuntimeError('callback bug')

    kumo.configure(progress='json')
    result = kumo.Trainer(plan).fit(table, snapshot=snapshot,
                                    callbacks=[broken, events.append])

    assert len(broken_calls) == 1
    assert 'callback bug' in caplog.text
    names = [event.name for event in events]
    assert names == [
        'fit_started',
        'trial_started', 'epoch_ended', 'epoch_ended', 'trial_ended',
        'trial_started', 'epoch_ended', 'epoch_ended', 'trial_ended',
        'fit_ended',
    ]
    assert len({event.run_id for event in events}) == 1
    assert [event.time for event in events] == sorted(
        event.time for event in events)
    assert set(events[0].params) == {'snapshot_id', 'query_sha256'}
    assert [(e.trial, e.epoch) for e in events if e.name == 'epoch_ended'
            ] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    for event in events:
        if event.name == 'epoch_ended':
            assert any(name.startswith('val_') for name in event.metrics)
            assert any(name.startswith('train_') for name in event.metrics)
        if event.name == 'trial_started':
            assert event.params['attempt'] == 1
        if event.name == 'trial_ended':
            assert event.params == {'status': 'succeeded', 'attempt': 1}
            assert {'val_mae', 'seconds'} <= set(event.metrics)
    assert events[-1].metrics == {**result.metrics,
                                  'seconds': events[-1].metrics['seconds']}
    assert set(result.metrics) == set(baseline.metrics)
    assert _bundle_files(result) == _bundle_files(baseline)
    assert events[-1].params == {'status': 'succeeded'}

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()
             if line.startswith('{"event": "kumo.engine"')]
    assert lines == [{'event': 'kumo.engine', **event.to_dict()}
                     for event in events]


def test_failed_fit_reports_fit_ended(table_and_snapshot, monkeypatch):
    import kumo.engine.trainer.trainer as trainer_module

    def fail(*args, **kwargs):
        raise ValueError('stop')

    monkeypatch.setattr(trainer_module, 'build_generated_training_task_dataset',
                        fail)
    table, snapshot = table_and_snapshot
    events = []

    with pytest.raises(ValueError, match='stop'):
        kumo.Trainer(ModelPlan()).fit(table, snapshot=snapshot,
                                      callbacks=[events.append])

    assert [event.name for event in events] == ['fit_started', 'fit_ended']
    assert events[-1].params == {'status': 'failed', 'error': 'ValueError'}


def test_event_errors_are_logged_not_raised(caplog):
    from kumo_core.automl.auto_trainer import AutoTrainer

    from kumo.engine._events import EventEmitter

    received = []
    emitter = EventEmitter('run', [received.append])
    emitter.emit('epoch_ended', metrics={'val_loss': 'n/a', 'val_mae': 0.5})
    emitter.emit('epoch_ended', params=5)
    emitter.core_sink({'name': 'epoch_ended', 'unknown': 1})
    auto_trainer = AutoTrainer.__new__(AutoTrainer)
    auto_trainer.on_event = emitter.core_sink
    auto_trainer._emit('trial_started', lambda: 1 / 0)

    assert [event.metrics for event in received] == [{'val_mae': 0.5}]
    assert [record.getMessage() for record in caplog.records] == [
        "Could not build progress event 'epoch_ended'",
        "Could not report progress event {'name': 'epoch_ended', "
        "'unknown': 1}",
        "Could not report progress event 'trial_started'",
    ]
