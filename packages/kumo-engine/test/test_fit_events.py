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
def table_and_snapshot(tmp_path):
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


def test_fit_reports_progress_events(table_and_snapshot, monkeypatch,
                                     caplog):
    from kumo_core.automl.optuna_searcher import TPESearcher

    update = TPESearcher.update

    def fail_second_trial(self, *args, **kwargs):
        if self._num_updates:
            raise RuntimeError('searcher down')
        self._num_updates = 1
        return update(self, *args, **kwargs)

    monkeypatch.setattr(TPESearcher, '_num_updates', 0, raising=False)
    monkeypatch.setattr(TPESearcher, 'update', fail_second_trial)
    table, snapshot = table_and_snapshot
    plan = ModelPlan()
    plan.training_job.num_experiments = 2
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 2
    plan.optimization.max_steps_per_epoch = 2
    events = []
    broken_calls = []

    def broken(event):
        broken_calls.append(event)
        raise RuntimeError('callback bug')

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
    assert set(events[0].params) == {'snapshot_id', 'query_sha256'}
    assert [(e.trial, e.epoch) for e in events if e.name == 'epoch_ended'
            ] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert any(name.startswith('val_') for name in events[2].metrics)
    assert events[1].params['attempt'] == 1
    assert events[4].params == {'status': 'succeeded', 'attempt': 1}
    assert {'val_mae', 'seconds'} <= set(events[4].metrics)
    assert events[8].params == {
        'status': 'failed',
        'attempt': 1,
        'error': 'RuntimeError: searcher down'
    }
    assert events[-1].metrics == {**result.metrics,
                                  'seconds': events[-1].metrics['seconds']}
    assert events[-1].params == {'status': 'succeeded'}
    assert [json.loads(json.dumps(event.to_dict()))['name']
            for event in events] == names


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
    assert events[-1].params == {
        'status': 'failed',
        'error': 'ValueError: stop'
    }


def test_event_metrics_keep_finite_numbers():
    from kumo.engine._events import EventEmitter

    received = []
    EventEmitter('run', [received.append]).emit(
        'epoch_ended', metrics={'val_loss': 'n/a', 'val_auc': float('nan'),
                                'val_mae': 0.5})

    assert received[0].metrics == {'val_mae': 0.5}
