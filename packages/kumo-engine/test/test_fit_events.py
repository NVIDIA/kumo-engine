# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import threading

import numpy as np
import pandas as pd
import pytest
from kumoapi.model_plan import ModelPlan

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.tracking import Event


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
    assert all(isinstance(event, Event) for event in events)
    names = [event.name for event in events]
    assert names == [
        'fit_started',
        'trial_started', 'epoch_ended', 'epoch_ended', 'trial_ended',
        'trial_started', 'epoch_ended', 'epoch_ended', 'trial_ended',
        'fit_ended',
    ]
    assert set(events[0].params) == {'snapshot_id', 'query_sha256'}
    assert [(e.trial, e.epoch) for e in events if e.name == 'epoch_ended'
            ] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert any(name.startswith('val_') for name in events[2].metrics)
    assert events[2].metrics['train_epoch_seconds'] > 0
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


@pytest.mark.parametrize('on_error', ['warn', 'raise'])
def test_an_unfinished_tracker_is_reported_by_the_error_policy(
        table_and_snapshot, tmp_path, monkeypatch, on_error):
    import kumo.engine._tracking as tracking

    monkeypatch.setattr(tracking, 'FINISH_SECONDS', 0.5)
    release = threading.Event()

    class Blocked:
        def start(self, record):
            pass

        def on_trial(self, record, event):
            pass

        def finish(self, record, error):
            release.wait(30)

    table, snapshot = table_and_snapshot
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 2
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(plan)
    cfg = kumo.MLOps(str(tmp_path / 'root'), tracking=Blocked(), store=False,
                     on_error=on_error)
    try:
        if on_error == 'warn':
            with pytest.warns(UserWarning, match='TimeoutError: experiment '
                              'tracking did not finish within 0.5s'):
                assert trainer.fit(table, snapshot=snapshot,
                                   mlops=cfg) is trainer.result
        else:
            with pytest.raises(RuntimeError) as raised:
                trainer.fit(table, snapshot=snapshot, mlops=cfg)
            assert isinstance(raised.value.__cause__, TimeoutError)
    finally:
        release.set()
    assert trainer.result.bundle_dir is not None


def test_event_metrics_keep_finite_numbers():
    from kumo.engine._events import EventEmitter

    received = []
    EventEmitter([received.append]).emit(
        'epoch_ended', metrics={'val_loss': 'n/a', 'val_auc': float('nan'),
                                'val_mae': 0.5})

    assert received[0].metrics == {'val_mae': 0.5}


def test_epoch_event_without_recorded_metrics_does_not_raise(tmp_path):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from kumo_core.callbacks.train_progress_logger import (
        AutoTrainingProgressLogger,
    )

    received = []
    logger = AutoTrainingProgressLogger(
        str(tmp_path), 1, on_event=lambda *a, **k: received.append(a))
    progress = SimpleNamespace(num_epochs_completed=1, metrics=[],
                               estimated_train_time=0.0)
    logger.set_current_trainer(
        SimpleNamespace(training_progress_logger=SimpleNamespace(
            progress=progress)), trial=1)
    logger.progress.start_time = datetime.now(timezone.utc)

    logger.on_train_epoch_end(None, None)

    assert received == []


def test_a_snapshot_reports_progress_events(graph, tmp_path):
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        events, again = [], []
        snapshot = graph.snapshot(callbacks=[events.append])
        graph.snapshot(callbacks=[again.append])
    finally:
        kumo.configure(scratch_dir=previous)

    assert [event.name for event in events] == [
        'snapshot_started', 'table_staged', 'table_staged', 'snapshot_ended']
    assert [event.params['table'] for event in events[1:3]] == [
        'users', 'events']
    assert events[-1].params['snapshot_id'] == snapshot.id
    assert events[-1].params['reused'] is False
    assert again[-1].params['reused'] is True
    assert all(event.params['seconds'] >= 0 for event in events)


def test_kumo_calls_drop_lightning_tips_and_only_those(caplog):
    from pytorch_lightning.utilities.rank_zero import rank_zero_info

    from kumo.engine.trainer.trainer import _without_torch_scatter_hint

    @_without_torch_scatter_hint
    def fit():
        rank_zero_info('💡 Tip: inside')
        rank_zero_info('GPU available: False')

    lightning = logging.getLogger('pytorch_lightning')
    lightning.addHandler(caplog.handler)  # it does not propagate
    try:
        with caplog.at_level('INFO', logger='pytorch_lightning'):
            fit()
            rank_zero_info('💡 Tip: outside')
    finally:
        lightning.removeHandler(caplog.handler)
    assert 'inside' not in caplog.text
    assert 'GPU available' in caplog.text and 'outside' in caplog.text


def test_an_event_prints_as_its_name_and_fields():
    event = Event(name='table_staged', time=1.0,
                  params={'table': 'users', 'seconds': 12.0})
    assert str(event) == 'table_staged table=users seconds=12.0'
    assert str(Event(name='epoch_ended', time=1.0, trial=1, epoch=2,
                     metrics={'val_auroc': 0.5})) == (
        'epoch_ended trial=1 epoch=2 val_auroc=0.5')


def test_kumo_calls_drop_the_fork_and_iterable_dataset_notes():
    import warnings

    from kumo.engine.trainer.trainer import _without_torch_scatter_hint

    @_without_torch_scatter_hint
    def fit():
        warnings.warn('This process (pid=1) is multi-threaded, use of fork() '
                      'may lead to deadlocks in the child.',
                      DeprecationWarning)
        warnings.warn('Your `IterableDataset` has `__len__` defined.',
                      UserWarning)
        warnings.warn('something else', UserWarning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit()
    assert [str(warning.message) for warning in caught] == ['something else']
