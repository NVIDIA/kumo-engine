# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every tracking adapter records the same run records as a recorded fit."""

import hashlib
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pytest
from kumoapi.plugin import PublishedModel

from kumo.engine import Event
from kumo.engine._tracking import (
    DROPPED_METRICS,
    DROPPED_PARAMS,
    FitContext,
    TrackingSession,
    run_record,
)
from kumo.engine.tracking import Capabilities, RunRecord

RUN = 'trainer-fit-1'
USER_TAGS = {'team': 'ents', 'kumo.run_id': 'overridden'}
QUERY = 'PREDICT ents.Y FOR EACH ents.ID'
CONTEXT = FitContext(
    run_id=RUN, task_type='regression', snapshot_id='snap@1',
    training_table_id='tt-1', query=QUERY, model_plan={
        'optimization': {'base_lr': [0.01, 0.001], 'max_epochs': 2},
        'training_job': {'tune_metric': 'mae'},
        'column_processing': {
            'encoder_overrides': {'ents.amount ($)': 'numerical'}
        },
    })


def _event(name, trial=None, epoch=None, metrics=None, **params):
    return Event(name=name, run_id=RUN, time=0.0, trial=trial, epoch=epoch,
                 metrics=metrics or {}, params=params)


def _epochs(trial, *values):
    return [
        _event('epoch_ended', trial, epoch, {'train_loss': loss,
                                             'val_mae': mae})
        for epoch, (loss, mae) in enumerate(values, 1)
    ]


# Trial 2's first attempt fails; epoch events carry no attempt, as in core.
EVENTS = [
    _event('trial_started', 1, attempt=1, **{'optimization.base_lr': 0.01}),
    *_epochs(1, (0.5, 0.4), (0.3, 0.35)),
    _event('trial_ended', 1, metrics={'val_mae': 0.35, 'seconds': 3.0},
           status='succeeded', attempt=1),
    _event('trial_started', 2, attempt=1, **{'optimization.base_lr': 0.001}),
    _event('trial_ended', 2, status='failed', attempt=1,
           error='OutOfMemoryError: CUDA out of memory'),
    _event('trial_started', 2, attempt=2, **{'optimization.base_lr': 0.001,
                                             'optimization.batch_size': 256}),
    *_epochs(2, (0.6, 0.5), (0.4, 0.3)),
    _event('trial_ended', 2, metrics={'val_mae': 0.3, 'seconds': 4.0},
           status='succeeded', attempt=2),
    _event('fit_ended', metrics={'val_mae': 0.3, 'test_mae': 0.32,
                                 'seconds': 9.0}, status='succeeded'),
]
RESULT = SimpleNamespace(
    metrics={'val_mae': 0.3, 'test_mae': 0.32},
    published_model=PublishedModel(name='ents-model', version='3',
                                   uri='models:/ents-model/3'))


class _Recorder:
    def __init__(self):
        self.calls: List[Tuple[str, Any]] = []

    def start(self, record):
        self.calls.append(('start', record))

    def on_trial(self, record, event):
        self.calls.append(('on_trial', record))

    def finish(self, record, error):
        self.calls.append(('finish', record))


def replay(trackers, events=EVENTS, result=RESULT,
           error=None) -> Dict[str, RunRecord]:
    r"""Feeds the recorded fit to `trackers`; returns the records expected
    of them."""
    recorder = _Recorder()
    session = TrackingSession([recorder, *trackers])
    session.start(lambda: CONTEXT)
    for event in events:
        session.on_event(event)
        if event.name == 'trial_started':
            session.trial_logger(event.trial, event.params['attempt'])
    session.finish(result, error)
    records = {
        record.tags['kumo.run_id']: record
        for hook, record in recorder.calls if hook == 'on_trial'
    }
    records[RUN] = run_record(CONTEXT, result, events[-1], tags=USER_TAGS)
    return records


@dataclass
class Observed:
    name: Optional[str]
    params: Mapping[str, str]
    metrics: Mapping[str, float]
    steps: Mapping[str, Tuple[Tuple[int, float], ...]]
    tags: Mapping[str, str]
    parent: Optional[str] = None
    links: Mapping[str, str] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Adapter:
    make: Callable[[], Any]
    observe: Callable[[], Dict[str, Observed]]
    kept_params: Callable[[Mapping[str, str]], Mapping[str, str]] = dict


def gaps(expected: Dict[str, RunRecord], observed: Dict[str, Observed],
         capabilities: Capabilities, kept_params: Callable) -> List[str]:
    found = []
    if set(observed) != set(expected):
        return [f'runs {sorted(observed)} != {sorted(expected)}']
    dropped = sum(len(record.params) - len(kept_params(record.params))
                  for record in expected.values())
    counted = sum(int(seen.tags.get(DROPPED_PARAMS, 0))
                  for seen in observed.values())
    if counted != dropped:
        found.append(f'{DROPPED_PARAMS}: {counted} != {dropped}')
    for run_id, record in expected.items():
        seen = observed[run_id]

        def differ(what, have, want):
            if have != want:
                found.append(f'{run_id} {what}: {have!r} != {want!r}')

        if record.name is not None:
            differ('name', seen.name, record.name)
        differ('params', dict(seen.params), dict(kept_params(record.params)))
        differ('metrics', {k: seen.metrics.get(k) for k in record.metrics},
               dict(record.metrics))
        names = set(record.metrics)
        if capabilities.step_metrics:
            names |= set(record.steps)
            differ('steps', {k: seen.steps.get(k) for k in record.steps},
                   dict(record.steps))
        differ('metric names', sorted(seen.metrics), sorted(names))
        differ('tags', {k: seen.tags.get(k) for k in record.tags},
               dict(record.tags))
        differ(
            'kumo tags',
            sorted(k for k in seen.tags if k.startswith('kumo.')
                   and k not in (DROPPED_PARAMS, DROPPED_METRICS)),
            sorted(k for k in record.tags if k.startswith('kumo.')))
        if capabilities.nested_runs:
            differ('parent', seen.parent, record.tags.get('kumo.parent_run'))
        if capabilities.model_link:
            differ('links', dict(seen.links), dict(record.links))
        if capabilities.artifacts:
            differ('artifacts', dict(seen.artifacts), dict(record.artifacts))
    return found


@pytest.fixture
def mlflow_adapter(tmp_path, monkeypatch):
    mlflow = pytest.importorskip('mlflow')
    from kumo.engine.integrations.mlflow import MlflowTracking, _mlflow_params

    monkeypatch.chdir(tmp_path)
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f'sqlite:///{tmp_path / "mlflow.db"}')

    def observe():
        client = mlflow.MlflowClient()
        runs = client.search_runs(
            [e.experiment_id for e in client.search_experiments()])
        kumo_ids = {r.info.run_id: r.data.tags['kumo.run_id'] for r in runs}

        def history(run_id, key):
            return tuple((m.step, m.value) for m in sorted(
                client.get_metric_history(run_id, key),
                key=lambda m: (m.step, m.timestamp)))

        return {
            kumo_ids[run.info.run_id]: Observed(
                name=run.info.run_name, params=run.data.params,
                metrics=run.data.metrics, tags=run.data.tags,
                steps={k: history(run.info.run_id, k)
                       for k in run.data.metrics},
                parent=kumo_ids.get(run.data.tags.get('mlflow.parentRunId')),
                artifacts={
                    a.path: mlflow.artifacts.load_text(
                        f'{run.info.artifact_uri}/{a.path}')
                    for a in client.list_artifacts(run.info.run_id)
                })
            for run in runs
        }

    try:
        yield Adapter(
            make=lambda: MlflowTracking(log_system_metrics=False,
                                        tags=USER_TAGS), observe=observe,
            kept_params=_mlflow_params)
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)


@pytest.fixture(params=['mlflow'])
def adapter(request) -> Adapter:
    return request.getfixturevalue(f'{request.param}_adapter')


def test_records_of_the_recorded_fit():
    records = replay([])

    fit = records[RUN]
    assert fit.tags == {
        'team': 'ents',
        'kumo.kind': 'fit',
        'kumo.engine_version': fit.tags['kumo.engine_version'],
        'kumo.run_id': RUN,
        'kumo.snapshot_id': 'snap@1',
        'kumo.training_table_id': 'tt-1',
        'kumo.query_sha256': hashlib.sha256(QUERY.encode()).hexdigest(),
        'kumo.model_name': 'ents-model',
        'kumo.model_version': '3',
        'kumo.model_uri': 'models:/ents-model/3',
    }
    assert fit.params == {
        'task_type': 'regression',
        'optimization.base_lr': '[0.01, 0.001]',
        'optimization.max_epochs': '2',
        'training_job.tune_metric': 'mae',
        'column_processing.encoder_overrides.ents.amount ($)': 'numerical',
    }
    with pytest.warns(UserWarning, match="'kumo.bundle_id', 'task_type'"):
        spoofed = run_record(CONTEXT, tags={'kumo.bundle_id': 'x',
                                            'task_type': 'y', 'a': 1})
    assert spoofed.tags.keys() - fit.tags.keys() == {'a'}
    assert fit.metrics == {'val_mae': 0.3, 'test_mae': 0.32, 'seconds': 9.0}
    assert fit.artifacts['query.pql'] == QUERY
    assert json.loads(fit.artifacts['model_plan.json']) == CONTEXT.model_plan

    oom = records[f'{RUN}-trial-2-attempt-1']
    assert oom == RunRecord(
        name='trial-2-attempt-1',
        params={'optimization.base_lr': '0.001', 'attempt': '1'}, tags={
            'kumo.kind': 'trial',
            'kumo.run_id': f'{RUN}-trial-2-attempt-1',
            'kumo.parent_run': RUN,
            'kumo.status': 'failed',
            'kumo.error': 'OutOfMemoryError: CUDA out of memory',
        })
    retry = records[f'{RUN}-trial-2-attempt-2']
    assert retry.params['attempt'] == '2'
    assert retry.metrics == {'best_val_mae': 0.3, 'seconds': 4.0}
    assert retry.steps == {'train_loss': ((1, 0.6), (2, 0.4)),
                           'val_mae': ((1, 0.5), (2, 0.3))}


def test_adapter_records_the_same_runs_beside_a_failing_tracker(
        adapter, caplog):
    class RaisesOnTrial(_Recorder):
        def on_trial(self, record, event):
            super().on_trial(record, event)
            raise RuntimeError('tracker down')

    broken, tracker = RaisesOnTrial(), adapter.make()
    expected = replay([broken, tracker])

    assert [hook for hook, _ in broken.calls] == [
        'start', 'on_trial', 'finish'
    ]
    assert broken.calls[-1] == ('finish', None)
    assert caplog.text.count('not called again') == 1
    assert gaps(expected, adapter.observe(), tracker.capabilities,
                adapter.kept_params) == []


def test_adapter_records_a_failed_fit(adapter):
    tracker = adapter.make()
    failed = _event('fit_ended', status='failed', error='ValueError: stop')
    expected = replay([tracker], [*EVENTS[:-1], failed], None,
                      ValueError('stop'))

    assert expected[RUN].tags['kumo.error'] == 'ValueError: stop'
    assert gaps(expected, adapter.observe(), tracker.capabilities,
                adapter.kept_params) == []


def test_only_rank_zero_tracks(monkeypatch):
    monkeypatch.setenv('LOCAL_RANK', '1')
    recorder = _Recorder()
    session = TrackingSession(recorder)

    session.start(lambda: CONTEXT)
    session.finish(RESULT, None)

    assert recorder.calls == []
