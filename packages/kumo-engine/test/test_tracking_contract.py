# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every tracking adapter records the same run records.

A recorded fake fit (fixed context, events and result) is replayed through
`TrackingSession` into an adapter; what the adapter stored is read back and
compared with `run_record`/`trial_record`, allowing only the gaps its
`Capabilities` declare and the params its platform limits drop (counted in
``kumo.dropped_params``). A new adapter is one more `adapter` fixture param.
"""

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
    TrackingSession,
)
from kumo.engine.tracking import (
    Capabilities,
    FitContext,
    RunRecord,
    run_record,
    trial_record,
)

RUN = 'trainer-fit-1'
USER_TAGS = {'team': 'ents', 'kumo.run_id': 'overridden'}
QUERY = 'PREDICT ents.Y FOR EACH ents.ID'
CONTEXT = FitContext(
    run_id=RUN, task_type='regression', snapshot_id='snap@1',
    training_table_id='tt-1', query=QUERY, model_plan={
        'optimization': {
            'base_lr': [0.01, 0.001],
            'max_epochs': 2
        },
        'training_job': {
            'tune_metric': 'mae'
        },
        'column_processing': {
            'encoder_overrides': {
                'ents.amount ($)': 'numerical'
            }
        },
    })


def _event(name, trial=None, epoch=None, metrics=None, **params):
    return Event(name=name, run_id=RUN, time=0.0, trial=trial, epoch=epoch,
                 metrics=metrics or {}, params=params)


def _epochs(trial, *values):
    return [
        _event('epoch_ended', trial, epoch, {
            'train_loss': loss,
            'val_mae': mae
        }) for epoch, (loss, mae) in enumerate(values, 1)
    ]


# Two trials of two epochs; trial 2's first attempt runs out of memory. Epoch
# events carry no attempt, as kumo-core emits them.
EVENTS = [
    _event('trial_started', 1, attempt=1, **{'optimization.base_lr': 0.01}),
    *_epochs(1, (0.5, 0.4), (0.3, 0.35)),
    _event('trial_ended', 1, metrics={
        'val_mae': 0.35,
        'seconds': 3.0
    }, status='succeeded', attempt=1),
    _event('trial_started', 2, attempt=1, **{'optimization.base_lr': 0.001}),
    _event('trial_ended', 2, status='failed', attempt=1,
           error='OutOfMemoryError: CUDA out of memory'),
    _event('trial_started', 2, attempt=2, **{
        'optimization.base_lr': 0.001,
        'optimization.batch_size': 256
    }),
    *_epochs(2, (0.6, 0.5), (0.4, 0.3)),
    _event('trial_ended', 2, metrics={
        'val_mae': 0.3,
        'seconds': 4.0
    }, status='succeeded', attempt=2),
    _event('fit_ended', metrics={
        'val_mae': 0.3,
        'test_mae': 0.32,
        'seconds': 9.0
    }, status='succeeded'),
]
RESULT = SimpleNamespace(
    metrics={
        'val_mae': 0.3,
        'test_mae': 0.32
    }, published_model=PublishedModel(name='ents-model', version='3',
                                      uri='models:/ents-model/3'))


class _Recorder:
    def __init__(self):
        self.calls: List[Tuple[str, Any]] = []

    def start(self, context):
        self.calls.append(('start', context))

    def on_event(self, event):
        self.calls.append(('on_event', event))

    def finish(self, result, error):
        self.calls.append(('finish', result))

    @property
    def events(self):
        return [value for hook, value in self.calls if hook == 'on_event']


class _Raising:
    def _raise(self, *args):
        raise RuntimeError('tracker down')

    start = on_event = trial_logger = finish = _raise


def replay(trackers, events=EVENTS, result=RESULT,
           error=None) -> List[Event]:
    r"""Feeds the recorded fit to `trackers` as `Trainer.fit` would; returns
    the events trackers saw. Trial loggers are requested but not fed: what
    Lightning logs through them is outside the record."""
    recorder = _Recorder()
    session = TrackingSession([recorder, *trackers])
    session.start(lambda: CONTEXT)
    for event in events:
        session.on_event(event)
        if event.name == 'trial_started':
            session.trial_logger(event.trial, event.params['attempt'])
    session.finish(result, error)
    return recorder.events


def expected_records(events: List[Event],
                     result=RESULT) -> Dict[str, RunRecord]:
    trials: Dict[Tuple[int, int], List[Event]] = {}
    for event in events:
        if event.trial is not None:
            trials.setdefault((event.trial, event.params['attempt']),
                              []).append(event)
    ended = next(event for event in events if event.name == 'fit_ended')
    records = [
        run_record(CONTEXT, result, ended, tags=USER_TAGS),
        *(trial_record(CONTEXT, group) for group in trials.values())
    ]
    return {record.tags['kumo.run_id']: record for record in records}


@dataclass
class Observed:
    r"""One run as an adapter stored it, keyed by its ``kumo.run_id``."""

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


def gaps(
        expected: Dict[str, RunRecord], observed: Dict[str, Observed],
        capabilities: Capabilities,
        kept_params: Callable[[Mapping[str, str]], Mapping[str,
                                                           str]]) -> List[str]:
    r"""Differences not explained by `capabilities` or the platform limits."""
    found = []
    if set(observed) != set(expected):
        return [f'runs {sorted(observed)} != {sorted(expected)}']
    dropped = sum(
        len(record.params) - len(kept_params(record.params))
        for record in expected.values())
    counted = sum(
        int(seen.tags.get(DROPPED_PARAMS, 0)) for seen in observed.values())
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
        differ('metrics', {k: seen.metrics.get(k)
                           for k in record.metrics}, dict(record.metrics))
        names = set(record.metrics)
        if capabilities.step_metrics:
            names |= set(record.steps)
            differ('steps', {k: seen.steps.get(k)
                             for k in record.steps}, dict(record.steps))
        differ('metric names', sorted(seen.metrics), sorted(names))
        differ('tags', {k: seen.tags.get(k)
                        for k in record.tags}, dict(record.tags))
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
        observed = {}
        for run in runs:
            run_id = run.info.run_id
            observed[kumo_ids[run_id]] = Observed(
                name=run.info.run_name, params=run.data.params,
                metrics=run.data.metrics, steps={
                    key:
                    tuple(
                        (m.step, m.value)
                        for m in sorted(client.get_metric_history(run_id, key),
                                        key=lambda m: (m.step, m.timestamp)))
                    for key in run.data.metrics
                }, tags=run.data.tags,
                parent=kumo_ids.get(run.data.tags.get('mlflow.parentRunId')),
                artifacts={
                    a.path:
                    mlflow.artifacts.load_text(
                        f'{run.info.artifact_uri}/{a.path}')
                    for a in client.list_artifacts(run_id)
                })
        return observed

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
    records = expected_records(replay([]))

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
        spoofed = run_record(CONTEXT, tags={
            'kumo.bundle_id': 'x',
            'task_type': 'y',
            'a': 1
        })
    assert spoofed.tags.keys() - fit.tags.keys() == {'a'}
    assert fit.metrics == {'val_mae': 0.3, 'test_mae': 0.32, 'seconds': 9.0}
    assert fit.links == {
        'model_name': 'ents-model',
        'model_version': '3',
        'model_uri': 'models:/ents-model/3'
    }
    assert fit.artifacts['query.pql'] == QUERY
    assert json.loads(fit.artifacts['model_plan.json']) == CONTEXT.model_plan

    oom = records[f'{RUN}-trial-2-attempt-1']
    assert oom == RunRecord(
        name='trial-2-attempt-1', params={
            'optimization.base_lr': '0.001',
            'attempt': '1'
        }, tags={
            'kumo.kind': 'trial',
            'kumo.run_id': f'{RUN}-trial-2-attempt-1',
            'kumo.parent_run': RUN,
            'kumo.status': 'failed',
            'kumo.error': 'OutOfMemoryError: CUDA out of memory',
        })
    retry = records[f'{RUN}-trial-2-attempt-2']
    assert retry.params['attempt'] == '2'
    assert retry.tags['kumo.status'] == 'succeeded'
    assert 'kumo.error' not in retry.tags
    assert retry.metrics == {'best_val_mae': 0.3, 'seconds': 4.0}
    assert retry.steps == {
        'train_loss': ((1, 0.6), (2, 0.4)),
        'val_mae': ((1, 0.5), (2, 0.3))
    }
    assert len(records) == 4


def test_adapter_records_the_same_runs(adapter):
    tracker = adapter.make()
    events = replay([tracker])

    assert gaps(expected_records(events), adapter.observe(),
                tracker.capabilities, adapter.kept_params) == []


def test_adapter_records_a_failed_fit(adapter):
    tracker = adapter.make()
    failed = _event('fit_ended', status='failed', error='ValueError: stop')
    events = replay([tracker], [*EVENTS[:-1], failed], None,
                    ValueError('stop'))

    expected = expected_records(events, None)
    assert expected[RUN].tags['kumo.error'] == 'ValueError: stop'
    assert gaps(expected, adapter.observe(), tracker.capabilities,
                adapter.kept_params) == []


def test_a_failing_tracker_does_not_affect_others(adapter, caplog):
    tracker = adapter.make()
    events = replay([_Raising(), tracker])

    assert gaps(expected_records(events), adapter.observe(),
                tracker.capabilities, adapter.kept_params) == []
    hooks = 2 + len(EVENTS) + sum(e.name == 'trial_started' for e in EVENTS)
    assert caplog.text.count('Experiment tracking failed') == hooks


def test_gaps_compare_step_histories_when_declared():
    expected = expected_records(replay([]))
    observed = {
        run_id:
        Observed(
            name=record.name, params=record.params, metrics={
                **record.metrics,
                **{
                    k: v[-1][1]
                    for k, v in record.steps.items()
                }
            }, steps=record.steps, tags=record.tags)
        for run_id, record in expected.items()
    }
    steps = Capabilities(step_metrics=True)
    assert gaps(expected, observed, steps, dict) == []

    retry = observed[f'{RUN}-trial-2-attempt-2']
    retry.steps = {**retry.steps, 'val_mae': ((10, 0.5), (20, 0.3))}
    assert len(gaps(expected, observed, steps, dict)) == 1
    undeclared = gaps(expected, observed, Capabilities(), dict)
    assert [gap.split()[1:3]
            for gap in undeclared] == [['metric', 'names:']] * 2


def test_session_merges_trial_loggers_and_skips_trackers_without_one():
    class WithLogger(_Recorder):
        def __init__(self, pl_logger):
            super().__init__()
            self.pl_logger = pl_logger

        def trial_logger(self, trial, attempt):
            return self.pl_logger

    one, two = object(), object()
    session = TrackingSession(_Recorder())
    assert session.trial_logger(1, 1) is None
    session = TrackingSession([_Recorder(), WithLogger(one), WithLogger(None)])
    assert session.trial_logger(1, 1) == [one]
    session = TrackingSession([WithLogger(one), WithLogger(two)])
    assert session.trial_logger(1, 1) == [one, two]


def test_only_rank_zero_tracks(monkeypatch):
    monkeypatch.setenv('LOCAL_RANK', '1')
    recorder = _Recorder()
    session = TrackingSession(recorder)

    session.start(lambda: CONTEXT)
    session.on_event(EVENTS[0])
    session.finish(RESULT, None)

    assert not session.enabled
    assert recorder.calls == []
