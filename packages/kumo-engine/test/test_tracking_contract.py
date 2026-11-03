# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pytest
from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.summaries import Dataset, Scalar, Table

from kumo.engine._tracking import (
    DROPPED_METRICS,
    DROPPED_PARAMS,
    TrackingSession,
    run_record,
)
from kumo.engine._version import __version__
from kumo.engine.tracking import Event, RunRecord
from kumo.engine.trainer._run_spec import FitContext, fit_spec, fitted_spec

FIT = 'fit'
USER_TAGS = {'team': 'ents', 'kumo.kind': 'overridden'}
QUERY = 'PREDICT ents.Y FOR EACH ents.ID'
INPUTS = {
    'ents': {'connector': 'S3', 'location': 's3://bucket/ents.parquet',
             'version': '1790000000', 'version_kind': 'timestamp'},
    'orders': {'connector': 'DATABRICKS',
               'location': '`main`.`sales`.`orders`',
               'version': '7_1790000000', 'version_kind': 'exact'},
    'users': {'connector': 'SNOWFLAKE', 'location': '"DB"."SCH"."USERS"',
              'version': '1790000000', 'version_kind': 'timestamp'},
}
SNAPSHOT = f"{'s' * 64}@{'v' * 16}"
CONTEXT = FitContext(
    query=QUERY, snapshot_id=SNAPSHOT, task_type='regression',
    started=datetime(2026, 10, 29, 6, 18, tzinfo=timezone.utc),
    model_plan={
        'optimization': {'base_lr': [0.01, 0.001], 'max_epochs': 2},
        'training_job': {'tune_metric': 'mae'},
        'column_processing': {
            'encoder_overrides': {'ents.amount ($)': 'numerical'}
        },
    })


def _event(name, trial=None, epoch=None, metrics=None, **params):
    return Event(name=name, time=0.0, trial=trial, epoch=epoch,
                 metrics=metrics or {}, params=params)


def _epochs(trial, *values):
    return [
        _event('epoch_ended', trial, epoch, {'train_loss': loss,
                                             'val_mae': mae})
        for epoch, (loss, mae) in enumerate(values, 1)
    ]


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
SELECTED = {'train_loss': ((1, 0.6), (2, 0.4)), 'val_mae': ((1, 0.5), (2, 0.3))}
BUNDLE = 'b' * 64
RESULT = SimpleNamespace(metrics={'val_mae': 0.3, 'test_mae': 0.32},
                         id=BUNDLE, bundle_dir=Path('no-bundle'),
                         registered_model=None, registered_alias=None)


class _Recorder:
    def __init__(self, tracker=None):
        self.calls: List[Tuple[str, Any]] = []
        self.tracker = tracker
        self.tags = getattr(tracker, 'tags', None)

    def prepare(self):
        getattr(self.tracker, 'prepare', lambda: None)()

    def start(self, record):
        self.calls.append(('start', record))
        if self.tracker is not None:
            self.tracker.start(record)

    def on_trial(self, record, event):
        self.calls.append(('on_trial', record))
        if self.tracker is not None:
            self.tracker.on_trial(record, event)

    def log_metrics(self, metrics, step):
        self.calls.append(('log_metrics', (metrics, step)))
        getattr(self.tracker, 'log_metrics', lambda *_: None)(metrics, step)

    def finish(self, record, error):
        self.calls.append(('finish', record))
        if self.tracker is not None:
            self.tracker.finish(record, error)


def join_tracking_threads():
    for thread in threading.enumerate():
        if thread.name == 'kumo-tracking':
            thread.join(30)


def replay(tracker=None, events=EVENTS, result=RESULT, error=None,
           context=CONTEXT, series=None,
           summaries=()) -> Dict[str, RunRecord]:
    recorder = _Recorder(tracker)
    session = TrackingSession(recorder)
    session.start(lambda: fit_spec(context))
    for event in events:
        session.on_event(event)
    session.finish(
        None if result is None else lambda: fitted_spec(
            session.spec, result, summaries=summaries), error)
    records = {
        record.name: record
        for hook, record in recorder.calls if hook == 'on_trial'
    }
    spec = replace(fit_spec(context), tags=USER_TAGS)
    if result is not None:
        spec = fitted_spec(spec, result)
    spec = replace(spec, steps=series if series is not None else {
        **({} if error else SELECTED)})
    if series is None and not error:
        spec = replace(spec, parents={**spec.parents, 'best_trial': 2})
    if 'error' in events[-1].params:
        spec = replace(spec, error=str(events[-1].params['error']))
    records[FIT] = run_record(spec)
    return records


@dataclass
class Observed:
    name: Optional[str]
    params: Mapping[str, str]
    metrics: Mapping[str, float]
    steps: Mapping[str, Tuple[Tuple[int, float], ...]]
    tags: Mapping[str, str]
    parent: Optional[str] = None
    artifacts: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Adapter:
    make: Callable[[], Any]
    observe: Callable[[], Dict[str, Observed]]
    kept_params: Callable[[Mapping[str, str]], Mapping[str, str]] = dict
    nested_runs: bool = False
    step_metrics: bool = False
    artifacts: bool = False
    run_name: Callable[[str], Optional[str]] = str
    expected: Callable[[Dict[str, RunRecord]], Dict[str, RunRecord]] = dict


def gaps(expected: Dict[str, RunRecord], observed: Dict[str, Observed],
         adapter: Adapter) -> List[str]:
    kept_params = adapter.kept_params
    expected = adapter.expected(expected)
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
            differ('name', adapter.run_name(seen.name),
                   adapter.run_name(record.name))
        differ('params', dict(seen.params), dict(kept_params(record.params)))
        differ('metrics', {k: seen.metrics.get(k) for k in record.metrics},
               dict(record.metrics))
        names = set(record.metrics)
        if adapter.step_metrics:
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
        if adapter.nested_runs:
            differ('parent', seen.parent,
                   FIT if record.tags['kumo.kind'] == 'trial' else None)
        if adapter.artifacts:
            differ('artifacts', dict(seen.artifacts), dict(record.artifacts))
    return found


def _trial(run):
    trial = run.info.run_name.split('-')[1]
    return f"trial-{trial}-attempt-{run.data.params['attempt']}"


@pytest.fixture
def mlflow_adapter(tracking_uri):
    import mlflow

    from kumo.engine.integrations.mlflow import MlflowTracking, _mlflow_params

    def observe():
        client = mlflow.MlflowClient()
        runs = client.search_runs(
            [e.experiment_id for e in client.search_experiments()])
        kumo_ids = {
            r.info.run_id:
            FIT if r.data.tags['kumo.kind'] == 'fit' else _trial(r)
            for r in runs
        }

        def history(run_id, key):
            return tuple((m.step, m.value) for m in sorted(
                client.get_metric_history(run_id, key),
                key=lambda m: (m.step, m.timestamp)))

        return {
            kumo_ids[run.info.run_id]: Observed(
                name=(run.info.run_name if run.data.tags['kumo.kind'] != 'trial'
                      else _trial(run)), params=run.data.params,
                metrics=run.data.metrics, tags=run.data.tags,
                steps={k: history(run.info.run_id, k)
                       for k in run.data.metrics},
                parent=kumo_ids.get(run.data.tags.get('mlflow.parentRunId')),
                artifacts={
                    a.path: mlflow.artifacts.load_text(
                        f'{run.info.artifact_uri}/{a.path}')
                    for a in client.list_artifacts(run.info.run_id)
                    if not a.is_dir
                })
            for run in runs
        }

    return Adapter(
        make=lambda: MlflowTracking(log_system_metrics=False, tags=USER_TAGS),
        observe=observe, kept_params=_mlflow_params, nested_runs=True,
        step_metrics=True, artifacts=True)


@pytest.fixture
def snowflake_adapter(monkeypatch):
    import _snowflake_experiment_fake

    fake = _snowflake_experiment_fake.install(monkeypatch)
    from kumo.engine.integrations.snowflake import SnowflakeTracking

    def observe():
        observed = {}
        for run_name, run in fake._instance.runs.items():
            names: Dict[Optional[str], Dict[str, Dict[str, Any]]] = {}

            def bucket(key):
                prefix, _, name = key.partition('/')
                if not prefix.startswith('TRIAL_'):
                    prefix, name = '', key
                return names.setdefault(prefix or None, {
                    'params': {}, 'tags': {}, 'metrics': {}, 'steps': {}
                }), name

            for key, value in run['params'].items():
                runs, name = bucket(key)
                if name.startswith('kumo.') or name in USER_TAGS:
                    runs['tags'][name] = value
                else:
                    runs['params'][name] = value
            for key, value, step in run['metrics']:
                if key == 'duration_seconds':
                    continue
                runs, name = bucket(key)
                runs['metrics'][name] = value
                runs['steps'][name] = (*runs['steps'].get(name, ()),
                                       (step, value))
            for prefix, runs in names.items():
                observed[prefix or FIT] = Observed(
                    name=prefix or run_name,
                    artifacts=run['artifacts'] if prefix is None else {},
                    **runs)
        return observed

    def expected(records):
        kept = {}
        for key, record in records.items():
            if record.tags['kumo.kind'] == 'trial':
                trial, attempt = key.split('-')[1::2]
                key = f'TRIAL_{trial}' + (
                    f'_ATTEMPT_{attempt}' if attempt != '1' else '')
                record = replace(record, steps={}, metrics={
                    name: value for name, value in record.metrics.items()
                    if name == 'val/best_mae'}, tags={
                        name: value for name, value in record.tags.items()
                        if name in ('kumo.status', 'kumo.error')})
            kept[key] = record
        return kept

    return Adapter(
        make=lambda: SnowflakeTracking(_snowflake_experiment_fake.FakeSession(),
                                       experiment_name='ents', tags=USER_TAGS),
        observe=observe, step_metrics=True, artifacts=True,
        run_name=lambda name: None, expected=expected)


@pytest.fixture(params=['mlflow', 'snowflake'])
def adapter(request) -> Adapter:
    return request.getfixturevalue(f'{request.param}_adapter')


def test_records_of_the_recorded_fit():
    records = replay()

    fit = records[FIT]
    assert fit.name == 'fit-ents-regression-20261029-061800'
    assert (fit.model_label, fit.started) == ('ents-regression',
                                              CONTEXT.started)
    assert fit.tags == {'team': 'ents', 'kumo.kind': 'fit',
                        'kumo.bundle_id': BUNDLE, 'kumo.snapshot_id': SNAPSHOT,
                        'kumo.task_type': 'regression',
                        'kumo.entity': 'ents', 'kumo.tune_metric': 'mae',
                        'kumo.engine_version': __version__,
                        'kumo.best_trial': '2'}
    assert fit.params == {
        'optimization.base_lr': '[0.01, 0.001]',
        'optimization.max_epochs': '2',
        'training_job.tune_metric': 'mae',
        'column_processing.encoder_overrides.ents.amount ($)': 'numerical',
    }
    with pytest.warns(UserWarning,
                      match="'kumo.bundle_id', 'training_job.tune_metric'"):
        spoofed = run_record(replace(fit_spec(CONTEXT), tags={
            'kumo.bundle_id': 'x', 'training_job.tune_metric': 'y', 'a': 1}))
    assert spoofed.tags.keys() - fit.tags.keys() == {'a'}
    assert fit.metrics == {'test/mae': 0.32}
    assert fit.steps == {'train/loss': ((1, 0.6), (2, 0.4)),
                         'val/mae': ((1, 0.5), (2, 0.3))}
    registered = SimpleNamespace(**{
        **vars(RESULT), 'registered_alias': 'candidate',
        'registered_model': SimpleNamespace(name='ents', version='3')})
    tags = run_record(fitted_spec(fit_spec(CONTEXT), registered)).tags
    assert tags.items() >= {'kumo.model_name': 'ents',
                            'kumo.model_version': '3'}.items()
    assert 'kumo.model_alias' not in tags
    assert fit.artifacts['query.pql'] == QUERY
    assert json.loads(fit.artifacts['model_plan.json']) == CONTEXT.model_plan

    oom = records['trial-2-attempt-1']
    assert oom == RunRecord(
        name='trial-2-attempt-1', model_label='ents-regression',
        params={'optimization.base_lr': '0.001', 'attempt': '1'}, tags={
            'kumo.kind': 'trial',
            'kumo.engine_version': __version__,
            'kumo.snapshot_id': SNAPSHOT,
            'kumo.task_type': 'regression',
            'kumo.status': 'failed',
            'kumo.error': 'OutOfMemoryError: CUDA out of memory',
        })
    retry = records['trial-2-attempt-2']
    assert retry.params['attempt'] == '2'
    assert retry.metrics == {'val/best_mae': 0.3}
    assert retry.steps == {'train/loss': ((1, 0.6), (2, 0.4)),
                           'val/mae': ((1, 0.5), (2, 0.3))}


def test_adapter_records_the_same_runs(adapter, tmp_path):
    (tmp_path / PROVENANCE_FILE).write_text('{"bundle": "b"}')
    result = SimpleNamespace(**{**vars(RESULT), 'bundle_dir': tmp_path})
    expected = replay(adapter.make(), result=result)

    assert expected[FIT].artifacts[PROVENANCE_FILE] == '{"bundle": "b"}'
    assert gaps(expected, adapter.observe(), adapter) == []


def test_adapter_records_a_single_trial_fit_epoch_by_epoch_without_trials(
        adapter):
    plan = {**CONTEXT.model_plan,
            'training_job': {'tune_metric': 'mae', 'num_experiments': 1}}
    events = [EVENTS[0], *_epochs(1, (0.5, 0.4), (0.3, 0.35)), EVENTS[3],
              EVENTS[-1]]
    expected = replay(adapter.make(), events,
                      context=replace(CONTEXT, model_plan=plan), series={
                          'train_loss': ((1, 0.5), (2, 0.3)),
                          'val_mae': ((1, 0.4), (2, 0.35))})

    assert list(expected) == [FIT]
    assert gaps(expected, adapter.observe(), adapter) == []


SUMMARIES = (
    Scalar('test/brier', 0.25),
    Table('graph/col_stats', ('table', 'rows'), (('ents', 400),)),
    Dataset('ents', INPUTS['ents'], {'ID': {'stype': 'id', 'dtype': 'int64'}},
            {'rows': 400}, 'source table'),
)


def test_adapter_logs_each_summary_type(adapter, mlflow_adapter):
    replay(adapter.make(), summaries=SUMMARIES)

    fit = adapter.observe()[FIT]
    assert fit.steps['test/brier'] == ((0, 0.25),)
    tables = {'col_stats.json': {'columns': ['table', 'rows'],
                                 'data': [['ents', 400]]}}
    if adapter is not mlflow_adapter:
        assert {name: json.loads(fit.artifacts[f'graph/{name}'])
                for name in tables} == tables
        return
    import mlflow

    client = mlflow.MlflowClient()
    run, = (r for r in client.search_runs(
        [e.experiment_id for e in client.search_experiments()])
        if r.data.tags['kumo.kind'] == 'fit')
    assert {name: json.loads(mlflow.artifacts.load_text(
        f'{run.info.artifact_uri}/{section}/{name}'))
        for section, name in (('graph', 'col_stats.json'),)} == tables
    assert json.loads(run.data.tags['mlflow.loggedArtifacts']) == [
        {'path': 'graph/col_stats.json', 'type': 'table'}]
    dataset, = run.inputs.dataset_inputs
    assert (dataset.dataset.name, dataset.dataset.digest,
            dataset.tags[0].value) == ('ents', hashlib.sha256(json.dumps(
                ['ents', INPUTS['ents'], SUMMARIES[-1].schema],
                sort_keys=True).encode()).hexdigest()[:8], 'source table')
    assert json.loads(dataset.dataset.schema) == {
        'mlflow_colspec': [{'name': 'ID', 'type': 'long'}]}
    assert json.loads(dataset.dataset.profile) == {'rows': 400}


def test_a_raising_tracker_is_ended_and_not_called_again(caplog):
    class RaisesOnTrial(_Recorder):
        def on_trial(self, record, event):
            super().on_trial(record, event)
            raise RuntimeError('tracker down')

    broken = RaisesOnTrial()
    replay(broken)

    assert [hook for hook, _ in broken.calls] == [
        'start', 'on_trial', 'finish'
    ]
    assert broken.calls[-1] == ('finish', None)
    assert caplog.text.count('off for the rest of this run') == 1


def test_adapter_records_a_failed_fit(adapter):
    failed = _event('fit_ended', status='failed', error='ValueError: stop')
    expected = replay(adapter.make(), [*EVENTS[:-1], failed], None,
                      ValueError('stop'))

    assert expected[FIT].tags['kumo.error'] == 'ValueError: stop'
    assert gaps(expected, adapter.observe(), adapter) == []


def test_only_rank_zero_tracks(monkeypatch):
    monkeypatch.setenv('LOCAL_RANK', '1')
    recorder = _Recorder()
    session = TrackingSession(recorder)

    session.start(lambda: fit_spec(CONTEXT))
    session.finish(lambda: fitted_spec(session.spec, RESULT), None)

    assert recorder.calls == []


def test_a_hung_tracker_does_not_stall_the_fit(monkeypatch):
    import kumo.engine._tracking as tracking

    monkeypatch.setattr(tracking, 'FINISH_SECONDS', 0.5)
    release = threading.Event()

    class Hung(_Recorder):
        def start(self, record):
            super().start(record)
            release.wait(30)

    session = TrackingSession(Hung())
    began = time.monotonic()
    try:
        session.start(lambda: fit_spec(CONTEXT))
        session.finish(None, None)
        elapsed = time.monotonic() - began
    finally:
        release.set()
        join_tracking_threads()

    assert elapsed < 1.5
    assert isinstance(session.error, TimeoutError)
