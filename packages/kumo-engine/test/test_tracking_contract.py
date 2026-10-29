# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every tracking adapter records the same run records as a recorded fit."""

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
from kumo_core.summaries import (
    Confusion,
    Curve,
    Dataset,
    Histogram,
    Scalar,
    Series,
    Table,
)

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
    query=QUERY, inputs=INPUTS, snapshot_id=SNAPSHOT, task_type='regression',
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
# The fit run's own series: the best val_mae after each trial that
# succeeded and, once the fit succeeds, the selected trial 2's epochs.
SEARCH = {'search/best_val_mae': ((1, 0.35), (2, 0.3))}
SELECTED = {'train_loss': ((1, 0.6), (2, 0.4)), 'val_mae': ((1, 0.5), (2, 0.3))}
BUNDLE = 'b' * 64
RESULT = SimpleNamespace(metrics={'val_mae': 0.3, 'test_mae': 0.32},
                         id=BUNDLE, bundle_dir=Path('no-bundle'),
                         registered_model=None)


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
    r"""Feeds the recorded fit to `tracker`; returns the records expected
    of it, with the fit run's ``series`` (default: the search's) and
    ``summaries``."""
    recorder = _Recorder(tracker)
    session = TrackingSession(recorder)
    session.start(lambda: fit_spec(context))
    for event in events:
        session.on_event(event)
    session.finish(
        None if result is None else lambda: fitted_spec(
            session.spec, result, events[-1].metrics.get('seconds'),
            summaries=summaries), error)
    records = {
        record.name: record
        for hook, record in recorder.calls if hook == 'on_trial'
    }
    spec = fit_spec(context, tags=USER_TAGS)
    if result is not None:
        spec = fitted_spec(spec, result, events[-1].metrics.get('seconds'))
    spec = replace(spec, steps=series if series is not None else {
        **SEARCH, **({} if error else SELECTED)})
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


def gaps(expected: Dict[str, RunRecord], observed: Dict[str, Observed],
         adapter: Adapter) -> List[str]:
    kept_params = adapter.kept_params
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
        kumo_ids = {
            r.info.run_id:
            FIT if r.data.tags['kumo.kind'] == 'fit' else r.info.run_name
            for r in runs
        }

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
                    if not a.is_dir
                })
            for run in runs
        }

    try:
        yield Adapter(
            make=lambda: MlflowTracking(log_system_metrics=False,
                                        tags=USER_TAGS), observe=observe,
            kept_params=_mlflow_params, nested_runs=True, step_metrics=True,
            artifacts=True)
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)


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
                if not prefix.startswith('trial-'):
                    prefix, name = '', key
                return names.setdefault(prefix or None, {
                    'params': {}, 'tags': {}, 'metrics': {}, 'steps': {}
                }), name

            for key, value in run['params'].items():
                runs, name = bucket(key)
                # Snowflake has no tags: they are params of the same name.
                if name.startswith('kumo.') or name in USER_TAGS:
                    runs['tags'][name] = value
                else:
                    runs['params'][name] = value
            for key, value, step in run['metrics']:
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

    return Adapter(
        make=lambda: SnowflakeTracking(_snowflake_experiment_fake.FakeSession(),
                                       experiment_name='ents', tags=USER_TAGS),
        observe=observe, step_metrics=True, artifacts=True)


@pytest.fixture(params=['mlflow', 'snowflake'])
def adapter(request) -> Adapter:
    return request.getfixturevalue(f'{request.param}_adapter')


def test_records_of_the_recorded_fit():
    records = replay()

    fit = records[FIT]
    assert fit.name == 'fit-20261029-0618'
    assert fit.tags == {'team': 'ents', 'kumo.kind': 'fit',
                        'kumo.bundle_id': BUNDLE, 'kumo.snapshot_id': SNAPSHOT,
                        'kumo.task_type': 'regression',
                        'kumo.tune_metric': 'mae',
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
        spoofed = run_record(fit_spec(CONTEXT, tags={
            'kumo.bundle_id': 'x', 'training_job.tune_metric': 'y', 'a': 1}))
    assert spoofed.tags.keys() - fit.tags.keys() == {'a'}
    assert fit.metrics == {'test/mae': 0.32, 'seconds': 9.0}
    assert fit.steps == {'search/best_val_mae': ((1, 0.35), (2, 0.3)),
                         'train/loss': ((1, 0.6), (2, 0.4)),
                         'val/mae': ((1, 0.5), (2, 0.3))}
    registered = SimpleNamespace(**{**vars(RESULT), 'registered_model':
                                    SimpleNamespace(name='ents', version='3')})
    tags = run_record(fitted_spec(fit_spec(CONTEXT), registered,
                                  alias='candidate')).tags
    assert tags.items() >= {'kumo.model_name': 'ents', 'kumo.model_version':
                            '3', 'kumo.model_alias': 'candidate'}.items()
    assert fit.artifacts['query.pql'] == QUERY
    assert json.loads(fit.artifacts['model_plan.json']) == CONTEXT.model_plan
    assert fit.inputs == INPUTS

    oom = records['trial-2-attempt-1']
    assert oom == RunRecord(
        name='trial-2-attempt-1',
        params={'optimization.base_lr': '0.001', 'attempt': '1'}, tags={
            'kumo.kind': 'trial',
            'kumo.status': 'failed',
            'kumo.error': 'OutOfMemoryError: CUDA out of memory',
        })
    retry = records['trial-2-attempt-2']
    assert retry.params['attempt'] == '2'
    assert retry.metrics == {'val/best_mae': 0.3, 'seconds': 4.0}
    assert retry.steps == {'train/loss': ((1, 0.6), (2, 0.4)),
                           'val/mae': ((1, 0.5), (2, 0.3))}


def test_adapter_records_the_same_runs(adapter, tmp_path):
    (tmp_path / PROVENANCE_FILE).write_text('{"bundle": "b"}')
    result = SimpleNamespace(**{**vars(RESULT), 'bundle_dir': tmp_path})
    expected = replay(adapter.make(), result=result)

    assert expected[FIT].artifacts[PROVENANCE_FILE] == '{"bundle": "b"}'
    assert gaps(expected, adapter.observe(), adapter) == []


def test_adapter_records_a_single_trial_fit_epoch_by_epoch(adapter):
    plan = {**CONTEXT.model_plan,
            'training_job': {'tune_metric': 'mae', 'num_experiments': 1}}
    events = [EVENTS[0], *_epochs(1, (0.5, 0.4), (0.3, 0.35)), EVENTS[3],
              EVENTS[-1]]
    expected = replay(adapter.make(), events,
                      context=replace(CONTEXT, model_plan=plan), series={
                          'train_loss': ((1, 0.5), (2, 0.3)),
                          'val_mae': ((1, 0.4), (2, 0.35))})

    assert gaps(expected, adapter.observe(), adapter) == []


SUMMARIES = (
    Scalar('test/brier', 0.25),
    Series('search/trials', ((1, 3.0), (2, 4.0)), x_label='trial'),
    Curve('test/roc', Series('test/roc/fpr', ((0, 1.0), (1, 0.5))),
          Series('test/roc/tpr', ((0, 1.0), (1, 0.75)))),
    Histogram('pred/True_PROB/hist', (0.5,), (3, 4)),
    Table('graph/col_stats', ('table', 'rows'), (('ents', 400),)),
    Confusion('test/confusion', ('a', 'b'), ((5, 1), (2, 7))),
    Dataset('ents', INPUTS['ents'], {'ID': {'stype': 'id', 'dtype': 'int64'}},
            {'rows': 400}, 'training'),
)


def test_adapter_logs_each_summary_type(adapter, mlflow_adapter):
    replay(adapter.make(), summaries=SUMMARIES)

    fit = adapter.observe()[FIT]
    assert {name: fit.steps[name] for name in (
        'test/brier', 'search/trials', 'test/roc/fpr', 'test/roc/tpr',
        'pred/True_PROB/hist/count', 'pred/True_PROB/hist/lower_edge')} == {
        'test/brier': ((0, 0.25),), 'search/trials': ((1, 3.0), (2, 4.0)),
        'test/roc/fpr': ((0, 1.0), (1, 0.5)),
        'test/roc/tpr': ((0, 1.0), (1, 0.75)),
        'pred/True_PROB/hist/count': ((0, 3.0), (1, 4.0)),
        'pred/True_PROB/hist/lower_edge': ((1, 0.5),)}
    tables = {'col_stats.json': {'columns': ['table', 'rows'],
                                 'data': [['ents', 400]]},
              'confusion.json': {'columns': ['actual', 'a', 'b'],
                                 'data': [['a', 5, 1], ['b', 2, 7]]}}
    if adapter is not mlflow_adapter:
        assert {name: json.loads(fit.artifacts[name])
                for name in tables} == tables
        return
    import mlflow

    client = mlflow.MlflowClient()
    run, = (r for r in client.search_runs(
        [e.experiment_id for e in client.search_experiments()])
        if r.data.tags['kumo.kind'] == 'fit')
    assert {name: json.loads(mlflow.artifacts.load_text(
        f'{run.info.artifact_uri}/{section}/{name}'))
        for section, name in (('graph', 'col_stats.json'),
                              ('test', 'confusion.json'))} == tables
    assert json.loads(run.data.tags['mlflow.loggedArtifacts']) == [
        {'path': 'graph/col_stats.json', 'type': 'table'},
        {'path': 'test/confusion.json', 'type': 'table'}]
    dataset, = run.inputs.dataset_inputs
    assert (dataset.dataset.name, dataset.dataset.digest,
            dataset.tags[0].value) == ('ents', '1790000000', 'training')
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


def test_a_hung_tracker_does_not_stall_the_fit(monkeypatch, caplog):
    import kumo.engine._tracking as tracking

    monkeypatch.setattr(tracking, 'FINISH_SECONDS', 0.5)
    release = threading.Event()

    class Hung(_Recorder):
        def start(self, record):
            super().start(record)
            release.wait(30)

    hung = Hung()
    began = time.monotonic()
    try:
        replay(hung)
        elapsed = time.monotonic() - began
    finally:
        release.set()
        join_tracking_threads()

    assert elapsed < 1.5
    assert 'did not finish within' in caplog.text
