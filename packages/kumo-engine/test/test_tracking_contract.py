# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every tracking adapter records the same run records as a recorded fit
and a recorded prediction run."""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pytest
from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.prediction_counts import count_predictions
from kumo_core.score_counts import PROBABILITY_EDGES, Counts
from kumo_core.summaries import (
    Confusion,
    Curve,
    Dataset,
    Histogram,
    Note,
    Scalar,
    Series,
    Table,
)
from kumo_core.types import TaskType
from kumoapi.plugin import PublishedModel

from kumo.engine._manifest import (
    ArtifactStats,
    PredictionManifest,
    PredictionParents,
    TrackingRef,
    stub,
)
from kumo.engine._tracking import (
    DROPPED_METRICS,
    DROPPED_PARAMS,
    TrackingSession,
    _Search,
    prediction_record,
    run_record,
    track_prediction,
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
# The fit run's own series: once the fit succeeds, the selected trial 2's
# epochs.
SELECTED = {'train_loss': ((1, 0.6), (2, 0.4)), 'val_mae': ((1, 0.5), (2, 0.3))}
BUNDLE = 'b' * 64
RESULT = SimpleNamespace(metrics={'val_mae': 0.3, 'test_mae': 0.32},
                         id=BUNDLE, bundle_dir=Path('no-bundle'),
                         registered_model=None, registered_alias=None)


PREDICTION_RUN = 'trainer-predict-1'
SCORES = np.array([0.2] * 19 + [0.6] * 31)


def _counted(scores):
    counts = Counts({'score': PROBABILITY_EDGES})
    count_predictions(counts, TaskType.BINARY_CLASSIFICATION,
                      prob=np.stack([1 - scores, scores], axis=1),
                      pred=scores > 0.4, target_pred=None,
                      columns=['False_PROB', 'True_PROB'])
    return counts.to_dict()


PREDICTION_MANIFEST = PredictionManifest.from_dict({
    'kind': 'prediction',
    'schema_version': 1,
    'id': PREDICTION_RUN,
    'parents': {
        'model': {'kind': 'model', 'schema_version': 1, 'id': BUNDLE,
                  'reconstructed': True},
        'snapshot': {
            'kind': 'snapshot', 'schema_version': 1, 'id': 'g@d',
            'parents': {'source_tables': INPUTS},
        },
    },
    'created_at': '2026-10-21T00:00:00+00:00',
    'environment': None,
    'prediction_table': {
        'query_sha256': '69dab700',
        'source': 'pql',
        'anchors': ['2024-04-29T00:00:00+00:00'],
        'rows': 50,
    },
    'scoring': {
        'binary_classification_threshold': 0.4,
        'threshold_source': 'argument',
        'num_classes_to_return': None,
        'count_source': 'none',
        'num_workers': None,
    },
    'outputs': [],
    'status': 'succeeded',
    'timings': {'started_at': '2026-10-21T00:00:00+00:00',
                'finished_at': '2026-10-21T00:01:00+00:00'},
    'stats': {
        'predictions': {'rows': 50},
    },
})


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
            session.spec, result, summaries=summaries), error)
    records = {
        record.name: record
        for hook, record in recorder.calls if hook == 'on_trial'
    }
    spec = fit_spec(context, tags=USER_TAGS)
    if result is not None:
        spec = fitted_spec(spec, result)
    spec = replace(spec, steps=series if series is not None else {
        **({} if error else SELECTED)})
    if not error:
        spec = replace(spec, parents={
            **spec.parents, **({} if series is not None else {'best_trial': 2}),
            'best_epoch': 2})
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
        metrics = {**record.metrics, **{
            summary.name: summary.value for summary in record.summaries
            if isinstance(summary, Scalar)}}
        differ('metrics', {k: seen.metrics.get(k) for k in metrics}, metrics)
        names = set(metrics)
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
    r"""An MLflow trial run's record name, ``trial-<i>-attempt-<j>``."""
    trial = run.info.run_name.split('-')[1]
    return f"trial-{trial}-attempt-{run.data.params['attempt']}"


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
            FIT if r.data.tags['kumo.kind'] == 'fit' else _trial(r)
            if r.data.tags['kumo.kind'] == 'trial' else
            r.data.tags['kumo.prediction_id']
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
                if not prefix.startswith('TRIAL_'):
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
            # Snowflake's own run duration and card are checked in
            # test_snowflake_tracking.
            for key, value, step in run['metrics']:
                if key == 'duration_seconds':
                    continue
                runs, name = bucket(key)
                runs['metrics'][name] = value
                runs['steps'][name] = (*runs['steps'].get(name, ()),
                                       (step, value))
            for prefix, runs in names.items():
                key = prefix or runs['tags'].get('kumo.prediction_id', FIT)
                observed[key] = Observed(
                    name=prefix or run_name,
                    artifacts={} if prefix else {
                        path: text for path, text in run['artifacts'].items()
                        if path != 'run_card.md'},
                    **runs)
        return observed

    def expected(records):
        r"""What Snowflake keeps of a trial: its params, status, error and
        best tuned validation metric, under ``TRIAL_<i>`` (a retry under
        ``TRIAL_<i>_ATTEMPT_<j>``)."""
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
        # Snowflake names are checked in test_snowflake_tracking.
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
                        'kumo.tune_metric': 'mae',
                        'kumo.engine_version': __version__,
                        'kumo.best_trial': '2', 'kumo.best_epoch': '2'}
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
    assert fit.artifacts == {}
    assert fit.description == f'`{QUERY}`'
    assert fit.inputs == INPUTS

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
    tables = {'col_stats.json': {'columns': ['table', 'rows'],
                                 'data': [['ents', 400]]},
              'confusion.json': {'columns': ['actual', 'a', 'b'],
                                 'data': [['a', 5, 1], ['b', 2, 7]]}}
    if adapter is not mlflow_adapter:
        assert {name for name in fit.steps if not name.startswith((
            'train/', 'val/', 'search/best'))} == {'test/mae', 'test/brier'}
        assert {path: json.loads(text)
                for path, text in fit.artifacts.items()} == {
            'search.json': {'columns': ['trial', 'trials'],
                            'data': [[1, 3.0], [2, 4.0]]},
            'test/roc.json': {'columns': ['step', 'fpr', 'tpr'],
                              'data': [[0, 1.0, 1.0], [1, 0.5, 0.75]]},
            'pred/True_PROB.json': {
                'columns': ['from', 'to', 'hist', 'hist cumulative share'],
                'data': [[None, 0.5, 3, 0.428571], [0.5, None, 4, 1.0]]},
            'graph/col_stats.json': tables['col_stats.json'],
            'test/confusion.json': tables['confusion.json']}
        return
    import mlflow

    assert {name for name in fit.steps if not name.startswith((
        'train/', 'val/', 'search/'))} == {'test/mae', 'test/brier'}
    client = mlflow.MlflowClient()
    run, = (r for r in client.search_runs(
        [e.experiment_id for e in client.search_experiments()])
        if r.data.tags['kumo.kind'] == 'fit')
    assert {name: json.loads(mlflow.artifacts.load_text(
        f'{run.info.artifact_uri}/{section}/{name}'))
        for section, name in (('graph', 'col_stats.json'),
                              ('test', 'confusion.json'))} == tables
    assert json.loads(run.data.tags['mlflow.loggedArtifacts']) == [
        {'path': f'{name}.json', 'type': 'table'} for name in (
            'search', 'test/roc', 'pred/True_PROB', 'graph/col_stats',
            'test/confusion')]
    dataset, = run.inputs.dataset_inputs
    assert (dataset.dataset.name, dataset.dataset.digest,
            dataset.tags[0].value) == ('ents', hashlib.sha256(json.dumps(
                ['ents', INPUTS['ents'], SUMMARIES[-1].schema],
                sort_keys=True).encode()).hexdigest()[:8], 'training')
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


def test_prediction_record_of_the_recorded_run():
    manifest = replace(PREDICTION_MANIFEST, stats={
        'predictions': ArtifactStats(rows=50, counts=_counted(SCORES))})
    record = prediction_record(manifest)

    assert (record.name, record.model_label) == (
        'predict-model-bbbbbbbb-2024-04-29', 'model')
    assert (record.started, record.finished) == (
        datetime(2026, 10, 21, tzinfo=timezone.utc),
        datetime(2026, 10, 21, 0, 1, tzinfo=timezone.utc))
    assert record.params == {
        'anchors': '["2024-04-29T00:00:00+00:00"]',
        'binary_classification_threshold': '0.4',
        'threshold_source': 'argument',
    }
    assert record.metrics == {}
    assert record.summaries == (
        Scalar('pred/rows', 50),
        Histogram('pred/histogram/True_PROB', PROBABILITY_EDGES, tuple(
            np.bincount(np.searchsorted(PROBABILITY_EDGES, SCORES, 'right'),
                        minlength=102).tolist()), (min(SCORES), max(SCORES))),
        Series('pred/True_PROB_count_by_score_pct', tuple(enumerate(
            np.bincount((SCORES * 100).astype(int), minlength=100).astype(
                float).tolist())), 'score_pct', chart=True),
        Table('pred/summary', ('column', 'mean', 'min', 'max'),
              (('True_PROB', SCORES.sum() / 50, 0.2, 0.6),), brief=True),
        Scalar('pred/positive_rate', 0.62))
    assert record.tags == {'kumo.kind': 'prediction',
                           'kumo.engine_version': __version__,
                           'kumo.prediction_id': PREDICTION_RUN,
                           'kumo.as_of': '2024-04-29',
                           'kumo.bundle_id': BUNDLE,
                           'kumo.snapshot_id': 'g@d'}
    assert record.description is None
    assert record.inputs == INPUTS
    assert json.loads(record.artifacts[PROVENANCE_FILE]) == manifest.to_dict()


@pytest.mark.parametrize('anchors,snapshot,as_of,name', [
    (['2026-10-28T23:00:00-02:00', '2026-10-28T00:00:00'], {},
     '2026-10-28..2026-10-29', 'predict-model-bbbbbbbb-2026-10-28..2026-10-29'),
    ([], {'created_at': '2026-10-27T10:00:00+00:00'}, '2026-10-27',
     'predict-model-bbbbbbbb-2026-10-27-000000'),
    ([], {}, '2026-10-21', 'predict-model-bbbbbbbb-2026-10-21-000000'),
])
def test_a_prediction_is_as_of_its_anchors_else_its_snapshot(
        anchors, snapshot, as_of, name):
    manifest = PredictionManifest.from_dict({
        **PREDICTION_MANIFEST.to_dict(),
        'prediction_table': {**PREDICTION_MANIFEST.prediction_table.to_dict(),
                             'anchors': anchors},
        'parents': {**PREDICTION_MANIFEST.to_dict()['parents'], 'snapshot': {
            'kind': 'snapshot', 'schema_version': 1, 'id': 'g@d',
            **snapshot}}})

    record = prediction_record(manifest)
    assert (record.tags['kumo.as_of'], record.name) == (as_of, name)


@pytest.mark.parametrize('kind,core,suffix,name', [
    ('fit', 'Fit-nightly', None, 'Fit-nightly'),
    ('fit', 'fitness', None, 'fit-fitness'),
    ('prediction', 'PREDICT_nightly', '-2024-04-29',
     'PREDICT_nightly-2024-04-29'),
])
def test_a_run_name_starts_with_its_kind_once(kind, core, suffix, name):
    from kumo.engine._tracking import run_name

    assert run_name(kind, core, suffix) == name


def test_prediction_record_of_a_partial_manifest():
    record = prediction_record(
        replace(PREDICTION_MANIFEST,
                parents=PredictionParents(model=stub('model', BUNDLE),
                                          snapshot=stub('snapshot', 'g@d')),
                prediction_table=replace(PREDICTION_MANIFEST.prediction_table,
                                         requested_rows=55, skipped_rows=5,
                                         skip_reason='no history'),
                scoring=replace(PREDICTION_MANIFEST.scoring,
                                num_classes_to_return=3),
                stats={'predictions': ArtifactStats(rows=150),
                       'embeddings': ArtifactStats(rows=150)}),
        RuntimeError('disk full'))

    assert record.params['num_classes_to_return'] == '3'
    assert record.summaries == (
        Scalar('pred/rows', 50), Scalar('pred/skipped_share', 5 / 55),
        Note('notes', '5 requested rows were not scored: no history.'))
    assert record.inputs == {}
    assert record.tags['kumo.error'] == 'RuntimeError: disk full'


def test_prediction_run_is_named_and_tagged_after_its_registered_model():
    model = SimpleNamespace(
        registered_model=PublishedModel('churn', '3', 'models:/churn/3'),
        registered_alias='champion',
        manifest=SimpleNamespace(holdout=None, tracking=TrackingRef(
            platform='mlflow', run_id='r1', url='https://mlflow/#/runs/r1')))
    record = prediction_record(PREDICTION_MANIFEST, model=model)

    assert record.name == 'predict-churn-v3-2024-04-29'
    assert record.tags.items() >= {
        'kumo.model_name': 'churn', 'kumo.model_version': '3',
        'kumo.fit_run_id': 'r1'}.items()


def test_adapter_records_the_same_prediction_run(adapter):
    track_prediction(PREDICTION_MANIFEST, adapter.make())

    expected = {PREDICTION_RUN: prediction_record(PREDICTION_MANIFEST)}
    assert gaps(expected, adapter.observe(), adapter) == []


def test_prediction_run_without_a_manifest_or_hook_is_skipped(caplog):
    recorder = _Recorder()
    recorder.log_prediction_run = recorder.calls.append
    track_prediction(None, recorder)
    track_prediction(PREDICTION_MANIFEST, _Recorder())

    assert recorder.calls == []
    assert 'not tracked' in caplog.text


def test_every_summary_type_has_a_sink_in_each_tracker():
    from typing import get_args

    from kumo_core.summaries import Summary

    from kumo.engine.integrations import mlflow, snowflake

    for sinks in (mlflow._SINKS, snowflake._SINKS):
        assert set(sinks) == set(get_args(Summary))


@pytest.mark.parametrize('metrics,refit_full,expected', [
    ({'test_auroc': 0.7, 'test_loss': 0.5, 'val_auroc': 0.6},
     False, {'test/auroc': 0.7, 'test/loss': 0.5, 'val/auroc': 0.6}),
    ({'test_auroc': 0.7, 'test_refit_auroc': 0.8, 'test_refit_loss': 0.4},
     False, {'before_refit/auroc': 0.7, 'test/auroc': 0.8, 'test/loss': 0.4}),
    ({'test_auroc': 0.7}, True, {'before_refit/auroc': 0.7}),
    ({'test_auroc': 0.7, 'test_refit_auroc': 0.8}, True,
     {'before_refit/auroc': 0.8}),
])
def test_holdout_metrics_are_named_after_the_shipped_model(
        metrics, refit_full, expected):
    from kumo.engine._tracking import logged_names

    assert logged_names(metrics, refit_full) == expected


def test_holdout_summaries_are_named_after_the_shipped_model():
    from kumo_core.summaries import Note

    from kumo.engine.trainer.trainer import _REFIT_FULL, _unlogged

    summaries = (Scalar('auroc', 0.7, role='model'),
                 Scalar('brier', 0.2, role='model'), Scalar('label/rows', 9.0))
    assert _unlogged(summaries, {'test_auroc': 0.7}) == (
        Scalar('test/brier', 0.2, role='model'), Scalar('label/rows', 9.0))
    assert _unlogged(summaries, {'test_auroc': 0.7}, refit_full=True) == (
        Scalar('before_refit/brier', 0.2, role='model'),
        Scalar('label/rows', 9.0), Note('notes', _REFIT_FULL))


def test_a_prediction_names_its_entity_and_when_its_labels_mature():
    from kumo.engine._tracking import _labels_mature

    trained = SimpleNamespace(query=QUERY, task_type='regression')
    model = SimpleNamespace(registered_model=None, registered_alias=None,
                            manifest=SimpleNamespace(
                                parents=SimpleNamespace(training_table=trained),
                                holdout=SimpleNamespace(window=[0, 7, 7, 'days'],
                                                        counts=None),
                                tracking=None))

    tags = prediction_record(PREDICTION_MANIFEST, model=model).tags
    assert (tags['kumo.entity'], tags['kumo.labels_mature']) == (
        'ents', '2024-05-06')
    assert _labels_mature(PREDICTION_MANIFEST, None) is None


def test_a_fit_logs_only_the_plan_options_it_sets():
    spec = fit_spec(replace(CONTEXT, model_plan={'optimization': {
        'base_lr': 0.01, 'max_epochs': '???', 'loss': None}}))

    assert spec.params == {'optimization.base_lr': 0.01}


class _Events(_Recorder):
    def on_trial(self, record, event):
        self.calls.append(('on_trial', event))


def _session(events, trials):
    recorder = _Events()
    session = TrackingSession(recorder)
    plan = {**CONTEXT.model_plan, 'training_job': {
        'tune_metric': 'mae', 'num_experiments': trials}}
    session.start(lambda: fit_spec(replace(CONTEXT, model_plan=plan)))
    for event in events:
        session.on_event(event)
    session.finish(lambda: fitted_spec(session.spec, RESULT), None)
    join_tracking_threads()
    return recorder.calls


def test_a_one_epoch_fit_writes_its_epoch_in_a_note_not_a_series():
    calls = _session([
        _event('trial_started', 1, attempt=1), *_epochs(1, (0.5, 0.4)),
        _event('trial_ended', 1, status='succeeded', attempt=1),
        _event('fit_ended', status='succeeded')], trials=1)

    assert [hook for hook, _ in calls] == ['start', 'finish']
    record = calls[-1][1]
    assert record.steps == {}
    assert Note('notes', 'Training (1 epoch): train loss 0.5 · val mae 0.4') \
        in record.summaries


def test_the_kept_epoch_is_the_first_with_the_best_validation_metric():
    search = _Search.of({'training_job.tune_metric': 'mae'})
    events = _epochs(1, (0.5, 0.4), (0.4, 0.3), (0.3, 0.3), (0.2, 0.5))

    assert search.best_epoch(events) == 2


def test_a_series_starts_once_it_has_two_points():
    calls = _session([
        _event('trial_started', 1, attempt=1), *_epochs(1, (0.5, 0.4)),
        _event('trial_ended', 1, metrics={'val_mae': 0.4}, status='succeeded',
               attempt=1),
        _event('trial_started', 2, attempt=1),
        *_epochs(2, (0.6, 0.5), (0.4, 0.3)),
        _event('trial_ended', 2, metrics={'val_mae': 0.3}, status='succeeded',
               attempt=1),
        _event('fit_ended', status='succeeded')], trials=2)

    epochs = [(event.trial, event.epoch) for hook, event in calls
              if hook == 'on_trial' and event.name == 'epoch_ended']
    assert epochs == [(2, 1), (2, 2)]
    assert [event.metrics for hook, event in calls if hook == 'on_trial'
            and event.trial == 1 and event.name == 'trial_ended'] == [
        {'val/mae': 0.4}]
    assert calls[-1][1].steps == {'train/loss': ((1, 0.6), (2, 0.4)),
                                  'val/mae': ((1, 0.5), (2, 0.3))}


def test_a_search_whose_selected_trial_ran_one_epoch_notes_it():
    calls = _session([
        _event('trial_started', 1, attempt=1), *_epochs(1, (0.5, 0.4)),
        _event('trial_ended', 1, metrics={'val_mae': 0.4}, status='succeeded',
               attempt=1),
        _event('fit_ended', status='succeeded')], trials=2)

    record = calls[-1][1]
    assert record.steps == {}
    assert Note('notes', 'Training (1 epoch): train loss 0.5 · val mae 0.4') \
        in record.summaries
