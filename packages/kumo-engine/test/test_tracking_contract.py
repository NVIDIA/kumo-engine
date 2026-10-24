# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every tracking adapter records the same run records as a recorded fit
and a recorded prediction run."""

import json
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pytest
from kumo_core.artifacts import PROVENANCE_FILE

from kumo.engine._manifest import (
    ArtifactStats,
    PredictionManifest,
    PredictionParents,
    stub,
)
from kumo.engine._tracking import (
    DROPPED_METRICS,
    DROPPED_PARAMS,
    FitContext,
    TrackingSession,
    prediction_record,
    run_record,
    track_prediction,
)
from kumo.engine.tracking import Event, RunRecord

RUN = 'trainer-fit-1'
USER_TAGS = {'team': 'ents', 'kumo.run_id': 'overridden'}
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
CONTEXT = FitContext(
    run_id=RUN, query=QUERY, inputs=INPUTS,
    model_plan={
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
BUNDLE = 'b' * 64
RESULT = SimpleNamespace(metrics={'val_mae': 0.3, 'test_mae': 0.32},
                         id=BUNDLE, bundle_dir=Path('no-bundle'))


PREDICTION_RUN = 'trainer-predict-1'
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
        'predictions': {
            'rows': 50,
            'columns': {
                'True_PROB': {
                    'count': 50,
                    'percentiles': [q / 100 for q in range(1, 100)],
                }
            },
            'predicted_positive_rate': 0.62,
        },
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

    def finish(self, record, error):
        self.calls.append(('finish', record))
        if self.tracker is not None:
            self.tracker.finish(record, error)


def join_tracking_threads():
    for thread in threading.enumerate():
        if thread.name == 'kumo-tracking':
            thread.join(30)


def replay(tracker=None, events=EVENTS, result=RESULT,
           error=None) -> Dict[str, RunRecord]:
    r"""Feeds the recorded fit to `tracker`; returns the records expected
    of it."""
    recorder = _Recorder(tracker)
    session = TrackingSession(recorder)
    session.start(lambda: CONTEXT)
    for event in events:
        session.on_event(event)
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
            differ('parent', seen.parent, record.tags.get('kumo.parent_run'))
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
                prefix, _, name = key.rpartition('/')
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
                observed[runs['tags']['kumo.run_id']] = Observed(
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

    fit = records[RUN]
    assert fit.tags == {'team': 'ents', 'kumo.kind': 'fit', 'kumo.run_id': RUN,
                        'kumo.bundle_id': BUNDLE}
    assert fit.params == {
        'optimization.base_lr': '[0.01, 0.001]',
        'optimization.max_epochs': '2',
        'training_job.tune_metric': 'mae',
        'column_processing.encoder_overrides.ents.amount ($)': 'numerical',
    }
    with pytest.warns(UserWarning,
                      match="'kumo.bundle_id', 'training_job.tune_metric'"):
        spoofed = run_record(CONTEXT, tags={'kumo.bundle_id': 'x',
                                            'training_job.tune_metric': 'y',
                                            'a': 1})
    assert spoofed.tags.keys() - fit.tags.keys() == {'a'}
    assert fit.metrics == {'val_mae': 0.3, 'test_mae': 0.32, 'seconds': 9.0}
    assert fit.artifacts['query.pql'] == QUERY
    assert json.loads(fit.artifacts['model_plan.json']) == CONTEXT.model_plan
    assert fit.inputs == INPUTS

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


def test_adapter_records_the_same_runs(adapter, tmp_path):
    (tmp_path / PROVENANCE_FILE).write_text('{"bundle": "b"}')
    result = SimpleNamespace(metrics=RESULT.metrics, id=BUNDLE,
                             bundle_dir=tmp_path)
    expected = replay(adapter.make(), result=result)

    assert expected[RUN].artifacts[PROVENANCE_FILE] == '{"bundle": "b"}'
    assert gaps(expected, adapter.observe(), adapter) == []


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

    assert expected[RUN].tags['kumo.error'] == 'ValueError: stop'
    assert gaps(expected, adapter.observe(), adapter) == []


def test_only_rank_zero_tracks(monkeypatch):
    monkeypatch.setenv('LOCAL_RANK', '1')
    recorder = _Recorder()
    session = TrackingSession(recorder)

    session.start(lambda: CONTEXT)
    session.finish(RESULT, None)

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


def test_prediction_record_of_the_recorded_run():
    record = prediction_record(PREDICTION_MANIFEST)

    assert record.name == PREDICTION_RUN
    assert record.params == {
        'anchors': '["2024-04-29T00:00:00+00:00"]',
        'binary_classification_threshold': '0.4',
        'threshold_source': 'argument',
        'count_source': 'none',
    }
    assert record.metrics == {
        'prediction_table.rows': 50.0,
        'predictions.rows': 50.0,
        'predictions.True_PROB.count': 50.0,
        **{
            f'predictions.True_PROB.q{q:02d}': q / 100
            for q in (1, 5, 25, 50, 75, 95, 99)
        },
        'predictions.predicted_positive_rate': 0.62,
    }
    assert record.tags == {'kumo.kind': 'prediction',
                           'kumo.run_id': PREDICTION_RUN,
                           'kumo.bundle_id': BUNDLE}
    assert record.inputs == INPUTS
    assert json.loads(record.artifacts[PROVENANCE_FILE]) == (
        PREDICTION_MANIFEST.to_dict())


def test_prediction_record_of_a_partial_manifest():
    record = prediction_record(
        replace(PREDICTION_MANIFEST,
                parents=PredictionParents(model=stub('model', BUNDLE),
                                          snapshot=stub('snapshot', 'g@d')),
                scoring=replace(PREDICTION_MANIFEST.scoring,
                                num_classes_to_return=3),
                stats={'predictions': ArtifactStats(rows=150, columns={
                    'top_score': {'count': 150, 'percentiles': []}}),
                       'embeddings': ArtifactStats(rows=150)}),
        RuntimeError('disk full'))

    assert record.params['num_classes_to_return'] == '3'
    assert record.metrics == {'prediction_table.rows': 50.0,
                              'predictions.rows': 150.0,
                              'predictions.top_score.count': 150.0,
                              'embeddings.rows': 150.0}
    assert record.inputs == {}
    assert record.tags['kumo.error'] == 'RuntimeError: disk full'


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
