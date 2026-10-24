# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest
from kumo_core.artifacts import PROVENANCE_FILE
from kumoapi.model_plan import ModelPlan
from kumoapi.plugin import PublishedModel
from test_mlflow_tracking import (  # noqa: F401
    _runs,
    table_and_snapshot,
    tracking_uri,
)

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

mlflow = pytest.importorskip('mlflow')


def _plan():
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 2
    plan.optimization.max_steps_per_epoch = 2
    return plan


def _sorted(frame):
    return frame.sort_values('ENTITY').reset_index(drop=True)


def test_fit_logs_registers_and_serves_the_model(table_and_snapshot,
                                                 tmp_path, caplog,
                                                 monkeypatch):
    from kumo.engine.integrations.mlflow import MlflowTracking, load_model

    table, snapshot = table_and_snapshot
    plan = _plan()
    cfg = kumo.MLOps(str(tmp_path / 'root'),
                     tracking=MlflowTracking(log_system_metrics=False),
                     registry=kumo.ModelRegistryLocation('m',
                                                         alias='champion'))
    trainer = kumo.Trainer(plan)
    result = trainer.fit(table, snapshot=snapshot, mlops=cfg)

    assert 'MLflow tracking failed' not in caplog.text
    parent, = [
        run for run in _runs(mlflow.MlflowClient())
        if 'mlflow.parentRunId' not in run.data.tags
    ]
    model, = mlflow.search_logged_models(output_format='list')
    assert model.source_run_id == parent.info.run_id
    metrics = {metric.key: metric.value for metric in model.metrics}
    assert metrics == {**result.metrics, 'seconds': metrics['seconds']}
    version = mlflow.MlflowClient().get_model_version_by_alias('m', 'champion')
    assert version.source == model.model_uri
    assert {k: v for k, v in parent.data.tags.items()
            if k.startswith('kumo.')} == {
                'kumo.kind': 'fit',
                'kumo.run_id': parent.data.tags['kumo.run_id'],
                'kumo.bundle_id': result.id,
            }
    assert result.registered_model == PublishedModel(
        name='m', version=str(version.version),
        uri=f'models:/m/{version.version}')
    assert trainer.result is result

    info = mlflow.models.get_model_info('models:/m@champion')
    assert info.metadata == {'kumo_bundle_id': result.id}
    provenance = json.loads(mlflow.artifacts.load_text(
        f'{model.artifact_location}/artifacts/{PROVENANCE_FILE}'))
    assert provenance == result.manifest.to_dict()
    assert [(c['name'], c['type']) for c in info.signature.inputs.to_dict()
            ] == [('ENTITY', 'long'), ('TIMESTAMP', 'datetime')]
    assert [(c['name'], c['type']) for c in info.signature.outputs.to_dict()
            ] == [('ENTITY', 'long'), ('TARGET_PRED', 'double')]

    location = snapshot.export(
        kumo.SnapshotOutputConfig(
            kumo.FileConnector(root_dir=str(tmp_path / 'exports')),
            'snapshots'))
    served = mlflow.pyfunc.load_model('models:/m@champion').predict(
        pd.DataFrame({'ENTITY': range(0, 400, 7)}),
        params={'snapshot': str(tmp_path / 'exports' / location)})
    assert list(served) == ['ENTITY', 'TARGET_PRED']
    assert len(served) == len(range(0, 400, 7))
    assert load_model('models:/m@champion').bundle_id == result.id

    original = mlflow.models.Model.load

    def other_bundle(path):
        model = original(path)
        model.metadata['kumo_bundle_id'] = 'other'
        return model

    with pytest.MonkeyPatch.context() as patch, pytest.warns(
            UserWarning, match='records bundle other'):
        patch.setattr(mlflow.models.Model, 'load', other_bundle)
        load_model('models:/m@champion')


def test_logged_model_scores_the_query_population_and_timed_rows(
        fitted, tracking_uri, tmp_path, monkeypatch):
    from kumo.engine.integrations.mlflow import log_model

    trainer, table, snapshot = fitted
    location = snapshot.export(
        kumo.SnapshotOutputConfig(
            kumo.FileConnector(root_dir=str(tmp_path / 'exports')),
            'snapshots'))

    info = log_model(trainer, snapshot=snapshot,
                     serving_snapshot=str(tmp_path / 'exports' / location))

    assert mlflow.get_run(info.run_id).info.status == 'FINISHED'
    requirements = mlflow.pyfunc.get_model_dependencies(info.model_uri)
    assert f'kumo-engine[gcs,s3]=={kumo.__version__}' in open(
        requirements).read().splitlines()
    assert {metric.key
            for metric in mlflow.get_logged_model(info.model_id).metrics
            } == set(trainer.result.metrics)
    assert [c['name'] for c in info.signature.outputs.to_dict()
            ] == ['ENTITY', 'TARGET_PRED', 'False_PROB', 'True_PROB']
    pyfunc = mlflow.pyfunc.load_model(info.model_uri)
    params = {
        'snapshot': str(tmp_path / 'exports' / location),
        'prediction_time': str(pd.Timestamp(table.timestamps[0])),
        'binary_classification_threshold': 0.5,
    }
    expected = _sorted(
        trainer.predict(table, snapshot=snapshot,
                        binary_classification_threshold=0.5).frame)
    empty = pd.DataFrame({'ENTITY': pd.Series([], dtype='int64')})
    tables = get_scratch_dir() / 'prediction_tables'
    before = set(tables.glob('*'))
    predict = kumo.Trainer.predict

    def predict_beside_another_call(self, *args, **kwargs):
        (tables / 'other-call').mkdir(parents=True, exist_ok=True)
        return predict(self, *args, **kwargs)

    monkeypatch.setattr(kumo.Trainer, 'predict', predict_beside_another_call)

    assert pyfunc.predict(empty, params=params).empty
    served = pyfunc.unwrap_python_model()._snapshots
    assert list(served) == [params['snapshot']]
    pd.testing.assert_frame_equal(
        _sorted(
            pyfunc.predict(empty, params={
                **params, 'snapshot': None, 'population': True
            })), expected)
    for rows, population in ((empty, True),
                             (pd.DataFrame({'ENTITY': range(50)}), False)):
        pd.testing.assert_frame_equal(
            _sorted(
                pyfunc.predict(rows,
                               params={**params, 'population': population})),
            expected)
    scratch = pyfunc.unwrap_python_model()._trainer.scratch_dir
    assert list(scratch.iterdir()) == []
    assert set(tables.glob('*')) == before | {tables / 'other-call'}


@pytest.mark.parametrize('tracking,registry', [
    (None, kumo.ModelRegistryLocation('m')),
    (SimpleNamespace(start=None), kumo.ModelRegistryLocation('m')),
    ('mlflow', kumo.ModelRegistryLocation('m', alias='SageMakerLifecycle-p')),
    ('mlflow', kumo.ModelRegistryLocation('m', version='3')),
])
def test_invalid_mlflow_registry_is_refused(tracking_uri, tmp_path, tracking,
                                            registry):
    from kumo.engine.integrations.mlflow import MlflowTracking

    if tracking == 'mlflow':
        tracking = MlflowTracking()
    with pytest.raises(ValueError):
        kumo.MLOps(str(tmp_path), tracking=tracking, registry=registry)


def test_fit_waits_for_model_logging_and_fails_the_run_if_it_fails(
        monkeypatch):
    import time

    from kumo.engine import _tracking
    from kumo.engine._tracking import FitContext, TrackingSession

    monkeypatch.setattr(_tracking, 'FINISH_SECONDS', 0.01)

    class Tracker:
        def __init__(self, fail):
            self.fail, self.calls = fail, []

        def start(self, record):
            self.calls.append('start')

        def on_trial(self, record, event):
            pass

        def log_model(self, result, snapshot, registry):
            time.sleep(0.2)
            self.calls.append('log_model')
            if self.fail:
                raise RuntimeError('upload failed')

        def finish(self, record, error):
            self.calls.append(('finish', error))

    result = SimpleNamespace(metrics={}, bundle_dir='missing', bundle_id='b')
    for fail in (False, True):
        tracker = Tracker(fail)
        session = TrackingSession(tracker)
        session.start(lambda: FitContext('r', 'q', {}))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, error = session.log_model(result, None)
        session.finish(result, error)
        session._worker.join(5)
        assert tracker.calls[:2] == ['start', 'log_model']
        assert tracker.calls[2] == ('finish', error)
        assert (error is not None) is fail
        assert any('upload failed' in str(w.message) for w in caught) is fail


@pytest.mark.parametrize('stall', ['slow upload', 'dead thread'])
def test_model_logging_wait_is_bounded(monkeypatch, stall):
    import threading

    from kumo.engine import _tracking
    from kumo.engine._tracking import FitContext, TrackingSession

    monkeypatch.setattr(_tracking, 'MODEL_SECONDS', 0.5)
    release = threading.Event()

    class Tracker:
        calls = []

        def start(self, record):
            pass

        def on_trial(self, record, event):
            pass

        def log_model(self, result, snapshot, registry):
            release.wait(30)

        def finish(self, record, error):
            self.calls.append((record, error))

    tracker = Tracker()
    session = TrackingSession(tracker)
    session.start(lambda: FitContext('r', 'q', {}))
    if stall == 'dead thread':
        session._submit(lambda: (_ for _ in ()).throw(SystemExit()))
        session._worker.join(5)
    with pytest.warns(UserWarning, match='Could not log the model'):
        _, error = session.log_model(SimpleNamespace(), None)
    release.set()
    session.finish(None, error)

    assert type(error) is (TimeoutError
                           if stall == 'slow upload' else RuntimeError)
    assert tracker.calls == [(None, None)]


def test_model_upload_gets_a_longer_http_timeout_and_a_late_one_is_dropped(
        tracking_uri, monkeypatch, caplog):
    import os

    from test_tracking_contract import CONTEXT

    from kumo.engine._tracking import TrackingSession
    from kumo.engine.integrations import mlflow as integration

    for name in integration._HTTP_LIMITS:
        monkeypatch.delenv(name, raising=False)
    tracker = integration.MlflowTracking(log_system_metrics=False)
    location = kumo.ModelRegistryLocation('m')
    seen = []

    def log_model(mlflow, result, snapshot, **kwargs):
        seen.append(os.environ.get('MLFLOW_HTTP_REQUEST_TIMEOUT'))
        if len(seen) == 2:
            tracker.finish(None, None)
        return SimpleNamespace(model_id=None, model_uri='runs:/x/model')

    monkeypatch.setattr(integration, '_log_model', log_model)
    monkeypatch.setattr(integration, '_register', lambda *args: 'registered')
    for _ in range(2):
        session = TrackingSession(tracker)
        session.start(lambda: CONTEXT)
        registered, error = session.log_model(SimpleNamespace(), None,
                                              location)
        assert error is None
        during = os.environ.get('MLFLOW_HTTP_REQUEST_TIMEOUT')
        session.finish(None, None)
        session._worker.join(5)
        assert registered == ('registered' if len(seen) == 1 else None)
        assert during == ('30' if len(seen) == 1 else None)

    assert seen == ['600', '600']
    assert 'MLFLOW_HTTP_REQUEST_TIMEOUT' not in os.environ
    assert 'the model is not registered' in caplog.text
