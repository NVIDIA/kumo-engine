# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest
from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.summaries import Scalar
from kumoapi.model_plan import ModelPlan
from kumoapi.plugin import PublishedModel
from test_mlflow_tracking import (  # noqa: F401
    _runs,
    table_and_snapshot,
    tracking_uri,
)

import kumo.engine as kumo
from kumo.engine import _card
from kumo.engine._scratch import get_scratch_dir
from kumo.engine._tracking import logged_names

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
                     registry=kumo.ModelRegistryLocation(
                         'm', alias='champion', comment='Mine.'))
    trainer = kumo.Trainer(plan)
    result = trainer.fit(table, snapshot=snapshot, mlops=cfg)

    assert 'MLflow tracking failed' not in caplog.text
    parent, = [
        run for run in _runs(mlflow.MlflowClient())
        if 'mlflow.parentRunId' not in run.data.tags
    ]
    model, = mlflow.search_logged_models(output_format='list')
    assert (model.name, model.source_run_id) == ('m', parent.info.run_id)
    metrics = {metric.key: metric.value for metric in model.metrics}
    holdout = {name: value for name, value in logged_names(
        result.metrics).items() if not name.startswith(('train/', 'val/'))}
    assert metrics == {
        **holdout,
        **{summary.name: summary.value for summary in result._summaries
           if isinstance(summary, Scalar) and summary.role == 'model'}}
    assert {(metric.dataset_name, metric.dataset_digest)
            for metric in model.metrics} == {(None, None)}
    version = mlflow.MlflowClient().get_model_version_by_alias('m', 'champion')
    assert version.source == model.model_uri
    assert parent.data.tags.items() >= {
        'kumo.kind': 'fit', 'kumo.bundle_id': result.id,
        'kumo.model_name': 'm', 'kumo.model_version': str(version.version),
        'kumo.model_alias': 'champion'}.items()
    assert result.registered_model == PublishedModel(
        name='m', version=str(version.version),
        uri=f'models:/m/{version.version}',
        metadata={'kumo_bundle_id': result.id}, metrics=result.metrics)
    assert version.tags == {'kumo_bundle_id': result.id}
    assert version.description == '\n'.join([
        'Mine.', '', '<!-- kumo:version -->',
        _card.version_headline(holdout, 'mae'),
        f'Trained by `{parent.info.run_name}`. Bundle '
        f"`{tmp_path / 'root' / 'models' / result.id}`.",
        '<!-- /kumo:version -->'])
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
    with pytest.warns(UserWarning, match='skipped 1 of 2 requested '
                      r'entities \(50\.0%\): not in snapshot'):
        partial = mlflow.pyfunc.load_model('models:/m@champion').predict(
            pd.DataFrame({'ENTITY': [7, 10**9]}),
            params={'snapshot': str(tmp_path / 'exports' / location)})
    assert list(partial['ENTITY']) == [7]
    assert load_model('models:/m@champion').result.id == result.id
    champion = kumo.Trainer.load_from_connector(
        cfg, location=kumo.ModelRegistryLocation('m', alias='champion'))
    assert champion.result.id == result.id
    assert champion.result.registered_model == PublishedModel(
        'm', str(version.version), f'models:/m/{version.version}')
    assert (champion.result.registered_alias,
            result.registered_alias) == ('champion', 'champion')
    assert champion.export(cfg).version == str(version.version)
    client = mlflow.MlflowClient()
    assert len(client.search_model_versions("name = 'm'")) == 1
    other = kumo.MLOps(str(tmp_path / 'root'), tracking=cfg.tracking,
                       registry=kumo.ModelRegistryLocation('m2'))
    first = champion.export(other)
    assert first.version == '1' and champion.export(other) == first
    assert load_model('models:/m2/1').result.id == result.id

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
    scratch = pyfunc.unwrap_python_model()._predictor.trainer.scratch_dir
    assert list(scratch.iterdir()) == []
    assert set(tables.glob('*')) == before | {tables / 'other-call'}


def test_pyfunc_names_a_missing_prediction_time_and_needs_no_threshold(
        fitted):
    from kumo.engine.integrations._mlflow_model import predict

    trainer, table, snapshot = fitted
    rows = pd.DataFrame({'ENTITY': range(5)})
    with pytest.raises(ValueError) as error:
        predict(trainer, snapshot, rows,
                {'binary_classification_threshold': .5})
    assert str(error.value) == ("temporal query: pass params={"
                                "'prediction_time': ...} or a TIMESTAMP "
                                "column")

    time = str(pd.Timestamp(table.timestamps[0]))
    scores = predict(trainer, snapshot, rows, {'prediction_time': time})
    assert len(scores) == 5
    assert scores['TARGET_PRED'].isna().all()
    assert scores['True_PROB'].between(0, 1).all()


def test_pyfunc_returns_rows_in_input_order(fitted):
    from kumo.engine.integrations._mlflow_model import predict

    trainer, table, snapshot = fitted
    time = pd.Timestamp(table.timestamps[0])
    entities = [7, 3, 9, 3, 999, 1, 40, 2, 33, 11, 3, 25] * 10
    rows = pd.DataFrame({
        'ENTITY': entities,
        'TIMESTAMP': [time - pd.Timedelta(days=i % 3)
                      for i in range(len(entities))],
    })
    scored = predict(trainer, snapshot, rows,
                     {'binary_classification_threshold': .5})
    assert scored['ENTITY'].tolist() == [e for e in entities if e != 999]
    assert 'ANCHOR_TIMESTAMP' not in scored


def test_scores_pair_with_their_rows_when_scoring_reorders():
    from kumo.engine.integrations._mlflow_model import _in_input_order

    day = pd.Timestamp('2024-05-01 12:00:00.250')
    rows = pd.DataFrame({
        'ENTITY1': ['b', 'a', 'b', 'c', 'a', 'gone'],
        'TIMESTAMP': [day, day, day - pd.Timedelta(days=1), day, day, day],
    })
    seconds = day.floor('s')
    scored = pd.DataFrame({
        'ENTITY': ['a', 'c', 'b', 'a', 'b'],
        'ANCHOR_TIMESTAMP': [seconds, seconds, seconds - pd.Timedelta(days=1),
                             seconds, seconds],
        'SCORE': [1.0, 3.0, 2.5, 1.0, 2.0],
    })

    ordered = _in_input_order(rows, scored)

    assert ordered.to_dict('list') == {
        'ENTITY': ['b', 'a', 'b', 'c', 'a'],
        'SCORE': [2.0, 1.0, 2.5, 3.0, 1.0],
    }


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
    from kumo.engine._tracking import TrackingSession
    from kumo.engine.trainer._run_spec import FitContext, fit_spec

    monkeypatch.setattr(_tracking, 'FINISH_SECONDS', 0.01)

    class Tracker:
        def __init__(self, fail):
            self.fail, self.calls = fail, []

        def start(self, record):
            self.calls.append('start')

        def on_trial(self, record, event):
            pass

        def log_model(self, bundle_dir, manifest, snapshot, registry):
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
        session.start(lambda: fit_spec(FitContext('q', {})))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, error = session.log_model(result.bundle_dir, None, None)
        session.finish(None, error)
        session._worker.join(5)
        assert tracker.calls[:2] == ['start', 'log_model']
        assert tracker.calls[2] == ('finish', error)
        assert (error is not None) is fail
        assert any('upload failed' in str(w.message) for w in caught) is fail


@pytest.mark.parametrize('stall', ['slow upload', 'dead thread'])
def test_model_logging_wait_is_bounded(monkeypatch, stall):
    import threading

    from kumo.engine import _tracking
    from kumo.engine._tracking import TrackingSession
    from kumo.engine.trainer._run_spec import FitContext, fit_spec

    monkeypatch.setattr(_tracking, 'MODEL_SECONDS', 0.5)
    release = threading.Event()

    class Tracker:
        calls = []

        def start(self, record):
            pass

        def on_trial(self, record, event):
            pass

        def log_model(self, bundle_dir, manifest, snapshot, registry):
            release.wait(30)

        def finish(self, record, error):
            self.calls.append((record, error))

    tracker = Tracker()
    session = TrackingSession(tracker)
    session.start(lambda: fit_spec(FitContext('q', {})))
    if stall == 'dead thread':
        session._submit(lambda: (_ for _ in ()).throw(SystemExit()))
        session._worker.join(5)
    with pytest.warns(UserWarning, match='Could not log the model'):
        _, error = session.log_model(None, None, None)
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
    from kumo.engine.trainer._run_spec import fit_spec

    for name in integration._HTTP_LIMITS:
        monkeypatch.delenv(name, raising=False)
    tracker = integration.MlflowTracking(log_system_metrics=False)
    location = kumo.ModelRegistryLocation('m')
    seen = []

    def log_model(target, client, bundle_dir, manifest, snapshot, **kwargs):
        seen.append(os.environ.get('MLFLOW_HTTP_REQUEST_TIMEOUT'))
        if len(seen) == 2:
            tracker.finish(None, None)
        return SimpleNamespace(model_id=None, model_uri='runs:/x/model')

    monkeypatch.setattr(integration, '_log_model', log_model)
    monkeypatch.setattr(integration, '_register', lambda *args: 'registered')
    for _ in range(2):
        session = TrackingSession(tracker)
        session.start(lambda: fit_spec(CONTEXT))
        registered, error = session.log_model(None, None, None, location)
        assert error is None
        during = os.environ.get('MLFLOW_HTTP_REQUEST_TIMEOUT')
        session.finish(None, None)
        session._worker.join(5)
        assert registered == ('registered' if len(seen) == 1 else None)
        assert during == ('30' if len(seen) == 1 else None)

    assert seen == ['600', '600']
    assert 'MLFLOW_HTTP_REQUEST_TIMEOUT' not in os.environ
    assert 'the model is not registered' in caplog.text


def test_a_model_with_text_columns_requires_the_text_extra(tmp_path):
    from kumo_core.encoders.config import EncodersConfig
    from kumo_core.io.json import save_json
    from kumoapi.encoder import Numerical, TextEmbedding

    from kumo.engine.integrations.mlflow import _requirement

    def requirement(encoders):
        path = tmp_path / 'encoders.json'
        save_json(EncodersConfig(encoders={'users': encoders}), str(path))
        return _requirement(path)

    assert requirement({'age': Numerical()}) == (
        f'kumo-engine[s3,gcs]=={kumo.__version__}')
    assert requirement({'age': Numerical(), 'bio': TextEmbedding('words')
                        }) == f'kumo-engine[s3,gcs,text]=={kumo.__version__}'


def test_a_tracker_dropped_earlier_says_the_model_was_not_logged():
    import time

    from kumo.engine._tracking import TrackingSession
    from kumo.engine.trainer._run_spec import FitContext, fit_spec

    class Tracker:
        def start(self, record):
            raise RuntimeError('server down')

        def log_model(self, bundle_dir, manifest, snapshot, registry):
            raise AssertionError('not called')

        def finish(self, record, error):
            pass

    session = TrackingSession(Tracker())
    session.start(lambda: fit_spec(FitContext('q', {})))
    deadline = time.monotonic() + 10
    while session._tracker is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.warns(UserWarning, match=r'Could not log the model: '
                      r'RuntimeError: tracking stopped after an earlier '
                      r'failure \(RuntimeError: server down\)'):
        registered, error = session.log_model(None, None, None)
    session.finish(None, error)
    assert registered is None and isinstance(error, RuntimeError)
    assert str(session.error) == 'server down'


@pytest.mark.parametrize('on_error', ['warn', 'raise'])
def test_a_registration_failure_is_tagged_on_the_fit_run_and_can_raise(
        table_and_snapshot, tmp_path, monkeypatch, on_error):
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    from kumo.engine.integrations.mlflow import MlflowTracking

    detail = ("CATALOG_DOES_NOT_EXIST: Catalog 'nope' does not exist. " +
              'x' * 300 + ' the actionable end')

    def missing_catalog(*args, **kwargs):
        raise MlflowException(detail)

    monkeypatch.setattr(MlflowClient, 'create_model_version',
                        missing_catalog)
    table, snapshot = table_and_snapshot
    cfg = kumo.MLOps(str(tmp_path / 'root'),
                     tracking=MlflowTracking(log_system_metrics=False),
                     registry=kumo.ModelRegistryLocation('nope.s.m'),
                     on_error=on_error)
    trainer = kumo.Trainer(_plan())
    cause = f'MlflowException: {detail}'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if on_error == 'warn':
            result = trainer.fit(table, snapshot=snapshot, mlops=cfg)
        else:
            with pytest.raises(RuntimeError) as raised:
                trainer.fit(table, snapshot=snapshot, mlops=cfg)
            assert str(raised.value).endswith(cause)
            assert isinstance(raised.value.__cause__, MlflowException)
            result = trainer.result
    reported = [str(w.message) for w in caught if 'nope' in str(w.message)]
    assert reported == ([f'Could not register the model: {cause}']
                        if on_error == 'warn' else [])
    assert result.id and result.registered_model is None

    run, = [run for run in _runs(mlflow.MlflowClient())
            if 'mlflow.parentRunId' not in run.data.tags]
    assert run.info.status == 'FAILED'
    assert run.data.tags['kumo.error'].startswith('MlflowException: '
                                                  'CATALOG_DOES_NOT_EXIST')
    assert run.data.tags['kumo.bundle_id'] == result.id


def test_a_tracking_start_failure_names_its_cause_and_can_raise(
        table_and_snapshot, tmp_path, monkeypatch, caplog):
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    from kumo.engine.integrations.mlflow import MlflowTracking

    def no_folder(self, name, *args, **kwargs):
        raise MlflowException(f'RESOURCE_DOES_NOT_EXIST: Parent of {name} '
                              f'does not exist.')

    table, snapshot = table_and_snapshot
    cfg = kumo.MLOps(str(tmp_path / 'root'),
                     tracking=MlflowTracking('/NoSuchFolder/churn',
                                             log_system_metrics=False),
                     on_error='raise')
    warn = dataclasses.replace(cfg, on_error='warn')
    monkeypatch.setattr(MlflowClient, 'get_experiment_by_name',
                        lambda self, name: None)
    monkeypatch.setattr(MlflowClient, 'create_experiment', no_folder)
    trainer = kumo.Trainer(_plan())
    with pytest.raises(RuntimeError, match='RESOURCE_DOES_NOT_EXIST') as raised:
        trainer.fit(table, snapshot=snapshot, mlops=cfg)

    assert isinstance(raised.value.__cause__, MlflowException)
    assert trainer.result.id
    off, = [record for record in caplog.records
            if 'off for the rest' in record.getMessage()]
    assert 'MlflowException: RESOURCE_DOES_NOT_EXIST' in off.getMessage()
    assert not off.exc_info

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        trainer.fit(table, snapshot=snapshot, mlops=warn)
    reported = [str(w.message) for w in caught
                if 'RESOURCE_DOES_NOT_EXIST' in str(w.message)]
    assert reported == [
        'Could not log the model: RuntimeError: tracking stopped after an '
        'earlier failure (MlflowException: RESOURCE_DOES_NOT_EXIST: Parent '
        'of /NoSuchFolder/churn does not exist.)'
    ]


def test_a_logged_model_is_named_after_its_model():
    from kumo.engine.integrations.mlflow import logged_model_name

    manifest = SimpleNamespace(parents=SimpleNamespace(
        training_table=SimpleNamespace(
            query='PREDICT users.CHURN FOR EACH users.ID',
            task_type='binary_classification')))
    assert logged_model_name('main.ml.churn', manifest) == 'churn'
    assert logged_model_name('a/b:c%d"e\'f', None) == 'a-b-c-d-e-f'
    assert logged_model_name(None, manifest) == 'users-binary_classification'
    assert logged_model_name(None, None) == 'model'
