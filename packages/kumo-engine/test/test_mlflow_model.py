# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest
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

    monkeypatch.setenv('KUMO_SCHEDULER', 'airflow')
    monkeypatch.setenv('KUMO_JOB_RUN_ID', 'run-7')
    table, snapshot = table_and_snapshot
    plan = _plan()
    tracker = MlflowTracking(log_system_metrics=False,
                             registered_model_name='m', aliases=('champion', ))
    result = kumo.Trainer(plan).fit(table, snapshot=snapshot, tracking=tracker)

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
                'kumo.bundle_id': result.bundle_id,
                'kumo.scheduler': 'airflow',
                'kumo.job_run_id': 'run-7',
            }
    assert tracker.registered == PublishedModel(
        name='m', version=str(version.version),
        uri=f'models:/m/{version.version}')

    info = mlflow.models.get_model_info('models:/m@champion')
    assert info.metadata == {'kumo_bundle_id': result.bundle_id}
    provenance = json.loads(mlflow.artifacts.load_text(
        f'{model.artifact_location}/artifacts/provenance.json'))
    assert provenance == result.provenance
    assert provenance['tracking'] == {
        'run_id': parent.data.tags['kumo.run_id']
    }
    assert [(c['name'], c['type']) for c in info.signature.inputs.to_dict()
            ] == [('ENTITY', 'long'), ('TIMESTAMP', 'datetime')]
    assert [(c['name'], c['type']) for c in info.signature.outputs.to_dict()
            ] == [('ENTITY', 'long'), ('TARGET_PRED', 'double')]

    location = snapshot.export(
        kumo.FileConnector(root_dir=str(tmp_path / 'exports')), 'snapshots')
    served = mlflow.pyfunc.load_model('models:/m@champion').predict(
        pd.DataFrame({'ENTITY': range(0, 400, 7)}),
        params={'snapshot': str(tmp_path / 'exports' / location)})
    assert list(served) == ['ENTITY', 'TARGET_PRED']
    assert len(served) == len(range(0, 400, 7))
    assert load_model('models:/m@champion').bundle_id == result.bundle_id

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
        fitted, tracking_uri, tmp_path):
    from kumo.engine.integrations.mlflow import log_model

    trainer, table, snapshot = fitted
    location = snapshot.export(
        kumo.FileConnector(root_dir=str(tmp_path / 'exports')), 'snapshots')

    info = log_model(trainer, snapshot=snapshot,
                     serving_snapshot=str(tmp_path / 'exports' / location))

    assert mlflow.get_run(info.run_id).info.status == 'FINISHED'
    requirements = mlflow.pyfunc.get_model_dependencies(info.model_uri)
    assert f'kumo-engine[gcs,s3]=={kumo.__version__}' in open(
        requirements).read().splitlines()
    assert {metric.key
            for metric in mlflow.get_logged_model(info.model_id).metrics
            } == set(trainer._training_result.metrics)
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
    tables = set((get_scratch_dir() / 'prediction_tables').glob('*'))

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
    assert not list((scratch / 'runs').glob('*'))
    assert set((get_scratch_dir() / 'prediction_tables').glob('*')) == tables


@pytest.mark.parametrize('options', [
    dict(registered_model_name='m', aliases=('SageMakerLifecycle-prod', )),
    dict(log_model=True, aliases=('champion', )),
])
def test_invalid_registration_is_refused(tracking_uri, options):
    from kumo.engine.integrations.mlflow import MlflowTracking

    with pytest.raises(ValueError):
        MlflowTracking(**options)


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

        def log_model(self, result, snapshot):
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
            error = session.log_model(result, None)
        session.finish(result, error)
        session._worker.join(5)
        assert tracker.calls[:2] == ['start', 'log_model']
        assert tracker.calls[2] == ('finish', error)
        assert (error is not None) is fail
        assert any('upload failed' in str(w.message) for w in caught) is fail
