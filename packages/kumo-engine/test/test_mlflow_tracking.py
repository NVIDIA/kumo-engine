# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import socket
import threading
import time
import warnings
from pathlib import Path

import pytest
from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.summaries import Dataset
from kumoapi.encoder import Numerical
from kumoapi.model_plan import ModelPlan
from test_tracking_contract import (
    CONTEXT,
    INPUTS,
    join_tracking_threads,
    replay,
)

import kumo.engine as kumo
from kumo.engine._manifest import TrackingRef
from kumo.engine._tracking import logged_names, run_record
from kumo.engine.trainer._run_spec import fit_spec

mlflow = pytest.importorskip('mlflow')
autologging_utils = pytest.importorskip('mlflow.utils.autologging_utils')
pytestmark = pytest.mark.usefixtures('tracking_uri')


def _runs(client):
    return client.search_runs(
        [e.experiment_id for e in client.search_experiments()])


def _fit_run():
    run, = [run for run in _runs(mlflow.MlflowClient())
            if 'mlflow.parentRunId' not in run.data.tags]
    return run


def test_fit_logs_a_run_per_fit_and_per_trial(table_and_snapshot, quick_plan,
                                              tmp_path, caplog):
    from kumo.engine.integrations.mlflow import MlflowTracking

    table, snapshot = table_and_snapshot
    client = mlflow.MlflowClient()
    plan = quick_plan
    plan.training_job.num_experiments = 2
    plan.optimization.base_lr = [0.01, 0.001]
    plan.column_processing.encoder_overrides = {
        'ents.amount ($)': Numerical('standard')
    }
    mlflow.pytorch.autolog()
    try:
        with mlflow.start_run() as outer:
            tracking = MlflowTracking(experiment_name='ents')
            mlops = kumo.MLOps(artifact_root=str(tmp_path / 'root'),
                               tracking=tracking)
            result = kumo.Trainer(plan).fit(table, snapshot=snapshot,
                                            mlops=mlops)
            assert mlflow.active_run().info.run_id == outer.info.run_id
        assert autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED is False
    finally:
        mlflow.pytorch.autolog(disable=True)

    runs = _runs(client)
    parent, = [
        run for run in runs
        if run.data.tags.get('mlflow.parentRunId') == outer.info.run_id
    ]
    assert tracking.run_id == parent.info.run_id
    assert re.fullmatch(r'fit-ents-regression-\d{8}-\d{6}',
                        parent.info.run_name)
    children = sorted(
        (run for run in runs
         if run.data.tags.get('mlflow.parentRunId') == parent.info.run_id),
        key=lambda run: run.info.run_name)
    assert len(runs) == 4
    assert client.get_experiment(parent.info.experiment_id).name == 'ents'
    assert parent.info.status == 'FINISHED'
    assert {k for k in parent.data.tags if k.startswith('kumo.')} == {
        'kumo.kind', 'kumo.bundle_id', 'kumo.dropped_params',
        'kumo.snapshot_id', 'kumo.task_type', 'kumo.entity',
        'kumo.tune_metric', 'kumo.engine_version', 'kumo.best_trial',
        'kumo.bundle_uri'}
    assert parent.data.tags['kumo.bundle_uri'] == str(
        tmp_path / 'root' / 'models' / result.id)
    assert parent.data.tags['kumo.snapshot_id'] == snapshot.id
    assert parent.data.tags['kumo.task_type'] == 'regression'
    assert parent.data.tags['kumo.bundle_id'] == result.id
    stored = tmp_path / 'root' / 'models' / result.id / PROVENANCE_FILE
    assert json.loads(stored.read_text())['tracking'] == {
        'platform': 'mlflow', 'run_id': parent.info.run_id,
        'experiment': 'ents', 'run_name': parent.info.run_name}
    assert result.tracking == TrackingRef(
        platform='mlflow', run_id=parent.info.run_id, experiment='ents',
        run_name=parent.info.run_name)
    logged = client.download_artifacts(parent.info.run_id, PROVENANCE_FILE,
                                       str(tmp_path))
    assert json.loads(Path(logged).read_text())['id'] == result.id
    assert int(parent.data.tags['kumo.dropped_params']) >= 1
    params = parent.data.params
    assert json.loads(params['optimization.base_lr']) == [0.01, 0.001]
    assert not any('$' in name for name in params)
    metrics = {
        name: value
        for name, value in parent.data.metrics.items()
        if not name.startswith('system/')
    }
    tested = {name: value for name, value in logged_names(
        result.metrics).items() if not name.startswith(('train/', 'val/'))}
    assert {name for name in metrics
            if name.startswith(('test/', 'before_refit/'))} == set(tested)
    health = json.loads(mlflow.artifacts.load_text(
        f'{parent.info.artifact_uri}/data/health.json'))
    assert health['data'][0][:2] == ['ents', 400]
    assert {name: metrics[name] for name in tested} == tested
    assert 'off for the rest of this run' not in caplog.text
    ents, = parent.inputs.dataset_inputs
    assert (ents.dataset.name, ents.dataset.source_type) == ('ents', 'local')
    assert json.loads(ents.dataset.source) == {
        'uri': str((tmp_path / 'data' / 'ents.parquet').resolve())
    }

    assert [run.info.run_name for run in children] == [
        f'trial-{i}-{parent.info.run_name}' for i in (1, 2)]
    for child in children:
        assert child.info.status == 'FINISHED'
        assert child.data.tags['kumo.status'] == 'succeeded'
        assert {'val/best_mae', 'val/mae'} <= set(child.data.metrics)
        history = client.get_metric_history(child.info.run_id, 'val/mae')
        assert sorted(metric.step for metric in history) == [1, 2]
    selected = min(children, key=lambda run: run.data.metrics['val/best_mae'])
    for name in ('train/loss', 'val/mae'):
        curve, trial = (sorted((metric.step, metric.value)
                               for metric in client.get_metric_history(
                                   run.info.run_id, name))
                        for run in (parent, selected))
        assert curve == trial and len(curve) == 2


def test_failed_fit_marks_the_run_failed(table_and_snapshot, tmp_path,
                                         monkeypatch):
    import kumo_core.training.orchestration as orchestration

    from kumo.engine.integrations.mlflow import MlflowTracking

    def fail(*args, **kwargs):
        raise ValueError('stop')

    monkeypatch.setattr(orchestration, 'run_callable_training', fail)
    table, snapshot = table_and_snapshot

    with kumo.mlops(kumo.MLOps(artifact_root=str(tmp_path / 'root'),
                               tracking=MlflowTracking())):
        with pytest.raises(ValueError, match='stop'):
            kumo.Trainer(ModelPlan()).fit(table, snapshot=snapshot)

    run, = _runs(mlflow.MlflowClient())
    assert run.info.status == 'FAILED'
    assert run.data.tags['kumo.error'] == 'ValueError: stop'
    assert mlflow.active_run() is None


def test_mlflow_error_ends_runs_failed_and_is_not_raised(monkeypatch, caplog):
    from mlflow.tracking import MlflowClient

    from kumo.engine.integrations.mlflow import MlflowTracking

    original = MlflowClient.log_batch

    def down(self, *args, metrics=(), **kwargs):
        if metrics:
            raise RuntimeError('tracking server down')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MlflowClient, 'log_batch', down)
    replay(MlflowTracking(log_system_metrics=False))

    assert caplog.text.count('off for the rest of this run') == 1
    assert mlflow.active_run() is None
    statuses = {run.info.status for run in _runs(mlflow.MlflowClient())}
    assert statuses == {'FAILED'}


def test_a_user_run_name_names_every_run():
    from kumo.engine.integrations.mlflow import MlflowTracking

    tracking = MlflowTracking(run_name='nightly', log_system_metrics=False)
    replay(tracking)
    join_tracking_threads()

    assert _fit_run().info.run_name == 'fit-nightly'


SAGEMAKER = ('arn:aws:sagemaker:us-west-2:account:'
             'mlflow-tracking-server/demo')


@pytest.mark.parametrize('uri,good,bad', [
    (SAGEMAKER, ['users-churn', 'u--1', 'a' * 57],
     ['users_churn', 'main.ml.churn', 'bad/name:x', '-churn', 'a' * 58]),
    ('arn:aws:sagemaker:us-east-1:account:mlflow-app/app-1',
     ['users-churn'], ['users_churn']),
    ('databricks-uc', ['main.ml.users_churn', 'ml.users-churn', 'churn'],
     ['main.ml.users churn', 'a.b.c.d', 'main..churn', 'main.ml.a/b']),
    ('sqlite:///mlflow.db', ['bad/name:x', 'users_churn', 'a.b.c.d'], []),
])
def test_registry_model_names_are_checked_when_mlops_is_built(
        tmp_path, uri, good, bad):
    from kumo.engine.integrations.mlflow import (
        MlflowTracking,
        check_model_name,
    )

    for name in good:
        check_model_name(name, uri)
    tracking = MlflowTracking(log_system_metrics=False,
                              target=f'sqlite:///{tmp_path}/m.db',
                              experiment_name='e')
    tracking.target.registry_uri = uri
    for name in bad:
        with pytest.raises(ValueError, match="e.g. '"):
            kumo.MLOps(str(tmp_path), tracking=tracking,
                       registry=kumo.ModelRegistryLocation(name))


def test_building_mlops_creates_the_experiment_or_says_why_it_cannot(
        tmp_path, monkeypatch):
    from kumo.engine.integrations.mlflow import MlflowTracking

    uri = f'sqlite:///{tmp_path}/m.db'
    kumo.MLOps(str(tmp_path), tracking=MlflowTracking(
        'churn', log_system_metrics=False, target=uri))
    assert mlflow.MlflowClient(uri).get_experiment_by_name('churn')

    def refuse(self, name, *args, **kwargs):
        raise mlflow.exceptions.MlflowException('folder does not exist')

    monkeypatch.setattr(mlflow.MlflowClient, 'create_experiment', refuse)
    with pytest.raises(RuntimeError, match=r"experiment '/Shared/x/churn'.*"
                       r"folder does not exist.*create its parent folder"):
        kumo.MLOps(str(tmp_path), tracking=MlflowTracking(
            '/Shared/x/churn', log_system_metrics=False, target=uri))


@pytest.mark.parametrize('uri,command', [
    (SAGEMAKER, 'aws sagemaker create-presigned-mlflow-tracking-server-url '
     '--tracking-server-name demo --region us-west-2'),
    ('sqlite:///mlflow.db', None),
])
def test_a_tracked_fit_on_sagemaker_says_how_to_open_the_ui(
        uri, command, monkeypatch):
    from kumo.engine.integrations import mlflow as integration
    from kumo.engine.integrations.mlflow import MlflowTracking

    monkeypatch.setattr(integration, '_HINTED', set())

    tracker = MlflowTracking(log_system_metrics=False)
    tracker.start(run_record(fit_spec(CONTEXT)))
    tracker.target.tracking_uri = uri
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tracker.finish(run_record(fit_spec(CONTEXT)), None)
    hints = [str(w.message) for w in caught if 'Studio' in str(w.message)]
    assert hints == ([] if command is None else [
        "MLflow run 'fit-ents-regression-20261029-061800' in experiment "
        "'Default': MLflow's run links do not open on SageMaker: open the "
        f'MLflow UI from SageMaker Studio or at the URL `{command}` prints.'])
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter('always')
        tracker.finish(run_record(fit_spec(CONTEXT)), None)
    assert not [w for w in again if 'Studio' in str(w.message)]


def test_user_disabled_autologging_stays_disabled():
    from kumo.engine.integrations.mlflow import MlflowTracking

    tracker = MlflowTracking(log_system_metrics=False)
    with autologging_utils.disable_autologging():
        tracker.start(run_record(fit_spec(CONTEXT)))
        tracker.finish(None, None)
        assert autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED is True


@pytest.fixture
def hung_server():
    server = socket.create_server(('127.0.0.1', 0))
    held = []

    def accept():
        while True:
            try:
                held.append(server.accept()[0])
            except OSError:
                return

    threading.Thread(target=accept, daemon=True).start()
    try:
        yield f'http://127.0.0.1:{server.getsockname()[1]}'
    finally:
        server.close()
        for connection in held:
            connection.close()
        join_tracking_threads()


def test_a_hung_server_does_not_stall_the_fit(hung_server, monkeypatch):
    import kumo.engine._tracking as tracking
    from kumo.engine.integrations.mlflow import MlflowTracking

    monkeypatch.setattr(tracking, 'FINISH_SECONDS', 0.5)
    monkeypatch.setenv('MLFLOW_HTTP_REQUEST_MAX_RETRIES', '0')
    monkeypatch.delenv('MLFLOW_HTTP_REQUEST_TIMEOUT', raising=False)
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(hung_server)
    try:
        began = time.monotonic()
        replay(MlflowTracking(log_system_metrics=False))
        elapsed = time.monotonic() - began
        limits = {name: os.environ.get(name) for name in (
            'MLFLOW_HTTP_REQUEST_MAX_RETRIES', 'MLFLOW_HTTP_REQUEST_TIMEOUT')}
    finally:
        mlflow.set_tracking_uri(previous)

    assert elapsed < 1.5
    assert limits == {'MLFLOW_HTTP_REQUEST_MAX_RETRIES': '0',
                      'MLFLOW_HTTP_REQUEST_TIMEOUT': '30'}


def test_source_tables_are_logged_as_native_inputs(monkeypatch):
    from mlflow.data.delta_dataset_source import DeltaDatasetSource

    from kumo.engine.integrations.mlflow import MlflowTracking

    monkeypatch.setattr(DeltaDatasetSource, '_lookup_table_id',
                        lambda self, name: 'table-id')
    replay(MlflowTracking(log_system_metrics=False), summaries=[
        Dataset(name, source, {}, {}, 'source table')
        for name, source in INPUTS.items()])

    run = _fit_run()
    inputs = {
        item.dataset.name: (item.dataset.source_type,
                            json.loads(item.dataset.source))
        for item in run.inputs.dataset_inputs
    }
    assert inputs == {
        'ents': ('s3', {'uri': 's3://bucket/ents.parquet'}),
        'orders': ('delta_table', {
            'delta_table_name': 'main.sales.orders',
            'delta_table_version': 7,
            'is_databricks_uc_table': True,
            'delta_table_id': 'table-id',
        }),
    }
    orders = next(item for item in run.inputs.dataset_inputs
                  if item.dataset.name == 'orders')
    assert mlflow.data.get_source(orders).delta_table_id == 'table-id'


def _fit_manifest():
    from kumo.engine._manifest import (
        ModelManifest,
        ModelParents,
        SnapshotManifest,
        SnapshotParents,
        SourceTableEntry,
        TrainingTableManifest,
        TrainingTableParents,
    )

    return ModelManifest(id='m', parents=ModelParents(
        training_table=TrainingTableManifest(
            id='tt-fadf04', created_at='2026-11-01T10:00:00+00:00',
            location='s3://root/training_tables/tt-fadf04',
            parents=TrainingTableParents(snapshot=SnapshotManifest(
                id='g@5ac245', created_at='2026-11-01T09:00:00+00:00',
                location='s3://root/snapshots/g/5ac245',
                parents=SnapshotParents(source_tables={
                    name: SourceTableEntry(
                        connector='S3', location=f's3://data/{name}',
                        version='3', version_kind='exact')
                    for name in ('users', 'orders')}))))))


def test_a_fits_inputs_are_logged_in_role_order():
    from kumo.engine._manifest import FIT_ROLES, artifact_summaries
    from kumo.engine.integrations.mlflow import MlflowTracking

    replay(MlflowTracking(log_system_metrics=False),
           summaries=artifact_summaries(_fit_manifest(), FIT_ROLES))

    assert [item.dataset.name for item in
            _fit_run().inputs.dataset_inputs] == [
        'training table 2026-11-01-fadf04',
        'graph snapshot 2026-11-01-5ac245', 'orders', 'users']


def test_the_snapshot_input_lists_its_tables_and_rows(tmp_path):
    from kumo.engine._data_summaries import table_details
    from kumo.engine._manifest import FIT_ROLES, artifact_summaries
    from kumo.engine.integrations.mlflow import MlflowTracking

    (tmp_path / 'ingest_stats.json').write_text(json.dumps(
        {'stats': {'node_counts': {'users': 300, 'orders': 6000}}}))
    (tmp_path / 'diskgraph_config.json').write_text(json.dumps(
        {'tables': [{'name': 'users'}, {'name': 'orders'}]}))
    replay(MlflowTracking(log_system_metrics=False),
           summaries=artifact_summaries(_fit_manifest(), FIT_ROLES,
                                        table_details(tmp_path, 'g@5ac245')))

    snapshot, = [item.dataset for item in _fit_run().inputs.dataset_inputs
                 if item.dataset.name.startswith('graph snapshot')]
    assert json.loads(snapshot.schema) == {'mlflow_colspec': [
        {'name': 'snapshot id', 'type': 'g@5ac245'},
        {'name': 'users', 'type': '300 rows'},
        {'name': 'orders', 'type': '6,000 rows'}]}
    assert json.loads(snapshot.profile) == {'tables': 2, 'rows': 6300}


def test_link_prediction_metric_names_are_logged():
    from kumo.engine._tracking import RunRecord
    from kumo.engine.integrations.mlflow import MlflowTracking

    tracker = MlflowTracking(experiment_name='lp', log_system_metrics=False)
    tags = {'kumo.kind': 'fit', 'kumo.run_id': 'r'}
    tracker.start(RunRecord(name='r', tags=tags))
    tracker.finish(
        RunRecord(name='r', tags=tags, metrics={
            'test_map@10': 0.5,
            'val_precision@10': 0.25
        }), None)

    run = mlflow.get_run(tracker.run_id)
    assert run.info.status == 'FINISHED'
    assert run.data.metrics == {
        'test_map_at_10': 0.5,
        'val_precision_at_10': 0.25
    }


class _Monitor:
    def collect_metrics(self):
        pass

    def aggregate_metrics(self):
        return {'cpu': 1.0}

    def clear_metrics(self):
        pass


def test_system_metrics_skip_a_failed_log_and_stop_after_repeated_ones(
        monkeypatch, caplog):
    import logging

    from kumo.engine.integrations import mlflow as integration

    batches = []

    def log(readings):
        batches.append([(values, step) for values, step, _ in readings])
        if len(batches) == 1 or len(batches) >= 3:
            raise ConnectionError('blip')

    monkeypatch.setattr(integration, '_metric_monitors', lambda: [_Monitor()])
    monkeypatch.setattr(integration, '_FLUSH_SECONDS', 0.0)
    caplog.set_level(logging.INFO, integration.logger.name)
    with integration._SystemMetrics(log, interval=0.001) as metrics:
        metrics._thread.join(10)

    reading = {'system/cpu': 1.0}
    assert batches == [[(reading, 0), (reading, 1)], *[
        [(reading, step)]
        for step in range(2, 3 + integration._SYSTEM_METRICS_FAILURES)]]
    assert caplog.text.count('Could not log system metrics') == 1
    assert 'ConnectionError: blip' in caplog.text
    assert 'stopped after 6 failed logs in a row' in caplog.text


def test_system_metrics_sample_every_second_unless_the_environment_says(
        monkeypatch):
    from kumo.engine.integrations import mlflow as integration

    monkeypatch.setattr(integration, '_metric_monitors', lambda: [_Monitor()])
    batches = []
    with integration._SystemMetrics(batches.append) as metrics:
        pass
    assert batches == []
    assert metrics._interval == 1.0
    monkeypatch.setenv('MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL', '2')
    assert integration._SystemMetrics(batches.append)._interval == 2.0
