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

import numpy as np
import pandas as pd
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
from kumo.engine._scratch import get_scratch_dir
from kumo.engine._tracking import logged_names, run_record
from kumo.engine.trainer._run_spec import fit_spec

mlflow = pytest.importorskip('mlflow')
autologging_utils = pytest.importorskip('mlflow.utils.autologging_utils')

QUERY = 'PREDICT ents.Y FOR EACH ents.ID'


@pytest.fixture
def tracking_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the default artifact root is ./mlruns
    previous = mlflow.get_tracking_uri()
    uri = f'sqlite:///{tmp_path / "mlflow.db"}'
    mlflow.set_tracking_uri(uri)
    try:
        yield uri
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)


@pytest.fixture
def table_and_snapshot(tmp_path, tracking_uri):
    previous_scratch = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        rng = np.random.default_rng(0)
        (tmp_path / 'data').mkdir()
        pd.DataFrame({
            'ID': np.arange(400),
            'Y': rng.random(400),
            'F1': rng.normal(size=400),
            'amount ($)': rng.normal(size=400),
        }).to_parquet(tmp_path / 'data' / 'ents.parquet')
        connector = kumo.FileConnector(root_dir=str(tmp_path / 'data'))
        ents = kumo.Table.from_source_table(connector['ents.parquet'],
                                            primary_key='ID').infer_metadata()
        graph = kumo.Graph(tables={'ents': ents}, edges=[])
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph=graph, query=QUERY)
        yield query.generate_training_table(snapshot), snapshot
    finally:
        kumo.configure(scratch_dir=previous_scratch)


def _runs(client):
    return client.search_runs(
        [e.experiment_id for e in client.search_experiments()])


def test_fit_logs_a_run_per_fit_and_per_trial(table_and_snapshot, tmp_path,
                                              caplog):
    from kumo.engine.integrations.mlflow import MlflowTracking

    table, snapshot = table_and_snapshot
    client = mlflow.MlflowClient()
    plan = ModelPlan()
    plan.training_job.num_experiments = 2
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 2
    plan.optimization.max_steps_per_epoch = 2
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
    assert re.fullmatch(r'fit-\d{8}-\d{4}', parent.info.run_name)
    children = sorted(
        (run for run in runs
         if run.data.tags.get('mlflow.parentRunId') == parent.info.run_id),
        key=lambda run: run.info.run_name)
    assert len(runs) == 4
    assert client.get_experiment(parent.info.experiment_id).name == 'ents'
    assert parent.info.status == 'FINISHED'
    assert {k for k in parent.data.tags if k.startswith('kumo.')} == {
        'kumo.kind', 'kumo.bundle_id', 'kumo.dropped_params',
        'kumo.snapshot_id', 'kumo.task_type', 'kumo.tune_metric',
        'kumo.engine_version', 'kumo.best_trial', 'kumo.bundle_uri',
        'kumo.artifact_root'}
    assert parent.data.tags['kumo.bundle_uri'] == str(
        tmp_path / 'root' / 'models' / result.id)
    assert parent.data.tags['kumo.snapshot_id'] == snapshot.id
    assert parent.data.tags['kumo.task_type'] == 'regression'
    assert parent.data.tags['kumo.bundle_id'] == result.id
    stored = tmp_path / 'root' / 'models' / result.id / PROVENANCE_FILE
    assert json.loads(stored.read_text())['tracking'] == {
        'platform': 'mlflow', 'run_id': parent.info.run_id}
    assert result.tracking.run_id == parent.info.run_id
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
        result.metrics).items() if name.startswith('test/')}
    assert {name for name in metrics if not name.startswith((
        'train/', 'val/', 'test/', 'data/', 'graph/'))} == {
            'seconds', 'search/best_val_mae'}
    assert metrics['graph/rows/ents'] == 400
    assert {name: metrics[name] for name in tested} == tested
    assert 'off for the rest of this run' not in caplog.text
    ents, = parent.inputs.dataset_inputs
    assert (ents.dataset.name, ents.dataset.source_type) == ('ents', 'local')
    assert json.loads(ents.dataset.source) == {
        'uri': str((tmp_path / 'data' / 'ents.parquet').resolve())
    }

    assert [run.info.run_name
            for run in children] == ['trial-1-attempt-1', 'trial-2-attempt-1']
    for child in children:
        assert child.info.status == 'FINISHED'
        assert child.data.tags['kumo.status'] == 'succeeded'
        assert {'val/best_mae', 'seconds',
                'val/mae'} <= set(child.data.metrics)
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


def test_mlflow_error_ends_runs_failed_and_is_not_raised(
        tracking_uri, monkeypatch, caplog):
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


def test_a_user_run_name_wins_over_the_default(tracking_uri):
    from kumo.engine.integrations.mlflow import MlflowTracking

    replay(MlflowTracking(run_name='nightly', log_system_metrics=False))
    join_tracking_threads()

    names = {run.info.run_name for run in _runs(mlflow.MlflowClient())
             if 'mlflow.parentRunId' not in run.data.tags}
    assert names == {'nightly'}


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
                       r"folder does not exist.*parent folder must exist"):
        kumo.MLOps(str(tmp_path), tracking=MlflowTracking(
            '/Shared/x/churn', log_system_metrics=False, target=uri))


@pytest.mark.parametrize('uri,command', [
    (SAGEMAKER, 'aws sagemaker create-presigned-mlflow-tracking-server-url '
     '--tracking-server-name demo --region us-west-2'),
    ('sqlite:///mlflow.db', None),
])
def test_a_tracked_fit_on_sagemaker_says_how_to_open_the_ui(
        tracking_uri, uri, command):
    from kumo.engine.integrations.mlflow import MlflowTracking

    tracker = MlflowTracking(log_system_metrics=False)
    tracker.start(run_record(fit_spec(CONTEXT)))
    tracker.target.tracking_uri = uri
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tracker.finish(run_record(fit_spec(CONTEXT)), None)
    hints = [str(w.message) for w in caught if 'Studio' in str(w.message)]
    assert hints == ([] if command is None else [
        "MLflow's run links do not open on SageMaker: open the MLflow UI "
        f'from SageMaker Studio or at the URL `{command}` prints.'])


def test_user_disabled_autologging_stays_disabled(tracking_uri):
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


def test_source_tables_are_logged_as_native_inputs(
        tracking_uri, monkeypatch):
    from mlflow.data.delta_dataset_source import DeltaDatasetSource

    from kumo.engine.integrations.mlflow import MlflowTracking

    monkeypatch.setattr(DeltaDatasetSource, '_lookup_table_id',
                        lambda self, name: 'table-id')
    replay(MlflowTracking(log_system_metrics=False), summaries=[
        Dataset(name, source, {}, {}, 'training')
        for name, source in INPUTS.items()])

    run, = [run for run in _runs(mlflow.MlflowClient())
            if run.data.tags['kumo.kind'] == 'fit']
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


def test_fit_inputs_come_from_the_embedded_parents():
    from kumo.engine._manifest import (
        ModelManifest,
        ModelParents,
        SnapshotManifest,
        SnapshotParents,
        SourceTableEntry,
        TrainingTableManifest,
        TrainingTableParents,
        stub,
    )
    from kumo.engine._tracking import fit_inputs

    entry = {'connector': 'LOCAL_FILE', 'location': '/data/ents.parquet',
             'version': '1790000000', 'version_kind': 'timestamp'}
    snapshot = SnapshotManifest(id='s', parents=SnapshotParents(
        source_tables={'ents': SourceTableEntry(**entry, sql='SELECT 1')}))
    custom = SourceTableEntry(**{**entry, 'version': None})
    table = TrainingTableManifest(id='t', parents=TrainingTableParents(
        snapshot=snapshot, source_table=custom))
    model = ModelManifest(id='m', parents=ModelParents(
        training_table=table, warm_start=ModelManifest(
            id='w', parents=ModelParents(training_table=TrainingTableManifest(
                id='u', parents=TrainingTableParents(snapshot=SnapshotManifest(
                    id='v', parents=SnapshotParents(source_tables={
                        'old': SourceTableEntry(**entry)}))))))))

    expected = {'ents': entry, 'training_table': {**entry, 'version': None}}
    assert fit_inputs(table) == expected
    assert fit_inputs(model) == expected
    assert fit_inputs(TrainingTableManifest(id='t', parents=TrainingTableParents(
        snapshot=stub('snapshot', 's'), source_table=custom))) == {
            'training_table': {**entry, 'version': None}}
    assert fit_inputs(stub('model', 'm')) == {}
    assert fit_inputs(None) == {}


def test_a_monitoring_run_is_new_and_tags_its_prediction_run(tracking_uri):
    from kumo.engine._tracking import RunRecord, log_run
    from kumo.engine.integrations.mlflow import MlflowTracking

    client = mlflow.MlflowClient()
    experiment = client.create_experiment('scores')
    predicted = client.create_run(experiment, tags={
        'kumo.kind': 'prediction', 'kumo.prediction_id': 'pred-1'
    }).info.run_id
    client.set_terminated(predicted)

    log_run(
        MlflowTracking(experiment_name='scores', log_system_metrics=False),
        'log_monitoring',
        RunRecord(name='monitor-pred-1', metrics={'psi': 0.1, 'ks': 0.2},
                  tags={'kumo.kind': 'monitoring', 'kumo.monitors': 'pred-1',
                        'kumo.bundle_id': 'b'}))

    monitor, = [run for run in _runs(client)
                if run.data.tags['kumo.kind'] == 'monitoring']
    assert monitor.info.run_name == 'monitor-pred-1'
    assert monitor.info.status == 'FINISHED'
    assert monitor.data.metrics == {'psi': 0.1, 'ks': 0.2}
    assert monitor.data.tags['kumo.monitors'] == 'pred-1'
    assert client.get_run(predicted).data.tags['kumo.monitored'] == (
        monitor.info.run_id)


def test_prediction_run_nests_under_the_active_run(tracking_uri):
    from dataclasses import replace

    from test_tracking_contract import PREDICTION_MANIFEST

    from kumo.engine._tracking import track_prediction
    from kumo.engine.integrations.mlflow import MlflowTracking

    client = mlflow.MlflowClient()
    tracking = MlflowTracking(experiment_name='scores',
                              log_system_metrics=False)
    with mlflow.start_run() as outer:
        track_prediction(PREDICTION_MANIFEST, tracking)
        assert mlflow.active_run().info.run_id == outer.info.run_id
    track_prediction(replace(PREDICTION_MANIFEST, id='trainer-predict-2'),
                     tracking)

    nested, top = sorted(
        (run for run in _runs(client) if 'kumo.kind' in run.data.tags),
        key=lambda run: 'mlflow.parentRunId' not in run.data.tags)
    assert nested.data.tags['mlflow.parentRunId'] == outer.info.run_id
    assert 'mlflow.parentRunId' not in top.data.tags
    assert client.get_experiment(top.info.experiment_id).name == 'scores'
    assert (top.info.run_name, top.info.status) == ('predict-20261021-0000',
                                                    'FINISHED')
    assert top.data.tags['mlflow.note.content'].startswith(
        'Scores of unlabelled rows')


def test_link_prediction_metric_names_are_logged(tracking_uri):
    from kumo.engine._tracking import RunRecord
    from kumo.engine.integrations.mlflow import MlflowTracking

    tracker = MlflowTracking(experiment_name='lp')
    tags = {'kumo.kind': 'fit'}
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


def test_system_metrics_skip_a_failed_log_and_stop_after_repeated_ones(
        monkeypatch, caplog):
    import logging

    from kumo.engine.integrations import mlflow as integration

    class Monitor:
        def collect_metrics(self):
            pass

        def aggregate_metrics(self):
            return {'cpu': 1.0}

        def clear_metrics(self):
            pass

    calls = []

    def log(metrics, step):
        calls.append(step)
        if step == 0 or step >= 2:
            raise ConnectionError('blip')

    monkeypatch.setattr(integration, '_metric_monitors', lambda: [Monitor()])
    caplog.set_level(logging.INFO, integration.logger.name)
    with integration._SystemMetrics(log, interval=0.001) as metrics:
        metrics._thread.join(10)

    assert calls == list(range(2 + integration._SYSTEM_METRICS_FAILURES))
    assert caplog.text.count('Could not log system metrics') == 1
    assert 'ConnectionError: blip' in caplog.text
    assert 'stopped after 6 failed logs in a row' in caplog.text


def test_link_prediction_monitoring_metrics_are_logged(tracking_uri):
    from kumo.engine._tracking import RunRecord
    from kumo.engine.integrations.mlflow import MlflowTracking

    tracker = MlflowTracking(experiment_name='lp')
    tracker.log_monitoring(
        RunRecord(
            name='m', tags={
                'kumo.kind': 'monitoring',
                'kumo.monitors': 'p'
            }, metrics={
                'map@10': 0.5,
                'precision@10': 0.25
            }))

    run = mlflow.get_run(tracker.run_id)
    assert run.info.status == 'FINISHED'
    assert run.data.metrics == {'map_at_10': 0.5, 'precision_at_10': 0.25}


@pytest.mark.parametrize('enabled', [True, False])
def test_prediction_run_logs_the_system_metrics_sampled_while_it_scored(
        tracking_uri, monkeypatch, enabled):
    from functools import partial

    from test_tracking_contract import PREDICTION_MANIFEST

    from kumo.engine._tracking import system_metrics, track_prediction
    from kumo.engine.integrations import mlflow as integration

    class Monitor:
        samples = 0

        def collect_metrics(self):
            self.samples += 1

        def aggregate_metrics(self):
            return {'cpu_utilization_percentage': float(self.samples)}

        def clear_metrics(self):
            pass

    monkeypatch.setattr(integration, '_metric_monitors', lambda: [Monitor()])
    monkeypatch.setattr(integration, '_SystemMetrics',
                        partial(integration._SystemMetrics, interval=0.01))
    tracking = integration.MlflowTracking(experiment_name='scores',
                                          log_system_metrics=enabled)
    with system_metrics(tracking):
        time.sleep(0.2)
    track_prediction(PREDICTION_MANIFEST, tracking)

    client = mlflow.MlflowClient()
    run, = _runs(client)
    history = [(point.step, point.value) for point in client.get_metric_history(
        run.info.run_id, 'system/cpu_utilization_percentage')]
    assert history == [(step, step + 1.0) for step in range(len(history))]
    assert bool(history) == enabled
