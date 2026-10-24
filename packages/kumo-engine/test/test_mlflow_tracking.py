# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from kumo_core.artifacts import PROVENANCE_FILE
from kumoapi.encoder import Numerical
from kumoapi.model_plan import ModelPlan
from test_tracking_contract import CONTEXT, join_tracking_threads, replay

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine._tracking import run_record

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
    children = sorted(
        (run for run in runs
         if run.data.tags.get('mlflow.parentRunId') == parent.info.run_id),
        key=lambda run: run.info.run_name)
    assert len(runs) == 4
    assert client.get_experiment(parent.info.experiment_id).name == 'ents'
    assert parent.info.status == 'FINISHED'
    assert {k for k in parent.data.tags if k.startswith('kumo.')} == {
        'kumo.kind', 'kumo.run_id', 'kumo.bundle_id', 'kumo.dropped_params'}
    assert parent.data.tags['kumo.bundle_id'] == result.id
    stored = tmp_path / 'root' / 'models' / result.id / PROVENANCE_FILE
    assert json.loads(stored.read_text())['tracking'] == {
        'run_id': parent.data.tags['kumo.run_id']}
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
    assert metrics == {**result.metrics, 'seconds': metrics['seconds']}
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
        assert {'best_val_mae', 'seconds',
                'val_mae'} <= set(child.data.metrics)
        history = client.get_metric_history(child.info.run_id, 'val_mae')
        assert sorted(metric.step for metric in history) == [1, 2]


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


def test_user_disabled_autologging_stays_disabled(tracking_uri):
    from kumo.engine.integrations.mlflow import MlflowTracking

    tracker = MlflowTracking(log_system_metrics=False)
    with autologging_utils.disable_autologging():
        tracker.start(run_record(CONTEXT))
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
    replay(MlflowTracking(log_system_metrics=False))

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
