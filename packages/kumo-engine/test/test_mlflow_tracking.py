# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from kumoapi.encoder import Numerical
from kumoapi.model_plan import ModelPlan

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

mlflow = pytest.importorskip('mlflow')

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


def _plan():
    plan = ModelPlan()
    plan.training_job.num_experiments = 2
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 2
    plan.optimization.max_steps_per_epoch = 2
    plan.optimization.base_lr = [0.01, 0.001]
    return plan


def _bundle_files(result):
    return sorted(
        str(path.relative_to(result.bundle_dir))
        for path in result.bundle_dir.rglob('*'))


def _runs(client):
    return client.search_runs(
        [e.experiment_id for e in client.search_experiments()])


def test_fit_logs_a_run_per_fit_and_per_trial(table_and_snapshot, caplog):
    from mlflow.utils import autologging_utils

    from kumo.engine.integrations.mlflow import MlflowTracking

    table, snapshot = table_and_snapshot
    client = mlflow.MlflowClient()
    plan = _plan()
    plan.column_processing.encoder_overrides = {
        'ents.amount ($)': Numerical('standard')
    }
    mlflow.pytorch.autolog()
    try:
        result = kumo.Trainer(plan).fit(
            table, snapshot=snapshot,
            tracking=MlflowTracking(experiment_name='ents', run_name='fit',
                                    tags={'team': 'a'}))
        assert autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED is False
    finally:
        mlflow.pytorch.autolog(disable=True)

    runs = _runs(client)
    parent, = [run for run in runs if 'mlflow.parentRunId' not in run.data.tags]
    children = sorted((run for run in runs if run is not parent),
                      key=lambda run: run.info.run_name)
    assert client.get_experiment(parent.info.experiment_id).name == 'ents'
    assert parent.info.run_name == 'fit'
    assert parent.info.status == 'FINISHED'
    assert parent.data.tags['team'] == 'a'
    assert parent.data.tags['kumo.snapshot_id'] == snapshot.id
    assert parent.data.tags['kumo.engine_version'] == kumo.__version__
    assert int(parent.data.tags['kumo.dropped_params']) >= 1
    assert parent.data.tags['kumo.kind'] == 'fit'
    assert parent.data.tags['kumo.query_sha256'] == hashlib.sha256(
        QUERY.encode()).hexdigest()
    params = parent.data.params
    assert params['task_type'] == 'regression'
    assert 'snapshot_id' not in params
    assert json.loads(params['optimization.base_lr']) == [0.01, 0.001]
    metrics = {
        name: value
        for name, value in parent.data.metrics.items()
        if not name.startswith('system/')
    }
    assert metrics == {**result.metrics, 'seconds': metrics['seconds']}
    artifacts = {a.path for a in client.list_artifacts(parent.info.run_id)}
    assert {'query.pql', 'model_plan.json'} <= artifacts
    plan = mlflow.artifacts.load_dict(
        f'{parent.info.artifact_uri}/model_plan.json')
    assert plan['optimization']['base_lr'] == [0.01, 0.001]
    # Keys MLflow would reject (column names) stay in the JSON only.
    assert 'amount ($)' in json.dumps(plan)
    assert not any('$' in name for name in params)
    assert 'MLflow tracking failed' not in caplog.text

    assert [run.info.run_name
            for run in children] == ['trial-1-attempt-1', 'trial-2-attempt-1']
    for child in children:
        assert child.info.status == 'FINISHED'
        assert child.data.tags['kumo.kind'] == 'trial'
        assert child.data.tags['kumo.status'] == 'succeeded'
        assert child.data.tags['mlflow.parentRunId'] == parent.info.run_id
        assert child.data.tags['kumo.parent_run'] == (
            parent.data.tags['kumo.run_id'])
        assert child.data.tags['kumo.run_id'] == (
            f"{parent.data.tags['kumo.run_id']}-{child.info.run_name}")
        assert child.data.params['attempt'] == '1'
        assert float(child.data.params['optimization.base_lr']) in (0.01, 0.001)
        assert {'best_val_mae', 'seconds', 'val_mae',
                'lightning-epoch'} <= set(child.data.metrics)
        history = client.get_metric_history(child.info.run_id, 'val_mae')
        assert sorted(metric.step for metric in history) == [1, 2]
        epochs = client.get_metric_history(child.info.run_id,
                                           'lightning-epoch')
        assert {0, 1} <= {metric.value for metric in epochs}


def test_fit_nests_under_the_active_run_beside_another_tracker(
        table_and_snapshot, tmp_path):
    from mlflow.utils import autologging_utils
    from pytorch_lightning.loggers import CSVLogger

    from kumo.engine.integrations.mlflow import MlflowTracking

    class CsvTracking:
        def __init__(self):
            self.events = []

        def start(self, context):
            pass

        def on_event(self, event):
            self.events.append(event)

        def finish(self, result, error):
            pass

        def trial_logger(self, trial, attempt):
            return CSVLogger(str(tmp_path / 'csv'), name=f'trial-{trial}')

    table, snapshot = table_and_snapshot
    client = mlflow.MlflowClient()
    plan = _plan()
    plan.training_job.num_experiments = 1
    csv = CsvTracking()
    with mlflow.start_run() as outer, \
            autologging_utils.disable_autologging():
        kumo.Trainer(plan).fit(
            table, snapshot=snapshot,
            tracking=[MlflowTracking(log_system_metrics=False), csv])
        assert mlflow.active_run().info.run_id == outer.info.run_id
        assert autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED is True

    runs = {run.info.run_id: run for run in _runs(client)}
    parent, = [
        run for run in runs.values()
        if run.data.tags.get('mlflow.parentRunId') == outer.info.run_id
    ]
    child, = [
        run for run in runs.values()
        if run.data.tags.get('mlflow.parentRunId') == parent.info.run_id
    ]
    assert len(runs) == 3
    assert parent.info.status == child.info.status == 'FINISHED'
    assert 'val_mae' in client.get_run(child.info.run_id).data.metrics
    assert list((tmp_path / 'csv' / 'trial-1').rglob('metrics.csv'))
    assert [
        event.params['attempt'] for event in csv.events
        if event.name == 'epoch_ended'
    ] == [1, 1]


def test_failed_fit_marks_the_run_failed(table_and_snapshot, monkeypatch):
    import kumo_core.training.orchestration as orchestration

    from kumo.engine.integrations.mlflow import MlflowTracking

    error = ValueError('stop')

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(orchestration, 'run_callable_training', fail)
    table, snapshot = table_and_snapshot

    with pytest.raises(ValueError) as raised:
        kumo.Trainer(_plan()).fit(table, snapshot=snapshot,
                                  tracking=MlflowTracking())

    assert raised.value is error
    run, = _runs(mlflow.MlflowClient())
    assert run.info.status == 'FAILED'
    assert run.data.tags['kumo.error'] == 'ValueError: stop'
    assert mlflow.active_run() is None


def test_mlflow_errors_do_not_change_the_fit(table_and_snapshot, monkeypatch,
                                             caplog):
    from mlflow.tracking import MlflowClient

    import kumo.engine.trainer._runtime as runtime_module
    from kumo.engine.integrations.mlflow import MlflowTracking

    table, snapshot = table_and_snapshot
    bind = runtime_module.local_training_bindings
    hooks = []

    def spy(*args, trial_logger=None, **kwargs):
        hooks.append(trial_logger)
        return bind(*args, trial_logger=trial_logger, **kwargs)

    monkeypatch.setattr(runtime_module, 'local_training_bindings', spy)
    baseline = kumo.Trainer(_plan()).fit(table, snapshot=snapshot)

    assert hooks == [None]
    assert list(baseline.bundle_dir.parent.rglob('tensorboard'))

    log_batch = MlflowClient.log_batch

    def broken(self, run_id, metrics=(), **kwargs):
        if metrics:
            raise RuntimeError('tracking server down')
        return log_batch(self, run_id, metrics=metrics, **kwargs)

    monkeypatch.setattr(MlflowClient, 'log_batch', broken)
    result = kumo.Trainer(_plan()).fit(
        table, snapshot=snapshot,
        tracking=MlflowTracking(log_system_metrics=False))

    # Fits are not seeded, so values differ run to run; the shape must not.
    assert set(result.metrics) == set(baseline.metrics)
    assert _bundle_files(result) == _bundle_files(baseline)
    assert caplog.text.count('MLflow tracking failed') == 1
    assert 'tracking server down' in caplog.text
    parent, = [
        run for run in _runs(mlflow.MlflowClient())
        if 'mlflow.parentRunId' not in run.data.tags
    ]
    assert parent.info.status == 'FINISHED'
    assert mlflow.active_run() is None


def test_lazy_lightning_lookups_failing_do_not_fail_the_fit(
        table_and_snapshot, monkeypatch, caplog):
    from mlflow.tracking import MlflowClient

    from kumo.engine.integrations.mlflow import MlflowTracking

    def down(self, run_id):
        raise RuntimeError('tracking server down')

    monkeypatch.setattr(MlflowClient, 'get_run', down)
    table, snapshot = table_and_snapshot
    plan = _plan()
    plan.training_job.num_experiments = 1

    result = kumo.Trainer(plan).fit(
        table, snapshot=snapshot,
        tracking=MlflowTracking(log_system_metrics=False))

    assert 'val_mae' in result.metrics
    assert caplog.text.count('MLflow tracking failed') == 1
    assert 'Trial failed' not in caplog.text
    assert mlflow.active_run() is None
