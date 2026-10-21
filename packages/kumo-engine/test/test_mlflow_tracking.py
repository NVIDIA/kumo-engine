# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pandas as pd
import pytest
from kumoapi.encoder import Numerical
from kumoapi.model_plan import ModelPlan
from test_tracking_contract import CONTEXT, replay

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

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


def test_fit_logs_a_run_per_fit_and_per_trial(table_and_snapshot, caplog):
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
            result = kumo.Trainer(plan).fit(
                table, snapshot=snapshot,
                tracking=MlflowTracking(experiment_name='ents'))
            assert mlflow.active_run().info.run_id == outer.info.run_id
        assert autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED is False
    finally:
        mlflow.pytorch.autolog(disable=True)

    runs = _runs(client)
    parent, = [
        run for run in runs
        if run.data.tags.get('mlflow.parentRunId') == outer.info.run_id
    ]
    children = sorted(
        (run for run in runs
         if run.data.tags.get('mlflow.parentRunId') == parent.info.run_id),
        key=lambda run: run.info.run_name)
    assert len(runs) == 4
    assert client.get_experiment(parent.info.experiment_id).name == 'ents'
    assert parent.info.status == 'FINISHED'
    assert parent.data.tags['kumo.snapshot_id'] == snapshot.id
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
    assert 'MLflow tracking failed' not in caplog.text

    assert [run.info.run_name
            for run in children] == ['trial-1-attempt-1', 'trial-2-attempt-1']
    for child in children:
        assert child.info.status == 'FINISHED'
        assert child.data.tags['kumo.status'] == 'succeeded'
        assert {'best_val_mae', 'seconds', 'val_mae',
                'lightning-epoch'} <= set(child.data.metrics)
        history = client.get_metric_history(child.info.run_id, 'val_mae')
        assert sorted(metric.step for metric in history) == [1, 2]


def test_failed_fit_marks_the_run_failed(table_and_snapshot, monkeypatch):
    import kumo_core.training.orchestration as orchestration

    from kumo.engine.integrations.mlflow import MlflowTracking

    def fail(*args, **kwargs):
        raise ValueError('stop')

    monkeypatch.setattr(orchestration, 'run_callable_training', fail)
    table, snapshot = table_and_snapshot

    with pytest.raises(ValueError, match='stop'):
        kumo.Trainer(ModelPlan()).fit(table, snapshot=snapshot,
                                      tracking=MlflowTracking())

    run, = _runs(mlflow.MlflowClient())
    assert run.info.status == 'FAILED'
    assert run.data.tags['kumo.error'] == 'ValueError: stop'
    assert mlflow.active_run() is None


@pytest.mark.parametrize('broken', ['log_batch', 'get_run'])
def test_mlflow_errors_are_logged_once_not_raised(tracking_uri, monkeypatch,
                                                  caplog, broken):
    from mlflow.tracking import MlflowClient

    from kumo.engine.integrations.mlflow import MlflowTracking

    original = getattr(MlflowClient, broken)

    def down(self, *args, metrics=(), **kwargs):
        if broken == 'get_run' or metrics:
            raise RuntimeError('tracking server down')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MlflowClient, broken, down)
    replay([MlflowTracking(log_system_metrics=False)])

    assert caplog.text.count('MLflow tracking failed') == 1
    assert mlflow.active_run() is None
    if broken == 'log_batch':
        parent, = [
            run for run in _runs(mlflow.MlflowClient())
            if 'mlflow.parentRunId' not in run.data.tags
        ]
        assert parent.info.status == 'FINISHED'


def test_user_disabled_autologging_stays_disabled(tracking_uri):
    from kumo.engine.integrations.mlflow import MlflowTracking

    tracker = MlflowTracking(log_system_metrics=False)
    with autologging_utils.disable_autologging():
        tracker.start(CONTEXT)
        tracker.finish(None, None)
        assert autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED is True
