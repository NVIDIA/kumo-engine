# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import py_compile
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from kumoapi.model_plan import ModelPlan

from kumo.engine.pquery import PredictiveQuery


def _examples() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'examples' / 'mlops'
        if (candidate / 'jobs.py').is_file():
            return candidate
    pytest.skip('examples/mlops is not in this checkout')


def _module(name: str):
    spec = importlib.util.spec_from_file_location(
        f'template_{name}', _examples() / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', [
    'jobs.py', 'fit_job.py', 'score_job.py', 'sagemaker/pipeline.py',
    'snowflake/register_jobs.py'
])
def test_python_compiles(name, tmp_path):
    py_compile.compile(str(_examples() / name), cfile=str(tmp_path / 'out'),
                       doraise=True)


def test_databricks_tasks_run_the_job_scripts():
    yaml = pytest.importorskip('yaml')
    bundle = yaml.safe_load(
        (_examples() / 'databricks' / 'databricks.yml').read_text())
    root = '${workspace.file_path}'
    tasks = [
        task['spark_python_task'] for job in bundle['resources']['jobs'].values()
        for task in job['tasks']
    ]
    assert tasks == [{
        'python_file': f'{root}/{script}',
        'parameters': [
            '--job-id', '{{job.id}}', '--job-run-id', '{{job.run_id}}'
        ]
    } for script in ('fit_job.py', 'score_job.py')]


def test_sagemaker_and_snowflake_name_the_job_scripts():
    pipeline = (_examples() / 'sagemaker' / 'pipeline.py').read_text()
    register = (_examples() / 'snowflake' / 'register_jobs.py').read_text()
    for script in ('fit_job.py', 'score_job.py'):
        assert (_examples() / script).is_file()
        assert f"/{script}'" in pipeline
        assert f"'entrypoint': '{script}'" in register


def _data(root: Path) -> None:
    root.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame({'USER_ID': np.arange(50)}).to_parquet(root / 'users.parquet')
    pd.DataFrame({'ITEM_ID': np.arange(20)}).to_parquet(root / 'items.parquet')
    pd.DataFrame({
        'ORDER_ID': np.arange(3000),
        'USER_ID': rng.integers(0, 50, 3000),
        'ITEM_ID': rng.integers(0, 20, 3000),
        'ORDER_TIME': pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 3000), unit='D'),
    }).to_parquet(root / 'orders.parquet')


def _summary(capsys):
    lines = capsys.readouterr().out.splitlines()
    return json.loads(lines[-1])


@pytest.mark.parametrize('registry', [None, 'mlflow'])
def test_fit_then_score_locally(registry, tmp_path, monkeypatch, capsys):
    if registry:
        mlflow = pytest.importorskip(registry)
        monkeypatch.setenv('MLFLOW_TRACKING_URI',
                           f'sqlite:///{tmp_path}/mlflow.db')
        monkeypatch.setenv('KUMO_REGISTRY', registry)
        monkeypatch.setenv('KUMO_MODEL_NAME', 'churn')
    _data(tmp_path / 'data')
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    monkeypatch.setattr(PredictiveQuery, 'suggest_model_plan',
                        lambda self, snapshot: plan)
    for name, value in (('KUMO_SOURCE', 'file'),
                        ('KUMO_DATA_ROOT', str(tmp_path / 'data')),
                        ('KUMO_ARTIFACT_ROOT', str(tmp_path / 'artifacts')),
                        ('KUMO_SCRATCH_DIR', str(tmp_path / 'scratch')),
                        ('KUMO_QUERY', 'PREDICT COUNT(orders.*, 0, 7, days) > 0 '
                         'FOR EACH users.USER_ID')):
        monkeypatch.setenv(name, value)
    for name in ('KUMO_PREDICTIONS_TABLE', 'KUMO_SNAPSHOT',
                 'KUMO_MODEL_PATH',
                 'KUMO_SCHEDULER', 'KUMO_JOB_ID', 'KUMO_JOB_RUN_ID'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.syspath_prepend(str(_examples()))
    monkeypatch.delitem(sys.modules, 'jobs', raising=False)

    _module('fit_job').main(['--job-id', 'local', '--job-run-id', 'run-1'])
    fit = _summary(capsys)
    assert fit['status'] == 'succeeded', fit
    assert fit['scheduler']['job_run_id'] == 'run-1'
    assert fit['outputs']['model']['uri']
    assert (fit['outputs']['model']['version'] is None) == (registry is None)

    monkeypatch.setenv('KUMO_MODEL_PATH', 'runs/run-1/model')
    _module('score_job').main(['--job-id', 'local', '--job-run-id', 'run-2'])
    score = _summary(capsys)
    assert score['status'] == 'succeeded', score
    assert (score['bundle_id'], score['snapshot_id']) != (None, None)
    assert score['bundle_id'] == fit['bundle_id']
    assert score['outputs']['predictions'] == str(
        tmp_path / 'artifacts' / 'runs' / 'run-2' / 'predictions') + '/'
    predictions = pd.read_parquet(score['outputs']['predictions'])
    assert len(predictions) == 50
    assert 'ANCHOR_TIMESTAMP' in predictions
    for column, value in (('KUMO_RUN_ID', score['run_id']),
                          ('KUMO_BUNDLE_ID', score['bundle_id']),
                          ('KUMO_SNAPSHOT_ID', score['snapshot_id'])):
        assert set(predictions[column]) == {value}
    if registry:
        runs = mlflow.search_runs(
            search_all_experiments=True,
            filter_string="tags.`kumo.kind` = 'prediction'")
        assert list(runs['tags.kumo.run_id']) == [score['run_id']]


def test_sagemaker_pipeline_needs_an_instance_type(capsys):
    pipeline = _module('sagemaker/pipeline')
    args = [
        '--role', 'r', '--image', 'i', '--code', 'c', '--data-root', 'd',
        '--artifact-root', 'a', '--tracking-arn', 't'
    ]
    with pytest.raises(SystemExit):
        pipeline.main(args)
    assert '--instance-type' in capsys.readouterr().err


@pytest.mark.parametrize('name', ['fit_job', 'score_job'])
def test_scripts_import_without_dunder_file(name, monkeypatch):
    path = _examples() / f'{name}.py'
    monkeypatch.setattr(sys, 'argv', [str(path)])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.delitem(sys.modules, 'jobs', raising=False)
    namespace = {'__name__': 'spark_python_task'}
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    assert callable(namespace['main'])


def test_snowflake_registry_scores_with_a_snowflake_tracking(monkeypatch):
    from types import SimpleNamespace

    from _snowflake_experiment_fake import FakeSession, install

    from kumo.engine.integrations.snowflake import SnowflakeTracking

    install(monkeypatch)
    monkeypatch.setenv('KUMO_REGISTRY', 'snowflake')
    monkeypatch.setenv('KUMO_MODEL_NAME', 'churn')
    monkeypatch.delenv('KUMO_SNAPSHOT', raising=False)
    monkeypatch.syspath_prepend(str(_examples()))
    monkeypatch.delitem(sys.modules, 'jobs', raising=False)
    score_job = _module('score_job')
    jobs = sys.modules['jobs']

    session = FakeSession()
    connector = object.__new__(jobs.SnowflakeConnector)
    monkeypatch.setattr(jobs.SnowflakeConnector, 'session',
                        lambda self: session, raising=False)
    seen = {}

    def predict_snapshot(snapshot, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(selection=SimpleNamespace(job_id='run-2'))

    trainer = SimpleNamespace(
        bundle_id='b', predict_snapshot=predict_snapshot,
        scoring_defaults=SimpleNamespace(
            binary_classification_threshold=None))
    graph = SimpleNamespace(snapshot=lambda: SimpleNamespace(id='s'))
    monkeypatch.setattr(jobs, 'source', lambda: connector)
    monkeypatch.setattr(jobs, 'load_champion', lambda c: trainer)
    monkeypatch.setattr(jobs, 'build_graph', lambda c: graph)
    monkeypatch.setattr(jobs, 'prediction_output', lambda c: 'output')
    monkeypatch.setattr(jobs, 'prediction_destinations', lambda o: {})

    score_job.main(['--job-id', 'local', '--job-run-id', 'run-2'])

    assert isinstance(seen['tracking'], SnowflakeTracking)
    assert seen['output_config'] == 'output'
