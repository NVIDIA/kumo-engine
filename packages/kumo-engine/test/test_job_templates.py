# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
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


def test_databricks_tasks_pass_flags_to_the_job_scripts():
    yaml = pytest.importorskip('yaml')
    bundle = yaml.safe_load(
        (_examples() / 'databricks' / 'databricks.yml').read_text())
    tasks = [
        task['spark_python_task'] for job in bundle['resources']['jobs'].values()
        for task in job['tasks']
    ]
    assert [task['python_file'].rsplit('/', 1)[-1]
            for task in tasks] == ['fit_job.py', 'score_job.py']
    fit, score = (dict(zip(task['parameters'][::2], task['parameters'][1::2]))
                  for task in tasks)
    assert fit['--registry'] == score['--registry'] == 'mlflow'
    assert fit['--artifact-root'] == score['--artifact-root']
    assert '--predictions-table' in score


def test_snowflake_tasks_run_the_jobs_with_flags(monkeypatch):
    calls = []

    class Definition:
        @staticmethod
        def register(*args, **kwargs):
            return Definition()

        def to_sql(self, job_args, use_async):
            calls.append(job_args)
            return 'CALL JOB'

    class Session:
        def sql(self, text):
            calls.append(text)
            return self

        def collect(self):
            pass

    jobs_module = type(sys)('snowflake.ml.jobs')
    jobs_module.MLJobDefinition = Definition
    monkeypatch.setitem(sys.modules, 'snowflake.ml.jobs', jobs_module)
    tasks = _module('snowflake/register_jobs').register(
        Session(), compute_pool='P', stage='@S', database='DB', schema='SC',
        warehouse='WH', artifact_root='s3://bucket/kumo', requirements=[],
        external_access_integrations=[])
    assert tasks == {'train': 'DB.SC.KUMO_TRAIN_TASK',
                     'score': 'DB.SC.KUMO_SCORE_TASK'}
    fit_args, fit_sql, score_args, score_sql = calls
    assert fit_args[:2] == ['--artifact-root', 's3://bucket/kumo']
    assert '--predictions-table' in score_args
    assert '--predictions-table' not in fit_args
    assert fit_sql.startswith('CREATE OR REPLACE TASK DB.SC.KUMO_TRAIN_TASK')
    assert score_sql.endswith('AS\nCALL JOB')


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


def test_fit_then_score_locally(tmp_path, monkeypatch, capsys):
    _data(tmp_path / 'data')
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    monkeypatch.setattr(PredictiveQuery, 'suggest_model_plan',
                        lambda self, snapshot: plan)
    monkeypatch.syspath_prepend(str(_examples()))
    monkeypatch.delitem(sys.modules, 'jobs', raising=False)
    root = tmp_path / 'artifacts'
    flags = [
        '--source',
        str(tmp_path / 'data'), '--artifact-root',
        str(root), '--scratch-dir',
        str(tmp_path / 'scratch')
    ]

    fit = _module('fit_job').main([
        *flags, '--query', 'PREDICT COUNT(orders.*, 0, 7, days) > 0 '
        'FOR EACH users.USER_ID'
    ])
    assert capsys.readouterr().out.splitlines()[-1] == f'model: models/{fit.id}'
    assert (root / 'models' / fit.id / '_provenance.json').is_file()
    assert any((root / 'snapshots').iterdir())
    assert any((root / 'training_tables').iterdir())

    score = _module('score_job').main(
        [*flags, '--model-path', f'models/{fit.id}'])
    assert score.num_rows == 50


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
