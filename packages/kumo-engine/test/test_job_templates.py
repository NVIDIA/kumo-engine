# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import py_compile
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _data(root: Path, prefix: str = '') -> None:
    root.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame({'USER_ID': np.arange(50)
                  }).to_parquet(root / f'{prefix}users.parquet')
    pd.DataFrame({'ITEM_ID': np.arange(20)
                  }).to_parquet(root / f'{prefix}items.parquet')
    pd.DataFrame({
        'ORDER_ID': np.arange(3000),
        'USER_ID': rng.integers(0, 50, 3000),
        'ITEM_ID': rng.integers(0, 20, 3000),
        'ORDER_TIME': pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 3000), unit='D'),
    }).to_parquet(root / f'{prefix}orders.parquet')


def test_fit_then_score_locally(tmp_path, monkeypatch, capsys):
    _data(tmp_path / 'data', prefix='demo_')
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
        str(tmp_path / 'scratch'), '--table-prefix=demo_'
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
    stored = root / 'predictions' / score.id / 'predictions'
    assert any(stored.glob('*.parquet'))
    assert json.loads((stored / '_provenance.json').read_text())[
        'parents']['model']['id'] == fit.id


def test_a_scoring_step_scores_its_fit_after_a_later_fit_moved_the_alias(
        tmp_path, monkeypatch):
    import mlflow

    _data(tmp_path / 'data')
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    monkeypatch.setattr(PredictiveQuery, 'suggest_model_plan',
                        lambda self, snapshot: plan)
    monkeypatch.syspath_prepend(str(_examples()))
    monkeypatch.delitem(sys.modules, 'jobs', raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MLFLOW_TRACKING_URI',
                       f'sqlite:///{tmp_path / "mlflow.db"}')
    flags = [
        '--source', str(tmp_path / 'data'), '--artifact-root',
        str(tmp_path / 'artifacts'), '--scratch-dir',
        str(tmp_path / 'scratch'), '--registry', 'mlflow', '--model-name',
        'users-churn'
    ]
    fit_a = _module('fit_job').main(
        [*flags, '--model-path-file', str(tmp_path / 'a')])
    fit_b = _module('fit_job').main(
        [*flags, '--model-path-file', str(tmp_path / 'b'), '--threshold',
         '0.7'])
    assert mlflow.MlflowClient().get_model_version_by_alias(
        'users-churn', 'champion').version == int(
            fit_b.registered_model.version)

    score = _module('score_job').main(
        [*flags, '--model-path-file', str(tmp_path / 'a')])
    assert score.manifest.parents.model.id == fit_a.id


def test_the_shared_graph_declares_text_columns(tmp_path, monkeypatch):
    import kumo.engine as kumo

    data = tmp_path / 'data'
    _data(data)
    users = pd.read_parquet(data / 'users.parquet')
    users.assign(BIO='likes red shoes').to_parquet(data / 'users.parquet')
    jobs = _module('jobs')
    monkeypatch.setattr(jobs, 'TEXT_COLUMNS', ('users.BIO', ))
    graph = jobs.build_graph(kumo.FileConnector(root_dir=str(data)))
    [bio] = [column for column in graph.tables['users'].definition.cols
             if column.name == 'BIO']
    assert bio.stype == 'text'


def test_jobs_fail_on_tracking_errors_unless_told_to_warn(tmp_path):
    jobs = _module('jobs')
    flags = ['--artifact-root', str(tmp_path), '--source', str(tmp_path)]
    for extra, expected in (([], 'raise'), (['--on-error', 'warn'], 'warn')):
        args = jobs.parse_args([*flags, *extra], score=False)
        assert jobs.mlops(args, None).on_error == expected


@pytest.mark.parametrize('name', ['fit_job', 'score_job'])
def test_scripts_import_without_dunder_file(name, monkeypatch):
    path = _examples() / f'{name}.py'
    monkeypatch.setattr(sys, 'argv', [str(path)])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.delitem(sys.modules, 'jobs', raising=False)
    namespace = {'__name__': 'spark_python_task'}
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    assert callable(namespace['main'])


def test_sagemaker_steps_take_their_own_instance_types_and_the_alias(
        monkeypatch):
    made = {}

    def fake(name):
        class Fake:
            properties = SimpleNamespace(ModelArtifacts=SimpleNamespace(
                S3ModelArtifacts='s3://fit/model/'))

            def __init__(self, *args, **kwargs):
                made[name] = kwargs

            def train(self, **kwargs):
                return kwargs

            def run(self, **kwargs):
                return kwargs

            def upsert(self, **kwargs):
                pass

        return Fake

    for module, names in {
            'sagemaker.core.processing': ['ProcessingInput',
                                          'ProcessingS3Input', 'Processor'],
            'sagemaker.core.workflow.pipeline_context': ['PipelineSession'],
            'sagemaker.mlops.workflow.pipeline': ['Pipeline'],
            'sagemaker.mlops.workflow.steps': ['ProcessingStep',
                                               'TrainingStep'],
            'sagemaker.train.configs': ['Compute', 'OutputDataConfig',
                                        'SourceCode'],
            'sagemaker.train.model_trainer': ['ModelTrainer'],
    }.items():
        monkeypatch.setitem(sys.modules, module,
                            SimpleNamespace(**{n: fake(n)
                                               for n in names}))
    flags = [
        f'--{flag}=x' for flag in ('role', 'image', 'code', 'data-root',
                                   'tracking-arn')
    ]
    monkeypatch.delenv('SAGEMAKER_VALIDATE_ROLE', raising=False)
    _module('sagemaker/pipeline').main([
        *flags, '--artifact-root=s3://bucket/kumo/', '--fit-instance-type',
        'ml.g5.xlarge', '--score-instance-type', 'ml.m5.xlarge', '--alias',
        'candidate'
    ])
    assert made['Compute']['instance_type'] == 'ml.g5.xlarge'
    assert made['Processor']['instance_type'] == 'ml.m5.xlarge'
    assert made['PipelineSession'] == {
        'default_bucket': 'bucket', 'default_bucket_prefix': 'sagemaker'}
    assert made['OutputDataConfig'] == {
        's3_output_path': 's3://bucket/sagemaker/output',
        'compression_type': 'NONE'}
    assert made['ProcessingS3Input']['s3_uri'] == 's3://fit/model/'
    assert made['SourceCode']['command'] == 'python fit_job.py'
    assert '--query' not in made['SourceCode']['args']
    assert os.environ['SAGEMAKER_VALIDATE_ROLE'] == 'false'
    for arguments in (made['SourceCode']['args'],
                      made['ProcessingStep']['step_args']['arguments']):
        assert arguments[arguments.index('--alias') + 1] == 'candidate'
        assert arguments[arguments.index('--on-error') + 1] == 'raise'
        assert arguments[arguments.index('--model-name') + 1] == 'users-churn'
    assert made['SourceCode']['args'][-2:] == [
        '--model-path-file', '/opt/ml/model/model_path']
    assert made['ProcessingStep']['step_args']['arguments'][-2:] == [
        '--model-path-file', '/opt/ml/processing/input/model/model_path']


def test_snowflake_tasks_pass_names_and_strict_mode(monkeypatch):
    calls = []

    class Definition:
        @classmethod
        def register(cls, *args, **kwargs):
            return cls()

        def to_sql(self, job_args, use_async):
            calls.append(job_args)
            return 'CALL x()'

    monkeypatch.setitem(sys.modules, 'snowflake.ml.jobs',
                        SimpleNamespace(MLJobDefinition=Definition))
    session = SimpleNamespace(
        sql=lambda sql: SimpleNamespace(collect=lambda: None))
    _module('snowflake/register_jobs').register(
        session, compute_pool='p', stage='s', database='D', schema='S',
        warehouse='W', artifact_root='s3://b/k/', requirements=[],
        external_access_integrations=[], model_name='CHURN',
        predictions_table='CHURN_OUT', table_prefix='DEMO_')
    train, score = calls
    for arguments in (train, score):
        assert arguments[arguments.index('--model-name') + 1] == 'CHURN'
        assert arguments[arguments.index('--on-error') + 1] == 'raise'
        assert '--table-prefix=DEMO_' in arguments
    assert score[score.index('--predictions-table') + 1] == 'CHURN_OUT'
