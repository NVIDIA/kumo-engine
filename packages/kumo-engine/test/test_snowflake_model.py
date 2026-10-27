# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Kumo bundle as a runnable Snowflake Model Registry version, packaged
and loaded by snowflake-ml's own packager, without a Snowflake account."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import kumo.engine as kumo
from kumo.engine.integrations import _serving

pytest.importorskip('snowflake.ml.model')

TARGET = {'warehouse': 'WH', 'database': 'DB', 'schema': 'ML'}


@pytest.fixture
def served(fitted, tmp_path, monkeypatch):
    monkeypatch.setattr(_serving, '_trainers', {})
    monkeypatch.setattr(_serving, '_snapshots', {})
    trainer, table, snapshot = fitted
    exports = tmp_path / 'exports'
    location = snapshot.export(
        kumo.SnapshotOutputConfig(kumo.FileConnector(root_dir=str(exports)),
                                  'snapshots'))
    return trainer, table, snapshot, str(exports / location)


def _package(trainer, metadata, tmp_path):
    from snowflake.ml.model._packager.model_packager import ModelPackager

    from kumo.engine.integrations._snowflake_model import log_model_args

    workdir = tmp_path / 'work'
    workdir.mkdir(parents=True)
    args = log_model_args(str(trainer.result.bundle_dir), metadata, TARGET,
                          str(workdir))
    ModelPackager(str(tmp_path / 'version')).save(
        name='kumo', model=args['model'], signatures=args['signatures'],
        pip_requirements=args['pip_requirements'],
        target_platforms=args['target_platforms'],
        python_version=args['python_version'], options=args['options'])
    loaded = ModelPackager(str(tmp_path / 'version'))
    loaded.load()
    return args, loaded.model


def test_a_packaged_version_scores_like_the_trainer(served, tmp_path):
    trainer, table, snapshot, uri = served
    args, model = _package(trainer, {'kumo.serving_snapshot': uri}, tmp_path)

    import torch

    major, minor = (int(p) for p in torch.__version__.split('.')[:2])
    requirements = args['pip_requirements']
    assert f'torch>={major}.{minor},<{major}.{minor + 1}' in requirements
    assert any(r.startswith('faiss-cpu==') for r in requirements)
    assert not [r for r in requirements if r.startswith(('kumo', 'diskgraph',
                                                         'torchmetrics'))]
    assert not [r for r in requirements if '*' in r]
    code = [Path(path) for path in args['code_paths']]
    assert [path.name for path in code] == [
        'kumo', 'kumo_core', 'kumoapi', 'kumopql', 'kumo_snowflake',
        'diskgraph', 'torchmetrics'
    ]
    assert (code[0] / 'engine' / 'integrations' / '_serving.py').is_file()
    assert not list(code[1].rglob('__pycache__'))
    assert args['target_platforms'] == ['SNOWPARK_CONTAINER_SERVICES']
    assert args['options'] == {
        'cuda_version': '12.6',
        'function_type': 'TABLE_FUNCTION'
    }
    assert args['python_version'] == '%d.%d' % sys.version_info[:2]
    assert 'artifact_repository_map' not in args
    assert [spec.name for spec in args['signatures']['predict'].outputs
            ] == ['ENTITY', 'TARGET_PRED', 'False_PROB', 'True_PROB']

    time = str(pd.Timestamp(table.timestamps[0]))
    rows = pd.DataFrame({'ENTITY': [str(i) for i in range(10)]})
    scored = model.predict(rows, prediction_time=time,
                           binary_classification_threshold=0.5)
    expected = _serving.predict(trainer, snapshot,
                                pd.DataFrame({'ENTITY': range(10)}), {
                                    'prediction_time': time,
                                    'binary_classification_threshold': 0.5
                                }).assign(ENTITY=lambda f: f.ENTITY.astype(str))
    pd.testing.assert_frame_equal(
        scored.sort_values('ENTITY').reset_index(drop=True),
        expected.sort_values('ENTITY').reset_index(drop=True), rtol=1e-4)
    assert list(_serving._snapshots) == [uri]
    assert list(model.predict(rows.iloc[:0]).columns) == list(scored.columns)


def test_serving_options_override_the_defaults_and_a_snapshot_is_required(
        served, tmp_path):
    from kumo.engine.integrations._snowflake_model import pip_requirements

    trainer, *_ = served
    args, model = _package(
        trainer, {
            'kumo.serving.pip_repository': 'DB.ML.REPO',
            'kumo.serving.cuda_version': '12.4',
            'kumo.serving.function_type': 'FUNCTION',
        }, tmp_path)
    assert args['artifact_repository_map'] == {'pip': 'DB.ML.REPO'}
    assert args['pip_requirements'] == pip_requirements()
    assert len(args['code_paths']) == 7
    assert args['options'] == {
        'cuda_version': '12.4',
        'function_type': 'FUNCTION'
    }
    with pytest.raises(ValueError, match='serving_snapshot='):
        model.predict(pd.DataFrame({'ENTITY': ['1']}))
    with pytest.raises(ValueError, match='same Python'):
        _package(trainer, {'kumo.serving.python_version': '2.7'},
                 tmp_path / 'other')


def test_a_bundled_wheel_supplies_the_requirements(monkeypatch):
    import importlib.metadata as metadata

    from kumo.engine.integrations._snowflake_model import pip_requirements

    bundled = [
        'pyarrow>=16,<25', 'pyarrow>=8.0.0', 'sqlglot==30.17.*',
        'torchmetrics==1.8.1', 'kumo-api==3.0.0',
        'faiss-gpu==1.14.3; sys_platform == "linux"',
        'faiss-cpu==1.14.3; sys_platform == "darwin"',
        'boto3>=1.34; extra == "s3"'
    ]
    monkeypatch.setattr(metadata, 'packages_distributions', lambda: {
        'kumo_core': ['kumo-engine'],
        'kumopql': ['kumo-engine']
    })
    monkeypatch.setattr(metadata, 'requires',
                        lambda dist: bundled if dist == 'kumo-engine' else [])
    monkeypatch.setattr(metadata, 'version', lambda dist: '2.12.1+cu126')
    monkeypatch.setattr('sys.platform', 'linux')
    assert pip_requirements() == [
        'pyarrow<25,>=16,>=8.0.0', 'sqlglot<30.18,>=30.17',
        'faiss-cpu==1.14.3', 'torch>=2.12,<2.13',
        'snowflake-snowpark-python', 'snowflake-connector-python'
    ]


def test_a_model_with_text_columns_also_requires_the_embeddings(tmp_path):
    from kumo_core.encoders.config import EncodersConfig
    from kumo_core.io.json import save_json
    from kumoapi.encoder import Numerical, TextEmbedding

    from kumo.engine.integrations._snowflake_model import (
        TEXT_REQUIREMENT,
        text_requirements,
    )

    def required(encoders):
        path = tmp_path / 'encoders.json'
        save_json(EncodersConfig(encoders={'users': encoders}), str(path))
        return text_requirements(str(path))

    assert required({'age': Numerical()}) == []
    assert required({'bio': TextEmbedding('words')}) == [TEXT_REQUIREMENT]


def test_kumo_snowflake_finds_the_engine_model_builder():
    from kumo_snowflake.registry import _model_builder

    from kumo.engine.integrations._snowflake_model import log_model_args

    assert _model_builder() is log_model_args


def test_a_stage_snapshot_is_read_with_the_container_identity(monkeypatch):
    from kumo.engine.connector.snowflake import SnowflakeConnector

    calls = []
    monkeypatch.setattr(SnowflakeConnector, 'from_spcs_env',
                        classmethod(lambda cls, **kw: calls.append(kw) or 'sf'))
    monkeypatch.setattr(
        kumo.GraphSnapshot, 'load_from_connector',
        classmethod(lambda cls, connector, path: (connector, path)))
    assert _serving.load_snapshot('@DB.ML.KUMO/snapshots/s1',
                                  TARGET) == ('sf', '@DB.ML.KUMO/snapshots/s1')
    assert calls == [{
        'warehouse': 'WH',
        'database': 'DB',
        'schema_name': 'ML'
    }]


def test_publish_passes_the_serving_snapshot_and_repository(fitted):
    from kumo.engine.trainer.trainer import _publish

    trainer, *_ = fitted
    published = []
    sink = SimpleNamespace(
        publish_model=lambda connector, bundle, **kw: published.append(kw) or
        SimpleNamespace(version='V1'))
    connector = SimpleNamespace(sink_backend=sink)
    location = kumo.ModelRegistryLocation('CHURN')
    _publish(trainer.result,
             kumo.ModelOutputConfig(connector, location=location))
    _publish(
        trainer.result,
        kumo.ModelOutputConfig(connector, location=location,
                               serving_snapshot='@DB.ML.KUMO/snapshots/s1',
                               serving_options={
                                   'pip_repository': 'DB.ML.REPO',
                                   'cuda_version': '12.4',
                               }))
    with pytest.raises(ValueError, match=r"Unknown serving_options \['cuda'\]"):
        kumo.ModelOutputConfig(connector, location=location,
                               serving_options={'cuda': '12.6'})
    plain, served = (call['metadata'] for call in published)
    assert 'kumo.serving_snapshot' not in plain
    assert {k: v for k, v in served.items() if k not in plain} == {
        'kumo.serving_snapshot': '@DB.ML.KUMO/snapshots/s1',
        'kumo.serving.pip_repository': 'DB.ML.REPO',
        'kumo.serving.cuda_version': '12.4',
    }
