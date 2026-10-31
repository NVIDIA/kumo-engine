# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`Trainer.fit(export_config=...)` / `Trainer.load()` against a fake
model-registry sink -- exercises the `training-spec.json` bundle marker
`kumo.engine.trainer.trainer` writes/reads, without running real GNN training.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pandas as pd
import pytest
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumoapi.plugin import ModelRegistryLocation, PublishedModel

from kumo.engine.graph import GraphSnapshot
from kumo.engine.pquery.prediction_table import PredictionTable
from kumo.engine.trainer._spec import (
    MANIFEST_FILE,
    SPEC_FILE,
    write_training_spec,
)
from kumo.engine.trainer.output_config import ModelOutputConfig
from kumo.engine.trainer.trainer import Trainer, TrainingResult


def _fake_published_bundle(root: Path) -> Path:
    r"""A bundle shaped like `kumo_core.artifacts.VitalArtifactPublicationPolicy`
    publishes: a `publication-manifest.json` naming a few artifact files.
    """
    bundle = root / 'published-model'
    (bundle / 'model').mkdir(parents=True)
    (bundle / 'model' / 'config.json').write_text('{}')
    (bundle / 'metrics.json').write_text(json.dumps({'auc': 0.9}))
    manifest = {
        'schema_version':
        1,
        'policy':
        'vital-training-artifacts-v1',
        'excluded_by_default': [],
        'artifacts': [
            {
                'role': 'trainer_config',
                'relative_path': 'model/config.json',
                'size_bytes': 2,
                'sha256': hashlib.sha256(b'{}').hexdigest(),
            },
            {
                'role':
                'metrics',
                'relative_path':
                'metrics.json',
                'size_bytes': (bundle / 'metrics.json').stat().st_size,
                'sha256':
                hashlib.sha256(
                    (bundle / 'metrics.json').read_bytes()).hexdigest(),
            },
        ],
    }
    (bundle / MANIFEST_FILE).write_text(json.dumps(manifest))
    return bundle


class _FakeSink:
    r"""A minimal `ModelPublishingSink`/`ModelFetchingSink` stand-in: publish
    copies the bundle into an in-memory dict, fetch copies it back out.
    """
    def __init__(self) -> None:
        self.published: dict = {}

    def write_table(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def publish_model(
        self,
        connector,
        bundle_dir: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        metadata=None,
        metrics=None,
        project_id=None,
    ) -> PublishedModel:
        assert location is not None
        key = f'{location.model_name}@{location.version or "latest"}'
        self.published[key] = bundle_dir
        return PublishedModel(name=location.model_name, version='V1',
                              uri=f'fake://{key}')

    def fetch_model(
        self,
        connector,
        local_dir: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        project_id=None,
    ) -> PublishedModel:
        assert location is not None
        key = f'{location.model_name}@{location.version or "latest"}'
        shutil.copytree(self.published[key], local_dir, dirs_exist_ok=True)
        return PublishedModel(name=location.model_name, version='V1',
                              uri=f'fake://{key}')


class _FakeConnector:
    def __init__(self, sink: _FakeSink) -> None:
        self.sink_backend = sink


def _generation_plan() -> TrainingTableGenerationPlan:
    return TrainingTableGenerationPlan(
        lag_timesteps=3,
        year_over_year=True,
        timeframe_step=2,
    )


def test_write_then_read_training_spec_round_trips(tmp_path: Path) -> None:
    bundle = _fake_published_bundle(tmp_path)
    model_plan = ModelPlan()

    write_training_spec(bundle, predictive_query='PREDICT COUNT(*)',
                        model_plan=model_plan, snapshot_id='snap-123',
                        training_table_generation=_generation_plan())

    assert (bundle / 'training-spec.json').is_file()
    manifest = json.loads((bundle / MANIFEST_FILE).read_text())
    roles = {entry['role'] for entry in manifest['artifacts']}
    assert roles == {'trainer_config', 'metrics', 'training_spec'}


def test_training_spec_manifest_update_is_atomic_and_retryable(
        tmp_path: Path, monkeypatch) -> None:
    import kumo_core.io.filesystem as fs_module

    bundle = _fake_published_bundle(tmp_path)
    original_manifest = (bundle / MANIFEST_FILE).read_bytes()
    real_replace = fs_module.os.replace
    failed = False

    def fail_first_manifest_replace(source, destination):
        nonlocal failed
        if Path(destination) == bundle / MANIFEST_FILE and not failed:
            failed = True
            raise OSError('simulated interrupted manifest commit')
        return real_replace(source, destination)

    monkeypatch.setattr(fs_module.os, 'replace', fail_first_manifest_replace)
    with pytest.raises(OSError, match='interrupted manifest commit'):
        write_training_spec(bundle, predictive_query='PREDICT COUNT(*)',
                            model_plan=ModelPlan(), snapshot_id='snap-123',
                            training_table_generation=_generation_plan())

    # Each destination is replaced as a complete file. A failed manifest
    # commit leaves the old valid manifest and no abandoned temporary file.
    assert (bundle / MANIFEST_FILE).read_bytes() == original_manifest
    assert json.loads(
        (bundle / SPEC_FILE).read_text())['snapshot_id'] == 'snap-123'
    assert not list(bundle.glob(f'.{MANIFEST_FILE}.*'))
    with pytest.raises(ValueError, match='integrity-covered'):
        Trainer.load(bundle)

    write_training_spec(bundle, predictive_query='PREDICT COUNT(*)',
                        model_plan=ModelPlan(), snapshot_id='snap-123',
                        training_table_generation=_generation_plan())
    manifest = json.loads((bundle / MANIFEST_FILE).read_text())
    entries = [
        entry for entry in manifest['artifacts']
        if entry['relative_path'] == SPEC_FILE
    ]
    assert len(entries) == 1
    assert entries[0]['size_bytes'] == (bundle / SPEC_FILE).stat().st_size
    assert entries[0]['sha256'] == hashlib.sha256(
        (bundle / SPEC_FILE).read_bytes()).hexdigest()
    assert Trainer.load(bundle)._snapshot_id == 'snap-123'


def test_load_rejects_unlisted_training_spec(tmp_path: Path) -> None:
    bundle = _fake_published_bundle(tmp_path)
    write_training_spec(bundle, predictive_query='PREDICT COUNT(*)',
                        model_plan=ModelPlan(), snapshot_id='snap-123',
                        training_table_generation=_generation_plan())
    manifest_path = bundle / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest['artifacts'] = [
        entry for entry in manifest['artifacts']
        if entry['role'] != 'training_spec'
    ]
    manifest_path.write_text(json.dumps(manifest))

    # The shared verifier accepts an unlisted legacy spec. Local trainer
    # metadata is mandatory now, so loading must require its declared digest.
    with pytest.raises(ValueError, match='integrity-covered'):
        Trainer.load(bundle)


def test_finish_fit_persists_spec_without_export_before_exposing_state(
        tmp_path: Path) -> None:
    bundle = _fake_published_bundle(tmp_path)
    trainer = Trainer(ModelPlan())
    result = TrainingResult({'auc': .9}, {}, bundle)

    finished = trainer._finish_fit(
        result, predictive_query='PREDICT COUNT(*)', snapshot_id='snap-123',
        training_table_generation=_generation_plan(), export_config=None)

    assert (bundle / SPEC_FILE).is_file()
    assert finished.bundle_id == Trainer.load(bundle).result.id
    assert trainer._training_result is finished
    assert trainer._snapshot_id == 'snap-123'
    assert trainer._training_query == 'PREDICT COUNT(*)'
    assert result.published_model is None


@pytest.mark.parametrize('path,location', [
    ('models/churn/v1', None),
    (None, ModelRegistryLocation('CHURN_GNN')),
])
def test_finish_fit_forwards_publication_details_and_returns_assigned_version(
        tmp_path, path, location):
    bundle = _fake_published_bundle(tmp_path)
    trainer = Trainer(ModelPlan())
    original = TrainingResult({'val_auroc': .9, 'test_auroc': .8}, {}, bundle)
    published = PublishedModel(name='CHURN_GNN', version='assigned-v7',
                               uri='provider://models/CHURN_GNN/assigned-v7')
    calls = []

    class Sink:
        def publish_model(self, connector, bundle_dir, **kwargs):
            spec = json.loads((Path(bundle_dir) / SPEC_FILE).read_text())
            assert {
                key: value
                for key, value in kwargs['metadata'].items()
                if not key.startswith('kumo_')
            } == {
                name: str(spec[name])
                for name in ('spec_version', 'predictive_query', 'snapshot_id')
            }
            calls.append((connector, bundle_dir, kwargs))
            return published

    connector = SimpleNamespace(sink_backend=Sink())
    result = trainer._finish_fit(
        original, predictive_query='PREDICT COUNT(*)', snapshot_id='snap-123',
        training_table_generation=_generation_plan(),
        export_config=SimpleNamespace(connector=connector, path=path,
                                      location=location))

    assert len(calls) == 1
    assert calls[0][0] is connector
    assert calls[0][1] == str(bundle)
    assert calls[0][2]['path'] == path
    assert calls[0][2]['location'] == location
    assert calls[0][2]['metrics'] == {'test/auroc': .8}
    assert calls[0][2]['metadata']['kumo_bundle_id'] == result.bundle_id
    assert result.published_model is published
    assert result.metrics == original.metrics
    assert result.bundle_dir == bundle
    assert trainer._training_result is result
    assert original.published_model is None


@pytest.mark.parametrize('failure', ['metadata', 'export'])
def test_finish_fit_failure_does_not_expose_state(tmp_path: Path, monkeypatch,
                                                  failure: str) -> None:
    import kumo.engine.trainer.trainer as trainer_module

    bundle = _fake_published_bundle(tmp_path)
    trainer = Trainer(ModelPlan())
    result = TrainingResult({}, {}, bundle)
    export_config = None
    if failure == 'metadata':
        monkeypatch.setattr(
            trainer_module, 'write_training_spec', lambda *args, **kwargs:
            (_ for _ in ()).throw(OSError('disk full')))
    else:

        class FailingSink:
            def publish_model(self, *args, **kwargs):
                assert (bundle / SPEC_FILE).is_file()
                raise OSError('registry unavailable')

        export_config = SimpleNamespace(
            connector=SimpleNamespace(sink_backend=FailingSink()), path='model',
            location=None)

    with pytest.raises(OSError):
        trainer._finish_fit(result, predictive_query='PREDICT COUNT(*)',
                            snapshot_id='snap-123',
                            training_table_generation=_generation_plan(),
                            export_config=export_config)
    assert trainer._training_result is None
    assert trainer._snapshot_id is None
    assert trainer._training_query is None


@pytest.mark.parametrize('accelerator', ['cpu', 'gpu', 'mps', 'auto'])
def test_load_reconstructs_trainer_from_published_bundle(
        tmp_path: Path, accelerator: str) -> None:
    bundle = _fake_published_bundle(tmp_path)
    model_plan = ModelPlan()
    write_training_spec(bundle, predictive_query='PREDICT COUNT(*)',
                        model_plan=model_plan, snapshot_id='snap-123',
                        training_table_generation=_generation_plan())

    loaded = Trainer.load(bundle, accelerator=accelerator)
    assert loaded._accelerator == accelerator

    assert loaded._snapshot_id == 'snap-123'
    assert loaded._training_result is not None
    assert loaded._training_result.metrics == {'auc': 0.9}
    assert 'trainer_config' in loaded._training_result.artifact_paths
    assert Path(
        loaded._training_result.artifact_paths['trainer_config']).is_file()
    assert 'training_spec' not in loaded._training_result.artifact_paths
    assert loaded._training_result.bundle_dir == bundle.resolve()
    assert loaded.training_table_generation_plan == _generation_plan()


@pytest.mark.parametrize('accelerator', ['cpu', 'gpu', 'mps', 'auto'])
def test_load_from_connector_fetches_then_uses_local_loader(
        tmp_path: Path, accelerator: str) -> None:
    bundle = _fake_published_bundle(tmp_path / 'source')
    write_training_spec(bundle, predictive_query='PREDICT COUNT(*)',
                        model_plan=ModelPlan(), snapshot_id='snap-123',
                        training_table_generation=_generation_plan())
    sink = _FakeSink()
    location = ModelRegistryLocation(model_name='CHURN_GNN')
    sink.publish_model(None, str(bundle), location=location)

    loaded = Trainer.load_from_connector(_FakeConnector(sink),
                                         location=location,
                                         accelerator=accelerator)
    assert loaded._accelerator == accelerator

    assert loaded._snapshot_id == 'snap-123'
    assert loaded._training_result is not None
    assert loaded._training_result.bundle_dir != bundle.resolve()
    assert loaded._training_result.bundle_dir.name == 'fetched-model'
    assert loaded.result.registered_model == PublishedModel(
        name='CHURN_GNN', version='V1', uri='fake://CHURN_GNN@latest')
    assert loaded.result.registered_alias is None


def test_local_bundle_reloads_in_a_fresh_process(tmp_path: Path) -> None:
    bundle = _fake_published_bundle(tmp_path)
    write_training_spec(bundle, predictive_query='PREDICT COUNT(*)',
                        model_plan=ModelPlan(), snapshot_id='snap-123',
                        training_table_generation=_generation_plan())
    code = """
import json
import sys
from kumo.engine.trainer.trainer import Trainer
trainer = Trainer.load(sys.argv[1])
print(json.dumps({
    'snapshot_id': trainer._snapshot_id,
    'query': trainer._training_query,
    'metrics': trainer._training_result.metrics,
    'bundle_dir': str(trainer._training_result.bundle_dir),
}))
"""
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(str(item) for item in sys.path)
    completed = subprocess.run(
        [sys.executable, '-c', code, str(bundle)], check=True,
        capture_output=True, text=True, env=env)
    restored = json.loads(completed.stdout)
    assert restored == {
        'snapshot_id': 'snap-123',
        'query': 'PREDICT COUNT(*)',
        'metrics': {
            'auc': .9
        },
        'bundle_dir': str(bundle.resolve()),
    }


def test_connector_load_requires_exactly_one_of_path_or_location() -> None:
    connector = _FakeConnector(_FakeSink())
    with pytest.raises(ValueError, match='Exactly one'):
        Trainer.load_from_connector(connector)
    with pytest.raises(ValueError, match='Exactly one'):
        Trainer.load_from_connector(
            connector, path='s3://bucket/model',
            location=ModelRegistryLocation(model_name='m'))


def test_load_refuses_a_connector_without_a_fetching_sink() -> None:
    class _WriteOnlySink:
        def write_table(self, *args, **kwargs):
            raise NotImplementedError

    connector = _FakeConnector(_WriteOnlySink())
    with pytest.raises(TypeError, match='ModelFetchingSink'):
        Trainer.load_from_connector(
            connector, location=ModelRegistryLocation(model_name='m'))


@pytest.mark.parametrize('defect', [
    'checkpoint_contents',
    'declared_size',
    'training_spec_contents',
    'parent_path',
    'absolute_path',
    'artifact_symlink',
    'spec_symlink',
])
def test_load_verifies_bundle_before_reading_spec(tmp_path, monkeypatch,
                                                  defect):
    import importlib

    trainer_module = importlib.import_module('kumo.engine.trainer.trainer')
    bundle = _fake_published_bundle(tmp_path)
    write_training_spec(bundle, predictive_query='PREDICT COUNT(*)',
                        model_plan=ModelPlan(), snapshot_id='snap-123',
                        training_table_generation=_generation_plan())
    manifest_path = bundle / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    entry = manifest['artifacts'][0]
    if defect == 'checkpoint_contents':
        (bundle / entry['relative_path']).write_text('[]')
    elif defect == 'declared_size':
        entry['size_bytes'] += 1
    elif defect == 'training_spec_contents':
        (bundle / 'training-spec.json').write_text('not JSON')
    else:
        outside = tmp_path / 'outside.json'
        outside.write_text('{}')
        if defect == 'parent_path':
            entry['relative_path'] = '../outside.json'
        elif defect == 'absolute_path':
            entry['relative_path'] = str(outside)
        elif defect == 'artifact_symlink':
            artifact = bundle / entry['relative_path']
            artifact.unlink()
            artifact.symlink_to(outside)
        else:
            spec = bundle / 'training-spec.json'
            outside.write_bytes(spec.read_bytes())
            spec.unlink()
            spec.symlink_to(outside)
    manifest_path.write_text(json.dumps(manifest))

    class ExistingBundleSink(_FakeSink):
        def fetch_model(self, connector, local_dir, **kwargs):
            # Preserve symlinks so this also covers a fetching plugin which
            # has already materialized an escaping link.
            shutil.copytree(bundle, local_dir, dirs_exist_ok=True,
                            symlinks=True)
            return PublishedModel(name='CHURN_GNN', version='V1',
                                  uri='fake://CHURN_GNN@V1')

    def must_not_parse(_bundle):
        pytest.fail('training spec must not be parsed before bundle validation')

    monkeypatch.setattr(trainer_module, 'read_training_spec', must_not_parse)
    with pytest.raises(ValueError,
                       match=('digest|size|absolute path|below the bundle')):
        Trainer.load_from_connector(
            _FakeConnector(ExistingBundleSink()),
            location=ModelRegistryLocation(model_name='CHURN_GNN'))


@pytest.fixture
def snapshot(tmp_path: Path) -> GraphSnapshot:
    config = {
        'tables': [{
            'table_name':
            'users',
            'source_table_name':
            'db.schema.users',
            'pkey': {
                'name': 'id',
                'dtype': 'int64',
                'stype': 'ID'
            },
            'cols': [{
                'name': 'label',
                'dtype': 'bool',
                'stype': 'categorical'
            }],
            'fkeys': [],
        }]
    }
    return GraphSnapshot('snap-123', tmp_path / 'index', config)


def test_predict_works_after_load_without_refitting(snapshot: GraphSnapshot,
                                                    tmp_path: Path,
                                                    monkeypatch) -> None:
    r"""The scenario `Trainer.load` exists for: publish, restart the
    process (a fresh `Trainer.load(...)` call, no `_training_query` carried
    over from an in-memory `fit()`), then `predict()` -- without another
    `fit()`. Regression test for `_training_query` not being restored by
    `load()`, which made `predict()`'s query-contract check
    (`trainer.py`'s `self._training_query is None`) always reject a
    freshly-loaded `Trainer`.
    """
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.task_dataset import DiskgraphTaskManifest
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    from kumo_core.types.enums import PredictionArtifactType
    from kumoapi.jobs import PredictionArtifactType as ApiType

    from kumo.engine.trainer.output_config import PredictionOutputConfig

    query = 'PREDICT users.label FOR EACH users.id'
    bundle = _fake_published_bundle(tmp_path / 'src')
    write_training_spec(bundle, predictive_query=query, model_plan=ModelPlan(),
                        snapshot_id=snapshot.id,
                        training_table_generation=_generation_plan())
    trainer = Trainer.load(bundle)
    monkeypatch.setattr(type(trainer), 'scratch_dir',
                        property(lambda _: tmp_path))

    @dataclass(frozen=True)
    class _TaskDataset:
        job_id: str = 'generated-job'
        artifact_root: str = 'generated-artifacts'

    table = PredictionTable(query, tmp_path, tmp_path, 0, 'tt-1', (), ())
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.build_generated_training_task_dataset',
        lambda **_: _TaskDataset(),
    )
    out_sink = SimpleNamespace(write_files=lambda *a, **k: None)
    output = PredictionOutputConfig(SimpleNamespace(sink_backend=out_sink),
                                    ApiType.PREDICTIONS, 'out')
    monkeypatch.setattr(
        'kumo_core.prediction.materialize_prediction_task', lambda *a, **k:
        SimpleNamespace(manifest_path='task', bundle_dir='bundle'))
    monkeypatch.setattr('kumo.engine.trainer.trainer._stage_model_bundle',
                        lambda *a: tmp_path)
    (tmp_path / 'config.json').write_text('{}')
    monkeypatch.setattr(
        DiskgraphTaskManifest, 'load', lambda *a: SimpleNamespace(
            task_type='binary_classification', stage_counts={'predict': 1}))
    monkeypatch.setattr(ResolvedNumericalTask, 'from_manifest',
                        lambda *a, **k: None)
    monkeypatch.setattr('kumo_core.train.trainer.load_encoders',
                        lambda *a: None)

    def fake_predict(**kwargs) -> None:
        from kumo_core.train.io import StreamingFrameWriter

        kind = PredictionArtifactType.PREDICTIONS
        worker = kwargs['partition_id']
        writer = StreamingFrameWriter(
            str(
                Path(kwargs['output_dir']) / kind.output_file_name /
                f'part-{worker:05d}'),
            max_file_size=1 << 20,
            max_buffer_size=1 << 20,
            index=False,
            stats_sample_rows=1000,
        )
        with writer:
            writer.write(
                pd.DataFrame({
                    'True_PROB': [.8],
                    'TARGET_PRED': [True],
                }))
        kwargs['artifact_stats_collector'].add_partition(
            worker, {kind: writer.stats})

    monkeypatch.setattr(DiskgraphNestedTrainer, 'load',
                        lambda *a, **k: SimpleNamespace(predict=fake_predict))

    result = trainer.predict(table, snapshot=snapshot, output_config=output,
                             binary_classification_threshold=.5)

    assert result.frame['TARGET_PRED'].tolist() == [True]


def _export(sink, bundle, location=None, artifact_paths=None):
    return Trainer(ModelPlan())._finish_fit(
        TrainingResult({}, artifact_paths or {},
                       bundle), predictive_query='PREDICT COUNT(*)',
        snapshot_id='snap-123', training_table_generation=_generation_plan(),
        export_config=SimpleNamespace(
            connector=SimpleNamespace(sink_backend=sink),
            path=None if location else 'm', location=location))


def _recording_sink(calls):
    return SimpleNamespace(publish_model=lambda *a, **k: calls.append(k) or
                           PublishedModel(name='m', version='1', uri='u'))


def test_publish_records_only_the_bundle_id_pointer(tmp_path: Path) -> None:
    calls = []
    result = _export(_recording_sink(calls), _fake_published_bundle(tmp_path))
    assert {k: v
            for k, v in calls[0]['metadata'].items()
            if k.startswith('kumo_')} == {'kumo_bundle_id': result.bundle_id}


class _AliasingSink(_FakeSink):
    def __init__(self, recorded: Optional[str] = None,
                 fail: bool = False) -> None:
        super().__init__()
        self.recorded = recorded
        self.fail = fail
        self.aliases: dict = {}

    def set_model_alias(self, connector, location, alias) -> None:
        if self.fail:
            raise RuntimeError('registry unavailable')
        self.aliases[alias] = location.version

    def fetch_model(self, connector, local_dir, **kwargs) -> PublishedModel:
        shutil.copytree(self.published['CHURN_GNN@latest'], local_dir,
                        dirs_exist_ok=True)
        return PublishedModel(name='CHURN_GNN', version='V1', uri='fake://',
                              metadata={'kumo_bundle_id': self.recorded})


@pytest.mark.parametrize('sink,warning', [
    (_AliasingSink(), None),
    (_AliasingSink(fail=True), 'registry unavailable'),
    (_FakeSink(), 'cannot set model aliases'),
])
def test_export_with_alias_publishes_then_sets_it_best_effort(
        tmp_path: Path, sink, warning) -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = _export(sink, _fake_published_bundle(tmp_path),
                         ModelRegistryLocation('CHURN_GNN', alias='prod'))
    assert result.published_model.version == 'V1'
    assert list(sink.published) == ['CHURN_GNN@latest']
    messages = [str(w.message) for w in caught if 'alias' in str(w.message)]
    if warning is None:
        assert sink.aliases == {'prod': 'V1'} and messages == []
    else:
        assert len(messages) == 1 and warning in messages[0]


def test_load_by_alias_warns_on_recorded_bundle_mismatch(
        tmp_path: Path) -> None:
    bundle = _fake_published_bundle(tmp_path)
    write_training_spec(bundle, predictive_query='PREDICT COUNT(*)',
                        model_plan=ModelPlan(), snapshot_id='snap-123',
                        training_table_generation=_generation_plan())
    sink = _AliasingSink('other')
    sink.publish_model(None, str(bundle),
                       location=ModelRegistryLocation('CHURN_GNN'))
    with pytest.warns(UserWarning, match='registry records bundle other'):
        loaded = Trainer.load_from_connector(
            _FakeConnector(sink),
            location=ModelRegistryLocation('CHURN_GNN', alias='prod'))
    assert loaded.result.id == Trainer.load(bundle).result.id


def test_training_result_exports_like_its_trainer(tmp_path: Path) -> None:
    bundle = _fake_published_bundle(tmp_path)
    write_training_spec(bundle, predictive_query='PREDICT COUNT(*)',
                        model_plan=ModelPlan(), snapshot_id='snap-123',
                        training_table_generation=_generation_plan())
    sink = _AliasingSink()
    published = TrainingResult({}, {}, bundle).export(
        ModelOutputConfig(_FakeConnector(sink),
                          location=ModelRegistryLocation('CHURN_GNN',
                                                         alias='prod')))
    assert published.version == 'V1' and sink.aliases == {'prod': 'V1'}
    assert list(sink.published) == ['CHURN_GNN@latest']


def test_alias_needs_an_aliasing_sink() -> None:
    connector = _FakeConnector(_FakeSink())
    with pytest.raises(NotImplementedError, match='alias'):
        Trainer.load_from_connector(
            connector, location=ModelRegistryLocation('m', alias='prod'))
    for kwargs in (dict(alias='prod', version='3'), dict(alias='')):
        with pytest.raises(ValueError, match='alias'):
            ModelRegistryLocation('m', **kwargs)


def test_invalid_export_alias_is_rejected_before_publishing(
        tmp_path: Path) -> None:
    sink = _AliasingSink()
    trainer = Trainer(ModelPlan())
    trainer._training_result = TrainingResult({}, {},
                                              _fake_published_bundle(tmp_path))
    with pytest.raises(ValueError, match='model alias'):
        trainer.export(
            SimpleNamespace(
                connector=SimpleNamespace(sink_backend=sink), path=None,
                location=ModelRegistryLocation('CHURN_GNN', alias='LATEST')))
    assert sink.published == {}


class _RegistryTracker:
    def __init__(self, sink):
        self.connector = _FakeConnector(sink)

    def start(self, record):
        pass

    def on_trial(self, record, event):
        pass

    def finish(self, record, error):
        pass


@pytest.mark.parametrize('sink,registered', [
    (_AliasingSink(), 'V1'),
    (SimpleNamespace(publish_model=lambda *a, **k: 1 / 0), None),
])
def test_mlops_registry_registers_after_the_fit(tmp_path: Path, monkeypatch,
                                                sink, registered) -> None:
    import warnings

    import kumo.engine as kumo

    bundle = _fake_published_bundle(tmp_path)
    write_training_spec(bundle, predictive_query='PREDICT COUNT(*)',
                        model_plan=ModelPlan(), snapshot_id='snap-123',
                        training_table_generation=_generation_plan())
    monkeypatch.setattr(
        Trainer, '_fit', lambda self, *a, **k: TrainingResult(
            {}, {}, bundle, bundle_id='b1'))
    tracker = _RegistryTracker(sink)
    registry = ModelRegistryLocation('CHURN_GNN', alias='prod')
    trainer = Trainer(ModelPlan())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = trainer.fit(
            SimpleNamespace(query='PREDICT COUNT(*)'),
            snapshot=SimpleNamespace(id='s'),
            mlops=kumo.MLOps(str(tmp_path / 'root'), tracking=tracker,
                             registry=registry, store=False))
    failures = [w for w in caught if 'Could not register' in str(w.message)]
    assert getattr(result.registered_model, 'version', None) == registered
    assert result.published_model is None
    if registered:
        assert trainer.result is result and not failures
        assert sink.aliases == {'prod': 'V1'}
    else:
        assert len(failures) == 1
        assert 'ZeroDivisionError' in str(failures[0].message)
        with pytest.raises(RuntimeError, match='ZeroDivisionError') as raised:
            trainer.fit(
                SimpleNamespace(query='PREDICT COUNT(*)'),
                snapshot=SimpleNamespace(id='s'),
                mlops=kumo.MLOps(str(tmp_path / 'root'), tracking=tracker,
                                 registry=registry, store=False,
                                 on_error='raise'))
        assert isinstance(raised.value.__cause__, ZeroDivisionError)


def test_mlops_registry_alias_is_checked_before_the_fit(tmp_path: Path,
                                                        monkeypatch) -> None:
    import kumo.engine as kumo

    monkeypatch.setattr(Trainer, '_fit', lambda *a, **k: 1 / 0)
    with pytest.raises(ValueError, match='model alias'):
        Trainer(ModelPlan()).fit(
            SimpleNamespace(query='q'), snapshot=SimpleNamespace(id='s'),
            mlops=kumo.MLOps(
                str(tmp_path), tracking=_RegistryTracker(_AliasingSink()),
                registry=ModelRegistryLocation('CHURN_GNN', alias='LATEST'),
                store=False))
