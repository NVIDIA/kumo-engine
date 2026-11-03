# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from kumo_core import prediction
from kumo_core.train.artifact_manifest import load_prediction_selection
from kumo_core.types import PredictionArtifactType
from kumoapi.model_plan import ModelPlan

from kumo.engine.pquery.prediction_table import PredictionTable
from kumo.engine.trainer.trainer import Trainer


@dataclass(frozen=True)
class _TaskDataset:
    job_id: str = 'generated-job'
    artifact_root: str = 'generated-artifacts'


@pytest.fixture
def prepared_prediction(snapshot, tmp_path, monkeypatch):
    import torch
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.task_dataset import DiskgraphTaskManifest
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    from kumo_core.train.io import StreamingFrameWriter
    from kumo_core.train.loss import BinaryCrossEntropyLoss
    from kumo_core.types import TaskType

    trainer = Trainer(ModelPlan())
    trainer._training_result = SimpleNamespace(artifact_paths={})
    trainer._snapshot_id = snapshot.id
    trainer._training_query = 'PREDICT users.label FOR EACH users.id'
    table = PredictionTable(trainer._training_query, tmp_path, tmp_path, 8,
                            'tt-1', (123, ), ())
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.build_generated_training_task_dataset',
        lambda **_: _TaskDataset())
    monkeypatch.setattr(type(trainer), 'scratch_dir',
                        property(lambda _: tmp_path))
    monkeypatch.setattr('kumo.engine.trainer.trainer._stage_model_bundle',
                        lambda *a: tmp_path)
    (tmp_path / 'config.json').write_text('{}')
    monkeypatch.setattr(
        prediction,
        'materialize_prediction_task', lambda *a, **kw: SimpleNamespace(
            manifest_path='task', bundle_dir='bundle'))
    monkeypatch.setattr(
        DiskgraphTaskManifest, 'load', lambda *_: SimpleNamespace(
            task_type='binary_classification', stage_counts={'predict': 8}))
    monkeypatch.setattr(ResolvedNumericalTask, 'from_manifest',
                        lambda *a, **kw: None)
    monkeypatch.setattr('kumo_core.train.trainer.load_encoders',
                        lambda *_: None)
    monkeypatch.setattr(prediction, '_prediction_worker_devices', lambda n, accelerator=None:
                        (None, ) * n)
    # Real process startup/error propagation is covered by test_prediction_workers;
    # run the same worker functions here so the real writers/manifests compose.
    monkeypatch.setattr(
        prediction, '_run_prediction_workers',
        lambda n, fn, kw: [fn(worker_id=i, **kw) for i in range(n)])
    loads = []

    def load(*args, **kwargs):
        loads.append(kwargs)
        loss = BinaryCrossEntropyLoss(TaskType.BINARY_CLASSIFICATION,
                                      threshold=kwargs['binary_threshold'])

        def predict(**options):
            worker = options['partition_id']
            total = options['total_partitions']
            ids = list(range(worker, 8, total))
            probs = torch.tensor([[.25], [.5], [.75], [float('nan')]] * 2)[ids]
            frame = pd.DataFrame({
                'ENTITY':
                ids,
                'True_PROB':
                probs[:, 0].numpy(),
                'TARGET_PRED':
                loss.to_prediction(probs.logit()).numpy()
            })
            frames = {}
            if options['return_predictions']:
                frames[PredictionArtifactType.PREDICTIONS] = frame
            if options['return_embeddings']:
                frames[PredictionArtifactType.EMBEDDINGS] = pd.DataFrame({
                    'ENTITY':
                    ids,
                    'EMBEDDING': [[float(i), 1.0] for i in ids]
                })
            stats = {}
            for kind, output_frame in frames.items():
                writer = StreamingFrameWriter(
                    str(
                        Path(options['output_dir']) / kind.output_file_name /
                        f'part-{worker:05d}'), max_file_size=1 << 20,
                    max_buffer_size=1 << 20, index=False,
                    stats_sample_rows=1000)
                with writer:
                    writer.write(output_frame)
                stats[kind] = writer.stats
            options['artifact_stats_collector'].add_partition(worker, stats)

        return SimpleNamespace(predict=predict)

    monkeypatch.setattr(DiskgraphNestedTrainer, 'load', load)
    return trainer, table, loads, snapshot


@pytest.mark.parametrize('workers', [1, 2])
def test_public_completion_uses_selected_files_without_collecting(
        prepared_prediction, monkeypatch, workers):
    trainer, table, loads, snapshot = prepared_prediction
    with monkeypatch.context() as guarded:
        guarded.setattr(pd, 'read_parquet',
                        lambda *a, **kw: pytest.fail('eager read'))
        guarded.setattr(pd, 'concat',
                        lambda *a, **kw: pytest.fail('eager concat'))
        result = trainer.predict(table, snapshot=snapshot, num_workers=workers,
                                 binary_classification_threshold=.5)
        assert result.num_rows == 8
        assert len(result.files) == workers
        selected = load_prediction_selection(result.selection)
        assert len(selected.shard_manifest_sha256s) == workers
    assert len(loads) == workers
    # Explicit collection stays available for small outputs and reflects the
    # producer's labels/probabilities, including its established NaN behavior.
    frame = result.frame.sort_values('ENTITY')
    assert frame['ENTITY'].tolist() == list(range(8))
    assert frame['TARGET_PRED'].tolist() == [False, False, True, False] * 2
    assert frame['True_PROB'].iloc[:3].tolist() == [.25, .5, .75]


def test_failed_worker_never_publishes_selection(prepared_prediction,
                                                 monkeypatch, tmp_path):
    trainer, table, _, snapshot = prepared_prediction

    def fail(n, fn, kwargs):
        fn(worker_id=0, **kwargs)
        raise RuntimeError('worker one failed')

    monkeypatch.setattr(prediction, '_run_prediction_workers', fail)
    with pytest.raises(RuntimeError, match='worker one failed'):
        trainer.predict(table, snapshot=snapshot, num_workers=2,
                        binary_classification_threshold=.5)
    assert not list(tmp_path.rglob('_selected_artifacts.json'))


@pytest.mark.parametrize('workers', [1, 2])
def test_connector_export_uses_selected_files_without_collecting(
        prepared_prediction, monkeypatch, tmp_path, workers):
    import pyarrow.parquet as pq
    from kumo_core.object_store_sink import ObjectStoreSinkBackend
    from kumoapi.jobs import PredictionArtifactType as ApiType
    from pyarrow.fs import LocalFileSystem

    from kumo.engine.trainer.output_config import PredictionOutputConfig

    trainer, table, _, snapshot = prepared_prediction

    class LocalSink(ObjectStoreSinkBackend):
        scheme = 'file'
        provider = 'test'

        def _destination(self, connector, coordinate):
            return LocalFileSystem(), coordinate

    target = tmp_path / 'exported'
    output = PredictionOutputConfig(SimpleNamespace(sink_backend=LocalSink()),
                                    ApiType.PREDICTIONS,
                                    str(target) + '/')
    monkeypatch.setattr(pd, 'read_parquet',
                        lambda *a, **kw: pytest.fail('eager read'))
    monkeypatch.setattr(pd, 'concat',
                        lambda *a, **kw: pytest.fail('eager concat'))
    result = trainer.predict(table, snapshot=snapshot, num_workers=workers,
                             output_config=output,
                             binary_classification_threshold=.5)
    assert result.num_rows == 8
    assert (target / '_SUCCESS').exists()
    exported = sorted(target.glob('*.parquet'))
    assert len(exported) == len(result.files) == workers
    assert sum(pq.ParquetFile(p).metadata.num_rows for p in exported) == 8
    assert sorted(p.read_bytes() for p in exported) == sorted(
        Path(p).read_bytes() for p in result.files)


def test_unsupported_export_rejected_before_prediction(prepared_prediction,
                                                       monkeypatch):
    from kumoapi.jobs import PredictionArtifactType as ApiType

    from kumo.engine.trainer.output_config import PredictionOutputConfig

    trainer, table, _, snapshot = prepared_prediction
    output = PredictionOutputConfig(
        SimpleNamespace(sink_backend=SimpleNamespace(
            write_table=lambda *a: None)), ApiType.PREDICTIONS, 'out')
    monkeypatch.setattr(
        'kumo_core.prediction.materialize_prediction_task',
        lambda *a, **kw: pytest.fail('must reject before materialization'))
    with pytest.raises(NotImplementedError,
                       match='FileArtifactSink.write_files'):
        trainer.predict(table, snapshot=snapshot, output_config=output,
                        binary_classification_threshold=.5)


@pytest.mark.parametrize('workers', [1, 2])
def test_separate_exact_output_paths(prepared_prediction, tmp_path, workers):
    import pyarrow.parquet as pq
    from kumo_core.object_store_sink import ObjectStoreSinkBackend
    from pyarrow.fs import LocalFileSystem

    from kumo.engine.trainer.output_config import PredictionOutputConfig

    class LocalSink(ObjectStoreSinkBackend):
        scheme = 'file'
        provider = 'test'

        def _destination(self, connector, coordinate):
            return LocalFileSystem(), coordinate

    trainer, table, _, snapshot = prepared_prediction
    connector = SimpleNamespace(source_type='LOCAL_FILE',
                                sink_backend=LocalSink())
    paths = {
        kind: str(tmp_path / kind / 'project') + '/'
        for kind in ('predictions', 'embeddings')
    }
    config = PredictionOutputConfig(connector, output_paths=paths)
    result = trainer.predict(table, snapshot=snapshot, output_config=config,
                             num_workers=workers,
                             binary_classification_threshold=.5)
    for kind, path in paths.items():
        files = sorted(Path(path).glob('*.parquet'))
        assert len(files) == workers
        assert sum(pq.ParquetFile(p).metadata.num_rows for p in files) == 8
        frame = pd.concat(pd.read_parquet(p) for p in files)
        assert sorted(frame['ENTITY']) == list(range(8))
        if kind == 'embeddings':
            assert all(len(vector) == 2 for vector in frame['EMBEDDING'])
        assert (Path(path) / '_SUCCESS').exists()
        assert not Path(path.rstrip('/') + '_' + kind).exists()
    result.export(config)
    with pytest.raises(ValueError, match='another export'):
        trainer.predict(table, snapshot=snapshot, output_config=config,
                        binary_classification_threshold=.5)


def test_export_missing_artifact_does_not_publish_partial_outputs(
        prepared_prediction, tmp_path):
    from kumo.engine.trainer.output_config import PredictionOutputConfig

    trainer, table, _, snapshot = prepared_prediction
    result = trainer.predict(table, snapshot=snapshot,
                             binary_classification_threshold=.5)
    writes = []
    connector = SimpleNamespace(
        source_type='LOCAL_FILE', sink_backend=SimpleNamespace(
            holds_files=lambda *a: True, filesystem=lambda *a: None,
            write_files=lambda *a, **kw: writes.append(a)))
    config = PredictionOutputConfig(
        connector, output_paths={
            'predictions': str(tmp_path / 'predictions') + '/',
            'embeddings': str(tmp_path / 'embeddings') + '/',
        })
    with pytest.raises(ValueError, match='did not generate embeddings'):
        result.export(config)
    assert writes == []


@pytest.mark.parametrize('provider', ['s3', 'gcs'])
@pytest.mark.parametrize('rooted', [False, True])
def test_exact_paths_reject_destination_aliases_before_writes(
        provider, rooted, monkeypatch):
    from kumo.engine import GCSConnector, PredictionOutputConfig, S3Connector

    connector_type, scheme = ((S3Connector, 's3') if provider == 's3' else
                              (GCSConnector, 'gs'))
    root = f'{scheme}://bucket/root'
    connector = connector_type(root_dir=root if rooted else None)
    monkeypatch.setattr(
        connector.sink_backend, '_filesystem',
        lambda *a: pytest.fail('must reject before cloud access'))
    monkeypatch.setattr(
        connector.sink_backend, 'write_files',
        lambda *a, **kw: pytest.fail('must reject before writes'))
    with pytest.raises(ValueError, match='distinct output paths'):
        PredictionOutputConfig(
            connector, output_paths={
                'predictions':
                f'{root}/out.parquet',
                'embeddings':
                ('out.parquet' if rooted else f'{root}/%6fut.parquet'),
            })


def test_retained_outputs_retry_after_partial_embedding_export(
        prepared_prediction, tmp_path, monkeypatch):
    from kumo_core.object_store_sink import _ArrowExportFiles

    from kumo.engine import FileConnector, PredictionOutputConfig
    from kumo.engine.trainer.trainer import PredictionResult

    trainer, table, _, snapshot = prepared_prediction
    connector = FileConnector(root_dir=str(tmp_path))
    initial = PredictionOutputConfig(
        connector, output_paths={
            'predictions': 'initial/predictions/',
            'embeddings': 'initial/embeddings/',
        })
    result = trainer.predict(table, snapshot=snapshot, output_config=initial,
                             num_workers=2, binary_classification_threshold=.5)
    retry = PredictionOutputConfig(
        connector, output_paths={
            'predictions': 'retry/predictions/',
            'embeddings': 'retry/embeddings/',
        })
    embeddings = PredictionResult(result.selection,
                                  PredictionArtifactType.EMBEDDINGS).files
    failed_part = Path(embeddings[1]).name
    copy = _ArrowExportFiles.copy_file

    def interrupt(storage, source, destination, **kwargs):
        assert not (Path(destination).parent / '_SUCCESS').exists()
        if '/embeddings/' in destination and Path(
                destination).name == failed_part:
            raise RuntimeError('embedding transfer interrupted')
        return copy(storage, source, destination, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_ArrowExportFiles, 'copy_file', interrupt)
        with pytest.raises(RuntimeError,
                           match='embedding transfer interrupted'):
            result.export(retry)
    prediction_dir = tmp_path / 'retry/predictions'
    embedding_dir = tmp_path / 'retry/embeddings'
    assert (prediction_dir / '_SUCCESS').exists()
    assert not (embedding_dir / '_SUCCESS').exists()
    assert len(list(embedding_dir.glob('*.parquet'))) == 1

    def recopy(storage, source, destination, **kwargs):
        assert not (Path(destination).parent / '_SUCCESS').exists()
        return copy(storage, source, destination, **kwargs)

    monkeypatch.setattr(_ArrowExportFiles, 'copy_file', recopy)
    result.export(retry)
    for directory in (prediction_dir, embedding_dir):
        assert (directory / '_SUCCESS').exists()
        frame = pd.concat(
            pd.read_parquet(path) for path in directory.glob('*.parquet'))
        assert sorted(frame.ENTITY) == list(range(8))
