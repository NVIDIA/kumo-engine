# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_core import prediction
from kumo_core.embedding_models import EmbeddingModelStore
from kumo_core.task_dataset import DiskgraphTaskManifest
from kumo_core.types import TaskType
from kumo_core.types.enums import PredictionArtifactType
from kumoapi.model_plan import ModelPlan

from kumo.engine.trainer.trainer import Trainer


def _spawn_probe(*, worker_id, directory, fail=False):
    import multiprocessing
    Path(directory,
         f'{worker_id}.txt').write_text(multiprocessing.get_start_method())
    if fail and worker_id == 0:
        raise ValueError('scorer failed')


@pytest.mark.parametrize('fail', [False, True])
def test_workers_spawn_and_join_every_process(tmp_path, fail):
    if fail:
        with pytest.raises(RuntimeError,
                           match='Prediction worker 0 failed: scorer failed'):
            prediction._run_prediction_workers(2, _spawn_probe, {
                'directory': str(tmp_path),
                'fail': True
            })
    else:
        prediction._run_prediction_workers(2, _spawn_probe,
                                           {'directory': str(tmp_path)})
    assert [p.read_text()
            for p in sorted(tmp_path.glob('*.txt'))] == ['spawn', 'spawn']


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_bad_worker_count_fails_at_public_entry(count, monkeypatch):
    materialize = Mock(side_effect=AssertionError('no materialization'))
    monkeypatch.setattr('kumo_core.prediction.materialize_prediction_task',
                        materialize)
    with pytest.raises(ValueError, match='num_workers must be an integer'):
        Trainer(ModelPlan()).predict(None, snapshot=None, output_config=None,
                                     num_workers=count)
    materialize.assert_not_called()


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_bad_class_count_fails_before_prediction_work(count, monkeypatch):
    materialize = Mock(side_effect=AssertionError('no materialization'))
    score = Mock(side_effect=AssertionError('no scoring'))
    monkeypatch.setattr('kumo_core.prediction.materialize_prediction_task',
                        materialize)
    monkeypatch.setattr('kumo_core.prediction.score_prediction', score)

    with pytest.raises(
            ValueError,
            match='num_classes_to_return must be a positive integer'):
        Trainer(ModelPlan()).predict(None, snapshot=None,
                                     num_classes_to_return=count)

    materialize.assert_not_called()
    score.assert_not_called()


@pytest.mark.parametrize('task_type', [
    TaskType.MULTICLASS_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
    TaskType.MULTILABEL_RANKING,
    TaskType.STATIC_LINK_PREDICTION,
])
@pytest.mark.parametrize('count', [None, 1, 3])
def test_valid_class_count_reaches_scorer(task_type, count, tmp_path,
                                          monkeypatch):
    trainer = Trainer(ModelPlan(), accelerator='cpu')
    trainer._training_result = SimpleNamespace(artifact_paths={})
    trainer._training_query = 'trained query'
    monkeypatch.setattr(Trainer, 'scratch_dir', property(lambda self: tmp_path))
    monkeypatch.setattr(
        'kumo_core.prediction_options.prediction_query_contract', lambda *args:
        (None, task_type.value))
    monkeypatch.setattr('kumo_core.pquery.normalize_graph', lambda *args: None)
    monkeypatch.setattr('kumo_core.pquery.parse_and_validate_query',
                        lambda *args: SimpleNamespace(
                            query=SimpleNamespace(top_k=None)))
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.build_generated_training_task_dataset',
        lambda **kwargs: object())
    monkeypatch.setattr('kumo_core.prediction.materialize_prediction_task',
                        lambda *args, **kwargs: object())
    monkeypatch.setattr('kumo.engine.trainer.trainer._stage_model_bundle',
                        lambda *args: tmp_path)
    monkeypatch.setattr('kumo_core.prediction.forecast_task_options',
                        lambda *args: None)
    score = Mock(return_value=object())
    monkeypatch.setattr('kumo_core.prediction.score_prediction', score)
    table = SimpleNamespace(query='prediction query', tt_id='table',
                            compiler_options=None, timestamps=(),
                            staging_dir=tmp_path, table_path=tmp_path,
                            row_count=1, required_stages=(),
                            forecast_timeframe_step=None)
    snapshot = SimpleNamespace(graph_config={}, index_path=tmp_path, id='snap')

    if task_type.is_link_pred and count is None:
        with pytest.raises(ValueError, match='specify the number of predictions'):
            trainer.predict(table, snapshot=snapshot,
                            num_classes_to_return=count)
        score.assert_not_called()
        return
    trainer.predict(table, snapshot=snapshot, num_classes_to_return=count)

    assert score.call_args.kwargs['top_k'] == count
    assert score.call_args.kwargs['accelerator'] == 'cpu'


@pytest.mark.parametrize('population,requested,expected', [(7, 3, 3), (1, 3, 1),
                                                           (0, 3, 0)])
@pytest.mark.parametrize('extra_fields', [None, {'PREDICTIONS': []}])
@pytest.mark.parametrize('accelerator', ['cpu', 'gpu', 'mps', 'auto'])
def test_partition_counts_devices_and_runtime_allocation(
        tmp_path, monkeypatch, population, requested, expected, extra_fields,
        accelerator):
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    loads, calls = [], []
    store = EmbeddingModelStore()
    monkeypatch.setattr(
        'kumo_core.train.artifact_manifest.hash_prediction_config',
        lambda *_: '1' * 64)
    monkeypatch.setattr(prediction, '_publish_scored_shard',
                        lambda *a, **kw: kw)
    monkeypatch.setattr(
        'kumo_core.train.artifact_manifest.publish_prediction_selection',
        lambda refs, **kw: refs)

    monkeypatch.setattr(
        DiskgraphTaskManifest, 'load', lambda *a: SimpleNamespace(
            task_type='regression', stage_counts={'predict': population}))
    monkeypatch.setattr(ResolvedNumericalTask, 'from_manifest',
                        lambda *a, **k: None)
    monkeypatch.setattr('kumo_core.train.trainer.load_encoders',
                        lambda *a: None)

    def load(*args, **kwargs):
        loads.append(kwargs)
        return SimpleNamespace(predict=lambda **options: calls.append(options))

    monkeypatch.setattr(DiskgraphNestedTrainer, 'load', load)
    monkeypatch.setattr(prediction, '_prediction_worker_devices',
                        lambda n, accelerator=None: tuple(range(n)))
    monkeypatch.setattr(
        prediction, '_run_prediction_workers',
        lambda n, score, kw: [score(worker_id=i, **kw) for i in range(n)])
    kwargs = dict(
        materialized=SimpleNamespace(manifest_path='manifest',
                                     bundle_dir='bundle'), model_dir='model',
        run_id='run', scratch_dir=str(tmp_path),
        artifacts=frozenset({PredictionArtifactType.PREDICTIONS}),
        num_workers=requested, extra_fields=extra_fields,
        accelerator=accelerator, graph_snapshot_id='snapshot',
        source_config_sha256='2' * 64, embedding_models=store)
    if not expected:
        with pytest.raises(RuntimeError, match='wrote no parquet'):
            prediction.score_prediction(**kwargs)
        assert not calls
        return
    result = prediction.score_prediction(**kwargs)
    assert len(result) == expected
    assert len(calls) == expected
    assert [load['devices'] for load in loads] == [[i] for i in range(expected)]
    assert all(load['embedding_models'] is store for load in loads)
    assert all(load.get('accelerator') == accelerator for load in loads)
    assert [call['partition_id'] for call in calls] == list(range(expected))
    assert [call['output_partition_ids']
            for call in calls] == [[i] for i in range(expected)]
    assert all(call['total_partitions'] == expected for call in calls)
    assert all(call['ignore_time_col'] for call in calls)
    assert all(call['runtime_num_workers'] == expected for call in calls)
    assert [call['runtime_worker_id']
            for call in calls] == list(range(expected))


def test_cuda_device_admission(monkeypatch):
    import torch
    monkeypatch.delenv('KUMO_ACCELERATOR', raising=False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    with pytest.raises(ValueError, match='only 1 are visible'):
        prediction._prediction_worker_devices(2)
    assert prediction._prediction_worker_devices(1) == (None, )
    monkeypatch.setenv('KUMO_ACCELERATOR', 'cpu')
    with pytest.raises(ValueError, match='only 1 are visible'):
        prediction._prediction_worker_devices(2)
    assert prediction._prediction_worker_devices(2, 'cpu') == (None, None)


@pytest.mark.parametrize('accelerator', ['cpu', 'mps'])
def test_explicit_accelerator_skips_cuda_admission(monkeypatch, accelerator):
    import torch
    monkeypatch.setenv('KUMO_ACCELERATOR', 'gpu')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    assert prediction._prediction_worker_devices(2, accelerator) == (None, None)


@pytest.mark.parametrize('accelerator', ['auto', 'gpu'])
def test_explicit_gpu_selection_overrides_cpu_environment(
        monkeypatch, accelerator):
    import torch
    monkeypatch.setenv('KUMO_ACCELERATOR', 'cpu')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    assert prediction._prediction_worker_devices(2, accelerator) == (0, 1)


@pytest.mark.parametrize('mismatch', [False, True])
def test_forecast_options_reach_every_scorer_before_output(
        tmp_path, monkeypatch, mismatch):
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer

    options = {
        'forecast_length': 3,
        'lag_timesteps': 1,
        'unix_start_offset': 0,
        'unix_end_offset': 86400000000000,
        'unix_step_size': 86400000000000,
    }
    captured, predicted = [], []
    task = SimpleNamespace(
        lag_timesteps=1,
        has_year_over_year_labels=False,
        unix_step_size=options['unix_step_size'],
        train_table=SimpleNamespace(name='train_table'),
        encoder_inputs={
            'train_table':
            [SimpleNamespace(column_name='autoregression_labels')]
        },
    )
    monkeypatch.setattr(
        'kumo_core.train.artifact_manifest.hash_prediction_config',
        lambda *_: '1' * 64)
    monkeypatch.setattr(
        DiskgraphTaskManifest, 'load', lambda *a: SimpleNamespace(
            task_type='forecasting', stage_counts={'predict': 2}))

    def resolve(*args, **kwargs):
        captured.append(kwargs)
        return task

    monkeypatch.setattr(ResolvedNumericalTask, 'from_manifest', resolve)
    monkeypatch.setattr('kumo_core.train.trainer.load_encoders',
                        lambda *a: None)
    trainer = SimpleNamespace(
        past_encoder=SimpleNamespace(length=2 if mismatch else 1),
        year_over_year_encoder=None,
        head=SimpleNamespace(unix_step_size=options['unix_step_size']),
        encoders=SimpleNamespace(
            encoders={'train_table': {
                'autoregression_labels': object()
            }}),
        predict=lambda **kwargs: predicted.append(kwargs),
    )
    monkeypatch.setattr(DiskgraphNestedTrainer, 'load', lambda *a, **k: trainer)
    monkeypatch.setattr(prediction, '_prediction_worker_devices', lambda n, accelerator=None:
                        (None, ) * n)
    monkeypatch.setattr(
        prediction, '_run_prediction_workers',
        lambda n, score, kw: [score(worker_id=i, **kw) for i in range(n)])
    monkeypatch.setattr(prediction, '_publish_scored_shard',
                        lambda *a, **kw: kw)
    monkeypatch.setattr(
        'kumo_core.train.artifact_manifest.publish_prediction_selection',
        lambda refs, **kw: refs)
    kwargs = dict(
        materialized=SimpleNamespace(manifest_path='manifest',
                                     bundle_dir='bundle'), model_dir='model',
        run_id='run', scratch_dir=str(tmp_path),
        artifacts=frozenset({PredictionArtifactType.PREDICTIONS}),
        num_workers=2, graph_snapshot_id='snapshot',
        source_config_sha256='2' * 64, forecast_options=options)

    if mismatch:
        with pytest.raises(ValueError, match='Forecast checkpoint'):
            prediction.score_prediction(**kwargs)
        assert not predicted
    else:
        prediction.score_prediction(**kwargs)
        assert len(predicted) == 2
    assert captured
    assert all({
        key: value
        for key, value in call.items() if key != 'name'
    } == options for call in captured)


@pytest.fixture(scope='module')
def predict_manifest(tmp_path_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumo_core.pquery import (
        build_generated_training_task_dataset,
        normalize_graph,
        parse_and_validate_query,
    )

    import kumo.engine as kumo
    from kumo.engine import FileConnector, Graph, PredictiveQuery, Table

    root = tmp_path_factory.mktemp('shards')
    kumo.configure(scratch_dir=root / 'scratch')
    (root / 'src').mkdir()
    pq.write_table(
        pa.table({
            'id': pa.array(range(50), pa.int64()),
            'label': [bool(i % 2) if i < 10 else None for i in range(50)],
        }), root / 'src' / 'users.parquet')
    graph = Graph([
        Table.from_source_table(
            FileConnector(root_dir=str(root / 'src'))['users'],
            primary_key='id')
    ])
    snapshot = graph.snapshot()
    table = PredictiveQuery(
        graph, 'PREDICT users.label FOR EACH users.id'
    ).generate_prediction_table(snapshot)
    native = normalize_graph(snapshot.graph_config)
    parsed = parse_and_validate_query(table.query, native)
    tt_id = f'{table.tt_id}-shards'
    request = prediction.PredictionTableRequest(
        query=parsed, graph=native, timestamps=table.timestamps,
        index_path=str(snapshot.index_path), tt_id=tt_id,
        task_dataset=build_generated_training_task_dataset(
            query=parsed, graph=native,
            graph_index_path=str(snapshot.index_path), run_id='shards',
            artifact_dir=str(root / 'run'), tt_id=tt_id,
            required_stages=table.required_stages, lag_timesteps=None,
            year_over_year=False))
    generated = prediction.GeneratedPredictionTable(
        staging_dir=str(table.staging_dir), table_path=str(table.table_path),
        row_count=table.row_count, tt_id=tt_id)
    return prediction.materialize_prediction_task(
        request, generated, graph_index_id=snapshot.id).manifest_path


@pytest.mark.parametrize('local_workers', [(1, 3), (2, 1, 3)])
def test_shards_with_different_loader_workers_cover_seeds_once(
        predict_manifest, local_workers):
    import torch
    from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule
    seeds = []
    for shard, workers in enumerate(local_workers):
        dataset = DiskgraphLightningDataModule(
            predict_manifest, {}, sampling_plan={'layers': []},
            train_batch_size=4, eval_batch_size=4,
            runtime_num_workers=len(local_workers), runtime_worker_id=shard,
            num_workers=workers,
            resolve_entity_node_type='users').predict_dataloader().dataset
        batches = list(
            torch.utils.data.DataLoader(dataset, batch_size=None,
                                        num_workers=workers))
        assert len(dataset) == len(batches)
        seeds += [
            seed for batch in batches
            for seed in batch['task']['tt_stage_node_ids'].to_pylist()
        ]
    population = DiskgraphTaskManifest.load(
        predict_manifest).stage_counts['predict']
    assert Counter(seeds) == Counter(range(population))
