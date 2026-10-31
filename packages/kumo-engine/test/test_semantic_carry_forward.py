# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from kumo_core.pquery import normalize_graph, parse_and_validate_query
from kumo_core.prediction_options import prediction_timestamps

from kumo.engine.connector.file import FileConnector
from kumo.engine.connector.gcs import GCSConnector
from kumo.engine.connector.s3 import S3Connector
from kumo.engine.pquery.prediction_table import (
    PredictionTable,
    _compile_prediction_table,
)
from kumo.engine.trainer._scoring import ScoringDefaults
from kumo.engine.trainer.trainer import Trainer, _stage_model_bundle

_SPLIT = SimpleNamespace(stage_counts={'test': 1})


@dataclass(frozen=True)
class _TaskDataset:
    job_id: str = 'generated-job'
    artifact_root: str = 'generated-artifacts'
    feature_schema = {'node_types': {}}


def _stub_generated_task_dataset_builder(monkeypatch):
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.build_generated_training_task_dataset',
        lambda *, run_id, artifact_dir, **_: _TaskDataset(
            job_id=run_id,
            artifact_root=str(Path(artifact_dir) / 'task_dataset'),
        ),
    )


def _best_trainer(log_dir):
    return SimpleNamespace(get_log_dir=lambda fit_mode: str(log_dir),
                           published_fit_mode=lambda: None)


def _write_prediction_shard(options, frame):
    from pathlib import Path

    from kumo_core.train.io import StreamingFrameWriter
    from kumo_core.types.enums import PredictionArtifactType

    kind = PredictionArtifactType.PREDICTIONS
    worker = options['partition_id']
    writer = StreamingFrameWriter(
        str(
            Path(options['output_dir']) / kind.output_file_name /
            f'part-{worker:05d}'),
        max_file_size=1 << 20,
        max_buffer_size=1 << 20,
        index=False,
        stats_sample_rows=1000,
    )
    with writer:
        writer.write(frame)
    options['artifact_stats_collector'].add_partition(worker,
                                                      {kind: writer.stats})


@pytest.mark.parametrize('connector_type',
                         [S3Connector, GCSConnector, FileConnector])
def test_object_store_format_comes_from_reader(connector_type, tmp_path):
    # A CSV directory need not end in .csv. Extension guessing loses the
    # format already resolved by the shared reader for a partitioned source.
    connector = (connector_type(root_dir=str(tmp_path))
                 if connector_type is FileConnector else connector_type())
    expected = object()
    backend = SimpleNamespace(get_table_config=Mock(
        return_value=SimpleNamespace(source_table=expected)))
    connector.__dict__['source_backend'] = backend
    assert connector.source_table_for('partitioned_events') is expected
    backend.get_table_config.assert_called_once_with(connector,
                                                     'partitioned_events')


def test_naive_prediction_time_is_utc_under_non_utc_host(monkeypatch):
    import os
    import time
    before = os.environ.get('TZ')
    try:
        monkeypatch.setenv('TZ', 'America/Los_Angeles')
        time.tzset()
        value = datetime(2026, 1, 1)
        assert prediction_timestamps(value) == (1767225600000000000, )
    finally:
        if before is None:
            monkeypatch.delenv('TZ', raising=False)
        else:
            monkeypatch.setenv('TZ', before)
        time.tzset()


@pytest.mark.parametrize('value, message', [
    (datetime(2026, 1, 1, tzinfo=timezone.utc), 'timezone'),
    (datetime(2026, 1, 1), 'static'),
])
def test_compile_rejects_invalid_prediction_time(snapshot, tmp_path, value,
                                                 message, monkeypatch):
    generate = Mock(side_effect=AssertionError('must not generate'))
    monkeypatch.setattr(
        'kumo.engine.pquery.prediction_table.generate_prediction_table',
        generate)
    parsed = parse_and_validate_query('PREDICT users.label FOR EACH users.id',
                                      normalize_graph(snapshot.graph_config))
    with pytest.raises(ValueError, match=message):
        _compile_prediction_table(snapshot, parsed, prediction_time=value,
                                  run_id='test', scratch_dir=tmp_path,
                                  artifact_dir=tmp_path)
    generate.assert_not_called()


@pytest.mark.parametrize('query, threshold, message', [
    ('PREDICT users.age FOR EACH users.id', .5, 'target/entity'),
    ('PREDICT users.label FOR EACH users.id', 1.1, 'between 0 and 1'),
])
def test_prediction_admission_before_materialization(snapshot, tmp_path, query,
                                                     threshold, message,
                                                     monkeypatch):
    from kumoapi.model_plan import ModelPlan
    trainer = Trainer(ModelPlan())
    trainer._training_result = SimpleNamespace(scoring_defaults=ScoringDefaults())
    trainer._snapshot_id = snapshot.id
    trainer._training_query = 'PREDICT users.label FOR EACH users.id'
    materialize = Mock(side_effect=AssertionError('must not materialize'))
    monkeypatch.setattr('kumo_core.prediction.materialize_prediction_task',
                        materialize)
    table = PredictionTable(query, tmp_path, tmp_path, 0, 'tt-1', (), ())
    with pytest.raises(ValueError, match=message):
        trainer.predict(table, snapshot=snapshot,
                        output_config=SimpleNamespace(),
                        binary_classification_threshold=threshold)
    materialize.assert_not_called()


def test_binary_predictions_preserve_probabilities_and_metadata_contract(
        snapshot, tmp_path, monkeypatch):
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.task_dataset import DiskgraphTaskManifest
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    from kumoapi.jobs import MetadataField
    from kumoapi.jobs import PredictionArtifactType as ApiType
    from kumoapi.model_plan import ModelPlan

    from kumo.engine.trainer.output_config import PredictionOutputConfig
    trainer = Trainer(ModelPlan())
    _stub_generated_task_dataset_builder(monkeypatch)
    monkeypatch.setattr(type(trainer), 'scratch_dir',
                        property(lambda _: tmp_path))
    trainer._training_result = SimpleNamespace(
        artifact_paths={}, manifest=None, scoring_defaults=ScoringDefaults())
    trainer._snapshot_id = snapshot.id
    trainer._training_query = 'PREDICT users.label FOR EACH users.id'
    table = PredictionTable(trainer._training_query, tmp_path, tmp_path, 0,
                            'tt-1', (), ())
    sink = SimpleNamespace(write_files=Mock())
    output = PredictionOutputConfig(SimpleNamespace(sink_backend=sink),
                                    ApiType.PREDICTIONS, 'out',
                                    extra_fields=[MetadataField.JOB_TIMESTAMP])
    monkeypatch.setattr(
        'kumo_core.prediction.materialize_prediction_task', lambda *a, **k:
        SimpleNamespace(manifest_path='task', bundle_dir='bundle'))
    monkeypatch.setattr('kumo.engine.trainer.trainer._stage_model_bundle',
                        lambda *a: tmp_path)
    (tmp_path / 'config.json').write_text('{}')
    monkeypatch.setattr(
        DiskgraphTaskManifest, 'load', lambda *a: SimpleNamespace(
            task_type='binary_classification', stage_counts={'predict': 3}))
    monkeypatch.setattr(ResolvedNumericalTask, 'from_manifest',
                        lambda *a, **k: None)
    monkeypatch.setattr('kumo_core.train.trainer.load_encoders',
                        lambda *a: None)
    calls = []

    def predict(**kwargs):
        calls.append(kwargs)
        _write_prediction_shard(
            kwargs,
            pd.DataFrame({
                'True_PROB': [.6, .8, None],
                'TARGET_PRED': [False, True, None]
            }))

    def load(*args, **kwargs):
        assert kwargs['binary_threshold'] == .7
        return SimpleNamespace(predict=predict)

    monkeypatch.setattr(DiskgraphNestedTrainer, 'load', load)
    result = trainer.predict(table, snapshot=snapshot, output_config=output,
                             binary_classification_threshold=.7)
    assert result.frame['TARGET_PRED'].iloc[:2].tolist() == [False, True]
    assert pd.isna(result.frame['TARGET_PRED'].iloc[2])
    assert result.frame['True_PROB'].iloc[:2].tolist() == [.6, .8]
    assert calls[0]['output_extra_fields'] == {'PREDICTIONS': ['JOB_TIMESTAMP']}
    assert calls[0]['job_start_time'].tzinfo is timezone.utc
    sink.write_files.assert_called_once_with(
        output.connector, output.table_name, list(result.files),
        output.write_mode, export_id=f'{result.selection.job_id}-PREDICTIONS')


def test_predict_reuses_generated_table_without_recompiling(
        snapshot, tmp_path, monkeypatch) -> None:
    r"""`Trainer.predict` must consume `prediction_table` as already
    compiled -- its generated rows and timestamps are carried straight into
    `materialize_prediction_task`, while task metadata comes from the graph
    selected for this prediction call. Its
    `table_path`/`staging_dir`/`row_count`/`tt_id` reconstructing the exact
    `GeneratedPredictionTable` `generate_prediction_table` produced --
    rather than recompiling the query against a fresh (and possibly
    different) anchor time. Regression test for the review's item 5.
    """
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.task_dataset import DiskgraphTaskManifest
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    from kumoapi.jobs import PredictionArtifactType as ApiType
    from kumoapi.model_plan import ModelPlan

    from kumo.engine.trainer.output_config import PredictionOutputConfig

    sentinel_timestamps = (1767225600000000000, )
    query = 'PREDICT users.label FOR EACH users.id'
    table = PredictionTable(query, tmp_path, tmp_path, 3, 'tt-fixed',
                            sentinel_timestamps, ())

    selected_config = copy.deepcopy(snapshot.graph_config)
    selected_config['tables'][0]['source_table_name'] = (
        'other.schema.selected_users')
    selected_config['tables'][0]['cols'][0]['encoder'] = {'name': 'Index'}
    selected_config['tables'][0]['cols'][1]['encoder'] = {'name': 'Numerical'}
    selected_config['tables'][0]['cols'].append({
        'name': 'income',
        'dtype': 'float64',
        'stype': 'numerical',
        'encoder': {
            'name': 'Numerical'
        },
    })
    from kumo.engine.graph import GraphSnapshot
    selected = GraphSnapshot(
        'independent-graph@selected-data',
        tmp_path / 'selected-index',
        selected_config,
    )

    def stats(index_path, table_name, mode):
        assert str(index_path) == str(selected.index_path)
        assert table_name == 'users'
        assert mode == 'full'
        return json.dumps({
            'label__CATEGORY_COUNTS': [],
            'age__MEAN': 42,
            'income__MEAN': 100,
        })

    monkeypatch.setattr('kumo_core.pquery.read_col_stats', stats)

    trainer = Trainer(ModelPlan())
    monkeypatch.setattr(type(trainer), 'scratch_dir',
                        property(lambda _: tmp_path))
    trainer._training_result = SimpleNamespace(
        artifact_paths={}, manifest=None, scoring_defaults=ScoringDefaults())
    trainer._snapshot_id = snapshot.id
    trainer._training_query = query

    calls = []

    def fake_materialize(request, generated, *, graph_index_id, **kwargs):
        calls.append((request, generated, graph_index_id))
        return SimpleNamespace(manifest_path='task', bundle_dir='bundle')

    monkeypatch.setattr('kumo_core.prediction.materialize_prediction_task',
                        fake_materialize)
    monkeypatch.setattr('kumo.engine.trainer.trainer._stage_model_bundle',
                        lambda *a: tmp_path)
    (tmp_path / 'config.json').write_text('{}')
    monkeypatch.setattr(
        DiskgraphTaskManifest, 'load', lambda *a: SimpleNamespace(
            task_type='binary_classification', stage_counts={'predict': 3}))
    monkeypatch.setattr(ResolvedNumericalTask, 'from_manifest',
                        lambda *a, **k: None)
    monkeypatch.setattr('kumo_core.train.trainer.load_encoders',
                        lambda *a: None)

    def fake_predict(**kwargs) -> None:
        _write_prediction_shard(
            kwargs, pd.DataFrame({
                'True_PROB': [.6],
                'TARGET_PRED': [True],
            }))

    monkeypatch.setattr(DiskgraphNestedTrainer, 'load',
                        lambda *a, **k: SimpleNamespace(predict=fake_predict))

    sink = SimpleNamespace(write_files=Mock())
    output = PredictionOutputConfig(SimpleNamespace(sink_backend=sink),
                                    ApiType.PREDICTIONS, 'out')
    trainer.predict(table, snapshot=selected, output_config=output,
                    binary_classification_threshold=.5)

    assert len(calls) == 1
    request, generated, graph_index_id = calls[0]
    assert request.task_dataset.job_id in request.tt_id
    assert Path(request.task_dataset.artifact_root).name == 'task_dataset'
    assert Path(request.task_dataset.artifact_root).parent.name == (
        request.task_dataset.job_id)
    assert request.timestamps == sentinel_timestamps
    # The ingest target is a fresh id scoped to this `predict()` call, not
    # the table's own (reusable, generation-time) `tt_id` -- see
    # `PredictionTable.tt_id`'s docstring: reusing it here would collide
    # with another `predict()` call against the same table in DiskGraph's
    # unlocked `index_path/train_tables/<tt_id>` extension path.
    assert request.tt_id != 'tt-fixed'
    assert request.tt_id == generated.tt_id
    assert request.index_path == str(selected.index_path)
    assert graph_index_id == selected.id
    selected_features = request.task_dataset.feature_schema['node_types'][
        'users']['feature_columns']
    assert selected_features == ['age', 'income']
    assert generated.staging_dir == str(table.staging_dir)
    assert generated.table_path == str(table.table_path)
    assert generated.row_count == 3

    assert not hasattr(table, 'snapshot')
    assert not hasattr(table, 'graph_index_id')
    assert not hasattr(table, 'task_dataset')


def test_predict_requires_an_explicit_snapshot(tmp_path) -> None:
    from kumoapi.model_plan import ModelPlan

    table = PredictionTable(
        'PREDICT users.label FOR EACH users.id',
        tmp_path,
        tmp_path,
        0,
        'tt-fixed',
        (),
        (),
    )
    with pytest.raises(TypeError, match='snapshot'):
        Trainer(ModelPlan()).predict(table)


def test_fit_consumes_generated_rows_with_fresh_task_metadata(
        snapshot, tmp_path, staged_seed_dir, monkeypatch) -> None:
    r"""Fit reuses generated rows and settings but builds task metadata fresh."""
    from kumo_core.model_plan import ModelPlan as CoreModelPlan
    from kumo_core.model_plan import TrainingTableGenerationPlan
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumoapi.model_plan import ModelPlan

    from kumo.engine.pquery.training_table import TrainingTable

    # Keep the sentinel lag setting: this adapter test mocks materialization
    # and checks metadata forwarding, not static-task model-plan admission.
    monkeypatch.setattr('kumo_core.training.workflow.resolve_model_plan',
                        lambda *args, **kwargs: None)

    generation_plan = TrainingTableGenerationPlan(lag_timesteps=3)
    train_table = TrainingTable(
        query='PREDICT users.label FOR EACH users.id',
        train_table_path=tmp_path / 'train.parquet',
        manifest_path=tmp_path / 'manifest.json',
        staging_dir=staged_seed_dir,
        required_stages=('train', 'val', 'test'),
        generation_plan=generation_plan,
    )
    assert not hasattr(train_table, 'plan')

    trainer = Trainer(ModelPlan())
    monkeypatch.setattr(type(trainer), 'scratch_dir',
                        property(lambda _: tmp_path))

    calls = []
    built = []

    def fake_build(**kwargs):
        built.append(kwargs)
        return _TaskDataset(
            job_id=kwargs['run_id'],
            artifact_root=str(Path(kwargs['artifact_dir']) / 'task_dataset'))

    def fake_materialize(request, task_dataset, generated, **kwargs):
        calls.append((request, task_dataset, generated, kwargs.get('tt_id')))
        return SimpleNamespace(task=Mock(spec=ResolvedNumericalTask, split=_SPLIT),
                               model_plan=CoreModelPlan(),
                               bundle_dir=str(tmp_path))

    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.build_generated_training_task_dataset',
        fake_build)
    monkeypatch.setattr(
        'kumo_core.training.workflow.materialize_generated_training_task',
        fake_materialize)
    monkeypatch.setattr(
        'kumo_core.training.orchestration.run_callable_training',
        lambda training_input, **kwargs: SimpleNamespace(
            metrics={},
            artifact_paths={},
            explain_disabled=True,
            best_trainer=_best_trainer(tmp_path),
        ))
    monkeypatch.setattr('kumo.engine.trainer.trainer.write_training_spec',
                        lambda *args, **kwargs: None)

    replacement = ModelPlan()
    replacement.optimization.max_epochs = 7
    trainer.model_plan = replacement
    trainer.fit(train_table, snapshot=snapshot)

    assert len(calls) == len(built) == 1
    request, task_dataset, generated, tt_id = calls[0]
    assert trainer.model_plan is replacement
    assert request.model_plan.optimization.max_epochs == 7
    assert task_dataset.job_id in tt_id
    assert Path(task_dataset.artifact_root).parent.name == task_dataset.job_id
    assert built[0]['required_stages'] == ('train', 'val', 'test')
    assert built[0]['graph_index_path'] == str(snapshot.index_path)
    assert tt_id == built[0]['tt_id']
    assert generated.train_table_path == str(train_table.train_table_path)
    assert generated.manifest_path == str(train_table.manifest_path)
    assert generated.staging_dir == str(train_table.staging_dir)
    assert request.training_table_generation_plan is generation_plan
    assert request.model_plan.training_table_generation == \
        CoreModelPlan().training_table_generation

    with pytest.raises(TypeError, match='snapshot'):
        trainer.fit(train_table)


@pytest.mark.parametrize(
    ('updated_snapshot_id', 'add_feature', 'expected_features'), [
        ('snapshot@new-data', False, ['label', 'age']),
        ('other-graph@new-data', True, ['label', 'age', 'income']),
    ])
def test_fit_materializes_existing_rows_against_explicit_snapshot(
        snapshot, tmp_path, staged_seed_dir, monkeypatch, updated_snapshot_id,
        add_feature, expected_features) -> None:
    r"""An explicit snapshot supplies graph features, not train-table rows."""
    from kumo_core.model_plan import ModelPlan as CoreModelPlan
    from kumo_core.model_plan import TrainingTableGenerationPlan
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumoapi.model_plan import ModelPlan

    from kumo.engine.graph import GraphSnapshot
    from kumo.engine.pquery.training_table import TrainingTable

    original_config = copy.deepcopy(snapshot.graph_config)
    original_config['tables'][0]['cols'][0]['encoder'] = {'name': 'Index'}
    original_config['tables'][0]['cols'][1]['encoder'] = {'name': 'Numerical'}
    original_config['tables'].append({
        'table_name':
        'orders',
        'source_table_name':
        'db.schema.orders',
        'pkey': {
            'name': 'order_id',
            'dtype': 'int64',
            'stype': 'ID',
        },
        'time_col': {
            'name': 'event_time',
            'dtype': 'date',
            'stype': 'timestamp',
        },
        'cols': [],
        'fkeys': [{
            'name': 'user_id',
            'dtype': 'int64',
            'stype': 'ID',
            'dst_table': 'users',
            'dst_key': 'id',
        }],
    })
    updated_config = copy.deepcopy(original_config)
    if add_feature:
        updated_config['tables'][0]['cols'].append({
            'name': 'income',
            'dtype': 'float64',
            'stype': 'numerical',
            'encoder': {
                'name': 'Numerical'
            },
        })
    updated_snapshot = GraphSnapshot(updated_snapshot_id,
                                     tmp_path / 'updated-index', updated_config)

    def stats(index_path, table_name, mode):
        assert mode == 'full'
        if table_name == 'orders':
            return json.dumps({
                'event_time__NA_FRACTION__False': 0.0,
                'event_time__MIN__False': 0,
                'event_time__MAX__False': 1_000,
            })
        assert table_name == 'users'
        values = {'age__MEAN': 42, 'label__CATEGORY_COUNTS': []}
        if add_feature and str(index_path) == str(updated_snapshot.index_path):
            values['income__MEAN'] = 100
        return json.dumps(values)

    monkeypatch.setattr('kumo_core.pquery.read_col_stats', stats)
    generation_plan = TrainingTableGenerationPlan(lag_timesteps=3)
    train_table = TrainingTable(
        query='PREDICT COUNT(orders.*, 0, 1, days) FOR EACH users.id',
        train_table_path=tmp_path / 'rows.parquet',
        manifest_path=tmp_path / 'rows-manifest.json',
        staging_dir=staged_seed_dir,
        required_stages=('train', 'val', 'test'),
        generation_plan=generation_plan,
    )
    assert not hasattr(train_table, 'plan')
    original = copy.deepcopy(train_table)
    calls = []

    def fake_materialize(request, task_dataset, generated, **kwargs):
        calls.append((request, task_dataset, generated))
        return SimpleNamespace(task=Mock(spec=ResolvedNumericalTask, split=_SPLIT),
                               model_plan=CoreModelPlan(),
                               bundle_dir=str(tmp_path))

    monkeypatch.setattr(
        'kumo_core.training.workflow.materialize_generated_training_task',
        fake_materialize)
    monkeypatch.setattr(
        'kumo_core.training.orchestration.run_callable_training',
        lambda training_input, **kwargs: SimpleNamespace(
            metrics={},
            artifact_paths={},
            explain_disabled=True,
            best_trainer=_best_trainer(tmp_path),
        ))
    monkeypatch.setattr('kumo.engine.trainer.trainer.write_training_spec',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(Trainer, 'scratch_dir',
                        property(lambda _: tmp_path / 'trainer'))

    trainer = Trainer(ModelPlan())
    trainer.fit(train_table, snapshot=updated_snapshot)

    assert train_table == original
    assert len(calls) == 1
    request, task_dataset, generated = calls[0]
    assert request.prepared_graph.index_id == updated_snapshot.id
    assert request.prepared_graph.index_path == str(updated_snapshot.index_path)
    features = task_dataset.feature_schema['node_types']['users']
    assert features['feature_columns'] == expected_features
    assert generated.train_table_path == str(train_table.train_table_path)
    assert generated.manifest_path == str(train_table.manifest_path)
    assert generated.staging_dir == str(train_table.staging_dir)
    assert task_dataset.task_family == 'node_prediction'
    assert task_dataset.is_temporal is True
    assert task_dataset.required_stages == train_table.required_stages
    assert trainer._snapshot_id == updated_snapshot.id


def test_fit_uses_selected_graph_task_semantics_without_regenerating_rows(
        snapshot, tmp_path, staged_seed_dir, monkeypatch) -> None:
    from kumo_core.model_plan import ModelPlan as CoreModelPlan
    from kumo_core.model_plan import TrainingTableGenerationPlan
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumoapi.model_plan import ModelPlan

    from kumo.engine.graph import GraphSnapshot
    from kumo.engine.pquery.training_table import TrainingTable

    monkeypatch.setattr('kumo_core.pquery.read_col_stats',
                        lambda *args: '{"age__MEAN": 42}')
    original_config = copy.deepcopy(snapshot.graph_config)
    original_config['tables'][0]['cols'][1]['encoder'] = {'name': 'Numerical'}
    original_snapshot = GraphSnapshot(snapshot.id, snapshot.index_path,
                                      original_config)
    train_table = TrainingTable(
        query='PREDICT users.label FOR EACH users.id',
        train_table_path=tmp_path / 'rows.parquet',
        manifest_path=tmp_path / 'rows-manifest.json',
        staging_dir=staged_seed_dir,
        required_stages=('train', 'val', 'test'),
        generation_plan=TrainingTableGenerationPlan(),
    )
    assert not hasattr(train_table, 'plan')
    incompatible_config = copy.deepcopy(original_snapshot.graph_config)
    incompatible_config['tables'][0]['cols'][0].update({
        'dtype': 'float64',
        'stype': 'numerical',
    })
    selected = GraphSnapshot('other@data', tmp_path / 'other-index',
                             incompatible_config)
    captured = {}

    def fake_materialize(request, task_dataset, generated, **kwargs):
        captured.update(request=request, task_dataset=task_dataset,
                        generated=generated)
        return SimpleNamespace(task=Mock(spec=ResolvedNumericalTask, split=_SPLIT),
                               model_plan=CoreModelPlan(),
                               bundle_dir=str(tmp_path))

    monkeypatch.setattr(
        'kumo_core.training.workflow.materialize_generated_training_task',
        fake_materialize)
    monkeypatch.setattr(
        'kumo_core.training.orchestration.run_callable_training',
        lambda training_input, **kwargs: SimpleNamespace(
            metrics={},
            artifact_paths={},
            explain_disabled=True,
            best_trainer=_best_trainer(tmp_path),
        ))
    monkeypatch.setattr('kumo.engine.trainer.trainer.write_training_spec',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(Trainer, 'scratch_dir',
                        property(lambda _: tmp_path / 'trainer'))

    Trainer(ModelPlan()).fit(train_table, snapshot=selected)

    task_dataset = captured['task_dataset']
    assert task_dataset.task_type == 'regression'
    assert captured['request'].prepared_graph.index_id == selected.id
    assert captured['request'].prepared_graph.index_path == str(
        selected.index_path)
    assert captured['generated'].staging_dir == str(train_table.staging_dir)


def test_repeated_predict_calls_get_distinct_ingest_targets(
        snapshot, tmp_path, monkeypatch) -> None:
    r"""Two `predict()` calls against the same `PredictionTable` -- from one
    `Trainer` called twice, or from two different `Trainer`s -- must not
    publish into the same DiskGraph `index_path/train_tables/<tt_id>`
    extension: that write has no lock or reuse check, so two calls reusing
    the table's own (generation-time, meant-to-be-shared) `tt_id` would race
    or clobber each other. The ingest `tt_id` is `<pquery tt_id>-<this
    predict() call's own run id>` -- unlike `fit()` (called once per
    `Trainer` by convention), `predict()` is routinely called more than
    once on one `Trainer`, so `self._id` alone wouldn't disambiguate
    repeated calls on the same `Trainer`. Regression test for the P1 review
    finding on items 4/5.
    """
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.task_dataset import DiskgraphTaskManifest
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    from kumoapi.jobs import PredictionArtifactType as ApiType
    from kumoapi.model_plan import ModelPlan

    from kumo.engine.trainer.output_config import PredictionOutputConfig

    query = 'PREDICT users.label FOR EACH users.id'
    _stub_generated_task_dataset_builder(monkeypatch)
    table = PredictionTable(query, tmp_path, tmp_path, 3, 'tt-fixed', (), ())

    tt_ids = []
    task_datasets = []
    model_dirs = []

    def fake_materialize(request, generated, *, graph_index_id, **kwargs):
        tt_ids.append(request.tt_id)
        task_datasets.append(request.task_dataset)
        return SimpleNamespace(manifest_path='task', bundle_dir='bundle')

    monkeypatch.setattr('kumo_core.prediction.materialize_prediction_task',
                        fake_materialize)

    def fake_stage(_artifacts, destination):
        model_dirs.append(destination)
        destination.mkdir(parents=True)
        (destination / 'config.json').write_text('{}')
        return destination

    monkeypatch.setattr('kumo.engine.trainer.trainer._stage_model_bundle',
                        fake_stage)
    monkeypatch.setattr(
        DiskgraphTaskManifest, 'load', lambda *a: SimpleNamespace(
            task_type='binary_classification', stage_counts={'predict': 3}))
    monkeypatch.setattr(ResolvedNumericalTask, 'from_manifest',
                        lambda *a, **k: None)
    monkeypatch.setattr('kumo_core.train.trainer.load_encoders',
                        lambda *a: None)

    def fake_predict(**kwargs) -> None:
        _write_prediction_shard(
            kwargs, pd.DataFrame({
                'True_PROB': [.6],
                'TARGET_PRED': [True],
            }))

    monkeypatch.setattr(DiskgraphNestedTrainer, 'load',
                        lambda *a, **k: SimpleNamespace(predict=fake_predict))

    sink = SimpleNamespace(write_files=Mock())
    output = PredictionOutputConfig(SimpleNamespace(sink_backend=sink),
                                    ApiType.PREDICTIONS, 'out')

    def run_predict() -> None:
        trainer = Trainer(ModelPlan())
        monkeypatch.setattr(type(trainer), 'scratch_dir',
                            property(lambda _: tmp_path))
        trainer._training_result = SimpleNamespace(
            artifact_paths={}, manifest=None,
            scoring_defaults=ScoringDefaults())
        trainer._snapshot_id = snapshot.id
        trainer._training_query = query
        trainer.predict(table, snapshot=snapshot, output_config=output,
                        binary_classification_threshold=.5)

    # One Trainer called twice against the same table, and a different
    # Trainer against that same table again -- none of the three should
    # collide, and none should reuse the table's own tt_id verbatim.
    same_trainer = Trainer(ModelPlan())
    monkeypatch.setattr(type(same_trainer), 'scratch_dir',
                        property(lambda _: tmp_path))
    same_trainer._training_result = SimpleNamespace(
        artifact_paths={}, manifest=None, scoring_defaults=ScoringDefaults())
    same_trainer._snapshot_id = snapshot.id
    same_trainer._training_query = query
    same_trainer.predict(table, snapshot=snapshot, output_config=output,
                         binary_classification_threshold=.5)
    same_trainer.predict(table, snapshot=snapshot, output_config=output,
                         binary_classification_threshold=.5)
    run_predict()

    assert len(tt_ids) == 3
    assert len(set(tt_ids)) == 3, f"collided ingest targets: {tt_ids}"
    assert 'tt-fixed' not in tt_ids
    assert len({item.job_id for item in task_datasets}) == 3
    assert all(
        len(item.job_id.rsplit('-', 1)[-1]) == 32 for item in task_datasets)
    assert len({item.artifact_root for item in task_datasets}) == 3
    assert len(set(model_dirs)) == 3
    for tt_id, task_dataset, model_dir in zip(tt_ids, task_datasets,
                                              model_dirs):
        assert task_dataset.job_id in tt_id
        assert Path(
            task_dataset.artifact_root).parent.name == (task_dataset.job_id)
        assert model_dir.parent.name == task_dataset.job_id


def test_repeated_fit_calls_get_distinct_ingest_targets(snapshot, tmp_path,
                                                        staged_seed_dir,
                                                        monkeypatch) -> None:
    r"""Every fit call owns its ingest and task-dataset coordinates."""
    from kumo_core.model_plan import ModelPlan as CoreModelPlan
    from kumo_core.model_plan import TrainingTableGenerationPlan
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumoapi.model_plan import ModelPlan

    from kumo.engine.pquery.training_table import TrainingTable

    # Keep the sentinel lag setting: this adapter test mocks materialization
    # and checks metadata forwarding, not static-task model-plan admission.
    monkeypatch.setattr('kumo_core.training.workflow.resolve_model_plan',
                        lambda *args, **kwargs: None)

    train_table = TrainingTable(
        query='PREDICT users.label FOR EACH users.id',
        train_table_path=tmp_path / 'train.parquet',
        manifest_path=tmp_path / 'manifest.json',
        staging_dir=staged_seed_dir,
        required_stages=('train', 'val', 'test'),
        generation_plan=TrainingTableGenerationPlan(lag_timesteps=3),
    )
    original = copy.deepcopy(train_table)

    tt_ids = []
    task_datasets = []

    def fake_build(**kwargs):
        return _TaskDataset(
            job_id=kwargs['run_id'],
            artifact_root=str(Path(kwargs['artifact_dir']) / 'task_dataset'))

    def fake_materialize(request, task_dataset, generated, **kwargs):
        tt_ids.append(kwargs.get('tt_id'))
        task_datasets.append(task_dataset)
        return SimpleNamespace(task=Mock(spec=ResolvedNumericalTask, split=_SPLIT),
                               model_plan=CoreModelPlan(),
                               bundle_dir=str(tmp_path))

    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.build_generated_training_task_dataset',
        fake_build)
    monkeypatch.setattr(
        'kumo_core.training.workflow.materialize_generated_training_task',
        fake_materialize)
    monkeypatch.setattr(
        'kumo_core.training.orchestration.run_callable_training',
        lambda training_input, **kwargs: SimpleNamespace(
            metrics={},
            artifact_paths={},
            explain_disabled=True,
            best_trainer=_best_trainer(tmp_path),
        ))
    monkeypatch.setattr('kumo.engine.trainer.trainer.write_training_spec',
                        lambda *args, **kwargs: None)

    # One Trainer called twice and an independent Trainer reuse the same
    # generated table. All calls need distinct mutable destinations.
    trainer1 = Trainer(ModelPlan())
    monkeypatch.setattr(type(trainer1), 'scratch_dir',
                        property(lambda _: tmp_path))
    trainer1.fit(train_table, snapshot=snapshot)
    trainer1.fit(train_table, snapshot=snapshot)
    assert len(tt_ids) == 2
    assert tt_ids[0] != tt_ids[1]
    assert 'tt-1' not in tt_ids

    # A different Trainer fit()-ing the same table gets a distinct id.
    trainer2 = Trainer(ModelPlan())
    monkeypatch.setattr(type(trainer2), 'scratch_dir',
                        property(lambda _: tmp_path))
    trainer2.fit(train_table, snapshot=snapshot)
    assert tt_ids[2] not in tt_ids[:2]
    assert 'tt-1' not in tt_ids
    assert len({item.job_id for item in task_datasets}) == 3
    assert all(
        len(item.job_id.rsplit('-', 1)[-1]) == 32 for item in task_datasets)
    assert len({item.artifact_root for item in task_datasets}) == 3
    for tt_id, task_dataset in zip(tt_ids, task_datasets):
        assert task_dataset.job_id in tt_id
        assert Path(
            task_dataset.artifact_root).parent.name == (task_dataset.job_id)
    assert train_table == original
    assert not hasattr(train_table, 'plan')


def test_concurrent_to_thread_fits_keep_loop_live_and_propagate_results(
        snapshot, tmp_path, staged_seed_dir, monkeypatch) -> None:
    from kumo_core.model_plan import ModelPlan as CoreModelPlan
    from kumo_core.model_plan import TrainingTableGenerationPlan
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumoapi.model_plan import ModelPlan

    from kumo.engine.pquery.training_table import TrainingTable

    # Keep the sentinel lag setting: this adapter test mocks materialization
    # and checks metadata forwarding, not static-task model-plan admission.
    monkeypatch.setattr('kumo_core.training.workflow.resolve_model_plan',
                        lambda *args, **kwargs: None)

    train_table = TrainingTable(
        query='PREDICT users.label FOR EACH users.id',
        train_table_path=tmp_path / 'train.parquet',
        manifest_path=tmp_path / 'manifest.json',
        staging_dir=staged_seed_dir,
        required_stages=('train', 'val', 'test'),
        generation_plan=TrainingTableGenerationPlan(lag_timesteps=3),
    )
    records = []
    records_lock = threading.Lock()

    def fake_build(**kwargs):
        return _TaskDataset(
            job_id=kwargs['run_id'],
            artifact_root=str(Path(kwargs['artifact_dir']) / 'task_dataset'))

    def fake_materialize(request, task_dataset, generated, **kwargs):
        with records_lock:
            records.append((kwargs['tt_id'], task_dataset, request.run_id))
        return SimpleNamespace(task=Mock(spec=ResolvedNumericalTask, split=_SPLIT),
                               model_plan=CoreModelPlan(),
                               bundle_dir=str(tmp_path))

    both_workers = threading.Barrier(2)

    def fake_train(training_input, **kwargs):
        both_workers.wait(timeout=5)
        time.sleep(.05)
        run_name = Path(training_input.home_dir).name
        return SimpleNamespace(metrics={'run_name': run_name},
                               artifact_paths={}, explain_disabled=True,
                               best_trainer=_best_trainer(tmp_path))

    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.build_generated_training_task_dataset',
        fake_build)
    monkeypatch.setattr(
        'kumo_core.training.workflow.materialize_generated_training_task',
        fake_materialize)
    monkeypatch.setattr(
        'kumo_core.training.orchestration.run_callable_training', fake_train)
    monkeypatch.setattr('kumo.engine.trainer.trainer.write_training_spec',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(Trainer, 'scratch_dir', property(lambda _: tmp_path))
    trainers = [Trainer(ModelPlan()), Trainer(ModelPlan())]

    async def exercise():
        ticks = 0
        finished = asyncio.Event()

        async def heartbeat():
            nonlocal ticks
            while not finished.is_set():
                ticks += 1
                await asyncio.sleep(.001)

        pulse = asyncio.create_task(heartbeat())
        try:
            results = await asyncio.gather(
                *(asyncio.to_thread(trainer.fit, train_table, snapshot=snapshot)
                  for trainer in trainers))
        finally:
            finished.set()
            await pulse
        return results, ticks

    results, heartbeat_ticks = asyncio.run(exercise())

    assert heartbeat_ticks > 1
    assert len(records) == 2
    tt_ids = [record[0] for record in records]
    task_datasets = [record[1] for record in records]
    run_ids = [record[2] for record in records]
    assert len(set(tt_ids)) == len(set(run_ids)) == 2
    assert len({item.job_id for item in task_datasets}) == 2
    assert len({item.artifact_root for item in task_datasets}) == 2
    assert {result.metrics['run_name'] for result in results} == set(run_ids)
    assert len({result.bundle_dir for result in results}) == 2
    assert all(trainer._training_result is result
               for trainer, result in zip(trainers, results))


def test_model_bundle_uses_shared_layout_and_requires_checkpoint(tmp_path):
    checkpoint = tmp_path / 'original.ckpt'
    checkpoint.write_bytes(b'checkpoint')
    model = _stage_model_bundle({'selected_checkpoint': str(checkpoint)},
                                tmp_path / 'bundle')
    assert (model / 'checkpoint/step=0.ckpt').read_bytes() == b'checkpoint'
    with pytest.raises(ValueError, match='checkpoint'):
        _stage_model_bundle({}, tmp_path / 'missing')
