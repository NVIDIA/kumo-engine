# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.model_plan import ModelPlan as CoreModelPlan
from kumo_core.model_plan_conversion import public_model_plan_from_internal
from kumo_core.pquery import target_to_entity_paths, to_plan_graph
from kumo_core.types import TaskType
from kumoapi.graph import Edge
from kumoapi.json_serde import to_json_dict
from kumoapi.model_plan import ModelPlan, RunMode, TrainingTableGenerationPlan
from kumoapi.pquery import QueryType
from kumoapi.source_table import (
    FileType,
    LocalFileSourceTable,
    S3RelativeSource,
    S3SourceTable,
)
from kumoapi.table import Column, TableDefinition
from kumoapi.train import TrainingTableSpec
from kumoapi.typing import Dtype, Stype
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem

from kumo.engine import (
    FileConnector,
    Graph,
    PredictionTable,
    PredictiveQuery,
    Table,
    Trainer,
    TrainingTable,
    configure,
)
from kumo.engine._local_schema import TableDefinition as LocalTableDefinition
from kumo.engine.connector.s3 import S3Connector
from kumo.engine.graph import GraphSnapshot
from kumo.engine.trainer.trainer import _to_core_model_plan


@pytest.mark.parametrize('connector_kind', ['file', 's3'])
def test_classify_top_k_boundaries_on_generated_table(connector_kind, tmp_path,
                                                      monkeypatch):
    storage = tmp_path / 'storage'
    source = storage / 'bucket'
    source.mkdir(parents=True)
    pq.write_table(
        pa.table({
            'id':
            pa.array(range(120), type=pa.int64()),
            'labels':
            pa.array([['a', 'b'] if i % 3 == 0 else ['b', 'c'] if i %
                      3 == 1 else ['a', 'c'] for i in range(120)]),
        }), source / 'users.parquet')
    configure(scratch_dir=str(tmp_path / 'scratch'))
    if connector_kind == 'file':
        connector = FileConnector(root_dir=str(source))
    else:

        class Filesystem(SubTreeFileSystem):
            def __init__(self, **kwargs):
                super().__init__(str(storage), LocalFileSystem())

            @property
            def region(self):
                return 'us-east-1'

        monkeypatch.setattr('kumo_s3.session.fs.S3FileSystem', Filesystem)
        monkeypatch.setattr('kumo_s3.backend.S3FileSystem', Filesystem)

        def download_file(bucket, key, target, **options):
            Path(target).write_bytes((storage / bucket / key).read_bytes())

        session = Mock(region_name='us-east-1')
        session.get_credentials.return_value = None
        session.client.return_value.download_file.side_effect = download_file
        monkeypatch.setattr(boto3, 'Session', Mock(return_value=session))
        connector = S3Connector(root_dir='s3://bucket', region='us-east-1')
    graph = Graph(
        [Table.from_source_table(connector['users'], primary_key='id')])

    for top_k in (0, -1):
        query = PredictiveQuery(
            graph, f'PREDICT users.labels CLASSIFY TOP {top_k} '
            'FOR EACH users.id')
        with pytest.raises(ValueError, match=f'got {top_k}'):
            query.validate()

    snapshot = graph.snapshot()
    oversized = PredictiveQuery(
        graph, 'PREDICT users.labels CLASSIFY TOP 4 FOR EACH users.id')
    with pytest.raises(ValueError, match='TOP K 4.*1\\.\\.3'):
        oversized.generate_training_table(snapshot)
    assert not list(snapshot.index_path.glob('train_tables/*.staging'))

    valid = PredictiveQuery(
        graph, 'PREDICT users.labels CLASSIFY TOP 3 FOR EACH users.id')
    result = valid.generate_training_table(snapshot)
    assert result.train_table_path.is_dir()
    assert (result.train_table_path / 'part-00000.parquet').is_file()
    assert pq.read_table(result.train_table_path).num_rows == 120


@pytest.fixture
def graph_snapshot(tmp_path):
    connector = S3Connector(root_dir='s3://schema-only-fixture')
    tables = []
    for name, cols, time_col in [
        ('users', [('label', Dtype.bool, Stype.categorical)], None),
        ('products', [], None),
        ('orders', [('user_id', Dtype.int64, Stype.ID),
                    ('product_id', Dtype.int64, Stype.ID),
                    ('ts', Dtype.date, Stype.timestamp),
                    ('amount', Dtype.float64, Stype.numerical)], 'ts'),
    ]:
        tables.append(
            Table(
                connector, name,
                TableDefinition(
                    source_table=S3SourceTable(
                        connector_id=connector.id, file_type=FileType.PARQUET,
                        source=S3RelativeSource(path=name)), pkey='id',
                    time_col=time_col,
                    cols=[Column(name='id', dtype=Dtype.int64, stype=Stype.ID)
                          ] + [
                              Column(name=col, dtype=dtype, stype=stype)
                              for col, dtype, stype in cols
                          ])))
    graph = Graph(tables, [
        Edge('orders', 'user_id', 'users'),
        Edge('orders', 'product_id', 'products')
    ])
    # An absent index proves suggestions only need resolved schema metadata.
    return graph, GraphSnapshot(f'{graph.id}@fixture', tmp_path / 'absent',
                                graph._graph_config)


@pytest.mark.parametrize('query', [
    'PREDICT users.label FOR EACH users.id',
    'PREDICT orders.amount FOR EACH orders.id',
    'PREDICT COUNT(orders.*, 0, 7, days) FOR EACH users.id',
    'PREDICT COUNT(orders.*, 0, 1, days) FORECAST 3 TIMEFRAMES '
    'FOR EACH users.id',
    'PREDICT LIST_DISTINCT(orders.product_id, 0, 7, days) RANK TOP 3 '
    'FOR EACH users.id',
])
@pytest.mark.parametrize('run_mode', list(RunMode))
@pytest.mark.parametrize('weight_col', [None, 'weight'])
def test_suggestions_match_query_defaults_and_validate(graph_snapshot, query,
                                                       run_mode, weight_col):
    graph, snapshot = graph_snapshot
    pquery = PredictiveQuery(graph, query)
    parsed = pquery._parsed
    native = graph.native_graph
    facts = dict(
        task_type=TaskType(parsed.task_type),
        query_type=QueryType(parsed.query_type),
        has_train_table_time_col=(parsed.query_type != 'static'
                                  or parsed.entity_table == 'orders'),
        run_mode=run_mode,
        graph=to_plan_graph(native),
        entity_table_names=(
            [parsed.entity_table] +
            ([parsed.rhs_node_type] if parsed.rhs_node_type else [])),
        target_to_entity_paths=target_to_entity_paths(parsed, native),
        has_train_table_weight_col=weight_col is not None,
    )
    expected = CoreModelPlan.from_default(**facts)
    if facts['task_type'].is_ranking:
        expected.training_job.metrics = list(
            set(expected.training_job.metrics +
                ['map@3', 'precision@3', 'recall@3', 'f1@3']))
    with patch('socket.socket.connect', side_effect=AssertionError('network')), \
         patch.object(Graph, 'snapshot', side_effect=AssertionError('ingest')), \
         patch.object(PredictiveQuery, 'generate_training_table',
                      side_effect=AssertionError('materialization')), \
         patch.object(Trainer, 'fit', side_effect=AssertionError('training')):
        plan = pquery.suggest_model_plan(
            snapshot, run_mode=run_mode,
            train_table_spec=TrainingTableSpec(weight_col=weight_col,
                                               split_col='ignored'))
    assert isinstance(plan, ModelPlan)
    actual = to_json_dict(plan)
    reference = to_json_dict(public_model_plan_from_internal(expected))
    actual['training_job']['metrics'].sort()
    reference['training_job']['metrics'].sort()
    assert actual == reference
    assert 'training_table_generation' not in actual
    plan.optimization.max_epochs = 2
    restored = _to_core_model_plan(plan)
    assert restored.optimization.max_epochs == 2
    errors, _ = restored.validate(**facts)
    assert errors == []


@pytest.mark.parametrize('query,run_mode,error', [
    ('not a predictive query', RunMode.FAST, 'does not start with the PREDICT'),
    ('PREDICT users.absent FOR EACH users.id', RunMode.FAST, 'absent'),
    ('PREDICT users.label FOR EACH users.id', 'invalid', 'RunMode'),
])
def test_suggestions_reject_invalid_input(graph_snapshot, query, run_mode,
                                          error):
    graph, snapshot = graph_snapshot
    with pytest.raises(ValueError, match=error):
        PredictiveQuery(graph,
                        query).suggest_model_plan(snapshot, run_mode=run_mode)


def test_suggestions_require_matching_snapshot_and_resolved_metadata(
        graph_snapshot):
    graph, snapshot = graph_snapshot
    query = 'PREDICT users.label FOR EACH users.id'
    with pytest.raises(ValueError, match='was not materialized'):
        PredictiveQuery(graph,
                        query).suggest_model_plan(replace(snapshot, id='other'))
    unresolved = Graph([replace(graph.tables[0], definition=None)])
    with pytest.raises(ValueError, match='definition|infer_metadata'):
        PredictiveQuery(unresolved, query).suggest_model_plan(
            replace(snapshot, id=f'{unresolved.id}@fixture'))


@pytest.mark.parametrize('connector_kind', ['file', 's3'])
def test_generated_weight_rejected_before_artifacts(connector_kind, tmp_path,
                                                    monkeypatch):
    columns = [
        Column(name='id', dtype=Dtype.int64, stype=Stype.ID),
        Column(name='label', dtype=Dtype.bool, stype=Stype.categorical),
        Column(name='sample_weight', dtype=Dtype.float64,
               stype=Stype.numerical),
    ]
    if connector_kind == 'file':
        source_dir = tmp_path / 'source'
        source_dir.mkdir()
        connector = FileConnector(root_dir=str(source_dir))
        definition = LocalTableDefinition(
            source_table=LocalFileSourceTable(
                path=str(source_dir / 'users.parquet'),
                file_type=FileType.PARQUET,
            ),
            pkey='id',
            cols=columns,
        )
    else:
        connector = S3Connector(root_dir='s3://schema-only-fixture')
        definition = TableDefinition(
            source_table=S3SourceTable(
                connector_id=connector.id,
                file_type=FileType.PARQUET,
                source=S3RelativeSource(path='users'),
            ),
            pkey='id',
            cols=columns,
        )
    graph = Graph([Table(connector, 'users', definition)])
    snapshot = GraphSnapshot(f'{graph.id}@fixture', tmp_path / 'index',
                             graph._graph_config)
    query = PredictiveQuery(graph, 'PREDICT users.label FOR EACH users.id')
    run_dir = tmp_path / 'generated-run'

    monkeypatch.setattr('kumo.engine.pquery.predictive_query.new_run_id',
                        lambda: pytest.fail('allocated a weighted run id'))
    monkeypatch.setattr(
        'kumo.engine.pquery.predictive_query.train_table_scratch_dir',
        lambda run_id: run_dir,
    )
    monkeypatch.setattr(
        'kumo.engine.pquery.predictive_query.build_training_task_plan',
        lambda workflow: pytest.fail('built a weighted native task plan'),
    )
    monkeypatch.setattr(
        'kumo.engine.pquery.predictive_query.generate_train_table',
        lambda request: pytest.fail('generated a weighted artifact'),
    )

    with pytest.raises(
            ValueError,
            match='Weighted training-table generation is not yet supported'):
        query.generate_training_table(
            snapshot,
            TrainingTableGenerationPlan(weight_col='sample_weight'),
        )

    assert not run_dir.exists()

    native_plan = SimpleNamespace(
        train_table=SimpleNamespace(lag_timesteps=None, compiler_options=None),
        task_dataset=SimpleNamespace(required_stages=('train', 'val')),
    )
    generated = SimpleNamespace(
        train_table_path=str(tmp_path / 'train.parquet'),
        manifest_path=str(tmp_path / 'manifest.json'),
        staging_dir=str(tmp_path / 'staging'),
    )
    monkeypatch.setattr('kumo.engine.pquery.predictive_query.new_run_id',
                        lambda: 'unweighted')
    monkeypatch.setattr(
        'kumo.engine.pquery.predictive_query.build_training_task_plan',
        lambda workflow: native_plan,
    )
    monkeypatch.setattr('kumo.engine.pquery.predictive_query.generate_train_table',
                        lambda request: generated)

    table = query.generate_training_table(snapshot)

    assert run_dir.is_dir()
    assert table.train_table_path == tmp_path / 'train.parquet'


def test_generated_temporal_split_requires_time_column(graph_snapshot,
                                                       monkeypatch):
    graph, snapshot = graph_snapshot
    monkeypatch.setattr(
        'kumo.engine.pquery.predictive_query.new_run_id',
        lambda: pytest.fail('allocated a run before rejecting missing time'),
    )
    with pytest.raises(ValueError, match="doesn't have a 'Create Date'"):
        PredictiveQuery(
            graph,
            'PREDICT users.label FOR EACH users.id').generate_training_table(
                snapshot,
                TrainingTableGenerationPlan(
                    split='TemporalSplit([0.6, 0.3, 0.1])'))


def test_generated_absolute_bounds_name_supported_alternative(
        graph_snapshot, monkeypatch):
    graph, snapshot = graph_snapshot
    monkeypatch.delenv('USE_DECOUPLED_SPLITS', raising=False)
    monkeypatch.setattr(
        'kumo.engine.pquery.predictive_query.new_run_id',
        lambda: pytest.fail('allocated a bounded-generation run id'),
    )

    with pytest.raises(
            ValueError,
            match="Use 'train_start_offset' and 'train_end_offset' instead"):
        PredictiveQuery(
            graph,
            'PREDICT COUNT(orders.*, 0, 1, days) > 0 FOR EACH users.id',
        ).generate_training_table(
            snapshot,
            TrainingTableGenerationPlan(
                start_time='2025-01-02T12:00:00',
                end_time='2025-01-21T12:00:00',
                timeframe_step=1,
                split='ApproxDateOffsetSplit([0.6, 0.2, 0.2])',
            ),
        )


@pytest.mark.parametrize('forecast_clause', ['', ' FORECAST 3 TIMEFRAMES'])
def test_forecast_generation_retains_effective_plan(graph_snapshot, tmp_path,
                                                    monkeypatch,
                                                    forecast_clause):
    graph, snapshot = graph_snapshot
    query = PredictiveQuery(
        graph,
        f'PREDICT COUNT(orders.*, 0, 1, days){forecast_clause} FOR EACH users.id',
    )
    native_plan = SimpleNamespace(
        train_table=SimpleNamespace(
            lag_timesteps=3,
            compiler_options={
                'lag_offsets_secs': (-1, -2, -3),
                'yoy_offsets_secs': (-365, ),
            },
        ),
        task_dataset=SimpleNamespace(required_stages=('train', 'val')),
    )
    generated = SimpleNamespace(
        train_table_path=str(tmp_path / 'train.parquet'),
        manifest_path=str(tmp_path / 'manifest.json'),
        staging_dir=str(tmp_path / 'staging'),
    )
    monkeypatch.setattr('kumo.engine.pquery.predictive_query.new_run_id',
                        lambda: 'run')
    monkeypatch.setattr(
        'kumo.engine.pquery.predictive_query.train_table_scratch_dir',
        lambda run_id: tmp_path / run_id,
    )
    monkeypatch.setattr(
        'kumo.engine.pquery.predictive_query.build_training_task_plan',
        lambda workflow: native_plan,
    )
    monkeypatch.setattr('kumo.engine.pquery.predictive_query.generate_train_table',
                        lambda plan: generated)

    table = query.generate_training_table(
        snapshot,
        plan=TrainingTableGenerationPlan(
            forecast_length=4,
            lag_timesteps=3,
            year_over_year=True,
            timeframe_step=2,
        ),
    )

    assert table.generation_plan.lag_timesteps == 3
    assert table.generation_plan.year_over_year is True
    assert table.generation_plan.timeframe_step == 2
    assert table.generation_plan.forecast_length == 4


def test_prediction_compiler_uses_plan_forecast_horizon(graph_snapshot,
                                                        tmp_path, monkeypatch):
    from kumo.engine.pquery.prediction_table import _compile_prediction_table
    graph, snapshot = graph_snapshot
    query = PredictiveQuery(
        graph, 'PREDICT COUNT(orders.*, 0, 1, days) FOR EACH users.id')
    monkeypatch.setattr('kumo_core.pquery.read_col_stats',
                        lambda *args: ('{"amount__MEAN": 1, '
                                       '"ts__MIN__False": 1704067200000000000, '
                                       '"ts__MAX__False": 1711843200000000000}'))
    monkeypatch.setattr('kumo_core.pquery._infer_encoder',
                        lambda column, stats: {'name': 'Numerical'})
    captured = []
    monkeypatch.setattr(
        'kumo.engine.pquery.prediction_table.generate_prediction_table',
        lambda request: captured.append(request) or object())
    request, _, _ = _compile_prediction_table(
        snapshot, query._parsed, prediction_time=None, run_id='prediction',
        scratch_dir=tmp_path, artifact_dir=tmp_path,
        generation_plan=TrainingTableGenerationPlan(forecast_length=4,
                                                    lag_timesteps=3,
                                                    year_over_year=True,
                                                    timeframe_step=2))
    assert captured == [request]
    assert request.query.task_type == 'forecasting'
    assert request.query.forecast_length == 4
    assert request.timestamps == (1_711_843_200_000_000_000, )
    assert request.task_dataset.task_type == 'forecasting'
    assert request.forecast_timeframe_step == 2
    assert request.compiler_options['future_offsets_secs'] == (0, 86400, 172800,
                                                               259200)


def test_custom_training_table_materializes_weight_and_manual_split(
        graph_snapshot, tmp_path, monkeypatch):
    graph, snapshot = graph_snapshot
    query = PredictiveQuery(graph,
                            'PREDICT users.label FOR EACH users.id')
    source = graph.tables[0].connector['custom_rows']
    rows = tmp_path / 'custom.parquet'
    rows.touch()
    staging = tmp_path / 'staging'
    captured = {}
    native_plan = SimpleNamespace(
        train_table=SimpleNamespace(
            lag_timesteps=None,
            compiler_options=None,
        ),
        task_dataset=SimpleNamespace(required_stages=('train', 'val', 'test')),
    )

    monkeypatch.setattr('kumo.engine.pquery._custom_source.stage_source_table',
                        lambda source_table, destination: rows)
    monkeypatch.setattr('kumo.engine.pquery._custom_source.validate_training_source',
                        lambda *args: None)

    def build(**kwargs):
        captured['plan_kwargs'] = kwargs
        return native_plan

    monkeypatch.setattr(
        'kumo_core.pquery.build_native_pquery_task_plan', build)

    def generate(request, custom):
        captured['custom'] = custom
        return SimpleNamespace(
            train_table_path=str(rows),
            manifest_path=str(staging / 'manifest.json'),
            staging_dir=str(staging),
        )

    monkeypatch.setattr(
        'kumo_core.training.table.generate_custom_train_table', generate)

    table = TrainingTable.from_source_table(
        source,
        TrainingTableSpec(weight_col='WEIGHT', split_col='SPLIT'),
        pq=query,
        snapshot=snapshot,
    )

    assert table.query == query.query
    assert table.required_stages == ('train', 'val', 'test')
    assert captured['custom'].parquet_path == str(rows)
    assert captured['custom'].weight_column == 'WEIGHT'
    generation = captured['plan_kwargs'][
        'model_plan'].training_table_generation
    assert generation.split == (
        "ManualSplit(col='SPLIT', train_value=0, val_value=1, test_value=2)")


def test_custom_training_table_uses_the_normal_inferred_split(
        graph_snapshot, tmp_path, monkeypatch):
    graph, snapshot = graph_snapshot
    query = PredictiveQuery(graph,
                            'PREDICT users.label FOR EACH users.id')
    rows = tmp_path / 'custom.parquet'
    rows.touch()
    staging = tmp_path / 'staging'
    captured = {}
    monkeypatch.setattr('kumo.engine.pquery._custom_source.stage_source_table',
                        lambda source_table, destination: rows)
    monkeypatch.setattr('kumo.engine.pquery._custom_source.validate_training_source',
                        lambda *args: None)

    def build(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            train_table=SimpleNamespace(lag_timesteps=None,
                                        compiler_options=None),
            task_dataset=SimpleNamespace(required_stages=('train', 'val')),
        )

    monkeypatch.setattr(
        'kumo_core.pquery.build_native_pquery_task_plan', build)
    monkeypatch.setattr(
        'kumo_core.training.table.generate_custom_train_table',
        lambda request, custom: SimpleNamespace(
            train_table_path=str(rows),
            manifest_path=str(staging / 'manifest.json'),
            staging_dir=str(staging),
        ),
    )

    table = TrainingTable.from_source_table(
        graph.tables[0].connector['custom_rows'],
        TrainingTableSpec(),
        pq=query,
        snapshot=snapshot,
    )

    assert str(captured['model_plan'].training_table_generation.split) in {
        'inferred', 'missing'
    }
    assert table.required_stages == ('train', 'val')


def test_prediction_generation_accepts_restored_training_plan(
        graph_snapshot, tmp_path, monkeypatch):
    graph, snapshot = graph_snapshot
    query = PredictiveQuery(
        graph,
        'PREDICT COUNT(orders.*, 0, 1, days) FORECAST 3 TIMEFRAMES '
        'FOR EACH users.id',
    )
    fitted_plan = TrainingTableGenerationPlan(
        lag_timesteps=3,
        year_over_year=True,
        timeframe_step=2,
    )
    request = SimpleNamespace(
        timestamps=(123, ),
        task_dataset=SimpleNamespace(required_stages=('predict', )),
        compiler_options={'lag_offsets_secs': (-1, -2, -3)},
        forecast_timeframe_step=2,
    )
    generated = SimpleNamespace(
        table_path=str(tmp_path / 'prediction.parquet'),
        staging_dir=str(tmp_path / 'staging'),
        row_count=4,
        tt_id='prediction',
    )
    captured = {}

    def compile_table(*args, **kwargs):
        captured.update(kwargs)
        return request, generated, snapshot.id

    monkeypatch.setattr('kumo.engine.pquery.predictive_query.new_run_id',
                        lambda: 'run')
    monkeypatch.setattr(
        'kumo.engine.pquery.predictive_query.prediction_table_scratch_dir',
        lambda run_id: tmp_path / run_id,
    )
    monkeypatch.setattr(
        'kumo.engine.pquery.predictive_query._compile_prediction_table',
        compile_table,
    )

    table = query.generate_prediction_table(snapshot, plan=fitted_plan)

    assert captured['generation_plan'] is fitted_plan
    assert table.compiler_options == {'lag_offsets_secs': (-1, -2, -3)}
    assert table.forecast_timeframe_step == 2


@pytest.mark.parametrize('operation', ['validate', 'snapshot'])
def test_mismatched_graph_keys_fail_before_staging(graph_snapshot, monkeypatch,
                                                   operation):
    graph, _ = graph_snapshot
    orders = graph.tables[2].with_column('user_id', dtype=Dtype.string)
    invalid = Graph([*graph.tables[:2], orders], graph.edges)

    def unexpected_staging(*args, **kwargs):
        pytest.fail('invalid graph reached connector staging')

    monkeypatch.setattr(type(orders.connector.source_backend),
                        'prepare_ingest', unexpected_staging)
    with pytest.raises(ValueError, match='orders.user_id.*users.id'):
        getattr(invalid, operation)()


def test_custom_prediction_table_materializes_exact_source_rows(
        graph_snapshot, tmp_path, monkeypatch):
    graph, snapshot = graph_snapshot
    query = PredictiveQuery(
        graph,
        'PREDICT COUNT(orders.*, 0, 1, days) FORECAST 3 TIMEFRAMES '
        'FOR EACH users.id',
    )
    source = graph.tables[0].connector['curated_population']
    rows = tmp_path / 'population.parquet'
    rows.touch()
    from kumo_core.prediction import PredictionTableRequest
    request = PredictionTableRequest(
        query=query._parsed,
        graph=graph.native_graph,
        timestamps=(123, ),
        index_path=str(snapshot.index_path),
        tt_id='custom-prediction',
        task_dataset=SimpleNamespace(required_stages=('train', 'val')),
        compiler_options={'lag_offsets_secs': (-1, -2, -3)},
        forecast_timeframe_step=2,
    )
    captured = {}
    monkeypatch.setattr('kumo.engine.pquery._custom_source.stage_source_table',
                        lambda source_table, destination: rows)
    monkeypatch.setattr(
        'kumo.engine.pquery._custom_source.validate_prediction_source',
        lambda *args: None,
    )

    def build(*args, **kwargs):
        captured['generation_plan'] = kwargs['generation_plan']
        captured['supplied_rows'] = kwargs['supplied_rows']
        return request, snapshot.id

    monkeypatch.setattr(
        'kumo.engine.pquery.prediction_table._build_prediction_table_request',
        build,
    )

    def generate(actual_request, custom):
        captured['request'] = actual_request
        captured['custom'] = custom
        return SimpleNamespace(
            table_path=str(rows),
            staging_dir=str(tmp_path / 'staging'),
            row_count=3,
            tt_id='custom-prediction',
        )

    monkeypatch.setattr(
        'kumo.engine.pquery.prediction_table.generate_custom_prediction_table',
        generate,
    )
    plan = TrainingTableGenerationPlan(lag_timesteps=3,
                                       timeframe_step=2)

    table = PredictionTable.from_source_table(
        source,
        pq=query,
        snapshot=snapshot,
        plan=plan,
    )

    assert captured['generation_plan'] is plan
    assert captured['supplied_rows'] is True
    assert captured['custom'].parquet_path == str(rows)
    assert captured['request'].timestamps == ()
    assert table.timestamps == ()
    assert table.row_count == 3
    assert table.required_stages == ('train', 'val')
    assert table.compiler_options == {'lag_offsets_secs': (-1, -2, -3)}
