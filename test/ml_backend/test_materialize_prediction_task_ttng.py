# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from datetime import datetime, timezone
from test.data.generic.ttng_v3_test_bundle import write_v3_bundle
from types import SimpleNamespace

import pytest
from kumo_core.job_config import PredictionConfig
from kumo_core.train.autoregression import AUTOREGRESSION_COL, YOY_COL
from kumo_core.types import PredictionArtifactType, TaskType

from statedb.config.jobs import MetadataField

mpt = importlib.import_module(
    'rest.backends.ml_backend.activities.materialize_prediction_task')


@pytest.mark.parametrize(
    ('artifact_types', 'expect_predictions', 'expect_embeddings'),
    [
        ([PredictionArtifactType.PREDICTIONS], True, False),
        ([PredictionArtifactType.EMBEDDINGS], False, True),
        ([
            PredictionArtifactType.PREDICTIONS,
            PredictionArtifactType.EMBEDDINGS
        ], True, True),
    ],
)
def test_prediction_artifact_flags_are_derived_from_requested_outputs(
        artifact_types, expect_predictions, expect_embeddings):
    job_input = SimpleNamespace(config=SimpleNamespace(outputs=[
        SimpleNamespace(artifact_type=artifact_type)
        for artifact_type in artifact_types
    ]))

    assert mpt.return_predictions(job_input) is expect_predictions
    assert mpt.return_embeddings(job_input) is expect_embeddings


def _patch_feature_schema(monkeypatch):
    materialize_training_task = importlib.import_module(
        'rest.backends.ml_backend.activities.materialize_training_task')
    monkeypatch.setattr(
        materialize_training_task,
        '_build_ttng_feature_schema',
        lambda task, _encoders: {
            'node_types': {
                table.table_name: {
                    'feature_columns': [],
                    'mapper_key': table.table_name,
                    'time_column': None,
                    'column_specs': [],
                }
                for table in task.graph.tables
            },
            'target_encoder_input': None,
        },
    )


def _make_job_input():
    return SimpleNamespace(
        id='bp-job',
        # Full-native predict derives the staging tt_id from the GENERATE job id;
        # None falls back to ``id`` (the single-job test case).
        config=SimpleNamespace(generate_pred_table_job_id=None),
        graph_data_compute_input=SimpleNamespace(graph_key='graph'),
        graph_version_info=SimpleNamespace(graph_data_version='v1'),
    )


def _make_pred_config():
    return PredictionConfig(
        home_dir='/h',
        train_dir='/t',
        task_name='task',
        model_dir='/m',
        output_dir='/o',
        return_embeddings=False,
        return_predictions=True,
        pred_table_path='/pred',
        num_partitions=1,
        output_num_partitions=6,
    )


def _patch_common(monkeypatch, calls, encoders=None):
    """Patch the service + encoder collection so the activity runs without a
    real diskgraph wheel. ``calls`` records the service materialize call args.

    One-gen predict: the generator already staged the predict seeds, so the
    activity is ingest-only — it asserts the kept staging parquet exists (patched
    True) and the service returns the ingested predict-stage row count."""
    class FakeService:
        def materialize_prediction_task_dataset(self, handle, spec, **kwargs):
            calls.append((handle, spec, kwargs))
            return SimpleNamespace(
                bundle_dir=spec.artifact_root,
                manifest_path=f'{spec.artifact_root}/m.json',
                predict_row_count=5,
            )

    monkeypatch.setattr('kumo.diskgraph.get_service',
                        lambda: FakeService())
    monkeypatch.setattr(
        'kumo.task.task.collect_encoders',
        lambda _: SimpleNamespace(encoders=encoders or
                                  {'users': {
                                      'age': object()
                                  }}),
    )
    # The kept predict staging parquet was written by the generator activity.
    monkeypatch.setattr(mpt.fsutil, 'exists', lambda _p: True)
    import kumo.data.generic.diskgraph_train_table as _dtt
    monkeypatch.setattr(_dtt, 'assert_ttng_bundle_complete', lambda _d: None)
    monkeypatch.setattr(_dtt, 'read_tt_id_from_staging_bundle',
                        lambda _d: 'pred_bp-job')
    monkeypatch.setattr(
        _dtt,
        'restore_durable_ttng_bundle',
        lambda _durable, local: local,
    )
    _patch_feature_schema(monkeypatch)


def _materialize(**kwargs):
    durable_staging_dir, source_tt_id = (
        mpt._load_ttng_prediction_bundle_identity(
            kwargs['prediction_config'].pred_table_path))
    return mpt._materialize_ttng_prediction_task_dataset(
        durable_staging_dir=durable_staging_dir,
        source_tt_id=source_tt_id,
        **kwargs,
    )


def _node_pquery():
    """A pquery double sufficient for the node-prediction path (static, so no
    pred timestamp needed)."""
    from kumoapi.pquery import QueryType
    bp_gen = SimpleNamespace(query_type=QueryType.STATIC)
    vquery = SimpleNamespace(target_stype=SimpleNamespace(value='ID'))
    return SimpleNamespace(
        validated_predictive_query=vquery,
        get_batch_pred_table_generator=lambda: bp_gen,
        list_distinct_target_is_array=False,
    )


def _node_task():
    users = SimpleNamespace(
        table_name='users',
        time_col=None,
        time_col_name=None,
        has_time_col=lambda: False,
        fkeys=[],
    )
    graph = SimpleNamespace(tables=[users], table=lambda _n: users)
    return SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        is_temporal=False,
        graph=graph,
        train_table=SimpleNamespace(
            fkeys=[SimpleNamespace(name='CUSTOMER_ID', dst_table='src.users')]),
    )


def _forecasting_pquery():
    from kumoapi.pquery import QueryType
    bp_gen = SimpleNamespace(
        query_type=QueryType.TEMPORAL,
        generate_pred_timestamp=lambda: SimpleNamespace(value=123),
    )
    vquery = SimpleNamespace(target_stype=SimpleNamespace(value='FLOAT'))
    return SimpleNamespace(
        validated_predictive_query=vquery,
        get_batch_pred_table_generator=lambda: bp_gen,
        list_distinct_target_is_array=False,
    )


def _forecasting_task():
    users = SimpleNamespace(
        table_name='users',
        time_col=None,
        time_col_name=None,
        has_time_col=lambda: False,
        fkeys=[],
    )
    graph = SimpleNamespace(tables=[users], table=lambda _n: users)
    return SimpleNamespace(
        task_type=TaskType.FORECASTING,
        is_temporal=True,
        graph=graph,
        train_table=SimpleNamespace(
            table_name='pred_table',
            fkeys=[SimpleNamespace(name='CUSTOMER_ID', dst_table='src.users')]),
    )


def test_ttng_node_prediction_materialize_writes_single_bundle(monkeypatch):
    calls = []
    _patch_common(monkeypatch, calls)
    task = _node_task()

    row_count = _materialize(
        task=task,
        pquery=_node_pquery(),
        prediction_config=(pred_config := _make_pred_config()),
        job_input=_make_job_input(),
        diskgraph_index_path='/dg',
        train_table_name='train_tt',
    )

    # The ingested predict-stage row count is returned (drives output parts).
    assert row_count == 5
    assert pred_config.diskgraph_task_dataset_bundle_dir == (
        '/dg/task_datasets/bp-job')
    assert len(calls) == 1
    handle, spec, kwargs = calls[0]
    # Ingest-only one-gen: node prediction now hands the kept staging dir to the
    # service (NOT a predict plan); the gen already ran in the generator.
    assert kwargs['predict_staging_dir'] == (
        '/dg/train_tables/.pred_bp-job.staging')
    assert spec.job_id == 'bp-job'
    assert spec.artifact_root == '/dg/task_datasets/bp-job'
    # Durable staging fields are recorded so every predict shard pod can
    # reconstruct its extension from the durable staging beside the pred table.
    assert pred_config.diskgraph_pred_staging_dir == '/pred.ttng_staging'
    # Single-job case (no gen job -> spec tt_id == pred_bp-job): SOURCE == TARGET.
    assert pred_config.diskgraph_pred_target_tt_id == 'pred_bp-job'
    # Node prediction: node_prediction family, no rhs, not temporal.
    assert spec.task_family == 'node_prediction'
    assert spec.rhs_node_type is None
    assert spec.is_temporal is False
    # Node prediction: no predict seed-time anchor is threaded (LP-only), so
    # the bundle stays byte-identical to before.
    assert kwargs.get('predict_seed_time_ns') is None


def test_ttng_forecasting_prediction_schema_uses_train_table_namespace(
        monkeypatch):
    calls = []
    _patch_common(monkeypatch, calls)
    materialize_training_task = importlib.import_module(
        'rest.backends.ml_backend.activities.materialize_training_task')
    monkeypatch.setattr(
        materialize_training_task,
        '_build_ttng_feature_schema',
        lambda _task, _encoders: {
            'node_types': {
                'users': {
                    'feature_columns': [],
                    'mapper_key': 'users',
                    'time_column': None,
                    'column_specs': [],
                },
                'pred_table': {
                    'feature_columns': [AUTOREGRESSION_COL, YOY_COL],
                    'mapper_key':
                    'pred_table',
                    'time_column':
                    None,
                    'column_specs': [
                        {
                            'column_name': AUTOREGRESSION_COL
                        },
                        {
                            'column_name': YOY_COL
                        },
                    ],
                },
            },
            'target_encoder_input': None,
        },
    )

    _materialize(
        task=_forecasting_task(),
        pquery=_forecasting_pquery(),
        prediction_config=_make_pred_config(),
        job_input=_make_job_input(),
        diskgraph_index_path='/dg',
        train_table_name='train_tt',
    )

    _, spec, _ = calls[0]
    node_schema = spec.feature_schema['node_types']
    assert set(node_schema) == {'users', 'train_tt'}
    assert node_schema['train_tt']['feature_columns'] == [
        AUTOREGRESSION_COL,
        YOY_COL,
    ]
    assert node_schema['train_tt']['mapper_key'] == 'train_tt'


@pytest.mark.parametrize(
    'task_type',
    [
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.MULTILABEL_CLASSIFICATION,
        TaskType.REGRESSION,
    ],
)
def test_ttng_node_subtype_prediction_materialize_uses_node_family(
    monkeypatch,
    task_type,
):
    calls = []
    _patch_common(monkeypatch, calls)
    task = _node_task()
    task.task_type = task_type

    _materialize(
        task=task,
        pquery=_node_pquery(),
        prediction_config=_make_pred_config(),
        job_input=_make_job_input(),
        diskgraph_index_path='/dg',
        train_table_name='train_tt',
    )

    _, spec, kwargs = calls[0]
    assert spec.task_family == 'node_prediction'
    assert spec.task_type == task_type.value
    assert spec.is_temporal is False
    assert spec.rhs_node_type is None
    assert kwargs.get('predict_seed_time_ns') is None


def test_ttng_node_prediction_bp_uses_temporal_prediction_anchor(monkeypatch):
    calls = []
    _patch_common(monkeypatch, calls)
    task = _node_task()
    task.task_type = TaskType.REGRESSION
    task.is_temporal = True

    _materialize(
        task=task,
        pquery=_forecasting_pquery(),
        prediction_config=_make_pred_config(),
        job_input=_make_job_input(),
        diskgraph_index_path='/dg',
        train_table_name='train_tt',
    )

    _, spec, kwargs = calls[0]
    assert spec.task_family == 'node_prediction'
    assert spec.task_type == TaskType.REGRESSION.value
    assert spec.is_temporal is True
    assert spec.rhs_node_type is None
    assert kwargs.get('predict_seed_time_ns') is None


def _lp_pquery():
    from kumoapi.pquery import QueryType
    bp_gen = SimpleNamespace(query_type=QueryType.STATIC)
    vquery = SimpleNamespace(target_stype=SimpleNamespace(value='ID'))
    return SimpleNamespace(
        validated_predictive_query=vquery,
        graph=SimpleNamespace(),
        get_batch_pred_table_generator=lambda: bp_gen,
        list_distinct_target_is_array=True,
    )


def _lp_task(task_type):
    users = SimpleNamespace(
        table_name='users',
        time_col=None,
        time_col_name=None,
        has_time_col=lambda: False,
        fkeys=[],
    )
    items = SimpleNamespace(
        table_name='items',
        time_col=None,
        time_col_name=None,
        has_time_col=lambda: False,
        fkeys=[],
    )
    graph = SimpleNamespace(
        tables=[users, items],
        table=lambda n: {
            'users': users,
            'items': items
        }[n],
    )
    return SimpleNamespace(
        task_type=task_type,
        is_temporal=task_type == TaskType.TEMPORAL_LINK_PREDICTION,
        graph=graph,
        train_table=SimpleNamespace(
            fkeys=[SimpleNamespace(name='USER_ID', dst_table='src.users')]),
    )


def test_ttng_lp_materialize_is_lp_aware(monkeypatch):
    """LP predict spec is link_prediction, carries rhs_node_type (FK-free via
    get_rhs_lp_table), and is_temporal True for temporal LP."""
    calls = []
    _patch_common(monkeypatch, calls)
    monkeypatch.setattr('kumo.pquery.utils.get_rhs_lp_table',
                        lambda *a, **k: 'items')

    _materialize(
        task=_lp_task(TaskType.TEMPORAL_LINK_PREDICTION),
        pquery=_lp_pquery(),
        prediction_config=_make_pred_config(),
        job_input=_make_job_input(),
        diskgraph_index_path='/dg',
        train_table_name='train_tt',
    )

    assert len(calls) == 1
    _, spec, kwargs = calls[0]
    assert spec.task_family == 'link_prediction'
    assert spec.rhs_node_type == 'items'
    assert spec.is_temporal is True
    assert spec.seed_node_type == 'users'
    assert spec.entity_table_name == 'users'
    # Ingest-only one-gen: the kept staging dir is threaded (no predict plan).
    assert kwargs['predict_staging_dir'] == (
        '/dg/train_tables/.pred_bp-job.staging')


def test_ttng_lp_static_not_temporal(monkeypatch):
    """Static LP is link_prediction but is_temporal False."""
    calls = []
    _patch_common(monkeypatch, calls)
    monkeypatch.setattr('kumo.pquery.utils.get_rhs_lp_table',
                        lambda *a, **k: 'items')

    _materialize(
        task=_lp_task(TaskType.STATIC_LINK_PREDICTION),
        pquery=_lp_pquery(),
        prediction_config=_make_pred_config(),
        job_input=_make_job_input(),
        diskgraph_index_path='/dg',
        train_table_name='train_tt',
    )

    _, spec, kwargs = calls[0]
    assert spec.task_family == 'link_prediction'
    assert spec.is_temporal is False
    assert spec.rhs_node_type == 'items'
    # Static LP: no real anchor, so a ``0`` sentinel predict seed time is
    # threaded so the RHS NodeRange loader gets a valid non-None seed time.
    assert kwargs['predict_seed_time_ns'] == 0


def _temporal_lp_pquery(anchor_ns):
    """Temporal LP pquery double whose batch-pred generator returns a fixed
    prediction anchor (ns)."""
    from kumoapi.pquery import QueryType
    bp_gen = SimpleNamespace(
        query_type=QueryType.TEMPORAL,
        generate_pred_timestamp=lambda: SimpleNamespace(value=anchor_ns),
    )
    vquery = SimpleNamespace(target_stype=SimpleNamespace(value='ID'))
    return SimpleNamespace(
        validated_predictive_query=vquery,
        graph=SimpleNamespace(),
        get_batch_pred_table_generator=lambda: bp_gen,
        list_distinct_target_is_array=True,
    )


def test_ttng_lp_temporal_threads_prediction_anchor(monkeypatch):
    """Temporal LP threads the real prediction anchor (ns) as the predict
    seed time so the RHS NodeRange loader seeds at the LHS query time."""
    calls = []
    _patch_common(monkeypatch, calls)
    monkeypatch.setattr('kumo.pquery.utils.get_rhs_lp_table',
                        lambda *a, **k: 'items')

    _materialize(
        task=_lp_task(TaskType.TEMPORAL_LINK_PREDICTION),
        pquery=_temporal_lp_pquery(anchor_ns=9_000_000_000),
        prediction_config=_make_pred_config(),
        job_input=_make_job_input(),
        diskgraph_index_path='/dg',
        train_table_name='train_tt',
    )

    _, spec, kwargs = calls[0]
    assert spec.task_family == 'link_prediction'
    assert spec.is_temporal is True
    assert kwargs['predict_seed_time_ns'] == 9_000_000_000


@pytest.mark.parametrize('fields', [
    [],
    [MetadataField.JOB_TIMESTAMP],
    [MetadataField.ANCHOR_TIMESTAMP],
    [MetadataField.ANCHOR_TIMESTAMP, MetadataField.JOB_TIMESTAMP],
])
@pytest.mark.parametrize(
    'task_type',
    [task_type for task_type in TaskType if task_type != TaskType.FORECASTING],
)
def test_prediction_config_writes_time_only_on_request(task_type, fields):
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pred_config = _make_pred_config()
    job_input = SimpleNamespace(
        start_time=start_time,
        config=SimpleNamespace(
            task_type=task_type,
            forecast_length=None,
            outputs=[
                SimpleNamespace(
                    artifact_type=PredictionArtifactType.PREDICTIONS,
                    extra_fields=fields,
                )
            ],
        ),
    )

    mpt._configure_prediction_output_fields(pred_config, job_input)

    anchor = MetadataField.ANCHOR_TIMESTAMP in fields
    assert pred_config.ignore_time_col is not anchor
    assert pred_config.pred_time_col_name == ('ANCHOR_TIMESTAMP'
                                              if anchor else None)
    assert pred_config.start_time == (start_time if MetadataField.JOB_TIMESTAMP
                                      in fields else None)
    assert pred_config.output_extra_fields == {
        'PREDICTIONS': sorted(field.value for field in fields)
    }


def test_forecasting_prediction_config_keeps_seed_time_without_metadata():
    pred_config = _make_pred_config()
    job_input = SimpleNamespace(
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        config=SimpleNamespace(
            task_type=TaskType.FORECASTING,
            forecast_length=None,
            outputs=[
                SimpleNamespace(
                    artifact_type=PredictionArtifactType.PREDICTIONS,
                    extra_fields=[],
                )
            ],
        ),
    )

    mpt._configure_prediction_output_fields(pred_config, job_input)

    assert pred_config.ignore_time_col is False
    assert pred_config.pred_time_col_name is None
    assert pred_config.output_extra_fields == {'PREDICTIONS': []}


def test_non_forecasting_plan_does_not_infer_task_type_from_horizon():
    pred_config = _make_pred_config()
    job_input = SimpleNamespace(
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        config=SimpleNamespace(
            task_type=TaskType.REGRESSION,
            forecast_length=3,
            outputs=[
                SimpleNamespace(
                    artifact_type=PredictionArtifactType.PREDICTIONS,
                    extra_fields=[],
                )
            ],
        ),
    )

    mpt._configure_prediction_output_fields(pred_config, job_input)

    assert pred_config.ignore_time_col is True
    assert pred_config.pred_time_col_name is None
    assert pred_config.output_extra_fields == {'PREDICTIONS': []}


def test_prediction_config_keeps_artifact_specific_requested_times():
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pred_config = _make_pred_config()
    job_input = SimpleNamespace(
        start_time=start_time,
        config=SimpleNamespace(
            task_type=TaskType.MULTILABEL_CLASSIFICATION,
            forecast_length=None,
            outputs=[
                SimpleNamespace(
                    artifact_type=PredictionArtifactType.PREDICTIONS,
                    extra_fields=[MetadataField.ANCHOR_TIMESTAMP],
                ),
                SimpleNamespace(
                    artifact_type=PredictionArtifactType.EMBEDDINGS,
                    extra_fields=[MetadataField.JOB_TIMESTAMP],
                ),
            ],
        ),
    )

    mpt._configure_prediction_output_fields(pred_config, job_input)

    assert pred_config.ignore_time_col is False
    assert pred_config.pred_time_col_name == 'ANCHOR_TIMESTAMP'
    assert pred_config.start_time == start_time
    assert pred_config.output_extra_fields == {
        'PREDICTIONS': ['ANCHOR_TIMESTAMP'],
        'EMBEDDINGS': ['JOB_TIMESTAMP'],
    }


def _partition_doubles():
    pred_config = SimpleNamespace(return_embeddings=False, top_k=None)
    settings = SimpleNamespace(pred_max_num_rows_per_partition=1000,
                               pred_max_num_entities_per_partition=1000)
    job_input = SimpleNamespace(
        id='bp-job',
        config=SimpleNamespace(num_parallel_workers_limit=1),
    )
    task = SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION)
    return pred_config, settings, job_input, task


def test_output_partitions_from_rows():
    """The TT-NG output partition count is derived from the ingested
    predict-stage row_count (no pred_table file read)."""
    pred_config, settings, job_input, task = _partition_doubles()

    from_rows = mpt._pred_num_partitions_from_rows(2500, pred_config, job_input,
                                                   settings, task,
                                                   id_is_string=False)

    assert from_rows == 3  # int(2500/1000)+1


# --------------------------------------------------------------------------- #
# C-mat: materialize re-ingests from the DURABLE staging when the ephemeral
# in-index staging was torn down (the exact reported failure), and falls back
# to a clear error when neither exists.
# --------------------------------------------------------------------------- #
def _patch_common_real_fs(monkeypatch, calls):
    class FakeService:
        def materialize_prediction_task_dataset(self, handle, spec, **kwargs):
            calls.append((handle, spec, kwargs))
            return SimpleNamespace(
                bundle_dir=spec.artifact_root,
                manifest_path=f'{spec.artifact_root}/m.json',
                predict_row_count=5,
            )

    monkeypatch.setattr('kumo.diskgraph.get_service',
                        lambda: FakeService())
    monkeypatch.setattr(
        'kumo.task.task.collect_encoders',
        lambda _: SimpleNamespace(encoders={'users': {
            'age': object()
        }}))
    _patch_feature_schema(monkeypatch)
    # NOTE: real fsutil.exists / copy here (no exists patch).


def _write_bundle(d, tt_id='pred_bp-job', *, publish=False):
    """Write a current, attempt-scoped no-split prediction bundle."""
    import os

    if publish:
        bundle_dir = write_v3_bundle(
            os.path.dirname(d),
            tt_id,
            ingest_spec_schema_version='diskgraph.tt_ingest_spec.v3',
        )
    else:
        expected = os.path.join(os.path.dirname(d), f'.{tt_id}.staging')
        assert os.path.normpath(d) == os.path.normpath(expected)
        bundle_dir = write_v3_bundle(
            os.path.dirname(d),
            tt_id,
            ingest_spec_schema_version='diskgraph.tt_ingest_spec.v3',
        )
    if publish:
        from kumo.data.generic import diskgraph_train_table
        diskgraph_train_table.persist_durable_ttng_bundle(bundle_dir, d)
    return bundle_dir


def test_ttng_materialize_restores_from_durable_when_index_torn_down(
        monkeypatch, tmp_path):
    """The reported bug: the ephemeral ``<index>/train_tables/.<tt_id>.staging``
    was wiped by delete_snapshot before materialize ran. With the durable
    staging present, materialize restores it into the index and ingests."""
    calls = []
    _patch_common_real_fs(monkeypatch, calls)

    index_path = str(tmp_path / 'index')
    pred_table_path = str(tmp_path / 'home' / 'pred_table')
    # Durable staging present; ephemeral absent (torn-down index).
    durable = f'{pred_table_path}.ttng_staging'
    unpublished = str(tmp_path / '.pred_bp-job.staging')
    _write_bundle(unpublished)
    from kumo.data.generic import diskgraph_train_table
    diskgraph_train_table.persist_durable_ttng_bundle(unpublished, durable)

    pred_config = PredictionConfig(home_dir='/h', train_dir='/t',
                                   task_name='task', model_dir='/m',
                                   output_dir='/o', return_embeddings=False,
                                   return_predictions=True,
                                   pred_table_path=pred_table_path)

    _materialize(task=_node_task(), pquery=_node_pquery(),
                 prediction_config=pred_config, job_input=_make_job_input(),
                 diskgraph_index_path=index_path, train_table_name='train_tt')

    _, _, kwargs = calls[0]
    ephemeral = f'{index_path}/train_tables/.pred_bp-job.staging'
    # Service is pointed at the index-local ephemeral staging (restored from
    # durable), so the DG publish (rename within the index FS) works.
    assert kwargs['predict_staging_dir'] == ephemeral
    import os
    assert os.path.isdir(f'{ephemeral}/tt.parquet')
    assert os.path.isfile(f'{ephemeral}/tt_ingest_spec.json')
    assert pred_config.diskgraph_pred_staging_dir == durable
    assert pred_config.diskgraph_pred_target_tt_id == 'pred_bp-job'


def test_ttng_materialize_decouples_source_and_target_tt_id_for_bp(
        monkeypatch, tmp_path):
    """Real batch prediction: generate_pred_table_job_id != BP job id. The SOURCE
    id is read from the durable bundle spec (gen stamped ``pred_gen-job`` there),
    while the TARGET extension tt_id recorded for predict shards is BP-job-keyed
    (what the materialize-side ingest publishes / the manifest reads). Source !=
    target is the intentional per-BP-job fan-out isolation; NO equality check
    and NO ``or job_input.id`` fallback."""
    calls = []
    _patch_common_real_fs(monkeypatch, calls)

    index_path = str(tmp_path / 'index')
    pred_table_path = str(tmp_path / 'home' / 'pred_table')
    # Gen wrote the staging under the GEN job id and stamped that id into the
    # bundle spec; the durable staging is located by the pred-table path
    # (job-id-independent). The SOURCE id is read back from the spec.
    durable = f'{pred_table_path}.ttng_staging'
    _write_bundle(durable, tt_id='pred_gen-job', publish=True)

    job_input = SimpleNamespace(
        id='bp-job',
        config=SimpleNamespace(generate_pred_table_job_id='gen-job'),
        graph_data_compute_input=SimpleNamespace(graph_key='graph'),
        graph_version_info=SimpleNamespace(graph_data_version='v1'),
    )

    pred_config = PredictionConfig(home_dir='/h', train_dir='/t',
                                   task_name='task', model_dir='/m',
                                   output_dir='/o', return_embeddings=False,
                                   return_predictions=True,
                                   pred_table_path=pred_table_path)

    _materialize(task=_node_task(), pquery=_node_pquery(),
                 prediction_config=pred_config, job_input=job_input,
                 diskgraph_index_path=index_path, train_table_name='train_tt')

    _, spec, kwargs = calls[0]
    # SOURCE id is read from the bundle spec (gen-job-keyed); the ephemeral
    # staging dir it drives is thus gen-job-keyed too.
    ephemeral = f'{index_path}/train_tables/.pred_gen-job.staging'
    assert kwargs['predict_staging_dir'] == ephemeral
    # TARGET extension key the predict subprocess reads == BP-job-keyed (matches
    # service.py's ``pred_<spec.job_id>`` publish, spec.job_id == job_input.id).
    assert spec.job_id == 'bp-job'
    assert pred_config.diskgraph_pred_target_tt_id == 'pred_bp-job'


def test_ttng_materialize_prefers_ephemeral_when_present(monkeypatch, tmp_path):
    """When the in-index ephemeral staging survives (same-pod), no restore copy
    happens — the service uses the ephemeral staging directly. The SOURCE id is
    still read from the durable bundle spec (present beside the pred table)."""
    calls = []
    _patch_common_real_fs(monkeypatch, calls)

    index_path = str(tmp_path / 'index')
    pred_table_path = str(tmp_path / 'home' / 'pred_table')
    durable = f'{pred_table_path}.ttng_staging'
    _write_bundle(durable, publish=True)  # spec source: tt_id == pred_bp-job
    ephemeral = f'{index_path}/train_tables/.pred_bp-job.staging'
    _write_bundle(ephemeral)  # same-pod: ephemeral present -> no restore copy

    pred_config = PredictionConfig(home_dir='/h', train_dir='/t',
                                   task_name='task', model_dir='/m',
                                   output_dir='/o', return_embeddings=False,
                                   return_predictions=True,
                                   pred_table_path=pred_table_path)

    _materialize(task=_node_task(), pquery=_node_pquery(),
                 prediction_config=pred_config, job_input=_make_job_input(),
                 diskgraph_index_path=index_path, train_table_name='train_tt')

    _, _, kwargs = calls[0]
    assert kwargs['predict_staging_dir'] == ephemeral
    # Still records the durable path for downstream shards even on the fast path.
    assert pred_config.diskgraph_pred_staging_dir == f'{pred_table_path}.ttng_staging'


def test_ttng_materialize_raises_when_no_staging_anywhere(
        monkeypatch, tmp_path):
    calls = []
    _patch_common_real_fs(monkeypatch, calls)

    index_path = str(tmp_path / 'index')
    pred_table_path = str(tmp_path / 'home' / 'pred_table')  # nothing written

    pred_config = PredictionConfig(home_dir='/h', train_dir='/t',
                                   task_name='task', model_dir='/m',
                                   output_dir='/o', return_embeddings=False,
                                   return_predictions=True,
                                   pred_table_path=pred_table_path)

    with pytest.raises(mpt.PredictionDataMaterializationError):
        _materialize(task=_node_task(), pquery=_node_pquery(),
                     prediction_config=pred_config, job_input=_make_job_input(),
                     diskgraph_index_path=index_path,
                     train_table_name='train_tt')
    assert calls == []  # service never called


def test_ttng_materialize_graph_key_consistency(monkeypatch, tmp_path):
    """Interaction with #29507: materialize and a simulated predict shard derive
    the SAME index-local extension dir from the same graph key, so the durable
    re-ingest writes/reads the same place on every pod."""
    from kumo.data.generic.diskgraph_staging import (
        predict_staging_tt_id,
        ttng_staging_dir,
    )
    from statedb.config.diskgraph import get_diskgraph_index_path_for_job

    # Same graph key + version threaded everywhere (what #29507 guarantees).
    job = SimpleNamespace(
        graph_data_compute_input=SimpleNamespace(graph_key='graphX'),
        graph_version_info=SimpleNamespace(graph_data_version='v9'),
    )
    idx_a = get_diskgraph_index_path_for_job(job)
    idx_b = get_diskgraph_index_path_for_job(job)
    assert idx_a == idx_b
    tt_id = predict_staging_tt_id('gen-job')
    assert ttng_staging_dir(idx_a, tt_id) == ttng_staging_dir(idx_b, tt_id)
