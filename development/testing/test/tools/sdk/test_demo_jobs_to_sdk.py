# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kumo_core.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumo_core.types import PredictionArtifactType
from kumo_core.types.schema import FPID
from kumoapi.typing import Dtype, Stype

from rest.config.graph_v2 import ColumnKey, ColumnKeyGroup
from rest.config.jobs import JobStatus
from rest.config.prediction_output import PredictionStorageType
from rest.config.table_v2 import Column, S3SourceTable, TableDefinition
from statedb.config.common import ProjectID
from tools.sdk.demo_jobs_to_sdk import (
    ExportError,
    align_query_column_case,
    batch_prediction_skip_reason,
    build_graph_bundle,
    compact_output_config,
    iter_statuses,
    load_metadata,
    render_batch_prediction_script,
    render_prediction_output_connector,
    render_training_script,
    render_tune_metric_check,
    resolve_state_db_root,
    training_skip_reason,
    validate_source_table_ref,
)

PQ_FPID = "a" * 32
GRAPH_FPID = "b" * 32
TABLE_FPID = "c" * 32
PROJECT_ID = ProjectID("project_" + "1" * 32)


def _key(fpid) -> tuple[str, str]:
    return str(PROJECT_ID), str(fpid)


def SOURCE_EXISTS(_path):
    return True


def test_resolve_state_db_root_uses_explicit_root():
    args = SimpleNamespace(
        state_db_root='/tmp/state-db/',
        dataplane='unused',
        customer='unused',
    )

    assert resolve_state_db_root(args) == '/tmp/state-db'


def test_resolve_state_db_root_uses_customer_dataplane_root():
    args = SimpleNamespace(
        state_db_root=None,
        dataplane='s3://dataplane/',
        customer='demo',
    )

    assert resolve_state_db_root(args) == 's3://dataplane/demo/prod/db'


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return False


def _tiny_dbs():
    table = TableDefinition(
        source_table=S3SourceTable(
            connector_id='connector_' + '1' * 32, source={
                'mode': 'uri',
                'uri': 's3://bucket/demo/customers.parquet'
            }, file_type="PARQUET"),
        cols=[
            Column(name="customer_id", stype=Stype.ID, dtype=Dtype.string),
            Column(name="label", stype=Stype.categorical, dtype=Dtype.int64),
        ],
        pkey="customer_id",
        time_col=None,
        end_time_col=None,
    )
    rest_db = SimpleNamespace(
        project_id=PROJECT_ID,
        pquery_definitions_map={
            _key(PQ_FPID):
            SimpleNamespace(
                pquery_name="customer_query",
                query_str=(
                    "predict customers.label = 1 for each customers.customer_id"
                ),
                graph_fpid=FPID(GRAPH_FPID),
            ),
        },
        graphs_v2_map={
            _key(GRAPH_FPID):
            SimpleNamespace(
                table_name_to_dbid={"customers": FPID(TABLE_FPID)},
                col_groups=[
                    ColumnKeyGroup(columns=(ColumnKey(
                        table_name="customers", col_name="customer_id"), ))
                ],
            ),
        },
        tables_v2_map={
            _key(TABLE_FPID): SimpleNamespace(config=table),
        },
    )
    job = SimpleNamespace(
        id="trainingjob-test",
        advanced_options=ModelPlan(),
        pquery_def_fpid=FPID(PQ_FPID),
        generate_train_table_job_id=None,
        execution=SimpleNamespace(
            status=JobStatus.DONE,
            start_time=datetime(2025, 1, 1),
            last_updated_at=datetime(2025, 1, 1),
        ),
    )
    ml_db = SimpleNamespace(
        training_job_map={"trainingjob-test": job},
        batch_prediction_jobs_map={},
        generate_train_table_jobs_map={},
    )
    return rest_db, ml_db, job


@pytest.mark.asyncio
async def test_load_metadata_scopes_duplicate_fpids_to_selected_project(
        monkeypatch):
    project_a = PROJECT_ID
    project_b = ProjectID("project_" + "2" * 32)
    shared_fpid = FPID(TABLE_FPID)

    async def list_graphs(project_id):
        return [SimpleNamespace(id=FPID(GRAPH_FPID), project_id=project_id)]

    async def list_tables(project_id):
        return [
            SimpleNamespace(
                id=shared_fpid,
                project_id=project_id,
                config=f"table-for-{project_id}",
            )
        ]

    jobs = SimpleNamespace(
        list_job_ids_by_type=AsyncMock(return_value=[]),
        get_training_jobs=AsyncMock(return_value=[]),
        get_batch_prediction_jobs=AsyncMock(return_value=[]),
        get_generate_train_table_jobs=AsyncMock(return_value=[]),
    )
    services = SimpleNamespace(
        pqueries=SimpleNamespace(list_definitions=AsyncMock(return_value=[])),
        graphs=SimpleNamespace(list_graphs=list_graphs),
        tables=SimpleNamespace(list_tables=list_tables),
        ml_jobs=jobs,
    )
    monkeypatch.setattr(
        "tools.sdk.demo_jobs_to_sdk.open_state_db_from_deploy_config",
        AsyncMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr("tools.sdk.demo_jobs_to_sdk.StateServices",
                        lambda _db: services)

    snapshot_a, _ = await load_metadata("/tmp/a", project_a)
    snapshot_b, _ = await load_metadata("/tmp/b", project_b)

    assert snapshot_a.tables_v2_map[_key(shared_fpid)].config == (
        f"table-for-{project_a}")
    assert snapshot_b.tables_v2_map[(
        str(project_b), str(shared_fpid))].config == (f"table-for-{project_b}")
    assert (str(project_b), str(shared_fpid)) not in snapshot_a.tables_v2_map


def test_training_job_discovery_marks_s3_job_exportable():
    rest_db, ml_db, _ = _tiny_dbs()

    statuses = list(
        iter_statuses(rest_db, ml_db, "training", [],
                      source_exists_fn=SOURCE_EXISTS))

    assert len(statuses) == 1
    assert statuses[0].exportable
    assert statuses[0].reason == "exportable"


def test_render_training_script_is_flat_s3_and_omits_inferred_defaults():
    rest_db, ml_db, job = _tiny_dbs()

    script = render_training_script(
        rest_db,
        ml_db,
        job,
        api_url="http://localhost:8000",
        auth_env_var="KUMO_API_KEY",
        source_exists_fn=SOURCE_EXISTS,
    )

    assert "def " not in script
    assert "kumo.S3Connector(" in script
    assert "get_by_name" not in script
    assert "s3_connector = kumo.S3Connector(\n)" in script
    assert ("source_table=s3_connector.table("
            "'s3://bucket/demo/customers.parquet')") in script
    assert "train_table = pquery.generate_training_table()" in script
    assert "'inferred'" not in script
    assert "model_plan = {}" in script


def test_training_job_discovery_keeps_latest_successful_per_dataset_pquery():
    rest_db, ml_db, job = _tiny_dbs()
    older_job = SimpleNamespace(**job.__dict__)
    older_job.id = "trainingjob-older"
    older_job.execution = SimpleNamespace(
        status=JobStatus.DONE,
        start_time=datetime(2024, 1, 1),
        last_updated_at=datetime(2024, 1, 1),
    )
    newer_job = SimpleNamespace(**job.__dict__)
    newer_job.id = "trainingjob-newer"
    newer_job.execution = SimpleNamespace(
        status=JobStatus.DONE,
        start_time=datetime(2025, 1, 1),
        last_updated_at=datetime(2025, 1, 1),
    )
    ml_db.training_job_map = {
        older_job.id: older_job,
        newer_job.id: newer_job,
    }

    statuses = list(
        iter_statuses(rest_db, ml_db, "training", [],
                      source_exists_fn=SOURCE_EXISTS))

    by_id = {status.job_id: status for status in statuses}
    assert by_id["trainingjob-newer"].exportable
    assert not by_id["trainingjob-older"].exportable
    assert "older successful duplicate" in by_id["trainingjob-older"].reason


def test_render_training_script_embeds_expected_tune_metric_partial_failure_check(
):
    rest_db, ml_db, job = _tiny_dbs()
    job.advanced_options.training_job.tune_metric = "auroc"
    job.execution.results = SimpleNamespace(
        test_metrics=[SimpleNamespace(name="auroc", val=0.8123)])

    script = render_training_script(
        rest_db,
        ml_db,
        job,
        api_url="http://localhost:8000",
        auth_env_var="KUMO_API_KEY",
        source_exists_fn=SOURCE_EXISTS,
    )

    assert "EXPECTED_TUNE_METRIC_NAME = 'auroc'" in script
    assert "EXPECTED_TUNE_METRIC_VALUE = 0.8123" in script
    assert "KUMO_TUNE_METRIC_REL_TOL" in script
    assert "non_blocking=False" in script
    assert "metrics_fn = getattr(candidate, \"metrics\", None)" in script
    assert "replay_test_metrics" in script
    assert "metric_value = _replay_field(metric, \"value\", \"val\")" in script
    assert "PARTIAL_FAILURES.append" in script
    assert "PARTIAL_FAILURE = bool(PARTIAL_FAILURES)" in script


def _job_with_expected_tune_metric():
    _rest_db, _ml_db, job = _tiny_dbs()
    job.advanced_options.training_job.tune_metric = "auroc"
    job.execution.results = SimpleNamespace(
        test_metrics=[SimpleNamespace(name="auroc", val=0.8123)])
    return job


def _run_tune_metric_check(training_job):
    namespace = {
        "os": os,
        "training_job": training_job,
    }
    exec(render_tune_metric_check(_job_with_expected_tune_metric()), namespace)
    return namespace


class _SdkTrainingJobResult:
    def metrics(self):
        return SimpleNamespace(test_metrics=[
            SimpleNamespace(name="auroc", value=0.8123),
        ])


@pytest.mark.parametrize(
    "training_job",
    [
        _SdkTrainingJobResult(),
        SimpleNamespace(result=SimpleNamespace(eval_metrics=SimpleNamespace(
            test_metrics=[SimpleNamespace(name="auroc", value=0.8123)]))),
        SimpleNamespace(results=SimpleNamespace(
            test_metrics=[SimpleNamespace(name="auroc", val=0.8123)])),
        {
            "result": {
                "eval_metrics": {
                    "test_metrics": [{
                        "name": "auroc",
                        "value": 0.8123
                    }],
                },
            },
        },
    ],
)
def test_rendered_tune_metric_lookup_accepts_sdk_and_compat_shapes(
        training_job):
    namespace = _run_tune_metric_check(training_job)

    assert not namespace["PARTIAL_FAILURES"]
    assert namespace["actual_test_tune_metric_value"] == 0.8123


def test_rendered_tune_metric_lookup_checks_test_metrics_only():
    training_job = SimpleNamespace(result=SimpleNamespace(
        eval_metrics=SimpleNamespace(
            validation_metrics=[SimpleNamespace(name="auroc", value=0.8123)],
            test_metrics=[],
        )))

    namespace = _run_tune_metric_check(training_job)

    assert namespace["PARTIAL_FAILURES"] == [
        "missing replay test tune metric 'auroc'"
    ]


def test_training_skip_reason_skips_forecasting_style_plan():
    rest_db, ml_db, job = _tiny_dbs()
    job.advanced_options.training_table_generation.forecast_length = 30

    reason = training_skip_reason(rest_db, ml_db, job,
                                  source_exists_fn=SOURCE_EXISTS)

    assert reason is not None
    assert "unsupported forecasting-style training job" in reason
    assert "forecast_length=30" in reason


def test_batch_prediction_skip_reason_skips_forecasting_style_plan():
    rest_db, ml_db, training_job = _tiny_dbs()
    batch_job = SimpleNamespace(
        id="bp-job-test",
        execution=SimpleNamespace(status=JobStatus.DONE),
        config=SimpleNamespace(
            outputs=[SimpleNamespace(storage_type=PredictionStorageType.S3)],
            training_job_id=training_job.id,
            pquery_def_fpid=FPID(PQ_FPID),
            forecast_length=7,
            lag_timesteps=None,
            year_over_year=False,
        ),
    )

    reason = batch_prediction_skip_reason(rest_db, ml_db, batch_job,
                                          source_exists_fn=SOURCE_EXISTS)

    assert reason is not None
    assert "unsupported forecasting-style batch prediction" in reason
    assert "forecast_length=7" in reason


def _product_sales_dbs():
    product_fpid = FPID("d" * 32)
    sales_fpid = FPID("e" * 32)
    products = TableDefinition(
        source_table=S3SourceTable(
            connector_id='connector_' + '1' * 32, source={
                'mode': 'uri',
                'uri': 's3://bucket/walmart/Products.parquet'
            }, file_type="PARQUET"),
        cols=[Column(name="ProductId", stype=Stype.ID, dtype=Dtype.int64)],
        pkey="ProductId",
        time_col=None,
        end_time_col=None,
    )
    sales = TableDefinition(
        source_table=S3SourceTable(
            connector_id='connector_' + '1' * 32, source={
                'mode': 'uri',
                'uri': 's3://bucket/walmart/Sales.parquet'
            }, file_type="PARQUET"),
        cols=[
            Column(name="SalesId", stype=Stype.ID, dtype=Dtype.int64),
            Column(name="ProductId", stype=Stype.ID, dtype=Dtype.int64),
        ],
        pkey="SalesId",
        time_col=None,
        end_time_col=None,
    )
    rest_db, ml_db, job = _tiny_dbs()
    rest_db.pquery_definitions_map[_key(PQ_FPID)].query_str = (
        "predict avg(sales.SalesId) for each products.ProductId")
    rest_db.graphs_v2_map[_key(GRAPH_FPID)] = SimpleNamespace(
        table_name_to_dbid={
            "products": product_fpid,
            "sales": sales_fpid
        },
        col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey(table_name="products", col_name="ProductId"),
                ColumnKey(table_name="sales", col_name="ProductId"),
            ))
        ],
    )
    rest_db.tables_v2_map = {
        _key(product_fpid): SimpleNamespace(config=products),
        _key(sales_fpid): SimpleNamespace(config=sales),
    }
    return rest_db, ml_db, job


def test_render_training_script_remaps_model_plan_tables_to_graph_aliases():
    rest_db, ml_db, job = _product_sales_dbs()
    job.advanced_options.neighbor_sampling.num_neighbors = [{
        "hop1": {
            "default": 12,
            "Products.ProductId->Sales.ProductId": "inferred",
        }
    }]

    script = render_training_script(
        rest_db,
        ml_db,
        job,
        api_url="http://localhost:8000",
        auth_env_var="KUMO_API_KEY",
        source_exists_fn=SOURCE_EXISTS,
    )

    assert "products.ProductId->sales.ProductId" in script
    assert "Products.ProductId->Sales.ProductId" not in script


def test_prediction_output_connector_uses_runtime_run_id():
    snippet = render_prediction_output_connector(
        "s3://bucket/demo_outputs/bp-job-test", "KUMO_DEMO_SDK_OUTPUT_RUN_ID")

    assert "KUMO_DEMO_SDK_OUTPUT_RUN_ID" in snippet
    assert "uuid.uuid4().hex" in snippet
    assert "prediction_output_run_id" in snippet
    assert "prediction_output_root.rstrip" in snippet


def test_missing_source_table_is_classified_before_script_execution():
    rest_db, ml_db, _job = _tiny_dbs()

    statuses = list(
        iter_statuses(rest_db, ml_db, "training", [],
                      source_exists_fn=lambda _path: False))

    assert len(statuses) == 1
    assert not statuses[0].exportable
    assert "missing source table customers" in statuses[0].reason


def test_file_upload_source_table_is_classified_as_unreplayable():
    source = S3SourceTable(
        connector_id="connector-project-upload",
        source={
            'mode': 'relative',
            'path': "uploaded_events_ID"
        },
        file_type="PARQUET",
    )

    with pytest.raises(ExportError, match="file-upload source table"):
        validate_source_table_ref("uploaded_events", source, SOURCE_EXISTS)


def _s3_output(artifact_type, file_path):
    return SimpleNamespace(
        storage_type=PredictionStorageType.S3,
        artifact_type=artifact_type,
        file_path=file_path,
    )


def _batch_job(training_job, job_id="bp-job-test", *, outputs=None,
               pquery_def_fpid=None, forecast_length=None):
    if outputs is None:
        outputs = [
            _s3_output(PredictionArtifactType.PREDICTIONS,
                       "s3://out/predictions.parquet")
        ]
    return SimpleNamespace(
        id=job_id,
        execution=SimpleNamespace(status=JobStatus.DONE),
        config=SimpleNamespace(
            outputs=outputs,
            training_job_id=training_job.id,
            pquery_def_fpid=pquery_def_fpid,
            prediction_time=None,
            forecast_length=forecast_length,
            lag_timesteps=None,
            year_over_year=False,
            filter_override_args=None,
            binary_classification_threshold=None,
            num_classes_to_return=None,
            num_parallel_workers_limit=1,
        ),
    )


def _set_single_source(rest_db, *, graph_name, source_table_name, s3_path,
                       connector_id=None):
    table = rest_db.tables_v2_map[_key(TABLE_FPID)].config
    table.source_table = S3SourceTable(
        connector_id=connector_id or 'connector_' + '1' * 32,
        source=({
            'mode': 'uri',
            'uri': s3_path,
        } if s3_path else {
            'mode': 'relative',
            'path': source_table_name,
        }),
        file_type="PARQUET",
    )
    rest_db.graphs_v2_map[_key(GRAPH_FPID)].table_name_to_dbid = {
        graph_name: FPID(TABLE_FPID)
    }
    rest_db.graphs_v2_map[_key(GRAPH_FPID)].col_groups = [
        ColumnKeyGroup(columns=(
            ColumnKey(table_name=graph_name, col_name="customer_id"), ))
    ]


@pytest.mark.parametrize(
    "job_id,forecast_length,lag_timesteps",
    [
        ("trainingjob-00000000000000000000000000000001", 12, None),
        ("trainingjob-00000000000000000000000000000002", 4, None),
        ("trainingjob-00000000000000000000000000000003", 30, 14),
        ("trainingjob-00000000000000000000000000000004", 2, 2),
    ],
)
def test_known_bug005_forecasting_jobs_are_skipped_by_id(
        job_id, forecast_length, lag_timesteps):
    rest_db, ml_db, job = _tiny_dbs()
    job.id = job_id
    job.advanced_options.training_table_generation.forecast_length = forecast_length
    job.advanced_options.training_table_generation.lag_timesteps = lag_timesteps
    ml_db.training_job_map = {job_id: job}

    statuses = list(
        iter_statuses(rest_db, ml_db, "training", [job_id],
                      source_exists_fn=SOURCE_EXISTS))

    assert len(statuses) == 1
    assert statuses[0].job_id == job_id
    assert not statuses[0].exportable
    assert "unsupported forecasting-style training job" in statuses[0].reason
    assert f"forecast_length={forecast_length}" in statuses[0].reason


def test_training_skip_reason_prefers_generated_training_table_plan():
    rest_db, ml_db, job = _tiny_dbs()
    job.advanced_options.training_table_generation.forecast_length = 1
    job.generate_train_table_job_id = "gen-tt-job"
    generated_plan = TrainingTableGenerationPlan()
    generated_plan.forecast_length = 12
    ml_db.generate_train_table_jobs_map = {
        "gen-tt-job":
        SimpleNamespace(config=SimpleNamespace(plan=generated_plan)),
    }

    reason = training_skip_reason(rest_db, ml_db, job,
                                  source_exists_fn=SOURCE_EXISTS)

    assert reason is not None
    assert "unsupported forecasting-style training job" in reason
    assert "forecast_length=12" in reason


def test_lag_timesteps_only_skip_is_not_labelled_forecasting():
    rest_db, ml_db, job = _tiny_dbs()
    job.advanced_options.training_table_generation.forecast_length = 1
    job.advanced_options.training_table_generation.lag_timesteps = 14

    reason = training_skip_reason(rest_db, ml_db, job,
                                  source_exists_fn=SOURCE_EXISTS)

    assert reason == "unsupported lag-timesteps training job: lag_timesteps=14"


def test_year_over_year_false_string_does_not_skip():
    rest_db, ml_db, job = _tiny_dbs()
    job.advanced_options.training_table_generation.year_over_year = "False"

    reason = training_skip_reason(rest_db, ml_db, job,
                                  source_exists_fn=SOURCE_EXISTS)

    assert reason is None


@pytest.mark.parametrize(
    "job_id,graph_name,source_table_name,s3_path",
    [
        (
            "trainingjob-00000000000000000000000000000005",
            "admissions",
            "admissions",
            "s3://example-bucket/demo/prod/MIMIC_III_v1.4/core/admissions.parquet",
        ),
        (
            "trainingjob-00000000000000000000000000000006",
            "admissions",
            "admissions",
            "s3://example-bucket/demo/prod/MIMIC_III_v1.4/core/admissions.parquet",
        ),
    ],
)
def test_known_bug011_missing_training_sources_are_skipped_by_id(
        job_id, graph_name, source_table_name, s3_path):
    rest_db, ml_db, job = _tiny_dbs()
    job.id = job_id
    ml_db.training_job_map = {job_id: job}
    _set_single_source(
        rest_db,
        graph_name=graph_name,
        source_table_name=source_table_name,
        s3_path=s3_path,
    )

    statuses = list(
        iter_statuses(rest_db, ml_db, "training", [job_id],
                      source_exists_fn=lambda _path: False))

    assert len(statuses) == 1
    assert statuses[0].job_id == job_id
    assert not statuses[0].exportable
    assert f"missing source table {graph_name}" in statuses[0].reason
    assert s3_path in statuses[0].reason


@pytest.mark.parametrize(
    "job_id,source_table_name",
    [
        ("bp-job-00000000000000000000000000000007", "transactions"),
        ("bp-job-00000000000000000000000000000008", "devices"),
    ],
)
def test_known_bug011_missing_batch_sources_are_skipped_by_id(
        job_id, source_table_name):
    rest_db, ml_db, training_job = _tiny_dbs()
    missing_path = f"s3://example-bucket/demo/prod/game_store/{source_table_name}.parquet"
    _set_single_source(
        rest_db,
        graph_name=source_table_name,
        source_table_name=source_table_name,
        s3_path=missing_path,
    )
    batch_job = _batch_job(training_job, job_id=job_id)
    ml_db.batch_prediction_jobs_map = {job_id: batch_job}

    statuses = list(
        iter_statuses(rest_db, ml_db, "batch_prediction", [job_id],
                      source_exists_fn=lambda _path: False))

    assert len(statuses) == 1
    assert statuses[0].job_id == job_id
    assert not statuses[0].exportable
    assert "training job cannot be exported: missing source table" in statuses[
        0].reason
    assert source_table_name in statuses[0].reason
    assert missing_path in statuses[0].reason


def test_known_bug011_uploaded_training_source_is_skipped_by_id():
    rest_db, ml_db, job = _tiny_dbs()
    job.id = "trainingjob-00000000000000000000000000000009"
    ml_db.training_job_map = {job.id: job}
    _set_single_source(
        rest_db,
        graph_name="uploaded_events_ID",
        source_table_name="uploaded_events_ID",
        s3_path="",
        connector_id="connector-project-upload",
    )

    statuses = list(
        iter_statuses(rest_db, ml_db, "training", [job.id],
                      source_exists_fn=SOURCE_EXISTS))

    assert len(statuses) == 1
    assert statuses[0].job_id == job.id
    assert not statuses[0].exportable
    assert "file-upload source table uploaded_events_ID" in statuses[0].reason
    assert "connector-project-upload" in statuses[0].reason


def _table_def(source_table_name, pkey, columns):
    names = []
    for name in (pkey, *columns):
        if name not in names:
            names.append(name)
    return TableDefinition(
        source_table=S3SourceTable(
            connector_id='connector_' + '1' * 32,
            source={
                'mode': 'uri',
                'uri': f's3://bucket/demo/{source_table_name}.parquet',
            },
            file_type="PARQUET",
        ),
        cols=[
            Column(name=name, stype=Stype.ID, dtype=Dtype.int64)
            for name in names
        ],
        pkey=pkey,
        time_col=None,
        end_time_col=None,
    )


def _model_plan_graph_dbs(table_specs, col_groups, query):
    rest_db, ml_db, job = _tiny_dbs()
    table_name_to_dbid = {}
    tables = {}
    for idx, spec in enumerate(table_specs):
        graph_name, source_table_name, pkey, columns = spec
        fpid = FPID(f"{idx + 10:032x}")
        table_name_to_dbid[graph_name] = fpid
        tables[_key(fpid)] = SimpleNamespace(
            config=_table_def(source_table_name, pkey, columns))
    rest_db.pquery_definitions_map[_key(PQ_FPID)].query_str = query
    rest_db.graphs_v2_map[_key(GRAPH_FPID)] = SimpleNamespace(
        table_name_to_dbid=table_name_to_dbid,
        col_groups=[
            ColumnKeyGroup(columns=tuple(
                ColumnKey(table_name=table_name, col_name=col_name)
                for table_name, col_name in group)) for group in col_groups
        ],
    )
    rest_db.tables_v2_map = tables
    return rest_db, ml_db, job


@pytest.mark.parametrize(
    "job_id,table_specs,col_groups,stale_key,stale_value,expected_key",
    [
        (
            "trainingjob-0000000000000000000000000000000a",
            (("products", "Products", "ProductId", ()),
             ("sales", "Sales", "SalesId", ("ProductId", ))),
            ((("products", "ProductId"), ("sales", "ProductId")), ),
            "Products.ProductId->Sales.ProductId",
            "inferred",
            "products.ProductId->sales.ProductId",
        ),
        (
            "trainingjob-0000000000000000000000000000000b",
            (("products", "Products", "ProductId", ()),
             ("sales", "Sales", "SalesId", ("ProductId", ))),
            ((("products", "ProductId"), ("sales", "ProductId")), ),
            "Products.ProductId->Sales.ProductId",
            "inferred",
            "products.ProductId->sales.ProductId",
        ),
        (
            "trainingjob-0000000000000000000000000000000c",
            (("stores", "Stores", "StoreId", ()), ("sales", "Sales", "SalesId",
                                                   ("StoreId", ))),
            ((("stores", "StoreId"), ("sales", "StoreId")), ),
            "Stores.StoreId->Sales.StoreId",
            "inferred",
            "stores.StoreId->sales.StoreId",
        ),
        (
            "trainingjob-0000000000000000000000000000000d",
            (("user_profiles", "user_profiles", "USER_ID", ()),
             ("user_events", "USER_EVENTS", "USER_ID", ())),
            ((("user_profiles", "USER_ID"), ("user_events", "USER_ID")), ),
            "user_profiles.USER_ID->USER_EVENTS.USER_ID",
            "inferred",
            "user_profiles.USER_ID->user_events.USER_ID",
        ),
        (
            "trainingjob-0000000000000000000000000000000e",
            (("copurchase", "CoPurchase", "ProductId1",
              ("ProductId2", )), ("products", "Products", "ProductId", ()),
             ("products_rhs", "Products", "ProductId", ())),
            ((("products", "ProductId"), ("copurchase", "ProductId1")),
             (("products_rhs", "ProductId"), ("copurchase", "ProductId2"))),
            "Products.ProductId->CoPurchase.ProductId1",
            "inferred",
            "products.ProductId->copurchase.ProductId1",
        ),
        (
            "trainingjob-spaced-alias",
            (("Sales Facts", "Sales_Facts", "ProductId", ()),
             ("Products", "Products", "ProductId", ())),
            ((("Products", "ProductId"), ("Sales Facts", "ProductId")), ),
            "Sales Facts.ProductId->Products.ProductId",
            3,
            "sales_facts.ProductId->products.ProductId",
        ),
    ],
)
def test_known_bug012_model_plan_refs_remap_to_generated_graph_aliases(
        job_id, table_specs, col_groups, stale_key, stale_value, expected_key):
    rest_db, ml_db, job = _model_plan_graph_dbs(
        table_specs, col_groups,
        "predict avg(sales.SalesId) for each products.ProductId")
    job.id = job_id
    ml_db.training_job_map = {job_id: job}
    job.advanced_options.neighbor_sampling.num_neighbors = [{
        "hop1": {
            "default": 12,
            stale_key: stale_value,
        },
    }]

    script = render_training_script(
        rest_db,
        ml_db,
        job,
        api_url="http://localhost:8000",
        auth_env_var="KUMO_API_KEY",
        source_exists_fn=SOURCE_EXISTS,
    )

    assert expected_key in script
    assert f"{expected_key!r}: {stale_value!r}" in script
    assert stale_key not in script
    assert "default" in script
    assert "training_job" in script


def test_compact_output_config_keeps_s3_multi_output_table_name_string():
    config = SimpleNamespace(outputs=[
        _s3_output(PredictionArtifactType.PREDICTIONS,
                   "s3://old/path/predictions.parquet"),
        _s3_output(PredictionArtifactType.EMBEDDINGS,
                   "s3://old/path/embeddings.parquet"),
    ])

    output_root, output_config = compact_output_config(config, "s3://new-root",
                                                       "bp-job-test")

    assert output_root == "s3://new-root/bp-job-test"
    assert output_config["output_types"] == ["EMBEDDINGS", "PREDICTIONS"]
    assert output_config["output_table_name"] == "bp_job_test"
    assert isinstance(output_config["output_table_name"], str)


def test_render_batch_prediction_script_supports_multi_output_s3_config():
    rest_db, ml_db, training_job = _tiny_dbs()
    batch_job = _batch_job(
        training_job,
        outputs=[
            _s3_output(PredictionArtifactType.PREDICTIONS,
                       "s3://old/path/predictions.parquet"),
            _s3_output(PredictionArtifactType.EMBEDDINGS,
                       "s3://old/path/embeddings.parquet"),
        ],
    )
    ml_db.batch_prediction_jobs_map = {batch_job.id: batch_job}

    script = render_batch_prediction_script(
        rest_db,
        ml_db,
        batch_job,
        api_url="http://localhost:8000",
        auth_env_var="KUMO_API_KEY",
        bp_output_root="s3://new-root",
        source_exists_fn=SOURCE_EXISTS,
    )

    assert "output_table_name" in script
    assert "bp_job_test" in script
    assert "EMBEDDINGS" in script
    assert "PREDICTIONS" in script
    assert "prediction_output_run_id" in script


def _aml_account_dbs():
    account_fpid = FPID("f" * 32)
    account = TableDefinition(
        source_table=S3SourceTable(
            connector_id='connector_' + '1' * 32,
            source={
                'mode': 'uri',
                'uri': 's3://bucket/aml/account.parquet',
            },
            file_type="PARQUET",
        ),
        cols=[
            Column(name="ACCOUNT_ID", stype=Stype.ID, dtype=Dtype.int64),
            Column(name="IS_LAUNDERING", stype=Stype.categorical,
                   dtype=Dtype.int64),
        ],
        pkey="ACCOUNT_ID",
        time_col=None,
        end_time_col=None,
    )
    rest_db, ml_db, job = _tiny_dbs()
    rest_db.pquery_definitions_map[_key(PQ_FPID)].query_str = (
        "PREDICT account.Is_Laundering=1 FOR EACH account.account_ID")
    rest_db.graphs_v2_map[_key(GRAPH_FPID)] = SimpleNamespace(
        table_name_to_dbid={"account": account_fpid},
        col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey(table_name="account", col_name="ACCOUNT_ID"), ))
        ],
    )
    rest_db.tables_v2_map = {
        _key(account_fpid): SimpleNamespace(config=account)
    }
    return rest_db, ml_db, job


def test_build_graph_bundle_aligns_pquery_column_case_with_emitted_graph():
    rest_db, _ml_db, _job = _aml_account_dbs()

    bundle = build_graph_bundle(rest_db, PQ_FPID,
                                source_exists_fn=SOURCE_EXISTS)

    assert "account.ACCOUNT_ID" in bundle.query
    assert "account.IS_LAUNDERING" in bundle.query
    assert "account_ID" not in bundle.query
    assert "Is_Laundering" not in bundle.query


def test_render_training_script_aligns_pquery_column_case_for_aml_bug008():
    rest_db, ml_db, job = _aml_account_dbs()

    script = render_training_script(
        rest_db,
        ml_db,
        job,
        api_url="http://localhost:8000",
        auth_env_var="KUMO_API_KEY",
        source_exists_fn=SOURCE_EXISTS,
    )

    assert "name='ACCOUNT_ID'" in script
    assert "name='IS_LAUNDERING'" in script
    assert "account.ACCOUNT_ID" in script
    assert "account.IS_LAUNDERING" in script
    assert "account.account_ID" not in script
    assert "account.Is_Laundering" not in script


def test_align_query_column_case_preserves_source_case_when_already_matched():
    rest_db, _ml_db, _job = _product_sales_dbs()

    bundle = build_graph_bundle(rest_db, PQ_FPID,
                                source_exists_fn=SOURCE_EXISTS)

    assert "sales.SalesId" in bundle.query
    assert "products.ProductId" in bundle.query
    assert "SALESID" not in bundle.query
    assert "PRODUCTID" not in bundle.query


def test_align_query_column_case_raises_when_column_absent_from_table():
    table_ref = SimpleNamespace(
        var_name="account",
        table=SimpleNamespace(cols=[
            SimpleNamespace(name="ACCOUNT_ID"),
        ]),
    )

    with pytest.raises(ExportError, match="account.does_not_exist"):
        align_query_column_case(
            "PREDICT account.does_not_exist FOR EACH account.ACCOUNT_ID",
            [table_ref],
        )


def test_align_query_column_case_leaves_unknown_table_references_alone():
    table_ref = SimpleNamespace(
        var_name="account",
        table=SimpleNamespace(cols=[
            SimpleNamespace(name="ACCOUNT_ID"),
        ]),
    )

    out = align_query_column_case(
        "PREDICT account.ACCOUNT_ID WHERE timestamp > date.parse('2024-01-01')",
        [table_ref],
    )

    assert "date.parse" in out
    assert "account.ACCOUNT_ID" in out


def test_batch_prediction_skip_reason_skips_alternate_pquery_graph():
    rest_db, ml_db, training_job = _tiny_dbs()
    alt_pq_fpid = FPID("1" * 32)
    alt_graph_fpid = FPID("2" * 32)
    rest_db.pquery_definitions_map[_key(alt_pq_fpid)] = SimpleNamespace(
        pquery_name="alternate_query",
        query_str="predict customers.label = 1 for each customers.customer_id",
        graph_fpid=alt_graph_fpid,
    )
    rest_db.graphs_v2_map[_key(alt_graph_fpid)] = SimpleNamespace(
        table_name_to_dbid={"customers": FPID(TABLE_FPID)},
        col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey(table_name="customers", col_name="customer_id"), )),
        ],
    )
    batch_job = _batch_job(training_job, pquery_def_fpid=alt_pq_fpid)

    reason = batch_prediction_skip_reason(rest_db, ml_db, batch_job,
                                          source_exists_fn=SOURCE_EXISTS)

    assert reason is not None
    assert "alternate batch prediction pquery graph is not supported" in reason
    with pytest.raises(ExportError,
                       match="alternate batch prediction pquery graph"):
        render_batch_prediction_script(
            rest_db,
            ml_db,
            batch_job,
            api_url="http://localhost:8000",
            auth_env_var="KUMO_API_KEY",
            bp_output_root="s3://new-root",
            source_exists_fn=SOURCE_EXISTS,
        )
