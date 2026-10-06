#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export persisted demo jobs as flat, replayable Kumo SDK scripts.

Examples:
    tools/sdk/demo_jobs_to_sdk.py --project-id project_... --dry-run
    tools/sdk/demo_jobs_to_sdk.py --project-id project_... --job-id trainingjob-...
    tools/sdk/demo_jobs_to_sdk.py --project-id project_... --customer demo
    tools/sdk/demo_jobs_to_sdk.py --project-id project_... --state-db-root /tmp/db --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import keyword
import pprint
import re
from dataclasses import dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import kumoapi.graph as graph_api
from kumo_core.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumo_core.types.schema import FPID
from pydantic import BaseModel

from io_utils import fsutil
from rest.config.batch_prediction_job import BatchPredictionJobConfig
from rest.config.jobs import JobStatus
from rest.config.prediction_output import PredictionStorageType
from rest.config.table_v2 import S3SourceTable, TableDefinition
from rest.utils.ui_to_sdk_utils import (
    clean_up_pquery_string,
    recursively_convert_dataclass_to_dict,
    render_str_vals_as_strings,
)
from service_framework.deploy_env_config import deploy_env
from statedb.config.common import (
    BatchPredictionJobID,
    GenerateTrainTableJobID,
    JobType,
    ProjectID,
    TrainingJobID,
)
from statedb.db.config import open_state_db_from_deploy_config
from statedb.db.services import StateServices
from statedb.rest import rest_api_db_entries

DEFAULT_CUSTOMER = "demo"
DEFAULT_DATAPLANE = "s3://kumo-prod-dataplane-bucket"
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_AUTH_ENV_VAR = "KUMO_API_KEY"
DEFAULT_BP_OUTPUT_ROOT = "s3://kumocloud-test-bucket/demo_jobs_to_sdk_outputs"
DEFAULT_BP_OUTPUT_RUN_ID_ENV = "KUMO_DEMO_SDK_OUTPUT_RUN_ID"
MISSING = object()
SourceExistsFn = Callable[[str], bool]


@dataclass(frozen=True)
class RestMetadataSnapshot:
    project_id: ProjectID
    pquery_definitions_map: Mapping[Any, Any]
    graphs_v2_map: Mapping[Any, Any]
    tables_v2_map: Mapping[Any, Any]


@dataclass(frozen=True)
class MLJobSnapshot:
    training_job_map: Mapping[Any, Any]
    batch_prediction_jobs_map: Mapping[Any, Any]
    generate_train_table_jobs_map: Mapping[Any, Any]


@dataclass(frozen=True)
class ExportStatus:
    job_type: str
    job_id: str
    exportable: bool
    reason: str


@dataclass(frozen=True)
class TableRef:
    graph_name: str
    var_name: str
    source_table_name: str
    connector_var: str
    connector_root_dir: str
    table: TableDefinition


@dataclass(frozen=True)
class GraphBundle:
    pquery_name: str
    query: str
    graph_fpid: str
    table_refs: list[TableRef]
    edges: list[graph_api.Edge]


@dataclass(frozen=True)
class QueryTableRef:
    source_table_name: str
    table_alias: Optional[str] = None
    table_alias_in_graph: Optional[str] = None


class ExportError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export demo StateDB jobs as Kumo SDK scripts.")
    parser.add_argument("--customer", default=DEFAULT_CUSTOMER)
    parser.add_argument("--dataplane", default=DEFAULT_DATAPLANE)
    parser.add_argument("--state-db-root")
    parser.add_argument(
        "--project-id",
        required=True,
        help="Project whose jobs and canonical table/graph metadata to export",
    )
    parser.add_argument("--output-dir", default="demo_sdk_jobs")
    parser.add_argument("--dry-run", "--list", action="store_true")
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument(
        "--job-type",
        choices=("all", "training", "batch_prediction"),
        default="all",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", dest="auth_env_var",
                        default=DEFAULT_AUTH_ENV_VAR)
    parser.add_argument("--bp-output-root", default=DEFAULT_BP_OUTPUT_ROOT)
    parser.add_argument("--bp-output-run-id-env",
                        default=DEFAULT_BP_OUTPUT_RUN_ID_ENV)
    parser.add_argument("--skip-source-table-check", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async())


async def main_async() -> None:
    args = parse_args()
    state_db_root = resolve_state_db_root(args)
    rest_db, ml_db = await load_metadata(
        state_db_root,
        ProjectID(args.project_id),
    )

    source_exists_fn = None if args.skip_source_table_check else fsutil.exists
    statuses = list(
        iter_statuses(
            rest_db,
            ml_db,
            args.job_type,
            args.job_id,
            source_exists_fn=source_exists_fn,
        ))
    for status in statuses:
        prefix = "OK" if status.exportable else "SKIP"
        print(f"{prefix}\t{status.job_type}\t{status.job_id}\t{status.reason}")

    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for status in statuses:
        if not status.exportable:
            continue
        if args.limit is not None and written >= args.limit:
            break
        script = export_job_script(
            rest_db,
            ml_db,
            status.job_type,
            status.job_id,
            api_url=args.api_url,
            auth_env_var=args.auth_env_var,
            bp_output_root=args.bp_output_root,
            bp_output_run_id_env=args.bp_output_run_id_env,
            source_exists_fn=source_exists_fn,
        )
        output_path = output_dir / f"{status.job_id}.py"
        # Replay scripts are local artifacts and use environment variables for credentials.
        # codeql[py/clear-text-storage-sensitive-data]
        output_path.write_text(script)
        print(f"WROTE\t{status.job_type}\t{status.job_id}\t{output_path}")
        written += 1
    print(f"Exported {written} script(s) to {output_dir}")


def resolve_state_db_root(args: argparse.Namespace) -> str:
    if args.state_db_root:
        return args.state_db_root.rstrip("/")
    return (f"{args.dataplane.rstrip('/')}/{args.customer}/prod/db")


async def load_metadata(
    state_db_root: str,
    project_id: ProjectID,
) -> tuple[RestMetadataSnapshot, MLJobSnapshot]:
    state_db = await open_state_db_from_deploy_config(
        deploy_env(),
        state_root=state_db_root,
    )
    services = StateServices(state_db)

    pquery_definitions = await services.pqueries.list_definitions()
    graphs = await services.graphs.list_graphs(project_id)
    tables = await services.tables.list_tables(project_id)
    rest_db = RestMetadataSnapshot(
        project_id=project_id,
        pquery_definitions_map={
            (str(project_id), str(definition.fpid)): definition
            for definition in pquery_definitions
        },
        graphs_v2_map={
            (str(project_id), str(graph.id)): graph
            for graph in graphs
        },
        tables_v2_map={
            (str(project_id), str(table.id)): table
            for table in tables
        },
    )

    training_ids = [
        TrainingJobID(str(job_id)) for job_id in await
        services.ml_jobs.list_job_ids_by_type(JobType.TRAINING_JOB)
    ]
    batch_prediction_ids = [
        BatchPredictionJobID(str(job_id)) for job_id in await
        services.ml_jobs.list_job_ids_by_type(JobType.BATCH_PREDICTION_JOB)
    ]
    generate_train_table_ids = [
        GenerateTrainTableJobID(str(job_id)) for job_id in await
        services.ml_jobs.list_job_ids_by_type(JobType.GENERATE_TRAIN_TABLE_JOB)
    ]
    training_jobs = [
        job for job in await services.ml_jobs.get_training_jobs(training_ids)
        if job.project_id == project_id
    ]
    batch_prediction_jobs = await services.ml_jobs.get_batch_prediction_jobs(
        batch_prediction_ids, project_ids={project_id})
    generate_train_table_jobs = (await
                                 services.ml_jobs.get_generate_train_table_jobs(
                                     generate_train_table_ids,
                                     project_ids={project_id},
                                 ))

    ml_db = MLJobSnapshot(
        training_job_map={str(job.id): job
                          for job in training_jobs},
        batch_prediction_jobs_map={
            str(job.id): job
            for job in batch_prediction_jobs
        },
        generate_train_table_jobs_map={
            str(job.id): job
            for job in generate_train_table_jobs
        },
    )
    return rest_db, ml_db


def iter_statuses(
    rest_db: RestMetadataSnapshot,
    ml_db: MLJobSnapshot,
    job_type: str,
    job_ids: Sequence[str],
    *,
    source_exists_fn: Optional[SourceExistsFn] = fsutil.exists,
) -> Iterable[ExportStatus]:
    selected = set(job_ids)
    if job_type in ("all", "training"):
        yield from dedupe_statuses(
            ml_db.training_job_map,
            selected,
            "training",
            lambda job: training_skip_reason(rest_db, ml_db, job,
                                             source_exists_fn=source_exists_fn),
            lambda job: training_dataset_pquery_key(
                rest_db, job, source_exists_fn=source_exists_fn),
        )

    if job_type in ("all", "batch_prediction"):
        yield from dedupe_statuses(
            ml_db.batch_prediction_jobs_map,
            selected,
            "batch_prediction",
            lambda job: batch_prediction_skip_reason(
                rest_db, ml_db, job, source_exists_fn=source_exists_fn),
            lambda job: batch_prediction_dataset_pquery_key(
                rest_db, ml_db, job, source_exists_fn=source_exists_fn),
        )


def dedupe_statuses(
    jobs: Mapping[str, Any],
    selected: set[str],
    job_type: str,
    skip_reason_fn: Callable[[Any], Optional[str]],
    group_key_fn: Callable[[Any], Any],
) -> Iterable[ExportStatus]:
    statuses: dict[str, ExportStatus] = {}
    group_keys: dict[str, Any] = {}
    keep_by_group: dict[Any, str] = {}
    jobs_by_id = {str(job_id): job for job_id, job in jobs.items()}

    for job_id, job in sorted(jobs_by_id.items()):
        if selected and job_id not in selected:
            continue
        reason = skip_reason_fn(job)
        if reason is not None:
            statuses[job_id] = ExportStatus(job_type, job_id, False, reason)
            continue
        group_key = group_key_fn(job)
        group_keys[job_id] = group_key
        statuses[job_id] = ExportStatus(job_type, job_id, True, "exportable")
        kept_job_id = keep_by_group.get(group_key)
        if kept_job_id is None or job_sort_value(job) > job_sort_value(
                jobs_by_id[kept_job_id]):
            keep_by_group[group_key] = job_id

    for job_id, status in sorted(statuses.items()):
        if not status.exportable:
            yield status
            continue
        kept_job_id = keep_by_group[group_keys[job_id]]
        if job_id == kept_job_id:
            yield status
        else:
            yield ExportStatus(
                job_type,
                job_id,
                False,
                f"older successful duplicate of {kept_job_id} for dataset/pquery",
            )


def training_dataset_pquery_key(
    rest_db: RestMetadataSnapshot,
    job: Any,
    *,
    source_exists_fn: Optional[SourceExistsFn] = fsutil.exists,
) -> tuple[Any, str]:
    assert job.pquery_def_fpid is not None
    graph = build_graph_bundle(rest_db, str(job.pquery_def_fpid),
                               source_exists_fn=source_exists_fn)
    return graph_dataset_key(graph), graph.query


def batch_prediction_dataset_pquery_key(
    rest_db: RestMetadataSnapshot,
    ml_db: MLJobSnapshot,
    job: Any,
    *,
    source_exists_fn: Optional[SourceExistsFn] = fsutil.exists,
) -> tuple[Any, ...]:
    assert job.config is not None
    training_job = ml_db.training_job_map[str(job.config.training_job_id)]
    pquery_def_fpid = job.config.pquery_def_fpid or training_job.pquery_def_fpid
    assert pquery_def_fpid is not None
    graph = build_graph_bundle(rest_db, str(pquery_def_fpid),
                               source_exists_fn=source_exists_fn)
    # Two BP runs against the same dataset+pquery may still differ in
    # prediction parameters (anchor time, filters, forecast horizon, output
    # destinations). Include those so distinct replays are not deduped away.
    config = job.config
    prediction_time = config.prediction_time
    prediction_time_key = (prediction_time.isoformat()
                           if prediction_time is not None else None)
    filter_overrides = config.filter_override_args
    filter_overrides_key = (tuple(sorted(
        to_primitive(filter_overrides).items())) if filter_overrides else None)
    output_paths = tuple(
        sorted((str(output.artifact_type.value), str(output.file_path))
               for output in config.outputs))
    return (
        graph_dataset_key(graph),
        graph.query,
        config.training_job_id,
        prediction_time_key,
        config.forecast_length,
        config.lag_timesteps,
        bool(config.year_over_year),
        filter_overrides_key,
        output_paths,
    )


def graph_dataset_key(graph: GraphBundle) -> tuple[tuple[str, str, str], ...]:
    return tuple((ref.graph_name, ref.connector_root_dir, ref.source_table_name)
                 for ref in graph.table_refs)


def job_sort_value(job: Any) -> str:
    execution = getattr(job, "execution", None)
    for value in (
            getattr(execution, "last_updated_at", None),
            getattr(execution, "start_time", None),
            getattr(getattr(job, "timestamps", None), "updated_at", None),
            getattr(getattr(job, "timestamps", None), "created_at", None),
            getattr(job, "id", ""),
    ):
        if value is not None:
            return str(value)
    return ""


def job_status(job: Any) -> Any:
    return getattr(getattr(job, "execution", None), "status", None)


def is_successful_job(job: Any) -> bool:
    status = job_status(job)
    return status == JobStatus.DONE or str(status) == JobStatus.DONE.value


def unsupported_forecasting_reason(query: str, plan: Any,
                                   job_type: str) -> Optional[str]:
    reasons: list[str] = []
    if re.search(r"\bFORECAST\b", query, flags=re.IGNORECASE):
        reasons.append("explicit FORECAST query")

    forecast_length = plan_field(plan, "forecast_length")
    if int_or_none(forecast_length) is not None and int_or_none(
            forecast_length) > 1:
        reasons.append(f"forecast_length={forecast_length}")

    if bool_or_false(plan_field(plan, "year_over_year")):
        reasons.append("year_over_year=True")

    if reasons:
        return f"unsupported forecasting-style {job_type}: " + ", ".join(
            reasons)

    lag_timesteps = plan_field(plan, "lag_timesteps")
    if int_or_none(lag_timesteps) not in (None, 0):
        return f"unsupported lag-timesteps {job_type}: lag_timesteps={lag_timesteps}"
    return None


def plan_field(plan: Any, name: str) -> Any:
    if isinstance(plan, Mapping):
        return plan.get(name)
    return getattr(plan, name, None)


def int_or_none(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def bool_or_false(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "t", "yes", "y")
    return bool(value)


def training_skip_reason(
    rest_db: RestMetadataSnapshot,
    ml_db: MLJobSnapshot,
    job: Any,
    *,
    source_exists_fn: Optional[SourceExistsFn] = fsutil.exists,
) -> Optional[str]:
    if not is_successful_job(job):
        return f"job status is {job_status(job)}"
    if job.advanced_options is None:
        return "missing advanced_options"
    if not job.pquery_def_fpid:
        return "missing pquery_def_fpid"
    try:
        graph = build_graph_bundle(rest_db, str(job.pquery_def_fpid),
                                   source_exists_fn=source_exists_fn)
        train_plan = get_training_table_plan(ml_db, job)
        reason = unsupported_forecasting_reason(graph.query, train_plan,
                                                "training job")
        if reason is not None:
            return reason
        get_model_plan(job)
    except ExportError as exc:
        return str(exc)
    return None


def batch_prediction_skip_reason(
    rest_db: RestMetadataSnapshot,
    ml_db: MLJobSnapshot,
    job: Any,
    *,
    source_exists_fn: Optional[SourceExistsFn] = fsutil.exists,
) -> Optional[str]:
    if not is_successful_job(job):
        return f"job status is {job_status(job)}"
    if job.config is None:
        return "missing config"
    if not job.config.outputs:
        return "missing outputs"
    if any(output.storage_type != PredictionStorageType.S3
           for output in job.config.outputs):
        return "non-S3 output"
    training_job = ml_db.training_job_map.get(str(job.config.training_job_id))
    if training_job is None:
        return f"missing training job {job.config.training_job_id}"
    reason = training_skip_reason(rest_db, ml_db, training_job,
                                  source_exists_fn=source_exists_fn)
    if reason:
        return f"training job cannot be exported: {reason}"
    training_pquery_def_fpid = training_job.pquery_def_fpid
    pquery_def_fpid = job.config.pquery_def_fpid or training_pquery_def_fpid
    if pquery_def_fpid:
        try:
            graph = build_graph_bundle(rest_db, str(pquery_def_fpid),
                                       source_exists_fn=source_exists_fn)
            if (job.config.pquery_def_fpid and training_pquery_def_fpid
                    and job.config.pquery_def_fpid != training_pquery_def_fpid):
                training_graph = build_graph_bundle(
                    rest_db, str(training_pquery_def_fpid),
                    source_exists_fn=source_exists_fn)
                if graph.graph_fpid != training_graph.graph_fpid:
                    return (
                        "alternate batch prediction pquery graph is not supported: "
                        f"training_graph={training_graph.graph_fpid}, "
                        f"prediction_graph={graph.graph_fpid}")
            reason = unsupported_forecasting_reason(graph.query, job.config,
                                                    "batch prediction")
            if reason is not None:
                return reason
        except ExportError as exc:
            return str(exc)
    return None


def export_job_script(
    rest_db: RestMetadataSnapshot,
    ml_db: MLJobSnapshot,
    job_type: str,
    job_id: str,
    *,
    api_url: str,
    auth_env_var: str,
    bp_output_root: str,
    bp_output_run_id_env: str = DEFAULT_BP_OUTPUT_RUN_ID_ENV,
    source_exists_fn: Optional[SourceExistsFn] = fsutil.exists,
) -> str:
    if job_type == "training":
        return render_training_script(rest_db, ml_db,
                                      ml_db.training_job_map[str(job_id)],
                                      api_url, auth_env_var,
                                      source_exists_fn=source_exists_fn)
    if job_type == "batch_prediction":
        return render_batch_prediction_script(
            rest_db, ml_db, ml_db.batch_prediction_jobs_map[str(job_id)],
            api_url, auth_env_var, bp_output_root,
            bp_output_run_id_env=bp_output_run_id_env,
            source_exists_fn=source_exists_fn)
    raise ValueError(f"Unsupported job type: {job_type}")


def render_training_script(
    rest_db: RestMetadataSnapshot,
    ml_db: MLJobSnapshot,
    job: Any,
    api_url: str,
    auth_env_var: str,
    *,
    source_exists_fn: Optional[SourceExistsFn] = fsutil.exists,
) -> str:
    assert job.pquery_def_fpid is not None
    graph = build_graph_bundle(rest_db, str(job.pquery_def_fpid),
                               source_exists_fn=source_exists_fn)
    train_plan = compact_training_table_plan(get_training_table_plan(
        ml_db, job))
    model_plan = compact_model_plan(get_model_plan(job), graph)
    existing_train_table_job_id = getattr(job, "generate_train_table_job_id",
                                          None) or None
    return (render_header(api_url, auth_env_var) + render_graph_setup(graph) +
            "\n"
            "pquery = kumo.PredictiveQuery(\n"
            "    graph=graph,\n"
            f"    query={quoted(graph.query)},\n"
            ")\n\n" + render_train_table(
                train_plan,
                existing_train_table_job_id=existing_train_table_job_id) +
            f"model_plan = {format_value(model_plan)}\n\n"
            "trainer = kumo.Trainer(model_plan)\n"
            "training_job = trainer.fit(\n"
            "    graph=graph,\n"
            "    train_table=train_table,\n"
            "    non_blocking=False,\n"
            ")\n" + render_tune_metric_check(job))


def render_batch_prediction_script(
    rest_db: RestMetadataSnapshot,
    ml_db: MLJobSnapshot,
    job: Any,
    api_url: str,
    auth_env_var: str,
    bp_output_root: str,
    *,
    bp_output_run_id_env: str = DEFAULT_BP_OUTPUT_RUN_ID_ENV,
    source_exists_fn: Optional[SourceExistsFn] = fsutil.exists,
) -> str:
    assert job.config is not None
    training_job = ml_db.training_job_map[str(job.config.training_job_id)]
    script = render_training_script(rest_db, ml_db, training_job, api_url,
                                    auth_env_var,
                                    source_exists_fn=source_exists_fn)
    query_var = "pquery"
    if job.config.pquery_def_fpid and job.config.pquery_def_fpid != training_job.pquery_def_fpid:
        prediction_graph = build_graph_bundle(rest_db,
                                              str(job.config.pquery_def_fpid),
                                              source_exists_fn=source_exists_fn)
        training_graph = build_graph_bundle(rest_db,
                                            str(training_job.pquery_def_fpid),
                                            source_exists_fn=source_exists_fn)
        if prediction_graph.graph_fpid != training_graph.graph_fpid:
            raise ExportError(
                "alternate batch prediction pquery graph is not supported: "
                f"training_graph={training_graph.graph_fpid}, "
                f"prediction_graph={prediction_graph.graph_fpid}")
        query_var = "prediction_pquery"
        script += ("\n"
                   "prediction_pquery = kumo.PredictiveQuery(\n"
                   "    graph=graph,\n"
                   f"    query={quoted(prediction_graph.query)},\n"
                   ")\n")

    output_root, output_config = compact_output_config(job.config,
                                                       bp_output_root, job.id)
    script += "\n" + render_prediction_table(
        query_var, compact_prediction_table_plan(job.config))
    script += render_prediction_output_connector(
        output_root, bp_output_run_id_env
    ) + (f"output_config = OutputConfig(\n"
         f"    **{format_value(output_config)},\n"
         "    output_connector=prediction_export_connector,\n"
         ")\n\n"
         "prediction_job = trainer.predict(\n"
         "    graph=graph,\n"
         "    prediction_table=prediction_table,\n"
         "    output_config=output_config,\n"
         "    training_job_id=training_job.job_id,\n" + render_call_kwargs(
             compact_predict_kwargs(job.config)) + "    non_blocking=False,\n"
         ")\n")
    return script


def render_prediction_output_connector(output_root: str,
                                       run_id_env: str) -> str:
    return (
        f"prediction_output_root = {output_root!r}\n"
        f"prediction_output_run_id = os.environ.get({run_id_env!r}) or uuid.uuid4().hex\n"
        "prediction_export_connector = kumo.S3Connector(\n"
        "    root_dir=f'{prediction_output_root.rstrip(\"/\")}/{prediction_output_run_id}',\n"
        ")\n\n")


def final_tune_metric(job: Any) -> Optional[tuple[str, float]]:
    tune_metric = get_model_plan(job).training_job.tune_metric
    if not tune_metric:
        return None
    results = getattr(job.execution, "results", None)
    test_metrics = getattr(results, "test_metrics", None)
    if not test_metrics:
        return None
    for metric in test_metrics:
        if getattr(metric, "name", None) == tune_metric:
            value = getattr(metric, "val", None)
            if value is not None:
                return tune_metric, float(value)
    return None


def render_tune_metric_check(job: Any) -> str:
    expected = final_tune_metric(job)
    if expected is None:
        return ""
    metric_name, metric_value = expected
    lines = [
        "",
        f"EXPECTED_TUNE_METRIC_NAME = {metric_name!r}",
        f"EXPECTED_TUNE_METRIC_VALUE = {metric_value!r}",
        'TUNE_METRIC_REL_TOL = float(os.environ.get("KUMO_TUNE_METRIC_REL_TOL", "0.05"))',
        "PARTIAL_FAILURES = []",
        "",
        "def _replay_field(obj, *names):",
        "    for name in names:",
        "        if isinstance(obj, dict) and name in obj:",
        "            return obj.get(name)",
        "        value = getattr(obj, name, None)",
        "        if value is not None:",
        "            return value",
        "    return None",
        "",
        "def _replay_metrics_from_result(result):",
        "    if result is None:",
        "        return None",
        "    for candidate in (",
        "        result,",
        "        _replay_field(result, \"result\", \"results\"),",
        "    ):",
        "        if candidate is None:",
        "            continue",
        "        metrics_fn = getattr(candidate, \"metrics\", None)",
        "        if callable(metrics_fn):",
        "            metrics = metrics_fn()",
        "            if metrics is not None:",
        "                return metrics",
        "        metrics = _replay_field(candidate, \"eval_metrics\", \"metrics\")",
        "        if metrics is not None:",
        "            return metrics",
        "        if _replay_field(candidate, \"test_metrics\") is not None:",
        "            return candidate",
        "    return None",
        "",
        "replay_training_result = training_job",
        "replay_metrics = _replay_metrics_from_result(replay_training_result)",
        "if replay_metrics is None and hasattr(replay_training_result, \"wait\"):",
        "    replay_training_result = replay_training_result.wait()",
        "    replay_metrics = _replay_metrics_from_result(replay_training_result)",
        "if replay_metrics is None and hasattr(replay_training_result, \"refresh\"):",
        "    replay_training_result = replay_training_result.refresh()",
        "    replay_metrics = _replay_metrics_from_result(replay_training_result)",
        "",
        "replay_test_metrics = _replay_field(replay_metrics, \"test_metrics\")",
        "actual_test_tune_metric_value = None",
        "if replay_test_metrics is not None:",
        "    for metric in replay_test_metrics:",
        "        metric_name = _replay_field(metric, \"name\")",
        "        metric_value = _replay_field(metric, \"value\", \"val\")",
        "        if metric_name == EXPECTED_TUNE_METRIC_NAME and metric_value is not None:",
        "            actual_test_tune_metric_value = float(metric_value)",
        "            break",
        "",
        "if actual_test_tune_metric_value is None:",
        "    PARTIAL_FAILURES.append(",
        '        f"missing replay test tune metric {EXPECTED_TUNE_METRIC_NAME!r}")',
        "else:",
        "    tune_metric_abs_delta = abs(actual_test_tune_metric_value - EXPECTED_TUNE_METRIC_VALUE)",
        "    tune_metric_rel_delta = tune_metric_abs_delta / max(abs(EXPECTED_TUNE_METRIC_VALUE), 1e-12)",
        "    if tune_metric_rel_delta > TUNE_METRIC_REL_TOL:",
        "        PARTIAL_FAILURES.append(",
        '            f"test tune metric {EXPECTED_TUNE_METRIC_NAME} diverged: "',
        '            f"expected={EXPECTED_TUNE_METRIC_VALUE}, "',
        '            f"actual={actual_test_tune_metric_value}, "',
        '            f"rel_delta={tune_metric_rel_delta:.6g}, "',
        '            f"tolerance={TUNE_METRIC_REL_TOL}")',
        "",
        "PARTIAL_FAILURE = bool(PARTIAL_FAILURES)",
        "if PARTIAL_FAILURE:",
        '    print("PARTIAL_FAILURE: " + "; ".join(PARTIAL_FAILURES))',
        "else:",
        "    print(",
        '        f"Test tune metric {EXPECTED_TUNE_METRIC_NAME} matched: "',
        '        f"expected={EXPECTED_TUNE_METRIC_VALUE}, actual={actual_test_tune_metric_value}")',
        "",
    ]
    return "\n".join(lines)


def validate_source_table_ref(
    graph_name: str,
    source: S3SourceTable,
    source_exists_fn: Optional[SourceExistsFn],
) -> None:
    connector_id = str(source.connector_id or "")
    source_uri = s3_source_uri(source)
    if source_uri is None:
        if connector_id:
            raise ExportError(
                f"file-upload source table {graph_name} uses {connector_id}; "
                "uploaded tables are not replayable as plain S3Connector "
                "tables")
        raise ExportError(f"S3 table {graph_name} is missing s3_path")
    if source_exists_fn is None:
        return
    try:
        exists = source_exists_fn(source_uri)
    except Exception as exc:
        raise ExportError(f"could not validate source table {graph_name} at "
                          f"{source_uri}: {exc}") from exc
    if not exists:
        raise ExportError(f"missing source table {graph_name}: {source.table} "
                          f"at {source_uri}")


def build_graph_bundle(
    rest_db: RestMetadataSnapshot,
    pquery_def_fpid: str,
    *,
    source_exists_fn: Optional[SourceExistsFn] = fsutil.exists,
) -> GraphBundle:
    project_id = str(rest_db.project_id)
    pquery_def = map_get(rest_db.pquery_definitions_map,
                         (project_id, str(pquery_def_fpid)))
    if pquery_def is None:
        raise ExportError(f"missing pquery definition {pquery_def_fpid}")
    graph = map_get(rest_db.graphs_v2_map,
                    (project_id, str(pquery_def.graph_fpid)))
    if graph is None:
        raise ExportError(f"missing graph {pquery_def.graph_fpid}")

    table_refs: list[TableRef] = []
    table_by_graph_name: dict[str, TableRef] = {}
    root_to_connector_var: dict[str, str] = {}
    used_connector_vars: set[str] = set()

    for graph_name, table_fpid in sorted(graph.table_name_to_dbid.items()):
        table_entry = map_get(rest_db.tables_v2_map,
                              (project_id, str(table_fpid)))
        if table_entry is None:
            raise ExportError(f"missing table {table_fpid}")
        table = table_entry.config
        source = table.source_table
        if not isinstance(source, S3SourceTable):
            source_type = getattr(source, "data_source_type", "UNKNOWN")
            raise ExportError(f"non-S3 graph table {graph_name}: {source_type}")
        validate_source_table_ref(str(graph_name), source, source_exists_fn)

        # Full-URI source tables replay through one rootless connector.  A
        # rooted connector would reject the full URI passed to ``table``.
        root_dir = ""
        if root_dir not in root_to_connector_var:
            connector_var = unique_var("s3_connector", used_connector_vars)
            root_to_connector_var[root_dir] = connector_var
            used_connector_vars.add(connector_var)

        ref = TableRef(
            graph_name=str(graph_name),
            var_name=safe_var(str(graph_name)),
            source_table_name=str(source.table_name),
            connector_var=root_to_connector_var[root_dir],
            connector_root_dir=root_dir,
            table=table,
        )
        table_refs.append(ref)
        table_by_graph_name[str(graph_name)] = ref

    edges: list[graph_api.Edge] = []
    for col_group in graph.col_groups:
        pkey_col = None
        for col in col_group.columns:
            table_ref = table_by_graph_name[str(col.table_name)]
            if str(col.col_name) == str(table_ref.table.pkey):
                pkey_col = col
                break
        if pkey_col is None:
            raise ExportError(
                "graph column group does not contain a primary key")
        for col in col_group.columns:
            if col == pkey_col:
                continue
            src_ref = table_by_graph_name[str(col.table_name)]
            dst_ref = table_by_graph_name[str(pkey_col.table_name)]
            edges.append(
                graph_api.Edge(src_ref.var_name, str(col.col_name),
                               dst_ref.var_name))

    query_refs = [
        QueryTableRef(source_table_name=ref.var_name,
                      table_alias_in_graph=ref.graph_name) for ref in table_refs
    ]
    query = clean_up_pquery_string(pquery_def.query_str, query_refs)
    query = align_query_column_case(query, table_refs)
    return GraphBundle(str(pquery_def.pquery_name), query,
                       str(pquery_def.graph_fpid), table_refs, edges)


def align_query_column_case(query: str, table_refs: Sequence[TableRef]) -> str:
    if not query:
        return query

    table_cols_by_var: dict[str, dict[str, str]] = {}
    for ref in table_refs:
        col_map: dict[str, str] = {}
        for col in ref.table.cols:
            col_map[str(col.name).lower()] = str(col.name)
        table_cols_by_var[ref.var_name] = col_map

    pattern = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")

    unresolved: list[str] = []

    def _replace(match: re.Match) -> str:
        table_token, col_token = match.group(1), match.group(2)
        col_map = table_cols_by_var.get(table_token)
        if col_map is None:
            return match.group(0)
        emitted = col_map.get(col_token.lower())
        if emitted is None:
            unresolved.append(f"{table_token}.{col_token}")
            return match.group(0)
        if emitted == col_token:
            return match.group(0)
        return f"{table_token}.{emitted}"

    rewritten = pattern.sub(_replace, query)

    if unresolved:
        seen: set[str] = set()
        unique_unresolved = []
        for ref_str in unresolved:
            if ref_str not in seen:
                seen.add(ref_str)
                unique_unresolved.append(ref_str)
        raise ExportError(
            "pquery references columns not present in the emitted graph: " +
            ", ".join(unique_unresolved))

    return rewritten


def render_header(api_url: str, auth_env_var: str) -> str:
    return ("import os\n"
            "import uuid\n\n"
            "import kumoai_studio as kumo\n"
            "from kumoai_studio.artifact_export import OutputConfig\n"
            "from kumoai_studio.graph import Column\n"
            "from kumoapi.typing import Dtype, Stype\n\n"
            f"API_URL = os.environ.get(\"KUMO_API_URL\", {api_url!r})\n"
            f"API_KEY = os.environ[\"{auth_env_var}\"]\n"
            "kumo.init(url=API_URL, api_key=API_KEY)\n")


def render_graph_setup(graph: GraphBundle) -> str:
    lines: list[str] = [""]
    emitted_connectors: set[str] = set()
    for table_ref in graph.table_refs:
        if table_ref.connector_var in emitted_connectors:
            continue
        lines.append(f"{table_ref.connector_var} = kumo.S3Connector(")
        if table_ref.connector_root_dir:
            lines.append(f"    root_dir={table_ref.connector_root_dir!r},")
        lines.extend([")", ""])
        emitted_connectors.add(table_ref.connector_var)

    for table_ref in graph.table_refs:
        lines.extend(render_table(table_ref))

    lines.extend(["graph = kumo.Graph(", "    tables={"])
    for table_ref in graph.table_refs:
        lines.append(f"        {table_ref.var_name!r}: {table_ref.var_name},")
    lines.extend(["    },", "    edges=["])
    for edge in graph.edges:
        lines.append("        " + format_value({
            "src_table": edge.src_table,
            "fkey": edge.fkey,
            "dst_table": edge.dst_table
        }) + ",")
    lines.extend(["    ],", ")", "graph.validate()"])
    return "\n".join(lines) + "\n"


def render_table(table_ref: TableRef) -> list[str]:
    table = table_ref.table
    lines = [
        f"{table_ref.var_name} = kumo.Table(",
        f"    source_table={table_ref.connector_var}.table({table_ref.source_table_name!r}),",
        f"    columns={render_columns(table.cols)},",
    ]
    if table.pkey is not None:
        lines.append(f"    primary_key={str(table.pkey)!r},")
    if table.time_col is not None:
        lines.append(f"    time_column={str(table.time_col)!r},")
    if table.end_time_col is not None:
        lines.append(f"    end_time_column={str(table.end_time_col)!r},")
    lines.extend([")", ""])
    return lines


def render_columns(cols: Sequence[Any]) -> str:
    if not cols:
        return "[]"
    rendered = []
    for col in cols:
        args = [
            f"name={str(col.name)!r}", f"stype=Stype.{enum_name(col.stype)}",
            f"dtype=Dtype.{enum_name(col.dtype)}"
        ]
        if col.timestamp_format is not None:
            args.append(f"timestamp_format={str(col.timestamp_format)!r}")
        rendered.append(f"Column({', '.join(args)})")
    return "[\n        " + ",\n        ".join(rendered) + ",\n    ]"


def render_train_table(
    plan: Mapping[str, Any],
    *,
    existing_train_table_job_id: Optional[str] = None,
) -> str:
    if existing_train_table_job_id:
        return (
            f"train_table = kumo.TrainingTableJob({existing_train_table_job_id!r})\n\n"
        )
    if not plan:
        return "train_table = pquery.generate_training_table()\n\n"
    return f"train_table_plan = {format_value(dict(plan))}\n" "train_table = pquery.generate_training_table(plan=train_table_plan)\n\n"


def render_prediction_table(query_var: str, plan: Mapping[str, Any]) -> str:
    if not plan:
        return f"prediction_table = {query_var}.generate_prediction_table()\n\n"
    return f"prediction_table_plan = {format_value(dict(plan))}\n" f"prediction_table = {query_var}.generate_prediction_table(plan=prediction_table_plan)\n\n"


def render_call_kwargs(kwargs: Mapping[str, Any]) -> str:
    return "".join(f"    {key}={format_value(value)},\n"
                   for key, value in kwargs.items())


def get_model_plan(job: Any) -> ModelPlan:
    advanced_options = job.advanced_options
    if isinstance(advanced_options, ModelPlan):
        return advanced_options
    if isinstance(advanced_options,
                  rest_api_db_entries.DeprecatedAdvancedAutoTrainerOptions):
        return migrate_deprecated_advanced_options(advanced_options)
    raise ExportError(
        f"unsupported advanced_options type {type(advanced_options).__name__}")


def get_training_table_plan(ml_db: MLJobSnapshot,
                            job: Any) -> TrainingTableGenerationPlan:
    if job.generate_train_table_job_id:
        train_table_job = ml_db.generate_train_table_jobs_map.get(
            str(job.generate_train_table_job_id))
        if train_table_job is None:
            raise ExportError(
                f"missing training table job {job.generate_train_table_job_id}")
        return train_table_job.config.plan

    model_plan = get_model_plan(job)
    if model_plan.training_table_generation is not None:
        return model_plan.training_table_generation
    raise ExportError("missing training table generation plan")


def migrate_deprecated_advanced_options(
    legacy: rest_api_db_entries.DeprecatedAdvancedAutoTrainerOptions
) -> ModelPlan:
    plan = ModelPlan()
    sections = {
        "training_table_generation": {
            "split": "split",
            "timeframe_step": "timeframe_step",
            "forecast_length": "forecast_length",
            "lag_timesteps": "lag_timesteps",
            "train_start_offset": "train_start_offset",
            "train_end_offset": "train_end_offset",
            "entity_candidate": "entity_candidate",
            "entity_candidate_aggregation": "entity_candidate_aggregation",
            "task_path_override": "task_path_override",
            "train_table_path_override": "train_table_path_override"
        },
        "training_job": {
            "num_experiments": "num_experiments",
            "tune_metric": "tune_metric",
            "disable_explain": "disable_explain",
            "refit_trainval": "refit_trainval",
            "refit_full": "refit_full"
        },
        "neighbor_sampling": {
            "num_neighbors": "num_neighbors",
            "max_target_neighbors_per_entity": "max_target_neighbors_per_entity"
        },
        "optimization": {
            "max_epochs": "max_epochs",
            "max_steps_per_epoch": "max_steps_per_epoch",
            "max_val_steps": "max_val_steps",
            "max_test_steps": "max_test_steps",
            "base_lr": "base_lr",
            "batch_size": "batch_size",
            "weight_decay": "weight_decay",
            "lr_scheduler": "lr_scheduler",
            "early_stopping": "early_stopping",
            "majority_sampling_ratio": "majority_sampling_ratio",
            "weight_mode": "weight_mode"
        },
        "model_architecture": {
            "channels": "channels",
            "num_pre_mp_layers": "num_pre_message_passing_layers",
            "num_post_mp_layers": "num_post_message_passing_layers",
            "aggr": "aggregation",
            "activation": "activation",
            "module": "module",
            "handle_new_target_entities": "handle_new_target_entities",
            "lp_target_embedding_mode": "target_embedding_mode",
            "lp_embedding_dim": "output_embedding_dim",
            "lp_ranking_emb_loss_coeff": "ranking_embedding_loss_coeff",
            "use_seq_id": "use_seq_id"
        },
        "column_processing": {
            "encoder_overrides": "encoder_overrides"
        },
    }
    for section_name, mapping in sections.items():
        section = getattr(plan, section_name)
        for old_name, new_name in mapping.items():
            value = getattr(legacy, old_name)
            if value not in (None, "", [], {}):
                setattr(section, new_name, value)
    metrics = list(legacy.extra_metrics)
    if legacy.tune_metric and legacy.tune_metric not in metrics:
        metrics.insert(0, legacy.tune_metric)
    if metrics:
        plan.training_job.metrics = metrics
    return plan


def compact_model_plan(model_plan: ModelPlan,
                       graph: GraphBundle) -> dict[str, Any]:
    actual = to_primitive(model_plan)
    default = to_primitive(ModelPlan())
    actual.pop("training_table_generation", None)
    default.pop("training_table_generation", None)
    compact = diff_dict(actual, default)
    return remap_model_plan_table_refs(compact, graph)


def remap_model_plan_table_refs(
    model_plan: dict[str, Any],
    graph: GraphBundle,
) -> dict[str, Any]:
    aliases = graph_table_aliases(graph)
    return rewrite_model_plan_value(model_plan, aliases)


def graph_table_aliases(graph: GraphBundle) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for ref in graph.table_refs:
        for candidate in (ref.graph_name, ref.var_name):
            aliases.setdefault(candidate.lower(), ref.var_name)
            aliases.setdefault(normalize_table_token(candidate), ref.var_name)
    for ref in graph.table_refs:
        aliases.setdefault(str(ref.source_table_name).lower(), ref.var_name)
        aliases.setdefault(normalize_table_token(ref.source_table_name),
                           ref.var_name)
    return aliases


def normalize_table_token(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(value)).lower()


def rewrite_model_plan_value(value: Any, aliases: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            rewrite_model_plan_key(str(k), aliases):
            rewrite_model_plan_value(v, aliases)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [rewrite_model_plan_value(item, aliases) for item in value]
    return value


def rewrite_model_plan_key(key: str, aliases: Mapping[str, str]) -> str:
    if "->" not in key:
        return rewrite_endpoint_key(key, aliases)
    return "->".join(
        rewrite_endpoint_key(part, aliases) for part in key.split("->"))


def rewrite_endpoint_key(value: str, aliases: Mapping[str, str]) -> str:
    table, dot, rest = value.partition(".")
    if not dot:
        return value
    mapped = aliases.get(table.lower()) or aliases.get(
        normalize_table_token(table))
    if mapped is None:
        return value
    return f"{mapped}.{rest}"


def compact_training_table_plan(
        plan: TrainingTableGenerationPlan) -> dict[str, Any]:
    compact = diff_dict(to_primitive(plan),
                        to_primitive(TrainingTableGenerationPlan()))
    if compact.get("forecast_length") == 1:
        compact.pop("forecast_length")
    return compact


def compact_prediction_table_plan(
        config: BatchPredictionJobConfig) -> dict[str, Any]:
    plan: dict[str, Any] = {}
    if config.prediction_time is not None:
        plan["anchor_time"] = config.prediction_time.isoformat()
    if config.forecast_length is not None:
        plan["forecast_length"] = config.forecast_length
    if config.lag_timesteps is not None:
        plan["lag_timesteps"] = config.lag_timesteps
    if config.year_over_year:
        plan["year_over_year"] = True
    if config.filter_override_args:
        filters = {
            k: v
            for k, v in to_primitive(config.filter_override_args).items()
            if v not in ("", None)
        }
        if filters:
            plan["filter_overrides"] = filters
    return plan


def compact_predict_kwargs(config: BatchPredictionJobConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.binary_classification_threshold is not None:
        kwargs[
            "binary_classification_threshold"] = config.binary_classification_threshold
    if config.num_classes_to_return is not None:
        kwargs["num_classes_to_return"] = config.num_classes_to_return
    if config.num_parallel_workers_limit != 1:
        kwargs["num_workers"] = config.num_parallel_workers_limit
    return kwargs


def compact_output_config(
    config: BatchPredictionJobConfig,
    bp_output_root: str,
    job_id: str,
) -> tuple[str, dict[str, Any]]:
    output_types: list[str] = []
    output_names: list[str] = []
    extra_fields: list[str] = []
    seen_extra_fields: set[str] = set()
    for output in config.outputs:
        output_types.append(str(output.artifact_type.value))
        _, name = str(output.file_path).rstrip("/").rsplit("/", 1)
        output_names.append(name.rsplit(".", 1)[0])
        for field in getattr(output, "extra_fields", None) or ():
            field_str = field.value if isinstance(field, Enum) else str(field)
            if field_str not in seen_extra_fields:
                seen_extra_fields.add(field_str)
                extra_fields.append(field_str)
    if not output_names:
        raise ExportError("missing S3 output path")
    output_root = f"{bp_output_root.rstrip('/')}/{job_id}"
    config_dict: dict[str, Any] = {"output_types": sorted(set(output_types))}
    config_dict["output_table_name"] = (output_names[0] if len(output_names)
                                        == 1 else safe_var(job_id))
    if extra_fields:
        config_dict["extra_fields"] = extra_fields
    return output_root, config_dict


def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        value = recursively_convert_dataclass_to_dict(value)
    elif isinstance(value, BaseModel):
        value = value.dict()
    if isinstance(value, dict):
        return {str(k): to_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def diff_dict(actual: Mapping[str, Any],
              default: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in actual.items():
        diff = diff_value(value, default.get(key, MISSING))
        if diff is not MISSING:
            out[key] = diff
    return render_str_vals_as_strings(out)


def diff_value(actual: Any, default: Any) -> Any:
    if default is not MISSING and actual == default:
        return MISSING
    if actual in (None, [], {}, "inferred", "???"):
        return MISSING
    if isinstance(actual, list):
        filtered = [
            value for value in actual if value not in (None, "inferred", "???")
        ]
        if not filtered:
            return MISSING
        actual = filtered
    if isinstance(actual, dict) and isinstance(default, dict):
        nested = diff_dict(actual, default)
        return nested if nested else MISSING
    return actual


def format_value(value: Any) -> str:
    return pprint.pformat(value, width=88, sort_dicts=False)


def enum_name(value: Any) -> str:
    if isinstance(value, Enum):
        return value.name
    return str(value).rsplit(".", 1)[-1]


def s3_source_uri(source: S3SourceTable) -> Optional[str]:
    uri = getattr(source.source, "uri", None)
    return str(uri) if uri else None


def safe_var(name: str) -> str:
    var = re.sub(r"\W+", "_", name).strip("_").lower()
    if not var:
        var = "value"
    if var[0].isdigit():
        var = f"_{var}"
    if keyword.iskeyword(var):
        var = f"{var}_"
    return var


def unique_var(name: str, used: set[str]) -> str:
    base = safe_var(name)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def map_get(mapping: Mapping[Any, Any], key: str) -> Any:
    return mapping.get(key) or mapping.get(FPID(key))


def quoted(value: str) -> str:
    # PQuery strings may contain quotes; ``repr`` picks a safe quoting
    # style (single/double) and escapes embedded quotes as needed.
    return repr(value)


if __name__ == "__main__":
    main()
