# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import os.path as osp
import shutil
from collections import defaultdict
from dataclasses import replace
from datetime import timedelta
from typing import Dict, List, NoReturn, Optional, Tuple, Union
from uuid import uuid4

from kumo_core.job_config import PredictionConfig
from kumo_core.types import ColStatType, PredictionArtifactType
from kumo_core.types.schema import BaseColumnConfig, VersionedSourceTableID
from kumo_core.utils.time import now_utc
from kumoapi.jobs import (
    FileTrainingTableOutputConfig,
    NamedTrainingTableOutputConfig,
    PredictionOutputConfig,
    TrainingTableArtifactType,
    TrainingTableOutputConfig,
)
from temporalio.client import WorkflowExecutionStatus
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

from async_utils.instrument import measure_blocking_code
from async_utils.sync import blocking_func
from io_utils import fsutil, json, logging
from io_utils.json import fast_from_json, load_many_json, save_json
from kumo.connector import MAX_SUPPORTED_FILES
from kumo.data import get_active_cache
from kumo.data.backend_cache import (
    AsyncDataTaskStatus,
    BackendCacheManager,
    TableDataTaskResult,
    TableRefreshStage,
)
from kumo.data.cache.graph_cache import GraphKey, VersionedGraphKey
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.graph.utils import get_minimal_stats
from kumo.home import HomeDirManager
from kumo.stats import Stats
from kumo.stats.utils import col_stat_dict_complete
from kumo.validation import ValidationResponse
from monitoring.tracing import TraceOptions, trace
from rest.backends import DATA_BACKEND_REFRESH_TIMEOUT_HOURS
from rest.backends.data_backend import connector_activity_registry
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.utils import (
    get_table_data_version,
    parse_edge_stats_result,
)
from rest.backends.data_backend.workflows import (
    GraphRefreshWorkflow,
    TableRefreshWorkflow,
    TableRefreshWorkflowInput,
)
from rest.backends.ml_backend.prediction_artifacts import (
    resolve_prediction_artifact,
)
from rest.config.artifact_export import (
    ARTIFACT_EXPORT_WORKFLOW_EXECUTION_TIMEOUT,
    ArtifactExportInput,
    DatabricksManagedVolumeStage,
)
from rest.config.batch_prediction_job import (
    BatchPredictionJobInput,
    direct_databricks_prediction_output_index,
    has_only_databricks_outputs,
    requested_prediction_artifact_types,
)
from rest.config.connector import (
    _CSV_SUFFIX,
    _PARQUET_SUFFIX,
    TEST_KUMO_FILE,
    S3File,
    S3FileList,
    S3Table,
)
from rest.config.data_backend import (
    ConnectorExecutionRef,
    ConnectorOpenSpec,
    GraphDataVersionInfo,
)
from rest.config.graph import EdgeHealth
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.table_v2 import TableDefinition
from rest.constants import DATA_BACKEND_TASK_QUEUE, ML_WORKFLOW_TASK_QUEUE
from rest.settings import SAMPLE_COL_STATS_WARMUP_CONCURRENCY
from rest.utils.driver_common import (
    async_connector_lifecycle,
    get_connector_from_spec,
    get_connector_from_spec_async,
)
from statedb.config.common import (
    ArtifactExportJobID,
    GenerateTrainTableJobID,
    JobID,
    JobStatus,
    TableName,
)
from statedb.config.table import TableDataVersion
from statedb.utils.errors import ArtifactNotFoundError

log = logging.getLogger(__name__)

_SINGLE_ACTIVITY_FAILURE_MESSAGE = (
    'Data backend operation failed. Please retry or contact Kumo support.')
_SINGLE_ACTIVITY_PUBLIC_ERROR_TYPES = frozenset({
    # Generic PublicException is intentional: request-validation activities
    # use it for messages whose complete text is constructed by Kumo.
    'PublicException',
    'ValidationException',
    'ArtifactNotFoundError',
    'CustomSqlQueryError',
    'DataRefreshError',
    'DatabricksClusterError',
    'DiskgraphError',
    'GraphRefreshError',
    'InvalidGraphError',
    'ProjectForbiddenError',
    'ProjectNotAvailableError',
    'ProjectNotFoundError',
    'SourceTableNotFoundError',
    'SourceTableUnavailableError',
    'TableInvalidColumnNameError',
    'UnsupportedInDiskgraphError',
})


def _extract_allowlisted_single_activity_public_exception(
    error: BaseException, ):
    """Accept only Kumo's exact Temporal PublicException envelope."""
    return _extract_allowlisted_temporal_public_exception(
        error,
        allowed_error_types=_SINGLE_ACTIVITY_PUBLIC_ERROR_TYPES,
        allow_timeout=False,
    )


def _extract_allowlisted_temporal_public_exception(
    error: BaseException,
    *,
    allowed_error_types: frozenset[str],
    allow_timeout: bool,
    exact_error_types: Optional[Dict[str, type]] = None,
):
    """Validate a public envelope against names and exact class bindings."""
    from temporalio.exceptions import ApplicationError, TimeoutError

    from common_utils.error import PublicException
    from temporalio_commons.error_utils import extract_public_exception

    exact_error_types = exact_error_types or {}
    current: BaseException | None = error
    for _ in range(20):
        if isinstance(current, TimeoutError):
            return (extract_public_exception(current)
                    if allow_timeout else None)
        if isinstance(current, PublicException):
            error_type = type(current).__name__
            expected_type = exact_error_types.get(error_type)
            if (error_type not in allowed_error_types
                    or (expected_type is not None
                        and type(current) is not expected_type)):
                return None
            return current
        if isinstance(current, ApplicationError):
            details = tuple(getattr(current, 'details', ()))
            if len(details) != 1 or not isinstance(details[0], dict):
                return None
            payload = details[0]
            if set(payload) != {
                    'type', 'message', 'template', 'kwargs', 'http_code',
                    'non_retryable', 'related_errors'
            }:
                return None
            error_type = payload['type']
            if (not isinstance(error_type, str)
                    or error_type not in allowed_error_types
                    or current.type != error_type
                    or not isinstance(payload['message'], str)
                    or not isinstance(payload['template'], str)
                    or not isinstance(payload['kwargs'], dict)
                    or not isinstance(payload['http_code'], int)
                    or not isinstance(payload['non_retryable'], bool)
                    or payload['related_errors'] != []):
                return None
            reconstructed = extract_public_exception(current)
            expected_type = exact_error_types.get(error_type)
            if (reconstructed is None
                    or type(reconstructed).__name__ != error_type
                    or (expected_type is not None
                        and type(reconstructed) is not expected_type)
                    or reconstructed.message != payload['message']
                    or reconstructed.template != payload['template']
                    or reconstructed.http_code != payload['http_code']
                    or reconstructed.non_retryable != payload['non_retryable']):
                return None
            return reconstructed
        cause = getattr(current, 'cause', None)
        if not isinstance(cause, BaseException) or cause is current:
            break
        current = cause
    return None


def _raise_safe_single_activity_failure(
    error: BaseException,
    activity_name: str,
    workflow_id: str,
) -> NoReturn:
    """Preserve typed public failures without exposing arbitrary causes."""
    from temporalio_commons.auto_rpc_exception import RpcError
    public_error = _extract_allowlisted_single_activity_public_exception(error)
    if public_error is not None:
        raise public_error from None

    # Unknown activity failures may contain credentials in their message,
    # details, or stack trace.  Log only safe correlation metadata and expose
    # the same workflow ID alongside the constant public payload.
    log.error(
        'Data backend activity failed (workflow_id=%s activity_name=%s '
        'error_type=%s)',
        workflow_id,
        activity_name,
        type(error).__name__,
    )
    raise RpcError(
        'data_backend',
        activity_name,
        'InternalError',
        f'{_SINGLE_ACTIVITY_FAILURE_MESSAGE} Reference: {workflow_id}.',
        '',
        False,
    ) from None


@blocking_func(DATA_BACKEND_THREAD_POOL)
def _compute_connector_col_stats(
    connector,
    source_table_name: str,
    stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
    data_version: TableDataVersion,
    approx_sample_size: Optional[int] = None,
) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
    kwargs = {
        "approx": True,
        "data_version": data_version,
    }
    if approx_sample_size is not None:
        kwargs["num_rows"] = approx_sample_size
    return connector.compute_col_stats(
        source_table_name,
        stats_dict,
        **kwargs,
    )


@blocking_func(DATA_BACKEND_THREAD_POOL)
def _compute_bp_sample_col_stats(
    sample,
    stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
    data_version: TableDataVersion,
    timestamp_precision: str,
) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
    """Reduce a bounded prediction sample, including SCORE grouped by CLASS."""
    from kumo.connector.connector import _compute_col_stats_from_sample

    result = _compute_col_stats_from_sample(
        stats_dict,
        sample,
        approx=True,
        data_version=data_version,
        timestamp_precision=timestamp_precision,
    )
    score_config = next(
        (config
         for config, requested in stats_dict.items() if config.name == 'SCORE'
         and ColStatType.SUMMARY_STATS_BY_CLASS in requested),
        None,
    )
    if score_config is None or 'CLASS' not in sample:
        return result

    valid = sample[[score_config.name, 'CLASS']].dropna()
    by_class = {}
    for class_value, frame in valid.groupby('CLASS', sort=False):
        values = frame[score_config.name].astype(float)
        by_class[str(class_value)] = {
            'min': float(values.min()),
            'max': float(values.max()),
            'mean': float(values.mean()),
        }
    result.setdefault(score_config,
                      {})[ColStatType.SUMMARY_STATS_BY_CLASS] = Stats(
                          by_class,
                          approx=True,
                          data_version=data_version,
                          num_entries=len(valid),
                          timestamp=now_utc().timestamp(),
                      )
    return result


def _merge_col_stats(
    primary: Dict[BaseColumnConfig, Dict[ColStatType, Stats]],
    fallback: Dict[BaseColumnConfig, Dict[ColStatType, Stats]],
) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
    merged = {col: dict(stats) for col, stats in primary.items()}
    for col, stats in fallback.items():
        if not stats:
            continue
        merged.setdefault(col, {})
        for stat_type, stat in stats.items():
            merged[col].setdefault(stat_type, stat)
    return merged


_BackgroundColStatsKey = Tuple[str, str, Tuple[Tuple[str, str, str,
                                                     Tuple[str, ...]], ...]]
_BACKGROUND_SAMPLE_COL_STATS_TASKS: Dict[_BackgroundColStatsKey,
                                         asyncio.Task] = {}


def _sample_col_stats_warmup_concurrency_limit() -> int:
    return max(0, SAMPLE_COL_STATS_WARMUP_CONCURRENCY.parse_value())


def _num_active_sample_col_stats_warmups() -> int:
    return sum(1 for task in _BACKGROUND_SAMPLE_COL_STATS_TASKS.values()
               if not task.done())


def _background_col_stats_key(
    table: TableDefinition,
    stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
    table_data_version: TableDataVersion,
) -> _BackgroundColStatsKey:
    stats_signature = tuple(
        sorted((
            col.name,
            str(col.stype),
            str(col.dtype),
            tuple(sorted(str(stat) for stat in stats)),
        ) for col, stats in stats_dict.items()))
    return (
        str(table.fingerprint()),
        str(table_data_version),
        stats_signature,
    )


class DataBackend:
    @staticmethod
    @trace()
    @measure_blocking_code
    async def start_refresh_table(
        ctx: DataBackendContext,
        connector_spec: ConnectorExecutionRef,
        table_def: TableDefinition,
        data_ver: TableDataVersion,
        alias: Optional[TableName] = None,
        defer_col_stats_validation: bool = False,
    ) -> Optional[str]:
        r"""Launch a table refresh workflow at specified data version.

        Args:
            connector_spec (ConnectorExecutionRef): connector transport for the table to
                be refreshed
            table_def (TableDefinition): definition of the table to be
                refreshed
            data_ver (TableDataVersion): data version of the table to refresh.
            alias (Optional[TableName]): User-facing table name passed to
                the workflow for use in validation error messages.
            defer_col_stats_validation (bool): When true, table refresh only
                performs ingest-time validation and lets the parent graph
                refresh compute and validate column stats at graph scope.
    """
        if ctx._settings.disable_async_ingestion:
            log.debug('Async table refresh is disabled!')
            return None

        log.debug(f"Start refresh workflow for "
                  f"{table_def.get_source_table_name()}")
        # TODO(Dulloor): start table or view refresh workflow
        temporal_client = ctx.temporal_client()
        client = await temporal_client.get()
        refresh_wf_id = TableRefreshWorkflow.compose_wf_id(table_def, data_ver)
        log.info(
            f"DataBackend.start_refresh_table: Starting TableRefreshWorkflow "
            f"for table={table_def.get_source_table_name()}, "
            f"wf_id={refresh_wf_id}")
        try:
            await client.start_workflow(
                TableRefreshWorkflow.run,
                args=[
                    ctx,
                    TableRefreshWorkflowInput(
                        connector_spec=connector_spec,
                        table=table_def,
                        data_version=data_ver,
                        alias=alias,
                        defer_col_stats_validation=defer_col_stats_validation,
                    ),
                ],
                id=refresh_wf_id,
                task_queue=DATA_BACKEND_TASK_QUEUE,
                execution_timeout=timedelta(
                    hours=DATA_BACKEND_REFRESH_TIMEOUT_HOURS),
            )
            log.info(
                'Started TableRefreshWorkflow %s and saved state under %s@%s',
                refresh_wf_id, table_def.cache_key(), data_ver)
            return refresh_wf_id
        except WorkflowAlreadyStartedError:
            log.info('Table refresh workflow %s is already running.',
                     refresh_wf_id)
            return None

    # Prepares a list of table files/directories that will be selected or
    # ignored as per requirement 4. of #2568. Returns S3FileList which
    # contains these lists and the counts that needs to be shown in
    # the UI.
    @staticmethod
    @trace(TraceOptions(trace_params=True))
    def s3_validate(s3_root_dir: str) -> S3FileList:
        # Get all files and directories in the root dir
        selected_list = []
        ignored_list = []
        # We check all suffixes to see if parquet or csv are present.
        sel_count = 0
        ign_count = 0

        top_level_tables = fsutil.listdir(path=s3_root_dir)

        # No tables found.
        if len(top_level_tables) == 0:
            return S3FileList(files_selected=[], files_ignored=[],
                              count_total=len(top_level_tables),
                              count_selected=0,
                              count_ignored=len(top_level_tables))

        # Default to parquet file and support all parquet / csv type.
        for entry_path in top_level_tables:
            table_path = entry_path.strip('/')
            table_name = os.path.basename(table_path)
            table_name = os.path.splitext(table_name)[0]
            if TEST_KUMO_FILE == table_name:
                continue

            file: S3File = S3File(file_name=table_path, details="")
            read_suffix = get_file_suffix(table_path,
                                          default_suffix=_PARQUET_SUFFIX)
            if (read_suffix in (_PARQUET_SUFFIX, _CSV_SUFFIX)
                    and sel_count < MAX_SUPPORTED_FILES):
                entry = S3Table(table_name=table_name, is_dir=False,
                                table_files=[file], file_type=read_suffix)
                selected_list.append(entry)
                sel_count += 1
            else:
                entry = S3Table(table_name=table_name, is_dir=False,
                                table_files=[file], file_type="")
                # Avoid overwhelmed number of ignored entries in the UI.
                if len(ignored_list) < 10:
                    ignored_list.append(entry)
                if sel_count >= MAX_SUPPORTED_FILES:
                    file.details = ("Too many files or directories behind the "
                                    "connector.")
                else:
                    file.details = (f"Default to parquet/csv connector, so all"
                                    f" {read_suffix} files are ignored.")
                ign_count += 1

        log.info(f"Validate s3 path {s3_root_dir}, found {sel_count} valid "
                 f"tables.")
        return S3FileList(files_selected=selected_list,
                          files_ignored=ignored_list,
                          count_total=len(top_level_tables),
                          count_selected=sel_count, count_ignored=ign_count)

    # helper to check if a stats request dict is covered by the minimal stats
    # generator used later to compose the request within the workflow
    @staticmethod
    def is_covered_by_minimal_stats(
            table: TableDefinition,
            stats_dict: Dict[BaseColumnConfig, List[ColStatType]]) -> bool:
        base_cols = table.to_base_columns()
        min_dict = get_minimal_stats(base_cols)
        min_dict = {k.base_config(): v for k, v in min_dict.items()}

        min_dict_by_name = {
            col_config.name: min_dict[col_config]
            for col_config in min_dict.keys()
        }
        for col_config, col_list in stats_dict.items():
            stats_list = min_dict_by_name.get(col_config.name)
            if not stats_list or not set(col_list).issubset(set(stats_list)):
                return False
        return True

    @staticmethod
    @measure_blocking_code
    async def save_table_refresh_result(
        table: VersionedSourceTableID,
        task_result: TableDataTaskResult,
    ) -> None:
        """Persist metadata about the execution of a single table ingestion
        workflow.
        """
        cache = get_active_cache()
        return await cache.table_refresh_result.save.coroutine(
            table.connector_name, table.source_table_name, table.version,
            task_result)

    @staticmethod
    @measure_blocking_code
    async def reset_table_refresh_markers(
        table: TableDefinition,
        data_version: TableDataVersion,
    ) -> None:
        """Synchronously reset *all* refresh markers for ``(table,
        data_version)`` to an in-progress state, durably, before a forced
        re-ingest (``refresh_source=true``) is launched.

        A forced re-ingest relaunches ``TableRefreshWorkflow``, which resets
        only the *overall* refresh marker (to INGEST/NOT_STARTED) at its own
        startup — and asynchronously. That leaves two inconsistencies a reader
        / waiter can observe:

          1. The per-stage ``COMPUTE_MIN_COL_STATS`` / ``COMPUTE_FULL_COL_STATS``
             marker files are *not* reset by the workflow, so they keep showing
             the previous refresh's ``DONE`` even while the overall marker is
             being reset.
          2. There is a window between the snapshot request returning and the
             relaunched workflow's reset becoming durable, during which a waiter
             keyed on ``(table, data_version)`` (identical across two forced
             refreshes of unchanged source data) still sees the *previous*
             refresh's fully-``DONE`` markers and returns prematurely.

        Resetting every marker here, in the request path before
        ``start_refresh_table``, closes both: by the time the snapshot request
        returns, all stages consistently read "in progress", so a subsequent
        ``_wait_for_graph_snapshot`` correctly waits the re-ingest out to
        COMPLETED instead of racing the stale ``DONE``. Force semantics are
        unchanged — the workflow still relaunches and re-ingests.
        """
        cache = get_active_cache()
        now = now_utc()

        # Overall marker -> INGEST_TABLE / IN_PROGRESS (mirrors the state the
        # relaunched TableRefreshWorkflow writes at startup). Keyed by
        # (connector_id, cache_key, version) to match get_many_table_refresh_
        # results(stage=None) and save_table_refresh_result.
        overall = TableDataTaskResult(
            status=AsyncDataTaskStatus.IN_PROGRESS,
            validation=ValidationResponse(),
            stage=TableRefreshStage.INGEST_TABLE,
            data_version=str(data_version),
            created_at=now,
            updated_at=now,
        )
        await cache.table_refresh_result.save.coroutine(
            table.get_connector_id(), table.cache_key(), str(data_version),
            overall)

        # Per-stage col-stat markers -> NOT_STARTED, so a waiter polling the
        # FULL_COL_STATS / MIN_COL_STATS stage no longer sees a terminal
        # (stale) result. These files are written at the same paths
        # get_many_table_refresh_results(stage=COMPUTE_*) reads.
        table_dir = cache.create_table_cache_dir(table.get_connector_id(),
                                                 table.cache_key(),
                                                 str(data_version),
                                                 table.fingerprint())
        for stage in (TableRefreshStage.COMPUTE_MIN_COL_STATS,
                      TableRefreshStage.COMPUTE_FULL_COL_STATS):
            await save_json.coroutine(
                TableDataTaskResult(
                    status=AsyncDataTaskStatus.NOT_STARTED,
                    validation=ValidationResponse(),
                    stage=stage,
                    data_version=str(data_version),
                    created_at=now,
                    updated_at=now,
                ),
                table_dir.colstat_task_result(stage),
            )

    @staticmethod
    async def get_many_table_refresh_results(
        tables: List[Tuple[TableDefinition, TableDataVersion]],
        stage: Optional[TableRefreshStage] = None
    ) -> List[Optional[TableDataTaskResult]]:
        """
        Batched operation for fetching table refresh results. If `stage` is
        specified, return the task result specific for that individual stage;
        otherwise return the overall table refresh result.
        """
        cache: BackendCacheManager = get_active_cache()
        if stage is None:
            return await cache.table_refresh_result.get_many.coroutine(
                [(table.get_connector_id(), table.cache_key(), ver)
                 for table, ver in tables], TableDataTaskResult)

        if stage == TableRefreshStage.INGEST_TABLE:
            return await cache.table_refresh_result.get_many.coroutine(
                [(table.get_connector_id(), table.get_source_table_name(), ver)
                 for table, ver in tables], TableDataTaskResult)

        if stage in [
                TableRefreshStage.COMPUTE_MIN_COL_STATS,
                TableRefreshStage.COMPUTE_FULL_COL_STATS
        ]:
            file_paths = []
            for table, data_version in tables:
                table_dir = get_active_cache().create_table_cache_dir(
                    table.get_connector_id(), table.cache_key(), data_version,
                    table.fingerprint())
                file_paths.append(table_dir.colstat_task_result(stage))
            log.debug('Load colstat task result from %s', file_paths)
            return await load_many_json.coroutine(file_paths,
                                                  TableDataTaskResult)

        raise AssertionError(
            f'Table task result for stage {stage} is not supported.')

    @staticmethod
    async def create_artifact_export_job(
        ctx: DataBackendContext,
        job_id: str,
        output_config: Union[PredictionOutputConfig, TrainingTableOutputConfig],
        connector_spec: ConnectorExecutionRef,
        artifact_export_job_id: ArtifactExportJobID,
    ) -> ArtifactExportJobID:
        """Creates an artifact job to export the
        output of prediction job `job_id` to the specified
        connector in `PredictionOutputConfig`.
        `credentials` need to be specified if the output connector is different
        from the connector specified when the prediction job was created."""

        temporal_client = ctx.temporal_client()
        client = await temporal_client.get()
        # Find the job output for the prediction artifact type.
        from rest.backends.ml_backend.data_util import (
            get_job_base_dir,
            get_job_dir,
        )
        home_dir = get_job_dir(get_job_base_dir(), JobID(job_id))
        home_dir_mgr = HomeDirManager(home_dir)

        if isinstance(output_config, (
                NamedTrainingTableOutputConfig,
                FileTrainingTableOutputConfig,
        )):
            train_table_job_id = GenerateTrainTableJobID(job_id)
            result = await temporal_client.result(train_table_job_id.wf_id)
            train_table_path: str = result.train_table_path
            artifact_paths = {output_config.artifact_type: train_table_path}
            artifact_export_input = ArtifactExportInput.from_out_config_n_spec(
                artifact_paths,
                output_config,
                connector_spec,
            )
        else:
            artifact_type = PredictionArtifactType(
                output_config.artifact_type)
            artifact = await resolve_prediction_artifact.coroutine(
                home_dir_mgr,
                artifact_type,
                validate_objects=False,
                include_selection_ref=True,
            )
            if artifact is None or artifact.selection_ref is None:
                raise ArtifactNotFoundError(path=osp.join(
                    home_dir_mgr.prediction_output_dir,
                    '_selected_artifacts.json',
                ))
            base_input = ArtifactExportInput.from_out_config_n_spec(
                {artifact_type: artifact.root},
                output_config,
                connector_spec,
            )
            artifact_export_input = replace(
                base_input,
                prediction_selection=artifact.selection_ref,
                prediction_artifact_types=(artifact_type, ),
            )

        try:
            from rest.backends.ml_backend import workflows as ml_workflows
            await client.start_workflow(
                ml_workflows.ArtifactExportWorkflow.run,
                args=[
                    artifact_export_input,
                ],
                id=artifact_export_job_id,
                task_queue=ML_WORKFLOW_TASK_QUEUE,
                execution_timeout=(ARTIFACT_EXPORT_WORKFLOW_EXECUTION_TIMEOUT),
                id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
            )
        except WorkflowAlreadyStartedError:
            log.info('Artifact export workflow %s was already dispatched.',
                     str(artifact_export_job_id))
        return artifact_export_job_id

    @staticmethod
    @trace()
    @measure_blocking_code
    async def get_temporal_job_status(
        ctx: DataBackendContext,
        job_id: str,
    ) -> JobStatus:
        temporal_client = ctx.temporal_client()
        wf_desc = await temporal_client.get_workflow_desc(job_id)
        if not wf_desc:
            return JobStatus.MISSING
        if wf_desc.status in {
                WorkflowExecutionStatus.FAILED,
                WorkflowExecutionStatus.TIMED_OUT
        }:
            return JobStatus.FAILED
        elif wf_desc.status in {
                WorkflowExecutionStatus.RUNNING,
                WorkflowExecutionStatus.CONTINUED_AS_NEW
        }:
            return JobStatus.RUNNING
        elif wf_desc.status == WorkflowExecutionStatus.COMPLETED:
            return JobStatus.DONE
        elif wf_desc.status in {
                WorkflowExecutionStatus.CANCELED,
                WorkflowExecutionStatus.TERMINATED
        }:
            return JobStatus.CANCELLED
        else:
            raise ValueError(
                f"Unsupported WorkflowExecutionStatus "
                f"{wf_desc.status}", )

    @staticmethod
    @trace()
    @measure_blocking_code
    async def cancel_temporal_job(
        ctx: DataBackendContext,
        job_id: str,
    ) -> JobStatus:
        temporal_client = ctx.temporal_client()
        await temporal_client.cancel(workflow_id=job_id)
        return await DataBackend.get_temporal_job_status(ctx=ctx, job_id=job_id)

    # Handles rest server request for col stats by checking if exact stats are
    # available in the cache and if so returning those. If provided, the job_id
    # is used to determine the table_data_version. Otherwise, the latest
    # version that has been ingested is used; if there is no such version the
    # latest version from the connector is used. If exact col stats are not
    # available a table refresh workflow is launched (which will not need to do
    # any actual compute for the ingest if that step is already done).
    @staticmethod
    @trace()
    @measure_blocking_code
    async def get_min_col_stats(
        connector_spec: ConnectorOpenSpec | None, table: TableDefinition,
        stats_dict: Dict[BaseColumnConfig,
                         List[ColStatType]], approx_sample_size: int,
        check_refresh_status: bool = True, compute_missing_stats: bool = True,
        allow_cached_approx_stats: bool = False,
        table_data_version: TableDataVersion | None = None
    ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
        connector_id = table.get_connector_id()
        source_table_name = table.get_source_table_name()
        cache_key = table.cache_key()
        fingerprint = table.fingerprint()

        # try to get the latest ingested version from backend cache.
        # Otherwise get the current version from the connector. For a Custom SQL
        # table the version hashes the SQL + referenced-table versions + output
        # cols (``table_def`` selects that branch), not a connector probe.
        if table_data_version is None:
            if connector_spec is None:
                raise ValueError(
                    'connector_spec is required to resolve a live data version')
            table_data_version = await get_table_data_version(
                connector_spec, source_table_name, table_def=table)

        minimal_stats_required = DataBackend.is_covered_by_minimal_stats(
            table, stats_dict)

        cache: BackendCacheManager = get_active_cache()
        results = await cache.table_refresh_result.get_many.coroutine([
            (connector_id, cache_key, table_data_version),
        ], TableDataTaskResult)
        refresh_status = results[0] or TableDataTaskResult(
            AsyncDataTaskStatus.NOT_STARTED, ValidationResponse(),
            TableRefreshStage.INGEST_TABLE, updated_at=now_utc())

        exact_stats_available = (minimal_stats_required
                                 and refresh_status.min_col_stats_computed
                                 ) or refresh_status.full_col_stats_computed
        exact_stats = {}

        def load_exact_min_stats(
        ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
            log.debug("Searching for exact stats")
            # DG-native: read exact min col stats from
            # BackendCacheManager directly (legacy
            # Backend.load_exact_min_col_stats_if_present was a thin
            # wrapper around this same load+get pair).
            cache.load_min_col_stats_cache(
                connector_name=fingerprint.connector_name,
                table_fingerprint=fingerprint,
                data_version=table_data_version,
            )
            loaded_stats = cache.get_min_col_stats(
                table_fingerprint=fingerprint,
                stats_dict=stats_dict,
                data_version=table_data_version,
                approx=False,
            )
            if len(loaded_stats) == 0:
                log.warning(
                    "get_exact_min_col_stats: exact stats not found for "
                    f"{fingerprint.connector_name}.{source_table_name}"
                    f" version: {table_data_version}")
            return loaded_stats

        def load_cached_approx_stats(
        ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
            loaded_stats = cache.get_min_col_stats(
                table_fingerprint=fingerprint,
                stats_dict=stats_dict,
                data_version=table_data_version,
                approx=True,
            )
            return _merge_col_stats(exact_stats, loaded_stats)

        if exact_stats_available or not check_refresh_status:
            exact_stats = load_exact_min_stats()
            if col_stat_dict_complete(exact_stats, stats_dict):
                log.debug("Exact stats returned and complete")
                return exact_stats
        elif check_refresh_status:
            # REST servers can cache an in-progress refresh result while a
            # worker later writes the exact stats and terminal task markers.
            # Treat the stats file itself as the freshness source here, and
            # only return once the requested exact stats are complete.
            exact_stats = load_exact_min_stats()
            if col_stat_dict_complete(exact_stats, stats_dict):
                log.debug("Exact stats returned and complete")
                return exact_stats

        cached_approx_stats = {}
        if allow_cached_approx_stats:
            cached_approx_stats = load_cached_approx_stats()
            if col_stat_dict_complete(cached_approx_stats, stats_dict):
                log.debug("Cached approximate stats returned and complete")
                return cached_approx_stats

        if not compute_missing_stats:
            return cached_approx_stats if allow_cached_approx_stats else exact_stats

        if connector_spec is None:
            raise ValueError(
                'connector_spec is required to compute missing column stats')

        log.debug('Exact stats not avilable for '
                  f'{connector_spec.connector_id}.{source_table_name}')

        # Pre-ingest resolution chain. Cache miss delegates to the connector
        # sample + Python stats fallback, then persists those approximate
        # stats without marking DG exact-stat stages complete.
        from statedb.utils.errors import UnsupportedInDiskgraphError

        # A Custom SQL table has no physical source table to probe by name: its
        # stats come exclusively from the ingested (materialized) snapshot read
        # above. Skip the connector fallback (a synthetic-name probe would fail)
        # and surface a materialization-oriented gap. The pre-materialization
        # preview stats surface lands separately (PR-6).
        if not table.is_sql:
            try:
                connector = await get_connector_from_spec_async(
                    connector_spec, resolved_factory=get_connector_from_spec)
                async with async_connector_lifecycle(connector):
                    tier2 = await _compute_connector_col_stats.coroutine(
                        connector,
                        source_table_name,
                        stats_dict,
                        data_version=table_data_version,
                        approx_sample_size=approx_sample_size,
                    )
            except Exception as exc:
                log.warning(
                    "DataBackend.get_min_col_stats: connector.compute_col_stats"
                    " failed for %s.%s@%s (error_type=%s)",
                    connector_spec.connector_id, source_table_name,
                    table_data_version,
                    type(exc).__name__)
                tier2 = {}

            if any(stats for stats in tier2.values()):
                cache.add_col_stats(fingerprint, tier2)
                await cache.save_col_stats_cache.coroutine(
                    fingerprint,
                    table_data_version,
                )
                return _merge_col_stats(
                    cached_approx_stats
                    if allow_cached_approx_stats else exact_stats,
                    tier2,
                )

        if table.is_sql:
            remediation = (
                'a Custom SQL table populates min col stats by materializing '
                '(running its query and ingesting the result)')
        else:
            remediation = ('DiskGraph ingest must populate the requested min '
                           'col stats first, or the connector must support '
                           'sample reads')
        raise UnsupportedInDiskgraphError(
            'DataBackend.get_min_col_stats: requested exact min col stats '
            f"unavailable for {connector_spec.connector_id}."
            f"{source_table_name}@{table_data_version}; DG cache miss. "
            f"{remediation}.")

    @staticmethod
    def start_min_col_stats_cache_warmup(
        connector_spec: ConnectorExecutionRef,
        table: TableDefinition,
        stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
        approx_sample_size: int,
        *,
        check_refresh_status: bool = True,
        table_data_version: TableDataVersion,
    ) -> bool:
        key = _background_col_stats_key(table, stats_dict, table_data_version)
        existing_task = _BACKGROUND_SAMPLE_COL_STATS_TASKS.get(key)
        if existing_task is not None and not existing_task.done():
            log.debug("Sample column stats cache warmup already running for %s",
                      key)
            return False

        concurrency_limit = _sample_col_stats_warmup_concurrency_limit()
        if concurrency_limit <= 0:
            log.debug("Sample column stats cache warmup disabled for %s", key)
            return False
        if _num_active_sample_col_stats_warmups() >= concurrency_limit:
            log.debug(
                "Skipping sample column stats cache warmup for %s; "
                "concurrency cap %s reached", key, concurrency_limit)
            return False

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            log.warning(
                "Cannot schedule sample column stats cache warmup without a "
                "running event loop for %s.%s@%s",
                connector_spec.connector_id,
                table.get_source_table_name(),
                table_data_version,
            )
            return False

        async def _warm_cache() -> None:
            try:
                await DataBackend.get_min_col_stats(
                    connector_spec,
                    table,
                    stats_dict,
                    approx_sample_size,
                    check_refresh_status=check_refresh_status,
                    compute_missing_stats=True,
                    allow_cached_approx_stats=True,
                    table_data_version=table_data_version,
                )
            except Exception:
                log.exception(
                    "Sample column stats cache warmup failed for %s.%s@%s",
                    connector_spec.connector_id,
                    table.get_source_table_name(),
                    table_data_version,
                )
            finally:
                _BACKGROUND_SAMPLE_COL_STATS_TASKS.pop(key, None)

        _BACKGROUND_SAMPLE_COL_STATS_TASKS[key] = loop.create_task(
            _warm_cache())
        log.info("Scheduled sample column stats cache warmup for %s.%s@%s",
                 connector_spec.connector_id, table.get_source_table_name(),
                 table_data_version)
        return True

    # Kick off graph refresh workflow.
    @staticmethod
    async def _execute_single_activity(
        ctx: DataBackendContext,
        activity_fn,
        activity_inputs: list,
        workflow_id_prefix: str,
        execution_timeout: Optional[timedelta] = None,
    ):
        from temporalio_commons.worker_registry import RunSingleActivityWorkflow  # isort: skip

        workflow_id = f'{workflow_id_prefix}:{uuid4().hex}'
        try:
            client = await ctx.temporal_client().get()
            return await client.execute_workflow(
                RunSingleActivityWorkflow.run,
                args=[activity_fn.__name__, activity_inputs],
                id=workflow_id,
                task_queue=DATA_BACKEND_TASK_QUEUE,
                execution_timeout=(
                    execution_timeout
                    or timedelta(hours=DATA_BACKEND_REFRESH_TIMEOUT_HOURS)),
            )
        except (Exception, asyncio.CancelledError) as error:
            failure = error
        # Translate outside the active ``except`` context so the sanitized
        # exception cannot retain the original Temporal failure via
        # ``__context__``.
        _raise_safe_single_activity_failure(
            failure,
            activity_fn.__name__,
            workflow_id,
        )

    @staticmethod
    async def refresh_graph_data_version_in_activity(
        ctx: DataBackendContext,
        graph_input: GraphDataComputeInput,
        job_id: Optional[str] = None,
    ) -> GraphDataVersionInfo:
        """Refresh source versions without resolving a connector in REST."""
        from rest.backends.data_backend.activities import DataBackendActivities

        return await DataBackend._execute_single_activity(
            ctx,
            DataBackendActivities.refresh_graph_data_version,
            [graph_input, job_id],
            f'refresh-graph-version:{graph_input.graph_key}',
        )

    @staticmethod
    async def get_table_data_version_in_activity(
        ctx: DataBackendContext,
        connector: ConnectorExecutionRef,
        source_table_name: str,
        *,
        refresh_table_version: bool = False,
        table_def: Optional[TableDefinition] = None,
    ) -> TableDataVersion:
        from rest.backends.data_backend.activities import DataBackendActivities

        # A Custom SQL table has no physical source table to probe by name: its
        # version is a content hash over the SQL + referenced-table versions +
        # output cols. Thread ``table_def`` (the refresh/version callers hold
        # it) into the activity so that branch runs; physical tables pass None
        # and keep the connector-probe path.
        return await DataBackend._execute_single_activity(
            ctx,
            DataBackendActivities.get_connector_table_data_version,
            [connector, source_table_name, refresh_table_version, table_def],
            f'table-data-version:{connector.project_id}:{connector.connector_id}',
        )

    @staticmethod
    @measure_blocking_code
    async def start_refresh_graph(
        ctx: DataBackendContext,
        graph_input: GraphDataComputeInput,
        graph_version_info: Optional[GraphDataVersionInfo] = None,
    ) -> Optional[str]:
        # TODO(Dulloor): Allow refresh of graphs without links
        if not graph_input.graph_definition.col_groups:
            return None

        if ctx._settings.disable_async_ingestion:
            log.debug('Async edge stats computation is disabled!')
            return None

        client = await ctx.temporal_client().get()
        graph_refresh_wf_id = GraphRefreshWorkflow.compose_wf_id(
            graph_input, graph_version_info)
        log.info(
            f"DataBackend.start_refresh_graph: Starting GraphRefreshWorkflow "
            f"for graph={graph_input.graph_display_name}, "
            f"wf_id={graph_refresh_wf_id}")
        try:
            await client.start_workflow(
                GraphRefreshWorkflow.run,
                args=[
                    ctx,
                    graph_input,
                    graph_version_info,
                    None,  # job_id
                ],
                id=graph_refresh_wf_id,
                task_queue=DATA_BACKEND_TASK_QUEUE,
                execution_timeout=timedelta(
                    hours=DATA_BACKEND_REFRESH_TIMEOUT_HOURS),
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
            )
            log.info('Triggered async graph refresh in wf_id=%s, input=%s',
                     graph_refresh_wf_id, graph_input)
        except WorkflowAlreadyStartedError:
            log.info('Graph refresh wf %s is already running',
                     graph_refresh_wf_id)

        return graph_refresh_wf_id

    @staticmethod
    @trace()
    @measure_blocking_code
    async def get_edge_stats(
        ctx: DataBackendContext, graph_input: GraphDataComputeInput,
        graph_version_info: GraphDataVersionInfo | None = None
    ) -> Optional[List[EdgeHealth]]:
        """Fetches edge stats for a single graph. If stats are not available,
        return none and trigger a graph ingestion workflow in the background.
        """
        graph_cache = get_active_cache().graph_cache

        if graph_version_info is None:
            versioned_key = (
                await graph_cache.get_many_latest_graph_versions.coroutine(
                    [graph_input.graph_key]))[0]

            # Graph refresh has not been started or failed before saving.
            if versioned_key is None:
                log.error("Graph has no data refreshed yet %s", graph_input)
                return None
        else:
            versioned_key = VersionedGraphKey(
                graph_input.graph_key, graph_version_info.graph_data_version)

        if not graph_input.graph_definition.col_groups:
            return []

        edge_stats = await graph_cache.get_graph_edge_stats.coroutine(
            versioned_key)

        if edge_stats is None:
            log.error(
                "Edge stats compute was completed for %s but result is "
                "missing, this might happen due to a bad workflow run or "
                "data loss in persist dir.", versioned_key)
            return None

        return parse_edge_stats_result(graph_input, edge_stats)

    @staticmethod
    @blocking_func(DATA_BACKEND_THREAD_POOL)
    def get_many_graph_validations(
            graph_keys: List[GraphKey]) -> Dict[GraphKey, ValidationResponse]:
        """
        Get many validation responses (if present), for each of provided graph
        keys. Each graph key should be global dedupe key of a graph, for more
        details, please refer to GraphDataComputeInput.graph_key field. Return
        empty ValidationResponse for graph that does not yet have validation
        result.
        """
        cache = get_active_cache()
        graph_version_keys = cache.graph_cache.get_many_latest_graph_versions(
            graph_keys)

        graph_validation_file_paths: Dict[GraphKey,
                                          List[str]] = defaultdict(list)
        validation_file_paths: List[str] = []
        for graph_id, key in zip(graph_keys, graph_version_keys):
            if key:
                graph_dir = cache.graph_cache.graph_dir(key.key, key.version)
                file_paths = [
                    HomeDirManager(graph_dir).validation_path('edge_stats'),
                    HomeDirManager(graph_dir).validation_path('table_stats'),
                    HomeDirManager(graph_dir).validation_path('table_col_stats')
                ]
                graph_validation_file_paths[graph_id] = file_paths
                validation_file_paths.extend(file_paths)
            else:
                log.info('No graph data version found for %s', graph_id)

        files = fsutil.get_many_if_present(validation_file_paths)
        validations = [
            fast_from_json(file, ValidationResponse)
            if file else ValidationResponse() for file in files
        ]
        path_to_validation = dict(zip(validation_file_paths, validations))
        result: Dict[GraphKey, ValidationResponse] = {}
        for graph_key in graph_keys:
            file_paths = graph_validation_file_paths[graph_key]
            validation = ValidationResponse()
            for file_path in file_paths:
                validation.merge(path_to_validation[file_path])
            result[graph_key] = validation

        return result

    @staticmethod
    @connector_activity_registry.connector_resolution_helper
    @measure_blocking_code
    async def start_bp_stats_activity(
        ctx: DataBackendContext,
        job_input: BatchPredictionJobInput,
        use_transformed_output: bool = False,
        direct_databricks_output_index: int | None = None,
        direct_databricks_stage: DatabricksManagedVolumeStage | None = None,
    ) -> None:
        """Compute Kumo-format column statistics for batch predictions.

        Durable outputs use DiskGraph over the local/staged Parquet artifact.
        Databricks-only outputs instead sample their producer-finalized UC
        volume stage before COPY/cleanup, so column stats never require a
        second full output copy through Kumo's control-plane storage.

        Downstream readers (``PredStats.parse``, the REST
        ``/prediction_stats`` endpoint, encoder inference for tasks that
        consume BP-output as a source table) light up automatically
        because we write into the same cache layout that ingest col-stats
        use.
        """
        from kumoapi.pquery import QueryType

        from kumo.diskgraph import get_service
        from kumo.diskgraph.service import _to_kumo_col_specs
        from rest.backends.ml_backend.data_util import (
            get_job_base_dir,
            get_job_dir,
        )
        from rest.utils.driver_common import generate_bp_stats_workflow_args

        if (PredictionArtifactType.PREDICTIONS
                not in requested_prediction_artifact_types(job_input.config)):
            log.info(
                'Skipping prediction column statistics because predictions '
                'were not requested: job=%s', job_input.id)
            return

        bp_job_home_dir = get_job_dir(get_job_base_dir(), JobID(job_input.id))
        job_outputs_dir = osp.join(bp_job_home_dir, 'outputs')
        job_stats_path = osp.join(job_outputs_dir, 'col_stats')
        databricks_only = has_only_databricks_outputs(job_input.config)

        # The awaited parent stats workflow runs after the output workflow.
        # Direct Databricks output computes these stats earlier while the
        # managed-volume stage still exists; make the later invocation a
        # cheap idempotent check.
        if (not use_transformed_output and direct_databricks_stage is None
                and databricks_only and fsutil.exists(job_stats_path)):
            log.info(
                'BP col-stats already available for direct output: job=%s',
                job_input.id,
            )
            return

        if direct_databricks_stage is None and databricks_only:
            if use_transformed_output:
                log.info(
                    'Skipping deprecated transformed BP stats for a '
                    'Databricks-only output with no durable artifact: job=%s',
                    job_input.id,
                )
                return
            if direct_databricks_prediction_output_index(
                    job_input.config) is not None:
                raise RuntimeError(
                    'Direct Databricks prediction stats were not computed '
                    'before the managed-volume stage was cleaned up.')

        if (direct_databricks_stage is None) != (direct_databricks_output_index
                                                 is None):
            raise ValueError(
                'Direct Databricks stats require both an output index and '
                'a managed-volume stage.')
        if direct_databricks_stage is not None:
            if use_transformed_output:
                raise ValueError(
                    'A raw Databricks stage cannot compute transformed stats.')
            expected_index = direct_databricks_prediction_output_index(
                job_input.config)
            if direct_databricks_output_index != expected_index:
                raise ValueError(
                    'Direct Databricks stats stage does not match the '
                    'configured raw-prediction output.')
            if not direct_databricks_stage.files:
                raise ValueError(
                    'Direct Databricks stats stage contains no Parquet files.')

        timestamp_precision = (job_input.graph_version_info.timestamp_precision
                               if job_input.graph_version_info is not None else
                               'seconds')

        entity_col_name = job_input.predictive_query.get_bp_entity_col_name()
        (bp_connector_spec, bp_table_def, bp_data_version,
         bp_col_stats_set) = await generate_bp_stats_workflow_args(
             job_id=job_input.id,
             job_home_dir=bp_job_home_dir,
             task_type=job_input.config.task_type,
             is_temporal=(
                 job_input.predictive_query.query_type == QueryType.TEMPORAL),
             entity_col_name=entity_col_name,
             use_transformed_output=use_transformed_output,
             data_version_override=(str(job_input.id) if direct_databricks_stage
                                    is not None else None),
         )

        home = HomeDirManager(bp_job_home_dir)
        selected_parquet_files: list[str] | None = None
        parquet_source: str
        if direct_databricks_stage is None:
            if use_transformed_output:
                parquet_source = home.internal_transformed_predictions_file
            else:
                predictions = await resolve_prediction_artifact.coroutine(
                    home, PredictionArtifactType.PREDICTIONS)
                if predictions is None:
                    raise RuntimeError(
                        'Batch prediction statistics found no selected '
                        'predictions')
                parquet_source = predictions.root
                selected_parquet_files = (list(predictions.files)
                                          if predictions.files is not None else
                                          None)
        backend_cache = get_active_cache()

        if direct_databricks_stage is not None:
            from kumo.connector import DatabricksConnector
            from kumo.stats.stats import NUM_ROWS_FOR_APPROX_STATS
            from rest.backends.data_backend import connector_broker
            from statedb.config.common import ConnectorID

            assert direct_databricks_output_index is not None
            output = job_input.config.outputs[direct_databricks_output_index]
            reference = job_input.output_connectors[ConnectorID(
                output.connector_id)]
            connector = get_connector_from_spec(
                await
                connector_broker.connector_credential_broker.resolve_async(
                    reference))
            if not isinstance(connector, DatabricksConnector):
                raise TypeError(
                    'Direct Databricks stats output has a non-Databricks '
                    'connector.')

            request = bp_col_stats_set.get_stats_types(bp_table_def)
            async with async_connector_lifecycle(connector,
                                                 open_connector=True):
                sample = await (
                    connector.read_managed_volume_parquet_sample.coroutine(
                        direct_databricks_stage.path,
                        list(request),
                        num_rows=NUM_ROWS_FOR_APPROX_STATS,
                    ))

            if sample.empty:
                fsutil.makedirs(job_outputs_dir, exist_ok=True)
                await save_json.coroutine({}, job_stats_path)
                log.info(
                    'BP col-stats unavailable for empty Databricks output: '
                    'job=%s files=%d',
                    job_input.id,
                    len(direct_databricks_stage.files),
                )
                return

            sampled_stats = await _compute_bp_sample_col_stats.coroutine(
                sample,
                request,
                bp_data_version,
                timestamp_precision,
            )
            if not col_stat_dict_complete(sampled_stats, request):
                raise RuntimeError(
                    'Databricks staged prediction sample did not produce all '
                    'required column statistics.')
            backend_cache.add_col_stats(
                bp_table_def.fingerprint(),
                sampled_stats,
            )
            cache_stats_path = await (
                backend_cache.save_col_stats_cache.coroutine(
                    bp_table_def.fingerprint(),
                    str(bp_data_version),
                ))
            if not fsutil.exists(cache_stats_path):
                raise RuntimeError(
                    'Databricks prediction statistics were not persisted.')

            fsutil.makedirs(job_outputs_dir, exist_ok=True)
            await fsutil.copy.coroutine(
                cache_stats_path,
                job_stats_path,
                overwrite=True,
                raise_on_exists=False,
            )
            log.info(
                'BP col-stats written: job=%s transformed=False source='
                'Databricks managed-volume sample (%d rows, %d files)',
                job_input.id,
                len(sample),
                len(direct_databricks_stage.files),
            )
            return

        kumo_column_specs = _to_kumo_col_specs(bp_table_def)

        # The DiskGraph Rust reader opens local files only (std::fs::File).
        # On a non-local dataplane (S3 / ADLS / GCS / UC Volumes) the
        # BP output lives at a remote URI, so stage it to local SSD first.
        # fsutil is the dataplane-agnostic layer (it covers every backend;
        # the Rust object_store path would only handle s3/gs/az), mirroring
        # kumo/diskgraph/prediction_artifacts.py.
        staging_dir: Optional[str] = None
        assert parquet_source is not None
        if fsutil.is_local(parquet_source):
            local_source = parquet_source
            parquet_paths = (selected_parquet_files
                             if selected_parquet_files is not None else
                             _list_parquet_files(local_source))
        else:
            variant = 'transformed' if use_transformed_output else 'predictions'
            local_source = await _stage_remote_parquet_local(
                parquet_source,
                str(job_input.id),
                variant,
                parquet_files=selected_parquet_files,
            )
            staging_dir = osp.dirname(local_source)
            parquet_paths = _list_parquet_files(local_source)

        try:
            if not parquet_paths:
                raise RuntimeError(
                    'Batch prediction column statistics found no Parquet '
                    f'files at {parquet_source}.')
            # @blocking_method, so from this async activity body it must be
            # invoked via .coroutine(...) rather than called directly on the
            # event loop. One call reads the parquet once and returns both
            # the full and min variants.
            svc = get_service()
            full_json, min_json = \
                await svc.compute_col_stats_for_parquet.coroutine(
                    parquet_paths=parquet_paths,
                    kumo_column_specs=kumo_column_specs,
                    data_version=str(bp_data_version),
                    timestamp_precision=timestamp_precision,
                )
        finally:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)

        table_dir = backend_cache.create_table_cache_dir(
            str(bp_connector_spec.connector_id),
            bp_table_def.get_source_table_name(),
            str(bp_data_version),
            bp_table_def.fingerprint(),
        )
        fsutil.write_text(table_dir.col_stats, full_json)
        fsutil.write_text(table_dir.min_col_stats, min_json)

        # Also write to the per-job directory for the fast reader path
        # (``try_load_col_stats_from_per_job_dir`` at
        # ``rest/driver/table_logic.py:2385``). When present, the
        # /v1/batch_prediction_jobs/{id}/prediction_stats endpoint picks
        # this up directly and skips the backend-cache fallback
        # round-trip. Reader is hardcoded to
        # ``{job_home_dir}/outputs/col_stats`` and has no
        # transformed-variant branch, so we only write here for the raw
        # predictions case.
        if not use_transformed_output:
            fsutil.makedirs(job_outputs_dir, exist_ok=True)
            fsutil.write_text(job_stats_path, full_json)

        # Mark refresh status as DONE so DataBackend.get_min_col_stats
        # (data_backend.py:488) sees the stats as ready. Without this the
        # reader's gating check (``min_col_stats_computed``) returns
        # False and the REST /prediction_stats endpoint raises 422 with
        # "DG cache miss, and connector.compute_col_stats produced no
        # stats" — even though the col_stats files are on disk. Mirrors
        # what the ingest path does in graph_refresh_workflow.py:326-374.
        # ``save`` is @blocking_method → call via .coroutine(...).
        now = now_utc()
        for stage in (TableRefreshStage.COMPUTE_MIN_COL_STATS,
                      TableRefreshStage.COMPUTE_FULL_COL_STATS,
                      TableRefreshStage.COMPLETED):
            await backend_cache.table_refresh_result.save.coroutine(
                bp_table_def.get_connector_id(),
                bp_table_def.cache_key(),
                str(bp_data_version),
                TableDataTaskResult(
                    status=AsyncDataTaskStatus.DONE,
                    stage=stage,
                    validation=ValidationResponse(),
                    data_version=str(bp_data_version),
                    created_at=now,
                    updated_at=now,
                ),
            )

        log.info(
            'DG BP col-stats written: job=%s transformed=%s parquet=%d files',
            job_input.id, use_transformed_output, len(parquet_paths))

    @staticmethod
    @measure_blocking_code
    async def compute_standalone_table_col_stats(
        connector_spec: ConnectorExecutionRef,
        table_def: TableDefinition,
        table_data_version: TableDataVersion,
    ) -> bool:
        """Compute exact col-stats for standalone table refreshes.

        Graph refresh computes stats at graph scope. A standalone table refresh
        only owns one table, so scan that table's ingested parquet directly with
        the DiskGraph wheel and write the same backend-cache artifacts.
        """
        from kumo.data.backend_cache import DirectDiskgraphSourceDecision
        from rest.config.table_v2 import is_file_source_table

        file_source_decision = None
        if is_file_source_table(table_def):
            file_source_decision = (await get_active_cache(
            ).direct_diskgraph_source_decision.get.coroutine(
                table_def.fingerprint(),
                str(table_data_version),
                cls=DirectDiskgraphSourceDecision,
            ))
            if (file_source_decision is not None
                    and file_source_decision.source is not None):
                log.info(
                    'Skipping standalone table col-stats for frozen direct '
                    'file source: connector=%s table=%s version=%s',
                    connector_spec.connector_id,
                    table_def.get_source_table_name(),
                    table_data_version,
                )
                return False

        connector = await get_connector_from_spec_async(
            connector_spec, resolved_factory=get_connector_from_spec)
        async with async_connector_lifecycle(connector):
            from kumo.connector.file_connector import FileConnector

            source_name = table_def.get_source_table_name()
            # A directly-read table stages no Kumo Parquet, so there is
            # nothing here to scan.
            direct_source = (
                isinstance(connector, FileConnector)
                and file_source_decision is None
                and connector.uses_direct_diskgraph_source_for(source_name))
            if direct_source:
                log.info(
                    'Skipping standalone table col-stats for a table '
                    'without staged local Parquet: connector=%s table=%s '
                    'version=%s', connector_spec.connector_id, source_name,
                    table_data_version)
                return False

            from diskgraph import DiskgraphSourceValidationError

            from common_utils.error import PublicException
            from kumo.data import MaterializationStage
            from kumo.data.utils import (
                get_diskgraph_local_ingest_path,
                get_materialize_path,
            )
            from kumo.diskgraph import get_service
            from kumo.diskgraph.service import _to_kumo_col_specs

            data_version = str(table_data_version)
            staging_dir: Optional[str] = None
            materialize_path = get_materialize_path(table_def.fingerprint(),
                                                    data_version,
                                                    MaterializationStage.INGEST)
            try:
                if fsutil.is_local(materialize_path):
                    parquet_paths = _list_parquet_files(materialize_path)
                elif _list_parquet_files(materialize_path):
                    local_source = await _stage_remote_table_parquet_local(
                        materialize_path, table_def, data_version)
                    staging_dir = osp.dirname(local_source)
                    parquet_paths = _list_parquet_files(local_source)
                else:
                    parquet_paths = []

                if not parquet_paths:
                    diskgraph_path = get_diskgraph_local_ingest_path(
                        table_def.fingerprint(), data_version)
                    parquet_paths = _list_parquet_files(diskgraph_path)
                if not parquet_paths:
                    raise RuntimeError(
                        "Standalone table refresh could not find ingested parquet "
                        f"for {table_def.get_source_table_name()}@{data_version}."
                    )

                full_json, min_json = \
                    await get_service().compute_col_stats_for_parquet.coroutine(
                        parquet_paths=parquet_paths,
                        kumo_column_specs=_to_kumo_col_specs(table_def),
                        data_version=data_version,
                    )
            except DiskgraphSourceValidationError as error:
                raise PublicException(
                    message=(f"Table '{source_name}' could not be refreshed: "
                             f"{error}"),
                    non_retryable=True,
                ) from None
            finally:
                if staging_dir is not None:
                    shutil.rmtree(staging_dir, ignore_errors=True)

            table_dir = get_active_cache().create_table_cache_dir(
                str(table_def.get_connector_id()),
                table_def.get_source_table_name(),
                data_version,
                table_def.fingerprint(),
            )
            fsutil.write_text(table_dir.col_stats, full_json)
            fsutil.write_text(table_dir.min_col_stats, min_json)

            log.info(
                'Standalone table col-stats written: table=%s version=%s '
                'parquet=%d files', table_def.get_source_table_name(),
                data_version, len(parquet_paths))
            return True


async def _stage_remote_table_parquet_local(
    parquet_source: str,
    table_def: TableDefinition,
    data_version: str,
) -> str:
    """Copy remote standalone table parquet to local SSD for DiskGraph stats."""
    from service_framework._raw_deploy_config import DiskgraphConfig
    from service_framework.deploy_env_config import deploy_env

    dc = deploy_env().data_compute
    if not isinstance(dc, DiskgraphConfig):
        raise RuntimeError(
            'compute_standalone_table_col_stats requires a DiskGraph '
            f'data_compute config; got {type(dc).__name__}')

    staging_dir = osp.join(dc.data_root, 'table_colstats',
                           table_def.fingerprint().ingest_path_fpid,
                           str(data_version))
    shutil.rmtree(staging_dir, ignore_errors=True)
    os.makedirs(staging_dir, exist_ok=True)
    local_target = osp.join(staging_dir, 'source.parquet')
    try:
        await fsutil.copy.coroutine(parquet_source, local_target,
                                    overwrite=True, raise_on_exists=False)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return local_target


async def _stage_remote_parquet_local(
    parquet_source: str,
    job_id: str,
    variant: str,
    *,
    parquet_files: list[str] | None = None,
) -> str:
    """Copy a remote-dataplane BP-output parquet to local SSD and return the
    local path.

    The BP output (``predictions.parquet``) is written straight to the remote
    dataplane storage and the stats activity runs on a different worker than
    the predictor, so there is no local copy to reuse — we download it. The
    diskgraph Rust reader is local-filesystem only; ``fsutil`` is the one
    dataplane-agnostic bridge (S3 / ADLS / GCS / UC Volumes), so we
    stage via ``fsutil.copy`` rather than teaching Rust each backend.

    Staging lives under the diskgraph local NVMe ``data_root`` in the
    job-scoped ``pred_artifact_cache/{job_id}`` namespace that
    ``cleanup_diskgraph_staging([job_id])`` already sweeps — a crash-leak
    backstop alongside the explicit cleanup the caller does after compute.
    This activity is diskgraph-only, so a non-DiskgraphConfig deploy is a
    hard error (no silent ``/tmp`` fallback that would ENOSPC on a
    billions-of-rows output).
    """
    from service_framework._raw_deploy_config import DiskgraphConfig
    from service_framework.deploy_env_config import deploy_env

    dc = deploy_env().data_compute
    if not isinstance(dc, DiskgraphConfig):
        raise RuntimeError(
            'start_bp_stats_activity requires a DiskGraph data_compute '
            f'config; got {type(dc).__name__}')
    staging_dir = osp.join(dc.data_root, 'pred_artifact_cache', job_id,
                           'bp_colstats', variant)
    # Fresh dir per attempt; the caller rmtree's it after compute.
    shutil.rmtree(staging_dir, ignore_errors=True)
    os.makedirs(staging_dir, exist_ok=True)
    local_target = osp.join(staging_dir, 'predictions.parquet')
    # fsutil.copy is @blocking_func; from this async activity it must be
    # awaited via .coroutine(...) rather than called on the event loop.
    if parquet_files is None:
        await fsutil.copy.coroutine(parquet_source, local_target,
                                    overwrite=True, raise_on_exists=False)
    else:
        os.makedirs(local_target, exist_ok=True)
        names = [osp.basename(path) for path in parquet_files]
        if len(set(names)) != len(names):
            raise ValueError('Selected prediction part names are duplicated')
        try:
            for source, name in zip(parquet_files, names, strict=True):
                await asyncio.to_thread(
                    fsutil.copy_file,
                    source,
                    osp.join(local_target, name),
                    overwrite=True,
                )
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    # Return the copied path itself (file or part-dir); the caller reads it
    # via _list_parquet_files and removes ``staging_dir`` after.
    return local_target


def _list_parquet_files(parquet_source: str) -> List[str]:
    """Resolve a BP-output parquet location into a flat file list.

    Handles three writer shapes:
      - Single ``predictions.parquet`` file.
      - Directory ``predictions.parquet/`` containing one or more
        ``part-*.parquet`` files (Spark-style partitioned writer).
      - Empty / non-existent path → empty list (the caller logs + skips).
    """
    if not fsutil.exists(parquet_source):
        return []
    if not fsutil.isdir(parquet_source):
        return [parquet_source]
    # Listing/suffix-filter/sort is delegated to fsutil.listdir (returns full,
    # sorted, '.parquet'-filtered paths — same usage as warehouse.py and
    # rest.utils.public); only the file-vs-dir-vs-missing resolution above is
    # specific to BP-output writer shapes.
    return fsutil.listdir(parquet_source, suffix='.parquet')


# Helper function to get file suffix.


def get_file_suffix(file_name: str, default_suffix: str = "") -> str:
    suffix = default_suffix
    dot_suffix = os.path.splitext(file_name)[1]
    if dot_suffix:
        suffix = dot_suffix[1:]
    return suffix


@blocking_func()
def get_output_path(home_dir_mgr: HomeDirManager, job_id: str,
                    prediction_output: PredictionOutputConfig) -> Dict:
    prediction_config = PredictionConfig.load(
        home_dir_mgr.job_config_path(job_id))
    out_dir = prediction_config.output_dir

    out_filename = f"{prediction_output.artifact_type.value.lower()}.parquet"
    output_path = os.path.join(out_dir, out_filename)
    if not fsutil.exists(output_path):
        raise ArtifactNotFoundError(path=output_path)
    return {prediction_output.artifact_type: output_path}


@blocking_func()
def get_train_table_artifact_path(
        home_dir_mgr: HomeDirManager,
        output_config: TrainingTableOutputConfig) -> Dict:
    artifact_type = output_config.artifact_type
    if artifact_type == TrainingTableArtifactType.FULL_TRAIN_TABLE:
        train_table_dir = home_dir_mgr.train_table_dir
        if not fsutil.exists(train_table_dir):
            raise ArtifactNotFoundError(path=train_table_dir)
        return {artifact_type: train_table_dir}
    else:
        raise ValueError(f"Unsupported artifact type: {artifact_type}")


@blocking_func()
def get_prediction_job_input_file(
        prediction_job_input_file: str) -> BatchPredictionJobInput:
    return json.load_json(prediction_job_input_file, BatchPredictionJobInput)
