# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from urllib.parse import urlsplit

from fastapi import HTTPException
from kumo_core.explain.schemas import EntityOutput, HistoricalItems
from kumo_core.explain.schemas import Subgraph as OldSubgraph
from kumo_core.model_plan import (
    TrainingTableGenerationPlan,
    inferred_or_missing_type_to_false,
    inferred_or_missing_type_to_none,
)
from kumo_core.prediction_options import resolve_classification_threshold
from kumo_core.types import PredictionArtifactType, TaskType
from kumo_core.types.schema import FPID
from kumoapi import jobs as jobs_api
from kumoapi.data_source import ConnectorType
from kumoapi.explain import ColumnAnalysisOutput, Subgraph
from kumoapi.pquery import QueryType
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype

from async_utils.instrument import measure_blocking_code
from common_utils.string import re
from io_utils import json, logging
from io_utils.json import load_json_async, load_many_json
from kumo.config.explain import StrategyMappings
from kumo.config.graph import TableConfig
from kumo.data import get_active_cache
from kumo.graph.table import Table
from kumo.home import HomeDirManager
from kumo.validation import ValidationError, ValidationResponse
from kumo.validation.validation import ValidationWarning
from rest.backends.data_backend.activities.prediction_output_validation import (
    validate_object_prediction_output,
    validate_warehouse_prediction_output,
)
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.data_backend import DataBackend
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.batch_prediction_job import (
    BatchPredictionJobConfig,
    BatchPredictionJobInput,
    BatchPredictionJobResponse,
    PaginatedBatchPredictionJobResponse,
    PaginationArgs,
    has_only_databricks_outputs,
)
from rest.config.data_backend import (
    ConnectorExecutionRef,
    ConnectorSpec,
    GraphDataVersionInfo,
)
from rest.config.evaluation import (
    ColumnExplanation,
    NavigationInfo,
    XAIResponse,
)
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.jobs import BatchPredictionRequest, MetadataField
from rest.config.ml_backend import MaterializationTaskInfo
from rest.config.prediction_output import (
    ADLSPredictionOutput,
    BigQueryPredictionOutput,
    DatabricksPredictionOutput,
    GCSPredictionOutput,
    LocalPredictionOutput,
    PredictionOutputConfig,
    S3PredictionOutput,
    SnowflakePredictionOutput,
    normalize_object_prediction_output,
)
from rest.config.predictive_query import pquery_runtime_options_or_none
from rest.config.predictive_query_v2 import (
    PQueryResource,
    PQueryValidationArgs,
)
from rest.config.table import ColStatsResponse
from rest.config.table_v2 import TableDefinition
from rest.deps.authentication import (
    RequestContext,
    visible_project_ids_or_none,
)
from rest.deps.context import _current_user_ctx
from rest.driver import graph_logic, pquery_logic
from rest.driver.data_snapshot_api_driver import get_graph_snapshot_info
from rest.driver.pquery_logic import column_cohorts_to_cube
from rest.driver.table_logic import (
    _get_col_stats_and_sample_rows,
    try_load_col_stats_from_per_job_dir,
)
from rest.utils.driver_common import generate_bp_stats_workflow_args
from rest.utils.pquery import forecasting_pass_through_plan
from rest.utils.table import ColStatsSet
from service_framework.deploy_env import send_slack_message
from service_framework.env_var import EnvVar
from statedb.config.common import (
    BatchPredictionJobID,
    CallToAction,
    CallToActionLevel,
    ColumnID,
    ConnectorID,
    DataSnapshot,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    GraphIDv2,
    GraphLink,
    JobSource,
    JobStatus,
    JobType,
    PredictiveQueryID,
    PredictiveQueryLink,
    ProjectID,
    TableID,
    TableName,
    TrainingJobID,
    to_graph_id,
    to_table_id,
)
from statedb.config.job_tag import (
    TAG_KEY_MODEL_ID,
    TAG_KEY_PARENT_JOB,
    TAG_KEY_PQUERY_ID,
    TAG_KEY_PQUERY_NAME,
    JobTagKey,
    JobTagValue,
)
from statedb.config.predictive_query import (
    FilterOverrideArgs,
    PQueryDefinition,
)
from statedb.config.table import TableDataVersion
from statedb.config.timestamp import ResourceTimestamps, now_utc
from statedb.db.services import StateServices
from statedb.ml.jobs import BatchPredictionJobDBEntry
from statedb.rest.connector_authorization import (
    CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS,
    CONNECTOR_GRAPH_ACTIONS,
    CONNECTOR_OUTPUT_ACTIONS,
)
from statedb.utils.errors import (
    BatchPredictionDoesNotExistError,
    ConnectorDoesNotExistError,
    IdentityForbiddenError,
    ProjectForbiddenError,
    ProjectNotFoundError,
    TrainingJobNotInProjectError,
)
from temporalio_commons.auto_rpc_exception import RpcError

log = logging.getLogger(__name__)

# The maximum number of batch prediction jobs to keep in the database.
# If there are more than this number of jobs, the oldest jobs will be deleted
# in batches.
DELETE_OLD_BP_JOBS = EnvVar(
    "DELETE_OLD_BP_JOBS",
    bool,
    default_value=False,
    description=(
        "If True Delete old bp jobs when number of BP jobs crosses a limit"),
)
MAX_BP_JOBS_PER_WORKFLOW = EnvVar(
    "MAX_BP_JOBS_PER_WORKFLOW",
    int,
    default_value=500,
    description=(
        "Delete BP jobs if total number of BP jobs per workflow exceeds this"),
)

# We limit the number of links to predict, to prevent the link prediction job
# from being too slow.
MAX_LINKS_TO_PREDICT = 500

# Maximum no.of supported parallel workers for a batch prediction job.
MAX_NUM_PARALLEL_WORKERS_BP = 32

# The maximum number of bp jobs to fetch from the ML backend
# TODO(jinu): This is a temporary fix to prevent the auto-rpc request from
# timing out when there are too many bp jobs. We should remove this limit
# once we have a pagination mechanism in place.
LIMIT_BP_JOBS_TO_FETCH = 500
LIMIT_BP_JOBS_PER_PAGE = 200


def _validate_trained_model_artifact_support(
    request: BatchPredictionRequest,
    task_type: TaskType,
    training_job_id: TrainingJobID,
) -> None:
    """Reject outputs the completed model cannot produce.

    ID-GNN link-prediction models have no embedding hook. Accepting an
    EMBEDDINGS destination and silently omitting it later violates the output
    contract, so validate the actual trained model before creating the job.
    """
    if (not task_type.is_link_pred
            or not any(output.artifact_type == PredictionArtifactType.EMBEDDINGS
                       for output in request.outputs)):
        return

    from kumo_core.training.config import LinkPredModuleType, TrainerConfig

    from io_utils.json import load_json
    from rest.backends.ml_backend.data_util import (
        get_job_base_dir,
        get_job_dir,
    )

    train_dir = get_job_dir(get_job_base_dir(), training_job_id)
    model_dir = HomeDirManager(train_dir).get_best_trainer_dir('')
    trainer_config = load_json(os.path.join(model_dir, 'config.json'),
                               TrainerConfig)
    module_kwargs = getattr(trainer_config, 'module_kwargs', None)
    if getattr(module_kwargs, 'module', None) == LinkPredModuleType.ID:
        raise ValueError(
            'Embeddings are not available for ID-GNN link-prediction models. '
            'Request predictions instead, or train a two-tower, shallow-RHS, '
            'or hybrid link-prediction model.')


@dataclass(frozen=True)
class _WarehouseTablePath:
    connector_id: ConnectorID
    table: str


@dataclass(frozen=True)
class _FileOutputPath:
    connector_id: Optional[ConnectorID]
    coordinate: str


@dataclass
class _ConflictInfo:
    is_other_job: bool
    bp_id: Optional[BatchPredictionJobID]


@dataclass
class _PendingOutputs:
    warehouse_tables: Dict[_WarehouseTablePath,
                           _ConflictInfo] = field(default_factory=dict)
    # file_paths stores both S3 (s3://) and GCS (gs://) output paths
    file_paths: Dict[_FileOutputPath,
                     _ConflictInfo] = field(default_factory=dict)


def _file_output_path(output) -> _FileOutputPath:
    coordinate = output.canonical_file_path
    connector_id = (None if urlsplit(coordinate).scheme else ConnectorID(
        output.connector_id))
    return _FileOutputPath(connector_id, coordinate)


@dataclass(frozen=True)
class _DecoupledPredictionGraphContext:
    """Immutable graph identity recorded by a gen-pred-table job."""
    graph_data_compute_input: GraphDataComputeInput
    graph_version_info: GraphDataVersionInfo
    graph_data_version: str


class BatchPredictionJobLogic:
    def __init__(
        self,
        ml_job_backend: MLJobBackend,
        data_backend_ctx: DataBackendContext,
        state_services: Optional[StateServices] = None,
    ):
        self._ml_backend = ml_job_backend
        self._data_backend_ctx = data_backend_ctx
        self._state_services = state_services

    def _ml_job_service_or_none(self):
        if self._state_services is None:
            return None
        return self._state_services.ml_jobs

    def _require_ml_job_service(self):
        ml_job_service = self._ml_job_service_or_none()
        if ml_job_service is None:
            raise RuntimeError(
                "BatchPredictionJobLogic requires StateServices.ml_jobs for "
                "batch prediction metadata mutations.")
        return ml_job_service

    def _pquery_service_or_none(self):
        if self._state_services is None:
            return None
        return self._state_services.pqueries

    def _require_pquery_service(self):
        pquery_service = self._pquery_service_or_none()
        if pquery_service is None:
            raise RuntimeError(
                "BatchPredictionJobLogic requires StateServices.pqueries for "
                "batch prediction pquery resolution.")
        return pquery_service

    async def _pqueries_by_id(
        self,
        pq_ids: Iterable[PredictiveQueryID],
    ) -> Dict[PredictiveQueryID, Optional[Any]]:
        pq_ids = list(dict.fromkeys(pq_ids))
        if not pq_ids:
            return {}
        pquery_service = self._require_pquery_service()

        definitions_by_id = await pquery_service.get_definitions_by_ids(pq_ids)
        return {pq_id: definitions_by_id.get(pq_id) for pq_id in pq_ids}

    async def _state_batch_prediction_entries(
        self,
        ctx: RequestContext,
        ids: Optional[Iterable[BatchPredictionJobID]] = None,
    ) -> List[BatchPredictionJobDBEntry]:
        ml_job_service = self._require_ml_job_service()

        if ids is None:
            ids = [
                BatchPredictionJobID(id)
                for id in await ml_job_service.list_job_ids_by_type(
                    JobType.BATCH_PREDICTION_JOB)
            ]

        return await ml_job_service.get_batch_prediction_rows(
            ids,
            project_ids=visible_project_ids_or_none(ctx.user_authorization),
        )

    async def _get_batch_prediction_entry_if_exists(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
    ) -> Optional[BatchPredictionJobDBEntry]:
        state_entries = await self._state_batch_prediction_entries(ctx, [id])
        return state_entries[0] if state_entries else None

    async def create_batch_prediction_job_v2(
        self,
        ctx: RequestContext,
        request: BatchPredictionRequest,
        pq_resource: PQueryResource,
        pq_def_fpid: FPID,
        task_type: TaskType,
        query_type: QueryType,
        training_job_id: TrainingJobID,
        *,
        predictive_query_id: PredictiveQueryID,
        filter_override_args: Optional[FilterOverrideArgs] = None,
        top_k: Optional[int] = None,
        generate_pred_table_job_id: Optional[
            GeneratePredictionTableJobID] = None,
        custom_pred_table_source: Optional[
            jobs_api.PredictionTableSource] = None,
        custom_pred_table_connector: Optional[ConnectorExecutionRef] = None,
        graph_snapshot_id: Optional[str] = None,
        job_source: Optional[JobSource] = None,
        custom_tags: Optional[List[str]] = None,
        user: Optional[str] = None,
        project_id: ProjectID,
    ) -> BatchPredictionJobID:
        """Drives BP launch from a `PQueryResource` + pre-frozen `pq_def_fpid`
        directly off a `BatchPredictionRequest`."""
        id = BatchPredictionJobID.make_random_id()

        training_job = await self._ml_backend.get_training_job(
            ctx, training_job_id)
        assert training_job
        assert training_job.execution.status == JobStatus.DONE

        if training_job.project_id != project_id:
            raise TrainingJobNotInProjectError(
                f"Training job with id {training_job.id} must belong to "
                "the same project as the initiated batch prediction job.")
        await asyncio.to_thread(
            _validate_trained_model_artifact_support,
            request,
            task_type,
            training_job_id,
        )

        # Compute the threshold without mutating the request. Multilabel
        # classification uses the same sigmoid-threshold output contract as
        # binary classification.
        requested_threshold = (
            request.predict_options.binary_classification_threshold)
        effective_threshold = resolve_classification_threshold(
            task_type,
            requested_threshold,
        )
        if requested_threshold is None and effective_threshold is not None:
            log.warning(
                'Classification threshold is set to 0.5 because it was missing.'
            )
        elif requested_threshold is not None and effective_threshold is None:
            log.warning(
                'binary_classification_threshold is only available for '
                'binary and multilabel classification queries. Reset to None.')

        # Compute effective num_classes_to_return (handles top_k default and
        # only_embeddings override).
        raw_num_classes = request.predict_options.num_classes_to_return
        effective_num_classes_to_return: Optional[int] = (
            raw_num_classes if raw_num_classes is not None else top_k)
        if request.only_embeddings:
            effective_num_classes_to_return = None

        gen_tt_job, train_table_options = await self._compute_bp_train_options(
            training_job, task_type, query_type, project_id=project_id)

        graph_id, graph_data_ver = await self._resolve_bp_graph_id(
            pq_resource, graph_snapshot_id, predictive_query_id)

        config = request.to_config(
            predictive_query_id=predictive_query_id,
            training_job_id=training_job_id,
            graph_id=graph_id,
            task_type=task_type,
            filter_override_args=filter_override_args,
            effective_num_classes_to_return=effective_num_classes_to_return,
            effective_threshold=effective_threshold,
        )
        config.graph_data_version = graph_data_ver
        self._apply_train_table_params_to_bp_config(config, gen_tt_job,
                                                    training_job)

        if custom_pred_table_source and generate_pred_table_job_id:
            raise ValueError(
                "Custom prediction table source and "
                "generate-prediction-table job ID cannot be specified at "
                "the same time.")
        if ((custom_pred_table_source is None) != (custom_pred_table_connector
                                                   is None)):
            raise ValueError(
                'Custom prediction source and connector reference must be '
                'provided together')
        if custom_pred_table_source is None:
            generate_pred_table_job_id = (
                generate_pred_table_job_id
                or GeneratePredictionTableJobID.as_child_of(id))
        config.generate_pred_table_job_id = generate_pred_table_job_id
        config.custom_pred_table_source = custom_pred_table_source

        try:
            # This BP may reuse a separate prediction-table generation ("gen")
            # job; BP must score on the graph version that gen job used, not the
            # current graph (#29507). The gen job may still be running now (the
            # SDK submits BP without waiting), so reading its version here would
            # fail -- leave these None and let the BP workflow read it once the
            # gen job finishes.
            graph_data_compute_input = None
            graph_version_info = None

            job_input = await self.make_batch_prediction_job_input_v2(
                pq_resource, pq_def_fpid, id, config,
                train_table_options=train_table_options,
                custom_pred_table_connector=custom_pred_table_connector,
                graph_data_compute_input=graph_data_compute_input,
                graph_version_info=graph_version_info, project_id=project_id)
        except Exception as ex:
            if isinstance(ex, ValueError):
                raise
            raise ValueError(
                f"Invalid batch prediction job input due to {type(ex).__name__}: {ex}"
            ) from ex

        # Submit must write the full row before the workflow starts; a separate
        # REST-metadata write afterwards left an early GET racing a 404.
        await self._ml_backend.start_batch_prediction(
            job_input,
            project_id=project_id,
            job_source=job_source,
            user=user,
            batch_prediction_workflow_id=request.batch_prediction_workflow_id,
        )

        await self._apply_bp_job_tags(id, config, custom_tags, training_job_id)

        if DELETE_OLD_BP_JOBS.parse_value():
            await self._delete_old_bp_jobs(
                ctx,
                request.batch_prediction_workflow_id,
                project_id=project_id,
            )

        return id

    async def _compute_bp_train_options(
        self,
        training_job,
        task_type: TaskType,
        query_type: QueryType,
        *,
        project_id: ProjectID,
    ) -> Tuple[Optional[object], Optional[TrainingTableGenerationPlan]]:
        """Look up the training job's gen-train-table job and carry its
        train-table options over to the BP query config. Returns
        (gen_tt_job, train_table_options)."""
        if not training_job.generate_train_table_job_id:
            return None, None
        gen_tt_jobs = await self._ml_backend.get_generate_train_table_jobs(
            job_ids={
                GenerateTrainTableJobID(
                    training_job.generate_train_table_job_id)
            },
            project_ids={project_id},
        )
        if len(gen_tt_jobs) != 1:
            return None, None
        gen_tt_job = gen_tt_jobs[0]
        return gen_tt_job, forecasting_pass_through_plan(
            task_type,
            query_type,
            forecast_length=gen_tt_job.config.plan.forecast_length,
            lag_timesteps=gen_tt_job.config.plan.lag_timesteps,
            year_over_year=gen_tt_job.config.plan.year_over_year,
        )

    async def _resolve_bp_graph_id(
        self,
        pq_resource: PQueryResource,
        graph_snapshot_id: Optional[str],
        pquery_id_for_log: str,
    ) -> Tuple[GraphIDv2, Optional[str]]:
        """Return the graph_id and graph_data_version to use for the BP job.

        Honors a `graph_snapshot_id` override when its graph differs from
        the pquery's own graph; otherwise returns the pquery's graph_id.
        """
        pq_graph_id = GraphIDv2.compose(fpid=pq_resource.graph.fpid)
        if not graph_snapshot_id:
            return pq_graph_id, None
        override_graph_id, graph_ver_info = await self._get_graph_snapshot_info(
            graph_snapshot_id)
        if override_graph_id != pq_graph_id:
            log.warning(f"Launching batch prediction with snapshot override: "
                        f"Graph data snapshot id={graph_snapshot_id} "
                        f"overrides the graph defined in predictive query "
                        f"{pquery_id_for_log}")
            return override_graph_id, graph_ver_info.graph_data_version
        return pq_graph_id, graph_ver_info.graph_data_version

    async def _get_graph_snapshot_info(
        self,
        graph_snapshot_id: str,
    ) -> Tuple[GraphIDv2, GraphDataVersionInfo]:
        return await get_graph_snapshot_info(graph_snapshot_id)

    async def _gen_pred_table_graph_context(
        self,
        gen_pred_table_job_id: GeneratePredictionTableJobID,
        *,
        project_id: ProjectID,
    ) -> _DecoupledPredictionGraphContext:
        """Return a decoupled gen-pred-table job's graph identity.

        Such a job may resolve a forked compute graph (e.g. graph-snapshot
        override) and write its staging table under a ``(graph_key,
        graph_data_version)`` pair. The BP job must reuse BOTH values so
        materialize reads from where the gen job wrote it. The generator
        workflow re-persists this input after graph refresh. If the input has
        not been written yet, fixed-snapshot jobs can fall back to StateDB
        graph metadata; incomplete saved inputs are not safe to reuse.
        """
        ml_job_service = self._ml_job_service_or_none()
        if ml_job_service is None:
            raise RuntimeError(
                "Generate-prediction-table lookup requires StateDB")
        matching_jobs = await ml_job_service.get_generate_pred_table_jobs(
            [gen_pred_table_job_id], project_ids={project_id})
        if not matching_jobs:
            raise ValueError('Cannot reuse generate-prediction-table job '
                             f'{gen_pred_table_job_id}: job not found.')
        gen_job = matching_jobs[0]

        gen_job_home_dir = await self._ml_backend.get_job_home_dir(
            gen_pred_table_job_id)
        input_path = HomeDirManager(gen_job_home_dir).prediction_job_input_file

        async def fallback_or_raise(
            reason: str, ) -> _DecoupledPredictionGraphContext:
            graph_context = await self._gen_pred_table_graph_context_from_state(
                gen_pred_table_job_id, gen_job=gen_job)
            if graph_context is not None:
                log.warning(
                    'Saved input %r for decoupled generate-prediction-table '
                    'job %s is %s; falling back to StateDB graph metadata.',
                    input_path, gen_pred_table_job_id, reason)
                return graph_context
            raise ValueError(
                'Cannot reuse decoupled generate-prediction-table job '
                f'{gen_pred_table_job_id}: saved input {input_path!r} is '
                f'{reason}. Regenerate the prediction table before starting '
                'batch prediction.')

        try:
            gen_job_input = await load_json_async(
                input_path,
                BatchPredictionJobInput,
                raise_on_missing=True,
            )
        except FileNotFoundError:
            return await fallback_or_raise('missing')
        if gen_job_input is None:
            return await fallback_or_raise('empty')
        graph_data_compute_input = gen_job_input.graph_data_compute_input
        graph_version_info = gen_job_input.graph_version_info
        if graph_data_compute_input is None or graph_version_info is None:
            raise ValueError(
                'Cannot reuse decoupled generate-prediction-table job '
                f'{gen_pred_table_job_id}: its saved input lacks the complete '
                '(graph_key, graph_data_version) identity. Regenerate the '
                'prediction table before starting batch prediction.')
        return _DecoupledPredictionGraphContext(
            graph_data_compute_input=graph_data_compute_input,
            graph_version_info=graph_version_info,
            graph_data_version=graph_version_info.graph_data_version,
        )

    async def _gen_pred_table_graph_context_from_state(
        self,
        gen_pred_table_job_id: GeneratePredictionTableJobID,
        *,
        gen_job: Any,
    ) -> Optional[_DecoupledPredictionGraphContext]:
        """Best-effort fallback when a gen-pred-table input file has not
        been persisted yet.

        A fixed-snapshot decoupled generator records enough identity in
        StateDB to recover the compute graph and graph version info from the
        graph cache. If any piece is missing, callers keep the original
        saved-input error.
        """
        if gen_job is None or gen_job.config is None:
            return None

        graph_data_version = gen_job.config.graph_data_version
        if graph_data_version is None:
            return None

        graph_id = to_graph_id(gen_job.config.graph_id)
        if not isinstance(graph_id, GraphIDv2):
            return None

        graph_data_compute_input = await _get_graph_data_compute_input_v2(
            graph_id,
            gen_job.project_id,
            self._state_services,
        )
        cache = get_active_cache()
        graph_version_info = (await load_many_json.coroutine(
            [cache.graph_cache.version_info_path(graph_id, graph_data_version)],
            GraphDataVersionInfo,
        ))[0]
        if graph_version_info is None:
            return None

        return _DecoupledPredictionGraphContext(
            graph_data_compute_input=graph_data_compute_input,
            graph_version_info=graph_version_info,
            graph_data_version=graph_version_info.graph_data_version,
        )

    def _apply_train_table_params_to_bp_config(
        self,
        config: BatchPredictionJobConfig,
        gen_tt_job,
        training_job,
    ) -> None:
        """Copy `lag_timesteps`, `year_over_year`, and `forecast_length` onto
        the BP config from the gen-train-table job's plan (preferred) or
        the training job's advanced options (fallback)."""
        if gen_tt_job is not None:
            plan = gen_tt_job.config.plan
            config.lag_timesteps = inferred_or_missing_type_to_none(
                plan.lag_timesteps)
            config.year_over_year = inferred_or_missing_type_to_false(
                plan.year_over_year)
            config.forecast_length = inferred_or_missing_type_to_none(
                plan.forecast_length)
        elif training_job.advanced_options is not None:
            ttg = training_job.advanced_options.training_table_generation
            config.lag_timesteps = inferred_or_missing_type_to_none(
                ttg.lag_timesteps)
            config.year_over_year = inferred_or_missing_type_to_false(
                ttg.year_over_year)
            config.forecast_length = inferred_or_missing_type_to_none(
                ttg.forecast_length)

    async def _apply_bp_job_tags(
        self,
        id: BatchPredictionJobID,
        config: BatchPredictionJobConfig,
        custom_tags: Optional[List[str]],
        training_job_id: TrainingJobID,
    ) -> None:
        """Apply pquery / model / parent-job tags to the BP job (and its
        implicit gen-pred-table child job, if any)."""
        job_tags: Dict[JobTagKey, JobTagValue] = {}
        if custom_tags is not None:
            job_tags.update(
                {JobTagKey(tag): JobTagValue(tag)
                 for tag in custom_tags})
        job_tags[TAG_KEY_PQUERY_NAME] = JobTagValue.of_pquery_name(
            config.predictive_query_id)
        job_tags[TAG_KEY_PQUERY_ID] = JobTagValue.of_pquery_id(
            config.pquery_def_fpid)
        job_tags[TAG_KEY_MODEL_ID] = JobTagValue(training_job_id)
        await self._ml_backend.apply_job_tags(JobType.BATCH_PREDICTION_JOB, id,
                                              dict(job_tags))
        pred_table_job_id = config.generate_pred_table_job_id
        if (pred_table_job_id and pred_table_job_id
                == GeneratePredictionTableJobID.as_child_of(id)):
            del job_tags[TAG_KEY_MODEL_ID]
            job_tags[TAG_KEY_PARENT_JOB] = JobTagValue(id)
            await self._ml_backend.apply_job_tags(
                JobType.GENERATE_PREDICTION_TABLE_JOB, pred_table_job_id,
                job_tags)

    async def _delete_old_bp_jobs(
        self,
        ctx: RequestContext,
        workflow_id: Optional[str] = None,
        *,
        project_id: ProjectID,
    ) -> None:
        """Per `worfklow_id` delete old BP jobs if the number of BP jobs
        exceeds the limit `MAX_BP_JOBS_PER_WORKFLOW`. If `workflow_id` is None,
        only consider jobs without a workflow id."""
        ml_job_service = self._require_ml_job_service()
        max_bp_jobs = MAX_BP_JOBS_PER_WORKFLOW.parse_value()
        job_ids_to_delete = set(
            await ml_job_service.list_batch_prediction_prune_candidate_ids(
                workflow_id,
                project_ids={project_id},
                keep_most_recent=max_bp_jobs,
            ))
        if job_ids_to_delete:
            await self.delete_many_batch_prediction_jobs(ctx, job_ids_to_delete)
            log.info(f"Deleted {len(job_ids_to_delete)} BP jobs, from "
                     f"workflow {workflow_id}, "
                     "to prevent overloading the system.")

    async def get_batch_prediction_job(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
    ) -> Optional[BatchPredictionJobResponse]:
        jobs = await self.list_batch_prediction_jobs(ctx, job_ids={id})
        if len(jobs) == 0:
            return None
        assert len(jobs) == 1
        return jobs[0]

    async def delete_batch_prediction(self, ctx: RequestContext,
                                      id: BatchPredictionJobID) -> bool:
        r"""Delete a batch prediction job with its metadata and tags.
        Return True if the job was deleted, False if no such job is visible
        to the caller."""
        ml_job_service = self._ml_job_service_or_none()
        if ml_job_service is None:
            raise RuntimeError(
                "BatchPredictionJobLogic requires StateServices.ml_jobs for "
                "batch prediction deletion.")
        try:
            deleted = await self._ml_backend.delete_batch_prediction(ctx, id)
        except RpcError as rpc_error:
            if rpc_error.error_type == ValueError.__name__:
                raise ValueError(rpc_error.message)
            raise
        if not deleted:
            return False
        await ml_job_service.delete_batch_prediction_metadata(id)
        await ml_job_service.delete_job_tags(JobType.BATCH_PREDICTION_JOB, id)
        return True

    async def delete_many_batch_prediction_jobs(
            self, ctx: RequestContext, ids: Set[BatchPredictionJobID]) -> bool:
        r"""Delete multiple batch prediction jobs.
        Return True if all jobs are deleted, False otherwise."""
        jobs_wont_delete = set()
        jobs_to_delete = set()
        ml_job_service = self._ml_job_service_or_none()
        if ml_job_service is not None:
            state_entries = await self._state_batch_prediction_entries(ctx, ids)
            assert state_entries is not None
            found_ids = {entry.id for entry in state_entries}
            jobs_wont_delete.update(ids - found_ids)
            jobs_to_delete.update(found_ids)
        else:
            raise RuntimeError(
                "BatchPredictionJobLogic requires StateServices.ml_jobs for "
                "batch prediction deletion.")

        # First delete from the ML backend
        deleted_from_ml_db = set()
        try:
            out = await self._ml_backend.delete_many_batch_prediction_jobs(
                ctx, jobs_to_delete)
            deleted_from_ml_db = out
        except RpcError as rpc_error:
            if rpc_error.error_type == ValueError.__name__:
                asyncio.create_task(
                    send_slack_message.coroutine(
                        f"Error deleting batch prediction jobs from StateDB "
                        f"Error: {rpc_error.message}"))
            else:
                raise

        # Then delete from the metadata DB. But only delete the jobs that are
        # successfully deleted from the ML backend.
        jobs_wont_delete.update(jobs_to_delete - deleted_from_ml_db)
        jobs_to_delete = deleted_from_ml_db
        await ml_job_service.delete_batch_prediction_metadata_and_tags(
            jobs_to_delete)
        return not jobs_wont_delete

    def list_batch_prediction_job_entries(
        self,
        ctx: RequestContext,
        job_entry_predicate: Callable[[BatchPredictionJobDBEntry], bool],
        sort_by_create_time=False,
    ) -> List[BatchPredictionJobDBEntry]:
        del ctx, job_entry_predicate, sort_by_create_time
        raise RuntimeError(
            "Synchronous batch prediction metadata listing is no longer backed "
            "by legacy REST metadata. Use _list_batch_prediction_job_entries "
            "with StateServices.ml_jobs.")

    async def _list_batch_prediction_job_entries(
        self,
        ctx: RequestContext,
        *,
        model_id: Optional[TrainingJobID] = None,
        pquery_id: Optional[PredictiveQueryID] = None,
        job_status: Optional[JobStatus] = None,
        job_ids: Optional[Set[BatchPredictionJobID]] = None,
        sort_by_create_time: bool = False,
        sort_desc: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
        match_pquery_def_fpid: bool = True,
        project_id: Optional[ProjectID] = None,
    ) -> List[BatchPredictionJobDBEntry]:
        project_ids = ({project_id} if project_id is not None else
                       visible_project_ids_or_none(ctx.user_authorization))
        rows = await self._require_ml_job_service().list_batch_prediction_rows(
            project_ids=project_ids,
            model_id=model_id,
            pquery_id=pquery_id,
            job_ids=job_ids,
            status=job_status,
            sort_created_at="desc" if sort_desc else "asc",
            limit=limit,
            offset=offset,
            match_pquery_def_fpid=match_pquery_def_fpid,
        )
        result = rows
        if sort_by_create_time:
            result.sort(
                key=lambda job: job.timestamps.created_at,
                reverse=sort_desc,
            )
        return result

    @staticmethod
    def _current_pred_job_ids_from_entries(
        pquery_ids: List[PredictiveQueryID],
        jobs: Iterable[BatchPredictionJobDBEntry],
    ) -> List[Optional[BatchPredictionJobID]]:
        pquery_id_set = set(pquery_ids)
        id_map: Dict[PredictiveQueryID,
                     List[BatchPredictionJobDBEntry]] = defaultdict(list)
        for job in jobs:
            pqid = job.config.predictive_query_id
            if pqid not in pquery_id_set:
                continue
            id_map[pqid].append(job)
        result: List[Optional[BatchPredictionJobID]] = []
        for pqid in pquery_ids:
            if pqid not in id_map:
                result.append(None)
                continue
            jobs_for_pquery = id_map[pqid]
            job = max(jobs_for_pquery,
                      key=lambda job: job.timestamps.created_at)
            result.append(job.id)
        return result

    def get_current_pred_job_ids(
        self,
        ctx: RequestContext,
        pquery_ids: List[PredictiveQueryID],
    ) -> List[Optional[BatchPredictionJobID]]:
        """Return the most recently launched batch prediction job for each
        predictive query.
        """
        del ctx, pquery_ids
        raise RuntimeError(
            "Synchronous current batch prediction lookup is no longer backed "
            "by legacy REST metadata. Use get_current_pred_job_ids_v2 with "
            "StateServices.ml_jobs.")

    async def get_current_pred_job_ids_v2(
        self,
        ctx: RequestContext,
        pquery_ids: List[PredictiveQueryID],
    ) -> List[Optional[BatchPredictionJobID]]:
        id_map = await self._require_ml_job_service(
        ).get_current_batch_prediction_job_ids(
            pquery_ids,
            project_ids=visible_project_ids_or_none(ctx.user_authorization),
        )
        return [id_map.get(pquery_id) for pquery_id in pquery_ids]

    async def get_current_pred_job_id_v2(
        self,
        ctx: RequestContext,
        pquery_id: PredictiveQueryID,
    ) -> Optional[BatchPredictionJobID]:
        return (await self.get_current_pred_job_ids_v2(ctx, [pquery_id]))[0]

    @staticmethod
    def _batch_prediction_links(
        config: BatchPredictionJobConfig,
        pq: Optional[Any],
        *,
        allow_missing_pquery: bool = False,
    ) -> Tuple[PredictiveQueryLink, GraphLink]:
        pq_id = config.predictive_query_id
        is_from_sdk = pq_id is not None and (pq_id.is_public
                                             or pq_id.name == "NA")
        if is_from_sdk and config.pquery_def_fpid is not None:
            public_pq_id = PredictiveQueryID.compose(
                fpid=config.pquery_def_fpid)
            graph_id = config.graph_id
            assert graph_id is not None
            return (
                PredictiveQueryLink(name=public_pq_id, id=public_pq_id),
                GraphLink(name=graph_id, id=graph_id),
            )

        if isinstance(pq, PQueryDefinition):
            resolved_pq_id = PredictiveQueryID.compose(pq.pquery_name, pq.fpid)
            graph_id = GraphIDv2.compose(fpid=pq.graph_fpid)
            return (
                PredictiveQueryLink(
                    name=pq.pquery_name or resolved_pq_id,
                    id=resolved_pq_id,
                ),
                GraphLink(name=graph_id, id=graph_id),
            )

        if pq is not None:
            return pq.to_pq_link(), pq.get_graph_link()

        if allow_missing_pquery:
            graph_id = config.graph_id
            assert isinstance(graph_id, GraphIDv2)
            assert graph_id.fpid is not None
            assert config.graph_data_version is not None
            snapshot_id = DataSnapshot.GRAPH.format_id(
                graph_id.fpid, config.graph_data_version)
            return (
                PredictiveQueryLink(
                    name=config.training_job_id,
                    training_job_id=config.training_job_id,
                ),
                GraphLink(
                    name=str(snapshot_id),
                    id=graph_id,
                    snapshot_id=snapshot_id,
                ),
            )

        assert pq is not None, f'Query {pq} not found'

    async def list_batch_prediction_jobs(
        self,
        ctx: RequestContext,
        *,
        model_id: Optional[TrainingJobID] = None,
        pquery_id: Optional[PredictiveQueryID] = None,
        job_status: Optional[JobStatus] = None,
        job_ids: Optional[Set[BatchPredictionJobID]] = None,
        limit: Optional[int] = None,
        project_id: Optional[ProjectID] = None,
    ) -> List[BatchPredictionJobResponse]:
        """Results are guaranteed to be sorted by created_at ascending (oldest
        elements first).
        Note: This method will only return the latest LIMIT_BP_JOBS_TO_FETCH
        jobs.
        """
        # Temporal fix to prevent fetching too many BP jobs, leading to AutoRpc
        # timeout, see #12991
        if not limit:
            limit = LIMIT_BP_JOBS_TO_FETCH
        limit = min(limit, LIMIT_BP_JOBS_TO_FETCH)

        jobs = await self._list_batch_prediction_job_entries(
            ctx,
            model_id=model_id,
            pquery_id=pquery_id,
            job_status=job_status,
            job_ids=job_ids,
            sort_by_create_time=True,
            sort_desc=True,
            limit=limit,
            project_id=project_id,
        )

        jobs_by_id = {job.id: job for job in jobs}
        job_outputs = await self._ml_backend.get_batch_prediction_jobs(
            ctx, set(job.id for job in jobs), job_status, limit)

        pqs_by_id = await self._pqueries_by_id(pq_ids=[
            jobs_by_id[job_id].config.predictive_query_id
            for job_id in job_outputs.keys()
        ])

        result: List[BatchPredictionJobResponse] = []
        for job_id, job_output in job_outputs.items():
            job = jobs_by_id[job_id]
            pq = pqs_by_id[job.config.predictive_query_id]
            pq_link, graph_link = self._batch_prediction_links(job.config, pq)

            result.append(
                BatchPredictionJobResponse(
                    id=job.id,
                    config=job.config,
                    timestamps=ResourceTimestamps(
                        created_at=job.timestamps.created_at,
                        updated_at=(job_output.last_updated_at
                                    or job.timestamps.created_at),
                    ),
                    predictive_query_link=pq_link,
                    execution=job_output,
                    graph_link=graph_link,
                    project_id=job.project_id,
                    description=job.description,
                    batch_prediction_workflow_id=job.
                    batch_prediction_workflow_id,
                    job_source=job.source,
                    user=job.user,
                    kumo_version=job.kumo_version,
                ))

        return sorted(result, key=lambda job: job.timestamps.created_at)

    async def list_paginated_batch_prediction_jobs(
        self,
        ctx: RequestContext,
        pagination_args: PaginationArgs,
        model_id: Optional[TrainingJobID] = None,
        pquery_id: Optional[PredictiveQueryID] = None,
        job_status: Optional[JobStatus] = None,
        ids: Optional[Set[BatchPredictionJobID]] = None,
    ) -> PaginatedBatchPredictionJobResponse:
        """
        Implement page-based pagination to list batch prediction
        jobs from most to least recent jobs.
        Args:
            ctx (RequestContext): Authentication context.
            pagination_args (PaginationArgs): The page (number) and
            pageSize (number of pages) to return.
            - 1 <= page < page_limit (calculated below).
            - 1 <= pageSize < 200.
            pquery_id (PredictiveQueryID).
            ids (BatchPredictionJobID).
        Returns:
            PaginatedBatchPredictionJobResponse
        """
        num_jobs_to_get: int = LIMIT_BP_JOBS_PER_PAGE
        if (pagination_args.page_size > LIMIT_BP_JOBS_PER_PAGE
                or pagination_args.page_size < 1):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="page_size has to be a number between 1 and 200.",
            )
        else:
            num_jobs_to_get = pagination_args.page_size

        total_jobs = await self._require_ml_job_service(
        ).count_batch_prediction_rows(
            project_ids=visible_project_ids_or_none(ctx.user_authorization),
            model_id=model_id,
            pquery_id=pquery_id,
            job_ids=ids,
            status=job_status,
            match_pquery_def_fpid=False,
        )
        page_limit: int = (total_jobs // num_jobs_to_get) + 1
        if pagination_args.page > page_limit:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"page cannot be larger than {page_limit}.",
            )
        elif pagination_args.page < 1:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="page cannot be smaller than 1.",
            )

        result: List[BatchPredictionJobResponse] = []

        # Fetch jobs from newest to oldest
        block: int = pagination_args.page - 1
        start: int = block * num_jobs_to_get
        jobs = await self._list_batch_prediction_job_entries(
            ctx=ctx,
            model_id=model_id,
            pquery_id=pquery_id,
            job_status=job_status,
            job_ids=ids,
            sort_by_create_time=True,
            sort_desc=True,
            limit=num_jobs_to_get,
            offset=start,
            match_pquery_def_fpid=False,
        )

        jobs_by_id = {job.id: job for job in jobs}
        job_outputs = await self._ml_backend.get_batch_prediction_jobs(
            ctx, set(job.id for job in jobs), job_status, num_jobs_to_get)

        pqs_by_id = await self._pqueries_by_id(pq_ids=[
            jobs_by_id[job_id].config.predictive_query_id
            for job_id in job_outputs.keys()
        ])

        for job_id, job_output in job_outputs.items():
            job = jobs_by_id[job_id]
            pq = pqs_by_id[job.config.predictive_query_id]
            predictive_query_link, graph_link = self._batch_prediction_links(
                job.config, pq, allow_missing_pquery=True)
            result.append(
                BatchPredictionJobResponse(
                    id=job.id,
                    config=job.config,
                    timestamps=ResourceTimestamps(
                        created_at=job.timestamps.created_at,
                        updated_at=(job_output.last_updated_at
                                    or job.timestamps.created_at),
                    ),
                    predictive_query_link=predictive_query_link,
                    execution=job_output,
                    graph_link=graph_link,
                    project_id=job.project_id,
                    description=job.description,
                    batch_prediction_workflow_id=job.
                    batch_prediction_workflow_id,
                    kumo_version=job.kumo_version,
                ))
        return PaginatedBatchPredictionJobResponse(
            jobs=sorted(result, key=lambda job: job.timestamps.created_at),
            current_page=pagination_args.page,
            number_of_pages=page_limit,
            total_number_of_jobs=total_jobs,
        )

    async def get_materialization_info(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
    ) -> Optional[MaterializationTaskInfo]:
        batch_pred = await self._get_batch_prediction_entry_if_exists(ctx, id)
        if not batch_pred:
            raise BatchPredictionDoesNotExistError(id)
        return await self._ml_backend.get_materialization_info(batch_pred.id)

    async def get_many_materialization_info(
        self,
        ctx: RequestContext,
        ids: List[BatchPredictionJobID],
    ) -> List[Optional[MaterializationTaskInfo]]:
        materialization_info = await self._ml_backend.get_many_materialization_info(
            ids)
        return materialization_info

    async def get_prediction_stat_args(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
        use_transformed_output: bool = False,
    ) -> Tuple[ConnectorSpec, TableDefinition, TableDataVersion, ColStatsSet]:
        bp_db_entry = await self._get_batch_prediction_entry_if_exists(ctx, id)
        if not bp_db_entry:
            raise BatchPredictionDoesNotExistError(id)
        bp_job_config = bp_db_entry.config
        if not bp_job_config:
            raise ValueError(f"Batch Prediction Job with ID {id} does not "
                             f"have an associated config object in the DB.")
        bp_job_home_dir = await self._ml_backend.get_job_home_dir(id)
        home_dir_mgr = HomeDirManager(bp_job_home_dir)
        bp_job_input = await load_json_async(
            home_dir_mgr.prediction_job_input_file, BatchPredictionJobInput)
        pquery_config = bp_job_input.predictive_query
        entity_col_name = pquery_config.get_bp_entity_col_name()

        return await generate_bp_stats_workflow_args(
            job_id=bp_db_entry.id,
            job_home_dir=await self._ml_backend.get_job_home_dir(id),
            task_type=bp_job_config.task_type,
            is_temporal=(pquery_config.query_type == QueryType.TEMPORAL),
            entity_col_name=entity_col_name,
            use_transformed_output=use_transformed_output,
            data_version_override=(str(bp_db_entry.id)
                                   if has_only_databricks_outputs(
                                       bp_job_input.config) else None),
        )

    async def get_prediction_stats(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
        use_transformed_output: bool = False,
    ) -> ColStatsResponse:
        (
            connector_spec,
            table_def,
            _,
            col_stats_set,
        ) = await self.get_prediction_stat_args(
            ctx=ctx, id=id, use_transformed_output=use_transformed_output)

        # Try to load stats from col_stats file first
        bp_job = await self._get_batch_prediction_entry_if_exists(ctx, id)
        if bp_job:
            job_home_dir = await self._ml_backend.get_job_home_dir(id)
            stats_response = await try_load_col_stats_from_per_job_dir(
                table_def=table_def,
                connector_spec=connector_spec,
                col_stats_set=col_stats_set,
                job_home_dir=job_home_dir,
            )
            if stats_response:
                log.info(
                    "we are able to retrieve stats from per job directory.")
                return stats_response

        log.info(
            "Falling back to previous logic to retrieve stats from data backend cache."
        )
        # Fall back to existing logic if stats couldn't be loaded from file
        return await _get_col_stats_and_sample_rows(
            table_def,
            connector_spec,
            col_stats_set,
            check_refresh_status=False,
        )

    async def validate_batch_prediction_args_v2(
        self,
        ctx: RequestContext,
        request: BatchPredictionRequest,
        allow_local_paths: bool,
        task_type: TaskType,
        query_type: QueryType,
        project_id: ProjectID,
    ) -> ValidationResponse:
        """v2 validator for BP jobs taking `BatchPredictionRequest` directly.

        Combines config-options, prediction-time, and outputs checks inline.
        """
        res = ValidationResponse()
        ctas: List[Optional[CallToAction]] = []

        # --- Config options (replaces _validate_config_options) ---
        num_workers = request.predict_options.num_workers
        if num_workers <= 0:
            ctas.append(
                CallToAction.crit(
                    f"num_workers must be a positive "
                    f"integer. Currently the supported values of "
                    f"parallelism are "
                    f"{set(range(1, MAX_NUM_PARALLEL_WORKERS_BP + 1))}"))
        if num_workers > MAX_NUM_PARALLEL_WORKERS_BP:
            ctas.append(
                CallToAction.crit(
                    "For efficiency reasons, our platform currently "
                    "supports a maximum job parallelism of "
                    f"{MAX_NUM_PARALLEL_WORKERS_BP} but you have requested "
                    f"{num_workers}. Please reduce the "
                    "parallelism of the batch prediction job and try again."))
        try:
            _validate_task_type_options_v2(task_type, request)
        except Exception as e:
            ctas.append(CallToAction.crit(str(e)))

        # --- Prediction time (replaces _validate_prediction_time) ---
        prediction_time = getattr(request.predict_options, 'prediction_time',
                                  None)
        if prediction_time:
            if prediction_time.tzinfo:
                res.errors.append(
                    ValidationError(
                        title="Timezone not allowed in prediction_time",
                        message=("The timezone included in the specified "
                                 "prediction time will be ignored. Please "
                                 "convert to UTC if data is timezone-aware, "
                                 "otherwise convert to the timezone of the "
                                 "data"),
                    ))
            if query_type == QueryType.STATIC:
                res.errors.append(
                    ValidationError(
                        title=("Prediction Anchor Time not "
                               "supported for static queries"),
                        message=("Your query is static, so it doesn't "
                                 "consider time when generating predictions, "
                                 "so specifying a prediction time is not "
                                 "allowed."),
                    ))

        # --- Outputs (replaces _validate_outputs) ---
        if request.outputs is not None:  # type: ignore
            pending_outputs = await self._get_outputs_of_running_jobs(
                ctx, request.outputs, project_id=project_id)
            for output in request.outputs:
                if (MetadataField.ANCHOR_TIMESTAMP in output.extra_fields
                        and query_type == QueryType.STATIC):
                    ctas.append(
                        CallToAction.crit(
                            "ANCHOR_TIMESTAMP is only supported for temporal "
                            "prediction tasks."))
                    continue
                if (isinstance(output, DatabricksPredictionOutput)
                        or isinstance(output, SnowflakePredictionOutput)
                        or isinstance(output, BigQueryPredictionOutput)):
                    if not output.table_name:
                        ctas.append(
                            CallToAction(
                                level=CallToActionLevel.CRITICAL,
                                message=("Table name is not specified for "
                                         "this batch prediction. Please "
                                         "specify a table name."),
                            ))
                        continue
                elif isinstance(output,
                                (S3PredictionOutput, GCSPredictionOutput,
                                 ADLSPredictionOutput)):
                    coordinate = output.file_path
                    if not coordinate:
                        ctas.append(
                            CallToAction(
                                level=CallToActionLevel.CRITICAL,
                                message=("File name is not specified for "
                                         "this batch prediction. Please "
                                         "specify a file name."),
                            ))
                        continue
                cta = await self._validate_output(
                    output,
                    allow_local_paths,
                    pending_outputs,
                    project_id,
                )
                ctas.append(cta)

        # --- Convert CTAs to ValidationResponse ---
        def _cta_to_validation(cta: CallToAction) -> ValidationResponse:
            if cta.level == CallToActionLevel.CRITICAL:
                return ValidationResponse(errors=[
                    ValidationError(title="Validation Error",
                                    message=cta.message)
                ])
            elif cta.level == CallToActionLevel.WARNING:
                return ValidationResponse(warnings=[
                    ValidationWarning(title="Validation Error",
                                      message=cta.message)
                ])
            return ValidationResponse()

        for cta in ctas:
            if cta:
                res.merge(_cta_to_validation(cta))
        return res

    async def cancel_batch_prediction_job(
        self,
        ctx: RequestContext,
        id: PredictiveQueryID,
        bp_job_id: Optional[BatchPredictionJobID] = None,
    ) -> bool:
        """
        Cancels the ongoing batch prediction job associated with bp_job_id.
        If bp_job_id is not supplied, the most recently triggered batch
        prediction job is cancelled.

        Args:
        ctx (RequestContext): API context associated with the request.
        id (PredictiveQueryID): ID of the predictive query under which the
            batch prediction job to be cancelled is running.
        bp_job_id (Optional[BatchPredictionJobID]): bp_job_id to cancel. If not
            supplied, the most recently triggered batch prediction job will be
            cancelled.
        """
        if bp_job_id:
            rows = await self._require_ml_job_service(
            ).list_batch_prediction_rows(
                project_ids=visible_project_ids_or_none(ctx.user_authorization),
                pquery_id=id,
                job_ids={bp_job_id},
                limit=1,
                match_pquery_def_fpid=False,
            )
            if not rows:
                raise ValueError(
                    f"No batch prediction job with id: {bp_job_id} found.")
            job_to_cancel = bp_job_id
        else:
            job_to_cancel = await self.get_current_pred_job_id_v2(ctx, id)
        if job_to_cancel is None:
            raise ValueError(
                f"No batch prediction job with pquery id: {id} found.")
        log.info(f"Cancelling batch predict job: {job_to_cancel}")
        return await self._ml_backend.cancel_batch_prediction_job(job_to_cancel)

    @measure_blocking_code
    async def make_batch_prediction_job_input_v2(
        self,
        pq_resource: PQueryResource,
        pq_def_fpid: FPID,
        job_id: BatchPredictionJobID,
        job_config: BatchPredictionJobConfig,
        *,
        train_table_options: Optional[TrainingTableGenerationPlan] = None,
        graph_data_compute_input: Optional["GraphDataComputeInput"] = None,
        graph_version_info: Optional[GraphDataVersionInfo] = None,
        custom_pred_table_connector: Optional[ConnectorExecutionRef] = None,
        project_id: ProjectID,
    ) -> BatchPredictionJobInput:
        """Thin wrapper that forwards a `PQueryResource` + `pq_def_fpid` to
        `_make_bp_input_v2`.

        Graph overrides pin the BP job to a decoupled generator's complete
        graph identity (see `_make_bp_input_v2`)."""
        await self._reauthorize_graph_compute_input(
            graph_data_compute_input,
            project_id,
        )
        output_connectors = await self._get_output_connectors(
            job_config.outputs, project_id)

        job_input = await _make_bp_input_v2(
            pq_resource,
            pq_def_fpid,
            job_id,
            job_config,
            self._data_backend_ctx,
            state_services=self._state_services,
            train_table_options=train_table_options,
            output_connectors=output_connectors,
            project_id=project_id,
            graph_data_compute_input=graph_data_compute_input,
            graph_version_info=graph_version_info,
            custom_pred_table_connector=custom_pred_table_connector,
        )

        return job_input

    async def _reauthorize_graph_compute_input(
        self,
        graph_input: Optional["GraphDataComputeInput"],
        project_id: ProjectID,
    ) -> None:
        """Recheck every project connector loaded from a prior job input."""
        if graph_input is None:
            return
        if not graph_input.connectors:
            return
        invalid = [
            connector_id
            for connector_id, transport in graph_input.connectors.items()
            if not isinstance(transport, ConnectorExecutionRef)
        ]
        if invalid:
            raise ValueError(
                'Graph inputs must contain only project connector references')
        if self._state_services is None:
            raise RuntimeError(
                'Batch prediction connector resolution requires StateServices')
        for connector_id, reference in graph_input.connectors.items():
            assert isinstance(reference, ConnectorExecutionRef)
            if reference.project_id != project_id:
                raise ValueError(
                    f'Connector {connector_id} was authorized for a '
                    'different project')
            graph_input.connectors[connector_id] = await (
                self._state_services.connectors.execution_ref_for_project(
                    _current_user_ctx.get(),
                    project_id,
                    connector_id,
                    CONNECTOR_GRAPH_ACTIONS,
                ))

    async def _get_output_connectors(
        self,
        outputs: List[PredictionOutputConfig],
        project_id: ProjectID,
    ) -> Dict[ConnectorID, ConnectorExecutionRef]:
        if self._state_services is None:
            raise RuntimeError(
                'Batch prediction connector resolution requires StateServices')
        expected_types: Dict[ConnectorID, ConnectorType] = {}
        for output in outputs:
            if isinstance(output, jobs_api.SnowflakePredictionOutput):
                expected_types[ConnectorID(
                    output.connector_id)] = ConnectorType.SNOWFLAKE
            elif isinstance(output, jobs_api.DatabricksPredictionOutput):
                expected_types[ConnectorID(
                    output.connector_id)] = ConnectorType.DATABRICKS
            elif isinstance(output, jobs_api.BigQueryPredictionOutput):
                expected_types[ConnectorID(
                    output.connector_id)] = ConnectorType.BIGQUERY
            elif isinstance(output, jobs_api.S3PredictionOutput):
                expected_types[ConnectorID(
                    output.connector_id)] = ConnectorType.S3
            elif isinstance(output, jobs_api.GCSPredictionOutput):
                expected_types[ConnectorID(
                    output.connector_id)] = ConnectorType.GCS
            elif isinstance(output, jobs_api.ADLSPredictionOutput):
                expected_types[ConnectorID(
                    output.connector_id)] = ConnectorType.ADLS

        if not expected_types:
            return {}
        if project_id is None:
            raise ValueError(
                'project_id is required for external prediction outputs')

        return {
            connector_id:
            await self._state_services.connectors.execution_ref_for_project(
                _current_user_ctx.get(),
                project_id,
                connector_id,
                (CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS
                 if expected_type == ConnectorType.DATABRICKS else
                 CONNECTOR_OUTPUT_ACTIONS),
                expected_type=expected_type,
            )
            for connector_id, expected_type in expected_types.items()
        }

    async def get_xai(
        self,
        home_mgr: HomeDirManager,
        tables: List[Table],
        entity: str,
        bfs_tree: Dict[TableID, List[TableID]],
    ) -> XAIResponse:
        resp = XAIResponse()

        # Load all XAI data in parallel.
        column_file_path = os.path.join(
            home_mgr.column_analysis_dir(TrainingStage.PRED),
            "column_analysis.json")
        column_analysis = await asyncio.gather(
            json.load_json_async(
                column_file_path,
                ColumnAnalysisOutput,
                default=None,
                raise_on_missing=False,
            ))

        root_table_id = None
        navigation_tree: Dict[str, List[str]] = defaultdict(list)
        col_stypes: Dict[str, Optional[Stype]] = {}
        for table in tables:
            table_id = to_table_id(table.name)
            resp.tables[table_id] = {}
            # Populate the root_table if we've found the entity table
            if f"{table_id.table_name}.{table.pkey_name}" == entity:
                root_table_id = table_id

            for col in table.cols:
                navigation_tree[table_id].append(f"{table_id}.{col.name}")
                col_stypes[str(ColumnID(table_id, col.name))] = col.stype

        assert root_table_id is not None

        # Merge BFS tree to navigation.
        for table_id, adj_table_ids in bfs_tree.items():
            navigation_tree[table_id].extend(adj_table_ids)
        resp.navigation = NavigationInfo(root_table_name=root_table_id,
                                         navigation_tree=navigation_tree)

        # Populate variance score and column analysis in response.
        name_to_table = {table.table_name: table for table in tables}
        column_analysis = column_analysis[0]
        if column_analysis is None or column_analysis.cohorts is None:
            log.warning(
                "Batch prediction XAI column analysis is unavailable for %s; "
                "returning navigation-only XAI response.",
                home_mgr.home_dir,
            )
            return resp
        for col_cohort in column_analysis.cohorts:
            table_id = to_table_id(name_to_table[col_cohort.table_name].name)
            col_id = ColumnID(table_id, col_cohort.col_name)
            if col_cohort.hop == 'all':
                hop_info = 'all hops'
            elif col_cohort.hop == 0:
                hop_info = 'seed entities'
            else:
                hop_info = f'{col_cohort.hop} hop'
            col_name_with_hop = f'{col_id.col_name} ({hop_info})'
            resp.tables[table_id][col_name_with_hop] = ColumnExplanation(
                col_id=str(col_id),
                stype=col_stypes.get(str(col_id)),
                variance_score=col_cohort.prediction_score,
                gradient_score=None,
                cohort=column_cohorts_to_cube(col_cohort),
            )
        return resp

    async def get_available_xai_entities(
        self,
        home_mgr: HomeDirManager,
    ) -> StrategyMappings:
        mappings_file_path = home_mgr.subgraph_mappings_path(TrainingStage.PRED)
        strategy_mappings = await load_json_async(
            mappings_file_path,
            StrategyMappings,
            default=None,
            raise_on_missing=False,
        )
        if strategy_mappings is None:
            log.warning(f"Strategy mappings could not be found on S3 at "
                        f"path '{mappings_file_path}'. Return empty mappings "
                        f"instead.")
            return StrategyMappings()
        return strategy_mappings

    async def get_entity_old_subgraph(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
    ) -> OldSubgraph:
        subgraph_file_path = home_mgr.subgraph_json_path(
            TrainingStage.PRED, entity_id, old=True)
        subgraph = await load_json_async(
            subgraph_file_path,
            OldSubgraph,
            default=None,
            raise_on_missing=False,
        )
        if subgraph is None:
            log.warning(
                f"Entity {entity_id} subgraph could not be found on S3 at "
                f"path '{subgraph_file_path}'. Return an empty subgraph "
                f"instead.")
            return OldSubgraph(id=entity_id, table_name=None, task_type=None)
        return subgraph

    async def get_subgraph(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
    ) -> Subgraph:
        subgraph_file_path = home_mgr.subgraph_json_path(
            TrainingStage.PRED, entity_id, old=False)
        subgraph = await load_json_async(
            subgraph_file_path,
            Subgraph,
            default=None,
            raise_on_missing=False,
        )
        if subgraph is None:
            log.warning(
                f"Entity {entity_id} subgraph could not be found on S3 at "
                f"path '{subgraph_file_path}'. Return an empty subgraph "
                f"instead.")
            return Subgraph()
        return subgraph

    async def get_entity_output(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
    ) -> EntityOutput:
        output_file_path = home_mgr.subgraph_output_path(
            TrainingStage.PRED, entity_id)
        output = await load_json_async(
            output_file_path,
            EntityOutput,
            default=None,
            raise_on_missing=False,
        )
        if output is None:
            log.warning(
                f"Entity {entity_id} output could not be found on S3 at "
                f"path '{output_file_path}'. Return an empty entity output "
                f"instead.")
            return EntityOutput(task_type=None)
        return output

    async def get_entity_historical_items(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
    ) -> HistoricalItems:
        items_file_path = home_mgr.subgraph_historical_items_path(
            TrainingStage.PRED, entity_id)
        items = await load_json_async(items_file_path, HistoricalItems,
                                      default=None, raise_on_missing=False)
        if items is None:
            log.warning(
                f"Entity {entity_id} historical items could not be found on "
                f"S3 at path '{items_file_path}'. Return empty historical "
                f"items instead.")
            return HistoricalItems()
        return items

    async def _get_outputs_of_running_jobs(
        self,
        ctx: RequestContext,
        outputs: List[PredictionOutputConfig],
        *,
        project_id: ProjectID,
    ) -> _PendingOutputs:
        ids = await self._ml_backend.get_running_batch_prediction_job_ids(
            project_ids={project_id})
        running_jobs = await self.list_batch_prediction_jobs(
            ctx, job_ids=set(ids))

        pending_outputs = _PendingOutputs()

        for job in running_jobs:
            for output in job.config.outputs:
                if (isinstance(output, DatabricksPredictionOutput)
                        or isinstance(output, SnowflakePredictionOutput)
                        or isinstance(output, BigQueryPredictionOutput)):
                    table = _WarehouseTablePath(output.connector_id,
                                                output.table_name)
                    pending_outputs.warehouse_tables[table] = _ConflictInfo(
                        is_other_job=True, bp_id=job.id)
                elif isinstance(output,
                                (S3PredictionOutput, GCSPredictionOutput,
                                 ADLSPredictionOutput)):
                    path = _file_output_path(output)
                    pending_outputs.file_paths[path] = _ConflictInfo(
                        is_other_job=True, bp_id=job.id)
                elif isinstance(output, LocalPredictionOutput):
                    continue
                else:
                    log.error(f"Invalid storage type {type(output)} for {job}")
        return pending_outputs

    async def _validate_output(
        self,
        output: PredictionOutputConfig,
        allow_local_paths: bool,
        pending_outputs: _PendingOutputs,
        project_id: ProjectID,
    ) -> Optional[CallToAction]:
        warehouse_output_types = (
            DatabricksPredictionOutput,
            SnowflakePredictionOutput,
            BigQueryPredictionOutput,
        )
        file_output_types = (
            S3PredictionOutput,
            GCSPredictionOutput,
            ADLSPredictionOutput,
        )
        if isinstance(output, warehouse_output_types):
            return await self._validate_warehouse_output(
                output, pending_outputs, project_id)
        elif isinstance(output, file_output_types):
            return await self._validate_object_output(output, allow_local_paths,
                                                      pending_outputs,
                                                      project_id)
        elif isinstance(output, LocalPredictionOutput):
            return None
        else:
            raise ValueError(f"Invalid storage type `{type(output)}`")

    async def _validate_object_output(
        self,
        output: Union[S3PredictionOutput, GCSPredictionOutput,
                      ADLSPredictionOutput],
        allow_local_paths: bool,
        pending_outputs: _PendingOutputs,
        project_id: ProjectID,
    ) -> Optional[CallToAction]:
        if self._state_services is None:
            return CallToAction.crit(
                'Object-store output connector validation is unavailable.')
        connector_id = ConnectorID(output.connector_id)
        expected_type = {
            S3PredictionOutput: ConnectorType.S3,
            GCSPredictionOutput: ConnectorType.GCS,
            ADLSPredictionOutput: ConnectorType.ADLS,
        }[type(output)]
        principal = _current_user_ctx.get()
        try:
            connector = await self._state_services.connectors.get_for_project(
                project_id, connector_id, principal)
        except (IdentityForbiddenError, ConnectorDoesNotExistError,
                ProjectForbiddenError, ProjectNotFoundError):
            return CallToAction.crit(
                f"Output connector with ID {output.connector_id} does not exist."
            )
        try:
            normalize_object_prediction_output(output, connector.config)
        except (TypeError, ValueError) as error:
            return CallToAction.crit(str(error))
        try:
            reference = await (
                self._state_services.connectors.execution_ref_for_project(
                    principal,
                    project_id,
                    connector_id,
                    CONNECTOR_OUTPUT_ACTIONS,
                    expected_type=expected_type,
                ))
            if reference.project_id != project_id:
                raise ValueError(
                    'Connector was authorized for a different project')
        except (IdentityForbiddenError, ConnectorDoesNotExistError,
                ProjectForbiddenError, ProjectNotFoundError, ValueError):
            return CallToAction.crit(
                f"Output connector with ID {output.connector_id} does not exist."
            )
        conflict = self._reserve_file_output(output, pending_outputs)
        if conflict is not None:
            return conflict
        return await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            validate_object_prediction_output,
            [reference, output, allow_local_paths],
            f'validate-object-output:{project_id}',
        )

    async def _validate_warehouse_output(
        self,
        output: Union[
            DatabricksPredictionOutput,
            SnowflakePredictionOutput,
            BigQueryPredictionOutput,
        ],
        pending_outputs: _PendingOutputs,
        project_id: ProjectID,
    ) -> Optional[CallToAction]:
        connector_id = ConnectorID(output.connector_id)
        table = _WarehouseTablePath(connector_id, output.table_name)
        if table in pending_outputs.warehouse_tables:
            conflict_info = pending_outputs.warehouse_tables[table]
            if conflict_info.is_other_job:
                return CallToAction.crit(
                    f"Job output has the same output table as a "
                    f"currently running job: '{table.connector_id}."
                    f"{table.table}'. Please pick a different "
                    f"output table, or wait for the current job to "
                    f"complete: {conflict_info.bp_id}")
            else:
                return CallToAction.crit(
                    f"Multiple outputs with the same output table were "
                    f"detected. Please ensure that all outputs are unique: "
                    f"{table.connector_id}.{table.table}")
        pending_outputs.warehouse_tables[table] = _ConflictInfo(
            is_other_job=False, bp_id=None)

        # TODO(zack): Validate whether table exists, and give error if the
        # schema doesn't match, and give warning that we will append to the
        # table if the schema matches

        if self._state_services is None:
            return CallToAction.crit(
                'Warehouse output connector validation is unavailable.')

        expected_type = {
            SnowflakePredictionOutput: ConnectorType.SNOWFLAKE,
            DatabricksPredictionOutput: ConnectorType.DATABRICKS,
            BigQueryPredictionOutput: ConnectorType.BIGQUERY,
        }[type(output)]
        try:
            reference = await self._state_services.connectors.execution_ref_for_project(
                _current_user_ctx.get(),
                project_id,
                connector_id,
                CONNECTOR_OUTPUT_ACTIONS,
                expected_type=expected_type,
            )
            if reference.project_id != project_id:
                raise ValueError(
                    'Connector was authorized for a different project')
        except (IdentityForbiddenError, ConnectorDoesNotExistError,
                ProjectForbiddenError, ProjectNotFoundError, ValueError):
            return CallToAction.crit(
                f"Output connector with ID {output.connector_id} does not exist."
            )
        result = await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            validate_warehouse_prediction_output,
            [reference, output],
            f'validate-warehouse-output:{project_id}',
        )
        if result is None and _has_non_uppercase_chars(output.table_name):
            return CallToAction.warn(
                'For the optimal experience, output table names should use '
                'only uppercase characters, numbers, and underscores.')
        return result

    @staticmethod
    def _reserve_file_output(
        output,
        pending_outputs: _PendingOutputs,
    ) -> Optional[CallToAction]:
        output_path = _file_output_path(output)
        if output_path in pending_outputs.file_paths:
            conflict_info = pending_outputs.file_paths[output_path]
            if conflict_info.is_other_job:
                return CallToAction.crit(
                    f"Job output has the same file path as a "
                    f"currently running job: '{conflict_info.bp_id}' Please "
                    f"pick a different output path: {output_path.coordinate}")
            else:
                return CallToAction.crit(
                    f"Multiple artifacts with the same file path were "
                    f"detected. Please ensure that all outputs paths "
                    f"are unique: {output_path.coordinate}")
        pending_outputs.file_paths[output_path] = _ConflictInfo(
            is_other_job=False, bp_id=None)
        return None

    def filters_are_query_compatible(
        self,
        filter_overrides: Optional[FilterOverrideArgs],
    ) -> Optional[ValidationResponse]:
        """
        Some filters like temporal_entity_filter is a legacy filter supported
        in PQL v1. This method validates that a PQL v1 filter is not used
        in PQL v2.

        Parameters:
        filter_overrides (FilterOverrideArgs): All the filter overrides used in
        the query.

        Returns:
        None if validation passes, str otherwise containing the reason for
        failure.
        """
        if filter_overrides is not None and (
                filter_overrides.temporal_entity_filter_override is not None
                and filter_overrides.temporal_entity_filter_override != ""):
            return ValidationResponse(errors=[
                ValidationError(
                    title="Incorrect filter override",
                    message="temporal_entity_filter is not supported in PQL "
                    "v2. Please use entity_filter instead even if your filter "
                    "contains an aggregation.",
                )
            ])
        return None


def _has_non_uppercase_chars(name: str) -> bool:
    return re.search(r"[^A-Z0-9_]", name) is not None


async def _make_bp_input_v2(
    pq_resource: PQueryResource,
    pq_def_fpid: FPID,
    id: BatchPredictionJobID,
    job_config: BatchPredictionJobConfig,
    data_backend_ctx: DataBackendContext,
    *,
    state_services: Optional[StateServices] = None,
    train_table_options: Optional[TrainingTableGenerationPlan] = None,
    output_connectors: Optional[Dict[ConnectorID,
                                     ConnectorExecutionRef]] = None,
    project_id: ProjectID,
    graph_data_compute_input: Optional["GraphDataComputeInput"] = None,
    graph_version_info: Optional[GraphDataVersionInfo] = None,
    custom_pred_table_connector: Optional[ConnectorExecutionRef] = None,
) -> BatchPredictionJobInput:
    """Builds a `BatchPredictionJobInput` from a `PQueryResource` +
    pre-frozen `pq_def_fpid` via the v2-native `make_predictive_config_v2`
    / `make_v2_query_with_overrides_v2`.

    Graph overrides are used verbatim instead of re-resolving from the pquery.
    This keeps a decoupled generator and its BP consumer on the same immutable
    ``(graph_key, graph_data_version)`` identity.
    """
    output_connectors = output_connectors or {}
    if state_services is None:
        raise RuntimeError(
            'Batch prediction input generation requires StateServices')

    graph_id = GraphIDv2.compose(fpid=pq_resource.graph.fpid)
    assert graph_id == job_config.graph_id

    pq_logic = pquery_logic.PQueryLogic(
        data_backend_ctx,
        state_services=state_services,
        project_id=project_id,
    )
    val_args = PQueryValidationArgs(
        query_string=pq_resource.query_string,
        graph=pq_resource.graph,
        name=pq_resource.name or "",
        desc=pq_resource.desc or "",
        train_table_options=train_table_options,
        runtime_options=pquery_runtime_options_or_none(
            list_distinct_target_is_array=(
                pq_resource.list_distinct_target_is_array),
            lp_bp_layout=pq_resource.lp_bp_layout,
        ),
    )
    graph = await graph_logic.GraphLogic(
        data_backend_ctx,
        state_services=state_services,
        project_id=project_id,
    )._make_graph_from_graph_def_v2(pq_resource.graph)
    train_query_config = await pq_logic.make_predictive_config_v2(
        val_args, graph=graph)

    job_config.pquery_def_fpid = pq_def_fpid

    if job_config.filter_override_args:
        (
            new_val_args,
            validation_response,
        ) = await pq_logic.make_v2_query_with_overrides_v2(
            val_args, job_config.filter_override_args)
        assert validation_response.ok
        assert new_val_args is not None
        query_config = await pq_logic.make_predictive_config_v2(
            new_val_args, graph=graph)
    else:
        query_config = train_query_config

    if graph_data_compute_input is None:
        graph_data_compute_input = await _get_graph_data_compute_input_v2(
            graph_id, project_id, state_services)
    if graph_version_info is None and job_config.graph_data_version:
        cache = get_active_cache()
        graph_version_info = (await load_many_json.coroutine(
            [
                cache.graph_cache.version_info_path(
                    graph_id, job_config.graph_data_version)
            ],
            GraphDataVersionInfo,
        ))[0]

    if graph_version_info:
        for table in query_config.graph.tables:
            assert isinstance(table, TableConfig)
            old_version = table.data_version
            new_version = graph_version_info.table_data_versions[TableName(
                table.table_name)]
            table.data_version = new_version
            log.info(f"Set table data version for {table.table_name} to "
                     f"{new_version} from {old_version}.")
    else:
        log.warning(f"Creating batch prediction job input for {id} without "
                    f"a graph version info object. Please be sure this is "
                    f"intended.")

    job_input = BatchPredictionJobInput(
        id=id,
        project_id=project_id,
        config=job_config,
        predictive_query=query_config,
        output_connectors=output_connectors,
        start_time=now_utc(),
        graph_data_compute_input=graph_data_compute_input,
        graph_version_info=graph_version_info,
        custom_pred_table_connector=custom_pred_table_connector,
    )

    return job_input


async def _get_graph_data_compute_input_v2(
    graph_id: GraphIDv2,
    project_id: ProjectID,
    state_services: Optional[StateServices] = None,
) -> GraphDataComputeInput:
    if state_services is not None:
        return await state_services.graphs.get_graph_data_compute_input(
            graph_id,
            project_id=project_id,
            user_authorization=_current_user_ctx.get(),
        )

    raise RuntimeError("Batch prediction input generation requires "
                       "StateServices.graphs.")


def _validate_task_type_options_v2(task_type: TaskType,
                                   request: BatchPredictionRequest) -> None:
    """Validate prediction options without mutating the request.

    The
    threshold-reset-to-None behavior (for non-binary tasks with a threshold)
    moves to `effective_threshold` computation in `create_batch_prediction_job_v2`,
    not here.
    """
    resolve_classification_threshold(
        task_type,
        request.predict_options.binary_classification_threshold,
        binary_required=True,
    )
    # NOTE: non-binary reset-to-None + warning is NOT here. That moves to the
    # effective_threshold computation in `create_batch_prediction_job_v2`.

    num_classes = request.predict_options.num_classes_to_return
    if not task_type.has_multiple_predictions:
        if num_classes is not None:
            raise ValueError("num_classes_to_return can only be specified "
                             "for multilabel, multiclass, and link "
                             "prediction queries. Please remove it "
                             "and try again.")
    elif task_type.is_link_pred:
        has_predictions = any(
            output.artifact_type == PredictionArtifactType.PREDICTIONS
            for output in request.outputs)

        if has_predictions:
            top_k = num_classes
            if top_k is None:
                raise ValueError("Please specify the number of predictions "
                                 "per entity when exporting predictions for "
                                 "a link prediction query.")
            if top_k > MAX_LINKS_TO_PREDICT:
                raise ValueError(f"For data efficiency reasons, "
                                 f"num_classes_to_return should be less than "
                                 f"or equal to "
                                 f"{MAX_LINKS_TO_PREDICT}. If you need "
                                 f"more, you can export embeddings and perform "
                                 f"maximum inner product search on the "
                                 f"embedding space.")
