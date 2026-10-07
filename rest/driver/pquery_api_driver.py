# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import dataclasses
import datetime
import hashlib
import os
from collections.abc import Iterable
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import kumoapi.data_source as data_source_api
import kumoapi.graph as graph_api
import kumoapi.jobs as jobs_api
import kumoapi.model_plan as model_plan_api
import kumoapi.source_table as source_table_api
import pandas as pd
from kumo_core.callbacks.train_progress_logger import AutoTrainerProgress
from kumo_core.model_plan import (
    ModelPlan,
    ModelPlanValueError,
    TrainingTableGenerationPlan,
    inferred_or_missing_type_to_false,
    inferred_or_missing_type_to_none,
)
from kumo_core.training.config import TrainerConfig
from kumo_core.types import (
    PredictionArtifactType,
    TaskType,
    TaskTypeWithReason,
)
from kumoapi.data_snapshot import GraphSnapshotID
from kumoapi.jobs import (
    GetEmbeddingsDfUrlResponse,
    GetPredictionsDfUrlResponse,
    PredictionProgress,
)
from kumoapi.model_plan import RunMode
from kumoapi.pquery import QueryType
from kumopql.parser import PQLParser
from kumopql.rewriting.bp_filter_transformer import (
    BPFilterTransformer,
    EditableQuery,
    EditableQueryResponse,
)
from kumopql.validator import PredictiveQueryValidator

from async_utils.sync import blocking_func, blocking_method
from io_utils import fsutil, logging
from io_utils.fsutil.impl_local import LocalPresignedHelper
from io_utils.json import from_json, load_json, load_json_async, to_json
from kumo.config.model_plan_yaml import model_plan_to_yaml
from kumo.config.predictive_query import PredictiveQueryConfigBase
from kumo.connector.connector import ConnectorError
from kumo.data import get_active_cache
from kumo.data.backend_cache import BackendCacheManager
from kumo.graph.utils import map_to_api_graph_definition
from kumo.home import HomeDirManager
from kumo.pquery.predictive_query import PredictiveQuery
from kumo.pquery.utils import (
    validate_forecasting_parameters_consistency,
    validator_kwargs,
)
from kumo.stats.stats import Stats
from kumo.stats.train_table_stats import (
    NUM_ENTITIES,
    SplitTrainTableStats,
    TrainTableGenerationProgress,
    TrainTableStats,
    load_train_table_stats_from_cache,
    train_table_generation_stats_from_cache_v2,
    train_table_generation_stats_from_job_dir,
    train_table_split_stats_from_cache,
    train_table_split_stats_from_job_dir,
)
from kumo.task.split import Split
from kumo.train.utils import progress_util
from kumo.utils.embedding_models import validate_text_embedding_models
from kumo.validation import ValidationException, ValidationResponse
from kumo.validation.validation import ValidationError, ValidationWarning
from rest.backends.data_backend.activities import custom_table_validation
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.data_backend import DataBackend
from rest.backends.ml_backend.data_util import get_job_base_dir, get_job_dir
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
# yapf: disable
from rest.backends.ml_backend.prediction_artifacts import (
    resolve_prediction_artifact,
)
# yapf: enable
from rest.config.connector import object_store_source_for_connector
from rest.config.data_backend import (
    ConnectorExecutionRef,
    GraphDataVersionInfo,
)
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.ml_backend import (
    GenerateTrainTableJobInput,
    MaterializationTaskInfo,
    TrainingJobInput,
)
from rest.driver.graph_driver import GraphDriver
from rest.driver.table_driver import TableDriver
from rest.utils.driver_common import (
    fail_closed_graph_from_config,
    get_current_timestamps,
)
from rest.utils.ui_to_sdk_utils import (
    clean_up_pquery_string,
    generate_conn_graph_or_table_sdk_script,
    generate_training_job_sdk_script,
    make_stype_and_dtype_sdk_compat,
    recursively_convert_dataclass_to_dict,
    render_str_vals_as_strings,
)
from service_framework.deploy_env_config import deploy_env
from statedb.config.common import ConnectorID
from statedb.config.predictive_query import FilterOverrideArgs
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS

# isort: off
from rest.backends.ml_backend.job_queue import (
    JobConcurrencyLimitError, )
# isort: on
from typing import Dict, List, Optional, Tuple

from kumo_core.types.schema import FPID

from rest.config.batch_prediction_job import (
    BatchPredictionJobAnchorTime,
    BatchPredictionJobConfig,
    BatchPredictionJobResponse,
    BatchPredictionJobStats,
    PaginatedBatchPredictionJobResponse,
    PaginationArgs,
)
from rest.config.batch_prediction_job_v2 import BatchPredictionFilterOverrides
from rest.config.graph_v2 import GraphDefinition, build_graph_definition
from rest.config.jobs import \
    BatchPredictionRequest as RestBatchPredictionRequest
from rest.config.mlops import MLOpsQueryConfig, MLOpsStats
from rest.config.predictive_query import (
    DefaultModelPlanInfo,
    DefaultModelPlanRequest,
    InsertFilterOverridesRequest,
    ModelPlansResponse,
    OutputPresignedUrlResponse,
    PQueryDefinition,
    RESTParsedPredictiveQuery,
    pquery_runtime_options_or_none,
)
from rest.config.predictive_query_v2 import (
    PQueryResource,
    PQueryValidationArgs,
    PQueryValidationRequest,
)
from rest.config.status import RestStatusCode
from rest.config.training_job import CustomTrainingTable, TrainingJobResponse
from rest.config.training_job_v2 import TrainingJobInternalRequest
from rest.deps.authentication import (
    RequestContext,
    visible_project_ids_or_none,
)
from rest.deps.context import _current_user_ctx
from rest.driver.batch_prediction_job_logic import BatchPredictionJobLogic
from rest.driver.data_snapshot_api_driver import get_graph_snapshot_info
from rest.driver.error_constants import (
    BATCH_PREDICTION_JOB_NOT_FOUND_MESSAGE,
    BATCH_PREDICTION_JOBS_DELETE_MESSAGE,
    INCOMPATIBLE_COLUMN_TYPES_MESSAGE,
    INCOMPATIBLE_COLUMN_TYPES_TITLE,
    INCOMPATIBLE_COLUMNS_MESSAGE,
    INCOMPATIBLE_COLUMNS_TITLE,
    INCOMPATIBLE_EDGES_MESSAGE,
    INCOMPATIBLE_EDGES_TITLE,
    INCOMPATIBLE_END_TIME_COLUMN_MESSAGE,
    INCOMPATIBLE_END_TIME_COLUMN_TITLE,
    INCOMPATIBLE_GRAPH_MESSAGE,
    INCOMPATIBLE_GRAPH_TITLE,
    INCOMPATIBLE_PRIMARY_KEY_MESSAGE,
    INCOMPATIBLE_PRIMARY_KEY_TITLE,
    INCOMPATIBLE_TIME_COLUMN_MESSAGE,
    INCOMPATIBLE_TIME_COLUMN_TITLE,
    PREDICTION_ARTIFACT_TYPE_NOT_FOUND_MESSAGE,
    PREDICTION_OUTPUT_SIZE_MESSAGE,
    PREDICTION_TABLE_JOB_FAILED_MESSAGE,
    PREDICTION_TABLE_JOB_FAILED_TITLE,
    TRAINING_JOB_INCOMPLETE_MESSAGE,
    TRAINING_JOB_INCOMPLETE_TITLE,
    TRAINING_JOB_NO_PQUERY_MESSAGE,
    TRAINING_JOB_NO_PQUERY_TITLE,
    TRAINING_JOB_NOT_FOUND_MESSAGE,
    TRAINING_JOB_NOT_FOUND_TITLE,
)
from rest.driver.graph_logic import GraphLogic
from rest.driver.mlops_logic import MLOpsLogic
from rest.driver.pquery_logic import PQueryLogic
from rest.driver.validators import (
    BatchPredictionValidator,
    TrainingJobValidator,
    _get_warn_details,
    _validation_to_warn_details,
)
from rest.settings import DataScaleLimits, PredictiveQuerySettings
from rest.utils.driver_common import extract_source_table_type
from rest.utils.pquery import forecasting_pass_through_plan
from rest.utils.routers import ValidationHTTPException
from statedb.config.common import (
    BatchPredictionJobID,
    ErrWarnDetails,
    ErrWarnResponse,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    GraphID,
    GraphIDv2,
    GraphLink,
    JobID,
    JobSource,
    JobStatus,
    JobType,
    PredictiveQueryID,
    PredictiveQueryLink,
    ProjectID,
    TrainingJobID,
    to_graph_id,
)
from statedb.config.job_tag import (
    TAG_KEY_PARENT_JOB,
    TAG_KEY_PQUERY_ID,
    TAG_KEY_PQUERY_NAME,
    TAG_KEY_WARM_START_JOB_ID,
    JobTagKey,
    JobTags,
    JobTagValue,
)
from statedb.config.timestamp import ResourceTimestamps
from statedb.db.ml_jobs import MLJobService
from statedb.db.services import GraphService, PQueryService, StateServices
from statedb.utils.errors import (
    BatchPredictionDoesNotExistError,
    BatchPredictionNotInProjectError,
    EntityDoesNotExistError,
    GraphDoesNotExistError,
    PredictiveQueryDoesNotExistError,
    TrainingJobDoesNotExistError,
    TrainingJobNotInProjectError,
)
from temporalio_commons.auto_rpc_exception import RpcError

log = logging.getLogger(__name__)

_DATA_DOWNLOAD_LIMIT = 1024 * 1024 * 1024

# Forecasting validation
_FORECASTING_PARAMS_TITLE = "Invalid Forecasting Configuration"
_FORECASTING_PARAMS_NO_TRAINING_JOB_MESSAGE = (
    "Forecasting parameters must be sourced from the training config. "
    "Please provide a valid training job ID.")
_FORECASTING_PARAMS_NO_GEN_TT_MESSAGE = (
    "Could not find the training table generation job for training job {}. "
    "The training job may be corrupted or missing its training table "
    "configuration.")


class TrainingJobEncodersNotAvailableError(EntityDoesNotExistError):
    """Raised when a terminal training job has no encoder artifact."""
    def __init__(self) -> None:
        Exception.__init__(
            self,
            RestStatusCode.ENCODERS_NOT_AVAILABLE.value,
        )


def _training_job_not_in_project(
        id: TrainingJobID,
        project_id: ProjectID) -> TrainingJobNotInProjectError:
    return TrainingJobNotInProjectError(
        f"Training Job with id='{id}' is not in project '{project_id}'")


def _batch_prediction_not_in_project(
        id: BatchPredictionJobID,
        project_id: ProjectID) -> BatchPredictionNotInProjectError:
    return BatchPredictionNotInProjectError(
        f"Batch Prediction with id='{id}' is not in project '{project_id}'")


@dataclasses.dataclass(frozen=True)
class ForecastingParams:
    """Forecasting parameters sourced from a training config."""
    lag_timesteps: Optional[int]
    year_over_year: bool
    forecast_length: Optional[int]

    @classmethod
    def from_plan(
        cls,
        plan: Union[TrainingTableGenerationPlan,
                    model_plan_api.PredictionTableGenerationPlan],
    ) -> 'ForecastingParams':
        """Extract forecasting params from a plan with forecasting fields."""
        return cls(
            lag_timesteps=inferred_or_missing_type_to_none(plan.lag_timesteps),
            year_over_year=inferred_or_missing_type_to_false(
                plan.year_over_year),
            forecast_length=inferred_or_missing_type_to_none(
                plan.forecast_length),
        )

    @classmethod
    def default(cls) -> 'ForecastingParams':
        return cls(lag_timesteps=None, year_over_year=False,
                   forecast_length=None)

    @property
    def is_forecasting(self) -> bool:
        return self.forecast_length is not None and self.forecast_length > 1

    @property
    def has_any(self) -> bool:
        return (self.forecast_length is not None
                or self.lag_timesteps is not None or self.year_over_year)


class PQueryApiDriver:
    # Internally used to filter out metrics from
    # TrainingJob.results.metrics
    FILTER_METRICS = ['val_.*', 'train_.*']

    def __init__(
        self,
        ml_job_backend: MLJobBackend,
        data_backend_ctx: DataBackendContext,
        allow_local_batch_pred_paths: bool,
        data_scale_limits: DataScaleLimits,
        pquery_settings: PredictiveQuerySettings,
        table_driver: TableDriver,
    ):
        """Construct the driver for PQuery related APIs.

        Args:
            ml_job_backend (MLJobBackend): Backend for ML train/predict jobs.

            data_backend_ctx (DataBackendContext): Backend context for data
            operations on table or graph
            (cache, metadata, ingestion and stats).

            allow_local_batch_pred_paths (bool): If true, both s3 and local
            file paths are allowed to be used for output of batch prediction;
            otherwise only s3 file paths are allowed.

        """
        self._allow_local_batch_pred_paths = allow_local_batch_pred_paths
        self._ml_job_backend = ml_job_backend
        self._data_backend_ctx = data_backend_ctx
        self._data_scale_limits = data_scale_limits
        self._pquery_settings = pquery_settings
        self._state_services: Optional[StateServices] = None
        # GraphDriver is constructed in-process because public API drivers are
        # injected into ``rest.public.app.AppBuilder`` and cannot be serialized
        # through a Temporal RPC boundary.
        self._graph_driver = GraphDriver(
            data_backend_ctx=data_backend_ctx,
            data_scale_limits=data_scale_limits,
            table_driver=table_driver,
        )

    def _configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services
        self._graph_driver._configure_state_services(state_services)

    @property
    def _ml_job_service(self) -> MLJobService:
        return self._require_state_services().ml_jobs

    @property
    def _pquery_service(self) -> PQueryService:
        return self._require_state_services().pqueries

    @property
    def _graph_service(self) -> GraphService:
        return self._require_state_services().graphs

    def _require_pquery_service(self) -> PQueryService:
        return self._pquery_service

    def _require_graph_service(self) -> GraphService:
        return self._graph_service

    def _require_ml_job_service(self) -> MLJobService:
        return self._ml_job_service

    def _require_state_services(self) -> StateServices:
        if self._state_services is None:
            raise RuntimeError("StateServices is required")
        return self._state_services

    async def _require_project_access(self, project_id: ProjectID) -> None:
        await self._require_state_services().projects.get_project(
            project_id,
            _current_user_ctx.get(),
        )

    async def require_project_access(self, ctx: RequestContext,
                                     project_id: ProjectID) -> None:
        await self._require_state_services().projects.get_project(
            project_id, ctx.user_authorization)

    @staticmethod
    def _job_pquery_fpid(job: Any) -> Optional[FPID]:
        """Return the immutable PQuery definition FPID recorded on a job."""
        direct = getattr(job, 'pquery_def_fpid', None)
        if direct is not None:
            return FPID(direct)

        config = getattr(job, 'config', None)
        if config is None:
            return None
        direct = getattr(config, 'pquery_def_fpid', None)
        if direct is not None:
            return FPID(direct)
        pquery_id = (getattr(config, 'pquery_id', None)
                     or getattr(config, 'predictive_query_id', None))
        if pquery_id is None:
            return None
        try:
            _, fpid = PredictiveQueryID.parse(PredictiveQueryID(pquery_id))
        except ValueError:
            return None
        return fpid

    async def _require_job_pquery_binding(
        self,
        job_id: JobID,
        pquery_id: PredictiveQueryID,
        project_id: ProjectID,
    ) -> FPID:
        """Verify a job, project, and immutable PQuery definition agree."""
        _, claimed_fpid = PredictiveQueryID.parse(pquery_id)
        if claimed_fpid is None:
            raise PredictiveQueryDoesNotExistError(pquery_id)
        job = await self._require_ml_job_service().get_job_if_exists_generic(
            job_id)
        if (job is None
                or str(getattr(job, 'project_id', '')) != str(project_id)
                or self._job_pquery_fpid(job) != claimed_fpid):
            # Use a resource-not-found response for every mismatch so this
            # authorization check cannot be used to enumerate another
            # project's jobs or PQueries.
            raise PredictiveQueryDoesNotExistError(pquery_id)
        return claimed_fpid

    def _batch_prediction_logic(self) -> BatchPredictionJobLogic:
        return BatchPredictionJobLogic(
            self._ml_job_backend,
            self._data_backend_ctx,
            state_services=self._state_services,
        )

    def _pquery_logic(
        self,
        project_id: ProjectID,
    ) -> PQueryLogic:
        return PQueryLogic(
            self._data_backend_ctx,
            state_services=self._require_state_services(),
            project_id=project_id,
        )

    def _graph_logic(
        self,
        project_id: ProjectID,
    ) -> GraphLogic:
        return GraphLogic(
            self._data_backend_ctx,
            state_services=self._state_services,
            project_id=project_id,
        )

    async def _get_graph_def_for_pquery(
        self,
        graph_fpid: FPID,
        project_id: ProjectID,
    ) -> Optional[GraphDefinition]:
        await self._require_project_access(project_id)
        return await self._require_graph_service().get_graph_definition(
            project_id, graph_fpid)

    @staticmethod
    def _pquery_resource_from_definition_and_graph(
        pq_def: PQueryDefinition,
        graph_def: Optional[GraphDefinition],
    ) -> PQueryResource:
        if graph_def is None:
            pq_id = PredictiveQueryID.compose(pq_def.pquery_name, pq_def.fpid)
            raise ValueError(f'Unable to find graph definition for {pq_id}')

        return PQueryResource(
            name=pq_def.pquery_name,
            query_string=pq_def.query_str,
            graph=graph_def,
            runtime_options=pquery_runtime_options_or_none(
                list_distinct_target_is_array=(
                    pq_def.list_distinct_target_is_array),
                lp_bp_layout=pq_def.lp_bp_layout,
            ),
        )

    async def _pquery_resource_from_definition(
        self,
        pq_def: PQueryDefinition,
        project_id: ProjectID,
    ) -> PQueryResource:
        graph_def = await self._get_graph_def_for_pquery(
            pq_def.graph_fpid, project_id)
        return self._pquery_resource_from_definition_and_graph(
            pq_def, graph_def)

    async def _pquery_resources_from_definitions(
        self,
        pq_defs: Iterable[PQueryDefinition],
        project_id: ProjectID,
    ) -> Dict[FPID, PQueryResource]:
        pq_defs = list(pq_defs)
        if not pq_defs:
            return {}

        graph_defs = await self._require_graph_service().get_graph_definitions(
            project_id, {pq_def.graph_fpid
                         for pq_def in pq_defs})
        resources: Dict[FPID, PQueryResource] = {}
        for pq_def in pq_defs:
            graph_def = graph_defs.get(pq_def.graph_fpid)
            if graph_def is not None:
                resources[pq_def.fpid] = (
                    self._pquery_resource_from_definition_and_graph(
                        pq_def, graph_def))
        return resources

    async def _pquery_definitions_by_ids(
        self,
        pq_ids: Iterable[PredictiveQueryID],
    ) -> Dict[PredictiveQueryID, Optional[PQueryDefinition]]:
        unique_pq_ids = list(dict.fromkeys(pq_ids))
        if not unique_pq_ids:
            return {}

        definitions_by_id = (await self._require_pquery_service().
                             get_definitions_by_ids(unique_pq_ids))
        return {pq_id: definitions_by_id.get(pq_id) for pq_id in unique_pq_ids}

    async def _get_graph_data_compute_input(
        self,
        graph_id: GraphIDv2,
        project_id: ProjectID,
    ) -> GraphDataComputeInput:
        return await self._require_graph_service().get_graph_data_compute_input(
            graph_id,
            project_id=project_id,
            user_authorization=_current_user_ctx.get(),
        )

    async def _apply_job_tags(
        self,
        job_type: JobType,
        job_id: JobID,
        tags: Dict[JobTagKey, JobTagValue],
    ) -> bool:
        return await self._require_ml_job_service().apply_job_tags(
            job_type, job_id, tags)

    async def _link_graph_to_project(
        self,
        graph_fpid: FPID,
        project_id: ProjectID,
    ) -> None:
        """Fail closed unless the predictive query graph belongs to project."""
        await self._graph_driver.get_graph_v2(graph_fpid, project_id)

    async def get_predictive_query_v2(
        self,
        pq_id: PredictiveQueryID,
        project_id: ProjectID,
    ) -> PQueryResource:
        await self._require_project_access(project_id)
        pq_def = await self._require_pquery_service().get_definition_by_id(pq_id
                                                                           )
        if pq_def is None:
            raise PredictiveQueryDoesNotExistError(pq_id)

        return await self._pquery_resource_from_definition(pq_def, project_id)

    async def get_many_predictive_queries_v2(self, job_to_pq_fpid: Dict[Union[
        TrainingJobID,
        BatchPredictionJobID,
        GeneratePredictionTableJobID,
        GenerateTrainTableJobID,
    ], PredictiveQueryID], project_id: ProjectID,
                                             skip_validation: bool = False):
        await self._require_project_access(project_id)
        id_to_fpid: Dict[PredictiveQueryID, FPID] = {}
        for job_id, pq_id in job_to_pq_fpid.items():
            id_to_fpid[pq_id] = await self._require_job_pquery_binding(
                JobID(job_id),
                pq_id,
                project_id,
            )

        pq_fpids = set(id_to_fpid.values())
        pqs_dict = await self._require_pquery_service().get_definitions(pq_fpids
                                                                        )

        # get task type for each predictive query
        fpid_to_task_type = {}
        for pq_fpid, pq_def in pqs_dict.items():
            fpid_to_task_type[pq_fpid] = pq_def.task_type
            if not pq_def.task_type:
                parsed_pq = await self.get_parsed_pq_v2_by_fpid(
                    pq_fpid, project_id, skip_validation)
                fpid_to_task_type[pq_fpid] = parsed_pq.task_type

        result: Dict[Union[
            TrainingJobID,
            BatchPredictionJobID,
            GeneratePredictionTableJobID,
            GenerateTrainTableJobID,
        ], Tuple[str, TaskType]] = {}

        # make a mapping between the job id and predictive query yaml
        for key, val in job_to_pq_fpid.items():
            query = pqs_dict[id_to_fpid[val]].query_str
            task_type = fpid_to_task_type[id_to_fpid[val]]
            result[key] = (query, task_type)

        return result

    async def get_allowed_bp_query_overrides_v2(
        self,
        request_context: RequestContext,
        training_job_id: TrainingJobID,
    ) -> EditableQuery:
        """
        Get the editable predictive query part such as entity/target filter
        with indices.
        """

        # get training job using self.get_training_job. Remove expensive fields
        training_job = await self.get_training_job(
            request_context,
            training_job_id,
            strip_expensive_fields=True,
        )
        if training_job is None:
            raise TrainingJobDoesNotExistError(training_job_id)

        pq_fpid = training_job.pquery_def_fpid
        # pquery_def_fpid refers to the pq_id used in training, this should
        # always be present for UI jobs. In SDK, if using custom training
        # table, this may not be present. In that case, we do not support
        # filter overrides.
        if pq_fpid is None:
            raise ValueError(
                'Predictive query not found for training job. Likely causes '
                ' for these are: \n1. The training job with id '
                f'{training_job_id} was trained using custom training table. '
                'Batch prediction filter overrides are not support with '
                'custom training table.\n2. The training job is associated '
                'with a legacy job which doesnt contain predictive_query '
                'definition')

        parsed_predictive_query = await self.get_parsed_pq_v2_by_fpid(
            pq_fpid, ProjectID(training_job.project_id))

        graph = fail_closed_graph_from_config(
            parsed_predictive_query.config.graph)

        parser = PQLParser()
        query_str = parsed_predictive_query.config.query_string()
        validator = PredictiveQueryValidator(
            **validator_kwargs(graph, query_string=query_str))

        parsed_query = parser.to_parsed_predictive_query(query_str)
        predictive_query, response = validator.validate_predictive_query(
            parsed_query)

        if predictive_query is None:
            raise ValueError(f'Unable to validate predictive query for {id}: '
                             f'{response.error_message()}')

        transformer = BPFilterTransformer()
        editable_query = transformer.get_bp_editable_query(
            predictive_query,
            parsed_predictive_query.task_type,
            map_to_api_graph_definition(graph),
            original_query_text="",
        )

        return editable_query

    async def insert_filter_overrides_v2(
        self,
        request_context: RequestContext,
        training_job_id: TrainingJobID,
        overrides: InsertFilterOverridesRequest,
    ) -> EditableQuery:
        """
        Insert user filter overrides into predictive query and return the new
        string to display.

        Args:
            request_context (RequestContext): The request context.
            training_job_id (TrainingJobID): The ID of the training job for
            which batch predictions is being created, where filter insertion
            is needed
            overrides (InsertFilterOverridesRequest): The filter overrides to
            insert.

        Returns:
            EditableQuery: The editable query object containing the updated
            predictive query if the insertion is possible. If the input
            doesn't result in a valid predictive query, validation error field
            in EditableQuery response will be populated.
        """
        training_job = await self.get_training_job(
            request_context,
            training_job_id,
            strip_expensive_fields=True,
        )
        if training_job is None:
            raise TrainingJobDoesNotExistError(training_job_id)

        pq_fpid = training_job.pquery_def_fpid
        # pquery_def_fpid refers to the pq_id used in training, this should
        # always be present for UI jobs. In SDK, if using custom training
        # table, this may not be present. In that case, we do not support
        # filter overrides.
        if pq_fpid is None:
            raise ValueError(
                'Predictive query not found for training job. Likely causes '
                ' for these are: \n1. The training job with id '
                f'{training_job_id} was trained using custom training table. '
                'Batch prediction filter overrides are not support with '
                'custom training table.\n2. The training job is associated '
                'with a legacy job which doesnt contain predictive_query '
                'definition')

        parsed_predictive_query = await self.get_parsed_pq_v2_by_fpid(
            pq_fpid, ProjectID(training_job.project_id))

        graph = fail_closed_graph_from_config(
            parsed_predictive_query.config.graph)

        parser = PQLParser()
        query_str = parsed_predictive_query.config.query_string()
        validator = PredictiveQueryValidator(
            **validator_kwargs(graph, query_string=query_str))

        parsed_query = parser.to_parsed_predictive_query(query_str)
        predictive_query, response = validator.validate_predictive_query(
            parsed_query)

        if predictive_query is None:
            raise ValueError(f'Unable to validate predictive query for {id}: '
                             f'{response.error_message()}')

        transformer = BPFilterTransformer()
        graph_def = map_to_api_graph_definition(graph)
        task_type = parsed_predictive_query.task_type
        editable_query = transformer.get_bp_editable_query(
            predictive_query,
            task_type,
            graph_def,
            original_query_text="",
        )

        user_changes = {}
        for field in editable_query.editable_fields:
            if field.field_name == 'Entity filter':
                user_changes[field.field_name] = overrides.entity_filter
            if field.field_name == 'Target filter':
                user_changes[field.field_name] = overrides.target_filter
        editable_query_user_response = EditableQueryResponse(
            editable_query, user_changes)
        return transformer.insert_filter_overrides(
            editable_query_user_response,
            graph_def,
            task_type,
        )[0]

    async def _from_custom_tt(
        self,
        project_id: ProjectID,
        custom_train_table: Optional[jobs_api.CustomTrainingTable] = None,
    ) -> Optional[CustomTrainingTable]:
        """Converts the external jobs_api.CustomTrainingTable to the internal
        CustomTrainingTable spec."""
        if custom_train_table is None:
            return None
        (source_table, source_table_name,
         connector_id) = extract_source_table_type(
             custom_train_table.source_table)

        assert source_table_name is not None
        # ``None`` names a table supplied as it stands, with no weight column
        # to read; every layer below here already takes an optional one.
        weight_col = custom_train_table.table_mod_spec.weight_col
        split_col = custom_train_table.table_mod_spec.split_col

        connector_spec = await self._get_custom_table_connector_spec(
            source_table, connector_id, project_id)
        if isinstance(source_table, source_table_api.CustomSQLSourceTable):
            from kumo_core.types import ConnectorType

            from kumo.data.custom_sql import referenced_tables_or_none

            connector = await self._require_state_services(
            ).connectors.get_for_project(
                project_id,
                ConnectorID(connector_id),
                _current_user_ctx.get(),
            )
            connector_type = ConnectorType(connector.config.type.value)
            if connector_type != ConnectorType.DATABRICKS:
                raise ValueError(
                    'Custom SQL training sources are currently supported '
                    'only for Databricks connectors.')
            source_table = dataclasses.replace(
                source_table,
                connector_type=source_table_api.ConnectorType(
                    connector_type.value),
                referenced_tables=(referenced_tables_or_none(
                    source_table.sql, connector_type) or []),
            )

        return CustomTrainingTable(
            connector_spec=connector_spec,
            source_table=source_table,
            weight_col_name=weight_col,
            split_col_name=split_col,
            validated=custom_train_table.validated,
        )

    async def _registered_custom_tt(
        self,
        project_id: ProjectID,
        custom_train_table: jobs_api.CustomTrainingTable,
    ) -> CustomTrainingTable:
        """The internal spec for rows recorded rather than generated.

        The same conversion a training job does. A Databricks physical table
        or Custom SQL query stages through the shared connector export path
        (``ingest_custom_train_table`` -> ``export_table_read_only``) exactly
        like any other warehouse source now -- this used to refuse both,
        because the direct, read-in-place Databricks path projected a
        registered source onto a *generated* table's columns
        (``databricks_custom_train_projection``) that a registration has
        none of. That direct path was retired (kumo#30914): every source,
        Databricks included, is copied into Kumo storage at registration the
        same way, so there is nothing left here that a registration cannot
        read. Custom SQL stays Databricks-only, but that restriction is
        ``_from_custom_tt``'s, not this method's.
        """
        spec = await self._from_custom_tt(project_id, custom_train_table)
        assert spec is not None
        return spec

    async def _get_custom_table_connector_spec(
        self,
        source_table: source_table_api.SourceTableType,
        connector_id: Optional[ConnectorID],
        project_id: ProjectID,
    ) -> ConnectorExecutionRef:
        if connector_id is None:
            raise ConnectorError(
                "A project-owned connector_id and project_id are required")
        connectors = self._require_state_services().connectors
        expected_type = None
        if not isinstance(source_table, source_table_api.CustomSQLSourceTable):
            source_type = getattr(source_table, 'data_source_type', None)
            source_type_name = getattr(source_type, 'name', None)
            if source_type_name is None:
                raise ValueError(
                    'Custom training source has no supported connector type')
            try:
                expected_type = data_source_api.ConnectorType[source_type_name]
            except KeyError:
                raise ValueError(
                    'Custom training source has no supported connector type'
                ) from None
            connector = await connectors.source_table_connector_for_project(
                project_id,
                ConnectorID(connector_id),
                _current_user_ctx.get(),
            )
            if connector is None:
                if not (isinstance(source_table, source_table_api.S3SourceTable)
                        and isinstance(source_table.source,
                                       source_table_api.S3RelativeSource)):
                    raise ValueError(
                        'Project upload connectors require a '
                        'connector-relative custom training source')
                expected_type = None
        return await connectors.execution_ref_for_project(
            _current_user_ctx.get(),
            project_id,
            ConnectorID(connector_id),
            CONNECTOR_GRAPH_ACTIONS,
            expected_type=expected_type,
        )

    async def validate_custom_train_table(
        self,
        custom_table: source_table_api.SourceTableType,
        weight_col: Optional[str],
        orig_train_table_path: str,
        extensive_validation: bool,
        project_id: ProjectID,
        split_col: Optional[str] = None,
    ) -> ValidationResponse:
        """
        Validate custom train table by
        1. Checks if custom_table exists
        2. Check if weight_col exists in source_table, when one is named
        3. Check if table in orig_train_table_path and custom_table have
        same number of rows. Enabled with `extensive_validation`.
        4. [TODO] Check if table in orig_train_table_path and custom_table have
        same column types and name other than weight_col

        Args:
            custom_table: A SourceTableType object with connector and
                table name of the custom training table.
            weight_col: Name of the weight column in the custom training
                table, or ``None`` when the table carries none. Every other
                check still runs.
            split_col: Name of the column carrying each row's split, or
                ``None`` when the run takes its split from the plan.
            orig_train_table_path: Original train table path
        """
        _, _, connector_id = extract_source_table_type(custom_table)
        try:
            reference = await self._get_custom_table_connector_spec(
                custom_table,
                connector_id,
                project_id,
            )

        except ValidationException as e:
            return ValidationResponse(errors=[
                ValidationError(
                    title="Connector not found",
                    message=str(e),
                )
            ])

        return await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            custom_table_validation.validate_custom_train_table_activity,
            [
                reference,
                custom_table,
                weight_col,
                orig_train_table_path,
                extensive_validation,
                split_col,
            ],
            f'validate-custom-table:{project_id}',
        )

    async def create_predictive_query_v2(
        self,
        args: PQueryResource,
        project_id: ProjectID,
    ) -> PredictiveQueryID:
        r"""Create or update a PQuery resource in StateDB, storing only the
        predictive query definition in the pquery definitions map."""
        await self._require_project_access(project_id)
        graph_def = build_graph_definition(args.graph)
        await self._graph_driver.create_full_graph_v2(
            graph_def,
            project_id=project_id,
            skip_validation=True,
        )

        if args.name:
            try:
                pq_id = PredictiveQueryID(args.name)
            except ValueError:
                # Display/template names are user-facing labels, not
                # necessarily PredictiveQueryID-shaped identifiers.
                pq_id = None
            if pq_id is not None and pq_id.fpid:
                raise ValueError('PQuery name is not valid')

        pquery_def = await self._pquery_logic(project_id).build_pquery_def_v2(
            pquery_resource=args, pquery_settings=self._pquery_settings)
        fpid = await self._require_pquery_service().add_definition(pquery_def)
        if args.name is not None:
            await self._require_pquery_service().set_alias(args.name, fpid)
        pq_id = PredictiveQueryID.public(pquery_def.fpid)

        return pq_id

    def convert_validation_response_to_error_details(
        self,
        response: jobs_api.ValidationResponse,
    ) -> jobs_api.ErrorDetails:
        """
                Convert a ValidationResponse object to an ErrorDetails object.

                Args:
                    response (jobs_api.ValidationResponse): The validation
                    response object.

                Returns:
                    jobs_api.ErrorDetails: The error details object.
                """
        items = [
            jobs_api.ErrorDetail(
                type=jobs_api.ErrorType.ERROR,
                title=error.title,
                description=error.message,
                cta=None,
            ) for error in response.errors
        ]
        items.extend([
            jobs_api.ErrorDetail(
                type=jobs_api.ErrorType.INFO,
                title=warn.title,
                description=warn.message,
                cta=None,
            ) for warn in response.warnings
        ])
        items.extend([
            jobs_api.ErrorDetail(
                type=jobs_api.ErrorType.INFO,
                title=info.title,
                description=info.message,
                cta=None,
            ) for info in response.info_items
        ])

        return jobs_api.ErrorDetails(items=items)

    async def validate_predictive_query_v2(
        self,
        args: PQueryValidationRequest,
        project_id: ProjectID,
        check_unique_id: bool = True,
    ) -> ValidationResponse:
        """
        Validate a predictive query using its supplied graph definition.

        The graph definition does not need to be persisted before validation,
        so callers can validate edits before saving the graph.

        Args:
            args (PQueryValidationRequest): Validation request containing
                query string, graph definition, model plan, and query
                dialect. Model plan is optional and only passed when we
                are validating a predictive query with a model plan present.
            check_unique_id (bool): Whether or not we should validate the name
                (for adding the query into the DB) or just the args.
                Defaults to True. (default: :obj:`True`)

        Returns:
            ValidationResponse: If query validation fails at any point,
                it will return an error or a warning.
        """
        await self._require_project_access(project_id)
        try:
            # Convert request to args
            query_name = ""
            if args.pquery_resource.name:
                query_name = args.pquery_resource.name
            if check_unique_id and query_name:
                existing = await self._require_pquery_service(
                ).get_definition_by_id(PredictiveQueryID(query_name))
                if existing is not None:
                    raise ValidationException(
                        title="Predictive query already exists",
                        message=(f"Predictive query with name={query_name} "
                                 "already exists"),
                    )
            val_args = PQueryValidationArgs(
                query_string=args.pquery_resource.query_string,
                graph=args.pquery_resource.graph,
                name=query_name,
                desc=args.pquery_resource.desc,
                model_plan=args.model_plan,
                runtime_options=pquery_runtime_options_or_none(
                    list_distinct_target_is_array=(
                        args.pquery_resource.list_distinct_target_is_array),
                    lp_bp_layout=args.pquery_resource.lp_bp_layout,
                ),
            )

            validation_response = await self._pquery_logic(
                project_id).validate_predictive_query_v2(
                    val_args,
                    pquery_settings=self._pquery_settings,
                    check_unique_id=False,
                )
            return validation_response
        except ValidationException as e:
            return ValidationResponse(errors=[
                ValidationError(
                    title='Invalid Predictive Query',
                    message=(str(e)),
                )
            ])

    async def get_pquery_metadata(
        self,
        pquery_id: PredictiveQueryID,
        project_id: ProjectID,
    ) -> Tuple[TaskType, QueryType]:
        if self._pquery_service is not None:
            pq_def = await self._pquery_service.get_definition_by_id(pquery_id)
            if pq_def is not None:
                graph_def = await self._get_graph_def_for_pquery(
                    pq_def.graph_fpid, project_id)
                if graph_def is None:
                    raise PredictiveQueryDoesNotExistError(pquery_id)
                if (pq_def.task_type is not None
                        and pq_def.query_type is not None):
                    return pq_def.task_type, pq_def.query_type

        _, pq_resource = await self._resolve_pquery_for_launch(
            pquery_id, project_id)
        val_args = PQueryValidationArgs(
            query_string=pq_resource.query_string,
            graph=pq_resource.graph,
            name=pq_resource.name or "",
            desc=pq_resource.desc or "",
            fpid=pquery_id.fpid,
            runtime_options=pquery_runtime_options_or_none(
                list_distinct_target_is_array=(
                    pq_resource.list_distinct_target_is_array),
                lp_bp_layout=pq_resource.lp_bp_layout,
            ),
        )
        parsed_pq, _ = await self._pquery_logic(
            project_id).parse_predictive_query_v2(
                val_args, config=None, auto_trainer_options=None,
                pquery_settings=self._pquery_settings, skip_validation=True)
        return (parsed_pq.task_type, parsed_pq.query_type)

    async def start_generate_pred_table(
        self,
        request: jobs_api.GeneratePredictionTableRequest,
        project_id: ProjectID,
        job_source: Optional[JobSource] = None,
        user: Optional[str] = None,
        is_child_job: Optional[bool] = None,
        training_job_id: Optional[TrainingJobID] = None,
    ) -> GeneratePredictionTableJobID:
        # NOTE: `is_child_job` field is added because currently all pred
        # table jobs are generated in standalone mode. This is a
        # temporary fix until the parent-child relationship is properly
        # restored.
        pq_def_fpid, pq_resource = await self._resolve_pquery_for_launch(
            request.pquery_id, project_id)

        compute_graph_fpid, graph_ver_info = (await
                                              self._handle_snapshot_override_v2(
                                                  pq_resource,
                                                  request.graph_snapshot_id))
        graph_data_ver = (graph_ver_info.graph_data_version
                          if graph_ver_info else None)

        if compute_graph_fpid != pq_resource.graph.fpid:
            pq_def_fpid, pq_resource = await self._fork_pquery_def_for_graph(
                pq_def_fpid, pq_resource, compute_graph_fpid, project_id)

        pq_id = PredictiveQueryID.compose(fpid=pq_def_fpid)

        await self._link_graph_to_project(
            pq_resource.graph.fpid,
            project_id,
        )

        val_args = PQueryValidationArgs(
            query_string=pq_resource.query_string,
            graph=pq_resource.graph,
            name=pq_resource.name or "",
            desc=pq_resource.desc or "",
            fpid=pq_def_fpid,
            runtime_options=pquery_runtime_options_or_none(
                list_distinct_target_is_array=(
                    pq_resource.list_distinct_target_is_array),
                lp_bp_layout=pq_resource.lp_bp_layout,
            ),
        )
        parsed_pq, _ = await self._pquery_logic(
            project_id).parse_predictive_query_v2(
                val_args, config=None, auto_trainer_options=None,
                pquery_settings=self._pquery_settings)
        task_type = parsed_pq.task_type

        forecasting_params = ForecastingParams.from_plan(request.plan)
        if not forecasting_params.has_any:
            if training_job_id is not None:
                forecasting_params = await self._get_forecasting_params(
                    training_job_id)
            else:
                forecasting_params = ForecastingParams.default()

        if forecasting_params.is_forecasting:
            task_type = TaskType.FORECASTING

        pass_through_train_options = forecasting_pass_through_plan(
            task_type,
            parsed_pq.query_type,
            forecast_length=forecasting_params.forecast_length,
            lag_timesteps=forecasting_params.lag_timesteps,
            year_over_year=forecasting_params.year_over_year,
        )

        bp_logic = self._batch_prediction_logic()
        binary_class_threshold = (0.5 if task_type
                                  == TaskType.BINARY_CLASSIFICATION else None)
        job_id = GeneratePredictionTableJobID.make_random_id()

        anchor_time = request.plan.anchor_time
        if isinstance(anchor_time, model_plan_api.InferredType):
            anchor_time = None
        else:
            anchor_time = pd.Timestamp(anchor_time).to_pydatetime()
            assert isinstance(anchor_time, datetime.datetime)

        try:
            job_input = await bp_logic.make_batch_prediction_job_input_v2(
                pq_resource,
                pq_def_fpid,
                BatchPredictionJobID(job_id),
                BatchPredictionJobConfig(
                    predictive_query_id=pq_id,
                    training_job_id=TrainingJobID('unused-job-id'),
                    task_type=task_type,
                    num_classes_to_return=None,
                    binary_classification_threshold=binary_class_threshold,
                    prediction_time=anchor_time,
                    graph_id=GraphIDv2.compose(fpid=compute_graph_fpid),
                    graph_data_version=graph_data_ver,
                    generate_pred_table_job_id=job_id,
                    lag_timesteps=forecasting_params.lag_timesteps,
                    year_over_year=forecasting_params.year_over_year,
                    forecast_length=forecasting_params.forecast_length,
                ),
                train_table_options=pass_through_train_options,
                project_id=project_id,
            )
        except Exception as ex:
            if isinstance(ex, ValueError):
                raise
            raise ValueError(f'Invalid batch prediction job input due to '
                             f'{type(ex).__name__}: {ex}') from ex

        job_id = await self._ml_job_backend.start_generate_pred_table(
            job_input,
            project_id=project_id,
            graph_version_info=graph_ver_info,
            job_source=job_source,
            user=user,
            is_child_job=is_child_job,
        )

        tags = self.create_ordered_job_tags(
            request.job_tags,
            pq_resource.name or "NA",
            job_input.config.pquery_def_fpid,
        )
        await self._apply_job_tags(JobType.GENERATE_PREDICTION_TABLE_JOB,
                                   job_id, tags)

        return job_id

    async def _get_forecasting_params(
        self,
        training_job_id: Optional[TrainingJobID],
    ) -> ForecastingParams:
        """Extract forecasting parameters from the training config.

        Tries the generate-train-table job first (authoritative source).
        Falls back to the training job's advanced_options if the gen-TT
        job is unavailable (e.g. SDK path with pre-generated train tables).
        Raises :class:`ValidationException` if the gen-TT job ID exists
        but the job entry itself is missing.
        """
        if training_job_id is None:
            return ForecastingParams.default()

        gen_tt_entry = await self._ml_job_backend.get_training_gen_tt_job(
            training_job_id)
        if gen_tt_entry is not None:
            return ForecastingParams.from_plan(gen_tt_entry.config.plan)

        # Fallback: gen-TT job not linked; try training job plan directly.
        training_plan = (await self._ml_job_backend.
                         get_training_job_training_table_plan(training_job_id))
        if training_plan is not None:
            return ForecastingParams.from_plan(training_plan)

        return ForecastingParams.default()

    async def _get_latest_training_job_for_pquery_from_services(
        self,
        pq_id: Union[str, PredictiveQueryID],
        *,
        project_ids: set[ProjectID] | None,
    ) -> Optional[TrainingJobID]:
        if isinstance(pq_id, PredictiveQueryID):
            name, fpid = pq_id.name, pq_id.fpid
        else:
            name, fpid = PredictiveQueryID.parse(pq_id)
        if fpid is not None:
            filters = {
                str(TAG_KEY_PQUERY_ID): str(JobTagValue.of_pquery_id(fpid))
            }
        elif name is not None:
            filters = {
                str(TAG_KEY_PQUERY_NAME): str(JobTagValue.of_pquery_name(name))
            }
        else:
            return None
        tagged_jobs = await self._list_job_tags(JobType.TRAINING_JOB, filters)
        if not tagged_jobs:
            return None
        rows = await self._require_ml_job_service().list_training_job_rows(
            job_ids={TrainingJobID(job_id)
                     for job_id in tagged_jobs},
            project_ids=project_ids,
            sort_created_at="desc",
            limit=1,
        )
        return rows[0].id if rows else None

    async def _training_job_ids_for_pquery_from_services(
        self,
        fpid: FPID,
        *,
        project_ids: set[ProjectID] | None,
    ) -> List[TrainingJobID]:
        filters = {str(TAG_KEY_PQUERY_ID): str(JobTagValue.of_pquery_id(fpid))}
        tagged_jobs = await self._list_job_tags(JobType.TRAINING_JOB, filters)
        rows = await self._require_ml_job_service().list_training_job_rows(
            job_ids={TrainingJobID(job_id)
                     for job_id in tagged_jobs},
            project_ids=project_ids,
            sort_created_at="asc",
        )
        return [row.id for row in rows]

    async def _cancel_inflight_training_jobs(
        self,
        ctx: RequestContext,
        training_job_ids: List[TrainingJobID],
    ) -> bool:
        if not training_job_ids:
            return True

        training_jobs_by_id = await self._ml_job_backend.get_training_jobs(
            ctx, set(training_job_ids))

        all_cancelled = True
        for training_job_id in training_job_ids:
            training_job = training_jobs_by_id.get(training_job_id)
            if (training_job is not None
                    and training_job.execution.status.is_submitted):
                cancelled = await self._ml_job_backend.cancel_training_job(
                    training_job_id)
                all_cancelled = all_cancelled and cancelled
        return all_cancelled

    async def _resolve_pquery_for_launch(
        self,
        pquery_name_or_id: str,
        project_id: ProjectID,
    ) -> Tuple[FPID, PQueryResource]:
        """Resolve a wire pquery_id (name or fpid form) to (FPID, PQueryResource).

        The single name->fpid boundary for v2 launch paths: aliases are
        resolved here and never propagate further. Raises
        PredictiveQueryDoesNotExistError if neither form resolves.
        """
        try:
            parsed = PredictiveQueryID.from_public(pquery_name_or_id)
            _, fpid = PredictiveQueryID.parse(parsed)
        except ValueError:
            fpid = None

        if fpid is None:
            fpid = await self._require_pquery_service().get_alias(
                pquery_name_or_id)

        if fpid is None:
            raise PredictiveQueryDoesNotExistError(pquery_name_or_id)

        pq_def = await self._require_pquery_service().get_definition(fpid)
        if pq_def is None:
            raise PredictiveQueryDoesNotExistError(pquery_name_or_id)

        graph_def = await self._get_graph_def_for_pquery(
            pq_def.graph_fpid, project_id)
        if graph_def is None:
            raise ValueError(f'Graph not found for pquery {pquery_name_or_id}')

        return fpid, PQueryResource(
            name=pq_def.pquery_name,
            query_string=pq_def.query_str,
            desc="",
            graph=graph_def,
            runtime_options=pquery_runtime_options_or_none(
                list_distinct_target_is_array=(
                    pq_def.list_distinct_target_is_array),
                lp_bp_layout=pq_def.lp_bp_layout,
            ),
        )

    def _get_parent_id(
        self,
        id: Union[GenerateTrainTableJobID, GeneratePredictionTableJobID],
    ) -> Union[TrainingJobID, BatchPredictionJobID]:
        """Get the parent job ID for a given job ID.

        Args:
            id (Union[
                GenerateTrainTableJobID,
                GeneratePredictionTableJobID,
            ]): The job ID to get the parent ID for.

        Returns:
            Union[TrainingJobID, BatchPredictionJobID]: The parent job ID.
        """
        if isinstance(id, GenerateTrainTableJobID):
            return TrainingJobID.as_parent_of(id)
        elif isinstance(id, GeneratePredictionTableJobID):
            return BatchPredictionJobID.as_parent_of(id)

    async def _get_job_dir(
        self, id: Union[
            GenerateTrainTableJobID,
            GeneratePredictionTableJobID,
            TrainingJobID,
            BatchPredictionJobID,
        ]
    ) -> Optional[str]:
        """Get the job directory for a given job ID. Resolve to parent if the
        job directory does not exist.

        Args:
            id (Union[
                GenerateTrainTableJobID,
                GeneratePredictionTableJobID,
                TrainingJobID,
                BatchPredictionJobID,
            ]): The job ID to get the job directory for.

        Returns:
            Optional[str]: The job directory if it exists, otherwise None.
        """
        job_dir = get_job_dir(get_job_base_dir(), id)
        job_dir_exists = await fsutil.exists_async(job_dir)
        if not job_dir_exists:
            log.warning(f"Job directory {job_dir} does not exist")
            if isinstance(id, GenerateTrainTableJobID) or isinstance(
                    id, GeneratePredictionTableJobID):
                parent_id = self._get_parent_id(id)
                job_dir = get_job_dir(get_job_base_dir(), parent_id)
                job_dir_exists = await fsutil.exists_async(job_dir)
                if not job_dir_exists:
                    return None
        return job_dir

    async def get_errors_for_job(
        self,
        id: Union[
            # TODO(Viman): GraphSnapshotID extends from str as opposed to
            # JobID, need to fix this before including it here so typesafe
            # linter is not highlighting it
            GenerateTrainTableJobID,
            GeneratePredictionTableJobID,
            TrainingJobID,
            BatchPredictionJobID,
        ],
    ) -> jobs_api.ErrorDetails:
        """
        Returns Errors and Info generated from the job in their corresponding
        jobs directory. This is typically jobs directory followed by the job
        id.

        Looks for errors in the following order (first non-empty wins):
        1. StateDB validation field
        2. ValidationResponse file
        3. StateDB failure_info.message
        4. StateDB log.detail

        Note: Multiple sources exist due to tech debt. Ideally, StateDB would be
        the single source of truth, but historically errors were also written
        to a ValidationResponse file. This file isn't written in all failure
        scenarios (e.g., early exceptions), so we cascade through sources to
        ensure errors are always surfaced when available.

        Arguments:
            id (Union[GenerateTrainTableJobID, GeneratePredictionTableJobID,
                TrainingJobID, BatchPredictionJobID,]): id of the job for which
                Errors and Info is required to be fetched.

        Returns:
            ErrorDetails containing errors and info associated with the
            workflow corresponding to input id.
        """
        # For GenerateTrainTableJobID and GeneratePredictionTableJobID,
        # try StateDB first as it's the most reliable source of truth.
        if isinstance(id,
                      (GenerateTrainTableJobID, GeneratePredictionTableJobID)):
            execution = await self._ml_job_backend.get_generate_table_job_execution(
                id)

            if execution is not None:
                # 1. Try validation from StateDB
                if execution.validation is not None:
                    err_warn_details = _validation_to_warn_details(
                        execution.validation)
                    if err_warn_details:
                        return self._transform_to_error_details(
                            err_warn_details)

                # 2. Try ValidationResponse file
                job_dir = await self._get_job_dir(id)
                if job_dir:
                    disk_err_warn_details = await _get_warn_details(
                        [HomeDirManager(job_dir).validation_path(id)])
                    if disk_err_warn_details and disk_err_warn_details[0]:
                        return self._transform_to_error_details(
                            disk_err_warn_details[0])

                # 3. Try failure_info.message from StateDB
                if execution.failure_info is not None and execution.failure_info.message:
                    return jobs_api.ErrorDetails(items=[
                        jobs_api.ErrorDetail(
                            title=None,
                            description=execution.failure_info.message,
                            cta=None,
                            type=jobs_api.ErrorType.ERROR,
                        )
                    ])

                # 4. Try log.detail from StateDB for failed jobs only. DONE
                # generated-table jobs store success text in log.detail.
                if (execution.status == JobStatus.FAILED
                        and execution.log.detail):
                    return jobs_api.ErrorDetails(items=[
                        jobs_api.ErrorDetail(
                            title=None,
                            description=execution.log.detail,
                            cta=None,
                            type=jobs_api.ErrorType.ERROR,
                        )
                    ])

            # Job not found in StateDB, return empty
            return jobs_api.ErrorDetails(items=[])

        # For other job types, use the original disk-based logic.
        # Note we do not return Object non existent error here if the dir is
        # not found since the dir could simply not be generated as workflow
        # ran successfully. Many of these workflows only conditionally create
        # ValidationResponse unlike kumo v0 workflows.
        job_dir = await self._get_job_dir(id)
        if not job_dir:
            return jobs_api.ErrorDetails(items=[])

        err_warn_details = await _get_warn_details(
            [HomeDirManager(job_dir).validation_path(id)])
        if len(err_warn_details) == 0:
            return jobs_api.ErrorDetails(items=[])
        # We are querying for 1 job only so at most can get one entry
        assert len(err_warn_details) == 1

        return self._transform_to_error_details(err_warn_details[0])

    def _transform_to_error_details(
        self,
        err_warn_details: List[ErrWarnDetails],
    ):
        """
        Transform the ErrorWarnDetails to the v2 API object ErrorDetails.

        Arguments:
        err_warn_details (List[ErrWarnDetails]): List of ErrWarnDetails to be
            transformed.
        """
        items = []
        for err in err_warn_details:
            type = (jobs_api.ErrorType.ERROR
                    if err.is_error else jobs_api.ErrorType.INFO)
            cta = None if err.cta is None else jobs_api.ErrorCTA(
                name=err.cta.name,
                url=err.cta.url,
            )

            error_detail = jobs_api.ErrorDetail(
                title=err.title,
                description=err.description,
                cta=cta,
                type=type,
            )
            items.append(error_detail)

        return jobs_api.ErrorDetails(items)

    async def start_generate_train_table(
        self,
        request: jobs_api.GenerateTrainTableRequest,
        project_id: ProjectID,
        job_source: Optional[JobSource] = None,
    ) -> GenerateTrainTableJobID:
        """Launch a job to generate training table."""
        pq_def_fpid, pq_resource = await self._resolve_pquery_for_launch(
            request.pquery_id, project_id)

        compute_graph_fpid, graph_version_info = (
            await self._handle_snapshot_override_v2(pq_resource,
                                                    request.graph_snapshot_id))

        if compute_graph_fpid != pq_resource.graph.fpid:
            pq_def_fpid, pq_resource = await self._fork_pquery_def_for_graph(
                pq_def_fpid, pq_resource, compute_graph_fpid, project_id)

        await self._link_graph_to_project(
            pq_resource.graph.fpid,
            project_id,
        )

        graph_data_compute_input = await self._get_graph_data_compute_input(
            GraphIDv2.compose(fpid=compute_graph_fpid), project_id)

        training_spec, parsed_pq = await self._pquery_logic(
            project_id)._make_training_spec_v2(
                pq_resource,
                self._pquery_settings,
                pq_def_fpid=pq_def_fpid,
                graph_version_info=graph_version_info,
            )

        # Here model plan is just a container for train-table-generation plan
        training_spec.advanced_options = ModelPlan()

        # TODO(siyang,manan): For now we use json serde to convert dataclass
        # of model plan from public API to internal. In the future, internal
        # dataclasses maybe reuse/inherit the public API dataclasses.
        plan = from_json(to_json(request.plan), TrainingTableGenerationPlan)
        training_spec.advanced_options.training_table_generation = plan

        # Validate train-table-generation plan.
        pquery = parsed_pq.predictive_query

        errors, _ = plan.validate(
            task_type=parsed_pq.task_type,
            query_type=parsed_pq.query_type,
            has_train_table_time_col=parsed_pq.has_train_table_time_col,
            run_mode=RunMode.NORMAL,
            graph=pquery.graph.config(),
            entity_table_names=pquery.entity_table_names,
            target_to_entity_paths=pquery.target_to_entity_paths,
        )
        if errors:
            raise ModelPlanValueError(errors)

        # Rows the caller supplies rather than rows this job generates. The
        # job is otherwise resolved identically -- same query, same graph
        # snapshot, same plan -- because those are what a training run reads
        # back off it, and a registered table has to name them the same way.
        if request.custom_train_table is not None:
            training_spec.custom_train_table = (await
                                                self._registered_custom_tt(
                                                    project_id,
                                                    request.custom_train_table))

        job_id = await self._ml_job_backend.start_generate_train_table(
            training_spec,
            graph_data_compute_input,
            pq_def_fpid,
            project_id=project_id,
            graph_version_info=graph_version_info,
            job_source=job_source,
            # As the caller named it. `training_spec` carries the internal
            # conversion, which drops everything the run does not read and
            # carries a connector token; the record has to answer what was
            # registered, so it keeps the request's own shape.
            registered_custom_table=request.custom_train_table,
        )

        tags = {
            JobTagKey(k): JobTagValue(v)
            for k, v in request.job_tags.items()
        }
        tags[TAG_KEY_PQUERY_NAME] = JobTagValue.of_pquery_name(pq_resource.name
                                                               or "NA")
        tags[TAG_KEY_PQUERY_ID] = JobTagValue.of_pquery_id(pq_def_fpid)
        await self._apply_job_tags(
            JobType.GENERATE_TRAIN_TABLE_JOB,
            job_id,
            tags,
        )

        return job_id

    async def validate_batch_prediction_inputs_v2(
        self,
        ctx: RequestContext,
        training_job_id: TrainingJobID,
        pred_table_job_id: Optional[GeneratePredictionTableJobID],
        graph_id: GraphIDv2,
        predictive_query_id: Optional[PredictiveQueryID],
    ) -> ValidationResponse:
        # 1. Ensure that the training job exists, and has been completed:
        training_job: Optional[TrainingJobResponse] = (
            await self.get_training_job(
                ctx,
                training_job_id,
                strip_expensive_fields=True,
            ))
        if not training_job:
            message = TRAINING_JOB_NOT_FOUND_MESSAGE.format(training_job_id)
            return ValidationResponse(errors=[
                ValidationError(title=TRAINING_JOB_NOT_FOUND_TITLE,
                                message=message)
            ])
        else:
            status = training_job.execution.status
            if status != JobStatus.DONE:
                message = TRAINING_JOB_INCOMPLETE_MESSAGE.format(
                    training_job_id, status)
                return ValidationResponse(errors=[
                    ValidationError(title=TRAINING_JOB_INCOMPLETE_TITLE,
                                    message=message)
                ])

        assert training_job
        assert training_job.execution.status == JobStatus.DONE
        project_id = ProjectID(training_job.project_id)

        training_job_pq_id: Optional[PredictiveQueryID] = (
            await self.get_training_job_pquery_id_v2(
                ctx,
                training_job_id,
            ))

        if not training_job_pq_id:
            message = TRAINING_JOB_NO_PQUERY_MESSAGE
            return ValidationResponse(errors=[
                ValidationError(title=TRAINING_JOB_NO_PQUERY_TITLE,
                                message=message)
            ])

        assert training_job_pq_id
        training_job_pquery: PQueryResource = (await
                                               self.get_predictive_query_v2(
                                                   pq_id=training_job_pq_id,
                                                   project_id=project_id))

        # 2. Ensure that the graph node and edge types are identical between
        # training and prediction. It is likely sufficient to loosen the
        # condition to simply ensure that the prediction node and edge types
        # are a subset of those present at training, but this should be fine
        # for now:
        training_job_graph_def = training_job_pquery.graph
        assert graph_id.fpid
        graph_def, _, _ = (await self._graph_driver.get_graph_v2(
            graph_id.fpid, project_id))
        prediction_job_graph_def = graph_def.graph

        training_node_types = set(training_job_graph_def.tables.keys())
        prediction_node_types = set(prediction_job_graph_def.tables.keys())
        response = ValidationResponse()
        if training_node_types != prediction_node_types:
            message = INCOMPATIBLE_GRAPH_MESSAGE.format(training_job_id,
                                                        training_node_types,
                                                        prediction_node_types)
            response.errors.append(
                ValidationError(title=INCOMPATIBLE_GRAPH_TITLE,
                                message=message))

        # Ignore column groups that contains less than 2 column keys,
        # due to legacy UI bug that accidentally allowed creating graph with
        # such meaningless column groups.
        # See https://github.com/kumo-ai/kumo/issues/22767
        training_edge_types = set(cg for cg in training_job_graph_def.col_groups
                                  if len(cg.columns) >= 2)
        prediction_edge_types = set(
            cg for cg in prediction_job_graph_def.col_groups
            if len(cg.columns) >= 2)
        if training_edge_types != prediction_edge_types:
            message = INCOMPATIBLE_EDGES_MESSAGE.format(training_job_id,
                                                        training_edge_types,
                                                        prediction_edge_types)
            response.errors.append(
                ValidationError(title=INCOMPATIBLE_EDGES_TITLE,
                                message=message))
        if not response.ok:
            return response

        # Check table columns and encoders for each table:
        for table_name, train_table_config in (
                training_job_graph_def.tables.items()):
            prediction_table_config = prediction_job_graph_def.tables.get(
                table_name)

            if train_table_config.pkey != prediction_table_config.pkey:
                message = INCOMPATIBLE_PRIMARY_KEY_MESSAGE.format(
                    training_job_id, table_name, train_table_config.pkey,
                    prediction_table_config.pkey)
                response.errors.append(
                    ValidationError(title=INCOMPATIBLE_PRIMARY_KEY_TITLE,
                                    message=message))

            if train_table_config.time_col != (
                    prediction_table_config.time_col):
                message = INCOMPATIBLE_TIME_COLUMN_MESSAGE.format(
                    training_job_id, table_name, train_table_config.time_col,
                    prediction_table_config.time_col)
                response.errors.append(
                    ValidationError(title=INCOMPATIBLE_TIME_COLUMN_TITLE,
                                    message=message))

            if train_table_config.end_time_col != (
                    prediction_table_config.end_time_col):
                message = INCOMPATIBLE_END_TIME_COLUMN_MESSAGE.format(
                    training_job_id, table_name,
                    train_table_config.end_time_col,
                    prediction_table_config.end_time_col)
                response.errors.append(
                    ValidationError(title=INCOMPATIBLE_END_TIME_COLUMN_TITLE,
                                    message=message))

            train_table_columns = {
                col.name: col
                for col in train_table_config.cols
            }
            prediction_table_columns = {
                col.name: col
                for col in prediction_table_config.cols
            }
            train_table_column_names = set(train_table_columns.keys())
            prediction_table_column_names = set(prediction_table_columns.keys())
            if train_table_column_names != prediction_table_column_names:
                message = INCOMPATIBLE_COLUMNS_MESSAGE.format(
                    training_job_id, table_name, train_table_column_names,
                    prediction_table_column_names)
                response.errors.append(
                    ValidationError(title=INCOMPATIBLE_COLUMNS_TITLE,
                                    message=message))
            for col_name in (train_table_column_names
                             & prediction_table_column_names):
                train_col = train_table_columns[col_name]
                prediction_col = prediction_table_columns[col_name]
                if (train_col.dtype != prediction_col.dtype
                        or train_col.stype != prediction_col.stype):
                    message = INCOMPATIBLE_COLUMN_TYPES_MESSAGE.format(
                        training_job_id, table_name, col_name, train_col.dtype,
                        train_col.stype, col_name, prediction_col.dtype,
                        prediction_col.stype)
                    response.errors.append(
                        ValidationError(title=INCOMPATIBLE_COLUMN_TYPES_TITLE,
                                        message=message))
        if not response.ok:
            return response

        # 3. Training job PQ compatibility (check after graph, as differing
        # graphs will always create a PQ FPID diff):
        # TODO(manan): raise error if different task types
        if predictive_query_id:
            prediction_job_pquery: PQueryResource = (
                await self.get_predictive_query_v2(
                    pq_id=predictive_query_id,
                    project_id=project_id,
                ))
            training_pq_fpid = training_job_pq_id.fpid
            training_pq_str = training_job_pquery.query_string
            prediction_pq_fpid = predictive_query_id.fpid
            prediction_pq_str = prediction_job_pquery.query_string
            if ((training_pq_fpid != prediction_pq_fpid)
                    and (training_pq_str != prediction_pq_str)):
                message = (
                    f"Training job {training_job_id} was trained with "
                    f"predictive query {training_pq_str}, which differs "
                    f"from the prediction job predictive query "
                    f"{prediction_pq_str}. Please ensure this is correct "
                    f"before proceeding.")
                response.warnings.append(
                    ValidationWarning(title="Mismatched Predictive Queries",
                                      message=message))

        # Get the prediction table job. If this is None, then the user is using
        # a custom prediction table
        pred_table_job = None
        if pred_table_job_id is not None:
            pred_table_job: Optional[
                jobs_api.GeneratePredictionTableJobResource] = (
                    await
                    self.get_generate_pred_table_job(ctx, pred_table_job_id,
                                                     project_id))
            # A still-running gen-pred-table job is fine: the BP workflow waits
            # for it to finish before materializing. Only a job that reached a
            # failed terminal state (FAILED/CANCELLED) can never yield a usable
            # prediction table, so reject those at submit with a clear message
            # rather than letting the BP job wait forever on a dead dependency.
            pred_table_status = pred_table_job.job_status_report.status
            if pred_table_status in (JobStatus.FAILED, JobStatus.CANCELLED):
                message = PREDICTION_TABLE_JOB_FAILED_MESSAGE.format(
                    pred_table_job_id, pred_table_status.value)
                return ValidationResponse(errors=[
                    ValidationError(title=PREDICTION_TABLE_JOB_FAILED_TITLE,
                                    message=message)
                ])

        train_table_job = None
        if training_job.generate_train_table_job_id is not None:
            train_table_job = await self.get_training_table_generation_job(
                ctx,
                GenerateTrainTableJobID(
                    training_job.generate_train_table_job_id))
        assert train_table_job is not None

        # 4. Forecasting parameter compatibility between the training and
        # prediction jobs:
        if pred_table_job is not None:
            ttg_plan = train_table_job.config.plan
            if ttg_plan is not None:
                iomttn = inferred_or_missing_type_to_none
                forecasting_param_validation = (
                    validate_forecasting_parameters_consistency(
                        training_forecast_length=iomttn(
                            ttg_plan.forecast_length),
                        training_lag_timesteps=iomttn(ttg_plan.lag_timesteps),
                        training_year_over_year=iomttn(ttg_plan.year_over_year),
                        prediction_forecast_length=iomttn(
                            pred_table_job.config.plan.forecast_length),
                        prediction_lag_timesteps=iomttn(
                            pred_table_job.config.plan.lag_timesteps),
                        prediction_year_over_year=iomttn(
                            pred_table_job.config.plan.year_over_year),
                    ))
                response.merge(forecasting_param_validation)

        return response

    async def start_batch_prediction_job_v2(
        self,
        ctx: RequestContext,
        request: jobs_api.BatchPredictionRequest,
        project_id: ProjectID,
        job_source: Optional[JobSource] = None,
        filter_overrides: Optional[BatchPredictionFilterOverrides] = None,
        custom_tags: Optional[List[str]] = None,
        user: Optional[str] = None,
    ) -> Tuple[Optional[BatchPredictionJobID], ValidationResponse]:
        """
        V2 implementation for start BP job, target for SDK/API usage.
        Compared to `start_training_job`, the main differences are:
        1. Takes a structured `BatchPredictionRequest` as argument.
        2. Allow concurrent BP jobs for the same pquery, typically using
           different trainer/model IDs.
        3. Support specifying a fixed graph data snapshot version.
        4. Support decoupling of pred-table-generation; i.e.,
           user can start a generate-pred-table job first, later on pass the
           generate-pred-table job's id to start a BP job.

        This API is intended to be called for the Public REST API/SDK purpose.
        """
        # Training Job ID:
        training_job_id = TrainingJobID(request.model_training_job_id)
        (request.pred_table_source, custom_pred_table_connector) = (
            await self._authorize_custom_prediction_table_source(
                request.pred_table_source, project_id))

        # Predictive Query ID (if not passed, will use Training PQID):
        if request.pred_table_job_id:
            pquery_id = PredictiveQueryID.from_public(
                (await
                 self.get_generate_pred_table_job(ctx,
                                                  request.pred_table_job_id,
                                                  project_id)).config.pquery_id)
        else:
            pquery_id = await self.get_training_job_pquery_id_v2(
                ctx, training_job_id)
        if pquery_id is None:
            raise ValueError(
                'Unable to find training job associated with the model '
                f'training ID={training_job_id}')

        pq_def_fpid, pq_resource = await self._resolve_pquery_for_launch(
            pquery_id, project_id)

        graph_fpid, _ = await self._handle_snapshot_override_v2(
            pq_resource, request.graph_snapshot_id)
        graph_id = GraphIDv2.public(graph_fpid)

        legacy_filter_overrides = None
        if filter_overrides is not None:
            legacy_filter_overrides = self._convert_to_legacy_filter_overrides(
                filter_overrides)

        # Parse the external kumoapi request into the internal, enriched
        # request type. Passing outputs/predict_options as dicts lets the
        # local discriminated union reconstruct the concrete
        # rest.config.jobs output subclasses (S3PredictionOutput, etc.) that
        # validation and `to_config` depend on.
        rest_request = RestBatchPredictionRequest(
            outputs=[dataclasses.asdict(o) for o in request.outputs],
            predict_options={
                **dataclasses.asdict(request.predict_options),
                "prediction_time":
                request.prediction_time,
            },
            batch_prediction_workflow_id=getattr(
                request, "batch_prediction_workflow_id", None),
            explanations=request.explanations,
        )

        validation_response = await self.validate_batch_prediction_inputs_v2(
            ctx=ctx,
            training_job_id=training_job_id,
            pred_table_job_id=request.pred_table_job_id,
            graph_id=graph_id,
            predictive_query_id=(pquery_id
                                 if request.pred_table_job_id else None),
        )

        if graph_fpid != pq_resource.graph.fpid:
            pq_def_fpid, pq_resource = await self._fork_pquery_def_for_graph(
                pq_def_fpid, pq_resource, graph_fpid, project_id)

        await self._link_graph_to_project(
            pq_resource.graph.fpid,
            project_id,
        )

        val_args = PQueryValidationArgs(
            query_string=pq_resource.query_string,
            graph=pq_resource.graph,
            name=pq_resource.name or "",
            desc=pq_resource.desc or "",
            fpid=pq_def_fpid,
            runtime_options=pquery_runtime_options_or_none(
                list_distinct_target_is_array=(
                    pq_resource.list_distinct_target_is_array),
                lp_bp_layout=pq_resource.lp_bp_layout,
            ),
        )
        parsed_pq, _ = await self._pquery_logic(
            project_id).parse_predictive_query_v2(
                val_args, config=None, auto_trainer_options=None,
                pquery_settings=self._pquery_settings)

        task_type = parsed_pq.task_type
        forecasting_params = await self._get_forecasting_params(training_job_id)
        if forecasting_params.is_forecasting:
            task_type = TaskType.FORECASTING
        pquery = parsed_pq.predictive_query

        logic_ro = self._batch_prediction_logic()

        validation_response.merge(await
                                  logic_ro.validate_batch_prediction_args_v2(
                                      ctx,
                                      rest_request,
                                      self._allow_local_batch_pred_paths,
                                      task_type,
                                      parsed_pq.query_type,
                                      project_id,
                                  ))

        if not validation_response.ok:
            return (None, validation_response)

        logic = self._batch_prediction_logic()
        bp_job_id = await logic.create_batch_prediction_job_v2(
            ctx,
            rest_request,
            pq_resource,
            pq_def_fpid,
            task_type,
            parsed_pq.query_type,
            training_job_id,
            predictive_query_id=pquery_id,
            filter_override_args=legacy_filter_overrides,
            top_k=pquery.validated_predictive_query.top_k,
            generate_pred_table_job_id=request.pred_table_job_id,
            custom_pred_table_source=request.pred_table_source,
            custom_pred_table_connector=custom_pred_table_connector,
            graph_snapshot_id=request.graph_snapshot_id,
            job_source=job_source,
            custom_tags=custom_tags,
            user=user,
            project_id=project_id,
        )

        tags = {
            JobTagKey(k): JobTagValue(v)
            for k, v in request.job_tags.items()
        }
        await self._apply_job_tags(JobType.BATCH_PREDICTION_JOB, bp_job_id,
                                   tags)
        if (request.pred_table_job_id is None
                and request.pred_table_source is None):
            # generate-prediction-table job is implicitly spawned as child
            # job of batch prediction job, so we need to update job_tags
            # (indexing).
            pred_table_job_id = GeneratePredictionTableJobID.as_child_of(
                bp_job_id)
            tags[TAG_KEY_PARENT_JOB] = JobTagValue(bp_job_id)
            await self._apply_job_tags(
                JobType.GENERATE_PREDICTION_TABLE_JOB,
                pred_table_job_id,
                tags,
            )

        return (bp_job_id, validation_response)

    async def _authorize_custom_prediction_table_source(
        self,
        source: jobs_api.PredictionTableSource | None,
        project_id: ProjectID,
    ) -> tuple[jobs_api.PredictionTableSource | None, ConnectorExecutionRef
               | None]:
        """Authorize and canonicalize a custom object-store coordinate."""
        if source is None:
            return None, None
        connector_id = ConnectorID(source.connector_id)
        principal = _current_user_ctx.get()
        connector = await self._require_state_services(
        ).connectors.get_for_project(project_id, connector_id, principal)
        config = connector.config
        expected_type = data_source_api.ConnectorType[source.storage_type.name]
        if config.type != expected_type:
            raise ValueError(
                'Custom prediction source type does not match its connector')
        if expected_type == data_source_api.ConnectorType.DATABRICKS:
            if not isinstance(
                    config, data_source_api.DatabricksConnectorResourceConfig):
                raise ValueError(
                    'Custom prediction source requires a Databricks connector')
        else:
            if not isinstance(config,
                              (data_source_api.S3ConnectorResourceConfig,
                               data_source_api.GCSConnectorResourceConfig,
                               data_source_api.ADLSConnectorResourceConfig)):
                raise ValueError(
                    'Custom prediction tables require an object-store or '
                    'Databricks connector')
            canonical_source = object_store_source_for_connector(
                config, source.source)
            source = dataclasses.replace(source, source=canonical_source)
        reference = await self._require_state_services(
        ).connectors.execution_ref_for_project(
            principal,
            project_id,
            connector_id,
            CONNECTOR_GRAPH_ACTIONS,
            expected_type=expected_type,
        )
        return source, reference

    def _convert_to_legacy_filter_overrides(
            self, filter_overrides: BatchPredictionFilterOverrides
    ) -> FilterOverrideArgs:
        return FilterOverrideArgs(
            entity_filter_override=filter_overrides.entity_filter,
            target_filter_override=filter_overrides.target_filter,
        )

    async def get_training_progress(
            self, ctx: RequestContext, training_job_id: TrainingJobID,
            project_id: ProjectID) -> jobs_api.AutoTrainerProgress:
        await self._require_training_job_in_project(ctx, training_job_id,
                                                    project_id)
        training_job_path = await self._ml_job_backend.get_job_home_dir(
            training_job_id)
        return await self._load_training_progress_from_job_path.coroutine(
            training_job_path)

    async def cancel_training_job(
        self,
        ctx: RequestContext,
        training_job_id: TrainingJobID,
        project_id: ProjectID,
    ) -> bool:
        await self._require_training_job_in_project(ctx, training_job_id,
                                                    project_id)
        return await self._ml_job_backend.cancel_training_job(training_job_id)

    @blocking_method()
    def _load_training_progress_from_job_path(
            self, training_job_path: str) -> AutoTrainerProgress:
        home = HomeDirManager(training_job_path)
        task_name = home.first_task_name()
        if task_name is None:
            # The task dir appears only once training starts, so a poll before
            # that (or after the job dir is reaped) finds no task. Return empty
            # progress rather than raising — a hard error is a non-retriable RPC
            # failure that wedges the SDK attach() poll loop (job unaffected).
            # This holds regardless of TT-NG: the "not ready yet" state is the
            # same, and the public GET /v1/training_jobs/{id}/progress endpoint
            # otherwise returns HTTP 500 for any poll during QUEUED/early
            # RUNNING (observed on a non-TT-NG single-host stack).
            return AutoTrainerProgress()
        return AutoTrainerProgress.load(home.get_automl_dir(task_name))

    async def start_training_job_v2(
        self,
        ctx: RequestContext,
        request: TrainingJobInternalRequest,
        project_id: ProjectID,
        job_source: Optional[JobSource] = None,
        training_table_generation_plan: Optional[
            TrainingTableGenerationPlan] = None,
        user_info: Optional[str] = None,
        reserved_training_job_id: Optional[TrainingJobID] = None,
    ) -> TrainingJobID:
        """
        V2 implementation for start training job, target for SDK/API usage.
        Compared to `start_training_job`, the main differences are:
        1. Takes a structured `TrainingJobInternalRequest` as argument,
           carrying an internal `ModelPlan` object, as opposed to a yaml of
           ModelPlan.
        2. Allow concurrent training jobs for the same pquery (still subject
           to global job concurrency limit), as opposed to forbidding this.
        3. Support training over a specific/fixed graph data snapshot.
        4. Support decoupling of train-table-generation from training; i.e.,
           user can start a generate-train-table job first, later on pass the
           generate-train-table job's id to start a training job.

        This API is intended to be called for the Public REST API/SDK purpose.

        Args:
            request (TrainingJobInternalRequest): The training job request
            containing pquery_id, graph_snapshot, model_plan etc.
            job_source (Optional[JobSource]): The source of the job, like UI
            or SDK.
            training_table_generation_plan (Optional[
            TrainingTableGenerationPlan]): The plan for generating the
            training table, if this is supplied, it is used for
            gen_train_table job triggered from this function. If this is
            not supplied, then default model plan corresponding to the
            predictive query is used.
            user_info (Optional[str]): User info for which user triggered
            the job.
        Returns:
            TrainingJobID: The ID of the started training job.
        """
        if reserved_training_job_id is None:
            reserved_training_job_id = TrainingJobID.make_random_id()

        (pq_id, pq_resource, pq_def_fpid, graph_version_info,
         graph_data_compute_input) = await self._get_pq_and_graph_v2(
             request.pquery_id, project_id, request.graph_snapshot_id)

        await self._link_graph_to_project(
            pq_resource.graph.fpid,
            project_id,
        )

        training_spec, parsed_pq = await self._pquery_logic(
            project_id)._make_training_spec_v2(
                pq_resource,
                self._pquery_settings,
                pq_def_fpid=pq_def_fpid,
            )

        # Copy so that mutations below (the training-table-generation
        # section, in particular) do not leak back into the caller's plan.
        training_spec.advanced_options = copy.deepcopy(request.model_plan)

        training_spec.custom_train_table = await self._from_custom_tt(
            project_id, request.custom_train_table)
        has_train_table_weight_col = False
        if training_spec.custom_train_table:
            has_train_table_weight_col = True
        model_plan_weight_col = (
            training_spec.advanced_options.training_table_generation.weight_col)
        if model_plan_weight_col is not None:
            has_train_table_weight_col = True

        training_spec.advanced_options.training_table_generation = (
            copy.deepcopy(training_table_generation_plan)
            or (await self._default_model_plan_with_graph_def(
                query_string=pq_resource.query_string,
                graph_def=pq_resource.graph,
                run_mode=RunMode.NORMAL,
                project_id=project_id,
            )).model_plan.training_table_generation)
        train_plan_weight_col = (
            training_spec.advanced_options.training_table_generation.weight_col)
        if train_plan_weight_col is None and model_plan_weight_col is not None:
            training_spec.advanced_options.training_table_generation.weight_col = (
                model_plan_weight_col)
            train_plan_weight_col = model_plan_weight_col
        if train_plan_weight_col is not None:
            has_train_table_weight_col = True
        training_spec.advanced_options.validate(
            task_type=parsed_pq.task_type,
            query_type=parsed_pq.query_type,
            has_train_table_time_col=parsed_pq.has_train_table_time_col,
            run_mode=RunMode.NORMAL,
            graph=parsed_pq.predictive_query.graph.config(),
            entity_table_names=parsed_pq.predictive_query.entity_table_names,
            target_to_entity_paths=(
                parsed_pq.predictive_query.target_to_entity_paths),
            has_train_table_weight_col=has_train_table_weight_col,
        )
        validate_text_embedding_models(training_spec.advanced_options)

        # Validate warm start job if provided
        if request.warm_start_job_id:
            await self._validate_warm_start_job(ctx, request.warm_start_job_id,
                                                parsed_pq.task_type)

        # Make a RPC call to MLJobBackend to kick off the training job.
        try:
            training_job_id = await self._ml_job_backend.start_training(
                pq_id.fpid,
                graph_data_compute_input,
                training_spec,
                graph_version_info=graph_version_info,
                gen_train_table_job_id=request.train_table_job_id,
                job_source=job_source,
                warm_start_job_id=request.warm_start_job_id,
                user_info=user_info,
                reserved_job_id=reserved_training_job_id,
                project_id=project_id,
            )
        except RpcError as rpc_error:
            if rpc_error.error_type == ValueError.__name__:
                raise ValueError(rpc_error.message) from rpc_error
            if rpc_error.error_type == JobConcurrencyLimitError.__name__:
                raise JobConcurrencyLimitError(rpc_error.message)
            raise

        await self._update_train_job_tags(request, training_job_id,
                                          pq_resource.name or "NA", pq_id)

        return training_job_id

    async def _update_train_job_tags(self, request: TrainingJobInternalRequest,
                                     training_job_id: TrainingJobID,
                                     pquery_name: str,
                                     pq_id: PredictiveQueryID) -> None:
        tags = self.create_ordered_job_tags(
            request.job_tags,
            pquery_name,
            pq_id.fpid,
            request.warm_start_job_id,
        )
        await self._apply_job_tags(
            JobType.TRAINING_JOB,
            training_job_id,
            tags,
        )
        if request.train_table_job_id is None:
            # generate-train-table job is implicitly spawned as child job
            # of training job, so we need to update job_tags (indexing).
            train_table_job_id = GenerateTrainTableJobID.as_child_of(
                training_job_id)
            tags[TAG_KEY_PARENT_JOB] = JobTagValue(training_job_id)
            await self._apply_job_tags(
                JobType.GENERATE_TRAIN_TABLE_JOB,
                train_table_job_id,
                tags,
            )

    async def _get_pq_and_graph_v2(
        self,
        pquery_id: str,
        project_id: ProjectID,
        graph_snapshot_id: Optional[GraphSnapshotID],
    ) -> Tuple[PredictiveQueryID, PQueryResource, FPID,
               Optional[GraphDataVersionInfo], GraphDataComputeInput]:
        """Resolve a wire pquery ID and build shared launch-time inputs."""
        pq_def_fpid, pq_resource = await self._resolve_pquery_for_launch(
            pquery_id, project_id)
        compute_graph_fpid, graph_version_info = (
            await self._handle_snapshot_override_v2(pq_resource,
                                                    graph_snapshot_id))
        if compute_graph_fpid != pq_resource.graph.fpid:
            pq_def_fpid, pq_resource = await self._fork_pquery_def_for_graph(
                pq_def_fpid, pq_resource, compute_graph_fpid, project_id)
        pq_id = PredictiveQueryID.compose(fpid=pq_def_fpid)
        graph_data_compute_input = await self._get_graph_data_compute_input(
            GraphIDv2.compose(fpid=compute_graph_fpid), project_id)
        return (pq_id, pq_resource, pq_def_fpid, graph_version_info,
                graph_data_compute_input)

    async def update_job_tags(
        self,
        job_type: JobType,
        job_id: JobID,
        tags: Dict[JobTagKey, JobTagValue],
        *,
        project_id: ProjectID,
    ) -> bool:
        """Update tags (add, mutate, or delete) for an existing job."""
        return await self._ml_job_backend.update_job_tags(
            job_type, job_id, tags, project_id=project_id)

    async def remove_job_tag(
        self,
        job_type: JobType,
        job_id: JobID,
        tag: JobTagKey,
        *,
        project_id: ProjectID,
    ) -> bool:
        return await self._ml_job_backend.delete_job_tag(
            job_type, job_id, tag, project_id=project_id)

    async def archive_job(
        self,
        job_id: JobID,
        archived: bool,
        project_id: ProjectID,
    ) -> bool:
        """Archive a job.

        Args:
            job_id: The ID of the job to archive
            archived: Boolean flag indicating whether the job should be archived.
                     If True, archives the job.
                     If False, unarchives the job.

        Returns:
            bool: True if job was archived, False otherwise
        """
        self._require_ml_job_service()
        return await self._ml_job_backend.archive_job(
            job_id,
            archived,
            project_id=project_id,
        )

    async def cancel_batch_prediction_job(
        self,
        ctx: RequestContext,
        id: PredictiveQueryID,
        bp_job_id: Optional[BatchPredictionJobID] = None,
    ) -> bool:
        """
        Cancel a batch prediction job and return true if it's known to have
        been successfully cancelled by the time the call returns.

        Args:
            ctx (RequestContext): API context associated with the request.
            id (PredictiveQueryID): ID of the predictive query under which the
                batch prediction job to be cancelled is running.
            bp_job_id (BatchPredictionJobID, optional): bp_job_id to cancel.
                If not supplied, the most recently triggered batch prediction
                job will be cancelled.
        """
        log.info('API driver starts to cancel prediction: %s', id)
        if await self.get_pquery_definition_by_id(id) is None:
            raise PredictiveQueryDoesNotExistError(id)
        return await self._batch_prediction_logic().cancel_batch_prediction_job(
            ctx, id, bp_job_id)

    async def get_current_training_job_id(
        self,
        ctx: RequestContext,
        id: PredictiveQueryID,
        project_id: ProjectID,
    ):
        return await self._get_current_training_job_id(ctx, id, project_id)

    async def _get_current_training_job_id(
        self,
        ctx: RequestContext,
        id: PredictiveQueryID,
        project_id: ProjectID,
    ):
        await self._require_state_services().projects.get_project(
            project_id, ctx.user_authorization)
        # v2 pqueries don't cache `current_training_job_id`; the pquery ->
        # training job link lives in job tags keyed on the definition fpid.
        pq_def = await self._require_pquery_service().get_definition_by_id(id)
        if pq_def is None:
            raise PredictiveQueryDoesNotExistError(id)
        name, _ = PredictiveQueryID.parse(id)
        return await self._get_latest_training_job_for_pquery_from_services(
            PredictiveQueryID.compose(name, pq_def.fpid),
            project_ids={project_id},
        )

    async def _get_successful_training_home_dir_for_pquery(
        self,
        ctx: RequestContext,
        id: PredictiveQueryID,
        project_id: ProjectID,
    ) -> tuple[str, ProjectID]:
        job_id = await self._get_current_training_job_id(ctx, id, project_id)
        if job_id is None:
            raise ValidationException(
                message=(
                    "This predictive query has needs to be trained, before "
                    "you can perform this operation."), )

        existing_job = await self._ml_job_backend.get_training_job(ctx, job_id)
        if not existing_job:
            raise RuntimeError(
                f"Training job {job_id} is missing from the "
                f"ML backend. Please check for database corruption, or "
                f"retrain the predictive query.")

        if existing_job.execution.status != JobStatus.DONE:
            raise ValidationException(
                message=(
                    "This predictive query needs to be trained successfully "
                    "before you can perform this operation. However, the last "
                    f"training job for this pquery has status = "
                    f"{existing_job.execution.status}."), )

        if ProjectID(existing_job.project_id) != project_id:
            raise TrainingJobDoesNotExistError(job_id)
        return (await self._ml_job_backend.get_job_home_dir(job_id), project_id)

    async def get_current_batch_prediction_job_id(
        self,
        ctx: RequestContext,
        id: PredictiveQueryID,
    ) -> Optional[BatchPredictionJobID]:
        """Return the most recently launched batch prediction job id for the
        given predictive query, or None if there is none."""
        return (
            await
            self._require_ml_job_service().get_current_batch_prediction_job_ids(
                [id],
                project_ids=visible_project_ids_or_none(ctx.user_authorization),
            )).get(id)

    async def get_predictive_query_object(
            self, id: PredictiveQueryID, project_id: ProjectID,
            pquery_settings: PredictiveQuerySettings,
            infer_encoders: bool = False) -> Optional[PredictiveQuery]:
        await self._require_project_access(project_id)
        return await self._pquery_logic(project_id).get_predictive_query_object(
            id,
            pquery_settings,
            infer_encoders=infer_encoders,
        )

    async def get_training_job(
        self,
        ctx: RequestContext,
        id: TrainingJobID,
        strip_expensive_fields: bool = False,
        project_id: Optional[ProjectID] = None,
    ) -> Optional[TrainingJobResponse]:
        if project_id is not None:
            await self.require_project_access(ctx, project_id)
        # Get the training job from the ML backend
        training_job = await self._ml_job_backend.get_training_job(
            ctx, id, strip_expensive_fields=strip_expensive_fields)
        if training_job is None:
            return None

        job_project = ProjectID(training_job.project_id)
        if project_id is None:
            await self.require_project_access(ctx, job_project)
        elif job_project != project_id:
            raise _training_job_not_in_project(id, project_id)

        # Back-fill legacy PQ ids. SDK/v2 jobs can be read from the ML job
        # row even if a previous partial submit missed the secondary tag row.
        job_tags = await self._get_job_tags_if_exists(JobType.TRAINING_JOB, id)
        if job_tags is not None:
            training_job.pquery_id = job_tags.pquery_name
            training_job.tags = job_tags.tags
        else:
            log.warning('Training job %s exists without REST job tags.', id)
            training_job.tags = {}
        return training_job

    async def get_training_table_generation_job(
            self, ctx: RequestContext, id: GenerateTrainTableJobID
    ) -> jobs_api.GenerateTrainTableJobResource:
        jobs = await self._ml_job_backend.get_generate_train_table_jobs(
            {id},
            project_ids=visible_project_ids_or_none(ctx.user_authorization),
        )
        if not jobs:
            raise KeyError(f'Job with id {id} not found in database')
        return jobs[0]

    async def get_training_job_pquery_id_v2(
        self,
        ctx: RequestContext,
        id: TrainingJobID,
    ) -> Optional[PredictiveQueryID]:
        job = await self._ml_job_backend.get_training_job(ctx, id)
        if job:
            if job.pquery_def_fpid is not None:
                return PredictiveQueryID.compose(fpid=job.pquery_def_fpid)
            # Fallback if needed to the predictive query name:
            pquery_name = await self.get_training_job_pquery_id(id)
            if pquery_name is None:
                return None
            return PredictiveQueryID.compose(pquery_name)
        return None

    # As temporary fallback, this approach is inefficient to retrieve pquery id
    # for a given TrainingJobID. Once ML platform SQL DB is fully rolled out,
    # we exect to lookup the JobModel from ML platform and get the pquery id.
    async def get_training_job_pquery_id(
        self,
        id: TrainingJobID,
    ) -> Optional[PredictiveQueryID]:
        job_tags = await self._get_job_tags(JobType.TRAINING_JOB, id)
        return job_tags.pquery_name

    @staticmethod
    def _batch_prediction_row_matches(
        row: Any,
        *,
        model_id: Optional[TrainingJobID],
        pquery_id: Optional[PredictiveQueryID],
        job_ids: Optional[set[BatchPredictionJobID]],
    ) -> bool:
        config = row.config
        if config is None:
            return False
        return ((model_id is None or model_id == config.training_job_id) and
                (pquery_id is None or config.predictive_query_id == pquery_id
                 or PredictiveQueryID.compose(fpid=config.pquery_def_fpid)
                 == pquery_id) and (job_ids is None or row.id in job_ids))

    @staticmethod
    def _batch_prediction_links_from_row(
        row: Any,
        pq: Optional[Any],
    ) -> tuple[PredictiveQueryLink, GraphLink]:
        config = row.config
        assert config is not None
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

        assert pq is not None, f'Query {pq} not found'
        if isinstance(pq, PQueryDefinition):
            pq_id = PredictiveQueryID.compose(pq.pquery_name, pq.fpid)
            graph_id = GraphIDv2.compose(fpid=pq.graph_fpid)
            return (
                PredictiveQueryLink(name=pq.pquery_name or pq_id, id=pq_id),
                GraphLink(name=graph_id, id=graph_id),
            )
        return pq.to_pq_link(), pq.get_graph_link()

    async def _pqueries_by_id_for_batch_prediction_rows(
        self,
        rows: Iterable[Any],
    ) -> Dict[PredictiveQueryID, Optional[Any]]:
        pq_ids = []
        for row in rows:
            config = row.config
            if config is None:
                continue
            pq_id = config.predictive_query_id
            is_from_sdk = pq_id is not None and (pq_id.is_public
                                                 or pq_id.name == "NA")
            if not (is_from_sdk and config.pquery_def_fpid is not None):
                pq_ids.append(pq_id)
        if not pq_ids:
            return {}
        return await self._pquery_definitions_by_ids(pq_ids)

    async def _batch_prediction_responses_from_state_rows(
        self,
        ctx: RequestContext,
        rows: Iterable[Any],
        *,
        job_status: Optional[JobStatus] = None,
        limit: Optional[int] = None,
        job_tags: Optional[Mapping[BatchPredictionJobID, JobTags]] = None,
    ) -> List[BatchPredictionJobResponse]:
        rows = [row for row in rows if row.config is not None]
        if not rows:
            return []
        rows_by_id = {row.id: row for row in rows}
        job_outputs = await self._ml_job_backend.get_batch_prediction_jobs(
            ctx, set(rows_by_id), job_status, limit)
        pqs_by_id = await self._pqueries_by_id_for_batch_prediction_rows(
            rows_by_id[job_id] for job_id in job_outputs)

        result = []
        for job_id, job_output in job_outputs.items():
            row = rows_by_id[job_id]
            config = row.config
            assert config is not None
            pq_link, graph_link = self._batch_prediction_links_from_row(
                row, pqs_by_id.get(config.predictive_query_id))
            response = BatchPredictionJobResponse(
                id=row.id,
                config=config,
                timestamps=ResourceTimestamps(
                    created_at=row.timestamps.created_at,
                    updated_at=(job_output.last_updated_at
                                or row.timestamps.created_at),
                ),
                predictive_query_link=pq_link,
                execution=job_output,
                graph_link=graph_link,
                description=row.description,
                batch_prediction_workflow_id=row.batch_prediction_workflow_id,
                job_source=row.source,
                user=row.user,
                project_id=row.project_id,
                kumo_version=row.kumo_version,
            )
            if job_tags is not None and row.id in job_tags:
                response.tags = job_tags[row.id].tags
            result.append(response)
        return sorted(result, key=lambda job: job.timestamps.created_at)

    async def get_batch_prediction_job(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
        project_id: Optional[ProjectID] = None,
    ) -> Optional[BatchPredictionJobResponse]:
        if project_id is not None:
            await self.require_project_access(ctx, project_id)
        rows = await self._require_ml_job_service().get_batch_prediction_rows(
            [id],
            project_ids=visible_project_ids_or_none(ctx.user_authorization),
        )
        if not rows:
            return None
        job_project = ProjectID(rows[0].project_id)
        if project_id is None:
            await self.require_project_access(ctx, job_project)
        elif job_project != project_id:
            raise _batch_prediction_not_in_project(id, project_id)
        tags = await self._get_job_tags(JobType.BATCH_PREDICTION_JOB, id)
        responses = await self._batch_prediction_responses_from_state_rows(
            ctx,
            rows,
            job_tags={id: tags},
        )
        return responses[0] if responses else None

    async def get_batch_prediction_progress(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
        project_id: ProjectID,
    ) -> Optional[PredictionProgress]:
        await self._require_batch_prediction_job_in_project(ctx, id, project_id)
        bp_job_path = get_job_dir(get_job_base_dir(), id)
        return await self._load_prediction_progress.coroutine(bp_job_path)

    async def cancel_batch_prediction_job_by_id(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
        project_id: ProjectID,
    ) -> bool:
        await self._require_batch_prediction_job_in_project(ctx, id, project_id)
        return await self._ml_job_backend.cancel_batch_prediction_job(id)

    async def _require_training_job_in_project(
        self,
        ctx: RequestContext,
        id: TrainingJobID,
        project_id: ProjectID,
    ) -> None:
        await self._require_state_services().projects.get_project(
            project_id, ctx.user_authorization)
        job = await self._require_ml_job_service().get_training_job(id)
        visible = visible_project_ids_or_none(ctx.user_authorization)
        if job is None or (visible is not None
                           and ProjectID(job.project_id) not in visible):
            raise TrainingJobDoesNotExistError(id)
        if ProjectID(job.project_id) != project_id:
            raise _training_job_not_in_project(id, project_id)

    async def _require_batch_prediction_job_in_project(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
        project_id: ProjectID,
    ) -> None:
        await self._require_state_services().projects.get_project(
            project_id, ctx.user_authorization)
        rows = await self._require_ml_job_service().get_batch_prediction_rows(
            [id],
            project_ids=visible_project_ids_or_none(ctx.user_authorization),
        )
        if not rows:
            raise BatchPredictionDoesNotExistError(id)
        if ProjectID(rows[0].project_id) != project_id:
            raise _batch_prediction_not_in_project(id, project_id)

    @blocking_method()
    def _load_prediction_progress(self,
                                  path: str) -> Optional[PredictionProgress]:
        return progress_util.get_prediction_progress(path)

    async def delete_predictive_query(
        self,
        ctx: RequestContext,
        id: PredictiveQueryID,
        project_id: ProjectID,
    ) -> None:
        await self._require_state_services().projects.get_project(
            project_id,
            ctx.user_authorization,
        )
        # Check if there is any batch prediction job associated with this query
        if await self._require_ml_job_service().has_batch_prediction_rows(
                project_ids={project_id},
                pquery_id=id,
                match_pquery_def_fpid=False,
        ):
            raise ValidationHTTPException(BATCH_PREDICTION_JOBS_DELETE_MESSAGE)

        pquery_service = self._require_pquery_service()
        pq_def = await pquery_service.get_definition_by_id(id)
        if pq_def is None:
            raise PredictiveQueryDoesNotExistError(id)
        if await self._require_graph_service().get_graph_entry(
                project_id, pq_def.graph_fpid) is None:
            # Definitions are content-addressed, but their graph ownership is
            # the authorization boundary.  Do not disclose or mutate a query
            # through a project that does not own its graph.
            raise PredictiveQueryDoesNotExistError(id)
        await self._cancel_inflight_training_jobs(
            ctx,
            await self._training_job_ids_for_pquery_from_services(
                pq_def.fpid, project_ids={project_id}),
        )
        await self._require_graph_service(
        ).decrement_or_delete_graph_definition(project_id, pq_def.graph_fpid)
        if pq_def.pquery_name is not None:
            await pquery_service.delete_alias(pq_def.pquery_name)
        await pquery_service.delete_definition(pq_def.fpid)

    async def delete_training_job(self, ctx: RequestContext,
                                  id: PredictiveQueryID,
                                  job_id: TrainingJobID) -> None:
        await self._require_ml_job_service().delete_job_tags(
            JobType.TRAINING_JOB, job_id)

        log.info('Deleting job %s from ML Job Backend.', job_id)
        try:
            # Always will exist:
            await self._ml_job_backend.delete_training_job(ctx, job_id)
        except RpcError as rpc_error:
            if rpc_error.error_type == ValueError.__name__:
                raise ValueError(rpc_error.message)
            raise
        log.info('Job %s has been deleted for %s in StateDB', job_id, id)

    async def delete_batch_prediction(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
    ) -> bool:
        return await self._batch_prediction_logic().delete_batch_prediction(
            ctx, id)

    async def list_predictive_queries_v2(
            self, project_id: ProjectID,
            name_pattern: str = '.*') -> List[PQueryResource]:
        """
        Returns a list of V2 PQueries (SDK/API resource model) that matches the
        name pattern.
        """
        await self._require_project_access(project_id)
        pquery_service = self._require_pquery_service()
        names = await pquery_service.list_aliases(name_pattern)
        definitions_by_id = await self._pquery_definitions_by_ids(
            [PredictiveQueryID(name) for name in names])
        definitions_by_name = {}
        for name in names:
            pq_def = definitions_by_id.get(PredictiveQueryID(name))
            if pq_def is not None:
                definitions_by_name[name] = pq_def

        missing_names = [
            name for name in names if name not in definitions_by_name
        ]
        if missing_names:
            raise PredictiveQueryDoesNotExistError(
                PredictiveQueryID(missing_names[0]))

        resources_by_fpid = await self._pquery_resources_from_definitions(
            definitions_by_name.values(), project_id)
        return [
            resources_by_fpid[definitions_by_name[name].fpid] for name in names
            if definitions_by_name[name].fpid in resources_by_fpid
        ]

    async def list_job_tags(
            self, job_type: JobType,
            tag_filters: Mapping[str, str]) -> Dict[JobID, JobTags]:
        return await self._list_job_tags(job_type, tag_filters)

    async def _list_job_tags(
        self,
        job_type: JobType,
        tag_filters: Mapping[str, str],
        *,
        job_ids: Iterable[JobID] | None = None,
    ) -> Dict[JobID, JobTags]:
        return await self._require_ml_job_service().list_job_tags(
            job_type, tag_filters, job_ids=job_ids)

    async def list_batch_prediction_jobs(
        self,
        ctx: RequestContext,
        model_id: Optional[TrainingJobID] = None,
        pquery_id: Optional[PredictiveQueryID] = None,
        job_status: Optional[JobStatus] = None,
        limit: Optional[int] = None,
        tag_filters: Mapping[JobTagKey, JobTagValue] = {},
        project_id: Optional[ProjectID] = None,
    ) -> List[BatchPredictionJobResponse]:
        job_ids = None
        job_tags: dict[BatchPredictionJobID, JobTags] = {}
        if tag_filters:
            job_tags = {
                BatchPredictionJobID(id): tags
                for id, tags in (await self._list_job_tags(
                    JobType.BATCH_PREDICTION_JOB, tag_filters)).items()
            }
            job_ids = set(job_tags)

        responses = await self._batch_prediction_logic(
        ).list_batch_prediction_jobs(
            ctx,
            model_id=model_id,
            pquery_id=pquery_id,
            job_status=job_status,
            limit=limit,
            job_ids=job_ids,
            project_id=project_id,
        )
        missing_tag_ids = [
            response.id for response in responses if response.id not in job_tags
        ]
        if missing_tag_ids:
            job_tags.update({
                BatchPredictionJobID(id): tag
                for id, tag in (await self._list_job_tags(
                    JobType.BATCH_PREDICTION_JOB,
                    {},
                    job_ids=missing_tag_ids,
                )).items()
            })
        for response in responses:
            if response.id in job_tags:
                response.tags = job_tags[response.id].tags
        return responses

    async def list_paginated_batch_prediction_jobs(
        self, ctx: RequestContext, pagination_args: PaginationArgs,
        model_id: Optional[TrainingJobID] = None,
        pquery_id: Optional[PredictiveQueryID] = None,
        job_status: Optional[JobStatus] = None
    ) -> PaginatedBatchPredictionJobResponse:
        return await self._batch_prediction_logic(
        ).list_paginated_batch_prediction_jobs(
            ctx,
            pagination_args,
            model_id=model_id,
            pquery_id=pquery_id,
            job_status=job_status,
        )

    async def get_generate_pred_table_job(
        self,
        ctx: RequestContext,
        job_id: GeneratePredictionTableJobID,
        project_id: ProjectID,
    ) -> jobs_api.GeneratePredictionTableJobResource:
        await self._require_state_services().projects.get_project(
            project_id, ctx.user_authorization)
        jobs = await self._ml_job_backend.get_generate_pred_table_jobs(
            {job_id}, project_ids={project_id})
        if not jobs:
            raise KeyError(f'Job with id {job_id} not found in database')
        return jobs[0]

    async def get_job_tags(self, job_type: JobType, job_id: JobID) -> JobTags:
        return await self._get_job_tags(job_type, job_id)

    async def _get_job_tags_if_exists(
        self,
        job_type: JobType,
        job_id: JobID,
    ) -> Optional[JobTags]:
        return await self._require_ml_job_service().get_job_tags(
            job_type, job_id)

    async def _get_job_tags(
        self,
        job_type: JobType,
        job_id: JobID,
    ) -> JobTags:
        job_tags = await self._get_job_tags_if_exists(job_type, job_id)
        if job_tags is None:
            raise KeyError(f"{job_type} id={job_id} is not found in DB")
        return job_tags

    async def get_training_job_materialization_info(
            self, id: TrainingJobID) -> Optional[MaterializationTaskInfo]:
        info = await self._ml_job_backend.get_materialization_info(id)
        if info is not None:
            info.clear_private_fields()
        return info

    async def get_parsed_pq_v2_by_fpid(
        self,
        fpid: FPID,
        project_id: ProjectID,
        skip_validation: bool = False,
    ) -> RESTParsedPredictiveQuery:
        pq_def = await self._require_pquery_service().get_definition(fpid)
        if pq_def is None:
            raise PredictiveQueryDoesNotExistError(
                PredictiveQueryID.compose(fpid=fpid))

        graph_def = await self._get_graph_def_for_pquery(
            pq_def.graph_fpid, project_id)
        if graph_def is None:
            raise ValueError(
                f'Unable to find graph for predictive query definition '
                f'{pq_def.fpid}')

        val_args = PQueryValidationArgs(
            query_string=pq_def.query_str,
            graph=graph_def,
            name=pq_def.pquery_name or "NA",
            desc="",
            fpid=fpid,
            runtime_options=pquery_runtime_options_or_none(
                list_distinct_target_is_array=(
                    pq_def.list_distinct_target_is_array),
                lp_bp_layout=pq_def.lp_bp_layout,
            ),
        )
        parsed_pq, _ = await self._pquery_logic(
            project_id).parse_predictive_query_v2(
                val_args, config=None, auto_trainer_options=None,
                pquery_settings=self._pquery_settings,
                skip_validation=skip_validation)
        return parsed_pq

    async def get_pquery_definition(self, fpid: FPID) -> PQueryDefinition:
        return await self._require_pquery_service().get_definition_required(fpid
                                                                            )

    async def get_pquery_definition_if_exists(
            self, fpid: FPID) -> Optional[PQueryDefinition]:
        return await self._require_pquery_service().get_definition(fpid)

    async def get_pquery_definition_by_id(
            self, id: PredictiveQueryID) -> Optional[PQueryDefinition]:
        return await self._require_pquery_service().get_definition_by_id(id)

    async def get_train_table_stats(
        self,
        job_id: TrainingJobID,
    ) -> Optional[TrainTableStats]:
        try:
            return await load_train_table_stats_from_cache.coroutine(job_id)
        except KeyError:
            log.debug('Exact training table stats not yet available')

    async def get_train_table_generation_stats(
        self,
        job_id: str,
    ) -> Optional[TrainTableGenerationProgress]:
        r"""Get training table generation statistics.

        Args:
            job_id (str): job id of the training job to get train table
                generation statistics.

        Returns:
            (Optional[TrainTableGenerationProgress]): if train table stats is
                available and the predictive query problem type is temporal,
                None otherwise.

        """
        # Get full job directory path
        job_dir = get_job_dir(get_job_base_dir(), JobID(job_id))

        # Try loading from job dir first
        stats = await train_table_generation_stats_from_job_dir.coroutine(
            job_dir)
        if stats is not None:
            return stats

        # TODO: We currently plan to deprecate this method by 2025/04/01.
        # Fall back to cache
        return await train_table_generation_stats_from_cache_v2.coroutine(job_id
                                                                          )

    async def get_train_table_split_stats(
        self,
        job_id: str,
    ) -> Optional[SplitTrainTableStats]:
        r"""Get training table split statistics.

        Args:
            job_id (str): job id of the training job to get train table split
                statistics.

        Returns:
            (Optional[SplitTrainTableStats]): if train table stats is
                available, None otherwise.
        """
        # Get full job directory path
        job_dir = get_job_dir(get_job_base_dir(), JobID(job_id))

        # Get the split
        split_fpid = None
        if TrainingJobID.is_valid_id(job_id):
            job_input_path = HomeDirManager(job_dir).training_job_input_file

            try:
                # Try to load directly as TrainingJobInput
                job_input = load_json(job_input_path, TrainingJobInput)
            except Exception as e:
                log.info(
                    f"Failed to load job input as TrainingJobInput directly: "
                    f"{e}. Trying to load via json components and see "
                    f"whether the failure is due to predictive query config "
                    f"is PredictiveQueryConfigBase which is the legacy "
                    f"format.")
                # Try loading as dictionary and then convert
                training_input_json = load_json(job_input_path, dict)
                pquery_config_json = training_input_json['spec'][
                    'predictive_query']

                # Check if we need to convert from PredictiveQueryConfigBase
                # to PredictiveQueryV2Config
                try:
                    pquery_config = from_json(to_json(pquery_config_json),
                                              PredictiveQueryConfigBase)
                except Exception as e:
                    log.error(
                        f"Failed to convert predictive query config "
                        f"{pquery_config_json} to PredictiveQueryConfigBase"
                        f": {e}")
                finally:
                    pquery_v2_config = (
                        pquery_config.to_predictive_query_v2_config())
                    # Replace with the V2Config in the original dict
                    training_input_json['spec'][
                        'predictive_query'] = dataclasses.asdict(
                            pquery_v2_config)

                    # Convert back to TrainingJobInput
                    job_input = from_json(to_json(training_input_json),
                                          TrainingJobInput)
                    log.info(
                        "Successfully converted to PredictiveQueryV2Config")

            if job_input.spec.split is not None:
                split_fpid = Split.get_fpid_from_config(job_input.spec.split)
        elif GenerateTrainTableJobID.is_valid_id(job_id):
            job_input_path = HomeDirManager(
                job_dir).generate_train_table_job_input_file
            job_input = load_json(job_input_path, GenerateTrainTableJobInput)
            if job_input.spec.split is not None:
                split_fpid = Split.get_fpid_from_config(job_input.spec.split)
        else:
            split_fpid = None

        if split_fpid is None:
            log.warning('Split fpid is not available for job %s', job_id)

        # Try loading from job dir first
        stats = await train_table_split_stats_from_job_dir.coroutine(
            job_dir,
            split_fpid,
        )
        if stats is not None:
            return stats

        # TODO: We currently plan to deprecate this method by 2025/04/01.
        # Fall back to existing cache-based method
        return await train_table_split_stats_from_cache.coroutine(job_id)

    async def infer_task_type_pqlv2(
        self,
        query: str,
        graph_id: GraphID,
        project_id: ProjectID,
    ) -> TaskTypeWithReason:
        """Infer the task type from a PQLv2 query string

        Args:
            query (str): Query string PQLv2
            graph_id (GraphID): Graph ID used for the query

        Returns:
            TaskTypeWithReason: Task type with associated inferral logic if the
                query is valid

        Raises:
            GraphDoesNotExistError: If the graph does not exist.
            ValidationException: If the query is semantically invalid for the
                graph.
        """
        logic = self._pquery_logic(project_id)
        graph = await self._graph_logic(project_id).get_graph_object(
            to_graph_id(graph_id))
        if graph is None:
            raise GraphDoesNotExistError(graph_id)

        return await logic.infer_task_type_pqlv2(query, graph)

    async def generate_holdout_presigned_url(
            self, training_job_id) -> OutputPresignedUrlResponse:
        r"""Generate the holdout dataset download url.

        Args:
            training_job_id (str): job id of the training job to get holdout
            data.

        Returns:
            OutputPresignedUrlResponse class which has the presigned url.
        """
        training_job_path = get_job_dir(get_job_base_dir(), training_job_id)
        path = await self._get_test_df.coroutine(training_job_path)

        helper = fsutil.get_presigned_helper(path)
        if helper is not None:
            presigned_url = await helper.generate_download_url.coroutine(path)
        else:
            presigned_url = fsutil.customer_facing_path(path)
        return OutputPresignedUrlResponse(presigned_url)

    @blocking_method()
    def _get_test_df(self, training_job_path: str) -> str:
        home = HomeDirManager(training_job_path)
        task_name = home.first_task_name()
        if task_name is None:
            raise ValueError("Task does not exist")
        best_train_dir = home.get_best_trainer_dir(str(task_name))

        # Download from refit train val directory if it is enabled.
        # Otherwise download from best train directory.
        trainer_config = load_json(os.path.join(best_train_dir, 'config.json'),
                                   TrainerConfig)
        if trainer_config.refit_cfg.refitted_trainval:
            best_train_dir = best_train_dir.rstrip('/') + '_refit_trainval'

        data_path = os.path.join(best_train_dir, 'test_df.parquet')
        if fsutil.exists(data_path):
            return data_path
        else:
            log.error("Holdout data does not exist for %s", training_job_path)
            raise ValueError("Holdout data does not exist.")

    async def generate_bp_presigned_url(
            self, batch_prediction_job_id) -> OutputPresignedUrlResponse:
        r"""Generate the batch prediction output url.

        For S3/GCS the output is zipped (if multi-partition) and a
        presigned/signed URL is returned so the UI can download via
        HTTP GET.

        Args:
            batch_prediction_job_id (str): job id of the batch prediction job
                to get output data.

        Returns:
            OutputPresignedUrlResponse class which has the presigned url.
        """
        batch_prediction_job_path = await self._ml_job_backend.get_job_home_dir(
            batch_prediction_job_id)

        @blocking_func()
        def _prepare_download_url() -> str:
            job_home = HomeDirManager(batch_prediction_job_path)
            artifact = resolve_prediction_artifact(
                job_home, PredictionArtifactType.PREDICTIONS)
            if artifact is None:
                raise ValueError('Batch prediction output was not found')
            output_dir = artifact.root
            files = (list(artifact.files) if artifact.files is not None else
                     fsutil.listdir(output_dir))
            data_size = sum(fsutil.filesize(path) or 0 for path in files)
            if data_size >= _DATA_DOWNLOAD_LIMIT:
                raise ValueError(
                    PREDICTION_OUTPUT_SIZE_MESSAGE.format(data_size))

            helper = fsutil.get_presigned_helper(output_dir)
            if helper is None or isinstance(helper, LocalPresignedHelper):
                raise ValueError(f"Download not supported for: {output_dir}")

            # Single partition → presign directly; multiple → zip first.
            if len(files) == 1:
                path = files[0]
            else:
                archive_name = hashlib.sha256(
                    str(batch_prediction_job_id).encode()).hexdigest() + '.zip'
                archive_path = os.path.join(
                    deploy_env().data_plane_storage_root,
                    'tmp_archives',
                    archive_name,
                )
                if artifact.files is not None:
                    path = fsutil.zip_files(files, output_path=archive_path)
                else:
                    path = fsutil.zip_directory(output_dir,
                                                output_path=archive_path)

            return helper.generate_download_url(path)

        presigned_url = await _prepare_download_url.coroutine()
        return OutputPresignedUrlResponse(presigned_url)

    async def supports_batch_prediction_download(
            self, batch_prediction_job_id: BatchPredictionJobID) -> bool:
        """Whether this deployment can expose a browser-safe prediction URL."""
        batch_prediction_job_path = await self._ml_job_backend.get_job_home_dir(
            batch_prediction_job_id)
        job_home = HomeDirManager(batch_prediction_job_path)

        @blocking_func()
        def _supports_download() -> bool:
            try:
                artifact = resolve_prediction_artifact(
                    job_home,
                    PredictionArtifactType.PREDICTIONS,
                    validate_objects=False,
                )
            except (FileNotFoundError, ValueError):
                return False
            if artifact is None:
                return False
            artifact_root = artifact.root
            helper = fsutil.get_presigned_helper(artifact_root)
            return helper is not None and not isinstance(
                helper, LocalPresignedHelper)

        return await _supports_download.coroutine()

    async def get_presigned_prediction_urls(
        self,
        ctx: RequestContext,
        job_id: str,
    ) -> GetPredictionsDfUrlResponse | None:
        """Returns GetPredictionsDfUrlResponse, containing a list of presigned
        URLs corresponding to partitions of a BP predictions output.

        Args:
            job_id (str): ID of the batch prediction job for which prediction
                partitions are requested.
        """
        res = await self._get_presigned_artifact_urls(
            ctx, job_id, PredictionArtifactType.PREDICTIONS)

        return None if res is None else GetPredictionsDfUrlResponse(res)

    async def get_presigned_embedding_urls(
        self,
        ctx: RequestContext,
        job_id: str,
    ) -> GetEmbeddingsDfUrlResponse | None:
        """Returns GetEmbeddingsDfUrlResponse, containing a list of presigned
        URLs corresponding to partitions of a BP embeddings output.

        Args:
            job_id (str): ID of the batch prediction job for which embedding
                partitions are requested.
        """
        res = await self._get_presigned_artifact_urls(
            ctx, job_id, PredictionArtifactType.EMBEDDINGS)

        return None if res is None else GetEmbeddingsDfUrlResponse(res)

    async def _get_presigned_artifact_urls(
            self, ctx: RequestContext, job_id: str,
            artifact_type: PredictionArtifactType) -> list[str] | None:

        bp_job_id = BatchPredictionJobID(job_id)
        bp_job_response = await self.get_batch_prediction_job(ctx, bp_job_id)

        # TODO(eric): There should be a check which looks at the status of the
        # job, not just the existence of the job. But a job could fail but
        # still have progressed far enough to have written the internal files.
        # Getting the results this way could allow a user to salvage a failed
        # job, and that's the reasoning for not locking things down quite yet.
        # What's needed is finer-grained job status codes which can accurately
        # indicate whether the internal files are available.
        if bp_job_response is None:
            raise ValueError(
                BATCH_PREDICTION_JOB_NOT_FOUND_MESSAGE.format(job_id))

        # TODO(eric): investigate whether identity comparison will be correct
        # despite there being two enums named PredictionArtifactType used
        # seemingly interchangeably
        configured_outputs = bp_job_response.config.outputs
        artifact_type_valid = any(o.artifact_type == artifact_type
                                  for o in configured_outputs)

        # An empty output list is the SDK's local-download contract: the
        # prediction artifacts remain in Kumo's data plane instead of being
        # copied to a customer connector. Discover those artifacts by their
        # internal path below. If external outputs were configured, continue
        # enforcing the exact requested artifact type.
        if configured_outputs and not artifact_type_valid:
            raise ValueError(
                PREDICTION_ARTIFACT_TYPE_NOT_FOUND_MESSAGE.format(
                    artifact_type))

        bp_job_path = await self._ml_job_backend.get_job_home_dir(bp_job_id)
        job_home = HomeDirManager(bp_job_path)

        @blocking_func()
        def _blocking_helper_for_fsutil_calls():
            artifact = resolve_prediction_artifact(job_home, artifact_type)
            if artifact is None:
                return None
            partitions: Iterable[str] = (artifact.files
                                         if artifact.files is not None else
                                         fsutil.listdir(artifact.root))
            artifact_root = artifact.root

            helper = fsutil.get_presigned_helper(artifact_root)
            if helper is not None:
                return [helper.generate_download_url(p) for p in partitions]
            return [fsutil.customer_facing_path(p) for p in partitions]

        return await _blocking_helper_for_fsutil_calls.coroutine()

    async def get_batch_pred_materialization_info(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
    ) -> Optional[MaterializationTaskInfo]:
        self._require_ml_job_service()
        if not await self._ml_job_backend._has_batch_prediction_job(ctx, id):
            raise BatchPredictionDoesNotExistError(id)
        info = await self._ml_job_backend.get_materialization_info(id)
        if info is not None:
            info.clear_private_fields()
        return info

    async def get_training_job_warnings(
            self, job_ids: List[TrainingJobID]) -> Dict[str, ErrWarnResponse]:

        response: Dict[str, ErrWarnResponse] = {}
        if len(job_ids) > 0:
            try:
                base_dir = await self._ml_job_backend.get_job_base_dir()
                tjv = TrainingJobValidator.from_state_services(
                    self._require_state_services(),
                    self._data_backend_ctx,
                    base_dir,
                    self._ml_job_backend,
                )
                response = await tjv.get_warnings(job_ids)
            except Exception as e:
                log.exception(e)
                # Generate an empty response for exceptions
                for job_id in job_ids:
                    response[job_id] = ErrWarnResponse(errors=[], warnings=[],
                                                       info_items=[])

        return response

    async def get_batch_prediction_warnings(
            self, bp_ids: List[BatchPredictionJobID],
            ctx: RequestContext) -> Dict[str, ErrWarnResponse]:

        response: Dict[str, ErrWarnResponse] = {}
        if len(bp_ids) > 0:
            try:
                base_dir = await self._ml_job_backend.get_job_base_dir()
                bpv = BatchPredictionValidator.from_state_services(
                    self._require_state_services(),
                    self._data_backend_ctx,
                    base_dir,
                    self._ml_job_backend,
                    ctx,
                )
                response = await bpv.get_warnings(bp_ids)
            except Exception as e:
                log.exception(e)
                # Generate an empty response for exceptions
                for bp in bp_ids:
                    response[bp] = ErrWarnResponse(errors=[], warnings=[],
                                                   info_items=[])

        return response

    async def get_prediction_anchor_time_from_per_job_dir(
        self,
        job_id: BatchPredictionJobID,
        home_dir: str,
    ) -> Optional[BatchPredictionJobAnchorTime]:
        return await BatchPredictionJobAnchorTime(
        ).get_stats_from_per_job_dir.coroutine(job_id, home_dir)

    async def get_prediction_anchor_time(
        self,
        job_id: BatchPredictionJobID,
        ctx: RequestContext,
    ) -> BatchPredictionJobAnchorTime:
        return await BatchPredictionJobAnchorTime().load_from_cache.coroutine(
            job_id)

    async def get_prediction_num_entities(
        self,
        job_id: BatchPredictionJobID,
        home_dir: str,
    ):
        """
        Get the number of entities for prediction table.

        Args:
            job_id (BatchPredictionJobID): OK to use the ID of a
            batch prediction job that has prediction table
            generation run in standalone mode is the same as that job's ID.
            home_dir (str): path for query stats in s3.
        Returns:
            The value for num entities if available otherwise None.
        """
        num_entities = await self._get_num_entities_from_job_dir(home_dir)
        if num_entities is None:
            num_entities = await self._get_num_entities_from_cache(job_id)
        return num_entities

    async def get_prediction_stats(
            self, id: BatchPredictionJobID,
            ctx: RequestContext) -> BatchPredictionJobStats:
        logic = self._batch_prediction_logic()
        job = await logic._get_batch_prediction_entry_if_exists(ctx, id)
        if job is None:
            raise BatchPredictionDoesNotExistError(id)
        if job.config is None:
            raise ValueError(f"Batch Prediction Job with ID {id} does not "
                             f"have an associated config object in the DB.")

        # Get the coroutine for raw stats
        raw_stats_coro = logic.get_prediction_stats(ctx, id)

        # If a transformation was specified
        if job.config.transformation is not None:
            # Get the coroutine for transformed stats
            transformed_stats_coro = logic.get_prediction_stats(
                ctx, id, use_transformed_output=True)

            # Execute them in parallel to decrease endpoint latency
            raw_stats, transformed_stats = await asyncio.gather(
                raw_stats_coro, transformed_stats_coro)

            return BatchPredictionJobStats(
                raw_predictions=raw_stats,
                transformed_predictions=transformed_stats,
            )
        else:
            # Otherwise just execute single coroutine
            return BatchPredictionJobStats(
                raw_predictions=(await raw_stats_coro),
                transformed_predictions=None,
            )

    async def get_mlops_stats(
        self,
        ctx: RequestContext,
        query: MLOpsQueryConfig,
    ) -> Optional[MLOpsStats]:
        return await MLOpsLogic.from_state_services(
            self._require_state_services(),
            self._ml_job_backend,
            self._data_backend_ctx,
        ).get_mlops_stats(query, ctx)

    async def _default_model_plan_with_graph_def(
        self,
        query_string: str,
        graph_def: graph_api.GraphDefinition,
        run_mode: RunMode,
        project_id: ProjectID,
        has_weight_col: bool = False,
    ) -> DefaultModelPlanInfo:
        """Build the default model plan straight from a graph definition the
        caller already holds, avoiding a redundant graph fetch by id.

        Mirrors ``get_model_plan_info`` (the by-id path) without the DB
        lookup. Deliberately uses the logic method directly rather than
        ``get_pquery_model_plan_with_graph_def`` so we don't add the latter's
        ``validate_graph_definition`` (which enforces graph connectivity) to
        callers operating on an already-persisted graph.
        """
        await self._require_project_access(project_id)
        return await self._pquery_logic(
            project_id).get_model_plan_info_with_graph_def(
                DefaultModelPlanRequest(
                    query_string=query_string,
                    graph_id="",
                    run_mode=run_mode,
                    has_train_table_weight_col=has_weight_col,
                    graph_def=build_graph_definition(graph_def),
                ), self._pquery_settings)

    async def get_pquery_default_model_plan(
        self,
        query_string: str,
        graph_id: GraphID,
        run_mode: RunMode,
        project_id: ProjectID,
        has_weight_col: bool = False,
    ) -> DefaultModelPlanInfo:
        await self._require_project_access(project_id)
        plan_info = await self._pquery_logic(project_id).get_model_plan_info(
            DefaultModelPlanRequest(query_string=query_string,
                                    graph_id=graph_id, run_mode=run_mode,
                                    has_train_table_weight_col=has_weight_col),
            self._pquery_settings)
        return plan_info

    async def get_pquery_model_plan_with_graph_def(
        self,
        query_string: str,
        graph_def: GraphDefinition,
        run_mode: RunMode,
        project_id: ProjectID,
        has_weight_col: bool = False,
    ) -> DefaultModelPlanInfo:
        """
        Get default model plan for a given predictive query
        using graph definition instead of graph ID.
        Using graph definition generalizes the approach is more
        light-weight (no need to store the graph in the DB) and
        is works with any graph definition.

        This method is introduced to reduce the reliance on
        the DB and accessing info from there.

        Args:
            query_string (str): The predictive query we want to make a
                model plan for.
            graph_def (GraphDefinition): The graph definition of the
                graph we want to use to construct the model plan.
            run_mode (RunMode): user defined run mode (fast, normal, slow).
            has_weight_col (bool): Yes/no option if there is a
                weight column present in the graph. (default: :obj:`False`)
        Returns:
            DefaultModelPlanInfo (DefaultModelPlanInfo) consitiong of
                model plan, task type, query type, and
                has_train_table_weight_col.
        """
        await self._require_project_access(project_id)
        # Validate graph defintion first
        validated_graph_def = build_graph_definition(graph_def)
        logic = self._graph_logic(project_id)
        graph = await logic._make_graph_from_graph_def_v2(validated_graph_def)
        await logic.validate_graph(graph)

        plan_info = await self._pquery_logic(
            project_id).get_model_plan_info_with_graph_def(
                DefaultModelPlanRequest(
                    query_string=query_string,
                    graph_id="",
                    run_mode=run_mode,
                    has_train_table_weight_col=has_weight_col,
                    graph_def=validated_graph_def,
                ), self._pquery_settings)
        return plan_info

    async def get_encoders_for_pquery(
            self, ctx: RequestContext, id: PredictiveQueryID,
            project_id: ProjectID) -> Optional[Dict[str, str]]:
        """
        Returns the encoders used for a specific predictive query.

        Output: Dictionary mapping from Fully Qualified Table Name to Encoder
        object representation

        {
            connector_id.table_id.column_name: Encoder(p1=2, p2=[41])
        }

        The encoder is meant to be used in the Model Planner and can
        be parsed by ModelPlan.parse_encoder_overrides
        into an encoder object in code

        Returns ``None`` when no encoder artifact exists. Backend retrieval
        failures are propagated to the caller.
        """
        job_id = await self._get_current_training_job_id(ctx, id, project_id)
        if job_id is None:
            log.debug('No training job is available for this Predictive Query')
            return None
        training_job = await self.get_training_job(ctx, job_id)
        if (training_job is None
                or training_job.execution.status != JobStatus.DONE):
            log.debug('Training job has not yet completed')
            return None
        return await self._ml_job_backend.get_encoders_for_job(ctx, job_id)

    async def get_encoders_for_training_job(
            self, ctx: RequestContext,
            id: TrainingJobID) -> Optional[Dict[str, str]]:
        """
        Return persisted encoders for a specific training job.

        A missing job, or a failed or cancelled job without an artifact,
        raises a not-found error. Missing artifacts for other states return
        ``None``. Backend retrieval failures are propagated.
        """
        training_job = await self.get_training_job(
            ctx,
            id,
            strip_expensive_fields=True,
        )
        if training_job is None:
            raise TrainingJobDoesNotExistError(id)

        encoders = await self._ml_job_backend.get_encoders_for_job(ctx, id)
        terminal_failure = training_job.execution.status in (
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        )
        if encoders is None and terminal_failure:
            raise TrainingJobEncodersNotAvailableError()
        return encoders

    def _has_weight_col(self, model_plan: ModelPlan,
                        custom_train_table: bool) -> bool:
        """While loading model plan we assume training table has
        a weight column if weight_mode is set and a custom_train_table is used.
        """
        has_weight_spec = (model_plan.training_table_generation.weight_col
                           is not None)
        return (model_plan.optimization.weight_mode is not None
                and (custom_train_table or has_weight_spec))

    async def get_model_plans_per_experiment(
        self,
        ctx: RequestContext,
        training_job_id: Optional[TrainingJobID],
        task_type: TaskType,
        query_type: Optional[QueryType],
    ) -> ModelPlansResponse:

        if training_job_id is None or query_type is None:
            log.warn('Training Job ID was not specified')
            return ModelPlansResponse([], [])
        training_job_dir = get_job_dir(
            get_job_base_dir(),
            JobID(training_job_id),
        )
        dir_mgr = HomeDirManager(training_job_dir)
        model_plan_paths = [
            os.path.join(trial_dir, 'model_plan.json')
            for trial_dir in dir_mgr.trials()
        ]
        model_plans = await asyncio.gather(*[
            load_json_async(path, ModelPlan, default=None,
                            raise_on_missing=False) for path in model_plan_paths
        ])
        model_plans = [mp for mp in model_plans if mp is not None]

        train_job_response = await self.get_training_job(ctx, training_job_id)
        custom_train_table = False
        if train_job_response and train_job_response.custom_train_table:
            custom_train_table = train_job_response.custom_train_table
        return ModelPlansResponse(
            model_plans=[
                model_plan_to_yaml(
                    mp,
                    task_type=task_type,
                    query_type=query_type,
                    has_train_table_weight_col=self._has_weight_col(
                        mp, custom_train_table),
                ) for mp in model_plans
            ],
            advanced_options=[],
        )

    async def _handle_snapshot_override_v2(
        self,
        pq_resource: PQueryResource,
        graph_snapshot_id: Optional[GraphSnapshotID],
    ) -> Tuple[FPID, Optional[GraphDataVersionInfo]]:
        if not graph_snapshot_id:
            return pq_resource.graph.fpid, None
        graph_id, graph_version_info = await self._get_graph_snapshot_info(
            graph_snapshot_id)
        if graph_id.fpid != pq_resource.graph.fpid:
            log.warning(f'Graph data snapshot id={graph_snapshot_id} '
                        f'overrides the graph defined in predictive query '
                        f'{pq_resource.name}')
            return graph_id.fpid, graph_version_info
        return pq_resource.graph.fpid, graph_version_info

    async def _get_graph_snapshot_info(
        self,
        graph_snapshot_id: GraphSnapshotID,
    ) -> Tuple[GraphIDv2, GraphDataVersionInfo]:
        return await get_graph_snapshot_info(graph_snapshot_id)

    async def _fork_pquery_def_for_graph(
        self,
        pq_def_fpid: FPID,
        pq_resource: PQueryResource,
        new_graph_fpid: FPID,
        project_id: ProjectID,
    ) -> Tuple[FPID, PQueryResource]:
        """Save a PQueryDefinition with `new_graph_fpid` (re-using an existing
        fpid if one already matches) and return its fpid along with a
        PQueryResource pointing at the new graph."""
        pquery_service = self._require_pquery_service()
        pq_def = await pquery_service.get_definition(pq_def_fpid)
        assert pq_def is not None
        forked = PQueryDefinition(
            get_current_timestamps(),
            pquery_name=pq_def.pquery_name,
            query_str=pq_def.query_str,
            graph_fpid=new_graph_fpid,
            list_distinct_target_is_array=(
                pq_def.list_distinct_target_is_array),
            task_type=pq_def.task_type,
            query_type=pq_def.query_type,
            lp_bp_layout=pq_def.lp_bp_layout,
        )
        forked_fpid = await pquery_service.add_definition(forked)
        new_graph_def = await self._get_graph_def_for_pquery(
            new_graph_fpid, project_id)
        if new_graph_def is None:
            raise ValueError(
                f'Graph {new_graph_fpid} not found for snapshot override')
        forked_resource = PQueryResource(
            name=pq_resource.name,
            query_string=pq_resource.query_string,
            desc=pq_resource.desc,
            graph=new_graph_def,
            runtime_options=pq_resource.runtime_options,
        )
        return forked_fpid, forked_resource

    def create_ordered_job_tags(
        self,
        request_tags: Dict[str, str],
        pquery_name: str,
        pquery_def_fpid: FPID,
        warm_start_job_id: Optional[TrainingJobID] = None,
    ) -> Dict[JobTagKey, JobTagValue]:
        """Create base job tags dictionary with request tags and standard
        pquery tags. Ensures that the custom tags coming from the request
        are applied first, since Python 3.7+ ensures order of insertion while
        reading the Dict keys. This is important to ensure that user defined
        (custom tags) are applied first, followed by the default tags.

        Args:
            request_tags: Dictionary of tags from the request
            pquery_name: Name of the predictive query
            pquery_def_fpid: FPID of the predictive query definition
            warm_start_job_id: Optional id of training job this job is a warm
                start of.
        Returns:
            Dictionary mapping JobTagKey to JobTagValue containing base tags
        """
        tags = {JobTagKey(k): JobTagValue(v) for k, v in request_tags.items()}
        # Create base tags with request tags first
        tags[TAG_KEY_PQUERY_NAME] = JobTagValue.of_pquery_name(pquery_name)
        tags[TAG_KEY_PQUERY_ID] = JobTagValue.of_pquery_id(pquery_def_fpid)
        if warm_start_job_id:
            tags[TAG_KEY_WARM_START_JOB_ID] = JobTagValue(warm_start_job_id)
        return tags

    async def _validate_warm_start_job(
        self,
        ctx: RequestContext,
        warm_start_job_id: str,
        current_task_type: TaskType,
    ) -> None:
        """Validates warm start job before starting a new training job.
        Raises an error if one of the following is NOT met:
        1. warm start job exists and is complete
        2. warm start job's predictive query task
        type is the same as the current training job's task type.
        3. the warm start job's predictive query task type is not forecasting.
        """
        from kumo_core.automl.runtime import validate_warm_start_task_type

        validate_warm_start_task_type(current_task_type)

        warm_start_job_id_obj = TrainingJobID(warm_start_job_id)
        warm_start_job = await self.get_training_job(
            ctx, warm_start_job_id_obj, strip_expensive_fields=True)
        if not warm_start_job:
            raise ValueError(
                f"Warm start training job {warm_start_job_id_obj} not found")
        if warm_start_job.execution.status != JobStatus.DONE:
            raise ValueError(
                f"Warm start training job {warm_start_job_id_obj} is not complete. "
                f"Current status: {warm_start_job.execution.status}")

        warm_start_pq_id = await self.get_training_job_pquery_id_v2(
            ctx, warm_start_job_id_obj)
        if warm_start_pq_id is None:
            raise ValueError(
                f"Unable to find predictive query for warm start job {warm_start_job_id_obj}"
            )
        warm_start_task_type, _ = await self.get_pquery_metadata(
            warm_start_pq_id, ProjectID(warm_start_job.project_id))

        validate_warm_start_task_type(current_task_type, warm_start_task_type)

    async def _get_num_entities_from_job_dir(self, home_dir: str):
        batch_pred_table_stats_path = HomeDirManager(home_dir).pred_table_stats
        if not fsutil.exists(batch_pred_table_stats_path):
            log.debug(f"Stats file not found at {batch_pred_table_stats_path}")
            return None
        pred_table_stats = load_json(batch_pred_table_stats_path,
                                     Dict[str, Dict[str, Any]])
        num_entities_stat = Stats.from_config(
            pred_table_stats.get(NUM_ENTITIES))
        if num_entities_stat is not None:
            assert isinstance(num_entities_stat, Stats)
            num_entities = num_entities_stat.value
            return num_entities
        else:
            log.warning(
                "Num entities prediction table stats are unavailable in the job directory."
            )
            return None

    async def _get_num_entities_from_cache(
        self,
        job_id: BatchPredictionJobID,
    ):
        cache: BackendCacheManager = get_active_cache()
        pred_table_stats = cache.pquery_table_stats_cache.get_stats(
            job_id, [NUM_ENTITIES])
        num_entities_stat = pred_table_stats.get(NUM_ENTITIES)
        if num_entities_stat is not None:
            assert isinstance(num_entities_stat, Stats)
            num_entities = num_entities_stat.value
            return num_entities
        else:
            log.warning(
                "Num entities prediction table stats are unavailable in the cache."
            )
            return None

    async def training_to_sdk(
        self,
        job_id: TrainingJobID,
        ctx: RequestContext,
    ):
        """
        Used to generate a fully executable Python SDK script to reproduce a training pipeline
        in the SDK. Everything up to generating a graph is covered by the method
        `GraphLogic.get_graph_setup_for_sdk`. This method generates:
            1. Predictive query config,
            2. Model plan config,
            3. Training table config,
            4. Training job config,
            5. Initiating a training table and training job with the configs.

        The output of this method is added to the output of `GraphLogic.get_graph_setup_for_sdk`.
        This method is part of the UI to SDK feature and will be called in the UI.

        Args:
            training_job_id (TrainingJobID): ID of the training job.
            ctx (RequestContext): Request context.
        Response:
            str: The complete python script for an SDK training workflow as a string.
        """
        training_job = await self.get_training_job(ctx, job_id,
                                                   strip_expensive_fields=True)

        if not training_job:
            raise TrainingJobDoesNotExistError(job_id)

        pq_fpid = training_job.pquery_def_fpid
        if not pq_fpid:
            raise ValueError(
                f"Predictive query definitiion with id='{pq_fpid}' does not exist."
            )
        pq_def = await self._require_pquery_service().get_definitions({pq_fpid})
        if pq_fpid not in pq_def:
            raise ValueError(
                f"Predictive query definitiion with id='{pq_fpid}' does not exist."
            )

        project_id = ProjectID(training_job.project_id)
        configs = await self._graph_logic(
            project_id).get_graph_metadata_for_sdk(pq_def[pq_fpid].graph_fpid)
        connector_configs, table_configs, edge_configs = configs

        pq_config: Dict[str, str] = {
            "graph":
            "\\" + "graph" + "\\",
            "query_string":
            clean_up_pquery_string(pq_def[pq_fpid].query_str, table_configs),
        }

        if not training_job.advanced_options:
            raise ValueError(
                f"Model plan for training job with id {job_id} cannot be found."
            )
        model_plan = training_job.advanced_options

        model_plan_as_dict: Dict[
            str, Any] = recursively_convert_dataclass_to_dict(model_plan)
        model_plan_as_dict = render_str_vals_as_strings(model_plan_as_dict)

        encoders = await self._ml_job_backend.get_encoders_for_job(ctx, job_id)

        # Remove `training_table_generation` as a separate object
        # as it is not part of SDK/API ModelPlan class.
        if model_plan.training_table_generation:
            training_table_gen_plan: Dict[str, Any] = model_plan_as_dict.pop(
                "training_table_generation")

        else:
            if not training_job.generate_train_table_job_id:
                raise ValueError(
                    f"Training table job for training job with id {job_id} cannot be found."
                )
            train_table_job = await self.get_training_table_generation_job(
                ctx,
                GenerateTrainTableJobID(
                    training_job.generate_train_table_job_id))

            training_table_gen_plan: Dict[
                str, Any] = recursively_convert_dataclass_to_dict(
                    train_table_job.config.plan)

        graph_script = generate_conn_graph_or_table_sdk_script(
            connector_configs,
            table_configs,
            edge_configs,
            project_id=project_id,
        )
        training_script = generate_training_job_sdk_script(
            pq_config, model_plan_as_dict, training_table_gen_plan,
            training_job.generate_train_table_job_id, encoders)
        return make_stype_and_dtype_sdk_compat(graph_script + training_script)
