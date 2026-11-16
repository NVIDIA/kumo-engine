# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import os.path as osp
from collections import defaultdict
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

from kumo_core.explain.schemas import EntityOutput, HistoricalItems
from kumo_core.explain.schemas import Subgraph as OldSubgraph
from kumo_core.model_plan import ModelPlan, ModelPlanValidationError
from kumo_core.train.output import TargetPredictionHeatMap
from kumo_core.train.stats import AllTargetRegressionStats
from kumo_core.training.config import TrainerConfig
from kumo_core.types import TaskTypeWithReason
from kumo_core.types.schema import FPID
from kumoapi.encoder import Encoder
from kumoapi.explain import (
    ColumnAnalysisOutput,
    ColumnCohorts,
    Feature,
    LinkPredRFM,
    NodePredRFM,
    Subgraph,
    SubgraphFeatureStore,
)
from kumoapi.model_plan import RunMode
from kumoapi.pquery import QueryType
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype
from kumopql.parser import PQLParser
from kumopql.rewriting.bp_filter_transformer import (
    BPFilterTransformer,
    EditableQueryResponse,
)
from kumopql.validator import PredictiveQueryValidator

from async_utils.instrument import measure_blocking_code
from async_utils.sync import blocking_func
from io_utils import fsutil, json, logging
from io_utils.json import load_json_async
from kumo.config.cohort import DataCube, DimensionSpec
from kumo.config.explain import StrategyMappings
from kumo.config.graph import TableConfig
from kumo.config.mixin import construct_object_string_from_config
from kumo.config.model_plan_adapter import to_predictive_query_kwargs
from kumo.config.model_plan_yaml import (
    model_plan_from_yaml,
    update_model_plan_from_yaml,
)
from kumo.config.predictive_query import PredictiveQueryV2Config
from kumo.config.task import SplitConfig
from kumo.graph import Graph
from kumo.graph.table import Table
from kumo.graph.utils import map_to_api_graph_definition
from kumo.home import HomeDirManager
from kumo.pquery.generator_utils import get_task_type
from kumo.pquery.predictive_query import PredictiveQuery, PredictiveQueryV2
from kumo.pquery.utils import validator_kwargs
from kumo.utils.embedding_models import validate_text_embedding_models
from kumo.validation import (
    ValidationError,
    ValidationException,
    ValidationResponse,
)
from monitoring.tracing import TraceOptions, trace
from rest.backends.data_backend import connector_activity_registry
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.data_backend import DataBackend
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.data_backend import (
    ConnectorExecutionRef,
    GraphDataVersionInfo,
    ValidatedPredictiveQuerySnapshot,
    bind_graph_config_to_execution_refs,
)
from rest.config.evaluation import (
    ColumnExplanation,
    NavigationInfo,
    RegressionHeatmapResponse,
    RegressionResiduals,
    XAIResponse,
)
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query import (
    DefaultModelPlanInfo,
    DefaultModelPlanRequest,
    PQueryDefinition,
    PQueryRuntimeOptions,
    RESTParsedPredictiveQuery,
    compute_pquery_fpid,
    pquery_runtime_options_or_none,
)
from rest.config.predictive_query_v2 import (
    PQueryResource,
    PQueryValidationArgs,
)
from rest.config.training_job import TrainingJobResponse, TrainingSpec
from rest.deps.authentication import RequestContext
from rest.deps.context import _current_user_ctx
from rest.driver import graph_logic
from rest.driver.error_constants import (
    DEPRECATED_TEMPORAL_FILTER_OVERRIDE_MESSAGE,
    DEPRECATED_TEMPORAL_FILTER_OVERRIDE_TITLE,
    NON_EDITABLE_TARGET_FILTER_MESSAGE,
    NON_EDITABLE_TARGET_FILTER_TITLE,
    PREDICTIVE_QUERY_CREATION_ERROR_MESSAGE,
    PREDICTIVE_QUERY_CREATION_ERROR_TITLE,
)
from rest.settings import PredictiveQuerySettings
from rest.utils.driver_common import (
    authorized_graph_for_activity,
    credential_free_graph_config_for_activity,
    fail_closed_graph_from_config,
    get_current_timestamps,
)
from rest.utils.explain import (
    AGGR_FEATURE_FILE_NAME,
    CATEGORY_RFM_DIR,
    GLOBAL_AGGR_FEATURE_DIR,
    aggr_list_features,
    group_merge_topk_feats,
)
from rest.utils.pquery import (
    make_predictive_config_with_graph,
    resolve_train_table_options,
)
from statedb.config.common import (
    ColumnID,
    ConnectorID,
    GraphIDv2,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TableID,
    TableName,
    TrainingJobID,
    to_graph_id,
    to_table_id,
)
from statedb.config.job_tag import TAG_KEY_PQUERY_ID, JobTagValue
from statedb.config.predictive_query import FilterOverrideArgs
from statedb.config.table import TableDataVersion
from statedb.db.services import StateServices
from statedb.rest import connector_authorization
from statedb.utils.errors import PredictiveQueryDoesNotExistError

log = logging.getLogger(__name__)

_INVALID_MODEL_PLAN_TITLE = 'Invalid model plan'
_INVALID_MODEL_PLAN_MESSAGE = (
    'Check its field names, values, and expression syntax.')


@connector_activity_registry.connector_authorized_activity
def validate_predictive_query_activity(
    args: PQueryValidationArgs,
    graph_config: Any,
    pquery_settings: PredictiveQuerySettings,
    connector_refs: Dict[ConnectorID, ConnectorExecutionRef],
    skip_validation: bool = False,
) -> ValidatedPredictiveQuerySnapshot:
    """Validate a PQuery with identity credentials inside the data worker.

    The result contains a canonical config and validation response, but no
    connector configurations or credentials.

    Raises:
        SourceTableNotFoundError: If a graph's physical source table no longer
            exists.
        SourceTableUnavailableError: If a physical source is missing or
            inaccessible and the provider cannot distinguish the cases.
        ValidationException: If graph or predictive-query validation fails.
    """
    with authorized_graph_for_activity(graph_config, connector_refs) as graph:
        if not skip_validation:
            graph.validate(raise_on_error=True, ignore_encoders=True)
        config: PredictiveQueryV2Config | None = None
        try:
            train_options, train_options_provided = (
                resolve_train_table_options(args.model_plan,
                                            args.train_table_options))
            config = make_predictive_config_with_graph(
                name=args.name or '',
                query_yaml=args.query_string,
                train_options=train_options,
                train_options_provided=train_options_provided,
                graph=graph,
                list_distinct_target_is_array=(
                    args.list_distinct_target_is_array),
                lp_bp_layout=args.lp_bp_layout,
            )
        except ModelPlanValidationError:
            pass
        # Raise outside the active exception context. The parser error can
        # contain arbitrary model-plan input and must not survive in the
        # public exception's cause/context chain.
        if config is None:
            raise ValidationException(
                title=_INVALID_MODEL_PLAN_TITLE,
                message=_INVALID_MODEL_PLAN_MESSAGE,
            ) from None
        pquery = PredictiveQueryV2.from_config(
            config,
            graph=graph,
            pquery_settings=pquery_settings.get_pquery_settings(),
            skip_validation=skip_validation,
        )
        # Preserve user/config semantics such as ``forecast_length=None``.
        # ``pquery.config()`` serializes constructor defaults (for example 1
        # for a non-forecast query), which would later look like an explicit
        # model-plan override. Only the graph needs activity-hydrated canonical
        # metadata and data versions.
        canonical_config = deepcopy(config)
        canonical_config.graph = pquery.graph.config()
        canonical_config.graph.data_source_dict = {}
        return ValidatedPredictiveQuerySnapshot(
            config=canonical_config,
            validation_response=pquery.get_validate_response(),
        )


@blocking_func()
def _build_request_predictive_query(
    config: PredictiveQueryV2Config,
    graph: Graph,
    pquery_settings: PredictiveQuerySettings,
    skip_validation: bool,
) -> PredictiveQueryV2:
    """Reconstruct a PQuery without installing credentials or provider I/O."""
    return PredictiveQueryV2._from_validated_config(
        config,
        graph=graph,
        pquery_settings=pquery_settings.get_pquery_settings(),
    )


class PQueryLogic:
    def __init__(
        self,
        data_backend_ctx: DataBackendContext,
        *,
        state_services: StateServices,
        project_id: ProjectID,
    ):
        if state_services is None:
            raise RuntimeError("PQueryLogic requires StateServices")
        self._data_backend_ctx = data_backend_ctx
        self._state_services = state_services
        self._project_id = project_id

    @property
    def _pquery_service(self):
        return self._state_services.pqueries

    @property
    def _graph_service(self):
        return self._state_services.graphs

    @property
    def _effective_ml_job_service(self):
        return self._state_services.ml_jobs

    def _require_pquery_service(self):
        return self._pquery_service

    def _require_graph_service(self):
        return self._graph_service

    def _require_ml_job_service(self):
        return self._effective_ml_job_service

    async def _pquery_exists(self, id: PredictiveQueryID) -> bool:
        return await self._require_pquery_service().get_definition_by_id(
            id) is not None

    async def _get_pquery_definition_by_id(
        self,
        id: PredictiveQueryID,
    ) -> Optional[PQueryDefinition]:
        return await self._require_pquery_service().get_definition_by_id(id)

    async def _get_graph_def(
        self,
        graph_id: GraphIDv2,
    ) -> Optional[GraphDefinition]:
        _, graph_fpid = GraphIDv2.parse(graph_id)
        if graph_fpid is None:
            return None
        return await self._graph_service.get_graph_definition(
            self._project_id, graph_fpid)

    def _new_graph_logic(self):
        return graph_logic.GraphLogic(
            self._data_backend_ctx,
            state_services=self._state_services,
            project_id=self._project_id,
        )

    async def _make_graph_from_graph_def_v2(
        self,
        graph_def: GraphDefinition,
    ) -> Graph:
        return await self._new_graph_logic()._make_graph_from_graph_def_v2(
            graph_def)

    async def _create_full_graph_def_v2(
        self,
        graph_def: GraphDefinition,
    ) -> Graph:
        return await self._new_graph_logic().create_full_graph_def_v2(graph_def)

    async def _delete_full_graph_def_v2(self, graph_id: GraphIDv2) -> bool:
        _, graph_fpid = GraphIDv2.parse(graph_id)
        if graph_fpid is None:
            return False
        return await self._require_graph_service(
        ).decrement_or_delete_graph_definition(self._project_id, graph_fpid)

    @staticmethod
    def _real_connector_ids_from_graph_definition(
        graph_def: GraphDefinition, ) -> Set[ConnectorID]:
        return {table.get_connector_id() for table in graph_def.tables.values()}

    @staticmethod
    def _real_connector_ids_from_graph_config(graph, ) -> Set[ConnectorID]:
        return {
            ConnectorID(connector_id)
            for connector_id in graph.data_source_dict
        }

    @trace(TraceOptions(trace_params=True))
    async def parse_predictive_query_v2(
        self,
        args: PQueryValidationArgs,
        config: Optional[PredictiveQueryV2Config],
        auto_trainer_options: Optional[ModelPlan],
        pquery_settings: PredictiveQuerySettings,
        has_train_table_weight_col: bool = False,
        skip_validation: bool = False,
        validated_snapshot: Optional[ValidatedPredictiveQuerySnapshot] = None,
    ) -> Tuple[RESTParsedPredictiveQuery, ValidationResponse]:
        warnings = []
        """
        Parse predictive query validation args with a graph definition.

        Args:
            args (PQueryValidationArgs): Args for PQ that will be
                validated.
            config (Optional[PredictiveQueryV2Config]): Used to
                create a V2 query object.
            auto_trainer_options (Optional[ModelPlan]): If available,
                pass model plan options.
            pquery_settings (PredictiveQuerySettings): Used to
                create a V2 query object.
            has_train_table_weight_col (bool): States if a
                weight column has been specified. (default :obj:`False`)
        Returns:
            Tuple[RESTParsedPredictiveQuery, ValidationResponse]: A tuple
                with the parsed query and validation.
        """

        # Every path below builds the plan from the task defaults plus the
        # user's model-plan YAML; pre-built train-table options have no
        # defined overlay semantics here, so refuse rather than drop them.
        # Raised rather than asserted so `python -O` cannot delete the guard.
        if args.train_table_options is not None:
            raise ValueError(
                'parse_predictive_query_v2 cannot apply pre-built '
                'train_table_options; pass model-plan YAML instead')

        if validated_snapshot is None:
            validated_snapshot = await self._validate_predictive_query_args(
                args,
                pquery_settings,
                skip_validation=skip_validation,
            )
        if config is None:
            config = deepcopy(validated_snapshot.config)
        if auto_trainer_options is None:
            auto_trainer_options = model_plan_from_yaml(args.model_plan or '')

        # Create PQ object
        pquery: PredictiveQuery = (await
                                   self._create_pq_object_from_args_and_config(
                                       config, pquery_settings,
                                       skip_validation=skip_validation,
                                       validated_snapshot=validated_snapshot))
        validation_response = pquery.get_validate_response()
        if len(validation_response.errors) != 0:
            raise ValidationException(
                message=validation_response.error_message())

        task_type = pquery.get_task_type()
        query_type: QueryType = pquery.query_type

        is_temporal_query = query_type == QueryType.TEMPORAL
        has_train_table_time_col = (await
                                    pquery.has_train_table_time_col.coroutine())
        has_train_table_time_col = (is_temporal_query
                                    or has_train_table_time_col)
        model_planner_yaml = ""
        if args.model_plan is not None and args.model_plan != '':
            model_planner_yaml = args.model_plan

        if isinstance(auto_trainer_options, ModelPlan):
            auto_trainer_options = update_model_plan_from_yaml(
                ModelPlan.from_default(
                    task_type=task_type,
                    query_type=query_type,
                    has_train_table_time_col=has_train_table_time_col,
                    run_mode=RunMode.FAST,
                    graph=pquery.graph.config(),
                    entity_table_names=pquery.entity_table_names,
                    target_to_entity_paths=pquery.target_to_entity_paths,
                    has_train_table_weight_col=has_train_table_weight_col,
                ), model_planner_yaml)

            if config.forecast_length is not None:
                auto_trainer_options.training_table_generation.forecast_length = (
                    config.forecast_length)

            _, warnings = auto_trainer_options.validate(
                task_type=task_type,
                query_type=query_type,
                has_train_table_time_col=has_train_table_time_col,
                run_mode=RunMode.FAST,
                graph=pquery.graph.config(),
                entity_table_names=pquery.entity_table_names,
                target_to_entity_paths=pquery.target_to_entity_paths,
                has_train_table_weight_col=has_train_table_weight_col,
            )
            validate_text_embedding_models(auto_trainer_options)

            if config.forecast_length is not None:
                auto_trainer_options.training_table_generation.forecast_length = (
                    config.forecast_length)
        else:
            auto_trainer_options.validate()

        # Update predictive query config with model planner train table
        # generation parameters
        config = to_predictive_query_kwargs(
            auto_trainer_options).apply_overrides(config)

        pquery = (await self._create_pq_object_from_args_and_config(
            config,
            pquery_settings,
            skip_validation=skip_validation,
            validated_snapshot=validated_snapshot,
        ))

        # Get the split configuration from the pquery object. At this point,
        # _create_pq_object_from_args_and_config has applied any split
        # specified in the model planner yaml.
        split_config: Optional[SplitConfig] = None
        assert isinstance(pquery, PredictiveQueryV2)
        split_config = pquery.split.config()

        validation_response = pquery.get_validate_response()
        if len(validation_response.errors) != 0:
            raise ValidationException(
                message=validation_response.error_message())

        is_temporal_query = query_type == QueryType.TEMPORAL
        has_train_table_time_col = (await
                                    pquery.has_train_table_time_col.coroutine())
        has_train_table_time_col = (is_temporal_query
                                    or has_train_table_time_col)

        pq_def_fpid = args.fpid or compute_pquery_fpid(
            pquery_name=args.name or None,
            query_str=args.query_string,
            graph_fpid=args.graph.fpid,
        )
        return RESTParsedPredictiveQuery(
            id=PredictiveQueryID.public(pq_def_fpid),
            config=config,
            predictive_query=pquery,
            auto_trainer_options=auto_trainer_options,
            task_type=task_type,
            query_type=query_type,
            has_train_table_time_col=has_train_table_time_col,
            split_config=split_config,
        ), ValidationResponse(warnings=warnings)

    async def validate_predictive_query_v2(
        self,
        args: PQueryValidationArgs,
        pquery_settings: PredictiveQuerySettings,
        filter_override_args: Optional[FilterOverrideArgs] = None,
        check_unique_id: bool = True,
    ) -> ValidationResponse:
        """
        Validate V2 predictive query using its graph definition.
        1. Making sure we can construct a PredictiveQueryV2Config object from
            the args passed in
        2. Attempting to apply filter overrides (if any) and trying to
            construct the PredictiveQueryV2Config object again
        3. Taking the PredictiveQueryV2Config object (potentially with
            overrides applied) and generating a PredictiveQuery object
        Args:
            args (PQueryValidationArgs): Query args
            pquery_settings (PredictiveQuerySettings): Global settings for what
                is allowed and not allowed in this environment
            filter_override_args (Optional[FilterOverrideArgs], optional): The
                filter overrides that we want to apply. This is not None only
                at Batch Prediction time. This is the method we use to validate
                overrides since they're dependent on the context of the query.
            check_unique_id (bool, optional): Whether or not we should
                validate the name (for adding the query into the DB) or just
                the args. Defaults to True.
        Raises:
            ValidationException: If ``check_unique_id`` is True and the query
                name already exists.
        Returns:
            ValidationResponse: Any other validations go in here and should be
                dealt with by the caller.
        """

        # Validation-only callers may send an unsaved query (empty name)
        # or a user-facing template/display name that is not yet a
        # `PredictiveQueryID`-shaped identifier. Only persisted create paths
        # need to coerce `args.name` into a PredictiveQueryID and check for
        # uniqueness.
        if check_unique_id:
            pq_id = PredictiveQueryID(args.name)
            if await self._pquery_exists(pq_id):
                raise ValidationException(
                    title="Predictive query already exists",
                    message=f"Predictive query with name={pq_id} already exists",
                )

        validation = ValidationResponse()
        try:
            parsed_pq, val = await self.parse_predictive_query_v2(
                args,
                config=None,
                auto_trainer_options=None,
                pquery_settings=pquery_settings,
            )
        except Exception as e:
            return ValidationResponse(errors=[
                ValidationError(
                    title=PREDICTIVE_QUERY_CREATION_ERROR_TITLE,
                    message=PREDICTIVE_QUERY_CREATION_ERROR_MESSAGE.format(
                        str(e)),
                )
            ])
        if not val.ok:
            return val
        validation.merge(val)

        val = (await self.make_v2_query_with_overrides_v2(
            args=args,
            override_args=filter_override_args,
            graph=parsed_pq.predictive_query.graph,
        ))[1]
        validation.merge(val)
        return validation

    async def make_v2_query_with_overrides_v2(
        self,
        args: PQueryValidationArgs,
        override_args: Optional[FilterOverrideArgs],
        graph: Optional[Graph] = None,
    ) -> Tuple[Optional[PQueryValidationArgs], ValidationResponse]:
        """
        Create predictive query validation args with filter overrides.
        """
        if graph is None:
            graph = await self._make_graph_from_graph_def_v2(args.graph)

        validation = ValidationResponse()
        parser = PQLParser()
        validator = PredictiveQueryValidator(
            **validator_kwargs(graph, query_string=args.query_string))

        # Validate the original query
        try:
            parsed_query = parser.to_parsed_predictive_query(args.query_string)
        except (ValidationException, ValueError) as e:
            validation.merge(
                ValidationResponse(errors=[
                    ValidationError(
                        title=PREDICTIVE_QUERY_CREATION_ERROR_TITLE,
                        message=PREDICTIVE_QUERY_CREATION_ERROR_MESSAGE.format(
                            str(e)),
                    )
                ]))
            return None, validation
        predictive_query, response = validator.validate_predictive_query(
            parsed_query)
        if predictive_query is None:
            validation.merge(response)
            return None, validation

        new_query_args = deepcopy(args)
        if override_args:
            if (override_args.temporal_entity_filter_override is not None
                    and len(override_args.temporal_entity_filter_override) > 0):
                validation.merge(
                    ValidationResponse(errors=[
                        ValidationError(
                            title=DEPRECATED_TEMPORAL_FILTER_OVERRIDE_TITLE,
                            message=DEPRECATED_TEMPORAL_FILTER_OVERRIDE_MESSAGE)
                    ]))
                return None, validation
            graph_def = map_to_api_graph_definition(graph)
            transformer = BPFilterTransformer()
            editable_query = transformer.get_bp_editable_query(
                predictive_query,
                (get_task_type(graph, predictive_query)).task_type,
                graph_def,
                original_query_text="",
            )
            user_changes = {}
            for field in editable_query.editable_fields:
                if (field.field_name == 'Entity filter'
                        and override_args.entity_filter_override is not None):
                    user_changes[
                        field.field_name] = override_args.entity_filter_override
                if (field.field_name == 'Target filter'
                        and override_args.target_filter_override is not None):
                    user_changes[
                        field.field_name] = override_args.target_filter_override

            if (override_args.target_filter_override
                    and 'Target filter' not in [
                        f.field_name for f in editable_query.editable_fields
                    ]):
                validation.merge(
                    ValidationResponse(errors=[
                        ValidationError(
                            title=NON_EDITABLE_TARGET_FILTER_TITLE,
                            message=NON_EDITABLE_TARGET_FILTER_MESSAGE)
                    ]))
            editable_query_user_response = EditableQueryResponse(
                editable_query, user_changes)
            task_type = get_task_type(graph, predictive_query).task_type
            editable_query_w_overrides = transformer.insert_filter_overrides(
                editable_query_user_response, graph_def, task_type)[0]
            validation.merge(editable_query_w_overrides.validation_response)
            query_string = editable_query_w_overrides.text
            new_query_args.query_string = query_string
        else:
            # these warnings are only merged if there are no overrides;
            # otherwise they'll get merged with corrected indices by
            # insert_filter_overrides()
            validation.merge(response)

        # Validate the query again after applying overrides
        try:
            parsed_query = parser.to_parsed_predictive_query(
                new_query_args.query_string)
        except (ValidationException, ValueError) as e:
            validation.merge(
                ValidationResponse(errors=[
                    ValidationError(title='Error in creating Predictive Query',
                                    message=str(e))
                ]))
            return None, validation
        predictive_query, response = validator.validate_predictive_query(
            parsed_query)
        if predictive_query is None:
            validation.merge(response)
            return None, validation

        return new_query_args, validation

    async def get_predictive_query_object(
        self,
        id: PredictiveQueryID,
        pquery_settings: PredictiveQuerySettings,
        infer_encoders: bool = False,
        pq_def: Optional[PQueryDefinition] = None,
    ) -> Optional[PredictiveQuery]:
        # Resolve the pquery id (bare name/alias or public fpid form) to its
        # v2 `PQueryDefinition`. Aliases are the single name->fpid boundary.
        query = pq_def
        if query is None:
            query = await self._get_pquery_definition_by_id(id)
            if query is None:
                raise ValidationException(
                    message=(f"The supplied pq id {id} is invalid. "
                             "Please make sure to pass a valid pq id."), )

        if not query:
            return None

        graph_id = GraphIDv2.compose(fpid=query.graph_fpid)
        graph_def = await self._get_graph_def(graph_id)
        if graph_def is None:
            raise ValidationException(
                message=(f"Graph definition with graph fpid "
                         f"{query.graph_fpid}. Please check the graph "
                         f"fpid associated with the query {id}."))
        args = PQueryValidationArgs(
            query_string=query.query_str,
            graph=graph_def,
            name=query.pquery_name if query.pquery_name is not None else "",
            model_plan="",
            runtime_options=pquery_runtime_options_or_none(
                list_distinct_target_is_array=(
                    query.list_distinct_target_is_array),
                lp_bp_layout=query.lp_bp_layout,
            ),
        )
        parsed_pq, _ = await self.parse_predictive_query_v2(
            args,
            config=None,
            auto_trainer_options=None,
            pquery_settings=pquery_settings,
        )
        pquery = parsed_pq.predictive_query
        auto_trainer_options = parsed_pq.auto_trainer_options
        # Apply user-specified encoder overrides
        if infer_encoders:
            if not isinstance(auto_trainer_options, ModelPlan):
                encoder_overrides: Dict[str, Encoder] = (
                    auto_trainer_options.get_parsed_encoder_overrides())
                pquery.graph.apply_encoder_overrides(encoder_overrides)
            await pquery.graph.infer_encoders.coroutine()
        return pquery

    async def _training_job_ids_for_pquery(self,
                                           fpid: FPID) -> List[TrainingJobID]:
        """Training job ids tagged to a pquery, ordered oldest to newest.

        v2 pqueries have no cached `current_training_job_id`; the pquery ->
        training job link lives in job tags.
        """
        filters = {str(TAG_KEY_PQUERY_ID): str(JobTagValue.of_pquery_id(fpid))}
        tagged_jobs = await self._require_ml_job_service().list_job_tags(
            JobType.TRAINING_JOB,
            filters,
        )
        rows = await self._require_ml_job_service().list_training_job_rows(
            job_ids={TrainingJobID(job_id)
                     for job_id in tagged_jobs},
            project_ids={self._project_id},
            sort_created_at="asc",
        )
        return [row.id for row in rows]

    async def _cancel_inflight_training_jobs(
            self, ctx: RequestContext, training_job_ids: List[TrainingJobID],
            ml_job_backend: MLJobBackend) -> bool:
        """Cancel every queued/running job among `training_job_ids` so they
        don't count towards our global concurrency limit.

        Returns True if all in-flight jobs were successfully cancelled
        (vacuously True when none are in-flight).
        """
        all_cancelled = True
        for training_job_id in training_job_ids:
            training_job = await ml_job_backend.get_training_job(
                ctx, training_job_id)
            if (training_job is not None
                    and training_job.execution.status.is_submitted):
                cancelled = await ml_job_backend.cancel_training_job(
                    training_job_id)
                all_cancelled = all_cancelled and cancelled
        return all_cancelled

    async def delete_predictive_query(self, ctx: RequestContext,
                                      id: PredictiveQueryID,
                                      ml_job_backend: MLJobBackend) -> bool:
        pq_def = await self._get_pquery_definition_by_id(id)
        if pq_def is None:
            return False

        # Cancel any in-flight training jobs tagged to this pquery so they
        # don't count towards our global concurrency limit.
        await self._cancel_inflight_training_jobs(
            ctx, await self._training_job_ids_for_pquery(pq_def.fpid),
            ml_job_backend)

        await self._delete_full_graph_def_v2(
            GraphIDv2.compose(fpid=pq_def.graph_fpid))
        if pq_def.pquery_name is not None:
            await self._require_pquery_service().delete_alias(pq_def.pquery_name
                                                              )
        await self._require_pquery_service().delete_definition(pq_def.fpid)
        return True

    @trace(TraceOptions(trace_params=True))
    @measure_blocking_code
    async def make_predictive_config_v2(
        self,
        args: PQueryValidationArgs,
        graph: Optional[Graph] = None,
        skip_validation: bool = False,
    ) -> PredictiveQueryV2Config:
        """
        Make predictive query config using validation args and a graph
        definition rather than a graph ID.

        Args:
            args (PQueryValidationArgs): The predictive query
                validation arguments that use graph definition
                for validation and config generation.

        Response:
            PredictiveQueryV2Config - Used to create a
                PredictiveQueryV2 object.
        """

        train_options, train_options_provided = resolve_train_table_options(
            args.model_plan, args.train_table_options)

        if not graph:
            graph = await self._make_graph_from_graph_def_v2(args.graph)

        # Pass the omitted-as-None sentinel through unchanged; the
        # PredictiveQuery constructor resolves None to the ALLOW_ARRAY_TARGETS
        # feature-flag default (predictive_query.py), so the FK-free LP
        # resolution mode still activates without collapsing the sentinel here.
        return await make_predictive_config_with_graph.coroutine(
            name=args.name if args.name else "",
            query_yaml=args.query_string,
            train_options=train_options,
            train_options_provided=train_options_provided,
            graph=graph,
            graph_config=credential_free_graph_config_for_activity(graph),
            list_distinct_target_is_array=args.list_distinct_target_is_array,
            lp_bp_layout=args.lp_bp_layout,
        )

    @trace(TraceOptions(trace_params=True))
    async def _create_pq_object_from_args_and_config(
        self,
        config: PredictiveQueryV2Config,
        pquery_settings: PredictiveQuerySettings,
        skip_validation: bool = False,
        validated_snapshot: Optional[ValidatedPredictiveQuerySnapshot] = None,
    ) -> PredictiveQueryV2:
        if validated_snapshot is None:
            raise RuntimeError('Request-side PQuery reconstruction requires '
                               'an activity-validated snapshot')

        expected_tables = {
            table.name: str(table.data_source)
            for table in config.graph.tables
        }
        validated_tables = {
            table.name: str(table.data_source)
            for table in validated_snapshot.config.graph.tables
        }
        if validated_tables != expected_tables:
            raise RuntimeError(
                'Predictive query validation returned unexpected graph tables')

        request_config = deepcopy(config)
        request_config.graph = deepcopy(validated_snapshot.config.graph)
        request_config.graph.data_source_dict = deepcopy(
            config.graph.data_source_dict)

        validated_versions = {
            table.name: TableDataVersion(table.data_version)
            for table in request_config.graph.tables
        }
        for table in config.graph.tables:
            table.data_version = validated_versions[table.name]

        # The request process retains only credential-free connector configs.
        # Make every reconstructed connector fail closed if a future code path
        # accidentally attempts provider I/O after the activity has returned.
        graph = fail_closed_graph_from_config(request_config.graph)
        return await _build_request_predictive_query.coroutine(
            request_config,
            graph=graph,
            pquery_settings=pquery_settings,
            skip_validation=skip_validation,
        )

    async def _validate_predictive_query_args(
        self,
        args: PQueryValidationArgs,
        pquery_settings: PredictiveQuerySettings,
        skip_validation: bool = False,
        graph: Optional[Graph] = None,
    ) -> ValidatedPredictiveQuerySnapshot:
        """Validate raw PQL and graph metadata exactly once in an activity."""
        if graph is None:
            graph = await self._make_graph_from_graph_def_v2(args.graph)
        request_graph_config = credential_free_graph_config_for_activity(graph)
        connector_ids = self._real_connector_ids_from_graph_definition(
            args.graph)
        connector_service = self._state_services.connectors
        principal = _current_user_ctx.get()
        references_list = await asyncio.gather(
            *(connector_service.execution_ref_for_project(
                principal,
                self._project_id,
                connector_id,
                connector_authorization.CONNECTOR_PQUERY_VALIDATION_ACTIONS,
            ) for connector_id in sorted(connector_ids)))
        references: Dict[ConnectorID, ConnectorExecutionRef] = {
            reference.connector_id: reference
            for reference in references_list
        }
        workflow_graph_config = bind_graph_config_to_execution_refs(
            request_graph_config,
            references,
        )
        snapshot = await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            validate_predictive_query_activity,
            [
                args,
                workflow_graph_config,
                pquery_settings,
                references,
                skip_validation,
            ],
            f'validate-pquery:{self._project_id}',
        )
        if not isinstance(snapshot, ValidatedPredictiveQuerySnapshot):
            raise TypeError('Predictive query validation returned an invalid '
                            'snapshot')
        if snapshot.config.graph.data_source_dict:
            raise RuntimeError(
                'Predictive query validation returned connector configurations')
        expected_tables = {
            table.name: str(table.data_source)
            for table in request_graph_config.tables
        }
        validated_tables = {
            table.name: str(table.data_source)
            for table in snapshot.config.graph.tables
        }
        if validated_tables != expected_tables:
            raise RuntimeError(
                'Predictive query validation returned unexpected graph tables')
        snapshot.config.graph.data_source_dict = deepcopy(
            request_graph_config.data_source_dict)
        return snapshot

    async def infer_task_type_pqlv2(
        self,
        query: str,
        graph: Graph,
    ) -> TaskTypeWithReason:
        """Infer the task type from a PQLv2 query string

        Args:
            query (str): Query string PQLv2
            graph (Graph): Graph object used for the query

        Returns:
            TaskTypeWithReason: Task type with associated inferral logic if the
                query is valid

        Raises:
            ValidationException: If the query is semantically invalid for the
                graph.
        """
        parser = PQLParser()
        parsed_pq = parser.to_parsed_predictive_query(query)

        validator = PredictiveQueryValidator(
            **validator_kwargs(graph, query_string=query))
        predictive_query, response = validator.validate_predictive_query(
            parsed_pq)

        if not response.ok:
            raise ValidationException(
                title="Invalid predictive query",
                message=response.message(),
            )

        assert predictive_query is not None
        return get_task_type(graph, predictive_query)

    def _set_table_version_in_pq_config(
            self, pq_config: PredictiveQueryV2Config,
            graph_version_info: GraphDataVersionInfo):
        for table in pq_config.graph.tables:
            assert isinstance(table, TableConfig)
            old_version = table.data_version
            new_version = (graph_version_info.table_data_versions[TableName(
                table.table_name)])
            table.data_version = new_version
            log.info(f"Set table data version for {table.table_name} to "
                     f"{new_version} from {old_version}.")
        return pq_config

    async def _make_training_spec_v2(
        self,
        pq_resource: PQueryResource,
        pquery_settings: PredictiveQuerySettings,
        *,
        pq_def_fpid: FPID,
        model_plan: Optional[ModelPlan] = None,
        graph_version_info: Optional[GraphDataVersionInfo] = None,
    ) -> Tuple[TrainingSpec, RESTParsedPredictiveQuery]:
        """Build a `TrainingSpec` from a `PQueryResource`."""
        val_args = PQueryValidationArgs(
            query_string=pq_resource.query_string,
            graph=pq_resource.graph,
            name=pq_resource.name or "NA",
            desc=pq_resource.desc or "",
            fpid=pq_def_fpid,
            runtime_options=pquery_runtime_options_or_none(
                list_distinct_target_is_array=(
                    pq_resource.list_distinct_target_is_array),
                lp_bp_layout=pq_resource.lp_bp_layout,
            ),
        )
        parsed_pq, _ = await self.parse_predictive_query_v2(
            val_args,
            config=None,
            auto_trainer_options=model_plan,
            pquery_settings=pquery_settings,
        )
        pq_config = parsed_pq.config
        auto_trainer_options = parsed_pq.auto_trainer_options
        assert isinstance(auto_trainer_options, ModelPlan)

        if graph_version_info:
            pq_config = self._set_table_version_in_pq_config(
                pq_config, graph_version_info)
        else:
            log.warning('Creating training job spec without a graph version '
                        'info object')
        pq_config.weight_col = (
            auto_trainer_options.training_table_generation.weight_col)

        pq_id = PredictiveQueryID.compose(fpid=pq_def_fpid)
        return TrainingSpec(
            predictive_query=pq_config,
            advanced_options=auto_trainer_options,
            predictive_query_id=pq_id,
            graph_id=GraphIDv2.compose(fpid=pq_resource.graph.fpid),
            task_type=parsed_pq.task_type,
            split=parsed_pq.split_config,
        ), parsed_pq

    async def _require_successful_training_job(
        self, ctx: RequestContext, id: PredictiveQueryID,
        ml_job_backend: MLJobBackend
    ) -> Tuple[TrainingJobResponse, HomeDirManager]:
        """Returns the home dir of the most recent successful training job, or
        a value error if none exists.
        """
        # Check that the predictive query exists.
        pq_def = await self._get_pquery_definition_by_id(id)
        if pq_def is None:
            raise PredictiveQueryDoesNotExistError(id)

        # The most recent training job tagged to this pquery (v2 has no cached
        # `current_training_job_id`; the link lives in job tags).
        job_ids = await self._training_job_ids_for_pquery(pq_def.fpid)
        if not job_ids:
            raise ValidationException(
                message=(
                    "This predictive query has needs to be trained, before "
                    "you can perform this operation."), )
        curr_training_job_id = job_ids[-1]

        existing_job = await ml_job_backend.get_training_job(
            ctx, curr_training_job_id)
        if not existing_job:
            # Return runtime error as this is due to a kumo bug, not user input
            # error.
            raise RuntimeError(
                f"Training job {curr_training_job_id} is missing from the "
                f"ML backend. Please check for database corruption, or "
                f"retrain the predictive query.")

        if existing_job.execution.status != JobStatus.DONE:
            raise ValidationException(
                message=(
                    "This predictive query needs to be trained successfully "
                    "before you can perform this operation. However, the last "
                    f"training job for this pquery has status = "
                    f"{existing_job.execution.status}."), )

        home_dir = await ml_job_backend.get_job_home_dir(curr_training_job_id)
        return (existing_job, HomeDirManager(home_dir))

    async def get_xai(
        self,
        job: TrainingJobResponse,
        home_mgr: HomeDirManager,
        tables: List[Table],
        entity: str,
        bfs_tree: Dict[TableID, List[TableID]],
        stage: TrainingStage,
    ) -> XAIResponse:
        resp = XAIResponse()
        task_name = home_mgr.first_task_name()
        if task_name is None or not fsutil.exists(
                home_mgr.get_best_trainer_config_path(task_name)):
            log.warning('Best trainer config is not available, please '
                        'make sure automl training is completed successfully')
            return resp
        best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
            task_name)
        trainer_config = await load_json_async(
            os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)
        if trainer_config and trainer_config.refit_cfg.refitted_trainval:
            best_trainer_dir = best_trainer_dir.rstrip('/') + '_refit_trainval'
        column_file_path = os.path.join(
            best_trainer_dir,
            f'column_analysis/{stage.value}/column_analysis.json')
        log.info(f"column_file_path: {column_file_path}")
        if not fsutil.exists(column_file_path):
            log.warning(
                f"Column analysis stats could not be found on S3 at "
                f"path '{column_file_path}'. It's possible that the query was "
                f"trained before migrating column analysis to explainer or "
                f"before renaming cohort analysis to column analysis.")
            column_file_path = os.path.join(
                best_trainer_dir,
                f'cohort_analysis/{stage.value}/cohort_analysis.json')
            if not fsutil.exists(column_file_path):
                log.warning(
                    f"Cohort analysis stats could not be found on S3 at "
                    f"path '{column_file_path}'. It's possible that the query "
                    f"was trained before migrating cohort analysis to "
                    f"explainer.")
                return resp

        job_results = job.execution.results
        assert job_results

        # Load all XAI data in parallel.
        column_analysis, explanation = await asyncio.gather(
            json.load_json_async(column_file_path, ColumnAnalysisOutput,
                                 default=None, raise_on_missing=False),
            job_results.load_test_explain.coroutine())

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

        # Populate gradient scores as column_id -> score.
        name_to_table = {
            **{
                table.table_name: table
                for table in tables
            },
            # Backwards compatibility:
            **{
                table.name: table
                for table in tables
            }
        }
        gradient_scores: Dict[str, float] = {}
        scores = explanation.contribution_summary.global_column_contributions
        for col_contrib in scores:
            table_id = to_table_id(name_to_table[col_contrib.table_name].name)
            gradient_scores[str(ColumnID(
                table_id, col_contrib.col_name))] = col_contrib.contribution

        # Populate variance score and column analysis in response.
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
                gradient_score=gradient_scores.get(str(col_id)),
                # TODO (Siyang) This potentially overwrites Cohorts for tables
                # that appear at multiple hops.
                cohort=column_cohorts_to_cube(col_cohort),
            )

        return resp

    async def get_xai_training_job(
        self,
        job: TrainingJobResponse,
        home_mgr: HomeDirManager,
        tables: List[Table],
        entity: str,
        bfs_tree: Dict[TableID, List[TableID]],
        stage: TrainingStage,
    ) -> XAIResponse:
        """
        This method is a duplicate of the `get_xai` method above, with the
        assumption that all table names are unique. It should be merged with
        the `get_xai` method to avoid code duplication and maintain a
        single source of truth.

        The main difference is that this method doesn't check for duplicate
        table names, as it assumes they are already unique. This assumption
        should be validated and the two methods should be combined into a
        single, more robust implementation.

        Args:
            job (TrainingJobResponse): The training job response object.
            home_mgr (HomeDirManager): The home directory manager.
            tables (List[Table]): List of tables in the graph.
            entity (str): The entity table name.
            bfs_tree (Dict[TableID, List[TableID]]): The breadth-first search
                tree of the graph.
            stage (TrainingStage): The training stage (e.g., TEST, TRAIN).

        Returns:
            XAIResponse: The XAI (Explainable AI) response object.
        """
        # This version of XAI only supports unique table names.
        if len(set(table.name for table in tables)) < len(tables):
            raise ValidationException(
                message=(
                    "Unable to parse XAI data due to duplicate table names. "
                    "Please go to the Predictive Query training page "
                    "to access XAI for this query."), )

        resp = XAIResponse()
        task_name = home_mgr.first_task_name()
        if task_name is None or not fsutil.exists(
                home_mgr.get_best_trainer_config_path(task_name)):
            log.warning('Best trainer config is not available, please '
                        'make sure automl training is completed successfully')
            return resp
        best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
            task_name)
        trainer_config = await load_json_async(
            os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)
        if trainer_config and trainer_config.refit_cfg.refitted_trainval:
            best_trainer_dir = best_trainer_dir.rstrip('/') + '_refit_trainval'
        column_file_path = os.path.join(
            best_trainer_dir,
            f'column_analysis/{stage.value}/column_analysis.json')
        if not fsutil.exists(column_file_path):
            log.warning(
                f"Column analysis stats could not be found on S3 at "
                f"path '{column_file_path}'. It's possible that the query was "
                f"trained before migrating column analysis to explainer or "
                f"before renaming cohort analysis to column analysis.")
            column_file_path = os.path.join(
                best_trainer_dir,
                f'cohort_analysis/{stage.value}/cohort_analysis.json')
            if not fsutil.exists(column_file_path):
                log.warning(
                    f"Cohort analysis stats could not be found on S3 at "
                    f"path '{column_file_path}'. It's possible that the query "
                    f"was trained before migrating cohort analysis to "
                    f"explainer.")
                return resp

        job_results = job.execution.results
        assert job_results

        # Load all XAI data in parallel.
        column_analysis, explanation = await asyncio.gather(
            json.load_json_async(column_file_path, ColumnAnalysisOutput,
                                 default=None, raise_on_missing=False),
            job_results.load_test_explain.coroutine())

        root_table_name = None
        navigation_tree: Dict[str, List[str]] = defaultdict(list)
        col_stypes: Dict[str, Optional[Stype]] = {}
        for table in tables:
            table_name = to_table_id(table.name).table_name
            resp.tables[table_name] = {}
            # Populate the root_table if we've found the entity table
            if f"{table_name}.{table.pkey_name}" == entity:
                root_table_name = table_name

            for col in table.cols:
                navigation_tree[table_name].append(f"{table_name}.{col.name}")
                col_stypes[f"{table_name}.{col.name}"] = col.stype

        assert root_table_name is not None

        # Merge BFS tree to navigation.
        for table_id, adj_table_ids in bfs_tree.items():
            adj_table_names = [
                table_id.table_name for table_id in adj_table_ids
            ]
            navigation_tree[table_id.table_name].extend(adj_table_names)
        resp.navigation = NavigationInfo(root_table_name=root_table_name,
                                         navigation_tree=navigation_tree)

        # Populate gradient scores as column_id -> score.
        name_to_table = {
            **{
                table.table_name: table
                for table in tables
            },
            # Backwards compatibility:
            **{
                table.name: table
                for table in tables
            }
        }
        gradient_scores: Dict[str, float] = {}
        scores = explanation.contribution_summary.global_column_contributions
        for col_contrib in scores:
            table_name = name_to_table[col_contrib.table_name].table_name
            gradient_scores[f"{table_name}.{col_contrib.col_name}"] = (
                col_contrib.contribution)

        # Populate variance score and column analysis in response.
        for col_cohort in column_analysis.cohorts:
            table_name = name_to_table[col_cohort.table_name].table_name
            col_id = f"{table_name}.{col_cohort.col_name}"
            if col_cohort.hop == 'all':
                hop_info = 'all hops'
            elif col_cohort.hop == 0:
                hop_info = 'seed entities'
            else:
                hop_info = f'{col_cohort.hop} hop'
            col_name_with_hop = f'{col_cohort.col_name} ({hop_info})'
            resp.tables[table_name][col_name_with_hop] = (
                ColumnExplanation(
                    col_id=col_id,
                    stype=col_stypes.get(col_id),
                    variance_score=col_cohort.prediction_score,
                    gradient_score=gradient_scores.get(col_id),
                    # TODO (Siyang) This potentially overwrites Cohorts for
                    # tables that appear at multiple hops.
                    cohort=column_cohorts_to_cube(col_cohort),
                ))

        return resp

    async def get_available_xai_entities(
        self,
        home_mgr: HomeDirManager,
        stage: TrainingStage,
    ) -> StrategyMappings:
        task_name = home_mgr.first_task_name()
        empty_mappings = StrategyMappings()

        if task_name is None or not fsutil.exists(
                home_mgr.get_best_trainer_config_path(task_name)):
            log.warning('Best trainer config is not available, please '
                        'make sure automl training is completed successfully')
            return empty_mappings

        best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
            task_name)

        mappings_file_path = home_mgr.subgraph_mappings_path(
            stage, best_trainer_dir)
        refitted_mappings_file_path = home_mgr.subgraph_mappings_path(
            stage,
            best_trainer_dir.rstrip('/') + '_refit_trainval')

        # speculatively load both possible StrategyMappings objects while
        # waiting for TrainerConfig
        mappings_task = asyncio.create_task(
            load_json_async(mappings_file_path, StrategyMappings,
                            default=empty_mappings))
        refitted_mappings_task = asyncio.create_task(
            load_json_async(refitted_mappings_file_path, StrategyMappings,
                            default=empty_mappings))

        trainer_config = await load_json_async(
            os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)

        if trainer_config and trainer_config.refit_cfg.refitted_trainval:
            selected_mappings_task = refitted_mappings_task
            discarded_mappings_task = mappings_task
            selected_mappings_file_path = refitted_mappings_file_path
        else:
            selected_mappings_task = mappings_task
            discarded_mappings_task = refitted_mappings_task
            selected_mappings_file_path = mappings_file_path

        discarded_mappings_task.cancel()
        await selected_mappings_task

        if (res := selected_mappings_task.result()) is empty_mappings:
            log.warning("Strategy mappings could not be found on S3 at path "
                        f"'{selected_mappings_file_path}'. Return empty "
                        "mappings instead.")

        return res

    async def get_entity_old_subgraph(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
        stage: TrainingStage,
    ) -> OldSubgraph:
        task_name = home_mgr.first_task_name()
        if task_name is None or not fsutil.exists(
                home_mgr.get_best_trainer_config_path(task_name)):
            log.warning('Best trainer config is not available, please '
                        'make sure automl training is completed successfully')
            return OldSubgraph(id=entity_id, table_name=None, task_type=None)
        best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
            task_name)
        trainer_config = await load_json_async(
            os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)
        if trainer_config and trainer_config.refit_cfg.refitted_trainval:
            best_trainer_dir = best_trainer_dir.rstrip('/') + '_refit_trainval'
        subgraph_file_path = home_mgr.subgraph_json_path(
            stage, entity_id, best_trainer_dir, old=True)
        subgraph = await load_json_async(subgraph_file_path, OldSubgraph,
                                         default=None, raise_on_missing=False)
        if subgraph is None:
            log.warning(
                f"Entity {entity_id} subgraph could not be found on S3 at "
                f"path '{subgraph_file_path}'. Returning an empty subgraph "
                f"instead.")
            return OldSubgraph(id=entity_id, table_name=None, task_type=None)
        return subgraph

    async def get_subgraph(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
        stage: TrainingStage,
    ) -> Subgraph:
        task_name = home_mgr.first_task_name()
        if task_name is None or not fsutil.exists(
                home_mgr.get_best_trainer_config_path(task_name)):
            log.warning('Best trainer config is not available, please '
                        'make sure automl training is completed successfully')
            return Subgraph()
        best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
            task_name)
        trainer_config = await load_json_async(
            os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)
        if trainer_config and trainer_config.refit_cfg.refitted_trainval:
            best_trainer_dir = best_trainer_dir.rstrip('/') + '_refit_trainval'
        subgraph_file_path = home_mgr.subgraph_json_path(
            stage, entity_id, best_trainer_dir, old=False)

        subgraph = await load_json_async(subgraph_file_path, Subgraph,
                                         default=None, raise_on_missing=False)
        if subgraph is None:
            log.warning(
                f"Entity {entity_id} subgraph could not be found on S3 at "
                f"path '{subgraph_file_path}'. Returning an empty subgraph "
                f"instead.")
            return Subgraph()
        return subgraph

    async def get_node_pred_rfm(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
        stage: TrainingStage,
    ) -> NodePredRFM:
        task_name = home_mgr.first_task_name()
        if task_name is None or not fsutil.exists(
                home_mgr.get_best_trainer_config_path(task_name)):
            log.warning('Best trainer config is not available, please '
                        'make sure automl training is completed successfully')
            return NodePredRFM()
        best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
            task_name)
        trainer_config = await load_json_async(
            os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)
        if trainer_config and trainer_config.refit_cfg.refitted_trainval:
            best_trainer_dir = best_trainer_dir.rstrip('/') + '_refit_trainval'
        rfm_file_path = home_mgr.rfm_path(stage, entity_id, best_trainer_dir)
        rfm = await load_json_async(rfm_file_path, NodePredRFM, default=None,
                                    raise_on_missing=False)
        if rfm is None:
            log.warning(f"Entity {entity_id} RFM could not be found on S3 at "
                        f"path '{rfm_file_path}'. Returning an empty rfm "
                        f"instead.")
            return NodePredRFM()
        return rfm

    async def get_category_node_pred_rfm(
        self,
        home_mgr: HomeDirManager,
        category: str,
        stage: TrainingStage,
    ) -> NodePredRFM:
        task_name = home_mgr.first_task_name()
        if task_name is None or not fsutil.exists(
                home_mgr.get_best_trainer_config_path(task_name)):
            log.warning('Best trainer config is not available, please '
                        'make sure automl training is completed successfully')
            return NodePredRFM()
        best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
            task_name)
        trainer_config = await load_json_async(
            os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)
        if trainer_config and trainer_config.refit_cfg.refitted_trainval:
            best_trainer_dir = best_trainer_dir.rstrip('/') + '_refit_trainval'
        subgraph_collector_dir = home_mgr.subgraph_collector_dir(
            stage, best_trainer_dir)
        cat_rfm_dir = osp.join(subgraph_collector_dir, CATEGORY_RFM_DIR)
        rfm_file_path = osp.join(cat_rfm_dir, f'{category}.json')
        rfm = await load_json_async(rfm_file_path, NodePredRFM, default=None,
                                    raise_on_missing=False)
        if rfm is None:
            log.error(f"Category {category} node prediction RFM could not be "
                      f"found on S3 at path '{rfm_file_path}'. Returning an "
                      f"empty rfm instead.")
            return NodePredRFM()
        return rfm

    async def get_category_link_pred_rfm(
        self,
        home_mgr: HomeDirManager,
        category: str,
        stage: TrainingStage,
    ) -> LinkPredRFM:
        task_name = home_mgr.first_task_name()
        if task_name is None or not fsutil.exists(
                home_mgr.get_best_trainer_config_path(task_name)):
            log.warning('Best trainer config is not available, please '
                        'make sure automl training is completed successfully')
            return LinkPredRFM()
        best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
            task_name)
        trainer_config = await load_json_async(
            os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)
        if trainer_config and trainer_config.refit_cfg.refitted_trainval:
            best_trainer_dir = best_trainer_dir.rstrip('/') + '_refit_trainval'
        subgraph_collector_dir = home_mgr.subgraph_collector_dir(
            stage, best_trainer_dir)
        cat_rfm_dir = osp.join(subgraph_collector_dir, CATEGORY_RFM_DIR)
        rfm_file_path = osp.join(cat_rfm_dir, f'{category}.json')
        rfm = await load_json_async(rfm_file_path, LinkPredRFM, default=None,
                                    raise_on_missing=False)
        if rfm is None:
            log.error(f"Link prediction RFM for {category} could not be "
                      f"found on S3 at path '{rfm_file_path}'. Returning an "
                      f"empty rfm instead.")
            return LinkPredRFM()
        return rfm

    async def get_link_pred_rfm(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
        stage: TrainingStage,
    ) -> LinkPredRFM:
        task_name = home_mgr.first_task_name()
        if task_name is None or not fsutil.exists(
                home_mgr.get_best_trainer_config_path(task_name)):
            log.warning('Best trainer config is not available, please '
                        'make sure automl training is completed successfully')
            return LinkPredRFM()
        best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
            task_name)
        trainer_config = await load_json_async(
            os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)
        if trainer_config and trainer_config.refit_cfg.refitted_trainval:
            best_trainer_dir = best_trainer_dir.rstrip('/') + '_refit_trainval'
        rfm_file_path = home_mgr.rfm_path(stage, entity_id, best_trainer_dir)
        rfm = await load_json_async(rfm_file_path, LinkPredRFM, default=None,
                                    raise_on_missing=False)
        if rfm is None:
            log.warning(f"Entity {entity_id} link prediction RFM could not be "
                        f"found on S3 at path '{rfm_file_path}'. Returning an "
                        f"empty rfm instead.")
            return LinkPredRFM()
        return rfm

    async def get_topk_feature(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
        topk: int,
        stage: TrainingStage,
    ) -> List[Feature]:
        r"""Get the topk features for a given entity based on the gradient
        score. If the feature has a null value or no score, it will be
        ignored. The features are sorted by the gradient score in descending
        order. If for example in topk there are features for the same column,
        will merge consecutive ones by using `mean` if the feature is
        numerical or `most_frequent` if the feature is categorical until
        reaching the topk unique features.

        Args:
            home_mgr (HomeDirManager): The home manager.
            entity_id (str): The entity ID.
            topk (int): The number of features to return.
            stage (TrainingStage): The stage such as val or test.

        Returns:
            The topk features.
        """
        subgraph = await self.get_subgraph(home_mgr, entity_id, stage)
        if subgraph.feature_store is None:
            log.warning(f"Entity {entity_id} subgraph could not be found or"
                        f"the feature store is empty. Returning an empty list "
                        f"instead.")
            return []
        return group_merge_topk_feats(subgraph, topk)

    async def get_aggr_feature(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
        table_name: str,
        column_name: str,
        op: str,
        stage: TrainingStage,
    ) -> Feature:
        r"""Get the aggregated feature for a given entity, table, column
        and operation. For numerical operations, if all the values are None,
        will return an empty feature.

        Args:
            home_mgr (HomeDirManager): The home manager.
            entity_id (str): The entity ID.
            table_name (str): The table name.
            column_name (str): The column name.
            op (str): The operation such as sum, mean, most_frequent,
                least_frequent, etc.
            stage (TrainingStage): The stage such as val or test.

        Returns:
            The aggregated feature.
        """
        subgraph = await self.get_subgraph(home_mgr, entity_id, stage)
        if subgraph.feature_store is None:
            log.warning(
                f"Entity {entity_id} subgraph could not be found or"
                f"the feature store is empty. Returning an empty feature "
                f"instead.")
            return Feature()
        table_nodes = subgraph.subgraph_store.get_table_node_ids(table_name)
        if len(table_nodes) == 0:
            log.warning(f"Table {table_name} not found in subgraph. "
                        f"Returning an empty feature instead.")
            return Feature()
        feat_store: SubgraphFeatureStore = subgraph.feature_store
        feats = [feat_store.features[node][column_name] for node in table_nodes]
        aggr_feat = aggr_list_features(feats, op)
        return aggr_feat

    async def get_category_aggr_feature(
        self,
        home_mgr: HomeDirManager,
        category: str,
        table_name: str,
        column_name: str,
        op: str,
        stage: TrainingStage,
    ) -> Feature:
        r"""Get the aggregated feature for a given selection category, table,
        column and operation.

        Args:
            home_mgr (HomeDirManager): The home manager.
            category (str): The entity selection category.
            table_name (str): The table name.
            column_name (str): The column name.
            op (str): The operation such as sum, mean, most_frequent,
                least_frequent, etc.
            stage (TrainingStage): The stage such as val or test.

        Returns:
            The aggregated feature.
        """
        # Operations that are pre-computed for random category
        if op in {'most_frequent', 'mean'} and category == 'random':
            task_name = home_mgr.first_task_name()
            if task_name is None or not fsutil.exists(
                    home_mgr.get_best_trainer_config_path(task_name)):
                log.error('Best trainer config is not available, please '
                          'make sure automl training is completed successfully')
                return Feature()
            best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
                task_name)
            trainer_config = await load_json_async(
                os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)
            if trainer_config and trainer_config.refit_cfg.refitted_trainval:
                best_trainer_dir = best_trainer_dir.rstrip(
                    '/') + '_refit_trainval'
            subgraph_collector_dir = home_mgr.subgraph_collector_dir(
                stage, best_trainer_dir)
            global_aggr_feat_dir = osp.join(subgraph_collector_dir,
                                            GLOBAL_AGGR_FEATURE_DIR)
            category_aggr_feat_dir = osp.join(global_aggr_feat_dir, category)
            table_dir = osp.join(category_aggr_feat_dir, table_name)
            col_dir = osp.join(table_dir, column_name)
            op_dir = osp.join(col_dir, op)
            aggr_feat_file_path = osp.join(op_dir,
                                           f"{AGGR_FEATURE_FILE_NAME}.json")
            feat = await load_json_async(aggr_feat_file_path, Feature,
                                         default=None, raise_on_missing=False)
            if feat is None:
                log.error(f"No aggregated feature found for "
                          f"{table_name}.{column_name} with op {op}")
                return Feature()
            return feat

        mappings = await self.get_available_xai_entities(home_mgr, stage)
        if len(mappings.mappings) == 0:
            log.warning('No available XAI entities found')
            return Feature()
        if category not in mappings.mappings:
            log.warning(f'Category {category} not found')
            return Feature()
        entity_ids = mappings.mappings[category][:10]
        feats: list[Feature] = await asyncio.gather(
            *(self.get_aggr_feature(home_mgr, entity_id, table_name,
                                    column_name, op, stage)
              for entity_id in entity_ids))
        if len(feats) == 0:
            log.warning(f"No features found for category {category}")
            return Feature()
        return aggr_list_features(feats, op)

    async def get_entity_output(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
        stage: TrainingStage,
    ) -> EntityOutput:
        task_name = home_mgr.first_task_name()
        if task_name is None or not fsutil.exists(
                home_mgr.get_best_trainer_config_path(task_name)):
            log.warning('Best trainer config is not available, please '
                        'make sure automl training is completed successfully')
            return EntityOutput(task_type=None)
        best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
            task_name)
        trainer_config = await load_json_async(
            os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)
        if trainer_config and trainer_config.refit_cfg.refitted_trainval:
            best_trainer_dir = best_trainer_dir.rstrip('/') + '_refit_trainval'
        output_file_path = home_mgr.subgraph_output_path(
            stage, entity_id, best_trainer_dir)
        output = await load_json_async(output_file_path, EntityOutput,
                                       default=None, raise_on_missing=False)
        if output is None:
            log.warning(
                f"Entity {entity_id} output could not be found on S3 at "
                f"path '{output_file_path}'. Returning an empty entity output "
                f"instead.")
            return EntityOutput(task_type=None)
        return output

    async def get_entity_historical_items(
        self,
        home_mgr: HomeDirManager,
        entity_id: str,
        stage: TrainingStage,
    ) -> HistoricalItems:
        task_name = home_mgr.first_task_name()
        if task_name is None or not fsutil.exists(
                home_mgr.get_best_trainer_config_path(task_name)):
            log.warning('Best trainer config is not available, please '
                        'make sure automl training is completed successfully')
            return HistoricalItems()
        best_trainer_dir = await home_mgr.get_best_trainer_dir.coroutine(
            task_name)
        trainer_config = await load_json_async(
            os.path.join(best_trainer_dir, 'config.json'), TrainerConfig)
        if trainer_config and trainer_config.refit_cfg.refitted_trainval:
            best_trainer_dir = best_trainer_dir.rstrip('/') + '_refit_trainval'
        items_file_path = home_mgr.subgraph_historical_items_path(
            stage, entity_id, best_trainer_dir)
        items = await load_json_async(items_file_path, HistoricalItems,
                                      default=None, raise_on_missing=False)
        if items is None:
            log.warning(
                f"Entity {entity_id} historical items could not be found on "
                f"S3 at path '{items_file_path}'. Return empty historical "
                f"items instead.")
            return HistoricalItems()
        return items

    async def get_regression_histogram(
        self,
        home_dir_mgr: HomeDirManager,
    ) -> AllTargetRegressionStats:
        stats = json.load_json(home_dir_mgr.all_target_regression_stats,
                               AllTargetRegressionStats, default=None,
                               raise_on_missing=False)

        # It is normal for regression histogram to be missing for queries prior
        # to v1.19, so don't throw an exception if the file is not found.
        if stats is None:
            return AllTargetRegressionStats.empty()

        return stats

    async def get_regression_heatmap(
        self,
        home_dir_mgr: HomeDirManager,
    ) -> RegressionHeatmapResponse:
        task_name = home_dir_mgr.first_task_name()

        # It is normal for regression heatmap to be missing for queries prior
        # to v1.19, so don't throw an exception if the file is not found.
        train_heatmap, val_heatmap, test_heatmap = None, None, None
        if task_name is not None:
            train_heatmap = await load_json_async(
                await home_dir_mgr.get_regression_heatmap_path.coroutine(
                    TrainingStage.TRAIN, task_name), TargetPredictionHeatMap,
                default=None, raise_on_missing=False)
            val_heatmap = await load_json_async(
                await home_dir_mgr.get_regression_heatmap_path.coroutine(
                    TrainingStage.VAL, task_name), TargetPredictionHeatMap,
                default=None, raise_on_missing=False)
            test_heatmap = await load_json_async(
                await home_dir_mgr.get_regression_heatmap_path.coroutine(
                    TrainingStage.TEST, task_name), TargetPredictionHeatMap,
                default=None, raise_on_missing=False)

        residuals = {}
        if train_heatmap is not None:
            residuals[
                TrainingStage.
                TRAIN] = RegressionResiduals.from_target_prediction_heatmap(
                    train_heatmap)
        if val_heatmap is not None:
            residuals[TrainingStage.
                      VAL] = RegressionResiduals.from_target_prediction_heatmap(
                          val_heatmap)
        if test_heatmap is not None:
            residuals[
                TrainingStage.
                TEST] = RegressionResiduals.from_target_prediction_heatmap(
                    test_heatmap)

        return RegressionHeatmapResponse(residuals_chart=residuals)

    async def get_model_plan_info(
        self,
        default_model_plan_request: DefaultModelPlanRequest,
        pquery_settings: PredictiveQuerySettings,
    ) -> DefaultModelPlanInfo:
        """Get the default model plan for a given query.

        The query references a persisted graph by ID; resolve its graph
        definition and delegate to the graph-definition-based model-plan path.
        """
        graph_id = to_graph_id(default_model_plan_request.graph_id)
        assert isinstance(graph_id, GraphIDv2)
        graph_def = await self._get_graph_def(graph_id)
        if graph_def is None:
            raise ValidationException(
                message=(f"Graph definition for graph_id {graph_id} not "
                         "found. Please check the graph id associated with "
                         "the query."))
        default_model_plan_request.graph_def = graph_def
        return await self.get_model_plan_info_with_graph_def(
            default_model_plan_request, pquery_settings)

    async def get_model_plan_info_with_graph_def(
        self,
        default_model_plan_request: DefaultModelPlanRequest,
        pquery_settings: PredictiveQuerySettings,
    ) -> DefaultModelPlanInfo:
        """
        Get default model plan for a given predictive query
        using graph definition instead of graph ID.
        Using graph definition generalizes the approach is more
        light-weight (no need to store the graph in the DB) and
        is works with any graph definition.

        Only used for v2 queries.

        Relies on passing a graph definition as opposed to a graph ID
        to generate the model plan.

        Args:
            query_string (str): The predictive query we want to make a
                model plan for.
            graph_def (GraphDefinition): The graph definition of the
                graph we want to use to construct the model plan.
            run_mode (RunMode): user defined run mode (fast, normal, slow).
            has_weight_col (bool): Yes/no option if there is a weight column
                present in the graph.
        Returns:
            DefaultModelPlanInfo consitiong of model plan, task type,
                query type, and has_train_table_weight_col.
        """
        # Convert request to valdation args
        from kumo_core.training.options import allow_array_targets
        args = PQueryValidationArgs(
            query_string=default_model_plan_request.query_string,
            graph=default_model_plan_request.graph_def,
            name="default",
            model_plan="",
            runtime_options=PQueryRuntimeOptions(
                list_distinct_target_is_array=allow_array_targets()),
        )

        assert default_model_plan_request.graph_def is not None

        parsed_pq, _ = await self.parse_predictive_query_v2(
            args=args,
            config=None,
            auto_trainer_options=None,
            pquery_settings=pquery_settings,
            has_train_table_weight_col=default_model_plan_request.
            has_train_table_weight_col,
        )
        pquery = parsed_pq.predictive_query

        model_plan = ModelPlan.from_default(
            task_type=parsed_pq.task_type,
            query_type=parsed_pq.query_type,
            has_train_table_time_col=parsed_pq.has_train_table_time_col,
            run_mode=RunMode(default_model_plan_request.run_mode),
            graph=pquery.graph.config(),
            entity_table_names=pquery.entity_table_names,
            target_to_entity_paths=pquery.target_to_entity_paths,
            has_train_table_weight_col=default_model_plan_request.
            has_train_table_weight_col,
            split=construct_object_string_from_config(
                await pquery.get_default_split_async()),
            forecast_length=(pquery.forecast_length
                             if pquery.is_forecasting else None),
            top_k=pquery.validated_predictive_query.top_k,
        )

        return DefaultModelPlanInfo(
            model_plan,
            parsed_pq.task_type,
            parsed_pq.query_type,
            has_train_table_weight_col=default_model_plan_request.
            has_train_table_weight_col,
        )

    async def build_pquery_def_v2(
        self,
        pquery_resource: PQueryResource,
        pquery_settings: Optional[PredictiveQuerySettings] = None,
        graph: Optional[Graph] = None,
    ) -> PQueryDefinition:
        # Canonicalize embedded Custom SQL tables so the pquery's ``graph_fpid``
        # (below) matches the canonical fingerprint the graph is persisted under:
        # both callers create/persist the graph first, and that step canonicalizes
        # the same way, so a Custom SQL table whose synthetic name is recomputed
        # here would otherwise leave the pquery referencing a pre-canonical graph
        # fpid that was never saved. Idempotent, and a no-op for a graph with no
        # Custom SQL tables (and for an already-canonical one).
        graph_def = await graph_logic.canonicalize_graph_custom_sql_tables(
            pquery_resource.graph, self._project_id, self._state_services,
            _current_user_ctx.get())
        if graph is None:
            graph = await self._make_graph_from_graph_def_v2(graph_def)
        name = pquery_resource.name if pquery_resource.name else ""
        args = PQueryValidationArgs(
            query_string=pquery_resource.query_string,
            graph=graph_def,
            name=name,
            runtime_options=pquery_runtime_options_or_none(
                list_distinct_target_is_array=(
                    pquery_resource.list_distinct_target_is_array),
                lp_bp_layout=pquery_resource.lp_bp_layout,
            ),
        )
        assert pquery_settings is not None
        snapshot = await self._validate_predictive_query_args(
            args,
            pquery_settings,
            graph=graph,
        )
        parsed_pq, _ = await self.parse_predictive_query_v2(
            args,
            config=None,
            auto_trainer_options=None,
            pquery_settings=pquery_settings,
            validated_snapshot=snapshot,
        )
        config = parsed_pq.config
        pquery = parsed_pq.predictive_query
        validated_lp_bp_layout = config.lp_bp_layout
        if pquery.lp_bp_layout != validated_lp_bp_layout:
            raise ValueError(
                'Validated pquery config lp_bp_layout does not match runtime '
                'pquery lp_bp_layout: '
                f'config={validated_lp_bp_layout!r}, '
                f'pquery={pquery.lp_bp_layout!r}')
        pq_def = PQueryDefinition(
            get_current_timestamps(),
            pquery_name=pquery_resource.name,
            query_str=pquery_resource.query_string,
            graph_fpid=graph_def.fpid,
            list_distinct_target_is_array=(
                pquery_resource.list_distinct_target_is_array),
            task_type=pquery.get_task_type(),
            query_type=pquery.query_type,
            lp_bp_layout=validated_lp_bp_layout,
        )
        return pq_def

    async def create_full_pquery_def_v2(
        self, pquery_resource: PQueryResource,
        pquery_settings: Optional[PredictiveQuerySettings] = None
    ) -> PQueryDefinition:
        graph_def = pquery_resource.graph
        graph = await self._create_full_graph_def_v2(graph_def)
        pq_def = await self.build_pquery_def_v2(
            pquery_resource=pquery_resource,
            pquery_settings=pquery_settings,
            graph=graph,
        )
        pquery_service = self._require_pquery_service()
        fpid = await pquery_service.add_definition(pq_def)
        if pquery_resource.name is not None:
            await pquery_service.set_alias(pquery_resource.name, fpid)
        return pq_def


def column_cohorts_to_cube(cohorts: ColumnCohorts) -> DataCube:
    data = {
        'Column Value': cohorts.cohorts,
        'Population Ratio': cohorts.populations,
        'Target Average Score': cohorts.targets,
        'Predicted Average Score': cohorts.predictions,
    }
    dimensions = [
        DimensionSpec(name='Data Field', all_values=list(data.keys()))
    ]
    # cohorts.targets is None for batch prediction
    if isinstance(cohorts.predictions, list) and \
        (isinstance(cohorts.targets, list) or cohorts.targets is None):
        return DataCube(data, dimensions)

    assert isinstance(cohorts.targets, dict) or cohorts.targets is None
    assert isinstance(cohorts.predictions, dict)
    multiclass_data = {}
    for cls_label in cohorts.predictions:
        if cohorts.targets is not None:
            data['Target Average Score'] = cohorts.targets[cls_label]
        data['Predicted Average Score'] = cohorts.predictions[cls_label]
        multiclass_data[cls_label] = copy(data)
    dimensions.insert(
        0,
        DimensionSpec(name='Class Label',
                      all_values=list(multiclass_data.keys())))
    return DataCube(multiclass_data, dimensions)
