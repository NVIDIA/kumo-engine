# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from http import HTTPStatus
from logging import getLogger
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from kumo_core.explain.schemas import EntityOutput, HistoricalItems
from kumo_core.explain.schemas import Subgraph as OldSubgraph
from kumo_core.train.stats import AllTargetRegressionStats
from kumoapi.data_snapshot import GraphSnapshotID
from kumoapi.explain import Feature, LinkPredRFM, NodePredRFM, Subgraph

from kumo.config.explain import StrategyMappings
from kumo.stats.train_table_stats import (
    SplitTrainTableStats,
    TrainTableGenerationProgress,
)
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.evaluation import (
    ClassificationEvaluationResponse,
    RegressionHeatmapResponse,
    XAIResponse,
)
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query import (
    MaterializationInfoResponse,
    ModelPlansResponse,
    OutputPresignedUrlResponse,
    RESTParsedPredictiveQuery,
)
from rest.config.status import RestStatusCode
from rest.config.training_job import TrainingJobResponse
from rest.config.training_job_v2 import (
    TrainingJobRequest,
    TrainingWorkflowResponse,
)
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.jobs_driver import JobsDriver
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.driver.rest_api_driver import RestApiDriver
from rest.routers import DISABLE_UI_WARNINGS
from rest.utils.project_scope import (
    require_project_id,
    require_requested_project_visible_for_user,
    require_visible_project_resource,
    require_visible_project_resource_for_user,
)
from rest.utils.routers import (
    ValidationHTTPException,
    nullable_response,
    wrap_validation_exceptions,
)
from statedb.config.common import (
    DataSnapshot,
    ErrWarnResponse,
    GenerateTrainTableJobID,
    GraphIDv2,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.utils.errors import (
    EntityDoesNotExistError,
    TrainingJobDoesNotExistError,
)

logger = getLogger(__name__)


class TrainingJobsResource:
    def __init__(
        self,
        auth: Authentication,
        rest_driver: RestApiDriver,
        driver: PQueryApiDriver,
        jobs_driver: JobsDriver,
        ml_job_backend: MLJobBackend,
    ) -> None:
        self._driver = driver
        self._rest_driver = rest_driver
        self._jobs_driver = jobs_driver
        self._ml_job_backend = ml_job_backend
        self._auth = auth
        self._pquery_settings = driver._pquery_settings

        self.router = APIRouter(
            prefix="/v0/training_jobs",
            tags=["TrainingJobs"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={HTTPStatus.NOT_FOUND: {
                "description": "Not found"
            }},
        )

        self.router.add_api_route(
            "/{id}",
            self.get_training_job,
            response_model=TrainingJobResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/train",
            self.train,
            response_model=TrainingJobID,
            methods=['POST'],
            summary="Create a new training job. This will cascadingly also "
            "create graph snapshot job and training table generation job.",
            responses={
                HTTPStatus.UNPROCESSABLE_ENTITY: {
                    'description':
                    "Unable to create a training job with the provided "
                    "inputs"
                }
            },
        )

        self.router.add_api_route(
            "/{id}/workflow",
            self.get_generation_and_training_jobs,
            response_model=TrainingWorkflowResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/materialization_info",
            self.get_materialization_info,
            response_model=MaterializationInfoResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/warnings",
            self.get_warnings,
            response_model=Dict[str, ErrWarnResponse],
            methods=['POST'],
        )

        self.router.add_api_route(
            "/{id}/get_train_table_generation_stats",
            self.get_train_table_generation_stats,
            response_model=TrainTableGenerationProgress,
            methods=['GET'],
            responses=nullable_response(),
        )

        self.router.add_api_route(
            "/{id}/get_train_table_split_stats",
            self.get_train_table_split_stats,
            # Optional: LP (and other pre-stats states) legitimately have no
            # per-stage split stats -- the handler returns None there, which a
            # non-Optional response_model would 500 on (nullable_response only
            # affects the schema docs, not runtime validation).
            response_model=Optional[SplitTrainTableStats],
            methods=['GET'],
            responses=nullable_response(),
        )

        self.router.add_api_route(
            "/{id}/download_holdout",
            self.generate_holdout_presigned_url,
            response_model=OutputPresignedUrlResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/model_plans",
            self.get_model_plans,
            response_model=ModelPlansResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/encoders",
            self.get_encoders_for_training_job,
            response_model=Optional[Dict[str, str]],
            methods=['GET'],
            responses={
                **nullable_response(),
                HTTPStatus.NOT_FOUND: {
                    'description':
                    "The training job does not exist, or it failed or was "
                    "cancelled without producing encoders."
                },
            },
        )

        self.router.add_api_route(
            "/{id}/eval/binary_classification",
            self.get_binary_classification,
            response_model=ClassificationEvaluationResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/regression_histogram",
            self.get_regression_histogram,
            response_model=AllTargetRegressionStats,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/regression_heatmap",
            self.get_regression_heatmap,
            response_model=RegressionHeatmapResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/xai",
            self.get_xai,
            response_model=XAIResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/available_xai_entities",
            self.get_available_xai_entities,
            response_model=StrategyMappings,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/entity_subgraph/{entity_id}",
            self.get_entity_old_subgraph,
            response_model=OldSubgraph,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/subgraph/{entity_id}",
            self.get_subgraph,
            response_model=Subgraph,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/node_pred_rfm/{entity_id}",
            self.get_node_pred_rfm,
            response_model=NodePredRFM,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/category_node_pred_rfm/{category}",
            self.get_category_node_pred_rfm,
            response_model=NodePredRFM,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/link_pred_rfm/{entity_id}",
            self.get_link_pred_rfm,
            response_model=LinkPredRFM,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/topk_feature/{entity_id}/{topk}",
            self.get_topk_feature,
            response_model=List[Feature],
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/category_link_pred_rfm/{category}",
            self.get_category_link_pred_rfm,
            response_model=LinkPredRFM,
            methods=['GET'],
        )

        self.router.add_api_route(
            ("/{id}/eval/category_aggr_feature/{category}"
             "/{table_name}/{column_name}/{op}"),
            self.get_category_aggr_feature,
            response_model=Feature,
            methods=['GET'],
        )

        self.router.add_api_route(
            ("/{id}/eval/aggr_feature/{entity_id}"
             "/{table_name}/{column_name}/{op}"),
            self.get_aggr_feature,
            response_model=Feature,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/entity_output/{entity_id}",
            self.get_entity_output,
            response_model=EntityOutput,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/eval/entity_historical_items/{entity_id}",
            self.get_entity_historical_items,
            response_model=HistoricalItems,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/cancel",
            self.cancel,
            methods=['POST'],
            description="Cancel a training job.",
            response_model=bool,
        )

        self.router.add_api_route(
            "/{training_job_id}/sdk",
            self.training_to_sdk,
            response_model=str,
            methods=['GET'],
        )

    async def cancel(
            self,
            id: TrainingJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> bool:
        project_id = require_project_id(project_id)
        await self.get_training_job(id=id, project_id=project_id, ctx=ctx)
        logger.info('Send cancellation request for job: %s', id)
        return await self._ml_job_backend.cancel_training_job(id)

    async def get_warnings(
            self,
            id: TrainingJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> Dict[str, ErrWarnResponse]:
        """Get all warnings and errors after validating the given TrainingJob

        Returns:
            Dict[ID, ErrWarnResponse] containing errors and warnings.
        """
        await self.get_training_job(id=id, project_id=project_id, ctx=ctx)
        if DISABLE_UI_WARNINGS:
            return {id: ErrWarnResponse(errors=[], warnings=[], info_items=[])}

        with wrap_validation_exceptions(
                not_exists_error=TrainingJobDoesNotExistError):
            return await self._driver.get_training_job_warnings([id])

    async def get_training_job(
        self, id: TrainingJobID, project_id: str,
        ctx: RequestContext = Depends(get_ctx)
    ) -> TrainingJobResponse:
        r"""Get the training job metadata and status

        Args:
            id (str): Id of the training job to get

        Returns: TrainingJobResponse

        Errors:
            HTTPStatus.NOT_FOUND: If the predictive query does not exist.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            require_requested_project_visible_for_user(
                project_id,
                user_authorization=ctx.user_authorization,
                not_found_error_factory=lambda: TrainingJobDoesNotExistError(id
                                                                             ),
            )
            training_job = await self._driver.get_training_job(
                ctx, id, strip_expensive_fields=True)
            return require_visible_project_resource_for_user(
                training_job,
                requested_project_id=project_id,
                user_authorization=ctx.user_authorization,
                not_found_error_factory=lambda: TrainingJobDoesNotExistError(id
                                                                             ),
            )

    async def train(
            self,
            request: TrainingJobRequest,
            ctx: RequestContext = Depends(get_ctx),
    ) -> TrainingJobID:
        """
        Handler for creating a new training job.

        It will create the following entities:
        1. A new graph if the current definition doesn't exist.
        2. A new predictive query.
        And trigger the following jobs:
        1. Graph snapshot job to snapshot the latest data for the Graph.
        2. Training table generation job to generate training tables. Training
        table specs from TrainingJobRequest.training_table_gen_plan will be
        used.
        3. A new training job. Graph data from (1), Training table from (2)
        and ModelPlan from request.model_plan will be used.

        Args:
            request (TrainingJobRequest): The request object containing all
            necessary information to start a training job, including graph
            definition, predictive query, and model plan.
        Returns:
            TrainingJobID: The ID of the initiated training job.
        """
        project_id = request.project_id
        project_id = require_project_id(project_id)
        request = dataclasses.replace(request, project_id=project_id)
        _id = await self._jobs_driver.train(ctx, request)
        return _id

    async def get_generation_and_training_jobs(
        self,
        id: TrainingJobID,
        project_id: str,
        ctx: RequestContext = Depends(get_ctx),
        skip_validation: bool = True,
    ) -> TrainingWorkflowResponse:
        r"""Get the training job metadata and status
        Args:
            id (str): Id of the training job to get
            skip_validation (bool): Skip validation. Defaults to True since it
                is a read-only operation. (default: :obj:`True`)
        Returns: TrainingWorkflowResponse with both job information and the pq
            definition
        Errors:
            HTTPStatus.NOT_FOUND: If the predictive query does not exist.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            require_requested_project_visible_for_user(
                project_id,
                user_authorization=ctx.user_authorization,
                not_found_error_factory=lambda: TrainingJobDoesNotExistError(id
                                                                             ),
            )
            training_job = await self._driver.get_training_job(ctx, id)
            training_job = require_visible_project_resource_for_user(
                training_job,
                requested_project_id=project_id,
                user_authorization=ctx.user_authorization,
                not_found_error_factory=lambda: TrainingJobDoesNotExistError(id
                                                                             ),
            )

            # Get predictive query info
            pq = await self._get_pq_obj(training_job,
                                        skip_validation=skip_validation)
            pqid = PredictiveQueryID.compose(fpid=training_job.pquery_def_fpid)

            pq_definition = await self._driver.get_predictive_query_v2(
                pqid, ProjectID(project_id))

            graph_snapshot_id: GraphSnapshotID | None = None
            graph_definition = GraphDefinition(
                **dataclasses.asdict(pq_definition.graph))

            if training_job.graph_data_version is not None:
                graph_snapshot_id = DataSnapshot.GRAPH.format_id(
                    graph_definition.fpid, training_job.graph_data_version)

            # Keep populating the existing response field for consumers that
            # use it to recover the graph FPID for follow-on jobs.
            legacy_graph_id = GraphIDv2.public(graph_definition.fpid)

            if training_job.generate_train_table_job_id is None:
                return TrainingWorkflowResponse(
                    pquery_definition=pq_definition, training_job=training_job,
                    task_type=pq.task_type, query_type=pq.query_type,
                    legacy_graph_id=legacy_graph_id,
                    job_source=training_job.job_source,
                    graph_snapshot_id=graph_snapshot_id)

            generation_job = (await
                              self._driver.get_training_table_generation_job(
                                  ctx,
                                  GenerateTrainTableJobID(
                                      training_job.generate_train_table_job_id),
                              ))
            return TrainingWorkflowResponse(
                pquery_definition=pq_definition, training_job=training_job,
                generation_job=generation_job, task_type=pq.task_type,
                query_type=pq.query_type, legacy_graph_id=legacy_graph_id,
                job_source=training_job.job_source,
                graph_snapshot_id=graph_snapshot_id)

    async def get_materialization_info(
        self,
        id: TrainingJobID,
        project_id: str,
        ctx: RequestContext = Depends(get_ctx),
    ) -> MaterializationInfoResponse:
        """Returns the materialization task info for training job if
        available.
        """
        await self.get_training_job(id=id, project_id=project_id, ctx=ctx)
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            info = await self._driver.get_training_job_materialization_info(id)
        if info is None:
            status = "Materialization info is not yet available."
        else:
            status = "Materialization info is available."
        return MaterializationInfoResponse(
            status=status,
            info=info,
        )

    async def get_train_table_generation_stats(
        self, id: TrainingJobID, ctx: RequestContext = Depends(get_ctx)
    ) -> Optional[TrainTableGenerationProgress]:
        r"""Get training table generation statistics.

        Args:
            id (str): ID of the training job to get train table generation
            statistics

        Returns:
            TrainTableGenerationProgress class which has training table
            generation statistics.

        Raises:
            - ValueError: if the query does not exists, or not yet trained
        """
        try:
            with wrap_validation_exceptions(TrainingJobDoesNotExistError):
                training_job = await self._driver.get_training_job(ctx, id)
                if training_job is None:
                    raise TrainingJobDoesNotExistError(id)
                # Get predictive query info
                return await self._driver.get_train_table_generation_stats(id)
        except (ValueError, KeyError, NotImplementedError) as err:
            raise ValidationHTTPException(str(err), err)

    async def get_train_table_split_stats(
        self,
        id: TrainingJobID,
        project_id: str,
        ctx: RequestContext = Depends(get_ctx),
    ) -> Optional[SplitTrainTableStats]:
        r"""Get split train table statistics.

        Args:
            id (str): ID of the training job to get split train table
            statistics

        Returns:
            SplitTrainTableStats class which has split train table statistics.

        Raises:
            - ValueError: if the query does not exists, or not yet trained
        """
        await self.get_training_job(id=id, project_id=project_id, ctx=ctx)
        try:
            with wrap_validation_exceptions(TrainingJobDoesNotExistError):
                response = await self._driver.get_train_table_split_stats(id)
                return response
        except (ValueError, KeyError, NotImplementedError) as err:
            raise ValidationHTTPException(str(err), err)

    async def generate_holdout_presigned_url(
            self,
            id: TrainingJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> OutputPresignedUrlResponse:
        r"""Generate the holdout dataset download url.

        Args:
            id (str): ID of the training job to download the holdout data.

        Returns:
            OutputPresignedUrlResponse class which has the presigned url.

        Raises:
            - ValueError: if the query does not exists, the holdout data does
                          not exist, or data expired.
        """
        await self.get_training_job(id=id, project_id=project_id, ctx=ctx)
        try:
            with wrap_validation_exceptions(TrainingJobDoesNotExistError):
                response = await self._driver.generate_holdout_presigned_url(id)
                return response
        except (ValueError, KeyError, NotImplementedError) as err:
            raise ValidationHTTPException(str(err), err)

    async def get_model_plans(
            self,
            id: TrainingJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> ModelPlansResponse:
        r"""Get the model plans for all existing experiments

        Args:
            id (str): ID of the training job to get model plans for

        Returns:
            List[Dict[str, Dict[str, str]]] list of model plans going from
                group name to option name to value

        Raises:
            - ValueError: if the query does not exist
        """
        try:
            with wrap_validation_exceptions(TrainingJobDoesNotExistError):
                training_job = await self._driver.get_training_job(ctx, id)
                training_job = require_visible_project_resource(
                    training_job,
                    requested_project_id=project_id,
                    not_found_error_factory=lambda:
                    TrainingJobDoesNotExistError(id),
                )

                # Get predictive query info
                # Skip validation since it is a read-only operation
                pq = await self._get_pq_obj(training_job, skip_validation=True)

                return (await self._driver.get_model_plans_per_experiment(
                    ctx=ctx,
                    training_job_id=id,
                    task_type=pq.task_type,
                    query_type=pq.query_type,
                ))
        except (ValueError, KeyError, NotImplementedError) as err:
            raise ValidationHTTPException(str(err), err)

    async def get_encoders_for_training_job(
        self, id: TrainingJobID, ctx: RequestContext = Depends(get_ctx)
    ) -> Optional[Dict[str, str]]:
        """
        Returns the encoders used for a specific training job.

        Output: Dictionary mapping from Fully Qualified Table Name to Encoder
        object representation

        {
            connector_id.table_id.column_name: Encoder(p1=2, p2=[41])
        }

        The encoder is meant to be used in the Model Planner and can
        be parsed by ModelPlan.parse_encoder_overrides
        into an encoder object in code

        Persisted encoders are returned regardless of job status. A missing
        training job, or a failed or cancelled job with no encoder artifact,
        returns 404. Missing artifacts for other states return null. Backend
        retrieval failures are propagated.
        """
        with wrap_validation_exceptions(EntityDoesNotExistError):
            return await self._driver.get_encoders_for_training_job(ctx, id)

    async def get_binary_classification(
        self, id: TrainingJobID, ctx: RequestContext = Depends(get_ctx)
    ) -> ClassificationEvaluationResponse:
        r"""Get the eval metrics for a training job .

        Args:
            id (str): Id of the trained predictive query to get eval stats for

        Returns:
            data (ClassificationEvaluationResponse): The requested data

        Errors:
            HTTPStatus.NOT_FOUND: If the training job does not exist.
        """

        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            training_job = await self._driver.get_training_job(ctx, id)

            if not training_job or not training_job.execution.results:
                raise HTTPException(
                    HTTPStatus.NOT_FOUND,
                    RestStatusCode.EVALUATION_NOT_YET_COMPLETED.value)
            if not training_job.execution.results.test_eval:
                raise HTTPException(
                    HTTPStatus.NOT_FOUND,
                    RestStatusCode.EVALUATION_NOT_AVAILABLE.value)

            return (
                ClassificationEvaluationResponse.from_binary_confusion_matrix(
                    training_job.execution.results.test_eval.confusion_matrix))

    async def get_regression_histogram(
            self,
            id: TrainingJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> AllTargetRegressionStats:
        r"""Get histograms for the target column of the training table for a
        regression task. See #7243

        Args:
            id (str): Id of the trained predictive query to get eval stats for

        Returns:
            data (ClassificationEvaluationResponse): The requested data

        Errors:
            HTTPStatus.NOT_FOUND: If the pquery does not exist.
            HTTPStatus.UNPROCESSABLE_ENTITY: for non-regression pqueries
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_regression_histogram(
                ctx=ctx, id=None, job_id=id)

    async def get_regression_heatmap(
            self,
            id: TrainingJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> RegressionHeatmapResponse:
        r"""Get a heatmap for predicted vs actual samples for each train, val,
        test split

        Args:
            id (str): Id of the trained predictive query to get eval stats for

        Returns:
            data (RegressionHeatmapesponse): The requested data

        Errors:
            HTTPStatus.NOT_FOUND: If the pquery does not exist.
            HTTPStatus.UNPROCESSABLE_ENTITY: for non-regression pqueries
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_regression_heatmap(
                ctx=ctx, id=None, job_id=id)

    async def get_xai(
        self, id: TrainingJobID,
        ctx: RequestContext = Depends(get_ctx)) -> XAIResponse:
        r"""Get explanation for a pquery.
        Args:
            id (str): Id of the trained predictive query to get eval stats for.

        Returns (XAIResponse): contain column-level contribution
            scores and column analysis, as well as navigation tree starting
            from root (entity) table.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            response = await self._rest_driver.get_xai(ctx=ctx, id=None,
                                                       job_id=id)
            assert isinstance(response, XAIResponse)
            return response

    async def get_available_xai_entities(
        self, id: TrainingJobID, ctx: RequestContext = Depends(get_ctx)
    ) -> StrategyMappings:
        r"""Get preselection strategy mappings of prediction-level XAI
        for a pquery.

        Args:
            id (str): Id of the trained predictive query to get eval stats for.

        Returns (StrategyMappings): Strategy mappings which contain the
            mappings of preselection strategy and entities.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_available_xai_entities(
                ctx=ctx, id=None, job_id=id)

    async def get_entity_old_subgraph(
        self, id: TrainingJobID, entity_id: str,
        ctx: RequestContext = Depends(get_ctx)) -> OldSubgraph:
        r"""Get XAI subgraph of an entity (old version).

        Args:
            id (str): Id of the trained predictive query to get eval stats for.
            entity_id (str): Id of the entity to get subgraph for.

        Returns (Subgraph): Subgraph of the entity.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_entity_old_subgraph(
                ctx=ctx, id=None, job_id=id, entity_id=entity_id)

    async def get_node_pred_rfm(
        self, id: TrainingJobID, entity_id: str,
        ctx: RequestContext = Depends(get_ctx)) -> NodePredRFM:
        r"""Get node prediction RFM of an entity.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_node_pred_rfm(
                ctx=ctx, id=None, job_id=id, entity_id=entity_id)

    async def get_category_node_pred_rfm(
        self, id: TrainingJobID, category: str,
        ctx: RequestContext = Depends(get_ctx)) -> NodePredRFM:
        r"""Get node prediction RFM of a category, where the category is
        how we select the entities to explain.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_category_node_pred_rfm(
                ctx=ctx, id=None, job_id=id, category=category)

    async def get_link_pred_rfm(
        self, id: TrainingJobID, entity_id: str,
        ctx: RequestContext = Depends(get_ctx)) -> LinkPredRFM:
        r"""Get link prediction RFM of an entity.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_link_pred_rfm(
                ctx=ctx, id=None, job_id=id, entity_id=entity_id)

    async def get_topk_feature(
        self, id: TrainingJobID, entity_id: str, topk: int,
        ctx: RequestContext = Depends(get_ctx)) -> List[Feature]:
        r"""Get topk feature of an entity.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_topk_feature(
                ctx=ctx, id=None, job_id=id, entity_id=entity_id, topk=topk)

    async def get_category_link_pred_rfm(
        self, id: TrainingJobID, category: str,
        ctx: RequestContext = Depends(get_ctx)) -> LinkPredRFM:
        r"""Get link prediction RFM of a category, where the category is
        how we select the entities to explain.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_category_link_pred_rfm(
                ctx=ctx, id=None, job_id=id, category=category)

    async def get_aggr_feature(
        self, id: TrainingJobID, entity_id: str, table_name: str,
        column_name: str, op: str,
        ctx: RequestContext = Depends(get_ctx)) -> Feature:
        r"""Get RFM aggregate feature of an entity.
        """
        if op not in [
                "sum",
                "mean",
                "max",
                "min",
                "most_frequent",
                "least_frequent",
        ]:
            raise HTTPException(
                status_code=400,
                detail=(f"Invalid operation: {op}. "
                        f"Valid operations are: sum, mean, max, "
                        f"min, most_frequent, least_frequent"))
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_aggr_feature(
                ctx=ctx, id=None, job_id=id, entity_id=entity_id,
                table_name=table_name, column_name=column_name, op=op)

    async def get_category_aggr_feature(
        self, id: TrainingJobID, category: str, table_name: str,
        column_name: str, op: str,
        ctx: RequestContext = Depends(get_ctx)) -> Feature:
        r"""Get RFM aggregate feature of a category, where the category is
        how we select the entities to explain.
        """
        if op not in [
                "sum",
                "mean",
                "max",
                "min",
                "most_frequent",
                "least_frequent",
        ]:
            raise HTTPException(
                status_code=400,
                detail=(f"Invalid operation: {op}. "
                        f"Valid operations are: sum, mean, max, "
                        f"min, most_frequent, least_frequent"))
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_category_aggr_feature(
                ctx=ctx, id=None, job_id=id, category=category,
                table_name=table_name, column_name=column_name, op=op)

    async def get_subgraph(
        self, id: TrainingJobID, entity_id: str,
        ctx: RequestContext = Depends(get_ctx)) -> Subgraph:
        r"""Get XAI subgraph of an entity.

        Args:
            id (str): Id of the trained predictive query to get eval stats for.
            entity_id (str): Id of the entity to get subgraph for.

        Returns (Subgraph): Subgraph of the entity.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_subgraph(ctx=ctx, id=None,
                                                        job_id=id,
                                                        entity_id=entity_id)

    async def get_entity_output(
        self, id: TrainingJobID, entity_id: str,
        ctx: RequestContext = Depends(get_ctx)) -> EntityOutput:
        r"""Get XAI output of an entity.

        Args:
            id (str): Id of the trained predictive query to get eval stats for.
            entity_id (str): Id of the entity to get subgraph for.

        Returns (EntityOutput): Output of the entity.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_entity_output(
                ctx=ctx, id=None, job_id=id, entity_id=entity_id)

    async def get_entity_historical_items(
        self, id: TrainingJobID, entity_id: str,
        ctx: RequestContext = Depends(get_ctx)) -> HistoricalItems:
        r"""Get XAI historical items of an entity for link prediction tasks.

        Args:
            id (str): Id of the trained predictive query to get eval stats for.
            entity_id (str): Id of the entity to get subgraph for.

        Returns (HistoricalItems): Historical items of the entity.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            return await self._rest_driver.get_entity_historical_items(
                ctx=ctx, id=None, job_id=id, entity_id=entity_id)

    async def _get_pq_obj(
        self,
        training_job: TrainingJobResponse,
        skip_validation: bool = False,
    ) -> RESTParsedPredictiveQuery:
        """Get the parsed (v2) Predictive Query object for a training job.

        Args:
            training_job (TrainingJobResponse): The training job to get a
                Predictive Query object for
            skip_validation (bool): Skip validation. (default: :obj:`False`)
        """
        assert training_job.pquery_def_fpid is not None
        return await self._driver.get_parsed_pq_v2_by_fpid(
            training_job.pquery_def_fpid,
            ProjectID(training_job.project_id),
            skip_validation=skip_validation,
        )

    async def training_to_sdk(
            self,
            training_job_id: TrainingJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> str:
        r"""
        Used to generate a fully executable Python SDK script to reproduce a training pipeline
        in the SDK. The script contains:
            1. Connector(s) config(s). If there are tables from multiple connectors,
            we load the config for each connector.
            2. Table configs structured as Python dictionaries.
            3. Graph edge configs.
            4. Using the aforementioned configs to setup a graph in the SDK.
            5. Predictive query config.
            6. Training table config.
            7. Model plan config.
            8. Use the graph and configs to generate a training table and training jobs.
               Additionally, give the option to load a training table with ID
               instead of generating a new one.

        This method is part of the UI to SDK feature and will be called in the UI.

        Args:
            training_job_id (TrainingJobID): ID of the training job.
            ctx (RequestContext): Request context.
        Response:
            str: The complete python script for an SDK training workflow as a string.

        Raises:
            HTTPException: If training job cannot be retrieved.
        """
        with wrap_validation_exceptions(TrainingJobDoesNotExistError):
            await self.get_training_job(id=training_job_id,
                                        project_id=project_id, ctx=ctx)
            return await self._driver.training_to_sdk(training_job_id, ctx)
