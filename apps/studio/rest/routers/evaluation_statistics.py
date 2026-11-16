# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Annotated, Dict

from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import HTTPException
from kumo_core.explain.schemas import EntityOutput, HistoricalItems
from kumo_core.explain.schemas import Subgraph as OldSubgraph
from kumo_core.train.stats import AllTargetRegressionStats

from io_utils import logging
from kumo.config.explain import ColumnContributionSummary, StrategyMappings
from rest.config.evaluation import (
    ClassificationEvaluationResponse,
    RegressionHeatmapResponse,
    XAIResponse,
)
from rest.config.status import RestStatusCode
from rest.config.training_job import TrainingJobOutput
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.driver.rest_api_driver import RestApiDriver
from rest.utils.routers import wrap_validation_exceptions
from statedb.config.common import PredictiveQueryID, ProjectID
from statedb.utils.errors import PredictiveQueryDoesNotExistError

TRUE_CLASS = 'True'
FALSE_CLASS = 'False'
EPS = 0.000001

ColumnContributionResponse = ColumnContributionSummary
MutualInformationResponse = Dict[str, float]

logger = logging.getLogger(__name__)


class EvaluationStatisticsResource:
    def __init__(self, auth: Authentication, driver: RestApiDriver,
                 pq_driver: PQueryApiDriver) -> None:
        self._driver = driver
        self._pq_driver = pq_driver
        self._auth = auth
        self._pquery_settings = pq_driver._pquery_settings

        self.router = APIRouter(
            prefix="/v0/eval",
            tags=["EvaluationStatistics"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={HTTPStatus.NOT_FOUND: {
                "description": "Not found"
            }},
        )

        self.router.add_api_route(
            "/{id}/binary_classification",
            self.get_binary_classification,
            response_model=ClassificationEvaluationResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/regression_histogram",
            self.get_regression_histogram,
            response_model=AllTargetRegressionStats,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/regression_heatmap",
            self.get_regression_heatmap,
            response_model=RegressionHeatmapResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/xai",
            self.get_xai,
            response_model=XAIResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/available_xai_entities",
            self.get_available_xai_entities,
            response_model=StrategyMappings,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/entity_subgraph/{entity_id}",
            self.get_old_subgraph,
            response_model=OldSubgraph,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/entity_output/{entity_id}",
            self.get_entity_output,
            response_model=EntityOutput,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/entity_historical_items/{entity_id}",
            self.get_entity_historical_items,
            response_model=HistoricalItems,
            methods=['GET'],
        )

    async def get_binary_classification(
        self,
        id: PredictiveQueryID,
        project_id: Annotated[str, Query()],
        ctx: RequestContext = Depends(get_ctx),
    ) -> ClassificationEvaluationResponse:
        r"""Get the eval metrics for a training job .

        Args:
            id (str): Id of the trained predictive query to get eval stats for

        Returns:
            data (ClassificationEvaluationResponse): The requested data

        Errors:
            HTTPStatus.NOT_FOUND: If the training job does not exist.
        """

        with wrap_validation_exceptions(PredictiveQueryDoesNotExistError):
            training_job = await self._get_completed_training_job_for_pq(
                id, project_id=ProjectID(project_id), ctx=ctx)

        if not training_job.results:
            raise HTTPException(
                HTTPStatus.NOT_FOUND,
                RestStatusCode.EVALUATION_NOT_YET_COMPLETED.value)
        if not training_job.results.test_eval:
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                RestStatusCode.EVALUATION_NOT_AVAILABLE.value)

        return ClassificationEvaluationResponse.from_binary_confusion_matrix(
            training_job.results.test_eval.confusion_matrix)

    async def get_regression_histogram(
            self,
            id: PredictiveQueryID,
            project_id: Annotated[str, Query()],
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
        with wrap_validation_exceptions(PredictiveQueryDoesNotExistError):
            return await self._driver.get_regression_histogram(
                ctx, id, project_id=ProjectID(project_id))

    async def get_regression_heatmap(
            self,
            id: PredictiveQueryID,
            project_id: Annotated[str, Query()],
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
        with wrap_validation_exceptions(PredictiveQueryDoesNotExistError):
            return await self._driver.get_regression_heatmap(
                ctx, id, project_id=ProjectID(project_id))

    async def get_xai(
            self,
            id: PredictiveQueryID,
            project_id: Annotated[str, Query()],
            ctx: RequestContext = Depends(get_ctx),
    ) -> XAIResponse:
        r"""Get explanation for a pquery.
        Args:
            id (str): Id of the trained predictive query to get eval stats for.

        Returns (XAIResponse): contain column-level contribution scores and
            column analysis, as well as navigation tree starting from root
            (entity) table.
        """
        with wrap_validation_exceptions(PredictiveQueryDoesNotExistError):
            return await self._driver.get_xai(ctx, id,
                                              project_id=ProjectID(project_id))

    async def get_available_xai_entities(
            self,
            id: PredictiveQueryID,
            project_id: Annotated[str, Query()],
            ctx: RequestContext = Depends(get_ctx),
    ) -> StrategyMappings:
        r"""Get preselection strategy mappings of prediction-level XAI
        for a pquery.

        Args:
            id (str): Id of the trained predictive query to get eval stats for.

        Returns (StrategyMappings): Strategy mappings which contain the
            mappings of preselection strategy and entities.
        """
        with wrap_validation_exceptions(PredictiveQueryDoesNotExistError):
            return await self._driver.get_available_xai_entities(
                ctx, id, project_id=ProjectID(project_id))

    async def get_old_subgraph(
            self,
            id: PredictiveQueryID,
            entity_id: str,
            project_id: Annotated[str, Query()],
            ctx: RequestContext = Depends(get_ctx),
    ) -> OldSubgraph:
        r"""Get XAI subgraph of an entity (old version).

        Args:
            id (str): Id of the trained predictive query to get eval stats for.
            entity_id (str): Id of the entity to get subgraph for.

        Returns (Subgraph): Subgraph of the entity.
        """
        with wrap_validation_exceptions(PredictiveQueryDoesNotExistError):
            return await self._driver.get_entity_old_subgraph(
                ctx, entity_id, id, project_id=ProjectID(project_id))

    async def get_entity_output(
            self,
            id: PredictiveQueryID,
            entity_id: str,
            project_id: Annotated[str, Query()],
            ctx: RequestContext = Depends(get_ctx),
    ) -> EntityOutput:
        r"""Get XAI output of an entity.

        Args:
            id (str): Id of the trained predictive query to get eval stats for.
            entity_id (str): Id of the entity to get subgraph for.

        Returns (EntityOutput): Output of the entity.
        """
        with wrap_validation_exceptions(PredictiveQueryDoesNotExistError):
            return await self._driver.get_entity_output(
                ctx, id, entity_id, project_id=ProjectID(project_id))

    async def get_entity_historical_items(
            self,
            id: PredictiveQueryID,
            entity_id: str,
            project_id: Annotated[str, Query()],
            ctx: RequestContext = Depends(get_ctx),
    ) -> HistoricalItems:
        r"""Get XAI historical items of an entity for link prediction tasks.

        Args:
            id (str): Id of the trained predictive query to get eval stats for.
            entity_id (str): Id of the entity to get subgraph for.

        Returns (HistoricalItems): Historical items of the entity.
        """
        with wrap_validation_exceptions(PredictiveQueryDoesNotExistError):
            return await self._driver.get_entity_historical_items(
                ctx, id, entity_id, project_id=ProjectID(project_id))

    async def _get_completed_training_job_for_pq(
            self,
            id: PredictiveQueryID,
            project_id: ProjectID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> TrainingJobOutput:
        training_job_id = await self._pq_driver.get_current_training_job_id(
            ctx, id, project_id)
        if training_job_id is None:
            raise ValueError("Predictive Query has not been trained yet")

        training_job = await self._driver.pquery_driver.get_training_job(
            ctx, training_job_id)
        if training_job is None:
            raise HTTPException(
                HTTPStatus.NOT_FOUND,
                RestStatusCode.TRAINING_JOB_DOES_NOT_EXIST.value)

        return training_job.execution
