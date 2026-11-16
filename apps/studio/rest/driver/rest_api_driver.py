# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

from kumo_core.explain.schemas import EntityOutput, HistoricalItems
from kumo_core.explain.schemas import Subgraph as OldSubgraph
from kumo_core.train.stats import AllTargetRegressionStats
from kumo_core.types.schema import FPID
from kumoapi.explain import Feature, LinkPredRFM, NodePredRFM, Subgraph
from kumoapi.train import TrainingStage

from async_utils.instrument import measure_blocking_code
from io_utils import logging
from kumo.config.explain import StrategyMappings
from kumo.home import HomeDirManager
from kumo.pquery.predictive_query import PredictiveQuery
from kumo.pquery.utils import table_name as pquery_table_name
from kumo.validation import ValidationException
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.api_command import FixtureDefinition, FixturePQueryRecord
from rest.config.evaluation import RegressionHeatmapResponse, XAIResponse
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query import (
    PQueryDefinition,
    pquery_runtime_options_or_none,
)
from rest.config.predictive_query_v2 import PQueryValidationArgs
from rest.config.table_v2 import TableDefinition
from rest.deps.authentication import Authentication, RequestContext
from rest.driver.artifact_api_driver import ArtifactApiDriver
from rest.driver.batch_prediction_job_logic import BatchPredictionJobLogic
from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.driver.data_snapshot_api_driver import DataSnapshotApiDriver
from rest.driver.graph_driver import GraphDriver
from rest.driver.jobs_driver import JobsDriver
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.driver.pquery_logic import PQueryLogic
from rest.driver.projects_driver import ProjectsDriver
from rest.driver.table_driver import TableDriver
from rest.driver.tensorboard_api_driver import TensorboardApiDriver
from rest.settings import RestApiDriverSettings
from statedb.config.common import (
    BatchPredictionJobID,
    GraphID,
    GraphIDv1,
    GraphIDv2,
    JobStatus,
    PredictiveQueryID,
    ProjectID,
    TableID,
    TableIDv2,
    TableName,
    TrainingJobID,
)
from statedb.db.auth import AuthPrincipal
from statedb.db.services import StateServices
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS
from statedb.utils.errors import (
    GraphDoesNotExistError,
    TableDoesNotExistError,
    TrainingJobDoesNotExistError,
)

log = logging.getLogger(__name__)


class RestApiDriver:
    def __init__(
        self,
        settings: RestApiDriverSettings,
        auth: Authentication,
        ml_job_backend: MLJobBackend,
        data_backend_ctx: DataBackendContext,
        projects_driver: Optional[ProjectsDriver] = None,
    ):
        self._settings = settings
        self._auth = auth
        self._ml_job_backend = ml_job_backend
        self._data_backend_ctx = data_backend_ctx
        self._state_services: Optional[StateServices] = None
        self._table_driver = TableDriver(
            self._data_backend_ctx,
            settings.data_scale_limits,
        )
        self._graph_driver = GraphDriver(
            self._data_backend_ctx,
            settings.data_scale_limits,
            self._table_driver,
        )
        self._pquery_api_driver = PQueryApiDriver(
            self._ml_job_backend,
            self._data_backend_ctx,
            self._settings.allow_local_batch_pred_paths,
            data_scale_limits=settings.data_scale_limits,
            pquery_settings=settings.pquery_settings,
            table_driver=self._table_driver,
        )
        self._tensorboard_driver = TensorboardApiDriver()
        self._connector_api_driver = ConnectorApiDriver(
            self._data_backend_ctx,
            settings.supported_connector_types,
            allow_dev_file_paths=settings.allow_local_connector_file_paths,
            data_scale_limits=settings.data_scale_limits,
        )
        self._data_snapshot_api_driver = DataSnapshotApiDriver(
            self._data_backend_ctx)
        self._artifact_export_api_driver = ArtifactApiDriver(
            self._ml_job_backend, self._data_backend_ctx)
        self._jobs_driver = JobsDriver(
            self._pquery_api_driver,
            self._data_snapshot_api_driver,
            self._ml_job_backend,
            self._graph_driver,
            self._connector_api_driver,
        )

    def _configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services
        self.connector_driver._configure_state_services(state_services)
        self.table_driver._configure_state_services(state_services)
        self.graph_driver._configure_state_services(state_services)
        self.pquery_driver._configure_state_services(state_services)
        self.data_snapshot_api_driver._configure_state_services(state_services)
        self._artifact_export_api_driver._configure_state_services(
            state_services)

    def _require_state_services(self) -> StateServices:
        if self._state_services is None:
            raise RuntimeError("RestApiDriver StateServices is not configured")
        return self._state_services

    @property
    def tensorboard_driver(self) -> TensorboardApiDriver:
        return self._tensorboard_driver

    @property
    def pquery_driver(self) -> PQueryApiDriver:
        return self._pquery_api_driver

    @property
    def connector_driver(self) -> ConnectorApiDriver:
        return self._connector_api_driver

    @property
    def data_snapshot_api_driver(self) -> DataSnapshotApiDriver:
        return self._data_snapshot_api_driver

    @property
    def table_driver(self) -> TableDriver:
        return self._table_driver

    @property
    def graph_driver(self) -> GraphDriver:
        return self._graph_driver

    @property
    def artifact_driver(self) -> ArtifactApiDriver:
        return self._artifact_export_api_driver

    async def _get_graph_definition_from_state_services(
        self,
        graph_id: GraphIDv2,
        project_id: ProjectID,
    ):
        _, graph_fpid = GraphIDv2.parse(graph_id)
        if graph_fpid is None:
            raise GraphDoesNotExistError(graph_id)
        graph_definition = await self._require_state_services(
        ).graphs.get_graph_definition(project_id, graph_fpid)
        if graph_definition is None:
            raise GraphDoesNotExistError(graph_id)
        return graph_definition

    def _get_bfs_tree_from_graph_definition(
        self,
        graph_def,
        graph_id: GraphIDv2,
        root_table_name: str,
    ) -> Dict[TableID, List[TableID]]:
        table_name_to_id = {
            TableName(table_name):
            TableIDv2.compose(None, TableName(table_name), table_def.fpid)
            for table_name, table_def in graph_def.tables.items()
        }

        root_table_id = table_name_to_id[TableName(root_table_name)]
        adj_list: Dict[TableID, Set[TableID]] = defaultdict(set)
        for col_group in graph_def.col_groups:
            connected_table_ids: Set[TableID] = {
                table_name_to_id[TableName(col.table_name)]
                for col in col_group.columns
            }
            for table_id in connected_table_ids:
                for adj_table_id in connected_table_ids:
                    if table_id != adj_table_id:
                        adj_list[table_id].add(adj_table_id)

        bfs_tree: Dict[TableID, List[TableID]] = defaultdict(list)
        queue = [root_table_id]
        visited = {root_table_id}
        while queue:
            table_id = queue.pop(0)
            for neighbor in adj_list[table_id]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    bfs_tree[table_id].append(neighbor)
        return bfs_tree

    async def _get_predictive_query_from_definition(
        self,
        pq_def: PQueryDefinition,
        project_id: ProjectID,
    ) -> PredictiveQuery:
        state_services = self._require_state_services()
        graph_def = await state_services.graphs.get_graph_definition(
            project_id, pq_def.graph_fpid)
        if graph_def is None:
            raise GraphDoesNotExistError(
                GraphIDv2.compose(fpid=pq_def.graph_fpid))

        parsed_pq, _ = await PQueryLogic(
            self._data_backend_ctx,
            state_services=state_services,
            project_id=project_id,
        ).parse_predictive_query_v2(
            PQueryValidationArgs(
                query_string=pq_def.query_str,
                graph=graph_def,
                name=pq_def.pquery_name or "NA",
                fpid=pq_def.fpid,
                runtime_options=pquery_runtime_options_or_none(
                    list_distinct_target_is_array=(
                        pq_def.list_distinct_target_is_array),
                    lp_bp_layout=pq_def.lp_bp_layout,
                ),
            ),
            config=None,
            auto_trainer_options=None,
            pquery_settings=self._pquery_api_driver._pquery_settings,
            skip_validation=True,
        )
        return parsed_pq.predictive_query

    @measure_blocking_code
    async def get_xai(
        self,
        ctx: RequestContext,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
        project_id: ProjectID | None = None,
    ) -> XAIResponse:
        state_services = self._require_state_services()
        home_dir_mgr, project_id, job = await self._get_home_dir(
            id=id,
            job_id=job_id,
            ctx=ctx,
            project_id=project_id,
        )
        logic = PQueryLogic(
            self._data_backend_ctx,
            state_services=state_services,
            project_id=project_id,
        )

        predictive_query: Optional[PredictiveQuery] = None
        graph_id: Optional[GraphID] = None
        # UI Jobs
        if id is not None:
            # Resolve the pquery id to its v2 `PQueryDefinition`.
            pq_def = await self._pquery_api_driver.get_pquery_definition_by_id(
                id)
            assert pq_def is not None
            pq_object = await self._get_predictive_query_from_definition(
                pq_def, project_id)
            assert pq_object is not None

            # Configure shared values
            predictive_query = pq_object
            graph_id = GraphIDv2.compose(fpid=pq_def.graph_fpid)
        # SDK Jobs
        else:
            # Get necessary data
            assert job_id is not None
            if job.pquery_def_fpid is None:
                raise ValueError(
                    'XAI is not supported for this Training Job because '
                    'it was trained on an old version of Kumo. '
                    'Please run a new Training Job.')
            pq_def: PQueryDefinition = (
                await self._pquery_api_driver.get_pquery_definition(
                    job.pquery_def_fpid))

            # Configure shared values
            predictive_query = await self._get_predictive_query_from_definition(
                pq_def, project_id)
            graph_id = GraphIDv2.compose(None, pq_def.graph_fpid)

        # Get the BFS tree
        assert isinstance(graph_id, GraphIDv2)
        graph_def = await self._get_graph_definition_from_state_services(
            graph_id, project_id)
        bfs_tree = self._get_bfs_tree_from_graph_definition(
            graph_def, graph_id, pquery_table_name(predictive_query.entity))

        if id is not None:
            return await logic.get_xai(
                job=job,
                home_mgr=home_dir_mgr,
                tables=predictive_query.graph.tables,
                entity=predictive_query.entity,
                bfs_tree=bfs_tree,
                stage=stage,
            )
        else:
            return await logic.get_xai_training_job(
                job=job,
                home_mgr=home_dir_mgr,
                tables=predictive_query.graph.tables,
                entity=predictive_query.entity,
                bfs_tree=bfs_tree,
                stage=stage,
            )

    @measure_blocking_code
    async def get_available_xai_entities(
        self,
        ctx: RequestContext,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
        project_id: ProjectID | None = None,
    ) -> StrategyMappings:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id, project_id=project_id))
        return await logic.get_available_xai_entities(home_dir_mgr, stage)

    @measure_blocking_code
    async def get_bp_xai(
        self,
        ctx: RequestContext,
        pq_id: Optional[PredictiveQueryID],
        bp_job_id: Optional[BatchPredictionJobID] = None,
    ) -> XAIResponse:
        state_services = self._require_state_services()
        home_dir_mgr, project_id, _job = await self._get_home_dir(
            id=None,
            job_id=bp_job_id,
            ctx=ctx,
            batch_prediction=True,
        )
        bp_logic = BatchPredictionJobLogic(
            self._ml_job_backend,
            self._data_backend_ctx,
            state_services=state_services,
        )

        predictive_query: Optional[PredictiveQuery] = None
        graph_id: Optional[GraphID] = None

        # UI Jobs
        if pq_id is not None:
            # Resolve the pquery id to its v2 `PQueryDefinition`.
            pq_def = await self._pquery_api_driver.get_pquery_definition_by_id(
                pq_id)
            assert pq_def is not None
            graph_id = GraphIDv2.compose(fpid=pq_def.graph_fpid)
            pq_object = await self._get_predictive_query_from_definition(
                pq_def, project_id)
            if pq_object is None:
                raise ValueError(
                    "Could not retrieve the predictive query object" +
                    f" for query wiht id {pq_id}.")

            predictive_query = pq_object
        else:
            raise ValueError("Predictive Query ID is required for "
                             "getting BP graph-level XAI")

        # Get the BFS tree
        try:
            assert isinstance(graph_id, GraphIDv2)
            graph_def = (await self._get_graph_definition_from_state_services(
                graph_id, project_id))
            bfs_tree = self._get_bfs_tree_from_graph_definition(
                graph_def, graph_id, pquery_table_name(predictive_query.entity))
        except Exception:
            raise ValueError(
                f"Could not generate BFS tree from graph with id {graph_id}" +
                f" for bp job {bp_job_id}.")

        return await bp_logic.get_xai(
            home_mgr=home_dir_mgr,
            tables=predictive_query.graph.tables,
            entity=predictive_query.entity,
            bfs_tree=bfs_tree,
        )

    @measure_blocking_code
    async def get_bp_available_xai_entities(
        self,
        ctx: RequestContext,
        id: Optional[BatchPredictionJobID],
    ) -> StrategyMappings:
        logic, home_dir_mgr = (
            await
            self._get_batch_prediction_artifact_logic_and_home_dir(ctx, id))
        return await logic.get_available_xai_entities(home_dir_mgr)

    @measure_blocking_code
    async def get_bp_entity_old_subgraph(
        self,
        ctx: RequestContext,
        id: Optional[BatchPredictionJobID],
        entity_id: str,
    ) -> OldSubgraph:
        logic, home_dir_mgr = (
            await
            self._get_batch_prediction_artifact_logic_and_home_dir(ctx, id))
        return await logic.get_entity_old_subgraph(home_dir_mgr, entity_id)

    @measure_blocking_code
    async def get_bp_subgraph(
        self,
        ctx: RequestContext,
        id: Optional[BatchPredictionJobID],
        entity_id: str,
    ) -> Subgraph:
        logic, home_dir_mgr = (
            await
            self._get_batch_prediction_artifact_logic_and_home_dir(ctx, id))
        return await logic.get_subgraph(home_dir_mgr, entity_id)

    async def get_bp_entity_output(
        self,
        ctx: RequestContext,
        id: Optional[BatchPredictionJobID],
        entity_id: str,
    ) -> EntityOutput:
        logic, home_dir_mgr = (
            await
            self._get_batch_prediction_artifact_logic_and_home_dir(ctx, id))
        return await logic.get_entity_output(home_mgr=home_dir_mgr,
                                             entity_id=entity_id)

    @measure_blocking_code
    async def get_bp_entity_historical_items(
        self,
        ctx: RequestContext,
        id: Optional[BatchPredictionJobID],
        entity_id: str,
    ) -> HistoricalItems:
        logic, home_dir_mgr = (
            await
            self._get_batch_prediction_artifact_logic_and_home_dir(ctx, id))
        return await logic.get_entity_historical_items(home_mgr=home_dir_mgr,
                                                       entity_id=entity_id)

    @measure_blocking_code
    async def get_entity_old_subgraph(
        self,
        ctx: RequestContext,
        entity_id: str,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
        project_id: ProjectID | None = None,
    ) -> OldSubgraph:
        r"""
        Get the subgraph for a given entity (will be deprecated soon).

        Args:
            ctx: The request context.
            entity_id: The entity ID.
            id: The predictive query ID.
            job_id: The training job ID.
            stage: The training stage.

        Returns:
            The subgraph object defined in :class:`kumo.config.subgraph`.
        """
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id, project_id=project_id))
        return await logic.get_entity_old_subgraph(home_mgr=home_dir_mgr,
                                                   entity_id=entity_id,
                                                   stage=stage)

    @measure_blocking_code
    async def get_subgraph(
        self,
        ctx: RequestContext,
        entity_id: str,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
    ) -> Subgraph:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id))
        return await logic.get_subgraph(home_mgr=home_dir_mgr,
                                        entity_id=entity_id, stage=stage)

    @measure_blocking_code
    async def get_node_pred_rfm(
        self,
        ctx: RequestContext,
        entity_id: str,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
    ) -> NodePredRFM:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id))
        return await logic.get_node_pred_rfm(home_mgr=home_dir_mgr,
                                             entity_id=entity_id, stage=stage)

    @measure_blocking_code
    async def get_category_node_pred_rfm(
        self,
        ctx: RequestContext,
        category: str,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
    ) -> NodePredRFM:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id))
        return await logic.get_category_node_pred_rfm(home_mgr=home_dir_mgr,
                                                      category=category,
                                                      stage=stage)

    @measure_blocking_code
    async def get_category_link_pred_rfm(
        self,
        ctx: RequestContext,
        category: str,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
    ) -> LinkPredRFM:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id))
        return await logic.get_category_link_pred_rfm(home_mgr=home_dir_mgr,
                                                      category=category,
                                                      stage=stage)

    @measure_blocking_code
    async def get_aggr_feature(
        self,
        ctx: RequestContext,
        entity_id: str,
        table_name: str,
        column_name: str,
        op: str,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
    ) -> Feature:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id))
        return await logic.get_aggr_feature(home_mgr=home_dir_mgr,
                                            entity_id=entity_id,
                                            table_name=table_name,
                                            column_name=column_name, op=op,
                                            stage=stage)

    @measure_blocking_code
    async def get_category_aggr_feature(
        self,
        ctx: RequestContext,
        category: str,
        table_name: str,
        column_name: str,
        op: str,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
    ) -> Feature:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id))
        return await logic.get_category_aggr_feature(home_mgr=home_dir_mgr,
                                                     category=category,
                                                     table_name=table_name,
                                                     column_name=column_name,
                                                     op=op, stage=stage)

    @measure_blocking_code
    async def get_link_pred_rfm(
        self,
        ctx: RequestContext,
        entity_id: str,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
    ) -> LinkPredRFM:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id))
        return await logic.get_link_pred_rfm(home_mgr=home_dir_mgr,
                                             entity_id=entity_id, stage=stage)

    @measure_blocking_code
    async def get_topk_feature(
        self,
        ctx: RequestContext,
        entity_id: str,
        topk: int,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
    ) -> List[Feature]:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id))
        return await logic.get_topk_feature(home_mgr=home_dir_mgr,
                                            entity_id=entity_id, topk=topk,
                                            stage=stage)

    @measure_blocking_code
    async def get_entity_output(
        self,
        ctx: RequestContext,
        id: Optional[PredictiveQueryID],
        entity_id: str,
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
        project_id: ProjectID | None = None,
    ) -> EntityOutput:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id, project_id=project_id))
        return await logic.get_entity_output(home_mgr=home_dir_mgr,
                                             entity_id=entity_id, stage=stage)

    @measure_blocking_code
    async def get_entity_historical_items(
        self,
        ctx: RequestContext,
        id: Optional[PredictiveQueryID],
        entity_id: str,
        job_id: Optional[TrainingJobID] = None,
        stage: TrainingStage = TrainingStage.TEST,
        project_id: ProjectID | None = None,
    ) -> HistoricalItems:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id, project_id=project_id))
        return await logic.get_entity_historical_items(home_mgr=home_dir_mgr,
                                                       entity_id=entity_id,
                                                       stage=stage)

    async def _get_pquery_artifact_logic_and_home_dir(
        self,
        ctx: RequestContext,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID],
        *,
        project_id: ProjectID | None = None,
    ) -> tuple[PQueryLogic, HomeDirManager]:
        state_services = self._require_state_services()
        home_dir_mgr, project_id, _job = await self._get_home_dir(
            id=id,
            job_id=job_id,
            ctx=ctx,
            project_id=project_id,
        )
        logic = PQueryLogic(
            self._data_backend_ctx,
            state_services=state_services,
            project_id=project_id,
        )
        return logic, home_dir_mgr

    async def _get_home_dir(
        self,
        id: Optional[PredictiveQueryID],
        job_id: Optional[Union[TrainingJobID, BatchPredictionJobID]],
        ctx: RequestContext,
        *,
        project_id: ProjectID | None = None,
        batch_prediction: bool = False,
    ) -> tuple[HomeDirManager, ProjectID, object]:
        if id is None and job_id is None:
            raise ValueError('At least one of Predictive Query or '
                             'Training Job ID must be specified.')

        # Get the home dir manager pointing to the right job
        existing_job = None
        if id is not None:
            if project_id is None:
                raise ValueError(
                    "project_id is required for predictive query artifacts")
            curr_training_job_id = (
                await self._pquery_api_driver.get_current_training_job_id(
                    ctx, id, project_id))
            if curr_training_job_id is None:
                raise ValidationException(
                    message=(
                        "This predictive query has needs to be trained, before "
                        "you can perform this operation."), )

            existing_job = await self._ml_job_backend.get_training_job(
                ctx, curr_training_job_id)
            if not existing_job:
                raise RuntimeError(
                    f"Training job {curr_training_job_id} is missing from the "
                    f"ML backend. Please check for database corruption, or "
                    f"retrain the predictive query.")
            if existing_job.execution.status != JobStatus.DONE:
                message = (
                    "This predictive query needs to be trained successfully "
                    "before you can perform this operation. However, the last "
                    f"training job for this pquery has status = "
                    f"{existing_job.execution.status}.")
                raise ValidationException(message=message)

            home_dir = await self._ml_job_backend.get_job_home_dir(
                curr_training_job_id)
        else:
            assert job_id is not None
            if batch_prediction:
                existing_job = (await
                                self._ml_job_backend.get_batch_prediction_job(
                                    ctx, job_id))
            else:
                existing_job = await self._ml_job_backend.get_training_job(
                    ctx, job_id)
            if existing_job is None:
                if batch_prediction:
                    raise ValueError(
                        f"Batch prediction job {job_id} does not exist")
                raise TrainingJobDoesNotExistError(job_id)
            home_dir = await self._ml_job_backend.get_job_home_dir(job_id)

        resolved_project_id = ProjectID(existing_job.project_id)
        if project_id is not None and resolved_project_id != project_id:
            raise TrainingJobDoesNotExistError(existing_job.id)
        await self._require_state_services().projects.get_project(
            resolved_project_id,
            ctx.user_authorization,
        )
        return HomeDirManager(home_dir), resolved_project_id, existing_job

    async def _get_batch_prediction_artifact_logic_and_home_dir(
        self,
        ctx: RequestContext,
        job_id: Optional[BatchPredictionJobID],
    ) -> tuple[BatchPredictionJobLogic, HomeDirManager]:
        state_services = self._require_state_services()
        logic = BatchPredictionJobLogic(
            self._ml_job_backend,
            self._data_backend_ctx,
            state_services=state_services,
        )
        home_dir_mgr, _project_id, _job = await self._get_home_dir(
            id=None,
            job_id=job_id,
            ctx=ctx,
            batch_prediction=True,
        )
        return logic, home_dir_mgr

    @measure_blocking_code
    async def get_regression_histogram(
        self,
        ctx: RequestContext,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        project_id: ProjectID | None = None,
    ) -> AllTargetRegressionStats:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id, project_id=project_id))

        return await logic.get_regression_histogram(home_dir_mgr)

    @measure_blocking_code
    async def get_regression_heatmap(
        self,
        ctx: RequestContext,
        id: Optional[PredictiveQueryID],
        job_id: Optional[TrainingJobID] = None,
        project_id: ProjectID | None = None,
    ) -> RegressionHeatmapResponse:
        logic, home_dir_mgr = (await
                               self._get_pquery_artifact_logic_and_home_dir(
                                   ctx, id, job_id, project_id=project_id))
        return await logic.get_regression_heatmap(home_dir_mgr)

    async def _collect_fixture_graph(
        self,
        graph_id: GraphID,
        project_id: ProjectID,
        tables_by_fpid: Dict[FPID, TableDefinition],
        graphs_by_fpid: Dict[FPID, GraphDefinition],
    ) -> FPID:
        if isinstance(graph_id, GraphIDv1):
            raise ValueError("Cannot export v2 fixtures from v1 graphs")

        state_services = self._require_state_services()
        graph_entry = None
        graph_id_str = str(graph_id)
        if isinstance(graph_id, GraphIDv2) and graph_id.fpid is not None:
            graph_entry = await state_services.graphs.get_graph_entry(
                project_id, graph_id.fpid)

        if graph_entry is None:
            try:
                graph_entry = await state_services.graphs.get_graph_entry(
                    project_id, FPID(graph_id_str))
            except ValueError:
                # Non-FPID graph identifiers are looked up via alias below.
                pass

        if graph_entry is None:
            graph_entry = await state_services.graphs.get_graph_entry_by_alias(
                project_id, graph_id_str)

        if graph_entry is None:
            raise ValueError(f"Graph {graph_id} not found")

        graph_fpid = graph_entry.id
        table_entries = await state_services.tables.get_table_entries(
            project_id, graph_entry.table_name_to_dbid.values())
        graph_tables = []
        for graph_table_name, fpid in graph_entry.table_name_to_dbid.items():
            table_entry = table_entries.get(fpid)
            if table_entry is None:
                raise TableDoesNotExistError(
                    TableIDv2.public(
                        table_name=graph_table_name,
                        fpid=fpid,
                        pquery_id=None,
                    ))
            graph_tables.append(table_entry.config)
        graph = graph_entry.to_graph_definition(graph_tables)
        graphs_by_fpid[graph_fpid] = graph
        for table_def_raw in graph.tables.values():
            table_def = (table_def_raw if isinstance(
                table_def_raw, TableDefinition) else TableDefinition(
                    **dataclasses.asdict(table_def_raw)))
            tables_by_fpid[table_def.fpid] = table_def
        return graph_fpid

    async def dump_as_fixtures_v2(
        self,
        pquery_ids: Optional[List[PredictiveQueryID]],
        project_id: ProjectID,
        principal: AuthPrincipal,
    ) -> FixtureDefinition:
        state_services = self._require_state_services()
        await state_services.projects.get_project(project_id, principal)
        if pquery_ids is None:
            pquery_defs = await state_services.pqueries.list_definitions()
        else:
            pquery_defs_by_id = (
                await
                state_services.pqueries.get_definitions_by_ids(pquery_ids))
            missing_pquery_ids = [
                pquery_id for pquery_id in pquery_ids
                if pquery_id not in pquery_defs_by_id
            ]
            if missing_pquery_ids:
                raise ValueError(
                    f'Predictive query id {missing_pquery_ids[0]} is not found.'
                )
            pquery_defs = [
                pquery_defs_by_id[pquery_id] for pquery_id in pquery_ids
            ]

        queries: List[FixturePQueryRecord] = []
        tables_by_fpid: Dict[FPID, TableDefinition] = {}
        graphs_by_fpid: Dict[FPID, GraphDefinition] = {}
        for pquery_def in pquery_defs:
            graph_fpid = pquery_def.graph_fpid
            if graph_fpid not in graphs_by_fpid:
                try:
                    graph_fpid = await self._collect_fixture_graph(
                        GraphIDv2.public(graph_fpid),
                        project_id,
                        tables_by_fpid,
                        graphs_by_fpid,
                    )
                except ValueError:
                    if pquery_ids is None:
                        continue
                    raise
            queries.append(
                FixturePQueryRecord(
                    query_str=pquery_def.query_str,
                    graph_fpid=graph_fpid,
                    pquery_name=(pquery_def.pquery_name or str(
                        PredictiveQueryID.public(pquery_def.fpid))),
                    list_distinct_target_is_array=(
                        pquery_def.list_distinct_target_is_array),
                    lp_bp_layout=pquery_def.lp_bp_layout,
                ))

        connector_ids = set()
        for graph in graphs_by_fpid.values():
            connector_ids.update(
                state_services.connectors.connector_ids_for_graph(graph))
        for connector_id in connector_ids:
            await state_services.connectors.execution_ref_for_project(
                principal, project_id, connector_id, CONNECTOR_GRAPH_ACTIONS)

        return FixtureDefinition(
            tables=list(tables_by_fpid.values()),
            graphs=list(graphs_by_fpid.values()),
            queries=queries,
        )
