# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from http import HTTPStatus
from typing import Any, Dict, List, Optional

import kumoapi.jobs as jobs_api
import kumoapi.model_plan as plan_api
from fastapi import HTTPException
from kumo_core.types import TaskType

from io_utils import logging
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.batch_prediction_job_v2 import (
    BatchPredictionInternalRequest,
    PredictionTableSchemaSampleColumn,
    SuggestPredictionSamples,
)
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query_v2 import PQueryResource
from rest.config.status import IDResponse
from rest.config.training_job_v2 import (
    TrainingJobInternalRequest,
    TrainingJobRequest,
)
from rest.deps.authentication import RequestContext
from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.driver.data_snapshot_api_driver import DataSnapshotApiDriver
from rest.driver.graph_driver import GraphDriver
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.utils.ui_to_sdk_utils import (
    bp_job_ouput_conf,
    clean_up_pquery_string,
    generate_header,
    generate_prediction_job_sdk_script,
    generate_prediction_only_sdk_script,
    generate_unsupported_diskgraph_forecasting_bp_sdk_script,
    is_diskgraph_forecasting_batch_prediction_sdk_unsupported,
    make_stype_and_dtype_sdk_compat,
)
from service_framework.deploy_env_config import deploy_env
from statedb.config.common import (
    BatchPredictionJobID,
    DataSnapshot,
    GenerateTrainTableJobID,
    GraphIDv2,
    JobSource,
    ProjectID,
    TrainingJobID,
)
from statedb.utils.errors import BatchPredictionDoesNotExistError

logger = logging.getLogger(__name__)


class JobsDriver:
    """
    Driver for orchestrating all logic behind running training in V2 APIs.

    Attributes:
        pquery_driver (PQueryApiDriver): Driver for handling predictive query
        business logic.
        data_snapshot_driver (DataSnapshotApiDriver): Driver for handling data
        snapshot business logic.

    """
    def __init__(
        self,
        pquery_driver: PQueryApiDriver,
        data_snapshot_driver: DataSnapshotApiDriver,
        ml_job_backend: MLJobBackend,
        graph_driver: GraphDriver,
        connector_driver: ConnectorApiDriver,
    ):
        self._pquery_driver = pquery_driver
        self._data_snapshot_driver = data_snapshot_driver
        self._ml_job_backend = ml_job_backend
        self._graph_driver = graph_driver
        self._connector_driver = connector_driver

    async def train(
        self,
        ctx: RequestContext,
        request: TrainingJobRequest,
    ) -> TrainingJobID:
        """
        Initiates a training job based on the provided training job request.
        It will create the following entities:
        1. A new graph if the current definition doesn't exist.
        2. A new predictive query.
        And trigger the following jobs:
        1. Graph snapshot job to snapshot the latest data for the Graph.
        2. Training table generation job to generate training tables. Training
           table specs from TrainingJobRequest.training_table_gen_plan will be
           used.
        3. A new training job. Graph data from (1), Training table from (2) and
           ModelPlan from request.model_plan will be used.

        Args:
            request (TrainingJobRequest): The request object containing all
            necessary information to start a training job, including graph
            definition, predictive query, and model plan.
        Returns:
            TrainingJobID: The ID of the initiated training job.
        """
        graph_fpid = request.graph_definition.fpid
        try:
            # Check if this graph already exists
            graph_response = await self._graph_driver.get_graph_v2(
                graph_fpid, ProjectID(request.project_id))
            graph_id = GraphIDv2.from_public(graph_response[0].id)
        except KeyError:
            # If doesn't exists, create it. This graph definition needs to be
            # in the DB before snapshotting.
            graph_id = await self._graph_driver.create_full_graph_v2(
                request.graph_definition,
                project_id=request.project_id,
            )
        # This should never happen anyway, doing this for linter
        assert graph_id is not None
        assert graph_id.fpid is not None

        # Pre-compute TrainingJobID before creating the graph snapshot.
        reserved_training_job_id = await self._ml_job_backend.reserve_training_job_id(
        )
        logger.info(f"jobs_driver.train: Reserved training job ID: "
                    f"{reserved_training_job_id}")

        logger.info("jobs_driver.train: creating graph snapshot for %s",
                    graph_id.fpid)
        graph_snapshot_id = await (
            self._data_snapshot_driver.create_graph_snapshot(
                graph_id.fpid,
                True,
                base_graph_name=request.base_graph_name,
                project_id=request.project_id,
            ))
        # Create a predictive query
        pquery_id = await self._pquery_driver.create_predictive_query_v2(
            PQueryResource(
                name=None,
                query_string=request.predictive_query,
                graph=request.graph_definition,
            ),
            project_id=request.project_id,
        )

        ui_tags = {
            **self._convert_to_tags_dict(request.tags),
            **(request.job_tags or {})
        }
        # Call the start_training_job_v2. This will depend on the graph
        # snapshot job instantiated above. It will automatically created a
        # training table generation job before triggering a training job.
        training_job_req = TrainingJobInternalRequest(
            job_tags=ui_tags,
            pquery_id=pquery_id,
            graph_snapshot_id=graph_snapshot_id,
            model_plan=request.model_plan,
        )

        logger.info(
            f"jobs_driver.train: About to call start_training_job_v2. "
            f"graph_snapshot_id={graph_snapshot_id}, pquery_id={pquery_id}, "
            f"reserved_training_job_id={reserved_training_job_id}")
        return await self._pquery_driver.start_training_job_v2(
            ctx=ctx,
            request=training_job_req,
            job_source=JobSource.UI,
            training_table_generation_plan=request.training_table_gen_plan,
            user_info=request.user,
            reserved_training_job_id=reserved_training_job_id,
            project_id=request.project_id,
        )

    async def create_batch_prediction_job(
        self,
        ctx: RequestContext,
        request: BatchPredictionInternalRequest,
        job_source: Optional[JobSource] = None,
    ) -> IDResponse:
        """
        Creates a batch prediction job.

        This method handles the creation of a batch prediction job, including
        the potential override of the predictive query and the generation of a
        prediction table if necessary.

        Args:
            ctx (RequestContext): The request context.
            request (BatchPredictionRequest): The batch prediction request
                containing details such as model training job ID, outputs, job
                tags, predictive query, graph snapshot ID, predict options, and
                explanations.
            job_source (Optional[JobSource]): The source of the job, if any.

        Returns:
            IDResponse: The response containing the ID of the created batch
                prediction job.

        Raises:
            HTTPException: If there are validation errors in the response.
        """
        graph_snapshot_id = None
        if request.graph_id is not None:
            # If graph_id is supplied, start a new data snapshot for this
            # graph and use that
            graph_snapshot_id = (
                await self._data_snapshot_driver.create_graph_snapshot(
                    request.graph_id,
                    True,
                    project_id=request.project_id,
                ))
        batch_prediction_api_request = jobs_api.BatchPredictionRequest(
            model_training_job_id=request.model_training_job_id,
            outputs=request.outputs,
            job_tags={
                **self._convert_to_tags_dict(request.tags),
                **(request.job_tags or {})
            },
            pred_table_job_id=None,
            graph_snapshot_id=graph_snapshot_id,
            predict_options=request.predict_options,
            explanations=request.explanations,
        )

        # get pquery_id from the model_training_job_id
        model_training_job = await self._ml_job_backend.get_training_job(
            ctx, request.model_training_job_id)  # type: ignore
        if model_training_job is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail="Model training job not found.",
            )
        pquery_id = await self._pquery_driver.get_training_job_pquery_id_v2(
            ctx,
            model_training_job.id,
        )
        if pquery_id is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail="Predictive query not found for training job.",
            )

        # If the predictive query is overridden, create a new predictive query
        pquery_overriden = request.predictive_query
        if pquery_overriden is not None:
            # In the UI, the user can potentially override the predictive
            # query. In that case, we will need to recreate it.
            graph_snapshot_id = graph_snapshot_id
            assert graph_snapshot_id is not None

            graph_fpid, _ = DataSnapshot.GRAPH.parse_id(graph_snapshot_id)
            _, graph_definition = (
                await self._data_snapshot_driver._get_graph_by_fpid(
                    request.project_id, graph_fpid))

            # Creating a new predictive query with overridden query.
            created_pquery_id = (await
                                 self._pquery_driver.create_predictive_query_v2(
                                     PQueryResource(
                                         query_string=pquery_overriden,
                                         graph=graph_definition,
                                     ),
                                     project_id=request.project_id,
                                 ))
            pquery_id = created_pquery_id
        assert pquery_id is not None

        gen_pred_table_request = jobs_api.GeneratePredictionTableRequest(
            job_tags={
                **self._convert_to_tags_dict(request.tags),
                **(request.job_tags or {})
            },
            pquery_id=str(pquery_id),
            plan=plan_api.PredictionTableGenerationPlan(
                anchor_time=request.anchor_time),
            graph_snapshot_id=graph_snapshot_id,
        )
        # Setting this will allow the bp job to leverage this gen pred
        # table job
        batch_prediction_api_request.pred_table_job_id = (
            await self._pquery_driver.start_generate_pred_table(
                gen_pred_table_request,
                job_source=job_source,
                user=request.user,
                is_child_job=True,
                training_job_id=model_training_job.id,
                project_id=request.project_id,
            ))

        id, validation_response = (
            await self._pquery_driver.start_batch_prediction_job_v2(
                ctx=ctx,
                request=batch_prediction_api_request,
                job_source=job_source,
                filter_overrides=request.filter_overrides,
                custom_tags=request.tags,
                user=request.user,
                project_id=request.project_id,
            ))

        if len(validation_response.errors) != 0:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=validation_response.error_message(),
            )
        assert id is not None
        return IDResponse(id)

    def _convert_to_tags_dict(self, tags: Optional[List[str]]):
        """
        Converts the tags from the request to a dictionary format. This is
        needed since the SDK supports tags in dict format but from the UI, you
        can simply add tag values associated with a job without a key.

        Args:
            tags: List of tags to convert.

        Returns:
            dict: Key will be incremental custom_ui_{tag_num}, where tag_num
            is 1,2... Value will be the tags coming in as input.
        """
        if tags is None:
            return {}
        return {tag: tag for tag in tags}

    async def get_training_job_samples(
        self,
        id: GenerateTrainTableJobID,
        task_type: TaskType,
    ) -> List[PredictionTableSchemaSampleColumn]:
        """
        Method to sample the first n rows of a train
        table job and use data to return a preview of
        what the prediction data will look like. Currenlty,
        this is only supported for SAAS.

        Args:
            - id (str): traing table id

        Returns:
            - List[SampleColumn] - an array of SampleColumn. Each
            item corresponds to one row in the preview schema.
            Each row has a name and an array of 10 samples.

        ** Note: we sample 40 rows for multiclass and multilabel
        classification to reduce the number of potential NA/missing items.
        """
        # get train table job path
        train_table_job_path = await \
            self._ml_job_backend.get_generate_train_table_job_path(
                id,
            )

        # Sample the first N rows for the preview. 40 for
        # multiclass/multilabel to reduce the chance of all-NA columns.
        sample_limit = 40 if task_type in (
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ) else 10

        try:
            import pyarrow.dataset as pds
            df = pds.dataset(train_table_job_path).head(
                sample_limit).to_pandas()
            samples_map: Dict[str, Any] = {
                column: df[column].tolist()
                for column in df.columns
            }
            pred_samples = SuggestPredictionSamples()
            return pred_samples.suggest_col_samples_by_task_type(
                task_type, samples_map)
        except Exception as e:
            logger.warning(f"Error while loading training table samples: {e}.")
            return []

    async def prediction_to_sdk(
        self,
        id: BatchPredictionJobID,
        ctx: RequestContext,
    ):
        """
        Used to generate a fully executable Python SDK script to reproduce a prediction pipeline
        in the SDK. Instead of regenerating the full training pipeline, this method loads the
        existing trained model via the SDK. This method generates:
            1. Header (imports and kumo.init),
            2. Load existing model script (Trainer.load, PredictiveQuery.load_from_training_job),
            3. (Optional) BP predictive query config (if different from training),
            4. (Optional) Export connector config,
            5. Prediction table config,
            6. Prediction job config.

        This method is part of the UI to SDK feature and will be called in the UI.

        Args:
            id (BatchPredictionJobID): ID of the prediction job.
            ctx (RequestContext): Request context.
        Response:
            str: The complete python script for an SDK prediction workflow as a string.
        """
        bp_job = await self._pquery_driver.get_batch_prediction_job(ctx, id)
        if not bp_job:
            raise BatchPredictionDoesNotExistError(id)

        bp_config = bp_job.config

        if is_diskgraph_forecasting_batch_prediction_sdk_unsupported(
                deploy_env().is_diskgraph, bp_config.task_type,
                bp_config.forecast_length):
            return generate_unsupported_diskgraph_forecasting_bp_sdk_script()

        pred_table_plan: Dict[str, Any] = {
            "anchor_time": bp_config.prediction_time,
            "forecast_length": bp_config.forecast_length,
            "lag_timesteps": bp_config.lag_timesteps,
            "year_over_year": bp_config.year_over_year,
        }

        if not pred_table_plan["anchor_time"]:
            pred_table_plan["anchor_time"] = "inferred"

        predict_options = {
            "binary_classification_threshold":
            bp_config.binary_classification_threshold,
            "num_classes_to_return": bp_config.num_classes_to_return,
            "num_workers": bp_config.num_parallel_workers_limit,
        }

        output_config, bp_connector_config = bp_job_ouput_conf(
            bp_config.outputs)

        export_connector_conf = await self._connector_driver.get_bp_connector_metadata_for_sdk(
            bp_connector_config,
            ProjectID(bp_job.project_id),
            ctx.user_authorization,
        )

        pq_bp_resource = await self._pquery_driver.get_predictive_query_v2(
            bp_config.predictive_query_id,
            ProjectID(bp_job.project_id),
        )
        graph_def = GraphDefinition(**dataclasses.asdict(pq_bp_resource.graph))
        _, table_refs, _ = await self._graph_driver.get_graph_metadata_for_sdk(
            graph_def.fpid, ProjectID(bp_job.project_id))

        # Generate header + load existing trained model
        header_and_model_script = generate_header(str(
            bp_job.project_id)) + generate_prediction_only_sdk_script(
                str(bp_config.training_job_id), )

        # Check if training and bp queries are different
        pq_config: Optional[Dict[str, str]] = None
        cleaned_bp_pq_str = clean_up_pquery_string(
            pq_bp_resource.query_string,
            table_refs,
        )
        training_pq_id = await self._pquery_driver.get_training_job_pquery_id_v2(
            ctx, bp_config.training_job_id)
        if training_pq_id:
            training_pq_resource = await self._pquery_driver.get_predictive_query_v2(
                training_pq_id, ProjectID(bp_job.project_id))
            cleaned_training_pq_str = clean_up_pquery_string(
                training_pq_resource.query_string,
                table_refs,
            )
            if cleaned_bp_pq_str != cleaned_training_pq_str:
                pq_config = {
                    "graph": "\\" + "graph" + "\\",
                    "query_string": cleaned_bp_pq_str,
                }

        return header_and_model_script + make_stype_and_dtype_sdk_compat(
            generate_prediction_job_sdk_script(
                pred_table_plan,
                predict_options,
                output_config,
                export_connector_conf,
                bp_config.generate_pred_table_job_id,
                pq_config,
                project_id=str(bp_job.project_id),
            ))
