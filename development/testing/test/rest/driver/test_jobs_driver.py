# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from datetime import datetime
from types import SimpleNamespace
from typing import List, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.jobs import (
    GraphSnapshotID,
    PredictionArtifactType,
    PredictionOutputConfig,
)

from kumo.validation.validation import ValidationResponse
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.batch_prediction_job import BatchPredictionJobConfig
from rest.config.batch_prediction_job_v2 import (
    BatchPredictionFilterOverrides,
    BatchPredictionInternalRequest,
)
from rest.config.graph_v2 import GraphDefinition
from rest.config.jobs import BatchPredictionOptions
from rest.config.prediction_output import S3PredictionOutput
from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.driver.data_snapshot_api_driver import DataSnapshotApiDriver
from rest.driver.graph_driver import GraphDriver
from rest.driver.jobs_driver import JobsDriver
from rest.driver.pquery_api_driver import PQueryApiDriver
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    GraphIDv2,
    JobSource,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)

_PROJECT_ID = ProjectID('project_0123456789abcdef0123456789abcdef')
_CONNECTOR_ID = 'connector_0123456789abcdef0123456789abcdef'


def _make_prediction_to_sdk_driver():
    pquery_driver: PQueryApiDriver = AsyncMock()
    data_snapshot_driver: DataSnapshotApiDriver = AsyncMock()
    ml_job_backend: MLJobBackend = AsyncMock()
    graph_driver: GraphDriver = AsyncMock()
    connector_driver: ConnectorApiDriver = AsyncMock()
    jobs_driver = JobsDriver(
        pquery_driver,
        data_snapshot_driver,
        ml_job_backend,
        graph_driver,
        connector_driver,
    )
    return jobs_driver, pquery_driver, graph_driver, connector_driver


def _make_bp_config(
    task_type: TaskType = TaskType.REGRESSION,
    forecast_length: int | None = None,
) -> BatchPredictionJobConfig:
    return BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID("pquery_test"),
        training_job_id=TrainingJobID("trainingjob-test"),
        task_type=task_type,
        forecast_length=forecast_length,
        lag_timesteps=30 if forecast_length else None,
        year_over_year=bool(forecast_length),
    )


@pytest.mark.asyncio
async def test_train_propagates_project_to_graph_snapshot_and_pquery(
        monkeypatch):
    pquery_driver: PQueryApiDriver = AsyncMock()
    data_snapshot_driver: DataSnapshotApiDriver = AsyncMock()
    ml_job_backend: MLJobBackend = AsyncMock()
    graph_driver: GraphDriver = AsyncMock()
    connector_driver: ConnectorApiDriver = AsyncMock()
    jobs_driver = JobsDriver(
        pquery_driver,
        data_snapshot_driver,
        ml_job_backend,
        graph_driver,
        connector_driver,
    )

    project_id = ProjectID('project_0123456789abcdef0123456789abcdef')
    graph_definition = GraphDefinition(tables={}, col_groups=[])
    graph_fpid = graph_definition.fpid
    graph_id = GraphIDv2.compose(fpid=graph_fpid)
    graph_driver.get_graph_v2.side_effect = KeyError
    graph_driver.create_full_graph_v2.return_value = graph_id
    reserved_id = TrainingJobID('trainingjob-reserved')
    ml_job_backend.reserve_training_job_id.return_value = reserved_id
    data_snapshot_driver.create_graph_snapshot.return_value = GraphSnapshotID(
        'graph-snapshot-test')
    pquery_id = PredictiveQueryID('pquery_test')
    pquery_driver.create_predictive_query_v2.return_value = pquery_id
    pquery_driver.start_training_job_v2.return_value = TrainingJobID(
        'trainingjob-created')
    monkeypatch.setattr(
        'rest.driver.jobs_driver.TrainingJobInternalRequest',
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    request = SimpleNamespace(
        graph_definition=graph_definition,
        predictive_query='PREDICT target FROM table',
        training_table_gen_plan=object(),
        model_plan=object(),
        tags=None,
        job_tags=None,
        user=None,
        base_graph_name=None,
        project_id=project_id,
    )
    ctx = AsyncMock()

    await jobs_driver.train(ctx, request)

    # Both plans reach the driver as the caller's internal objects, without a
    # round trip through the public contract.
    assert (pquery_driver.start_training_job_v2.await_args.kwargs['request'].
            model_plan is request.model_plan)
    assert (pquery_driver.start_training_job_v2.await_args.
            kwargs['training_table_generation_plan']
            is request.training_table_gen_plan)

    graph_driver.create_full_graph_v2.assert_awaited_once_with(
        graph_definition,
        project_id=project_id,
    )
    data_snapshot_driver.create_graph_snapshot.assert_awaited_once_with(
        graph_fpid,
        True,
        base_graph_name=None,
        project_id=project_id,
    )
    pquery_driver.create_predictive_query_v2.assert_awaited_once()
    assert (pquery_driver.create_predictive_query_v2.await_args.
            kwargs['project_id'] == project_id)
    assert (pquery_driver.start_training_job_v2.await_args.kwargs['project_id']
            == project_id)


@dataclasses.dataclass
class _GraphForSdk:
    fpid: FPID


def _patch_prediction_to_sdk_normal_path(monkeypatch, is_diskgraph: bool):
    monkeypatch.setattr(
        "rest.driver.jobs_driver.deploy_env",
        lambda: SimpleNamespace(is_diskgraph=is_diskgraph),
    )
    monkeypatch.setattr(
        "rest.driver.jobs_driver.bp_job_ouput_conf",
        lambda outputs: ({}, "bp-connector"),
    )
    monkeypatch.setattr(
        "rest.driver.jobs_driver.GraphDefinition",
        lambda **kwargs: SimpleNamespace(fpid=kwargs["fpid"]),
    )
    monkeypatch.setattr(
        "rest.driver.jobs_driver.clean_up_pquery_string",
        lambda query, table_refs: query,
    )
    generate_header = MagicMock(return_value="HEADER")
    monkeypatch.setattr(
        "rest.driver.jobs_driver.generate_header",
        generate_header,
    )
    monkeypatch.setattr(
        "rest.driver.jobs_driver.generate_prediction_only_sdk_script",
        lambda training_job_id: f"LOAD_MODEL({training_job_id})",
    )
    generate_prediction = MagicMock(return_value="PREDICTION_SCRIPT")
    monkeypatch.setattr(
        "rest.driver.jobs_driver.generate_prediction_job_sdk_script",
        generate_prediction,
    )
    monkeypatch.setattr(
        "rest.driver.jobs_driver.make_stype_and_dtype_sdk_compat",
        lambda script: script,
    )
    return generate_header, generate_prediction


@pytest.mark.asyncio
async def test_prediction_to_sdk_skips_diskgraph_forecasting_before_training(
        monkeypatch):
    monkeypatch.setattr(
        "rest.driver.jobs_driver.deploy_env",
        lambda: SimpleNamespace(is_diskgraph=True),
    )
    jobs_driver, pquery_driver, graph_driver, connector_driver = (
        _make_prediction_to_sdk_driver())
    pquery_driver.get_batch_prediction_job.return_value = SimpleNamespace(
        config=_make_bp_config(forecast_length=30))

    script = await jobs_driver.prediction_to_sdk(
        BatchPredictionJobID("bp-job-test"),
        AsyncMock(),
    )

    assert "SKIPPED_UNSUPPORTED" in script
    assert "generate_training_table" not in script
    assert "generate_prediction_table" not in script
    pquery_driver.training_to_sdk.assert_not_called()
    pquery_driver.get_predictive_query_v2.assert_not_called()
    connector_driver.get_bp_connector_metadata_for_sdk.assert_not_called()
    graph_driver.get_graph_metadata_for_sdk.assert_not_called()


@pytest.mark.parametrize(
    "is_diskgraph,bp_config",
    [
        (False,
         _make_bp_config(task_type=TaskType.FORECASTING, forecast_length=30)),
        (True, _make_bp_config()),
    ],
)
@pytest.mark.asyncio
async def test_prediction_to_sdk_normal_path_for_non_diskgraph_or_non_forecast(
        monkeypatch, is_diskgraph, bp_config):
    generate_header, generate_prediction = (
        _patch_prediction_to_sdk_normal_path(monkeypatch, is_diskgraph))
    jobs_driver, pquery_driver, graph_driver, connector_driver = (
        _make_prediction_to_sdk_driver())
    ctx = AsyncMock()
    pquery_driver.get_batch_prediction_job.return_value = SimpleNamespace(
        config=bp_config,
        project_id=_PROJECT_ID,
    )
    connector_driver.get_bp_connector_metadata_for_sdk.return_value = {}
    pquery_driver.get_predictive_query_v2.return_value = SimpleNamespace(
        graph=_GraphForSdk(FPID("37371883384758475869584735748685")),
        query_string="PREDICT SUM(t.y, 0, 1) FOR EACH e.id",
    )
    graph_driver.get_graph_metadata_for_sdk.return_value = ([], [], None)
    pquery_driver.get_training_job_pquery_id_v2.return_value = None

    script = await jobs_driver.prediction_to_sdk(
        BatchPredictionJobID("bp-job-test"),
        ctx,
    )

    assert script == "HEADERLOAD_MODEL(trainingjob-test)PREDICTION_SCRIPT"
    generate_header.assert_called_once_with(str(_PROJECT_ID))
    assert generate_prediction.call_args.kwargs["project_id"] == str(
        _PROJECT_ID)
    pquery_driver.training_to_sdk.assert_not_called()
    pquery_driver.get_training_job_pquery_id_v2.assert_awaited_once_with(
        ctx,
        bp_config.training_job_id,
    )
    connector_driver.get_bp_connector_metadata_for_sdk.assert_awaited_once_with(
        "bp-connector",
        _PROJECT_ID,
        ctx.user_authorization,
    )
    graph_driver.get_graph_metadata_for_sdk.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_batch_prediction_job_without_pquery():
    pquery_driver: PQueryApiDriver = AsyncMock()
    data_snapshot_driver: DataSnapshotApiDriver = AsyncMock()
    ml_job_backend: MLJobBackend = AsyncMock()
    graph_driver: GraphDriver = AsyncMock()
    connector_driver: ConnectorApiDriver = AsyncMock()
    jobs_driver = JobsDriver(
        pquery_driver,
        data_snapshot_driver,
        ml_job_backend,
        graph_driver,
        connector_driver,
    )

    filter_overrides_input = BatchPredictionFilterOverrides(
        entity_filter="customer.Active = '1'",
        target_filter="articles.product_colour_group = 'Black'",
    )

    request = BatchPredictionInternalRequest(
        tags=['batch_pred_job', 'with_entity_filter'],
        model_training_job_id='training-job-383789',
        predict_options=BatchPredictionOptions(num_workers=1),
        project_id=_PROJECT_ID,
        outputs=cast(List[PredictionOutputConfig], [
            S3PredictionOutput(
                artifact_type=PredictionArtifactType.PREDICTIONS,
                connector_id=_CONNECTOR_ID,
                file_path=('s3://kumo-public-datasets/test/prediction.parquet'),
            )
        ]),
        filter_overrides=filter_overrides_input,
        explanations=False,
        anchor_time=datetime(2021, 1, 1),
    )

    result = ('bp-job-diudiddui', ValidationResponse())
    pquery_driver.start_batch_prediction_job_v2.return_value = result

    result = await jobs_driver.create_batch_prediction_job(
        AsyncMock(),
        request,
        JobSource.UI,
    )

    pquery_driver.start_batch_prediction_job_v2.assert_called_once()
    pquery_driver.start_generate_pred_table.assert_called_once()
    gen_table_request = pquery_driver.start_generate_pred_table.call_args[0][0]
    ns_timestamp = request.anchor_time.timestamp() * 1e9  # type:ignore
    assert gen_table_request.plan.anchor_time == ns_timestamp  # type: ignore
    data_snapshot_driver.assert_not_called()
    call_args = pquery_driver.start_batch_prediction_job_v2.call_args
    assert call_args is not None  # Ensure call_args is captured

    # Validate the arguments
    _, call_args = call_args

    assert isinstance(call_args['ctx'], AsyncMock)
    assert call_args['job_source'] == JobSource.UI
    request = call_args['request']
    filter_overrides = call_args['filter_overrides']
    assert request is not None
    assert request.model_training_job_id == 'training-job-383789'
    assert request.graph_snapshot_id is None
    assert len(request.outputs) == 1
    assert isinstance(request.outputs[0], S3PredictionOutput)
    assert request.outputs[
        0].artifact_type == PredictionArtifactType.PREDICTIONS
    assert request.pred_table_job_id is not None
    assert 'batch_pred_job' in request.job_tags
    assert 'with_entity_filter' in request.job_tags
    assert request.job_tags['batch_pred_job'] == 'batch_pred_job'
    assert request.job_tags['with_entity_filter'] == 'with_entity_filter'
    assert filter_overrides == filter_overrides_input


@pytest.mark.asyncio
async def test_create_batch_prediction_job_with_pquery():
    pquery_driver: PQueryApiDriver = AsyncMock()
    data_snapshot_driver: DataSnapshotApiDriver = AsyncMock()
    ml_job_backend: MLJobBackend = AsyncMock()
    graph_driver: GraphDriver = AsyncMock()
    graph_definition = AsyncMock(spec=GraphDefinition)
    connector_driver: ConnectorApiDriver = AsyncMock()
    jobs_driver = JobsDriver(
        pquery_driver,
        data_snapshot_driver,
        ml_job_backend,
        graph_driver,
        connector_driver,
    )

    project_id = ProjectID('project_0123456789abcdef0123456789abcdef')
    request = BatchPredictionInternalRequest(
        tags=['custom_tag1'], model_training_job_id='training-job-383789',
        predict_options=BatchPredictionOptions(num_workers=1),
        outputs=cast(List[PredictionOutputConfig], [
            S3PredictionOutput(
                artifact_type=PredictionArtifactType.PREDICTIONS,
                connector_id=_CONNECTOR_ID,
                file_path=('s3://kumo-public-datasets/test/prediction.parquet'),
            )
        ]), explanations=False,
        predictive_query='PREDICT COUNT(t.*, 0, 10) FOR EACH c.id',
        graph_id=FPID('37371883384758475869584735748685'),
        project_id=project_id, anchor_time=datetime(2021, 1, 1))

    result = ('bp-job-diudiddui', ValidationResponse())
    pquery_driver.start_batch_prediction_job_v2.return_value = result
    data_snapshot_driver._get_graph_by_fpid.return_value = (
        None,
        graph_definition,
    )
    pquery_driver.create_predictive_query_v2.return_value = PredictiveQueryID(
        'pq-30983849384839483848495838294959')
    gen_pred_table_id = GeneratePredictionTableJobID('gen-pred-2897')
    graph_snapshot_id = GraphSnapshotID(
        'graph-37371883384758475869584735748688@47843')
    pquery_driver.start_generate_pred_table.return_value = (gen_pred_table_id)
    data_snapshot_driver.create_graph_snapshot.return_value = graph_snapshot_id

    result = await jobs_driver.create_batch_prediction_job(
        AsyncMock(),
        request,
        JobSource.UI,
    )

    pquery_driver.start_batch_prediction_job_v2.assert_called_once()
    data_snapshot_driver._get_graph_by_fpid.assert_called_once()
    data_snapshot_driver.create_graph_snapshot.assert_called_once()
    pquery_driver.create_predictive_query_v2.assert_called_once()
    pquery_driver.start_generate_pred_table.assert_called_once()

    gen_table_request = pquery_driver.start_generate_pred_table.call_args[0][0]
    job_source = pquery_driver.start_generate_pred_table.call_args[1]
    assert 'job_source' in job_source
    assert job_source['job_source'] == JobSource.UI
    ns_timestamp = request.anchor_time.timestamp() * 1e9  # type: ignore
    assert gen_table_request.plan.anchor_time == ns_timestamp
    snapshot_args = data_snapshot_driver.create_graph_snapshot.call_args
    assert snapshot_args is not None

    snapshot_args, _ = snapshot_args
    assert snapshot_args[0] == FPID('37371883384758475869584735748685')
    assert snapshot_args[1]
    assert data_snapshot_driver.create_graph_snapshot.call_args.kwargs == {
        'project_id': project_id
    }
    assert (pquery_driver.create_predictive_query_v2.await_args.
            kwargs['project_id'] == project_id)

    call_args = pquery_driver.start_batch_prediction_job_v2.call_args
    assert call_args is not None

    # Validate the arguments
    _, call_args = call_args

    assert isinstance(call_args['ctx'], AsyncMock)
    assert call_args['job_source'] == JobSource.UI
    request = call_args['request']
    assert request is not None
    assert request.model_training_job_id == 'training-job-383789'
    assert len(request.outputs) == 1
    assert isinstance(request.outputs[0], S3PredictionOutput)
    assert request.outputs[
        0].artifact_type == PredictionArtifactType.PREDICTIONS
    assert request.pred_table_job_id == gen_pred_table_id
    assert request.graph_snapshot_id == graph_snapshot_id
    assert call_args['custom_tags'] == ['custom_tag1']
