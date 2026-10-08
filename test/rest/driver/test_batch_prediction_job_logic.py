# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import kumoapi.data_source as data_source_api
import pytest
from kumo_core.prediction_options import resolve_classification_threshold
from kumo_core.types import PredictionArtifactType, TaskType
from kumo_core.types.schema import FPID
from kumoapi.data_source import ConnectorType
from kumoapi.pquery import QueryType

from rest.config.batch_prediction_job import (
    BatchPredictionJobConfig,
    BatchPredictionJobInput,
)
from rest.config.connector import (
    ADLSConnectorResourceConfig,
    GCSConnectorResourceConfig,
    S3ConnectorResourceConfig,
)
from rest.config.data_backend import (
    ConnectorExecutionRef,
    ConnectorSpec,
    GraphDataVersionInfo,
)
from rest.config.graph_v2 import GraphDataComputeInput, GraphDefinition
from rest.config.prediction_output import (
    ADLSPredictionOutput,
    BigQueryPredictionOutput,
    DatabricksPredictionOutput,
    GCSPredictionOutput,
    PredictionStorageType,
    S3PredictionOutput,
    SnowflakePredictionOutput,
)
from rest.driver import batch_prediction_job_logic
from rest.driver.batch_prediction_job_logic import (
    BatchPredictionJobLogic,
    _file_output_path,
    _PendingOutputs,
    _validate_trained_model_artifact_support,
)
from statedb.config.common import (
    BatchPredictionJobID,
    ConnectorID,
    GeneratePredictionTableJobID,
    GraphIDv2,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.config.job_tag import (
    TAG_KEY_MODEL_ID,
    TAG_KEY_PARENT_JOB,
    TAG_KEY_PQUERY_ID,
    TAG_KEY_PQUERY_NAME,
    JobTagValue,
)
from statedb.config.predictive_query import FilterOverrideArgs
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.connector_authorization import (
    CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS,
    CONNECTOR_GRAPH_ACTIONS,
    CONNECTOR_OUTPUT_ACTIONS,
)
from statedb.utils.errors import ConnectorDoesNotExistError
from temporalio_commons.auto_rpc_exception import RpcError

_PROJECT_ID = ProjectID.generate()


@pytest.mark.parametrize('task_type', [TaskType.STATIC_LINK_PREDICTION,
                                       TaskType.TEMPORAL_LINK_PREDICTION])
@pytest.mark.parametrize('count', [None, 500, 501])
@pytest.mark.parametrize('has_predictions', [False, True])
def test_link_prediction_count_validation(task_type, count, has_predictions):
    request = SimpleNamespace(
        predict_options=SimpleNamespace(binary_classification_threshold=None,
                                        num_classes_to_return=count),
        outputs=[SimpleNamespace(artifact_type=(
            PredictionArtifactType.PREDICTIONS if has_predictions else
            PredictionArtifactType.EMBEDDINGS))],
    )
    if has_predictions and count != 500:
        message = ('specify the number of predictions' if count is None else
                   'less than or equal to 500')
        with pytest.raises(ValueError, match=message):
            batch_prediction_job_logic._validate_task_type_options_v2(
                task_type, request)
    else:
        batch_prediction_job_logic._validate_task_type_options_v2(
            task_type, request)
    assert request.predict_options.num_classes_to_return == count


@pytest.mark.parametrize(
    'task_type',
    [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION],
)
def test_effective_classification_threshold_defaults_thresholded_tasks(
    task_type: TaskType, ) -> None:
    assert resolve_classification_threshold(task_type, None) == 0.5


@pytest.mark.parametrize('threshold', [0.0, 0.25, 1.0])
def test_effective_multilabel_threshold_preserves_explicit_value(
    threshold: float, ) -> None:
    assert resolve_classification_threshold(
        TaskType.MULTILABEL_CLASSIFICATION,
        threshold,
    ) == threshold


@pytest.mark.parametrize('threshold', [-0.01, 1.01])
def test_effective_multilabel_threshold_rejects_out_of_range(
    threshold: float, ) -> None:
    with pytest.raises(ValueError, match='between 0 and 1'):
        resolve_classification_threshold(
            TaskType.MULTILABEL_CLASSIFICATION,
            threshold,
        )


def test_id_gnn_embeddings_request_has_actionable_validation(
        monkeypatch) -> None:
    from kumo_core.training.config import LinkPredModuleType

    monkeypatch.setattr(
        batch_prediction_job_logic.HomeDirManager,
        'get_best_trainer_dir',
        lambda *_args: '/model',
    )
    monkeypatch.setattr(
        'io_utils.json.load_json',
        lambda *_args, **_kwargs: SimpleNamespace(module_kwargs=SimpleNamespace(
            module=LinkPredModuleType.ID)),
    )
    request = SimpleNamespace(outputs=[
        SimpleNamespace(artifact_type=PredictionArtifactType.EMBEDDINGS)
    ])

    with pytest.raises(ValueError,
                       match='Embeddings are not available for ID-GNN'):
        _validate_trained_model_artifact_support(
            request,
            TaskType.STATIC_LINK_PREDICTION,
            TrainingJobID.make_random_id(),
        )


_OUTPUT_CASES = (
    (
        'snowflake',
        ConnectorType.SNOWFLAKE,
        SnowflakePredictionOutput(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            connector_id=ConnectorID.generate(),
            table_name='predictions',
        ),
    ),
    (
        'databricks',
        ConnectorType.DATABRICKS,
        DatabricksPredictionOutput(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            connector_id=ConnectorID.generate(),
            table_name='schema.predictions',
        ),
    ),
    (
        'bigquery',
        ConnectorType.BIGQUERY,
        BigQueryPredictionOutput(
            storage_type=PredictionStorageType.BIGQUERY,
            artifact_type=PredictionArtifactType.PREDICTIONS,
            connector_id=ConnectorID.generate(),
            table_name='predictions',
        ),
    ),
    (
        's3',
        ConnectorType.S3,
        S3PredictionOutput(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            connector_id=ConnectorID.generate(),
            file_path='s3://bucket/root/predictions/',
        ),
    ),
    (
        'gcs',
        ConnectorType.GCS,
        GCSPredictionOutput(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            connector_id=ConnectorID.generate(),
            file_path='gs://bucket/root/predictions/',
        ),
    ),
    (
        'adls',
        ConnectorType.ADLS,
        ADLSPredictionOutput(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            connector_id=ConnectorID.generate(),
            file_path=('abfss://filesystem@account.dfs.core.windows.net/root/'),
        ),
    ),
)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture(autouse=True, scope="session")
def setup_rest_test_session():
    yield


def _logic(
    ml_backend=None,
    data_backend=None,
    *,
    state_services=None,
    **services,
) -> BatchPredictionJobLogic:
    if state_services is None and services:
        state_services = SimpleNamespace(**services)
    return BatchPredictionJobLogic(
        ml_backend if ml_backend is not None else AsyncMock(),
        data_backend if data_backend is not None else AsyncMock(),
        state_services=state_services,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1])
async def test_v2_threshold_errors_use_the_validation_response_adapter(
        threshold) -> None:
    request = SimpleNamespace(
        predict_options=SimpleNamespace(
            num_workers=1,
            binary_classification_threshold=threshold,
            prediction_time=None,
            num_classes_to_return=None,
        ),
        outputs=None,
    )

    response = await _logic().validate_batch_prediction_args_v2(
        MagicMock(),
        request,
        allow_local_paths=False,
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        project_id=_PROJECT_ID,
    )

    assert not response.ok
    assert len(response.errors) == 1
    assert "between 0 and 1" in response.errors[0].message


def test_tef_is_not_used_in_pql_v2():
    bp_job_logic = _logic()
    response = bp_job_logic.filters_are_query_compatible(
        FilterOverrideArgs(
            temporal_entity_filter_override='COUNT(TRANSACTION.*, -60, 0) > 0'),
    )

    assert response is not None, "Response should not be None"
    assert hasattr(response,
                   'errors'), "Response should have an 'errors' attribute"
    assert len(response.errors) == 1, "There should be exactly one error"
    assert response.errors[0].title == 'Incorrect filter override'
    assert response.errors[0].message.startswith('temporal_entity_filter is')


@pytest.mark.asyncio
async def test_get_graph_data_compute_input_uses_state_services_without_legacy(
    monkeypatch, ):
    legacy_graph_logic = MagicMock(
        side_effect=AssertionError('legacy GraphLogic should not be opened'))
    monkeypatch.setattr(
        batch_prediction_job_logic.graph_logic,
        'GraphLogic',
        legacy_graph_logic,
    )

    graph_input = GraphDataComputeInput(
        graph_display_name='graph',
        graph_key='graph-key',
        tables={},
        base_tables={},
        connectors={},
        graph_definition=GraphDefinition(tables={}, col_groups=[]),
    )
    graph_service = SimpleNamespace(get_graph_data_compute_input=AsyncMock(
        return_value=graph_input))
    state_services = SimpleNamespace(graphs=graph_service)
    graph_id = GraphIDv2.compose(fpid=FPID('a' * 32))

    result = await batch_prediction_job_logic._get_graph_data_compute_input_v2(
        graph_id,
        _PROJECT_ID,
        state_services,
    )

    assert result is graph_input
    graph_service.get_graph_data_compute_input.assert_awaited_once_with(
        graph_id,
        project_id=_PROJECT_ID,
        user_authorization=None,
    )
    legacy_graph_logic.assert_not_called()


@pytest.mark.asyncio
async def test_delete_old_bp_jobs_uses_state_services_workflow_rows(
    monkeypatch, ):
    monkeypatch.setattr(
        batch_prediction_job_logic.MAX_BP_JOBS_PER_WORKFLOW,
        "parse_value",
        lambda: 2,
    )

    oldest_id = BatchPredictionJobID("bp-oldest")
    ml_jobs = SimpleNamespace(
        list_batch_prediction_prune_candidate_ids=AsyncMock(
            return_value=[oldest_id]))
    logic = _logic(ml_jobs=ml_jobs)
    logic.delete_many_batch_prediction_jobs = AsyncMock()
    ctx = MagicMock()

    await logic._delete_old_bp_jobs(ctx, "workflow-one", project_id=_PROJECT_ID)

    ml_jobs.list_batch_prediction_prune_candidate_ids.assert_awaited_once_with(
        "workflow-one",
        project_ids={_PROJECT_ID},
        keep_most_recent=2,
    )
    logic.delete_many_batch_prediction_jobs.assert_awaited_once_with(
        ctx, {oldest_id})


@pytest.mark.asyncio
async def test_resolve_bp_graph_id_uses_snapshot_info(monkeypatch, ):
    snapshot_info = AsyncMock()
    monkeypatch.setattr(
        batch_prediction_job_logic,
        "get_graph_snapshot_info",
        snapshot_info,
    )
    graph_fpid = FPID("a" * 32)
    graph_id = GraphIDv2.compose(fpid=graph_fpid)
    version_info = SimpleNamespace(graph_data_version="version-one")
    snapshot_info.return_value = (graph_id, version_info)
    logic = _logic()
    pq_resource = SimpleNamespace(graph=SimpleNamespace(fpid=graph_fpid))

    result = await logic._resolve_bp_graph_id(
        pq_resource,
        "snapshot-one",
        "pquery-one",
    )

    assert result == (graph_id, "version-one")
    snapshot_info.assert_awaited_once_with("snapshot-one")


@pytest.mark.asyncio
async def test_cancel_batch_prediction_job():
    ml_backend = AsyncMock()
    pquery_id = PredictiveQueryID('pquery')
    rows = [
        create_state_bp_row('bp1', pquery_id),
        create_state_bp_row('bp2', pquery_id),
    ]
    ml_jobs = SimpleNamespace(
        list_batch_prediction_rows=AsyncMock(return_value=[rows[1]]), )
    bp_job_logic = _logic(ml_backend, ml_jobs=ml_jobs)
    ml_backend.cancel_batch_prediction_job.return_value = True

    response = await bp_job_logic.cancel_batch_prediction_job(
        AsyncMock(),
        pquery_id,
        BatchPredictionJobID('bp2'),
    )

    assert response
    ml_jobs.list_batch_prediction_rows.assert_awaited_once_with(
        project_ids=None,
        pquery_id=pquery_id,
        job_ids={BatchPredictionJobID('bp2')},
        limit=1,
        match_pquery_def_fpid=False,
    )
    bp_job_logic._ml_backend.cancel_batch_prediction_job.\
        assert_awaited_once_with(BatchPredictionJobID('bp2'))  # type: ignore


@pytest.mark.asyncio
async def test_apply_bp_job_tags_uses_ml_backend():
    bp_job_id = BatchPredictionJobID.make_random_id()
    training_job_id = TrainingJobID.make_random_id()
    pred_table_job_id = GeneratePredictionTableJobID.as_child_of(bp_job_id)
    ml_backend = AsyncMock()
    bp_job_logic = _logic(ml_backend, )
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('pquery'),
        training_job_id=training_job_id,
        task_type=TaskType.REGRESSION,
        generate_pred_table_job_id=pred_table_job_id,
    )

    await bp_job_logic._apply_bp_job_tags(
        bp_job_id,
        config,
        ["custom"],
        training_job_id,
    )

    assert ml_backend.apply_job_tags.await_count == 2
    bp_call, pred_table_call = ml_backend.apply_job_tags.await_args_list
    assert bp_call.args[0] == JobType.BATCH_PREDICTION_JOB
    assert bp_call.args[1] == bp_job_id
    assert bp_call.args[2][TAG_KEY_MODEL_ID] == JobTagValue(training_job_id)
    assert bp_call.args[2][TAG_KEY_PQUERY_NAME] == JobTagValue.of_pquery_name(
        config.predictive_query_id)
    assert bp_call.args[2][TAG_KEY_PQUERY_ID] == JobTagValue.of_pquery_id(
        config.pquery_def_fpid)
    assert pred_table_call.args[0] == JobType.GENERATE_PREDICTION_TABLE_JOB
    assert pred_table_call.args[1] == pred_table_job_id
    assert TAG_KEY_MODEL_ID not in pred_table_call.args[2]
    assert pred_table_call.args[2][TAG_KEY_PARENT_JOB] == JobTagValue(bp_job_id)


def create_state_bp_row(bp_job_id: str | BatchPredictionJobID,
                        pquery_id: PredictiveQueryID) -> SimpleNamespace:
    id = (bp_job_id if isinstance(bp_job_id, BatchPredictionJobID) else
          BatchPredictionJobID(bp_job_id))
    return SimpleNamespace(
        id=id,
        config=BatchPredictionJobConfig(
            predictive_query_id=pquery_id,
            training_job_id=TrainingJobID("tjob1"),
            task_type=TaskType.REGRESSION,
        ),
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        description="",
        batch_prediction_workflow_id=None,
        source=None,
        user=None,
        project_id=_PROJECT_ID,
        kumo_version=None,
        storage_path=None,
        input_data_stats=None,
        registry_metadata=None,
    )


@pytest.mark.asyncio
async def test_get_materialization_info_uses_state_services_without_legacy():
    bp_job_id = BatchPredictionJobID("bp-state")
    expected = object()
    ml_backend = AsyncMock()
    ml_backend.get_materialization_info.return_value = expected
    ml_jobs = SimpleNamespace(get_batch_prediction_rows=AsyncMock(return_value=[
        create_state_bp_row(bp_job_id, PredictiveQueryID("pquery"))
    ]))
    logic = _logic(ml_backend, ml_jobs=ml_jobs)

    result = await logic.get_materialization_info(MagicMock(), bp_job_id)

    assert result is expected
    ml_jobs.get_batch_prediction_rows.assert_awaited_once_with([bp_job_id],
                                                               project_ids=None)
    ml_backend.get_materialization_info.assert_awaited_once_with(bp_job_id)


@pytest.mark.asyncio
async def test_cancel_batch_prediction_job_uses_state_services_without_legacy():
    pquery_id = PredictiveQueryID("pquery")
    bp_job_id = BatchPredictionJobID("bp2")
    ml_backend = AsyncMock()
    ml_backend.cancel_batch_prediction_job.return_value = True
    rows = [
        create_state_bp_row("bp1", pquery_id),
        create_state_bp_row("bp2", pquery_id),
        create_state_bp_row("bp3", PredictiveQueryID("other")),
    ]
    ml_jobs = SimpleNamespace(
        list_batch_prediction_rows=AsyncMock(return_value=[rows[1]]), )
    logic = _logic(ml_backend, ml_jobs=ml_jobs)

    result = await logic.cancel_batch_prediction_job(MagicMock(), pquery_id,
                                                     bp_job_id)

    assert result is True
    ml_jobs.list_batch_prediction_rows.assert_awaited_once_with(
        project_ids=None,
        pquery_id=pquery_id,
        job_ids={bp_job_id},
        limit=1,
        match_pquery_def_fpid=False,
    )
    ml_backend.cancel_batch_prediction_job.assert_awaited_once_with(bp_job_id)


@pytest.mark.asyncio
async def test_delete_batch_prediction_uses_state_services_without_legacy():
    bp_job_id = BatchPredictionJobID("bp-state")
    ml_backend = AsyncMock()
    ml_backend.delete_batch_prediction.return_value = True
    ml_jobs = SimpleNamespace(
        get_batch_prediction_rows=AsyncMock(return_value=[
            create_state_bp_row(bp_job_id, PredictiveQueryID("pquery"))
        ]),
        delete_batch_prediction_metadata=AsyncMock(return_value=True),
        delete_job_tags=AsyncMock(return_value=True),
    )
    logic = _logic(ml_backend, ml_jobs=ml_jobs)
    ctx = MagicMock()

    assert await logic.delete_batch_prediction(ctx, bp_job_id)

    ml_jobs.get_batch_prediction_rows.assert_not_awaited()
    ml_jobs.delete_batch_prediction_metadata.assert_awaited_once_with(bp_job_id)
    ml_jobs.delete_job_tags.assert_awaited_once_with(
        JobType.BATCH_PREDICTION_JOB, bp_job_id)
    ml_backend.delete_batch_prediction.assert_awaited_once_with(ctx, bp_job_id)


@pytest.mark.asyncio
async def test_delete_batch_prediction_returns_false_when_backend_finds_no_job(
):
    bp_job_id = BatchPredictionJobID("bp-missing")
    ml_backend = AsyncMock()
    ml_backend.delete_batch_prediction.return_value = False
    ml_jobs = SimpleNamespace(
        get_batch_prediction_rows=AsyncMock(return_value=[]),
        delete_batch_prediction_metadata=AsyncMock(return_value=True),
        delete_job_tags=AsyncMock(return_value=True),
    )
    logic = _logic(ml_backend, ml_jobs=ml_jobs)
    ctx = MagicMock()

    assert await logic.delete_batch_prediction(ctx, bp_job_id) is False

    ml_backend.delete_batch_prediction.assert_awaited_once_with(ctx, bp_job_id)
    ml_jobs.get_batch_prediction_rows.assert_not_awaited()
    ml_jobs.delete_batch_prediction_metadata.assert_not_called()
    ml_jobs.delete_job_tags.assert_not_called()


@pytest.mark.asyncio
async def test_delete_batch_prediction_does_not_mutate_state_metadata_when_backend_fails(
):
    bp_job_id = BatchPredictionJobID("bp-state")
    ml_backend = AsyncMock()
    ml_backend.delete_batch_prediction.side_effect = RpcError(
        "ml",
        "delete_batch_prediction",
        ValueError.__name__,
        "backend rejected delete",
        "",
        True,
    )
    ml_jobs = SimpleNamespace(
        get_batch_prediction_rows=AsyncMock(return_value=[
            create_state_bp_row(bp_job_id, PredictiveQueryID("pquery"))
        ]),
        delete_batch_prediction_metadata=AsyncMock(return_value=True),
        delete_job_tags=AsyncMock(return_value=True),
    )
    logic = _logic(ml_backend, ml_jobs=ml_jobs)
    ctx = MagicMock()

    with pytest.raises(ValueError, match="backend rejected delete"):
        await logic.delete_batch_prediction(ctx, bp_job_id)

    ml_jobs.get_batch_prediction_rows.assert_not_awaited()
    ml_backend.delete_batch_prediction.assert_awaited_once_with(ctx, bp_job_id)
    ml_jobs.delete_batch_prediction_metadata.assert_not_called()
    ml_jobs.delete_job_tags.assert_not_called()


@pytest.mark.asyncio
async def test_delete_many_batch_prediction_jobs_uses_state_services_without_legacy(
):
    pquery_id = PredictiveQueryID("pquery")
    bp_job_id = BatchPredictionJobID("bp-state")
    missing_id = BatchPredictionJobID("bp-missing")
    ml_backend = AsyncMock()
    ml_backend.delete_many_batch_prediction_jobs.return_value = {bp_job_id}
    ml_jobs = SimpleNamespace(
        get_batch_prediction_rows=AsyncMock(
            return_value=[create_state_bp_row(bp_job_id, pquery_id)]),
        delete_batch_prediction_metadata_and_tags=AsyncMock(
            return_value={bp_job_id}),
    )
    logic = _logic(ml_backend, ml_jobs=ml_jobs)
    ctx = MagicMock()

    result = await logic.delete_many_batch_prediction_jobs(
        ctx, {bp_job_id, missing_id})

    assert result is False
    ml_jobs.get_batch_prediction_rows.assert_awaited_once()
    assert set(ml_jobs.get_batch_prediction_rows.await_args.args[0]) == {
        bp_job_id, missing_id
    }
    ml_backend.delete_many_batch_prediction_jobs.assert_awaited_once()
    backend_ctx, backend_ids = (
        ml_backend.delete_many_batch_prediction_jobs.await_args.args)
    assert backend_ctx is ctx
    assert backend_ids == {bp_job_id}
    ml_jobs.delete_batch_prediction_metadata_and_tags.assert_awaited_once_with(
        {bp_job_id})


@pytest.mark.asyncio
async def test_get_current_pred_job_ids_v2_uses_state_services_without_legacy():
    pquery_id = PredictiveQueryID("pquery")
    other_pquery_id = PredictiveQueryID("other")
    ml_jobs = SimpleNamespace(get_current_batch_prediction_job_ids=AsyncMock(
        return_value={
            pquery_id: BatchPredictionJobID("bp-new"),
            other_pquery_id: BatchPredictionJobID("bp-other"),
        }))
    logic = _logic(ml_jobs=ml_jobs)

    assert await logic.get_current_pred_job_ids_v2(
        MagicMock(),
        [pquery_id, other_pquery_id,
         PredictiveQueryID("missing")],
    ) == [
        BatchPredictionJobID("bp-new"),
        BatchPredictionJobID("bp-other"),
        None,
    ]
    ml_jobs.get_current_batch_prediction_job_ids.assert_awaited_once_with(
        [pquery_id, other_pquery_id,
         PredictiveQueryID("missing")],
        project_ids=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('failure', 'error_type', 'message'),
    [
        pytest.param('wrong-type', ValueError, 'snowflake connector',
                     id='wrong-type'),
        pytest.param('missing', ConnectorDoesNotExistError, 'does not exist',
                     id='missing'),
    ],
)
async def test_output_connectors_propagate_authorized_lookup_errors(
    failure,
    error_type,
    message,
):
    connector_id = ConnectorID.generate()
    project_id = ProjectID.generate()
    error = (
        ValueError(f'Connector {connector_id} is not a snowflake connector') if
        failure == 'wrong-type' else ConnectorDoesNotExistError(connector_id))
    connector_service = SimpleNamespace(execution_ref_for_project=AsyncMock(
        side_effect=error))
    logic = _logic(connectors=connector_service)

    output = SnowflakePredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=connector_id,
        table_name="preds",
    )
    with pytest.raises(error_type, match=message):
        await logic._get_output_connectors([output], project_id)


@pytest.mark.asyncio
async def test_pqueries_by_id_uses_bulk_id_lookup_when_available():
    risk_fpid = FPID("0123456789abcdef0123456789abcdef")
    churn_fpid = FPID("11111111111111111111111111111111")
    risk_id = PredictiveQueryID.compose("riskpquery", risk_fpid)
    churn_id = PredictiveQueryID.compose("churnpquery", churn_fpid)
    missing_id = PredictiveQueryID("missing")
    risk_def = object()
    churn_def = object()
    pqueries = SimpleNamespace(
        get_definitions_by_ids=AsyncMock(return_value={
            risk_id: risk_def,
            churn_id: churn_def,
        }),
        get_definitions=AsyncMock(
            side_effect=AssertionError("fpid pquery lookup used")),
        get_definition_by_id=AsyncMock(
            side_effect=AssertionError("scalar pquery lookup used")),
    )
    logic = _logic(pqueries=pqueries)

    assert await logic._pqueries_by_id([risk_id, risk_id, churn_id,
                                        missing_id]) == {
                                            risk_id: risk_def,
                                            churn_id: churn_def,
                                            missing_id: None,
                                        }
    pqueries.get_definitions_by_ids.assert_awaited_once_with(
        [risk_id, churn_id, missing_id])
    pqueries.get_definitions.assert_not_awaited()
    pqueries.get_definition_by_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_output_connectors_use_authorized_secret_free_references():
    project_id = ProjectID.generate()
    references = {
        ConnectorID(output.connector_id):
        ConnectorExecutionRef(connector_id=ConnectorID(output.connector_id),
                              project_id=project_id)
        for _, _, output in _OUTPUT_CASES
    }
    connector_service = SimpleNamespace(
        execution_ref_for_project=AsyncMock(
            side_effect=lambda _principal, _project_id, connector_id, _actions,
            **_kwargs: references[connector_id]), )
    state_services = SimpleNamespace(connectors=connector_service)
    logic = _logic(state_services=state_services)

    outputs = [output for _, _, output in _OUTPUT_CASES]
    snowflake_id = ConnectorID(outputs[0].connector_id)
    outputs.append(
        SnowflakePredictionOutput(
            artifact_type=PredictionArtifactType.EMBEDDINGS,
            connector_id=snowflake_id,
            table_name='embeddings',
        ))
    output_connectors = await logic._get_output_connectors(outputs, project_id)

    assert output_connectors == references
    assert connector_service.execution_ref_for_project.await_count == 6
    for _, connector_type, output in _OUTPUT_CASES:
        actions = (CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS
                   if connector_type == ConnectorType.DATABRICKS else
                   CONNECTOR_OUTPUT_ACTIONS)
        connector_service.execution_ref_for_project.assert_any_await(
            None,
            project_id,
            ConnectorID(output.connector_id),
            actions,
            expected_type=connector_type,
        )


@pytest.mark.parametrize(
    ('output', 'expected_type'),
    [
        pytest.param(output, connector_type, id=name)
        for name, connector_type, output in _OUTPUT_CASES[:3]
    ],
)
@pytest.mark.asyncio
async def test_warehouse_output_validation_uses_typed_project_reference(
    output,
    expected_type,
    monkeypatch,
):
    project_id = ProjectID.generate()
    connector_id = ConnectorID(output.connector_id)
    reference = ConnectorExecutionRef(
        connector_id=connector_id,
        project_id=project_id,
    )
    connector_service = SimpleNamespace(execution_ref_for_project=AsyncMock(
        return_value=reference))
    logic = _logic(connectors=connector_service)
    execute_activity = AsyncMock(return_value=None)
    monkeypatch.setattr(
        batch_prediction_job_logic.DataBackend,
        '_execute_single_activity',
        execute_activity,
    )

    await logic._validate_warehouse_output(
        output,
        _PendingOutputs(),
        project_id,
    )

    connector_service.execution_ref_for_project.assert_awaited_once_with(
        None,
        project_id,
        connector_id,
        CONNECTOR_OUTPUT_ACTIONS,
        expected_type=expected_type,
    )
    execute_activity.assert_awaited_once()


def _object_output(provider, *, connector_id=None, path='predictions/',
                   absolute=False, artifact=PredictionArtifactType.PREDICTIONS):
    connector_id = connector_id or ConnectorID.generate()
    if provider == 's3':
        return S3PredictionOutput(artifact_type=artifact,
                                  connector_id=connector_id, file_path=path)
    output_type = (GCSPredictionOutput
                   if provider == 'gcs' else ADLSPredictionOutput)
    return output_type(artifact_type=artifact, connector_id=connector_id,
                       file_path=path)


def _object_connector_config(provider, *, rooted):
    if provider == 's3':
        target = (data_source_api.S3RootedConnectorTarget(
            bucket='customer-data', prefix='root')
                  if rooted else data_source_api.S3AnyURIConnectorTarget())
        return S3ConnectorResourceConfig(name=provider, target=target)
    if provider == 'gcs':
        target = (data_source_api.GCSRootedConnectorTarget(
            bucket='customer-data', prefix='root')
                  if rooted else data_source_api.GCSAnyURIConnectorTarget())
        return GCSConnectorResourceConfig(name=provider, target=target)
    target = (data_source_api.ADLSRootedConnectorTarget(
        storage_account='account', filesystem='filesystem', prefix='root')
              if rooted else data_source_api.ADLSAnyURIConnectorTarget())
    return ADLSConnectorResourceConfig(name=provider, target=target)


@pytest.mark.parametrize(
    ('output', 'config', 'relative_path'),
    [
        pytest.param(
            _object_output('s3', path='s3://customer-data/root/predictions/'),
            _object_connector_config('s3', rooted=True),
            'predictions/',
            id='s3',
        ),
        pytest.param(
            _object_output('gcs', path='gs://customer-data/root/predictions/'),
            _object_connector_config('gcs', rooted=True),
            'predictions/',
            id='gcs',
        ),
        pytest.param(
            _object_output(
                'adls', path=('abfss://filesystem@account.dfs.core.windows.net/'
                              'root/predictions/')),
            _object_connector_config('adls', rooted=True),
            'predictions/',
            id='adls',
        ),
    ],
)
@pytest.mark.asyncio
async def test_full_rooted_object_output_normalized_after_project_lookup(
    output,
    config,
    relative_path,
    monkeypatch,
):
    project_id = ProjectID.generate()
    connector_id = ConnectorID(output.connector_id)
    reference = ConnectorExecutionRef(connector_id=connector_id,
                                      project_id=project_id)
    connector_service = SimpleNamespace(
        get_for_project=AsyncMock(return_value=SimpleNamespace(config=config)),
        execution_ref_for_project=AsyncMock(return_value=reference),
    )
    logic = _logic(connectors=connector_service)
    execute_activity = AsyncMock(return_value=None)
    monkeypatch.setattr(batch_prediction_job_logic.DataBackend,
                        '_execute_single_activity', execute_activity)

    assert await logic._validate_object_output(output, False, _PendingOutputs(),
                                               project_id) is None

    connector_service.get_for_project.assert_awaited_once_with(
        project_id, connector_id, None)
    assert output.file_path == relative_path
    execute_activity.assert_awaited_once()


@pytest.mark.asyncio
async def test_object_output_authorizes_before_normalization_or_io(monkeypatch):
    connector_id = ConnectorID.generate()
    output = GCSPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=connector_id,
        file_path='gs://customer-data/root/predictions/',
    )
    connector_service = SimpleNamespace(
        get_for_project=AsyncMock(
            side_effect=ConnectorDoesNotExistError(connector_id)),
        execution_ref_for_project=AsyncMock(),
    )
    normalize = MagicMock(side_effect=AssertionError('must not normalize'))
    execute_activity = AsyncMock(side_effect=AssertionError('must not run'))
    monkeypatch.setattr(batch_prediction_job_logic,
                        'normalize_object_prediction_output', normalize)
    monkeypatch.setattr(batch_prediction_job_logic.DataBackend,
                        '_execute_single_activity', execute_activity)

    result = await _logic(connectors=connector_service
                          )._validate_object_output(output, False,
                                                    _PendingOutputs(),
                                                    ProjectID.generate())

    assert result is not None
    normalize.assert_not_called()
    connector_service.execution_ref_for_project.assert_not_awaited()
    execute_activity.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('output', 'config'),
    [
        (
            _object_output('s3', path='s3://customer-data/predictions/',
                           absolute=True),
            _object_connector_config('s3', rooted=False),
        ),
        (
            _object_output('gcs', path='gs://customer-data/predictions/',
                           absolute=True),
            _object_connector_config('gcs', rooted=False),
        ),
        (
            _object_output(
                'adls', path=('abfss://data@account.dfs.core.windows.net/'
                              'predictions/'), absolute=True),
            _object_connector_config('adls', rooted=False),
        ),
    ],
    ids=['s3', 'gcs', 'adls'],
)
async def test_rootless_prediction_output_is_authorized_and_validated(
        monkeypatch, output, config):
    project_id = ProjectID.generate()
    reference = ConnectorExecutionRef(
        connector_id=ConnectorID(output.connector_id), project_id=project_id)
    connector_service = SimpleNamespace(
        get_for_project=AsyncMock(return_value=SimpleNamespace(config=config)),
        execution_ref_for_project=AsyncMock(return_value=reference),
    )
    execute_activity = AsyncMock(return_value=None)
    monkeypatch.setattr(batch_prediction_job_logic.DataBackend,
                        '_execute_single_activity', execute_activity)

    result = await _logic(connectors=connector_service
                          )._validate_object_output(output, False,
                                                    _PendingOutputs(),
                                                    project_id)

    assert result is None, result.message
    connector_service.execution_ref_for_project.assert_awaited_once()
    execute_activity.assert_awaited_once()


def test_rooted_object_output_conflicts_are_scoped_to_connector():
    shared_path = 'predictions/'
    first_connector = ConnectorID.generate()
    second_connector = ConnectorID.generate()
    outputs = [
        _object_output('s3', connector_id=first_connector, path=shared_path),
        _object_output('gcs', connector_id=second_connector, path=shared_path),
        _object_output('gcs', path=shared_path),
        _object_output('adls', path=shared_path),
    ]
    pending = _PendingOutputs()

    for output in outputs:
        assert BatchPredictionJobLogic._reserve_file_output(output,
                                                            pending) is None

    duplicate = _object_output('gcs', connector_id=second_connector,
                               path=shared_path,
                               artifact=PredictionArtifactType.EMBEDDINGS)
    conflict = BatchPredictionJobLogic._reserve_file_output(duplicate, pending)
    assert conflict is not None
    assert 'Multiple artifacts with the same file path' in conflict.message


@pytest.mark.parametrize(('provider', 'uri'), [
    ('s3', 's3://bucket/predictions/'),
    ('gcs', 'gs://bucket/predictions/'),
    ('adls', 'abfss://data@account.dfs.core.windows.net/predictions/'),
])
def test_absolute_object_output_has_global_conflict_key(provider, uri):
    first_connector_id = ConnectorID.generate()
    second_connector_id = ConnectorID.generate()
    rooted = _object_output(provider, connector_id=first_connector_id)
    first_absolute = _object_output(provider, connector_id=first_connector_id,
                                    path=uri, absolute=True)
    second_absolute = _object_output(
        provider,
        connector_id=second_connector_id,
        path=uri,
        absolute=True,
        artifact=PredictionArtifactType.EMBEDDINGS,
    )
    pending = _PendingOutputs()

    assert _file_output_path(rooted).connector_id == first_connector_id
    assert _file_output_path(first_absolute).connector_id is None
    assert BatchPredictionJobLogic._reserve_file_output(first_absolute,
                                                        pending) is None
    conflict = BatchPredictionJobLogic._reserve_file_output(
        second_absolute, pending)
    assert conflict is not None
    assert 'Multiple artifacts with the same file path' in conflict.message


def _compute_input(graph_key: str) -> GraphDataComputeInput:
    return GraphDataComputeInput(
        graph_display_name="graph",
        graph_key=graph_key,
        tables={},
        base_tables={},
        connectors={},
        graph_definition=GraphDefinition(tables={}, col_groups=[]),
    )


def _gen_context_runtime(
    gen_job_id,
    project_id,
    *,
    rows=None,
    home='/jobs/gen-home',
    **services,
):
    if rows is None:
        rows = [
            SimpleNamespace(id=gen_job_id, project_id=project_id, config=None)
        ]
    ml_backend = SimpleNamespace(get_job_home_dir=AsyncMock(return_value=home))
    ml_jobs = SimpleNamespace(get_generate_pred_table_jobs=AsyncMock(
        return_value=rows))
    return SimpleNamespace(
        logic=_logic(ml_backend, ml_jobs=ml_jobs, **services),
        ml_backend=ml_backend,
        ml_jobs=ml_jobs,
    )


@pytest.mark.asyncio
async def test_saved_graph_warehouse_references_are_reauthorized():
    connector_id = ConnectorID.generate()
    project_id = ProjectID.generate()
    saved_reference = ConnectorExecutionRef(
        connector_id=connector_id,
        project_id=project_id,
    )
    refreshed_reference = ConnectorExecutionRef(
        connector_id=connector_id,
        project_id=project_id,
    )
    connector_service = SimpleNamespace(execution_ref_for_project=AsyncMock(
        return_value=refreshed_reference))
    logic = _logic(connectors=connector_service)
    graph_input = _compute_input("graph-key")
    graph_input.connectors[connector_id] = saved_reference

    await logic._reauthorize_graph_compute_input(graph_input, project_id)

    assert graph_input.connectors[connector_id] is refreshed_reference
    connector_service.execution_ref_for_project.assert_awaited_once_with(
        None,
        project_id,
        connector_id,
        CONNECTOR_GRAPH_ACTIONS,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('invalid_reference', 'message'),
    [
        pytest.param('foreign-project', 'different project',
                     id='foreign-project'),
        pytest.param('legacy-spec', 'project connector references',
                     id='legacy-spec'),
    ],
)
async def test_saved_graph_rejects_invalid_connector_reference(
    invalid_reference,
    message,
):
    connector_id = ConnectorID.generate()
    graph_input = _compute_input("graph-key")
    if invalid_reference == 'foreign-project':
        reference = ConnectorExecutionRef(
            connector_id=connector_id,
            project_id=ProjectID.generate(),
        )
    else:
        reference = ConnectorSpec(
            connector_id=connector_id,
            connector_config=object(),
        )
    graph_input.connectors[connector_id] = reference
    logic = _logic(connectors=SimpleNamespace())

    with pytest.raises(ValueError, match=message):
        await logic._reauthorize_graph_compute_input(graph_input,
                                                     ProjectID.generate())


@pytest.mark.asyncio
async def test_gen_pred_table_graph_context_reuses_complete_gen_identity(
    monkeypatch, ):
    """A decoupled gen-pred-table job and the BP job that consumes it must
    resolve the SAME ``(graph_key, graph_data_version)`` identity, so
    materialization reads the staging table from where the gen job wrote it.
    """
    gen_job_id = GeneratePredictionTableJobID.make_random_id()
    project_id = ProjectID.generate()
    compute_input = _compute_input("NA-forked-compute-key")
    version_info = SimpleNamespace(graph_data_version='forked-version')
    gen_job_input = SimpleNamespace(
        graph_data_compute_input=compute_input,
        graph_version_info=version_info,
    )

    load_spy = AsyncMock(return_value=gen_job_input)
    monkeypatch.setattr(batch_prediction_job_logic, "load_json_async", load_spy)
    runtime = _gen_context_runtime(gen_job_id, project_id)

    result = await runtime.logic._gen_pred_table_graph_context(
        gen_job_id, project_id=project_id)

    assert result is not None
    assert result.graph_data_compute_input is compute_input
    assert result.graph_version_info is version_info
    assert result.graph_data_version == 'forked-version'
    runtime.ml_backend.get_job_home_dir.assert_awaited_once_with(gen_job_id)
    load_spy.assert_awaited_once_with(
        '/jobs/gen-home/inputs/prediction_job_input.json',
        BatchPredictionJobInput,
        raise_on_missing=True,
    )


@pytest.mark.parametrize(
    ('input_state', 'message'),
    [
        pytest.param('missing', 'saved input.*missing', id='missing'),
        pytest.param('empty', 'saved input.*empty', id='empty'),
        pytest.param('version-info', 'complete', id='missing-version-info'),
        pytest.param('graph-key', 'complete', id='missing-graph-key'),
    ],
)
@pytest.mark.asyncio
async def test_gen_pred_table_graph_context_rejects_incomplete_input(
    monkeypatch,
    input_state,
    message,
):
    gen_job_id = GeneratePredictionTableJobID.make_random_id()
    project_id = ProjectID.generate()
    if input_state == 'missing':
        load_spy = AsyncMock(side_effect=FileNotFoundError())
    elif input_state == 'empty':
        load_spy = AsyncMock(return_value=None)
    else:
        load_spy = AsyncMock(return_value=SimpleNamespace(
            graph_data_compute_input=(None if input_state == 'graph-key' else
                                      _compute_input('NA-forked-compute-key')),
            graph_version_info=(
                None if input_state == 'version-info' else SimpleNamespace(
                    graph_data_version='saved-version')),
            config=SimpleNamespace(graph_data_version='saved-version'),
        ))
    monkeypatch.setattr(batch_prediction_job_logic, "load_json_async", load_spy)
    runtime = _gen_context_runtime(gen_job_id, project_id)

    with pytest.raises(ValueError, match=message):
        await runtime.logic._gen_pred_table_graph_context(
            gen_job_id, project_id=project_id)
    load_spy.assert_awaited_once()


@pytest.mark.asyncio
async def test_gen_pred_table_graph_context_rejects_foreign_job_before_file_read(
):
    gen_job_id = GeneratePredictionTableJobID.make_random_id()
    project_id = ProjectID.generate()
    runtime = _gen_context_runtime(gen_job_id, project_id, rows=[])

    with pytest.raises(ValueError, match="job not found"):
        await runtime.logic._gen_pred_table_graph_context(
            gen_job_id, project_id=project_id)

    runtime.ml_jobs.get_generate_pred_table_jobs.assert_awaited_once_with(
        [gen_job_id], project_ids={project_id})
    runtime.ml_backend.get_job_home_dir.assert_not_awaited()


@pytest.mark.parametrize('load_result', [FileNotFoundError(), None])
@pytest.mark.asyncio
async def test_gen_pred_table_graph_context_falls_back_to_state_when_input_missing(
        monkeypatch, load_result):
    gen_job_id = GeneratePredictionTableJobID.make_random_id()
    project_id = ProjectID.generate()
    graph_id = GraphIDv2.compose(fpid=FPID('a' * 32))
    compute_input = _compute_input(str(graph_id))
    graph_version_info = GraphDataVersionInfo(
        graph_data_version='saved-version',
        table_data_versions={},
    )
    ml_backend = SimpleNamespace(get_job_home_dir=AsyncMock(
        return_value='s3://bucket/jobs/gen-home'))
    ml_jobs = SimpleNamespace(get_generate_pred_table_jobs=AsyncMock(
        return_value=[
            SimpleNamespace(
                project_id=project_id, config=SimpleNamespace(
                    graph_id=graph_id,
                    graph_data_version='saved-version',
                ))
        ]))
    graph_service = SimpleNamespace(get_graph_data_compute_input=AsyncMock(
        return_value=compute_input))
    state_services = SimpleNamespace(
        ml_jobs=ml_jobs,
        graphs=graph_service,
    )
    load_spy = (AsyncMock(side_effect=load_result) if isinstance(
        load_result, Exception) else AsyncMock(return_value=load_result))
    monkeypatch.setattr(batch_prediction_job_logic, 'load_json_async', load_spy)
    version_info_path = MagicMock(return_value='/cache/version_info')
    graph_cache = SimpleNamespace(version_info_path=version_info_path)
    monkeypatch.setattr(
        batch_prediction_job_logic,
        'get_active_cache',
        lambda: SimpleNamespace(graph_cache=graph_cache),
    )
    load_many = AsyncMock(return_value=[graph_version_info])
    monkeypatch.setattr(batch_prediction_job_logic, 'load_many_json',
                        SimpleNamespace(coroutine=load_many))

    logic = _logic(ml_backend, state_services=state_services)  # type: ignore

    result = await logic._gen_pred_table_graph_context(gen_job_id,
                                                       project_id=project_id)

    assert result.graph_data_compute_input is compute_input
    assert result.graph_version_info is graph_version_info
    assert result.graph_data_version == 'saved-version'
    load_spy.assert_awaited_once()
    ml_jobs.get_generate_pred_table_jobs.assert_awaited_once_with(
        [gen_job_id], project_ids={project_id})
    graph_service.get_graph_data_compute_input.assert_awaited_once_with(
        graph_id,
        project_id=project_id,
        user_authorization=None,
    )
    version_info_path.assert_called_once_with(graph_id, 'saved-version')
    load_many.assert_awaited_once_with(['/cache/version_info'],
                                       GraphDataVersionInfo)


@pytest.mark.asyncio
async def test_existing_model_can_score_new_generator_graph(monkeypatch):
    """The generator graph binds prediction data only; it does not require
    the already-trained model to have been trained on that same snapshot.

    Under TT-NG-only the gen job may still be running when BP is submitted
    (the SDK submits BP without waiting), so ``create_batch_prediction_job_v2``
    no longer eagerly reads the gen graph context. Binding the gen graph
    version is deferred to the BP workflow, which reads it off the finished
    gen job. Here we assert the parent leaves the gen graph fields None and
    does not bind ``config.graph_data_version`` eagerly."""
    training_job_id = TrainingJobID('trained-on-old-graph')
    gen_pred_table_job_id = GeneratePredictionTableJobID('new-graph-table')
    config = SimpleNamespace(
        graph_data_version=None,
        generate_pred_table_job_id=None,
        custom_pred_table_source=None,
    )
    request = SimpleNamespace(
        predict_options=SimpleNamespace(
            binary_classification_threshold=None,
            num_classes_to_return=None,
        ),
        only_embeddings=False,
        batch_prediction_workflow_id=None,
        to_config=MagicMock(return_value=config),
    )
    training_job = SimpleNamespace(
        id=training_job_id,
        execution=SimpleNamespace(status=JobStatus.DONE),
        project_id=_PROJECT_ID,
        # Deliberately a different model-training snapshot.
        graph_data_version='old-training-graph-version',
        generate_train_table_job_id=None,
        advanced_options=None,
    )
    ml_backend = SimpleNamespace(
        get_training_job=AsyncMock(return_value=training_job),
        start_batch_prediction=AsyncMock(),
    )
    logic = _logic(ml_backend)  # type: ignore
    logic._compute_bp_train_options = AsyncMock(return_value=(None, None))
    logic._resolve_bp_graph_id = AsyncMock(return_value=(GraphIDv2.compose(
        fpid=FPID('a' * 32)), None))
    logic._gen_pred_table_graph_context = AsyncMock()
    generated_input = object()
    logic.make_batch_prediction_job_input_v2 = AsyncMock(
        return_value=generated_input)
    logic._apply_bp_job_tags = AsyncMock()
    monkeypatch.setattr(batch_prediction_job_logic.DELETE_OLD_BP_JOBS,
                        'parse_value', lambda: False)

    await logic.create_batch_prediction_job_v2(
        SimpleNamespace(is_customer=False),
        request,
        SimpleNamespace(graph=SimpleNamespace(fpid=FPID('a' * 32))),
        FPID('b' * 32),
        TaskType.BINARY_CLASSIFICATION,
        MagicMock(),
        training_job_id,
        predictive_query_id=PredictiveQueryID('pquery'),
        generate_pred_table_job_id=gen_pred_table_job_id,
        project_id=_PROJECT_ID,
    )

    # The gen graph binding is deferred to the BP workflow: the parent does
    # not eagerly read the gen graph context and leaves the gen graph fields
    # unset. The BP workflow later reads them off the finished gen job.
    logic._gen_pred_table_graph_context.assert_not_awaited()
    make_kwargs = logic.make_batch_prediction_job_input_v2.await_args.kwargs
    assert make_kwargs['graph_data_compute_input'] is None
    assert make_kwargs['graph_version_info'] is None
    assert config.graph_data_version is None
    ml_backend.start_batch_prediction.assert_awaited_once_with(
        generated_input,
        job_source=None,
        project_id=_PROJECT_ID,
        user=None,
        batch_prediction_workflow_id=None,
    )
