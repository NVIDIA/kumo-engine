# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import kumoapi.jobs as api
import pytest
from fastapi import FastAPI, HTTPException
from kumo_core.types import TaskType
from pydantic import TypeAdapter, ValidationError

from io_utils.json import to_json_dict
from rest.config import prediction_output
from rest.config.batch_prediction_job import (
    BatchPredictionJobConfig,
    BatchPredictionJobOutput,
    BatchPredictionJobResponse,
)
from rest.config.file_type import FileType
from rest.public.routers.prediction import (
    PredictionRouterV1,
    _make_prediction_job_resource,
)
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    GraphLink,
    JobEventLogEntry,
    JobFailureInfo,
    JobStatus,
    PredictiveQueryID,
    PredictiveQueryLink,
    ProjectID,
    TrainingJobID,
)
from statedb.config.predictive_query import FilterOverrideArgs
from statedb.config.timestamp import ResourceTimestamps, now_utc

_PROJECT_ID = ProjectID(f'project_{"1" * 32}')
_CONNECTOR_ID = f'connector_{"2" * 32}'
_JOB_ID = BatchPredictionJobID('batch-prediction-job')
_EXTRA_FIELDS = [api.MetadataField.JOB_TIMESTAMP]


def _output(output_type, **kwargs):
    return output_type(
        artifact_type=api.PredictionArtifactType.PREDICTIONS,
        connector_id=_CONNECTOR_ID,
        extra_fields=_EXTRA_FIELDS,
        **kwargs,
    )


_OUTPUT_CASES = {
    'Snowflake':
    dict(table_name='PUBLIC.PREDICTIONS', staging_table_name='INTERNAL.STAGING',
         write_mode=api.WriteMode.APPEND),
    'S3':
    dict(file_path='predictions/', file_type=FileType.CSV),
    'GCS':
    dict(file_path='gs://bucket/predictions/', file_type=FileType.CSV),
    'ADLS':
    dict(
        file_path='abfss://container@account.dfs.core.windows.net/predictions/',
        file_type=FileType.CSV),
    'Databricks':
    dict(table_name='catalog.schema.predictions'),
    'BigQuery':
    dict(storage_type=api.PredictionStorageType.BIGQUERY,
         table_name='dataset.predictions', staging_table_name='dataset.staging',
         write_mode=api.WriteMode.APPEND),
}


def _canonical_public_output_payload(output, public_type) -> dict:
    """Validate the internal payload through the exact public output type."""
    internal = to_json_dict(output)
    public_fields = {field.name for field in dataclasses.fields(public_type)}
    projected = TypeAdapter(public_type).validate_python({
        name: value
        for name, value in internal.items() if name in public_fields
    })
    return to_json_dict(projected)


def test_public_prediction_route_uses_exact_api_request_contract() -> None:
    auth = SimpleNamespace()
    router = PredictionRouterV1(
        lambda: None,
        MagicMock(),
        auth,
        MagicMock(),
    )
    app = FastAPI()
    app.include_router(router.router)

    request_schema = app.openapi()['paths']['/prediction_jobs']['post'][
        'requestBody']['content']['application/json']['schema']
    assert request_schema == {
        '$ref': '#/components/schemas/BatchPredictionRequest'
    }
    properties = app.openapi(
    )['components']['schemas']['BatchPredictionRequest']['properties']
    assert set(properties) == {
        field.name
        for field in dataclasses.fields(api.BatchPredictionRequest)
    }
    assert 'batch_prediction_workflow_id' not in properties

    with pytest.raises(ValidationError, match='batch_prediction_workflow_id'):
        TypeAdapter(api.BatchPredictionRequest).validate_python({
            'job_tags': {},
            'model_training_job_id':
            'training-job',
            'predict_options': {},
            'batch_prediction_workflow_id':
            'internal-workflow-id',
        })


@pytest.mark.asyncio
async def test_prediction_download_route_returns_internal_partitions() -> None:
    response = api.GetPredictionsDfUrlResponse(
        prediction_partitions=['/customer-facing/part-0.parquet'])
    router = PredictionRouterV1.__new__(PredictionRouterV1)
    router._driver = SimpleNamespace(get_presigned_prediction_urls=AsyncMock(
        return_value=response))

    assert await router.get_prediction_df_urls(_JOB_ID) == response


@pytest.mark.asyncio
async def test_prediction_download_route_returns_not_found_when_absent(
) -> None:
    router = PredictionRouterV1.__new__(PredictionRouterV1)
    router._driver = SimpleNamespace(get_presigned_prediction_urls=AsyncMock(
        return_value=None))

    with pytest.raises(HTTPException) as exc_info:
        await router.get_prediction_df_urls(_JOB_ID)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_prediction_delete_route_second_delete_returns_404() -> None:
    ctx = object()
    delete = AsyncMock(side_effect=[True, False])
    router = PredictionRouterV1.__new__(PredictionRouterV1)
    router._driver = SimpleNamespace(delete_batch_prediction=delete)

    first = await router.delete_prediction_job(_JOB_ID, ctx)
    with pytest.raises(HTTPException) as exc_info:
        await router.delete_prediction_job(_JOB_ID, ctx)

    assert dataclasses.asdict(first) == {}
    assert exc_info.value.status_code == 404
    assert str(_JOB_ID) in exc_info.value.detail
    assert [args.args for args in delete.await_args_list] == [
        (ctx, _JOB_ID),
        (ctx, _JOB_ID),
    ]


def _job(config: BatchPredictionJobConfig) -> BatchPredictionJobResponse:
    return BatchPredictionJobResponse(
        id=_JOB_ID,
        project_id=_PROJECT_ID,
        config=config,
        timestamps=ResourceTimestamps(
            created_at=now_utc(),
            updated_at=now_utc(),
        ),
        predictive_query_link=PredictiveQueryLink(
            id=PredictiveQueryID('pquery'),
            name=PredictiveQueryID('pquery'),
        ),
        execution=BatchPredictionJobOutput(
            status=JobStatus.RUNNING,
            log=JobEventLogEntry.not_started(),
        ),
        graph_link=GraphLink(id='graph', name='graph'),
    )


def test_prediction_resource_projects_only_public_prediction_options() -> None:
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('pquery'),
        training_job_id=TrainingJobID('training-job'),
        task_type=TaskType.BINARY_CLASSIFICATION,
        binary_classification_threshold=0.5,
        num_classes_to_return=7,
        prediction_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        filter_override_args=FilterOverrideArgs(
            entity_filter_override='customers.active = true'),
        num_parallel_workers_limit=3,
        generate_pred_table_job_id=GeneratePredictionTableJobID.as_child_of(
            _JOB_ID),
        graph_id=None,
        explanations=True,
    )
    resource = _make_prediction_job_resource(_job(config))
    payload = to_json_dict(resource)

    assert resource.config.predict_options.num_workers == 3
    assert resource.config.prediction_time == datetime(2026, 1, 1,
                                                       tzinfo=timezone.utc)
    assert resource.config.explanations is True
    assert resource.config.pred_table_job_id is None
    assert payload['config']['predict_options'] == {
        'binary_classification_threshold': 0.5,
        'num_classes_to_return': 7,
        'num_workers': 3,
    }
    assert payload['config']['prediction_time'] == '2026-01-01T00:00:00Z'
    assert payload['config']['explanations'] is True
    assert payload['config']['pred_table_job_id'] is None
    assert 'filter_override_args' not in payload['config']
    assert 'customers.active = true' not in repr(payload)
    assert 'predict_options' not in payload
    assert 'outputs' not in payload


def test_prediction_resource_exposes_message_without_internal_stack() -> None:
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('pquery'),
        training_job_id=TrainingJobID('training-job'),
        task_type=TaskType.REGRESSION,
        generate_pred_table_job_id=GeneratePredictionTableJobID.as_child_of(
            _JOB_ID),
        graph_id=None,
    )
    job = _job(config)
    job.execution = BatchPredictionJobOutput(
        status=JobStatus.FAILED,
        log=JobEventLogEntry.failed(),
        failure_info=JobFailureInfo(
            message=('Batch prediction failed. '
                     'Custom prediction table is missing ENTITY1.'),
            stacktrace='/private/backend/path.py:123 internal traceback',
        ),
    )

    resource = _make_prediction_job_resource(job)
    validation = resource.job_status_report.validation_response

    assert validation is not None
    assert validation.errors[0].title == 'Batch prediction failed'
    assert validation.errors[0].message == (
        'Custom prediction table is missing ENTITY1.')
    assert '/private/backend/path.py' not in repr(resource)


def test_prediction_resource_preserves_explicit_prediction_table_job() -> None:
    pred_table_job_id = GeneratePredictionTableJobID(
        'explicit-prediction-table')
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('pquery'),
        training_job_id=TrainingJobID('training-job'),
        task_type=TaskType.REGRESSION,
        prediction_time=None,
        generate_pred_table_job_id=pred_table_job_id,
        graph_id=None,
    )

    resource = _make_prediction_job_resource(_job(config))
    payload = to_json_dict(resource)

    assert resource.config.prediction_time is None
    assert resource.config.pred_table_job_id == pred_table_job_id
    assert payload['config']['prediction_time'] is None
    assert payload['config']['pred_table_job_id'] == pred_table_job_id


def test_prediction_resource_rejects_prediction_time_with_explicit_table_job(
) -> None:
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('pquery'),
        training_job_id=TrainingJobID('training-job'),
        task_type=TaskType.REGRESSION,
        prediction_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        generate_pred_table_job_id=GeneratePredictionTableJobID(
            'explicit-prediction-table'),
        graph_id=None,
    )

    with pytest.raises(ValueError, match='non-child prediction-table job'):
        _make_prediction_job_resource(_job(config))


@pytest.mark.parametrize(('provider', 'output_kwargs'), _OUTPUT_CASES.items())
def test_prediction_resource_projects_persisted_outputs(
    provider,
    output_kwargs,
) -> None:
    internal_type = getattr(prediction_output, f'{provider}PredictionOutput')
    public_type = getattr(api, f'{provider}PredictionOutput')
    output = _output(internal_type, **output_kwargs)
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('pquery'),
        training_job_id=TrainingJobID('training-job'),
        task_type=TaskType.REGRESSION,
        outputs=[output],
        graph_id=None,
    )

    resource = _make_prediction_job_resource(_job(config))
    payload = to_json_dict(resource)

    assert type(resource.config.outputs[0]) is public_type
    assert payload['config']['outputs'] == [
        _canonical_public_output_payload(output, public_type)
    ]


def test_prediction_resource_projects_local_output() -> None:
    output = prediction_output.LocalPredictionOutput(
        artifact_type=api.PredictionArtifactType.EMBEDDINGS, )
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('pquery'),
        training_job_id=TrainingJobID('training-job'),
        task_type=TaskType.REGRESSION,
        outputs=[output],
        graph_id=None,
    )

    resource = _make_prediction_job_resource(_job(config))

    assert type(resource.config.outputs[0]) is api.LocalPredictionOutput
    assert resource.config.outputs[0].artifact_type == (
        api.PredictionArtifactType.EMBEDDINGS)
