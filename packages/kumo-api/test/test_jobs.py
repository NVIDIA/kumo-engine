# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from kumoapi.jobs import (
    ADLSPredictionOutput,
    ADLSPredictionTableSource,
    ArtifactExportRequest,
    BatchPredictionJobResource,
    BatchPredictionOptions,
    BatchPredictionRequest,
    DatabricksPredictionOutput,
    DatabricksPredictionTableSource,
    FileTrainingTableOutputConfig,
    GCSPredictionOutput,
    GCSPredictionTableSource,
    GeneratePredictionTableRequest,
    GenerateTrainTableRequest,
    JobResourceBase,
    JobStatus,
    JobStatusReport,
    LocalPredictionOutput,
    NamedTrainingTableOutputConfig,
    PredictionArtifactType,
    S3PredictionOutput,
    S3PredictionTableSource,
    TrainerProgress,
    TrainingJobRequest,
    TrainingTableOutputConfig,
    WriteMode,
)
from kumoapi.json_serde import from_json, to_json, to_json_dict
from kumoapi.model_plan import (
    PredictionTableGenerationPlan,
    TrainingTableGenerationPlan,
)
from kumoapi.source_table import (
    ADLSRelativeSource,
    GCSURISource,
    S3RelativeSource,
)


def test_artifact_export_request_requires_canonical_idempotency_key():
    output = LocalPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS)
    request = ArtifactExportRequest(
        request_id='8f14e45f-ea5e-4a40-a79b-3f55a4df7c4d',
        job_id='job',
        prediction_output=output,
    )
    missing_request_id = json.loads(to_json(request))
    del missing_request_id['request_id']
    with pytest.raises((ValidationError, TypeError)):
        from_json(json.dumps(missing_request_id), ArtifactExportRequest)
    with pytest.raises(ValidationError, match='canonical UUID'):
        ArtifactExportRequest(request_id='NOT-A-UUID', job_id='job',
                              prediction_output=output)

    decoded = from_json(to_json(request), ArtifactExportRequest)
    assert decoded.request_id == request.request_id

    with pytest.raises(ValidationError, match='unknown_digest_input'):
        from_json({
            **to_json_dict(request),
            'unknown_digest_input': 'typo',
        }, ArtifactExportRequest)


@pytest.mark.parametrize(
    'job_request',
    [
        GenerateTrainTableRequest(
            job_tags={},
            pquery_id='pquery',
            plan=TrainingTableGenerationPlan(),
            graph_snapshot_id=None,
        ),
        GeneratePredictionTableRequest(
            job_tags={},
            pquery_id='pquery',
            plan=PredictionTableGenerationPlan(),
            graph_snapshot_id=None,
        ),
        TrainingJobRequest(job_tags={}, pquery_id='pquery'),
        BatchPredictionRequest(
            job_tags={},
            model_training_job_id='training-job',
            predict_options=BatchPredictionOptions(),
        ),
    ],
)
def test_job_requests_reject_unknown_fields(job_request):
    payload = to_json_dict(job_request)
    payload['optional_field_typo'] = {'value': True}

    with pytest.raises(ValidationError, match='optional_field_typo'):
        from_json(payload, type(job_request))


def test_trainer_progress_roundtrips_actual_accelerator():
    progress = TrainerProgress(accelerator='cuda')
    assert from_json(to_json(progress), TrainerProgress) == progress


def test_custom_prediction_table_source_roundtrips_without_raw_path():
    request = BatchPredictionRequest(
        job_tags={},
        model_training_job_id='training-job',
        predict_options=BatchPredictionOptions(),
        pred_table_source=S3PredictionTableSource(
            connector_id='connector-s3',
            source=S3RelativeSource(path='custom/deep/prediction.parquet'),
        ),
    )

    restored = from_json(to_json(request), BatchPredictionRequest)
    assert restored == request
    assert restored.pred_table_source is not None
    assert restored.pred_table_source.connector_id == 'connector-s3'
    assert not hasattr(restored, 'pred_table_path')


@pytest.mark.parametrize(
    'prediction_source',
    [
        GCSPredictionTableSource(
            connector_id='connector-gcs-any',
            source=GCSURISource(
                uri='gs://company-data/custom/prediction.parquet'),
        ),
        ADLSPredictionTableSource(
            connector_id='connector-adls',
            source=ADLSRelativeSource(path='custom/prediction.parquet'),
        ),
    ],
)
def test_generalized_custom_prediction_source_union_roundtrips(
        prediction_source):
    request = BatchPredictionRequest(
        job_tags={},
        model_training_job_id='training-job',
        predict_options=BatchPredictionOptions(),
        pred_table_source=prediction_source,
    )
    payload = to_json_dict(request)
    assert payload['pred_table_source']['storage_type'] in {'GCS', 'ADLS'}
    assert from_json(payload, BatchPredictionRequest) == request


@pytest.mark.parametrize(
    'prediction_source',
    [
        DatabricksPredictionTableSource(
            connector_id='connector-databricks',
            table='catalog.schema.prediction_input',
        ),
        DatabricksPredictionTableSource(
            connector_id='connector-databricks',
            sql='SELECT entity_id, event_time FROM catalog.schema.events',
        ),
    ],
)
def test_databricks_custom_prediction_source_roundtrips(prediction_source):
    request = BatchPredictionRequest(
        job_tags={},
        model_training_job_id='training-job',
        predict_options=BatchPredictionOptions(),
        pred_table_source=prediction_source,
    )
    payload = to_json_dict(request)
    assert payload['pred_table_source']['storage_type'] == 'DATABRICKS'
    assert from_json(payload, BatchPredictionRequest) == request


@pytest.mark.parametrize('write_mode', list(WriteMode))
def test_databricks_prediction_output_write_mode_roundtrips(write_mode):
    output = DatabricksPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id='connector-databricks',
        table_name='schema__kumo__predictions',
        write_mode=write_mode,
    )

    payload = to_json_dict(output)

    assert payload['write_mode'] == write_mode.value
    assert from_json(payload, DatabricksPredictionOutput) == output


@pytest.mark.parametrize(
    'kwargs',
    [
        {},
        {
            'table': 'table',
            'sql': 'SELECT 1'
        },
        {
            'table': ''
        },
        {
            'sql': ' SELECT 1'
        },
        {
            'connector_id': ' connector',
            'table': 'table'
        },
    ],
)
def test_databricks_custom_prediction_source_validates_coordinates(kwargs):
    kwargs.setdefault('connector_id', 'connector-databricks')
    with pytest.raises((ValueError, ValidationError)):
        DatabricksPredictionTableSource(**kwargs)


def test_legacy_custom_prediction_source_without_provider_tag_remains_s3():
    payload = {
        'job_tags': {},
        'model_training_job_id': 'training-job',
        'predict_options': {},
        'pred_table_source': {
            'connector_id': 'connector-s3',
            'source': {
                'mode': 'relative',
                'path': 'custom/prediction.parquet',
            },
        },
    }
    original_source = dict(payload['pred_table_source'])
    restored = from_json(payload, BatchPredictionRequest)
    assert isinstance(restored.pred_table_source, S3PredictionTableSource)
    assert to_json_dict(restored)['pred_table_source']['storage_type'] == 'S3'
    assert payload['pred_table_source'] == original_source


@pytest.mark.parametrize(
    'storage_type,uri',
    [
        ('GCS', 's3://company-data/custom.parquet'),
        ('ADLS', 'gs://company-data/custom.parquet'),
        ('S3', ('abfss://curated@companylake.dfs.core.windows.net/'
                'custom.parquet')),
    ],
)
def test_custom_prediction_source_rejects_provider_mismatch(storage_type, uri):
    with pytest.raises(ValidationError):
        from_json(
            {
                'job_tags': {},
                'model_training_job_id': 'training-job',
                'predict_options': {},
                'pred_table_source': {
                    'storage_type': storage_type,
                    'connector_id': 'connector',
                    'source': {
                        'mode': 'uri',
                        'uri': uri,
                    },
                },
            }, BatchPredictionRequest)


@pytest.mark.parametrize(
    'payload',
    [
        {
            'pred_table_job_id': 'prediction-job',
            'pred_table_source': {
                'connector_id': 'connector-s3',
                'source': {
                    'mode': 'relative',
                    'path': 'custom.parquet',
                },
            },
        },
        {
            'pred_table_source': {
                'connector_id': 'connector-s3',
                'source': {
                    'mode': 'relative',
                    'path': 'custom.parquet',
                },
            },
            'prediction_time': '2026-07-06T12:00:00Z',
        },
        {
            'pred_table_job_id': 'prediction-job',
            'prediction_time': '2026-07-06T12:00:00Z',
        },
        {
            'pred_table_job_id': 'prediction-job',
            'pred_table_source': {
                'connector_id': 'connector-s3',
                'source': {
                    'mode': 'relative',
                    'path': 'custom.parquet',
                },
            },
            'prediction_time': '2026-07-06T12:00:00Z',
        },
    ],
)
def test_batch_prediction_rejects_raw_or_ambiguous_custom_source(payload):
    with pytest.raises((ValidationError, ValueError, TypeError)):
        from_json(
            {
                'job_tags': {},
                'model_training_job_id': 'training-job',
                'predict_options': {},
                **payload,
            },
            BatchPredictionRequest,
        )


def test_batch_prediction_rejects_removed_legacy_coordinate_field():
    with pytest.raises(ValidationError, match='pred_table_path'):
        from_json(
            {
                'job_tags': {},
                'model_training_job_id': 'training-job',
                'predict_options': {},
                'pred_table_path': 's3://bucket/legacy.parquet',
            },
            BatchPredictionRequest,
        )


def test_batch_prediction_uses_correct_post_init_hook():
    request_attributes = BatchPredictionRequest.__dict__
    assert '__post_init__' in request_attributes


@pytest.mark.parametrize(
    'output_type',
    [GCSPredictionOutput, ADLSPredictionOutput],
)
def test_object_store_prediction_output_requires_connector(output_type):
    payload = {
        'artifact_type': PredictionArtifactType.PREDICTIONS,
        'file_path': 'predictions/output',
    }

    with pytest.raises((ValidationError, TypeError)):
        from_json(payload, output_type)

    parsed = from_json({
        **payload, 'connector_id': 'connector-output'
    }, output_type)
    assert parsed.connector_id == 'connector-output'


@pytest.mark.parametrize(
    'output_type,file_path',
    [
        (GCSPredictionOutput,
         'gs://company-output/predictions/2026/output.parquet'),
        (ADLSPredictionOutput,
         ('abfss://predictions@companylake.dfs.core.usgovcloudapi.net/'
          '2026/output.parquet')),
    ],
)
def test_object_store_prediction_output_preserves_full_path_wire_shape(
        output_type, file_path):
    output = from_json(
        {
            'artifact_type': PredictionArtifactType.PREDICTIONS,
            'connector_id': 'connector-output',
            'file_path': file_path,
        }, output_type)

    assert output.file_path == file_path
    payload = to_json_dict(output)
    assert payload['file_path'] == file_path
    assert 'destination' not in payload
    assert from_json(to_json(output), output_type) == output


@pytest.mark.parametrize(
    'output_type,file_path',
    [
        (GCSPredictionOutput, 's3://company-output/predictions/output'),
        (GCSPredictionOutput, 'GS://company-output/predictions/output'),
        (GCSPredictionOutput, 'gs://company-output'),
        (GCSPredictionOutput,
         'gs://company-output/predictions/output?generation=1'),
        (ADLSPredictionOutput, 'gs://company-output/predictions/output'),
        (ADLSPredictionOutput,
         ('ABFSS://predictions@companylake.dfs.core.windows.net/'
          'output')),
        (ADLSPredictionOutput,
         'abfss://predictions@companylake.dfs.core.windows.net'),
        (ADLSPredictionOutput,
         ('abfss://predictions@companylake.dfs.core.windows.net/'
          'output?sig=secret')),
    ],
)
def test_object_store_prediction_output_rejects_noncanonical_full_uri(
        output_type, file_path):
    with pytest.raises((ValidationError, ValueError, UnicodeError)):
        from_json(
            {
                'artifact_type': PredictionArtifactType.PREDICTIONS,
                'connector_id': 'connector-output',
                'file_path': file_path,
            }, output_type)


def test_s3_prediction_output_relative_path_roundtrip():
    output = S3PredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id='connector-output',
        file_path='predictions/2026/output',
    )
    assert from_json(to_json(output), S3PredictionOutput) == output
    assert to_json_dict(output)['file_path'] == 'predictions/2026/output'
    assert 'destination' not in to_json_dict(output)


def test_s3_prediction_output_rejects_deprecated_destination_shape():
    payload = {
        'artifact_type': PredictionArtifactType.PREDICTIONS,
        'connector_id': 'connector-output',
        'destination': {
            'mode': 'uri',
            'uri': 's3://company-output/predictions/2026/output',
        },
    }
    with pytest.raises((ValidationError, TypeError)):
        from_json(payload, S3PredictionOutput)


def test_s3_prediction_output_schema_uses_string_file_path():
    from pydantic import TypeAdapter
    schema = TypeAdapter(S3PredictionOutput).json_schema()
    assert schema['properties']['file_path']['type'] == 'string'
    assert 'destination' not in schema['properties']


@pytest.mark.parametrize(
    'destination',
    [
        {
            'mode': 'relative',
            'path': 's3://bucket/output'
        },
        {
            'mode': 'relative',
            'path': '../output'
        },
        {
            'mode': 'uri',
            'uri': 's3://bucket/output?signature=x'
        },
        {
            'mode': 'uri',
            'uri': 'https://bucket.s3.amazonaws.com/output'
        },
        {
            'mode': 'uri',
            'path': 'wrong-field'
        },
    ],
)
def test_s3_prediction_output_rejects_unsafe_or_incomplete_destination(
        destination):
    with pytest.raises((ValidationError, ValueError)):
        from_json(
            {
                'artifact_type': PredictionArtifactType.PREDICTIONS,
                'connector_id': 'connector-output',
                'destination': destination,
            }, S3PredictionOutput)


@pytest.mark.parametrize(
    'destination',
    [
        {
            'mode': 'relative',
            'path': 'valid/output',
            'uri': 's3://company-output/typo',
        },
        {
            'mode': 'uri',
            'uri': 's3://company-output/valid',
            'path': 'typo',
        },
    ],
)
def test_s3_destination_rejects_unknown_coordinates(destination):
    with pytest.raises((ValidationError, ValueError)):
        from_json(
            {
                'artifact_type': PredictionArtifactType.PREDICTIONS,
                'connector_id': 'connector-output',
                'destination': destination,
            }, S3PredictionOutput)


def test_training_table_output_requires_connector_and_target_name():
    with pytest.raises((ValidationError, TypeError, ValueError)):
        from_json({
            'output_kind': 'named',
            'table_name': 'training'
        }, TrainingTableOutputConfig)
    with pytest.raises(ValueError, match='table_name'):
        NamedTrainingTableOutputConfig(connector_id='connector-output',
                                       table_name='')
    output = NamedTrainingTableOutputConfig(
        connector_id='connector-output',
        table_name='training',
    )
    assert not hasattr(output, 's3_path')


_TRAINING_TABLE_OUTPUT_CASES = (
    (
        {
            'output_kind': 'named',
            'connector_id': 'warehouse',
            'table_name': 'training',
        },
        NamedTrainingTableOutputConfig,
        {
            'path': 'ignored',
            'destination': {
                'mode': 'relative',
                'path': 'ignored'
            },
        },
    ),
    (
        {
            'output_kind': 'file',
            'connector_id': 'gcs',
            'path': 'deep/training',
        },
        FileTrainingTableOutputConfig,
        {
            'table_name': 'ignored',
            'destination': {
                'mode': 'relative',
                'path': 'ignored'
            },
        },
    ),
)


@pytest.mark.parametrize(
    'output,expected_type,cross_kind_fields',
    _TRAINING_TABLE_OUTPUT_CASES,
)
def test_training_table_output_discriminated_union_is_strict(
        output, expected_type, cross_kind_fields):
    parsed = from_json(output, TrainingTableOutputConfig)
    assert isinstance(parsed, expected_type)
    with pytest.raises(ValidationError):
        from_json({**output, **cross_kind_fields}, TrainingTableOutputConfig)


@pytest.mark.parametrize('path', [
    's3://bucket/deep/training',
    'gs://bucket/deep/training',
    'abfss://data@account.dfs.core.windows.net/deep/training',
])
def test_file_training_output_keeps_path_wire_shape(path):
    output = FileTrainingTableOutputConfig(connector_id='connector', path=path)
    payload = to_json_dict(output)
    assert payload['path'] == path
    assert 'destination' not in payload


@pytest.mark.parametrize(
    'payload',
    [
        {
            'output_kind': 's3',
            'connector_id': 's3',
            'table_name': 'legacy',
        },
        {
            'output_kind': 'named',
            'connector_id': 'warehouse',
            'destination': {
                'mode': 'relative',
                'path': 'training',
            },
        },
        {
            'output_kind': 'file',
            'connector_id': 'gcs',
            'path': 'https://bucket.example/training',
        },
        *[{
            'output_kind': 'file',
            'connector_id': 'gcs',
            'path': path,
        } for path in (
            '../escape',
            '.',
            'deep//training',
            'deep\\training',
            'deep/training?version=1',
            'deep/training#fragment',
            'deep/training?',
            'deep/training#',
            'deep/\x00training',
            '%2e%2e/escape',
            'deep%2Ftraining',
            'deep%5Ctraining',
            'deep/%00training',
            'deep%252Ftraining',
        )],
    ],
)
def test_training_table_output_rejects_missing_or_unsafe_required_fields(
        payload):
    with pytest.raises((ValidationError, ValueError, TypeError)):
        from_json(payload, TrainingTableOutputConfig)


def test_durable_job_resource_requires_project():
    now = datetime.now(timezone.utc)
    payload = {
        'job_id': 'job-1',
        'job_status_report': {
            'status': JobStatus.RUNNING,
            'tracking_url': 'https://example.test/jobs/job-1',
            'start_time': now.isoformat(),
        },
        'created_at': now.isoformat(),
        'tags': {},
    }

    with pytest.raises((ValidationError, TypeError)):
        from_json(payload, JobResourceBase)

    resource = JobResourceBase(
        job_id='job-1',
        job_status_report=JobStatusReport(
            status=JobStatus.RUNNING,
            tracking_url='https://example.test/jobs/job-1',
            start_time=now,
        ),
        created_at=now,
        tags={},
        project_id='project-1',
    )
    assert resource.project_id == 'project-1'

    payload.update(
        project_id='project-1', config={
            'job_tags': {},
            'model_training_job_id':
            'model-1',
            'predict_options': {},
            'outputs': [{
                'artifact_type': 'PREDICTIONS',
                'storage_type': 'GCS',
                'connector_id': 'gcs',
                'file_path': 'predictions/output',
            }],
        })
    restored = from_json(payload, BatchPredictionJobResource)
    output = restored.config.outputs[0]
    assert isinstance(output, GCSPredictionOutput)
    assert output.file_path == 'predictions/output'


def test_a_training_table_spec_may_describe_an_unmodified_table():
    """A supplied table needs no modification to be describable."""
    from kumoapi.train import TrainingTableSpec

    spec = TrainingTableSpec()

    assert spec.weight_col is None
    assert spec.split_col is None


def test_a_training_table_spec_names_the_column_it_splits_on():
    from kumoapi.json_serde import from_json, to_json
    from kumoapi.train import TrainingTableSpec

    spec = TrainingTableSpec(weight_col='weight', split_col='SPLIT')

    assert from_json(to_json(spec), TrainingTableSpec) == spec
