# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
from pydantic import ValidationError

from kumoapi.jobs import (
    ADLSModelArtifactOutput,
    DatabricksModelArtifactOutput,
    GCSModelArtifactOutput,
    ModelArtifactOutput,
    ModelOutputConfig,
    S3ModelArtifactOutput,
    SnowflakeModelArtifactOutput,
)
from kumoapi.json_serde import from_json, to_json, to_json_dict

_MODEL_ARTIFACT_OUTPUT_CASES = (
    (
        {
            'output_kind': 's3',
            'connector_id': 's3',
            'path': 'models/deep/job-1/',
        },
        S3ModelArtifactOutput,
        {
            'destination': {
                'mode': 'relative',
                'path': 'ignored'
            }
        },
    ),
    (
        {
            'output_kind': 'gcs',
            'connector_id': 'gcs',
            'path': 'models/deep/job-1/',
        },
        GCSModelArtifactOutput,
        {
            'destination': {
                'mode': 'relative',
                'path': 'ignored'
            }
        },
    ),
    (
        {
            'output_kind': 'adls',
            'connector_id': 'adls',
            'path': 'models/deep/job-1/',
        },
        ADLSModelArtifactOutput,
        {
            'destination': {
                'mode': 'relative',
                'path': 'ignored'
            }
        },
    ),
    (
        {
            'output_kind': 'snowflake',
            'connector_id': 'snowflake',
            'model_name': 'CHURN_GNN',
            'version_name': 'V3',
            'comment': 'production candidate',
        },
        SnowflakeModelArtifactOutput,
        {
            'path': 'ignored'
        },
    ),
    (
        {
            'output_kind': 'databricks',
            'connector_id': 'databricks',
            'model_name': 'catalog.schema.churn_gnn',
            'version_name': None,
            'comment': None,
        },
        DatabricksModelArtifactOutput,
        {
            'path': 'ignored'
        },
    ),
)


@pytest.mark.parametrize(
    'payload,expected_type,cross_kind_fields',
    _MODEL_ARTIFACT_OUTPUT_CASES,
)
def test_model_artifact_output_discriminated_union_is_strict(
        payload, expected_type, cross_kind_fields):
    output = from_json(payload, ModelArtifactOutput)
    assert isinstance(output, expected_type)
    ignored_field = next(iter(cross_kind_fields))
    with pytest.raises(ValidationError, match=ignored_field):
        from_json({**payload, **cross_kind_fields}, ModelArtifactOutput)


@pytest.mark.parametrize('output', [
    S3ModelArtifactOutput(connector_id='s3',
                          path='s3://model-artifacts/models/job-1/'),
    GCSModelArtifactOutput(connector_id='gcs',
                           path='gs://bucket/models/job-1'),
    ADLSModelArtifactOutput(
        connector_id='adls',
        path='abfss://models@companylake.dfs.core.windows.net/job-1'),
])
def test_model_artifact_full_path_roundtrip_keeps_wire_shape(output):
    assert from_json(to_json(output), ModelArtifactOutput) == output
    payload = to_json_dict(output)
    assert payload['path'] == output.path
    assert 'destination' not in payload


@pytest.mark.parametrize('output', [
    SnowflakeModelArtifactOutput(
        connector_id='snowflake',
        model_name='CHURN_GNN',
        version_name='V3',
        comment='production candidate',
    ),
    DatabricksModelArtifactOutput(
        connector_id='databricks',
        model_name='catalog.schema.churn_gnn',
    ),
])
def test_model_registry_artifact_roundtrip_keeps_wire_shape(output):
    assert from_json(to_json(output), ModelArtifactOutput) == output
    payload = to_json_dict(output)
    assert payload['model_name'] == output.model_name
    assert 'path' not in payload


@pytest.mark.parametrize('output_kind', ['snowflake', 'databricks'])
@pytest.mark.parametrize(
    'invalid_fields',
    [
        {
            'connector_id': ''
        },
        {
            'model_name': ''
        },
        {
            'model_name': ' padded'
        },
        {
            'version_name': ''
        },
        {
            'version_name': 'V3 '
        },
    ],
)
def test_model_registry_artifact_rejects_invalid_names(output_kind,
                                                       invalid_fields):
    payload = {
        'output_kind': output_kind,
        'connector_id': 'registry',
        'model_name': 'CHURN_GNN',
        **invalid_fields,
    }
    with pytest.raises((ValidationError, ValueError, TypeError)):
        from_json(payload, ModelArtifactOutput)


def test_s3_model_artifact_rejects_deprecated_destination_shape():
    payload = {
        'output_kind': 's3',
        'connector_id': 's3',
        'destination': {
            'mode': 'uri',
            'uri': 's3://model-artifacts/models/job-1/',
        },
    }
    with pytest.raises((ValidationError, TypeError)):
        from_json(payload, ModelArtifactOutput)


@pytest.mark.parametrize(
    'output',
    [
        {
            'output_kind': 'gcs',
            'connector_id': 'gcs',
            'path': '../job-1',
        },
        {
            'output_kind': 's3',
            'connector_id': 's3',
            'output_path': 's3://bucket/job-1',
        },
        {
            'output_kind': 's3',
            'connector_id': '',
            'destination': {
                'mode': 'relative',
                'path': 'job-1',
            },
        },
    ],
)
def test_model_artifact_output_rejects_raw_unsafe_or_incomplete(output):
    with pytest.raises((ValidationError, ValueError, TypeError)):
        from_json(output, ModelArtifactOutput)


def test_model_output_rejects_removed_raw_output_path():
    with pytest.raises((ValidationError, ValueError, TypeError)):
        from_json(
            {
                'training_job_id': 'job-1',
                'output_path': 's3://bucket/job-1',
            },
            ModelOutputConfig,
        )
