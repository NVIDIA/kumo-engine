# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.types import PredictionArtifactType
from pydantic import ValidationError

from io_utils.json import from_json
from rest.config.file_type import FileType
from rest.config.jobs import (
    BatchPredictionOptions,
    BatchPredictionRequest,
    MetadataField,
)
from rest.config.prediction_output import (
    ADLSPredictionOutput,
    GCSPredictionOutput,
    PredictionStorageType,
    S3PredictionOutput,
)

_CONNECTOR_ID = f'connector_{"1" * 32}'


def test_s3_output_ftype_inference():
    # Should infer file type if missing
    text = """
    {
        "storage_type": "S3",
        "artifact_type": "PREDICTIONS",
        "connector_id": "connector_11111111111111111111111111111111",
        "file_path": "s3://bucket/path.parquet"
    }
    """
    output = from_json(text, S3PredictionOutput)
    assert output.validated_file_type == FileType.PARQUET

    # Should not infer file type if present
    text = """
    {
        "storage_type": "S3",
        "artifact_type": "PREDICTIONS",
        "connector_id": "connector_11111111111111111111111111111111",
        "file_path": "s3://bucket/path.parquet",
        "file_type": "CSV"
    }
    """
    output = from_json(text, S3PredictionOutput)
    assert output.validated_file_type == FileType.CSV
    assert output.is_partitioned is False
    assert output.directory_name == "s3://bucket/"
    output = output.require_partitioned()
    assert output.is_partitioned is False
    assert output.file_path == "s3://bucket/path.parquet"

    # Should default to parquet if no extension is provided
    text = """
    {
        "storage_type": "S3",
        "artifact_type": "PREDICTIONS",
        "connector_id": "connector_11111111111111111111111111111111",
        "file_path": "s3://bucket/path"
    }
    """
    output = from_json(text, S3PredictionOutput)
    assert output.validated_file_type == FileType.PARQUET

    # Should parse extension from directory
    text = """
    {
        "storage_type": "S3",
        "artifact_type": "PREDICTIONS",
        "connector_id": "connector_11111111111111111111111111111111",
        "file_path": "s3://bucket/path.csv/"
    }
    """
    output = from_json(text, S3PredictionOutput)
    assert output.validated_file_type == FileType.CSV
    assert output.is_partitioned is True
    assert output.directory_name == "s3://bucket/path.csv/"


@pytest.mark.parametrize('path', ['output.CSV', 'output.PARQUET/'])
def test_file_type_inference_is_case_insensitive(path):
    expected = FileType.CSV if path.endswith('CSV') else FileType.PARQUET
    assert FileType.parse_extension(path) == expected


@pytest.mark.parametrize('output', [
    S3PredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=_CONNECTOR_ID,
        file_path='s3://bucket/output.CSV',
    ),
    GCSPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=_CONNECTOR_ID,
        file_path='gs://bucket/output.PARQUET',
    ),
    ADLSPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=_CONNECTOR_ID,
        file_path=('abfss://data@account.dfs.core.windows.net/output.CSV'),
    ),
])
def test_explicit_uppercase_file_output_is_not_rewritten_as_directory(output):
    assert output.require_partitioned() is output


def test_bp_request_outputs():
    predict_options = BatchPredictionOptions()

    # 2 prediction outputs with different extra_fields should fail.
    o1 = S3PredictionOutput(
        storage_type=PredictionStorageType.S3,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=_CONNECTOR_ID,
        file_path="s3://random1/output",
        extra_fields=[MetadataField.ANCHOR_TIMESTAMP],
    )
    o2 = S3PredictionOutput(
        storage_type=PredictionStorageType.S3,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=_CONNECTOR_ID,
        file_path="s3://random1/output",
    )
    with pytest.raises(ValidationError):
        BatchPredictionRequest(
            predict_options=predict_options,
            outputs=[o1, o2],
        )

    # If one of the output is EMBEDDINGS, it should not fail.
    o2 = S3PredictionOutput(
        storage_type=PredictionStorageType.S3,
        artifact_type=PredictionArtifactType.EMBEDDINGS,
        connector_id=_CONNECTOR_ID,
        file_path="s3://random1/output",
    )
    BatchPredictionRequest(
        predict_options=predict_options,
        outputs=[o1, o2],
    )

    # Both outputs having the same extra_fields should not fail.
    o2 = S3PredictionOutput(
        storage_type=PredictionStorageType.S3,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=_CONNECTOR_ID,
        file_path="s3://random1/output",
        extra_fields=[MetadataField.ANCHOR_TIMESTAMP],
    )
    BatchPredictionRequest(
        predict_options=predict_options,
        outputs=[o1, o2],
    )
