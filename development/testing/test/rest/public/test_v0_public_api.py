# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import kumoapi.jobs as api
import pytest
from kumoapi.jobs import PredictionArtifactType, PredictionStorageType

from rest.public.routers.prediction import _require_partitioned_s3_output

_CONNECTOR_ID = 'connector_' + '1' * 32


def test_require_partitioned_s3_output_adds_trailing_slash():
    """Test that S3 output paths without trailing slashes get one added."""
    # Case 1: Path without trailing slash should get one added
    output = api.S3PredictionOutput(
        storage_type=PredictionStorageType.S3,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        file_path="s3://bucket/path/to/predictions",
        connector_id=_CONNECTOR_ID,
    )
    result = _require_partitioned_s3_output([output])
    assert len(result) == 1
    assert isinstance(result[0], api.S3PredictionOutput)
    assert result[0].file_path == "s3://bucket/path/to/predictions/"
    assert result[0].file_path.endswith('/')

    # Case 2: Path with trailing slash should remain unchanged
    output_with_slash = api.S3PredictionOutput(
        storage_type=PredictionStorageType.S3,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        file_path="s3://bucket/path/to/embeddings/",
        connector_id=_CONNECTOR_ID,
    )
    result_with_slash = _require_partitioned_s3_output([output_with_slash])
    assert len(result_with_slash) == 1
    assert isinstance(result_with_slash[0], api.S3PredictionOutput)
    assert (result_with_slash[0].file_path == "s3://bucket/path/to/embeddings/")
    assert result_with_slash[0].file_path.endswith('/')


@pytest.mark.parametrize('output,coordinate', [
    (api.S3PredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        file_path='s3://bucket/output.PARQUET',
        connector_id=_CONNECTOR_ID,
    ), 's3://bucket/output.PARQUET'),
    (api.GCSPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        file_path='gs://bucket/output.CSV',
        connector_id=_CONNECTOR_ID,
    ), 'gs://bucket/output.CSV'),
    (api.ADLSPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        file_path=('abfss://data@account.dfs.core.windows.net/'
                   'output.PARQUET'),
        connector_id=_CONNECTOR_ID,
    ), 'abfss://data@account.dfs.core.windows.net/output.PARQUET'),
])
def test_object_output_preserves_uppercase_file_extension(output, coordinate):
    result = _require_partitioned_s3_output([output])[0]
    assert result.file_path == coordinate
