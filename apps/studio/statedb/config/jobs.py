# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

import kumo_core.types
import kumoapi.jobs as api
from kumoapi.source_table import FileType
from pydantic import Field, field_validator
from typing_extensions import Annotated

from io_utils import kumo_dataclass

PredictionStorageType = api.PredictionStorageType
MetadataField = api.MetadataField
WriteMode = api.WriteMode


@kumo_dataclass
class BigQueryPredictionOutput(api.BigQueryPredictionOutput):
    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return kumo_core.types.PredictionArtifactType(v)


@kumo_dataclass
class SnowflakePredictionOutput(api.SnowflakePredictionOutput):
    # Persisted Kumo extension over the generated API output shape.
    staging_table_name: Optional[str] = None

    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return kumo_core.types.PredictionArtifactType(v)


@kumo_dataclass
class DatabricksPredictionOutput(api.DatabricksPredictionOutput):
    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return kumo_core.types.PredictionArtifactType(v)


@kumo_dataclass
class S3PredictionOutput(api.S3PredictionOutput):
    file_type: Optional[FileType] = None

    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return kumo_core.types.PredictionArtifactType(v)


@kumo_dataclass
class GCSPredictionOutput(api.GCSPredictionOutput):
    file_type: Optional[FileType] = None

    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return kumo_core.types.PredictionArtifactType(v)


@kumo_dataclass
class ADLSPredictionOutput(api.ADLSPredictionOutput):
    file_type: Optional[FileType] = None

    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return kumo_core.types.PredictionArtifactType(v)


@kumo_dataclass
class LocalPredictionOutput(api.LocalPredictionOutput):
    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return kumo_core.types.PredictionArtifactType(v)


PredictionOutputConfig = Annotated[Union[BigQueryPredictionOutput,
                                         SnowflakePredictionOutput,
                                         DatabricksPredictionOutput,
                                         S3PredictionOutput,
                                         GCSPredictionOutput,
                                         ADLSPredictionOutput,
                                         LocalPredictionOutput],
                                   Field(discriminator='storage_type')]


def validate_prediction_outputs(outputs: List[PredictionOutputConfig]) -> None:
    # One producer artifact is shared by every destination of that type, so
    # all destinations for the same artifact must request the same schema.
    extra_fields_by_artifact: dict[kumo_core.types.PredictionArtifactType,
                                   set] = {}
    for output in outputs:
        extra_fields = set(output.extra_fields)
        expected = extra_fields_by_artifact.setdefault(output.artifact_type,
                                                       extra_fields)
        if extra_fields != expected:
            raise ValueError(
                "All outputs for the same artifact type must have the same "
                f"set of extra_fields. Got: {outputs}")
