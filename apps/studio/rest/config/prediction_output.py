# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prediction-output config classes shared across job configs.

Extracted into a leaf module so both ``rest.config.jobs`` and
``rest.config.batch_prediction_job`` can depend on these output types without
importing one another. This breaks the jobs <-> batch_prediction_job import
cycle: the shared low-level output types now live below both modules.
"""
import copy
import os
from typing import Optional, Union

import kumoapi.data_source as data_source_api
from kumo_core.types import PredictionArtifactType
from pydantic import Field, field_validator
from typing_extensions import Annotated

from io_utils import fsutil, kumo_dataclass
from rest.config.connector import ConnectorResourceConfigDB
from rest.config.file_type import FileType
from statedb.config import jobs as serialized_jobs
from statedb.config.jobs import \
    ADLSPredictionOutput as SerializedADLSPredictionOutput
from statedb.config.jobs import \
    BigQueryPredictionOutput as SerializedBigQueryPredictionOutput
from statedb.config.jobs import \
    DatabricksPredictionOutput as SerializedDatabricksPredictionOutput
from statedb.config.jobs import \
    GCSPredictionOutput as SerializedGCSPredictionOutput
from statedb.config.jobs import \
    LocalPredictionOutput as SerializedLocalPredictionOutput
from statedb.config.jobs import \
    S3PredictionOutput as SerializedS3PredictionOutput
from statedb.config.jobs import \
    SnowflakePredictionOutput as SerializedSnowflakePredictionOutput

PredictionStorageType = serialized_jobs.PredictionStorageType


@kumo_dataclass
class BigQueryPredictionOutput(SerializedBigQueryPredictionOutput):
    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return PredictionArtifactType(v)


@kumo_dataclass
class SnowflakePredictionOutput(SerializedSnowflakePredictionOutput):
    # There are two ways to write to Snowflake:
    # 1. Write directly to the table_name. This is the default behavior
    # with staging_table_name=None. The output is *APPENDED* to the table_name.
    # It's a safe way to not overwrite the existing data in the table_name.
    # 2. Write to a staging table and then copy the staging table to the
    # table_name. After the staging table is written, the table_name will be
    # *OVERWRITTEN* with the staging table. This is done by replacing the
    # table_name with the staging table in an atomic operation. This is
    # recommended when the table_name already exists and we want to overwrite
    # the existing data in the table_name while ensuring that the table_name
    # is never corrupted.
    # Note that if the table_name already exists with right permissions, after
    # it is overwritten, the table_name will still have the same permissions.
    # Otherwise, the table_name will have the default permissions.
    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return PredictionArtifactType(v)


@kumo_dataclass
class DatabricksPredictionOutput(SerializedDatabricksPredictionOutput):
    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return PredictionArtifactType(v)


@kumo_dataclass
class S3PredictionOutput(SerializedS3PredictionOutput):
    # If not provided, then the file_path must either end in `.csv` or
    # `.parquet`, and we will parse the file type from there. Please use the
    # `validated_file_type` proper to access the parsed & validated file type.
    file_type: Optional[FileType] = None

    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return PredictionArtifactType(v)

    @property
    def validated_file_type(self) -> FileType:
        return _validated_prediction_file_type(self.file_type, self.file_path)

    @property
    def canonical_file_path(self) -> str:
        """Returns canonical string that can be used to detect conflicts
        between files/directories that have different names, but refer to the
        same s3 location.
        """
        return self.file_path.rstrip('/')

    @property
    def is_partitioned(self) -> bool:
        return self.file_path.endswith('/')

    def require_partitioned(self) -> 'S3PredictionOutput':
        """In the public API, we will require that is_partitioned == True, in
        order to reduce the chance of orchestrated jobs failing due to
        excessively large file sizes. is_partitioned False is still supported
        through the UI for convenience, where reliability is less of a concern.
        """
        return _require_partitioned_file_output(self,
                                                fsutil.is_s3(self.file_path))

    @property
    def directory_name(self) -> str:
        return _file_output_directory(self.file_path)


def _validated_prediction_file_type(file_type: Optional[FileType],
                                    file_path: str) -> FileType:
    if file_type is not None:
        return file_type
    try:
        return FileType.parse_extension(file_path)
    except ValueError:
        return FileType.PARQUET


def _require_partitioned_file_output(output, is_provider_path: bool):
    if output.is_partitioned or not is_provider_path:
        return output
    if output.file_path.casefold().endswith(('.parquet', '.csv')):
        return output
    partitioned = copy.copy(output)
    partitioned.file_path = f'{output.file_path}/'
    return partitioned


def _file_output_directory(file_path: str) -> str:
    return file_path if file_path.endswith(
        '/') else os.path.dirname(file_path) + '/'


@kumo_dataclass
class GCSPredictionOutput(SerializedGCSPredictionOutput):
    # If not provided, then the file_path must either end in `.csv` or
    # `.parquet`, and we will parse the file type from there. Please use the
    # `validated_file_type` property to access the parsed & validated file type.
    file_type: Optional[FileType] = None

    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return PredictionArtifactType(v)

    @property
    def validated_file_type(self) -> FileType:
        return _validated_prediction_file_type(self.file_type, self.file_path)

    @property
    def canonical_file_path(self) -> str:
        """Returns canonical string that can be used to detect conflicts
        between files/directories that have different names, but refer to the
        same GCS location.
        """
        return self.file_path.rstrip('/')

    @property
    def is_partitioned(self) -> bool:
        return self.file_path.endswith('/')

    def require_partitioned(self) -> 'GCSPredictionOutput':
        """In the public API, we will require that is_partitioned == True, in
        order to reduce the chance of orchestrated jobs failing due to
        excessively large file sizes. is_partitioned False is still supported
        through the UI for convenience, where reliability is less of a concern.
        """
        return _require_partitioned_file_output(self,
                                                fsutil.is_gcs(self.file_path))

    @property
    def directory_name(self) -> str:
        return _file_output_directory(self.file_path)


@kumo_dataclass
class ADLSPredictionOutput(SerializedADLSPredictionOutput):
    # If not provided, then the file_path must either end in `.csv` or
    # `.parquet`, and we will parse the file type from there. Please use the
    # `validated_file_type` property to access the parsed & validated file type.
    file_type: Optional[FileType] = None

    @field_validator('artifact_type')
    @classmethod
    def _coerce_artifact_type(cls, v):
        return PredictionArtifactType(v)

    @property
    def validated_file_type(self) -> FileType:
        return _validated_prediction_file_type(self.file_type, self.file_path)

    @property
    def canonical_file_path(self) -> str:
        """Returns canonical string for ADLS file output conflict detection."""
        return self.file_path.rstrip('/')

    @property
    def is_partitioned(self) -> bool:
        return self.file_path.endswith('/')

    def require_partitioned(self) -> 'ADLSPredictionOutput':
        """Require directory outputs for ADLS paths unless a file is explicit."""
        return _require_partitioned_file_output(
            self,
            fsutil.get_scheme(self.file_path) in ('abfss', 'abfs'))

    @property
    def directory_name(self) -> str:
        return _file_output_directory(self.file_path)


@kumo_dataclass
class LocalPredictionOutput(SerializedLocalPredictionOutput):
    pass


PredictionOutputConfig = Annotated[Union[BigQueryPredictionOutput,
                                         SnowflakePredictionOutput,
                                         DatabricksPredictionOutput,
                                         S3PredictionOutput,
                                         GCSPredictionOutput,
                                         ADLSPredictionOutput,
                                         LocalPredictionOutput],
                                   Field(discriminator='storage_type')]

validate_prediction_outputs = serialized_jobs.validate_prediction_outputs


def normalize_object_prediction_output(
    output: S3PredictionOutput | GCSPredictionOutput | ADLSPredictionOutput,
    connector_config: ConnectorResourceConfigDB,
) -> None:
    """Authorize and normalize one object-store prediction output.

    Provider type must match the selected connector. Rooted targets retain a
    connector-relative coordinate; rootless targets retain the canonical full
    provider URI already supplied through the existing output field.
    """
    from rest.config.connector import normalize_file_destination

    if isinstance(output, S3PredictionOutput):
        if connector_config.type != data_source_api.ConnectorType.S3:
            raise ValueError('S3 prediction output requires an S3 connector')
        output.file_path = normalize_file_destination(connector_config,
                                                      output.file_path)
        return
    if isinstance(output, GCSPredictionOutput):
        if connector_config.type != data_source_api.ConnectorType.GCS:
            raise ValueError('GCS prediction output requires a GCS connector')
        output.file_path = normalize_file_destination(connector_config,
                                                      output.file_path)
        return
    if isinstance(output, ADLSPredictionOutput):
        if connector_config.type != data_source_api.ConnectorType.ADLS:
            raise ValueError(
                'ADLS prediction output requires an ADLS connector')
        output.file_path = normalize_file_destination(connector_config,
                                                      output.file_path)
        return
    raise TypeError(f'Unsupported object prediction output: {type(output)}')
