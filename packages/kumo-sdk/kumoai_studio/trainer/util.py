# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Mapping, Optional, Set, Tuple, Union

from kumoapi.jobs import (
    ADLSPredictionOutput,
    BigQueryPredictionOutput,
    DatabricksPredictionOutput,
    GCSPredictionOutput,
    LocalPredictionOutput,
    MetadataField,
    PredictionArtifactType,
    PredictionOutputConfig,
    PredictionStorageType,
    S3PredictionOutput,
    SnowflakePredictionOutput,
    WriteMode,
)

from kumoai_studio.artifact_export.config import OutputConfig
from kumoai_studio.connector import (
    ADLSConnector,
    BigQueryConnector,
    Connector,
    DatabricksConnector,
    GCSConnector,
    S3Connector,
    SnowflakeConnector,
)
from kumoai_studio.databricks import DB_SEP


def validate_output_arguments(
    output_types: Optional[Set[str]],
    output_connector: Optional[Connector] = None,
    output_table_name: Optional[str] = None,
    output_paths: Optional[Mapping[str, str]] = None,
) -> None:
    r"""Validate the output arguments for a prediction job or an export job."""
    assert output_types is not None
    if output_paths is not None:
        for path in output_paths.values():
            validate_output_arguments(output_types, output_connector, path)
        return
    output_types = {x.lower() for x in output_types}
    assert output_types.issubset({'predictions', 'embeddings'})
    if output_connector is not None:
        assert output_table_name is not None
        supported_output_connectors = (
            S3Connector,
            GCSConnector,
            SnowflakeConnector,
            DatabricksConnector,
            BigQueryConnector,
            ADLSConnector,
        )
        if not isinstance(output_connector, supported_output_connectors):
            raise ValueError(
                f"Connector type {type(output_connector)} is not supported for"
                f" outputs. Supported output connector types are S3, GCS, "
                f"ADLS, Snowflake, Databricks, and BigQuery.")
        if not isinstance(output_table_name, str):
            raise ValueError(
                f"The output table name must be a string for all "
                f"non-Databricks connectors. Got '{output_table_name}'.")

        if isinstance(output_connector,
                      (S3Connector, GCSConnector, ADLSConnector)):
            output_connector._rooted_destination(output_table_name)
        if isinstance(output_connector, DatabricksConnector):
            assert DB_SEP in output_table_name


def build_prediction_output_config(
    output_type: str,
    output_connector: Optional[Connector] = None,
    output_table_name: Optional[Union[str, Tuple]] = None,
    output_metadata_fields: Optional[List[MetadataField]] = None,
    output_config: Optional[OutputConfig] = None,
) -> PredictionOutputConfig:
    r"""Build the prediction output config."""
    assert output_config is not None
    artifact_type = PredictionArtifactType(output_type.upper())
    output_name = f"{output_table_name}_{output_type}"
    file_output_name = output_name
    if isinstance(output_table_name, str):
        stem, separator, extension = output_table_name.rpartition('.')
        if separator and extension.lower() in {'csv', 'parquet'}:
            file_output_name = f'{stem}_{output_type}.{extension}'
    if output_config.output_paths is not None:
        file_output_name = output_config.output_paths[output_type]
    output_metadata_fields = output_metadata_fields or []
    if output_connector is None:
        return LocalPredictionOutput(
            artifact_type=artifact_type,
            extra_fields=output_metadata_fields,
        )
    elif isinstance(output_connector, S3Connector):
        return S3PredictionOutput(
            artifact_type=artifact_type,
            connector_id=output_connector.id,
            file_path=output_connector._rooted_destination(file_output_name),
            extra_fields=output_metadata_fields,
        )
    elif isinstance(output_connector, GCSConnector):
        return GCSPredictionOutput(
            artifact_type=artifact_type,
            connector_id=output_connector.id,
            file_path=output_connector._rooted_destination(file_output_name),
            extra_fields=output_metadata_fields,
        )
    elif isinstance(output_connector, ADLSConnector):
        return ADLSPredictionOutput(
            artifact_type=artifact_type,
            connector_id=output_connector.id,
            file_path=output_connector._rooted_destination(file_output_name),
            extra_fields=output_metadata_fields,
        )
    elif isinstance(output_connector, SnowflakeConnector):
        return SnowflakePredictionOutput(
            artifact_type=artifact_type,
            connector_id=output_connector.id,
            table_name=output_name,
            extra_fields=output_metadata_fields,
            write_mode=output_config.connector_specific_config.write_mode
            if output_config.connector_specific_config is not None else
            WriteMode.OVERWRITE,
        )
    elif isinstance(output_connector, DatabricksConnector):
        return DatabricksPredictionOutput(
            artifact_type=artifact_type,
            connector_id=output_connector.id,
            table_name=output_name,
            extra_fields=output_metadata_fields,
            write_mode=output_config.connector_specific_config.write_mode
            if output_config.connector_specific_config is not None else
            WriteMode.OVERWRITE,
        )
    elif isinstance(output_connector, BigQueryConnector):
        return BigQueryPredictionOutput(
            storage_type=PredictionStorageType.BIGQUERY,
            artifact_type=artifact_type,
            connector_id=output_connector.id,
            table_name=output_name,
            extra_fields=output_metadata_fields,
            write_mode=output_config.connector_specific_config.write_mode
            if output_config.connector_specific_config is not None else
            WriteMode.OVERWRITE,
        )
    else:
        raise NotImplementedError()
