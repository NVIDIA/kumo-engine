# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from io_utils import fsutil
from kumo.connector import (
    BigQueryConnector,
    DatabricksConnector,
    SnowflakeConnector,
)
from kumo.connector.connector import ConnectorError
from kumo.connector.sql_connector import InternalWriteMode
from rest.backends.data_backend import connector_activity_registry
from rest.config.artifact_export import (
    file_output_config,
    resolve_file_output_path,
)
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.jobs import WriteMode
from rest.config.prediction_output import (
    ADLSPredictionOutput,
    BigQueryPredictionOutput,
    DatabricksPredictionOutput,
    GCSPredictionOutput,
    S3PredictionOutput,
    SnowflakePredictionOutput,
)
from rest.utils.driver_common import get_connector_from_spec
from statedb.config.common import CallToAction

ObjectOutput = Union[S3PredictionOutput, GCSPredictionOutput,
                     ADLSPredictionOutput]
WarehouseOutput = Union[DatabricksPredictionOutput, SnowflakePredictionOutput,
                        BigQueryPredictionOutput]


def _validate_file_path(
    output_path: str,
    allow_local_paths: bool,
    is_partitioned: bool,
) -> CallToAction | None:
    if not allow_local_paths and fsutil.is_local(output_path):
        return CallToAction.crit(
            'The output location must be a cloud storage path.')
    try:
        if fsutil.exists(output_path):
            if is_partitioned:
                if fsutil.isdir(output_path) and fsutil.listdir(output_path):
                    return CallToAction.crit(
                        'The selected directory already exists and is '
                        'non-empty. Please clear it or select another path.')
                return CallToAction.crit(
                    'A file already exists at this location.')
            if fsutil.isdir(output_path):
                return CallToAction.crit(
                    'A directory already exists at the selected location.')
            return CallToAction.warn(
                'A file already exists and will be overwritten.')
    except (PermissionError, OSError):
        pass
    if not fsutil.has_write_permission(output_path):
        return CallToAction.crit(
            'The selected identity cannot write to this output location.')
    return None


@connector_activity_registry.connector_authorized_activity
def validate_object_prediction_output(
    reference: ConnectorExecutionRef,
    output: ObjectOutput,
    allow_local_paths: bool,
) -> CallToAction | None:
    if str(reference.connector_id) != str(output.connector_id):
        return CallToAction.crit(
            'Output connector reference does not match the selected '
            'connector.')
    connector = get_connector_from_spec(reference)
    try:
        if not hasattr(connector, 'filesystem_credential_scope'):
            return CallToAction.crit(
                'Selected output connector is not an object-store connector.')
        try:
            config = file_output_config(reference, output.file_path,
                                        output.file_type)
            output_path = resolve_file_output_path(config, connector)
            type_matches = (isinstance(output, S3PredictionOutput)
                            and fsutil.is_s3(output_path)
                            or isinstance(output, GCSPredictionOutput)
                            and fsutil.is_gcs(output_path)
                            or isinstance(output, ADLSPredictionOutput)
                            and fsutil.is_adls(output_path))
            if not type_matches:
                return CallToAction.crit(
                    'Selected output connector type does not match the '
                    'output type.')
        except ValueError as error:
            return CallToAction.crit(str(error))
        try:
            with connector.filesystem_credential_scope(output_path):
                return _validate_file_path(
                    output_path,
                    allow_local_paths,
                    output.is_partitioned,
                )
        except Exception:
            return CallToAction.crit(
                'Unable to validate write access with the selected '
                'object-store identity.')
    finally:
        connector.close()


@connector_activity_registry.connector_authorized_activity
def validate_warehouse_prediction_output(
    reference: ConnectorExecutionRef,
    output: WarehouseOutput,
) -> CallToAction | None:
    if str(reference.connector_id) != str(output.connector_id):
        return CallToAction.crit(
            'Output connector reference does not match the selected '
            'connector.')
    if (isinstance(output,
                   (BigQueryPredictionOutput, SnowflakePredictionOutput))
            and output.write_mode == WriteMode.APPEND
            and output.staging_table_name is not None):
        return CallToAction.crit(
            'A staging table may be used only with OVERWRITE mode.')
    connector = get_connector_from_spec(reference)
    try:
        if isinstance(output, BigQueryPredictionOutput):
            if not isinstance(connector, BigQueryConnector):
                return CallToAction.crit(
                    'Connector is not a BigQuery connector.')
            try:
                connector.open()
                mode = (InternalWriteMode.APPEND if output.write_mode
                        == WriteMode.APPEND else InternalWriteMode.OVERWRITE)
                connector.check_table_write_permissions(
                    table_name=output.table_name,
                    table_schema=connector.dataset_id,
                    mode=mode,
                )
                if output.staging_table_name is not None:
                    connector.check_table_write_permissions(
                        table_name=output.staging_table_name,
                        table_schema=connector.dataset_id,
                        mode=InternalWriteMode.OVERWRITE,
                    )
            except ConnectorError:
                return CallToAction.crit(
                    'The selected identity cannot write to the configured '
                    'BigQuery dataset and output table.')
            except Exception:
                return CallToAction.crit(
                    'Unable to validate BigQuery output write access.')
        if isinstance(output, SnowflakePredictionOutput):
            if not isinstance(connector, SnowflakeConnector):
                return CallToAction.crit(
                    'Connector is not a Snowflake connector.')
            try:
                connector.open()
                connector.check_schema_exists(connector.schema_name)
                connector.check_table_write_permissions(
                    table_name=output.table_name,
                    table_schema=connector.schema_name,
                    mode=(InternalWriteMode.APPEND if output.write_mode
                          == WriteMode.APPEND else InternalWriteMode.OVERWRITE),
                )
                if output.staging_table_name is not None:
                    connector.check_table_write_permissions(
                        table_name=output.staging_table_name,
                        table_schema=connector.schema_name,
                        mode=InternalWriteMode.OVERWRITE,
                    )
            except ConnectorError:
                return CallToAction.crit(
                    'The selected identity cannot create or modify tables '
                    'in the configured Snowflake schema.')
            except Exception:
                return CallToAction.crit(
                    'Unable to validate Snowflake output write access.')
        if isinstance(output, DatabricksPredictionOutput):
            if not isinstance(connector, DatabricksConnector):
                return CallToAction.crit(
                    'Connector is not a Databricks connector.')
            try:
                schema_name, table_name = (
                    connector.require_output_table_within_connector_target(
                        output.table_name))
            except ConnectorError:
                return CallToAction.crit(
                    'Databricks output table must use the connector schema.')
            try:
                connector.open()
                connector.check_schema_exists(schema_name)
                connector.check_table_write_permissions(
                    table_name=table_name,
                    table_schema=schema_name,
                    mode=(InternalWriteMode.APPEND if output.write_mode
                          == WriteMode.APPEND else InternalWriteMode.OVERWRITE),
                )
            except ConnectorError:
                return CallToAction.crit(
                    'The selected identity cannot create or modify tables '
                    'in the configured Databricks schema.')
            except Exception:
                return CallToAction.crit(
                    'Unable to validate Databricks output write access.')
        return None
    finally:
        try:
            connector.close()
        except Exception:
            pass
