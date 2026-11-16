# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
from google.api_core.exceptions import Forbidden
from kumo_core.types import PredictionArtifactType

from kumo.connector import (
    BigQueryConnector,
    DatabricksConnector,
    SnowflakeConnector,
)
from kumo.connector.connector import ConnectorError
from kumo.connector.sql_connector import InternalWriteMode
from rest.backends.data_backend.activities.prediction_output_validation import (
    validate_object_prediction_output,
    validate_warehouse_prediction_output,
)
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.jobs import WriteMode
from rest.config.prediction_output import (
    BigQueryPredictionOutput,
    DatabricksPredictionOutput,
    GCSPredictionOutput,
    PredictionStorageType,
    S3PredictionOutput,
    SnowflakePredictionOutput,
)
from statedb.config.common import ConnectorID, ProjectID


def _reference(connector_id: ConnectorID) -> ConnectorExecutionRef:
    return ConnectorExecutionRef(
        connector_id=connector_id,
        project_id=ProjectID.generate(),
    )


def test_object_output_requires_matching_reference_and_provider_type():
    connector_id = ConnectorID.generate()
    output = S3PredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=connector_id,
        file_path='predictions/',
    )
    wrong_reference = _reference(ConnectorID.generate())

    with patch(
            'rest.backends.data_backend.activities.'
            'prediction_output_validation.get_connector_from_spec',
    ) as get_connector:
        result = validate_object_prediction_output(
            wrong_reference,
            output,
            False,
        )
    assert result is not None
    assert 'does not match' in result.message
    get_connector.assert_not_called()

    connector = Mock(
        root_dir='gs://bucket/root',
        filesystem_credential_scope=Mock(return_value=nullcontext()),
    )
    with patch(
            'rest.backends.data_backend.activities.'
            'prediction_output_validation.get_connector_from_spec',
            return_value=connector,
    ):
        result = validate_object_prediction_output(
            _reference(connector_id),
            output,
            False,
        )
    assert result is not None
    assert 'type does not match' in result.message


def test_object_output_rejects_path_outside_connector_target():
    connector_id = ConnectorID.generate()
    output = GCSPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=connector_id,
        file_path='gs://other-bucket/predictions/',
    )
    connector = Mock(
        root_dir='gs://authorized-bucket/root',
        filesystem_credential_scope=Mock(return_value=nullcontext()),
    )

    with patch(
            'rest.backends.data_backend.activities.'
            'prediction_output_validation.get_connector_from_spec',
            return_value=connector,
    ):
        result = validate_object_prediction_output(
            _reference(connector_id),
            output,
            False,
        )

    assert result is not None
    assert 'outside' in result.message or 'relative' in result.message
    connector.filesystem_credential_scope.assert_not_called()


def test_object_output_uses_explicit_connector_credential_scope():
    connector_id = ConnectorID.generate()
    output = S3PredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=connector_id,
        file_path='predictions/',
    )
    scope = Mock()
    scope.__enter__ = Mock(return_value=None)
    scope.__exit__ = Mock(return_value=False)
    connector = Mock(
        root_dir='s3://bucket/root',
        filesystem_credential_scope=Mock(return_value=scope),
    )

    with patch(
            'rest.backends.data_backend.activities.'
            'prediction_output_validation.get_connector_from_spec',
            return_value=connector,
    ), patch(
            'rest.backends.data_backend.activities.'
            'prediction_output_validation._validate_file_path',
            return_value=None,
    ):
        result = validate_object_prediction_output(
            _reference(connector_id),
            output,
            False,
        )

    assert result is None
    connector.filesystem_credential_scope.assert_called_once_with(
        's3://bucket/root/predictions/')
    scope.__enter__.assert_called_once()
    scope.__exit__.assert_called_once()


def test_bigquery_output_validation_probes_write_and_cleans_up():
    connector_id = ConnectorID.generate()
    output = BigQueryPredictionOutput(
        storage_type=PredictionStorageType.BIGQUERY,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=connector_id,
        table_name='predictions',
    )
    connector = BigQueryConnector(
        name=str(connector_id),
        project_id='customer-project',
        dataset_id='authorized_dataset',
    )
    context = Mock()
    connector._context = context
    from google.api_core.exceptions import NotFound
    context.get_table.side_effect = NotFound('missing')

    with patch(
            'rest.backends.data_backend.activities.'
            'prediction_output_validation.get_connector_from_spec',
            return_value=connector,
    ):
        result = validate_warehouse_prediction_output(_reference(connector_id),
                                                      output)

    assert result is None
    context.create_table.assert_called_once()
    context.delete_table.assert_called_once()


@pytest.mark.parametrize(
    ('output_type', 'storage_type'),
    [
        (BigQueryPredictionOutput, PredictionStorageType.BIGQUERY),
        (SnowflakePredictionOutput, PredictionStorageType.SNOWFLAKE),
    ],
)
def test_warehouse_output_rejects_staging_table_with_append_before_provider_io(
    output_type,
    storage_type,
):
    connector_id = ConnectorID.generate()
    output = output_type(
        storage_type=storage_type,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=connector_id,
        table_name='predictions',
        staging_table_name='predictions_staging',
        write_mode=WriteMode.APPEND,
    )

    with patch(
            'rest.backends.data_backend.activities.'
            'prediction_output_validation.get_connector_from_spec',
    ) as get_connector:
        result = validate_warehouse_prediction_output(
            _reference(connector_id),
            output,
        )

    assert result is not None
    assert result.message == (
        'A staging table may be used only with OVERWRITE mode.')
    get_connector.assert_not_called()


def test_bigquery_output_denial_is_sanitized(caplog):
    connector_id = ConnectorID.generate()
    output = BigQueryPredictionOutput(
        storage_type=PredictionStorageType.BIGQUERY,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=connector_id,
        table_name='predictions',
    )
    connector = BigQueryConnector(
        name=str(connector_id),
        project_id='customer-project',
        dataset_id='authorized_dataset',
    )
    connector._context = Mock()
    secret = 'provider-payload-with-client-secret'
    connector.context.create_table.side_effect = Forbidden(secret)

    with patch(
            'rest.backends.data_backend.activities.'
            'prediction_output_validation.get_connector_from_spec',
            return_value=connector,
    ):
        result = validate_warehouse_prediction_output(_reference(connector_id),
                                                      output)

    assert result is not None
    assert 'cannot write' in result.message
    assert secret not in result.message
    assert secret not in caplog.text


@pytest.mark.parametrize(
    ('table_name', 'write_mode', 'expected_mode'),
    [
        ('other_schema__kumo__predictions', WriteMode.OVERWRITE, None),
        (
            'authorized_schema__kumo__predictions',
            WriteMode.OVERWRITE,
            InternalWriteMode.OVERWRITE,
        ),
        (
            'authorized_schema__kumo__predictions',
            WriteMode.APPEND,
            InternalWriteMode.APPEND,
        ),
    ],
)
def test_databricks_output_validation_enforces_schema_and_write_mode(
    table_name,
    write_mode,
    expected_mode,
):
    connector_id = ConnectorID.generate()
    output = DatabricksPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=connector_id,
        table_name=table_name,
        write_mode=write_mode,
    )
    connector = DatabricksConnector(
        name=str(connector_id),
        host='https://workspace.example.com',
        cluster_id='',
        warehouse_id='warehouse',
        catalog='catalog',
        schema_name='authorized_schema',
    )

    with (
            patch(
                'rest.backends.data_backend.activities.'
                'prediction_output_validation.get_connector_from_spec',
                return_value=connector,
            ),
            patch.object(connector, 'open'),
            patch.object(connector, 'check_schema_exists'),
            patch.object(connector, 'check_table_write_permissions') as
            check_write,
    ):
        result = validate_warehouse_prediction_output(_reference(connector_id),
                                                      output)

    if expected_mode is None:
        assert result is not None
        assert 'connector schema' in result.message
        check_write.assert_not_called()
    else:
        assert result is None
        check_write.assert_called_once_with(
            table_name='predictions',
            table_schema='authorized_schema',
            mode=expected_mode,
        )


def test_snowflake_provider_error_is_not_returned_or_logged(caplog):
    connector_id = ConnectorID.generate()
    output = SnowflakePredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=connector_id,
        table_name='PREDICTIONS',
    )
    connector = SnowflakeConnector(
        name=str(connector_id),
        account='account',
        warehouse='warehouse',
        database='database',
        schema_name='authorized_schema',
    )
    connector._context = Mock()
    secret = 'provider-payload-with-session-token'

    with patch(
            'rest.backends.data_backend.activities.'
            'prediction_output_validation.get_connector_from_spec',
            return_value=connector,
    ), patch.object(
            connector,
            'check_schema_exists',
    ), patch.object(
            connector,
            'check_table_write_permissions',
            side_effect=ConnectorError(secret),
    ):
        result = validate_warehouse_prediction_output(_reference(connector_id),
                                                      output)

    assert result is not None
    assert 'configured Snowflake schema' in result.message
    assert secret not in result.message
    assert secret not in caplog.text
