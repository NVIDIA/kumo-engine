# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from kumo_core.types import PredictionArtifactType

from kumo.connector.connector import ConnectorError
from rest.backends.ml_backend.activities.parquet_io_activities import (
    write_artifacts_to_customer,
)
from rest.config.artifact_export import (
    ArtifactExportInput,
    ArtifactOutTable,
    NamedOutputConfig,
)
from rest.config.data_backend import ConnectorExecutionRef, ConnectorSpec
from rest.config.jobs import WriteMode
from statedb.config.common import ConnectorID, ProjectID
from statedb.utils.errors import ArtifactExportError
from temporalio_commons.process_pool_utils import HealingProcessPoolExecutor


def test_write_artifacts_to_customer_sanitizes_connector_error(
        caplog: pytest.LogCaptureFixture) -> None:
    provider_payload = 'provider said password=customer-secret-token'
    resolved_connectors = {PredictionArtifactType.PREDICTIONS: [Mock()]}

    with (
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                '_write_artifacts_to_customer',
                side_effect=ConnectorError(message=provider_payload),
            ),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                '_resolve_output_connectors',
                return_value=resolved_connectors,
            ),
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'activity.info',
                return_value=SimpleNamespace(
                    attempt=1,
                    activity_type='write-artifacts',
                    activity_id='test-activity',
                ),
            ),
    ):
        with pytest.raises(ArtifactExportError) as exc_info:
            write_artifacts_to_customer(Mock(), {})

    error = exc_info.value
    assert error.get_user_message() == (
        'The selected output connector rejected the write. '
        'Verify its native target permissions.')
    assert error.__cause__ is None
    assert provider_payload not in str(error)
    assert provider_payload not in caplog.text
    assert resolved_connectors == {}


def test_write_artifacts_resolves_connector_inside_activity() -> None:
    connector_id = ConnectorID('connector_' + 'a' * 32)
    reference = ConnectorExecutionRef(
        connector_id=connector_id,
        project_id=ProjectID('project_' + 'b' * 32),
    )
    artifact_type = PredictionArtifactType.PREDICTIONS
    input = ArtifactExportInput(
        artifact_paths={artifact_type: '/internal/predictions'},
        output_config={
            artifact_type: [
                NamedOutputConfig(
                    connector=reference,
                    out_table=ArtifactOutTable(
                        table_name='predictions',
                        write_mode=WriteMode.OVERWRITE,
                    ),
                )
            ],
        },
    )
    resolved = ConnectorSpec(
        connector_id=connector_id,
        connector_config=Mock(),
    )
    seen_connectors = {}

    def write(*_args, **kwargs):
        seen_connectors.update(kwargs['resolved_connectors'])
        return []

    with (
            patch(
                'rest.backends.data_backend.connector_broker.'
                'connector_credential_broker.resolve',
                return_value=resolved,
            ) as resolve,
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                '_write_artifacts_to_customer',
                side_effect=write,
            ),
            patch.object(HealingProcessPoolExecutor, 'submit') as
            process_pool_submit,
            patch(
                'rest.backends.ml_backend.activities.parquet_io_activities.'
                'activity.info',
                return_value=SimpleNamespace(
                    attempt=1,
                    activity_type='write-artifacts',
                    activity_id='test-activity',
                ),
            ),
    ):
        assert write_artifacts_to_customer(input, {}) == []

    resolve.assert_called_once_with(reference)
    assert seen_connectors == {artifact_type: [resolved]}
    process_pool_submit.assert_not_called()
