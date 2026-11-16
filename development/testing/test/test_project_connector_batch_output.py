# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kumo_core.types import ConnectorType, PredictionArtifactType

from rest.config.data_backend import ConnectorExecutionRef
from rest.config.prediction_output import DatabricksPredictionOutput
from rest.deps.context import _current_user_ctx
from rest.driver.batch_prediction_job_logic import (
    BatchPredictionJobLogic,
    _PendingOutputs,
)
from statedb.config.common import ConnectorID, ProjectID
from statedb.utils.errors import ConnectorDoesNotExistError


@pytest.mark.asyncio
async def test_warehouse_output_validation_dispatches_project_reference(
) -> None:
    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()
    reference = ConnectorExecutionRef(project_id=project_id,
                                      connector_id=connector_id)
    connector_service = SimpleNamespace(execution_ref_for_project=AsyncMock(
        return_value=reference))
    logic = BatchPredictionJobLogic(
        ml_job_backend=None,
        data_backend_ctx=MagicMock(),
        state_services=SimpleNamespace(connectors=connector_service),
    )
    output = DatabricksPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=str(connector_id),
        table_name='PREDICTIONS',
    )
    principal = MagicMock()
    token = _current_user_ctx.set(principal)
    try:
        with patch(
                'rest.driver.batch_prediction_job_logic.DataBackend.'
                '_execute_single_activity',
                new=AsyncMock(return_value=None),
        ) as execute_activity:
            result = await logic._validate_warehouse_output(
                output,
                _PendingOutputs(),
                project_id,
            )
    finally:
        _current_user_ctx.reset(token)

    assert result is None
    connector_service.execution_ref_for_project.assert_awaited_once_with(
        principal,
        project_id,
        connector_id,
        expected_type=ConnectorType.DATABRICKS,
    )
    assert execute_activity.await_args.args[2] == [reference, output]


@pytest.mark.asyncio
async def test_warehouse_output_validation_rejects_hidden_connector() -> None:
    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()
    connector_service = SimpleNamespace(execution_ref_for_project=AsyncMock(
        side_effect=ConnectorDoesNotExistError(connector_id)))
    logic = BatchPredictionJobLogic(
        ml_job_backend=None,
        data_backend_ctx=MagicMock(),
        state_services=SimpleNamespace(connectors=connector_service),
    )
    output = DatabricksPredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=str(connector_id),
        table_name='PREDICTIONS',
    )

    with patch(
            'rest.driver.batch_prediction_job_logic.DataBackend.'
            '_execute_single_activity',
            new=AsyncMock(),
    ) as execute_activity:
        result = await logic._validate_warehouse_output(
            output,
            _PendingOutputs(),
            project_id,
        )

    assert result is not None
    assert 'does not exist' in result.message
    execute_activity.assert_not_awaited()
